import json
import sqlite3
import threading
from contextlib import contextmanager

from .config import DB_PATH

_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    album TEXT,
    duration_ms INTEGER,
    spotify_id TEXT,
    deezer_id INTEGER,
    preview_url TEXT,
    genre TEXT,
    release_date TEXT,
    popularity INTEGER,
    enriched INTEGER DEFAULT 0,
    UNIQUE(title, artist)
);

CREATE TABLE IF NOT EXISTS plays (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    played_at TEXT NOT NULL,
    source TEXT NOT NULL,
    UNIQUE(track_id, played_at, source)
);

CREATE TABLE IF NOT EXISTS audio_features (
    track_id INTEGER PRIMARY KEY REFERENCES tracks(id),
    bpm REAL,
    key TEXT,
    mode TEXT,
    energy REAL,
    loudness_db REAL,
    danceability REAL,
    valence REAL,
    acousticness REAL,
    brightness REAL,
    spectral_centroid REAL,
    spectral_rolloff REAL,
    zero_crossing_rate REAL,
    dynamic_range REAL,
    mfcc TEXT,
    -- Phase 1 deep features
    tempo_confidence REAL,
    beat_regularity REAL,
    onset_rate REAL,
    time_signature INTEGER,
    harmonic_ratio REAL,
    key_confidence REAL,
    camelot TEXT,
    spectral_bandwidth REAL,
    spectral_contrast REAL,
    spectral_flatness REAL,
    instrumentalness REAL,
    speechiness REAL,
    liveness REAL,
    analyzed_at TEXT DEFAULT CURRENT_TIMESTAMP,
    error TEXT
);

CREATE TABLE IF NOT EXISTS top_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    item_type TEXT NOT NULL,      -- 'track' | 'artist'
    time_range TEXT NOT NULL,     -- 'short_term' | 'medium_term' | 'long_term'
    rank INTEGER NOT NULL,
    name TEXT NOT NULL,
    artist TEXT,
    track_id INTEGER REFERENCES tracks(id),
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, item_type, time_range, rank)
);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    artist TEXT NOT NULL,
    deezer_id INTEGER,
    preview_url TEXT,
    reason TEXT,
    score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(title, artist)
);

CREATE TABLE IF NOT EXISTS playlists (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source TEXT DEFAULT 'custom',    -- 'spotify' | 'ytmusic' | 'custom'
    external_id TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(source, external_id)
);

CREATE TABLE IF NOT EXISTS playlist_tracks (
    playlist_id INTEGER NOT NULL REFERENCES playlists(id) ON DELETE CASCADE,
    track_id INTEGER NOT NULL REFERENCES tracks(id),
    position INTEGER NOT NULL,
    PRIMARY KEY (playlist_id, position)
);

CREATE TABLE IF NOT EXISTS lyrics (
    track_id INTEGER PRIMARY KEY REFERENCES tracks(id),
    text TEXT,
    sentiment REAL,
    word_count INTEGER,
    unique_ratio REAL,
    top_words TEXT,
    explicit INTEGER,
    fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
    error TEXT
);

CREATE INDEX IF NOT EXISTS idx_plays_track ON plays(track_id);
CREATE INDEX IF NOT EXISTS idx_plays_time ON plays(played_at);
CREATE INDEX IF NOT EXISTS idx_pltracks ON playlist_tracks(playlist_id);
"""

# Columns added after the original release — applied to existing DBs by _migrate().
_ADDED_COLUMNS = {
    "audio_features": [
        ("tempo_confidence", "REAL"), ("beat_regularity", "REAL"),
        ("onset_rate", "REAL"), ("time_signature", "INTEGER"),
        ("harmonic_ratio", "REAL"), ("key_confidence", "REAL"),
        ("camelot", "TEXT"), ("spectral_bandwidth", "REAL"),
        ("spectral_contrast", "REAL"), ("spectral_flatness", "REAL"),
        ("instrumentalness", "REAL"), ("speechiness", "REAL"),
        ("liveness", "REAL"),
    ],
}


def _migrate(con):
    for table, cols in _ADDED_COLUMNS.items():
        existing = {r["name"] for r in con.execute(f"PRAGMA table_info({table})")}
        for name, coltype in cols:
            if name not in existing:
                con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {coltype}")


def init_db():
    with connect() as con:
        con.executescript(SCHEMA)
        _migrate(con)


@contextmanager
def connect():
    with _lock:
        con = sqlite3.connect(DB_PATH, timeout=15)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA busy_timeout=15000")
        try:
            yield con
            con.commit()
        finally:
            con.close()


def upsert_track(con, title, artist, album=None, duration_ms=None,
                 spotify_id=None, popularity=None, release_date=None):
    """Insert a track if new, return its id. Fills in missing fields on conflict."""
    title = (title or "").strip()
    artist = (artist or "").strip()
    if not title or not artist:
        return None
    con.execute(
        """INSERT INTO tracks (title, artist, album, duration_ms, spotify_id, popularity, release_date)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(title, artist) DO UPDATE SET
             album = COALESCE(tracks.album, excluded.album),
             duration_ms = COALESCE(tracks.duration_ms, excluded.duration_ms),
             spotify_id = COALESCE(tracks.spotify_id, excluded.spotify_id),
             popularity = COALESCE(tracks.popularity, excluded.popularity),
             release_date = COALESCE(tracks.release_date, excluded.release_date)
        """,
        (title, artist, album, duration_ms, spotify_id, popularity, release_date),
    )
    row = con.execute(
        "SELECT id FROM tracks WHERE title=? AND artist=?", (title, artist)
    ).fetchone()
    return row["id"] if row else None


def add_play(con, track_id, played_at, source):
    con.execute(
        "INSERT OR IGNORE INTO plays (track_id, played_at, source) VALUES (?,?,?)",
        (track_id, played_at, source),
    )


def rows_to_dicts(rows):
    return [dict(r) for r in rows]


def json_field(value):
    return json.dumps(value) if value is not None else None
