"""Storage: SQLite for facts, memory-mapped float16 matrices for vectors.

There is no vector database here and that is a deliberate choice. A brute-force
cosine over 100k tracks in a 2304-dim fp16 matrix is roughly 460 MB of memory
traffic and lands in tens of milliseconds — faster than the HNSW build would
amortise, exact rather than approximate, and with no server to run. The ANN
path exists behind the same interface for when a library genuinely outgrows
that, and pgvector is there if the index needs to be shared between machines.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS tracks (
    id           INTEGER PRIMARY KEY,
    digest       TEXT UNIQUE NOT NULL,
    path         TEXT NOT NULL,
    title        TEXT,
    artist       TEXT,
    album        TEXT,
    albumartist  TEXT,
    year         INTEGER,
    track_no     INTEGER,
    duration_s   REAL,
    codec        TEXT,
    bitrate      INTEGER,
    lossless     INTEGER DEFAULT 0,
    mbid         TEXT,
    isrc         TEXT,
    acoustid     TEXT,
    added_at     REAL,
    analysed_at  REAL,
    embedded_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
CREATE INDEX IF NOT EXISTS idx_tracks_mbid   ON tracks(mbid);

CREATE TABLE IF NOT EXISTS features (
    track_id INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    data     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slop (
    track_id   INTEGER PRIMARY KEY REFERENCES tracks(id) ON DELETE CASCADE,
    score      REAL NOT NULL,
    confidence REAL NOT NULL,
    data       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS tags (
    track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    tag      TEXT NOT NULL,
    weight   REAL NOT NULL,
    source   TEXT NOT NULL,
    PRIMARY KEY (track_id, tag, source)
);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

CREATE TABLE IF NOT EXISTS plays (
    id        INTEGER PRIMARY KEY,
    track_id  INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    played_at REAL NOT NULL,
    played_ms INTEGER,
    skipped   INTEGER DEFAULT 0,
    context   TEXT
);
CREATE INDEX IF NOT EXISTS idx_plays_time ON plays(played_at);

CREATE TABLE IF NOT EXISTS labels (
    track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    axis     TEXT NOT NULL,
    value    REAL NOT NULL,
    noted_at REAL,
    PRIMARY KEY (track_id, axis)
);

CREATE TABLE IF NOT EXISTS vec_rows (
    space    TEXT NOT NULL,
    track_id INTEGER REFERENCES tracks(id) ON DELETE CASCADE,
    row      INTEGER NOT NULL,
    PRIMARY KEY (space, track_id)
);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class Track:
    id: int
    digest: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    duration_s: float | None
    year: int | None = None
    lossless: bool = False
    bitrate: int | None = None
    codec: str | None = None

    @property
    def label(self) -> str:
        return f"{self.artist or 'Unknown'} — {self.title or Path(self.path).stem}"


class Library:
    def __init__(self, paths):
        self.paths = paths
        paths.ensure()
        self.db = sqlite3.connect(paths.db)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.vectors = VectorStore(paths.vectors)

    # -- tracks -----------------------------------------------------------
    def upsert_track(self, digest: str, path: Path, info, tags: dict) -> int:
        row = self.db.execute("SELECT id FROM tracks WHERE digest=?", (digest,)).fetchone()
        fields = dict(
            path=str(path),
            title=tags.get("title"),
            artist=tags.get("artist"),
            album=tags.get("album"),
            albumartist=tags.get("album_artist") or tags.get("albumartist"),
            year=_year(tags.get("date") or tags.get("year")),
            track_no=_int(tags.get("track")),
            duration_s=info.duration_s,
            codec=info.codec,
            bitrate=info.bit_rate,
            lossless=int(info.lossless),
            isrc=tags.get("isrc"),
            mbid=tags.get("musicbrainz_trackid") or tags.get("musicbrainz_releasetrackid"),
        )
        if row:
            sets = ", ".join(f"{k}=?" for k in fields)
            self.db.execute(f"UPDATE tracks SET {sets} WHERE id=?",
                            (*fields.values(), row["id"]))
            self.db.commit()
            return int(row["id"])
        cols = ", ".join(["digest", *fields, "added_at"])
        marks = ", ".join("?" * (len(fields) + 2))
        cur = self.db.execute(f"INSERT INTO tracks ({cols}) VALUES ({marks})",
                              (digest, *fields.values(), time.time()))
        self.db.commit()
        return int(cur.lastrowid)

    def get(self, track_id: int) -> Track | None:
        r = self.db.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
        return _to_track(r) if r else None

    def find(self, query: str, limit: int = 10) -> list[Track]:
        like = f"%{query}%"
        rows = self.db.execute(
            "SELECT * FROM tracks WHERE title LIKE ? OR artist LIKE ? OR path LIKE ?"
            " ORDER BY artist, title LIMIT ?", (like, like, like, limit)).fetchall()
        return [_to_track(r) for r in rows]

    def all_tracks(self) -> list[Track]:
        return [_to_track(r) for r in
                self.db.execute("SELECT * FROM tracks ORDER BY id").fetchall()]

    def pending(self, stage: str) -> list[Track]:
        col = {"analyse": "analysed_at", "embed": "embedded_at"}[stage]
        rows = self.db.execute(
            f"SELECT * FROM tracks WHERE {col} IS NULL ORDER BY id").fetchall()
        return [_to_track(r) for r in rows]

    def mark(self, track_id: int, stage: str) -> None:
        col = {"analyse": "analysed_at", "embed": "embedded_at"}[stage]
        self.db.execute(f"UPDATE tracks SET {col}=? WHERE id=?", (time.time(), track_id))
        self.db.commit()

    # -- derived data -----------------------------------------------------
    def put_features(self, track_id: int, features: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO features (track_id, data) VALUES (?, ?)",
                        (track_id, json.dumps(features)))
        self.db.commit()

    def features(self, track_id: int) -> dict | None:
        r = self.db.execute("SELECT data FROM features WHERE track_id=?", (track_id,)).fetchone()
        return json.loads(r["data"]) if r else None

    def all_features(self) -> dict[int, dict]:
        return {int(r["track_id"]): json.loads(r["data"])
                for r in self.db.execute("SELECT track_id, data FROM features")}

    def put_slop(self, track_id: int, verdict) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO slop (track_id, score, confidence, data) VALUES (?,?,?,?)",
            (track_id, verdict.score, verdict.confidence, json.dumps(verdict.to_dict())))
        self.db.commit()

    def slop_scores(self) -> dict[int, float]:
        return {int(r["track_id"]): float(r["score"])
                for r in self.db.execute("SELECT track_id, score FROM slop")}

    def slop(self, track_id: int) -> dict | None:
        r = self.db.execute("SELECT data FROM slop WHERE track_id=?", (track_id,)).fetchone()
        return json.loads(r["data"]) if r else None

    def put_tags(self, track_id: int, tags: dict[str, float], source: str) -> None:
        self.db.executemany(
            "INSERT OR REPLACE INTO tags (track_id, tag, weight, source) VALUES (?,?,?,?)",
            [(track_id, t, float(w), source) for t, w in tags.items()])
        self.db.commit()

    def tags(self, track_id: int, limit: int = 12) -> list[tuple[str, float]]:
        rows = self.db.execute(
            "SELECT tag, weight FROM tags WHERE track_id=? ORDER BY weight DESC LIMIT ?",
            (track_id, limit)).fetchall()
        return [(r["tag"], float(r["weight"])) for r in rows]

    # -- behaviour --------------------------------------------------------
    def log_play(self, track_id: int, played_ms: int, skipped: bool,
                 context: str | None = None, at: float | None = None) -> None:
        self.db.execute(
            "INSERT INTO plays (track_id, played_at, played_ms, skipped, context)"
            " VALUES (?,?,?,?,?)",
            (track_id, at or time.time(), played_ms, int(skipped), context))
        self.db.commit()

    def play_counts(self) -> dict[int, int]:
        return {int(r["track_id"]): int(r["n"]) for r in self.db.execute(
            "SELECT track_id, COUNT(*) n FROM plays WHERE skipped=0 GROUP BY track_id")}

    def history(self, limit: int | None = None) -> list[tuple[int, float]]:
        sql = "SELECT track_id, played_at FROM plays WHERE skipped=0 ORDER BY played_at"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [(int(r["track_id"]), float(r["played_at"])) for r in self.db.execute(sql)]

    def set_label(self, track_id: int, axis: str, value: float) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO labels (track_id, axis, value, noted_at) VALUES (?,?,?,?)",
            (track_id, axis, float(value), time.time()))
        self.db.commit()

    def labels(self, axis: str) -> dict[int, float]:
        return {int(r["track_id"]): float(r["value"]) for r in self.db.execute(
            "SELECT track_id, value FROM labels WHERE axis=?", (axis,))}

    def label_axes(self) -> list[str]:
        return [r["axis"] for r in self.db.execute(
            "SELECT axis, COUNT(*) n FROM labels GROUP BY axis ORDER BY n DESC")]

    def close(self) -> None:
        self.vectors.flush()
        self.db.close()


class VectorStore:
    """One append-only float16 matrix per space, plus a row map in SQLite."""

    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, np.ndarray] = {}
        self._dirty: dict[str, list[np.ndarray]] = {}

    def _file(self, space: str) -> Path:
        return self.root / f"{space}.f16.npy"

    def matrix(self, space: str) -> np.ndarray | None:
        if space in self._cache:
            return self._cache[space]
        f = self._file(space)
        if not f.exists():
            return None
        m = np.load(f, mmap_mode="r")
        self._cache[space] = m
        return m

    def append(self, db: sqlite3.Connection, space: str, track_id: int,
               vec: np.ndarray) -> None:
        vec = np.asarray(vec, dtype=np.float16).reshape(-1)
        existing = db.execute("SELECT row FROM vec_rows WHERE space=? AND track_id=?",
                              (space, track_id)).fetchone()
        current = self.matrix(space)
        if existing is not None and current is not None:
            m = np.load(self._file(space), mmap_mode="r+")
            m[int(existing["row"])] = vec
            m.flush()
            self._cache.pop(space, None)
            return
        rows = 0 if current is None else current.shape[0]
        self._dirty.setdefault(space, []).append(vec)
        db.execute("INSERT OR REPLACE INTO vec_rows (space, track_id, row) VALUES (?,?,?)",
                   (space, track_id, rows + len(self._dirty[space]) - 1))

    def flush(self) -> None:
        for space, vecs in self._dirty.items():
            if not vecs:
                continue
            new = np.stack(vecs)
            f = self._file(space)
            if f.exists():
                old = np.load(f)
                new = np.concatenate([old, new])
            np.save(f, new)
            self._cache.pop(space, None)
        self._dirty.clear()

    def ids_and_matrix(self, db: sqlite3.Connection, space: str
                       ) -> tuple[np.ndarray, np.ndarray] | None:
        m = self.matrix(space)
        if m is None:
            return None
        rows = db.execute("SELECT track_id, row FROM vec_rows WHERE space=? ORDER BY row",
                          (space,)).fetchall()
        if not rows:
            return None
        ids = np.array([int(r["track_id"]) for r in rows], dtype=np.int64)
        idx = np.array([int(r["row"]) for r in rows], dtype=np.int64)
        idx = idx[idx < m.shape[0]]
        return ids[: len(idx)], np.asarray(m)[idx]

    def vector(self, db: sqlite3.Connection, space: str, track_id: int) -> np.ndarray | None:
        r = db.execute("SELECT row FROM vec_rows WHERE space=? AND track_id=?",
                       (space, track_id)).fetchone()
        m = self.matrix(space)
        if r is None or m is None or int(r["row"]) >= m.shape[0]:
            return None
        return np.asarray(m[int(r["row"])], dtype=np.float32)


def cosine_search(matrix: np.ndarray, query: np.ndarray, k: int
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Exact top-k cosine. Rows and query are assumed L2-normalised."""
    q = np.asarray(query, dtype=np.float32).reshape(-1)
    q = q / (np.linalg.norm(q) + 1e-9)
    scores = np.asarray(matrix, dtype=np.float32) @ q
    k = min(k, len(scores))
    top = np.argpartition(-scores, k - 1)[:k]
    top = top[np.argsort(-scores[top])]
    return top, scores[top]


def _to_track(r: sqlite3.Row) -> Track:
    return Track(id=int(r["id"]), digest=r["digest"], path=r["path"], title=r["title"],
                 artist=r["artist"], album=r["album"], duration_s=r["duration_s"],
                 year=r["year"], lossless=bool(r["lossless"]), bitrate=r["bitrate"],
                 codec=r["codec"])


def _year(v) -> int | None:
    if not v:
        return None
    digits = "".join(c for c in str(v) if c.isdigit())[:4]
    return int(digits) if len(digits) == 4 else None


def _int(v) -> int | None:
    try:
        return int(str(v).split("/")[0])
    except (TypeError, ValueError):
        return None
