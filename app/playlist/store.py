"""Create, list, and populate playlists (real imports + smart/virtual builds)."""
from ..db import connect, rows_to_dicts

FEATURE_SELECT = """
    t.id, t.title, t.artist, t.album, t.genre, t.release_date, t.preview_url,
    f.bpm, f.key, f.mode, f.camelot, f.energy, f.danceability, f.valence,
    f.acousticness, f.brightness, f.instrumentalness, f.speechiness, f.liveness,
    f.loudness_db, f.mfcc
"""


def list_playlists():
    with connect() as con:
        return rows_to_dicts(con.execute(
            """SELECT p.id, p.name, p.source,
                      COUNT(pt.track_id) n_tracks,
                      SUM(CASE WHEN f.track_id IS NOT NULL AND f.error IS NULL THEN 1 ELSE 0 END) n_analyzed
               FROM playlists p
               LEFT JOIN playlist_tracks pt ON pt.playlist_id = p.id
               LEFT JOIN audio_features f ON f.track_id = pt.track_id
               GROUP BY p.id ORDER BY p.created_at DESC""").fetchall())


def create(name, track_ids, source="custom", external_id=None):
    with connect() as con:
        cur = con.execute(
            "INSERT OR IGNORE INTO playlists (name, source, external_id) VALUES (?,?,?)",
            (name, source, external_id))
        if cur.lastrowid and cur.rowcount:
            pid = cur.lastrowid
        else:  # existing (source, external_id) — replace its tracks
            pid = con.execute(
                "SELECT id FROM playlists WHERE source=? AND external_id=?",
                (source, external_id)).fetchone()["id"]
            con.execute("DELETE FROM playlist_tracks WHERE playlist_id=?", (pid,))
        for pos, tid in enumerate(track_ids):
            con.execute(
                "INSERT OR IGNORE INTO playlist_tracks (playlist_id, track_id, position) VALUES (?,?,?)",
                (pid, tid, pos))
    return pid


def delete(pid):
    with connect() as con:
        con.execute("DELETE FROM playlist_tracks WHERE playlist_id=?", (pid,))
        con.execute("DELETE FROM playlists WHERE id=?", (pid,))


def get_tracks(pid):
    with connect() as con:
        meta = con.execute("SELECT * FROM playlists WHERE id=?", (pid,)).fetchone()
        if not meta:
            return None, None
        rows = con.execute(
            f"""SELECT {FEATURE_SELECT}, pt.position
               FROM playlist_tracks pt
               JOIN tracks t ON t.id = pt.track_id
               LEFT JOIN audio_features f ON f.track_id = t.id AND f.error IS NULL
               WHERE pt.playlist_id = ? ORDER BY pt.position""", (pid,)).fetchall()
    return dict(meta), rows_to_dicts(rows)


def build_smart(kind="top", value=None, limit=40):
    """Create a virtual playlist from your own library.

    kind: 'top' (most-played) · 'source' (value=spotify|ytmusic|apple) ·
          'genre' (value=genre name) · 'recent' (latest plays)
    """
    with connect() as con:
        if kind == "source":
            rows = con.execute(
                """SELECT t.id, COUNT(*) n FROM plays p JOIN tracks t ON t.id=p.track_id
                   WHERE p.source=? GROUP BY t.id ORDER BY n DESC LIMIT ?""",
                (value, limit)).fetchall()
            name = f"Top from {value}"
        elif kind == "genre":
            rows = con.execute(
                """SELECT t.id, COUNT(p.id) n FROM tracks t
                   LEFT JOIN plays p ON p.track_id=t.id
                   WHERE t.genre=? GROUP BY t.id ORDER BY n DESC LIMIT ?""",
                (value, limit)).fetchall()
            name = f"{value} tracks"
        elif kind == "recent":
            rows = con.execute(
                """SELECT t.id, MAX(p.played_at) m FROM plays p JOIN tracks t ON t.id=p.track_id
                   GROUP BY t.id ORDER BY m DESC LIMIT ?""", (limit,)).fetchall()
            name = "Recently played"
        else:  # top
            rows = con.execute(
                """SELECT t.id, COUNT(*) n FROM plays p JOIN tracks t ON t.id=p.track_id
                   GROUP BY t.id ORDER BY n DESC LIMIT ?""", (limit,)).fetchall()
            name = "Most played"
    track_ids = [r["id"] for r in rows]
    if not track_ids:
        return None
    return create(name, track_ids, source="custom")
