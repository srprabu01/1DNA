"""Enrich tracks via the Deezer public API (no key required).

Gives us: 30-second preview MP3s (input for signal analysis), album genre,
release date, and artist-relation data used by the recommender.
"""
import time

import requests

from ..db import connect

API = "https://api.deezer.com"
_genre_cache = {}


def _get(path, params=None):
    r = requests.get(f"{API}{path}", params=params or {}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and data.get("error"):
        code = data["error"].get("code")
        if code == 4:  # quota — back off once and retry
            time.sleep(5)
            r = requests.get(f"{API}{path}", params=params or {}, timeout=20)
            data = r.json()
            if isinstance(data, dict) and data.get("error"):
                raise RuntimeError(str(data["error"]))
        else:
            raise RuntimeError(str(data["error"]))
    return data


def fresh_preview(deezer_id):
    """Deezer preview URLs are signed and expire (~24h). Re-fetch a live one."""
    try:
        tr = _get(f"/track/{deezer_id}")
        return tr.get("preview") or None
    except Exception:
        return None


def search_track(title, artist):
    q = f'track:"{title}" artist:"{artist}"'
    data = _get("/search", {"q": q, "limit": 1})
    hits = data.get("data", [])
    if not hits:
        # looser search
        data = _get("/search", {"q": f"{artist} {title}", "limit": 1})
        hits = data.get("data", [])
    return hits[0] if hits else None


def album_genre(album_id):
    if album_id in _genre_cache:
        return _genre_cache[album_id]
    try:
        alb = _get(f"/album/{album_id}")
        genres = [g["name"] for g in (alb.get("genres") or {}).get("data", [])]
        genre = genres[0] if genres else None
        _genre_cache[album_id] = (genre, alb.get("release_date"))
    except Exception:
        _genre_cache[album_id] = (None, None)
    return _genre_cache[album_id]


def artist_related(artist_name, limit=5):
    data = _get("/search/artist", {"q": artist_name, "limit": 1})
    hits = data.get("data", [])
    if not hits:
        return []
    rel = _get(f"/artist/{hits[0]['id']}/related", {"limit": limit})
    return rel.get("data", [])


def artist_top_tracks(artist_id, limit=5):
    data = _get(f"/artist/{artist_id}/top", {"limit": limit})
    return data.get("data", [])


def enrich_batch(limit=40):
    """Match un-enriched tracks against Deezer. Returns summary."""
    matched = failed = 0
    with connect() as con:
        rows = con.execute(
            "SELECT id, title, artist FROM tracks WHERE enriched=0 LIMIT ?", (limit,)
        ).fetchall()
    for row in rows:
        try:
            hit = search_track(row["title"], row["artist"])
        except Exception:
            hit = None
        with connect() as con:
            if hit:
                genre, release = (None, None)
                alb = hit.get("album") or {}
                if alb.get("id"):
                    genre, release = album_genre(alb["id"])
                con.execute(
                    """UPDATE tracks SET deezer_id=?, preview_url=?, genre=COALESCE(?, genre),
                       release_date=COALESCE(release_date, ?), duration_ms=COALESCE(duration_ms, ?),
                       enriched=1 WHERE id=?""",
                    (hit["id"], hit.get("preview") or None, genre, release,
                     (hit.get("duration") or 0) * 1000 or None, row["id"]),
                )
                matched += 1
            else:
                con.execute("UPDATE tracks SET enriched=-1 WHERE id=?", (row["id"],))
                failed += 1
        time.sleep(0.15)  # stay well under Deezer's 50 req / 5 s quota
    with connect() as con:
        remaining = con.execute(
            "SELECT COUNT(*) c FROM tracks WHERE enriched=0"
        ).fetchone()["c"]
    return {"matched": matched, "failed": failed, "remaining": remaining}
