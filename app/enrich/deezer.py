"""Enrich tracks via the Deezer public API (no key required).

Gives us: 30-second preview MP3s (input for signal analysis), album genre,
release date, and artist-relation data used by the recommender.
"""
import re
import time

import requests

from ..db import connect

API = "https://api.deezer.com"
_genre_cache = {}

_BRACKETS = re.compile(r"[\(\[\{][^)\]\}]*[\)\]\}]")
# YouTube titles bury the song name: "Nijamellam Video Song | Singers | Movie".
# Everything from the first of these markers on is noise, so the leading phrase
# is the song. Used only as a fallback when the raw title fails to match.
_SONG_CUT = re.compile(
    r"\s*(?:\||[-–—]| video| song| official| lyric| promo| teaser| from | ft\.?"
    r"| feat\.?| with lyrics| 4k| 8k| hd)\b", re.I)


_NOISE_WORDS = {"official", "music", "video", "audio", "lyric", "lyrical", "full",
                "hd", "4k", "8k", "song", "songs", "new", "presenting", "presents",
                "the", "from", "feat", "ft", "cover"}


def _clean_song(text):
    t = _BRACKETS.sub(" ", text or "")
    core = _SONG_CUT.split(t, maxsplit=1)[0]
    words = re.sub(r"\s+", " ", core).strip(" -–—|:\"").split()
    while words and words[0].lower() in _NOISE_WORDS:
        words.pop(0)
    while words and words[-1].lower() in _NOISE_WORDS:
        words.pop()
    core = " ".join(words)
    # reject an all-noise or too-short leftover (e.g. a title that *starts* with
    # "Official Lyric Video | …" leaves nothing usable)
    return core if len(core) >= 3 and any(w.lower() not in _NOISE_WORDS for w in words) else ""


def _tok(s):
    return {w for w in re.split(r"[^a-z0-9]+", (s or "").lower()) if len(w) >= 3}


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


def _first(q):
    try:
        hits = _get("/search", {"q": q, "limit": 1}).get("data") or []
        return hits[0] if hits else None
    except Exception:
        return None


def _plausible(hit, song, artist):
    """Guard the loose fallbacks: the hit's title must LEAD with our song name
    (so 'So Baby' won't grab 'Baby Songs To Go To Sleep'), or the artist must
    genuinely match (covers the movie-as-artist rows where the composer differs)."""
    ht = (hit.get("title") or "").lower().strip()
    s = (song or "").lower().strip()
    if s and (ht.startswith(s) or (len(ht) >= 4 and s.startswith(ht))):
        return True
    return bool(_tok(artist) & _tok((hit.get("artist") or {}).get("name")))


def search_track(title, artist):
    song = _clean_song(title)
    # 1) exact-field query is self-validating — trust it directly
    hit = _first(f'track:"{title}" artist:"{artist}"')
    if hit:
        return hit
    # 2) every looser query must pass the plausibility guard (a bare
    #    "artist title" search on a garbled title can return the wrong song)
    queries = [f"{artist} {title}"]
    if song and song.lower() != (title or "").strip().lower():
        queries.append(f"{song} {artist}")
        if len(_tok(song)) >= 2:   # a bare song search is only safe when distinctive
            queries.append(f'track:"{song}"')
    for q in queries:
        hit = _first(q)
        if hit and _plausible(hit, song or title, artist):
            return hit
    return None


def enrich_normalized_batch(limit=60):
    """Retry the previously-unmatched (enriched=-1) tracks using their NORMALIZED
    clean_title + movie/album — the fields the ETL recovered from garbled titles.
    Skips Shorts and non-music. Requires the normalize ETL to have run.

    Marks a still-unmatched track enriched=-2 so it isn't retried forever."""
    matched = failed = 0
    pool = ("enriched=-1 AND clean_title IS NOT NULL AND length(clean_title)>=3 "
            "AND COALESCE(is_short,0)=0 AND COALESCE(is_non_music,0)=0")
    with connect() as con:
        rows = con.execute(
            f"SELECT id, clean_title, movie_album, artist FROM tracks WHERE {pool} LIMIT ?",
            (limit,)).fetchall()
    for row in rows:
        song = row["clean_title"]
        hit = None
        for ctx in (row["movie_album"], row["artist"], song):
            if ctx:
                hit = search_track(song, ctx)
                if hit:
                    break
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
                     (hit.get("duration") or 0) * 1000 or None, row["id"]))
                matched += 1
            else:
                con.execute("UPDATE tracks SET enriched=-2 WHERE id=?", (row["id"],))
                failed += 1
        time.sleep(0.15)
    with connect() as con:
        remaining = con.execute(f"SELECT COUNT(*) c FROM tracks WHERE {pool}").fetchone()["c"]
    return {"matched": matched, "failed": failed, "remaining": remaining}


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
