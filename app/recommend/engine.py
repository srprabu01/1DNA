"""Recommendation engine.

Strategy: take your most-played artists, walk Deezer's related-artist graph,
pull those artists' top tracks, drop anything already in your library, then
rank candidates by (a) similarity of their analyzed audio features to your
play-weighted taste vector when available, and (b) artist affinity + Deezer rank.
"""
import time

import numpy as np

from ..analytics.engine import FEATURE_COLS
from ..db import connect
from ..enrich import deezer


def _taste_vector():
    with connect() as con:
        rows = con.execute(
            """SELECT f.energy, f.danceability, f.valence, f.acousticness,
                      f.brightness, COALESCE(p.n,1) w
               FROM audio_features f
               LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                 ON p.track_id = f.track_id
               WHERE f.error IS NULL"""
        ).fetchall()
    vecs, weights = [], []
    for r in rows:
        v = [r[c] for c in FEATURE_COLS]
        if all(x is not None for x in v):
            vecs.append(v)
            weights.append(r["w"])
    if not vecs:
        return None
    return np.average(np.array(vecs), axis=0, weights=np.array(weights))


def _known_pairs(con):
    return {
        (r["title"].lower(), r["artist"].lower())
        for r in con.execute("SELECT title, artist FROM tracks").fetchall()
    }


def generate(n_seed_artists=6, per_artist=4):
    taste = _taste_vector()
    with connect() as con:
        seeds = [r["artist"] for r in con.execute(
            """SELECT t.artist, COUNT(*) n FROM plays p
               JOIN tracks t ON t.id = p.track_id
               GROUP BY t.artist ORDER BY n DESC LIMIT ?""",
            (n_seed_artists,),
        ).fetchall()]
        if not seeds:
            seeds = [r["name"] for r in con.execute(
                """SELECT name FROM top_items WHERE item_type='artist'
                   AND time_range='medium_term' ORDER BY rank LIMIT ?""",
                (n_seed_artists,),
            ).fetchall()]
        known = _known_pairs(con)

    if not seeds:
        return {"error": "No listening data yet — sync or import a source first."}

    candidates = {}
    for seed in seeds:
        try:
            related = deezer.artist_related(seed, limit=4)
        except Exception:
            continue
        for depth, rel in enumerate(related):
            try:
                tops = deezer.artist_top_tracks(rel["id"], limit=per_artist)
            except Exception:
                continue
            for rank, tr in enumerate(tops):
                key = (tr["title"].lower(), rel["name"].lower())
                if key in known or key in candidates:
                    continue
                affinity = 1.0 / (1 + depth) * 1.0 / (1 + 0.3 * rank)
                candidates[key] = {
                    "title": tr["title"],
                    "artist": rel["name"],
                    "deezer_id": tr["id"],
                    "preview_url": tr.get("preview") or None,
                    "reason": f"Because you listen to {seed}",
                    "affinity": affinity,
                }
            time.sleep(0.12)
        time.sleep(0.12)

    ranked = list(candidates.values())

    # feature-similarity re-ranking on the top candidates by affinity
    if taste is not None:
        from ..audio.features import analyze_preview
        ranked.sort(key=lambda c: -c["affinity"])
        for c in ranked[:20]:
            if not c["preview_url"]:
                c["similarity"] = None
                continue
            try:
                feats = analyze_preview(c["preview_url"], f"cand_{c['deezer_id']}")
                vec = np.array([feats[col] for col in FEATURE_COLS])
                dist = float(np.linalg.norm(vec - taste))
                c["similarity"] = round(1.0 / (1.0 + dist), 3)
                c["features"] = {k: feats[k] for k in
                                 ("bpm", "key", "mode", *FEATURE_COLS)}
            except Exception:
                c["similarity"] = None
        for c in ranked:
            sim = c.get("similarity")
            c["score"] = round(0.5 * c["affinity"] + 0.5 * (sim if sim else 0.4), 3)
    else:
        for c in ranked:
            c["score"] = round(c["affinity"], 3)

    ranked.sort(key=lambda c: -c["score"])
    ranked = ranked[:30]

    with connect() as con:
        con.execute("DELETE FROM recommendations")
        for c in ranked:
            con.execute(
                """INSERT OR IGNORE INTO recommendations
                   (title, artist, deezer_id, preview_url, reason, score)
                   VALUES (?,?,?,?,?,?)""",
                (c["title"], c["artist"], c["deezer_id"],
                 c["preview_url"], c["reason"], c["score"]),
            )
    return {"count": len(ranked), "recommendations": ranked}


def latest():
    with connect() as con:
        rows = con.execute(
            "SELECT * FROM recommendations ORDER BY score DESC"
        ).fetchall()
    return [dict(r) for r in rows]
