"""Debiased similar-track retrieval + the popularity-leakage diagnostic.

Uses the corrected feature space (popularity/anisotropy/hubness removed) so
"similar" means genuinely similar-sounding, not "also popular". Adds MMR so the
list is diverse rather than ten near-duplicates, and caps repeats per artist.
"""
import numpy as np

from . import vectors
from .space import SpaceCorrection, popularity_leakage


def _corrected():
    ids, X, plays, source = vectors.load_best()
    if len(ids) < 5:
        return None
    # High-dim (embedding) spaces get the full deepcut treatment; the low-dim
    # DSP feature space gets popularity-debiasing only (anisotropy/CSLS hurt it).
    high = source == "embedding"
    corr = SpaceCorrection.fit(X, popularity=plays,
                               n_components=2 if high else 0,
                               csls_k=10 if high else 0)
    return ids, corr.apply(X), plays, corr


def diagnostics():
    ids, X, plays, _, _ = vectors.load_matrix()
    if len(ids) < 8:
        return {"available": False}
    before = popularity_leakage(X, plays)
    corr = SpaceCorrection.fit(X, popularity=plays, n_components=1)
    after = popularity_leakage(corr.apply(X), plays)
    return {"available": True, "n_tracks": int(len(ids)),
            "popularity_r2_before": round(before, 3),
            "popularity_r2_after": round(after, 3),
            "verdict": ("Your raw 'similar' list was substantially a popularity chart"
                        if before > 0.15 else
                        "Popularity bias was modest") +
                       f" — debiasing cut it from {round(before*100)}% to {round(after*100)}%."}


def similar_to(track_id: int, k: int = 12, mmr: float = 0.6, artist_cap: int = 2):
    got = _corrected()
    if got is None:
        return {"available": False, "note": "Need more analyzed tracks."}
    ids, C, plays, corr = got
    pos = {int(t): i for i, t in enumerate(ids)}
    if track_id not in pos:
        return {"available": False, "note": "Track not analyzed yet."}
    qi = pos[track_id]
    sims = C @ C[qi]
    sims = corr.csls(sims, np.arange(len(ids)))
    sims[qi] = -np.inf

    meta = vectors.track_meta(ids)
    seed_artist = (meta.get(track_id, {}).get("artist") or "").lower()

    # candidate pool by raw corrected similarity, then MMR for diversity
    pool = np.argsort(-sims)[:k * 8]
    picked, artist_n = [], {}
    while pool.size and len(picked) < k:
        if not picked:
            best = pool[0]
        else:
            chosen = np.array([p for p in picked])
            div = (C[pool] @ C[chosen].T).max(axis=1)
            mmr_score = mmr * sims[pool] - (1 - mmr) * div
            best = pool[int(np.argmax(mmr_score))]
        pool = pool[pool != best]
        m = meta.get(int(ids[best]), {})
        a = (m.get("artist") or "").lower()
        if a and a == seed_artist:                 # don't just return the same artist
            continue
        if artist_n.get(a, 0) >= artist_cap:
            continue
        artist_n[a] = artist_n.get(a, 0) + 1
        picked.append(best)

    out = []
    for i in picked:
        m = meta.get(int(ids[i]), {})
        out.append({"id": int(ids[i]), "title": m.get("title"), "artist": m.get("artist"),
                    "genre": m.get("genre"), "preview_url": m.get("preview_url"),
                    "similarity": round(float(sims[i]), 3), "plays": int(plays[i])})
    return {"available": True, "seed": meta.get(track_id, {}), "similar": out}
