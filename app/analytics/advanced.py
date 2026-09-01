"""Phase 3 — advanced macro analytics: mood-over-time, listening calendar,
discovery & concentration metrics, and k-means clustering into mood groups."""
import math

import numpy as np

from ..db import connect, rows_to_dicts

CLUSTER_FEATS = ["energy", "danceability", "valence", "acousticness",
                 "brightness", "instrumentalness"]


def mood_over_time():
    """Play-weighted monthly averages of valence / energy / tempo."""
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            """SELECT strftime('%Y-%m', p.played_at) month,
                      AVG(f.valence) valence, AVG(f.energy) energy,
                      AVG(f.bpm) bpm, COUNT(*) plays
               FROM plays p JOIN audio_features f ON f.track_id = p.track_id
               WHERE f.error IS NULL AND f.valence IS NOT NULL
               GROUP BY month ORDER BY month""").fetchall())
    for r in rows:
        for k in ("valence", "energy"):
            r[k] = round(r[k], 3) if r[k] is not None else None
        r["bpm"] = round(r["bpm"], 1) if r["bpm"] is not None else None
    return rows


def calendar():
    """Daily play counts (GitHub-style heatmap) + summary streak stats."""
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            """SELECT date(played_at) day, COUNT(*) plays
               FROM plays GROUP BY day ORDER BY day""").fetchall())
    days = {r["day"]: r["plays"] for r in rows if r["day"]}
    busiest = max(rows, key=lambda r: r["plays"]) if rows else None
    return {"days": rows, "active_days": len(days),
            "busiest_day": busiest, "max_in_day": busiest["plays"] if busiest else 0}


def discovery():
    """New-artist / new-track first-appearances per month + discovery ratio."""
    with connect() as con:
        artist_first = rows_to_dicts(con.execute(
            """SELECT strftime('%Y-%m', first) month, COUNT(*) new_artists FROM (
                   SELECT t.artist, MIN(p.played_at) first
                   FROM plays p JOIN tracks t ON t.id=p.track_id
                   GROUP BY t.artist) GROUP BY month ORDER BY month""").fetchall())
        track_first = rows_to_dicts(con.execute(
            """SELECT strftime('%Y-%m', first) month, COUNT(*) new_tracks FROM (
                   SELECT p.track_id, MIN(p.played_at) first
                   FROM plays p GROUP BY p.track_id) GROUP BY month ORDER BY month""").fetchall())
        totals = con.execute(
            "SELECT COUNT(*) plays, COUNT(DISTINCT track_id) tracks FROM plays").fetchone()
    disc_ratio = (totals["tracks"] / totals["plays"]) if totals["plays"] else None
    new_by_month = {r["month"]: r["new_artists"] for r in artist_first}
    trk_by_month = {r["month"]: r["new_tracks"] for r in track_first}
    months = sorted(set(new_by_month) | set(trk_by_month))
    series = [{"month": m, "new_artists": new_by_month.get(m, 0),
               "new_tracks": trk_by_month.get(m, 0)} for m in months]
    return {"series": series,
            "discovery_ratio": round(disc_ratio, 3) if disc_ratio else None,
            "explorer_label": _explorer_label(disc_ratio)}


def _explorer_label(ratio):
    if ratio is None:
        return None
    if ratio > 0.8:
        return "Explorer — you rarely repeat, always seeking new"
    if ratio > 0.5:
        return "Balanced — mix of new and familiar"
    if ratio > 0.3:
        return "Comfort listener — you replay favourites a lot"
    return "Loyalist — you loop a tight set of songs"


def concentration():
    """How concentrated your listening is: Gini, HHI, and top-N share of artists."""
    with connect() as con:
        counts = [r["n"] for r in con.execute(
            """SELECT COUNT(*) n FROM plays p JOIN tracks t ON t.id=p.track_id
               GROUP BY t.artist ORDER BY n DESC""").fetchall()]
    if not counts:
        return {"available": False}
    arr = np.array(counts, dtype=float)
    total = arr.sum()
    shares = arr / total
    hhi = float(np.sum(shares ** 2))                       # Herfindahl index
    # Gini coefficient
    s = np.sort(arr)
    n = len(s)
    gini = float((2 * np.sum((np.arange(1, n + 1)) * s) / (n * s.sum())) - (n + 1) / n) if n > 1 else 0.0
    top = np.cumsum(shares)
    return {
        "available": True, "n_artists": n,
        "gini": round(gini, 3), "hhi": round(hhi, 4),
        "top1_pct": round(100 * float(shares[0]), 1),
        "top5_pct": round(100 * float(top[min(4, n - 1)]), 1),
        "top10_pct": round(100 * float(top[min(9, n - 1)]), 1),
        "label": ("Very concentrated — a few artists dominate" if gini > 0.6 else
                  "Concentrated" if gini > 0.45 else
                  "Fairly even" if gini > 0.3 else "Very diverse spread"),
    }


def _kmeans(X, k, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            if (labels == j).any():
                centroids[j] = X[labels == j].mean(axis=0)
    return labels, centroids


def clusters(k=4):
    """Cluster analyzed library into mood groups (smart playlists)."""
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            f"""SELECT t.id, t.title, t.artist,
                       {', '.join('f.' + c for c in CLUSTER_FEATS)}, f.bpm
                FROM tracks t JOIN audio_features f ON f.track_id=t.id
                WHERE f.error IS NULL AND f.energy IS NOT NULL""").fetchall())
    usable = [r for r in rows if all(r[c] is not None for c in CLUSTER_FEATS)]
    if len(usable) < k:
        return {"available": False, "n": len(usable),
                "note": f"Need at least {k} analyzed tracks to form {k} clusters."}
    X = np.array([[r[c] for c in CLUSTER_FEATS] for r in usable], dtype=float)
    labels, centroids = _kmeans(X, k)
    out = []
    for j in range(k):
        members = [usable[i] for i in range(len(usable)) if labels[i] == j]
        if not members:
            continue
        c = centroids[j]
        profile = {CLUSTER_FEATS[m]: round(float(c[m]), 3) for m in range(len(CLUSTER_FEATS))}
        avg_bpm = (round(float(np.mean([m["bpm"] for m in members if m["bpm"]])), 1)
                   if any(m["bpm"] for m in members) else None)
        out.append({
            "id": j, "size": len(members),
            "label": _cluster_label(profile, avg_bpm),
            "profile": profile, "avg_bpm": avg_bpm,
            "samples": [{"id": m["id"], "title": m["title"], "artist": m["artist"]}
                        for m in members[:8]],
        })
    out.sort(key=lambda c: -c["size"])
    # de-duplicate identical labels by appending a distinguishing tempo tag
    seen = {}
    for c in out:
        if c["label"] in seen and c["avg_bpm"]:
            c["label"] += f" · {int(c['avg_bpm'])} BPM"
        seen[c["label"]] = True
    return {"available": True, "k": k, "n": len(usable), "clusters": out}


def _cluster_label(p, bpm=None):
    e, v, ac, ins, d = (p["energy"], p["valence"], p["acousticness"],
                        p["instrumentalness"], p["danceability"])
    fast = bpm is not None and bpm >= 130
    slow = bpm is not None and bpm < 95
    if ins > 0.6 and e < 0.5:
        return "Instrumental / ambient"
    if ac > 0.55 and e < 0.55:
        return "Acoustic / mellow" if not slow else "Slow & acoustic"
    if e >= 0.6 and v >= 0.55:
        if fast and d >= 0.55:
            return "Dancefloor bangers"
        return "Feel-good & fast" if fast else "Upbeat & bright"
    if e >= 0.6:
        return "Dark & driving" if fast else "Dark & energetic"
    if v < 0.45:
        return "Moody / introspective"
    if slow:
        return "Slow & warm"
    return "Chill / easy"
