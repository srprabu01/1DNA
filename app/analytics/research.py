"""Deep Research engine.

Library-wide statistical study of your analyzed music:
  - PCA 2-D "library map" (every track positioned in sonic space)
  - artist sonic signatures (what makes each artist distinctive vs your baseline)
  - track-to-track similarity network (MFCC timbre + feature distance)
  - feature statistics, era drift, loudness-war audit
  - auto-written prose findings

Per-track: a research dossier with library percentiles and nearest sonic siblings.
"""
import json

import numpy as np

from ..db import connect, rows_to_dicts

FEATS = ["energy", "danceability", "valence", "acousticness", "brightness",
         "instrumentalness", "speechiness", "liveness"]


def _library(min_feats=True):
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            """SELECT t.id, t.title, t.artist, t.genre, t.release_date, t.popularity,
                      f.bpm, f.key, f.mode, f.camelot, f.loudness_db, f.dynamic_range,
                      f.harmonic_ratio, f.beat_regularity, f.onset_rate, f.mfcc,
                      f.energy, f.danceability, f.valence, f.acousticness, f.brightness,
                      f.instrumentalness, f.speechiness, f.liveness,
                      COALESCE(p.n,0) plays
               FROM tracks t JOIN audio_features f ON f.track_id=t.id
               LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                 ON p.track_id=t.id
               WHERE f.error IS NULL""").fetchall())
    if min_feats:
        rows = [r for r in rows if all(r[c] is not None for c in FEATS)]
    return rows


def _matrix(rows, cols=FEATS):
    return np.array([[r[c] for c in cols] for r in rows], dtype=float)


def _mfcc_matrix(rows):
    out = []
    for r in rows:
        try:
            v = json.loads(r["mfcc"] or "[]")
            out.append(v if len(v) == 13 else [0.0] * 13)
        except Exception:
            out.append([0.0] * 13)
    return np.array(out, dtype=float)


def _distance_matrix(rows):
    """Combined sonic distance: standardized features + standardized MFCC timbre."""
    X = _matrix(rows)
    M = _mfcc_matrix(rows)
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    Ms = (M - M.mean(axis=0)) / (M.std(axis=0) + 1e-9)
    combo = np.hstack([Xs, 0.5 * Ms])
    diff = combo[:, None, :] - combo[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2))


# ---------- per-track dossier ----------

def track_dossier(track_id):
    rows = _library()
    idx = next((i for i, r in enumerate(rows) if r["id"] == track_id), None)
    if idx is None:
        return {"available": False,
                "note": "Track not analyzed yet — run the pipeline first."}
    me = rows[idx]

    # percentiles vs library
    percentiles = {}
    extra = [("bpm", "tempo"), ("loudness_db", "loudness"), ("dynamic_range", "dynamics")]
    for col in FEATS + [e[0] for e in extra]:
        vals = np.array([r[col] for r in rows if r[col] is not None], dtype=float)
        if me[col] is not None and len(vals) > 1:
            percentiles[col] = round(float((vals < me[col]).mean() * 100), 0)

    # nearest sonic siblings
    similar = []
    if len(rows) > 2:
        D = _distance_matrix(rows)
        order = np.argsort(D[idx])
        dmax = float(D[idx].max()) or 1.0
        for j in order[1:6]:
            r = rows[j]
            similar.append({"id": r["id"], "title": r["title"], "artist": r["artist"],
                            "similarity": round(1 - float(D[idx][j]) / dmax, 3)})

    # structure (computed live from the cached preview)
    try:
        from ..audio.structure import analyze_structure
        structure = analyze_structure(track_id)
    except Exception as e:
        structure = {"error": str(e)[:120]}

    return {"available": True, "percentiles": percentiles,
            "similar": similar, "structure": structure,
            "library_size": len(rows)}


# ---------- library-wide deep research ----------

def deep_research():
    rows = _library()
    if len(rows) < 5:
        return {"available": False,
                "note": "Need at least 5 fully analyzed tracks — run the pipeline first."}

    X = _matrix(rows)
    findings = []

    # --- feature statistics table ---
    stats = {}
    for i, c in enumerate(FEATS):
        col = X[:, i]
        stats[c] = {"mean": round(float(col.mean()), 3), "std": round(float(col.std()), 3),
                    "min": round(float(col.min()), 3), "q25": round(float(np.percentile(col, 25)), 3),
                    "median": round(float(np.median(col)), 3),
                    "q75": round(float(np.percentile(col, 75)), 3),
                    "max": round(float(col.max()), 3)}

    # --- PCA library map ---
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    U, S, Vt = np.linalg.svd(Xs, full_matrices=False)
    coords = U[:, :2] * S[:2]
    var_explained = (S ** 2 / (S ** 2).sum())[:2]
    # name the axes by their strongest feature loadings
    def axis_name(v):
        top = np.argsort(-np.abs(v))[:2]
        return " / ".join(f"{'+' if v[t] > 0 else '−'}{FEATS[t]}" for t in top)
    pca_map = {
        "x_label": axis_name(Vt[0]), "y_label": axis_name(Vt[1]),
        "var_explained": [round(float(v) * 100, 1) for v in var_explained],
        "points": [{"id": r["id"], "title": r["title"], "artist": r["artist"],
                    "genre": r["genre"], "plays": r["plays"],
                    "x": round(float(coords[i, 0]), 3), "y": round(float(coords[i, 1]), 3)}
                   for i, r in enumerate(rows)],
    }
    findings.append(
        f"Two latent dimensions explain {round(float(var_explained.sum())*100)}% of the "
        f"sonic variance in your library. Axis 1 ({axis_name(Vt[0])}) separates your music "
        f"most strongly — it's the main fault line in your taste.")

    # --- artist sonic signatures ---
    lib_mean, lib_std = X.mean(axis=0), X.std(axis=0) + 1e-9
    by_artist = {}
    for i, r in enumerate(rows):
        by_artist.setdefault(r["artist"], []).append(i)
    signatures = []
    for artist, idxs in by_artist.items():
        if len(idxs) < 1:
            continue
        mean_vec = X[idxs].mean(axis=0)
        z = (mean_vec - lib_mean) / lib_std
        top = int(np.argmax(np.abs(z)))
        signatures.append({
            "artist": artist, "n_tracks": len(idxs),
            "plays": int(sum(rows[i]["plays"] for i in idxs)),
            "profile": {FEATS[j]: round(float(mean_vec[j]), 3) for j in range(len(FEATS))},
            "distinctive": FEATS[top],
            "distinctive_z": round(float(z[top]), 2),
            "direction": "higher" if z[top] > 0 else "lower",
        })
    signatures.sort(key=lambda s: -s["plays"])
    sig_top = [s for s in signatures if abs(s["distinctive_z"]) > 1.2][:3]
    for s in sig_top:
        findings.append(
            f"{s['artist']}'s sonic signature: markedly {s['direction']} "
            f"{s['distinctive']} than your library baseline (z = {s['distinctive_z']}).")

    # --- similarity network ---
    D = _distance_matrix(rows)
    edges = []
    for i in range(len(rows)):
        for j in np.argsort(D[i])[1:3]:      # 2 nearest neighbours each
            j = int(j)
            if i < j:
                edges.append({"a": rows[i]["id"], "b": rows[j]["id"],
                              "w": round(float(D[i][j]), 3)})
    edges.sort(key=lambda e: e["w"])
    edges = edges[:80]
    # most "connected" (central) track = smallest mean distance to everything
    central = int(np.argmin(D.mean(axis=1)))
    findings.append(
        f"“{rows[central]['title']}” by {rows[central]['artist']} is the sonic centre of "
        f"your library — the single track most representative of everything you play.")
    lonely = int(np.argmax(D.mean(axis=1)))
    findings.append(
        f"Your biggest outlier is “{rows[lonely]['title']}” by {rows[lonely]['artist']} — "
        f"nothing else you listen to sounds like it.")

    # --- correlations (top 5 strongest only, to keep findings sharp) ---
    corr = np.corrcoef(Xs.T)
    pairs = [(abs(float(corr[i, j])), float(corr[i, j]), i, j)
             for i in range(len(FEATS)) for j in range(i + 1, len(FEATS))
             if abs(corr[i, j]) >= 0.6]
    for _, c, i, j in sorted(pairs, reverse=True)[:5]:
        rel = "rise and fall together" if c > 0 else "are opposed"
        findings.append(f"In your taste, {FEATS[i]} and {FEATS[j]} {rel} (r = {round(c, 2)}).")

    # --- era drift ---
    eras = {}
    for i, r in enumerate(rows):
        y = (r["release_date"] or "")[:4]
        if y.isdigit():
            eras.setdefault(y[:3] + "0s", []).append(i)
    era_drift = [{"era": e, "n": len(ix),
                  "energy": round(float(X[ix, 0].mean()), 3),
                  "valence": round(float(X[ix, 2].mean()), 3),
                  "loudness": round(float(np.mean([rows[i]["loudness_db"] for i in ix
                                                   if rows[i]["loudness_db"] is not None])), 1)}
                 for e, ix in sorted(eras.items()) if len(ix) >= 2]
    if len(era_drift) >= 2:
        l0, l1 = era_drift[0], era_drift[-1]
        if l1["loudness"] - l0["loudness"] > 2:
            findings.append(
                f"The loudness war is visible in your library: your {l1['era']} tracks average "
                f"{l1['loudness']} dB vs {l0['loudness']} dB in the {l0['era']} — "
                f"{round(l1['loudness']-l0['loudness'],1)} dB louder, typically at the cost of dynamics.")

    # --- loudness-war audit (compression score) ---
    dr = [(r["dynamic_range"], r["title"], r["artist"]) for r in rows
          if r["dynamic_range"] is not None]
    if dr:
        dr.sort()
        most_crushed = dr[0]
        most_dynamic = dr[-1]
        findings.append(
            f"Most compressed master you play: “{most_crushed[1]}” ({most_crushed[0]} dB of "
            f"dynamic range). Most dynamic: “{most_dynamic[1]}” ({most_dynamic[0]} dB).")

    # --- tempo modes ---
    bpms = np.array([r["bpm"] for r in rows if r["bpm"]])
    if len(bpms) > 4:
        hist, edges_ = np.histogram(bpms, bins=8)
        peak = int(np.argmax(hist))
        findings.append(
            f"Your tempo sweet spot is {int(edges_[peak])}–{int(edges_[peak+1])} BPM "
            f"({hist[peak]} of {len(bpms)} analyzed tracks live there).")

    return {
        "available": True,
        "n_tracks": len(rows),
        "findings": findings,
        "stats": stats,
        "pca_map": pca_map,
        "signatures": signatures[:12],
        "network": {"edges": edges,
                    "nodes": [{"id": r["id"], "title": r["title"], "artist": r["artist"],
                               "genre": r["genre"], "plays": r["plays"]} for r in rows]},
        "era_drift": era_drift,
    }
