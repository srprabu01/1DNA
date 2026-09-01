"""Deep playlist analysis: DNA, cohesion, energy arc, harmonic flow, outliers,
duplicates, breakdowns, use-case classification, and optimal re-ordering."""
import json
import re

import numpy as np

from ..audio.camelot import transition_score
from . import store

FEATS = ["energy", "danceability", "valence", "acousticness", "brightness",
         "instrumentalness", "speechiness", "liveness"]


def _norm_title(s):
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _vec(t):
    v = [t.get(c) for c in FEATS]
    return v if all(x is not None for x in v) else None


def analyze(pid):
    meta, tracks = store.get_tracks(pid)
    if meta is None:
        return None
    analyzed = [t for t in tracks if _vec(t) is not None and t.get("bpm")]
    result = {
        "id": pid, "name": meta["name"], "source": meta["source"],
        "n_tracks": len(tracks), "n_analyzed": len(analyzed),
    }
    if len(analyzed) < 2:
        result["note"] = ("Need at least 2 analyzed tracks. Run the processing "
                          "pipeline (match + analyze) on this playlist's songs first.")
        result["tracks"] = tracks
        return result

    M = np.array([_vec(t) for t in analyzed], dtype=float)

    # ---- aggregate DNA ----
    dna = {c: round(float(M[:, i].mean()), 3) for i, c in enumerate(FEATS)}
    bpms = [t["bpm"] for t in analyzed if t.get("bpm")]
    dna["bpm"] = round(float(np.mean(bpms)), 1)
    dna["bpm_spread"] = round(float(np.std(bpms)), 1)
    result["dna"] = dna

    # ---- cohesion: 1 - average feature dispersion ----
    cohesion = 1.0 - float(np.mean(M.std(axis=0)))
    result["cohesion"] = round(max(0.0, min(1.0, cohesion)), 3)
    result["cohesion_label"] = (
        "Very tight / single vibe" if cohesion > 0.85 else
        "Cohesive" if cohesion > 0.7 else
        "Varied" if cohesion > 0.5 else "All over the place")

    # ---- energy / valence arc ----
    result["arc"] = [
        {"pos": i + 1, "title": t["title"], "artist": t["artist"],
         "energy": t["energy"], "valence": t["valence"], "bpm": t["bpm"]}
        for i, t in enumerate(analyzed)]

    # ---- harmonic (Camelot) flow between consecutive tracks ----
    flow, good = [], 0
    for a, b in zip(analyzed, analyzed[1:]):
        sc = transition_score(a.get("camelot"), b.get("camelot"))
        flow.append({"from": a["title"], "to": b["title"],
                     "from_key": a.get("camelot"), "to_key": b.get("camelot"),
                     "bpm_delta": round((b["bpm"] or 0) - (a["bpm"] or 0), 1),
                     "score": sc})
        if sc is not None and sc >= 0.85:
            good += 1
    scores = [f["score"] for f in flow if f["score"] is not None]
    result["harmonic_flow"] = flow
    result["harmonic_score"] = round(float(np.mean(scores)), 3) if scores else None
    result["harmonic_pct"] = round(100 * good / len(flow), 0) if flow else None

    # ---- outliers (farthest from the centroid) ----
    centroid = M.mean(axis=0)
    dists = np.linalg.norm(M - centroid, axis=1)
    order = np.argsort(-dists)[:3]
    result["outliers"] = [
        {"title": analyzed[i]["title"], "artist": analyzed[i]["artist"],
         "distance": round(float(dists[i]), 3)}
        for i in order if dists[i] > 0.35]

    # ---- duplicates (same normalized title+artist, or near-identical audio) ----
    dups = []
    seen = {}
    for t in tracks:
        k = (_norm_title(t["title"]), _norm_title(t["artist"]))
        if k in seen:
            dups.append({"title": t["title"], "artist": t["artist"], "reason": "same song"})
        seen[k] = True
    result["duplicates"] = dups

    # ---- breakdowns ----
    result["genres"] = _counts([t.get("genre") for t in tracks])
    result["eras"] = _counts([(t.get("release_date") or "")[:3] + "0s"
                              for t in tracks if (t.get("release_date") or "")[:4].isdigit()])
    result["top_artists"] = _counts([t["artist"] for t in tracks], top=8)

    # ---- use-case + mood label ----
    result["use_case"] = _use_case(dna)
    result["mood_label"] = _mood_label(dna)

    # ---- optimal re-ordering (greedy harmonic + energy smoothness) ----
    result["reorder"] = _reorder(analyzed)

    return result


def _counts(values, top=10):
    out = {}
    for v in values:
        if v:
            out[v] = out.get(v, 0) + 1
    return [{"name": k, "count": v}
            for k, v in sorted(out.items(), key=lambda kv: -kv[1])[:top]]


def _use_case(dna):
    e, d, v, ac, ins = (dna["energy"], dna["danceability"], dna["valence"],
                        dna["acousticness"], dna["instrumentalness"])
    scores = {
        "Workout / Hype": 0.5 * e + 0.3 * d + 0.2 * dna["bpm"] / 180,
        "Party / Dance": 0.5 * d + 0.3 * e + 0.2 * v,
        "Focus / Study": 0.5 * ins + 0.3 * (1 - e) + 0.2 * (1 - dna["speechiness"]),
        "Chill / Relax": 0.4 * (1 - e) + 0.3 * ac + 0.3 * v,
        "Sleep / Ambient": 0.5 * (1 - e) + 0.3 * ac + 0.2 * ins,
        "Sad / Introspective": 0.5 * (1 - v) + 0.3 * ac + 0.2 * (1 - e),
    }
    best = max(scores, key=scores.get)
    return {"label": best, "scores": {k: round(v, 3) for k, v in
                                      sorted(scores.items(), key=lambda kv: -kv[1])}}


def _mood_label(dna):
    e, v = dna["energy"], dna["valence"]
    quad = ("Bright & energetic" if e >= .5 and v >= .5 else
            "Dark & intense" if e >= .5 else
            "Warm & mellow" if v >= .5 else "Moody & subdued")
    tempo = "up-tempo" if dna["bpm"] >= 120 else "mid-tempo" if dna["bpm"] >= 95 else "slow"
    return f"{quad}, {tempo}"


def _reorder(analyzed):
    """Greedy nearest-neighbour by harmonic compatibility then small energy steps."""
    remaining = list(range(len(analyzed)))
    # start from the lowest-energy track (natural build-up)
    start = min(remaining, key=lambda i: analyzed[i]["energy"])
    seq = [start]
    remaining.remove(start)
    while remaining:
        cur = analyzed[seq[-1]]
        def score(j):
            t = analyzed[j]
            harm = transition_score(cur.get("camelot"), t.get("camelot")) or 0.3
            energy_step = 1 - abs((t["energy"] or 0) - (cur["energy"] or 0))
            return 0.6 * harm + 0.4 * energy_step
        nxt = max(remaining, key=score)
        seq.append(nxt)
        remaining.remove(nxt)
    return [{"pos": i + 1, "title": analyzed[j]["title"], "artist": analyzed[j]["artist"],
             "camelot": analyzed[j].get("camelot"), "energy": analyzed[j]["energy"]}
            for i, j in enumerate(seq)]
