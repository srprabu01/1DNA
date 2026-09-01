"""Per-track feature vectors — the shared space that debiasing, taste probes,
and similarity all operate over.

Assembles the analyzed audio features (the 0-1 estimates + normalized tempo /
loudness / dynamics + the 13-band MFCC timbre) into one standardized matrix.
This is the app's equivalent of deepcut's embedding matrix — lower-dimensional
(no GPU/foundation model), but the debias / taste / retrieval math is identical.
"""
import json

import numpy as np

from ..db import connect, rows_to_dicts

# 0-1 features that already share a scale
UNIT_COLS = ["energy", "danceability", "valence", "acousticness", "brightness",
             "instrumentalness", "speechiness", "liveness", "harmonic_ratio",
             "beat_regularity", "tempo_confidence", "key_confidence"]

# human-readable names for every dimension (for taste-driver explanations)
DIM_NAMES = UNIT_COLS + ["tempo", "loudness", "dynamics"] + [f"mfcc{i+1}" for i in range(13)]


def load_matrix(with_mfcc: bool = True):
    """Returns (ids, X_standardized, plays, mean, std). One row per analyzed track."""
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            f"""SELECT f.track_id, {', '.join('f.' + c for c in UNIT_COLS)},
                       f.bpm, f.loudness_db, f.dynamic_range, f.mfcc,
                       COALESCE(p.n, 0) plays
                FROM audio_features f
                LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                  ON p.track_id = f.track_id
                WHERE f.error IS NULL AND f.energy IS NOT NULL""").fetchall())

    ids, feats, plays = [], [], []
    for r in rows:
        base = [r[c] for c in UNIT_COLS]
        if any(v is None for v in base):
            continue
        extra = [(r["bpm"] or 120) / 200.0,
                 ((r["loudness_db"] if r["loudness_db"] is not None else -12) + 35) / 35.0,
                 (r["dynamic_range"] or 8) / 30.0]
        vec = base + extra
        if with_mfcc:
            try:
                mf = json.loads(r["mfcc"] or "[]")
            except Exception:
                mf = []
            vec += list(mf) if len(mf) == 13 else [0.0] * 13
        ids.append(r["track_id"])
        feats.append(vec)
        plays.append(r["plays"])

    if not feats:
        return np.array([]), np.zeros((0, 0)), np.array([]), None, None
    X = np.array(feats, dtype=np.float32)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    return np.array(ids), (X - mean) / std, np.array(plays, dtype=np.float32), mean, std


def load_best():
    """Prefer CLAP embeddings (semantic, high-dim) when they exist; else the DSP
    feature matrix. Returns (ids, X, plays, source)."""
    try:
        from ..audio import embed
        cur = embed.clap_matrix()
    except Exception:
        cur = None
    if cur is not None and len(cur[0]) >= 20:
        ids, X = cur
        meta = track_meta(ids)
        plays = np.array([meta.get(int(i), {}).get("plays", 0) for i in ids],
                         dtype=np.float32)
        return ids, X.astype(np.float32), plays, "embedding"
    ids, X, plays, _, _ = load_matrix()
    return ids, X, plays, "dsp"


def track_meta(ids):
    """title/artist/genre/plays for a list of track ids, keyed by id."""
    if len(ids) == 0:
        return {}
    q = ",".join("?" * len(ids))
    with connect() as con:
        rows = con.execute(
            f"""SELECT t.id, t.title, t.artist, t.genre, t.preview_url,
                       COALESCE(p.n,0) plays
                FROM tracks t
                LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                  ON p.track_id = t.id
                WHERE t.id IN ({q})""", [int(i) for i in ids]).fetchall()
    return {r["id"]: dict(r) for r in rows}
