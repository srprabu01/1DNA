"""Your taste, as a linear probe over the feature space — ported from deepcut.

deepcut fits a logistic regression over frozen embeddings from a few hundred
explicit clicks. We don't have clicks, but we have years of play counts, so the
labels are IMPLICIT: your heavily-played tracks are positives, your rarely-played
ones are negatives. The probe learns what separates your rotation from the rest,
which gives (a) a personal "for you" score for every track and (b) the signed
feature drivers of your taste.
"""
import numpy as np

from . import vectors
from .space import SpaceCorrection


def _fit_logistic(X, y, w, l2=1.0, iters=500, lr=0.5):
    n, d = X.shape
    W = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(X @ W + b)))
        g = (p - y) * w
        W -= lr * ((X.T @ g) / n + l2 * W / n)
        b -= lr * (g.sum() / n)
    return W, b


def _auc(y, s):
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def fit(min_tracks: int = 40):
    """Fit the taste probe. Returns a summary dict (weights, quality, top picks)."""
    ids, X, plays, source = vectors.load_best()
    if len(ids) < min_tracks:
        return {"available": False,
                "note": f"Need at least {min_tracks} analyzed tracks (have {len(ids)}). "
                        "Run the processing pipeline on more of your library."}

    # implicit labels from play counts: top third = loved, bottom third = not
    order = np.argsort(plays)
    n = len(ids)
    third = max(1, n // 3)
    neg_idx = order[:third]
    pos_idx = order[-third:]
    if plays[pos_idx].max() <= plays[neg_idx].max():
        return {"available": False, "note": "Not enough play-count spread to learn taste yet."}

    rows = np.concatenate([pos_idx, neg_idx])
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))])
    # weight positives by how heavily played (log), so your anthems count most
    w = np.concatenate([1.0 + np.log1p(plays[pos_idx]) / 3.0, np.ones(len(neg_idx))])

    # held-out AUC via a simple split
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(rows))
    cut = int(len(rows) * 0.75)
    tr, te = perm[:cut], perm[cut:]
    W, b = _fit_logistic(X[rows][tr], y[tr], w[tr])
    auc = _auc(y[te], X[rows][te] @ W + b)

    # refit on all labels for the deployed model
    W, b = _fit_logistic(X[rows], y, w)

    scores = 1.0 / (1.0 + np.exp(-(X @ W + b)))
    _save(W, b)

    # feature drivers (signed) — only interpretable for the named DSP dims
    if source == "dsp" and len(W) == len(vectors.DIM_NAMES):
        drivers = sorted(zip(vectors.DIM_NAMES, W.tolist()), key=lambda kv: -abs(kv[1]))
    else:
        drivers = []
    meta = vectors.track_meta(ids)

    def pack(i):
        m = meta.get(int(ids[i]), {})
        return {"id": int(ids[i]), "title": m.get("title"), "artist": m.get("artist"),
                "genre": m.get("genre"), "plays": int(plays[i]),
                "taste": round(float(scores[i]), 3)}

    top = [pack(i) for i in np.argsort(-scores)[:25]]
    # "hidden gems" = high taste score but you've barely played them
    gem_order = [i for i in np.argsort(-scores) if plays[i] <= 2][:15]
    gems = [pack(i) for i in gem_order]

    return {
        "available": True, "n_tracks": int(n), "space": source,
        "auc": round(auc, 3) if auc == auc else None,
        "drivers": [{"feature": f, "weight": round(wt, 3),
                     "direction": "you like more" if wt > 0 else "you like less"}
                    for f, wt in drivers[:10]],
        "top": top, "gems": gems,
    }


def taste_scores():
    """{track_id: taste_prob} using the saved probe, or None if unfit."""
    W, b = _load()
    if W is None:
        return None
    ids, X, _, _ = vectors.load_best()
    if len(ids) == 0 or X.shape[1] != len(W):   # probe trained on a different space
        return {}
    s = 1.0 / (1.0 + np.exp(-(X @ W + b)))
    return {int(i): float(v) for i, v in zip(ids, s)}


# ---- persistence ----
from ..config import DATA_DIR
_PATH = DATA_DIR / "taste_probe.npz"


def _save(W, b):
    np.savez(_PATH, W=W, b=np.array([b]))


def _load():
    if not _PATH.exists():
        return None, None
    d = np.load(_PATH)
    return d["W"].astype(np.float32), float(d["b"][0])


def fitted():
    return _PATH.exists()
