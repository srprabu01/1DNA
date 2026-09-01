"""Your taste, as a linear probe over frozen vectors.

This is the payoff for never fine-tuning. Because the embeddings are fixed, a
personal preference model is a logistic regression with a few hundred
parameters that fits in under a second and can be thrown away and refit
whenever you change your mind. You can have many of them — one per axis, not
one global "like" score — and an axis can be anything you can label
consistently: *would I put this on at 2am*, *is the drummer human*, *too clean*.

Labels come from two places:

* **Explicit**, from ``deepcut label``. Around 200 examples is where an axis
  usually becomes useful; the active-learning picker gets you there faster by
  asking about the tracks it is least sure of.
* **Implicit**, from listening. Finished plays are weak positives, sub-30-second
  skips are weak negatives. Free, noisy, and worth roughly a tenth of an
  explicit label each — which is how they are weighted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TasteHead:
    axis: str
    space: str
    weights: np.ndarray
    bias: float
    cv_score: float
    n_labels: int
    kind: str = "binary"          # binary | continuous
    layer_profile: dict[str, float] | None = None

    def score(self, vector: np.ndarray) -> float:
        v = np.asarray(vector, dtype=np.float32).ravel()
        if v.shape != self.weights.shape:
            raise ValueError(f"{self.axis}: expected dim {self.weights.shape[0]}, got {v.shape[0]}")
        z = float(v @ self.weights + self.bias)
        return float(1 / (1 + np.exp(-z))) if self.kind == "binary" else z

    def score_matrix(self, matrix: np.ndarray) -> np.ndarray:
        z = np.asarray(matrix, dtype=np.float32) @ self.weights + self.bias
        return 1 / (1 + np.exp(-z)) if self.kind == "binary" else z

    def save(self, models_dir: Path) -> None:
        models_dir.mkdir(parents=True, exist_ok=True)
        np.save(models_dir / f"taste_{self.axis}.npy", self.weights)
        (models_dir / f"taste_{self.axis}.json").write_text(json.dumps({
            "axis": self.axis, "space": self.space, "bias": self.bias,
            "cv_score": self.cv_score, "n_labels": self.n_labels, "kind": self.kind,
            "layer_profile": self.layer_profile,
        }, indent=2))

    @classmethod
    def load(cls, models_dir: Path, axis: str) -> "TasteHead | None":
        w = models_dir / f"taste_{axis}.npy"
        m = models_dir / f"taste_{axis}.json"
        if not (w.exists() and m.exists()):
            return None
        meta = json.loads(m.read_text())
        return cls(axis=meta["axis"], space=meta["space"], weights=np.load(w),
                   bias=meta["bias"], cv_score=meta["cv_score"],
                   n_labels=meta["n_labels"], kind=meta.get("kind", "binary"),
                   layer_profile=meta.get("layer_profile"))

    @staticmethod
    def list_axes(models_dir: Path) -> list[str]:
        return sorted(p.stem.removeprefix("taste_") for p in models_dir.glob("taste_*.json"))


def implicit_labels(lib, completion_threshold: float = 0.8,
                    skip_seconds: float = 30.0) -> dict[int, tuple[float, float]]:
    """(label, weight) per track from listening behaviour.

    The 30-second skip is the industry's oldest behavioural signal and it is
    still the sharpest one available locally: it is a decision, not a rating.
    """
    out: dict[int, tuple[float, float]] = {}
    rows = lib.db.execute(
        "SELECT p.track_id, p.played_ms, p.skipped, t.duration_s"
        " FROM plays p JOIN tracks t ON t.id = p.track_id").fetchall()
    tally: dict[int, list[float]] = {}
    for r in rows:
        dur = float(r["duration_s"] or 0) * 1000
        played = float(r["played_ms"] or 0)
        if r["skipped"] and played < skip_seconds * 1000:
            tally.setdefault(int(r["track_id"]), []).append(0.0)
        elif dur > 0 and played / dur >= completion_threshold:
            tally.setdefault(int(r["track_id"]), []).append(1.0)
    for tid, vals in tally.items():
        out[tid] = (float(np.mean(vals)), min(len(vals), 5) * 0.1)
    return out


def fit(lib, axis: str, space: str = "mert", *, use_implicit: bool = True,
        C: float = 0.4) -> TasteHead:
    """Fit one axis. Returns the head; caller decides whether to save it."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import cross_val_score

    loaded = lib.vectors.ids_and_matrix(lib.db, space)
    if loaded is None:
        raise RuntimeError(f"no vectors in space '{space}'. Run `deepcut embed` first.")
    ids, matrix = loaded
    matrix = matrix.astype(np.float32)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    pos = {int(t): i for i, t in enumerate(ids)}

    labelled = lib.labels(axis)
    rows: list[int] = []
    y: list[float] = []
    w: list[float] = []
    for tid, value in labelled.items():
        if tid in pos:
            rows.append(pos[tid]); y.append(float(value)); w.append(1.0)

    if use_implicit and axis in {"like", "keep", "affinity"}:
        for tid, (value, weight) in implicit_labels(lib).items():
            if tid in pos and tid not in labelled:
                rows.append(pos[tid]); y.append(value); w.append(weight)

    if len(rows) < 12:
        raise RuntimeError(
            f"axis '{axis}' has {len(rows)} usable labels; fit needs at least 12. "
            f"Try `deepcut label --axis {axis} --suggest 20`.")

    X = matrix[rows]
    y_arr = np.array(y, dtype=np.float32)
    w_arr = np.array(w, dtype=np.float32)
    binary = set(np.unique(y_arr)).issubset({0.0, 1.0})

    if binary:
        yb = y_arr.astype(int)
        clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
        folds = int(min(5, np.bincount(yb).min()))
        cv = (float(cross_val_score(clf, X, yb, cv=folds, scoring="roc_auc").mean())
              if folds >= 2 else float("nan"))
        clf.fit(X, yb, sample_weight=w_arr)
        head = TasteHead(axis, space, clf.coef_[0].astype(np.float32),
                         float(clf.intercept_[0]), cv, len(rows), "binary")
    else:
        reg = Ridge(alpha=1.0 / max(C, 1e-6))
        folds = min(5, max(2, len(rows) // 4))
        cv = float(cross_val_score(reg, X, y_arr, cv=folds, scoring="r2").mean())
        reg.fit(X, y_arr, sample_weight=w_arr)
        head = TasteHead(axis, space, reg.coef_.astype(np.float32),
                         float(reg.intercept_), cv, len(rows), "continuous")

    head.layer_profile = _layer_profile(lib, axis, rows, ids, y_arr, w_arr)
    return head


def suggest_labels(lib, axis: str, models_dir: Path, n: int = 20,
                   space: str = "mert") -> list[int]:
    """Which tracks to label next.

    Uncertainty alone clusters your questions in one corner of the space, so
    candidates are picked by uncertainty and then thinned for diversity — the
    same MMR idea used in retrieval, applied to the labelling queue.
    """
    loaded = lib.vectors.ids_and_matrix(lib.db, space)
    if loaded is None:
        return []
    ids, matrix = loaded
    matrix = matrix.astype(np.float32)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
    already = set(lib.labels(axis))
    mask = np.array([int(t) not in already for t in ids])
    if not mask.any():
        return []

    head = TasteHead.load(models_dir, axis)
    if head is None or head.space != space:
        # Cold start: spread the first labels across the space rather than
        # asking about twenty near-identical tracks.
        return _spread(ids[mask], matrix[mask], n)

    probs = head.score_matrix(matrix[mask])
    uncertainty = 1.0 - np.abs(probs - 0.5) * 2.0
    order = np.argsort(-uncertainty)[: n * 6]
    return _spread(ids[mask][order], matrix[mask][order], n)


def _spread(ids: np.ndarray, matrix: np.ndarray, n: int) -> list[int]:
    picked: list[int] = []
    chosen_rows: list[int] = []
    for i in range(len(ids)):
        if len(picked) >= n:
            break
        if chosen_rows:
            sims = matrix[chosen_rows] @ matrix[i]
            if float(sims.max()) > 0.9:
                continue
        picked.append(int(ids[i]))
        chosen_rows.append(i)
    return picked


def _layer_profile(lib, axis: str, rows: list[int], ids: np.ndarray,
                   y: np.ndarray, w: np.ndarray) -> dict[str, float] | None:
    """Where in MERT's depth does this axis live?

    Fits a probe per depth band on the stored layer stack. An axis that is
    predicted best by the production band is about how records are *made*; one
    that lives in the structure band is about how they are *written*. Knowing
    which changes what you should retrieve on.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    from .explain import LAYER_BANDS

    loaded = lib.vectors.ids_and_matrix(lib.db, "mert_layers")
    if loaded is None or not set(np.unique(y)).issubset({0.0, 1.0}):
        return None
    lids, stack = loaded
    stack = stack.astype(np.float32)
    pos = {int(t): i for i, t in enumerate(lids)}
    keep = [(pos[int(ids[r])], int(y[i])) for i, r in enumerate(rows) if int(ids[r]) in pos]
    if len(keep) < 20:
        return None
    idx = [k[0] for k in keep]
    yy = np.array([k[1] for k in keep])
    if len(np.unique(yy)) < 2:
        return None

    n_layers = 13 if stack.shape[1] % 13 == 0 else 25
    dim = stack.shape[1] // n_layers
    out: dict[str, float] = {}
    folds = int(min(5, np.bincount(yy).min()))
    if folds < 2:
        return None
    for name, (lo, hi) in LAYER_BANDS.items():
        hi = min(hi, n_layers)
        if lo >= hi:
            continue
        sl = stack[idx][:, lo * dim: hi * dim]
        sl = sl / (np.linalg.norm(sl, axis=1, keepdims=True) + 1e-9)
        clf = LogisticRegression(max_iter=2000, C=0.4, class_weight="balanced")
        out[name] = round(float(cross_val_score(clf, sl, yy, cv=folds,
                                                scoring="roc_auc").mean()), 3)
    return out
