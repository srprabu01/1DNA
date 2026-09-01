"""Three corrections applied to the vector space before anything is retrieved.

Raw cosine over a foundation-model space is a worse similarity metric than it
looks, for three separate and individually fixable reasons.

1. **Popularity leaks into the geometry.** Anything trained or filtered on
   engagement puts famous tracks in the middle of the space, so "similar to X"
   quietly means "famous and vaguely like X". Regress popularity on the
   embeddings and subtract the resulting direction.

2. **The space is anisotropic.** Frozen transformer embeddings concentrate in a
   narrow cone; a few dominant directions account for most of the variance and
   every pair looks similar. Removing the top principal components restores the
   dynamic range of the cosine.

3. **Hubs.** In high dimensions a handful of vectors are close to *everything*
   and appear in every neighbour list. CSLS penalises a candidate by how
   crowded its own neighbourhood is, which is the cheapest known fix.

All three are fitted once over the library and stored. None of them touch the
encoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class SpaceCorrection:
    mean: np.ndarray | None = None
    popularity_dir: np.ndarray | None = None
    top_components: np.ndarray | None = None   # (n_components, dim)
    csls_radius: np.ndarray | None = None      # per-row mean neighbour similarity
    n_components: int = 0

    # -- fitting ----------------------------------------------------------
    @classmethod
    def fit(cls, matrix: np.ndarray, popularity: np.ndarray | None = None,
            n_components: int = 2, csls_k: int = 10) -> "SpaceCorrection":
        X = np.asarray(matrix, dtype=np.float32)
        mean = X.mean(axis=0)
        Xc = X - mean

        pop_dir = None
        if popularity is not None and len(popularity) == len(X):
            p = np.asarray(popularity, dtype=np.float32)
            if np.std(p) > 1e-6:
                # log1p because play counts are heavy-tailed; a track played 500
                # times is not 500x more "popular" in any useful geometric sense.
                p = np.log1p(p)
                p = (p - p.mean()) / (p.std() + 1e-9)
                # Least-squares direction that best predicts popularity.
                pop_dir = (Xc.T @ p) / len(p)
                norm = np.linalg.norm(pop_dir)
                pop_dir = pop_dir / norm if norm > 1e-9 else None

        Xp = _project_out(Xc, pop_dir)
        comps = None
        if n_components > 0 and len(Xp) > n_components + 1:
            # Randomised SVD is plenty for the top 2-4 directions.
            k = min(n_components, min(Xp.shape) - 1)
            _, _, Vt = np.linalg.svd(_subsample(Xp, 20_000), full_matrices=False)
            comps = Vt[:k].astype(np.float32)

        corr = cls(mean=mean, popularity_dir=pop_dir, top_components=comps,
                   n_components=0 if comps is None else comps.shape[0])
        corrected = corr.apply(X)
        corr.csls_radius = _csls_radius(corrected, k=csls_k)
        return corr

    # -- applying ---------------------------------------------------------
    def apply(self, X: np.ndarray) -> np.ndarray:
        V = np.asarray(X, dtype=np.float32)
        single = V.ndim == 1
        if single:
            V = V[None, :]
        if self.mean is not None:
            V = V - self.mean
        V = _project_out(V, self.popularity_dir)
        if self.top_components is not None:
            V = V - (V @ self.top_components.T) @ self.top_components
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        return V[0] if single else V

    def csls_scores(self, sims: np.ndarray, rows: np.ndarray,
                    query_radius: float | None = None) -> np.ndarray:
        """Penalise candidates that are close to everything."""
        if self.csls_radius is None:
            return sims
        r = self.csls_radius[rows]
        qr = float(query_radius) if query_radius is not None else float(self.csls_radius.mean())
        return 2.0 * sims - r - qr

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=_or_empty(self.mean),
            popularity_dir=_or_empty(self.popularity_dir),
            top_components=_or_empty(self.top_components),
            csls_radius=_or_empty(self.csls_radius),
        )
        path.with_suffix(".json").write_text(json.dumps({"n_components": self.n_components}))

    @classmethod
    def load(cls, path: Path) -> "SpaceCorrection | None":
        if not path.exists():
            return None
        d = np.load(path)
        return cls(
            mean=_or_none(d["mean"]),
            popularity_dir=_or_none(d["popularity_dir"]),
            top_components=_or_none(d["top_components"]),
            csls_radius=_or_none(d["csls_radius"]),
            n_components=int(json.loads(path.with_suffix(".json").read_text())["n_components"])
            if path.with_suffix(".json").exists() else 0,
        )


def popularity_leakage(matrix: np.ndarray, popularity: np.ndarray) -> float:
    """How much of the space is popularity, as an R^2 in [0, 1].

    Worth printing before and after correction. If this is above ~0.15 your
    "similar tracks" list is substantially a popularity chart.
    """
    X = np.asarray(matrix, dtype=np.float32)
    p = np.log1p(np.asarray(popularity, dtype=np.float32))
    if np.std(p) < 1e-6 or len(X) < 8:
        return 0.0
    p = (p - p.mean()) / (p.std() + 1e-9)
    Xc = X - X.mean(axis=0)
    # Ridge, because dim >> samples is common early on.
    lam = 1e-2 * len(X)
    G = Xc.T @ Xc + lam * np.eye(Xc.shape[1], dtype=np.float32)
    w = np.linalg.solve(G, Xc.T @ p)
    pred = Xc @ w
    ss_res = float(np.sum((p - pred) ** 2))
    ss_tot = float(np.sum(p ** 2))
    return float(np.clip(1.0 - ss_res / (ss_tot + 1e-9), 0.0, 1.0))


# ---------------------------------------------------------------------------

def _project_out(X: np.ndarray, direction: np.ndarray | None) -> np.ndarray:
    if direction is None:
        return X
    d = direction / (np.linalg.norm(direction) + 1e-9)
    return X - np.outer(X @ d, d)


def _subsample(X: np.ndarray, n: int) -> np.ndarray:
    if len(X) <= n:
        return X
    rng = np.random.default_rng(0)
    return X[rng.choice(len(X), n, replace=False)]


def _csls_radius(X: np.ndarray, k: int = 10, block: int = 2048) -> np.ndarray:
    """Mean similarity to each vector's k nearest neighbours."""
    n = len(X)
    k = min(k, max(n - 1, 1))
    out = np.zeros(n, dtype=np.float32)
    for start in range(0, n, block):
        chunk = X[start:start + block]
        sims = chunk @ X.T
        for i in range(len(chunk)):
            sims[i, start + i] = -np.inf
        part = np.partition(sims, -k, axis=1)[:, -k:]
        out[start:start + block] = part.mean(axis=1)
    return out


def _or_empty(a): return np.array([]) if a is None else a
def _or_none(a): return None if a.size == 0 else a
