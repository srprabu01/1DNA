"""Space corrections before similarity — ported from deepcut's debias.py.

Raw cosine over a feature space is a worse similarity metric than it looks:
  1. Popularity leaks into the geometry — "similar to X" quietly becomes
     "popular and vaguely like X". We regress play-count on the vectors and
     subtract that direction.
  2. The space is anisotropic — a couple of directions dominate and make every
     pair look alike. We optionally strip the top principal component(s).
  3. Hubs — a few vectors are near everything and pollute every neighbour list.
     CSLS penalises a candidate by how crowded its own neighbourhood is.

Adapted for this app's ~25-dim feature space (deepcut runs it on 768+-dim
foundation-model embeddings, so we default to removing only 1 component here).
"""
import numpy as np


def project_out(X, direction):
    if direction is None:
        return X
    d = direction / (np.linalg.norm(direction) + 1e-9)
    return X - np.outer(X @ d, d)


class SpaceCorrection:
    def __init__(self, mean=None, pop_dir=None, top_components=None, csls_radius=None):
        self.mean = mean
        self.pop_dir = pop_dir
        self.top_components = top_components
        self.csls_radius = csls_radius

    @classmethod
    def fit(cls, X, popularity=None, n_components=1, csls_k=10):
        X = np.asarray(X, dtype=np.float32)
        mean = X.mean(axis=0)
        Xc = X - mean

        pop_dir = None
        if popularity is not None and len(popularity) == len(X):
            p = np.log1p(np.asarray(popularity, dtype=np.float32))  # plays are heavy-tailed
            if np.std(p) > 1e-6:
                p = (p - p.mean()) / (p.std() + 1e-9)
                pop_dir = (Xc.T @ p) / len(p)
                n = np.linalg.norm(pop_dir)
                pop_dir = pop_dir / n if n > 1e-9 else None

        Xp = project_out(Xc, pop_dir)
        comps = None
        if n_components > 0 and len(Xp) > n_components + 1:
            k = min(n_components, min(Xp.shape) - 1)
            _, _, Vt = np.linalg.svd(Xp, full_matrices=False)
            comps = Vt[:k].astype(np.float32)

        corr = cls(mean=mean, pop_dir=pop_dir, top_components=comps)
        # CSLS hubness correction only helps high-dim (transformer) spaces; on a
        # low-dim feature space it hurts, so callers pass csls_k=0 to skip it.
        corr.csls_radius = _csls_radius(corr.apply(X), k=csls_k) if csls_k > 0 else None
        return corr

    def apply(self, X):
        V = np.asarray(X, dtype=np.float32)
        single = V.ndim == 1
        if single:
            V = V[None, :]
        if self.mean is not None:
            V = V - self.mean
        V = project_out(V, self.pop_dir)
        if self.top_components is not None:
            V = V - (V @ self.top_components.T) @ self.top_components
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        return V[0] if single else V

    def csls(self, sims, rows):
        """Hubness-corrected scores: 2*sim - own_radius - candidate_radius."""
        if self.csls_radius is None:
            return sims
        r = self.csls_radius[rows]
        qr = float(self.csls_radius.mean())
        return 2.0 * sims - r - qr


def popularity_leakage(X, popularity):
    """R^2 in [0,1] of how much of the space is popularity. Above ~0.15 means
    'similar tracks' is substantially a popularity chart."""
    X = np.asarray(X, dtype=np.float32)
    p = np.log1p(np.asarray(popularity, dtype=np.float32))
    if np.std(p) < 1e-6 or len(X) < 8:
        return 0.0
    p = (p - p.mean()) / (p.std() + 1e-9)
    Xc = X - X.mean(axis=0)
    lam = 1e-2 * len(X)                       # ridge — dims can exceed samples early
    G = Xc.T @ Xc + lam * np.eye(Xc.shape[1], dtype=np.float32)
    w = np.linalg.solve(G, Xc.T @ p)
    pred = Xc @ w
    ss_res = float(np.sum((p - pred) ** 2))
    ss_tot = float(np.sum(p ** 2))
    return float(np.clip(1.0 - ss_res / (ss_tot + 1e-9), 0.0, 1.0))


def _csls_radius(X, k=10, block=2048):
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
