"""Retrieval, in two stages, over two spaces.

Stage one is recall: a wide sweep over the corrected vector spaces, blended and
hubness-corrected, filtered by hard constraints (tempo window, key
compatibility, slop threshold, artist caps). Stage two is selection: maximal
marginal relevance for diversity, then a small explicitly-budgeted exploration
share, then optional sequencing into a listenable order.

Optimising similarity alone produces a set that scores well and is boring to
listen to. The knobs that fix that — ``mmr_lambda`` and ``epsilon`` — are the
two most important parameters in the system and are deliberately exposed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RetrieveConfig
from .debias import SpaceCorrection
from .dsp import camelot_distance
from .store import Library, Track

SPACES = ("mert", "clap")


@dataclass
class Constraints:
    bpm_range: tuple[float, float] | None = None
    allow_half_double: bool = True
    camelot_seed: str | None = None
    camelot_max_distance: int = 2
    max_slop: float | None = None
    year_range: tuple[int, int] | None = None
    min_duration_s: float = 45.0
    exclude_track_ids: set[int] = field(default_factory=set)
    exclude_artists: set[str] = field(default_factory=set)
    artist_cap: int = 2
    unheard_only: bool = False
    require_tags: set[str] = field(default_factory=set)


@dataclass
class Candidate:
    track: Track
    score: float
    per_space: dict[str, float]
    features: dict
    origin: str = "relevance"   # relevance | exploration

    @property
    def bpm(self) -> float:
        return float(self.features.get("bpm", 0.0))

    @property
    def camelot(self) -> str:
        return str(self.features.get("camelot", ""))


class Engine:
    def __init__(self, lib: Library, cfg: RetrieveConfig, models_dir):
        self.lib = lib
        self.cfg = cfg
        self.models_dir = models_dir
        self._spaces: dict[str, tuple[np.ndarray, np.ndarray, SpaceCorrection | None]] = {}
        self._features = lib.all_features()
        self._slop = lib.slop_scores()
        self._plays = lib.play_counts()

    # -- space loading ----------------------------------------------------
    def space(self, name: str):
        if name in self._spaces:
            return self._spaces[name]
        loaded = self.lib.vectors.ids_and_matrix(self.lib.db, name)
        if loaded is None:
            return None
        ids, matrix = loaded
        corr = SpaceCorrection.load(self.models_dir / f"correction_{name}.npz") \
            if self.cfg.debias_popularity else None
        matrix = corr.apply(matrix) if corr else _norm(matrix.astype(np.float32))
        self._spaces[name] = (ids, matrix, corr)
        return self._spaces[name]

    def fit_corrections(self, n_components: int = 2) -> dict[str, dict]:
        """Fit and persist the popularity/anisotropy/hubness corrections."""
        from .debias import popularity_leakage
        report: dict[str, dict] = {}
        for name in SPACES:
            loaded = self.lib.vectors.ids_and_matrix(self.lib.db, name)
            if loaded is None:
                continue
            ids, matrix = loaded
            matrix = matrix.astype(np.float32)
            pop = np.array([self._plays.get(int(i), 0) for i in ids], dtype=np.float32)
            before = popularity_leakage(matrix, pop) if pop.sum() > 0 else 0.0
            corr = SpaceCorrection.fit(matrix, pop if pop.sum() > 0 else None,
                                       n_components=n_components)
            corr.save(self.models_dir / f"correction_{name}.npz")
            after = popularity_leakage(corr.apply(matrix), pop) if pop.sum() > 0 else 0.0
            report[name] = {"tracks": len(ids), "dim": int(matrix.shape[1]),
                            "popularity_r2_before": round(before, 4),
                            "popularity_r2_after": round(after, 4),
                            "components_removed": corr.n_components}
            self._spaces.pop(name, None)
        return report

    # -- query construction -----------------------------------------------
    def vector_for_tracks(self, name: str, track_ids: list[int]) -> np.ndarray | None:
        loaded = self.space(name)
        if loaded is None:
            return None
        ids, matrix, _ = loaded
        pos = {int(t): i for i, t in enumerate(ids)}
        rows = [pos[t] for t in track_ids if t in pos]
        if not rows:
            return None
        return _norm(matrix[rows].mean(axis=0))

    def vector_for_text(self, text: str, encoder) -> np.ndarray:
        """CLAP text tower. The prompt becomes a point in the audio space."""
        loaded = self.space("clap")
        vec = encoder.clap.encode_text([_prompt_template(text)])[0]
        if loaded is not None and loaded[2] is not None:
            vec = loaded[2].apply(vec)
        return _norm(vec)

    # -- search ------------------------------------------------------------
    def search(self, queries: dict[str, np.ndarray], constraints: Constraints,
               k: int | None = None, weights: dict[str, float] | None = None,
               ) -> list[Candidate]:
        k = k or self.cfg.results
        weights = weights or {"mert": self.cfg.weight_mert, "clap": self.cfg.weight_clap}
        weights = {s: w for s, w in weights.items() if s in queries and w > 0}
        if not weights:
            return []
        total = sum(weights.values())
        weights = {s: w / total for s, w in weights.items()}

        blended: dict[int, dict[str, float]] = {}
        for name, weight in weights.items():
            loaded = self.space(name)
            if loaded is None:
                continue
            ids, matrix, corr = loaded
            q = _norm(np.asarray(queries[name], dtype=np.float32))
            sims = matrix @ q
            if corr is not None and corr.csls_radius is not None:
                rows = np.arange(len(ids))
                qr = float(np.sort(sims)[-min(10, len(sims)):].mean())
                sims = corr.csls_scores(sims, rows, qr)
            take = min(self.cfg.candidates * 3, len(sims))
            top = np.argpartition(-sims, take - 1)[:take]
            for row in top:
                tid = int(ids[row])
                blended.setdefault(tid, {})[name] = float(sims[row])

        scored: list[Candidate] = []
        for tid, per_space in blended.items():
            if not self._passes(tid, per_space, constraints):
                continue
            track = self.lib.get(tid)
            if track is None:
                continue
            score = sum(weights[s] * v for s, v in per_space.items() if s in weights)
            scored.append(Candidate(track, score, per_space, self._features.get(tid, {})))

        scored.sort(key=lambda c: -c.score)
        pool = scored[: self.cfg.candidates]
        chosen = self._mmr(pool, k, constraints)
        chosen = self._explore(chosen, pool, k, constraints)
        return chosen

    # -- filters -----------------------------------------------------------
    def _passes(self, tid: int, per_space: dict[str, float], c: Constraints) -> bool:
        if tid in c.exclude_track_ids:
            return False
        f = self._features.get(tid)
        if f is None:
            return False
        if f.get("duration_s", 0) < c.min_duration_s:
            return False
        if c.max_slop is not None and self._slop.get(tid, 0.0) > c.max_slop:
            return False
        if c.unheard_only and self._plays.get(tid, 0) > 0:
            return False
        if c.bpm_range:
            lo, hi = c.bpm_range
            bpm = float(f.get("bpm", 0))
            options = [bpm, bpm * 2, bpm / 2] if c.allow_half_double else [bpm]
            if not any(lo <= b <= hi for b in options):
                return False
        if c.camelot_seed:
            if camelot_distance(c.camelot_seed, f.get("camelot", "")) > c.camelot_max_distance:
                return False
        if c.year_range:
            track = self.lib.get(tid)
            year = track.year if track else None
            if year is None or not (c.year_range[0] <= year <= c.year_range[1]):
                return False
        if c.require_tags:
            have = {t for t, _ in self.lib.tags(tid, limit=30)}
            if not (c.require_tags & have):
                return False
        if c.exclude_artists:
            track = self.lib.get(tid)
            if track and (track.artist or "").lower() in c.exclude_artists:
                return False
        return True

    # -- selection ---------------------------------------------------------
    def _mmr(self, pool: list[Candidate], k: int, c: Constraints) -> list[Candidate]:
        """Maximal marginal relevance over the MERT space, with an artist cap."""
        if not pool:
            return []
        loaded = self.space("mert") or self.space("clap")
        if loaded is None:
            return pool[:k]
        ids, matrix, _ = loaded
        pos = {int(t): i for i, t in enumerate(ids)}
        rows = [pos.get(cand.track.id) for cand in pool]
        usable = [i for i, r in enumerate(rows) if r is not None]
        if not usable:
            return pool[:k]
        V = matrix[[rows[i] for i in usable]]
        idx_map = {i: n for n, i in enumerate(usable)}

        lam = self.cfg.mmr_lambda
        selected: list[int] = []
        artist_counts: dict[str, int] = {}
        best_sim = np.full(len(usable), -1.0, dtype=np.float32)
        remaining = set(usable)
        cap = c.artist_cap or self.cfg.artist_cap

        while remaining and len(selected) < k:
            best_i, best_val = None, -1e9
            for i in remaining:
                artist = (pool[i].track.artist or "?").lower()
                if artist_counts.get(artist, 0) >= cap:
                    continue
                val = lam * pool[i].score - (1 - lam) * float(best_sim[idx_map[i]])
                if val > best_val:
                    best_i, best_val = i, val
            if best_i is None:
                break
            selected.append(best_i)
            remaining.discard(best_i)
            artist = (pool[best_i].track.artist or "?").lower()
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            sims = V @ V[idx_map[best_i]]
            best_sim = np.maximum(best_sim, sims)
        return [pool[i] for i in selected]

    def _explore(self, chosen: list[Candidate], pool: list[Candidate],
                 k: int, c: Constraints) -> list[Candidate]:
        """Spend an explicit epsilon of the slots on the tail.

        Sampled from the deep end of the candidate pool, weighted toward tracks
        you have not played. This is the only place novelty is injected, and
        keeping it in one place means it can be turned off and measured.
        """
        budget = int(round(self.cfg.epsilon * k))
        if budget <= 0 or len(pool) <= len(chosen):
            return chosen[:k]
        picked = {cnd.track.id for cnd in chosen}
        tail = [cnd for cnd in pool[len(chosen):] if cnd.track.id not in picked]
        if not tail:
            return chosen[:k]
        weights = np.array([1.0 / (1.0 + self._plays.get(cnd.track.id, 0)) for cnd in tail])
        weights = weights / weights.sum()
        rng = np.random.default_rng()
        n = min(budget, len(tail))
        idx = rng.choice(len(tail), size=n, replace=False, p=weights)
        explorers = []
        for i in idx:
            cnd = tail[int(i)]
            explorers.append(Candidate(cnd.track, cnd.score, cnd.per_space,
                                       cnd.features, origin="exploration"))
        keep = chosen[: max(k - len(explorers), 0)]
        return keep + explorers


# ---------------------------------------------------------------------------
# Sequencing
# ---------------------------------------------------------------------------

def sequence(candidates: list[Candidate], *, energy_arc: str = "rise",
             tempo_weight: float = 1.0, key_weight: float = 1.0) -> list[Candidate]:
    """Order a set into something you would actually let play.

    Greedy nearest-transition walk under three costs: tempo continuity, Camelot
    distance, and deviation from a target loudness arc. Playlists with smooth
    transitions get skipped less; this is where that shows up.
    """
    if len(candidates) < 3:
        return candidates
    n = len(candidates)
    targets = _arc(n, energy_arc)
    remaining = list(range(n))
    start = min(remaining, key=lambda i: abs(_energy(candidates[i]) - targets[0]))
    order = [start]
    remaining.remove(start)
    for step in range(1, n):
        prev = candidates[order[-1]]
        best, best_cost = None, 1e18
        for i in remaining:
            cnd = candidates[i]
            tempo_cost = abs(_bpm(cnd) - _bpm(prev)) / 12.0
            key_cost = camelot_distance(prev.camelot, cnd.camelot) / 2.0
            arc_cost = abs(_energy(cnd) - targets[step]) * 2.0
            cost = tempo_weight * tempo_cost + key_weight * key_cost + arc_cost
            if cost < best_cost:
                best, best_cost = i, cost
        order.append(best)
        remaining.remove(best)
    return [candidates[i] for i in order]


def cadence_query(spm: float, tolerance: float = 0.04) -> tuple[float, float]:
    """A tempo window for a target cadence in steps per minute.

    Runners hold a fairly narrow cadence, so the window is tight; half-time is
    handled by the half/double allowance in the constraint check rather than by
    widening the window, which would let unrelated tempos through.
    """
    return (spm * (1 - tolerance), spm * (1 + tolerance))


def _arc(n: int, shape: str) -> np.ndarray:
    x = np.linspace(0, 1, n)
    if shape == "rise":
        return 0.35 + 0.5 * x
    if shape == "fall":
        return 0.85 - 0.5 * x
    if shape == "peak":
        return 0.35 + 0.55 * np.sin(np.pi * x)
    return np.full(n, 0.6)


def _energy(c: Candidate) -> float:
    lufs = float(c.features.get("integrated_lufs", -14.0))
    return float(np.clip((lufs + 26.0) / 18.0, 0.0, 1.0))


def _bpm(c: Candidate) -> float:
    return float(c.features.get("bpm", 120.0)) or 120.0


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    if v.ndim == 1:
        return v / (np.linalg.norm(v) + 1e-9)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)


def _prompt_template(text: str) -> str:
    """CLAP's text tower was trained on caption-like strings, not bare queries."""
    t = text.strip().rstrip(".")
    lowered = t.lower()
    if lowered.startswith(("this is", "a ", "an ", "the sound of")):
        return t
    return f"This is a recording of {t}."
