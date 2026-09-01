"""Why did this track come back?

The majors will not ship this, because an explanation is an invitation to
argue with the recommendation. Locally it is the best part: a similarity score
you can interrogate is a similarity score you can tune.

Three sources of explanation, in increasing order of interest:

* **Measured deltas.** Tempo, key distance, loudness, brightness. Cheap,
  exact, and enough on its own for most "why is this here" questions.
* **Space disagreement.** MERT and CLAP encode different things. When they
  disagree the disagreement is the story: high MERT and low CLAP means it
  *sounds* alike but would be described differently — a cover, a genre
  crossing, a producer's fingerprint on a different scene.
* **Layer attribution.** MERT's lower layers carry timbre and production, its
  upper layers carry musical structure. Comparing similarity at each depth
  separates "recorded the same way" from "written the same way", which no
  single cosine can do.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import camelot_distance

# Rough reading of what MERT's depth encodes. Layer indices are into the
# hidden_states tuple, which includes the embedding output at index 0.
LAYER_BANDS = {
    "production": (1, 4),   # timbre, recording character, mix
    "playing": (5, 8),      # articulation, groove, instrumentation
    "structure": (9, 13),   # harmony, form, higher-level musical identity
}


@dataclass
class Explanation:
    seed_label: str
    candidate_label: str
    space_similarity: dict[str, float] = field(default_factory=dict)
    layer_similarity: dict[str, float] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    shared_tags: list[str] = field(default_factory=list)
    taste: dict[str, float] = field(default_factory=dict)
    slop: float | None = None

    def lines(self) -> list[str]:
        out: list[str] = []
        mert = self.space_similarity.get("mert")
        clap = self.space_similarity.get("clap")
        if mert is not None and clap is not None:
            gap = mert - clap
            if gap > 0.18:
                out.append(f"sounds alike ({mert:.2f}) but sits in a different scene ({clap:.2f})")
            elif gap < -0.18:
                out.append(f"described alike ({clap:.2f}) but built differently ({mert:.2f})")
            else:
                out.append(f"close in both sound ({mert:.2f}) and description ({clap:.2f})")

        if self.layer_similarity:
            ranked = sorted(self.layer_similarity.items(), key=lambda kv: -kv[1])
            top, bottom = ranked[0], ranked[-1]
            if top[1] - bottom[1] > 0.08:
                out.append(f"nearest in {top[0]} ({top[1]:.2f}), furthest in {bottom[0]} ({bottom[1]:.2f})")

        bpm = self.deltas.get("bpm_delta")
        if bpm is not None:
            out.append("same tempo" if abs(bpm) < 2 else f"{abs(bpm):.0f} BPM {'faster' if bpm > 0 else 'slower'}")
        cam = self.deltas.get("camelot_distance")
        if cam is not None:
            out.append("same key" if cam == 0 else
                       "harmonically compatible" if cam == 1 else
                       f"{int(cam)} steps around the wheel")
        lufs = self.deltas.get("lufs_delta")
        if lufs is not None and abs(lufs) > 3:
            out.append(f"{abs(lufs):.0f} LU {'louder' if lufs > 0 else 'quieter'}")
        bright = self.deltas.get("centroid_ratio")
        if bright is not None and (bright > 1.25 or bright < 0.8):
            out.append("brighter" if bright > 1 else "darker")
        micro = self.deltas.get("microtiming_delta")
        if micro is not None and abs(micro) > 6:
            out.append("looser played" if micro > 0 else "tighter to the grid")

        if self.shared_tags:
            out.append("shares " + ", ".join(self.shared_tags[:3]))
        for axis, value in self.taste.items():
            out.append(f"{axis} {value:+.2f}")
        if self.slop is not None and self.slop > 0.5:
            out.append(f"synthetic-audio score {self.slop:.2f}")
        return out

    def __str__(self) -> str:
        return "; ".join(self.lines())


def explain(engine, seed_id: int, candidate_id: int,
            taste_heads: dict | None = None) -> Explanation:
    lib = engine.lib
    seed, cand = lib.get(seed_id), lib.get(candidate_id)
    if seed is None or cand is None:
        raise KeyError("unknown track")
    e = Explanation(seed.label, cand.label)

    for space in ("mert", "clap"):
        a = lib.vectors.vector(lib.db, space, seed_id)
        b = lib.vectors.vector(lib.db, space, candidate_id)
        if a is not None and b is not None:
            e.space_similarity[space] = _cos(a, b)

    la = lib.vectors.vector(lib.db, "mert_layers", seed_id)
    lb = lib.vectors.vector(lib.db, "mert_layers", candidate_id)
    if la is not None and lb is not None:
        e.layer_similarity = _layer_bands(la, lb)

    fa = lib.features(seed_id) or {}
    fb = lib.features(candidate_id) or {}
    if fa and fb:
        e.deltas = {
            "bpm_delta": round(float(fb.get("bpm", 0)) - float(fa.get("bpm", 0)), 1),
            "camelot_distance": camelot_distance(fa.get("camelot", ""), fb.get("camelot", "")),
            "lufs_delta": round(float(fb.get("integrated_lufs", 0)) - float(fa.get("integrated_lufs", 0)), 1),
            "centroid_ratio": round(float(fb.get("spectral_centroid_hz", 1)) /
                                    max(float(fa.get("spectral_centroid_hz", 1)), 1e-6), 2),
            "microtiming_delta": round(float(fb.get("microtiming_ms", 0)) -
                                       float(fa.get("microtiming_ms", 0)), 1),
            "harmonic_delta": round(float(fb.get("harmonic_ratio", 0)) -
                                    float(fa.get("harmonic_ratio", 0)), 2),
        }

    ta = {t for t, _ in lib.tags(seed_id, 20)}
    tb = {t for t, _ in lib.tags(candidate_id, 20)}
    e.shared_tags = sorted(ta & tb)

    if taste_heads:
        for axis, head in taste_heads.items():
            v = lib.vectors.vector(lib.db, head.space, candidate_id)
            if v is not None:
                e.taste[axis] = round(float(head.score(v)), 3)

    slop = lib.slop(candidate_id)
    if slop:
        e.slop = float(slop["score"])
    return e


def contrast(engine, track_ids: list[int], k: int = 6) -> dict[str, list[str]]:
    """What distinguishes a set of tracks from the rest of the library.

    Useful for naming a cluster or a generated playlist: report the measured
    features on which the set is most unlike everything else, as z-scores.
    """
    feats = engine._features
    keys = ["bpm", "integrated_lufs", "spectral_centroid_hz", "harmonic_ratio",
            "microtiming_ms", "hf_ratio", "sub_ratio", "chroma_entropy",
            "dynamic_spread_lu", "repetition_index", "onset_rate_hz"]
    inside, outside = [], []
    chosen = set(track_ids)
    for tid, f in feats.items():
        row = [float(f.get(k, 0.0)) for k in keys]
        (inside if tid in chosen else outside).append(row)
    if len(inside) < 2 or len(outside) < 8:
        return {}
    A, B = np.array(inside), np.array(outside)
    mu, sd = B.mean(axis=0), B.std(axis=0) + 1e-9
    z = (A.mean(axis=0) - mu) / sd
    order = np.argsort(-np.abs(z))[:k]
    high = [f"{keys[i]} {'high' if z[i] > 0 else 'low'} ({z[i]:+.1f}σ)" for i in order]
    return {"distinguishing": high}


def _layer_bands(a: np.ndarray, b: np.ndarray, dim: int | None = None) -> dict[str, float]:
    """Cosine per depth band from the flattened (layers, dim) stacks."""
    if a.shape != b.shape or a.ndim != 1:
        return {}
    for n_layers in (13, 25):  # MERT-95M has 12 blocks, 330M has 24
        if a.size % n_layers == 0:
            dim = a.size // n_layers
            A = a.reshape(n_layers, dim)
            B = b.reshape(n_layers, dim)
            out: dict[str, float] = {}
            for name, (lo, hi) in LAYER_BANDS.items():
                hi = min(hi, n_layers)
                if lo >= hi:
                    continue
                out[name] = round(float(np.mean([_cos(A[i], B[i]) for i in range(lo, hi)])), 3)
            return out
    return {}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    return round(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)), 3)
