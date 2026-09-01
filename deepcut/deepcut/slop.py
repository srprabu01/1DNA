"""A synthetic-audio prior, with its evidence shown.

If most of what arrives in a new-music feed is machine-generated filler, the
filter is not a footnote — it is most of the precision. But a detector that
returns a bare number is useless and slightly dangerous: the same signals that
mark a generated track (rigid grid, loop repetition, narrow dynamics) also mark
perfectly good sequenced electronic music. So this module returns a score
*plus* the contribution of every signal, and the interface is expected to show
the evidence rather than silently drop tracks.

Two modes:

* **Prior** (default) — hand-weighted logistic over interpretable signals.
  Works with zero labels. Calibrated to be conservative.
* **Calibrated** — the same signals plus the frozen embedding, fit as logistic
  regression on your own labelled examples. A few hundred labels beats the
  prior comfortably. ``deepcut slop fit`` does this.

Nothing here detects *watermarks*. If a generator embeds one, a watermark
reader is strictly better evidence than any of this, and should be preferred.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SlopVerdict:
    score: float                       # 0 = confidently human, 1 = confidently generated
    confidence: float                  # how much evidence was actually available
    signals: dict[str, float] = field(default_factory=dict)      # raw measurements
    contributions: dict[str, float] = field(default_factory=dict)  # signed logit terms
    mode: str = "prior"

    @property
    def verdict(self) -> str:
        if self.confidence < 0.35:
            return "unclear"
        if self.score >= 0.65:
            return "likely generated"
        if self.score <= 0.30:
            return "likely human-performed"
        return "machine-produced, origin unclear"

    def top_evidence(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.contributions.items(), key=lambda kv: -abs(kv[1]))[:n]

    def to_dict(self) -> dict:
        return {"score": round(self.score, 4), "confidence": round(self.confidence, 3),
                "verdict": self.verdict, "mode": self.mode,
                "signals": {k: round(v, 4) for k, v in self.signals.items()},
                "contributions": {k: round(v, 4) for k, v in self.contributions.items()}}


# Hand-set weights. Deliberately modest on the "sounds sequenced" family so a
# techno record does not get filtered out of a techno query.
PRIOR_WEIGHTS = {
    "phase_incoherence": 1.9,
    "spectral_checkerboard": 1.7,
    "hf_cliff_unexplained": 1.5,
    "hf_noise_flatness": 1.1,
    "edge_truncation": 0.9,
    "grid_rigidity": 0.8,
    "loop_rigidity": 0.7,
    "static_dynamics": 0.5,
    "stereo_degenerate": 0.4,
}
PRIOR_BIAS = -2.4


def score(features, y: np.ndarray, sr: int, *, lossless_source: bool,
          source_bitrate: int | None = None,
          model_dir: Path | None = None,
          embedding: np.ndarray | None = None) -> SlopVerdict:
    """Score one track. ``features`` is a :class:`~deepcut.dsp.SignalFeatures`."""
    sig: dict[str, float] = {}

    sig["phase_incoherence"] = _phase_incoherence(y, sr)
    sig["spectral_checkerboard"] = _checkerboard(y, sr)
    sig["hf_cliff_unexplained"] = _hf_cliff(features, sr, lossless_source, source_bitrate)
    sig["hf_noise_flatness"] = _hf_noise_flatness(y, sr)
    sig["edge_truncation"] = _edge_truncation(y, sr, features.duration_s)
    sig["grid_rigidity"] = _squash(4.0 - features.microtiming_ms, 0.0, 4.0)
    sig["loop_rigidity"] = _squash(features.repetition_index - 0.55, 0.0, 0.35)
    sig["static_dynamics"] = _squash(1.6 - features.dynamic_spread_lu, 0.0, 1.6)
    sig["stereo_degenerate"] = 1.0 if features.side_ratio < 0.005 else 0.0

    model = _load_calibrated(model_dir) if model_dir else None
    if model is not None:
        return _score_calibrated(model, sig, embedding)

    contributions = {k: PRIOR_WEIGHTS[k] * v for k, v in sig.items() if k in PRIOR_WEIGHTS}
    logit = PRIOR_BIAS + sum(contributions.values())
    available = sum(1 for v in sig.values() if v is not None and np.isfinite(v))
    return SlopVerdict(
        score=float(1 / (1 + np.exp(-logit))),
        confidence=float(np.clip(available / len(PRIOR_WEIGHTS), 0, 1)
                         * (0.6 + 0.4 * min(features.duration_s / 60.0, 1.0))),
        signals=sig, contributions=contributions, mode="prior",
    )


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def _phase_incoherence(y: np.ndarray, sr: int) -> float:
    """Frame-to-frame phase-advance irregularity in steady harmonic partials.

    A real recorded partial advances in phase almost linearly. Vocoder-style
    and diffusion decoders reconstruct magnitude well and phase less well, so
    the second difference of unwrapped phase stays noisier than it should on
    bins that are otherwise stable in magnitude.
    """
    n_fft, hop = 2048, 512
    seg = y[: sr * 45]
    if len(seg) < n_fft * 8:
        return 0.0
    win = np.hanning(n_fft).astype(np.float32)
    frames = 1 + (len(seg) - n_fft) // hop
    frames = min(frames, 900)
    spec = np.empty((n_fft // 2 + 1, frames), dtype=np.complex64)
    for i in range(frames):
        spec[:, i] = np.fft.rfft(seg[i * hop: i * hop + n_fft] * win)
    mag = np.abs(spec)
    if mag.max() <= 0:
        return 0.0
    # Only judge bins that are loud and magnitude-stable.
    strength = mag.mean(axis=1)
    stability = mag.std(axis=1) / (mag.mean(axis=1) + 1e-9)
    keep = np.argsort(-strength)[:48]
    keep = keep[stability[keep] < 1.0]
    if len(keep) < 6:
        return 0.0
    phase = np.unwrap(np.angle(spec[keep]), axis=1)
    dev = np.diff(phase, n=2, axis=1)
    # Wrap deviations into (-pi, pi] before measuring spread.
    dev = (dev + np.pi) % (2 * np.pi) - np.pi
    spread = float(np.mean(np.std(dev, axis=1)))
    return _squash(spread - 1.0, 0.0, 1.0)


def _checkerboard(y: np.ndarray, sr: int) -> float:
    """Periodic banding along the frequency axis from transposed convolutions."""
    n_fft, hop = 2048, 512
    seg = y[: sr * 45]
    if len(seg) < n_fft * 8:
        return 0.0
    win = np.hanning(n_fft).astype(np.float32)
    frames = min(1 + (len(seg) - n_fft) // hop, 600)
    acc = np.zeros(n_fft // 2 + 1, dtype=np.float64)
    for i in range(frames):
        acc += np.abs(np.fft.rfft(seg[i * hop: i * hop + n_fft] * win))
    profile = np.log(acc / frames + 1e-9)
    profile = profile - _smooth(profile, 33)  # remove the spectral envelope
    profile -= profile.mean()
    if np.allclose(profile, 0):
        return 0.0
    ac = np.correlate(profile, profile, mode="full")[len(profile) - 1:]
    ac /= ac[0] + 1e-12
    peak = float(np.max(ac[4:64])) if len(ac) > 64 else 0.0
    return _squash(peak - 0.25, 0.0, 0.45)


def _hf_cliff(features, sr: int, lossless: bool, bitrate: int | None) -> float:
    """A brick wall well below Nyquist that the container does not explain.

    A 16 kHz ceiling on a 128 kbps MP3 is the codec. The same ceiling on a FLAC
    is either a generator or an upscaled lossy file, and both are worth flagging.
    """
    nyquist = sr / 2
    cliff = features.rolloff99_hz
    if cliff <= 0 or cliff > nyquist * 0.92:
        return 0.0
    expected = {None: 0.0}
    if not lossless and bitrate:
        if bitrate < 130_000:
            expected_cutoff = 16_000
        elif bitrate < 200_000:
            expected_cutoff = 18_000
        else:
            expected_cutoff = 19_500
        if cliff >= expected_cutoff * 0.92:
            return 0.0
    del expected
    shortfall = (nyquist * 0.92 - cliff) / (nyquist * 0.92)
    return _squash(shortfall, 0.05, 0.45) * (1.0 if lossless else 0.5)


def _hf_noise_flatness(y: np.ndarray, sr: int) -> float:
    """Featureless hiss above 12 kHz instead of transient-linked HF content."""
    from numpy.fft import rfft, rfftfreq
    n_fft = 4096
    seg = y[: sr * 30]
    if len(seg) < n_fft * 4:
        return 0.0
    freqs = rfftfreq(n_fft, 1 / sr)
    band = (freqs > 12_000) & (freqs < min(18_000, sr / 2))
    if band.sum() < 8:
        return 0.0
    win = np.hanning(n_fft).astype(np.float32)
    vals = []
    for i in range(0, len(seg) - n_fft, n_fft):
        mag = np.abs(rfft(seg[i:i + n_fft] * win))[band] + 1e-12
        geo = np.exp(np.mean(np.log(mag)))
        vals.append(geo / (mag.mean() + 1e-12))
    if not vals:
        return 0.0
    return _squash(float(np.mean(vals)) - 0.45, 0.0, 0.35)


def _edge_truncation(y: np.ndarray, sr: int, duration: float) -> float:
    """Hard cuts at the boundaries, or a suspiciously exact runtime."""
    if len(y) < sr:
        return 0.0
    head = float(np.max(np.abs(y[: int(0.02 * sr)])))
    tail = float(np.max(np.abs(y[-int(0.02 * sr):])))
    peak = float(np.max(np.abs(y)) + 1e-12)
    hard = 0.0
    if head / peak > 0.15:
        hard += 0.5
    if tail / peak > 0.15:
        hard += 0.5
    # Generators often emit exactly N seconds.
    if abs(duration - round(duration)) < 0.02 and duration > 20:
        hard = min(hard + 0.35, 1.0)
    return min(hard, 1.0)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def fit(examples: list[tuple[dict[str, float], np.ndarray | None, int]],
        model_dir: Path, use_embedding: bool = True) -> dict:
    """Fit a logistic model on labelled examples.

    ``examples`` is a list of ``(signals, embedding, label)`` where label is 1
    for generated. Returns a dict of fit diagnostics.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    keys = sorted(PRIOR_WEIGHTS)
    X_sig = np.array([[ex[0].get(k, 0.0) for k in keys] for ex in examples], dtype=np.float32)
    y = np.array([ex[2] for ex in examples], dtype=np.int32)
    embs = [ex[1] for ex in examples]
    use_emb = use_embedding and all(e is not None for e in embs)
    X = np.hstack([X_sig, np.stack(embs)]) if use_emb else X_sig

    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    folds = int(min(5, np.bincount(y).min()))
    auc = (cross_val_score(clf, X, y, cv=folds, scoring="roc_auc").mean()
           if folds >= 2 else float("nan"))
    clf.fit(X, y)

    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez(model_dir / "slop_model.npz", coef=clf.coef_, intercept=clf.intercept_)
    meta = {"keys": keys, "use_embedding": bool(use_emb), "n": len(y),
            "positives": int(y.sum()), "cv_auc": float(auc)}
    (model_dir / "slop_model.json").write_text(json.dumps(meta, indent=2))
    return meta


def _load_calibrated(model_dir: Path):
    meta_path = model_dir / "slop_model.json"
    weights_path = model_dir / "slop_model.npz"
    if not (meta_path.exists() and weights_path.exists()):
        return None
    meta = json.loads(meta_path.read_text())
    data = np.load(weights_path)
    return {"meta": meta, "coef": data["coef"][0], "intercept": float(data["intercept"][0])}


def _score_calibrated(model, sig: dict[str, float], embedding: np.ndarray | None) -> SlopVerdict:
    meta = model["meta"]
    keys = meta["keys"]
    x = np.array([sig.get(k, 0.0) for k in keys], dtype=np.float32)
    if meta["use_embedding"]:
        if embedding is None:
            return SlopVerdict(0.5, 0.0, sig, {}, mode="calibrated-unavailable")
        x = np.concatenate([x, embedding.astype(np.float32)])
    coef = model["coef"]
    logit = float(model["intercept"] + float(coef @ x))
    contributions = {k: float(coef[i] * x[i]) for i, k in enumerate(keys)}
    if meta["use_embedding"]:
        contributions["embedding"] = float(coef[len(keys):] @ x[len(keys):])
    return SlopVerdict(float(1 / (1 + np.exp(-logit))),
                       confidence=min(1.0, meta["n"] / 200.0),
                       signals=sig, contributions=contributions, mode="calibrated")


# ---------------------------------------------------------------------------

def _squash(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(np.clip((x - lo) / (hi - lo + 1e-9), 0.0, 1.0))


def _smooth(x: np.ndarray, k: int) -> np.ndarray:
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")
