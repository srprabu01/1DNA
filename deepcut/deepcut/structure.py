"""Where the track's parts are, and which of them to show the encoders.

Two jobs:

1. Segment the track into repeating sections (a Laplacian spectral-clustering
   segmentation in the McFee & Ellis style, over beat-synchronous features).
2. Use that segmentation to choose the excerpts the semantic encoders see.

(2) is the part that matters most for retrieval quality and is where this
departs from the usual advice. Fixed percentage offsets are a guess about where
the interesting part of a track is. The repeated, high-energy section *is* the
interesting part — it is the bit a person would hum — and segmentation finds it
for a few hundred milliseconds of extra CPU. Fixed offsets remain the fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ANALYSIS_SR, ExcerptPolicy


@dataclass
class Section:
    start: float
    end: float
    label: int
    energy: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {"start": round(self.start, 2), "end": round(self.end, 2),
                "label": self.label, "energy": round(self.energy, 4)}


@dataclass
class Excerpt:
    start: float
    seconds: float
    reason: str  # "section:3" | "fallback:0.50"


def segment(y: np.ndarray, sr: int = ANALYSIS_SR, max_types: int = 6) -> list[Section]:
    """Beat-synchronous Laplacian segmentation. Returns time-ordered sections."""
    import librosa
    from sklearn.cluster import KMeans

    hop = 512
    if len(y) < sr * 8:
        return []

    try:
        _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False)
        if len(beats) < 12:
            return []
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop, n_mfcc=20)
        csync = librosa.util.sync(chroma, beats, aggregate=np.median)
        msync = librosa.util.sync(mfcc, beats, aggregate=np.mean)

        # Repetition graph (which beats sound like which other beats) ...
        rec = librosa.segment.recurrence_matrix(
            csync, width=3, mode="affinity", sym=True)
        rec = _median_filter(rec, 7)
        # ... plus a local path graph (adjacent beats belong together).
        path = _path_affinity(msync)

        mu = float(np.clip(path.mean() / (path.mean() + rec.mean() + 1e-9), 0.05, 0.95))
        A = mu * rec + (1 - mu) * path

        # Normalised Laplacian; the low eigenvectors encode segment membership.
        deg = A.sum(axis=1)
        Dinv = np.diag(1.0 / np.sqrt(deg + 1e-9))
        L = np.eye(A.shape[0]) - Dinv @ A @ Dinv
        evals, evecs = np.linalg.eigh(L)
        k = _choose_k(evals, max_types)
        X = evecs[:, :k]
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        labels = KMeans(n_clusters=k, n_init=8, random_state=0).fit_predict(X)
    except Exception:
        return []

    times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_sync = librosa.util.sync(rms[np.newaxis, :], beats, aggregate=np.mean)[0]

    sections: list[Section] = []
    start_idx = 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start_idx]:
            # sync yields len(beats)+1 labels but times has len(beats); clamp both
            s = float(times[min(start_idx, len(times) - 1)])
            e = float(times[min(i, len(times) - 1)])
            if e - s >= 1.0:
                energy = float(np.mean(rms_sync[start_idx:i]))
                sections.append(Section(s, e, int(labels[start_idx]), energy))
            start_idx = i
    return _merge_short(sections, min_len=6.0)


def repetition_index(sections: list[Section]) -> float:
    """How much of the track is repeated material. 1.0 = one idea on loop.

    High values are a loop-based arrangement; near-zero is through-composed.
    Combined with low microtiming variance it is a decent machine-made prior.
    """
    if not sections:
        return 0.0
    total = sum(s.duration for s in sections)
    if total <= 0:
        return 0.0
    by_label: dict[int, float] = {}
    for s in sections:
        by_label[s.label] = by_label.get(s.label, 0.0) + s.duration
    repeated = sum(d for lab, d in by_label.items()
                   if sum(1 for s in sections if s.label == lab) > 1)
    return float(repeated / total)


def choose_excerpts(duration: float, sections: list[Section],
                    policy: ExcerptPolicy) -> list[Excerpt]:
    """Pick the windows the encoders will see."""
    win = policy.seconds
    usable_start = min(policy.skip_head_s, max(duration - win, 0) / 2)
    usable_end = max(duration - policy.skip_tail_s - win, usable_start)

    if policy.structure_aware and sections:
        counts: dict[int, int] = {}
        for s in sections:
            counts[s.label] = counts.get(s.label, 0) + 1
        scored = sorted(
            sections,
            key=lambda s: (counts[s.label] * s.energy * min(s.duration / win, 1.5)),
            reverse=True,
        )
        picks: list[Excerpt] = []
        seen_labels: set[int] = set()
        for s in scored:
            if len(picks) >= policy.count:
                break
            # Centre the window in the section, then clamp into the usable span.
            start = np.clip(s.start + max(s.duration - win, 0) / 2, usable_start, usable_end)
            if any(abs(start - p.start) < policy.min_gap_s for p in picks):
                continue
            # Prefer a distinct section type for the second and third window so
            # the fused vector covers the verse as well as the chorus.
            if s.label in seen_labels and len(seen_labels) < min(policy.count, len(counts)):
                continue
            seen_labels.add(s.label)
            picks.append(Excerpt(float(start), win, f"section:{s.label}"))
        if len(picks) >= min(policy.count, 2):
            return sorted(picks, key=lambda e: e.start)

    return [
        Excerpt(float(np.clip(duration * pos - win / 2, usable_start, usable_end)),
                win, f"fallback:{pos:.2f}")
        for pos in policy.fallback_positions[: policy.count]
    ]


# ---------------------------------------------------------------------------

def _median_filter(A: np.ndarray, size: int) -> np.ndarray:
    from scipy.ndimage import median_filter
    return median_filter(A, size=(1, size))


def _path_affinity(feat: np.ndarray) -> np.ndarray:
    n = feat.shape[1]
    P = np.zeros((n, n), dtype=np.float32)
    d = np.linalg.norm(np.diff(feat, axis=1), axis=0)
    sigma = float(np.median(d) + 1e-9)
    w = np.exp(-d / sigma)
    idx = np.arange(n - 1)
    P[idx, idx + 1] = w
    P[idx + 1, idx] = w
    return P


def _choose_k(evals: np.ndarray, max_types: int) -> int:
    """Largest eigengap in the low end of the spectrum."""
    upper = min(max_types + 1, len(evals) - 1)
    if upper < 3:
        return 2
    gaps = np.diff(evals[:upper])
    return int(np.argmax(gaps[1:]) + 2)


def _merge_short(sections: list[Section], min_len: float) -> list[Section]:
    if not sections:
        return []
    out = [sections[0]]
    for s in sections[1:]:
        if s.duration < min_len and out:
            prev = out[-1]
            out[-1] = Section(prev.start, s.end, prev.label,
                              (prev.energy * prev.duration + s.energy * s.duration)
                              / max(prev.duration + s.duration, 1e-9))
        else:
            out.append(s)
    return out
