"""The measured layer.

Everything in this module is a *measurement* — reproducible, explainable, and
independent of any model weights. It is deliberately separated from the learned
layer (``embed.py``) because the two fail in completely different ways, and a
recommendation you can defend needs to know which kind of number it is standing
on.

Depends on librosa + pyloudnorm. Nothing here needs a GPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from .config import ANALYSIS_SR

# ---------------------------------------------------------------------------
# Tonal
# ---------------------------------------------------------------------------

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler probe-tone profiles.
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot wheel. Neighbouring numbers are a fifth apart; A/B share a tonic set.
_CAMELOT_MAJOR = {"B": 1, "F#": 2, "C#": 3, "G#": 4, "D#": 5, "A#": 6,
                  "F": 7, "C": 8, "G": 9, "D": 10, "A": 11, "E": 12}
_CAMELOT_MINOR = {"G#": 1, "D#": 2, "A#": 3, "F": 4, "C": 5, "G": 6,
                  "D": 7, "A": 8, "E": 9, "B": 10, "F#": 11, "C#": 12}


def to_camelot(pitch_class: str, scale: str) -> str:
    """'F#', 'minor' -> '11A'. The lingua franca of harmonic mixing."""
    table = _CAMELOT_MINOR if scale == "minor" else _CAMELOT_MAJOR
    letter = "A" if scale == "minor" else "B"
    return f"{table[pitch_class]}{letter}"


def camelot_distance(a: str, b: str) -> int:
    """Steps around the wheel. 0 = same key, 1 = a compatible mix, >2 = a clash.

    Same number across A/B (relative major/minor) counts as 1, which is how DJs
    actually treat it.
    """
    if not a or not b:
        return 99
    na, la = int(a[:-1]), a[-1]
    nb, lb = int(b[:-1]), b[-1]
    ring = min((na - nb) % 12, (nb - na) % 12)
    return ring if la == lb else max(ring, 1) if ring == 0 else ring + 1


def estimate_key(chroma: np.ndarray) -> tuple[str, str, float]:
    """Correlate a mean chroma vector against all 24 rotated profiles."""
    profile = chroma.mean(axis=1)
    if profile.sum() <= 0:
        return "C", "major", 0.0
    profile = profile / profile.sum()
    scores: list[tuple[float, str, str]] = []
    for i in range(12):
        for name, template in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            rotated = np.roll(template, i)
            r = float(np.corrcoef(profile, rotated)[0, 1])
            scores.append((r, PITCH_CLASSES[i], name))
    scores.sort(reverse=True)
    best, runner = scores[0], scores[1]
    # Confidence is the margin over the next-best hypothesis, not the raw
    # correlation: a track that correlates 0.9 with two neighbouring keys is
    # ambiguous, and saying so is more useful than a high number.
    confidence = float(np.clip((best[0] - runner[0]) * 4.0, 0.0, 1.0))
    return best[1], best[2], confidence


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SignalFeatures:
    duration_s: float = 0.0

    # rhythm
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    bpm_half_double_ambiguous: bool = False
    beat_count: int = 0
    onset_rate_hz: float = 0.0
    microtiming_ms: float = 0.0     # std of onset deviation from the beat grid
    swing_ratio: float = 0.0        # 0.5 = straight, ~0.66 = triplet swing
    pulse_clarity: float = 0.0

    # tonal
    key: str = ""
    scale: str = ""
    camelot: str = ""
    key_confidence: float = 0.0
    key_stability: float = 0.0      # 1.0 = never modulates
    chroma_entropy: float = 0.0     # low = tonal/diatonic, high = atonal/noisy

    # loudness and dynamics
    integrated_lufs: float = 0.0
    loudness_range_lu: float = 0.0
    true_peak_dbtp: float = 0.0
    plr_db: float = 0.0             # peak-to-loudness; low = heavily limited
    crest_factor_db: float = 0.0
    dynamic_spread_lu: float = 0.0

    # spectrum
    spectral_centroid_hz: float = 0.0
    spectral_bandwidth_hz: float = 0.0
    rolloff85_hz: float = 0.0
    rolloff99_hz: float = 0.0
    spectral_flatness: float = 0.0
    hf_ratio: float = 0.0           # energy above 8 kHz
    sub_ratio: float = 0.0          # energy below 60 Hz
    harmonic_ratio: float = 0.0

    # stereo (measured on the original file, not the mono master)
    stereo_width: float = 0.0
    side_ratio: float = 0.0
    mono_compatible: bool = True

    # structure, filled in by structure.py
    section_count: int = 0
    mean_section_s: float = 0.0
    repetition_index: float = 0.0
    sections: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse(master_audio: np.ndarray, sr: int, stereo: np.ndarray | None = None) -> SignalFeatures:
    """Full measured-feature pass over one track.

    ``master_audio`` is mono float32 at ``sr``. ``stereo`` is an optional short
    2xN window from the original file, used only for the stereo image.
    """
    import librosa

    from .decode import resample_poly

    y = resample_poly(master_audio, sr, ANALYSIS_SR) if sr != ANALYSIS_SR else master_audio
    y = np.ascontiguousarray(y, dtype=np.float32)
    f = SignalFeatures(duration_s=float(len(master_audio) / sr))
    if len(y) < ANALYSIS_SR:  # under a second of audio; nothing to say
        return f

    hop = 512
    harm, perc = librosa.effects.hpss(y)

    # -- rhythm ---------------------------------------------------------
    onset_env = librosa.onset.onset_strength(y=perc, sr=ANALYSIS_SR, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=ANALYSIS_SR,
                                           hop_length=hop, units="frames")
    f.bpm = float(np.atleast_1d(tempo)[0])
    f.beat_count = int(len(beats))
    beat_times = librosa.frames_to_time(beats, sr=ANALYSIS_SR, hop_length=hop)

    # Tempogram ridge sharpness is a better confidence signal than beat count.
    tg = librosa.feature.tempogram(onset_envelope=onset_env, sr=ANALYSIS_SR, hop_length=hop)
    tg_mean = tg.mean(axis=1)
    if tg_mean.max() > 0:
        norm = tg_mean / tg_mean.max()
        f.pulse_clarity = float(1.0 - norm.mean())
    f.bpm_confidence = float(np.clip(f.pulse_clarity * 1.3, 0.0, 1.0))

    # Half/double ambiguity: check whether the tempogram likes 2x or 0.5x nearly
    # as much. Flagging it beats silently picking one, because a 174 BPM d'n'b
    # track logged as 87 will never match a running cadence query.
    ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
    f.bpm_half_double_ambiguous = _half_double_ambiguous(ac, f.bpm, ANALYSIS_SR, hop)

    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=ANALYSIS_SR,
                                        hop_length=hop, units="time")
    f.onset_rate_hz = float(len(onsets) / max(f.duration_s, 1e-9))
    f.microtiming_ms, f.swing_ratio = _groove(onsets, beat_times)

    # -- tonal ----------------------------------------------------------
    chroma = librosa.feature.chroma_cqt(y=harm, sr=ANALYSIS_SR, hop_length=hop)
    f.key, f.scale, f.key_confidence = estimate_key(chroma)
    f.camelot = to_camelot(f.key, f.scale)
    f.key_stability = _key_stability(chroma)
    col = chroma / (chroma.sum(axis=0, keepdims=True) + 1e-9)
    f.chroma_entropy = float(np.mean(-(col * np.log2(col + 1e-9)).sum(axis=0)) / np.log2(12))

    # -- loudness -------------------------------------------------------
    f.integrated_lufs, f.loudness_range_lu, f.dynamic_spread_lu = _loudness(master_audio, sr)
    peak = float(np.max(np.abs(_oversample(master_audio))) + 1e-12)
    f.true_peak_dbtp = float(20 * np.log10(peak))
    f.plr_db = float(f.true_peak_dbtp - f.integrated_lufs)
    rms = float(np.sqrt(np.mean(master_audio ** 2)) + 1e-12)
    f.crest_factor_db = float(20 * np.log10(np.max(np.abs(master_audio)) / rms + 1e-12))

    # -- spectrum -------------------------------------------------------
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=ANALYSIS_SR, n_fft=2048)
    power = S ** 2
    total = power.sum() + 1e-12
    f.spectral_centroid_hz = float(librosa.feature.spectral_centroid(S=S, sr=ANALYSIS_SR).mean())
    f.spectral_bandwidth_hz = float(librosa.feature.spectral_bandwidth(S=S, sr=ANALYSIS_SR).mean())
    f.rolloff85_hz = float(librosa.feature.spectral_rolloff(S=S, sr=ANALYSIS_SR, roll_percent=0.85).mean())
    f.rolloff99_hz = float(librosa.feature.spectral_rolloff(S=S, sr=ANALYSIS_SR, roll_percent=0.99).mean())
    f.spectral_flatness = float(librosa.feature.spectral_flatness(S=S).mean())
    f.hf_ratio = float(power[freqs >= 8000].sum() / total)
    f.sub_ratio = float(power[freqs <= 60].sum() / total)
    he = float(np.sum(harm ** 2))
    pe = float(np.sum(perc ** 2))
    f.harmonic_ratio = float(he / (he + pe + 1e-12))

    # -- stereo ---------------------------------------------------------
    if stereo is not None and stereo.ndim == 2 and stereo.shape[0] == 2:
        left, right = stereo[0], stereo[1]
        mid = (left + right) / 2
        side = (left - right) / 2
        me = float(np.mean(mid ** 2)) + 1e-12
        se = float(np.mean(side ** 2))
        f.side_ratio = float(se / (me + se))
        denom = np.sqrt(np.mean(left ** 2) * np.mean(right ** 2)) + 1e-12
        corr = float(np.mean(left * right) / denom)
        f.stereo_width = float(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))
        f.mono_compatible = corr > -0.2

    return f


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _half_double_ambiguous(ac: np.ndarray, bpm: float, sr: int, hop: int) -> bool:
    if bpm <= 0 or len(ac) < 4:
        return False

    def lag_for(b: float) -> int:
        return int(round(60.0 / b * sr / hop))

    def strength(b: float) -> float:
        lag = lag_for(b)
        return float(ac[lag]) if 0 < lag < len(ac) else 0.0

    base = strength(bpm)
    if base <= 0:
        return False
    rivals = [strength(bpm * 2), strength(bpm / 2)]
    return max(rivals) > 0.9 * base


def _groove(onsets: np.ndarray, beats: np.ndarray) -> tuple[float, float]:
    """Deviation of onsets from the beat grid, in ms, plus a swing estimate.

    This is the rubato signal: a sequenced loop lands within a couple of ms of
    the grid every time, a played performance does not. It feeds both the
    "human vs machine" prior and the synthetic-audio score.
    """
    if len(onsets) < 8 or len(beats) < 4:
        return 0.0, 0.0
    period = float(np.median(np.diff(beats)))
    if period <= 0:
        return 0.0, 0.0
    grid = (onsets - beats[0]) / period
    frac = grid - np.round(grid)
    microtiming = float(np.std(frac) * period * 1000.0)

    # Swing: where do the off-beat onsets sit between two beats?
    offbeat = grid - np.floor(grid)
    mid = offbeat[(offbeat > 0.3) & (offbeat < 0.8)]
    swing = float(np.median(mid)) if len(mid) >= 4 else 0.0
    return microtiming, swing


def _key_stability(chroma: np.ndarray, window: int = 400) -> float:
    """Fraction of windows agreeing with the global key. 1.0 = no modulation."""
    if chroma.shape[1] < window * 2:
        return 1.0
    global_key = estimate_key(chroma)[:2]
    agree = 0
    total = 0
    for start in range(0, chroma.shape[1] - window, window):
        local = estimate_key(chroma[:, start:start + window])[:2]
        agree += int(local == global_key)
        total += 1
    return float(agree / max(total, 1))


def _oversample(x: np.ndarray, factor: int = 4) -> np.ndarray:
    """Cheap true-peak estimate: inter-sample peaks live between the samples."""
    from scipy.signal import resample_poly
    if len(x) > 8_000_000:  # cap the cost on very long files
        x = x[:8_000_000]
    return resample_poly(x, factor, 1)


def _loudness(audio: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Integrated LUFS, EBU R128 loudness range, and short-term spread."""
    try:
        import pyloudnorm as pyln
    except ImportError:
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
        return float(20 * np.log10(rms)), 0.0, 0.0

    meter = pyln.Meter(sr)
    try:
        integrated = float(meter.integrated_loudness(audio))
    except Exception:
        integrated = -70.0
    if not np.isfinite(integrated):
        integrated = -70.0

    # 3 s short-term blocks, 1 s hop, gated at -20 LU below the mean.
    block = int(3 * sr)
    hop = int(1 * sr)
    st: list[float] = []
    for start in range(0, max(len(audio) - block, 0) + 1, hop):
        seg = audio[start:start + block]
        if len(seg) < block:
            break
        with np.errstate(divide="ignore"):
            val = float(meter.integrated_loudness(seg))
        if np.isfinite(val) and val > -70:
            st.append(val)
    if len(st) < 3:
        return integrated, 0.0, 0.0
    arr = np.array(st)
    gated = arr[arr > arr.mean() - 20.0]
    lra = float(np.percentile(gated, 95) - np.percentile(gated, 10))
    spread = float(np.std(arr))
    return integrated, lra, spread
