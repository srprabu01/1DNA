"""Musical structure analysis (MIR) from the cached preview audio.

Self-similarity based: detects section boundaries, estimates the most-repeated
segment (chorus/hook candidate), scores repetitiveness, and extracts the
energy envelope over time.
"""
import numpy as np

from ..config import CACHE_DIR


def _load(track_id):
    import librosa
    mp3 = CACHE_DIR / f"{track_id}.mp3"
    if not mp3.exists():
        raise FileNotFoundError("no cached preview — analyze this track first")
    y, sr = librosa.load(str(mp3), sr=22050, mono=True)
    return y, sr


def analyze_structure(track_id, n_segments=6):
    """Returns segments, chorus candidate, repetition score, energy curve."""
    import librosa

    y, sr = _load(track_id)
    dur = len(y) / sr

    # beat-synchronous chroma + mfcc stack for structure
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    if beats is not None and len(beats) > 8:
        chroma_s = librosa.util.sync(chroma, beats)
        mfcc_s = librosa.util.sync(mfcc, beats)
        frame_times = librosa.frames_to_time(beats, sr=sr)
    else:
        chroma_s, mfcc_s = chroma, mfcc
        frame_times = librosa.times_like(chroma[0], sr=sr)

    X = np.vstack([librosa.util.normalize(chroma_s, axis=0),
                   librosa.util.normalize(mfcc_s, axis=0)])

    # segment boundaries (agglomerative clustering on the feature stack)
    k = min(n_segments, max(2, X.shape[1] // 6))
    try:
        bounds = librosa.segment.agglomerative(X, k)
        bound_times = [float(frame_times[min(b, len(frame_times) - 1)]) for b in bounds]
    except Exception:
        bound_times = [0.0]
    bound_times = sorted(set([0.0] + bound_times + [dur]))

    # self-similarity (recurrence) on chroma for repetition + chorus detection
    R = librosa.segment.recurrence_matrix(
        librosa.util.normalize(chroma_s, axis=0), mode="affinity", sym=True)
    repetition = float(np.mean(R)) if R.size else 0.0

    # segment stats + chorus candidate = segment whose frames recur the most
    rms = librosa.feature.rms(y=y)[0]
    rms_t = librosa.times_like(rms, sr=sr)
    rec_strength = R.mean(axis=1) if R.size else np.zeros(X.shape[1])
    # sync() yields one more frame than boundary times — align the two
    n = min(len(frame_times), len(rec_strength))
    frame_times, rec_strength = frame_times[:n], rec_strength[:n]

    segments = []
    for a, b in zip(bound_times, bound_times[1:]):
        if b - a < 0.75:
            continue
        mask = (frame_times >= a) & (frame_times < b)
        emask = (rms_t >= a) & (rms_t < b)
        seg_energy = float(np.mean(rms[emask])) if emask.any() else 0.0
        seg_rec = float(np.mean(rec_strength[mask])) if mask.any() else 0.0
        segments.append({"start": round(a, 2), "end": round(b, 2),
                         "energy": seg_energy, "recurrence": seg_rec})
    if segments:
        emax = max(s["energy"] for s in segments) or 1.0
        for s in segments:
            s["energy"] = round(s["energy"] / emax, 3)
            s["recurrence"] = round(s["recurrence"], 3)
        chorus = max(segments, key=lambda s: 0.6 * s["recurrence"] + 0.4 * s["energy"])
        chorus_span = [chorus["start"], chorus["end"]]
    else:
        chorus_span = None

    # 1-second energy curve for the timeline
    step = max(1, int(len(rms) / max(1, int(dur))))
    curve = [round(float(np.mean(rms[i:i + step])), 4) for i in range(0, len(rms), step)]
    cmax = max(curve) or 1
    curve = [round(c / cmax, 3) for c in curve]

    return {
        "duration": round(dur, 1),
        "segments": segments,
        "chorus": chorus_span,
        "repetition_score": round(repetition, 3),
        "energy_curve": curve,
    }
