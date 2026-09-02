"""DSP worker — runs inside a child process.

Imports only numpy / librosa / the pure feature math (never app.main), so a
spawned child never re-runs init_db or the schema migration. Every value that
crosses the process boundary is a plain picklable type (int, str, dict).
"""
from __future__ import annotations

import os

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS")


def pin_threads():
    """Pin every BLAS/numba/OMP pool to one thread. Must run before numpy import
    to take effect, so we also set it in the parent before the pool is created."""
    for var in _THREAD_VARS:
        os.environ.setdefault(var, "1")


def warm_worker():
    """Pool initializer: pin threads, then pay the librosa import + numba JIT once
    per worker instead of on its first real track."""
    pin_threads()
    try:
        import librosa
        import numpy as np
        y = np.zeros(22050, dtype=np.float32)
        env = librosa.onset.onset_strength(y=y, sr=22050)
        librosa.beat.beat_track(onset_envelope=env, sr=22050)
        librosa.feature.chroma_cqt(y=y, sr=22050)
    except Exception:
        pass


def analyze_file(track_id, mp3_path, npy_path=None):
    """Load a cached preview and extract features.

    Returns (track_id, feats_dict | None, error | None). Catches Python errors;
    a hard C-level decoder crash kills the process instead — pebble replaces it
    and surfaces ProcessExpired for this one task.
    """
    pin_threads()
    try:
        import librosa
        import numpy as np

        from .features import _extract_features
        if npy_path and os.path.exists(npy_path):
            y = np.asarray(np.load(npy_path), dtype=np.float32)
            sr = 22050
        else:
            y, sr = librosa.load(str(mp3_path), sr=22050, mono=True)
        if len(y) < sr:
            return track_id, None, "preview too short"
        return track_id, _extract_features(y, sr), None
    except Exception as e:  # decode/DSP error (not a hard crash)
        return track_id, None, str(e)[:200]
