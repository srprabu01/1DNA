"""Decoding, content hashing, and the 48 kHz master cache.

The rest of the system asks this module three things: *what is this file*
(:func:`probe`), *give me the mono 48 kHz master* (:func:`load_master`), and
*give me a short stereo window* (:func:`load_stereo`). Everything else in
deepcut works on the arrays those return, never on the file directly.

Decoding goes through ``soundfile`` (libsndfile) first — modern libsndfile
decodes MP3, FLAC, WAV, AIFF and Ogg with no external binary — and falls back to
``librosa``/``audioread`` (ffmpeg) only for the formats libsndfile will not open.
The master is cached as fp16 keyed by content digest, so re-analysing or
re-embedding a track never decodes it twice.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MASTER_SR

_LOSSLESS_FORMATS = {"FLAC", "WAV", "WAVEX", "AIFF", "AIFC", "AU", "CAF", "W64"}
_LOSSLESS_EXT = {".flac", ".wav", ".aif", ".aiff", ".alac", ".ape", ".wv", ".tta"}


@dataclass
class AudioInfo:
    duration_s: float
    sample_rate: int
    channels: int
    codec: str
    bit_rate: int | None
    lossless: bool


# ---------------------------------------------------------------------------
# Resampling — the one function embed.py and dsp.py both import.
# ---------------------------------------------------------------------------

def resample_poly(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resample of a mono signal. Identity when the rates match."""
    x = np.asarray(x, dtype=np.float32)
    if int(sr_from) == int(sr_to) or x.size == 0:
        return x
    from math import gcd

    from scipy.signal import resample_poly as _rp

    g = gcd(int(sr_from), int(sr_to)) or 1
    up = int(sr_to) // g
    down = int(sr_from) // g
    return np.asarray(_rp(x, up, down), dtype=np.float32)


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def content_digest(path: Path, chunk: int = 1 << 20) -> str:
    """SHA-1 of the raw file bytes. Stable across renames and tag edits."""
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _safe_key(digest: str) -> str:
    """A digest can be an app id like ``app:1234`` — make it a legal filename."""
    if all(c.isalnum() or c in "-_" for c in digest):
        return digest
    return hashlib.sha1(digest.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Probe
# ---------------------------------------------------------------------------

def probe(path: Path) -> AudioInfo:
    """Duration, rate, channels, codec, and a lossless flag — cheaply."""
    path = Path(path)
    try:
        import soundfile as sf

        info = sf.info(str(path))
        fmt = (info.format or "").upper()
        subtype = (info.subtype or "").upper()
        lossless = fmt in _LOSSLESS_FORMATS or path.suffix.lower() in _LOSSLESS_EXT \
            or subtype.startswith("PCM") or subtype.startswith("FLOAT")
        size = path.stat().st_size if path.exists() else 0
        bit_rate = int(size * 8 / info.duration) if info.duration and size else None
        codec = fmt or path.suffix.lstrip(".").upper()
        return AudioInfo(
            duration_s=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
            codec=codec,
            bit_rate=bit_rate,
            lossless=bool(lossless),
        )
    except Exception:
        pass
    # Fallback: let librosa (audioread/ffmpeg) at least give a duration.
    try:
        import librosa

        dur = float(librosa.get_duration(path=str(path)))
    except Exception:
        dur = 0.0
    return AudioInfo(
        duration_s=dur,
        sample_rate=0,
        channels=0,
        codec=path.suffix.lstrip(".").upper(),
        bit_rate=None,
        lossless=path.suffix.lower() in _LOSSLESS_EXT,
    )


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

def _decode(path: Path, always_2d: bool = True) -> tuple[np.ndarray, int]:
    """Return (samples, sr). Samples are (N, channels) float32 when 2d."""
    path = Path(path)
    try:
        import soundfile as sf

        data, sr = sf.read(str(path), dtype="float32", always_2d=always_2d)
        return np.asarray(data, dtype=np.float32), int(sr)
    except Exception:
        import librosa

        y, sr = librosa.load(str(path), sr=None, mono=not always_2d)
        y = np.asarray(y, dtype=np.float32)
        if always_2d:
            y = y[:, None] if y.ndim == 1 else y.T  # librosa mono→(N,1), stereo (2,N)→(N,2)
        return y, int(sr)


def load_master(path: Path, cache_dir: Path | None = None, *,
                sr: int = MASTER_SR, digest: str | None = None) -> tuple[np.ndarray, int]:
    """Mono float32 at ``sr`` (48 kHz by default), cached as fp16 by digest."""
    cache_file: Path | None = None
    if cache_dir is not None and digest:
        cache_dir = Path(cache_dir)
        cache_file = cache_dir / f"{_safe_key(digest)}.f16.npy"
        if cache_file.exists():
            try:
                return np.asarray(np.load(cache_file), dtype=np.float32), sr
            except Exception:
                pass  # corrupt cache entry; decode afresh below

    data, native_sr = _decode(path, always_2d=True)
    mono = data.mean(axis=1) if data.ndim == 2 else data
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if native_sr != sr:
        mono = resample_poly(mono, native_sr, sr)

    if cache_file is not None:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, mono.astype(np.float16))
        except Exception:
            pass
    return mono, sr


def load_stereo(path: Path, seconds: float = 30.0) -> np.ndarray | None:
    """A short (2, N) window at native rate for the stereo-image measurement.

    Returns ``None`` for mono sources or on any decode failure — the stereo
    features are optional and the caller treats absence as "not measured".
    """
    try:
        data, sr = _decode(path, always_2d=True)  # (N, channels)
    except Exception:
        return None
    if data.ndim != 2 or data.shape[1] < 2:
        return None
    n = int(seconds * sr) if seconds else data.shape[0]
    window = data[:n] if n else data
    return np.ascontiguousarray(window[:, :2].T, dtype=np.float32)  # (2, N)


# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------

def read_tags(path: Path) -> dict:
    """Best-effort metadata via mutagen. Returns ``{}`` when it is unavailable
    or the file carries no tags (the app's cached previews have none)."""
    try:
        import mutagen  # noqa: F401
        from mutagen import File as MutagenFile
    except Exception:
        return {}
    try:
        audio = MutagenFile(str(path), easy=True)
    except Exception:
        return {}
    if audio is None or not getattr(audio, "tags", None):
        return {}

    def first(*keys: str) -> str | None:
        for key in keys:
            val = audio.tags.get(key)
            if val:
                return str(val[0]) if isinstance(val, list) else str(val)
        return None

    tags = {
        "title": first("title"),
        "artist": first("artist"),
        "album": first("album"),
        "album_artist": first("albumartist", "album artist"),
        "date": first("date", "year", "originaldate"),
        "track": first("tracknumber"),
        "isrc": first("isrc"),
        "musicbrainz_trackid": first("musicbrainz_trackid"),
    }
    return {k: v for k, v in tags.items() if v}
