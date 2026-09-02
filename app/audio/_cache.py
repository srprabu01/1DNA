"""Preview download + on-disk cache.

Keyed by deezer_id (so the cache survives a DB rebuild), with the legacy
per-track-id filename honoured for back-compat. Writes are atomic (tempfile +
os.replace) and size-checked, so a killed run never leaves a torn file that reads
as a valid cache hit, and a corrupt file is never a permanent hit.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time

import requests

from ..config import CACHE_DIR

_SESSION = requests.Session()
_ADAPTER = requests.adapters.HTTPAdapter(pool_connections=16, pool_maxsize=16)
_SESSION.mount("https://", _ADAPTER)
_SESSION.mount("http://", _ADAPTER)

# Deezer's public API caps at ~50 req / 5 s; keep refreshes well under that
# regardless of how many download threads are running.
_REFRESH_SEM = threading.Semaphore(3)
_GAP_LOCK = threading.Lock()
_LAST_REFRESH = [0.0]
_MIN_GAP = 0.125  # ~8 refresh req/s

WANT_NPY = os.environ.get("MUSIC_DECODED_CACHE") == "1"
_MIN_BYTES = 4096


def _key(row):
    dz = row.get("deezer_id")
    return f"dz{dz}" if dz else f"t{row['id']}"


def _atomic_write(path, data):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _download(url):
    try:
        r = _SESSION.get(url, timeout=30)
        if r.status_code >= 400:
            return None, str(r.status_code)
        return r.content, None
    except Exception as e:
        return None, str(e)[:120]


def _throttled_refresh(deezer_id):
    from ..enrich.deezer import fresh_preview
    with _REFRESH_SEM:
        with _GAP_LOCK:
            gap = time.time() - _LAST_REFRESH[0]
            if gap < _MIN_GAP:
                time.sleep(_MIN_GAP - gap)
            _LAST_REFRESH[0] = time.time()
        return fresh_preview(deezer_id)


def ensure_cached(row):
    """Return (mp3_path | None, npy_path | None, error | None). Never raises."""
    dz_mp3 = CACHE_DIR / (_key(row) + ".mp3")
    npy = CACHE_DIR / (_key(row) + ".npy")
    legacy = CACHE_DIR / f"{row['id']}.mp3"  # the original per-track-id cache

    for cached in (dz_mp3, legacy):
        if cached.exists() and cached.stat().st_size > _MIN_BYTES:
            return cached, (npy if WANT_NPY and npy.exists() else None), None

    url = row.get("preview_url")
    if not url:
        return None, None, "no preview url"

    data, err = _download(url)
    if err and row.get("deezer_id") and any(c in err for c in ("403", "404", "Forbidden")):
        fresh = _throttled_refresh(row["deezer_id"])
        if fresh:
            data, err = _download(fresh)
            if not err:
                try:
                    from ..db import connect
                    with connect() as con:
                        con.execute("UPDATE tracks SET preview_url=? WHERE id=?",
                                    (fresh, row["id"]))
                except Exception:
                    pass
    if err:
        return None, None, err
    if not data or len(data) <= _MIN_BYTES:
        return None, None, "preview too small"
    _atomic_write(dz_mp3, data)
    return dz_mp3, None, None
