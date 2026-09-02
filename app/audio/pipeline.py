"""Parallel audio analysis.

Stage 1 (threads) downloads/caches previews; stage 2 (processes) runs the CPU-bound
DSP across cores. The PARENT process is the sole SQLite writer (the threading.Lock
in db.py gives no cross-process exclusion), batching results with executemany. A
corrupt preview that hard-crashes a worker is quarantined to audio_features.error
via pebble instead of aborting the whole run; without pebble it falls back to a
correct serial pass.
"""
from __future__ import annotations

import os
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from ..db import connect
from . import _cache
from ._worker import analyze_file, pin_threads, warm_worker

_COLS = ("track_id", "bpm", "key", "mode", "camelot", "key_confidence", "energy",
         "loudness_db", "danceability", "valence", "acousticness", "brightness",
         "instrumentalness", "speechiness", "liveness", "spectral_centroid",
         "spectral_rolloff", "spectral_bandwidth", "spectral_contrast",
         "spectral_flatness", "zero_crossing_rate", "dynamic_range", "harmonic_ratio",
         "tempo_confidence", "beat_regularity", "onset_rate", "time_signature", "mfcc")


def _select(retry_errors, limit):
    cond = "(f.track_id IS NULL OR f.error IS NOT NULL)" if retry_errors else "f.track_id IS NULL"
    lim = f"LIMIT {int(limit)}" if limit else ""
    with connect() as con:
        rows = con.execute(
            f"""SELECT t.id, t.preview_url, t.deezer_id FROM tracks t
                LEFT JOIN audio_features f ON f.track_id = t.id
                LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                  ON p.track_id = t.id
                WHERE t.preview_url IS NOT NULL AND {cond}
                ORDER BY COALESCE(p.n, 0) DESC {lim}""").fetchall()
    return [dict(r) for r in rows]


def _remaining():
    with connect() as con:
        return con.execute(
            """SELECT COUNT(*) c FROM tracks t LEFT JOIN audio_features f ON f.track_id = t.id
               WHERE t.preview_url IS NOT NULL AND f.track_id IS NULL""").fetchone()["c"]


def _feat_tuple(track_id, f):
    return (track_id,) + tuple(f[c] for c in _COLS[1:])


def _flush(feat_rows, err_rows):
    if not feat_rows and not err_rows:
        return
    with connect() as con:
        if feat_rows:
            con.executemany(
                f"INSERT OR REPLACE INTO audio_features ({','.join(_COLS)}) "
                f"VALUES ({','.join('?' * len(_COLS))})", feat_rows)
        if err_rows:
            con.executemany(
                "INSERT OR REPLACE INTO audio_features (track_id, error) VALUES (?,?)", err_rows)


def analyze_batch_parallel(retry_errors=False, limit=None, dsp_workers=None,
                           dl_workers=12, flush_every=100, timeout=45):
    """Analyze all (or `limit`) enriched-but-unanalyzed tracks in parallel.
    Returns {analyzed, errors, remaining} — same shape as the serial analyze_batch."""
    rows = _select(retry_errors, limit)
    if not rows:
        return {"analyzed": 0, "errors": 0, "remaining": _remaining()}

    pin_threads()  # set in the parent so spawned children inherit the pin
    workers = dsp_workers or max(1, (os.cpu_count() or 2) - 1)
    feat_rows, err_rows = [], []
    counts = {"analyzed": 0, "errors": 0}

    def collect(track_id, feats, err):
        if feats:
            feat_rows.append(_feat_tuple(track_id, feats))
            counts["analyzed"] += 1
        else:
            err_rows.append((track_id, err or "unknown"))
            counts["errors"] += 1
        if len(feat_rows) + len(err_rows) >= flush_every:
            _flush(feat_rows, err_rows)
            feat_rows.clear()
            err_rows.clear()

    def download(row):
        mp3, npy, err = _cache.ensure_cached(row)
        return row, (str(mp3) if mp3 else None), (str(npy) if npy else None), err

    try:
        from pebble import ProcessExpired, ProcessPool
        from concurrent.futures import TimeoutError as FutureTimeout
        pool = ProcessPool(max_workers=workers, max_tasks=300, initializer=warm_worker)
    except Exception:
        pool = None

    if pool is None:                     # ---- serial fallback (still cached DSP) ----
        with ThreadPoolExecutor(max_workers=dl_workers) as dlx:
            for row, mp3, npy, err in dlx.map(download, rows):
                if err or not mp3:
                    collect(row["id"], None, err or "download failed")
                    continue
                _, feats, e = analyze_file(row["id"], mp3, npy)
                collect(row["id"], feats, e)
        _flush(feat_rows, err_rows)
        return {**counts, "remaining": _remaining()}

    max_inflight = workers * 3
    inflight = {}

    def harvest(target):
        while len(inflight) > target:
            done, _ = wait(list(inflight), timeout=timeout * 2, return_when=FIRST_COMPLETED)
            if not done:
                break
            for fut in done:
                tid = inflight.pop(fut)
                try:
                    rid, feats, e = fut.result()
                    collect(rid, feats, e)
                except ProcessExpired:
                    collect(tid, None, "decode crash (isolated)")
                except FutureTimeout:
                    collect(tid, None, "analysis timeout")
                except Exception as ex:
                    collect(tid, None, str(ex)[:120])

    with pool, ThreadPoolExecutor(max_workers=dl_workers) as dlx:
        for row, mp3, npy, err in dlx.map(download, rows):
            if err or not mp3:
                collect(row["id"], None, err or "download failed")
                continue
            fut = pool.schedule(analyze_file, args=(row["id"], mp3, npy), timeout=timeout)
            inflight[fut] = row["id"]
            harvest(max_inflight)
        harvest(0)

    _flush(feat_rows, err_rows)
    return {**counts, "remaining": _remaining()}
