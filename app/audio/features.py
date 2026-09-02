"""Signal-level ("micro") audio analysis with librosa.

Downloads each track's 30-second preview MP3 and extracts:
tempo, musical key/mode, energy, loudness, danceability & valence estimates,
acousticness, brightness, spectral shape, dynamics, and MFCC timbre vector.
"""
import json

import numpy as np
import requests

from ..config import CACHE_DIR
from ..db import connect
from .camelot import to_camelot

KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Schmuckler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _estimate_key(chroma_mean):
    """Returns (key, mode, confidence 0..1). Confidence = margin of best over 2nd best."""
    scores = []
    for shift in range(12):
        rolled = np.roll(chroma_mean, -shift)
        for profile, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            scores.append((float(np.corrcoef(rolled, profile)[0, 1]), KEYS[shift], mode))
    scores.sort(reverse=True)
    best = scores[0]
    second = scores[1][0]
    conf = _clip01((best[0] - second) / 0.35)
    return best[1], best[2], round(conf, 3)


def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))


def analyze_preview(preview_url, track_id):
    import librosa  # deferred: heavy import

    mp3 = CACHE_DIR / f"{track_id}.mp3"
    if not mp3.exists():
        r = requests.get(preview_url, timeout=30)
        r.raise_for_status()
        mp3.write_bytes(r.content)

    y, sr = librosa.load(str(mp3), sr=22050, mono=True)
    if len(y) < sr:  # under 1 second of audio — junk
        raise ValueError("preview too short")
    return _extract_features(y, sr)


def _extract_features(y, sr):
    """Pure DSP over a mono 22.05 kHz signal -> the feature dict. Identical math to
    before; no network/disk/DB, so it is safe to run inside a worker process."""
    import librosa

    # ---- harmonic / percussive separation (used by several features) ----
    y_h, y_p = librosa.effects.hpss(y)
    e_h = float(np.mean(y_h ** 2))
    e_p = float(np.mean(y_p ** 2))
    harmonic_ratio = round(e_h / (e_h + e_p + 1e-9), 3)

    # ---- rhythm ----
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times) > 3:
        ibi = np.diff(beat_times)                       # inter-beat intervals
        beat_regularity = round(_clip01(1.0 - (np.std(ibi) / (np.mean(ibi) + 1e-9))), 3)
        # tempo confidence: strength of the dominant peak in the tempogram autocorr
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env) // 2)
        tempo_confidence = round(_clip01(float(np.max(ac[1:]) / (ac[0] + 1e-9))), 3)
    else:
        beat_regularity = tempo_confidence = 0.0
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_rate = round(len(onsets) / (len(y) / sr), 2)   # onsets per second
    # crude time-signature guess from beats-per-bar via onset autocorr on beats
    time_signature = 4
    if len(beat_times) > 8:
        strengths = onset_env[np.clip(beats, 0, len(onset_env) - 1)]
        for cand in (3, 4):
            groups = strengths[:len(strengths) // cand * cand].reshape(-1, cand)
            if len(groups) and np.argmax(groups.mean(axis=0)) == 0:
                time_signature = cand
                break

    # ---- loudness / dynamics ----
    rms = librosa.feature.rms(y=y)[0]
    loudness_db = float(20 * np.log10(max(float(np.mean(rms)), 1e-6)))
    energy = _clip01((loudness_db + 35.0) / 30.0)
    dynamic_range = float(20 * np.log10(max(float(np.max(rms)), 1e-6) /
                                        max(float(np.percentile(rms, 10)), 1e-6)))

    # ---- spectrum / timbre ----
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    brightness = _clip01(cent / 4000.0)

    # ---- harmony / key ----
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr)
    key, mode, key_confidence = _estimate_key(np.mean(chroma, axis=1))
    camelot = to_camelot(key, mode)

    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)

    # ---- heuristic 0-1 scores (consistent within a library) ----
    beat_strength = _clip01(float(np.mean(onset_env)) / 3.0)
    tempo_fit = float(np.exp(-((tempo - 120.0) ** 2) / (2 * 35.0 ** 2)))
    danceability = _clip01(0.45 * beat_strength + 0.30 * tempo_fit +
                           0.15 * energy + 0.10 * beat_regularity)

    mode_bonus = 0.20 if mode == "major" else 0.0
    valence = _clip01(0.35 * energy + 0.25 * brightness +
                      0.20 * _clip01(tempo / 160.0) + mode_bonus)

    acousticness = _clip01(1.0 - 0.6 * _clip01(rolloff / 6000.0) - 0.4 * _clip01(zcr / 0.15))

    # instrumentalness: high harmonic content + low speech-like modulation
    instrumentalness = _clip01(0.6 * harmonic_ratio + 0.4 * (1 - _clip01(flatness / 0.1)))
    # speechiness: high ZCR + high flatness + irregular beats ≈ spoken/rap
    speechiness = _clip01(0.5 * _clip01(zcr / 0.12) + 0.3 * _clip01(flatness / 0.05) +
                          0.2 * (1 - beat_regularity))
    # liveness: high spectral flatness + wide dynamics + noise floor ≈ live/crowd
    liveness = _clip01(0.5 * _clip01(flatness / 0.06) + 0.5 * _clip01(dynamic_range / 25.0))

    return {
        "bpm": round(tempo, 1),
        "key": key,
        "mode": mode,
        "camelot": camelot,
        "key_confidence": key_confidence,
        "energy": round(energy, 3),
        "loudness_db": round(loudness_db, 1),
        "danceability": round(danceability, 3),
        "valence": round(valence, 3),
        "acousticness": round(acousticness, 3),
        "brightness": round(brightness, 3),
        "instrumentalness": round(instrumentalness, 3),
        "speechiness": round(speechiness, 3),
        "liveness": round(liveness, 3),
        "spectral_centroid": round(cent, 1),
        "spectral_rolloff": round(rolloff, 1),
        "spectral_bandwidth": round(bandwidth, 1),
        "spectral_contrast": round(contrast, 2),
        "spectral_flatness": round(flatness, 4),
        "zero_crossing_rate": round(zcr, 4),
        "dynamic_range": round(dynamic_range, 1),
        "harmonic_ratio": harmonic_ratio,
        "tempo_confidence": tempo_confidence,
        "beat_regularity": beat_regularity,
        "onset_rate": onset_rate,
        "time_signature": time_signature,
        "mfcc": json.dumps([round(float(v), 2) for v in mfcc]),
    }


def analyze_batch(limit=8, retry_errors=False):
    """Analyze the next batch of enriched-but-unanalyzed tracks, most-played first.

    retry_errors=True also re-attempts tracks whose previous analysis errored
    (e.g. an expired preview URL that can now be refreshed).
    """
    cond = "(f.track_id IS NULL OR f.error IS NOT NULL)" if retry_errors else "f.track_id IS NULL"
    with connect() as con:
        rows = con.execute(
            f"""SELECT t.id, t.preview_url, t.deezer_id FROM tracks t
               LEFT JOIN audio_features f ON f.track_id = t.id
               LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                 ON p.track_id = t.id
               WHERE t.preview_url IS NOT NULL AND {cond}
               ORDER BY COALESCE(p.n, 0) DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    done = errors = 0
    for row in rows:
        try:
            try:
                feats = analyze_preview(row["preview_url"], row["id"])
            except Exception as first:
                # Deezer preview URLs are signed & expire → refresh and retry once
                if row["deezer_id"] and ("403" in str(first) or "Forbidden" in str(first)
                                         or "404" in str(first)):
                    from ..enrich.deezer import fresh_preview
                    fresh = fresh_preview(row["deezer_id"])
                    if fresh:
                        with connect() as con:
                            con.execute("UPDATE tracks SET preview_url=? WHERE id=?",
                                        (fresh, row["id"]))
                        feats = analyze_preview(fresh, row["id"])
                    else:
                        raise
                else:
                    raise
            with connect() as con:
                con.execute(
                    """INSERT OR REPLACE INTO audio_features
                       (track_id, bpm, key, mode, camelot, key_confidence, energy,
                        loudness_db, danceability, valence, acousticness, brightness,
                        instrumentalness, speechiness, liveness, spectral_centroid,
                        spectral_rolloff, spectral_bandwidth, spectral_contrast,
                        spectral_flatness, zero_crossing_rate, dynamic_range,
                        harmonic_ratio, tempo_confidence, beat_regularity, onset_rate,
                        time_signature, mfcc)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (row["id"], feats["bpm"], feats["key"], feats["mode"],
                     feats["camelot"], feats["key_confidence"], feats["energy"],
                     feats["loudness_db"], feats["danceability"], feats["valence"],
                     feats["acousticness"], feats["brightness"], feats["instrumentalness"],
                     feats["speechiness"], feats["liveness"], feats["spectral_centroid"],
                     feats["spectral_rolloff"], feats["spectral_bandwidth"],
                     feats["spectral_contrast"], feats["spectral_flatness"],
                     feats["zero_crossing_rate"], feats["dynamic_range"],
                     feats["harmonic_ratio"], feats["tempo_confidence"],
                     feats["beat_regularity"], feats["onset_rate"],
                     feats["time_signature"], feats["mfcc"]),
                )
            done += 1
        except Exception as e:
            with connect() as con:
                con.execute(
                    "INSERT OR REPLACE INTO audio_features (track_id, error) VALUES (?,?)",
                    (row["id"], str(e)[:200]),
                )
            errors += 1

    with connect() as con:
        remaining = con.execute(
            """SELECT COUNT(*) c FROM tracks t
               LEFT JOIN audio_features f ON f.track_id = t.id
               WHERE t.preview_url IS NOT NULL AND f.track_id IS NULL"""
        ).fetchone()["c"]
    return {"analyzed": done, "errors": errors, "remaining": remaining}
