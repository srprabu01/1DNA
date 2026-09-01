"""Macro (listening behavior) and micro (audio DNA) analytics."""
import json
import math

from ..db import connect, rows_to_dicts

FEATURE_COLS = ["energy", "danceability", "valence", "acousticness", "brightness"]


def overview():
    with connect() as con:
        t = con.execute("SELECT COUNT(*) c FROM tracks").fetchone()["c"]
        p = con.execute("SELECT COUNT(*) c FROM plays").fetchone()["c"]
        a = con.execute("SELECT COUNT(DISTINCT artist) c FROM tracks").fetchone()["c"]
        f = con.execute(
            "SELECT COUNT(*) c FROM audio_features WHERE error IS NULL"
        ).fetchone()["c"]
        e = con.execute("SELECT COUNT(*) c FROM tracks WHERE enriched=1").fetchone()["c"]
        hours = con.execute(
            """SELECT SUM(COALESCE(t.duration_ms, 210000)) / 3600000.0 h
               FROM plays p JOIN tracks t ON t.id = p.track_id"""
        ).fetchone()["h"]
        by_source = rows_to_dicts(con.execute(
            "SELECT source, COUNT(*) plays FROM plays GROUP BY source"
        ).fetchall())
        span = con.execute(
            "SELECT MIN(played_at) lo, MAX(played_at) hi FROM plays"
        ).fetchone()
    return {
        "tracks": t, "plays": p, "artists": a, "analyzed": f, "enriched": e,
        "listening_hours": round(hours or 0, 1), "by_source": by_source,
        "first_play": span["lo"], "last_play": span["hi"],
    }


def top(kind="tracks", limit=25, days=None):
    where = ""
    params = []
    if days:
        where = "WHERE p.played_at >= datetime('now', ?)"
        params.append(f"-{int(days)} days")
    with connect() as con:
        if kind == "artists":
            rows = con.execute(
                f"""SELECT t.artist name, COUNT(*) plays,
                    COUNT(DISTINCT t.id) unique_tracks
                    FROM plays p JOIN tracks t ON t.id = p.track_id {where}
                    GROUP BY t.artist ORDER BY plays DESC LIMIT ?""",
                (*params, limit),
            ).fetchall()
        else:
            rows = con.execute(
                f"""SELECT t.id, t.title, t.artist, t.genre, COUNT(*) plays,
                    f.bpm, f.energy, f.valence, f.key, f.mode
                    FROM plays p JOIN tracks t ON t.id = p.track_id
                    LEFT JOIN audio_features f ON f.track_id = t.id {where}
                    GROUP BY t.id ORDER BY plays DESC LIMIT ?""",
                (*params, limit),
            ).fetchall()
    return rows_to_dicts(rows)


def patterns():
    """Hour-of-day and day-of-week listening heatmap + monthly trend."""
    with connect() as con:
        hod = rows_to_dicts(con.execute(
            """SELECT CAST(strftime('%H', played_at) AS INT) hour, COUNT(*) plays
               FROM plays GROUP BY hour ORDER BY hour"""
        ).fetchall())
        dow = rows_to_dicts(con.execute(
            """SELECT CAST(strftime('%w', played_at) AS INT) day, COUNT(*) plays
               FROM plays GROUP BY day ORDER BY day"""
        ).fetchall())
        monthly = rows_to_dicts(con.execute(
            """SELECT strftime('%Y-%m', played_at) month, COUNT(*) plays,
               COUNT(DISTINCT track_id) unique_tracks
               FROM plays GROUP BY month ORDER BY month"""
        ).fetchall())
        heat = rows_to_dicts(con.execute(
            """SELECT CAST(strftime('%w', played_at) AS INT) day,
               CAST(strftime('%H', played_at) AS INT) hour, COUNT(*) plays
               FROM plays GROUP BY day, hour"""
        ).fetchall())
    return {"by_hour": hod, "by_weekday": dow, "monthly": monthly, "heatmap": heat}


def genres():
    with connect() as con:
        dist = rows_to_dicts(con.execute(
            """SELECT t.genre, COUNT(*) plays FROM plays p
               JOIN tracks t ON t.id = p.track_id
               WHERE t.genre IS NOT NULL
               GROUP BY t.genre ORDER BY plays DESC LIMIT 15"""
        ).fetchall())
        eras = rows_to_dicts(con.execute(
            """SELECT substr(t.release_date, 1, 3) || '0s' era, COUNT(*) plays
               FROM plays p JOIN tracks t ON t.id = p.track_id
               WHERE length(t.release_date) >= 4
               GROUP BY era ORDER BY era"""
        ).fetchall())
    return {"distribution": dist, "eras": eras}


def audio_profile():
    """Your aggregate 'audio DNA': play-weighted means, distributions, keys, tempo."""
    with connect() as con:
        rows = con.execute(
            """SELECT f.*, COALESCE(p.n, 1) plays FROM audio_features f
               LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                 ON p.track_id = f.track_id
               WHERE f.error IS NULL"""
        ).fetchall()
    if not rows:
        return {"available": False}

    total_w = sum(r["plays"] for r in rows)
    profile = {}
    for col in FEATURE_COLS:
        vals = [(r[col], r["plays"]) for r in rows if r[col] is not None]
        if vals:
            profile[col] = round(sum(v * w for v, w in vals) / sum(w for _, w in vals), 3)

    bpms = [r["bpm"] for r in rows if r["bpm"]]
    keys = {}
    for r in rows:
        if r["key"]:
            label = f"{r['key']} {r['mode'] or ''}".strip()
            keys[label] = keys.get(label, 0) + r["plays"]

    hist = {}
    for col in FEATURE_COLS:
        buckets = [0] * 10
        for r in rows:
            if r[col] is not None:
                buckets[min(int(r[col] * 10), 9)] += 1
        hist[col] = buckets

    # taste diversity: normalized entropy of play distribution across tracks
    with connect() as con:
        counts = [r["n"] for r in con.execute(
            "SELECT COUNT(*) n FROM plays GROUP BY track_id"
        ).fetchall()]
    diversity = None
    if len(counts) > 1:
        tot = sum(counts)
        ent = -sum((c / tot) * math.log(c / tot) for c in counts)
        diversity = round(ent / math.log(len(counts)), 3)

    return {
        "available": True,
        "n_analyzed": len(rows),
        "profile": profile,
        "avg_bpm": round(sum(bpms) / len(bpms), 1) if bpms else None,
        "avg_loudness": round(
            sum(r["loudness_db"] for r in rows if r["loudness_db"] is not None)
            / max(1, sum(1 for r in rows if r["loudness_db"] is not None)), 1),
        "keys": dict(sorted(keys.items(), key=lambda kv: -kv[1])[:12]),
        "histograms": hist,
        "diversity": diversity,
        "mood_quadrant": _mood_quadrant(profile),
    }


def _mood_quadrant(profile):
    e, v = profile.get("energy"), profile.get("valence")
    if e is None or v is None:
        return None
    if e >= 0.5 and v >= 0.5:
        return "Energetic & Happy (hype)"
    if e >= 0.5:
        return "Energetic & Tense (intense)"
    if v >= 0.5:
        return "Calm & Happy (chill)"
    return "Calm & Sad (melancholic)"


def track_detail(track_id):
    with connect() as con:
        t = con.execute("SELECT * FROM tracks WHERE id=?", (track_id,)).fetchone()
        if not t:
            return None
        f = con.execute(
            "SELECT * FROM audio_features WHERE track_id=?", (track_id,)
        ).fetchone()
        plays = rows_to_dicts(con.execute(
            "SELECT played_at, source FROM plays WHERE track_id=? ORDER BY played_at DESC LIMIT 50",
            (track_id,),
        ).fetchall())
        lyr = con.execute(
            """SELECT sentiment, word_count, unique_ratio, top_words, explicit, error
               FROM lyrics WHERE track_id=?""", (track_id,)).fetchone()
    out = dict(t)
    out["features"] = dict(f) if f else None
    if out["features"] and out["features"].get("mfcc"):
        out["features"]["mfcc"] = json.loads(out["features"]["mfcc"])
    if lyr and lyr["word_count"] and not lyr["error"]:
        out["lyrics"] = dict(lyr)
        out["lyrics"]["top_words"] = json.loads(out["lyrics"]["top_words"] or "[]")
    else:
        out["lyrics"] = None
    out["plays"] = plays
    out["play_count"] = len(plays)
    return out
