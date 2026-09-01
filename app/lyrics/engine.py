"""Lyrics layer — fetch lyrics (lyrics.ovh, free/no-key) and analyse them:
sentiment, vocabulary richness, top words, explicit flag, and the
audio-vs-lyric mood mismatch (happy sound / sad words and vice-versa)."""
import re
import time
from collections import Counter

import requests

from ..db import connect, rows_to_dicts

_STOP = set("""a an and the to of in on at is it its it's i i'm im you your youre my me we
us he she they them his her their our so if or but for with as be been am are was were do
does did done have has had not no yes oh yeah la na uh ooh gonna wanna gotta cause 'cause
this that these those there here what when where who why how all any some just now then
up down out off over under again more most very can will would could should get got go
going know like love got up down all out""".split())

_EXPLICIT = set("""fuck fucking shit bitch nigga niggas ass dick pussy cunt motherfucker
whore slut damn hell bastard cock""".split())


def _analyze_text(text):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    words = re.findall(r"[a-zA-Z']+", text.lower())
    total = len(words)
    if total < 5:
        raise ValueError("lyrics too short / instrumental")
    content = [w for w in words if w not in _STOP and len(w) > 2]
    counts = Counter(content)
    unique_ratio = len(set(words)) / total
    sentiment = SentimentIntensityAnalyzer().polarity_scores(text)["compound"]
    explicit = 1 if any(w in _EXPLICIT for w in words) else 0
    return {
        "word_count": total,
        "unique_ratio": round(unique_ratio, 3),
        "sentiment": round(sentiment, 3),
        "explicit": explicit,
        "top_words": [{"word": w, "n": n} for w, n in counts.most_common(12)],
    }


def fetch_one(track_id, title, artist):
    primary = artist.split(",")[0].strip()
    r = requests.get(
        f"https://api.lyrics.ovh/v1/{requests.utils.quote(primary)}/{requests.utils.quote(title)}",
        timeout=20)
    if r.status_code != 200:
        raise LookupError("lyrics not found")
    text = (r.json().get("lyrics") or "").strip()
    if not text:
        raise LookupError("lyrics not found")
    import json
    stats = _analyze_text(text)
    with connect() as con:
        con.execute(
            """INSERT OR REPLACE INTO lyrics
               (track_id, text, sentiment, word_count, unique_ratio, top_words, explicit, error)
               VALUES (?,?,?,?,?,?,?,NULL)""",
            (track_id, text, stats["sentiment"], stats["word_count"],
             stats["unique_ratio"], json.dumps(stats["top_words"]), stats["explicit"]))
    return stats


def fetch_batch(limit=10):
    """Fetch + analyse lyrics for the most-played tracks lacking them."""
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            """SELECT t.id, t.title, t.artist FROM tracks t
               LEFT JOIN lyrics l ON l.track_id = t.id
               LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                 ON p.track_id = t.id
               WHERE l.track_id IS NULL
               ORDER BY COALESCE(p.n,0) DESC LIMIT ?""", (limit,)).fetchall())
    done = missing = 0
    for r in rows:
        try:
            fetch_one(r["id"], r["title"], r["artist"])
            done += 1
        except Exception as e:
            with connect() as con:
                con.execute(
                    "INSERT OR REPLACE INTO lyrics (track_id, error) VALUES (?,?)",
                    (r["id"], str(e)[:120]))
            missing += 1
        time.sleep(0.3)
    with connect() as con:
        remaining = con.execute(
            "SELECT COUNT(*) c FROM tracks t LEFT JOIN lyrics l ON l.track_id=t.id WHERE l.track_id IS NULL"
        ).fetchone()["c"]
    return {"analyzed": done, "missing": missing, "remaining": remaining}


def overview():
    """Aggregate lyric analytics across your library, incl. audio-vs-lyric mismatch."""
    import json
    with connect() as con:
        rows = rows_to_dicts(con.execute(
            """SELECT l.*, f.valence, t.title, t.artist,
                      COALESCE(p.n,1) plays
               FROM lyrics l JOIN tracks t ON t.id=l.track_id
               LEFT JOIN audio_features f ON f.track_id=l.track_id AND f.error IS NULL
               LEFT JOIN (SELECT track_id, COUNT(*) n FROM plays GROUP BY track_id) p
                 ON p.track_id=l.track_id
               WHERE l.error IS NULL AND l.word_count IS NOT NULL""").fetchall())
    if not rows:
        return {"available": False}

    wordbag = Counter()
    for r in rows:
        for w in json.loads(r["top_words"] or "[]"):
            wordbag[w["word"]] += w["n"] * r["plays"]

    sentiments = [r["sentiment"] for r in rows if r["sentiment"] is not None]
    explicit_n = sum(1 for r in rows if r["explicit"])

    # audio-vs-lyric mismatch: bright audio (valence>.55) but sad words (sentiment<-.2)
    mismatches = []
    for r in rows:
        if r["valence"] is None or r["sentiment"] is None:
            continue
        if r["valence"] >= 0.55 and r["sentiment"] <= -0.2:
            kind = "happy sound, sad words"
        elif r["valence"] <= 0.4 and r["sentiment"] >= 0.3:
            kind = "dark sound, upbeat words"
        else:
            continue
        gap = abs(r["valence"] - (r["sentiment"] + 1) / 2)
        mismatches.append({"title": r["title"], "artist": r["artist"],
                           "kind": kind, "gap": round(gap, 3),
                           "valence": r["valence"], "sentiment": r["sentiment"]})
    mismatches.sort(key=lambda m: -m["gap"])

    return {
        "available": True,
        "n_tracks": len(rows),
        "avg_sentiment": round(sum(sentiments) / len(sentiments), 3) if sentiments else None,
        "explicit_pct": round(100 * explicit_n / len(rows), 1),
        "avg_vocab_richness": round(
            sum(r["unique_ratio"] for r in rows) / len(rows), 3),
        "word_cloud": [{"word": w, "weight": n} for w, n in wordbag.most_common(40)],
        "mismatches": mismatches[:10],
    }
