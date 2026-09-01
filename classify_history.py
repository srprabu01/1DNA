"""Loop over watch-history entries NOT yet in the library, ask the YouTube Data
API what CATEGORY each video is (authoritative), and add the ones that are Music.

Run: python classify_history.py <path-to-watch-history.json>
"""
import json
import re
import sys
import time

import requests

sys.path.insert(0, ".")
from app.db import connect, upsert_track, add_play
from app.ingest import youtube as Y
from app.ingest import youtube_api

CATS = {"1": "Film & Animation", "2": "Autos", "10": "Music", "15": "Pets",
        "17": "Sports", "18": "Short Movies", "19": "Travel", "20": "Gaming",
        "21": "Videoblogging", "22": "People & Blogs", "23": "Comedy",
        "24": "Entertainment", "25": "News & Politics", "26": "Howto & Style",
        "27": "Education", "28": "Science & Tech", "29": "Nonprofits",
        "30": "Movies", "43": "Shows"}


def _dur(iso):
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mn, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mn * 60 + s


def main(path):
    data = json.load(open(path, encoding="utf-8"))
    valid = [e for e in data if e.get("title", "").startswith("Watched") and e.get("time")]

    # candidates = entries NOT already classified as music by our heuristic
    cand = {}   # videoId -> list of watch dicts
    for e in valid:
        if Y._is_music(e):
            continue
        m = re.search(r"v=([\w-]{6,})", e.get("titleUrl", "") or "")
        if not m:
            continue
        subs = e.get("subtitles") or []
        cand.setdefault(m.group(1), []).append({
            "title": e["title"], "time": e["time"],
            "channel": subs[0].get("name", "") if subs else ""})

    ids = list(cand)
    print(f"not-yet-added videos to classify: {len(ids)}", flush=True)

    token = youtube_api._token()
    if not token:
        print("ERROR: YouTube not signed in.", flush=True)
        return

    cat_of = {}   # videoId -> categoryId
    dur_of = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        for attempt in range(3):
            r = requests.get("https://www.googleapis.com/youtube/v3/videos",
                             headers={"Authorization": f"Bearer {token}"},
                             params={"part": "snippet,contentDetails",
                                     "id": ",".join(chunk), "maxResults": 50},
                             timeout=30)
            if r.status_code == 401:
                token = youtube_api._token()
                continue
            r.raise_for_status()
            break
        for it in r.json().get("items", []):
            cat_of[it["id"]] = (it.get("snippet") or {}).get("categoryId")
            dur_of[it["id"]] = _dur((it.get("contentDetails") or {}).get("duration"))
        print(f"  classified {min(i+50,len(ids))}/{len(ids)}", flush=True)
        time.sleep(0.1)

    # tally what the skipped history actually IS
    from collections import Counter
    breakdown = Counter()
    for vid in ids:
        breakdown[CATS.get(cat_of.get(vid), "Unavailable/removed")] += 1
    print("\n=== what your not-yet-added history actually is ===", flush=True)
    for name, n in breakdown.most_common():
        print(f"  {n:>5}  {name}", flush=True)

    # add the Music ones (category 10). Songs (>=60s) added; count shorts.
    added_plays = added_songs = shorts = 0
    with connect() as con:
        for vid in ids:
            if cat_of.get(vid) != "10":
                continue
            if dur_of.get(vid, 0) and dur_of[vid] < 60:
                shorts += 1
                continue
            watches = cand[vid]
            artist, song = Y._split_artist_title(
                Y._clean_title(watches[0]["title"]), watches[0]["channel"])
            tid = upsert_track(con, title=song, artist=artist,
                               duration_ms=dur_of.get(vid, 0) * 1000 or None)
            if not tid:
                continue
            added_songs += 1
            for w in watches:
                add_play(con, tid, w["time"], "youtube")
                added_plays += 1

    with connect() as con:
        t = con.execute("SELECT COUNT(*) c FROM tracks").fetchone()["c"]
        p = con.execute("SELECT COUNT(*) c FROM plays").fetchone()["c"]
    print(f"\n=== added {added_songs} music videos ({added_plays} plays); "
          f"{shorts} music-shorts skipped ===", flush=True)
    print(f"library now: {t} unique videos, {p} total plays", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
