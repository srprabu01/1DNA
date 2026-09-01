"""List the videos YouTube classified as NON-music (removed from the library).
Writes a CSV and prints a grouped summary with examples."""
import csv
import json
import re
import sys
import warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from app.ingest import youtube, youtube_api

CATS = {"1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
        "15": "Pets & Animals", "17": "Sports", "18": "Short Movies",
        "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging",
        "22": "People & Blogs (vlogs)", "23": "Comedy", "24": "Entertainment",
        "25": "News & Politics", "26": "Howto & Style", "27": "Education",
        "28": "Science & Technology", "29": "Nonprofits", "30": "Movies",
        "43": "Shows"}

path, out_csv = sys.argv[1], sys.argv[2]
data = json.load(open(path, encoding="utf-8"))

by_vid = {}
for e in data:
    if not e.get("title", "").startswith("Watched ") or not e.get("time"):
        continue
    m = re.search(r"[?&]v=([\w-]{6,})", e.get("titleUrl", "") or "")
    if not m:
        continue
    subs = e.get("subtitles") or []
    v = by_vid.setdefault(m.group(1), {"title": e["title"][8:],
                          "channel": subs[0].get("name", "") if subs else "",
                          "watches": 0, "url": e.get("titleUrl", ""), "entry": e})
    v["watches"] += 1

print(f"classifying {len(by_vid)} unique videos…", flush=True)
cls = youtube_api.classify_ids(list(by_vid))

# REMOVED = NOT (heuristic-music OR YouTube-category-Music). i.e. genuinely non-music.
groups = defaultdict(list)
for vid, info in by_vid.items():
    c = cls.get(vid)
    if youtube._is_music(info["entry"]) or (c and c["category"] == "10"):
        continue  # music — kept (union)
    cat = "unavailable/removed" if not c else CATS.get(c["category"], f"cat {c['category']}")
    groups[cat].append(info)

# write full CSV (non-music only), most-watched first
rows = []
for cat, items in groups.items():
    for it in items:
        rows.append((cat, it["watches"], it["title"], it["channel"], it["url"]))
rows.sort(key=lambda r: (-r[1]))
with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Category", "TimesWatched", "Title", "Channel", "URL"])
    w.writerows(rows)

total = sum(len(v) for v in groups.values())
print(f"\n=== {total} NON-MUSIC videos removed (grouped) ===", flush=True)
for cat, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
    top = sorted(items, key=lambda i: -i["watches"])[:6]
    print(f"\n▶ {cat}: {len(items)} videos", flush=True)
    for it in top:
        t = "".join(ch for ch in it["title"] if ch.isprintable())[:58]
        c = "".join(ch for ch in it["channel"] if ch.isprintable())[:24]
        print(f"    {it['watches']:>2}x  {t}  —  {c}", flush=True)
print(f"\nfull list ({total} rows) -> {out_csv}", flush=True)
