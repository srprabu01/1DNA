"""List every YouTube video that was NOT added to the library (non-music, Shorts,
deleted/private), with category + play count + reason. Writes a CSV and a summary.

Usage: python list_excluded.py <watch-history.json>
"""
import sys, csv, warnings
from collections import Counter
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from app.ingest import youtube

path = sys.argv[1] if len(sys.argv) > 1 else (
    r"C:\Users\srpra\AppData\Local\Temp\claude"
    r"\C--Users-srpra-OneDrive-Desktop-Claude"
    r"\76e85bae-f9ee-4c64-bf7b-ed413e825a9f\scratchpad\takeout"
    r"\Takeout\YouTube and YouTube Music\history\watch-history.json")

content = open(path, "rb").read()
res = youtube.import_authoritative(content, dry_run=True)
excluded = sorted(res["excluded"], key=lambda x: -x["plays"])

out = r"C:\Users\srpra\OneDrive\Desktop\Claude\music-analyzer\youtube_not_added.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["plays", "category", "reason", "title", "channel", "video_id"])
    for e in excluded:
        w.writerow([e["plays"], e["category"], e["reason"],
                    e["title"], e["channel"], e["video_id"]])

tot_vids = len(excluded)
tot_plays = sum(e["plays"] for e in excluded)
print("=== NOT ADDED FROM YOUTUBE ===", flush=True)
print(f"kept as music: {res['music_videos']} videos / {res['plays']} plays", flush=True)
print(f"EXCLUDED: {tot_vids} videos / {tot_plays} plays", flush=True)
print(f"  (non-music dropped: {res['nonmusic_dropped']}, shorts skipped: {res['shorts_skipped']})", flush=True)
print("\n-- excluded by category --", flush=True)
cat = Counter()
for e in excluded:
    cat[e["category"]] += e["plays"]
for c, n in cat.most_common():
    print(f"  {n:>5} plays  {c}", flush=True)
print("\n-- top 40 most-watched videos that were NOT added --", flush=True)
for e in excluded[:40]:
    t = "".join(ch for ch in e["title"] if ch.isprintable())[:52]
    print(f"  {e['plays']:>3}x  [{e['category'][:18]:18}] {t}", flush=True)
print(f"\nfull CSV -> {out}", flush=True)
