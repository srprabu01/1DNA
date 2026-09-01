"""Prepare the 'not added' videos for LLM music-classification.
Drops deleted/private and Shorts (<60s). Writes full data (with play timestamps)
locally, and a compact list for the workflow.
"""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from app.ingest import youtube

path = (r"C:\Users\srpra\AppData\Local\Temp\claude"
        r"\C--Users-srpra-OneDrive-Desktop-Claude"
        r"\76e85bae-f9ee-4c64-bf7b-ed413e825a9f\scratchpad\takeout"
        r"\Takeout\YouTube and YouTube Music\history\watch-history.json")

res = youtube.import_authoritative(open(path, "rb").read(), dry_run=True)
exc = [e for e in res["excluded"] if e.get("reason") == "non-music"]

cand = []
shorts = deleted = 0
for e in exc:
    if e["category"] == "deleted/private" or e["title"].startswith("http"):
        deleted += 1
        continue
    if e.get("seconds") and e["seconds"] < 60:   # avoid Shorts
        shorts += 1
        continue
    cand.append(e)

full = r"C:\Users\srpra\OneDrive\Desktop\Claude\music-analyzer\data\excluded_full.json"
json.dump({e["video_id"]: e for e in cand}, open(full, "w", encoding="utf-8"))

compact = [{"id": e["video_id"], "t": e["title"][:90], "ch": e["channel"][:35],
            "cat": e["category"], "sec": e.get("seconds")} for e in cand]
json.dump(compact, open(r"C:\Users\srpra\OneDrive\Desktop\Claude\music-analyzer\data\excluded_compact.json",
                        "w", encoding="utf-8"))

print(f"excluded non-music: {len(exc)}")
print(f"dropped: {deleted} deleted/private, {shorts} shorts (<60s)")
print(f"CANDIDATES to classify: {len(cand)}")
print(f"compact bytes: {len(json.dumps(compact))}")
