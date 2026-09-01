"""Full processing pipeline, standalone (survives web-server restarts).
Enrich (Deezer) -> analyze (librosa, most-played first) -> lyrics. WAL for concurrency.
"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from app.db import connect
with connect() as con:
    con.execute("PRAGMA journal_mode=WAL")

from app.enrich.deezer import enrich_batch
from app.audio.features import analyze_batch
from app.lyrics import engine as ly

print("=== ENRICH (Deezer match) ===", flush=True)
while True:
    r = enrich_batch(limit=60)
    print(f"  matched {r['matched']} failed {r['failed']} — {r['remaining']} left", flush=True)
    if r["remaining"] == 0:
        break

print("=== ANALYZE (librosa, most-played first) ===", flush=True)
done = 0
while True:
    r = analyze_batch(limit=6)
    done += r["analyzed"]
    print(f"  +{r['analyzed']} (err {r['errors']}) — {r['remaining']} left ({done} done)", flush=True)
    if r["remaining"] == 0:
        break

print("=== LYRICS (top tracks) ===", flush=True)
for _ in range(8):
    r = ly.fetch_batch(limit=25)
    print(f"  lyrics +{r['analyzed']} ({r['missing']} missing) — {r['remaining']} left", flush=True)
    if r["remaining"] == 0:
        break
print("=== PIPELINE COMPLETE ===", flush=True)
