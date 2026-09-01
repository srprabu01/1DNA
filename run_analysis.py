"""Standalone background analysis runner — resilient to the web server restarting.
Enables WAL so it can write audio_features while the dashboard reads concurrently.
Run: python run_analysis.py
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

print("=== enriching new tracks (Deezer match) ===", flush=True)
while True:
    r = enrich_batch(limit=60)
    print(f"  matched {r['matched']}, failed {r['failed']} — {r['remaining']} left", flush=True)
    if r["remaining"] == 0:
        break

print("=== analyzing (most-played first) ===", flush=True)
done = 0
while True:
    r = analyze_batch(limit=6)
    done += r["analyzed"]
    print(f"  +{r['analyzed']} (errors {r['errors']}) — {r['remaining']} left, {done} done", flush=True)
    if r["remaining"] == 0:
        break

print("=== fetching lyrics (top tracks) ===", flush=True)
for _ in range(6):
    r = ly.fetch_batch(limit=25)
    print(f"  lyrics +{r['analyzed']} ({r['missing']} missing) — {r['remaining']} left", flush=True)
    if r["remaining"] == 0:
        break
print("=== ANALYSIS COMPLETE ===", flush=True)
