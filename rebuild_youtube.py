"""Rebuild the YouTube portion of the library authoritatively: classify every
watched video by YouTube's own category and keep only Music. Leaves Spotify
data untouched. Run: python rebuild_youtube.py <watch-history.json>
"""
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

from app.db import connect
from app.ingest import youtube

path = sys.argv[1]

with connect() as con:
    con.execute("PRAGMA busy_timeout=30000")
    before_yt = con.execute("SELECT COUNT(*) c FROM plays WHERE source='youtube'").fetchone()["c"]
    before_t = con.execute("SELECT COUNT(*) c FROM tracks").fetchone()["c"]
    con.execute("DELETE FROM plays WHERE source='youtube'")
print(f"cleared {before_yt} heuristic youtube plays (tracks before: {before_t})", flush=True)

r = youtube.import_authoritative(open(path, "rb").read())
print("import result:", flush=True)
for k, v in r.items():
    print(f"  {k}: {v}", flush=True)

# remove now-orphaned tracks (0 plays from any source) + their side tables
with connect() as con:
    orphans = "(SELECT t.id FROM tracks t LEFT JOIN plays p ON p.track_id=t.id WHERE p.id IS NULL)"
    con.execute(f"DELETE FROM audio_features WHERE track_id IN {orphans}")
    con.execute(f"DELETE FROM lyrics WHERE track_id IN {orphans}")
    con.execute(f"DELETE FROM playlist_tracks WHERE track_id IN {orphans}")
    con.execute(f"DELETE FROM tracks WHERE id IN {orphans}")
    t = con.execute("SELECT COUNT(*) c FROM tracks").fetchone()["c"]
    yt = con.execute("SELECT COUNT(*) c FROM plays WHERE source='youtube'").fetchone()["c"]
    sp = con.execute("SELECT COUNT(*) c FROM plays WHERE source='spotify'").fetchone()["c"]
print(f"\n=== FINAL: {t} tracks | youtube {yt} plays | spotify {sp} plays ===", flush=True)
