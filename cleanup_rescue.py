"""Purge false-positive rescued tracks: any recovered song whose ARTIST cannot be
confirmed in the original video's title/channel is a coincidental match — remove
its track, plays, features and lyrics. Reads youtube_recovered.csv."""
import sys, csv, re, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from app.db import connect


def toks(s):
    return {w for w in re.split(r"[^a-z0-9가-힯]+", (s or "").lower()) if len(w) >= 3}


rows = list(csv.DictReader(open("youtube_recovered.csv", encoding="utf-8-sig")))
keep, remove = [], []
for r in rows:
    vid_text = (r["original_video"] + " " + r["channel"]).lower()
    da = r["artist"]
    atoks = {t for t in toks(da) if len(t) >= 4}
    artist_in = (da.lower() in vid_text) or bool(atoks & toks(vid_text))
    (keep if artist_in else remove).append(r)

deleted_tracks = deleted_plays = 0
with connect() as con:
    con.execute("PRAGMA busy_timeout=15000")
    for r in remove:
        row = con.execute("SELECT id FROM tracks WHERE title=? AND artist=?",
                          (r["recovered_as"], r["artist"])).fetchone()
        if not row:
            continue
        tid = row["id"]
        deleted_plays += con.execute("DELETE FROM plays WHERE track_id=?", (tid,)).rowcount
        con.execute("DELETE FROM audio_features WHERE track_id=?", (tid,))
        con.execute("DELETE FROM lyrics WHERE track_id=?", (tid,))
        con.execute("DELETE FROM tracks WHERE id=?", (tid,))
        deleted_tracks += 1

print(f"kept {len(keep)} real songs, removed {deleted_tracks} false positives ({deleted_plays} plays)", flush=True)
print("\n-- removed (false positives) --", flush=True)
for r in remove:
    s = f"  {r['recovered_as']} - {r['artist']}   <= {r['original_video'][:45]}"
    print("".join(c for c in s if c.isprintable()), flush=True)

# rewrite CSV to the clean kept set
with open("youtube_recovered.csv", "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    for r in sorted(keep, key=lambda x: -int(x["plays"])):
        w.writerow(r)
print(f"\nclean recovered CSV rewritten with {len(keep)} songs", flush=True)
