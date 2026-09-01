"""Robust rescue: verify each excluded (non-short, non-deleted) video against
Deezer and add the ones that are genuinely music. Fixes the brittle matching that
missed songs like 'Chiri Thottu (From "Sarvam Maya")' and 'Ey Pulla':
  - strip Deezer's "(From ...)/(feat ...)" suffix before comparing titles
  - partial-artist match (video says "Justin"; Deezer says "Justin Prabhakaran")
  - try multiple query forms (before '|', after ' - ', whole title)
Shorts already removed upstream (<60s). Reads data/excluded_full.json.
"""
import sys, csv, re, time, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from app.enrich import deezer
from app.db import connect, upsert_track, add_play

cand = json.load(open(r"data\excluded_full.json", encoding="utf-8"))

VETO = re.compile(
    r"\b(workout|glute|abs|hourglass|snowcam|snowboard|webcam|ski area|gameplay|"
    r"patch \d|tier list|walkthrough|tutorial|recipe|unboxing|how to|\bcourse\b|"
    r"davinci|resolve|money method|star citizen|league of legends|build is|"
    r"cargo|trading|automation|timberborn|co-?op|exercise|challenge day|"
    r"gym|deadlift|squat|highlights|full match|patch notes)\b", re.I)
_PAREN = re.compile(r"\((?:from|feat|ft|with|prod|out now|audio|video|lyric\w*|"
                    r"official|4k|hd)[^)]*\)", re.I)
_NOISE = re.compile(
    r"\b(official music video|official video|official audio|music video|lyric\w* video|"
    r"video song|full video|title track|dance (?:practice|cover|performance)|"
    r"choreography|live performance|4k|hd|out now)\b", re.I)


def toks(s):
    return {w for w in re.split(r"[^a-z0-9가-힯]+", (s or "").lower()) if len(w) >= 3}


def clean(t):
    t = _PAREN.sub("", t)
    t = _NOISE.sub("", t)
    t = re.sub(r"[#@]\w+", "", t)
    return t.strip(" -–—|·•\"'")


def queries(title):
    base = clean(title)
    qs = []
    seg = base.split("|")[0].strip()
    if len(re.sub(r"[^a-z0-9]", "", seg.lower())) >= 3:
        qs.append(seg)
    if " - " in base:                    # "Artist x Artist - Song" -> "Song"
        qs.append(clean(base.split(" - ")[-1]))
    qs.append(base[:60])
    seen, out = set(), []
    for q in qs:
        q = q.strip()
        k = q.lower()
        if q and k not in seen and len(re.sub(r"[^a-z0-9가-힯]", "", k)) >= 3:
            seen.add(k); out.append(q)
    return out


def try_match(e):
    vid_text = (e["title"] + " " + e["channel"]).lower()
    vt = toks(e["title"])
    for q in queries(e["title"]):
        try:
            hits = deezer._get("/search", {"q": q, "limit": 3}).get("data", [])
        except Exception:
            continue
        for hit in hits:
            dt_raw = hit.get("title", "")
            da = (hit.get("artist") or {}).get("name", "")
            dtt = toks(_PAREN.sub("", dt_raw))
            if not dtt:
                continue
            sim = len(dtt & vt) / len(dtt)
            atoks = {t for t in toks(da) if len(t) >= 4}
            artist_in = (da.lower() in vid_text) or bool(atoks & toks(vid_text))
            # Require the artist to be confirmable in the video — a generic title
            # matching some song (Xbox, White Pass, Latest Headlines) is NOT enough.
            if artist_in and sim >= 0.45:
                return dt_raw, da, hit
        time.sleep(0.1)
    return None


recovered, searched = [], 0
for vid, e in cand.items():
    if VETO.search(e["title"]):
        continue
    searched += 1
    m = try_match(e)
    if m:
        recovered.append((e, *m))

added = 0
with connect() as con:
    con.execute("PRAGMA busy_timeout=15000")
    for e, dt, da, hit in recovered:
        tid = upsert_track(con, title=dt, artist=da,
                           album=(hit.get("album") or {}).get("title"))
        if not tid:
            continue
        for ts in e["times"]:
            add_play(con, tid, ts, "youtube")
            added += 1

out = r"youtube_recovered.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["plays", "recovered_as", "artist", "original_video", "channel", "category"])
    for e, dt, da, hit in sorted(recovered, key=lambda r: -r[0]["plays"]):
        w.writerow([e["plays"], dt, da, e["title"], e["channel"], e["category"]])

print(f"searched {searched} candidates — RECOVERED {len(recovered)} songs, {added} plays", flush=True)
print("-- top recovered --", flush=True)
for e, dt, da, hit in sorted(recovered, key=lambda r: -r[0]["plays"])[:35]:
    s = f"  {e['plays']:>2}x  {dt} - {da}"
    print("".join(c for c in s if c.isprintable()), flush=True)
