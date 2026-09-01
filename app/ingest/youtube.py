"""Parse a Google Takeout YouTube / YouTube Music watch-history.json export.

Get yours at https://takeout.google.com -> deselect all -> YouTube and YouTube Music
-> history -> JSON format. Upload watch-history.json in the dashboard.
"""
import json
import re

from ..db import connect, upsert_track, add_play

_FEAT = re.compile(r"\s*[\(\[](?:feat|ft|with|prod)\.?[^\)\]]*[\)\]]", re.I)
_NOISE = re.compile(
    r"\s*[\(\[][^\)\]]*(official|video|audio|lyric|lyrics|visualizer|hd|4k|mv)[^\)\]]*[\)\]]",
    re.I,
)


def _clean_title(raw):
    t = raw.removeprefix("Watched ").strip()
    t = _NOISE.sub("", t)
    t = _FEAT.sub("", t)
    return t.strip(" -–|")


_SONG_LABEL = re.compile(
    r"^(?P<song>.+?)\s+(?:full\s+)?(?:video\s+|lyric(?:al)?\s+|audio\s+)?song\b"
    r"\s*(?=[:|\-–—(]|$)", re.I)


def _split_artist_title(title, channel):
    # Indian-music titles are "<Song> Song: singers | music director | movie …"
    # — the song name leads, so pull it out before the generic "- Artist - Song"
    # rule mistakes a singer list for the artist.
    m = _SONG_LABEL.match(title)
    if m and " - " not in m.group("song"):
        song = m.group("song").strip(" -–|:")
        artist = (channel or "").removesuffix(" - Topic").strip()
        return artist or "Unknown", song
    # "Artist - Song" pattern is the YouTube norm
    for sep in (" - ", " – ", " — ", " | "):
        if sep in title:
            artist, song = title.split(sep, 1)
            return artist.strip(), song.strip()
    # fall back to channel name ("Artist - Topic" channels are auto-generated music)
    artist = (channel or "").removesuffix(" - Topic").strip()
    return artist or "Unknown", title


# Takeout gives no video category, so we classify by source signals. This is
# deliberately RECALL-biased so film/OST/K-pop/anime/J-pop music on label channels
# (which don't say "official video") still gets counted as music.
_MUSIC_CHANNEL_HINTS = (
    "vevo", " - topic",
    # generic label words
    "music", "records", "recordz", "soundtrack",
    # Indian labels
    "t-series", "saregama", "think music", "sony music", "aditya", "lahari",
    "divo", "junglee", "u1 records", "zee music", "speed records", "tips",
    "sun tv", "sony music south", "muzik247", "east coast", "trend music",
    # K-pop / J-pop / C-pop / anime labels & aggregators
    "entertainment", "1thek", "smtown", "hybe", "bighit", "jyp", "yg ", "ygex",
    "stone music", "kakao", "genie", "starship", "pledis", "cube", "ador",
    "ani-one", "muse asia", "muse india", "aniplex", "sme", "avex", "lantis",
    "pony canyon", "flyingdog", "universal music", "warner music", "sacra music",
    "1thek", "stone", "kozentertain", "ka:me")

_MUSIC_TITLE_HINTS = (
    "official music video", "official video", "official audio", "official m/v",
    "lyric video", "lyrics video", "lyrical", "lyric", "(audio)", "audio)",
    "music video", " m/v", "(m/v", "mv)", "[mv]", " mv ", "|mv",
    "video song", "full song", "full video", "title track", "theme song",
    " ost", "ost ", "ost)", "ost]", "(ost", "soundtrack", "original sound",
    "opening", "ending", "op)", "ed)", "character song", "insert song",
    "unplugged", "acoustic", "(cover", "cover)", "feat.", " ft.", "audiotrack",
    "song |", "| song", "song)", "jukebox", "tv size", "full ver",
    # live performances (K-pop stages, gigs) & remixes — kept SPECIFIC so gaming
    # livestreams ("🔴LIVE - PATCH") don't match a bare "live".
    "live performance", "live at", "afterhours", "after hours", "waterbomb",
    "unplugged live", "concert", "remix", "mashup", "prod.", "prod ",
    "hd song", "hd songs", "video songs", "pieces album", "audio track", "singles")

# Korean / Japanese / Chinese scripts — anime/K-drama/J-pop OSTs often have
# CJK-only titles. Treat as music when paired with any music word or label.
_CJK = None


def _has_cjk(s):
    global _CJK
    if _CJK is None:
        import re as _re
        _CJK = _re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")
    return bool(_CJK.search(s or ""))


def _is_music(entry):
    if entry.get("header", "") == "YouTube Music":
        return True
    subs = entry.get("subtitles") or []
    channel = (subs[0].get("name", "") if subs else "").lower()
    title = entry.get("title", "").lower()
    if any(h in channel for h in _MUSIC_CHANNEL_HINTS):
        return True
    if any(h in title for h in _MUSIC_TITLE_HINTS):
        return True
    # CJK title with a music cue (channel or word) — anime/K-drama/J-pop OSTs
    if _has_cjk(title) and (_has_cjk(channel) or "music" in channel):
        return True
    # CJK title in an "Artist - Song" shape (e.g. "비비 (BIBI) - ...") is music
    # even on odd channels; K/J gaming/vlogs rarely use that romanised shape.
    raw = entry.get("title", "")
    if _has_cjk(raw) and any(s in raw for s in (" - ", " – ", " — ", "│", " | ", "~")):
        return True
    return False


def _parse_html(text):
    """Parse a Takeout watch-history.html export into JSON-shaped entries.

    Google exports this format by default. Each watch is a 'content-cell' with
    a video link (title), a channel link, and a timestamp.
    """
    import html as _html

    cell = re.compile(
        r'<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">(.*?)</div>',
        re.S)
    vid = re.compile(r'<a href="([^"]*watch\?v=[^"]*)">(.*?)</a>', re.S)
    chan = re.compile(r'<a href="[^"]*(?:/channel/|/@)[^"]*">(.*?)</a>', re.S)
    # date sits after the last <br>: "Jan 5, 2026, 10:00:00 PM IST"
    date = re.compile(r'(\w{3}\s+\d{1,2},\s+\d{4},\s+[\d:]+(?:\s*[AP]M)?[^<]*)$')

    entries = []
    for block in cell.findall(text):
        block = _html.unescape(block)          # &#8239; -> narrow space, &amp; -> & etc.
        vm = vid.search(block)
        if not vm:
            continue
        url = vm.group(1).strip()
        title = vm.group(2).strip()
        if not title or title.startswith("http"):   # removed/private video
            continue
        cm = chan.search(block)
        channel = cm.group(1).strip() if cm else ""
        tail = re.split(r'<br\s*/?>', block)[-1]
        dm = date.search(tail.strip())
        when = _parse_html_date(dm.group(1)) if dm else None
        if not when:
            continue
        # header hint: YouTube Music blocks say so earlier in the cell
        header = "YouTube Music" if "music.youtube.com" in block else "YouTube"
        entries.append({"title": "Watched " + title, "time": when,
                        "header": header, "titleUrl": url,
                        "subtitles": [{"name": channel}]})
    return entries


def _parse_html_date(s):
    from datetime import datetime
    s = s.replace(" ", " ").replace(" ", " ").strip()
    s = re.sub(r"\s+[A-Z]{2,4}$", "", s)  # drop trailing tz abbrev
    for fmt in ("%b %d, %Y, %I:%M:%S %p", "%b %d, %Y, %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).isoformat()
        except ValueError:
            continue
    return None


def _video_id(entry):
    m = re.search(r"[?&]v=([\w-]{6,})", entry.get("titleUrl", "") or "")
    return m.group(1) if m else None


def _parse(content: bytes):
    head = content.lstrip()[:1]
    if head in (b"[", b"{"):
        return json.loads(content)
    return _parse_html(content.decode("utf-8", errors="replace"))


# Words that are NOT artist identity — never let these propagate music-ness.
_PROP_STOP = set("""official video audio music songs song lyric lyrical lyrics
visualiser visualizer live from the and feat ft with mix remix cover version
movie movies films film arts entertainment records recordz channel presents
studio dance choreography full new latest best top part vol hits playlist mix
tamil telugu hindi malayalam kannada korean japanese kpop jpop bollywood
super sound india world times media network productions production creations
release official4k reaction shorts short teaser trailer scene scenes""".split())

_VER = re.compile(
    r"\((?:live|lyrical?|lyrics|visuali[sz]er|audio|video|official|remix|cover|"
    r"unplugged|acoustic|4k|hd|full|reprise|slowed|reverb|extended|teaser)[^)]*\)",
    re.I)


def _song_core(d):
    """Normalized song-name key so 'X (Lyrical)', 'X (Live)', 'X Visualiser' all
    collapse to the same key — used to match alternate versions of one song."""
    _, song = _split_artist_title(_clean_title(d["w"][0]["title"]), d["channel"])
    song = _VER.sub("", song.lower())
    return re.sub(r"[^a-z0-9가-힣]", "", song)


def _name_tokens(d):
    """Identity tokens (artist words + channel words) that can vouch for other
    videos as 'a song by that artist'. Filtered to name-like words."""
    artist, _ = _split_artist_title(_clean_title(d["w"][0]["title"]), d["channel"])
    text = (artist + " " + d["channel"]).lower()
    return {t for t in re.split(r"[^a-z0-9가-힣]+", text)
            if len(t) >= 5 and t not in _PROP_STOP}


def _title_tokens(d):
    return {t for t in re.split(r"[^a-z0-9가-힣]+", d["w"][0]["title"].lower())
            if len(t) >= 5 and t not in _PROP_STOP}


def import_authoritative(content: bytes, token=None, short_sec: int = 45,
                         propagate: bool = True, dry_run: bool = False):
    """Import a Takeout history, keeping music by three layers:

      1. DIRECT signal — title/channel says music (heuristic) OR YouTube category
         is Music (10). (YT category alone is unreliable for Indian/indie music.)
      2. PROPAGATION — once an artist/channel is confirmed as music, their OTHER
         videos are treated as songs too ("more by that artist"), and alternate
         versions of a known song (lyrical / live / visualiser) match by song-core.
      3. Genuine non-music (gaming/sports/tutorials with none of the above) is
         dropped; obvious Shorts are dropped unless the title itself says music.

    dry_run=True classifies but writes nothing (returns the breakdown + samples).
    """
    from collections import Counter
    from . import youtube_api

    data = _parse(content)
    by_vid = {}
    for e in data:
        if not e.get("title", "").startswith("Watched ") or not e.get("time"):
            continue
        vid = _video_id(e)
        if not vid:
            continue
        subs = e.get("subtitles") or []
        d = by_vid.setdefault(vid, {"entry": e, "channel":
                              subs[0].get("name", "") if subs else "", "w": []})
        d["w"].append({"title": e["title"], "time": e["time"]})

    try:
        cls = youtube_api.classify_ids(list(by_vid), token=token)
    except Exception:
        cls = {}   # not signed in — heuristic only

    # ---- pass 1: direct signals ----
    reason = {}
    for vid, d in by_vid.items():
        info = cls.get(vid)
        if _is_music(d["entry"]) or (info and info["category"] == "10"):
            reason[vid] = "signal"

    # ---- build propagation vocabulary from DIRECT-signal music ONLY (no cascade,
    # so one stray match can't drag a whole gaming channel in) ----
    ch_total, ch_music = Counter(), Counter()
    for vid, d in by_vid.items():
        ch = d["channel"].lower()
        if ch:
            ch_total[ch] += 1
            if vid in reason:
                ch_music[ch] += 1
    # an "artist channel" = a channel that is PREDOMINANTLY confirmed music
    music_channels = {ch for ch, n in ch_music.items()
                      if n >= 2 and n / ch_total[ch] >= 0.5}
    # Music/label channels also post trailers, full movies, tutorials, gameplay —
    # veto those even when the channel is a music channel.
    veto = re.compile(
        r"\b(trailer|teaser trailer|full movie|tutorial|guide|review|gameplay|"
        r"unboxing|vfx|how to|behind the scenes|making of|interview|explained|"
        r"reaction|podcast|vlog|highlights|patch|tier list|walkthrough|"
        r"episode \d|ep\.? \d)\b", re.I)

    # ---- pass 2: single, non-cascading propagation (artist's OWN channel only) ----
    prop_samples = []
    if propagate:
        for vid, d in by_vid.items():
            if vid in reason:
                continue
            ch = d["channel"].lower()
            if not (ch and ch in music_channels):
                continue
            if veto.search(d["w"][0]["title"]):
                continue
            reason[vid] = "same-artist channel"
            if len(prop_samples) < 40:
                prop_samples.append(("same-artist channel",
                                     d["w"][0]["title"][:58], d["channel"][:24]))

    # ---- import (or dry-run) ----
    breakdown = Counter()
    excluded = []
    plays = songs = shorts = dropped = 0
    con_cm = connect() if not dry_run else None
    con = con_cm.__enter__() if con_cm else None
    try:
        for vid, d in by_vid.items():
            if vid not in reason:
                dropped += 1
                if dry_run:
                    info = cls.get(vid)
                    cat = info["category"] if info else None
                    excluded.append({
                        "title": d["w"][0]["title"].replace("Watched ", ""),
                        "channel": d["channel"], "plays": len(d["w"]),
                        "category": _CATEGORY_NAMES.get(cat, cat) if cat else "deleted/private",
                        "reason": "non-music", "video_id": vid,
                        "seconds": info["seconds"] if info else None,
                        "times": [w["time"] for w in d["w"]]})
                continue
            info = cls.get(vid)
            if (reason[vid] == "signal" and info and info["seconds"]
                    and info["seconds"] < short_sec and not _is_music(d["entry"])):
                shorts += 1
                if dry_run:
                    excluded.append({
                        "title": d["w"][0]["title"].replace("Watched ", ""),
                        "channel": d["channel"], "plays": len(d["w"]),
                        "category": _CATEGORY_NAMES.get(info["category"], info["category"]),
                        "reason": "short (<%ds)" % short_sec, "video_id": vid})
                continue
            breakdown[reason[vid]] += 1
            songs += 1
            if dry_run:
                plays += len(d["w"])
                continue
            artist, song = _split_artist_title(
                _clean_title(d["w"][0]["title"]), d["channel"])
            tid = upsert_track(con, title=song, artist=artist,
                               duration_ms=(info["seconds"] * 1000 if info and info["seconds"] else None))
            if not tid:
                continue
            for w in d["w"]:
                add_play(con, tid, w["time"], "youtube")
                plays += 1
    finally:
        if con_cm:
            con_cm.__exit__(None, None, None)

    return {"videos_classified": len(by_vid), "music_videos": songs,
            "plays": plays, "shorts_skipped": shorts, "nonmusic_dropped": dropped,
            "kept_breakdown": dict(breakdown), "propagation_samples": prop_samples,
            "excluded": excluded}


# YouTube video category ids -> names (for the "not added" report)
_CATEGORY_NAMES = {
    "1": "Film & Animation", "2": "Autos & Vehicles", "10": "Music",
    "15": "Pets & Animals", "17": "Sports", "18": "Short Movies",
    "19": "Travel & Events", "20": "Gaming", "21": "Videoblogging",
    "22": "People & Blogs", "23": "Comedy", "24": "Entertainment",
    "25": "News & Politics", "26": "Howto & Style", "27": "Education",
    "28": "Science & Technology", "29": "Nonprofits & Activism",
    "30": "Movies", "43": "Shows", "44": "Trailers"}


def import_file(content: bytes, music_only: bool = True):
    """Import a Takeout watch-history export (JSON or HTML). Returns summary."""
    head = content.lstrip()[:1]
    if head in (b"[", b"{"):
        data = json.loads(content)
    else:
        data = _parse_html(content.decode("utf-8", errors="replace"))
    imported = skipped = 0
    with connect() as con:
        for entry in data:
            title = entry.get("title", "")
            if not title.startswith("Watched ") or not entry.get("time"):
                skipped += 1
                continue
            if music_only and not _is_music(entry):
                skipped += 1
                continue
            subs = entry.get("subtitles") or []
            channel = subs[0].get("name", "") if subs else ""
            artist, song = _split_artist_title(_clean_title(title), channel)
            tid = upsert_track(con, title=song, artist=artist)
            if tid:
                add_play(con, tid, entry["time"], "youtube")
                imported += 1
            else:
                skipped += 1
    return {"imported": imported, "skipped": skipped}
