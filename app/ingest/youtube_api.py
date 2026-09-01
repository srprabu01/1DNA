"""Official YouTube Data API v3 via a real 'Sign in with Google' (Web OAuth).

This is the SUPPORTED path (unlike the YT Music internal API, which rejects
OAuth). It pulls the data Google exposes officially:
  - liked music videos          -> library tracks
  - your playlists + their items -> analyzable playlists
  - your subscriptions          -> favourite/affinity artists

What it CANNOT get (no Google API exposes it): your watch/listening HISTORY
with timestamps. For that, use the header-paste method or a Takeout import.
"""
import json
import time
from urllib.parse import urlencode

import requests

from .. import config
from ..db import connect, upsert_track
from .youtube import _clean_title, _split_artist_title, _MUSIC_CHANNEL_HINTS, _MUSIC_TITLE_HINTS

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
API = "https://www.googleapis.com/youtube/v3"
SCOPE = "https://www.googleapis.com/auth/youtube.readonly"
MUSIC_CATEGORY = "10"  # YouTube's "Music" video category id


# ---------- OAuth ----------

def oauth_configured():
    return bool(config.YTDATA_CLIENT_ID and config.YTDATA_CLIENT_SECRET)


def connected():
    return config.YTDATA_TOKEN_PATH.exists()


def auth_url():
    return AUTH_URL + "?" + urlencode({
        "client_id": config.YTDATA_CLIENT_ID,
        "redirect_uri": config.YTDATA_REDIRECT_URI,
        "response_type": "code",
        "scope": SCOPE,
        "access_type": "offline",   # get a refresh token -> stays signed in
        "prompt": "consent",
        "include_granted_scopes": "true",
    })


def exchange_code(code):
    r = requests.post(TOKEN_URL, timeout=30, data={
        "code": code, "client_id": config.YTDATA_CLIENT_ID,
        "client_secret": config.YTDATA_CLIENT_SECRET,
        "redirect_uri": config.YTDATA_REDIRECT_URI,
        "grant_type": "authorization_code"})
    r.raise_for_status()
    t = r.json()
    t["expires_at"] = time.time() + t.get("expires_in", 3600)
    config.YTDATA_TOKEN_PATH.write_text(json.dumps(t))
    return True


def _token():
    if not config.YTDATA_TOKEN_PATH.exists():
        return None
    t = json.loads(config.YTDATA_TOKEN_PATH.read_text())
    if time.time() > t.get("expires_at", 0) - 60:
        if not t.get("refresh_token"):
            return t.get("access_token")
        r = requests.post(TOKEN_URL, timeout=30, data={
            "client_id": config.YTDATA_CLIENT_ID,
            "client_secret": config.YTDATA_CLIENT_SECRET,
            "refresh_token": t["refresh_token"], "grant_type": "refresh_token"})
        r.raise_for_status()
        f = r.json()
        t["access_token"] = f["access_token"]
        t["expires_at"] = time.time() + f.get("expires_in", 3600)
        config.YTDATA_TOKEN_PATH.write_text(json.dumps(t))
    return t["access_token"]


# ---------- API helpers ----------

def _get(path, token, params):
    r = requests.get(f"{API}{path}", timeout=30,
                     headers={"Authorization": f"Bearer {token}"}, params=params)
    if r.status_code == 401:
        raise RuntimeError("YouTube sign-in expired — sign in again.")
    r.raise_for_status()
    return r.json()


def classify_ids(ids, token=None):
    """Authoritatively classify YouTube video IDs by category via the Data API.

    Returns {video_id: {"category": id_str, "seconds": int}}. Category "10" is
    Music. Videos that are deleted/private simply won't appear in the result.
    Batches 50 ids per call (1 quota unit each).
    """
    import re as _re
    token = token or _token()
    if not token:
        raise RuntimeError("Sign in with Google (YouTube) first.")
    out = {}
    ids = list(dict.fromkeys(ids))  # de-dupe, preserve order
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        r = requests.get(f"{API}/videos", timeout=30,
                         headers={"Authorization": f"Bearer {token}"},
                         params={"part": "snippet,contentDetails",
                                 "id": ",".join(chunk), "maxResults": 50})
        if r.status_code == 401:
            token = _token()
            continue
        r.raise_for_status()
        for it in r.json().get("items", []):
            d = (it.get("contentDetails") or {}).get("duration", "")
            m = _re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", d)
            secs = (int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60
                    + int(m.group(3) or 0)) if m else 0
            out[it["id"]] = {"category": (it.get("snippet") or {}).get("categoryId"),
                             "seconds": secs}
    return out


def _paged(path, token, params, cap=4000):
    params = dict(params)
    items = []
    while len(items) < cap:
        d = _get(path, token, params)
        items += d.get("items", [])
        nxt = d.get("nextPageToken")
        if not nxt:
            break
        params["pageToken"] = nxt
    return items


def _is_music_meta(title, channel):
    c = (channel or "").lower()
    t = (title or "").lower()
    if any(h in c for h in _MUSIC_CHANNEL_HINTS):
        return True
    return any(h in t for h in _MUSIC_TITLE_HINTS)


import re as _re
_EMOJI = _re.compile(
    "[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "←-⇿⬀-⯿ -⁯️‍〰]")
_HASHTAG = _re.compile(r"#\w+")
_SHORTS = _re.compile(r"#?shorts?\b", _re.I)


def _clean_yt_title(raw):
    """Strip hashtags, emoji, #shorts, 'By @user' and trailing junk from a
    YouTube video title before we try to parse artist/song out of it."""
    t = _SHORTS.sub("", raw or "")
    t = _HASHTAG.sub("", t)
    t = _EMOJI.sub("", t)
    t = _re.sub(r"\b(by|feat\.?|ft\.?)\s+@\S+", "", t, flags=_re.I)
    t = _re.sub(r"@\S+", "", t)
    t = _clean_title(t)          # existing (official video/audio/feat) cleaner
    return t.strip(" -–—|·•\"'")


def _is_junk(title):
    """A parsed title is junk if it has no real word content."""
    return len(_re.sub(r"[^a-z0-9]", "", (title or "").lower())) < 2


def _iso_seconds(iso):
    m = _re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    h, mn, s = (int(x) if x else 0 for x in m.groups())
    return h * 3600 + mn * 60 + s


# A real song preview sits roughly in this window; below = Short/clip,
# above = mix/livestream/podcast.
MIN_SONG_SEC = 75
MAX_SONG_SEC = 900


# ---------- sync ----------

def sync():
    token = _token()
    if not token:
        raise RuntimeError("Sign in with Google first.")
    summary = {"liked": 0, "playlists": 0, "playlist_tracks": 0,
               "subscriptions": 0, "skipped_shorts": 0, "skipped_nonmusic": 0}

    # 1) liked videos — keep genuine songs only. A liked *video* feed is full of
    # Shorts/clips/reactions, so we require: Music category OR a music channel
    # (VEVO / "- Topic"), a song-length duration (drops Shorts & mixes), and a
    # non-junk title after stripping emoji/hashtags.
    with connect() as con:
        for it in _paged("/videos", token,
                         {"part": "snippet,contentDetails", "myRating": "like",
                          "maxResults": 50}):
            sn = it.get("snippet", {})
            channel = sn.get("videoOwnerChannelTitle") or sn.get("channelTitle", "")
            is_music = (sn.get("categoryId") == MUSIC_CATEGORY
                        or _is_music_meta(sn.get("title", ""), channel))
            if not is_music:
                summary["skipped_nonmusic"] += 1
                continue
            dur = _iso_seconds((it.get("contentDetails") or {}).get("duration"))
            if dur and (dur < MIN_SONG_SEC or dur > MAX_SONG_SEC):
                summary["skipped_shorts"] += 1
                continue
            artist, song = _split_artist_title(_clean_yt_title(sn.get("title", "")), channel)
            if _is_junk(song):
                summary["skipped_nonmusic"] += 1
                continue
            if upsert_track(con, title=song, artist=artist,
                            duration_ms=dur * 1000 if dur else None):
                summary["liked"] += 1

    # 2) subscriptions -> favourite artists (affinity for the recommender)
    with connect() as con:
        con.execute("DELETE FROM top_items WHERE source='youtube' AND time_range='subscriptions'")
        for i, it in enumerate(_paged("/subscriptions", token,
                               {"part": "snippet", "mine": "true",
                                "maxResults": 50, "order": "alphabetical"}), 1):
            name = it["snippet"]["title"].removesuffix(" - Topic").strip()
            con.execute(
                """INSERT OR IGNORE INTO top_items
                   (source, item_type, time_range, rank, name)
                   VALUES ('youtube','artist','subscriptions',?,?)""", (i, name))
            summary["subscriptions"] += 1

    # 3) playlists + their items -> analyzable playlists
    from ..playlist import store
    for pl in _paged("/playlists", token, {"part": "snippet", "mine": "true", "maxResults": 50}):
        ext = pl["id"]
        name = pl["snippet"]["title"]
        items = _paged("/playlistItems", token,
                       {"part": "snippet", "playlistId": ext, "maxResults": 50})
        track_ids = []
        with connect() as con:
            for it in items:
                sn = it.get("snippet", {})
                title_raw = sn.get("title", "")
                if title_raw in ("Deleted video", "Private video", "", None):
                    continue
                channel = sn.get("videoOwnerChannelTitle", "")
                artist, song = _split_artist_title(_clean_yt_title(title_raw), channel)
                if _is_junk(song):
                    continue
                tid = upsert_track(con, title=song, artist=artist)
                if tid:
                    track_ids.append(tid)
        if track_ids:
            store.create(name, track_ids, source="youtube", external_id=ext)
            summary["playlists"] += 1
            summary["playlist_tracks"] += len(track_ids)

    return summary
