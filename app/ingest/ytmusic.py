"""YouTube Music via logged-in session (ytmusicapi) — no API keys, no Takeout.

You paste your browser request headers ONCE (from a request to music.youtube.com
while logged in). We convert them to an auth token and reuse the session to pull
your listening history, liked songs, and library live.

How to get the headers (one time):
  1. Open https://music.youtube.com in your browser, logged in.
  2. Open DevTools (F12) -> Network tab. Filter for "browse".
  3. Play something / click around so a POST to /youtubei/v1/browse appears.
  4. Right-click that request -> Copy -> Copy request headers.
  5. Paste the whole block into the app's YouTube Music box.
"""
import json

from .. import config
from ..db import connect, upsert_track, add_play

AUTH_PATH = config.DATA_DIR / "ytmusic_auth.json"


def connected():
    return AUTH_PATH.exists() or config.YT_OAUTH_PATH.exists()


# ---------- OAuth device-code sign-in (recommended) ----------

def oauth_configured():
    return bool(config.YT_OAUTH_CLIENT_ID and config.YT_OAUTH_CLIENT_SECRET)


def _oauth_creds():
    from ytmusicapi import OAuthCredentials
    return OAuthCredentials(client_id=config.YT_OAUTH_CLIENT_ID,
                            client_secret=config.YT_OAUTH_CLIENT_SECRET)


def oauth_start():
    """Begin the device flow: returns a short code + a Google URL to approve at."""
    if not oauth_configured():
        raise RuntimeError("Add YT_OAUTH_CLIENT_ID / SECRET to .env first (see README).")
    code = _oauth_creds().get_code()
    config.YT_DEVICECODE_PATH.write_text(json.dumps(dict(code)))
    return {"user_code": code["user_code"],
            "verification_url": code["verification_url"],
            "expires_in": code["expires_in"]}


def oauth_finish():
    """Exchange the approved device code for a saved, auto-refreshing token."""
    if not config.YT_DEVICECODE_PATH.exists():
        raise RuntimeError("Start the sign-in first.")
    code = json.loads(config.YT_DEVICECODE_PATH.read_text())
    token = dict(_oauth_creds().token_from_code(code["device_code"]))
    if token.get("error") == "authorization_pending":
        # user hasn't finished approving on google.com/device yet — keep the
        # device code so they can simply click Finish again
        raise RuntimeError("Google hasn't seen your approval yet — finish the "
                           "approval in the Google tab, wait a few seconds, "
                           "then click Finish again.")
    if "access_token" not in token or "refresh_token" not in token:
        raise RuntimeError(f"Google sign-in failed: "
                           f"{token.get('error_description') or token.get('error') or 'unknown error'}. "
                           f"Click Sign in to start over.")
    import time
    token.setdefault("expires_at", int(time.time()) + int(token.get("expires_in", 3600)))
    config.YT_OAUTH_PATH.write_text(json.dumps(token))
    config.YT_DEVICECODE_PATH.unlink(missing_ok=True)
    return True


def save_auth(headers_raw: str):
    """Parse pasted browser headers into a ytmusicapi auth blob and store it."""
    import ytmusicapi

    auth = ytmusicapi.setup(filepath=None, headers_raw=headers_raw)
    # setup returns a JSON string; validate it parses
    data = json.loads(auth) if isinstance(auth, str) else auth
    if "Cookie" not in json.dumps(data) and "cookie" not in json.dumps(data).lower():
        raise ValueError("Those headers don't contain a login cookie — make sure you "
                         "copied request headers from a logged-in music.youtube.com request.")
    AUTH_PATH.write_text(json.dumps(data))
    return True


def _client():
    from ytmusicapi import YTMusic

    # Prefer OAuth sign-in; fall back to pasted browser headers.
    if config.YT_OAUTH_PATH.exists():
        tok = json.loads(config.YT_OAUTH_PATH.read_text())
        if "access_token" not in tok:      # corrupt/error blob — discard it
            config.YT_OAUTH_PATH.unlink(missing_ok=True)
            raise RuntimeError("Stored YouTube sign-in was invalid and has been "
                               "cleared — click Sign in with YouTube again.")
        return YTMusic(str(config.YT_OAUTH_PATH), oauth_credentials=_oauth_creds())
    if AUTH_PATH.exists():
        return YTMusic(json.loads(AUTH_PATH.read_text()))
    raise RuntimeError("YouTube Music not connected — sign in first.")


def _artists(item):
    arts = item.get("artists") or []
    names = [a["name"] for a in arts if a.get("name")]
    return ", ".join(names) if names else (item.get("author") or "Unknown")


def _dur_ms(item):
    secs = item.get("duration_seconds")
    if secs:
        return int(secs) * 1000
    return None


# YouTube Music tags each item with a videoType. These are the authoritative
# music/non-music signal:
#   MUSIC_VIDEO_TYPE_ATV                  -> a song (audio track)      = music
#   MUSIC_VIDEO_TYPE_OMV                  -> official music video      = music
#   MUSIC_VIDEO_TYPE_OFFICIAL_SOURCE_MUSIC-> licensed music video      = music
#   MUSIC_VIDEO_TYPE_UGC                  -> user-uploaded music clip  = music
#   MUSIC_VIDEO_TYPE_PODCAST_EPISODE / episodes / anything else        = NOT music
_MUSIC_TYPES = {
    "MUSIC_VIDEO_TYPE_ATV",
    "MUSIC_VIDEO_TYPE_OMV",
    "MUSIC_VIDEO_TYPE_OFFICIAL_SOURCE_MUSIC",
    "MUSIC_VIDEO_TYPE_UGC",
}


def classify(item):
    """Return 'song', 'music_video', or 'nonmusic' for a YT Music item.

    Even though the data comes from music.youtube.com, its history can now
    include podcasts/episodes and stray non-song clips — this keeps genuine
    music (songs AND music videos) and drops everything else.
    """
    vt = (item.get("videoType") or "").upper()
    if item.get("episodeId") or "PODCAST" in vt or "EPISODE" in vt:
        return "nonmusic"
    if vt in _MUSIC_TYPES:
        return "music_video" if vt in (
            "MUSIC_VIDEO_TYPE_OMV", "MUSIC_VIDEO_TYPE_OFFICIAL_SOURCE_MUSIC") else "song"

    # No/unknown videoType → fall back to heuristics.
    arts = [a for a in (item.get("artists") or []) if a.get("name")]
    dur = item.get("duration_seconds")
    # Long content with no artist attribution ≈ podcast / mix / lecture, not a song.
    if dur and dur > 1200 and not arts:
        return "nonmusic"
    if not arts and not item.get("album"):
        return "nonmusic"
    return "song"


def sync(include_nonmusic=False):
    """Pull play history + liked + library from the logged-in session.

    Non-music items (podcasts/episodes/stray clips) are filtered out by default;
    genuine music — songs and music videos — is kept. Set include_nonmusic=True
    to keep everything.
    """
    yt = _client()
    summary = {"history": 0, "music_videos": 0, "skipped_nonmusic": 0,
               "liked": 0, "library": 0}
    from datetime import datetime, timedelta, timezone

    now = datetime.now(timezone.utc)
    rel = {"Today": 0, "Yesterday": 1}

    with connect() as con:
        # History — YT Music gives a relative "played" label, not an exact time.
        for i, item in enumerate(yt.get_history()):
            title = item.get("title")
            if not title:
                continue
            kind = classify(item)
            if kind == "nonmusic" and not include_nonmusic:
                summary["skipped_nonmusic"] += 1
                continue
            tid = upsert_track(con, title=title, artist=_artists(item),
                               album=(item.get("album") or {}).get("name"),
                               duration_ms=_dur_ms(item))
            if not tid:
                continue
            days = rel.get(item.get("played", ""), 0)
            # spread items within a day by index so plays don't collapse to one timestamp
            when = (now - timedelta(days=days, seconds=i)).isoformat()
            add_play(con, tid, when, "ytmusic")
            summary["history"] += 1
            if kind == "music_video":
                summary["music_videos"] += 1

        # Liked songs
        try:
            liked = yt.get_liked_songs(limit=500).get("tracks", [])
        except Exception:
            liked = []
        for item in liked:
            if not item.get("title"):
                continue
            if classify(item) == "nonmusic" and not include_nonmusic:
                continue
            if upsert_track(con, title=item["title"], artist=_artists(item),
                            album=(item.get("album") or {}).get("name"),
                            duration_ms=_dur_ms(item)):
                summary["liked"] += 1

        # Library songs
        try:
            lib = yt.get_library_songs(limit=1000)
        except Exception:
            lib = []
        for item in lib:
            if item.get("title"):
                if upsert_track(con, title=item["title"], artist=_artists(item),
                                album=(item.get("album") or {}).get("name"),
                                duration_ms=_dur_ms(item)):
                    summary["library"] += 1

    return summary
