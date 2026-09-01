"""Spotify ingestion.

Primary path: real "Sign in with Spotify" OAuth (authorization-code flow). Needs a
one-time free app registration (SPOTIFY_CLIENT_ID / SECRET in .env). After the user
clicks Sign in and approves, we hold an auto-refreshing token — no re-pasting.

Fallback paths (keyless, unofficial): paste your `sp_dc` cookie, or a short-lived
Bearer token from DevTools. Used only if you'd rather not register an app.
"""
import base64
import json
import time
import urllib.parse

import requests

from .. import config
from ..db import connect, upsert_track, add_play

API = "https://api.spotify.com/v1"
AUTH_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = "user-read-recently-played user-top-read user-library-read playlist-read-private"
COOKIE_PATH = config.DATA_DIR / "spotify_cookie.json"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/125.0 Safari/537.36")


# ---------- OAuth sign-in (recommended) ----------

def oauth_configured():
    return bool(config.SPOTIFY_CLIENT_ID and config.SPOTIFY_CLIENT_SECRET)


def auth_url():
    return AUTH_URL + "?" + urllib.parse.urlencode({
        "client_id": config.SPOTIFY_CLIENT_ID,
        "response_type": "code",
        "redirect_uri": config.SPOTIFY_REDIRECT_URI,
        "scope": SCOPES,
    })


def _basic_auth():
    raw = f"{config.SPOTIFY_CLIENT_ID}:{config.SPOTIFY_CLIENT_SECRET}".encode()
    return {"Authorization": "Basic " + base64.b64encode(raw).decode()}


def exchange_code(code):
    r = requests.post(TOKEN_URL, headers=_basic_auth(), timeout=30, data={
        "grant_type": "authorization_code", "code": code,
        "redirect_uri": config.SPOTIFY_REDIRECT_URI})
    r.raise_for_status()
    t = r.json()
    t["expires_at"] = time.time() + t.get("expires_in", 3600)
    t["source"] = "oauth"
    config.SPOTIFY_OAUTH_PATH.write_text(json.dumps(t))
    return True


def _oauth_token():
    if not config.SPOTIFY_OAUTH_PATH.exists():
        return None
    t = json.loads(config.SPOTIFY_OAUTH_PATH.read_text())
    if time.time() > t.get("expires_at", 0) - 60:
        r = requests.post(TOKEN_URL, headers=_basic_auth(), timeout=30, data={
            "grant_type": "refresh_token", "refresh_token": t["refresh_token"]})
        r.raise_for_status()
        fresh = r.json()
        t["access_token"] = fresh["access_token"]
        if fresh.get("refresh_token"):
            t["refresh_token"] = fresh["refresh_token"]
        t["expires_at"] = time.time() + fresh.get("expires_in", 3600)
        config.SPOTIFY_OAUTH_PATH.write_text(json.dumps(t))
    return t["access_token"]


# ---------- credential storage ----------

def save_cookie(sp_dc: str):
    sp_dc = sp_dc.strip().strip('"')
    if sp_dc.lower().startswith("sp_dc="):
        sp_dc = sp_dc.split("=", 1)[1]
    # validate BEFORE saving so a bad cookie doesn't get persisted
    try:
        tok, _ = _token_from_cookie(sp_dc)
    except Exception as e:
        raise RuntimeError(
            f"{e}. If your cookie is definitely fresh, Spotify may have changed its "
            "token flow — use the 'Token fallback' option instead."
        )
    COOKIE_PATH.write_text(json.dumps({"sp_dc": sp_dc}))
    return bool(tok)


def save_bearer(token: str):
    token = token.strip()
    if token.lower().startswith("bearer "):
        token = token.split(" ", 1)[1]
    config.TOKENS_PATH.write_text(json.dumps({
        "access_token": token,
        "expires_at": time.time() + 3300,  # assume ~55 min of life
        "source": "bearer",
    }))
    return True


def connected():
    return (config.SPOTIFY_OAUTH_PATH.exists() or COOKIE_PATH.exists()
            or config.TOKENS_PATH.exists())


# ---------- cookie -> token exchange ----------

def _totp():
    import pyotp
    secret_cipher = [12, 56, 76, 33, 88, 44, 88, 33, 78, 78, 11, 66, 22, 22, 55, 69, 54]
    transformed = [e ^ ((t % 33) + 9) for t, e in enumerate(secret_cipher)]
    joined = "".join(str(n) for n in transformed)
    secret = base64.b32encode(bytes.fromhex(joined.encode().hex())).decode().rstrip("=")
    return pyotp.TOTP(secret, digits=6, interval=30)


def _token_from_cookie(sp_dc):
    s = requests.Session()
    s.headers.update({"User-Agent": UA, "App-Platform": "WebPlayer",
                      "Accept": "application/json"})
    s.cookies.set("sp_dc", sp_dc, domain=".spotify.com")
    try:
        server_time = s.get("https://open.spotify.com/api/server-time",
                            timeout=15).json().get("serverTime", int(time.time()))
    except Exception:
        server_time = int(time.time())
    otp = _totp().at(int(time.time()))
    params = {"reason": "init", "productType": "web-player", "totp": otp,
              "totpServer": otp, "totpVer": 5, "sTime": server_time,
              "cTime": int(time.time() * 1000)}
    r = s.get("https://open.spotify.com/api/token", params=params, timeout=15)
    r.raise_for_status()
    data = r.json()
    if data.get("isAnonymous") or "accessToken" not in data:
        raise RuntimeError("Cookie rejected or expired — re-copy sp_dc from a "
                           "logged-in open.spotify.com session.")
    return data["accessToken"], data.get("accessTokenExpirationTimestampMs", 0)


def _access_token():
    # 0) OAuth sign-in (preferred, auto-refreshing)
    tok = _oauth_token()
    if tok:
        return tok
    # 1) cookie flow (auto-refreshing)
    if COOKIE_PATH.exists():
        sp_dc = json.loads(COOKIE_PATH.read_text())["sp_dc"]
        # cached web token still valid?
        if config.TOKENS_PATH.exists():
            cached = json.loads(config.TOKENS_PATH.read_text())
            if cached.get("source") == "cookie" and time.time() < cached.get("expires_at", 0) - 60:
                return cached["access_token"]
        token, exp_ms = _token_from_cookie(sp_dc)
        config.TOKENS_PATH.write_text(json.dumps({
            "access_token": token,
            "expires_at": (exp_ms / 1000) if exp_ms else time.time() + 3300,
            "source": "cookie",
        }))
        return token
    # 2) manually pasted bearer token
    if config.TOKENS_PATH.exists():
        cached = json.loads(config.TOKENS_PATH.read_text())
        if time.time() < cached.get("expires_at", 0):
            return cached["access_token"]
        raise RuntimeError("Pasted Spotify token expired — paste a fresh one, or use the cookie method.")
    return None


# ---------- API calls ----------

def _get(path, token, params=None):
    r = requests.get(f"{API}{path}", headers={"Authorization": f"Bearer {token}"},
                     params=params or {}, timeout=30)
    if r.status_code == 429:
        time.sleep(int(r.headers.get("Retry-After", 2)) + 1)
        return _get(path, token, params)
    if r.status_code == 401:
        raise RuntimeError("Spotify rejected the session (401) — cookie/token expired.")
    r.raise_for_status()
    return r.json()


def _store_track(con, item):
    artists = ", ".join(a["name"] for a in item.get("artists", []))
    album = item.get("album") or {}
    return upsert_track(
        con, title=item.get("name"), artist=artists, album=album.get("name"),
        duration_ms=item.get("duration_ms"), spotify_id=item.get("id"),
        popularity=item.get("popularity"), release_date=album.get("release_date"),
    )


def sync():
    """Pull recently played, top items, and saved library. Returns a summary dict."""
    token = _access_token()
    if not token:
        raise RuntimeError("Spotify not connected — paste your sp_dc cookie first.")

    summary = {"plays": 0, "saved": 0, "top_tracks": 0, "top_artists": 0}
    with connect() as con:
        recent = _get("/me/player/recently-played", token, {"limit": 50})
        for it in recent.get("items", []):
            tid = _store_track(con, it["track"])
            if tid:
                add_play(con, tid, it["played_at"], "spotify")
                summary["plays"] += 1

        for rng in ("short_term", "medium_term", "long_term"):
            top_t = _get("/me/top/tracks", token, {"limit": 50, "time_range": rng})
            for rank, it in enumerate(top_t.get("items", []), 1):
                tid = _store_track(con, it)
                con.execute(
                    """INSERT INTO top_items (source, item_type, time_range, rank, name, artist, track_id)
                       VALUES ('spotify','track',?,?,?,?,?)
                       ON CONFLICT(source, item_type, time_range, rank) DO UPDATE SET
                         name=excluded.name, artist=excluded.artist,
                         track_id=excluded.track_id, fetched_at=CURRENT_TIMESTAMP""",
                    (rng, rank, it["name"],
                     ", ".join(a["name"] for a in it.get("artists", [])), tid),
                )
                summary["top_tracks"] += 1

            top_a = _get("/me/top/artists", token, {"limit": 50, "time_range": rng})
            for rank, it in enumerate(top_a.get("items", []), 1):
                genres = ", ".join(it.get("genres", [])[:3]) or None
                con.execute(
                    """INSERT INTO top_items (source, item_type, time_range, rank, name, artist)
                       VALUES ('spotify','artist',?,?,?,?)
                       ON CONFLICT(source, item_type, time_range, rank) DO UPDATE SET
                         name=excluded.name, artist=excluded.artist, fetched_at=CURRENT_TIMESTAMP""",
                    (rng, rank, it["name"], genres),
                )
                summary["top_artists"] += 1

        offset = 0
        while offset < 500:
            saved = _get("/me/tracks", token, {"limit": 50, "offset": offset})
            items = saved.get("items", [])
            if not items:
                break
            for it in items:
                if _store_track(con, it["track"]):
                    summary["saved"] += 1
            offset += 50
            if not saved.get("next"):
                break

    return summary
