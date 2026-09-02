import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env FIRST so its values are available below. load_dotenv does NOT override
# real environment variables, so container/compose-injected values always win.
load_dotenv(BASE_DIR / ".env")


def _path_env(var, default):
    return Path(os.getenv(var, str(default)))


# Paths are env-overridable so a container can put the DB + cache on a mounted volume.
DATA_DIR = _path_env("MUSIC_DATA_DIR", BASE_DIR / "data")
CACHE_DIR = _path_env("MUSIC_CACHE_DIR", DATA_DIR / "audio_cache")
DB_PATH = _path_env("MUSIC_DB_PATH", DATA_DIR / "music.db")
TOKENS_PATH = DATA_DIR / "spotify_tokens.json"
SPOTIFY_OAUTH_PATH = DATA_DIR / "spotify_oauth.json"
YT_OAUTH_PATH = DATA_DIR / "ytmusic_oauth.json"
YT_DEVICECODE_PATH = DATA_DIR / "ytmusic_devicecode.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Server binding — default to loopback for a bare local run; Docker sets 0.0.0.0.
HOST = os.getenv("MUSIC_HOST", "127.0.0.1")
PORT = int(os.getenv("MUSIC_PORT", "8000"))

SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "")
SPOTIFY_REDIRECT_URI = os.getenv(
    "SPOTIFY_REDIRECT_URI", "http://127.0.0.1:8000/callback/spotify"
)
LASTFM_API_KEY = os.getenv("LASTFM_API_KEY", "")

# YouTube Music OAuth (Google Cloud "TV and Limited Input" OAuth client)
# NOTE: dead end — YT Music internal API rejects TV-client tokens. Kept for compat.
YT_OAUTH_CLIENT_ID = os.getenv("YT_OAUTH_CLIENT_ID", "")
YT_OAUTH_CLIENT_SECRET = os.getenv("YT_OAUTH_CLIENT_SECRET", "")

# YouTube Data API v3 — official "Sign in with Google" (Web application OAuth client)
YTDATA_CLIENT_ID = os.getenv("YTDATA_CLIENT_ID", "")
YTDATA_CLIENT_SECRET = os.getenv("YTDATA_CLIENT_SECRET", "")
YTDATA_REDIRECT_URI = os.getenv("YTDATA_REDIRECT_URI",
                                "http://127.0.0.1:8000/callback/youtube")
YTDATA_TOKEN_PATH = DATA_DIR / "youtube_data_token.json"
