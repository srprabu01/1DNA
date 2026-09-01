import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "audio_cache"
DB_PATH = DATA_DIR / "music.db"
TOKENS_PATH = DATA_DIR / "spotify_tokens.json"
SPOTIFY_OAUTH_PATH = DATA_DIR / "spotify_oauth.json"
YT_OAUTH_PATH = DATA_DIR / "ytmusic_oauth.json"
YT_DEVICECODE_PATH = DATA_DIR / "ytmusic_devicecode.json"

DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")

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
