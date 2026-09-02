"""Single entrypoint: ``python -m app`` — used by the Docker CMD and the local
launch scripts. Serves the FastAPI app with one uvicorn worker.

Only ONE worker: each worker process holds its own db.connect() lock, and multiple
would break the single-writer SQLite serialization. Scale by machine, not workers.
"""
import os

import uvicorn

from . import config
from .main import app  # noqa: F401  (import triggers init_db once)


def main():
    if os.getenv("MUSIC_OPEN_BROWSER") == "1":
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://127.0.0.1:{config.PORT}")).start()
    uvicorn.run(app, host=config.HOST, port=config.PORT, workers=1,
                proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()
