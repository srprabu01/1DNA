#!/usr/bin/env bash
# One-command local launch on macOS/Linux/WSL:  ./run.sh
# Creates/reuses a .venv, installs deps once, then serves the dashboard.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating .venv (first run)..."
  python3 -m venv .venv
  ./.venv/bin/python -m pip install --upgrade pip
  ./.venv/bin/python -m pip install -r requirements.txt
fi

export MUSIC_OPEN_BROWSER=1
echo "Starting Music DNA Analyzer at http://127.0.0.1:8000 ..."
exec ./.venv/bin/python -m app
