#!/bin/bash
# One command: GRIDTECH optimiser + website on http://127.0.0.1:8770  (builds the site first if needed)
cd "$(dirname "$0")"
export PATH="$PWD/.tools/node-v22.23.4-darwin-arm64/bin:$PATH"
[ -d dist ] || { [ -d node_modules ] || npm install --ignore-scripts --no-audit --no-fund; npx vite build; }
# Python with numpy, scipy, pandas, highspy, matplotlib: $PYTHON, else the repo .venv, else python3 on PATH
PY="${PYTHON:-}"
[ -z "$PY" ] && [ -x ../.venv/bin/python ] && PY=../.venv/bin/python
[ -z "$PY" ] && PY=python3
exec "$PY" ../team_app/gridtech_server.py "$@"
