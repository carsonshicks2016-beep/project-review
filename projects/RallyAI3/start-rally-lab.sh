#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  echo "Create the project Python environment and install rallylab/requirements.txt first."
  exit 1
fi
.venv/bin/python -c 'import fastapi, uvicorn, psutil, yaml, onnx' 2>/dev/null || {
  .venv/bin/python -m pip install -r rallylab/requirements.txt
}

if [ ! -d "dashboard/node_modules" ]; then
  npm --prefix dashboard install
fi
npm --prefix dashboard run build
exec .venv/bin/python -m rallylab --port "${RALLY_LAB_PORT:-8765}"
