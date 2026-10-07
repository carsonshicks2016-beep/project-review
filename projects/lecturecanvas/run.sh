#!/usr/bin/env bash
# LectureCanvas launcher: starts the Whisper transcription server and the web UI.
set -uo pipefail

WHISPER_PORT="${PORT:-5175}"
UI_PORT="${UI_PORT:-5174}"

cd "$(dirname "$0")"

echo "=========================================================="
echo " LectureCanvas: Classroom Transcriber & Word Visualizer    "
echo "=========================================================="

cleanup() {
  trap - SIGINT SIGTERM EXIT
  echo ""
  echo "Shutting down LectureCanvas services..."
  local pids
  pids="$(jobs -p)"
  [ -n "$pids" ] && kill $pids 2>/dev/null
  wait 2>/dev/null
  exit 0
}
trap cleanup SIGINT SIGTERM EXIT

# --- Preflight ------------------------------------------------------------
if [ ! -d node_modules ]; then
  echo "Installing web dependencies (npm install)..."
  npm install || { echo "npm install failed."; exit 1; }
fi

missing_python_deps=""
for module in flask whisper torch; do
  python3 -c "import $module" 2>/dev/null || missing_python_deps="$missing_python_deps $module"
done

if [ -n "$missing_python_deps" ]; then
  echo ""
  echo "Missing Python packages:$missing_python_deps"
  echo "The web app will still run, but importing recorded audio needs them:"
  echo ""
  echo "    python3 -m pip install -r requirements.txt"
  echo ""
  START_SERVER=0
else
  START_SERVER=1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Note: ffmpeg was not found on PATH. Whisper needs it to decode audio files."
  echo "      macOS: brew install ffmpeg"
fi

# --- Whisper server -------------------------------------------------------
if [ "$START_SERVER" = "1" ]; then
  echo "Starting Whisper transcription server on port ${WHISPER_PORT}..."
  PORT="$WHISPER_PORT" python3 server.py &

  # Wait for the port to answer rather than guessing with a fixed sleep; loading
  # torch alone can take several seconds on a cold start.
  for _ in $(seq 1 40); do
    if curl -sf "http://127.0.0.1:${WHISPER_PORT}/api/health" >/dev/null 2>&1; then
      echo "Whisper server is ready (http://127.0.0.1:${WHISPER_PORT})."
      break
    fi
    sleep 0.5
  done
fi

# --- Web UI ---------------------------------------------------------------
echo "Starting web UI on port ${UI_PORT}..."
npm run dev -- --host --port "$UI_PORT" --open &

wait
