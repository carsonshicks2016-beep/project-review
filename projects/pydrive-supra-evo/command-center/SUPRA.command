#!/bin/bash
# Supra Command Center launcher — double-click to start the dashboard.
cd "$(dirname "$0")/.." || exit 1
PORT=8770

# already running? just open the browser.
if curl -s "http://localhost:$PORT/api/status" >/dev/null 2>&1; then
  echo "Command Center already running."
  open "http://localhost:$PORT"
  exit 0
fi

echo "Starting Supra Command Center…"
python3 command-center/server.py &
SERVER_PID=$!

# wait for the server to answer, then open the browser.
for i in $(seq 1 30); do
  if curl -s "http://localhost:$PORT/api/status" >/dev/null 2>&1; then
    break
  fi
  sleep 0.4
done
open "http://localhost:$PORT"

echo ""
echo "  Dashboard:  http://localhost:$PORT"
echo "  Server PID: $SERVER_PID   (close this window or Ctrl-C to stop)"
echo ""
wait $SERVER_PID
