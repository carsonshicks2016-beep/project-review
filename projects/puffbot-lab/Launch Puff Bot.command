#!/bin/zsh
# Opens the Puff Bot Lab dashboard, starting its server if it is not already running.
# Training runs are separate processes; closing this window or the browser never stops one.
cd "${0:A:h}"
if ! curl -fsS http://127.0.0.1:8777/api/system >/dev/null 2>&1; then
  mkdir -p .runtime
  nohup "$PWD/.venv/bin/python" -u -m puffbot serve --no-browser --port 8777 > .runtime/dashboard.log 2>&1 &
  for attempt in {1..50}; do
    curl -fsS http://127.0.0.1:8777/api/system >/dev/null 2>&1 && break
    sleep 0.2
  done
fi
open http://127.0.0.1:8777
