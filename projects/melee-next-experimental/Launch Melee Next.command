#!/bin/zsh
cd "${0:A:h}"
if ! curl -fsS http://127.0.0.1:8776/api/state >/dev/null 2>&1; then
  mkdir -p .runtime
  nohup "$PWD/.venv/bin/python" -u -m melee_next serve > .runtime/dashboard.log 2>&1 &
  for attempt in {1..50}; do
    curl -fsS http://127.0.0.1:8776/api/state >/dev/null 2>&1 && break
    sleep .2
  done
fi
open http://127.0.0.1:8776
