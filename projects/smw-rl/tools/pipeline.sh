#!/usr/bin/env bash
# Full per-level pipeline: explore -> train -> evaluate.
#
#   bash tools/pipeline.sh YoshiIsland1 YoshiIsland2
#
# Exploration must come first. PPO on its own gets stuck at the first obstacle
# it cannot stumble past and stays stuck (YoshiIsland3: 600k steps, 6,230
# episodes, no movement), because the curriculum it would learn from is
# harvested from its own rollouts. The frontier archive needs no policy, so it
# can break through; training then learns from the states it found.
#
# Environment overrides:
#   ITERS=12000 STEPS=4000000 NENVS=12 EPISODES=50 bash tools/pipeline.sh LEVEL

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ITERS="${ITERS:-8000}"
ROLLOUT="${ROLLOUT:-200}"
STEPS="${STEPS:-4000000}"
NENVS="${NENVS:-12}"
EPISODES="${EPISODES:-50}"
PY="./.venv/bin/python"

if [ $# -eq 0 ]; then
  echo "usage: bash tools/pipeline.sh LEVEL [LEVEL ...]" >&2
  exit 1
fi

$PY -m smwrl.setup_core --check || {
  echo "emulator core is broken; repairing"
  $PY -m smwrl.setup_core
}

for LEVEL in "$@"; do
  echo "############ $LEVEL ############"

  if [ -f "checkpoints/$LEVEL/archive.pkl" ]; then
    echo "==> extending existing archive"
    $PY -m smwrl.explore --level "$LEVEL" --iters "$ITERS" --rollout "$ROLLOUT" --resume
  else
    echo "==> exploring (frontier archive)"
    $PY -m smwrl.explore --level "$LEVEL" --iters "$ITERS" --rollout "$ROLLOUT"
  fi

  echo "==> verifying the curriculum is usable"
  if ! $PY -m smwrl.verify --level "$LEVEL"; then
    echo "!! goal-side stages are not finishable; explore more before training" >&2
    continue
  fi

  echo "==> training"
  $PY -m smwrl.train --level "$LEVEL" --steps "$STEPS" --n-envs "$NENVS" \
      --curriculum --resume --save-every 100000

  echo "==> evaluating (unaided, fresh starts)"
  $PY -m smwrl.evaluate --level "$LEVEL" --episodes "$EPISODES"
done
