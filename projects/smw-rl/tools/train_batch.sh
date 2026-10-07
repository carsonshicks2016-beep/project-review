#!/usr/bin/env bash
# Train several levels concurrently.
#
# Four concurrent runs at 12 envs each is the measured throughput peak on a
# 12-core M2 Pro (~2,092 steps/s aggregate, vs ~1,180 for one wide 32-env run).
# See the performance table in README.md, or re-measure with
# tools/bench_scaling.py.
#
#   bash tools/train_batch.sh YoshiIsland1 YoshiIsland2 YoshiIsland3 YoshiIsland4
#
# Logs land in runs/train_<level>.log. Stop everything with:
#   pkill -f smwrl.train
#
# Note: with --resume, --steps is the number of ADDITIONAL steps to train,
# not an absolute target.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STEPS="${STEPS:-10000000}"
NENVS="${NENVS:-12}"
PY="./.venv/bin/python"

if [ $# -eq 0 ]; then
  echo "usage: bash tools/train_batch.sh LEVEL [LEVEL ...]" >&2
  exit 1
fi

for LEVEL in "$@"; do
  if [ ! -f "checkpoints/$LEVEL/curriculum.meta.json" ]; then
    echo "missing clear-proven curriculum for $LEVEL; run tools/pipeline.sh first" >&2
    exit 1
  fi
  if ! $PY -m smwrl.verify --level "$LEVEL" >/dev/null; then
    echo "curriculum verification failed for $LEVEL; refusing batch training" >&2
    exit 1
  fi
done

for LEVEL in "$@"; do
  echo "==> launching $LEVEL ($NENVS envs, +$STEPS steps)"
  nohup $PY -m smwrl.train --level "$LEVEL" --steps "$STEPS" --n-envs "$NENVS" \
    --curriculum --resume --save-every 100000 > "runs/train_$LEVEL.log" 2>&1 &
  sleep 3
done

echo
echo "launched $# trainers. watch any of them with:"
echo "    $PY -m smwrl.live --level $1 --brain"
echo "progress:  tail -f runs/train_<level>.log"
