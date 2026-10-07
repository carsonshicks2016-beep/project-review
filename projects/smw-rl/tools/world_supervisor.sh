#!/usr/bin/env bash
# Keep whole-game exploration and training alive, and merge shards as they land.
#
#   bash tools/world_supervisor.sh 4        # 4 explorer shards + 1 trainer
#   pkill -f world_supervisor               # stop supervising (children keep running)
#
# Exists because the whole-game stack is new and has already died three ways in
# one day: an overworld TimeoutError, a missing select() after a refactor, and a
# missing env.reset() on the seeded path. Each would have cost a whole night.
#
# Every child is resumable -- explorers checkpoint their archive every 250
# excursions and training saves every 100k steps -- so a restart costs minutes.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

SHARDS="${1:-4}"
PY="./.venv/bin/python"
ITERS="${ITERS:-6000}"
TRAIN_STEPS="${TRAIN_STEPS:-20000000}"
NENVS="${NENVS:-20}"
MERGE_EVERY="${MERGE_EVERY:-900}"     # seconds
PROMOTE_EVERY="${PROMOTE_EVERY:-1800}"   # seconds
PROMOTE_EPISODES="${PROMOTE_EPISODES:-20}"

mkdir -p runs/logs checkpoints
log() { echo "[$(date +%H:%M:%S)] supervisor: $*"; }

alive() { pgrep -f "$1" >/dev/null 2>&1; }

start_shard() {
  local i="$1"
  log "starting explorer shard $i"
  nohup $PY -m smwrl.worldrun --iters "$ITERS" --rollout 220 \
    --resume-from checkpoints/world_archive.pkl \
    --out "checkpoints/world_shard_$i.pkl" \
    >> "runs/logs/worldrun_$i.log" 2>&1 &
}

start_trainer() {
  log "starting whole-game trainer"
  nohup $PY -m smwrl.world_train --steps "$TRAIN_STEPS" --n-envs "$NENVS" --resume \
    >> runs/logs/world_train.log 2>&1 &
}

log "supervising $SHARDS explorer shard(s) + 1 trainer"
last_merge=$SECONDS
last_promote=$SECONDS

while true; do
  for i in $(seq 1 "$SHARDS"); do
    alive "worldrun.*world_shard_$i.pkl" || start_shard "$i"
    sleep 1
  done
  alive "smwrl.world_train" || start_trainer

  # Fold shard discoveries back into the shared archive so the next restart
  # (and the next trainer run) inherits everything found so far.
  if (( SECONDS - last_merge >= MERGE_EVERY )); then
    last_merge=$SECONDS
    log "merging shards"
    $PY tools/world_merge.py checkpoints/world_shard_*.pkl 2>&1 \
      | grep -E "^merged|^translevels|^frontier" | sed 's/^/           /'
  fi

  # Keep the best policy this run has produced. Training writes latest.zip in
  # place, so without this a policy that gets worse destroys the one that was
  # good -- which is exactly how the 0.49-levels checkpoint was lost.
  if (( SECONDS - last_promote >= PROMOTE_EVERY )); then
    last_promote=$SECONDS
    log "evaluating latest for promotion"
    $PY -m smwrl.world_evaluate --episodes "$PROMOTE_EPISODES" --promote --quiet 2>&1 \
      | grep -E "^episodes|^levels/episode|^promoted|^not promoted|^evaluation failed" \
      | sed 's/^/           /'
  fi
  sleep 30
done
