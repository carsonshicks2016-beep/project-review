#!/usr/bin/env bash
# Headless training of the gear-head race policy -> ppo_supra_v2.pt
#   ./train_gears.sh [iterations] [workers]
# Examples:
#   ./train_gears.sh                # 8000 iters, 8 workers (~3h, coexists w/ other apps)
#   ./train_gears.sh 15000 10       # longer run, more cores
#
# Resumes ppo_supra_v2.pt if it exists (continues training); the best-so-far is
# kept in ppo_supra_v2.best.pt.  Logs to v2_train.log.  Stop anytime with
# `kill -INT <PID>` (or Ctrl-C if foreground) -- it checkpoints on exit.
cd "$(dirname "$0")"

ITERS="${1:-8000}"
WORKERS="${2:-8}"

if pgrep -f "ppo_supra_v2.pt" >/dev/null; then
  echo "!! a process is already writing ppo_supra_v2.pt -- not starting a second one."
  echo "   (check:  pgrep -fl ppo_supra_v2.pt )"
  exit 1
fi

echo "training ppo_supra_v2.pt: $ITERS iters, $WORKERS workers  ->  v2_train.log"
nohup python3 run.py --ppo "$ITERS" --ppo-save ppo_supra_v2.pt --workers "$WORKERS" \
      > v2_train.log 2>&1 &
echo "started PID $!"
echo "watch the numbers:   tail -f v2_train.log"
echo "watch it drive live: ./watch_gears.sh reload"
