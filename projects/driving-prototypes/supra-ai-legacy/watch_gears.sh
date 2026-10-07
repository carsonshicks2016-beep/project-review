#!/usr/bin/env bash
# Live viewer for the gear-head race policy (ppo_supra_v2.pt).
#   ./watch_gears.sh           # watch the saved policy drive (deterministic car 1)
#   ./watch_gears.sh reload    # hot-reload the policy as ./train_gears.sh trains it
#   ./watch_gears.sh best      # watch the best-so-far checkpoint
#
# Controls: A = follow leader, F = next car, T = 8x fast, +/- zoom, SPACE pause, ESC quit
cd "$(dirname "$0")"

case "$1" in
  reload) python3 run.py --watch-ppo --ppo-save ppo_supra_v2.pt --reload ;;
  best)   python3 run.py --watch-ppo --ppo-save ppo_supra_v2.best.pt ;;
  *)      python3 run.py --watch-ppo --ppo-save ppo_supra_v2.pt ;;
esac
