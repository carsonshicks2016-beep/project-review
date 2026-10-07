#!/usr/bin/env bash
# ------------------------------------------------------------
# run_live_ensemble.sh – quick launcher for the PPO council
#
# Requirements:
#   • Alpaca API keys in a .env file (ALPACA_API_KEY / ALPACA_SECRET_KEY)
#   • ALPACA_PAPER=true for paper‑trading (default) or false for live
#   • Python 3.10+ and the project’s dependencies installed
#
# Usage examples:
#   # 1️⃣ Paper‑trading (no real money) – safe default
#   ./run_live_ensemble.sh --dry-run
#
#   # 2️⃣ Live trading (real money) – only if you understand the risk
#   ./run_live_ensemble.sh --allow-live
#
#   # 3️⃣ Change the polling interval (seconds) – e.g. every 5 min
#   ./run_live_ensemble.sh --interval 300
# ------------------------------------------------------------

set -euo pipefail

# ---- Helper -------------------------------------------------
usage() {
  echo "Usage: $0 [--dry-run] [--allow-live] [--interval <seconds>]"
  echo
  echo "Options:"
  echo "  --dry-run           Run in paper‑mode without sending orders to Alpaca"
  echo "  --allow-live        Permit real‑money trading when ALPACA_PAPER=false"
  echo "  --interval <sec>    Seconds between each evaluation loop (default 300)"
  exit 1
}

# ---- Parse arguments ----------------------------------------
DRY_RUN=false
ALLOW_LIVE=false
INTERVAL=300

while (( "$#" )); do
  case "$1" in
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --allow-live)
      ALLOW_LIVE=true
      shift
      ;;
    --interval)
      if [ -z "${2:-}" ]; then
        echo "Error: --interval requires a numeric argument"
        usage
      fi
      INTERVAL=$2
      shift 2
      ;;
    -h|--help)
      usage
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
done

# ---- Build the command ---------------------------------------
CMD=(python3 live_ensemble.py)

if $DRY_RUN; then
  CMD+=(--dry-run)
fi

if $ALLOW_LIVE; then
  CMD+=(--allow-live)
fi

CMD+=(--interval "$INTERVAL")

# ---- Run ----------------------------------------------------
echo "=== Launching PPO Council ==="
echo "  Dry‑run mode:      $DRY_RUN"
echo "  Allow live money: $ALLOW_LIVE"
echo "  Evaluation period: $INTERVAL seconds"
echo "  Command: ${CMD[*]}"
echo "--------------------------------"

# Execute the Python script
"${CMD[@]}"
