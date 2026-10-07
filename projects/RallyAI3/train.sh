#!/usr/bin/env bash
# Start a training run that will still be alive in the morning.
#
# WHY THIS IS A SCRIPT AND NOT A COMMAND YOU REMEMBER
#
# Every long run so far has been ended by something other than a decision.
#
#   rally10 was stopped at 6.9M of 25M steps and answered nothing. The whole reason
#   it existed was to see the annealing phase — the part where the policy stops
#   experimenting and settles — and it never reached it.
#
#   Before that, training stopped SILENTLY: the Mac was set to sleep after a minute and
#   Unity was set to pause when it lost focus. The trainer looked healthy, the editor
#   looked frozen, and no error was printed by anything. That is what `caffeinate` below
#   is for, and it is not optional.
#
#   And rally08 ate its own best policy, because keep_checkpoints was smaller than
#   max_steps / checkpoint_interval, so the checkpoint at the reward peak was deleted
#   long before the run ended.
#
# Each of those is now either prevented here or asserted before the run starts.
#
#   ./train.sh rally11                      the full 25M run
#   ./train.sh rally11 --resume             pick a stopped run back up
#   ./train.sh rally11-probe --steps 2000000    a short one, to check a change
set -uo pipefail

RUN_ID="${1:-}"
shift 2>/dev/null || true

if [ -z "$RUN_ID" ]; then
    echo "Usage: ./train.sh <run-id> [--resume] [--steps N] [--envs N]"
    exit 1
fi

CONFIG="config/rally_ppo.yaml"
APP="Builds/RallyTraining.app"
ENVS=6
STEPS=""
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --resume) EXTRA+=("--resume"); shift ;;
        --steps)  STEPS="$2"; shift 2 ;;
        --envs)   ENVS="$2";  shift 2 ;;
        *) EXTRA+=("$1"); shift ;;
    esac
done

# ── Pre-flight ───────────────────────────────────────────────────────────────

[ -f "$CONFIG" ] || { echo "No config at $CONFIG"; exit 1; }
[ -x ".venv/bin/mlagents-learn" ] || { echo "No mlagents-learn in .venv"; exit 1; }
[ -d "$APP" ] || {
    echo "No player at $APP."
    echo "Build one: Unity > Rally > Training > Build Player (for headless training)."
    exit 1
}

# The player is a SNAPSHOT of the scripts at build time. Training against a stale one
# measures a game that no longer exists — which is exactly how the rally05 policy came
# to report 85 % stage completion on a car that had since been rewritten underneath it.
# Named after Unity's product name, not the bundle — see the note in evaluate.sh.
BIN="$(ls "$APP/Contents/MacOS/" 2>/dev/null | head -1)"
BIN="$APP/Contents/MacOS/$BIN"
STALE="$(find Assets -name '*.cs' -newer "$BIN" -print -quit 2>/dev/null)"
if [ -n "$STALE" ]; then
    echo
    echo "STOP: $STALE is newer than the built player."
    echo "      This run would train against the old code. Rebuild the player first:"
    echo "      Unity > Rally > Training > Build Player (for headless training)"
    echo
    read -r -p "Train anyway? [y/N] " reply
    [ "$reply" = "y" ] || exit 1
fi

if [ -d "results/$RUN_ID" ] && [[ ! " ${EXTRA[*]:-} " =~ " --resume " ]]; then
    echo "results/$RUN_ID already exists. Use --resume, or pick another run id."
    exit 1
fi

# ── The checkpoint trap ──────────────────────────────────────────────────────
#
# keep_checkpoints must be at least max_steps / checkpoint_interval or the earliest
# checkpoints are silently deleted as the run goes. rally08's reward peaked at 87.8
# around 920k steps and that checkpoint was gone before the run ended, leaving nothing
# on disk earlier than 5.5M.
KEEP=$(grep -E '^\s*keep_checkpoints:' "$CONFIG" | grep -oE '[0-9]+' | head -1)
INTERVAL=$(grep -E '^\s*checkpoint_interval:' "$CONFIG" | grep -oE '[0-9]+' | head -1)
MAXSTEPS=$(grep -E '^\s*max_steps:' "$CONFIG" | grep -oE '[0-9]+' | head -1)
[ -n "$STEPS" ] && MAXSTEPS="$STEPS"

if [ -n "$KEEP" ] && [ -n "$INTERVAL" ] && [ -n "$MAXSTEPS" ]; then
    NEEDED=$(( MAXSTEPS / INTERVAL ))
    if [ "$KEEP" -lt "$NEEDED" ]; then
        echo "STOP: keep_checkpoints is $KEEP but this run produces $NEEDED checkpoints."
        echo "      The early ones — including, historically, the best one — will be deleted."
        echo "      Raise keep_checkpoints in $CONFIG to at least $NEEDED."
        exit 1
    fi
fi

# ── A shorter run ────────────────────────────────────────────────────────────
#
# max_steps lives in the YAML and mlagents-learn has no flag for it, so --steps writes
# a copy of the config with the number changed. A copy, not an edit: a probe run must
# not quietly leave the real config set to two million steps.
if [ -n "$STEPS" ]; then
    CONFIG_COPY="$(mktemp -t rally_ppo).yaml"
    sed -E "s/^([[:space:]]*max_steps:).*/\1 $STEPS/" "$CONFIG" > "$CONFIG_COPY"
    echo "Short run: max_steps $STEPS, via $CONFIG_COPY"
    CONFIG="$CONFIG_COPY"
fi

mkdir -p results
LOG="results/$RUN_ID.log"

echo "Run        $RUN_ID"
echo "Config     $CONFIG"
echo "Player     $APP  x $ENVS"
echo "Steps      ${MAXSTEPS:-from config}"
echo "Log        $LOG"
echo "Ghosts     results/ghosts/  (labelled $RUN_ID)"
echo

# The best lap of this run is kept as a path, labelled with the run id, so it can be
# raced against the next run's best on the same stage — see Rally > Watch > Race Ghosts.
# Ghosts survive observation-space changes; policies do not.
export RALLY_GHOST_LABEL="$RUN_ID"

# ── caffeinate ───────────────────────────────────────────────────────────────
#
#   -i  no idle sleep       -m  no disk sleep       -s  no system sleep
#
# All three, and they only hold while this process runs. A run that dies because the
# machine slept leaves no error anywhere: the trainer simply stops receiving
# observations, and both halves sit there looking fine.
caffeinate -i -m -s \
    .venv/bin/mlagents-learn "$CONFIG" \
        --run-id="$RUN_ID" \
        --env="$APP" \
        --num-envs="$ENVS" \
        --no-graphics \
        "${EXTRA[@]}" 2>&1 | tee "$LOG"
