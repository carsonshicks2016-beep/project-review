#!/usr/bin/env bash
# Drive a policy over a set of stages, headless, and record why every episode ended.
#
# WHY THIS EXISTS
#
# Three runs in a row plateaued against two failure modes — obstacle strikes stuck at
# 45-53 %, rollovers at 15-20 % — and neither was ever measured, because measuring them
# needs something training cannot give you: the same stage, twice, against a policy that
# is not moving underneath you. During training the stage is random, the curriculum is
# ramping, and the policy changes every few thousand steps, so any two episodes differ
# in three ways at once.
#
# This runs a FIXED policy over NAMED stages with FIXED difficulty. Everything varies
# except the thing being measured.
#
#   ./evaluate.sh                          60 random stages, 2 rocks/100 m
#   ./evaluate.sh --seeds 20260727,4711    those two stages, alternating
#   ./evaluate.sh --episodes 200 --rocks 0 200 stages with no road rocks at all
#
# That last one is the control the obstacle question has always needed: if the rollover
# and off-stage rates are unchanged with zero rocks, the rocks are not what is capping
# the finish rate.
#
# Results land in results/eval/<timestamp>/ as one JSON line per episode — the same
# format training writes, so analyse-episodes.py reads either.
#
#   .venv/bin/python analyse-episodes.py results/eval/<timestamp>
set -uo pipefail

APP="Builds/RallyTraining.app"

# The executable inside the bundle is named after Unity's PRODUCT name, not the bundle —
# here "RallyAI3" inside RallyTraining.app. Assuming they match gives a "no player" error
# on a player that exists and is perfectly fine, so ask the bundle instead of guessing.
BIN="$(ls "$APP/Contents/MacOS/" 2>/dev/null | head -1)"
[ -n "$BIN" ] && BIN="$APP/Contents/MacOS/$BIN"

EPISODES=60
SEEDS=""
ROCKS="2"
REFRESH=""
POLICY=""

usage() {
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
    cat <<'EOF'

Options:
  --episodes N      Episodes to run, then quit.        (default 60)
  --seeds a,b,c     Stage seeds to drive, in turn.     (default: random each time)
  --rocks X         Road rocks per 100 m.              (default 2)
  --refresh N       Episodes per stage before the next seed. (default 1)
  --policy FILE     Copy this .onnx into the current-policy slot first.
                    Requires a player rebuild afterwards — see the note below.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --episodes) EPISODES="$2"; shift 2 ;;
        --seeds)    SEEDS="$2";    shift 2 ;;
        --rocks)    ROCKS="$2";    shift 2 ;;
        --refresh)  REFRESH="$2";  shift 2 ;;
        --policy)   POLICY="$2";   shift 2 ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

# ── The policy under test ────────────────────────────────────────────────────
#
# The player is a SNAPSHOT. The policy it drives with was baked into the scene when it
# was built, so pointing this script at a different .onnx means rebuilding — there is no
# way to swap a brain into a built player from outside. Copying the file is the first
# half of that; Unity has to do the second.
SLOT="Assets/ML-Agents/Models/RallyDriver-current.onnx"
if [ -n "$POLICY" ]; then
    [ -f "$POLICY" ] || { echo "No such policy: $POLICY"; exit 1; }
    cp "$POLICY" "$SLOT"
    echo "Copied $(basename "$POLICY") into $SLOT."
    echo "Now rebuild the player (Rally > Training > Build Player) before this measures anything."
fi

[ -x "$BIN" ] || {
    echo "No player at $BIN."
    echo "Build one first: Unity > Rally > Training > Build Player (for headless training)."
    exit 1
}

# ── Staleness ────────────────────────────────────────────────────────────────
#
# A player built before the last script change measures the OLD game and says nothing
# about the current one. This has already cost a day once, in the other direction: the
# rally05 policy was evaluated against a physics rewrite it predated and scored 85 %.
NEWEST_SOURCE="$(find Assets -name '*.cs' -newer "$BIN" -print -quit 2>/dev/null)"
if [ -n "$NEWEST_SOURCE" ]; then
    echo
    echo "WARNING: $BIN is older than $NEWEST_SOURCE."
    echo "         It does not contain the current code, and the forensics this script"
    echo "         exists to collect are part of that code. Rebuild the player first."
    echo
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="results/eval/$STAMP"
mkdir -p "$OUT"

export RALLY_EPISODE_LOG_DIR="$PWD/$OUT"
# Ghosts from this run are labelled with it, so a lap kept here is not confused with a
# lap kept by training. Without this they all land in results/ghosts as "unlabelled".
export RALLY_GHOST_LABEL="eval-$STAMP"
export RALLY_EVAL_EPISODES="$EPISODES"
export RALLY_EVAL_ROCKS="$ROCKS"
[ -n "$SEEDS" ]   && export RALLY_EVAL_SEEDS="$SEEDS"
[ -n "$REFRESH" ] && export RALLY_EVAL_REFRESH="$REFRESH"

echo "Evaluating $EPISODES episodes at $ROCKS rocks/100 m${SEEDS:+ on seeds $SEEDS}"
echo "  -> $OUT"

# -batchmode -nographics: no window, no rendering, and it still runs the full physics
# and the full policy. The episode log is the output; the Unity log carries the stage
# report, which is where a stage that generated wrongly shows up.
#
# ABSOLUTE path. A player does not resolve -logFile against the shell's directory, so a
# relative one silently wrote nothing here and the fallback below had no log to print.
"$BIN" -batchmode -nographics -logFile "$PWD/$OUT/player.log"
STATUS=$?

# Unity exits non-zero on Application.Quit from batchmode in some versions; the episodes
# on disk are what says whether the run worked, so count them rather than trusting $?.
WRITTEN="$(cat "$OUT"/episodes-*.jsonl 2>/dev/null | wc -l | tr -d ' ')"
echo "Wrote $WRITTEN episodes (player exit $STATUS)."

if [ "$WRITTEN" = "0" ]; then
    echo "Nothing was recorded. The tail of $OUT/player.log:"
    tail -20 "$OUT/player.log" 2>/dev/null
    exit 1
fi

echo
echo "Analyse it with:"
echo "    .venv/bin/python analyse-episodes.py $OUT"
