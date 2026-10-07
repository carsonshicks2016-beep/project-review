#!/bin/bash
set -e

APP_DIR="/Users/REVIEW_USER/Desktop/01 Core Work/PythonDojo"
cd "$APP_DIR"

themes=("Homebrew" "Pro" "Ocean" "Novel")
RANDOM_INDEX=$((RANDOM % 4))
SELECTED_THEME=${themes[$RANDOM_INDEX]}
osascript -e "tell application \"Terminal\" to set current settings of front window to settings set \"$SELECTED_THEME\"" >/dev/null 2>&1 || true

if [ ! -d ".venv" ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -e ".[dev]" >/dev/null

echo "Python Dojo"
echo "1) Browser app at http://127.0.0.1:8797"
echo "2) Terminal dojo"
printf "Choose mode [1/2]: "
read MODE

if [ "$MODE" = "2" ]; then
  python-dojo term
else
  python-dojo web --port 8797
fi
