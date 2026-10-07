#!/usr/bin/env bash
# One-glance status for an unattended run.
#
#   bash tools/status.sh
#
# The column that matters is "unaided clear" -- the rate from a fresh start.
# The training log's plain `clear` includes curriculum episodes that begin next
# to the goal and clear trivially; that once read 25% while the true rate was 0%.

set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

echo "=== processes ==="
echo "  autopilots: $(pgrep -f 'smwrl.autopilot' | wc -l | tr -d ' ')" \
     " trainers: $(pgrep -f 'smwrl.train' | wc -l | tr -d ' ')" \
     " explorers: $(pgrep -f 'smwrl.explore' | wc -l | tr -d ' ')"
ps -Ao command | grep -E "smwrl.(train|explore)" | grep -v grep \
  | grep -oE "smwrl\.[a-z]+ --level [A-Za-z0-9]+" | sort | uniq -c | sed 's/^/   /'

echo
echo "=== what each worker is doing ==="
for f in runs/autopilot_*.log; do
  [ -f "$f" ] || continue
  printf "  %s\n" "$(basename "$f" .log)"
  grep -E "steps ->|=====|PASSED|skipping|budget|extending to|reaches back" "$f" \
    | tail -3 | sed 's/^/     /'
done

echo
echo "=== finished / in-flight levels ==="
./.venv/bin/python - <<'PYEOF'
import json, glob, time
seen = False
for f in sorted(glob.glob("checkpoints/autopilot_*.json")):
    try:
        j = json.load(open(f))
    except Exception:
        continue
    for e in j:
        seen = True
        chunks = e.get("chunks") or []
        last = chunks[-1] if chunks else {}
        rate = e.get("clear_rate", last.get("clear", 0.0))
        print(f"  {e['level']:<18} {e.get('result','running'):<18} "
              f"unaided clear {rate:5.0%}   progress {last.get('progress',0):6.0f}")
if not seen:
    print("  (nothing recorded yet)")
PYEOF

echo
echo "=== live policies (stale entries marked) ==="
./.venv/bin/python - <<'PYEOF'
import json, glob, time
now = time.time()
rows = []
for f in sorted(glob.glob("checkpoints/*/status.json")):
    lv = f.split("/")[1]
    try:
        s = json.load(open(f))
    except Exception:
        continue
    age = now - s.get("updated", 0)
    rows.append((age, lv, s))
for age, lv, s in sorted(rows):
    mark = "" if age < 600 else f"   (stale {age/3600:.1f}h)"
    print(f"  {lv:<18} {s['timesteps']:>12,} steps  stage {str(s.get('curriculum_stage')):>4}"
          f"  unaided x {s.get('solo_mean_x',0):6.0f}"
          f"  unaided clear {s.get('solo_clear_rate',0):5.0%}{mark}")
PYEOF

echo
echo "=== machine ==="
top -l 2 -n 0 -s 1 2>/dev/null | grep "CPU usage" | tail -1 | sed 's/^/  /'
memory_pressure 2>/dev/null | tail -1 | sed 's/^/  /'
df -h . | tail -1 | awk '{print "  disk free "$4}'
echo
echo "watch:  ./.venv/bin/python -m smwrl.live --level YoshiIsland1 --brain"
echo "stop:   pkill -f smwrl.autopilot && pkill -f smwrl.train"
