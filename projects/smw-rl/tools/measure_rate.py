"""Measure exact training throughput from the published status.json.

Sampling the console log is too coarse: it prints every 20k steps, so a 75s
window carries +/-266 steps/s of quantisation error. status.json records both
the exact timestep and the exact wall-clock time it was written, so two samples
give an exact rate.

    python tools/measure_rate.py --level YoshiIsland1 --publishes 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(path: Path):
    try:
        d = json.loads(path.read_text())
        return int(d["timesteps"]), float(d["updated"])
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="YoshiIsland1")
    ap.add_argument("--publishes", type=int, default=2,
                    help="how many publish intervals to average over")
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()

    path = ROOT / "checkpoints" / args.level / "status.json"
    first = read(path)
    while first is None:
        time.sleep(2)
        first = read(path)

    # Wait for the next publish so the window starts on a clean boundary.
    start = first
    t_end = time.time() + args.timeout
    while time.time() < t_end:
        cur = read(path)
        if cur and cur[0] != start[0]:
            start = cur
            break
        time.sleep(2)

    seen = 0
    last = start
    while time.time() < t_end and seen < args.publishes:
        cur = read(path)
        if cur and cur[0] != last[0]:
            last = cur
            seen += 1
        time.sleep(2)

    d_steps = last[0] - start[0]
    d_time = last[1] - start[1]
    if d_steps <= 0 or d_time <= 0:
        print("no progress observed (is training running?)")
        sys.exit(1)
    rate = d_steps / d_time
    print(f"{rate:8.0f} steps/s   ({d_steps:,} steps over {d_time:.1f}s)"
          f"   -> {5e6 / rate / 3600:.2f}h per 5M")


if __name__ == "__main__":
    main()
