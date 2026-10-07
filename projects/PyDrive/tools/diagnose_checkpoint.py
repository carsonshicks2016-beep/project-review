"""Export deterministic agent telemetry for diagnosis.

Examples:
  PYTHONPATH="$PWD" python3 tools/diagnose_checkpoint.py ridge_race_best.pt --track club --episodes 1 --max-steps 20 --out /tmp/supra_diag_60
  PYTHONPATH="$PWD" python3 tools/diagnose_checkpoint.py 787btrue_best.pt --track akina --scenario auto --episodes 4
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from supra.diagnostics import DiagnosticPolicy, run_diagnostics


def _split_csv(value: str) -> list[str]:
    return [x.strip() for x in (value or "").split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoint", help=".pt PPO checkpoint or .npz GA champion")
    ap.add_argument("--track", default="club", help="named/legacy track to replay")
    ap.add_argument("--scenario", default="auto",
                    choices=("auto", "solo", "frozen_opponents", "true_multi"),
                    help="replay arena (auto: 78-dim PPO -> true_multi; opponents -> frozen)")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--starts", default="line,quarter,half,threequarter",
                    help="comma list: line,quarter,half,threequarter or fractions")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="control steps per episode (default full episode budget)")
    ap.add_argument("--out", default=None,
                    help="output folder (default diagnostics/<checkpoint>_<track>_<timestamp>)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--flat", action="store_true",
                    help="force flat terrain; ordinary tracks are flat by default")
    ap.add_argument("--hills", action="store_true",
                    help="opt into procedural elevation/grade/crest physics")
    ap.add_argument("--hill-scale", type=float, default=None)
    ap.add_argument("--opponents", default="",
                    help="comma-separated opponent .pt checkpoints; switches to frozen_opponents")
    ap.add_argument("--raw-trace", default="gzip", choices=("gzip", "plain", "none"),
                    help="per-step JSONL trace: gzip (default, ~15x smaller), "
                         "plain (uncompressed, 200+ MB), or none (npz only)")
    args = ap.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"FAIL: checkpoint not found: {args.checkpoint}")
        return 2

    policy = DiagnosticPolicy(args.checkpoint)
    if policy.legacy and args.hills:
        print("[diagnostics] legacy checkpoint detected; forcing flat replay and obs slicing")
    out_dir = run_diagnostics(
        checkpoint=args.checkpoint,
        track_name=args.track,
        scenario=args.scenario,
        episodes=max(1, int(args.episodes)),
        starts=_split_csv(args.starts),
        max_steps=args.max_steps,
        out=args.out,
        seed=args.seed,
        flat=args.flat,
        hills=args.hills,
        hill_scale=args.hill_scale,
        opponents=_split_csv(args.opponents),
        raw_trace=args.raw_trace,
    )
    trace_name = {"gzip": "raw_trace.jsonl.gz", "plain": "raw_trace.jsonl",
                  "none": "raw_trace.npz (jsonl skipped)"}[args.raw_trace]
    print(f"[diagnostics] wrote telemetry bundle -> {out_dir}")
    print(f"[diagnostics] raw trace: {out_dir / trace_name}")
    print(f"[diagnostics] summary:   {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
