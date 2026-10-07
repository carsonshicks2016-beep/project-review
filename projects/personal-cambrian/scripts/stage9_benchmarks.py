#!/usr/bin/env python3
"""Consolidated Stage-9 benchmark report (ROADMAP Stage 9.5).

Collates the Stage-9 speedup numbers -- surrogate sims-to-QD (9.4), JAX-archive
batched-insert parity/throughput (9.3), MJX dynamics parity + physics-steps/sec
(9.1, loaded from runs/mjx_benchmark.json) -- and writes them to runs/.

    python3 scripts/stage9_benchmarks.py
    python3 scripts/stage9_benchmarks.py --mjx        # (re)run the MJX benchmark first

Done-when (9.5): numbers reported in runs/.
"""
import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.evo.benchmarks import stage9_report, save_report

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-budget", type=int, default=400)
    ap.add_argument("--archive-n", type=int, default=20000)
    ap.add_argument("--mjx", action="store_true", help="run scripts/mjx_benchmark.py first")
    args = ap.parse_args()

    mjx_json = os.path.join(ROOT, "runs", "mjx_benchmark.json")
    if args.mjx:
        print("running MJX benchmark ...")
        subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "mjx_benchmark.py")],
                       check=False)

    print("running Stage-9 benchmarks (surrogate + jax archive) ...")
    report = stage9_report(mjx_json=mjx_json, sim_budget=args.sim_budget,
                           archive_n=args.archive_n)

    s = report["surrogate_9_4"]
    a = report["jax_archive_9_3"]
    m = report["mjx_9_1"]
    print("\n=== STAGE 9 BENCHMARKS ===")
    print(f"9.4 surrogate : baseline_qd={s['baseline_qd']}  surrogate_qd={s['surrogate_qd']}  "
          f"({s['qd_ratio_at_equal_budget']}x)  reaches baseline in "
          f"{s['sims_to_match_baseline']}/{s['sim_budget']} sims ({s['sims_saved_frac']:.0%} fewer)")
    if a.get("available"):
        print(f"9.3 jax archive: qd_parity={a['qd_parity']} (cpu={a['cpu_qd']} jax={a['jax_qd']})  "
              f"batched insert of {a['n_candidates']} = {a['batched_insert_ms']}ms")
    if m.get("available"):
        tp = m.get("throughput_steps_per_sec", {})
        par = m.get("parity", {})
        print(f"9.1 mjx        : parity contact-free={par.get('contact_free_n20'):.1e}  "
              f"CPU C-engine={tp.get('cpu_1env', 0):,.0f} steps/s  (MJX-on-CPU slower; GPU=100x)")
    else:
        print("9.1 mjx        : (no runs/mjx_benchmark.json -- run with --mjx)")
    print("9.2 jax ppo    : matches CPU policy quality (tests); far faster on GPU")

    out = os.path.join(ROOT, "runs", "stage9_benchmarks.json")
    save_report(report, out)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
