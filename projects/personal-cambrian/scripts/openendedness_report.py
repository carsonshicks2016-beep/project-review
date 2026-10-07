#!/usr/bin/env python3
"""Open-endedness report over a long run (ROADMAP Stage 8.4).

Computes the three open-endedness signals and identifies any plateau:

  * COVERAGE GROWTH + INNOVATION RATE from a real MAP-Elites QD run (a finite
    archive is *expected* to plateau -- this is where the detector earns its keep).
  * ANNECS (accumulated novel-and-solved environments) from a long, RL-free toy
    POET run, shown as the open-ended contrast (it should NOT plateau).

    python3 scripts/openendedness_report.py
    python3 scripts/openendedness_report.py --iterations 150 --fine-tune 1500

The plateau verdicts are printed; with matplotlib a combined accumulation plot is
saved to renders/openendedness.png. The fast proof of the metrics is
tests/test_openendedness.py.
"""
import argparse
import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import SEEDS
from personal_cambrian.evo import (
    run_qd, QDConfig, POET, POETConfig,
    cumulative, detect_plateau, analyze, annecs_curve, coverage_growth,
    innovation_series, plot_openendedness, has_matplotlib,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# --- a fast RL-free open-ended baseline (toy POET) -------------------------
@dataclass
class _Env:
    target: float


@dataclass
class _Agent:
    skill: float


def _toy_annecs(steps, seed=0):
    def opt(a, e, n):
        s = a.skill
        for _ in range(n):
            s += 0.5 * max(0.0, (e.target + 0.5) - s)
        return _Agent(s)

    def mut(e, rng, step=0.5):
        return _Env(max(0.0, e.target + (step if rng.random() < 0.85 else -0.25 * step)))

    cfg = POETConfig(max_active=5, reproduce_every=1, transfer_every=3, n_children=4,
                     max_admit=1, optimize_steps=2, mc_low=-0.3, mc_high=0.3,
                     repro_threshold=0.3, novelty_k=3)
    p = POET(cfg, optimize=opt, evaluate=lambda a, e: a.skill - e.target, env_mutate=mut,
             env_descriptor=lambda e: np.array([e.target]),
             env_difficulty=lambda e: e.target, rng=np.random.default_rng(seed))
    p.add_env(_Env(0.0), _Agent(0.0))
    p.run(steps)
    return annecs_curve(p.env_outcomes(), solve_threshold=0.0, novelty_threshold=0.25,
                        n_steps=steps)


def _report(name, cum, window):
    onset = detect_plateau(cum, window=window, rel_threshold=0.15)
    verdict = "SUSTAINED" if onset is None else f"PLATEAU at step {onset}"
    print(f"  {name:<18} total={cum[-1]:6.0f}   {verdict}")
    return onset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", default="quadruped", choices=list(SEEDS))
    ap.add_argument("--iterations", type=int, default=120)
    ap.add_argument("--seed-steps", type=int, default=8_000)
    ap.add_argument("--fine-tune", type=int, default=1_000)
    ap.add_argument("--ep-steps", type=int, default=150)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"open-endedness report: QD on {args.seed_creature}, {args.iterations} iters")
    t0 = time.time()
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=12, iterations=args.iterations,
                   seed_steps=args.seed_steps, fine_tune_steps=args.fine_tune,
                   n_envs=4, n_steps=256, ep_steps=args.ep_steps, hidden=64,
                   macro_rate=0.4, seed=args.seed)
    logs = []
    archive, phylo = run_qd([SEEDS[args.seed_creature]()], cfg, log_fn=logs.append)
    print(f"  QD done in {time.time()-t0:.0f}s\n")

    cov = cumulative(coverage_growth(logs, key="cells"))
    innov = cumulative(innovation_series(phylo))
    annecs = _toy_annecs(args.iterations, seed=args.seed)

    print("metric              accumulation      plateau verdict")
    p_cov = _report("QD coverage", cov, args.window)
    p_innov = _report("QD innovations", innov, args.window)
    p_ann = _report("POET ANNECS", annecs, args.window)

    print("\nfull analyze() on QD coverage growth:")
    print(" ", analyze(coverage_growth(logs, key="cells"), window=args.window))

    if has_matplotlib():
        out = os.path.join(ROOT, "renders", "openendedness.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_openendedness({"QD coverage": cov, "QD innovations": innov,
                            "POET ANNECS": annecs}, out,
                           plateaus={"QD coverage": p_cov, "QD innovations": p_innov,
                                     "POET ANNECS": p_ann})
        print(f"\nsaved plot -> {out}")


if __name__ == "__main__":
    main()
