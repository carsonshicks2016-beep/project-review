#!/usr/bin/env python3
"""Surrogate-assisted QD ablation (ROADMAP Stage 9.4).

Runs MAP-Elites with and without the genome-feature surrogate on a controllable
domain (fitness predictable from features), and shows the surrogate reaches the
no-surrogate QD-score with MATERIALLY FEWER expensive sims -- the Stage-9.4 done-when.
This payoff is hardware-independent (sims are a count), so the result is real on CPU.

    python3 scripts/surrogate_ablation.py --sim-budget 400

The mechanism is unit-tested in tests/test_surrogate.py; `genome_features` +
`surrogate_assisted_qd` plug into the real creature QD by passing a creature
`evaluate_fn` (an RL rollout) in place of the toy below.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.evo import (
    surrogate_assisted_qd, SurrogateQDConfig, sims_to_reach, has_matplotlib,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_G = 8


def _cell(x):
    b = np.clip(((np.clip(x[:2], -2, 2) + 2) / 4 * _G).astype(int), 0, _G - 1)
    return (int(b[0]), int(b[1]))


def _evaluate(x):
    fit = 2.0 * np.exp(-((x[2] - 0.5) ** 2 + (x[3] - 0.5) ** 2) / 0.3) - 0.05 * (x[0] ** 2 + x[1] ** 2)
    return float(fit), _cell(x)


def _mutate(x, rng):
    return np.clip(x + rng.normal(0, 0.4, 4), -2.0, 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-budget", type=int, default=300)
    ap.add_argument("--candidates", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    seeds = [rng.uniform(-2, 2, 4) for _ in range(4)]
    feats = lambda x: np.asarray(x, float)

    base = surrogate_assisted_qd(seeds, _evaluate, _mutate, feats,
                                 SurrogateQDConfig(sim_budget=args.sim_budget,
                                                   use_surrogate=False, seed=args.seed))
    surr = surrogate_assisted_qd(seeds, _evaluate, _mutate, feats,
                                 SurrogateQDConfig(sim_budget=args.sim_budget, use_surrogate=True,
                                                   n_candidates=args.candidates, beta=args.beta,
                                                   warmup=20, seed=args.seed))
    reach = sims_to_reach(surr.qd_curve, base.qd_score)
    saved = (1 - reach / base.n_sims) if reach else 0.0
    print(f"baseline : qd={base.qd_score:7.2f}  coverage={base.coverage:3d}  sims={base.n_sims}")
    print(f"surrogate: qd={surr.qd_score:7.2f}  coverage={surr.coverage:3d}  sims={surr.n_sims}")
    print(f"\nsurrogate reaches the no-surrogate QD-score in {reach}/{base.n_sims} sims "
          f"({saved:.0%} fewer)")
    print(f"and at equal budget scores {surr.qd_score/base.qd_score:.1f}x the QD-score.")

    if has_matplotlib():
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for r, lbl in [(base, "no surrogate"), (surr, "surrogate-assisted")]:
            xs, ys = zip(*r.qd_curve)
            ax.plot(xs, ys, label=lbl)
        ax.axhline(base.qd_score, ls="--", alpha=0.4, color="gray")
        if reach:
            ax.axvline(reach, ls="--", alpha=0.4, color="green")
        ax.set_xlabel("expensive sims"); ax.set_ylabel("QD-score")
        ax.set_title("Surrogate-assisted QD: same QD-score, fewer sims")
        ax.legend()
        out = os.path.join(ROOT, "renders", "surrogate_ablation.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fig.tight_layout(); fig.savefig(out, dpi=90); plt.close(fig)
        print(f"saved plot -> {out}")


if __name__ == "__main__":
    main()
