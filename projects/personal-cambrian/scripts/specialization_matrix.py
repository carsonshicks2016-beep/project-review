#!/usr/bin/env python3
"""Per-niche specialization tradeoff matrix (ROADMAP Stage 7.6 demonstration).

Trains one specialist policy per niche on a seed creature, then cross-evaluates
every specialist on every niche. A clear *specialist diagonal* (each niche's best
performer is its own specialist) is the evidence that the niches select for
genuinely different, non-interchangeable behaviours.

    python3 scripts/specialization_matrix.py
    python3 scripts/specialization_matrix.py --niches track ballistics iron_zone --steps 40000
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped, agent_zero, SEEDS
from personal_cambrian.evo import (
    tradeoff_matrix, column_normalize, specialist_diagonal_fraction,
    plot_tradeoff_matrix, has_matplotlib,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", default="quadruped", choices=list(SEEDS))
    ap.add_argument("--niches", nargs="+",
                    default=["track", "ballistics", "iron_zone", "endurance"])
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--ep-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    genome = SEEDS[args.seed_creature]()
    print(f"specialization matrix: {args.seed_creature}, niches={args.niches}, "
          f"{args.steps} steps each")
    t0 = time.time()
    M, names = tradeoff_matrix(genome, args.niches, train_steps=args.steps,
                               ep_steps=args.ep_steps, seed=args.seed,
                               log_fn=lambda s: print(f"  {s}  ({time.time()-t0:.0f}s)"))

    print(f"\nraw tradeoff matrix (rows=specialist, cols=evaluated-on), "
          f"wall={time.time()-t0:.0f}s")
    hdr = "            " + "".join(f"{n[:9]:>11s}" for n in names)
    print(hdr)
    for i, n in enumerate(names):
        print(f"{n[:11]:>11s} " + "".join(f"{M[i, j]:>11.2f}" for j in range(len(names))))

    frac = specialist_diagonal_fraction(M)
    winners = [names[int(np.argmax(M[:, j]))] for j in range(len(names))]
    print(f"\ncolumn winners: {dict(zip(names, winners))}")
    print(f"specialist-diagonal fraction = {frac:.0%} "
          f"({'CLEAR diagonal' if frac >= 0.5 else 'weak / no diagonal'})")

    if has_matplotlib():
        out = os.path.join(ROOT, "renders", "specialization_matrix.png")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plot_tradeoff_matrix(M, names, out)
        print(f"saved heatmap -> {out}")


if __name__ == "__main__":
    main()
