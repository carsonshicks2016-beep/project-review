#!/usr/bin/env python3
"""POET open-ended curriculum on the real creature pipeline (ROADMAP Stage 8.3).

Co-evolves a population of (niche-environment, controller) pairs for a fixed seed
morphology: agents are warm-started PPO policies, environments are NicheTasks whose
difficulty is mutated upward, gated by minimal-criterion coevolution + novelty, and
agents transfer between niches. The Stage-8.3 done-when is visible in the log: the
active-env set turns over while frontier difficulty rises, and transfers happen.

    python3 scripts/poet_curriculum.py
    python3 scripts/poet_curriculum.py --niche terrain --iterations 40 --train 1500

The fast, RL-free proof of the same loop is tests/test_poet.py; this is the slow
demonstration on actual trained creatures.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import SEEDS
from personal_cambrian.sim.tasks import NICHES
from personal_cambrian.evo import POETConfig, make_creature_poet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-creature", default="quadruped", choices=list(SEEDS))
    ap.add_argument("--niche", default="terrain", choices=list(NICHES))
    ap.add_argument("--iterations", type=int, default=30)
    ap.add_argument("--train", type=int, default=1_500)
    ap.add_argument("--ep-steps", type=int, default=150)
    ap.add_argument("--max-active", type=int, default=6)
    ap.add_argument("--mc-low", type=float, default=-2.0)
    ap.add_argument("--mc-high", type=float, default=8.0)
    ap.add_argument("--repro-threshold", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import numpy as np
    genome = SEEDS[args.seed_creature]()
    niche_cls = NICHES[args.niche]

    cfg = POETConfig(max_active=args.max_active, reproduce_every=2, transfer_every=2,
                     n_children=3, max_admit=1, optimize_steps=1,
                     mc_low=args.mc_low, mc_high=args.mc_high,
                     repro_threshold=args.repro_threshold, novelty_k=3)
    poet = make_creature_poet(genome, cfg, train_steps=args.train, ep_steps=args.ep_steps,
                              hidden=64, n_envs=4, n_steps=256, seed=args.seed,
                              rng=np.random.default_rng(args.seed))
    poet.add_env(niche_cls(difficulty=0.0, max_steps=args.ep_steps), None)

    print(f"POET: {args.seed_creature} on '{args.niche}', {args.iterations} iters, "
          f"train={args.train} steps/opt")
    t0 = time.time()

    def _log(r):
        print(f"  t={r['t']:3d}  active={r['n_active']}  "
              f"frontier_diff={r['max_difficulty']:5.2f}  "
              f"+{r['added']} -{r['removed']} xfer+{r['transfers']}  "
              f"best={r['best_score']:+7.2f}  ({time.time()-t0:.0f}s)")

    poet.run(args.iterations, log_fn=_log)

    h = poet.history
    rose = h[-1]["max_difficulty"] > h[0]["max_difficulty"]
    print(f"\nfrontier difficulty: {h[0]['max_difficulty']:.2f} -> {h[-1]['max_difficulty']:.2f}")
    print(f"environments created/retired: {poet.n_added} / {poet.n_removed}")
    print(f"agent transfers: {len(poet.transfers)}")
    met = rose and poet.n_removed > 0 and len(poet.transfers) > 0
    print("DONE-WHEN MET: curriculum turns over with rising difficulty + transfers"
          if met else "no clear open-ended turnover at this budget (try more iters)")


if __name__ == "__main__":
    main()
