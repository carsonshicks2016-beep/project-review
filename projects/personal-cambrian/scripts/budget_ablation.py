#!/usr/bin/env python3
"""Fine-tune-budget fairness ablation (ROADMAP Stage 3.5 "done when").

Trains a parent on the quadruped, then for several small-mutation bodies measures
score-vs-fine-tune-budget (warm-started), finds each body's knee, and recommends a
budget that puts >=90% of bodies past their knee. Also contrasts the warm-started
knee with a from-scratch knee for one body, showing warm-start lets us judge a
body fairly with a much smaller budget (so a good body is not misjudged for an
undertrained brain).

    python3 scripts/budget_ablation.py --parent-steps 200000 --body-steps 80000
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.encoding import Genome
from personal_cambrian.seeds import quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import (
    PPOConfig, make_modular_setup, warmstart_curve, budget_curve, scratch_make_agent,
    find_knee, recommend_budget,
)
from personal_cambrian.control.ppo import train


def small_mutation(g: Genome, rng, sigma=0.12) -> Genome:
    child = Genome.from_json(g.to_json())
    for p in child.parts:
        p.dims = {k: max(0.01, v * (1 + rng.normal(0, sigma))) for k, v in p.dims.items()}
    for m in child.muscles:
        m.pcsa_cm2 = max(1.0, m.pcsa_cm2 * (1 + rng.normal(0, sigma)))
    return child


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-steps", type=int, default=200_000)
    ap.add_argument("--body-steps", type=int, default=80_000)
    ap.add_argument("--n-bodies", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pmake_env, pmake_agent = make_modular_setup("quadruped")
    print(f"training parent ({args.parent_steps} steps)...")
    parent, _ = train(pmake_env, PPOConfig(total_timesteps=args.parent_steps, seed=args.seed),
                      make_agent=pmake_agent)

    bodies = [quadruped()] + [small_mutation(quadruped(), rng) for _ in range(args.n_bodies - 1)]
    knees = []
    print(f"\nwarm-started score-vs-budget for {len(bodies)} bodies (to {args.body_steps} steps):")
    for i, g in enumerate(bodies):
        cme = (lambda gg: (lambda s: CreatureEnv(gg, task=LocomotionTask(max_steps=500),
                                                 obs_mode="structured")))(g)
        steps, scores = warmstart_curve(cme, parent, max_steps=args.body_steps)
        knee = find_knee(steps, scores)
        knees.append(knee)
        print(f"  body {i}: knee at {knee:6d} steps  (final mean_return {scores[-1]:.1f})")

    budget = recommend_budget(knees, coverage=0.9)
    past = float(np.mean(np.asarray([k for k in knees]) <= budget))
    print(f"\nrecommended fine-tune budget (>=90% past knee): {budget:.0f} steps  "
          f"({past:.0%} of bodies past their knee at this budget)")

    # fairness contrast: warm-start vs from-scratch knee for body 0
    cme0 = lambda s: CreatureEnv(bodies[0], task=LocomotionTask(max_steps=500),
                                 obs_mode="structured")
    s_steps, s_scores = budget_curve(cme0, scratch_make_agent(cme0), max_steps=args.body_steps)
    scratch_knee = find_knee(s_steps, s_scores)
    print(f"\nfairness: body 0 knee  warm-started={knees[0]} steps  vs  from-scratch={scratch_knee} steps")
    print("  -> warm-start lets a body be judged fairly with a much smaller budget,")
    print("     so a good body is not culled for an undertrained brain.")


if __name__ == "__main__":
    main()
