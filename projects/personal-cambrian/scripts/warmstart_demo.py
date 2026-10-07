#!/usr/bin/env python3
"""Demonstrate parent->child warm-start (ROADMAP Stage 3.4 "done when").

Trains a parent modular policy on the quadruped, makes a small-mutation child
(perturbed limb dimensions + muscle PCSA, same topology), then shows:
  (1) the warm-started child starts ABOVE random (0 fine-tune), and
  (2) after a short fine-tune it beats a from-scratch policy at the same budget.

    python3 scripts/warmstart_demo.py --parent-steps 250000 --child-steps 60000
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
    PPOConfig, ModularPolicy, policy_for_env, evaluate,
    warm_start, warm_started_make_agent, make_modular_setup,
)
from personal_cambrian.control.ppo import train


def small_mutation(g: Genome, rng, sigma=0.12) -> Genome:
    """Perturb part dims + muscle PCSA (topology preserved => same actuator count)."""
    child = Genome.from_json(g.to_json())
    for p in child.parts:
        p.dims = {k: max(0.01, v * (1 + rng.normal(0, sigma))) for k, v in p.dims.items()}
    for m in child.muscles:
        m.pcsa_cm2 = max(1.0, m.pcsa_cm2 * (1 + rng.normal(0, sigma)))
    return child


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-steps", type=int, default=250_000)
    ap.add_argument("--child-steps", type=int, default=60_000)
    ap.add_argument("--ep-steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    # --- train the parent on the quadruped --------------------------------
    pmake_env, pmake_agent = make_modular_setup("quadruped", ep_steps=args.ep_steps)
    print(f"training parent ({args.parent_steps} steps)...")
    parent, ph = train(pmake_env, PPOConfig(total_timesteps=args.parent_steps, seed=args.seed),
                       make_agent=pmake_agent)
    parent_level = ph[-1]["mean_return"]

    # --- a small-mutation child -------------------------------------------
    child_genome = small_mutation(quadruped(), rng)
    cme = lambda s: CreatureEnv(child_genome, task=LocomotionTask(max_steps=args.ep_steps),
                                obs_mode="structured")
    child_env = cme(0)
    node_dim = child_env.obs_builder.node_dim
    n_nodes = child_env.obs_builder.n_nodes
    adj = child_env.actuator_adjacency

    # --- learning-curve comparison: warm-started vs from-scratch -----------
    # Stochastic TRAINING return is the honest discriminator: a random policy
    # thrashes and falls; a warm-started one already has a competent gait. (A
    # deterministic eval is misleading here — a ~zero policy just stands and
    # banks the alive-bonus.)
    cfg = PPOConfig(total_timesteps=args.child_steps, seed=args.seed)
    warm_ma = warm_started_make_agent(cme, parent)
    scratch_ma = lambda o, a: ModularPolicy(node_dim, n_nodes, adj, parent.hidden, parent.rounds)

    print(f"\nparent reached mean_return {parent_level:.1f}; fine-tuning the mutated child "
          f"for {args.child_steps} steps (warm vs scratch)...")
    _, wh = train(cme, cfg, make_agent=warm_ma)
    _, sh = train(cme, cfg, make_agent=scratch_ma)

    def first(h):
        return next((r["mean_return"] for r in h if np.isfinite(r["mean_return"])), float("nan"))
    thr = 0.7 * parent_level

    def steps_to(h):
        return next((r["global_step"] for r in h
                     if np.isfinite(r["mean_return"]) and r["mean_return"] >= thr), None)

    print(f"\n  start-of-finetune mean_return:  scratch {first(sh):6.2f}   warm {first(wh):6.2f}"
          f"   <- warm starts above random")
    print(f"  end-of-finetune   mean_return:  scratch {sh[-1]['mean_return']:6.2f}   warm {wh[-1]['mean_return']:6.2f}")
    print(f"  steps to reach {thr:5.1f} return:    scratch {steps_to(sh)}   warm {steps_to(wh)}")
    child_env.close()


if __name__ == "__main__":
    main()
