#!/usr/bin/env python3
"""JAX PPO training demo (ROADMAP Stage 9.2).

Trains the JAX PPO either on a fast toy reacher (sanity) or on a real creature via
the MJX vectorized env (`MjxVecEnv`), reporting the learning curve + wall time.

    python3 scripts/jax_train.py --task reacher
    python3 scripts/jax_train.py --task creature --seed-creature quadruped --steps 40000

HONEST NOTE: this box is CPU-only. The JAX PPO matches the torch CPU PPO's policy
QUALITY (see tests/test_jax_ppo.py), but the "far faster" half of Stage 9 is a GPU
property -- MJX rollouts are slow on CPU. Run on an accelerator for the speedup.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.control.jax_ppo import jax_ppo_available


def _reacher(n_envs=16, D=2, H=20):
    import jax, jax.numpy as jp

    class ReachJax:
        obs_dim = D
        act_dim = D

        def __init__(self):
            self.n_envs = n_envs

        def reset(self, key):
            pos = jax.random.uniform(key, (n_envs, D), minval=-1.0, maxval=1.0)
            return (pos, jp.zeros(n_envs)), pos

        def step(self, state, act, key):
            pos, t = state
            pos = jp.clip(pos + 0.1 * jp.tanh(act), -2.0, 2.0)
            rew = -jp.linalg.norm(pos, axis=1)
            t = t + 1
            done = (t >= H).astype(jp.float32)
            newpos = jax.random.uniform(key, pos.shape, minval=-1.0, maxval=1.0)
            return (jp.where(done[:, None] > 0, newpos, pos), jp.where(done > 0, 0.0, t)), \
                pos, rew, done

    return ReachJax()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="reacher", choices=["reacher", "creature"])
    ap.add_argument("--seed-creature", default="quadruped")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=20)
    ap.add_argument("--substeps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not jax_ppo_available():
        print("jax not installed: pip install jax mujoco-mjx")
        return
    from personal_cambrian.control.jax_ppo import train, JaxPPOConfig

    if args.task == "reacher":
        env = _reacher(n_envs=args.n_envs)
        lr = 3e-3
    else:
        from personal_cambrian.control.jax_ppo import MjxVecEnv
        from personal_cambrian.seeds import SEEDS
        env = MjxVecEnv(SEEDS[args.seed_creature](), n_envs=args.n_envs,
                        n_substeps=args.substeps)
        lr = 3e-4
        print(f"creature '{args.seed_creature}': obs_dim={env.obs_dim} act_dim={env.act_dim}")

    cfg = JaxPPOConfig(total_timesteps=args.steps, n_steps=args.n_steps,
                       hidden=64, lr=lr, seed=args.seed)
    print(f"JAX PPO on '{args.task}': {args.steps} steps, {args.n_envs} parallel envs")
    t0 = time.time()
    _, hist = train(env, cfg, verbose=True)
    print(f"\nwall_time={time.time()-t0:.0f}s  "
          f"mean_return {hist[0]['mean_return']:.2f} -> {hist[-1]['mean_return']:.2f}")
    if args.task == "creature":
        print("(forward-velocity reward; CPU-slow -- run on a GPU for the speedup)")


if __name__ == "__main__":
    main()
