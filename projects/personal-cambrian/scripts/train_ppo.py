#!/usr/bin/env python3
"""Train a creature to locomote with PPO (ROADMAP Stage 3.1/3.2 demonstration).

    python3 scripts/train_ppo.py                          # quadruped, 200k steps
    python3 scripts/train_ppo.py --creature quadruped --steps 300000
    python3 scripts/train_ppo.py --resume runs/<dir>      # continue a run
    python3 scripts/train_ppo.py --render                 # save a walked PNG

Uses control.train.run_training: logs the return curve + periodic eval to
runs/<dir>/metrics.jsonl, writes best.pt / last.pt checkpoints, and reports
trained-vs-untrained forward distance.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from personal_cambrian.seeds import SEEDS
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import (
    PPOConfig, ActorCritic, evaluate, run_training, make_creature_env, new_run_dir,
    make_modular_setup,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--creature", default="quadruped", choices=list(SEEDS))
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ep-steps", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--policy", default="mlp", choices=["mlp", "modular"])
    ap.add_argument("--resume", default=None, help="run dir to resume (uses its last.pt)")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    cfg = PPOConfig(total_timesteps=args.steps, n_envs=args.n_envs, seed=args.seed)
    if args.policy == "modular":
        make_env, make_agent = make_modular_setup(args.creature, ep_steps=args.ep_steps,
                                                  hidden=cfg.hidden)
    else:
        make_env = make_creature_env(args.creature, ep_steps=args.ep_steps)
        make_agent = None

    if args.resume:
        run_dir = args.resume
        resume_from = os.path.join(run_dir, "last.pt")
    else:
        run_dir = new_run_dir(args.creature)
        resume_from = None

    env0 = make_env(args.seed)
    untrained = (make_agent(None, None) if make_agent else
                 ActorCritic(env0.observation_space.shape[0], env0.action_space.shape[0], cfg.hidden))
    base = evaluate(untrained, env0, max_steps=args.ep_steps, seed=args.seed)
    env0.close()

    print(f"training PPO ({args.policy}) on {args.creature} for {args.steps} steps "
          f"(n_envs={args.n_envs}) -> {run_dir}")
    t0 = time.time()
    agent, history = run_training(make_env, cfg, run_dir, eval_every=args.eval_every,
                                  eval_seed=args.seed, checkpoint_every=max(1, args.eval_every),
                                  resume_from=resume_from, make_agent=make_agent)

    env = make_env(args.seed)
    trained = evaluate(agent, env, max_steps=args.ep_steps, seed=args.seed)
    first = next((h["mean_return"] for h in history if np.isfinite(h["mean_return"])), float("nan"))
    print(f"\nwall_time={time.time()-t0:.0f}s   checkpoints: {run_dir}/best.pt, last.pt")
    print(f"mean_return: first={first:.2f} -> last={history[-1]['mean_return']:.2f}")
    print(f"forward distance: untrained={base['distance']:+.2f}m  trained={trained['distance']:+.2f}m "
          f"(return {trained['return']:+.1f}, survived {trained['steps']})")

    if args.render:
        sys.path.insert(0, os.path.join(ROOT, "scripts"))
        from view_creature import write_png
        renv = CreatureEnv(SEEDS[args.creature](), task=LocomotionTask(max_steps=args.ep_steps),
                           obs_mode=("structured" if args.policy == "modular" else "flat"),
                           render_mode="rgb_array")
        obs, _ = renv.reset(seed=args.seed)
        for _ in range(150):
            obs, _, term, trunc, _ = renv.step(agent.act(obs, deterministic=True).astype(np.float32))
            if term or trunc:
                break
        out = os.path.join(ROOT, "renders", f"{args.creature}_ppo.png")
        write_png(out, renv.render())
        print(f"rendered -> {out}")
        renv.close()
    env.close()


if __name__ == "__main__":
    main()
