"""Train a drift-trick agent with PPO over the arena curriculum."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from drift.config import CURRICULUM
from drift.curriculum import CurriculumCallback
from drift.env import DriftEnv

INFO_KEYS = ("score", "best_combo", "best_multiplier", "norm_score", "wipeouts",
             "crashed", "n_switchback", "n_donut", "n_clip", "n_manji")


def make_env(rank: int, seed: int, stage_idx: int):
    def _init():
        env = DriftEnv(stage=CURRICULUM[stage_idx])
        env.reset(seed=seed + rank)
        return Monitor(env, info_keywords=INFO_KEYS)

    return _init


def build_vec_env(n_envs: int, seed: int, stage_idx: int, subproc: bool):
    fns = [make_env(i, seed, stage_idx) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if subproc and n_envs > 1 else DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", type=int, default=0, help="curriculum rung to start on")
    p.add_argument("--out", type=Path, default=Path("runs/ppo"))
    p.add_argument("--subproc", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=0)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    venv = build_vec_env(args.n_envs, args.seed, args.stage, args.subproc)

    if args.algo == "ppo":
        model = PPO(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=2048,
            n_epochs=10,
            gamma=0.995,          # ~8 s horizon at 25 Hz: long enough to value a combo
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=3e-3,        # drifting needs sustained exploration to find
            vf_coef=0.5,
            max_grad_norm=0.5,
            # Full lock is 50 deg, so unit-variance noise would saturate the
            # steering constantly. Start tight and let the policy open up.
            policy_kwargs=dict(net_arch=[256, 256], log_std_init=-0.8),
            tensorboard_log=str(args.out / "tb"),
            seed=args.seed,
            verbose=1,
        )
    else:
        model = SAC("MlpPolicy", venv, learning_rate=3e-4, buffer_size=600_000,
                    batch_size=512, gamma=0.995, train_freq=8, gradient_steps=8,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    tensorboard_log=str(args.out / "tb"), seed=args.seed, verbose=1)

    callbacks = [CurriculumCallback()]
    callbacks[0].stage_idx = args.stage
    if args.checkpoint_every:
        callbacks.append(CheckpointCallback(
            save_freq=max(1, args.checkpoint_every // args.n_envs),
            save_path=str(args.out / "ckpt"), name_prefix="drift"))

    model.learn(total_timesteps=args.steps, callback=callbacks, progress_bar=False)

    model.save(args.out / "model")
    venv.save(str(args.out / "vecnorm.pkl"))
    print(f"\nsaved {args.out/'model'}.zip and {args.out/'vecnorm.pkl'}")
    if callbacks[0].history:
        print("curriculum:", " -> ".join(f"{n}@{s:,}" for s, n, _ in callbacks[0].history))


if __name__ == "__main__":
    main()
