"""Train a docking agent with PPO over the rendezvous curriculum."""

from __future__ import annotations

import argparse
from pathlib import Path

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from rendezvous.config import CURRICULUM
from rendezvous.curriculum import CurriculumCallback
from rendezvous.env import DockingEnv


def make_env(rank: int, seed: int, stage_idx: int):
    def _init():
        env = DockingEnv(stage=CURRICULUM[stage_idx])
        env.reset(seed=seed + rank)
        return Monitor(env, info_keywords=("is_success", "dv_used"))

    return _init


def build_vec_env(n_envs: int, seed: int, stage_idx: int, subproc: bool):
    fns = [make_env(i, seed, stage_idx) for i in range(n_envs)]
    venv = SubprocVecEnv(fns) if subproc and n_envs > 1 else DummyVecEnv(fns)
    return VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.997)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3_000_000)
    p.add_argument("--n-envs", type=int, default=16)
    p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stage", type=int, default=0, help="curriculum rung to start on")
    p.add_argument("--out", type=Path, default=Path("runs/ppo"))
    p.add_argument("--subproc", action="store_true")
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
            gamma=0.997,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=1e-3,
            vf_coef=0.5,
            max_grad_norm=0.5,
            # Start with a tight action distribution: coarse exploration noise
            # is larger than the docking tolerance and drowns out fine control.
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0
            ),
            tensorboard_log=str(args.out / "tb"),
            seed=args.seed,
            verbose=1,
        )
    else:
        model = SAC(
            "MlpPolicy",
            venv,
            learning_rate=3e-4,
            buffer_size=1_000_000,
            batch_size=512,
            gamma=0.997,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=str(args.out / "tb"),
            seed=args.seed,
            verbose=1,
        )

    curriculum = CurriculumCallback()
    curriculum.stage_idx = args.stage
    model.learn(total_timesteps=args.steps, callback=curriculum, progress_bar=False)

    model.save(args.out / "model")
    venv.save(str(args.out / "vecnorm.pkl"))
    print(f"saved -> {args.out/'model'}.zip")
    if curriculum.history:
        print("curriculum promotions:")
        for step, name, rate in curriculum.history:
            print(f"  {step:>9,} steps -> {name} (prev stage {rate:.0%})")


if __name__ == "__main__":
    main()
