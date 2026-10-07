"""
Reinforcement Learning training script for drift car agents using PPO and SAC.
"""

import argparse
import os
from typing import Optional
import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from drift_rl.env import DriftGymkhanaEnv


def make_env(track_name: str = "touge", seed: int = 42):
    def _init():
        env = DriftGymkhanaEnv(track_name=track_name)
        env.reset(seed=seed)
        return env
    return _init


def train(
    algo: str = "ppo",
    track: str = "touge",
    total_timesteps: int = 100_000,
    save_dir: str = "models",
    log_dir: str = "logs",
    seed: int = 42,
    learning_rate: float = 3e-4,
    batch_size: int = 128,
):
    """Train a drift car RL agent."""
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_env(track_name=track, seed=seed)])
    eval_env = DummyVecEnv([make_env(track_name=track, seed=seed + 100)])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=max(1000, total_timesteps // 20),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(2000, total_timesteps // 10),
        save_path=save_dir,
        name_prefix=f"drift_{algo}_{track}",
    )

    print(f"============================================================")
    print(f" Starting Drift RL Training")
    print(f" Algorithm:    {algo.upper()}")
    print(f" Track:        {track}")
    print(f" Timesteps:    {total_timesteps:,}")
    print(f" Learning Rate: {learning_rate}")
    print(f" Save Dir:     {save_dir}")
    print(f"============================================================")

    if algo.lower() == "sac":
        model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=200_000,
            batch_size=batch_size,
            ent_coef="auto",
            gamma=0.99,
            tau=0.005,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=dict(net_arch=[256, 256]),
        )
    else:
        # PPO default
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=log_dir,
            seed=seed,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
    )

    final_model_path = os.path.join(save_dir, f"drift_{algo}_{track}_final.zip")
    model.save(final_model_path)
    print(f"\n[DONE] Training complete! Model saved to: {final_model_path}")
    return model, final_model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML Drift Agent")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"], help="RL algorithm")
    parser.add_argument("--track", type=str, default="touge", choices=["touge", "gymkhana", "stadium"], help="Track")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total training timesteps")
    parser.add_argument("--save-dir", type=str, default="models", help="Model checkpoint directory")
    parser.add_argument("--log-dir", type=str, default="logs", help="Tensorboard log directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    train(
        algo=args.algo,
        track=args.track,
        total_timesteps=args.timesteps,
        save_dir=args.save_dir,
        log_dir=args.log_dir,
        learning_rate=args.lr,
    )
