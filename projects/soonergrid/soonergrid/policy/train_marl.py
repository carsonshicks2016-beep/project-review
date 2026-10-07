"""
Train Parameter-Shared PPO (IPPO) Agents for Traffic Signal Control.

Uses StableBaselines3 to train a single neural network policy that controls
all 7 Norman intersections simultaneously (Parameter Sharing).
"""

import os
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from soonergrid.gym.env import NormanTrafficEnv
from soonergrid.gym.sb3_wrapper import MARLToVecEnv

ROOT = Path(__file__).resolve().parents[2]


def make_env(dt_s: float = 15.0, duration_s: float = 28800.0, shock: str = None):
    env = NormanTrafficEnv(
        dt_s=dt_s,
        episode_duration_s=duration_s,
        incident_shock=shock,
        enable_self_healing=True,
    )
    return MARLToVecEnv(env)


def train():
    print("Initializing MAPPO (IPPO) Training Environment...")
    set_random_seed(42)
    
    # 8-hour game day simulation at 15s dt = 1920 steps per episode
    # 7 agents * 1920 steps = 13,440 agent-transitions per episode
    env = make_env(dt_s=15.0, duration_s=28800.0)
    
    # Optional: evaluation environment
    eval_env = make_env(dt_s=15.0, duration_s=28800.0)

    model_dir = ROOT / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    log_dir = ROOT / "data" / "tb_logs"

    # PPO Hyperparameters tuned for MARL continuous/discrete proxy
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1920,          # Update once per episode (1920 * 7 = 13440 batch size)
        batch_size=3360,       # 4 minibatches per update
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,         # Encourage exploration
        tensorboard_log=str(log_dir),
        device="cpu",          # Force CPU to stay below fan threshold
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir),
        log_path=str(model_dir),
        eval_freq=1920,        # Evaluate every episode
        n_eval_episodes=1,     # Just 1 episode to save time
        deterministic=True,
        render=False,
    )

    print("Starting training (target: ~1 episode for rapid benchmark)...")
    start_time = time.time()
    
    # Train for 1 full episode as a proof-of-concept
    model.learn(
        total_timesteps=15000,
        callback=eval_callback,
        progress_bar=True,
    )
    
    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.1f}s")
    
    final_path = model_dir / "marl_ppo_final.zip"
    model.save(str(final_path))
    print(f"Saved final model to {final_path}")


if __name__ == "__main__":
    train()
