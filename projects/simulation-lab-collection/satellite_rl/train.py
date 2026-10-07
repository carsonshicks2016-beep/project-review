"""
PPO Training script for Autonomous Satellite Rendezvous and Docking.
Uses Stable-Baselines3 with a curriculum progression from terminal docking to full rendezvous.
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from satellite_rl.env import SatelliteDockingEnv


class CurriculumCallback(BaseCallback):
    """
    Progressively increases scenario distance and difficulty as the agent's
    docking success rate reaches milestone thresholds.
    """

    def __init__(self, check_freq: int = 1000, window_size: int = 40, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.window_size = window_size
        self.episode_successes = []
        self.current_level = 0.0

    def _on_step(self) -> bool:
        # Check infos for episode completions across all vectorized envs
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if done and "success" in info:
                self.episode_successes.append(1.0 if info["success"] else 0.0)
                if len(self.episode_successes) > self.window_size:
                    self.episode_successes.pop(0)

        # Periodically adjust curriculum level
        if self.n_calls % self.check_freq == 0 and len(self.episode_successes) >= 15:
            success_rate = float(np.mean(self.episode_successes))
            prev_level = self.current_level

            # Advance curriculum when success rate is high
            if success_rate >= 0.70 and self.current_level < 1.0:
                self.current_level = min(1.0, self.current_level + 0.15)
            elif success_rate < 0.25 and self.current_level > 0.0:
                self.current_level = max(0.0, self.current_level - 0.05)

            if abs(self.current_level - prev_level) > 1e-4:
                # Update all vectorized envs
                for env in self.training_env.envs:
                    env.set_curriculum_level(self.current_level)
                if self.verbose > 0:
                    print(
                        f"[Curriculum Step {self.n_calls}] Success Rate: {success_rate * 100:.1f}% -> "
                        f"Adjusted Curriculum Level to {self.current_level:.2f}"
                    )

        return True


def make_env(stage: str = "curriculum", level: float = 0.0):
    def _init():
        return SatelliteDockingEnv(stage=stage, curriculum_level=level)
    return _init


def train(
    total_timesteps: int = 300_000,
    n_envs: int = 4,
    stage: str = "curriculum",
    model_dir: str = "models",
    model_name: str = "ppo_satellite_docking"
):
    os.makedirs(model_dir, exist_ok=True)
    save_path = os.path.join(model_dir, model_name)

    print(f"--- Launching Satellite RVD PPO Training ---")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Vectorized environments: {n_envs}")
    print(f"Mode: {stage}")

    # Vectorized environments
    env_fns = [make_env(stage=stage, level=0.0) for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # PPO Hyperparameters tuned for continuous orbital mechanics
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[128, 128])
    )

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=policy_kwargs,
        verbose=1
    )

    from stable_baselines3.common.callbacks import CallbackList, EvalCallback

    # Separate evaluation environment for best brain tracking
    eval_env = DummyVecEnv([make_env(stage=stage, level=0.0)])
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=model_dir,
        eval_freq=max(1000, 4000 // n_envs),
        n_eval_episodes=10,
        deterministic=True,
        render=False,
        verbose=1
    )

    curriculum_cb = CurriculumCallback(check_freq=2000, window_size=50, verbose=1) if stage == "curriculum" else None
    callbacks = [eval_callback]
    if curriculum_cb:
        callbacks.append(curriculum_cb)

    try:
        model.learn(total_timesteps=total_timesteps, callback=CallbackList(callbacks))
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")

    # Save final model
    model.save(save_path)
    print(f"\nModel successfully saved to: {save_path}.zip")
    print(f"Best evaluation model checkpoint saved in: {model_dir}/best_model.zip")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train PPO for Satellite Rendezvous and Docking")
    parser.add_argument("--timesteps", type=int, default=250_000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--stage", type=str, default="curriculum", choices=["curriculum", "docking", "proximity", "rendezvous"])
    parser.add_argument("--dir", type=str, default="models", help="Model directory")
    parser.add_argument("--name", type=str, default="ppo_satellite_docking", help="Model filename")

    args = parser.parse_args()
    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        stage=args.stage,
        model_dir=args.dir,
        model_name=args.name
    )
