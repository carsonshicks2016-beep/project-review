"""Train one PPO policy over the whole game, using the world archive as curriculum.

    python -m smwrl.world_train --steps 5_000_000 --n-envs 12

Unlike `smwrl.train`, there is no per-level checkpoint and no dispatcher: a
single policy plays continuously and the environment scripts the map between
levels. The score that matters is **levels cleared per episode from the first
playable state**, not distance inside one level.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage

from smwrl.policy import publish_policy
from smwrl.world_env import WorldEnv, WorldEpisodeConfig, curriculum_from_archive
from smwrl.wrappers import ObsConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "world"
RUNS = ROOT / "runs" / "tb"

# One definition, used by both the envs and the published contract. Declaring it
# twice is how a checkpoint ends up describing an observation it was not trained
# on, which no consumer can detect.
OBS = ObsConfig()
FRAME_STACK = 4


def make_env(curriculum, ratio, seed):
    def _init():
        env = WorldEnv(curriculum, RewardConfig(), WorldEpisodeConfig(),
                       OBS, curriculum_ratio=ratio)
        env.reset(seed=seed)
        return Monitor(env, info_keywords=("levels_cleared", "progress"))
    return _init


class WorldProgress(BaseCallback):
    """Reports what matters here: levels chained from an unaided start."""

    def __init__(self, outdir: Path, every: int = 20_000, save_every: int = 100_000):
        super().__init__()
        self.outdir = outdir
        self.every, self.save_every = every, save_every
        self._next, self._next_save = every, save_every
        self._t0 = time.time()
        self._start = None
        self.best = 0
        self._recent: list[int] = []
        self._solo: list[int] = []

    def _on_step(self) -> bool:
        if self._start is None:
            self._start = self.num_timesteps
            # Resuming keeps the global step count, so thresholds seeded at
            # `every` / `save_every` are already millions of steps in the past.
            # Left alone the callback fires on every single step until they catch
            # up: restarting at 2.9M steps printed 145 progress lines and
            # republished the checkpoint 29 times back to back.
            self._next = self.num_timesteps + self.every
            self._next_save = self.num_timesteps + self.save_every
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            n = int(info.get("levels_cleared", 0))
            self.best = max(self.best, n)
            self._recent = (self._recent + [n])[-100:]
            if not info.get("from_checkpoint", False):
                self._solo = (self._solo + [n])[-100:]

        if self.num_timesteps >= self._next_save:
            self._next_save += self.save_every
            self.outdir.mkdir(parents=True, exist_ok=True)
            # Publish through the contract boundary, not a bare model.save: a
            # checkpoint with no embedded observation contract cannot be loaded
            # by anything without guessing its input shape, which is exactly
            # what policy.py refuses to do. `level` is None on purpose -- this
            # policy belongs to no single level.
            publish_policy(self.model, self.outdir / "latest.zip", OBS,
                           FRAME_STACK, None)
            (self.outdir / "status.json").write_text(json.dumps({
                "timesteps": int(self.num_timesteps),
                "mean_levels": float(np.mean(self._recent)) if self._recent else 0.0,
                "solo_mean_levels": float(np.mean(self._solo)) if self._solo else 0.0,
                "best_levels": self.best, "updated": time.time()}, indent=2))

        if self.num_timesteps >= self._next:
            self._next += self.every
            rate = (self.num_timesteps - self._start) / max(1e-6, time.time() - self._t0)
            mean = float(np.mean(self._recent)) if self._recent else 0.0
            solo = float(np.mean(self._solo)) if self._solo else 0.0
            self.logger.record("world/mean_levels", mean)
            self.logger.record("world/solo_levels", solo)
            print(f"[world] {self.num_timesteps:>10,} steps | {rate:6.0f} steps/s | "
                  f"levels/episode {mean:5.2f} | SOLO {solo:5.2f} | best {self.best}",
                  flush=True)
        return True


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--archive", type=Path,
                   default=ROOT / "checkpoints" / "world_archive.pkl")
    p.add_argument("--steps", type=int, default=5_000_000)
    p.add_argument("--n-envs", type=int, default=12)
    p.add_argument("--curriculum-size", type=int, default=64)
    p.add_argument("--curriculum-ratio", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.02)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    curriculum = curriculum_from_archive(args.archive, args.curriculum_size)
    if not curriculum:
        raise SystemExit(f"no usable archive at {args.archive}; run smwrl.worldrun first")
    print(f"curriculum: {len(curriculum)} states "
          f"({sum(len(c) for c in curriculum)/1e6:.1f} MB compressed), "
          f"{args.curriculum_ratio:.0%} of resets")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    venv = SubprocVecEnv([make_env(curriculum, args.curriculum_ratio, i)
                          for i in range(args.n_envs)])
    venv = VecTransposeImage(VecFrameStack(venv, n_stack=FRAME_STACK))

    CKPT.mkdir(parents=True, exist_ok=True)
    latest = CKPT / "latest.zip"
    if args.resume and latest.exists():
        model = PPO.load(latest, env=venv, device=device)
        print(f"resuming from {latest}")
    else:
        model = PPO("CnnPolicy", venv, learning_rate=2.5e-4, n_steps=256,
                    batch_size=512, n_epochs=4, gamma=0.99, gae_lambda=0.95,
                    ent_coef=args.ent_coef, tensorboard_log=str(RUNS),
                    device=device, verbose=0)

    print(f"training whole-game on {device} with {args.n_envs} envs")
    cb = WorldProgress(CKPT)
    try:
        model.learn(total_timesteps=args.steps, callback=cb,
                    reset_num_timesteps=not args.resume, tb_log_name="world")
    except KeyboardInterrupt:
        print("\ninterrupted -- saving")
    finally:
        publish_policy(model, latest, OBS, FRAME_STACK, None)
        print(f"saved {latest}  (best levels in one episode: {cb.best})")
        venv.close()


if __name__ == "__main__":
    main()
