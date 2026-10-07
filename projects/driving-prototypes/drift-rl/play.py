"""Watch a policy (or the scripted pilot, or yourself) drift.

  python3 play.py --model runs/ppo1            # a trained agent
  python3 play.py --pilot                      # the equilibrium baseline
  python3 play.py --keyboard                   # arrows + space, have a go
  python3 play.py --model runs/ppo1 --gif out.gif --episodes 1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from drift.config import CURRICULUM, stage_by_name
from drift.env import DriftEnv
from drift.pilot import DriftPilot
from drift.viewer import Viewer


def load_policy(path: Path):
    from stable_baselines3 import PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    zip_path = path if path.suffix == ".zip" else path / "model.zip"
    try:
        model = PPO.load(zip_path, device="cpu")
    except Exception:
        model = SAC.load(zip_path, device="cpu")

    norm_path = (path.parent if path.suffix == ".zip" else path) / "vecnorm.pkl"
    norm = None
    if norm_path.exists():
        shell = DummyVecEnv([lambda: DriftEnv()])
        norm = VecNormalize.load(str(norm_path), shell)
        norm.training = False
    return model, norm


def keyboard_action(pygame) -> np.ndarray:
    k = pygame.key.get_pressed()
    steer = (1.0 if k[pygame.K_LEFT] else 0.0) - (1.0 if k[pygame.K_RIGHT] else 0.0)
    pedal = (1.0 if k[pygame.K_UP] else 0.0) - (1.0 if k[pygame.K_DOWN] else 0.0)
    hand = 1.0 if k[pygame.K_SPACE] else -1.0
    return np.array([steer, pedal, hand], dtype=np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--model", type=Path, help="run directory or .zip")
    src.add_argument("--pilot", action="store_true", help="scripted equilibrium driver")
    src.add_argument("--keyboard", action="store_true", help="drive it yourself")
    p.add_argument("--stage", default="gymkhana",
                   choices=[s.name for s in CURRICULUM])
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--chase", action="store_true", help="camera follows the car")
    p.add_argument("--gif", type=Path, help="write the run to an animated gif")
    p.add_argument("--fps", type=int, default=25)
    args = p.parse_args()

    stage = stage_by_name(args.stage)
    env = DriftEnv(stage=stage)
    viewer = Viewer(headless=bool(args.gif), chase=args.chase,
                    title=f"drift-rl - {args.stage}")
    import pygame

    model = norm = None
    if args.model:
        model, norm = load_policy(args.model)
        label = f"policy {args.model}"
    elif args.pilot:
        label = "scripted pilot"
    else:
        label = "keyboard: arrows + space"

    frames, scores = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        viewer.reset(env)
        pilot = DriftPilot(slip_deg=38.0, speed=16.0, flip_every=2.5,
                           arena=env.arena) if args.pilot else None
        while True:
            if model is not None:
                o = norm.normalize_obs(obs[None]) if norm is not None else obs[None]
                action, _ = model.predict(o, deterministic=True)
                action = action[0]
            elif pilot is not None:
                st, th, hb = pilot.act(env.state)
                action = np.array([st, th, hb * 2 - 1])
            else:
                action = keyboard_action(pygame)

            obs, _r, term, trunc, info = env.step(action)
            viewer.draw(env, f"{label}   ep {ep + 1}/{args.episodes}")
            if args.gif:
                frames.append(viewer.frame_rgb())
            elif not viewer.flip(args.fps):
                term = True
            if term or trunc:
                break

        scores.append(info.get("score", 0.0))
        tricks = " ".join(f"{k[2:]}={v}" for k, v in sorted(info.items())
                          if k.startswith("n_") and v)
        print(f"ep {ep + 1}: score {info.get('score', 0):9,.0f}  "
              f"best combo {info.get('best_combo', 0):8,.0f}  "
              f"max x{info.get('best_multiplier', 1):.1f}  "
              f"{'CRASHED' if info.get('crashed') else 'clean  '}  {tricks}")

    print(f"\nmean score over {len(scores)} runs: {np.mean(scores):,.0f}")
    if args.gif:
        import imageio.v2 as iio
        iio.mimsave(args.gif, frames[::2], fps=args.fps // 2, loop=0)
        print(f"wrote {args.gif} ({len(frames) // 2} frames)")
    viewer.close()


if __name__ == "__main__":
    main()
