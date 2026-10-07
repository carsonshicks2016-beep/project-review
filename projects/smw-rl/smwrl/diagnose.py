"""Find where a policy gets stuck, and show you the screen at that point.

A flat mean_x tells you the agent stopped improving but not why. This runs the
current policy, records where progress stalls across episodes, and dumps frames
from the stall point so the obstacle is visible.

    python -m smwrl.diagnose --level YoshiIsland1 --episodes 8
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from smwrl.env import make_env
from smwrl.levels import LEVELS
from smwrl.policy import load_policy_runtime
from smwrl.wrappers import EpisodeConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="YoshiIsland1", choices=sorted(LEVELS))
    ap.add_argument("--episodes", type=int, default=8)
    ap.add_argument("--outdir", default="/tmp/smw_diag")
    ap.add_argument("--deterministic", action="store_true")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    path = CKPT / args.level / "latest.zip"
    runtime = load_policy_runtime(path, expected_level=args.level)
    model = runtime.model
    print(f"policy: {path}")

    env = make_env(args.level, RewardConfig(), EpisodeConfig(), monitor=False,
                   obs_cfg=runtime.obs_config)
    outcomes: Counter[str] = Counter()
    stall_x: list[int] = []
    saved = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        stack = runtime.initial_stack(obs)
        best, since, step = 0, 0, 0
        shot_taken = False

        while True:
            batch = runtime.batch(stack)
            a, _ = model.predict(batch, deterministic=args.deterministic)
            obs, _, term, trunc, info = env.step(int(np.asarray(a).flat[0]))
            stack = runtime.advance(stack, obs)
            step += 1

            x = int(info.get("max_x", 0))
            if x > best:
                best, since = x, 0
            else:
                since += 1

            # 90 steps of no progress: we are at the wall, grab the screen.
            if since == 90 and not shot_taken and saved < 6:
                frame = env.unwrapped.em.get_screen()
                f = out / f"stuck_ep{ep}_x{best}.png"
                cv2.imwrite(str(f), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                print(f"  ep {ep}: stalled at x={best} -> {f.name}")
                shot_taken = True
                saved += 1

            if term or trunc:
                why = ("cleared" if info.get("level_cleared")
                       else "death" if info.get("death")
                       else "stuck" if info.get("stuck") else "timeout")
                outcomes[why] += 1
                # Deaths never trigger the stall shot, and for levels that die
                # rather than idle the terminal frame is the whole story.
                if not shot_taken and saved < 8:
                    frame = env.unwrapped.em.get_screen()
                    f = out / f"{why}_ep{ep}_x{best}.png"
                    cv2.imwrite(str(f), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    print(f"  ep {ep}: {why} at x={best} -> {f.name}")
                    saved += 1
                stall_x.append(best)
                break

    env.close()
    print(f"\noutcomes: {dict(outcomes)}")
    print(f"stall x: mean {np.mean(stall_x):.0f}  min {min(stall_x)}  max {max(stall_x)}"
          f"  spread {np.std(stall_x):.1f}")
    print(f"level is ~{LEVELS[args.level].length} px long -- "
          f"reached {np.mean(stall_x) / LEVELS[args.level].length:.0%}")


if __name__ == "__main__":
    main()
