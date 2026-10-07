"""Watch trained agents in isolated save-state levels with splits.

    python -m smwrl.watch --level YoshiIsland1
    python -m smwrl.watch --route
    python -m smwrl.watch --route --record run.mp4

This is a specialist-policy playlist, not a continuous game. Use
``smwrl.playthrough`` for a power-on session with persistent lives/map state.
Missing policies and failures stop by default; explicit flags opt into a partial
diagnostic playlist.
"""

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import numpy as np
import pygame

from smwrl.env import make_env
from smwrl.levels import LEVELS, ROUTE
from smwrl.policy import load_policy_runtime
from smwrl.wrappers import EpisodeConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"

SCALE = 3
PANEL_W = 260
NATIVE_W, NATIVE_H = 256, 224


def fmt_time(seconds: float) -> str:
    m, s = divmod(seconds, 60)
    return f"{int(m)}:{s:06.3f}"


class Viewer:
    """Scaled game view plus a split panel, drawn with pygame."""

    def __init__(self, record: str | None = None, fps: int = 60):
        pygame.init()
        pygame.display.set_caption("SMW - PPO speedrun")
        self.w = NATIVE_W * SCALE + PANEL_W
        self.h = NATIVE_H * SCALE
        self.screen = pygame.display.set_mode((self.w, self.h))
        self.font = pygame.font.SysFont("Menlo", 16)
        self.big = pygame.font.SysFont("Menlo", 28, bold=True)
        self.small = pygame.font.SysFont("Menlo", 13)
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.proc = None
        if record:
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{self.w}x{self.h}", "-r", str(fps), "-i", "-",
                 "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", record],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

    def pump(self) -> bool:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                return False
        return True

    def draw(self, frame: np.ndarray, hud: dict, splits: list[tuple[str, float | None]]) -> None:
        self.screen.fill((16, 16, 20))
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(pygame.transform.scale(surf, (NATIVE_W * SCALE, NATIVE_H * SCALE)), (0, 0))

        x0 = NATIVE_W * SCALE + 14
        self.screen.blit(self.big.render(hud["total"], True, (255, 255, 255)), (x0, 16))
        self.screen.blit(self.font.render(hud["level"], True, (255, 210, 90)), (x0, 58))
        self.screen.blit(self.small.render(hud["detail"], True, (150, 150, 160)), (x0, 80))

        y = 116
        self.screen.blit(self.small.render("SPLITS", True, (120, 120, 130)), (x0, y))
        y += 20
        for name, t in splits[-16:]:
            colour = (120, 230, 140) if t is not None else (110, 110, 120)
            label = name if len(name) <= 20 else name[:19] + "…"
            self.screen.blit(self.small.render(label, True, colour), (x0, y))
            txt = fmt_time(t) if t is not None else "--:--.---"
            self.screen.blit(self.small.render(txt, True, colour), (x0 + 150, y))
            y += 18

        pygame.display.flip()
        if self.proc:
            raw = pygame.image.tostring(self.screen, "RGB")
            try:
                self.proc.stdin.write(raw)
            except (BrokenPipeError, ValueError):
                self.proc = None
        self.clock.tick(self.fps)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        pygame.quit()


def load_policy(level: str):
    best = CKPT / level / "best.zip"
    path = best if best.exists() else CKPT / level / "latest.zip"
    if not path.exists():
        return None
    return load_policy_runtime(path, expected_level=level)


def play_level(level: str, viewer: Viewer, model, splits, elapsed_before: float,
               max_steps: int, deterministic: bool) -> tuple[bool, float]:
    """Play one level. Returns (cleared, seconds spent)."""
    runtime = model
    env = make_env(level, RewardConfig(), EpisodeConfig(max_steps=max_steps),
                   monitor=False,
                   obs_cfg=runtime.obs_config if runtime is not None else None)
    obs, _ = env.reset()
    stack = (runtime.initial_stack(obs) if runtime is not None
             else np.repeat(obs[None], 4, axis=0))

    t0 = time.time()
    cleared = False
    steps = 0
    while True:
        if not viewer.pump():
            env.close()
            raise KeyboardInterrupt

        if model is None:
            action = 2  # no policy yet: just hold run-right so there is something to see
        else:
            batch = runtime.batch(stack)
            action, _ = runtime.model.predict(batch, deterministic=deterministic)
            action = int(np.asarray(action).flat[0])

        obs, _, term, trunc, info = env.step(action)
        stack = (runtime.advance(stack, obs) if runtime is not None
                 else np.concatenate([stack[1:], obs[None]], axis=0))
        steps += 1

        frame = env.unwrapped.em.get_screen()
        now = time.time() - t0
        st = env.last_state
        viewer.draw(
            frame,
            {
                "total": fmt_time(elapsed_before + now),
                "level": LEVELS[level].label,
                "detail": f"x {st.x if st else 0:5d}   timer {st.timer if st else 0:3d}   "
                          f"step {steps}",
            },
            splits,
        )

        if term or trunc:
            cleared = bool(info.get("level_cleared"))
            break

    env.close()
    return cleared, time.time() - t0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=sorted(LEVELS))
    p.add_argument("--route", action="store_true")
    p.add_argument("--record", metavar="OUT.mp4")
    p.add_argument("--fps", type=int, default=15,
                   help="policy decisions per second; 15 x frameskip 4 is realtime")
    p.add_argument("--max-steps", type=int, default=1200)
    # Sampling is the default because these policies are not converged: taking
    # the argmax of the YoshiIsland1 policy stalls at x=1890 every single run,
    # while sampling from the same weights reaches the goal at 4826.
    p.add_argument("--deterministic", action="store_true",
                   help="take the argmax action instead of sampling; only "
                        "worth it once a policy has sharpened")
    p.add_argument("--allow-partial", action="store_true",
                   help="diagnostic only: skip levels without a policy")
    p.add_argument("--continue-on-failure", action="store_true",
                   help="diagnostic only: load the next independent state after a failure")
    args = p.parse_args()

    if not args.route and not args.level:
        args.level = "YoshiIsland1"

    levels = ROUTE if args.route else [args.level]
    trained = [lv for lv in levels if any(
        (CKPT / lv / name).exists() for name in ("best.zip", "latest.zip"))]
    if args.route:
        missing = [lv for lv in levels if lv not in trained]
        if missing:
            message = (f"no policy for {len(missing)} listed level(s): "
                       f"{', '.join(missing)}")
            if not args.allow_partial:
                raise SystemExit(message + "\nRefusing to call this a route. "
                                 "Pass --allow-partial for a diagnostic playlist.")
            print(message + "; explicitly skipping for partial diagnostics")
            levels = trained
    if not levels:
        print("Nothing to watch yet -- train a level first:\n"
              "    python -m smwrl.train --level YoshiIsland1 --steps 5_000_000")
        return

    viewer = Viewer(record=args.record, fps=args.fps)
    splits: list[tuple[str, float | None]] = []
    total = 0.0
    try:
        for lv in levels:
            model = load_policy(lv)
            cleared, secs = play_level(lv, viewer, model, splits, total,
                                       args.max_steps, args.deterministic)
            total += secs
            splits.append((LEVELS[lv].label, total if cleared else None))
            print(f"{LEVELS[lv].label:<24} {'CLEARED' if cleared else 'failed ':>8}  "
                  f"{secs:6.2f}s   total {fmt_time(total)}")
            if not cleared and not args.continue_on_failure:
                print("stopping on the first failed specialist (use "
                      "--continue-on-failure for diagnostics)")
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        viewer.close()
        done = sum(1 for _, t in splits if t is not None)
        print(f"\n{done}/{len(splits)} isolated levels cleared   playlist time {fmt_time(total)}")


if __name__ == "__main__":
    main()
