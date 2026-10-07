"""Watch several levels train at once, in a grid.

Each cell runs its own emulator and hot-reloads that level's published policy,
so you see all the agents improving side by side.

    python -m smwrl.live_grid                                  # the 4 Yoshi's Island levels
    python -m smwrl.live_grid --levels YoshiIsland1,DonutPlains1
    python -m smwrl.live_grid --scale 2 --fps 60

stable-retro allows only one emulator instance per process, so each level runs
in its own worker process that steps the game, runs its policy on CPU, and ships
frames back for compositing. Saliency is off here; use `smwrl.live --brain` when
you want to look inside one policy in detail.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import queue
import time
from collections import Counter

import numpy as np
import pygame

from smwrl.levels import LEVELS
from smwrl.live import ACCENT, BG, DIM, FG, GOOD, PolicyWatcher

NATIVE_W, NATIVE_H = 256, 224
HUD_H = 26
PANEL_W = 300


def _worker(level: str, max_steps: int, deterministic: bool, fps: int, q, stop) -> None:
    """Run one level and stream frames + stats to the parent."""
    import warnings

    warnings.filterwarnings("ignore")
    from smwrl.env import make_env
    from smwrl.wrappers import EpisodeConfig, RewardConfig

    watcher = PolicyWatcher(level)
    watcher.poll()
    env = make_env(level, RewardConfig(), EpisodeConfig(max_steps=max_steps), monitor=False,
                   obs_cfg=(watcher.runtime.obs_config if watcher.runtime else None))
    obs, _ = env.reset()
    stack = (watcher.runtime.initial_stack(obs) if watcher.runtime
             else np.repeat(obs[None], 4, axis=0))

    outcomes: Counter[str] = Counter()
    episodes = best_x = step_i = 0
    period = 1.0 / max(1, fps)
    next_t = time.monotonic()

    try:
        while not stop.is_set():
            watcher.poll()
            model = watcher.model
            if model is None:
                action = 4 if (step_i % 10) < 3 else 2
            else:
                batch = watcher.runtime.batch(stack)
                a, _ = model.predict(batch, deterministic=deterministic)
                action = int(np.asarray(a).flat[0])

            obs, _, term, trunc, info = env.step(action)
            stack = (watcher.runtime.advance(stack, obs) if watcher.runtime
                     else np.concatenate([stack[1:], obs[None]], axis=0))
            step_i += 1
            best_x = max(best_x, int(info.get("max_x", 0)))
            st = env.last_state

            payload = {
                "frame": env.unwrapped.em.get_screen(),
                "x": st.x if st else 0,
                "best_x": best_x,
                "episodes": episodes,
                "clears": outcomes.get("CLEARED", 0),
                "outcomes": dict(outcomes),
                "status": watcher.status,
                "generation": watcher.generation,
            }
            # Keep only the newest frame; the parent may render slower than we step.
            try:
                q.put_nowait(payload)
            except queue.Full:
                try:
                    q.get_nowait()
                    q.put_nowait(payload)
                except (queue.Empty, queue.Full):
                    pass

            if term or trunc:
                outcomes["CLEARED" if info.get("level_cleared")
                         else "death" if info.get("death")
                         else "stuck" if info.get("stuck") else "timeout"] += 1
                episodes += 1
                obs, _ = env.reset()
                stack = (watcher.runtime.initial_stack(obs) if watcher.runtime
                         else np.repeat(obs[None], 4, axis=0))

            # Pace to roughly realtime so we do not burn a core racing ahead.
            next_t += period
            delay = next_t - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.monotonic()
    finally:
        env.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels",
                    default="YoshiIsland1,YoshiIsland2,YoshiIsland3,YoshiIsland4")
    ap.add_argument("--scale", type=float, default=1.5)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (smoke test)")
    ap.add_argument("--shot", metavar="OUT.png", help="save a screenshot before exiting")
    args = ap.parse_args()

    levels = [x for x in (s.strip() for s in args.levels.split(",")) if x]
    for lv in levels:
        if lv not in LEVELS:
            raise SystemExit(f"unknown level: {lv}")

    cols = 2 if len(levels) > 1 else 1
    rows = math.ceil(len(levels) / cols)
    gw, gh = int(NATIVE_W * args.scale), int(NATIVE_H * args.scale)
    cell_h = gh + HUD_H
    width, height = cols * gw + PANEL_W, rows * cell_h

    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    procs, queues = [], []
    print(f"starting {len(levels)} emulator processes…")
    for lv in levels:
        q = ctx.Queue(maxsize=1)
        p = ctx.Process(target=_worker,
                        args=(lv, args.max_steps, args.deterministic, args.fps, q, stop),
                        daemon=True)
        p.start()
        procs.append(p)
        queues.append(q)

    pygame.init()
    pygame.display.set_caption("SMW live - " + " | ".join(levels))
    screen = pygame.display.set_mode((width, height))
    font = pygame.font.SysFont("Menlo", 13)
    bold = pygame.font.SysFont("Menlo", 15, bold=True)
    small = pygame.font.SysFont("Menlo", 12)
    clock = pygame.time.Clock()

    latest: list[dict | None] = [None] * len(levels)
    running, frames = True, 0
    print("running. Esc or close the window to stop.")

    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (
                    e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
                ):
                    running = False

            for i, q in enumerate(queues):
                try:
                    latest[i] = q.get_nowait()
                except queue.Empty:
                    pass

            screen.fill(BG)
            for i, lv in enumerate(levels):
                cx, cy = (i % cols) * gw, (i // cols) * cell_h
                d = latest[i]
                if d is not None:
                    surf = pygame.surfarray.make_surface(d["frame"].swapaxes(0, 1))
                    screen.blit(pygame.transform.scale(surf, (gw, gh)), (cx, cy))
                else:
                    screen.blit(small.render("booting emulator…", True, DIM),
                                (cx + 12, cy + gh // 2))

                pygame.draw.rect(screen, (24, 24, 30), (cx, cy + gh, gw, HUD_H))
                clears = d["clears"] if d else 0
                hud = (f"{LEVELS[lv].label[:22]}  x{(d['x'] if d else 0):5d}"
                       f"  best {(d['best_x'] if d else 0):5d}"
                       f"  ep {(d['episodes'] if d else 0):4d}  clr {clears:3d}")
                screen.blit(font.render(hud, True, GOOD if clears else FG),
                            (cx + 6, cy + gh + 5))
                pygame.draw.rect(screen, (44, 44, 54), (cx, cy, gw, cell_h), 1)

            # -- shared training status column ----------------------------
            x0 = cols * gw + 12
            pygame.draw.rect(screen, (22, 22, 28), (cols * gw, 0, PANEL_W, height))
            y = 12
            screen.blit(bold.render("TRAINING", True, ACCENT), (x0, y))
            y += 24
            for i, lv in enumerate(levels):
                d = latest[i]
                s = (d or {}).get("status") or {}
                screen.blit(font.render(LEVELS[lv].label[:24], True, FG), (x0, y))
                y += 17
                if s:
                    screen.blit(small.render(
                        f"  {s.get('timesteps', 0):>10,} steps", True, DIM), (x0, y))
                    y += 15
                    solo = s.get("solo_clear_rate")
                    solo_txt = f"{solo:.0%}" if solo is not None else "n/a"
                    screen.blit(small.render(
                        f"  solo clear {solo_txt:>5}  x {s.get('solo_mean_x', 0):>6.0f}",
                        True, GOOD if (solo or 0) > 0 else DIM), (x0, y))
                    y += 15
                    screen.blit(small.render(
                        f"  gen {(d or {}).get('generation', 0)}"
                        f"   {s.get('steps_per_sec', 0):.0f}/s", True, DIM), (x0, y))
                    y += 20
                else:
                    screen.blit(small.render("  waiting for policy…", True, DIM), (x0, y))
                    y += 35

            pygame.display.flip()
            clock.tick(args.fps)
            frames += 1
            if args.max_frames and frames >= args.max_frames:
                running = False
    except KeyboardInterrupt:
        pass
    finally:
        if args.shot:
            pygame.image.save(screen, args.shot)
            print("wrote", args.shot)
        stop.set()
        for p in procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
        pygame.quit()
        print()
        for i, lv in enumerate(levels):
            d = latest[i] or {}
            print(f"{LEVELS[lv].label:<24} {d.get('episodes', 0):>4} eps  "
                  f"best_x {d.get('best_x', 0):>5}  {d.get('outcomes', {})}")


if __name__ == "__main__":
    main()
