"""Watch the whole-game policy play, with the same engineer panel as `smwrl.live`.

`smwrl.live` is per-level by construction: it takes a level name, loads that
level's checkpoint and resets to that level's save state. The whole-game policy
has none of those -- one policy plays every level in one continuous session, and
its checkpoint belongs to no level at all -- so it needs its own entry point.
The panel, the policy loader and the saliency pass are shared; only the
environment, the status readout and the HUD differ.

    python -m smwrl.world_live --brain

It hot-reloads checkpoints/world/latest.zip as `smwrl.world_train` publishes it,
so you watch the policy improve rather than a frozen snapshot.

Episodes start from the first playable state by default (`--curriculum-ratio 0`)
because the unaided run is the number that matters -- training's own average
mixes in episodes seeded next to a goal. Raise the ratio to watch it play from
somewhere deeper in the archive instead.

This is not the same thing as `smwrl.world_watch`, which replays the archive's
frontier using random actions and involves no policy at all. That one shows you
where the *search* has reached; this one shows you what the *agent* has learned.

Like the per-level viewer it runs its own emulator and keeps the policy on CPU,
so it costs a core but never touches the GPU.
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import pygame

from smwrl.actions import N_ACTIONS
from smwrl.live import (BG, FG, GAME_H, GAME_W, PANEL_W, Panel, PolicyWatcher,
                        policy_introspect)
from smwrl.world_env import WorldEnv, WorldEpisodeConfig, curriculum_from_archive
from smwrl.wrappers import ObsConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARCHIVE = ROOT / "checkpoints" / "world_archive.pkl"


def status_lines(status: dict, generation: int) -> list[str]:
    """Whole-game training reports levels chained, not distance into one level."""
    if not status:
        return []
    age = time.time() - status.get("updated", time.time())
    return [
        f"training  {status.get('timesteps', 0):>12,} steps",
        f"levels/episode {status.get('mean_levels', 0):5.2f}"
        f"   SOLO {status.get('solo_mean_levels', 0):5.2f}"
        f"   best {status.get('best_levels', 0)}",
        f"policy gen {generation}   published {age:>4.0f}s ago",
    ]


def why_ended(info: dict) -> str:
    if info.get("transition_failed"):
        return "map transition failed"
    if info.get("death"):
        return "death"
    if info.get("stuck"):
        return "stuck"
    return "timeout"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--brain", action="store_true", help="show the engineer panel")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--brain-every", type=int, default=3,
                    help="compute saliency every N frames (it needs a backward pass)")
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--curriculum-size", type=int, default=64)
    ap.add_argument("--curriculum-ratio", type=float, default=0.0,
                    help="fraction of episodes seeded from deeper in the archive; "
                         "0 means always start from the first playable state")
    ap.add_argument("--deterministic", action="store_true",
                    help="take the argmax action instead of sampling; sampling is "
                         "the default here because it is what training actually does")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = run until closed); for smoke tests")
    ap.add_argument("--shot", metavar="OUT.png", help="save a screenshot before exiting")
    args = ap.parse_args()

    curriculum = curriculum_from_archive(args.archive, args.curriculum_size)
    if not curriculum:
        raise SystemExit(f"no usable archive at {args.archive}; "
                         "run smwrl.worldrun first")

    # expected_level=None: the whole-game contract claims no level, and asking
    # it to claim "world" would reject every checkpoint the trainer publishes.
    watcher = PolicyWatcher("world", expected_level=None)
    watcher.poll()
    if watcher.model is None:
        print(f"No whole-game policy published yet at {watcher.path}.\n"
              "Start training first:\n"
              "    python -m smwrl.world_train --steps 20_000_000\n"
              "Showing a scripted run-right agent until one appears.")

    pygame.init()
    pygame.display.set_caption("SMW live - whole game")
    width = GAME_W + (PANEL_W if args.brain else 0)
    screen = pygame.display.set_mode((width, GAME_H))
    fonts = (
        pygame.font.SysFont("Menlo", 15),
        pygame.font.SysFont("Menlo", 18, bold=True),
        pygame.font.SysFont("Menlo", 13),
    )
    panel = Panel(screen, fonts)
    clock = pygame.time.Clock()

    env = WorldEnv(curriculum, RewardConfig(),
                   WorldEpisodeConfig(max_steps=args.max_steps),
                   watcher.runtime.obs_config if watcher.runtime else ObsConfig(),
                   curriculum_ratio=args.curriculum_ratio)
    obs, _ = env.reset()
    stack = (watcher.runtime.initial_stack(obs) if watcher.runtime
             else np.repeat(obs[None], 4, axis=0))
    value_hist: deque[float] = deque(maxlen=220)
    last_sal = None
    ret = 0.0
    frame_i = 0
    episodes = 0
    running = True

    try:
        while running:
            for e in pygame.event.get():
                if e.type == pygame.QUIT or (
                    e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE
                ):
                    running = False
            watcher.poll()

            probs = np.zeros(N_ACTIONS, np.float32)
            value, sal, entropy = 0.0, None, 0.0
            if watcher.model is None:
                action = 4 if (frame_i % 10) < 3 else 2
                probs[action] = 1.0
            else:
                want_sal = args.brain and (frame_i % args.brain_every == 0)
                action, probs, value, s_ = policy_introspect(watcher.runtime, stack, want_sal)
                if s_ is not None:
                    last_sal = s_
                sal = last_sal
                if not args.deterministic:
                    action = int(np.random.choice(N_ACTIONS, p=probs / probs.sum()))
                entropy = float(-(probs * np.log(probs + 1e-8)).sum())
                value_hist.append(value)

            obs, reward, term, trunc, info = env.step(action)
            stack = (watcher.runtime.advance(stack, obs) if watcher.runtime
                     else np.concatenate([stack[1:], obs[None]], axis=0))
            ret += reward
            frame_i += 1

            screen.fill(BG)
            # WorldEnv holds the emulator directly; there is no gym wrapper stack
            # to unwrap the way the per-level viewer does.
            frame = env.env.em.get_screen()
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (GAME_W, GAME_H)), (0, 0))

            st = env.last_state
            hud = (f"translevel {st.translevel if st else 0:3d}   "
                   f"x {st.x if st else 0:5d}   "
                   f"cleared {info.get('levels_cleared', 0)}   "
                   f"events {info.get('events', 0)}   ep {episodes}")
            strip = pygame.Surface((GAME_W, 30))
            strip.set_alpha(190)
            strip.fill((0, 0, 0))
            screen.blit(strip, (0, GAME_H - 30))
            screen.blit(fonts[0].render(hud, True, FG), (10, GAME_H - 24))

            if args.brain:
                panel.draw(GAME_W + 20, {
                    "status_lines": status_lines(watcher.status, watcher.generation),
                    "generation": watcher.generation,
                    "obs84": stack[-1],
                    "saliency": sal,
                    "value": value,
                    "value_hist": list(value_hist),
                    "probs": probs,
                    "action": action,
                    "reward": reward,
                    "ret": ret,
                    "entropy": entropy,
                })

            pygame.display.flip()
            clock.tick(args.fps)
            if args.max_frames and frame_i >= args.max_frames:
                running = False

            if term or trunc:
                episodes += 1
                print(f"ep {episodes:>4}  levels {info.get('levels_cleared', 0)}  "
                      f"progress {info.get('progress', 0):>6}  "
                      f"return {ret:>9.1f}  {why_ended(info)}", flush=True)
                obs, _ = env.reset()
                stack = (watcher.runtime.initial_stack(obs) if watcher.runtime
                         else np.repeat(obs[None], 4, axis=0))
                ret = 0.0
                value_hist.clear()
    except KeyboardInterrupt:
        pass
    finally:
        if args.shot:
            pygame.image.save(screen, args.shot)
            print("wrote", args.shot)
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
