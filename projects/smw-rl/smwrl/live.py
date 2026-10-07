"""Watch training happen live, with an engineer panel showing the policy's internals.

Run this in a second terminal while `smwrl.train` is going. It hot-reloads the
policy every time training publishes a new one, so you see the agent actually
getting better rather than a frozen snapshot.

    python -m smwrl.live --level YoshiIsland1 --brain

The panel shows, per frame:
  * what the agent actually sees (84x84 grayscale, the newest of 4 stacked frames)
  * a saliency map -- which pixels the chosen action was most sensitive to
  * the full action distribution, not just the sampled action
  * the critic's value estimate, tracked over the episode
  * the live reward breakdown and training status

The spectator runs its own emulator and keeps the policy on CPU, so it competes
with training for a core but never for the GPU. If training slows noticeably,
drop it to `--fps 30` or train with `--n-envs 11`.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import pygame
import torch

from smwrl.actions import ACTION_LABELS, N_ACTIONS
from smwrl.env import make_env
from smwrl.levels import LEVELS
from smwrl.policy import load_policy_runtime
from smwrl.wrappers import EpisodeConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"

SCALE = 3
GAME_W, GAME_H = 256 * SCALE, 224 * SCALE
PANEL_W = 520

BG = (14, 14, 18)
FG = (232, 232, 238)
DIM = (128, 128, 140)
ACCENT = (255, 205, 90)
GOOD = (120, 230, 150)
BAD = (240, 110, 110)


_SAME = object()


class PolicyWatcher:
    """Loads checkpoints/<level>/latest.zip and reloads it whenever it changes.

    `expected_level` is the level the checkpoint's contract must claim, and
    defaults to the directory it was loaded from. Pass None for a policy that
    belongs to no single level -- the whole-game one publishes its contract with
    a null level, and demanding it match "world" would reject every checkpoint.
    """

    def __init__(self, level: str, expected_level=_SAME):
        self.level = level
        self.expected_level = level if expected_level is _SAME else expected_level
        self.path = CKPT / level / "latest.zip"
        self.status_path = CKPT / level / "status.json"
        self.model = None
        self.runtime = None
        self.mtime = 0.0
        self.generation = 0
        self.status: dict = {}
        self._last_poll = 0.0

    def poll(self) -> bool:
        """Reload if a newer policy has been published. Returns True on reload."""
        now = time.time()
        if now - self._last_poll < 2.0:
            return False
        self._last_poll = now

        if self.status_path.exists():
            try:
                self.status = json.loads(self.status_path.read_text())
            except (json.JSONDecodeError, OSError):
                pass

        if not self.path.exists():
            return False
        m = self.path.stat().st_mtime
        if m <= self.mtime:
            return False
        try:
            # CPU on purpose: leave the GPU entirely to the trainer.
            self.runtime = load_policy_runtime(
                self.path, device="cpu", expected_level=self.expected_level
            )
            self.model = self.runtime.model
            self.mtime = m
            self.generation += 1
            return True
        except (EOFError, OSError, RuntimeError, ValueError):
            return False  # caught it mid-write; try again next poll


def policy_introspect(runtime, stack: np.ndarray, want_saliency: bool):
    """Run the policy and pull out the internals worth showing.

    Returns (action, probs, value, saliency84 or None).
    """
    model = runtime.model
    batch = runtime.batch(stack)
    obs_t, _ = model.policy.obs_to_tensor(batch)
    x = obs_t.float().clone().requires_grad_(want_saliency)

    if not want_saliency:
        with torch.no_grad():
            dist = model.policy.get_distribution(x)
            probs = dist.distribution.probs[0].cpu().numpy()
            value = float(model.policy.predict_values(x)[0, 0])
        return int(probs.argmax()), probs, value, None

    dist = model.policy.get_distribution(x)
    probs_t = dist.distribution.probs[0]
    action = int(torch.argmax(probs_t))
    value = float(model.policy.predict_values(x)[0, 0].detach())

    # "Which pixels was this decision most sensitive to?" -- gradient of the
    # chosen action's log-probability w.r.t. the input, max-pooled over the
    # 4 stacked frames.
    model.policy.zero_grad(set_to_none=True)
    torch.log(probs_t[action] + 1e-8).backward()
    grad = x.grad.detach().abs()[0]
    sal = grad.amax(dim=0).cpu().numpy()
    if sal.max() > 0:
        sal = sal / sal.max()
    return action, probs_t.detach().cpu().numpy(), value, sal


def heatmap(sal: np.ndarray, size: int) -> np.ndarray:
    img = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    coloured = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)


class Panel:
    def __init__(self, screen, fonts):
        self.s = screen
        self.f, self.fb, self.fs = fonts

    def text(self, txt, x, y, colour=FG, font=None):
        self.s.blit((font or self.f).render(txt, True, colour), (x, y))

    def draw(self, x0, info: dict) -> None:
        s = self.s
        pygame.draw.rect(s, (22, 22, 28), (x0 - 10, 0, PANEL_W, GAME_H))

        y = 12
        self.text("ENGINEER PANEL", x0, y, ACCENT, self.fb)
        y += 26
        # Pre-formatted by the caller: what counts as training progress differs
        # per policy. Per-level reports mean_x and clear rate; the whole-game
        # one reports levels chained per episode, and neither has the other's
        # fields to render.
        for line in info["status_lines"] or ["waiting for trainer to publish a policy…"]:
            self.text(line, x0, y, DIM, self.fs)
            y += 16
        y += 10

        # -- what the agent sees, and what it looked at --------------------
        thumb = 150
        self.text("AGENT VIEW (what it sees)", x0, y, DIM, self.fs)
        self.text("SALIENCY", x0 + thumb + 20, y, DIM, self.fs)
        y += 16
        ob = info["obs84"]
        # the observation may be grayscale (1ch) or colour (3ch)
        obs_img = np.repeat(ob, 3, axis=2) if ob.shape[2] == 1 else ob
        surf = pygame.surfarray.make_surface(
            cv2.resize(obs_img, (thumb, thumb), interpolation=cv2.INTER_NEAREST).swapaxes(0, 1)
        )
        s.blit(surf, (x0, y))
        if info["saliency"] is not None:
            hm = heatmap(info["saliency"], thumb)
            s.blit(pygame.surfarray.make_surface(hm.swapaxes(0, 1)), (x0 + thumb + 20, y))
        else:
            self.text("(off)", x0 + thumb + 20, y + thumb // 2, DIM, self.fs)
        y += thumb + 18

        # -- critic --------------------------------------------------------
        self.text(f"VALUE  {info['value']:+8.2f}", x0, y, FG, self.fb)
        y += 24
        hist = info["value_hist"]
        if len(hist) > 1:
            w, h = PANEL_W - 30, 46
            lo, hi = min(hist), max(hist)
            rng = max(1e-6, hi - lo)
            pts = [
                (x0 + i * w / (len(hist) - 1), y + h - (v - lo) / rng * h)
                for i, v in enumerate(hist)
            ]
            pygame.draw.rect(s, (30, 30, 38), (x0, y, w, h))
            pygame.draw.lines(s, GOOD, False, pts, 2)
        y += 62

        # -- policy distribution -------------------------------------------
        self.text("ACTION DISTRIBUTION", x0, y, DIM, self.fs)
        y += 16
        probs = info["probs"]
        chosen = info["action"]
        bar_w = PANEL_W - 130
        for i in range(N_ACTIONS):
            colour = ACCENT if i == chosen else (70, 110, 170)
            row_y = y + i * 19
            self.text(ACTION_LABELS[i], x0, row_y, FG if i == chosen else DIM, self.fs)
            pygame.draw.rect(s, (34, 34, 42), (x0 + 92, row_y + 2, bar_w, 12))
            pygame.draw.rect(s, colour, (x0 + 92, row_y + 2, int(bar_w * probs[i]), 12))
            self.text(f"{probs[i]:4.0%}", x0 + 92 + bar_w + 6, row_y, DIM, self.fs)
        y += N_ACTIONS * 19 + 10

        r = info["reward"]
        self.text(f"reward {r:+7.2f}   return {info['ret']:+9.2f}   "
                  f"entropy {info['entropy']:4.2f}", x0, y, DIM, self.fs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="YoshiIsland1", choices=sorted(LEVELS))
    ap.add_argument("--tag", default="",
                    help="watch a tagged run, e.g. --tag _s2 for a second seed. "
                         "The level state is unchanged; only the checkpoint dir differs.")
    ap.add_argument("--brain", action="store_true", help="show the engineer panel")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--brain-every", type=int, default=3,
                    help="compute saliency every N frames (it needs a backward pass)")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--deterministic", action="store_true",
                    help="take the argmax action instead of sampling; sampling is "
                         "the default here because it is what training actually does")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="stop after N frames (0 = run until closed); for smoke tests")
    ap.add_argument("--shot", metavar="OUT.png", help="save a screenshot before exiting")
    args = ap.parse_args()

    pygame.init()
    pygame.display.set_caption(f"SMW live - {LEVELS[args.level].label}{args.tag}")
    width = GAME_W + (PANEL_W if args.brain else 0)
    screen = pygame.display.set_mode((width, GAME_H))
    fonts = (
        pygame.font.SysFont("Menlo", 15),
        pygame.font.SysFont("Menlo", 18, bold=True),
        pygame.font.SysFont("Menlo", 13),
    )
    panel = Panel(screen, fonts)
    clock = pygame.time.Clock()

    watcher = PolicyWatcher(args.level + args.tag)
    watcher.poll()
    if watcher.model is None:
        print(f"No policy published yet at {watcher.path}.\n"
              "Start training first:\n"
              f"    python -m smwrl.train --level {args.level} --steps 5_000_000\n"
              "Showing a scripted run-right agent until one appears.")

    env = make_env(args.level, RewardConfig(),
                   EpisodeConfig(max_steps=args.max_steps), monitor=False,
                   obs_cfg=(watcher.runtime.obs_config if watcher.runtime else None))
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
            frame = env.unwrapped.em.get_screen()
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(pygame.transform.scale(surf, (GAME_W, GAME_H)), (0, 0))

            st = env.last_state
            hud = (f"x {st.x if st else 0:5d}   max_x {info.get('max_x', 0):5}   "
                   f"timer {st.timer if st else 0:3d}   ep {episodes}")
            strip = pygame.Surface((GAME_W, 30))
            strip.set_alpha(190)
            strip.fill((0, 0, 0))
            screen.blit(strip, (0, GAME_H - 30))
            screen.blit(fonts[0].render(hud, True, FG), (10, GAME_H - 24))

            if args.brain:
                st_ = watcher.status
                age = time.time() - st_.get("updated", time.time())
                panel.draw(GAME_W + 20, {
                    "status_lines": [
                        f"training  {st_.get('timesteps', 0):>12,} steps"
                        f"   {st_.get('steps_per_sec', 0):.0f}/s",
                        f"mean_x {st_.get('mean_x', 0):>6.0f}"
                        f"   best_x {st_.get('best_x', 0):>5}"
                        f"   clear {st_.get('clear_rate', 0):5.1%}",
                        f"policy gen {watcher.generation}"
                        f"   published {age:>4.0f}s ago",
                    ] if st_ else [],
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
                why = ("CLEARED" if info.get("level_cleared")
                       else "death" if info.get("death")
                       else "stuck" if info.get("stuck") else "timeout")
                print(f"ep {episodes:>4}  max_x {info.get('max_x', 0):>5}  "
                      f"return {ret:>9.1f}  {why}", flush=True)
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
