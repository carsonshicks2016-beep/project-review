"""Play one continuous Super Mario World session from power-on toward credits.

Unlike ``smwrl.watch --route``, this never resets to a per-level save state.
Menus and the overworld are scripted, while each gameplay segment is dispatched
by the live translevel ID to its specialist policy.  Lives, powerups, items,
event flags, and map progress therefore carry forward exactly as they do for a
human player.

The runner is deliberately fail-closed: an unknown translevel, missing policy,
incompatible checkpoint, exhausted lives, or transition timeout stops the run.
The stable ``THE END`` mode (0x29) is the only whole-game success signal.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from smwrl.actions import ACTION_TABLE
from smwrl.levels import LEVELS, LEVEL_BY_TRANSLEVEL
from smwrl.overworld import (
    MODE_LEVEL,
    MODE_OVERWORLD,
    NOOP,
    TRANSLEVEL,
    boot_to_map,
    enter_level,
    mode,
    move,
    position,
    read,
    wait_settled,
)
from smwrl.policy import load_policy_runtime
from smwrl.ram import decode
from smwrl.retro_env import INTEGRATION_DIR, GAME, make_raw_env
from smwrl.wrappers import preprocess_frame

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"
JOURNAL = CKPT / "playthrough_latest.json"

MODE_GAME_OVER = 0x17
MODE_ENDING_FIRST = 0x18
# SMWDisX's RunGameMode table maps 0x18..0x29 across the final cutscene,
# credits/enemy list, fades, and stable THE END screen.  0x1B is only the first
# cutscene and must not be reported as whole-game completion.
MODE_ENDING_FINAL = 0x29
SNES_FPS = 60.0988
FRAMESKIP = 4


def is_ending_mode(value: int) -> bool:
    return MODE_ENDING_FIRST <= value <= MODE_ENDING_FINAL


@dataclass(frozen=True)
class RouteTransition:
    """One verified post-exit map transition in the intended game route."""

    destination: tuple[int, int]
    directions: tuple[str, ...]
    next_translevel: int


# Evidence-backed and deliberately incomplete.  This normal exit was driven
# through its complete live event.  Missing entries stop instead of guessing a
# branch or re-entering a completed Switch Palace tile.
VERIFIED_ROUTE_TRANSITIONS: dict[tuple[int, str], RouteTransition] = {
    (0x2A, "normal"): RouteTransition((152, 104), (), 0x27),
}

MANDATORY_CONTENT_GAPS = (
    "verified normal/secret exit route graph",
    "castles and Koopalings",
    "fortresses and Switch Palaces",
    "remaining required specialist controllers",
    "Bowser controller and live THE END certification",
)


@dataclass
class SegmentResult:
    level: str
    translevel: int
    outcome: str
    decisions: int
    frames: int
    max_x: int
    lives: int
    exit_kind: str = ""
    detail: str = ""


def checkpoint_for(level: str) -> Path | None:
    directory = CKPT / level
    for name in ("best.zip", "latest.zip"):
        path = directory / name
        if path.exists():
            return path
    return None


class Viewer:
    """Render every raw emulator frame at SNES speed."""

    def __init__(self, enabled: bool, scale: int = 3):
        self.enabled = enabled
        self.scale = scale
        self.pygame = None
        if enabled:
            import pygame

            self.pygame = pygame
            pygame.init()
            pygame.display.set_caption("SMW continuous AI playthrough")
            self.screen = pygame.display.set_mode((256 * scale, 224 * scale))
            self.clock = pygame.time.Clock()

    def draw(self, frame: np.ndarray) -> None:
        if not self.enabled:
            return
        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
            ):
                raise KeyboardInterrupt
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        self.screen.blit(
            pygame.transform.scale(surf, (256 * self.scale, 224 * self.scale)), (0, 0)
        )
        pygame.display.flip()
        self.clock.tick(SNES_FPS)

    def close(self) -> None:
        if self.enabled:
            self.pygame.quit()


class TrackedEnv:
    """Transparent raw-env proxy that accounts for and renders every frame."""

    def __init__(self, raw, viewer: Viewer):
        self.raw = raw
        self.viewer = viewer
        self.frames = 0

    @property
    def unwrapped(self):
        return self.raw.unwrapped

    def reset(self, *args, **kwargs):
        result = self.raw.reset(*args, **kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        self.viewer.draw(obs)
        return result

    def step(self, action):
        result = self.raw.step(action)
        self.frames += 1
        self.viewer.draw(result[0])
        return result

    def __getattr__(self, name):
        return getattr(self.raw, name)


class ContinuousRunner:
    def __init__(self, render: bool, deterministic: bool, max_decisions: int,
                 capture_missing: bool = False):
        self.viewer = Viewer(render)
        self.env = TrackedEnv(make_raw_env(None, render_mode=None), self.viewer)
        self.deterministic = deterministic
        self.max_decisions = max_decisions
        self.capture_missing = capture_missing
        self.results: list[SegmentResult] = []
        self.started = time.time()

    def close(self) -> None:
        self.env.raw.close()
        self.viewer.close()

    def _write_journal(self, status: str, detail: str = "") -> None:
        payload = {
            "status": status,
            "detail": detail,
            "wall_seconds": time.time() - self.started,
            "emulated_frames": self.env.frames,
            "emulated_seconds": self.env.frames / SNES_FPS,
            "segments": [asdict(x) for x in self.results],
            "updated": time.time(),
        }
        tmp = JOURNAL.with_suffix(".tmp.json")
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        os.replace(tmp, JOURNAL)

    def _capture_unknown(self, translevel: int) -> Path | None:
        if not self.capture_missing:
            return None
        out = INTEGRATION_DIR / GAME / f"Discovered{translevel:02X}.state"
        raw_state = self.env.unwrapped.em.get_state()
        tmp = out.with_suffix(".state.tmp")
        tmp.write_bytes(gzip.compress(raw_state, compresslevel=6))
        os.replace(tmp, out)
        return out

    def boot(self, first_direction: str) -> None:
        result = boot_to_map(self.env)
        if not result.ok:
            raise RuntimeError(f"new-game boot failed: {result.detail}")
        before = position(self.env)
        after = move(self.env, first_direction)
        if after == before:
            raise RuntimeError(f"no overworld path {first_direction} from {before}")
        if not enter_level(self.env):
            raise RuntimeError(f"could not enter the first level at {after}")

    def _play_segment(self, name: str, runtime) -> SegmentResult:
        frame = self.env.unwrapped.em.get_screen()
        obs = preprocess_frame(frame, runtime.obs_config)
        stack = runtime.initial_stack(obs)
        initial_frames = self.env.frames
        max_x = 0
        last_lives = None
        latest_lives = 0

        for decision in range(1, self.max_decisions + 1):
            if mode(self.env) == MODE_LEVEL:
                action, _ = runtime.model.predict(
                    runtime.batch(stack), deterministic=self.deterministic
                )
                buttons = ACTION_TABLE[int(np.asarray(action).flat[0])]
            else:
                # Fade/door/course-clear sequences are not gameplay decisions.
                buttons = NOOP

            raw_frames = []
            info = {}
            for _ in range(FRAMESKIP):
                frame, _, _, _, info = self.env.step(buttons)
                raw_frames.append(frame)
                state = decode(info)
                max_x = max(max_x, state.x)
                latest_lives = state.lives
                current_mode = mode(self.env)
                if current_mode == MODE_ENDING_FINAL:
                    return SegmentResult(name, read(self.env, TRANSLEVEL), "credits",
                                         decision, self.env.frames - initial_frames,
                                         max_x, latest_lives)
                if is_ending_mode(current_mode):
                    self._finish_ending()
                    return SegmentResult(name, read(self.env, TRANSLEVEL), "credits",
                                         decision, self.env.frames - initial_frames,
                                         max_x, latest_lives)
                if current_mode == MODE_OVERWORLD:
                    exit_kind = ("secret" if state.overworld_exit_mode == 1
                                 else "normal" if state.overworld_exit_mode == 0
                                 else f"mode_0x{state.overworld_exit_mode:02X}")
                    return SegmentResult(name, read(self.env, TRANSLEVEL), "cleared",
                                         decision, self.env.frames - initial_frames,
                                         max_x, latest_lives, exit_kind,
                                         "returned to overworld")
                if state.cleared:
                    exit_kind = "secret" if state.secret_exit else "normal"
                    return SegmentResult(name, state.translevel, "cleared", decision,
                                         self.env.frames - initial_frames, max_x,
                                         latest_lives, exit_kind, "goal/exit trigger")
                if state.dying or (last_lives is not None and state.lives < last_lives):
                    reported_lives = state.lives - 1 if state.dying else state.lives
                    return SegmentResult(name, state.translevel, "death", decision,
                                         self.env.frames - initial_frames, max_x,
                                         reported_lives,
                                         detail="death animation; life decrement accounted")
                last_lives = state.lives

            pooled = np.maximum(raw_frames[-2], raw_frames[-1])
            obs = preprocess_frame(pooled, runtime.obs_config)
            stack = runtime.advance(stack, obs)

        return SegmentResult(name, read(self.env, TRANSLEVEL), "timeout",
                             self.max_decisions, self.env.frames - initial_frames,
                             max_x, latest_lives,
                             detail=f"policy exceeded {self.max_decisions} decisions")

    def _wait_for_retry(self, translevel: int, timeout: int = 6000) -> bool:
        """Wait through a death sequence until the same level is playable again."""
        left_gameplay = False
        for _ in range(timeout):
            _, _, _, _, info = self.env.step(NOOP)
            current_mode = mode(self.env)
            state = decode(info)
            if is_ending_mode(current_mode):
                return False
            if current_mode == MODE_GAME_OVER:
                return False
            if current_mode != MODE_LEVEL:
                left_gameplay = True
            if current_mode == MODE_OVERWORLD:
                # SMW returns to the map after an ordinary death; a human
                # presses A on the same node to retry.  Preserve the live map
                # state and do exactly that instead of waiting for an automatic
                # reload that never comes.
                wait_settled(self.env, timeout=timeout)
                if not enter_level(self.env, timeout=1200):
                    return False
                return read(self.env, TRANSLEVEL) == translevel
            if (left_gameplay and current_mode == MODE_LEVEL and not state.dying
                    and state.translevel == translevel):
                for _ in range(30):
                    self.env.step(NOOP)
                return True
        return False

    def _finish_ending(self, timeout: int = 120_000) -> None:
        """Render the complete noninteractive ending pipeline through THE END."""
        for _ in range(timeout):
            current = mode(self.env)
            if current == MODE_ENDING_FINAL:
                return
            if current == MODE_GAME_OVER:
                raise RuntimeError("game over during the ending pipeline")
            if not is_ending_mode(current):
                raise RuntimeError(
                    f"ending pipeline unexpectedly left modes 0x18..0x29 at 0x{current:02X}"
                )
            self.env.step(NOOP)
        raise TimeoutError("ending did not reach the stable THE END mode 0x29")

    def _advance_after_clear(self, source_translevel: int, exit_kind: str) -> str:
        # Bowser enters a long ending pipeline and never returns to the map.
        # Inspect each frame so none of its transient modes can be missed.
        for _ in range(12_000):
            current = mode(self.env)
            if is_ending_mode(current):
                self._finish_ending()
                return "credits"
            if current == MODE_GAME_OVER:
                raise RuntimeError("game over while leaving a cleared stage")
            if current == MODE_OVERWORLD:
                break
            self.env.step(NOOP)
        else:
            raise TimeoutError("level exit reached neither overworld nor ending")

        wait_settled(self.env, timeout=12_000)
        key = (source_translevel, exit_kind)
        transition = VERIFIED_ROUTE_TRANSITIONS.get(key)
        if transition is None:
            raise RuntimeError(
                f"no verified route transition for translevel 0x{source_translevel:02X} "
                f"{exit_kind} exit; refusing to guess an overworld branch"
            )
        for direction in transition.directions:
            move(self.env, direction)
        actual = position(self.env)
        if actual != transition.destination:
            raise RuntimeError(
                f"route transition {key} settled at {actual}, expected "
                f"{transition.destination}"
            )
        if not enter_level(self.env, timeout=1200):
            raise RuntimeError(f"could not enter the next node at {actual}")
        entered = read(self.env, TRANSLEVEL)
        if entered != transition.next_translevel:
            raise RuntimeError(
                f"entered translevel 0x{entered:02X}, expected "
                f"0x{transition.next_translevel:02X}"
            )
        return "level"

    def run(self, first_direction: str = "RIGHT", max_segments: int = 100) -> bool:
        self._write_journal("booting")
        self.boot(first_direction)

        for _ in range(max_segments):
            if mode(self.env) == MODE_ENDING_FINAL:
                self._write_journal("complete", "credits reached")
                return True

            translevel = read(self.env, TRANSLEVEL)
            name = LEVEL_BY_TRANSLEVEL.get(translevel)
            if name is None:
                captured = self._capture_unknown(translevel)
                detail = (f"unknown translevel 0x{translevel:02X} at {position(self.env)}"
                          + (f"; captured {captured}" if captured else ""))
                self._write_journal("blocked", detail)
                print("BLOCKED:", detail)
                return False

            expected = LEVELS[name]
            actual_pos = position(self.env)
            if actual_pos != expected.map_pos:
                detail = (f"{name} translevel matched but map position {actual_pos} "
                          f"!= manifest {expected.map_pos}")
                self._write_journal("blocked", detail)
                print("BLOCKED:", detail)
                return False

            checkpoint = checkpoint_for(name)
            if checkpoint is None:
                detail = f"no promoted/latest policy for required live level {name}"
                self._write_journal("blocked", detail)
                print("BLOCKED:", detail)
                return False
            runtime = load_policy_runtime(checkpoint, expected_level=name)
            print(f"{name}: policy {checkpoint.name}, lives carry forward, "
                  f"map={actual_pos}, translevel=0x{translevel:02X}")

            while True:
                result = self._play_segment(name, runtime)
                self.results.append(result)
                self._write_journal("running", f"{name}: {result.outcome}")
                print(f"  {result.outcome:<8} x={result.max_x:<5} "
                      f"decisions={result.decisions:<4} lives={result.lives}")
                if result.outcome == "credits":
                    self._write_journal("complete", "credits reached")
                    return True
                if result.outcome == "cleared":
                    break
                if result.outcome != "death" or not self._wait_for_retry(translevel):
                    detail = f"{name} ended with {result.outcome}; no playable retry"
                    self._write_journal("failed", detail)
                    print("FAILED:", detail)
                    return False

            if self._advance_after_clear(translevel, result.exit_kind) == "credits":
                self._write_journal("complete", "credits reached")
                return True

        detail = f"stopped at safety cap of {max_segments} segments without credits"
        self._write_journal("blocked", detail)
        print("BLOCKED:", detail)
        return False


def readiness() -> bool:
    policies = {name: checkpoint_for(name) for name in LEVELS}
    available = {name: p for name, p in policies.items() if p is not None}
    print(f"continuous runner: available ({len(available)}/{len(LEVELS)} known levels)")
    invalid = []
    for name, path in available.items():
        try:
            runtime = load_policy_runtime(path, expected_level=name)
            print(f"  OK  {name:<20} {path.name:<10} obs="
                  f"{runtime.model.observation_space.shape}")
        except Exception as e:
            invalid.append(name)
            print(f"  BAD {name:<20} {type(e).__name__}: {e}")
    missing = [name for name, path in policies.items() if path is None]
    print(f"missing known policies: {len(missing)}")
    if missing:
        print("  " + ", ".join(missing))
    print("mandatory full-game gaps:")
    for gap in MANDATORY_CONTENT_GAPS:
        print(f"  - {gap}")
    manifest_complete = not MANDATORY_CONTENT_GAPS
    return manifest_complete and not missing and not invalid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report fail-closed full-game readiness without starting")
    ap.add_argument("--headless", action="store_true", help="run faster than realtime")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--first-direction", choices=("LEFT", "RIGHT"), default="RIGHT",
                    help="RIGHT starts the normal Yoshi's Island 2 route; LEFT visits YI1")
    ap.add_argument("--max-decisions", type=int, default=3000)
    ap.add_argument("--max-segments", type=int, default=100)
    ap.add_argument("--capture-missing", action="store_true",
                    help="save the first unknown live stage as a training state")
    args = ap.parse_args()

    if args.check:
        raise SystemExit(0 if readiness() else 1)

    runner = ContinuousRunner(not args.headless, args.deterministic,
                              args.max_decisions, args.capture_missing)
    try:
        ok = runner.run(args.first_direction, args.max_segments)
    except KeyboardInterrupt:
        runner._write_journal("stopped", "user stopped the run")
        ok = False
    except Exception as e:
        runner._write_journal("failed", f"{type(e).__name__}: {e}")
        raise
    finally:
        runner.close()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
