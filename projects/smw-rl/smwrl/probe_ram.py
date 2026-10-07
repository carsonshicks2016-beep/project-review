"""Empirically verify the SMW RAM address map.

Every reward term in this project reads these addresses, so they get checked
against a live emulator before anything is built on top of them. Runs a
scripted "hold right + run" on a level and reports how each variable behaves.

    python -m smwrl.probe_ram [--state YoshiIsland1] [--frames 900]
"""

from __future__ import annotations

import argparse

import numpy as np

from smwrl.retro_env import make_raw_env

# Button order for the SNES core in stable-retro.
SNES_BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]


def button_mask(*names: str) -> np.ndarray:
    a = np.zeros(len(SNES_BUTTONS), np.uint8)
    for n in names:
        a[SNES_BUTTONS.index(n)] = 1
    return a


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="YoshiIsland1")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--every", type=int, default=60)
    args = ap.parse_args()

    env = make_raw_env(state=args.state)
    _, info = env.reset()

    print(f"state={args.state}  initial info:")
    for k, v in sorted(info.items()):
        print(f"    {k:18s} {v}")
    print()

    # Hold right + Y (run). Tap B (jump) periodically so we clear small gaps
    # and can watch the airborne / animation variables move.
    run_right = button_mask("RIGHT", "Y")
    jump_right = button_mask("RIGHT", "Y", "B")

    # reset() hands back an empty info dict; the variables only appear after
    # the first step, so the history keys are discovered lazily.
    history: dict[str, list[int]] = {}
    header_done = False

    for f in range(args.frames):
        act = jump_right if (f % 48) < 14 else run_right
        _, _, term, trunc, info = env.step(act)
        for k, v in info.items():
            history.setdefault(k, []).append(int(v))
        if not header_done and info:
            print(f"{'frame':>6} " + " ".join(f"{k:>10}" for k in sorted(info)))
            header_done = True
        if f % args.every == 0:
            print(f"{f:>6} " + " ".join(f"{int(info[k]):>10}" for k in sorted(info)))
        if term or trunc:
            print(f"--- env reported done at frame {f} ---")
            break

    env.close()

    print("\n=== behaviour summary ===")
    print(f"{'variable':<18} {'min':>10} {'max':>10} {'first':>10} {'last':>10} {'changes':>8}")
    for k in sorted(history):
        h = history[k]
        changes = int(np.sum(np.diff(h) != 0))
        print(f"{k:<18} {min(h):>10} {max(h):>10} {h[0]:>10} {h[-1]:>10} {changes:>8}")

    # The single most important assertion: x_pos must climb while holding right.
    x = history.get("x_pos", [])
    if x:
        print()
        if max(x) - x[0] > 200:
            print(f"PASS  x_pos advanced {x[0]} -> {max(x)} while holding right.")
        else:
            print(f"FAIL  x_pos barely moved ({x[0]} -> {max(x)}). Address is wrong.")


if __name__ == "__main__":
    main()
