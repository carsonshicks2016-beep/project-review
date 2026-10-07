"""Discrete action set for Super Mario World.

The raw SNES action space is MultiBinary(12), i.e. 4096 combinations, almost
all of them meaningless. Learning is far faster over a curated set of the
combinations a player actually holds.

SMW specifics: Y is run/carry, B is jump, A is spin jump, X also dashes. A
speedrun is essentially "hold Y and right, jump at the right moments", so the
run-modified variants matter most.
"""

from __future__ import annotations

import numpy as np
import hashlib

# Button order for the SNES core, taken from stable_retro/cores/snes9x.json.
BUTTONS = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]

ACTION_COMBOS: list[list[str]] = [
    [],                          # 0  no-op
    ["RIGHT"],                   # 1  walk right
    ["RIGHT", "Y"],              # 2  run right
    ["RIGHT", "B"],              # 3  jump right
    ["RIGHT", "Y", "B"],         # 4  run-jump right  (the speedrun workhorse)
    ["RIGHT", "Y", "A"],         # 5  run spin-jump right
    ["RIGHT", "DOWN"],           # 6  duck-slide right
    ["B"],                       # 7  jump in place
    ["Y", "B"],                  # 8  jump holding run (carry/momentum)
    ["LEFT"],                    # 9  walk left
    ["LEFT", "Y"],               # 10 run left
    ["LEFT", "B"],               # 11 jump left
    ["DOWN"],                    # 12 duck / enter pipe
    ["UP"],                      # 13 up (doors, vines)
]

N_ACTIONS = len(ACTION_COMBOS)

# Compact labels for the live policy-distribution display.
_GLYPH = {"RIGHT": "→", "LEFT": "←", "UP": "↑", "DOWN": "↓"}
ACTION_LABELS: list[str] = [
    "idle" if not combo else "+".join(_GLYPH.get(b, b) for b in combo)
    for combo in ACTION_COMBOS
]


def build_action_table() -> np.ndarray:
    """(N_ACTIONS, 12) uint8 matrix mapping a discrete action to button presses."""
    table = np.zeros((N_ACTIONS, len(BUTTONS)), dtype=np.uint8)
    for i, combo in enumerate(ACTION_COMBOS):
        for name in combo:
            table[i, BUTTONS.index(name)] = 1
    return table


ACTION_TABLE = build_action_table()
ACTION_SCHEMA_VERSION = 1
ACTION_SCHEMA_SHA256 = hashlib.sha256(
    ACTION_TABLE.tobytes() + "|".join(BUTTONS).encode("ascii")
).hexdigest()
