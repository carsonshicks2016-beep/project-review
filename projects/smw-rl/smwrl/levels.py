"""Registry of legacy specialist/save-state keys and their live identities.

``ROUTE`` is an isolated-state diagnostic inventory, not a traversable game
route.  Some stable-retro filenames are historical aliases; ``label`` is the
canonical in-game stage name while ``state`` and the dictionary key stay stable
so existing checkpoints remain usable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Level:
    state: str          # save state name, matches integration/*.state
    label: str          # human-readable, used in the split display
    length: int         # approximate x extent, for progress display only
    translevel: int     # live $7E13BF ID used by the continuous dispatcher
    map_pos: tuple[int, int]  # verified 16-bit overworld coordinates


LEVELS: dict[str, Level] = {
    "YoshiIsland1": Level("YoshiIsland1", "Yoshi's Island 1", 4600, 0x29, (56, 136)),
    "YoshiIsland2": Level("YoshiIsland2", "Yoshi's Island 2", 4800, 0x2A, (152, 136)),
    "YoshiIsland3": Level("YoshiIsland3", "Yoshi's Island 3", 3800, 0x27, (152, 104)),
    "YoshiIsland4": Level("YoshiIsland4", "Yoshi's Island 4", 4400, 0x26, (184, 72)),
    "DonutPlains1": Level("DonutPlains1", "Donut Plains 1", 5000, 0x15, (88, 280)),
    "DonutPlains2": Level("DonutPlains2", "Donut Plains 2", 4600, 0x09, (56, 216)),
    "DonutPlains3": Level("DonutPlains3", "Donut Ghost House", 3600, 0x04, (88, 168)),
    "DonutPlains4": Level("DonutPlains4", "Donut Plains 3", 4200, 0x05, (152, 168)),
    "DonutPlains5": Level("DonutPlains5", "Donut Plains 4", 3400, 0x06, (184, 200)),
    "VanillaDome1": Level("VanillaDome1", "Vanilla Dome 1", 4800, 0x3E, (88, 296)),
    "VanillaDome2": Level("VanillaDome2", "Vanilla Dome 2", 5200, 0x3C, (136, 264)),
    "VanillaDome3": Level("VanillaDome3", "Vanilla Dome 3", 3800, 0x2B, (136, 200)),
    "VanillaDome4": Level("VanillaDome4", "Vanilla Dome 4", 4000, 0x2E, (200, 232)),
    "VanillaDome5": Level("VanillaDome5", "Vanilla Dome 5", 4400, 0x3D, (200, 264)),
    "Bridges1": Level("Bridges1", "Cheese Bridge Area", 4200, 0x0F, (328, 88)),
    "Bridges2": Level("Bridges2", "Cookie Mountain", 5400, 0x10, (376, 88)),
    "Forest1": Level("Forest1", "Forest of Illusion 1", 4600, 0x42, (136, 376)),
    "Forest2": Level("Forest2", "Forest of Illusion 2", 4200, 0x44, (168, 424)),
    "Forest3": Level("Forest3", "Forest of Illusion 3", 4000, 0x47, (136, 456)),
    "Forest4": Level("Forest4", "Forest of Illusion 4", 4400, 0x41, (104, 376)),
    "Forest5": Level("Forest5", "Forest of Illusion 5", 3800, 0x43, (72, 424)),
    "ChocolateIsland1": Level("ChocolateIsland1", "Chocolate Island 1", 4600, 0x22, (392, 360)),
    "ChocolateIsland2": Level("ChocolateIsland2", "Chocolate Ghost House", 5600, 0x21, (344, 360)),
    "ChocolateIsland3": Level("ChocolateIsland3", "Chocolate Island 3", 4000, 0x24, (344, 440)),
}

LEVEL_BY_TRANSLEVEL = {level.translevel: name for name, level in LEVELS.items()}
assert len(LEVEL_BY_TRANSLEVEL) == len(LEVELS), "translevel IDs must be unique"

ROUTE: list[str] = [
    "YoshiIsland1", "YoshiIsland2", "YoshiIsland3", "YoshiIsland4",
    "DonutPlains1", "DonutPlains2", "DonutPlains3", "DonutPlains4", "DonutPlains5",
    "VanillaDome1", "VanillaDome2", "VanillaDome3", "VanillaDome4", "VanillaDome5",
    "Bridges1", "Bridges2",
    "Forest1", "Forest2", "Forest3", "Forest4", "Forest5",
    "ChocolateIsland1", "ChocolateIsland2", "ChocolateIsland3",
]

# Sensible order to train in: the early levels are the simplest and make the
# best smoke tests.
TRAINING_ORDER = ROUTE
