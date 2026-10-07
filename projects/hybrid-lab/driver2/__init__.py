"""Driver 2.0 — Progressive Curriculum GA with LSTM brains.

A standalone training package for the 919 Evo on the Nordschleife.
Splits the track into 64 segments and progressively condenses them
(64 → 32 → 16 → 8 → 4 → 2 → 1) as the population masters each level.

Usage:
    python -m driver2 train [--pop 200] [--segments 64]
    python -m driver2 eval  --genome <path> [--laps 3]
    python -m driver2 watch --run <run-id>
"""

__version__ = "0.1.0"
