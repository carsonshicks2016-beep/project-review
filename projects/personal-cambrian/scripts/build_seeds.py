#!/usr/bin/env python3
"""Write the hand-authored seed genomes to data/seeds/*.genome.json.

    python3 scripts/build_seeds.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import SEEDS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "seeds")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, fn in SEEDS.items():
        g = fn().assert_valid()
        path = os.path.join(OUT, f"{name}.genome.json")
        with open(path, "w") as f:
            f.write(g.to_json())
        print(f"wrote {path}  ({len(g.parts)} parts, {len(g.muscles)} muscles, "
              f"hash={g.hash()[:12]})")


if __name__ == "__main__":
    main()
