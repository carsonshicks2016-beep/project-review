"""Merge parallel exploration archives into the main one.

Exploration is single-threaded, so using more than one core means running
several explorers against their own archive files and unioning the results.
Each shard keeps the cheaper route to any cell both of them found.

    python tools/world_merge.py checkpoints/world_shard_*.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smwrl.world_archive import WorldArchive, load_archive  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", type=Path)
    ap.add_argument("--into", type=Path,
                    default=ROOT / "checkpoints" / "world_archive.pkl")
    args = ap.parse_args()

    main_archive = load_archive(args.into) if args.into.exists() else WorldArchive()
    before = len(main_archive.cells)
    print(f"main archive: {before} cells")

    for shard in args.shards:
        if not shard.exists() or shard.resolve() == args.into.resolve():
            continue
        try:
            other = load_archive(shard)
        except Exception as e:
            print(f"  {shard.name}: unreadable ({type(e).__name__}), skipped")
            continue
        added = main_archive.absorb(other)
        print(f"  {shard.name}: {len(other.cells)} cells -> {added} new/improved")

    main_archive.save(args.into)
    best = main_archive.best_cell
    print(f"\nmerged: {before} -> {len(main_archive.cells)} cells")
    print(f"translevels: {sorted(main_archive.translevels)}")
    if best:
        print(f"frontier: translevel {best.translevel} room {best.room} "
              f"x {best.x}, {best.cleared} level(s) cleared")


if __name__ == "__main__":
    main()
