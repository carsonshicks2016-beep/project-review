"""Write stages to ``packages/shared/stages``.

One copy, in one place, read by both sides. v1 kept a byte-identical duplicate
under the viewer and synced it with a ``shutil.copy`` buried in a demo script;
editing one and forgetting the other rendered geometry the physics never
stepped. The viewer serves this directory — it does not get its own copy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rallyai import contracts
from rallyai.stage.fixtures import ALL
from rallyai.stage.generator import N_TIERS, generate


def stages_dir() -> Path:
    return Path(contracts.schema_dir()).parent / "stages"


def _report(path: Path, doc: dict) -> None:
    shown = path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path
    print(f"wrote {shown}  ({doc['length_m']:.1f} m, {len(doc['centerline'])} knots, "
          f"{len(doc['obstacles'])} obstacles)")
    print(f"      {doc['content_hash']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=None,
                        help=f"authored stages to write (default: all). "
                             f"choices: {', '.join(ALL)}")
    parser.add_argument("--seed", type=int, nargs="+", default=None,
                        help="generate procedural stages for these seeds instead")
    parser.add_argument("--tier", type=int, default=0,
                        help=f"curriculum tier 0..{N_TIERS - 1} (default 0)")
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    args = parser.parse_args(argv)

    out_dir = args.out or stages_dir()

    if args.seed is not None:
        for seed in args.seed:
            stage = generate(seed, args.tier)
            path = contracts.write_json(stage, out_dir / f"{stage['id']}.json", kind="stage")
            _report(path, contracts.read_json(path, kind="stage"))
        return 0

    for name in args.names or list(ALL):
        if name not in ALL:
            parser.error(f"unknown stage {name!r}; choices: {', '.join(ALL)}")
        path = contracts.write_json(ALL[name](), out_dir / f"{name}.json", kind="stage")
        _report(path, contracts.read_json(path, kind="stage"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
