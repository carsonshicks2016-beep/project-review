#!/usr/bin/env python3
"""Compare IBAMR's six manufactured-flow error norms with its golden output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

NORM_RE = re.compile(r"\s+(L1-norm|L2-norm|max-norm):\s+([-+0-9.eE]+)")
FIELDS = ("L1-norm", "L2-norm", "max-norm")


def extract(path: Path) -> dict[str, float]:
    lines = path.read_text(errors="replace").splitlines()
    found: dict[str, float] = {}
    field = None
    for line in lines:
        if line.startswith("Error in u at time"):
            field = "velocity"
        elif line.startswith("Error in p at time"):
            field = "pressure"
        match = NORM_RE.fullmatch(line)
        if match and field:
            found[f"{field}_{match.group(1)}"] = float(match.group(2))
    expected = {f"{field}_{norm}" for field in ("velocity", "pressure") for norm in FIELDS}
    missing = sorted(expected.difference(found))
    if missing:
        raise ValueError(f"missing expected norm entries in {path}: {missing}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=1.0e-6)
    args = parser.parse_args()
    observed = extract(args.run_dir / "solver.stdout.log")
    reference = extract(args.run_dir / "expected.stdout.log")
    comparisons = {}
    passed = True
    for key, expected in reference.items():
        actual = observed[key]
        relative = abs(actual - expected) / max(abs(expected), 1.0e-14)
        ok = relative <= args.relative_tolerance
        passed &= ok
        comparisons[key] = {"expected": expected, "observed": actual, "relative_error": relative, "pass": ok}
    result = {"gate": "PASS" if passed else "FAIL", "relative_tolerance": args.relative_tolerance,
              "norms": comparisons}
    (args.run_dir / "solver_reference_comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
