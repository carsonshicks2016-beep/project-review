#!/usr/bin/env python3
"""Prepare one named row of docs/run_matrix.csv as an immutable run directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

from generate_geometry import read_outline, sample_wing, write_vertex, sha256

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "case" / "ibamr" / "input3d"
MATRIX = ROOT / "docs" / "run_matrix.csv"
FREQUENCY = 26.1
RADIUS = 0.0483


def read_case(case_id: str) -> dict[str, str]:
    with MATRIX.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["case_id"] == case_id:
                return row
    raise ValueError(f"unknown case_id {case_id!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_id")
    parser.add_argument("--output-root", type=Path, default=ROOT / "case" / "runs")
    parser.add_argument("--cycles", type=int, default=40)
    args = parser.parse_args()
    if args.cycles < 4:
        parser.error("--cycles must be at least 4 to capture the final three cycles")

    row = read_case(args.case_id)
    run_dir = args.output_root / args.case_id
    if run_dir.exists():
        parser.error(f"refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    max_levels = int(row["max_levels"])
    steps = int(row["dt_max_steps_per_period"])
    dx = RADIUS / (16 * 2 ** (max_levels - 1))
    base = TEMPLATE.read_text()
    replacements = (
        (r"(?m)^MAX_LEVELS = \d+$", f"MAX_LEVELS = {max_levels}"),
        (r"(?m)^STEPS_PER_PERIOD = \d+$", f"STEPS_PER_PERIOD = {steps}"),
        (r"(?m)^END_CYCLES = \d+$", f"END_CYCLES = {args.cycles}"),
    )
    for pattern, replacement in replacements:
        base, count = re.subn(pattern, replacement, base, count=1)
        if count != 1:
            raise ValueError(f"template does not contain one line matching {pattern!r}")
    input_path = run_dir / "input3d"
    input_path.write_text(base)

    outline_path = ROOT / "reference_data" / "wing_outline.csv"
    outline = read_outline(outline_path)
    one_wing, _ = sample_wing(outline, dx)
    right = [(x, y, z) for x, y, z in one_wing]
    left = [(x, -y, z) for x, y, z in one_wing]
    # Apply the same initial pose as the generator, preserving node ordering.
    from generate_geometry import pose
    right = [pose(point, 1) for point in right]
    left = [pose(point, -1) for point in left]
    write_vertex(run_dir / "right_wing.vertex", right)
    write_vertex(run_dir / "left_wing.vertex", left)
    for name in ("right_wing.vertex", "left_wing.vertex"):
        if not (run_dir / name).is_file():
            raise RuntimeError(f"failed to create {name}")

    manifest = {
        "case_id": args.case_id,
        "input_sha256": sha256(input_path),
        "outline_sha256": sha256(outline_path),
        "right_wing_vertex_sha256": sha256(run_dir / "right_wing.vertex"),
        "left_wing_vertex_sha256": sha256(run_dir / "left_wing.vertex"),
        "spatial_level": row["spatial_level"],
        "time_step_level": row["time_step_level"],
        "max_levels": max_levels,
        "base_cells_per_R": int(row["base_cells_per_R"]),
        "dt_max_steps_per_period": steps,
        "finest_nominal_dx_m": dx,
        "point_spacing_m": dx,
        "cycles_requested": args.cycles,
        "maximum_local_mpi_ranks": 8,
        "maximum_solver_memory_gb": 8,
        "launch_status": "prepared_not_run",
        "notes": row["notes"],
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
