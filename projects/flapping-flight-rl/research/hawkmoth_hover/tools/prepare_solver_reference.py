#!/usr/bin/env python3
"""Prepare the pinned IBAMR 0.19.0 3-D staggered-flow regression case."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).parents[1]
REFERENCE = ROOT / "reference_data" / "ibamr_0.19.0_navier_stokes"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "stage_a_ibamr_reference")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite reference run directory: {args.output}")
    args.output.mkdir(parents=True)
    shutil.copy2(REFERENCE / "navier_stokes_01_3d.input", args.output / "input3d")
    shutil.copy2(REFERENCE / "navier_stokes_01_3d.output", args.output / "expected.stdout.log")
    source_revision = subprocess.run(
        ["git", "-C", str(Path.home() / "Applications/ibamr-research/source"), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "case": "IBAMR navier_stokes_01_3d staggered manufactured-flow regression",
        "ibamr_version": "0.19.0",
        "ibamr_source_revision": source_revision,
        "source_sha256": sha256(REFERENCE / "navier_stokes_01.cpp"),
        "input_sha256": sha256(args.output / "input3d"),
        "expected_output_sha256": sha256(args.output / "expected.stdout.log"),
        "run_status": "prepared_not_run",
        "comparison": "compare six final velocity and pressure L1, L2, and max norms against the bundled upstream output",
    }
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
