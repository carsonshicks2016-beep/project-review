#!/usr/bin/env python3
"""Prepare a hashed, coarse-grid Stage A coupling-diagnosis study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from generate_geometry import sha256

ROOT = Path(__file__).parents[1]
PREPARE = ROOT / "tools" / "prepare_verification.py"
SCENARIOS = ("stationary", "zero_amplitude", "stationary_uniform", "matched_translation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "stage_a_diagnosis")
    parser.add_argument("--duration-s", type=float, default=(1.0 / 26.1) / 8192.0)
    parser.add_argument("--target-stiffness", type=float, default=10.0,
                        help="previously best one-step diagnostic value; not an accepted calibration")
    args = parser.parse_args()
    if args.duration_s <= 0.0 or args.target_stiffness <= 0.0:
        parser.error("duration and target stiffness must be positive")
    if args.output_root.exists():
        parser.error(f"refusing to overwrite diagnosis study: {args.output_root}")
    args.output_root.mkdir(parents=True)

    # The base preparer accepts one case at a time. Prepare each scenario into
    # its own child and keep a single top-level manifest over those immutable inputs.
    scenarios = []
    for name in SCENARIOS:
        child = args.output_root / name
        subprocess.run([sys.executable, str(PREPARE), "--output-root", str(child),
                        "--case", name, "--spatial", "coarse", "--timestep", "dt_coarse",
                        "--duration-s", f"{args.duration_s:.16g}",
                        "--target-stiffness", f"{args.target_stiffness:.16g}"], check=True)
        case_dir = child / f"{name}_coarse_dt_coarse"
        scenarios.append({
            "name": name,
            "case_dir": str(case_dir.relative_to(args.output_root)),
            "manifest_sha256": sha256(case_dir / "run_manifest.json"),
            "input_sha256": sha256(case_dir / "input3d"),
        })

    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT.parents[1],
                                  check=True, capture_output=True, text=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    manifest = {
        "study": "Stage A moving-boundary diagnostic controls",
        "status": "PREPARED_NOT_RUN",
        "interpretation": "numerical diagnostics only; not a hawkmoth result or validation",
        "diagnostic_design": {
            "stationary": "quiescent fluid and stationary paired wing",
            "zero_amplitude": "zero-amplitude prescribed hover control in quiescent fluid",
            "stationary_uniform": "10 m/s initialized flow and stationary wing; nonzero relative-flow control",
            "matched_translation": "10 m/s initialized flow and wing translation at 10 m/s; zero-relative-flow control",
            "factor_interpretation": "stationary_uniform versus matched_translation isolates the prescribed target-motion change under identical initial flow",
        },
        "duration_s": args.duration_s,
        "target_stiffness_N_m_per_point": args.target_stiffness,
        "grid": "coarse Stage A hierarchy only; no convergence claim",
        "resource_limits": {"mpi_ranks_max": 8, "solver_rss_gb_max": 8, "free_disk_gb_min": 40},
        "git_revision_at_preparation": revision,
        "preparer_sha256": sha256(PREPARE),
        "application_source_sha256": sha256(ROOT / "src" / "main.cpp"),
        "scenario_inputs": scenarios,
    }
    path = args.output_root / "diagnosis_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    (args.output_root / "diagnosis_manifest.sha256").write_text(f"{sha256(path)}  diagnosis_manifest.json\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
