#!/usr/bin/env python3
"""Prepare a one-step uniform-flow control with all IB force inputs disabled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

from generate_geometry import sha256

ROOT = Path(__file__).parents[1]
PREPARE = ROOT / "tools" / "prepare_verification.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path,
                        default=ROOT / "results" / "stage_a_force_isolation")
    parser.add_argument("--duration-s", type=float, default=(1.0 / 26.1) / 8192.0)
    parser.add_argument("--amr-levels", type=int, choices=(1, 3), default=3,
                        help="maximum hierarchy levels; 1 disables finer AMR levels")
    args = parser.parse_args()
    if args.duration_s <= 0.0:
        parser.error("duration must be positive")
    if args.output_root.exists():
        parser.error(f"refusing to overwrite force-isolation study: {args.output_root}")
    args.output_root.mkdir(parents=True)

    child = args.output_root / "prepared"
    subprocess.run([
        sys.executable, str(PREPARE), "--output-root", str(child),
        "--case", "stationary_uniform", "--spatial", "coarse",
        "--timestep", "dt_coarse", "--duration-s", f"{args.duration_s:.16g}",
        "--target-stiffness", "0",
    ], check=True)
    case_dir = child / "stationary_uniform_coarse_dt_coarse"
    input_path = case_dir / "input3d"
    deck = input_path.read_text()
    deck, count = re.subn(r"^ENABLE_IB_FORCING = TRUE$", "ENABLE_IB_FORCING = FALSE",
                          deck, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("expected exactly one ENABLE_IB_FORCING input setting")
    deck, count = re.subn(r"^ENABLE_FLOW_STATE_DIAGNOSTICS = FALSE$",
                          "ENABLE_FLOW_STATE_DIAGNOSTICS = TRUE", deck, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError("expected exactly one ENABLE_FLOW_STATE_DIAGNOSTICS input setting")
    if args.amr_levels == 1:
        deck, count = re.subn(r"^MAX_LEVELS = 3$", "MAX_LEVELS = 1", deck,
                              count=1, flags=re.MULTILINE)
        if count != 1:
            raise RuntimeError("expected the coarse base case to have three AMR levels")
    input_path.write_text(deck)

    run_manifest_path = case_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest.update({
        "verification_case": "stationary_uniform_force_disabled",
        "ib_forcing_enabled": False,
        "flow_state_diagnostics_enabled": True,
        "max_levels": args.amr_levels,
        "nominal_finest_dx_m": 0.0483 / (16 * 2 ** (args.amr_levels - 1)),
        "target_point_stiffness_N_m_per_point": 0.0,
        "diagnostic_purpose": "check uniform-flow preservation and control-volume force with no immersed-boundary force spreading",
        "input_sha256": sha256(input_path),
        "launch_status": "prepared_not_run",
        "interpretation": "diagnostic null control only; not physical validation",
    })
    if args.amr_levels == 1:
        run_manifest["amr_note"] = (
            "AMR finer levels disabled; base mesh and already-generated dense Lagrangian geometry retained"
        )
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2) + "\n")
    (case_dir / "run_manifest.sha256").write_text(
        f"{sha256(run_manifest_path)}  run_manifest.json\n")

    study = {
        "study": "Stage A no-immersed-boundary-forcing uniform-flow isolation",
        "status": "PREPARED_NOT_RUN",
        "question": "Does the first-step force/divergence anomaly persist when IB target springs and standard Lagrangian force generation are disabled?",
        "case_dir": str(case_dir.relative_to(args.output_root)),
        "duration_s": args.duration_s,
        "controls_held_fixed": ["base mesh", "initial 10 m/s uniform velocity", "physical boundary conditions", "fluid parameters", "time-step ceiling", "diagnostic force evaluator", "Lagrangian geometry file"],
        "amr_levels": args.amr_levels,
        "single_intended_change": ("ENABLE_IB_FORCING=FALSE, TARGET_STIFFNESS=0, and MAX_LEVELS=1; added diagnostics do not alter the numerical scheme"
                                   if args.amr_levels == 1 else
                                   "ENABLE_IB_FORCING=FALSE and TARGET_STIFFNESS=0; added diagnostics do not alter the numerical scheme"),
        "force_definition": "zero target-point spring stiffness and IBStandardForceGen callback not registered; force evaluator remains enabled as a diagnostic measurement",
        "resource_limits": {"mpi_ranks_max": 8, "solver_rss_gb_max": 8, "free_disk_gb_min": 40},
        "preparer_sha256": sha256(Path(__file__)),
        "base_preparer_sha256": sha256(PREPARE),
        "case_input_sha256": sha256(input_path),
        "case_manifest_sha256": sha256(run_manifest_path),
        "solver_source_sha256": sha256(ROOT / "src" / "main.cpp"),
    }
    study_path = args.output_root / "force_isolation_manifest.json"
    study_path.write_text(json.dumps(study, indent=2) + "\n")
    (args.output_root / "force_isolation_manifest.sha256").write_text(
        f"{sha256(study_path)}  force_isolation_manifest.json\n")
    print(json.dumps(study, indent=2))


if __name__ == "__main__":
    main()
