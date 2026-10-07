#!/usr/bin/env python3
"""Summarize the immutable one-step Stage A diagnosis cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from generate_geometry import sha256
import summarize_momentum_budget


TARGET_RMS_LIMIT_DX = 0.02
TARGET_MAX_LIMIT_DX = 0.10
NO_SLIP_RMS_LIMIT = 0.05
NO_SLIP_MAX_LIMIT = 0.20
DIVERGENCE_RMS_LIMIT = 1.0e-4
DIVERGENCE_MAX_LIMIT = 1.0e-3
REFERENCE_RADIUS_M = 0.0483


def summarize(root: Path, extra_cases: list[tuple[str, Path]] | None = None) -> dict:
    study = json.loads((root / "diagnosis_manifest.json").read_text())
    scenarios = []
    configured_cases = [(case["name"], root / case["case_dir"], case["input_sha256"])
                        for case in study["scenario_inputs"]]
    configured_cases.extend((name, path if path.is_absolute() else root / path, None)
                            for name, path in extra_cases or [])
    for case_name, run_dir, prepared_input_hash in configured_cases:
        resource_path = run_dir / "resource_record.json"
        history_path = run_dir / "force_history.csv"
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        if not resource_path.exists() or not history_path.exists():
            scenarios.append({"name": case_name, "gate": "NOT_RUN"})
            continue
        resource = json.loads(resource_path.read_text())
        with history_path.open(newline="") as stream:
            history = list(csv.DictReader(stream))
        dx = float(manifest["nominal_finest_dx_m"])
        uref = float(manifest["translation_velocity_m_s"][0])
        if uref <= 0.0:
            uref = 1.0
        step_metrics = []
        for row in history:
            values = {
                "target_rms_over_dx": float(row["target_error_rms_m"]) / dx,
                "target_max_over_dx": float(row["target_error_max_m"]) / dx,
                "no_slip_rms_over_Uref": float(row["no_slip_velocity_rms_m_s"]) / uref,
                "no_slip_max_over_Uref": float(row["no_slip_velocity_max_m_s"]) / uref,
                "divergence_rms_times_R_over_Uref": float(row["divergence_rms_s_inv"]) * REFERENCE_RADIUS_M / uref,
                "divergence_max_times_R_over_Uref": float(row["divergence_max_s_inv"]) * REFERENCE_RADIUS_M / uref,
            }
            values["gates"] = {
                "target_rms": values["target_rms_over_dx"] <= TARGET_RMS_LIMIT_DX,
                "target_max": values["target_max_over_dx"] <= TARGET_MAX_LIMIT_DX,
                "no_slip_rms": values["no_slip_rms_over_Uref"] <= NO_SLIP_RMS_LIMIT,
                "no_slip_max": values["no_slip_max_over_Uref"] <= NO_SLIP_MAX_LIMIT,
                "divergence_rms": values["divergence_rms_times_R_over_Uref"] <= DIVERGENCE_RMS_LIMIT,
                "divergence_max": values["divergence_max_times_R_over_Uref"] <= DIVERGENCE_MAX_LIMIT,
            }
            values["force_N"] = [float(row[f"F{axis}_N"]) for axis in "xyz"]
            values["time_s"] = float(row["time_s"])
            step_metrics.append(values)
        all_gates = all(all(step["gates"].values()) for step in step_metrics)
        budget_path = run_dir / "momentum_budget.csv"
        budget = summarize_momentum_budget.summarize(budget_path) if budget_path.exists() else None
        scenarios.append({
            "name": case_name,
            "run_id": manifest["run_id"],
            "gate": "PASS_SHORT_CONTROL" if resource["status"] == "COMPLETED" and all_gates else
                    "FAIL_DIAGNOSTIC_GATES" if resource["status"] == "COMPLETED" else resource["status"],
            "input_sha256": prepared_input_hash or manifest["input_sha256"],
            "executable_sha256": resource["run_identity"]["executable_sha256_at_launch"],
            "source_sha256": resource["run_identity"]["application_source_sha256"]["research/hawkmoth_hover/src/main.cpp"],
            "resource": {
                "elapsed_seconds": resource["elapsed_seconds"],
                "peak_process_tree_rss_gb": resource["peak_process_tree_rss_gb"],
                "disk_bytes_after": resource["disk_bytes_after"],
                "free_disk_bytes_after": resource["free_disk_bytes_after"],
            },
            "step_metrics": step_metrics,
            "momentum_budget": budget,
        })
    completed = [item for item in scenarios if item["gate"] != "NOT_RUN"]
    return {
        "study": study["study"],
        "status": "DIAGNOSTIC_ONLY",
        "stage_a_gate": "FAILED / UNVALIDATED",
        "scope": "one-step diagnostic controls; no refinement trend or hawkmoth result",
        "acceptance_thresholds": {
            "target_rms_over_dx": TARGET_RMS_LIMIT_DX,
            "target_max_over_dx": TARGET_MAX_LIMIT_DX,
            "no_slip_rms_over_Uref": NO_SLIP_RMS_LIMIT,
            "no_slip_max_over_Uref": NO_SLIP_MAX_LIMIT,
            "divergence_rms_times_R_over_Uref": DIVERGENCE_RMS_LIMIT,
            "divergence_max_times_R_over_Uref": DIVERGENCE_MAX_LIMIT,
        },
        "cases_completed": len(completed),
        "cases": scenarios,
        "study_manifest_sha256": sha256(root / "diagnosis_manifest.json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--extra-case", action="append", default=[], metavar="NAME=RUN_DIR",
                        help="include another immutable run directory; may be repeated")
    args = parser.parse_args()
    extras = []
    for item in args.extra_case:
        if "=" not in item:
            parser.error("--extra-case values must have the form NAME=RUN_DIR")
        name, directory = item.split("=", 1)
        extras.append((name, Path(directory)))
    result = summarize(args.study_root, extras)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
