#!/usr/bin/env python3
"""Summarize Stage A verification matrices without upgrading failures to passes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from typing import Any

TARGET_RMS_DX_LIMIT = 0.02
TARGET_MAX_DX_LIMIT = 0.10
NO_SLIP_RMS_LIMIT = 0.05
NO_SLIP_MAX_LIMIT = 0.20
DIV_RMS_LIMIT = 1.0e-4
DIV_MAX_LIMIT = 1.0e-3
RADIUS_M = 0.0483


def run_summary(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    resource_path = run_dir / "resource_record.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    resource = json.loads(resource_path.read_text()) if resource_path.exists() else {"status": "NOT_RUN"}
    history_path = run_dir / "force_history.csv"
    if not history_path.exists() or resource.get("status") != "COMPLETED":
        return {"run_id": manifest.get("run_id", run_dir.name), "status": resource.get("status", "NOT_RUN"),
                "run_dir": str(run_dir), "manifest": manifest, "resource": resource,
                "gate": "NOT_EVALUATED"}
    with history_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return {"run_id": manifest.get("run_id", run_dir.name), "status": "FAILED_EMPTY_HISTORY",
                "run_dir": str(run_dir), "manifest": manifest, "resource": resource, "gate": "FAIL"}
    values = {key: [float(row[key]) for row in rows] for key in rows[0] if key != "time_s"}
    dx = float(manifest["nominal_finest_dx_m"])
    velocity = manifest.get("translation_velocity_m_s", (0.0, 0.0, 0.0))
    u_ref = (sum(float(x) ** 2 for x in velocity) ** 0.5) or 1.0
    cfl_values = values.get("realized_velocity_cfl", [])
    if not cfl_values:
        log_path = run_dir / "hawkmoth_hover.log"
        if log_path.exists():
            cfl_values = [float(value) for value in re.findall(r"CFL number = ([+\-0-9.eE]+)", log_path.read_text())]
    metrics = {
        "rows": len(rows),
        "target_rms_over_dx_max": max(values["target_error_rms_m"]) / dx,
        "target_max_over_dx_max": max(values["target_error_max_m"]) / dx,
        "minimum_surface_domain_margin_over_dx": (min(values["minimum_surface_domain_margin_m"]) / dx
                                                   if "minimum_surface_domain_margin_m" in values else None),
        "no_slip_rms_over_Uref_max": max(values["no_slip_velocity_rms_over_Uref"]),
        "no_slip_max_over_Uref_max": max(values["no_slip_velocity_max_over_Uref"]),
        "divergence_rms_times_R_over_Uref_max": max(values["divergence_rms_s_inv"]) * RADIUS_M / u_ref,
        "divergence_max_times_R_over_Uref_max": max(values["divergence_max_s_inv"]) * RADIUS_M / u_ref,
        "surface_band_width_over_finest_dx": (max(values["divergence_surface_band_width_m"]) / dx
                                               if "divergence_surface_band_width_m" in values else None),
        "surface_band_divergence_rms_times_R_over_Uref_max": (max(values["divergence_surface_band_rms_s_inv"]) * RADIUS_M / u_ref
                                                              if "divergence_surface_band_rms_s_inv" in values else None),
        "surface_band_divergence_max_times_R_over_Uref_max": (max(values["divergence_surface_band_max_s_inv"]) * RADIUS_M / u_ref
                                                              if "divergence_surface_band_max_s_inv" in values else None),
        "max_force_N": max((sum(float(row[k]) ** 2 for k in ("Fx_N", "Fy_N", "Fz_N")) ** 0.5) for row in rows),
        "max_power_W": max(abs(value) for value in values["power_W"]),
        "dt_min_s": min(values["dt_s"]),
        "dt_max_s": max(values["dt_s"]),
        "realized_velocity_cfl_max": max(cfl_values) if cfl_values else None,
    }
    required_diagnostics = (
        "minimum_surface_domain_margin_m",
        "no_slip_velocity_rms_over_Uref",
        "no_slip_velocity_max_over_Uref",
        "divergence_surface_band_width_m",
        "divergence_surface_band_rms_s_inv",
        "divergence_surface_band_max_s_inv",
    )
    missing_diagnostics = [key for key in required_diagnostics if key not in values]
    if missing_diagnostics:
        return {"run_id": manifest.get("run_id", run_dir.name), "status": resource.get("status"),
                "run_dir": str(run_dir), "manifest": manifest, "resource": resource,
                "metrics": metrics, "missing_diagnostics": missing_diagnostics,
                "gate": "INCOMPLETE_FIELDS"}
    checks = {
        "target_rms": metrics["target_rms_over_dx_max"] <= TARGET_RMS_DX_LIMIT,
        "target_max": metrics["target_max_over_dx_max"] <= TARGET_MAX_DX_LIMIT,
        "surface_points_inside_hierarchy_with_2h_margin": (metrics["minimum_surface_domain_margin_over_dx"] is not None
                                                             and metrics["minimum_surface_domain_margin_over_dx"] > 2.0),
        "no_slip_rms": metrics["no_slip_rms_over_Uref_max"] <= NO_SLIP_RMS_LIMIT,
        "no_slip_max": metrics["no_slip_max_over_Uref_max"] <= NO_SLIP_MAX_LIMIT,
        "divergence_rms": metrics["divergence_rms_times_R_over_Uref_max"] <= DIV_RMS_LIMIT,
        "divergence_max": metrics["divergence_max_times_R_over_Uref_max"] <= DIV_MAX_LIMIT,
        "surface_band_divergence_rms": (metrics["surface_band_divergence_rms_times_R_over_Uref_max"] is not None
                                        and metrics["surface_band_divergence_rms_times_R_over_Uref_max"] <= DIV_RMS_LIMIT),
        "surface_band_divergence_max": (metrics["surface_band_divergence_max_times_R_over_Uref_max"] is not None
                                        and metrics["surface_band_divergence_max_times_R_over_Uref_max"] <= DIV_MAX_LIMIT),
    }
    # A local pass is only a necessary condition. Refinement trends and the
    # upstream regression gate are evaluated at the matrix/report level.
    return {"run_id": manifest.get("run_id", run_dir.name), "status": resource.get("status"),
            "run_dir": str(run_dir), "manifest": manifest, "resource": resource,
            "metrics": metrics, "checks": checks,
            "gate": "PASS_LOCAL_THRESHOLDS" if all(checks.values()) else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="prepared verification roots")
    parser.add_argument("--output", type=Path, help="write combined JSON summary here")
    parser.add_argument("--solver-reference-json", type=Path,
                        help="comparison JSON produced by verify_solver_reference.py")
    args = parser.parse_args()
    report = {"overall_gate": "INCOMPLETE", "tolerances": {
                  "target_rms_over_dx": TARGET_RMS_DX_LIMIT, "target_max_over_dx": TARGET_MAX_DX_LIMIT,
                  "no_slip_rms_over_Uref": NO_SLIP_RMS_LIMIT, "no_slip_max_over_Uref": NO_SLIP_MAX_LIMIT,
                  "divergence_rms_times_R_over_Uref": DIV_RMS_LIMIT,
                  "divergence_max_times_R_over_Uref": DIV_MAX_LIMIT},
              "runs": [],
              "independent_momentum_closure_gate": "NOT_PASSED: outer-boundary flux and traction are not independently integrated"}
    if args.solver_reference_json:
        report["solver_reference_regression"] = json.loads(args.solver_reference_json.read_text())
    else:
        report["solver_reference_regression"] = {"gate": "NOT_PROVIDED"}
    for root in args.roots:
        matrix_path = root / "prepared_matrix.csv"
        if not matrix_path.exists():
            continue
        with matrix_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                report["runs"].append(run_summary(Path(row["run_dir"])))
    passed = sum(run["gate"] == "PASS_LOCAL_THRESHOLDS" for run in report["runs"])
    not_evaluated = sum(run["gate"] in {"NOT_EVALUATED", "INCOMPLETE_FIELDS"} for run in report["runs"])
    failed = sum(run["gate"] == "FAIL" for run in report["runs"])
    failed_execution = sum(run["status"] in {"FAILED", "STOPPED_RESOURCE_CAP", "FAILED_EMPTY_HISTORY"}
                           for run in report["runs"])
    report["counts"] = {"passed_local_only": passed, "not_evaluated": not_evaluated,
                         "incomplete_fields": sum(run["gate"] == "INCOMPLETE_FIELDS" for run in report["runs"]),
                         "failed_local_thresholds": failed, "failed_executions": failed_execution}
    if failed or failed_execution:
        report["overall_gate"] = "FAILED"
        report["overall_gate_note"] = (
            "At least one verification case violated a predeclared local threshold or failed to execute. "
            "This is a Stage A failure; remaining matrix and independent momentum checks may still be incomplete."
        )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
