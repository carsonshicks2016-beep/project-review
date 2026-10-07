#!/usr/bin/env python3
"""Compare diagnostic force measurements with and without AMR after regrid refresh."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize_case(case_dir: Path) -> dict:
    case_dir = case_dir.resolve()
    resource_path = case_dir / "resource_record.json"
    resource = json.loads(resource_path.read_text())
    run = json.loads((case_dir / "run_manifest.json").read_text())
    force = read_csv(case_dir / "force_history.csv")[-1]
    flow = read_csv(case_dir / "flow_state_summary.csv")
    flow_post = next(row for row in flow if row["stage"] == "post_advance")
    trace = read_csv(case_dir / "control_volume_momentum_trace.csv")
    refreshed = [row for row in trace if row["stage"] == "post_regrid_lagged_integral"]
    stress = read_csv(case_dir / "control_volume_stress.csv")
    budget = read_csv(case_dir / "momentum_budget.csv")[-1]
    layout = read_csv(case_dir / "hierarchy_patch_layout.csv")
    return {
        "case_dir": str(case_dir),
        "max_levels": int(run["max_levels"]),
        "solver_status": resource["status"],
        "elapsed_seconds": resource["elapsed_seconds"],
        "peak_process_tree_rss_gb": resource["peak_process_tree_rss_gb"],
        "free_disk_bytes_after": resource["free_disk_bytes_after"],
        "post_regrid_lagged_momentum_by_structure_kg_m_s": [
            {"structure_id": int(row["structure_id"]),
             "momentum": [float(row[f"P_composite_{axis}_kg_m_s"]) for axis in "xyz"]}
            for row in refreshed
        ],
        "force_N": [float(force[f"F{axis}_N"]) for axis in "xyz"],
        "target_error_rms_dx": float(force["target_error_rms_m"]) /
        float(run["nominal_finest_dx_m"]),
        "no_slip_rms_over_Uref": float(force["no_slip_velocity_rms_over_Uref"]),
        "global_divergence_rms_s_inv": float(force["divergence_rms_s_inv"]),
        "global_divergence_max_s_inv": float(force["divergence_max_s_inv"]),
        "flow_post_advance": {
            key: float(flow_post[key]) for key in (
                "mean_u_x_m_s", "rms_u_y_m_s", "rms_u_z_m_s", "divergence_rms_s_inv", "divergence_max_s_inv")
        },
        "momentum_budget": {
            key: float(budget[key]) for key in (
                "dP_fluid_dt_x_N", "dP_fluid_dt_y_N", "dP_fluid_dt_z_N",
                "force_on_body_x_N", "force_on_body_y_N", "force_on_body_z_N",
                "pressure_traction_x_N", "pressure_traction_y_N", "pressure_traction_z_N",
                "advective_momentum_flux_x_N", "advective_momentum_flux_y_N", "advective_momentum_flux_z_N",
                "viscous_traction_x_N", "viscous_traction_y_N", "viscous_traction_z_N",
                "signed_residual_x_N", "signed_residual_y_N", "signed_residual_z_N", "normalized_residual")
        },
        "ibamr_implied_surface_flux_by_structure_N": [
            [float(row[f"ibamr_implied_surface_flux_{axis}_N"]) for axis in "xyz"] for row in stress
        ],
        "independent_level_zero_surface_flux_by_structure_N": [
            [float(row[f"coarse_surface_total_{axis}_N"]) for axis in "xyz"] for row in stress
        ],
        "patch_box_counts_by_stage": {
            stage: sum(row["stage"] == stage for row in layout)
            for stage in ("before_lagged_integral", "after_advance")
        },
        "artifact_sha256": {
            "input3d": run["input_sha256"],
            "resource_record": sha256(resource_path),
            "patch_layout": sha256(case_dir / "hierarchy_patch_layout.csv"),
            "executable": resource["run_identity"]["executable_sha256_at_launch"],
            "application_source": resource["run_identity"]["application_source_sha256"][
                "research/hawkmoth_hover/src/main.cpp"],
        },
    }


def summarize_no_diagnostics_case(case_dir: Path) -> dict:
    case_dir = case_dir.resolve()
    resource_path = case_dir / "resource_record.json"
    resource = json.loads(resource_path.read_text())
    run = json.loads((case_dir / "run_manifest.json").read_text())
    force = read_csv(case_dir / "force_history.csv")[-1]
    return {
        "case_dir": str(case_dir),
        "max_levels": int(run["max_levels"]),
        "solver_status": resource["status"],
        "elapsed_seconds": resource["elapsed_seconds"],
        "peak_process_tree_rss_gb": resource["peak_process_tree_rss_gb"],
        "diagnostic_output_enabled": False,
        "force_N": [float(force[f"F{axis}_N"]) for axis in "xyz"],
        "artifact_sha256": {
            "input3d": resource["run_identity"]["input3d_sha256"],
            "resource_record": sha256(resource_path),
            "executable": resource["run_identity"]["executable_sha256_at_launch"],
            "application_source": resource["run_identity"]["application_source_sha256"][
                "research/hawkmoth_hover/src/main.cpp"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("single_level_case", type=Path)
    parser.add_argument("amr_case", type=Path)
    parser.add_argument("--no-diagnostics-case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    one = summarize_case(args.single_level_case)
    amr = summarize_case(args.amr_case)
    result = {
        "study": "Post-regrid refresh of the IBAMR lagged momentum integral",
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "stage_a_gate": "FAILED / UNVALIDATED",
        "purpose": "test whether refreshing old-state CV momentum after regrid projection/synchronization removes the initial force transient",
        "cases": {"single_level": one, "three_level_amr": amr},
        "diagnostics_disabled_control": summarize_no_diagnostics_case(args.no_diagnostics_case)
        if args.no_diagnostics_case else None,
        "interpretation": (
            "After the application regrid callback synchronizes CURRENT_DATA and recomputes the lagged control-volume "
            "momentum, the one-level uniform-flow null force fell from -1.185 kN in the prior run to roundoff. "
            "In the three-level case, summed force also fell from the prior +4.435 kN anomaly to a small O(0.1 N) "
            "vector. This is evidence that the pre-regrid lagged integral caused the kilonewton startup transient in "
            "these tested cases. It is not a validated force result: the three-level flow remains badly nonuniform, "
            "global divergence is high, and the independently integrated momentum ledger has a large residual. "
            "Stage A therefore remains FAILED / UNVALIDATED."
        ),
        "next_gate": "Keep the callback refresh, then independently validate force and momentum closure and repair AMR uniform-flow preservation before any moving-wing refinement study.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
