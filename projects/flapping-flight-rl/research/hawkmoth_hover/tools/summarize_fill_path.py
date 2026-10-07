#!/usr/bin/env python3
"""Summarize the one-step IBAMR lagged-momentum fill-path diagnosis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_dir", type=Path)
    args = parser.parse_args()
    case_dir = args.case_dir.resolve()
    audit_path = case_dir / "control_volume_face_audit.csv"
    stress_path = case_dir / "control_volume_stress.csv"
    force_path = case_dir / "force_history.csv"
    layout_path = case_dir / "hierarchy_patch_layout.csv"
    resource_path = case_dir / "resource_record.json"
    run_manifest_path = case_dir / "run_manifest.json"
    audit = csv_rows(audit_path)

    def momentum(stage: str, sid: str = "0", component: str = "0") -> float:
        rows = [row for row in audit if row["stage"] == stage and row["structure_id"] == sid
                and row["component"] == component]
        if len(rows) != 1:
            raise ValueError(f"expected one {stage}/{sid}/{component} audit row, found {len(rows)}")
        return float(rows[0]["total_momentum_kg_m_s"])

    stress = csv_rows(stress_path)
    force = csv_rows(force_path)
    layout = csv_rows(layout_path)
    momentum_trace = csv_rows(case_dir / "control_volume_momentum_trace.csv")
    resource = json.loads(resource_path.read_text())
    manifest = json.loads(run_manifest_path.read_text())
    input_deck = (case_dir / "input3d").read_text()
    rho = next(float(line.split("=", 1)[1].strip()) for line in input_deck.splitlines()
               if line.strip().startswith("RHO ="))
    velocity = float(manifest["translation_velocity_m_s"][0])
    bounds = stress[0]
    lengths = [float(bounds[f"box_upper_current_{axis}_m"]) - float(bounds[f"box_lower_current_{axis}_m"])
               for axis in "xyz"]
    analytic = rho * velocity
    for length in lengths:
        analytic *= length
    stages = {
        "old_copy_only": momentum("ibamr_copy_only_replica"),
        "old_fill_at_current_time": momentum("ibamr_fill_path_replica"),
        "old_field_refilled_at_new_time": momentum("ibamr_preadvance_newtime_fill"),
        "new_copy_only": momentum("ibamr_new_copy_only_replica"),
        "new_fill_at_new_time": momentum("ibamr_new_fill_path_replica"),
        "evaluator_old_integral": float(stress[0]["P_box_current_x_kg_m_s"]),
        "evaluator_new_integral": float(stress[0]["P_box_new_x_kg_m_s"]),
        "analytic_uniform_integral": analytic,
    }
    force_n = [float(force[0][f"F{axis}_N"]) for axis in "xyz"]
    layout_by_stage = {
        stage: sorted(
            tuple(row[key] for key in ("level", "patch_id", "x_lower", "x_upper", "y_lower", "y_upper",
                                       "z_lower", "z_upper"))
            for row in layout if row["stage"] == stage
        )
        for stage in ("before_lagged_integral", "after_advance")
    }
    post_regrid = [row for row in momentum_trace if row["stage"] == "post_regrid_lagged_integral"]
    independent_surface = [
        sum(float(row[f"coarse_surface_total_{axis}_N"]) for row in stress) for axis in "xyz"
    ]
    result = {
        "study": "IBAMR lagged-momentum fill-path isolation",
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "stage_a_gate": "FAILED / UNVALIDATED",
        "scope": "one timestep, single AMR level, zero IB force spreading; no convergence or moth-physics claim",
        "case_dir": str(case_dir),
        "solver_status": resource["status"],
        "elapsed_seconds": resource["elapsed_seconds"],
        "peak_process_tree_rss_gb": resource["peak_process_tree_rss_gb"],
        "free_disk_bytes_after": resource["free_disk_bytes_after"],
        "staggered_x_momentum_kg_m_s": stages,
        "post_regrid_refreshed_integrals_by_structure": [
            {"structure_id": int(row["structure_id"]),
             "momentum_kg_m_s": [float(row[f"P_composite_{axis}_kg_m_s"]) for axis in "xyz"]}
            for row in post_regrid
        ],
        "level_zero_patch_layout_unchanged_across_advance":
            layout_by_stage["before_lagged_integral"] == layout_by_stage["after_advance"],
        "patch_boxes": {
            stage: layout_by_stage[stage] for stage in layout_by_stage
        },
        "fill_path_relative_changes_percent": {
            "old_fill_vs_copy": 100.0 * (stages["old_fill_at_current_time"] - stages["old_copy_only"])
            / stages["old_copy_only"],
            "old_refilled_at_new_time_vs_copy": 100.0 * (stages["old_field_refilled_at_new_time"]
            - stages["old_copy_only"]) / stages["old_copy_only"],
            "new_fill_vs_copy": 100.0 * (stages["new_fill_at_new_time"] - stages["new_copy_only"])
            / stages["new_copy_only"],
            "evaluator_old_vs_analytic": 100.0 * (stages["evaluator_old_integral"] - analytic) / analytic,
            "evaluator_new_vs_analytic": 100.0 * (stages["evaluator_new_integral"] - analytic) / analytic,
        },
        "measured_force_N": force_n,
        "momentum_budget_normalized_residual": (
            float(csv_rows(case_dir / "momentum_budget.csv")[0]["normalized_residual"])
            if (case_dir / "momentum_budget.csv").exists() else None
        ),
        "post_advance_global_divergence_rms_s_inv": float(force[0]["divergence_rms_s_inv"]),
        "post_advance_global_divergence_max_s_inv": float(force[0]["divergence_max_s_inv"]),
        "independent_level_zero_surface_stress_flux_N": independent_surface,
        "interpretation": (
            "The source field copy has the expected uniform-flow CV momentum before ghost filling. "
            "IBAMR's lagged (old-state) velocity copy followed by its velocity ghost fill loses the two "
            "x-normal CV-edge contributions, and repeating that fill at the next timestamp does not restore them. "
            "The corresponding post-advance copy-plus-fill retains them, and the evaluator's new momentum matches "
            "the independent composite integral. This isolates the old/new discrepancy to the lagged-state fill "
            "path in this one-level control. The recorded level-zero patch boxes are unchanged across the advance, "
            "so box repartitioning is not the explanation in this case; why identical interpolation changes the "
            "old-state samples but not the post-advance samples remains unresolved. This does not establish a general "
            "IBAMR defect. The app-level callback recomputes lagged momentum after regrid synchronization. "
            f"It recorded {force_n[0]:.6g} N total x force in this run, versus -1,185.324 N in the same one-level "
            "control before the callback refresh. This corrects the startup force transient in this one case, "
            "but is not a Stage A pass; AMR moving-boundary and independent momentum-closure gates remain failed."
        ),
        "excluded_crash_probes": [
            "stage_a_fill_path_final: scratch pressure-fill experiment exited with SIGSEGV; partial files retained",
            "stage_a_fill_path_final2: scratch pressure-fill experiment exited with SIGSEGV; partial files retained",
        ],
        "artifact_sha256": {
            "input3d": sha256(case_dir / "input3d"),
            "run_manifest": sha256(run_manifest_path),
            "resource_record": sha256(resource_path),
            "patch_layout": sha256(layout_path),
            "executable": resource["run_identity"]["executable_sha256_at_launch"],
            "application_source": resource["run_identity"]["application_source_sha256"][
                "research/hawkmoth_hover/src/main.cpp"],
        },
    }
    output = case_dir.parent.parent / "fill_path_diagnosis_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
