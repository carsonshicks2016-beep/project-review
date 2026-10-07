#!/usr/bin/env python3
"""Summarize the no-IB-force uniform-flow isolation against its baseline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


def read_one_csv(path: Path) -> dict[str, float]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError(f"expected one diagnostic step in {path}, got {len(rows)}")
    return {key: float(value) for key, value in rows[0].items() if value != ""}


def summarize(study_root: Path, baseline_dir: Path) -> dict:
    study_root = study_root.resolve()
    case_dir = study_root / "prepared" / "stationary_uniform_coarse_dt_coarse"
    study_manifest_path = study_root / "force_isolation_manifest.json"
    study_manifest = json.loads(study_manifest_path.read_text()) if study_manifest_path.exists() else {}
    baseline_manifest_path = baseline_dir / "run_manifest.json"
    baseline_manifest = json.loads(baseline_manifest_path.read_text()) if baseline_manifest_path.exists() else {}
    force_off = read_one_csv(case_dir / "force_history.csv")
    baseline = read_one_csv(baseline_dir / "force_history.csv")
    budget = read_one_csv(case_dir / "momentum_budget.csv")
    resource = json.loads((case_dir / "resource_record.json").read_text())
    flow_path = case_dir / "flow_state_summary.csv"
    flow_states = []
    if flow_path.exists():
        with flow_path.open(newline="") as stream:
            flow_states = list(csv.DictReader(stream))
    surface_path = case_dir / "control_volume_stress.csv"
    surface_fluxes = []
    if surface_path.exists():
        with surface_path.open(newline="") as stream:
            surface_fluxes = list(csv.DictReader(stream))
    momentum_trace = []
    momentum_trace_path = case_dir / "control_volume_momentum_trace.csv"
    if momentum_trace_path.exists():
        with momentum_trace_path.open(newline="") as stream:
            momentum_trace = [
                {key: (value if key == "stage" else int(value) if key == "structure_id" else float(value))
                 for key, value in row.items()}
                for row in csv.DictReader(stream)
            ]
    raw_cv_momentum = []
    if surface_fluxes and "P_box_current_x_kg_m_s" in surface_fluxes[0]:
        run_manifest = json.loads((case_dir / "run_manifest.json").read_text())
        deck = (case_dir / "input3d").read_text()
        rho_match = re.search(r"^RHO\s*=\s*([0-9.eE+-]+)\s*$", deck, re.MULTILINE)
        rho = float(rho_match.group(1)) if rho_match else None
        uref = abs(float(run_manifest.get("translation_velocity_m_s", [0.0])[0]))
        for row in surface_fluxes:
            lo = [float(row[f"box_lower_current_{axis}_m"]) for axis in "xyz"]
            hi = [float(row[f"box_upper_current_{axis}_m"]) for axis in "xyz"]
            expected = rho * uref if rho is not None else None
            if expected is not None:
                for a, b in zip(lo, hi):
                    expected *= b - a
            current = [float(row[f"P_box_current_{axis}_kg_m_s"]) for axis in "xyz"]
            new = [float(row[f"P_box_new_{axis}_kg_m_s"]) for axis in "xyz"]
            raw_cv_momentum.append({
                "structure_id": int(row["structure_id"]),
                "P_box_current_kg_m_s": current,
                "P_box_new_kg_m_s": new,
                "P_box_rate_N": [float(row[f"box_momentum_rate_{axis}_N"]) for axis in "xyz"],
                "analytic_uniform_x_momentum_kg_m_s": expected,
                "current_x_momentum_error_percent": 100.0 * (current[0] - expected) / expected
                if expected is not None and expected != 0.0 else None,
                "new_x_momentum_error_percent": 100.0 * (new[0] - expected) / expected
                if expected is not None and expected != 0.0 else None,
                "CV_bounds_current_m": {"lower": lo, "upper": hi},
            })
    flow_by_stage = {
        row["stage"]: {key: float(value) for key, value in row.items() if key != "stage"}
        for row in flow_states
    }
    baseline_flow = {}
    baseline_flow_path = baseline_dir / "flow_state_summary.csv"
    if baseline_flow_path.exists():
        with baseline_flow_path.open(newline="") as stream:
            baseline_flow = {row["stage"]: row for row in csv.DictReader(stream)}

    def state_metrics(row: dict[str, str]) -> dict[str, float]:
        return {key: float(row[key]) for key in (
            "mean_u_x_m_s", "rms_u_y_m_s", "rms_u_z_m_s", "rms_pressure_Pa",
            "divergence_rms_s_inv", "divergence_max_s_inv")}
    composite_trace_by_key = {(row["stage"], row["structure_id"]): row for row in momentum_trace}
    for row in raw_cv_momentum:
        sid = row["structure_id"]
        old_trace = composite_trace_by_key.get(("pre_advance_before_lagged_integral", sid))
        new_trace = composite_trace_by_key.get(("post_advance_before_force_evaluation", sid))
        if old_trace:
            row["independent_pre_advance_momentum_kg_m_s"] = [
                old_trace[f"P_composite_{axis}_kg_m_s"] for axis in "xyz"]
            row["old_integral_vs_composite_x_error_percent"] = (
                100.0 * (row["P_box_current_kg_m_s"][0] - old_trace["P_composite_x_kg_m_s"])
                / old_trace["P_composite_x_kg_m_s"] if old_trace["P_composite_x_kg_m_s"] else None)
        if new_trace:
            row["independent_post_advance_momentum_kg_m_s"] = [
                new_trace[f"P_composite_{axis}_kg_m_s"] for axis in "xyz"]
            row["new_integral_vs_composite_x_error_percent"] = (
                100.0 * (row["P_box_new_kg_m_s"][0] - new_trace["P_composite_x_kg_m_s"])
                / new_trace["P_composite_x_kg_m_s"] if new_trace["P_composite_x_kg_m_s"] else None)
    def sum_surface_vectors(prefix: str) -> list[float]:
        return [sum(float(row[f"{prefix}_{axis}_N"]) for row in surface_fluxes) for axis in "xyz"]
    baseline_fx = baseline["Fx_N"]
    force_off_fx = force_off["Fx_N"]
    relative_force_change = abs(force_off_fx - baseline_fx) / max(abs(baseline_fx), 1.0e-30)
    stdout = (case_dir / "solver.stdout.log").read_text(errors="replace")
    stderr = (case_dir / "solver.stderr.log").read_text(errors="replace")
    solver_log = (case_dir / "hawkmoth_hover.log").read_text(errors="replace")
    residuals = [float(value) for value in re.findall(r"stokes solve residual norm\s*=\s*([+\-0-9.eE]+)", solver_log)]
    result = {
        "study": "Stage A no-immersed-boundary-forcing uniform-flow isolation",
        "status": "DIAGNOSTIC_ONLY",
        "stage_a_gate": "FAILED / UNVALIDATED",
        "scope": "one timestep on a base-grid-only hierarchy compared with the prior three-level AMR control; no convergence, periodicity, or moth result",
        "question": "Does the first-step force/divergence anomaly persist with zero Lagrangian force spreading?",
        "force_disabled_definition": "TARGET_STIFFNESS=0 and no IBStandardForceGen callback; the IB force vector is therefore initialized to zero by IBMethod",
        "baseline_run": str(baseline_dir.resolve()),
        "amr_levels": study_manifest.get("amr_levels"),
        "baseline_amr_levels": baseline_manifest.get("max_levels"),
        "amr_comparison": {
            "single_level_post_regrid": state_metrics(flow_by_stage["post_regrid_projection_and_sync"])
            if "post_regrid_projection_and_sync" in flow_by_stage else None,
            "single_level_post_advance": state_metrics(flow_by_stage["post_advance"])
            if "post_advance" in flow_by_stage else None,
            "amr_post_regrid": state_metrics(baseline_flow["post_regrid_projection_and_sync"])
            if "post_regrid_projection_and_sync" in baseline_flow else None,
            "amr_post_advance": state_metrics(baseline_flow["post_advance"])
            if "post_advance" in baseline_flow else None,
            "single_level_uniform_state_preserved_through_regrid": (
                abs(float(flow_by_stage["post_regrid_projection_and_sync"]["mean_u_x_m_s"]) - 10.0) < 1e-8
                and float(flow_by_stage["post_regrid_projection_and_sync"]["divergence_rms_s_inv"]) < 1e-8
            ) if "post_regrid_projection_and_sync" in flow_by_stage else None,
        },
        "force_disabled_run": str(case_dir),
        "baseline_force_N": [baseline[f"F{axis}_N"] for axis in "xyz"],
        "force_disabled_force_N": [force_off[f"F{axis}_N"] for axis in "xyz"],
        "relative_x_force_change": relative_force_change,
        "force_persists_without_ib_forcing": relative_force_change < 0.01,
        "force_disabled_diagnostics": {
            "target_error_rms_m": force_off["target_error_rms_m"],
            "target_error_max_m": force_off["target_error_max_m"],
            "no_slip_rms_over_Uref": force_off["no_slip_velocity_rms_over_Uref"],
            "no_slip_max_over_Uref": force_off["no_slip_velocity_max_over_Uref"],
            "divergence_rms_s_inv": force_off["divergence_rms_s_inv"],
            "divergence_max_s_inv": force_off["divergence_max_s_inv"],
            "surface_band_divergence_rms_s_inv": force_off["divergence_surface_band_rms_s_inv"],
            "surface_band_divergence_max_s_inv": force_off["divergence_surface_band_max_s_inv"],
        },
        "flow_state_stages": [
            {key: (value if key == "stage" else float(value)) for key, value in row.items()}
            for row in flow_states
        ],
        "control_surface_stress_comparison": [
            {key: (int(value) if key == "structure_id" else float(value)) for key, value in row.items()}
            for row in surface_fluxes
        ],
        "control_volume_raw_momentum_terms": raw_cv_momentum,
        "independent_composite_cv_momentum_trace": momentum_trace,
        "surface_integral_method": (
            "Independent closed-surface pressure, advective momentum-flux, and viscous-traction integral on level-zero faces. "
            "It uses the same rectangular CV bounds but does not reuse IBAMR's AMR face weights; this is diagnostic, not grid converged."
        ),
        "force_balance_decomposition_N": {
            "ibamr_force_sum": sum_surface_vectors("ibamr_force"),
            "control_volume_momentum_rate_sum": sum_surface_vectors("box_momentum_rate"),
            "ibamr_implied_surface_flux_sum": sum_surface_vectors("ibamr_implied_surface_flux"),
            "independent_level_zero_surface_flux_sum": sum_surface_vectors("coarse_surface_total"),
            "independent_minus_ibamr_surface_flux_sum": [
                sum(float(row[f"coarse_minus_ibamr_{axis}_N"]) for row in surface_fluxes) for axis in "xyz"
            ],
        },
        "momentum_ledger": {
            "fluid_momentum_rate_N": [budget[f"dP_fluid_dt_{axis}_N"] for axis in "xyz"],
            "measured_force_on_body_N": [budget[f"force_on_body_{axis}_N"] for axis in "xyz"],
            "pressure_traction_N": [budget[f"pressure_traction_{axis}_N"] for axis in "xyz"],
            "advective_flux_N": [budget[f"advective_momentum_flux_{axis}_N"] for axis in "xyz"],
            "viscous_traction_N": [budget[f"viscous_traction_{axis}_N"] for axis in "xyz"],
            "normalized_residual": budget["normalized_residual"],
        },
        "solver_stokes_residual_norms": residuals,
        "resource_record": resource,
        "source_logs_nonempty": {"stdout": bool(stdout.strip()), "stderr": bool(stderr.strip())},
        "interpretation": (
            "With one AMR level, the initialized uniform 10 m/s field remains unchanged through the regrid callback and fluid advance, "
            "with zero measured divergence. The prior three-level no-force run lost uniformity and developed divergence during the "
            "regrid/projection/synchronization path. This localizes the observed field corruption to behavior introduced by the refined "
            "hierarchy or its regridding/projection path, but does not identify the internal cause. The force evaluator remains suspect: "
            "it reports -1.185 kN even though the no-force fluid field is unchanged, while the prior three-level case reported +4.435 kN. "
            "Those force values are diagnostic artifacts until the force/control-volume measurement is independently understood. "
            "The one-level case is not a pass: its dense Lagrangian point cloud was retained while its Eulerian hierarchy was coarsened, "
            "and no-slip error is not meaningful with IB forcing disabled. Stage A remains failed."
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study_root", type=Path)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.study_root, args.baseline)
    output = args.study_root / "force_isolation_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
