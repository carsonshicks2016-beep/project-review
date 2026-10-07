#!/usr/bin/env python3
"""Summarize immutable CIB pilot run folders without changing their records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_constant(text: str, key: str, default: float | None = None) -> float | None:
    match = re.search(rf"(?m)^\s*{re.escape(key)}\s*=\s*([-+0-9.eE]+)\b", text)
    return float(match.group(1)) if match else default


def read_last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return rows[-1] if rows else None


def value(row: dict[str, str] | None, key: str) -> float | None:
    if not row or key not in row or row[key] == "":
        return None
    try:
        number = float(row[key])
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def vector_norm(row: dict[str, str] | None, prefix: str) -> float | None:
    parts = [value(row, f"{prefix}_{axis}") for axis in "xyz"]
    if any(part is None for part in parts):
        return None
    return math.sqrt(sum(part * part for part in parts if part is not None))


def solver_outcome(log_path: Path) -> str:
    if not log_path.is_file():
        return "MISSING_LOG"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"CIBStaggeredStokesSolver:\s*(converged|diverged):", text, re.IGNORECASE)
    return matches[-1].upper() if matches else "NO_TERMINAL_STATUS"


def summarize_run(run_dir: Path) -> dict[str, Any]:
    record_path = run_dir / "run_record.json"
    record = json.loads(record_path.read_text(encoding="utf-8")) if record_path.is_file() else {}
    input_path = run_dir / "cib_open_disk.input"
    input_text = input_path.read_text(encoding="utf-8", errors="replace") if input_path.is_file() else ""
    n = numeric_constant(input_text, "N")
    length = numeric_constant(input_text, "L")
    ratio = numeric_constant(input_text, "REF_RATIO")
    levels = numeric_constant(input_text, "MAX_LEVELS")
    finest_dx = length / (n * ratio ** (levels - 1.0)) if None not in (n, length, ratio, levels) else None
    radius = numeric_constant(input_text, "R")
    u_ref = numeric_constant(input_text, "U_REFERENCE", 1.0)

    force = read_last_csv_row(run_dir / "force_history.csv")
    budget = read_last_csv_row(run_dir / "momentum_budget.csv")
    outcome = solver_outcome(run_dir / "CIB3d.log")
    return_code = record.get("return_code")

    if outcome == "DIVERGED":
        interpretation = "FAILED_SOLVER_NONCONVERGENCE"
    elif return_code not in (None, 0):
        interpretation = "FAILED_RUNTIME_AFTER_SOLVER" if outcome == "CONVERGED" else "FAILED_RUNTIME"
    elif outcome == "CONVERGED":
        interpretation = "COMPLETED_DIAGNOSTIC_ONLY"
    else:
        interpretation = "UNQUALIFIED_SOLVER_STATUS"

    target_rms = value(force, "position_error_rms_code")
    target_max = value(force, "position_error_max_code")
    slip_rms = value(force, "newtime_interp_slip_rms_over_Uref")
    slip_max = value(force, "newtime_interp_slip_max_over_Uref")
    slip_metric = "new_time_eulerian_interpolation" if slip_rms is not None else None
    if slip_metric is None:
        slip_rms = value(force, "no_slip_rms_over_Uref")
        slip_max = value(force, "no_slip_max_over_Uref")
        if slip_rms is not None:
            slip_metric = "legacy_or_non_newtime_metric_do_not_treat_as_current"

    divergence_rms = value(force, "divergence_rms_code")
    divergence_max = value(force, "divergence_max_code")
    normalized_div_rms = divergence_rms * radius / u_ref if None not in (divergence_rms, radius, u_ref) else None
    normalized_div_max = divergence_max * radius / u_ref if None not in (divergence_max, radius, u_ref) else None
    position_rms_dx = target_rms / finest_dx if None not in (target_rms, finest_dx) else None
    position_max_dx = target_max / finest_dx if None not in (target_max, finest_dx) else None

    eulerian_explicit_integral = None
    force_column_interpretation = None
    if budget and all(f"explicit_ib_body_force_{axis}" in budget for axis in "xyz"):
        eulerian_explicit_integral = {
            axis: value(budget, f"explicit_ib_body_force_{axis}") for axis in "xyz"
        }
        force_column_interpretation = "explicit_IB_body_force_field_not_the_CIB_constraint_multiplier_spread"
    elif budget and all(f"eulerian_ib_force_{axis}" in budget for axis in "xyz"):
        # This schema predates the column-name correction. The recorded capture
        # was of the ordinary explicit IB body-force field, not CIB's implicit
        # saddle-point multiplier force.
        eulerian_explicit_integral = {axis: value(budget, f"eulerian_ib_force_{axis}") for axis in "xyz"}
        force_column_interpretation = "legacy_misleading_column_name_corrected_by_interpretation"

    return {
        "run": run_dir.name,
        "run_record_status": record.get("status"),
        "scientific_interpretation": interpretation,
        "solver_terminal_status_from_CIB3d_log": outcome,
        "process_return_code": return_code,
        "run_record": record_path.name if record_path.is_file() else None,
        "input_sha256": record.get("input_sha256") or sha256(input_path),
        "binary_sha256": record.get("binary_sha256"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "peak_process_tree_rss_gib": record.get("peak_process_tree_rss_gib"),
        "minimum_observed_free_disk_gib": record.get("minimum_observed_free_disk_gib"),
        "mesh": {
            "N": n,
            "L": length,
            "refinement_ratio": ratio,
            "levels": levels,
            "finest_dx_code_units": finest_dx,
        },
        "last_force_history_row": {
            "position_error_rms_dx": position_rms_dx,
            "position_error_max_dx": position_max_dx,
            "target_tracking_gate_for_recorded_interval": (
                "PASS" if position_rms_dx is not None and position_max_dx is not None
                and position_rms_dx <= 0.02 and position_max_dx <= 0.10 else "NOT_PASSED_OR_NOT_MEASURED"
            ),
            "velocity_metric": slip_metric,
            "slip_rms_over_Uref": slip_rms,
            "slip_max_over_Uref": slip_max,
            "slip_gate_for_recorded_interval": (
                "PASS" if slip_rms is not None and slip_max is not None
                and slip_metric == "new_time_eulerian_interpolation"
                and slip_rms <= 0.05 and slip_max <= 0.20 else "NOT_PASSED_OR_NOT_MEASURED"
            ),
            "slip_gate_interpretation": (
                "assessed with the actual new-time Eulerian field"
                if slip_metric == "new_time_eulerian_interpolation" else
                "legacy/stale velocity sample: do not treat its apparent threshold pass as a coupling result"
                if slip_metric else "no slip metric recorded"
            ),
            "divergence_rms_times_R_over_Uref": normalized_div_rms,
            "divergence_max_times_R_over_Uref": normalized_div_max,
            "divergence_gate_for_recorded_interval": (
                "PASS" if normalized_div_rms is not None and normalized_div_max is not None
                and normalized_div_rms <= 1.0e-4 and normalized_div_max <= 1.0e-3 else "NOT_PASSED_OR_NOT_MEASURED"
            ),
            "lambda_resultant_code_force": {
                axis: value(force, f"lambda_force_on_fluid_{axis}") for axis in "xyz"
            } if force else None,
        },
        "last_momentum_budget_row": {
            "signed_residual_code_force_vector_norm": vector_norm(budget, "residual"),
            "term_magnitude_normalized_residual": value(budget, "normalized_residual"),
            "lambda_on_fluid_code_force": {
                axis: value(budget, f"lambda_force_on_fluid_{axis}") for axis in "xyz"
            } if budget else None,
            "ordinary_explicit_IB_body_force_integral_code_force": eulerian_explicit_integral,
            "explicit_force_column_interpretation": force_column_interpretation,
            "interpretation_of_explicit_field": (
                "Separate explicit-force channel; does not capture the CIB saddle-point multiplier or verify its spread."
                if eulerian_explicit_integral is not None else None
            ),
        } if budget else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.study_dir.resolve()
    output = (args.output or root / "study_summary.json").resolve()
    runs = [
        summarize_run(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name != "build" and (path / "run_record.json").is_file()
    ]

    primary = next((run for run in runs if run["run"] == "matched_translation_1step_newtime_sample_v2"), None)
    force_probe = next((run for run in runs if run["run"] == "matched_translation_force_integral_v3"), None)
    summary = {
        "study": "cib_open_disk_method_pilot",
        "status": "DIAGNOSTIC_ONLY_NOT_PASSED",
        "ibamr_version": "0.19.0",
        "interpretation": (
            "The corrected one-step matched-translation sample satisfies the recorded target, new-time slip, and "
            "divergence scales for that single interval. Its momentum-budget residual is 0.3309 under the current "
            "term-magnitude normalization, so the ledger and force interpretation remain unresolved. The later "
            "explicit IB body-force field capture is a separate channel and does not measure the CIB saddle-point "
            "multiplier spread. No extension or crossflow result is authorized by this summary."
        ),
        "stage_a_status": "FAILED / UNVALIDATED",
        "primary_matched_translation_run": primary,
        "explicit_force_field_capture_run": force_probe,
        "runs": runs,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "run_count": len(runs), "status": summary["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
