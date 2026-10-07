#!/usr/bin/env python3
"""Summarize the independently integrated outer-boundary momentum ledger."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


VECTOR_TERMS = (
    "dP_fluid_dt",
    "force_on_body",
    "pressure_traction",
    "advective_momentum_flux",
    "viscous_traction",
    "signed_residual",
)


def vector(row: dict[str, str], term: str) -> list[float]:
    return [float(row[f"{term}_{axis}_N"]) for axis in "xyz"]


def norm(value: list[float]) -> float:
    return sum(component * component for component in value) ** 0.5


def summarize(path: Path) -> dict:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    steps = []
    for row in rows:
        terms = {name: vector(row, name) for name in VECTOR_TERMS}
        scale = sum(norm(terms[name]) for name in VECTOR_TERMS[:5])
        refined_boundary = bool(int(row["refined_physical_boundary"]))
        steps.append({
            "time_s": float(row["time_s"]),
            "dt_s": float(row["dt_s"]),
            "dP_fluid_dt_N": terms["dP_fluid_dt"],
            "hydrodynamic_force_on_body_N": terms["force_on_body"],
            "pressure_traction_N": terms["pressure_traction"],
            "advective_momentum_flux_N": terms["advective_momentum_flux"],
            "viscous_traction_N": terms["viscous_traction"],
            "signed_residual_N": terms["signed_residual"],
            "residual_magnitude_N": norm(terms["signed_residual"]),
            "normalization_sum_term_magnitudes_N": scale,
            "normalized_residual": norm(terms["signed_residual"]) / scale if scale else 0.0,
            "outer_boundary_refined": refined_boundary,
        "usable": True,
        })
    refined = any(step["outer_boundary_refined"] for step in steps)
    return {
        "source_csv": str(path),
        "budget_equation": "dP_fluid/dt + F_hydrodynamic_on_body - (pressure_traction + advective_momentum_flux + viscous_traction) = 0",
        "sign_convention": "pressure/viscous terms are outward stress traction; advective term is -rho*u*(u dot n); body force is positive on the immersed body",
        "outer_terms_independently_integrated": True,
        "outer_boundary_refined": refined,
        "gate": "MEASURED_DIAGNOSTIC_WITH_AMR_BOUNDARY" if refined and steps else "MEASURED_DIAGNOSTIC" if steps else "NO_STEPS",
        "interpretation": "The three outer-boundary terms are integrated directly from the physical-boundary MAC fields on each AMR level; covered coarse boundary cells are omitted. The force comes from IBAMR's moving-control-volume evaluator; this ledger is independent of that evaluator's aggregate outer-surface term but shares its velocity and pressure solution.",
        "steps": steps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("momentum_budget_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.momentum_budget_csv)
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
