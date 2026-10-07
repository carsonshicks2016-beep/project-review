#!/usr/bin/env python3
"""Print the protocol-owned acquisition matrix for faithful-v2 evidence."""
from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.faithful.evidence import REQUIRED_EVIDENCE_ROLES  # noqa: E402
from supra.faithful.schemas import EvidenceReadinessEvaluator  # noqa: E402


ROLE_GROUPS = {
    "vehicle.mass-properties": ("geometry", "mass_properties"),
    "vehicle.suspension-pitch-link": ("suspension",),
    "tyre.michelin-310-710-18": ("tyres",),
    "vehicle.aero-active-controls": ("aero",),
    "vehicle.ice-gearbox-driveline": ("ice", "gearbox"),
    "vehicle.mgu-battery-exhaust-ers": ("mgu", "battery", "thermal_limits"),
    "vehicle.brakes-embedded-controls": ("brakes", "embedded_controllers"),
}


def acquisition_matrix() -> dict[str, object]:
    groups = dict(EvidenceReadinessEvaluator.REQUIRED_PARAMETERS)
    semantics = EvidenceReadinessEvaluator.PARAMETER_SEMANTICS
    sensitive = EvidenceReadinessEvaluator.PROTOCOL_HIGH_SENSITIVITY
    roles = []
    for role in sorted(REQUIRED_EVIDENCE_ROLES):
        parameter_names = sorted(
            name for group in ROLE_GROUPS.get(role, ()) for name in groups[group]
        )
        if role == "track.nordschleife-june-2018-survey":
            purpose = "surveyed surface, boundaries, curbs, barriers, materials and timing planes"
        elif role == "calibration.component-spa":
            purpose = "component-rig and Spa calibration telemetry"
        elif role == "holdout.nordschleife-record-telemetry":
            purpose = "sealed Nord record telemetry opened only after parameter freeze"
        elif role == "scenario.record-conditions-preparation":
            purpose = "weather, fuel, SOC, tyre and brake preparation state"
            parameter_names = [item[1] for item in EvidenceReadinessEvaluator.SCENARIO_PARAMETERS]
        else:
            purpose = "licensed vehicle or tyre source data"
        roles.append({
            "role": role,
            "purpose": purpose,
            "parameter_groups": list(ROLE_GROUPS.get(role, ())),
            "parameters": [{
                "name": name,
                "unit": semantics[name][0] if name in semantics else "protocol-defined",
                "arity": semantics[name][1] if name in semantics else None,
                "high_sensitivity": name in sensitive,
            } for name in parameter_names],
            "required_confidence": (
                "licensed-measured-or-calibrated"
                if role != "scenario.record-conditions-preparation"
                else "licensed-measured"
            ),
            "adapter_status": "vendor-format-adapter-required",
        })
    return {
        "schema": "faithful-evidence-acquisition-matrix-v1",
        "roles": roles,
        "survey_tolerances": {
            "lap_length_error_m_max": 0.1,
            "boundary_error_m_max": 0.05,
            "vertical_error_m_max": 0.01,
            "bank_error_deg_max": 0.1,
        },
        "holdout_rule": EvidenceReadinessEvaluator.REQUIRED_HOLDOUT_POLICY,
        "validation_metrics": [
            {
                "artifact_kind": kind.value,
                "metrics": [
                    {"name": name, "comparison": comparison, "threshold": threshold}
                    for name, comparison, threshold in metrics
                ],
            }
            for kind, metrics in EvidenceReadinessEvaluator.VALIDATION_METRICS
        ],
    }


def main() -> None:
    print(json.dumps(acquisition_matrix(), sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
