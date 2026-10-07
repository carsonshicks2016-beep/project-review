#!/usr/bin/env python3
"""Hermetic validation gate for the faithful-v2 schema/backend foundation."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.faithful import (  # noqa: E402
    CarAssetRegistry,
    Confidence,
    DriverActionV2,
    EvidenceReadinessEvaluator,
    FIXED_TIMESTEP_S,
    MJXTrainingTwin,
    MuJoCoAuthority,
    PhysicsValidationEvidenceV1,
    ShiftRequest,
    TruthfulAvAdapter,
    TrackSampleV2,
    TrackSurveyValidationV1,
    ValidationArtifactKind,
    ValidationArtifactV1,
    ValidationMetricV1,
    content_sha256,
    default_record_scenario,
    official_919evo_vehicle_spec,
    training_fallback_track,
)


FAILED: list[str] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def raises(exception_type: type[BaseException], fn) -> bool:
    try:
        fn()
    except exception_type:
        return True
    return False


def main() -> None:
    vehicle = official_919evo_vehicle_spec()
    track = training_fallback_track()
    scenario = default_record_scenario(vehicle, track)

    print("== immutable evidence schemas ==")
    gate("vehicle schema versioned", vehicle.schema_version == "vehicle-spec-v2")
    gate("track schema versioned", track.schema_version == "track-surface-v2")
    gate("scenario schema versioned", scenario.schema_version == "record-scenario-v1")
    gate("schemas are frozen", raises(FrozenInstanceError, lambda: setattr(vehicle, "vehicle_id", "changed")))
    gate("hashes deterministic", vehicle.sha256 == content_sha256(vehicle))
    gate("scenario binds exact vehicle hash", scenario.vehicle_spec_sha256 == vehicle.sha256)
    gate("scenario binds exact track hash", scenario.track_surface_sha256 == track.sha256)

    expected_anchors = {
        "body_length_m": 5.078,
        "body_width_m": 1.900,
        "body_height_m": 1.050,
        "vehicle_mass_kg": 849.0,
        "driver_ballast_mass_kg": 39.0,
        "ice_max_power_w": 720.0 * 735.49875,
        "front_mgu_max_power_w": 440.0 * 735.49875,
        "ice_redline_rpm": 9000.0,
        "forward_gear_count": 7.0,
    }
    for name, expected in expected_anchors.items():
        parameter = vehicle.parameter(name)
        gate(
            f"official anchor {name}",
            math.isclose(parameter.scalar(), expected, rel_tol=0.0, abs_tol=1e-9)
            and parameter.evidence.confidence.value == "official",
        )
    gate("record target exact", scenario.target_lap_time_s == 319.546)
    gate("boost request disabled by default", not scenario.controls.boost_request)
    gate("DRS request disabled by default", not scenario.controls.drs_request)

    print("== fail-closed evidence readiness ==")
    readiness = EvidenceReadinessEvaluator.evaluate(vehicle, track, scenario)
    gate("public bootstrap cannot claim faithful", not readiness.faithful_claim_allowed)
    gate("public bootstrap not physics validated", not readiness.physics_validated)
    gate("capability explicitly approximate", readiness.capability_label == "telemetry-constrained approximation")
    gate(
        "unsigned caller evidence can never open readiness",
        any("signed evidence freeze" in item for item in readiness.blockers),
    )
    gate("missing survey is a blocker", any("track is training-only" in item for item in readiness.blockers))
    gate("inferred sensitive values identified", len(readiness.high_sensitivity_inferred) > 20)
    neutralized_groups = {
        group_name: tuple(replace(parameter, high_sensitivity=False) for parameter in parameters)
        for group_name, parameters in vehicle.groups()
    }
    neutralized = replace(vehicle, **neutralized_groups)
    neutralized_scenario = replace(scenario, vehicle_spec_sha256=neutralized.sha256)
    neutralized_readiness = EvidenceReadinessEvaluator.evaluate(
        neutralized, track, neutralized_scenario
    )
    gate(
        "caller cannot clear protocol sensitivity flags",
        "tyre_peak_mu" in neutralized_readiness.high_sensitivity_inferred,
    )
    sparse_vehicle = replace(vehicle, tyres=(vehicle.parameter("front_tyre_size_mm"),))
    sparse_scenario = replace(scenario, vehicle_spec_sha256=sparse_vehicle.sha256)
    gate(
        "dummy required group cannot replace named protocol parameters",
        any(
            "missing protocol parameter tyres.tyre_peak_mu" in item
            for item in EvidenceReadinessEvaluator.evaluate(
                sparse_vehicle, track, sparse_scenario
            ).blockers
        ),
    )

    validator_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    captured_evidence = replace(
        track.survey_evidence,
        source_uri=Path(__file__).resolve().as_uri(),
        source_sha256=validator_digest,
        digest_scope="content",
        confidence=Confidence.LICENSED_MEASURED,
        license_id="validator-fixture",
        calibration_dataset="validator-fixture",
    )
    sparse_track = replace(
        track,
        lap_length=replace(track.lap_length, evidence=captured_evidence),
        survey_evidence=captured_evidence,
        samples=(
            TrackSampleV2(0.0, (0.0, 0.0, 0.0), (0.0, 5.0, 0.0), (0.0, -5.0, 0.0), "asphalt", "main"),
            TrackSampleV2(20832.0, (20832.0, 0.0, 0.0), (20832.0, 5.0, 0.0), (20832.0, -5.0, 0.0), "asphalt", "main"),
        ),
        curbs_sha256=validator_digest,
        barriers_sha256=validator_digest,
        materials_sha256=validator_digest,
        certification_eligible=True,
        approximation_reason=None,
        survey_validation=TrackSurveyValidationV1(
            protocol_version="track-survey-validation-v1",
            passed=True,
            artifact_uri=Path(__file__).resolve().as_uri(),
            artifact_sha256=validator_digest,
            verifier_signature_sha256="2" * 64,
            evidence=captured_evidence,
            lap_length_error_m=0.0,
            max_boundary_error_m=0.0,
            max_vertical_error_m=0.0,
            max_bank_error_deg=0.0,
        ),
    )
    sparse_track_scenario = replace(scenario, track_surface_sha256=sparse_track.sha256)
    gate(
        "two-point track cannot become a certification survey",
        any(
            "sample density" in item
            for item in EvidenceReadinessEvaluator.evaluate(
                vehicle, sparse_track, sparse_track_scenario
            ).blockers
        ),
    )

    metric_values = {
        ValidationArtifactKind.COMPONENT: (("all_within_measurement_uncertainty", 1.0, "bool"),),
        ValidationArtifactKind.SPA: (("lap_time_error_fraction", 0.0, "1"), ("speed_error_fraction", 0.0, "1"), ("lateral_accel_error_g", 0.0, "g")),
        ValidationArtifactKind.NORD_HOLDOUT: (("lap_time_error_fraction", 0.0, "1"), ("speed_trace_nrmse", 0.0, "1"), ("peak_speed_error_fraction", 0.0, "1"), ("average_speed_error_fraction", 0.0, "1")),
        ValidationArtifactKind.NUMERICS: (("half_ms_lap_delta_s", 0.0, "s"), ("mjx_force_error_fraction", 0.0, "1"), ("mjx_speed_error_mps", 0.0, "m/s"), ("mjx_position_error_m", 0.0, "m")),
        ValidationArtifactKind.ENERGY: (("full_lap_residual_fraction", 0.0, "1"),),
    }
    artifacts = {
        kind: ValidationArtifactV1(
            kind=kind,
            protocol_version="physics-validation-v1",
            passed=True,
            artifact_uri=f"/definitely/not/materialized/{kind.value}.json",
            artifact_sha256="3" * 64,
            verifier_signature_sha256="4" * 64,
            source_evidence=captured_evidence,
            metrics=tuple(ValidationMetricV1(*metric) for metric in metrics),
        )
        for kind, metrics in metric_values.items()
    }
    declared_only_validation = PhysicsValidationEvidenceV1(
        component=artifacts[ValidationArtifactKind.COMPONENT],
        spa=artifacts[ValidationArtifactKind.SPA],
        nord_holdout=artifacts[ValidationArtifactKind.NORD_HOLDOUT],
        numerics=artifacts[ValidationArtifactKind.NUMERICS],
        energy=artifacts[ValidationArtifactKind.ENERGY],
        freeze_manifest_sha256="5" * 64,
        parameters_frozen_before_holdout=True,
    )
    declared_only_scenario = replace(scenario, physics_validation=declared_only_validation)
    gate(
        "dummy validation hashes without materialized artifacts cannot pass",
        any(
            "not materialized" in item
            for item in EvidenceReadinessEvaluator.evaluate(
                vehicle, track, declared_only_scenario
            ).blockers
        ),
    )
    gate(
        "opening Nord holdout before freeze blocks readiness",
        any(
            "holdout telemetry" in item
            for item in EvidenceReadinessEvaluator.evaluate(
                vehicle, track, replace(scenario, record_telemetry_opened=True)
            ).blockers
        ),
    )
    forbidden_boost = DriverActionV2(boost_request=True)
    forbidden_drs = DriverActionV2(drs_request=True)
    gate("unknown boost control rejected", raises(PermissionError, lambda: forbidden_boost.assert_permitted(scenario.controls)))
    gate("unknown DRS control rejected", raises(PermissionError, lambda: forbidden_drs.assert_permitted(scenario.controls)))
    gate(
        "truthy string cannot certify a track",
        raises(TypeError, lambda: replace(track, certification_eligible="false")),
    )
    gate(
        "truthy string cannot pass validation evidence",
        raises(TypeError, lambda: replace(
            artifacts[ValidationArtifactKind.COMPONENT], passed="false"
        )),
    )
    gate(
        "truthy string cannot enable an undocumented action",
        raises(TypeError, lambda: DriverActionV2(boost_request="false")),
    )
    boost_scenario = replace(
        scenario,
        controls=replace(scenario.controls, boost_request=True),
    )
    gate(
        "enabled boost requires a trusted licensed interface role",
        any(
            "boost request" in item
            for item in EvidenceReadinessEvaluator.evaluate(
                vehicle, track, boost_scenario
            ).blockers
        ),
    )
    wrong_vehicle = replace(vehicle, vehicle_id="attacker-car")
    wrong_vehicle_scenario = replace(
        scenario, vehicle_spec_sha256=wrong_vehicle.sha256
    )
    gate(
        "wrong vehicle identity cannot claim 919 readiness",
        any(
            "vehicle_id must be" in item
            for item in EvidenceReadinessEvaluator.evaluate(
                wrong_vehicle, track, wrong_vehicle_scenario
            ).blockers
        ),
    )
    bad_length = replace(vehicle.parameter("body_length_m"), unit="furlong")
    bad_geometry = tuple(
        bad_length if item.name == bad_length.name else item
        for item in vehicle.geometry
    )
    wrong_units_vehicle = replace(vehicle, geometry=bad_geometry)
    wrong_units_scenario = replace(
        scenario, vehicle_spec_sha256=wrong_units_vehicle.sha256
    )
    gate(
        "protocol-owned units and arity reject semantic substitution",
        any(
            "parameter semantics mismatch: body_length_m" in item
            for item in EvidenceReadinessEvaluator.evaluate(
                wrong_units_vehicle, track, wrong_units_scenario
            ).blockers
        ),
    )

    print("== 1 ms MuJoCo public-anchor authority baseline ==")
    authority = MuJoCoAuthority(vehicle, track, scenario)
    gate(
        "backend fixed at 1 ms",
        authority.identity.timestep_s == FIXED_TIMESTEP_S == authority.model_metadata.timestep_s,
    )
    gate("backend uses float64", authority.identity.precision == "float64")
    gate("mutable live MjModel is not exposed", not hasattr(authority, "model"))
    gate("baseline runnable", authority.capability.runnable)
    gate("baseline cannot certify", not authority.capability.certification_authority)
    gate("baseline cannot claim faithful", not authority.capability.faithful_claim_allowed)

    start = authority.reset(speed_mps=30.0)
    start_snapshot = authority.snapshot()
    for _ in range(500):
        deployed = authority.step(DriverActionV2(accelerator=1.0))
    deployment_snapshot = authority.snapshot()
    gate("MuJoCo advances exactly 500 steps", deployed.step_index == 500 and math.isclose(deployed.time_s, 0.5, abs_tol=1e-12))
    gate("accelerator increases speed", deployed.velocity_body_mps[0] > start.velocity_body_mps[0])
    gate("ICE telemetry comes from backend state", (deployed.engine_power_w or 0.0) > 0.0)
    gate("front MGU deployment is signed positive", (deployed.front_mgu_power_w or 0.0) > 0.0)
    gate("battery discharge is signed positive", (deployed.battery_power_w or 0.0) > 0.0)
    gate("deployment reduces battery energy", deployment_snapshot.energy.battery_energy_j < start_snapshot.energy.battery_energy_j)
    gate("fuel energy decreases under load", deployment_snapshot.energy.fuel_chemical_energy_j < start_snapshot.energy.fuel_chemical_energy_j)

    # A held categorical request produces one rising-edge shift, not hundreds.
    authority.step(DriverActionV2(shift_request=ShiftRequest.UPSHIFT))
    first_shift_gear = authority.snapshot().telemetry.gear
    for _ in range(20):
        authority.step(DriverActionV2(shift_request=ShiftRequest.UPSHIFT))
    held_shift_gear = authority.snapshot().telemetry.gear
    authority.step(DriverActionV2(shift_request=ShiftRequest.HOLD))
    authority.step(DriverActionV2(shift_request=ShiftRequest.UPSHIFT))
    second_shift_gear = authority.snapshot().telemetry.gear
    gate("shift request is edge-triggered", first_shift_gear == held_shift_gear and second_shift_gear == first_shift_gear + 1)

    battery_before_regen = authority.snapshot().energy.battery_energy_j
    for _ in range(300):
        braking = authority.step(DriverActionV2(brake=0.8))
    battery_after_regen = authority.snapshot().energy.battery_energy_j
    gate("front MGU regen is signed negative", (braking.front_mgu_power_w or 0.0) < 0.0)
    gate("recovered power is nonnegative", (braking.regen_power_w or 0.0) > 0.0)
    gate("battery charge is signed negative", (braking.battery_power_w or 0.0) < 0.0)
    gate("regeneration increases battery energy", battery_after_regen > battery_before_regen)
    projection = TruthfulAvAdapter(CarAssetRegistry.discover()).project(
        braking, vehicle_id="porsche_919evo"
    )
    gate(
        "backend regen semantics bind directly to truthful A/V",
        projection.audio.front_mgu is not None
        and projection.audio.front_mgu.regen_power_w == braking.regen_power_w,
    )

    final_snapshot = authority.snapshot()
    transferred = max(
        1.0,
        start_snapshot.energy.stored_energy_j - final_snapshot.energy.stored_energy_j
        + final_snapshot.energy.accounted_loss_j,
    )
    residual_fraction = abs(final_snapshot.energy.cumulative_numerical_residual_j) / transferred
    gate("energy ledger finite", math.isfinite(final_snapshot.energy.cumulative_numerical_residual_j))
    gate("baseline energy residual bounded", residual_fraction < 1e-3, f"fraction={residual_fraction:.3e}")
    observation = authority.observe()
    gate("driver observation contains no tyre-force truth", not hasattr(observation, "wheel_force_xyz_n"))
    gate("unsupported wheel speeds remain unavailable", all(value is None for value in observation.wheel_speed_rad_s))
    gate("unsupported contact state remains unavailable", all(value is None for value in final_snapshot.telemetry.wheel_contact))
    gate("unsupported active aero remains unavailable", final_snapshot.telemetry.front_aero_position is None and final_snapshot.telemetry.rear_aero_position is None)
    gate(
        "closed controller status rejects arbitrary privileged labels",
        raises(TypeError, lambda: replace(observation, controller_status=(("hidden_friction", 2.1),))),
    )
    gate("driver observation includes onboard hybrid state", 0.0 <= observation.battery_soc <= 1.0)
    gate("snapshot hash stable without stepping", final_snapshot.state_sha256 == authority.snapshot().state_sha256)
    original_timestep = float(authority._model.opt.timestep)
    authority._model.opt.timestep = original_timestep * 2.0
    gate("model mutation invalidates frozen physics identity", raises(RuntimeError, authority.snapshot))
    authority._model.opt.timestep = original_timestep
    gate("restored model passes integrity assertion", authority.snapshot().state_sha256 == final_snapshot.state_sha256)
    original_friction = float(authority._model.geom_friction[0, 0])
    authority._model.geom_friction[0, 0] = original_friction * 1.1
    gate(
        "contact/friction mutation invalidates frozen physics identity",
        raises(RuntimeError, authority.snapshot),
    )
    authority._model.geom_friction[0, 0] = original_friction
    gate(
        "restored contact model passes integrity assertion",
        authority.snapshot().state_sha256 == final_snapshot.state_sha256,
    )

    print("== MJX fail-closed interface ==")
    twin = MJXTrainingTwin(vehicle, track, scenario)
    gate("MJX interface identifies vectorization", twin.capability.vectorized)
    gate("MJX interface never certifies", not twin.capability.certification_authority)
    gate("uncorrelated MJX refuses to run", not twin.capability.runnable)
    gate("uncorrelated MJX cannot claim faithful", not twin.capability.faithful_claim_allowed)
    gate("MJX reset fails closed", raises(RuntimeError, twin.reset))
    gate("MJX snapshot identifies no state", not twin.snapshot().initialized)

    if FAILED:
        print(f"\n{len(FAILED)} gate(s) FAILED: {FAILED}")
        raise SystemExit(1)
    print("\nall faithful-v2 core gates green")


if __name__ == "__main__":
    main()
