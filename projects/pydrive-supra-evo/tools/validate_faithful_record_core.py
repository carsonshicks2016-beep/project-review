#!/usr/bin/env python3
"""Deterministic regression gates for the faithful-v2 record core."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import math
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from supra.record import (  # noqa: E402
    ArtifactPaths,
    CheckpointIdentity,
    CheckpointUse,
    FeasibilityEvidence,
    FlyingLapProtocolV1,
    FlyingLapTraceV1,
    LapEvent,
    LapEventKind,
    LapPlanSample,
    LapPlanV1,
    OraclePrerequisiteError,
    ProtocolSample,
    RecordCertificateV1,
    RunIdentity,
    StageMetrics,
    TrainingStage,
    checkpoint_compatibility,
    create_evidence_bundle_manifest,
    evaluate_feasibility,
    evaluate_stage_gate,
    generate_ed25519_keypair,
    hash_evidence_artifacts,
    oracle_runtime_status,
    require_oracle_prerequisites,
    verify_evidence_bundle,
)
from supra.record._canonical import (  # noqa: E402
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
)


def digest(label: str) -> str:
    return sha256_bytes(label.encode("utf-8"))


VEHICLE_SHA256 = digest("vehicle")
TRACK_SHA256 = digest("track")
SOURCE_SHA256 = digest("source")
DEPENDENCIES_SHA256 = digest("dependencies")
PHYSICS_IDENTITY_PAYLOAD = {
    "backend_name": "mujoco-authority",
    "backend_version": "faithful-v2-certification-1",
    "precision": "float64",
    "timestep_s": 0.001,
    "vehicle_spec_sha256": VEHICLE_SHA256,
    "track_surface_sha256": TRACK_SHA256,
    "source_sha256": SOURCE_SHA256,
    "dependency_sha256": DEPENDENCIES_SHA256,
    "capability_label": "licensed validated certification authority",
}
PHYSICS_IDENTITY_BYTES = canonical_json_bytes(PHYSICS_IDENTITY_PAYLOAD)
PHYSICS_IDENTITY_SHA256 = sha256_bytes(PHYSICS_IDENTITY_BYTES)
PROTOCOL_PAYLOAD = {
    "schema": "faithful-flying-lap-protocol-v1",
    "target_lap_s": 319.546,
    "official_lap_length_m": 20_832.0,
    "minimum_sample_interval_s": 0.0005,
    "maximum_sample_interval_s": 0.1,
}
PROTOCOL_BYTES = canonical_json_bytes(PROTOCOL_PAYLOAD)
PROTOCOL_SHA256 = sha256_bytes(PROTOCOL_BYTES)


def gate(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))
    if not condition:
        raise SystemExit(f"gate failed: {name} {detail}")


def identity_gates(tmp: Path) -> RunIdentity:
    print("== identity and artifact namespace ==")
    identity = RunIdentity(
        edition="porsche-919evo-faithful-v2",
        run_id="record-program-0001",
        vehicle_spec_sha256=VEHICLE_SHA256,
        track_surface_sha256=TRACK_SHA256,
        physics_identity_sha256=PHYSICS_IDENTITY_SHA256,
        controller_sha256=digest("controller"),
        observation_schema_sha256=digest("observations"),
        action_schema_sha256=digest("actions"),
        source_sha256=SOURCE_SHA256,
        dependencies_sha256=DEPENDENCIES_SHA256,
        protocol_sha256=PROTOCOL_SHA256,
    )
    try:
        identity.run_id = "mutated"  # type: ignore[misc]
        immutable = False
    except (FrozenInstanceError, AttributeError):
        immutable = True
    gate("run identity is immutable", immutable)
    paths = ArtifactPaths(tmp / "runtime", identity)
    gate("artifacts are edition and run scoped",
         paths.manifest.as_posix().endswith(
             "fable5/editions/porsche-919evo-faithful-v2/"
             "runs/record-program-0001/manifest.json"))
    try:
        paths.assert_scoped(tmp / "runtime" / "fable5_ring_pipeline.json")
        escaped = False
    except ValueError:
        escaped = True
    gate("legacy/global artifact path is rejected", escaped)
    return identity


def checkpoint_gates(identity: RunIdentity) -> None:
    print("== checkpoint compatibility ==")
    common = dict(
        run=identity,
        policy_architecture_sha256=digest("policy-arch"),
        policy_state_sha256=digest("policy-state"),
        normalizer_schema_sha256=digest("normalizer"),
        normalizer_state_sha256=digest("normalizer-state"),
        optimizer_schema_sha256=digest("optimizer"),
        optimizer_state_sha256=digest("optimizer-state"),
        scheduler_state_sha256=digest("scheduler-state"),
        rng_state_sha256=digest("rng-state"),
        training_budget_sha256=digest("budget"),
    )
    foundation = CheckpointIdentity(stage=TrainingStage.FOUNDATION, **common)
    same = CheckpointIdentity(stage=TrainingStage.FOUNDATION, **common)
    flow = CheckpointIdentity(stage=TrainingStage.FLOW, **common)
    exact = checkpoint_compatibility(foundation, same, CheckpointUse.EXACT_RESUME)
    gate("exact resume carries complete state and evidence",
         exact.allowed and exact.load_optimizer and exact.load_rng
         and exact.carry_evaluation and exact.certifiable_lineage)
    changed_rng = replace(same, rng_state_sha256=digest("different-rng-state"))
    gate("exact resume rejects optimizer/RNG transaction drift",
         not checkpoint_compatibility(
             foundation, changed_rng, CheckpointUse.EXACT_RESUME).allowed)
    warm = checkpoint_compatibility(foundation, flow, CheckpointUse.STAGE_WARM_START)
    gate("stage warm start resets optimizer, RNG, and evaluation",
         warm.allowed and warm.load_policy and warm.load_normalizer
         and not warm.load_optimizer and not warm.load_rng
         and not warm.carry_evaluation and warm.certifiable_lineage)
    forged_warm_target = replace(
        flow, policy_state_sha256=digest("undeclared-post-load-policy")
    )
    gate(
        "stage warm-start identity must match the state actually carried",
        not checkpoint_compatibility(
            foundation, forged_warm_target, CheckpointUse.STAGE_WARM_START
        ).allowed,
    )

    migrated_run = RunIdentity(
        edition="porsche-919evo-faithful-v2", run_id="migration-probe",
        vehicle_spec_sha256=digest("different-vehicle"),
        track_surface_sha256=identity.track_surface_sha256,
        physics_identity_sha256=identity.physics_identity_sha256,
        controller_sha256=identity.controller_sha256,
        observation_schema_sha256=identity.observation_schema_sha256,
        action_schema_sha256=identity.action_schema_sha256,
        source_sha256=identity.source_sha256,
        dependencies_sha256=identity.dependencies_sha256,
        protocol_sha256=identity.protocol_sha256,
    )
    migrated = CheckpointIdentity(
        run=migrated_run, stage=TrainingStage.FOUNDATION,
        policy_architecture_sha256=foundation.policy_architecture_sha256,
        policy_state_sha256=foundation.policy_state_sha256,
        normalizer_schema_sha256=foundation.normalizer_schema_sha256,
        normalizer_state_sha256=foundation.normalizer_state_sha256,
        optimizer_schema_sha256=foundation.optimizer_schema_sha256,
        optimizer_state_sha256=foundation.optimizer_state_sha256,
        scheduler_state_sha256=foundation.scheduler_state_sha256,
        rng_state_sha256=foundation.rng_state_sha256,
        training_budget_sha256=foundation.training_budget_sha256,
    )
    denied = checkpoint_compatibility(foundation, migrated,
                                      CheckpointUse.EXPLICIT_MIGRATION)
    allowed = checkpoint_compatibility(
        foundation, migrated, CheckpointUse.EXPLICIT_MIGRATION,
        migration_adapter_id="legacy-policy-only-v1",
    )
    gate("cross-identity load requires an explicit adapter", not denied.allowed)
    gate("migration is permanently noncertifiable",
         allowed.allowed and allowed.load_policy and not allowed.certifiable_lineage
         and not allowed.load_normalizer and not allowed.load_optimizer)


def oracle_and_feasibility_gates(identity: RunIdentity) -> None:
    print("== oracle plan and conservative feasibility ==")
    sample0 = LapPlanSample(
        0.0, 0.0, (0.0, 0.0, 0.0), 80.0, 0.0, 1.0, 0.0, 4,
        250_000.0, 0.0, 0.9, 0.0, 0.5, 0.1,
    )
    sample1 = LapPlanSample(
        318.0, 20_832.0, (1.0, 0.0, 0.0), 85.0, 0.0, 0.8, 0.0, 7,
        0.0, 100_000.0, 0.2, 1.0, 0.4, 0.05,
    )
    plan = LapPlanV1(
        identity.sha256, digest("scenario"), identity.physics_identity_sha256,
        "casadi-ipopt", "frozen-probe", (sample0, sample1),
    )
    gate("lap plan is immutable, monotonic, and hash bound",
         plan.predicted_lap_time_s == 318.0 and len(plan.sha256) == 64)

    evidence = FeasibilityEvidence(
        run_identity_sha256=identity.sha256,
        lap_plan_sha256=plan.sha256,
        authoritative_replay_sha256=digest("authority-replay"),
        oracle_lap_time_s=318.0,
        authoritative_replay_lap_time_s=318.1,
        numerical_error_bound_s=0.1,
        model_uncertainty_one_sided_95_s=0.5,
        solver_prerequisites_satisfied=True,
        physics_validated=True,
    )
    decision = evaluate_feasibility(evidence)
    gate(
        "bounded 318.7 second arithmetic cannot open the absent oracle runner",
        not decision.passed
        and abs(decision.conservative_upper_bound_s - 318.7) < 1e-9
        and any("execution verifier is not implemented" in item
                for item in decision.blockers),
    )
    try:
        replace(evidence, replay_agreement_limit_s=0.200001)
        widened = True
    except ValueError:
        widened = False
    gate("oracle/replay agreement threshold cannot be widened", not widened)
    blocked = evaluate_feasibility(FeasibilityEvidence(
        run_identity_sha256=identity.sha256,
        lap_plan_sha256=plan.sha256,
        authoritative_replay_sha256=digest("bad-replay"),
        oracle_lap_time_s=318.0,
        authoritative_replay_lap_time_s=319.0,
        numerical_error_bound_s=0.2,
        model_uncertainty_one_sided_95_s=0.5,
        constraint_violations=("front_mgu_current_limit",),
        missing_evidence=("licensed_tyre_model",),
    ))
    gate("missing evidence and replay disagreement fail closed",
         not blocked.passed and len(blocked.blockers) >= 5)

    required = ("licensed_tyre_model", "2018_track_survey")
    available = ("casadi-ipopt",)
    status = oracle_runtime_status(required, available)
    try:
        require_oracle_prerequisites(required, available)
        raised = False
    except OraclePrerequisiteError as exc:
        raised = True
        detail = str(exc)
    gate("optional solver/evidence boundary fails explicitly when incomplete",
         raised and not status.ready and "missing physical evidence" in detail)


def protocol_gates() -> None:
    print("== continuous flying-lap protocol ==")
    samples_list: list[ProtocolSample] = []
    dt = 0.05
    lap_speed_mps = 20_832.0 / 300.0
    for index in range(int(320.0 / dt) + 1):
        time_s = index * dt
        if time_s <= 20.0:
            distance_m = 50.0 * time_s
            speed_mps = 50.0
        else:
            distance_m = 1_000.0 + lap_speed_mps * (time_s - 20.0)
            speed_mps = lap_speed_mps
        samples_list.append(ProtocolSample(
            time_s, (distance_m, 0.0, 0.0), speed_mps, distance_m,
            50.0 - 0.02 * time_s, max(0.2, 0.95 - 0.002 * time_s), 7,
        ))
    samples = tuple(samples_list)
    warm = LapEvent(LapEventKind.WARMUP_START, 0.0, 0.0)
    start = LapEvent(LapEventKind.START_LINE_CROSSING, 20.0, 1_000.0,
                     "t13_start_finish", 1)
    finish = LapEvent(LapEventKind.FINISH_LINE_CROSSING, 320.0, 21_832.0,
                      "t13_start_finish", 1)
    protocol = FlyingLapProtocolV1()
    decision = protocol.evaluate(FlyingLapTraceV1(samples, (warm, start, finish)))
    gate(
        "kinematically plausible straight trace cannot impersonate the surveyed lap",
        not decision.legal and not decision.beats_record
        and any("authority verifier is not implemented" in item
                for item in decision.blockers),
    )

    reset = LapEvent(LapEventKind.RESET, 100.0, 7_000.0)
    bad_events = protocol.evaluate(
        FlyingLapTraceV1(samples, (warm, start, reset, finish)))
    gate("reset during warmup/lap is rejected",
         not bad_events.legal and any("reset" in item for item in bad_events.blockers))
    teleported_samples = list(samples)
    probe_index = int(170.0 / dt)
    probe = teleported_samples[probe_index]
    teleported_samples[probe_index] = replace(
        probe, position_m=(100_000.0, 0.0, 0.0))
    bad_motion = protocol.evaluate(
        FlyingLapTraceV1(tuple(teleported_samples), (warm, start, finish)))
    gate("unreported positional teleport is rejected",
         not bad_motion.legal and any("teleport" in item for item in bad_motion.blockers))

    fixed_pose = tuple(replace(sample, position_m=(0.0, 0.0, 0.0))
                       for sample in samples)
    fixed_decision = protocol.evaluate(
        FlyingLapTraceV1(fixed_pose, (warm, start, finish)))
    gate("fixed-position trace cannot fake a lap with a typed odometer",
         not fixed_decision.legal
         and any("stationary" in item for item in fixed_decision.blockers))

    fake_odometer: list[ProtocolSample] = []
    for sample in samples:
        offset = 0.0
        if sample.time_s > 20.0:
            offset = 1_000.0 * math.sin(
                math.pi * (sample.time_s - 20.0) / 300.0)
        fake_odometer.append(replace(
            sample, cumulative_distance_m=sample.cumulative_distance_m + offset))
    fake_odo_decision = protocol.evaluate(
        FlyingLapTraceV1(tuple(fake_odometer), (warm, start, finish)))
    gate("fake cumulative distance fails two-sided kinematic consistency",
         not fake_odo_decision.legal
         and any("odometer" in item for item in fake_odo_decision.blockers))

    sparse = samples[::20]
    sparse_decision = protocol.evaluate(
        FlyingLapTraceV1(sparse, (warm, start, finish)))
    gate("sparse telemetry gaps cannot conceal discontinuities",
         not sparse_decision.legal
         and any("sample gap" in item for item in sparse_decision.blockers))

    fuel_pulses = tuple(replace(
        sample,
        fuel_kg=sample.fuel_kg + (0.006 if index % 2 else 0.0),
    ) for index, sample in enumerate(samples))
    fuel_decision = protocol.evaluate(
        FlyingLapTraceV1(fuel_pulses, (warm, start, finish)))
    gate("distributed sub-tolerance refuelling is summed and rejected",
         not fuel_decision.legal
         and any("total fuel increase" in item for item in fuel_decision.blockers))


def stage_gates() -> None:
    print("== training stage gates ==")
    probes = (
        StageMetrics(TrainingStage.FOUNDATION, recovered_sectors=16),
        StageMetrics(TrainingStage.FLOW, valid_perturbed_chains=95,
                     perturbed_chain_trials=100),
        StageMetrics(TrainingStage.FINISH, consecutive_legal_full_laps=5),
        StageMetrics(TrainingStage.FAST, best_legal_lap_s=329.999),
        StageMetrics(TrainingStage.FRONTIER, best_legal_lap_s=319.545),
    )
    decisions = tuple(evaluate_stage_gate(probe) for probe in probes)
    gate("all five exact stage thresholds advance", all(item.passed for item in decisions))
    gate("record equality does not pass frontier",
         not evaluate_stage_gate(StageMetrics(
             TrainingStage.FRONTIER, best_legal_lap_s=319.546)).passed)
    try:
        evaluate_stage_gate(
            StageMetrics(TrainingStage.FRONTIER, best_legal_lap_s=500.0),
            record_target_s=999.0,
        )
        widened = True
    except ValueError:
        widened = False
    gate("frontier record benchmark cannot be widened", not widened)
    try:
        StageMetrics(TrainingStage.FRONTIER, best_legal_lap_s=True)
        bool_lap_accepted = True
    except TypeError:
        bool_lap_accepted = False
    try:
        StageMetrics(TrainingStage.FOUNDATION, recovered_sectors=16.0)
        float_count_accepted = True
    except TypeError:
        float_count_accepted = False
    gate(
        "bool lap times and noninteger stage counts are rejected",
        not bool_lap_accepted and not float_count_accepted,
    )


def evidence_bundle_gates(tmp: Path, identity: RunIdentity) -> None:
    print("== independently verifiable evidence bundle ==")
    bundle = tmp / "evidence"
    bundle.mkdir()
    role_paths: dict[str, str] = {}

    def write_role(role: str, content: bytes, suffix: str = "bin") -> None:
        relative = f"artifacts/{role}.{suffix}"
        path = bundle / relative
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
        role_paths[role] = relative

    scenario_payload = {
        "schema_version": "record-scenario-v1",
        "scenario_id": "nordschleife-2018-t13-flying-lap",
        "vehicle_spec_sha256": identity.vehicle_spec_sha256,
        "track_surface_sha256": identity.track_surface_sha256,
        "target_lap_time_s": 319.546,
    }
    source_bundle_payload = {
        "schema": "faithful-source-bundle-v1",
        "source_sha256": identity.source_sha256,
        "dependencies_sha256": identity.dependencies_sha256,
        "container_sha256": digest("container"),
    }
    base_contents = {
        "checkpoint": b"immutable policy checkpoint",
        "scenario": canonical_json_bytes(scenario_payload),
        "action_trace": b"driver-equivalent action trace",
        "telemetry": b"complete vehicle telemetry",
        "replay": b"independent action-only replay",
        "energy_ledger": b"energy residual below 0.1 percent",
        "per_wheel_forces": b"four wheel force and slip histories",
        "thermal_trace": b"tyre brake battery and powertrain temperatures",
        "aero_trace": b"aero actuator states and loads",
        "contact_trace": b"four wheel legal contacts",
        "parameter_posterior": b"frozen QMC parameter posterior",
        "source_bundle": canonical_json_bytes(source_bundle_payload),
        "run_identity": canonical_json_bytes(identity),
        "physics_identity": PHYSICS_IDENTITY_BYTES,
        "protocol": PROTOCOL_BYTES,
        "lap_plan": b"immutable CasADi/IPOPT lap plan",
        "authoritative_replay": b"one millisecond MuJoCo MPC replay",
        "standalone_verifier": b"independent verifier source and environment",
    }
    for role, content in base_contents.items():
        write_role(role, content, "json" if role in {
            "scenario", "source_bundle", "run_identity", "physics_identity", "protocol"
        } else "bin")

    hashes = {role: sha256_file(bundle / relative)
              for role, relative in role_paths.items()}
    common = {
        "run_identity_sha256": identity.sha256,
        "physics_identity_sha256": identity.physics_identity_sha256,
        "protocol_sha256": identity.protocol_sha256,
        "scenario_sha256": hashes["scenario"],
    }
    physics_validation = {
        "schema": "faithful-physics-validation-result-v1",
        **common,
        "passed": True,
        "component_fidelity_passed": True,
        "holdout_telemetry_passed": True,
    }
    write_role("physics_validation", canonical_json_bytes(physics_validation), "json")
    hashes["physics_validation"] = sha256_file(
        bundle / role_paths["physics_validation"])
    oracle_result = {
        "schema": "faithful-oracle-result-v1",
        **common,
        "passed": True,
        "physics_validation_sha256": hashes["physics_validation"],
        "lap_plan_sha256": hashes["lap_plan"],
        "authoritative_replay_sha256": hashes["authoritative_replay"],
        "conservative_upper_bound_s": 318.7,
        "replay_agreement_s": 0.1,
        "constraint_violations": [],
    }
    write_role("oracle_result", canonical_json_bytes(oracle_result), "json")
    lap_protocol_result = {
        "schema": "faithful-lap-protocol-result-v1",
        **common,
        "legal": True,
        "no_teleport": True,
        "action_trace_sha256": hashes["action_trace"],
        "telemetry_sha256": hashes["telemetry"],
        "authoritative_replay_sha256": hashes["authoritative_replay"],
        "lap_time_s": 318.0,
    }
    write_role("lap_protocol_trace", canonical_json_bytes(lap_protocol_result), "json")
    robustness_result = {
        "schema": "faithful-robustness-result-v1",
        **common,
        "robust_record": True,
        "parameter_posterior_sha256": hashes["parameter_posterior"],
        "sample_count": 1_000,
        "valid_fraction": 0.99,
        "p95_lap_time_s": 318.5,
        "p95_bootstrap_upper_s": 319.0,
    }
    write_role("robustness_result", canonical_json_bytes(robustness_result), "json")

    private_key, public_key = generate_ed25519_keypair()

    def make_certificate(
        artifact_links: tuple[tuple[str, str], ...],
    ) -> RecordCertificateV1:
        link_map = dict(artifact_links)
        return RecordCertificateV1(
            run_identity_sha256=link_map["run_identity"],
            checkpoint_sha256=link_map["checkpoint"],
            scenario_sha256=link_map["scenario"],
            physics_identity_sha256=link_map["physics_identity"],
            protocol_sha256=link_map["protocol"],
            action_trace_sha256=link_map["action_trace"],
            telemetry_sha256=link_map["telemetry"],
            replay_sha256=link_map["replay"],
            lap_time_s=318.0,
            target_lap_s=319.546,
            numerical_error_bound_s=0.1,
            model_uncertainty_one_sided_95_s=0.5,
            legal_flying_lap=True,
            no_teleport=True,
            physics_validated=True,
            oracle_feasible=True,
            robust_record=True,
            verifier_id="independent-test-verifier",
            issued_utc="2026-07-16T22:00:00Z",
            deterministic_replay_count=5,
            deterministic_replay_max_delta_s=0.001,
            action_only_replay_delta_s=0.001,
            telemetry_hash_reproduced=True,
            artifact_sha256_by_role=artifact_links,
            robust_sample_count=1_000,
            robust_valid_fraction=0.99,
            robust_p95_lap_time_s=318.5,
            robust_p95_bootstrap_upper_s=319.0,
        ).sign(private_key)

    artifact_links = hash_evidence_artifacts(bundle, role_paths)
    certificate = make_certificate(artifact_links)
    gate("certificate contract requires five deterministic and action-only replays",
         certificate.contract_fields_satisfied
         and not replace(certificate, deterministic_replay_count=4).contract_fields_satisfied
         and not replace(certificate, telemetry_hash_reproduced=False).contract_fields_satisfied)
    gate(
        "self-authored contract fields cannot claim a record without the independent runner",
        not certificate.certifies_record and not certificate.certifies_robust_record,
    )
    try:
        replace(certificate, legal_flying_lap="false")
        bool_bypass = True
    except TypeError:
        bool_bypass = False
    try:
        replace(certificate, deterministic_replay_count=True)
        count_bypass = True
    except TypeError:
        count_bypass = False
    gate("string booleans and bool-as-count certificate bypasses are rejected",
         not bool_bypass and not count_bypass)
    try:
        replace(certificate, robust_p95_lap_time_s=400.0,
                robust_p95_bootstrap_upper_s=319.0)
        reversed_bound = True
    except ValueError:
        reversed_bound = False
    gate("robust bootstrap upper bound cannot sit below measured p95",
         not reversed_bound)

    manifest = create_evidence_bundle_manifest(
        bundle, certificate, role_paths, private_key)
    verified = verify_evidence_bundle(bundle, public_key)
    gate(
        "signed scaffold bundle remains noncertifying until the independent runner exists",
        not verified.valid
        and any("record contract" in item for item in verified.blockers),
        "; ".join(verified.blockers),
    )
    gate("certificate and bundle use domain-separated signatures",
         not replace(
             manifest, bundle_signature=certificate.verifier_signature
         ).verify_signature(public_key))
    wrong_private, wrong_public = generate_ed25519_keypair()
    gate("verification uses the public key, not a shared signing secret",
         manifest.verify_signature(public_key)
         and certificate.verify_signature(public_key)
         and not manifest.verify_signature(wrong_public)
         and not certificate.verify_signature(wrong_public)
         and not manifest.verify_signature(private_key))
    manifest_bytes = (bundle / "RECORD_EVIDENCE.json").read_bytes()
    gate("private signing key is absent from the evidence bundle",
         private_key not in manifest_bytes
         and private_key.hex().encode("ascii") not in manifest_bytes
         and wrong_private not in manifest_bytes)
    try:
        replace(certificate, signature_scheme="hmac-sha256-v1")
        hmac_accepted = True
    except ValueError:
        hmac_accepted = False
    gate("legacy shared-secret HMAC certificates are rejected", not hmac_accepted)

    mutation_cases = {
        "run_identity": ("physics_identity_sha256", digest("wrong-physics")),
        "scenario": ("target_lap_time_s", 400.0),
        "source_bundle": ("dependencies_sha256", digest("wrong-dependencies")),
        "physics_identity": ("vehicle_spec_sha256", digest("wrong-vehicle")),
        "protocol": ("maximum_sample_interval_s", 999.0),
        "physics_validation": ("passed", False),
        "oracle_result": ("physics_validation_sha256", digest("wrong-validation")),
        "lap_protocol_trace": ("legal", False),
        "robustness_result": ("parameter_posterior_sha256", digest("wrong-posterior")),
    }
    rejected_roles: list[str] = []
    for role, (field, bad_value) in mutation_cases.items():
        path = bundle / role_paths[role]
        original = path.read_bytes()
        payload = json.loads(original)
        payload[field] = bad_value
        path.write_bytes(canonical_json_bytes(payload))
        bad_links = hash_evidence_artifacts(bundle, role_paths)
        bad_certificate = make_certificate(bad_links)
        try:
            create_evidence_bundle_manifest(
                bundle, bad_certificate, role_paths, private_key)
        except ValueError:
            rejected_roles.append(role)
        finally:
            path.write_bytes(original)
    gate("run/physics/protocol/validation/oracle/lap/robust reports are cross-linked",
         set(rejected_roles) == set(mutation_cases), ", ".join(rejected_roles))

    restored = verify_evidence_bundle(bundle, public_key)
    gate(
        "adversarial probes leave original signatures and hashes intact",
        manifest.verify_signature(public_key)
        and certificate.verify_signature(public_key)
        and not any("signature mismatch" in item or "digest mismatch" in item
                    for item in restored.blockers),
        "; ".join(restored.blockers),
    )

    telemetry_path = bundle / role_paths["telemetry"]
    telemetry_real = telemetry_path.with_suffix(".real")
    telemetry_path.replace(telemetry_real)
    telemetry_path.symlink_to(telemetry_real.name)
    symlinked = verify_evidence_bundle(bundle, public_key)
    gate(
        "in-bundle artifact symlinks are rejected before resolution",
        not symlinked.valid
        and any("symlink" in item for item in symlinked.blockers),
        "; ".join(symlinked.blockers),
    )
    telemetry_path.unlink()
    telemetry_real.replace(telemetry_path)

    telemetry_path.write_bytes(b"tampered telemetry")
    tampered = verify_evidence_bundle(bundle, public_key)
    gate("one-byte-class artifact tampering is detected",
         not tampered.valid and any("telemetry" in item for item in tampered.blockers))


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="faithful-record-core-") as directory:
        tmp = Path(directory)
        identity = identity_gates(tmp)
        checkpoint_gates(identity)
        oracle_and_feasibility_gates(identity)
        protocol_gates()
        stage_gates()
        evidence_bundle_gates(tmp, identity)
    print("\nAll faithful-v2 record-core gates passed.")


if __name__ == "__main__":
    main()
