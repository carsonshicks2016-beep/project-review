"""Fail-closed control plane for the Porsche 919 faithful-v2 program.

This module deliberately does not infer readiness from a run manifest.  The
public bootstrap is an approximation, the authoritative model/oracle/training
twin are incomplete, and the licensed evidence has not been ingested.  Until
independent verifiers for those artifacts are wired here, every record and
long-run-training gate remains closed regardless of files a caller creates.

The only write exposed by this module creates an explicitly noncertifying run
skeleton in the edition-scoped namespace.  It contains no ``RunIdentity``, no
checkpoint lineage, and no evidence claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.util import find_spec
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .defaults import (
    default_record_scenario,
    official_919evo_vehicle_spec,
    training_fallback_track,
)
from .schemas import EvidenceReadinessEvaluator
from .evidence import (
    DEFAULT_TRUST_STORE,
    EvidenceTrustStoreV1,
    evidence_inventory,
    resolve_evidence_vault,
)


PROGRAM_VERSION = "faithful-v2"
EDITION = "porsche-919evo-faithful-v2"
RECORD_BENCHMARK_S = 319.546
OFFICIAL_LAP_LENGTH_M = 20_832.0
SKELETON_SCHEMA = "faithful-run-skeleton-v1"
_SAFE_RUN_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,79}$")

# These are acquisition requirements, not package-presence guesses.  Keep the
# list concrete so the dashboard never turns an absent input into a vague
# "needs more data" message.
MISSING_LICENSED_EVIDENCE = (
    "licensed Porsche mass properties, CG and inertia tensor",
    "licensed Porsche suspension K&C, damper and active pitch-link maps",
    "licensed Michelin 310/710-18 combined-slip, transient, thermal and wear data",
    "licensed Porsche aero maps and active-element actuator/control evidence",
    "licensed ICE, turbo, fuel, driveline and seven-speed gearbox maps",
    "licensed MGU, battery, exhaust-ERS, regen and thermal-limit maps",
    "licensed brake hydraulics, brake-by-wire, yaw-control and carbon-disc data",
    "June 2018 Nordschleife laser survey with legal boundaries and surface materials",
    "Spa calibration telemetry and component-rig measurements",
    "frozen Nordschleife holdout telemetry, driver inputs and racing line",
    "record weather, fuel, SOC, tyre and brake preparation evidence",
)


class FaithfulProgramBlockedError(RuntimeError):
    """Raised when a command attempts to cross a closed faithful-v2 gate."""


@dataclass(frozen=True, slots=True)
class SkeletonInspection:
    exists: bool
    valid: bool
    detail: str


def _module_available(module_name: str) -> bool:
    try:
        return find_spec(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _optional_file_sha256(path: Path) -> str | None:
    if not path.is_file() or path.is_symlink():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_development_runtime_probe(path: Path, lock_sha256: str | None) -> tuple[bool, str]:
    if not path.is_file() or path.is_symlink():
        return False, "no materialized Linux runtime probe"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        dependencies = value["dependencies"]
        expected_dependencies = {
            "casadi": "3.7.2", "cryptography": "46.0.6", "jax": "0.10.2",
            "mujoco": "3.9.0", "numpy": "2.4.4", "scipy": "1.17.1",
            "trimesh": "4.12.2",
        }
        checks = (
            value.get("schema") == "faithful-authority-runtime-identity-v1",
            value.get("python") == "3.12.10",
            value.get("machine") == "x86_64",
            value.get("jax_x64") is True,
            value.get("dependency_lock_sha256") == lock_sha256,
            dependencies == expected_dependencies,
            type(value.get("mujoco_timestep_s")) in {int, float},
            type(value.get("mujoco_time_after_10_steps_s")) in {int, float},
            math.isclose(float(value.get("mujoco_timestep_s")), 0.001,
                         rel_tol=0.0, abs_tol=1e-15),
            math.isclose(float(value.get("mujoco_time_after_10_steps_s")), 0.01,
                         rel_tol=0.0, abs_tol=1e-12),
            value.get("ipopt") == {"solver": "ipopt", "solution": 3.0},
        )
        if not all(checks):
            return False, "Linux runtime probe does not match the pinned contract"
    except (KeyError, OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
        return False, "Linux runtime probe is invalid"
    return True, "Linux x86-64 MuJoCo/JAX/IPOPT development probe passed"


def _validate_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError(
            "run_id must be a lowercase artifact-safe identifier of at most 80 characters"
        )
    return run_id


def _runtime_root(project_root: str | Path) -> Path:
    project = Path(project_root).expanduser().resolve()
    candidate = project / "runtime"
    resolved = candidate.resolve()
    if resolved != candidate:
        raise ValueError("runtime must be a real directory inside the project, not a symlink")
    return resolved


def _scoped_run_root(project_root: str | Path, run_id: str) -> Path:
    """Resolve the exact edition namespace and reject symlink/path escapes."""
    run_id = _validate_run_id(run_id)
    runtime_root = _runtime_root(project_root)
    edition_candidate = runtime_root / "fable5" / "editions" / EDITION
    edition = edition_candidate.resolve()
    if edition != edition_candidate:
        raise ValueError("faithful edition namespace must not contain symlinks")
    if edition != runtime_root and runtime_root not in edition.parents:
        raise ValueError("faithful edition namespace escapes runtime root")
    run_candidate = edition / "runs" / run_id
    run_root = run_candidate.resolve()
    if run_root != run_candidate:
        raise ValueError("faithful run namespace must not contain symlinks")
    if run_root != edition and edition not in run_root.parents:
        raise ValueError("faithful run namespace escapes its edition")
    return run_root


def _read_skeleton(manifest: Path, run_id: str) -> SkeletonInspection:
    if not manifest.exists():
        return SkeletonInspection(False, False, "run skeleton is not initialized")
    try:
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError("manifest is not a regular file")
        if manifest.stat().st_size > 32 * 1024:
            raise ValueError("manifest exceeds the 32 KiB skeleton limit")
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("manifest root must be an object")
        required = {
            "schema": SKELETON_SCHEMA,
            "program_version": PROGRAM_VERSION,
            "edition": EDITION,
            "run_id": run_id,
            "state": "noncertifying_skeleton",
            "model_class": "telemetry-constrained approximation",
            "certification_eligible": False,
            "training_authorized": False,
            "legacy_import_policy": "forbidden",
            "run_identity": None,
            "evidence_bundle": None,
        }
        for key, expected in required.items():
            if payload.get(key) != expected:
                raise ValueError(f"skeleton field {key!r} does not match the fail-closed contract")
        if type(payload["certification_eligible"]) is not bool:
            raise TypeError("certification_eligible must be an actual bool")
        if type(payload["training_authorized"]) is not bool:
            raise TypeError("training_authorized must be an actual bool")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return SkeletonInspection(True, False, f"invalid noncertifying skeleton: {exc}")
    return SkeletonInspection(True, True, "noncertifying run skeleton initialized")


def faithful_program_status(
    project_root: str | Path,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Return computed public-bootstrap status without trusting run booleans.

    No legacy/global Fable artifact is read.  A selected skeleton contributes
    only its existence/path; it cannot influence any gate or claim.
    """
    vehicle = official_919evo_vehicle_spec()
    track = training_fallback_track()
    scenario = default_record_scenario(vehicle, track)
    if scenario.target_lap_time_s != RECORD_BENCHMARK_S:
        raise RuntimeError("faithful-v2 scenario benchmark drifted from 319.546 seconds")
    readiness = EvidenceReadinessEvaluator.evaluate(vehicle, track, scenario)

    casadi_available = _module_available("casadi")
    mujoco_available = _module_available("mujoco")
    jax_available = _module_available("jax")
    mjx_module_available = _module_available("mujoco.mjx")
    # Package presence does not make the twin runnable.  Correlation and the
    # shared faithful dynamics adapter are mandatory and are currently absent.
    mjx_dependency_available = (
        jax_available and mujoco_available and mjx_module_available
    )

    evidence_pipeline: dict[str, Any] = {
        "implemented": True,
        "trust_store": str(DEFAULT_TRUST_STORE),
        "trusted_key_count": 0,
        "vault_configured": False,
        "inventory": None,
        "blockers": [],
    }
    try:
        trust_store = EvidenceTrustStoreV1.load(DEFAULT_TRUST_STORE)
        evidence_pipeline["trusted_key_count"] = len(trust_store.keys)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        trust_store = None
        evidence_pipeline["blockers"].append(f"invalid protocol trust store: {exc}")
    configured_vault = os.environ.get("FAITHFUL_EVIDENCE_ROOT")
    if configured_vault:
        evidence_pipeline["vault_configured"] = True
        if trust_store is not None:
            try:
                vault = resolve_evidence_vault(project_root, configured_vault)
                evidence_pipeline["inventory"] = evidence_inventory(vault, trust_store)
            except (OSError, TypeError, ValueError) as exc:
                evidence_pipeline["blockers"].append(f"invalid evidence vault: {exc}")

    project_path = Path(project_root).expanduser().resolve()
    runtime_lock = project_path / "requirements-faithful-lock.txt"
    runtime_dockerfile = project_path / "containers" / "faithful" / "Dockerfile"
    runtime_probe = project_path / "runtime" / "faithful" / "linux-amd64-runtime-identity.json"
    runtime_lock_sha256 = _optional_file_sha256(runtime_lock)
    runtime_probe_verified, runtime_probe_detail = _verify_development_runtime_probe(
        runtime_probe, runtime_lock_sha256
    )
    authority_runtime = {
        "target": "linux/amd64",
        "base_image": (
            "python:3.12.10-slim-bookworm@"
            "sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db"
        ),
        "dependency_lock_sha256": runtime_lock_sha256,
        "container_spec_sha256": _optional_file_sha256(runtime_dockerfile),
        "development_probe_path": str(runtime_probe),
        "development_probe_sha256": _optional_file_sha256(runtime_probe),
        "development_probe_verified": runtime_probe_verified,
        "development_probe_detail": runtime_probe_detail,
        # A package smoke test is not equivalent to a successfully built,
        # exported and independently identified Linux image.
        "linux_image_identity_verified": False,
    }

    artifact_root: str | None = None
    skeleton = SkeletonInspection(False, False, "no faithful run selected")
    if run_id is not None:
        run_root = _scoped_run_root(project_root, run_id)
        artifact_root = str(run_root)
        skeleton = _read_skeleton(run_root / "manifest.json", run_id)

    blockers = [
        "No certification-eligible signed evidence freeze is bound to this run",
        "June 2018 certification track survey is absent; the current track is training-only",
        "MuJoCo authority is a public-anchor planar baseline, not the licensed 6-DOF model",
        "MJX shared-model dynamics and MuJoCo correlation gates are not implemented",
        "CasADi/IPOPT whole-lap collocation plus 1 ms nonlinear-MPC replay is not implemented",
        "No verified oracle result satisfies time + numerical error + model uncertainty < 319.546 s",
        "No independently materialized actor-boundary audit authorizes long-run training",
        "No independently signed policy certificate or robust 1,000-sample result exists",
        "Legacy Mazda/919 checkpoints, optimizers, normalizers, HOF, heat and evaluations are forbidden",
    ]
    if evidence_pipeline["trusted_key_count"] == 0:
        blockers.append("No protocol-trusted evidence custodian public key is configured")
    if not evidence_pipeline["vault_configured"]:
        blockers.append("FAITHFUL_EVIDENCE_ROOT is not configured")
    if not authority_runtime["development_probe_verified"]:
        blockers.append(authority_runtime["development_probe_detail"])
    if not authority_runtime["linux_image_identity_verified"]:
        blockers.append(
            "Linux development probe is not an independently signed container identity bound to this run"
        )
    inventory = evidence_pipeline["inventory"]
    if isinstance(inventory, dict) and inventory.get("missing_roles"):
        blockers.append(
            f"External evidence vault is missing {len(inventory['missing_roles'])} required roles"
        )
    blockers.extend(evidence_pipeline["blockers"])
    if not runtime_probe_verified:
        if not casadi_available:
            blockers.append(
                "CasADi/IPOPT is unavailable locally and has no verified authority-runtime probe"
            )
        else:
            blockers.append(
                "Local CasADi is present, but IPOPT has no pinned authority-runtime evidence"
            )
    if not mjx_dependency_available:
        blockers.append("JAX/MuJoCo dependencies required by the MJX twin are incomplete")
    if run_id is not None and not skeleton.valid:
        blockers.append(skeleton.detail)

    # These values are computed here and intentionally hard-false for the
    # public bootstrap.  Do not replace them with fields read from manifest.json.
    gates = {
        "training": False,
        "physics_validated": False,
        "oracle_feasible": False,
        "sim_record": False,
        "robust_record": False,
    }
    return {
        "program_version": PROGRAM_VERSION,
        "edition": EDITION,
        "benchmark_s": RECORD_BENCHMARK_S,
        "lap_length_m": OFFICIAL_LAP_LENGTH_M,
        "model_class": readiness.capability_label,
        "current_stage": "evidence acquisition",
        "run_id": run_id,
        "artifact_root": artifact_root,
        "run_skeleton_initialized": skeleton.valid,
        "run_skeleton_detail": skeleton.detail,
        "authority": (
            "MuJoCo public-anchor baseline · float64 · 1 ms · noncertifying"
            if mujoco_available else
            "MuJoCo unavailable · authoritative backend blocked"
        ),
        "training_twin": (
            "MJX dependencies present · dynamics uncorrelated and not runnable"
            if mjx_dependency_available else
            "MJX dependencies incomplete · dynamics not runnable"
        ),
        "oracle_runtime": {
            "casadi_available": casadi_available or runtime_probe_verified,
            "local_casadi_available": casadi_available,
            "authority_container_casadi_available": runtime_probe_verified,
            "ipopt_smoke_test_evidenced": runtime_probe_verified,
            "whole_lap_solver_implemented": False,
            "authority_mpc_replay_implemented": False,
        },
        "dependencies": {
            "mujoco_available": mujoco_available,
            "jax_available": jax_available,
            "mujoco_mjx_available": mjx_module_available,
        },
        "authority_runtime": authority_runtime,
        "evidence_pipeline": evidence_pipeline,
        "bootstrap_hashes": {
            "vehicle_spec_sha256": vehicle.sha256,
            "track_surface_sha256": track.sha256,
            "scenario_sha256": scenario.sha256,
        },
        "gates": gates,
        "missing_evidence": list(MISSING_LICENSED_EVIDENCE),
        "readiness_blocker_count": len(readiness.blockers),
        "blockers": blockers,
        "oracle_lap_s": None,
        "certified_lap_s": None,
        "certified_champion": None,
        "recommended_next_action": (
            "Designate a trusted evidence custodian key, configure the external vault, "
            "and ingest the licensed vehicle, tyre, track and telemetry packages"
        ),
    }


def initialize_run_skeleton(project_root: str | Path, run_id: str) -> Path:
    """Create an edition-scoped, noncertifying skeleton without fake evidence."""
    run_root = _scoped_run_root(project_root, run_id)
    manifest = run_root / "manifest.json"
    existing = _read_skeleton(manifest, run_id)
    if existing.exists:
        if existing.valid:
            return manifest
        raise FileExistsError(existing.detail)
    if run_root.exists():
        raise FileExistsError(
            f"refusing to adopt existing run directory without a valid skeleton: {run_root}"
        )

    parent = run_root.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after mkdir to catch an edition/runs symlink introduced before
    # the write.  The run leaf itself must be newly created.
    checked = _scoped_run_root(project_root, run_id)
    if checked != run_root:
        raise ValueError("faithful run namespace changed during initialization")
    run_root.mkdir(mode=0o700, exist_ok=False)

    payload = {
        "schema": SKELETON_SCHEMA,
        "program_version": PROGRAM_VERSION,
        "edition": EDITION,
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "state": "noncertifying_skeleton",
        "model_class": "telemetry-constrained approximation",
        "benchmark_s": RECORD_BENCHMARK_S,
        "lap_length_m": OFFICIAL_LAP_LENGTH_M,
        "certification_eligible": False,
        "training_authorized": False,
        "legacy_import_policy": "forbidden",
        "run_identity": None,
        "evidence_bundle": None,
        "note": (
            "Namespace reservation only. This file is not evidence and cannot open "
            "physics, oracle, training, simulation-record or robust-record gates."
        ),
    }
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(manifest, flags, 0o600)
        try:
            remaining = memoryview(encoded)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short write while creating faithful run skeleton")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except Exception:
        try:
            run_root.rmdir()
        except OSError:
            pass
        raise
    return manifest


def _blocked_message(command: str, status: dict[str, Any]) -> str:
    reasons = status["blockers"]
    rendered = "\n  - ".join(reasons)
    return (
        f"faithful-v2 {command} is blocked; no process or checkpoint was started.\n"
        f"  - {rendered}"
    )


def require_oracle_launch_ready(
    project_root: str | Path,
    run_id: str | None = None,
) -> None:
    """Fail until verified physics and a real authority-replayed oracle exist."""
    status = faithful_program_status(project_root, run_id=run_id)
    # Deliberately ignore every caller-authored manifest boolean.  A future
    # implementation may return only after cryptographic artifact verification
    # is added here; the public bootstrap can never reach that branch.
    raise FaithfulProgramBlockedError(_blocked_message("oracle", status))


def require_training_launch_ready(
    project_root: str | Path,
    run_id: str | None = None,
    *,
    iterations: int,
) -> None:
    """Fail before allocating workers unless physics, oracle and actor audit pass."""
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("faithful-v2 training iterations must be a positive integer")
    status = faithful_program_status(project_root, run_id=run_id)
    raise FaithfulProgramBlockedError(_blocked_message("training", status))


__all__ = [
    "EDITION", "FaithfulProgramBlockedError", "MISSING_LICENSED_EVIDENCE",
    "OFFICIAL_LAP_LENGTH_M", "PROGRAM_VERSION", "RECORD_BENCHMARK_S",
    "SKELETON_SCHEMA", "faithful_program_status", "initialize_run_skeleton",
    "require_oracle_launch_ready", "require_training_launch_ready",
]
