"""Conservative feasibility, asymmetric certificates, and evidence bundles.

The verifier deliberately never accepts a shared signing secret.  A private
Ed25519 key is used only by the independent certifier; bundles contain a hash
of the corresponding public key and are verified with that public key.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Mapping

from ._canonical import (
    canonical_json_bytes,
    canonicalize,
    require_sha256,
    secure_digest_equal,
    sha256_bytes,
    sha256_file,
    sha256_object,
)
from .protocol import OFFICIAL_LAP_LENGTH_M, RECORD_TARGET_S

try:  # Fail closed at the signing/verification boundary if unavailable.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - exercised on minimal verifier hosts
    InvalidSignature = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]


CERTIFICATE_SCHEMA = "faithful-record-certificate-v1"
BUNDLE_SCHEMA = "faithful-record-evidence-bundle-v1"
SIGNATURE_SCHEME = "ed25519-v1"
MANIFEST_NAME = "RECORD_EVIDENCE.json"
MAX_REPLAY_AGREEMENT_S = 0.2
# Interfaces and arithmetic are implemented, but the repo does not yet contain
# the CasADi/MPC authority execution or the action-only replay/QMC certifier.
# Keep claim-producing properties closed until those runners independently
# materialize and verify the artifacts they summarize.
AUTHORITY_ORACLE_EXECUTION_VERIFIER_IMPLEMENTED = False
INDEPENDENT_CERTIFICATION_RUNNER_IMPLEMENTED = False
CERTIFICATE_SIGNATURE_DOMAIN = b"SUPRA/FAITHFUL-V2/CERTIFICATE/V1\x00"
BUNDLE_SIGNATURE_DOMAIN = b"SUPRA/FAITHFUL-V2/EVIDENCE-BUNDLE/V1\x00"

# Every semantic input or result needed to reproduce a claim has a dedicated
# artifact.  The complete role->digest map is signed inside the certificate;
# it is not enough for a manifest to say that validation/oracle checks passed.
REQUIRED_ROLES = frozenset((
    "checkpoint", "scenario", "action_trace", "telemetry", "replay",
    "energy_ledger", "per_wheel_forces", "thermal_trace", "aero_trace",
    "contact_trace", "parameter_posterior", "source_bundle",
    "run_identity", "physics_identity", "protocol", "physics_validation",
    "lap_plan", "authoritative_replay", "oracle_result",
    "lap_protocol_trace", "robustness_result", "standalone_verifier",
))


def _require_ed25519() -> None:
    if Ed25519PrivateKey is None or Ed25519PublicKey is None:
        raise RuntimeError(
            "Ed25519 support is unavailable; install cryptography to sign or "
            "verify faithful-v2 record evidence"
        )


def _private_key(value: bytes | Any) -> Any:
    _require_ed25519()
    if isinstance(value, Ed25519PrivateKey):
        return value
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError("Ed25519 private key must be a 32-byte raw seed")
    return Ed25519PrivateKey.from_private_bytes(value)


def _public_key(value: bytes | Any) -> Any:
    _require_ed25519()
    if isinstance(value, Ed25519PublicKey):
        return value
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError("Ed25519 public key must be exactly 32 raw bytes")
    return Ed25519PublicKey.from_public_bytes(value)


def _raw_public_key(value: bytes | Any, *, private: bool = False) -> bytes:
    key = _private_key(value).public_key() if private else _public_key(value)
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def generate_ed25519_keypair() -> tuple[bytes, bytes]:
    """Return ``(private_seed, public_key)`` in raw 32-byte form."""
    _require_ed25519()
    private = Ed25519PrivateKey.generate()
    private_bytes = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_bytes = _raw_public_key(private, private=True)
    return private_bytes, public_bytes


def _signature_bytes(value: str) -> bytes:
    if len(value) != 128:
        raise ValueError("Ed25519 signature must be 128 hexadecimal characters")
    try:
        signature = bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("Ed25519 signature must be hexadecimal") from exc
    if len(signature) != 64:
        raise ValueError("Ed25519 signature must contain 64 bytes")
    return signature


def _verify_signature(public_key: bytes | Any, signature: str,
                      domain: bytes, payload: Any) -> bool:
    if not signature:
        return False
    try:
        _public_key(public_key).verify(
            _signature_bytes(signature), domain + canonical_json_bytes(payload)
        )
        return True
    except (ValueError, TypeError):
        return False
    except Exception as exc:  # cryptography raises InvalidSignature on mismatch
        if InvalidSignature is not None and isinstance(exc, InvalidSignature):
            return False
        return False


@dataclass(frozen=True, slots=True)
class FeasibilityEvidence:
    run_identity_sha256: str
    lap_plan_sha256: str
    authoritative_replay_sha256: str
    oracle_lap_time_s: float
    authoritative_replay_lap_time_s: float
    numerical_error_bound_s: float
    model_uncertainty_one_sided_95_s: float
    replay_agreement_limit_s: float = MAX_REPLAY_AGREEMENT_S
    constraint_violations: tuple[str, ...] = ()
    missing_evidence: tuple[str, ...] = ()
    solver_prerequisites_satisfied: bool = False
    physics_validated: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "constraint_violations",
                           tuple(self.constraint_violations))
        object.__setattr__(self, "missing_evidence", tuple(self.missing_evidence))
        for name in ("run_identity_sha256", "lap_plan_sha256",
                     "authoritative_replay_sha256"):
            require_sha256(getattr(self, name), name)
        for name in ("oracle_lap_time_s", "authoritative_replay_lap_time_s",
                     "numerical_error_bound_s",
                     "model_uncertainty_one_sided_95_s",
                     "replay_agreement_limit_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.replay_agreement_limit_s <= MAX_REPLAY_AGREEMENT_S:
            raise ValueError("oracle/replay agreement limit cannot exceed 0.2 seconds")
        for name in ("solver_prerequisites_satisfied", "physics_validated"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an actual bool")


@dataclass(frozen=True, slots=True)
class FeasibilityDecision:
    passed: bool
    conservative_upper_bound_s: float
    nominal_time_s: float
    blockers: tuple[str, ...]


def evaluate_feasibility(evidence: FeasibilityEvidence,
                         target_s: float = RECORD_TARGET_S) -> FeasibilityDecision:
    """Apply the fixed conservative long-run gate without manufacturing pace."""
    nominal = max(evidence.oracle_lap_time_s,
                  evidence.authoritative_replay_lap_time_s)
    upper = (nominal + evidence.numerical_error_bound_s
             + evidence.model_uncertainty_one_sided_95_s)
    blockers: list[str] = []
    if not AUTHORITY_ORACLE_EXECUTION_VERIFIER_IMPLEMENTED:
        blockers.append(
            "authoritative CasADi/MPC oracle execution verifier is not implemented"
        )
    if abs(target_s - RECORD_TARGET_S) > 1e-12:
        blockers.append("faithful-v2 benchmark is fixed at 319.546 seconds")
    if not evidence.solver_prerequisites_satisfied:
        blockers.append("oracle solver prerequisites are not satisfied")
    if not evidence.physics_validated:
        blockers.append("physics validation gate has not passed")
    if evidence.missing_evidence:
        blockers.append("missing evidence: " + ", ".join(evidence.missing_evidence))
    if evidence.constraint_violations:
        blockers.append("authoritative replay has constraint violations: "
                        + ", ".join(evidence.constraint_violations))
    disagreement = abs(evidence.oracle_lap_time_s
                       - evidence.authoritative_replay_lap_time_s)
    # The global contract remains authoritative even if an object is produced
    # by an older/deserialization path.
    agreement_limit = min(evidence.replay_agreement_limit_s,
                          MAX_REPLAY_AGREEMENT_S)
    if disagreement > agreement_limit:
        blockers.append(
            f"oracle/replay disagreement {disagreement:.6f}s exceeds "
            f"{agreement_limit:.6f}s"
        )
    if upper >= RECORD_TARGET_S:
        blockers.append(
            f"conservative upper bound {upper:.6f}s does not beat "
            f"{RECORD_TARGET_S:.3f}s"
        )
    return FeasibilityDecision(not blockers, upper, nominal, tuple(blockers))


def _normalize_artifact_hashes(
    values: tuple[tuple[str, str], ...] | list[list[str]] | Any,
) -> tuple[tuple[str, str], ...]:
    links: list[tuple[str, str]] = []
    for item in values:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise TypeError("artifact hashes must be (role, sha256) pairs")
        role, digest = item
        if not isinstance(role, str) or not role:
            raise TypeError("artifact hash role must be a nonempty string")
        if not isinstance(digest, str):
            raise TypeError("artifact hash digest must be a string")
        require_sha256(digest, f"artifact hash for {role}")
        links.append((role, digest))
    roles = [role for role, _ in links]
    if len(set(roles)) != len(roles):
        raise ValueError("artifact hash roles must be unique")
    missing = REQUIRED_ROLES - set(roles)
    extras = set(roles) - REQUIRED_ROLES
    if missing or extras:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(sorted(missing)))
        if extras:
            detail.append("unrecognized: " + ", ".join(sorted(extras)))
        raise ValueError("certificate must bind exactly all evidence roles ("
                         + "; ".join(detail) + ")")
    return tuple(sorted(links))


@dataclass(frozen=True, slots=True)
class RecordCertificateV1:
    run_identity_sha256: str
    checkpoint_sha256: str
    scenario_sha256: str
    physics_identity_sha256: str
    protocol_sha256: str
    action_trace_sha256: str
    telemetry_sha256: str
    replay_sha256: str
    lap_time_s: float
    target_lap_s: float
    numerical_error_bound_s: float
    model_uncertainty_one_sided_95_s: float
    legal_flying_lap: bool
    no_teleport: bool
    physics_validated: bool
    oracle_feasible: bool
    robust_record: bool
    verifier_id: str
    issued_utc: str
    deterministic_replay_count: int
    deterministic_replay_max_delta_s: float
    action_only_replay_delta_s: float
    telemetry_hash_reproduced: bool
    artifact_sha256_by_role: tuple[tuple[str, str], ...]
    robust_sample_count: int = 0
    robust_valid_fraction: float = 0.0
    robust_p95_lap_time_s: float | None = None
    robust_p95_bootstrap_upper_s: float | None = None
    verifier_signature: str = ""
    verifier_public_key_sha256: str = ""
    signature_scheme: str = SIGNATURE_SCHEME
    schema: str = CERTIFICATE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_sha256_by_role",
                           _normalize_artifact_hashes(
                               self.artifact_sha256_by_role))
        if self.schema != CERTIFICATE_SCHEMA:
            raise ValueError(f"unsupported certificate schema: {self.schema}")
        for name in (
            "run_identity_sha256", "checkpoint_sha256", "scenario_sha256",
            "physics_identity_sha256", "protocol_sha256", "action_trace_sha256",
            "telemetry_sha256", "replay_sha256",
        ):
            require_sha256(getattr(self, name), name)
        for name in ("lap_time_s", "target_lap_s", "numerical_error_bound_s",
                     "model_uncertainty_one_sided_95_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if abs(self.target_lap_s - RECORD_TARGET_S) > 1e-12:
            raise ValueError("faithful-v2 benchmark is fixed at 319.546 seconds")
        for name in (
            "legal_flying_lap", "no_teleport", "physics_validated",
            "oracle_feasible", "robust_record", "telemetry_hash_reproduced",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be an actual bool")
        for name in ("deterministic_replay_count", "robust_sample_count"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be an actual int, not bool/float")
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")
        for name in ("deterministic_replay_max_delta_s",
                     "action_only_replay_delta_s"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 <= self.robust_valid_fraction <= 1:
            raise ValueError("robust_valid_fraction must be in [0, 1]")
        for name in ("robust_p95_lap_time_s", "robust_p95_bootstrap_upper_s"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive when present")
        if (self.robust_p95_lap_time_s is not None
                and self.robust_p95_bootstrap_upper_s is not None
                and self.robust_p95_bootstrap_upper_s
                < self.robust_p95_lap_time_s):
            raise ValueError("bootstrap upper bound cannot be below the measured p95")
        if not self.verifier_id or not self.issued_utc:
            raise ValueError("verifier_id and issued_utc are required")
        if self.signature_scheme != SIGNATURE_SCHEME:
            raise ValueError("unsupported verifier signature scheme")
        if self.verifier_signature:
            _signature_bytes(self.verifier_signature)
        if self.verifier_public_key_sha256:
            require_sha256(self.verifier_public_key_sha256,
                           "verifier_public_key_sha256")

        links = self.artifact_hashes
        direct_links = {
            "run_identity": self.run_identity_sha256,
            "checkpoint": self.checkpoint_sha256,
            "scenario": self.scenario_sha256,
            "physics_identity": self.physics_identity_sha256,
            "protocol": self.protocol_sha256,
            "action_trace": self.action_trace_sha256,
            "telemetry": self.telemetry_sha256,
            "replay": self.replay_sha256,
        }
        for role, expected in direct_links.items():
            if links[role] != expected:
                raise ValueError(f"certificate {role} hash is not cross-linked")

    @property
    def artifact_hashes(self) -> dict[str, str]:
        return dict(self.artifact_sha256_by_role)

    @property
    def contract_fields_satisfied(self) -> bool:
        conservative_time = (self.lap_time_s + self.numerical_error_bound_s
                             + self.model_uncertainty_one_sided_95_s)
        return (
            self.legal_flying_lap and self.no_teleport and self.physics_validated
            and self.oracle_feasible and conservative_time < self.target_lap_s
            and self.deterministic_replay_count >= 5
            and self.deterministic_replay_max_delta_s <= 0.001
            and self.action_only_replay_delta_s <= 0.001
            and self.telemetry_hash_reproduced
            and bool(self.verifier_signature)
            and bool(self.verifier_public_key_sha256)
        )

    @property
    def certifies_record(self) -> bool:
        return (
            INDEPENDENT_CERTIFICATION_RUNNER_IMPLEMENTED
            and self.contract_fields_satisfied
        )

    @property
    def certifies_robust_record(self) -> bool:
        return (
            INDEPENDENT_CERTIFICATION_RUNNER_IMPLEMENTED
            and self.robust_record and self.contract_fields_satisfied
            and self.robust_sample_count >= 1_000
            and self.robust_valid_fraction >= 0.99
            and self.robust_p95_lap_time_s is not None
            and self.robust_p95_bootstrap_upper_s is not None
            and self.robust_p95_bootstrap_upper_s >= self.robust_p95_lap_time_s
            and self.robust_p95_bootstrap_upper_s < self.target_lap_s
        )

    def unsigned_payload(self) -> dict:
        return {
            field: getattr(self, field) for field in self.__dataclass_fields__
            if field != "verifier_signature"
        }

    def to_dict(self) -> dict:
        return canonicalize(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RecordCertificateV1":
        values = dict(payload)
        values["artifact_sha256_by_role"] = tuple(
            tuple(item) for item in values["artifact_sha256_by_role"]
        )
        return cls(**values)

    def sign(self, private_key: bytes | Any) -> "RecordCertificateV1":
        key = _private_key(private_key)
        public_bytes = _raw_public_key(key, private=True)
        unsigned = replace(
            self,
            verifier_signature="",
            verifier_public_key_sha256=sha256_bytes(public_bytes),
        )
        signature = key.sign(
            CERTIFICATE_SIGNATURE_DOMAIN
            + canonical_json_bytes(unsigned.unsigned_payload())
        ).hex()
        return replace(unsigned, verifier_signature=signature)

    def verify_signature(self, public_key: bytes | Any) -> bool:
        try:
            public_bytes = _raw_public_key(public_key)
        except (RuntimeError, ValueError, TypeError):
            return False
        if (not self.verifier_public_key_sha256
                or not secure_digest_equal(
                    sha256_bytes(public_bytes), self.verifier_public_key_sha256)):
            return False
        return _verify_signature(
            public_key, self.verifier_signature, CERTIFICATE_SIGNATURE_DOMAIN,
            self.unsigned_payload(),
        )


@dataclass(frozen=True, slots=True)
class EvidenceArtifact:
    role: str
    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError("artifact role is required")
        pure = PurePosixPath(self.path)
        if pure.is_absolute() or ".." in pure.parts or self.path == MANIFEST_NAME:
            raise ValueError(f"unsafe evidence path: {self.path!r}")
        if type(self.bytes) is not int or self.bytes < 0:
            raise ValueError("artifact size must be a nonnegative integer")
        require_sha256(self.sha256, "artifact sha256")


@dataclass(frozen=True, slots=True)
class EvidenceBundleManifestV1:
    certificate: RecordCertificateV1
    artifacts: tuple[EvidenceArtifact, ...]
    bundle_signature: str = ""
    signing_public_key_sha256: str = ""
    signature_scheme: str = SIGNATURE_SCHEME
    schema: str = BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.schema != BUNDLE_SCHEMA:
            raise ValueError(f"unsupported bundle schema: {self.schema}")
        roles = [artifact.role for artifact in self.artifacts]
        paths = [artifact.path for artifact in self.artifacts]
        if len(set(roles)) != len(roles) or len(set(paths)) != len(paths):
            raise ValueError("evidence roles and paths must be unique")
        if self.signature_scheme != SIGNATURE_SCHEME:
            raise ValueError("unsupported bundle signature scheme")
        if self.bundle_signature:
            _signature_bytes(self.bundle_signature)
        if self.signing_public_key_sha256:
            require_sha256(self.signing_public_key_sha256,
                           "signing_public_key_sha256")

    def unsigned_payload(self) -> dict:
        return {
            "certificate": self.certificate,
            "artifacts": self.artifacts,
            "signing_public_key_sha256": self.signing_public_key_sha256,
            "signature_scheme": self.signature_scheme,
            "schema": self.schema,
        }

    def sign(self, private_key: bytes | Any) -> "EvidenceBundleManifestV1":
        key = _private_key(private_key)
        public_bytes = _raw_public_key(key, private=True)
        unsigned = replace(
            self,
            bundle_signature="",
            signing_public_key_sha256=sha256_bytes(public_bytes),
        )
        signature = key.sign(
            BUNDLE_SIGNATURE_DOMAIN
            + canonical_json_bytes(unsigned.unsigned_payload())
        ).hex()
        return replace(unsigned, bundle_signature=signature)

    def verify_signature(self, public_key: bytes | Any) -> bool:
        try:
            public_bytes = _raw_public_key(public_key)
        except (RuntimeError, ValueError, TypeError):
            return False
        if (not self.signing_public_key_sha256
                or not secure_digest_equal(
                    sha256_bytes(public_bytes), self.signing_public_key_sha256)):
            return False
        return _verify_signature(
            public_key, self.bundle_signature, BUNDLE_SIGNATURE_DOMAIN,
            self.unsigned_payload(),
        )

    def to_dict(self) -> dict:
        return canonicalize(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "EvidenceBundleManifestV1":
        return cls(
            certificate=RecordCertificateV1.from_dict(payload["certificate"]),
            artifacts=tuple(EvidenceArtifact(**item) for item in payload["artifacts"]),
            bundle_signature=payload.get("bundle_signature", ""),
            signing_public_key_sha256=payload.get(
                "signing_public_key_sha256", ""),
            signature_scheme=payload.get("signature_scheme", ""),
            schema=payload.get("schema", BUNDLE_SCHEMA),
        )


@dataclass(frozen=True, slots=True)
class BundleVerification:
    valid: bool
    blockers: tuple[str, ...]
    manifest_sha256: str | None = None


def _safe_bundle_file(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or ".." in pure.parts:
        raise ValueError(f"unsafe evidence path: {relative!r}")
    resolved_root = root.resolve()
    lexical = resolved_root.joinpath(*pure.parts)
    # Path.resolve() erases symlink identity. Check every lexical component
    # first so an in-bundle symlink is not mistaken for an ordinary immutable
    # evidence file merely because its target also lies inside the bundle.
    cursor = resolved_root
    for part in pure.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"evidence path contains a symlink: {relative}")
    candidate = lexical.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError(f"evidence path escapes bundle: {relative}")
    return candidate


def _artifacts_for_paths(bundle_root: Path,
                         role_paths: Mapping[str, str]) -> tuple[EvidenceArtifact, ...]:
    missing = REQUIRED_ROLES - set(role_paths)
    extras = set(role_paths) - REQUIRED_ROLES
    if missing or extras:
        detail = []
        if missing:
            detail.append("missing: " + ", ".join(sorted(missing)))
        if extras:
            detail.append("unrecognized: " + ", ".join(sorted(extras)))
        raise ValueError("evidence roles must match the contract ("
                         + "; ".join(detail) + ")")
    artifacts: list[EvidenceArtifact] = []
    for role, relative in sorted(role_paths.items()):
        path = _safe_bundle_file(bundle_root, relative)
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"evidence artifact is not a regular file: {relative}")
        artifacts.append(EvidenceArtifact(
            role, PurePosixPath(relative).as_posix(), path.stat().st_size,
            sha256_file(path),
        ))
    return tuple(artifacts)


def hash_evidence_artifacts(bundle_root: Path,
                            role_paths: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    """Hash all and only the required roles for certificate construction."""
    return tuple((artifact.role, artifact.sha256)
                 for artifact in _artifacts_for_paths(bundle_root, role_paths))


def _json_artifact(role: str, paths: Mapping[str, Path],
                   blockers: list[str]) -> dict[str, Any] | None:
    path = paths.get(role)
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("root must be a JSON object")
        return payload
    except Exception as exc:
        blockers.append(f"{role} semantic artifact is invalid JSON: {exc}")
        return None


def _check_fields(role: str, payload: dict[str, Any] | None,
                  expected: Mapping[str, Any], blockers: list[str]) -> None:
    if payload is None:
        return
    for field, value in expected.items():
        if payload.get(field) != value or type(payload.get(field)) is not type(value):
            blockers.append(f"{role} semantic mismatch: {field}")


def _semantic_cross_link_blockers(
    certificate: RecordCertificateV1,
    paths: Mapping[str, Path],
) -> tuple[str, ...]:
    """Validate signed report relationships, not just certificate booleans."""
    blockers: list[str] = []
    hashes = certificate.artifact_hashes
    run = _json_artifact("run_identity", paths, blockers)
    if run is not None:
        _check_fields("run_identity", run, {
            "schema": "faithful-run-identity-v1",
            "physics_identity_sha256": certificate.physics_identity_sha256,
            "protocol_sha256": certificate.protocol_sha256,
        }, blockers)

    scenario = _json_artifact("scenario", paths, blockers)
    if scenario is not None and run is not None:
        _check_fields("scenario", scenario, {
            "schema_version": "record-scenario-v1",
            "vehicle_spec_sha256": run.get("vehicle_spec_sha256"),
            "track_surface_sha256": run.get("track_surface_sha256"),
            "target_lap_time_s": RECORD_TARGET_S,
        }, blockers)

    source_bundle = _json_artifact("source_bundle", paths, blockers)
    if source_bundle is not None and run is not None:
        _check_fields("source_bundle", source_bundle, {
            "schema": "faithful-source-bundle-v1",
            "source_sha256": run.get("source_sha256"),
            "dependencies_sha256": run.get("dependencies_sha256"),
        }, blockers)
        container_hash = source_bundle.get("container_sha256")
        if not isinstance(container_hash, str):
            blockers.append("source_bundle semantic mismatch: container_sha256")
        else:
            try:
                require_sha256(container_hash, "source_bundle container_sha256")
            except ValueError:
                blockers.append("source_bundle semantic mismatch: container_sha256")

    physics = _json_artifact("physics_identity", paths, blockers)
    if physics is not None and run is not None:
        _check_fields("physics_identity", physics, {
            "precision": "float64",
            "timestep_s": 0.001,
            "vehicle_spec_sha256": run.get("vehicle_spec_sha256"),
            "track_surface_sha256": run.get("track_surface_sha256"),
            "source_sha256": run.get("source_sha256"),
            "dependency_sha256": run.get("dependencies_sha256"),
        }, blockers)

    protocol = _json_artifact("protocol", paths, blockers)
    _check_fields("protocol", protocol, {
        "schema": "faithful-flying-lap-protocol-v1",
        "target_lap_s": RECORD_TARGET_S,
        "official_lap_length_m": OFFICIAL_LAP_LENGTH_M,
    }, blockers)
    if protocol is not None:
        gap = protocol.get("maximum_sample_interval_s")
        if (type(gap) not in (int, float) or isinstance(gap, bool)
                or not math.isfinite(float(gap)) or not 0 < float(gap) <= 0.1):
            blockers.append("protocol semantic mismatch: maximum_sample_interval_s")

    common = {
        "run_identity_sha256": certificate.run_identity_sha256,
        "physics_identity_sha256": certificate.physics_identity_sha256,
        "protocol_sha256": certificate.protocol_sha256,
        "scenario_sha256": certificate.scenario_sha256,
    }
    validation = _json_artifact("physics_validation", paths, blockers)
    _check_fields("physics_validation", validation, {
        "schema": "faithful-physics-validation-result-v1",
        **common,
        "passed": certificate.physics_validated,
    }, blockers)

    oracle = _json_artifact("oracle_result", paths, blockers)
    _check_fields("oracle_result", oracle, {
        "schema": "faithful-oracle-result-v1",
        **common,
        "passed": certificate.oracle_feasible,
        "physics_validation_sha256": hashes["physics_validation"],
        "lap_plan_sha256": hashes["lap_plan"],
        "authoritative_replay_sha256": hashes["authoritative_replay"],
    }, blockers)
    if oracle is not None:
        upper = oracle.get("conservative_upper_bound_s")
        agreement = oracle.get("replay_agreement_s")
        if (type(upper) not in (int, float) or isinstance(upper, bool)
                or not math.isfinite(float(upper))
                or float(upper) >= RECORD_TARGET_S):
            blockers.append("oracle_result semantic mismatch: conservative_upper_bound_s")
        if (type(agreement) not in (int, float) or isinstance(agreement, bool)
                or not math.isfinite(float(agreement))
                or not 0 <= float(agreement) <= MAX_REPLAY_AGREEMENT_S):
            blockers.append("oracle_result semantic mismatch: replay_agreement_s")
        if oracle.get("constraint_violations") != []:
            blockers.append("oracle_result semantic mismatch: constraint_violations")

    lap_protocol = _json_artifact("lap_protocol_trace", paths, blockers)
    _check_fields("lap_protocol_trace", lap_protocol, {
        "schema": "faithful-lap-protocol-result-v1",
        **common,
        "legal": certificate.legal_flying_lap,
        "no_teleport": certificate.no_teleport,
        "action_trace_sha256": certificate.action_trace_sha256,
        "telemetry_sha256": certificate.telemetry_sha256,
        "authoritative_replay_sha256": hashes["authoritative_replay"],
        "lap_time_s": certificate.lap_time_s,
    }, blockers)

    robustness = _json_artifact("robustness_result", paths, blockers)
    _check_fields("robustness_result", robustness, {
        "schema": "faithful-robustness-result-v1",
        **common,
        "robust_record": certificate.robust_record,
        "parameter_posterior_sha256": hashes["parameter_posterior"],
        "sample_count": certificate.robust_sample_count,
        "valid_fraction": certificate.robust_valid_fraction,
        "p95_lap_time_s": certificate.robust_p95_lap_time_s,
        "p95_bootstrap_upper_s": certificate.robust_p95_bootstrap_upper_s,
    }, blockers)
    return tuple(blockers)


def create_evidence_bundle_manifest(
    bundle_root: Path,
    certificate: RecordCertificateV1,
    role_paths: Mapping[str, str],
    verifier_key: bytes | Any,
) -> EvidenceBundleManifestV1:
    """Hash, cross-link and atomically write an Ed25519-signed manifest.

    ``verifier_key`` is retained as the parameter name for compatibility, but
    it must now be an Ed25519 private seed/object.  It is never serialized.
    """
    public_key = _raw_public_key(verifier_key, private=True)
    if not certificate.verify_signature(public_key):
        raise ValueError("certificate must have a valid Ed25519 signature")
    artifacts = _artifacts_for_paths(bundle_root, role_paths)
    actual_hashes = tuple((artifact.role, artifact.sha256)
                          for artifact in artifacts)
    if actual_hashes != certificate.artifact_sha256_by_role:
        raise ValueError("certificate does not cross-link every evidence artifact")
    paths = {artifact.role: _safe_bundle_file(bundle_root, artifact.path)
             for artifact in artifacts}
    semantic_blockers = _semantic_cross_link_blockers(certificate, paths)
    if semantic_blockers:
        raise ValueError("semantic evidence cross-link failure: "
                         + "; ".join(semantic_blockers))
    manifest = EvidenceBundleManifestV1(certificate, artifacts).sign(verifier_key)
    payload = json.dumps(json.loads(canonical_json_bytes(manifest)), indent=2,
                         sort_keys=True, allow_nan=False) + "\n"
    bundle_root.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".record-evidence-", dir=bundle_root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, bundle_root / MANIFEST_NAME)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return manifest


def _manifest_from_dict(payload: dict) -> EvidenceBundleManifestV1:
    return EvidenceBundleManifestV1.from_dict(payload)


def verify_evidence_bundle(bundle_root: Path, verifier_key: bytes | Any,
                           *, reject_unlisted_files: bool = True) -> BundleVerification:
    """Verify a bundle using only the independent verifier's public key."""
    blockers: list[str] = []
    manifest_path = bundle_root / MANIFEST_NAME
    if manifest_path.is_symlink():
        return BundleVerification(False, ("evidence manifest must not be a symlink",))
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _manifest_from_dict(raw)
    except Exception as exc:
        return BundleVerification(False, (f"invalid evidence manifest: {exc}",))

    if not manifest.verify_signature(verifier_key):
        blockers.append("bundle Ed25519 signature mismatch")
    if not manifest.certificate.verify_signature(verifier_key):
        blockers.append("certificate Ed25519 signature mismatch")
    if not manifest.certificate.certifies_record:
        blockers.append("certificate does not satisfy the record contract")
    if (manifest.certificate.robust_record
            and not manifest.certificate.certifies_robust_record):
        blockers.append("claimed robust record does not satisfy robustness thresholds")

    roles = {artifact.role: artifact for artifact in manifest.artifacts}
    missing_roles = REQUIRED_ROLES - set(roles)
    extra_roles = set(roles) - REQUIRED_ROLES
    if missing_roles:
        blockers.append("missing required roles: " + ", ".join(sorted(missing_roles)))
    if extra_roles:
        blockers.append("unrecognized evidence roles: " + ", ".join(sorted(extra_roles)))
    resolved_paths: dict[str, Path] = {}
    for artifact in manifest.artifacts:
        try:
            path = _safe_bundle_file(bundle_root, artifact.path)
            resolved_paths[artifact.role] = path
            if not path.is_file() or path.is_symlink():
                blockers.append(f"artifact is missing or not regular: {artifact.path}")
                continue
            if path.stat().st_size != artifact.bytes:
                blockers.append(f"artifact size mismatch: {artifact.path}")
            if not secure_digest_equal(sha256_file(path), artifact.sha256):
                blockers.append(f"artifact digest mismatch: {artifact.path}")
        except Exception as exc:
            blockers.append(f"artifact verification failed ({artifact.path}): {exc}")

    certificate_hashes = manifest.certificate.artifact_hashes
    for role in sorted(REQUIRED_ROLES):
        artifact = roles.get(role)
        if artifact and not secure_digest_equal(
                artifact.sha256, certificate_hashes[role]):
            blockers.append(f"certificate/{role} digest mismatch")
    blockers.extend(_semantic_cross_link_blockers(
        manifest.certificate, resolved_paths))

    if reject_unlisted_files:
        listed = {artifact.path for artifact in manifest.artifacts} | {MANIFEST_NAME}
        actual = {
            path.relative_to(bundle_root).as_posix()
            for path in bundle_root.rglob("*") if path.is_file()
        }
        extras = actual - listed
        if extras:
            blockers.append("unlisted files in evidence bundle: "
                            + ", ".join(sorted(extras)))

    return BundleVerification(not blockers, tuple(blockers), sha256_object(manifest))
