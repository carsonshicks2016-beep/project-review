"""Signed, external evidence-vault primitives for faithful-v2.

Licensed source material never belongs in the repository.  This module treats
an external directory as an immutable content-addressed vault, verifies
Ed25519-signed package manifests against a protocol-owned trust store, and
creates signed freezes that bind the exact physical inputs used by a run.

The implementation deliberately distinguishes structurally valid fixtures
from certification-eligible licensed packages.  A fixture remains
noncertifying even when correctly signed and complete.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from typing import Any, Iterable, Mapping

try:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey,
        Ed25519PublicKey,
    )
except ImportError:  # pragma: no cover - fail-closed boundary
    InvalidSignature = None  # type: ignore[assignment]
    serialization = None  # type: ignore[assignment]
    Ed25519PrivateKey = None  # type: ignore[assignment,misc]
    Ed25519PublicKey = None  # type: ignore[assignment,misc]


PACKAGE_SCHEMA = "faithful-evidence-package-v1"
FREEZE_SCHEMA = "faithful-evidence-freeze-v1"
TRUST_STORE_SCHEMA = "faithful-evidence-trust-store-v1"
AUDIT_SCHEMA = "faithful-holdout-audit-event-v1"
SIGNATURE_SCHEME = "ed25519-v1"
PACKAGE_MANIFEST = "EVIDENCE_PACKAGE.json"
FREEZE_MANIFEST = "EVIDENCE_FREEZE.json"
HOLDOUT_AUDIT = "HOLDOUT_AUDIT.jsonl"
PACKAGE_SIGNATURE_DOMAIN = b"SUPRA/FAITHFUL-V2/EVIDENCE-PACKAGE/V1\x00"
FREEZE_SIGNATURE_DOMAIN = b"SUPRA/FAITHFUL-V2/EVIDENCE-FREEZE/V1\x00"
AUDIT_SIGNATURE_DOMAIN = b"SUPRA/FAITHFUL-V2/HOLDOUT-AUDIT/V1\x00"
ZERO_SHA256 = "0" * 64

_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_MEDIA_TYPE = re.compile(r"^[a-z0-9.+-]+/[a-z0-9.+-]+$")

# These roles are protocol-owned.  A caller cannot rename a convenient data
# file to bypass an absent physical evidence class.
REQUIRED_EVIDENCE_ROLES = frozenset({
    "vehicle.mass-properties",
    "vehicle.suspension-pitch-link",
    "tyre.michelin-310-710-18",
    "vehicle.aero-active-controls",
    "vehicle.ice-gearbox-driveline",
    "vehicle.mgu-battery-exhaust-ers",
    "vehicle.brakes-embedded-controls",
    "track.nordschleife-june-2018-survey",
    "calibration.component-spa",
    "holdout.nordschleife-record-telemetry",
    "scenario.record-conditions-preparation",
})
HOLDOUT_ROLE = "holdout.nordschleife-record-telemetry"


def _actual_bool(value: Any, field: str) -> None:
    if type(value) is not bool:
        raise TypeError(f"{field} must be an actual bool")


def _require_identifier(value: str, field: str) -> None:
    if not isinstance(value, str) or not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase artifact-safe identifier")


def _require_sha256(value: str, field: str, *, allow_zero: bool = False) -> None:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    if not allow_zero and value == ZERO_SHA256:
        raise ValueError(f"{field} must not be the zero digest")


def _utc_timestamp(value: str, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short write")
        remaining = remaining[written:]


def _safe_relative_path(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path is required")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix():
        raise ValueError(f"artifact path must be normalized and relative: {value!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe artifact path: {value!r}")
    if path.name in {PACKAGE_MANIFEST, FREEZE_MANIFEST, HOLDOUT_AUDIT}:
        raise ValueError(f"artifact path uses a reserved manifest name: {value!r}")
    return path


def _assert_no_symlink_components(root: Path, relative: PurePosixPath) -> Path:
    candidate = root
    for part in relative.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"evidence path contains a symlink: {relative.as_posix()}")
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError(f"evidence path escapes package: {relative.as_posix()}")
    return resolved


def _require_ed25519() -> None:
    if Ed25519PrivateKey is None or Ed25519PublicKey is None:
        raise RuntimeError("cryptography with Ed25519 support is required")


def _private_key(value: bytes | Any) -> Any:
    _require_ed25519()
    if isinstance(value, Ed25519PrivateKey):
        return value
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError("Ed25519 private key must be a raw 32-byte seed")
    return Ed25519PrivateKey.from_private_bytes(value)


def _public_key(value: bytes | Any) -> Any:
    _require_ed25519()
    if isinstance(value, Ed25519PublicKey):
        return value
    if not isinstance(value, bytes) or len(value) != 32:
        raise ValueError("Ed25519 public key must contain exactly 32 raw bytes")
    return Ed25519PublicKey.from_public_bytes(value)


def _raw_public_key(value: bytes | Any, *, private: bool = False) -> bytes:
    key = _private_key(value).public_key() if private else _public_key(value)
    return key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )


def generate_evidence_keypair() -> tuple[bytes, bytes]:
    """Return ``(private_seed, public_key)`` for an external custodian."""
    _require_ed25519()
    private = Ed25519PrivateKey.generate()
    seed = private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return seed, _raw_public_key(private, private=True)


def load_private_key_file(path: str | Path,
                          *, forbidden_root: str | Path | None = None) -> bytes:
    """Load a raw or lowercase-hex private seed without printing it."""
    candidate = Path(path).expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise ValueError("private key path must be a regular non-symlink file")
    resolved = candidate.resolve()
    if forbidden_root is not None:
        forbidden = Path(forbidden_root).expanduser().resolve()
        if resolved == forbidden or forbidden in resolved.parents:
            raise ValueError("private key must remain outside the repository")
    raw = candidate.read_bytes()
    if len(raw) == 32:
        return raw
    try:
        text = raw.decode("ascii").strip()
        decoded = bytes.fromhex(text)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("private key file must contain 32 raw bytes or 64 hex characters") from exc
    if len(decoded) != 32 or text != text.lower():
        raise ValueError("private key file must contain 32 raw bytes or lowercase hex")
    return decoded


def public_key_id(public_key: bytes | Any) -> str:
    return _sha256_bytes(_raw_public_key(public_key))


def _sign(private_key: bytes | Any, domain: bytes, payload: Mapping[str, Any]) -> str:
    return _private_key(private_key).sign(domain + _canonical_bytes(payload)).hex()


def _verify(public_key: bytes | Any, signature: str, domain: bytes,
            payload: Mapping[str, Any]) -> bool:
    if not isinstance(signature, str) or len(signature) != 128:
        return False
    try:
        raw_signature = bytes.fromhex(signature)
        _public_key(public_key).verify(raw_signature, domain + _canonical_bytes(payload))
        return True
    except (ValueError, TypeError):
        return False
    except Exception as exc:
        if InvalidSignature is not None and isinstance(exc, InvalidSignature):
            return False
        raise


@dataclass(frozen=True, slots=True)
class TrustedEvidenceKeyV1:
    key_id: str
    public_key_hex: str
    label: str
    status: str
    permitted_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_sha256(self.key_id, "key_id")
        if not isinstance(self.public_key_hex, str) or len(self.public_key_hex) != 64:
            raise ValueError("public_key_hex must contain 32 bytes")
        try:
            public = bytes.fromhex(self.public_key_hex)
        except ValueError as exc:
            raise ValueError("public_key_hex must be hexadecimal") from exc
        if self.public_key_hex != self.public_key_hex.lower() or len(public) != 32:
            raise ValueError("public_key_hex must contain lowercase raw Ed25519 bytes")
        if public_key_id(public) != self.key_id:
            raise ValueError("trusted key_id does not match public key")
        if not self.label:
            raise ValueError("trusted key label is required")
        if self.status not in {"active", "revoked"}:
            raise ValueError("trusted key status must be active or revoked")
        if not self.permitted_kinds or any(
            kind not in {"licensed", "fixture", "freeze", "holdout-audit"}
            for kind in self.permitted_kinds
        ):
            raise ValueError("trusted key has invalid permitted_kinds")
        if len(set(self.permitted_kinds)) != len(self.permitted_kinds):
            raise ValueError("trusted key permitted_kinds must be unique")

    @property
    def public_key(self) -> bytes:
        return bytes.fromhex(self.public_key_hex)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key_id": self.key_id,
            "public_key_hex": self.public_key_hex,
            "label": self.label,
            "status": self.status,
            "permitted_kinds": list(self.permitted_kinds),
        }


@dataclass(frozen=True, slots=True)
class EvidenceTrustStoreV1:
    schema: str
    keys: tuple[TrustedEvidenceKeyV1, ...]

    def __post_init__(self) -> None:
        if self.schema != TRUST_STORE_SCHEMA:
            raise ValueError(f"trust store schema must be {TRUST_STORE_SCHEMA}")
        ids = [key.key_id for key in self.keys]
        if len(ids) != len(set(ids)):
            raise ValueError("trust store key IDs must be unique")

    def key_for(self, key_id: str, kind: str) -> TrustedEvidenceKeyV1 | None:
        for key in self.keys:
            if (key.key_id == key_id and key.status == "active"
                    and kind in key.permitted_kinds):
                return key
        return None

    def to_dict(self) -> dict[str, Any]:
        return {"schema": self.schema, "keys": [key.to_dict() for key in self.keys]}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceTrustStoreV1":
        if set(value) != {"schema", "keys"} or not isinstance(value.get("keys"), list):
            raise ValueError("invalid evidence trust store fields")
        keys = []
        for raw in value["keys"]:
            if not isinstance(raw, dict) or set(raw) != {
                "key_id", "public_key_hex", "label", "status", "permitted_kinds"
            }:
                raise ValueError("invalid trusted evidence key fields")
            kinds = raw["permitted_kinds"]
            if not isinstance(kinds, list) or not all(isinstance(x, str) for x in kinds):
                raise ValueError("permitted_kinds must be a string list")
            keys.append(TrustedEvidenceKeyV1(
                key_id=raw["key_id"], public_key_hex=raw["public_key_hex"],
                label=raw["label"], status=raw["status"],
                permitted_kinds=tuple(kinds),
            ))
        return cls(schema=value["schema"], keys=tuple(keys))

    @classmethod
    def load(cls, path: str | Path) -> "EvidenceTrustStoreV1":
        candidate = Path(path)
        if candidate.is_symlink() or not candidate.is_file():
            raise ValueError("trust store must be a regular non-symlink file")
        value = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("trust store root must be an object")
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class EvidenceArtifactV1:
    artifact_id: str
    role: str
    path: str
    sha256: str
    size_bytes: int
    media_type: str
    units: str
    source_uri: str
    license_id: str
    redistribution: str
    confidence: str
    calibration_role: str
    high_sensitivity: bool
    holdout_sealed: bool

    def __post_init__(self) -> None:
        _require_identifier(self.artifact_id, "artifact_id")
        if self.role not in REQUIRED_EVIDENCE_ROLES:
            raise ValueError(f"unrecognized protocol evidence role: {self.role!r}")
        _safe_relative_path(self.path)
        _require_sha256(self.sha256, "artifact sha256")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise ValueError("artifact size_bytes must be a nonnegative integer")
        if not isinstance(self.media_type, str) or not _MEDIA_TYPE.fullmatch(self.media_type):
            raise ValueError("artifact media_type must be a normalized MIME type")
        for field in ("units", "source_uri", "license_id", "redistribution"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"artifact {field} is required")
        if self.confidence not in {
            "official", "licensed-measured", "calibrated", "inferred"
        }:
            raise ValueError("artifact confidence is invalid")
        if self.calibration_role not in {
            "source", "calibration", "holdout", "scenario"
        }:
            raise ValueError("artifact calibration_role is invalid")
        _actual_bool(self.high_sensitivity, "artifact high_sensitivity")
        _actual_bool(self.holdout_sealed, "artifact holdout_sealed")
        if self.role == HOLDOUT_ROLE:
            if self.calibration_role != "holdout" or not self.holdout_sealed:
                raise ValueError("Nord holdout evidence must be sealed and marked holdout")
        elif self.calibration_role == "holdout" or self.holdout_sealed:
            raise ValueError("only the protocol holdout role may be sealed as holdout")

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id, "role": self.role,
            "path": self.path, "sha256": self.sha256,
            "size_bytes": self.size_bytes, "media_type": self.media_type,
            "units": self.units, "source_uri": self.source_uri,
            "license_id": self.license_id, "redistribution": self.redistribution,
            "confidence": self.confidence, "calibration_role": self.calibration_role,
            "high_sensitivity": self.high_sensitivity,
            "holdout_sealed": self.holdout_sealed,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceArtifactV1":
        fields = {
            "artifact_id", "role", "path", "sha256", "size_bytes", "media_type", "units",
            "source_uri", "license_id", "redistribution", "confidence",
            "calibration_role", "high_sensitivity", "holdout_sealed",
        }
        if set(value) != fields:
            raise ValueError("invalid evidence artifact fields")
        return cls(**{field: value[field] for field in fields})


@dataclass(frozen=True, slots=True)
class EvidencePackageV1:
    schema: str
    package_id: str
    package_kind: str
    created_at_utc: str
    intended_use: str
    signer_key_id: str
    artifacts: tuple[EvidenceArtifactV1, ...]
    signature_scheme: str
    signature: str

    def __post_init__(self) -> None:
        if self.schema != PACKAGE_SCHEMA:
            raise ValueError(f"evidence package schema must be {PACKAGE_SCHEMA}")
        _require_identifier(self.package_id, "package_id")
        if self.package_kind not in {"licensed", "fixture"}:
            raise ValueError("package_kind must be licensed or fixture")
        _utc_timestamp(self.created_at_utc, "created_at_utc")
        if self.intended_use != "private-research":
            raise ValueError("faithful evidence intended_use must be private-research")
        _require_sha256(self.signer_key_id, "signer_key_id")
        if self.signature_scheme != SIGNATURE_SCHEME:
            raise ValueError(f"signature_scheme must be {SIGNATURE_SCHEME}")
        if not isinstance(self.signature, str) or len(self.signature) != 128:
            raise ValueError("package signature must be 128 hex characters")
        try:
            bytes.fromhex(self.signature)
        except ValueError as exc:
            raise ValueError("package signature must be hexadecimal") from exc
        artifact_ids = [artifact.artifact_id for artifact in self.artifacts]
        paths = [artifact.path for artifact in self.artifacts]
        if not self.artifacts:
            raise ValueError("evidence package must contain at least one artifact")
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("evidence package artifact IDs must be unique")
        if len(paths) != len(set(paths)):
            raise ValueError("evidence package paths must be unique")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "package_id": self.package_id,
            "package_kind": self.package_kind, "created_at_utc": self.created_at_utc,
            "intended_use": self.intended_use, "signer_key_id": self.signer_key_id,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "signature_scheme": self.signature_scheme,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidencePackageV1":
        fields = {
            "schema", "package_id", "package_kind", "created_at_utc",
            "intended_use", "signer_key_id", "artifacts",
            "signature_scheme", "signature",
        }
        if set(value) != fields or not isinstance(value.get("artifacts"), list):
            raise ValueError("invalid evidence package fields")
        artifacts = tuple(
            EvidenceArtifactV1.from_dict(raw) if isinstance(raw, dict)
            else (_ for _ in ()).throw(ValueError("artifact must be an object"))
            for raw in value["artifacts"]
        )
        return cls(
            schema=value["schema"], package_id=value["package_id"],
            package_kind=value["package_kind"], created_at_utc=value["created_at_utc"],
            intended_use=value["intended_use"], signer_key_id=value["signer_key_id"],
            artifacts=artifacts, signature_scheme=value["signature_scheme"],
            signature=value["signature"],
        )


@dataclass(frozen=True, slots=True)
class EvidenceVerificationV1:
    valid: bool
    certification_eligible: bool
    package_id: str | None
    roles: tuple[str, ...]
    blockers: tuple[str, ...]


def build_signed_package(package_id: str, package_kind: str,
                         artifacts: Iterable[EvidenceArtifactV1],
                         private_key: bytes | Any,
                         *, created_at_utc: str | None = None) -> EvidencePackageV1:
    public = _raw_public_key(private_key, private=True)
    placeholder = EvidencePackageV1(
        schema=PACKAGE_SCHEMA, package_id=package_id, package_kind=package_kind,
        created_at_utc=created_at_utc or datetime.now(timezone.utc).isoformat(),
        intended_use="private-research", signer_key_id=public_key_id(public),
        artifacts=tuple(artifacts), signature_scheme=SIGNATURE_SCHEME,
        signature="0" * 128,
    )
    signature = _sign(private_key, PACKAGE_SIGNATURE_DOMAIN, placeholder.unsigned_payload())
    return EvidencePackageV1(**{
        **placeholder.unsigned_payload(), "artifacts": placeholder.artifacts,
        "signature": signature,
    })


def write_package_manifest(package_root: str | Path,
                           package: EvidencePackageV1) -> Path:
    root = Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("package root must be a regular non-symlink directory")
    path = root / PACKAGE_MANIFEST
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to replace package manifest: {path}")
    encoded = json.dumps(package.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        _write_all(descriptor, encoded.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return path


def load_package(package_root: str | Path) -> EvidencePackageV1:
    root = Path(package_root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("package root must be a regular non-symlink directory")
    manifest = root / PACKAGE_MANIFEST
    if manifest.is_symlink() or not manifest.is_file():
        raise ValueError(f"missing regular {PACKAGE_MANIFEST}")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("evidence package manifest root must be an object")
    return EvidencePackageV1.from_dict(value)


def verify_package_directory(package_root: str | Path,
                             trust_store: EvidenceTrustStoreV1) -> EvidenceVerificationV1:
    blockers: list[str] = []
    package: EvidencePackageV1 | None = None
    try:
        package = load_package(package_root)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return EvidenceVerificationV1(False, False, None, (), (f"invalid package: {exc}",))
    root = Path(package_root).resolve()
    key = trust_store.key_for(package.signer_key_id, package.package_kind)
    if key is None:
        blockers.append("package signer is absent, revoked, or not permitted for this package kind")
    elif not _verify(key.public_key, package.signature, PACKAGE_SIGNATURE_DOMAIN,
                     package.unsigned_payload()):
        blockers.append("package signature is invalid")

    listed = {PACKAGE_MANIFEST}
    for artifact in package.artifacts:
        relative = _safe_relative_path(artifact.path)
        listed.add(relative.as_posix())
        try:
            path = _assert_no_symlink_components(root, relative)
        except ValueError as exc:
            blockers.append(str(exc))
            continue
        if not path.is_file():
            blockers.append(f"artifact is not a regular file: {artifact.path}")
            continue
        if path.stat().st_size != artifact.size_bytes:
            blockers.append(f"artifact size mismatch: {artifact.path}")
        if _sha256_file(path) != artifact.sha256:
            blockers.append(f"artifact hash mismatch: {artifact.path}")
        if artifact.high_sensitivity and artifact.confidence == "inferred":
            blockers.append(f"high-sensitivity artifact is inferred: {artifact.role}")
        if package.package_kind == "licensed" and artifact.license_id.startswith("fixture"):
            blockers.append(f"licensed package uses a fixture licence: {artifact.role}")

    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            blockers.append(f"unlisted or listed symlink in package: {path.relative_to(root)}")
        elif path.is_file():
            actual.add(path.relative_to(root).as_posix())
    extras = sorted(actual - listed)
    missing = sorted(listed - actual)
    if extras:
        blockers.append("unlisted files in evidence package: " + ", ".join(extras))
    if missing:
        blockers.append("listed files missing from evidence package: " + ", ".join(missing))

    valid = not blockers
    eligible = valid and package.package_kind == "licensed"
    return EvidenceVerificationV1(
        valid, eligible, package.package_id,
        tuple(sorted(artifact.role for artifact in package.artifacts)),
        tuple(blockers),
    )


def resolve_evidence_vault(project_root: str | Path,
                           configured_root: str | Path | None = None,
                           *, create: bool = False) -> Path:
    raw = configured_root if configured_root is not None else os.environ.get(
        "FAITHFUL_EVIDENCE_ROOT"
    )
    if raw is None:
        raise ValueError("set FAITHFUL_EVIDENCE_ROOT or pass --faithful-evidence-root")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise ValueError("faithful evidence root must be an absolute path")
    project = Path(project_root).expanduser().resolve()
    # Resolve through the nearest existing parent so a not-yet-created path
    # cannot hide beneath a symlink into the repository.
    ancestor = candidate
    suffix: list[str] = []
    while not ancestor.exists():
        suffix.append(ancestor.name)
        ancestor = ancestor.parent
    if ancestor.is_symlink():
        raise ValueError("evidence root ancestor must not be a symlink")
    resolved = ancestor.resolve().joinpath(*reversed(suffix))
    if resolved == project or project in resolved.parents:
        raise ValueError("faithful evidence vault must be outside the repository")
    if candidate.exists() and candidate.is_symlink():
        raise ValueError("faithful evidence root must not be a symlink")
    if create:
        resolved.mkdir(mode=0o700, parents=True, exist_ok=True)
        for name in ("packages", "freezes", "audit"):
            (resolved / name).mkdir(mode=0o700, exist_ok=True)
    if not resolved.is_dir():
        raise ValueError("faithful evidence root does not exist")
    metadata = resolved.stat()
    if metadata.st_uid != os.getuid():
        raise PermissionError("faithful evidence root must be owned by the current user")
    if metadata.st_mode & 0o077:
        raise PermissionError(
            "faithful evidence root must not grant group or other permissions"
        )
    for name in ("packages", "freezes", "audit"):
        child = resolved / name
        if child.exists():
            if child.is_symlink() or not child.is_dir():
                raise ValueError(f"evidence vault {name} must be a regular directory")
            child_metadata = child.stat()
            if child_metadata.st_uid != os.getuid() or child_metadata.st_mode & 0o077:
                raise PermissionError(
                    f"evidence vault {name} must be private and current-user owned"
                )
    return resolved


def ingest_evidence_package(source_root: str | Path, vault_root: str | Path,
                            trust_store: EvidenceTrustStoreV1) -> Path:
    source = Path(source_root)
    verification = verify_package_directory(source, trust_store)
    if not verification.valid or verification.package_id is None:
        raise ValueError("evidence package verification failed: " + "; ".join(verification.blockers))
    vault = Path(vault_root).resolve()
    packages = vault / "packages"
    if not packages.is_dir() or packages.is_symlink():
        raise ValueError("evidence vault packages directory is invalid")
    target = packages / verification.package_id
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"evidence package already exists: {verification.package_id}")
    temporary = Path(tempfile.mkdtemp(prefix=".ingest-", dir=packages))
    try:
        package = load_package(source)
        for artifact in package.artifacts:
            relative = _safe_relative_path(artifact.path)
            src = _assert_no_symlink_components(source.resolve(), relative)
            dst = temporary.joinpath(*relative.parts)
            dst.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            shutil.copyfile(src, dst, follow_symlinks=False)
            os.chmod(dst, 0o400)
        shutil.copyfile(source / PACKAGE_MANIFEST, temporary / PACKAGE_MANIFEST,
                        follow_symlinks=False)
        os.chmod(temporary / PACKAGE_MANIFEST, 0o400)
        copied = verify_package_directory(temporary, trust_store)
        if not copied.valid:
            raise ValueError("copied evidence failed verification: " + "; ".join(copied.blockers))
        os.replace(temporary, target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def evidence_inventory(vault_root: str | Path,
                       trust_store: EvidenceTrustStoreV1) -> dict[str, Any]:
    vault = Path(vault_root).resolve()
    packages = vault / "packages"
    if packages.is_symlink():
        raise ValueError("evidence vault packages directory must not be a symlink")
    rows = []
    observed_roles: set[str] = set()
    if packages.is_dir():
        for path in sorted(packages.iterdir(), key=lambda item: item.name):
            if not path.is_dir() or path.is_symlink():
                rows.append({
                    "package_id": path.name, "valid": False,
                    "certification_eligible": False,
                    "roles": [], "blockers": ["invalid package directory"],
                })
                continue
            result = verify_package_directory(path, trust_store)
            if result.valid:
                observed_roles.update(result.roles)
            rows.append({
                "package_id": result.package_id or path.name,
                "valid": result.valid,
                "certification_eligible": result.certification_eligible,
                "roles": list(result.roles), "blockers": list(result.blockers),
            })
    return {
        "schema": "faithful-evidence-inventory-v1",
        "vault_root": str(vault), "packages": rows,
        "required_roles": sorted(REQUIRED_EVIDENCE_ROLES),
        "observed_roles": sorted(observed_roles),
        "missing_roles": sorted(REQUIRED_EVIDENCE_ROLES - observed_roles),
        "certification_ready": (
            not (REQUIRED_EVIDENCE_ROLES - observed_roles)
            and bool(rows)
            and all(row["valid"] and row["certification_eligible"] for row in rows)
        ),
    }


@dataclass(frozen=True, slots=True)
class EvidenceFreezeV1:
    schema: str
    freeze_id: str
    created_at_utc: str
    signer_key_id: str
    package_manifest_sha256: tuple[tuple[str, str], ...]
    vehicle_spec_sha256: str
    track_surface_sha256: str
    scenario_sha256: str
    parameter_registry_sha256: str
    signature_scheme: str
    signature: str

    def __post_init__(self) -> None:
        if self.schema != FREEZE_SCHEMA:
            raise ValueError(f"freeze schema must be {FREEZE_SCHEMA}")
        _require_identifier(self.freeze_id, "freeze_id")
        _utc_timestamp(self.created_at_utc, "created_at_utc")
        _require_sha256(self.signer_key_id, "signer_key_id")
        package_ids = []
        for package_id, digest in self.package_manifest_sha256:
            _require_identifier(package_id, "freeze package_id")
            _require_sha256(digest, "freeze package manifest sha256")
            package_ids.append(package_id)
        if not package_ids or package_ids != sorted(package_ids) or len(package_ids) != len(set(package_ids)):
            raise ValueError("freeze packages must be a nonempty unique sorted tuple")
        for field in (
            "vehicle_spec_sha256", "track_surface_sha256", "scenario_sha256",
            "parameter_registry_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if self.signature_scheme != SIGNATURE_SCHEME:
            raise ValueError(f"signature_scheme must be {SIGNATURE_SCHEME}")
        if not isinstance(self.signature, str) or len(self.signature) != 128:
            raise ValueError("freeze signature must be 128 hex characters")
        try:
            bytes.fromhex(self.signature)
        except ValueError as exc:
            raise ValueError("freeze signature must be hexadecimal") from exc

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "freeze_id": self.freeze_id,
            "created_at_utc": self.created_at_utc,
            "signer_key_id": self.signer_key_id,
            "package_manifest_sha256": {
                key: value for key, value in self.package_manifest_sha256
            },
            "vehicle_spec_sha256": self.vehicle_spec_sha256,
            "track_surface_sha256": self.track_surface_sha256,
            "scenario_sha256": self.scenario_sha256,
            "parameter_registry_sha256": self.parameter_registry_sha256,
            "signature_scheme": self.signature_scheme,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "signature": self.signature}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceFreezeV1":
        fields = {
            "schema", "freeze_id", "created_at_utc", "signer_key_id",
            "package_manifest_sha256", "vehicle_spec_sha256",
            "track_surface_sha256", "scenario_sha256",
            "parameter_registry_sha256", "signature_scheme", "signature",
        }
        packages = value.get("package_manifest_sha256")
        if set(value) != fields or not isinstance(packages, dict):
            raise ValueError("invalid evidence freeze fields")
        if not all(isinstance(k, str) and isinstance(v, str) for k, v in packages.items()):
            raise ValueError("freeze package hashes must be a string map")
        return cls(
            schema=value["schema"], freeze_id=value["freeze_id"],
            created_at_utc=value["created_at_utc"], signer_key_id=value["signer_key_id"],
            package_manifest_sha256=tuple(sorted(packages.items())),
            vehicle_spec_sha256=value["vehicle_spec_sha256"],
            track_surface_sha256=value["track_surface_sha256"],
            scenario_sha256=value["scenario_sha256"],
            parameter_registry_sha256=value["parameter_registry_sha256"],
            signature_scheme=value["signature_scheme"], signature=value["signature"],
        )

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.to_dict()))


@dataclass(frozen=True, slots=True)
class FreezeVerificationV1:
    valid: bool
    certification_eligible: bool
    freeze_id: str | None
    freeze_sha256: str | None
    missing_roles: tuple[str, ...]
    blockers: tuple[str, ...]


def create_evidence_freeze(vault_root: str | Path, freeze_id: str,
                           package_ids: Iterable[str], trust_store: EvidenceTrustStoreV1,
                           private_key: bytes | Any, *, vehicle_spec_sha256: str,
                           track_surface_sha256: str, scenario_sha256: str,
                           parameter_registry_sha256: str) -> Path:
    _require_identifier(freeze_id, "freeze_id")
    vault = Path(vault_root).resolve()
    packages_root = vault / "packages"
    if packages_root.is_symlink() or not packages_root.is_dir():
        raise ValueError("evidence vault packages directory is invalid")
    package_hashes = []
    observed_roles: set[str] = set()
    for package_id in sorted(set(package_ids)):
        _require_identifier(package_id, "package_id")
        package_root = packages_root / package_id
        result = verify_package_directory(package_root, trust_store)
        if not result.valid:
            raise ValueError(f"cannot freeze invalid package {package_id}: " + "; ".join(result.blockers))
        observed_roles.update(result.roles)
        package_hashes.append((package_id, _sha256_file(package_root / PACKAGE_MANIFEST)))
    if not package_hashes:
        raise ValueError("freeze requires at least one package")
    missing_roles = REQUIRED_EVIDENCE_ROLES - observed_roles
    if missing_roles:
        raise ValueError(
            "cannot freeze incomplete evidence roles: " + ", ".join(sorted(missing_roles))
        )
    public = _raw_public_key(private_key, private=True)
    if trust_store.key_for(public_key_id(public), "freeze") is None:
        raise PermissionError("freeze signer is not trusted for freeze manifests")
    placeholder = EvidenceFreezeV1(
        schema=FREEZE_SCHEMA, freeze_id=freeze_id,
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        signer_key_id=public_key_id(public),
        package_manifest_sha256=tuple(package_hashes),
        vehicle_spec_sha256=vehicle_spec_sha256,
        track_surface_sha256=track_surface_sha256,
        scenario_sha256=scenario_sha256,
        parameter_registry_sha256=parameter_registry_sha256,
        signature_scheme=SIGNATURE_SCHEME, signature="0" * 128,
    )
    freeze = EvidenceFreezeV1(**{
        **placeholder.unsigned_payload(),
        "package_manifest_sha256": placeholder.package_manifest_sha256,
        "signature": _sign(private_key, FREEZE_SIGNATURE_DOMAIN,
                           placeholder.unsigned_payload()),
    })
    freezes = vault / "freezes"
    if not freezes.is_dir() or freezes.is_symlink():
        raise ValueError("evidence vault freezes directory is invalid")
    destination = freezes / freeze_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"evidence freeze already exists: {freeze_id}")
    destination.mkdir(mode=0o700)
    path = destination / FREEZE_MANIFEST
    encoded = json.dumps(freeze.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            _write_all(descriptor, encoded.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        verification = verify_evidence_freeze(vault, freeze_id, trust_store)
        if not verification.valid:
            raise RuntimeError(
                "new evidence freeze failed self-verification: "
                + "; ".join(verification.blockers)
            )
    except Exception:
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return path


def verify_evidence_freeze(vault_root: str | Path, freeze_id: str,
                           trust_store: EvidenceTrustStoreV1) -> FreezeVerificationV1:
    blockers: list[str] = []
    roles: set[str] = set()
    vault = Path(vault_root).resolve()
    freeze_root = vault / "freezes" / freeze_id
    path = freeze_root / FREEZE_MANIFEST
    try:
        if freeze_root.is_symlink():
            raise ValueError("freeze directory must not be a symlink")
        resolved_freeze = freeze_root.resolve()
        if resolved_freeze != vault and vault not in resolved_freeze.parents:
            raise ValueError("freeze directory escapes the evidence vault")
        if path.is_symlink() or not path.is_file():
            raise ValueError("freeze manifest is not a regular file")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("freeze manifest root must be an object")
        freeze = EvidenceFreezeV1.from_dict(value)
        if freeze.freeze_id != freeze_id:
            raise ValueError("freeze ID does not match its directory")
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return FreezeVerificationV1(False, False, None, None,
                                    tuple(sorted(REQUIRED_EVIDENCE_ROLES)),
                                    (f"invalid freeze: {exc}",))
    key = trust_store.key_for(freeze.signer_key_id, "freeze")
    if key is None:
        blockers.append("freeze signer is absent, revoked, or not permitted")
    elif not _verify(key.public_key, freeze.signature, FREEZE_SIGNATURE_DOMAIN,
                     freeze.unsigned_payload()):
        blockers.append("freeze signature is invalid")

    all_packages_eligible = True
    for package_id, expected in freeze.package_manifest_sha256:
        package_root = vault / "packages" / package_id
        manifest = package_root / PACKAGE_MANIFEST
        if manifest.is_symlink() or not manifest.is_file():
            blockers.append(f"freeze package manifest missing: {package_id}")
            all_packages_eligible = False
            continue
        if _sha256_file(manifest) != expected:
            blockers.append(f"freeze package manifest hash mismatch: {package_id}")
        result = verify_package_directory(package_root, trust_store)
        if not result.valid:
            blockers.append(f"freeze package invalid: {package_id}: " + "; ".join(result.blockers))
        if not result.certification_eligible:
            all_packages_eligible = False
        for role in result.roles:
            roles.add(role)
    missing = tuple(sorted(REQUIRED_EVIDENCE_ROLES - roles))
    if missing:
        blockers.append("freeze is missing required evidence roles: " + ", ".join(missing))
    valid = not blockers
    eligible = valid and all_packages_eligible and not missing
    return FreezeVerificationV1(
        valid, eligible, freeze.freeze_id, freeze.sha256, missing, tuple(blockers)
    )


@dataclass(frozen=True, slots=True)
class HoldoutAuditEventV1:
    schema: str
    sequence: int
    created_at_utc: str
    event: str
    freeze_id: str
    freeze_sha256: str
    actor: str
    prior_event_sha256: str
    signer_key_id: str
    signature_scheme: str
    signature: str

    def __post_init__(self) -> None:
        if self.schema != AUDIT_SCHEMA:
            raise ValueError(f"audit schema must be {AUDIT_SCHEMA}")
        if type(self.sequence) is not int or self.sequence < 1:
            raise ValueError("audit sequence must be a positive integer")
        _utc_timestamp(self.created_at_utc, "created_at_utc")
        if self.event != "holdout-opened":
            raise ValueError("unsupported holdout audit event")
        _require_identifier(self.freeze_id, "freeze_id")
        _require_sha256(self.freeze_sha256, "freeze_sha256")
        if not isinstance(self.actor, str) or not self.actor.strip():
            raise ValueError("holdout audit actor is required")
        _require_sha256(self.prior_event_sha256, "prior_event_sha256", allow_zero=True)
        _require_sha256(self.signer_key_id, "signer_key_id")
        if self.signature_scheme != SIGNATURE_SCHEME:
            raise ValueError(f"signature_scheme must be {SIGNATURE_SCHEME}")
        if not isinstance(self.signature, str) or len(self.signature) != 128:
            raise ValueError("audit signature must be 128 hex characters")

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema, "sequence": self.sequence,
            "created_at_utc": self.created_at_utc, "event": self.event,
            "freeze_id": self.freeze_id, "freeze_sha256": self.freeze_sha256,
            "actor": self.actor, "prior_event_sha256": self.prior_event_sha256,
            "signer_key_id": self.signer_key_id,
            "signature_scheme": self.signature_scheme,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_payload(), "signature": self.signature}

    @property
    def sha256(self) -> str:
        return _sha256_bytes(_canonical_bytes(self.to_dict()))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HoldoutAuditEventV1":
        fields = {
            "schema", "sequence", "created_at_utc", "event", "freeze_id",
            "freeze_sha256", "actor", "prior_event_sha256", "signer_key_id",
            "signature_scheme", "signature",
        }
        if set(value) != fields:
            raise ValueError("invalid holdout audit event fields")
        return cls(**{field: value[field] for field in fields})


def _parse_audit_text(raw: str) -> list[HoldoutAuditEventV1]:
    events = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        try:
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError("event must be an object")
            events.append(HoldoutAuditEventV1.from_dict(value))
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid holdout audit line {line_number}: {exc}") from exc
    return events


def _read_audit(vault: Path) -> list[HoldoutAuditEventV1]:
    path = vault / "audit" / HOLDOUT_AUDIT
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise ValueError("holdout audit must be a regular non-symlink file")
    return _parse_audit_text(path.read_text(encoding="utf-8"))


def _audit_blockers(events: list[HoldoutAuditEventV1],
                    trust_store: EvidenceTrustStoreV1) -> list[str]:
    blockers: list[str] = []
    previous = ZERO_SHA256
    opened: set[str] = set()
    for expected_sequence, event in enumerate(events, 1):
        if event.sequence != expected_sequence:
            blockers.append("holdout audit sequence is not contiguous")
        if event.prior_event_sha256 != previous:
            blockers.append(f"holdout audit chain mismatch at sequence {event.sequence}")
        if event.freeze_id in opened:
            blockers.append(f"holdout opened more than once for freeze {event.freeze_id}")
        opened.add(event.freeze_id)
        key = trust_store.key_for(event.signer_key_id, "holdout-audit")
        if key is None:
            blockers.append(f"untrusted holdout audit signer at sequence {event.sequence}")
        elif not _verify(key.public_key, event.signature, AUDIT_SIGNATURE_DOMAIN,
                         event.unsigned_payload()):
            blockers.append(f"invalid holdout audit signature at sequence {event.sequence}")
        previous = event.sha256
    return blockers


def verify_holdout_audit(vault_root: str | Path,
                         trust_store: EvidenceTrustStoreV1) -> tuple[bool, tuple[str, ...]]:
    vault = Path(vault_root).resolve()
    try:
        events = _read_audit(vault)
    except ValueError as exc:
        return False, (str(exc),)
    blockers = _audit_blockers(events, trust_store)
    return not blockers, tuple(blockers)


def append_holdout_open_event(vault_root: str | Path, freeze_id: str, actor: str,
                              trust_store: EvidenceTrustStoreV1,
                              private_key: bytes | Any) -> Path:
    vault = Path(vault_root).resolve()
    verification = verify_evidence_freeze(vault, freeze_id, trust_store)
    if not verification.valid or verification.freeze_sha256 is None:
        raise ValueError("cannot open holdout for invalid freeze: " + "; ".join(verification.blockers))
    public = _raw_public_key(private_key, private=True)
    key_id = public_key_id(public)
    if trust_store.key_for(key_id, "holdout-audit") is None:
        raise PermissionError("signer is not trusted for holdout audit events")
    audit = vault / "audit" / HOLDOUT_AUDIT
    if audit.parent.is_symlink() or not audit.parent.is_dir():
        raise ValueError("evidence vault audit directory is invalid")
    audit.parent.mkdir(mode=0o700, exist_ok=True)
    flags = os.O_RDWR | os.O_APPEND | os.O_CREAT
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(audit, flags, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        size = os.fstat(descriptor).st_size
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = bytearray()
        remaining = size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                break
            raw.extend(block)
            remaining -= len(block)
        events = _parse_audit_text(raw.decode("utf-8"))
        existing_blockers = _audit_blockers(events, trust_store)
        if existing_blockers:
            raise ValueError(
                "existing holdout audit is invalid: " + "; ".join(existing_blockers)
            )
        if any(event.freeze_id == freeze_id for event in events):
            raise FileExistsError(f"holdout already opened for freeze {freeze_id}")
        prior = events[-1].sha256 if events else ZERO_SHA256
        placeholder = HoldoutAuditEventV1(
            schema=AUDIT_SCHEMA, sequence=len(events) + 1,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
            event="holdout-opened", freeze_id=freeze_id,
            freeze_sha256=verification.freeze_sha256, actor=actor.strip(),
            prior_event_sha256=prior, signer_key_id=key_id,
            signature_scheme=SIGNATURE_SCHEME, signature="0" * 128,
        )
        event = HoldoutAuditEventV1(**{
            **placeholder.unsigned_payload(),
            "signature": _sign(private_key, AUDIT_SIGNATURE_DOMAIN,
                               placeholder.unsigned_payload()),
        })
        os.write(descriptor, _canonical_bytes(event.to_dict()) + b"\n")
        os.fsync(descriptor)
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(descriptor)
    valid, blockers = verify_holdout_audit(vault, trust_store)
    if not valid:
        raise RuntimeError("holdout audit failed after append: " + "; ".join(blockers))
    return audit


DEFAULT_TRUST_STORE = Path(__file__).with_name("trusted_evidence_keys.json")


__all__ = [
    "AUDIT_SCHEMA", "DEFAULT_TRUST_STORE", "EvidenceArtifactV1",
    "EvidenceFreezeV1", "EvidencePackageV1", "EvidenceTrustStoreV1",
    "EvidenceVerificationV1", "FREEZE_SCHEMA", "FreezeVerificationV1",
    "HOLDOUT_AUDIT", "HOLDOUT_ROLE", "HoldoutAuditEventV1",
    "PACKAGE_MANIFEST", "PACKAGE_SCHEMA", "REQUIRED_EVIDENCE_ROLES",
    "SIGNATURE_SCHEME", "TRUST_STORE_SCHEMA", "TrustedEvidenceKeyV1",
    "append_holdout_open_event", "build_signed_package",
    "create_evidence_freeze", "evidence_inventory", "generate_evidence_keypair",
    "ingest_evidence_package", "load_private_key_file", "load_package",
    "public_key_id", "resolve_evidence_vault", "verify_evidence_freeze",
    "verify_holdout_audit", "verify_package_directory", "write_package_manifest",
]
