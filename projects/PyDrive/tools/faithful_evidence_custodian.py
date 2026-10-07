#!/usr/bin/env python3
"""Offline custodian utilities for faithful-v2 evidence packages.

This tool never edits the protocol trust store.  Key generation writes the
private seed only to an explicitly selected external path and emits a public
trust-entry file for human review.  Package signing computes size and SHA-256
from the materialized files instead of accepting caller-supplied digests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.faithful.evidence import (  # noqa: E402
    EvidenceArtifactV1,
    build_signed_package,
    generate_evidence_keypair,
    load_private_key_file,
    public_key_id,
    write_package_manifest,
)


REQUEST_SCHEMA = "faithful-evidence-package-request-v1"
ARTIFACT_FIELDS = {
    "artifact_id", "role", "path", "media_type", "units", "source_uri", "license_id",
    "redistribution", "confidence", "calibration_role", "high_sensitivity",
    "holdout_sealed",
}


def _require_outside_repository(path: Path, field: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_absolute():
        raise ValueError(f"{field} must be an absolute path")
    ancestor = candidate
    suffix: list[str] = []
    while not ancestor.exists():
        suffix.append(ancestor.name)
        ancestor = ancestor.parent
    resolved = ancestor.resolve().joinpath(*reversed(suffix))
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError(f"{field} must remain outside the repository")
    return resolved


def _write_exclusive(path: Path, data: bytes, mode: int) -> None:
    if not path.is_absolute():
        raise ValueError(f"output path must be absolute: {path}")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        remaining = memoryview(data)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("short write")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _generate(args: argparse.Namespace) -> None:
    private_path = _require_outside_repository(Path(args.private_key), "private key")
    public_path = _require_outside_repository(Path(args.public_key), "public key")
    trust_path = _require_outside_repository(Path(args.trust_entry), "trust entry")
    private, public = generate_evidence_keypair()
    key_id = public_key_id(public)
    entry = {
        "key_id": key_id,
        "public_key_hex": public.hex(),
        "label": args.label,
        "status": "active",
        "permitted_kinds": ["licensed", "fixture", "freeze", "holdout-audit"],
    }
    _write_exclusive(private_path, private, 0o600)
    try:
        _write_exclusive(public_path, public.hex().encode("ascii") + b"\n", 0o644)
        _write_exclusive(
            trust_path,
            (json.dumps(entry, sort_keys=True, indent=2) + "\n").encode("utf-8"),
            0o644,
        )
    except Exception:
        private_path.unlink(missing_ok=True)
        public_path.unlink(missing_ok=True)
        trust_path.unlink(missing_ok=True)
        raise
    print(json.dumps({
        "key_id": key_id,
        "private_key_path": str(private_path),
        "public_key_path": str(public_path),
        "trust_entry_path": str(trust_path),
        "next_step": "Review the public trust entry before adding it to the protocol trust store.",
    }, sort_keys=True, indent=2))


def _artifact_path(package_root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    if (not relative or pure.is_absolute() or relative != pure.as_posix()
            or any(part in {"", ".", ".."} for part in pure.parts)):
        raise ValueError(f"unsafe artifact path: {relative!r}")
    candidate = package_root
    for part in pure.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError(f"artifact path contains a symlink: {relative}")
    resolved = candidate.resolve()
    root = package_root.resolve()
    if root not in resolved.parents or not resolved.is_file():
        raise ValueError(f"artifact is not a regular file inside package root: {relative}")
    return resolved


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _sign_package(args: argparse.Namespace) -> None:
    package_root = _require_outside_repository(Path(args.package_root), "package root")
    request_path = _require_outside_repository(Path(args.request), "package request")
    if package_root.is_symlink() or not package_root.is_dir():
        raise ValueError("package root must be a regular non-symlink directory")
    if request_path.is_symlink() or not request_path.is_file():
        raise ValueError("package request must be a regular non-symlink JSON file")
    root = package_root.resolve()
    request_resolved = request_path.resolve()
    if request_resolved == root or root in request_resolved.parents:
        raise ValueError("package request must remain outside the signed package directory")
    request = json.loads(request_path.read_text(encoding="utf-8"))
    fields = {"schema", "package_id", "package_kind", "artifacts"}
    if not isinstance(request, dict) or set(request) != fields:
        raise ValueError("package request has invalid fields")
    if request["schema"] != REQUEST_SCHEMA:
        raise ValueError(f"package request schema must be {REQUEST_SCHEMA}")
    if not isinstance(request["artifacts"], list) or not request["artifacts"]:
        raise ValueError("package request artifacts must be a nonempty list")

    artifacts = []
    for raw in request["artifacts"]:
        if not isinstance(raw, dict) or set(raw) != ARTIFACT_FIELDS:
            raise ValueError("package request artifact has invalid fields")
        path = _artifact_path(root, raw["path"])
        artifacts.append(EvidenceArtifactV1(
            artifact_id=raw["artifact_id"], role=raw["role"],
            path=raw["path"], sha256=_digest(path),
            size_bytes=path.stat().st_size, media_type=raw["media_type"],
            units=raw["units"], source_uri=raw["source_uri"],
            license_id=raw["license_id"], redistribution=raw["redistribution"],
            confidence=raw["confidence"], calibration_role=raw["calibration_role"],
            high_sensitivity=raw["high_sensitivity"],
            holdout_sealed=raw["holdout_sealed"],
        ))
    private = load_private_key_file(args.private_key, forbidden_root=ROOT)
    package = build_signed_package(
        request["package_id"], request["package_kind"], artifacts, private
    )
    manifest = write_package_manifest(root, package)
    print(json.dumps({
        "manifest": str(manifest), "package_id": package.package_id,
        "package_kind": package.package_kind, "signer_key_id": package.signer_key_id,
        "artifact_count": len(package.artifacts),
    }, sort_keys=True, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate-key", help="create an external custodian keypair")
    generate.add_argument("--private-key", required=True)
    generate.add_argument("--public-key", required=True)
    generate.add_argument("--trust-entry", required=True)
    generate.add_argument("--label", required=True)
    generate.set_defaults(handler=_generate)
    package = commands.add_parser("sign-package", help="hash and sign one package directory")
    package.add_argument("--package-root", required=True)
    package.add_argument("--request", required=True)
    package.add_argument("--private-key", required=True)
    package.set_defaults(handler=_sign_package)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        args.handler(args)
    except (FileExistsError, OSError, UnicodeError, json.JSONDecodeError,
            TypeError, ValueError) as exc:
        raise SystemExit(f"custodian command failed: {exc}") from exc


if __name__ == "__main__":
    main()
