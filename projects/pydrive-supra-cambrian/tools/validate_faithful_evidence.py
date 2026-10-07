#!/usr/bin/env python3
"""Adversarial validation for the faithful-v2 external evidence vault."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.faithful.evidence import (  # noqa: E402
    EvidenceArtifactV1,
    EvidenceTrustStoreV1,
    REQUIRED_EVIDENCE_ROLES,
    TRUST_STORE_SCHEMA,
    TrustedEvidenceKeyV1,
    append_holdout_open_event,
    build_signed_package,
    create_evidence_freeze,
    evidence_inventory,
    generate_evidence_keypair,
    ingest_evidence_package,
    public_key_id,
    resolve_evidence_vault,
    verify_evidence_freeze,
    verify_holdout_audit,
    verify_package_directory,
    write_package_manifest,
)


FAILED: list[str] = []


def gate(name: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        FAILED.append(name)


def raises(kind: type[BaseException], callback) -> bool:
    try:
        callback()
    except kind:
        return True
    return False


def trust_store(public: bytes) -> EvidenceTrustStoreV1:
    key = TrustedEvidenceKeyV1(
        key_id=public_key_id(public), public_key_hex=public.hex(),
        label="validator-only custodian", status="active",
        permitted_kinds=("licensed", "fixture", "freeze", "holdout-audit"),
    )
    return EvidenceTrustStoreV1(schema=TRUST_STORE_SCHEMA, keys=(key,))


def build_source(root: Path, package_id: str, package_kind: str,
                 private: bytes, *, all_roles: bool = True) -> Path:
    source = root / package_id
    source.mkdir()
    roles = sorted(REQUIRED_EVIDENCE_ROLES) if all_roles else [
        "vehicle.mass-properties"
    ]
    artifacts = []
    for index, role in enumerate(roles):
        relative = f"artifacts/{index:02d}-{role.replace('.', '-')}.bin"
        path = source / relative
        path.parent.mkdir(exist_ok=True)
        data = f"validator fixture {package_id} {role}\n".encode()
        path.write_bytes(data)
        is_holdout = role == "holdout.nordschleife-record-telemetry"
        artifacts.append(EvidenceArtifactV1(
            artifact_id=f"artifact-{index:02d}", role=role, path=relative,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data), media_type="application/octet-stream",
            units="mixed", source_uri=f"vault-source:{package_id}/{role}",
            license_id=("licensed-validator-v1" if package_kind == "licensed"
                        else "fixture-validator-v1"),
            redistribution="none", confidence="licensed-measured",
            calibration_role=("holdout" if is_holdout else
                              "calibration" if role == "calibration.component-spa"
                              else "scenario" if role.startswith("scenario.")
                              else "source"),
            high_sensitivity=True, holdout_sealed=is_holdout,
        ))
    package = build_signed_package(package_id, package_kind, artifacts, private)
    write_package_manifest(source, package)
    return source


def main() -> None:
    private, public = generate_evidence_keypair()
    trust = trust_store(public)
    other_private, _ = generate_evidence_keypair()

    example = json.loads((
        ROOT / "supra" / "faithful" / "evidence_package_request.example.json"
    ).read_text(encoding="utf-8"))
    gate("checked-in package request enumerates the exact protocol roles",
         example.get("package_kind") == "fixture"
         and {item["role"] for item in example.get("artifacts", [])}
         == REQUIRED_EVIDENCE_ROLES
         and len({item["artifact_id"] for item in example["artifacts"]})
         == len(example["artifacts"]))

    print("== external vault boundary ==")
    with tempfile.TemporaryDirectory(prefix="faithful-project-") as project_dir, \
            tempfile.TemporaryDirectory(prefix="faithful-vault-parent-") as vault_parent:
        project = Path(project_dir)
        gate("relative vault root rejected", raises(
            ValueError, lambda: resolve_evidence_vault(project, "relative-vault", create=True)
        ))
        gate("in-repository vault rejected", raises(
            ValueError, lambda: resolve_evidence_vault(project, project / "evidence", create=True)
        ))
        vault = resolve_evidence_vault(
            project, Path(vault_parent) / "private-evidence", create=True
        )
        gate("vault is external and initialized",
             vault.is_dir() and all((vault / name).is_dir()
                                    for name in ("packages", "freezes", "audit")))
        permissive = Path(vault_parent) / "permissive-evidence"
        permissive.mkdir(mode=0o755)
        permissive.chmod(0o755)
        gate("group-readable evidence vault rejected", raises(
            PermissionError, lambda: resolve_evidence_vault(project, permissive)
        ))

        print("== signed packages and permanent fixture label ==")
        source_root = Path(vault_parent) / "incoming"
        source_root.mkdir()
        fixture = build_source(source_root, "fixture-complete-v1", "fixture", private)
        result = verify_package_directory(fixture, trust)
        gate("trusted fixture package verifies", result.valid)
        gate("signed fixture remains noncertifying", not result.certification_eligible)
        destination = ingest_evidence_package(fixture, vault, trust)
        gate("verified package ingested into immutable namespace",
             destination == vault / "packages" / "fixture-complete-v1")
        gate("duplicate package ingestion rejected", raises(
            FileExistsError, lambda: ingest_evidence_package(fixture, vault, trust)
        ))
        inventory = evidence_inventory(vault, trust)
        gate("fixture can exercise a complete role inventory",
             not inventory["missing_roles"] and len(inventory["observed_roles"])
             == len(REQUIRED_EVIDENCE_ROLES))
        gate("complete fixture inventory cannot claim readiness",
             inventory["certification_ready"] is False)

        forged = build_source(source_root, "forged-package-v1", "licensed", other_private)
        forged_result = verify_package_directory(forged, trust)
        gate("untrusted package signer rejected",
             not forged_result.valid and any("signer" in item for item in forged_result.blockers))

        tampered = build_source(source_root, "tampered-package-v1", "licensed", private,
                                all_roles=False)
        (tampered / "artifacts" / "00-vehicle-mass-properties.bin").write_bytes(b"changed")
        tampered_result = verify_package_directory(tampered, trust)
        gate("one-byte-class artifact tampering detected",
             not tampered_result.valid and any("mismatch" in item
                                               for item in tampered_result.blockers))

        extra = build_source(source_root, "smuggled-file-v1", "licensed", private,
                             all_roles=False)
        (extra / "unlisted.bin").write_bytes(b"not in signed manifest")
        extra_result = verify_package_directory(extra, trust)
        gate("unlisted package file rejected",
             not extra_result.valid and any("unlisted files" in item
                                             for item in extra_result.blockers))

        if hasattr(Path, "symlink_to"):
            linked = build_source(source_root, "symlink-package-v1", "licensed", private,
                                  all_roles=False)
            artifact = linked / "artifacts" / "00-vehicle-mass-properties.bin"
            artifact.unlink()
            artifact.symlink_to(extra / "unlisted.bin")
            linked_result = verify_package_directory(linked, trust)
            gate("artifact symlink rejected", not linked_result.valid and any(
                "symlink" in item for item in linked_result.blockers
            ))

        print("== signed freeze and holdout audit ==")
        freeze_path = create_evidence_freeze(
            vault, "fixture-freeze-v1", ["fixture-complete-v1"], trust, private,
            vehicle_spec_sha256=hashlib.sha256(b"vehicle").hexdigest(),
            track_surface_sha256=hashlib.sha256(b"track").hexdigest(),
            scenario_sha256=hashlib.sha256(b"scenario").hexdigest(),
            parameter_registry_sha256=hashlib.sha256(b"parameters").hexdigest(),
        )
        freeze = verify_evidence_freeze(vault, "fixture-freeze-v1", trust)
        gate("fixture freeze verifies structurally", freeze.valid and freeze_path.is_file())
        gate("fixture freeze remains noncertifying", not freeze.certification_eligible)
        audit = append_holdout_open_event(
            vault, "fixture-freeze-v1", "validator", trust, private
        )
        audit_valid, audit_blockers = verify_holdout_audit(vault, trust)
        gate("signed hash-chained holdout-open event verifies",
             audit.is_file() and audit_valid and not audit_blockers)
        gate("holdout cannot be opened twice for one freeze", raises(
            FileExistsError, lambda: append_holdout_open_event(
                vault, "fixture-freeze-v1", "validator", trust, private
            )
        ))

        print("== project CLI integration ==")
        trust_path = Path(vault_parent) / "trust-store.json"
        trust_path.write_text(
            json.dumps(trust.to_dict(), sort_keys=True), encoding="utf-8"
        )
        environment = os.environ.copy()
        environment["FAITHFUL_EVIDENCE_ROOT"] = str(vault)
        inventory_cli = subprocess.run(
            [sys.executable, str(ROOT / "run.py"),
             "--faithful-evidence-inventory",
             "--faithful-evidence-trust-store", str(trust_path)],
            cwd=ROOT, env=environment, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        cli_inventory = json.loads(inventory_cli.stdout) if inventory_cli.returncode == 0 else {}
        gate("CLI inventories configured external vault",
             inventory_cli.returncode == 0
             and cli_inventory.get("certification_ready") is False
             and not cli_inventory.get("missing_roles"))
        verify_cli = subprocess.run(
            [sys.executable, str(ROOT / "run.py"),
             "--faithful-evidence-verify", "fixture-complete-v1",
             "--faithful-evidence-trust-store", str(trust_path)],
            cwd=ROOT, env=environment, check=False,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        cli_verification = json.loads(verify_cli.stdout) if verify_cli.returncode == 0 else {}
        gate("CLI verifies fixture without laundering certification",
             verify_cli.returncode == 0
             and cli_verification.get("valid") is True
             and cli_verification.get("certification_eligible") is False)

        audit_lines = audit.read_text(encoding="utf-8").splitlines()
        event = json.loads(audit_lines[0])
        event["actor"] = "tampered"
        audit.write_text(json.dumps(event) + "\n", encoding="utf-8")
        audit_valid, audit_blockers = verify_holdout_audit(vault, trust)
        gate("holdout audit tampering detected",
             not audit_valid and any("signature" in item for item in audit_blockers))

    print("== schema type safety ==")
    data = b"x"
    base = EvidenceArtifactV1(
        artifact_id="mass-properties-v1", role="vehicle.mass-properties",
        path="artifacts/mass.bin",
        sha256=hashlib.sha256(data).hexdigest(), size_bytes=1,
        media_type="application/octet-stream", units="kg",
        source_uri="vault-source:mass", license_id="licensed-v1",
        redistribution="none", confidence="licensed-measured",
        calibration_role="source", high_sensitivity=True,
        holdout_sealed=False,
    )
    gate("truthy string cannot impersonate high-sensitivity bool", raises(
        TypeError, lambda: replace(base, high_sensitivity="false")
    ))
    gate("path traversal rejected", raises(
        ValueError, lambda: replace(base, path="../mass.bin")
    ))
    gate("non-holdout role cannot claim sealed status", raises(
        ValueError, lambda: replace(base, holdout_sealed=True)
    ))
    second = replace(
        base, artifact_id="mass-properties-supporting-v1",
        path="artifacts/mass-supporting.bin",
    )
    multi_file = build_signed_package(
        "multi-file-role-v1", "fixture", (base, second), private
    )
    gate("one protocol role may contain multiple uniquely identified files",
         len(multi_file.artifacts) == 2
         and {artifact.role for artifact in multi_file.artifacts}
         == {"vehicle.mass-properties"})
    gate("duplicate artifact IDs rejected", raises(
        ValueError, lambda: build_signed_package(
            "duplicate-artifact-id-v1", "fixture",
            (base, replace(second, artifact_id=base.artifact_id)), private,
        )
    ))

    if FAILED:
        raise SystemExit("faithful evidence validation failed: " + ", ".join(FAILED))
    print("faithful evidence validation: PASS")


if __name__ == "__main__":
    main()
