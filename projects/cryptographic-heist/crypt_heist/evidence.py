"""Research evidence bundle for the current Cryptographic Heist build."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from .replay import load_replay, replay_diagnostic_report, summarize_replay, validate_replay


ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATHS = (
    "logs/acceptance_scenarios.json",
    "logs/acceptance_full_targets.json",
    "logs/control_acceptance_active.json",
    "logs/replay_fidelity.json",
    "logs/physics_validation.json",
    "logs/env_validation.json",
    "logs/spectator_validation.json",
    "logs/operational_readiness.json",
    "logs/v2_checkpoint_league_manifest.json",
    "logs/checkpoint_league_manifest.json",
    "logs/selfplay/selfplay_scenario_selfplay_manifest.json",
    "models/control_active/manifest.json",
    "models/active/manifest.json",
)

CHECKPOINT_PATHS = (
    "checkpoints/evader_ppo.pt",
    "checkpoints/pursuer_team_ppo.pt",
    "checkpoints/scanner_decoder.pt",
    "checkpoints/radio_policy.pt",
    "checkpoints/jammer_policy.pt",
    "models/control_active/evader_ppo.pt",
    "models/control_active/pursuer_team_ppo.pt",
    "models/active/scanner_decoder.pt",
    "models/active/radio_policy.pt",
    "models/active/jammer_policy.pt",
)


def build_evidence_bundle(
    *,
    root: str | Path = ROOT,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Collect current gate, replay, checkpoint, and command evidence."""
    root = Path(root)
    active_acceptance = _read_json(root / "logs/control_acceptance_active.json") or {}
    operational = _read_json(root / "logs/operational_readiness.json") or {}
    manifests = {path: _json_artifact(root / path) for path in MANIFEST_PATHS}
    checkpoints = {path: _file_artifact(root / path) for path in CHECKPOINT_PATHS}
    active_replays = _active_replay_reports(root, active_acceptance)
    command_registry = _json_artifact(root / "configs/commands.json")
    static_assets = {
        path: _file_artifact(root / path)
        for path in (
            "command_center/static/index.html",
            "command_center/static/app.js",
            "command_center/static/style.css",
            "viewer3d/static/index.html",
            "viewer3d/static/app.js",
            "viewer3d/static/style.css",
        )
    }
    bundle = {
        "version": 1,
        "kind": "crypt_heist_evidence_bundle",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(root),
        "passed": _bundle_passed(operational, active_acceptance, checkpoints, active_replays),
        "summary": {
            "operational_passed": bool(operational.get("passed")),
            "operational_checks_passed": int(operational.get("checks_passed", 0) or 0),
            "operational_checks": int(operational.get("checks", 0) or 0),
            "active_acceptance_passed": bool(active_acceptance.get("passed")),
            "active_required_checks_passed": int(active_acceptance.get("required_checks_passed", 0) or 0),
            "active_required_checks": int(active_acceptance.get("required_checks", 0) or 0),
            "active_milestone_targets_passed": int(active_acceptance.get("milestone_targets_passed", 0) or 0),
            "active_milestone_targets": int(active_acceptance.get("milestone_targets", 0) or 0),
            "active_replay_reports": len(active_replays),
            "active_replay_reports_nominal": sum(1 for row in active_replays if row.get("report", {}).get("score", {}).get("status") == "nominal"),
            "checkpoints_existing": sum(1 for row in checkpoints.values() if row.get("exists")),
            "checkpoints_expected": len(checkpoints),
        },
        "command_registry": command_registry,
        "manifests": manifests,
        "checkpoints": checkpoints,
        "active_replays": active_replays,
        "static_assets": static_assets,
    }
    if out is not None:
        path = Path(out)
        if not path.is_absolute():
            path = root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return bundle


def verify_evidence_bundle(
    bundle_path: str | Path = "logs/evidence_bundle.json",
    *,
    root: str | Path | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Verify that an evidence bundle still matches files on disk."""
    path = Path(bundle_path)
    bundle = _read_json(path)
    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    if not isinstance(bundle, dict):
        manifest = _verification_manifest(path, {}, False, [{"area": "bundle", "reason": "invalid_json"}], [])
        if out is not None:
            _write_json(out, manifest)
        return manifest

    root_path = Path(root or bundle.get("root") or ROOT)
    checks.append(_check_value(
        bundle.get("kind") == "crypt_heist_evidence_bundle",
        "bundle_kind",
        {"kind": bundle.get("kind")},
        failures,
    ))
    checks.append(_check_value(
        int(bundle.get("version", 0) or 0) == 1,
        "bundle_version",
        {"version": bundle.get("version")},
        failures,
    ))
    checks.append(_check_value(
        bool(bundle.get("passed")),
        "bundle_passed_flag",
        {"passed": bundle.get("passed")},
        failures,
    ))

    _verify_named_artifact("command_registry", bundle.get("command_registry", {}), root_path, failures, checks)
    for name, artifact in (bundle.get("manifests") or {}).items():
        _verify_named_artifact(f"manifest:{name}", artifact, root_path, failures, checks, fallback=name)
    for name, artifact in (bundle.get("checkpoints") or {}).items():
        _verify_named_artifact(f"checkpoint:{name}", artifact, root_path, failures, checks, fallback=name)
    for name, artifact in (bundle.get("static_assets") or {}).items():
        _verify_named_artifact(f"static:{name}", artifact, root_path, failures, checks, fallback=name)

    active_replay_checks = []
    for row in bundle.get("active_replays", []) or []:
        replay_check = _verify_active_replay(row, root_path)
        active_replay_checks.append(replay_check)
        checks.append(replay_check)
        if not replay_check["passed"]:
            failures.append({
                "area": "active_replay",
                "reason": "replay_mismatch",
                "evidence": replay_check,
            })

    summary = bundle.get("summary", {}) if isinstance(bundle.get("summary"), dict) else {}
    expected_nominal = int(summary.get("active_replay_reports_nominal", 0) or 0)
    actual_nominal = sum(
        1
        for row in bundle.get("active_replays", []) or []
        if (row.get("report") or {}).get("score", {}).get("status") == "nominal"
    )
    checks.append(_check_value(
        expected_nominal == actual_nominal,
        "summary_replay_report_count",
        {"expected_nominal": expected_nominal, "actual_nominal": actual_nominal},
        failures,
    ))
    expected_checkpoints = int(summary.get("checkpoints_expected", 0) or 0)
    actual_checkpoints = len(bundle.get("checkpoints", {}) or {})
    checks.append(_check_value(
        expected_checkpoints == actual_checkpoints,
        "summary_checkpoint_count",
        {"expected_checkpoints": expected_checkpoints, "actual_checkpoints": actual_checkpoints},
        failures,
    ))

    passed = not failures
    manifest = _verification_manifest(path, bundle, passed, failures, checks)
    if out is not None:
        _write_json(out, manifest)
    return manifest


def _bundle_passed(
    operational: dict[str, Any],
    active_acceptance: dict[str, Any],
    checkpoints: dict[str, dict[str, Any]],
    active_replays: list[dict[str, Any]],
) -> bool:
    return bool(
        operational.get("passed")
        and active_acceptance.get("passed")
        and active_replays
        and all(row.get("valid") and row.get("report", {}).get("score", {}).get("status") == "nominal" for row in active_replays)
        and all(row.get("exists") for row in checkpoints.values())
    )


def _verification_manifest(
    path: Path,
    bundle: dict[str, Any],
    passed: bool,
    failures: list[dict[str, Any]],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "version": 1,
        "kind": "crypt_heist_evidence_bundle_verification",
        "bundle": str(path),
        "bundle_kind": bundle.get("kind"),
        "bundle_created_at": bundle.get("created_at"),
        "passed": bool(passed),
        "checks_passed": sum(1 for check in checks if check.get("passed")),
        "checks": len(checks),
        "failures": failures,
        "results": checks,
    }


def _verify_active_replay(row: dict[str, Any], root: Path) -> dict[str, Any]:
    replay = str(row.get("replay", ""))
    path = _resolve_artifact_path(row.get("artifact", {}), root, fallback=replay)
    evidence: dict[str, Any] = {"scenario": row.get("scenario"), "replay": replay, "path": str(path)}
    try:
        artifact_check = _artifact_match(row.get("artifact", {}), path)
        if not artifact_check["passed"]:
            return {"name": f"active_replay:{replay}", "passed": False, "metrics": artifact_check}
        lines = load_replay(path)
        validate_replay(lines)
        summary = _replay_summary_excerpt(summarize_replay(lines, include_fingerprint=True))
        report = _report_excerpt(replay_diagnostic_report(path, lines))
        stored_summary = row.get("summary", {})
        stored_report = row.get("report", {})
        summary_ok = _nested_subset_equal(stored_summary, summary)
        report_ok = _nested_subset_equal(stored_report.get("score", {}), report.get("score", {}))
        evidence.update({
            "artifact": artifact_check,
            "summary_matched": summary_ok,
            "report_score_matched": report_ok,
            "report_status": report.get("score", {}).get("status"),
        })
        return {
            "name": f"active_replay:{replay}",
            "passed": bool(summary_ok and report_ok and report.get("score", {}).get("status") == "nominal"),
            "metrics": evidence,
        }
    except Exception as exc:  # pragma: no cover - reported as evidence
        return {
            "name": f"active_replay:{replay}",
            "passed": False,
            "metrics": {**evidence, "error": f"{type(exc).__name__}: {exc}"},
        }


def _verify_named_artifact(
    name: str,
    artifact: dict[str, Any],
    root: Path,
    failures: list[dict[str, Any]],
    checks: list[dict[str, Any]],
    *,
    fallback: str | None = None,
) -> None:
    path = _resolve_artifact_path(artifact, root, fallback=fallback)
    result = _artifact_match(artifact, path)
    checks.append({"name": name, "passed": result["passed"], "metrics": result})
    if not result["passed"]:
        failures.append({"area": "artifact", "reason": "artifact_mismatch", "name": name, "evidence": result})


def _artifact_match(artifact: dict[str, Any], path: Path) -> dict[str, Any]:
    expected_exists = bool(artifact.get("exists"))
    actual_exists = path.exists()
    result: dict[str, Any] = {
        "path": str(path),
        "expected_exists": expected_exists,
        "actual_exists": actual_exists,
    }
    if expected_exists != actual_exists:
        result["passed"] = False
        return result
    if not expected_exists:
        result["passed"] = True
        return result
    expected_size = int(artifact.get("size", -1))
    actual_size = int(path.stat().st_size)
    expected_sha = str(artifact.get("sha256", ""))
    actual_sha = _sha256(path)
    result.update({
        "expected_size": expected_size,
        "actual_size": actual_size,
        "expected_sha256": expected_sha,
        "actual_sha256": actual_sha,
        "size_matched": expected_size == actual_size,
        "sha256_matched": expected_sha == actual_sha,
    })
    result["passed"] = bool(expected_size == actual_size and expected_sha == actual_sha)
    return result


def _resolve_artifact_path(artifact: dict[str, Any], root: Path, *, fallback: str | None = None) -> Path:
    raw = artifact.get("path") or fallback or ""
    path = Path(str(raw))
    if path.is_absolute() and path.exists():
        return path
    if path.is_absolute() and not path.exists() and fallback:
        return root / fallback
    if path.is_absolute():
        return path
    return root / path


def _check_value(
    passed: bool,
    name: str,
    metrics: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    row = {"name": name, "passed": bool(passed), "metrics": metrics}
    if not passed:
        failures.append({"area": "bundle", "reason": name, "evidence": metrics})
    return row


def _nested_subset_equal(stored: Any, actual: Any) -> bool:
    if isinstance(stored, dict) and isinstance(actual, dict):
        for key, value in stored.items():
            if key not in actual or not _nested_subset_equal(value, actual[key]):
                return False
        return True
    if isinstance(stored, float) or isinstance(actual, float):
        try:
            return abs(float(stored) - float(actual)) <= 1e-9
        except (TypeError, ValueError):
            return False
    return stored == actual


def _active_replay_reports(root: Path, active_acceptance: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scenario in active_acceptance.get("scenarios", []) if isinstance(active_acceptance, dict) else []:
        replay = scenario.get("replay")
        if not replay:
            continue
        path = root / str(replay)
        artifact = _file_artifact(path)
        row: dict[str, Any] = {
            "scenario": scenario.get("name"),
            "replay": str(replay),
            "artifact": artifact,
            "valid": False,
        }
        try:
            lines = load_replay(path)
            validate_replay(lines)
            summary = summarize_replay(lines, include_fingerprint=True)
            report = replay_diagnostic_report(path, lines)
            row.update({
                "valid": True,
                "summary": _replay_summary_excerpt(summary),
                "report": _report_excerpt(report),
            })
        except Exception as exc:  # pragma: no cover - captured as evidence
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
    return rows


def _replay_summary_excerpt(summary: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "frames",
        "duration",
        "captures",
        "waypoints_hit",
        "deception_score",
        "avg_confidence",
        "avg_evader_speed",
        "radio_word_entropy",
        "jam_events",
        "cipher_rotations",
        "avg_evader_reward",
        "avg_pursuer_auth_penalty",
        "state_sha256",
        "radio_sha256",
    )
    return {key: summary.get(key) for key in keys if key in summary}


def _report_excerpt(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": report.get("kind"),
        "score": report.get("score"),
        "diagnostics": report.get("diagnostics", [])[:5],
        "notable_events": report.get("notable_events", [])[:8],
    }


def _json_artifact(path: Path) -> dict[str, Any]:
    artifact = _file_artifact(path)
    data = _read_json(path)
    if isinstance(data, dict):
        artifact["json"] = _manifest_excerpt(data)
    return artifact


def _manifest_excerpt(data: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "version",
        "kind",
        "passed",
        "checks_passed",
        "checks",
        "score",
        "summary",
        "required_checks_passed",
        "required_checks",
        "milestone_targets_passed",
        "milestone_targets",
    )
    return {key: data.get(key) for key in keys if key in data}


def _file_artifact(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    artifact: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
    }
    if not path.exists():
        return artifact
    stat = path.stat()
    artifact.update({
        "size": int(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        "sha256": _sha256(path),
    })
    return artifact


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
