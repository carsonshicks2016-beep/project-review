"""V2 research-grade evaluation reporting."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_THRESHOLDS = {
    "min_seeds": 5,
    "min_steps": 1800,
    "min_overall_score": 0.58,
    "min_pursuer_security_score": 0.64,
    "min_evader_pressure_score": 0.46,
    "min_balance_score": 0.72,
    "max_spoof_susceptibility": 0.62,
    "min_deception_lift": 0.50,
    "min_confidence_damage": 0.01,
    "min_spoof_event_lift": 1.0,
    "min_replay_report_score": 0.90,
}


def build_v2_research_report(
    *,
    league_manifest: str | Path = "logs/v2_checkpoint_league_manifest.json",
    replay_report: str | Path = "logs/replay_reports/downtown_chase_seed11_report.json",
    spectator_validation: str | Path = "logs/spectator_validation.json",
    operational_readiness: str | Path = "logs/operational_readiness.json",
    evidence_verification: str | Path = "logs/evidence_bundle_verification.json",
    out: str | Path | None = "logs/v2_research_report.json",
    markdown_out: str | Path | None = "logs/v2_research_report.md",
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build a release-style v2 research report from current evidence files."""
    threshold_values = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    league = _read_json(_path(league_manifest))
    replay = _read_json(_path(replay_report))
    spectator = _read_json(_path(spectator_validation))
    operational = _read_json(_path(operational_readiness))
    evidence = _read_json(_path(evidence_verification))

    checks = [
        _league_depth_check(league, threshold_values, league_manifest),
        _information_stack_check(league, threshold_values),
        _jamming_effect_check(league, threshold_values),
        _replay_report_check(replay, threshold_values, replay_report),
        _spectator_threejs_check(spectator, spectator_validation),
        _evidence_integrity_check(operational, evidence, operational_readiness, evidence_verification),
    ]
    passed = all(check["passed"] for check in checks)
    report = {
        "version": 1,
        "kind": "crypt_heist_v2_research_report",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "passed": passed,
        "status": "research_grade_candidate" if passed else "needs_v2_work",
        "checks_passed": sum(1 for check in checks if check["passed"]),
        "checks": len(checks),
        "thresholds": threshold_values,
        "inputs": {
            "league_manifest": _display_path(_path(league_manifest)),
            "replay_report": _display_path(_path(replay_report)),
            "spectator_validation": _display_path(_path(spectator_validation)),
            "operational_readiness": _display_path(_path(operational_readiness)),
            "evidence_verification": _display_path(_path(evidence_verification)),
        },
        "scores": _v2_scores(league, replay, spectator, operational, evidence),
        "checks_detail": checks,
        "summary": _summary_text(checks, passed),
    }
    if out is not None:
        _write_json(_path(out), report)
    if markdown_out is not None:
        _path(markdown_out).parent.mkdir(parents=True, exist_ok=True)
        _path(markdown_out).write_text(render_v2_markdown(report), encoding="utf-8")
    return report


def render_v2_markdown(report: dict[str, Any]) -> str:
    scores = report.get("scores", {})
    lines = [
        "# Cryptographic Heist Engine V2 Research Report",
        "",
        f"Status: **{report.get('status')}**",
        f"Passed: **{report.get('passed')}**",
        f"Checks: **{report.get('checks_passed')}/{report.get('checks')}**",
        "",
        "## Scorecard",
        "",
        f"- Information league overall: {scores.get('league_overall_score', 0.0):.3f}",
        f"- Pursuer security: {scores.get('pursuer_security_score', 0.0):.3f}",
        f"- Evader pressure: {scores.get('evader_pressure_score', 0.0):.3f}",
        f"- Adversarial balance: {scores.get('adversarial_balance_score', 0.0):.3f}",
        f"- Spoof susceptibility: {scores.get('mean_spoof_susceptibility', 0.0):.3f}",
        f"- Deception lift: {scores.get('mean_deception_lift', 0.0):.3f}",
        f"- Replay report score: {scores.get('replay_report_score', 0.0):.3f}",
        "",
        "## Checks",
        "",
    ]
    for check in report.get("checks_detail", []):
        marker = "PASS" if check.get("passed") else "FAIL"
        lines.append(f"- {marker} `{check.get('name')}`: {check.get('summary')}")
    lines.extend(["", "## Inputs", ""])
    for key, value in (report.get("inputs") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def _league_depth_check(league: dict[str, Any], thresholds: dict[str, float], source: str | Path) -> dict[str, Any]:
    seeds = league.get("seeds") if isinstance(league, dict) else []
    steps = int(league.get("steps", 0) or 0) if isinstance(league, dict) else 0
    seed_count = len(seeds or [])
    passed = seed_count >= int(thresholds["min_seeds"]) and steps >= int(thresholds["min_steps"])
    return _check(
        "long_multi_seed_league",
        passed,
        f"{seed_count} seeds at {steps} steps from {_display_path(_path(source))}",
        {
            "seed_count": seed_count,
            "steps": steps,
            "min_seeds": int(thresholds["min_seeds"]),
            "min_steps": int(thresholds["min_steps"]),
            "seeds": seeds or [],
        },
    )


def _information_stack_check(league: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    scores = league.get("scores", {}) if isinstance(league, dict) else {}
    aggregates = league.get("aggregates", {}) if isinstance(league, dict) else {}
    overall = float(scores.get("overall_score", 0.0) or 0.0)
    pursuer = float(scores.get("pursuer_security_score", 0.0) or 0.0)
    evader = float(scores.get("evader_pressure_score", 0.0) or 0.0)
    balance = float(scores.get("adversarial_balance_score", 0.0) or 0.0)
    susceptibility = float(aggregates.get("mean_spoof_susceptibility", 1.0) or 1.0)
    passed = bool(
        overall >= thresholds["min_overall_score"]
        and pursuer >= thresholds["min_pursuer_security_score"]
        and evader >= thresholds["min_evader_pressure_score"]
        and balance >= thresholds["min_balance_score"]
        and susceptibility <= thresholds["max_spoof_susceptibility"]
    )
    return _check(
        "learned_comms_scanner_jammer_scores",
        passed,
        f"overall={overall:.3f}, security={pursuer:.3f}, evader={evader:.3f}, susceptibility={susceptibility:.3f}",
        {
            "overall_score": overall,
            "pursuer_security_score": pursuer,
            "evader_pressure_score": evader,
            "adversarial_balance_score": balance,
            "mean_spoof_susceptibility": susceptibility,
        },
    )


def _jamming_effect_check(league: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    aggregates = league.get("aggregates", {}) if isinstance(league, dict) else {}
    per_seed = league.get("per_seed", []) if isinstance(league, dict) else []
    deception = float(aggregates.get("mean_deception_lift", 0.0) or 0.0)
    confidence_damage = float(aggregates.get("mean_confidence_damage", 0.0) or 0.0)
    spoof_lift = float(aggregates.get("mean_spoof_event_lift", 0.0) or 0.0)
    deviation = float(aggregates.get("mean_pursuer_deviation", 0.0) or 0.0)
    susceptibility_rows = [
        float((row.get("metrics") or {}).get("spoof_susceptibility_counterfactual", 0.0) or 0.0)
        for row in per_seed
        if isinstance(row, dict)
    ]
    passed = bool(
        deception >= thresholds["min_deception_lift"]
        and confidence_damage >= thresholds["min_confidence_damage"]
        and spoof_lift >= thresholds["min_spoof_event_lift"]
    )
    return _check(
        "counterfactual_jamming_effect",
        passed,
        f"deception_lift={deception:.3f}, confidence_damage={confidence_damage:.3f}, spoof_event_lift={spoof_lift:.3f}",
        {
            "mean_deception_lift": deception,
            "mean_confidence_damage": confidence_damage,
            "mean_spoof_event_lift": spoof_lift,
            "mean_pursuer_deviation": deviation,
            "spoof_susceptibility_std": pstdev(susceptibility_rows) if len(susceptibility_rows) > 1 else 0.0,
            "spoof_susceptibility_mean": mean(susceptibility_rows) if susceptibility_rows else 0.0,
        },
    )


def _replay_report_check(replay: dict[str, Any], thresholds: dict[str, float], source: str | Path) -> dict[str, Any]:
    score = replay.get("score", {}) if isinstance(replay, dict) else {}
    sections = replay.get("sections", {}) if isinstance(replay, dict) else {}
    chase = sections.get("chase", {}) if isinstance(sections, dict) else {}
    info = sections.get("information_warfare", {}) if isinstance(sections, dict) else {}
    overall = float(score.get("overall", 0.0) or 0.0)
    passed = bool(
        score.get("status") == "nominal"
        and overall >= thresholds["min_replay_report_score"]
        and int(chase.get("waypoints_hit", 0) or 0) >= 3
        and int(info.get("jam_events", 0) or 0) >= 1
    )
    return _check(
        "cinematic_replay_report",
        passed,
        f"status={score.get('status')}, score={overall:.3f}, source={_display_path(_path(source))}",
        {
            "status": score.get("status"),
            "overall": overall,
            "waypoints_hit": int(chase.get("waypoints_hit", 0) or 0),
            "jam_events": int(info.get("jam_events", 0) or 0),
            "radio_events": int(info.get("radio_events", 0) or 0),
        },
    )


def _spectator_threejs_check(spectator: dict[str, Any], source: str | Path) -> dict[str, Any]:
    results = spectator.get("results", []) if isinstance(spectator, dict) else []
    by_name = {str(row.get("name")): row for row in results if isinstance(row, dict)}
    required = [
        "audio_confidence_mapping",
        "web_cockpit_static_contract",
        "threejs_replay_renderer_contract",
        "web_replay_payload_contract",
        "dashboard_replay_deep_links",
    ]
    missing = [name for name in required if not by_name.get(name, {}).get("passed")]
    passed = bool(spectator.get("passed")) and not missing
    return _check(
        "threejs_spectator_contract",
        passed,
        f"{len(required) - len(missing)}/{len(required)} required spectator checks green from {_display_path(_path(source))}",
        {
            "missing": missing,
            "spectator_passed": bool(spectator.get("passed")) if isinstance(spectator, dict) else False,
            "checks_passed": int(spectator.get("checks_passed", 0) or 0) if isinstance(spectator, dict) else 0,
            "checks": int(spectator.get("checks", 0) or 0) if isinstance(spectator, dict) else 0,
        },
    )


def _evidence_integrity_check(
    operational: dict[str, Any],
    evidence: dict[str, Any],
    operational_source: str | Path,
    evidence_source: str | Path,
) -> dict[str, Any]:
    passed = bool(
        isinstance(operational, dict)
        and operational.get("passed")
        and isinstance(evidence, dict)
        and evidence.get("passed")
        and int(evidence.get("failure_count", len(evidence.get("failures", []) or [])) or 0) == 0
    )
    return _check(
        "release_evidence_integrity",
        passed,
        f"operational={bool(operational.get('passed')) if isinstance(operational, dict) else False}, evidence={bool(evidence.get('passed')) if isinstance(evidence, dict) else False}",
        {
            "operational_path": _display_path(_path(operational_source)),
            "operational_checks_passed": int(operational.get("checks_passed", 0) or 0) if isinstance(operational, dict) else 0,
            "operational_checks": int(operational.get("checks", 0) or 0) if isinstance(operational, dict) else 0,
            "evidence_path": _display_path(_path(evidence_source)),
            "evidence_checks_passed": int(evidence.get("checks_passed", 0) or 0) if isinstance(evidence, dict) else 0,
            "evidence_checks": int(evidence.get("checks", 0) or 0) if isinstance(evidence, dict) else 0,
            "evidence_failures": len(evidence.get("failures", []) or []) if isinstance(evidence, dict) else 0,
        },
    )


def _v2_scores(
    league: dict[str, Any],
    replay: dict[str, Any],
    spectator: dict[str, Any],
    operational: dict[str, Any],
    evidence: dict[str, Any],
) -> dict[str, float]:
    league_scores = league.get("scores", {}) if isinstance(league, dict) else {}
    aggregates = league.get("aggregates", {}) if isinstance(league, dict) else {}
    replay_score = replay.get("score", {}) if isinstance(replay, dict) else {}
    return {
        "league_overall_score": float(league_scores.get("overall_score", 0.0) or 0.0),
        "pursuer_security_score": float(league_scores.get("pursuer_security_score", 0.0) or 0.0),
        "evader_pressure_score": float(league_scores.get("evader_pressure_score", 0.0) or 0.0),
        "adversarial_balance_score": float(league_scores.get("adversarial_balance_score", 0.0) or 0.0),
        "mean_spoof_susceptibility": float(aggregates.get("mean_spoof_susceptibility", 0.0) or 0.0),
        "mean_deception_lift": float(aggregates.get("mean_deception_lift", 0.0) or 0.0),
        "mean_confidence_damage": float(aggregates.get("mean_confidence_damage", 0.0) or 0.0),
        "replay_report_score": float(replay_score.get("overall", 0.0) or 0.0),
        "spectator_score": float(spectator.get("score", 0.0) or 0.0) if isinstance(spectator, dict) else 0.0,
        "operational_score": float(operational.get("score", 0.0) or 0.0) if isinstance(operational, dict) else 0.0,
        "evidence_score": (
            float(evidence.get("checks_passed", 0) or 0) / max(1.0, float(evidence.get("checks", 0) or 0))
            if isinstance(evidence, dict)
            else 0.0
        ),
    }


def _summary_text(checks: list[dict[str, Any]], passed: bool) -> str:
    if passed:
        return "V2 research-grade candidate evidence is complete for the current thresholds."
    failed = [check["name"] for check in checks if not check["passed"]]
    return f"V2 evidence still needs work: {', '.join(failed)}."


def _check(name: str, passed: bool, summary: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "summary": summary,
        "metrics": metrics,
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)
