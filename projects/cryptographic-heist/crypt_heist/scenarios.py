"""Replay-backed acceptance scenarios for the operational dashboard."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .comms import VOCABULARY
from .policy_runtime import EvaderCheckpointController, PursuerTeamCheckpointController
from .replay import ReplayRecorder, load_replay, summarize_replay, validate_replay
from .sim import HeistSim

ACCEPTANCE_VERSION = 1
SCENARIO_NAMES = ("downtown_chase", "roadblock", "spoof_burst", "cipher_shift")


@dataclass
class ScenarioThresholds:
    """Tunable gate values for early smoke checks and later milestone checks."""

    downtown_min_waypoints: int = 0
    downtown_target_waypoints: int = 3
    downtown_min_radio_events: int = 24
    downtown_min_avg_speed: float = 4.0
    roadblock_min_capture_events: int = 1
    roadblock_min_boxed_frames: int = 90
    roadblock_max_capture_time: float = 2.5
    spoof_min_spoofed_events: int = 5
    spoof_min_deception: float = 1.0
    spoof_min_confidence_drop: float = 0.08
    spoof_min_active_pursuers: int = 1
    cipher_min_rotations: int = 1
    cipher_min_confidence_drop: float = 0.15
    cipher_min_recovery: float = 0.02


def run_acceptance_suite(
    *,
    scenarios: list[str] | tuple[str, ...] | None = None,
    seed: int = 11,
    steps: int | None = None,
    record_dir: str | Path | None = "replays/acceptance",
    out: str | Path | None = "logs/acceptance_scenarios.json",
    thresholds: ScenarioThresholds | None = None,
    action_factory: Callable[[str], Callable[[HeistSim, int], dict[str, Any] | None] | None] | None = None,
) -> dict[str, Any]:
    """Run named scenarios and write a manifest with replay-backed evidence."""
    thresholds = thresholds or ScenarioThresholds()
    requested = list(scenarios or SCENARIO_NAMES)
    unknown = [name for name in requested if name not in SCENARIO_NAMES]
    if unknown:
        raise ValueError(f"unknown scenario(s): {', '.join(unknown)}")

    results = [
        run_acceptance_scenario(
            name=name,
            seed=seed,
            steps=steps,
            record_dir=record_dir,
            thresholds=thresholds,
            action_fn=action_factory(name) if action_factory is not None else None,
        )
        for name in requested
    ]
    score = score_acceptance_manifest({"scenarios": results})
    diagnostics = _aggregate_diagnostics(results)
    manifest = {
        "version": ACCEPTANCE_VERSION,
        "seed": int(seed),
        "requested_scenarios": requested,
        "passed": all(result["passed"] for result in results),
        "required_checks_passed": _count_checks(results, required=True, passed=True),
        "required_checks": _count_checks(results, required=True, passed=None),
        "milestone_targets_passed": _count_checks(results, required=False, passed=True),
        "milestone_targets": _count_checks(results, required=False, passed=None),
        "score": score,
        "diagnostics": diagnostics,
        "scenarios": results,
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["manifest"] = str(path)
    return manifest


def run_acceptance_scenario(
    *,
    name: str,
    seed: int = 11,
    steps: int | None = None,
    record_dir: str | Path | None = "replays/acceptance",
    thresholds: ScenarioThresholds | None = None,
    action_fn: Callable[[HeistSim, int], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    """Run one named scenario and return a serializable result row."""
    thresholds = thresholds or ScenarioThresholds()
    if name == "downtown_chase":
        return _run_downtown_chase(seed, steps, record_dir, thresholds, action_fn)
    if name == "roadblock":
        return _run_roadblock(seed, steps, record_dir, thresholds, action_fn)
    if name == "spoof_burst":
        return _run_spoof_burst(seed, steps, record_dir, thresholds, action_fn)
    if name == "cipher_shift":
        return _run_cipher_shift(seed, steps, record_dir, thresholds, action_fn)
    raise ValueError(f"unknown scenario: {name}")


def run_control_acceptance_suite(
    *,
    evader_checkpoint: str | Path,
    pursuer_team_checkpoint: str | Path,
    scenarios: list[str] | tuple[str, ...] | None = None,
    seed: int = 31,
    steps: int | None = None,
    record_dir: str | Path | None = "replays/control_acceptance",
    out: str | Path | None = "logs/control_acceptance_scenarios.json",
    thresholds: ScenarioThresholds | None = None,
) -> dict[str, Any]:
    """Run acceptance scenarios with a learned evader and learned pursuer team."""
    return run_acceptance_suite(
        scenarios=scenarios,
        seed=seed,
        steps=steps,
        record_dir=record_dir,
        out=out,
        thresholds=thresholds,
        action_factory=lambda _name: checkpoint_control_action_factory(
            evader_checkpoint=evader_checkpoint,
            pursuer_team_checkpoint=pursuer_team_checkpoint,
        ),
    )


def run_checkpoint_acceptance_suite(
    *,
    evader_checkpoint: str | Path | None = None,
    pursuer_team_checkpoint: str | Path | None = None,
    scenarios: list[str] | tuple[str, ...] | None = None,
    seed: int = 31,
    steps: int | None = None,
    record_dir: str | Path | None = "replays/checkpoint_acceptance",
    out: str | Path | None = "logs/checkpoint_acceptance_scenarios.json",
    thresholds: ScenarioThresholds | None = None,
) -> dict[str, Any]:
    """Run acceptance scenarios with any available learned-control checkpoint(s)."""
    return run_acceptance_suite(
        scenarios=scenarios,
        seed=seed,
        steps=steps,
        record_dir=record_dir,
        out=out,
        thresholds=thresholds,
        action_factory=lambda _name: checkpoint_control_action_factory(
            evader_checkpoint=evader_checkpoint,
            pursuer_team_checkpoint=pursuer_team_checkpoint,
        ),
    )


def score_acceptance_manifest(manifest: dict[str, Any]) -> dict[str, float]:
    """Convert scenario check pass ratios into a compact promotion score."""
    scenarios = manifest.get("scenarios", [])
    required_total = sum(int(row.get("required_checks", 0)) for row in scenarios)
    required_passed = sum(int(row.get("required_checks_passed", 0)) for row in scenarios)
    milestone_total = sum(int(row.get("milestone_targets", 0)) for row in scenarios)
    milestone_passed = sum(int(row.get("milestone_targets_passed", 0)) for row in scenarios)
    required_ratio = required_passed / max(1, required_total)
    milestone_ratio = milestone_passed / max(1, milestone_total)
    overall = 0.86 * required_ratio + 0.14 * milestone_ratio if milestone_total else required_ratio
    by_name = {str(row.get("name", "")): _score_scenario_row(row) for row in scenarios}
    evader_score = by_name.get("downtown_chase", overall)
    team_score = by_name.get("roadblock", overall)
    information_scores = [
        by_name[name]
        for name in ("spoof_burst", "cipher_shift")
        if name in by_name
    ]
    information_score = _mean(information_scores) if information_scores else overall
    return {
        "overall_score": float(np.clip(overall, 0.0, 1.0)),
        "required_score": float(np.clip(required_ratio, 0.0, 1.0)),
        "milestone_score": float(np.clip(milestone_ratio, 0.0, 1.0)),
        "evader_control_score": float(np.clip(evader_score, 0.0, 1.0)),
        "pursuer_team_control_score": float(np.clip(team_score, 0.0, 1.0)),
        "information_warfare_score": float(np.clip(information_score, 0.0, 1.0)),
        "required_checks": float(required_total),
        "required_checks_passed": float(required_passed),
        "milestone_targets": float(milestone_total),
        "milestone_targets_passed": float(milestone_passed),
    }


def _run_downtown_chase(
    seed: int,
    steps: int | None,
    record_dir: str | Path | None,
    thresholds: ScenarioThresholds,
    action_fn: Callable[[HeistSim, int], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    frames = int(steps or 1200)
    record = _record_path(record_dir, "downtown_chase", seed)
    sim = HeistSim(seed=seed, reset_on_capture=True)
    run = _run_recorded(sim, frames, record, action_fn=action_fn)
    metrics = run["metrics"]
    summary = run["replay_summary"]
    checks = [
        _check("frames_recorded", summary["frames"], frames, ">="),
        _check("radio_events", summary["radio_events"], thresholds.downtown_min_radio_events, ">="),
        _check("avg_evader_speed", summary["avg_evader_speed"], thresholds.downtown_min_avg_speed, ">="),
        _check("waypoints_hit", metrics["waypoints_hit"], thresholds.downtown_min_waypoints, ">="),
        _check(
            "original_plan_waypoint_target",
            metrics["waypoints_hit"],
            thresholds.downtown_target_waypoints,
            ">=",
            required=False,
            note="Full milestone target from the build plan; not required for the early operational smoke gate.",
        ),
    ]
    return _scenario_result("downtown_chase", seed, frames, record, metrics, summary, checks)


def _run_roadblock(
    seed: int,
    steps: int | None,
    record_dir: str | Path | None,
    thresholds: ScenarioThresholds,
    action_fn: Callable[[HeistSim, int], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    frames = int(steps or 320)
    record = _record_path(record_dir, "roadblock", seed)
    sim = HeistSim(seed=seed, reset_on_capture=False)
    _configure_roadblock(sim)
    run = _run_recorded(sim, frames, record, action_fn=action_fn or _brake_all_actions, stop_on_done=True)
    metrics = run["metrics"]
    summary = run["replay_summary"]
    checks = [
        _check("capture_events", metrics["capture_events"], thresholds.roadblock_min_capture_events, ">="),
        _check("boxed_frames", metrics["boxed_frames"], thresholds.roadblock_min_boxed_frames, ">="),
        _check("capture_time", metrics["first_capture_time"], thresholds.roadblock_max_capture_time, "<="),
        _check("max_impact_finite", summary["max_impact"], 1.0e9, "<="),
    ]
    return _scenario_result("roadblock", seed, frames, record, metrics, summary, checks)


def _run_spoof_burst(
    seed: int,
    steps: int | None,
    record_dir: str | Path | None,
    thresholds: ScenarioThresholds,
    action_fn: Callable[[HeistSim, int], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    frames = int(steps or 300)
    record = _record_path(record_dir, "spoof_burst", seed)
    sim = HeistSim(seed=seed, reset_on_capture=True)
    sim.jam_cooldown = 0.0
    pre_confidence = float(sim.channel.confidence)
    run = _run_recorded(sim, frames, record, action_fn=_combine_action_fns(action_fn, _spoof_once_actions))
    metrics = run["metrics"]
    summary = run["replay_summary"]
    confidence_drop = max(0.0, pre_confidence - metrics["min_confidence"])
    metrics["initial_confidence"] = pre_confidence
    metrics["confidence_drop"] = confidence_drop
    checks = [
        _check("spoofed_radio_events", summary["spoofed_radio_events"], thresholds.spoof_min_spoofed_events, ">="),
        _check("deception_score", metrics["deception_score"], thresholds.spoof_min_deception, ">="),
        _check("confidence_drop", confidence_drop, thresholds.spoof_min_confidence_drop, ">="),
        _check("max_active_spoofed_pursuers", metrics["max_active_spoofed_pursuers"], thresholds.spoof_min_active_pursuers, ">="),
    ]
    return _scenario_result("spoof_burst", seed, frames, record, metrics, summary, checks)


def _run_cipher_shift(
    seed: int,
    steps: int | None,
    record_dir: str | Path | None,
    thresholds: ScenarioThresholds,
    action_fn: Callable[[HeistSim, int], dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    frames = int(steps or 420)
    record = _record_path(record_dir, "cipher_shift", seed)
    sim = HeistSim(seed=seed, reset_on_capture=True)
    sim.channel.confidence = 0.92
    sim.channel.cipher_timer = 17.96
    pre_confidence = float(sim.channel.confidence)
    run = _run_recorded(sim, frames, record, action_fn=action_fn)
    metrics = run["metrics"]
    summary = run["replay_summary"]
    confidence_drop = max(0.0, pre_confidence - metrics["min_confidence"])
    recovery = max(0.0, metrics["final_confidence"] - metrics["min_confidence"])
    metrics["initial_confidence"] = pre_confidence
    metrics["confidence_drop"] = confidence_drop
    metrics["confidence_recovery"] = recovery
    checks = [
        _check("cipher_rotations", metrics["cipher_rotations"], thresholds.cipher_min_rotations, ">="),
        _check("confidence_drop", confidence_drop, thresholds.cipher_min_confidence_drop, ">="),
        _check("confidence_recovery", recovery, thresholds.cipher_min_recovery, ">="),
        _check("radio_events", summary["radio_events"], 6, ">="),
    ]
    return _scenario_result("cipher_shift", seed, frames, record, metrics, summary, checks)


def _run_recorded(
    sim: HeistSim,
    steps: int,
    record: Path | None,
    *,
    action_fn: Callable[[HeistSim, int], dict[str, Any] | None] | None = None,
    stop_on_done: bool = False,
) -> dict[str, Any]:
    confidence_values: list[float] = []
    evader_speeds: list[float] = []
    min_distances: list[float] = []
    waypoint_distances: list[float] = []
    active_spoofed_counts: list[int] = []
    boxed_frames = 0
    pinned_frames = 0
    capture_events = 0
    waypoint_events = 0
    first_capture_time: float | None = None
    cipher_keys: set[tuple[float, str]] = set()

    recorder_cm = ReplayRecorder(record, seed=int(sim.seed or 0), dt=sim.sim.dt) if record is not None else nullcontext(None)
    executed_frames = 0
    with recorder_cm as recorder:
        for i in range(int(steps)):
            actions = action_fn(sim, i) if action_fn is not None else None
            sim.step(actions=actions)
            executed_frames += 1
            confidence_values.append(float(sim.channel.confidence))
            evader_speeds.append(float(sim.evader.vehicle.speed))
            dists = _pursuer_distances(sim)
            min_distances.append(min(dists))
            waypoint = sim.evader_planner.waypoint
            waypoint_distances.append(float(np.hypot(
                waypoint[0] - sim.evader.vehicle.x,
                waypoint[1] - sim.evader.vehicle.y,
            )))
            boxed_frames += int(sum(distance < 22.0 for distance in dists) >= 3)
            pinned_frames += int(min(dists) < 11.0 and sim.evader.vehicle.speed < 6.5)
            active_count = sum(int(p.spoof_timer > 0.0) for p in sim.pursuers)
            active_spoofed_counts.append(active_count)
            capture_events += int(sim.capture_event)
            waypoint_events += int(sim.waypoint_event)
            if sim.capture_event and first_capture_time is None:
                first_capture_time = float(sim.time)
            for event in sim.channel.events:
                if event.speaker == "dispatch" and event.word == "cipher":
                    cipher_keys.add((round(float(event.time), 6), event.word))
            if recorder is not None:
                recorder.record(sim.snapshot())
            if stop_on_done and sim.episode_done:
                break

    replay_summary: dict[str, Any]
    if record is not None:
        lines = load_replay(record)
        validate_replay(lines)
        replay_summary = summarize_replay(lines)
    else:
        replay_summary = {
            "frames": executed_frames,
            "radio_events": len(sim.channel.events),
            "spoofed_radio_events": sum(1 for event in sim.channel.events if event.spoofed),
            "avg_evader_speed": _mean(evader_speeds),
            "max_impact": float(sim.impact_energy),
        }

    metrics = {
        "requested_steps": int(steps),
        "frames": int(replay_summary.get("frames", executed_frames)),
        "sim_time": float(sim.time),
        "captures": int(sim.captures),
        "capture_events": int(capture_events),
        "first_capture_time": float(first_capture_time if first_capture_time is not None else 1.0e9),
        "waypoints_hit": int(sim.waypoints_hit),
        "waypoint_events": int(waypoint_events),
        "deception_score": float(sim.deception_score),
        "avg_confidence": _mean(confidence_values),
        "min_confidence": min(confidence_values) if confidence_values else float(sim.channel.confidence),
        "final_confidence": float(sim.channel.confidence),
        "avg_evader_speed": _mean(evader_speeds),
        "final_evader_speed": float(sim.evader.vehicle.speed),
        "min_pursuer_distance": min(min_distances) if min_distances else 0.0,
        "closest_waypoint_distance": min(waypoint_distances) if waypoint_distances else 0.0,
        "final_waypoint_distance": waypoint_distances[-1] if waypoint_distances else 0.0,
        "boxed_frames": int(boxed_frames),
        "pinned_frames": int(pinned_frames),
        "cipher_rotations": int(len(cipher_keys)),
        "max_active_spoofed_pursuers": int(max(active_spoofed_counts) if active_spoofed_counts else 0),
        "jamming_budget": int(sim.channel.jamming_budget),
        "last_spoof_words": list(sim.channel.last_spoof_words),
    }
    return {"metrics": metrics, "replay_summary": replay_summary}


def _configure_roadblock(sim: HeistSim) -> None:
    ev = sim.evader.vehicle
    ev.vx = 2.0
    ev.vy = 0.0
    ev.r = 0.0
    ev.yaw = 0.0
    placements = [
        (9.0, 0.0),
        (-15.5, 0.0),
        (0.0, 15.5),
        (0.0, -15.5),
        (20.5, 0.0),
    ]
    for i, (dx, dy) in enumerate(placements):
        p = sim.pursuers[i].vehicle
        p.reset(ev.x + dx, ev.y + dy, yaw=float(np.arctan2(-dy, -dx)), speed=0.5)
        sim.pursuers[i].spoof_timer = 0.0
        sim.pursuers[i].spoof_offset[:] = 0.0
    sim.capture_timer = 0.0


def _brake_all_actions(sim: HeistSim, i: int) -> dict[str, Any]:
    del sim, i
    brake = {"control": np.asarray([0.0, -1.0, -1.0], dtype=np.float32)}
    actions = {"evader_0": dict(brake)}
    for idx in range(5):
        actions[f"pursuer_{idx}"] = dict(brake)
    return actions


def _spoof_once_actions(sim: HeistSim, i: int) -> dict[str, Any] | None:
    del sim
    if i != 0:
        return None
    return {
        "evader_0": {
            "jam": 1,
            "spoof_tokens": np.asarray([
                _token("alpha"),
                _token("gate"),
                _token("east"),
                _token("switch"),
                _token("seal"),
            ], dtype=np.int64),
            "target_mask": np.asarray([1, 1, 1, 0, 0], dtype=np.float32),
        }
    }


def checkpoint_control_action_factory(
    *,
    evader_checkpoint: str | Path | None = None,
    pursuer_team_checkpoint: str | Path | None = None,
) -> Callable[[HeistSim, int], dict[str, Any]]:
    if evader_checkpoint is None and pursuer_team_checkpoint is None:
        raise ValueError("at least one checkpoint is required")
    evader = EvaderCheckpointController(evader_checkpoint) if evader_checkpoint is not None else None
    pursuers = PursuerTeamCheckpointController(pursuer_team_checkpoint) if pursuer_team_checkpoint is not None else None

    def _action(sim: HeistSim, i: int) -> dict[str, Any]:
        del i
        return _merge_action_dicts(
            evader.action(sim) if evader is not None else None,
            pursuers.action(sim) if pursuers is not None else None,
        )

    return _action


def _combine_action_fns(
    *fns: Callable[[HeistSim, int], dict[str, Any] | None] | None,
) -> Callable[[HeistSim, int], dict[str, Any] | None]:
    live = [fn for fn in fns if fn is not None]
    if not live:
        return lambda _sim, _i: None

    def _action(sim: HeistSim, i: int) -> dict[str, Any] | None:
        return _merge_action_dicts(*(fn(sim, i) for fn in live))

    return _action


def _merge_action_dicts(*action_dicts: dict[str, Any] | None) -> dict[str, Any] | None:
    merged: dict[str, Any] = {}
    for actions in action_dicts:
        if not actions:
            continue
        for agent, payload in actions.items():
            existing = merged.setdefault(agent, {})
            if isinstance(existing, dict) and isinstance(payload, dict):
                existing.update(dict(payload))
            else:
                merged[agent] = payload
    return merged or None


def _scenario_result(
    name: str,
    seed: int,
    steps: int,
    record: Path | None,
    metrics: dict[str, Any],
    replay_summary: dict[str, Any],
    checks: list[dict[str, Any]],
) -> dict[str, Any]:
    required_checks = [check for check in checks if check["required"]]
    diagnostics = _scenario_diagnostics(name, metrics, replay_summary, checks)
    return {
        "name": name,
        "seed": int(seed),
        "steps": int(steps),
        "passed": all(check["passed"] for check in required_checks),
        "required_checks_passed": sum(int(check["passed"]) for check in required_checks),
        "required_checks": len(required_checks),
        "milestone_targets_passed": sum(int(check["passed"]) for check in checks if not check["required"]),
        "milestone_targets": sum(1 for check in checks if not check["required"]),
        "replay": str(record) if record is not None else None,
        "metrics": metrics,
        "replay_summary": replay_summary,
        "checks": checks,
        "diagnostics": diagnostics,
        "primary_diagnostic": diagnostics[0] if diagnostics else None,
    }


def _check(
    name: str,
    actual: float | int,
    expected: float | int,
    op: str,
    *,
    required: bool = True,
    note: str = "",
) -> dict[str, Any]:
    actual_f = float(actual)
    expected_f = float(expected)
    if op == ">=":
        passed = actual_f >= expected_f
    elif op == "<=":
        passed = actual_f <= expected_f
    else:
        raise ValueError(f"unsupported check operator: {op}")
    return {
        "name": name,
        "passed": bool(passed),
        "actual": _number(actual),
        "expected": _number(expected),
        "op": op,
        "required": bool(required),
        "note": note,
    }


def _count_checks(results: list[dict[str, Any]], *, required: bool, passed: bool | None) -> int:
    count = 0
    for result in results:
        for check in result["checks"]:
            if check["required"] != required:
                continue
            if passed is not None and check["passed"] != passed:
                continue
            count += 1
    return count


def _scenario_diagnostics(
    name: str,
    metrics: dict[str, Any],
    replay_summary: dict[str, Any],
    checks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    failed_checks = [check for check in checks if not check["passed"]]
    for check in failed_checks:
        diagnostics.append(_check_diagnostic(name, check))

    if name == "downtown_chase":
        _diagnose_downtown(metrics, replay_summary, checks, diagnostics)
    elif name == "roadblock":
        _diagnose_roadblock(metrics, checks, diagnostics)
    elif name == "spoof_burst":
        _diagnose_spoof(metrics, replay_summary, checks, diagnostics)
    elif name == "cipher_shift":
        _diagnose_cipher(metrics, checks, diagnostics)

    if not diagnostics:
        diagnostics.append({
            "owner": _scenario_owner(name),
            "reason": "scenario_nominal",
            "severity": "info",
            "message": f"{name} passed all required checks.",
            "evidence": {},
            "suggestion": "Keep this scenario in the promotion gate and raise thresholds when training stabilizes.",
        })
    return sorted(diagnostics, key=lambda item: _severity_rank(item["severity"]), reverse=True)


def _check_diagnostic(name: str, check: dict[str, Any]) -> dict[str, Any]:
    severity = "high" if check.get("required") else "medium"
    reason = "failed_required_check" if check.get("required") else "missed_milestone_target"
    return {
        "owner": _scenario_owner(name),
        "reason": reason,
        "severity": severity,
        "message": f"{check['name']} was {check['actual']} but expected {check['op']} {check['expected']}.",
        "evidence": {
            "check": check["name"],
            "actual": check["actual"],
            "expected": check["expected"],
            "op": check["op"],
        },
        "suggestion": _scenario_suggestion(name),
    }


def _diagnose_downtown(
    metrics: dict[str, Any],
    replay_summary: dict[str, Any],
    checks: list[dict[str, Any]],
    out: list[dict[str, Any]],
) -> None:
    waypoint_check = _check_by_name(checks, "waypoints_hit")
    milestone_check = _check_by_name(checks, "original_plan_waypoint_target")
    speed_check = _check_by_name(checks, "avg_evader_speed")
    max_impact = float(replay_summary.get("max_impact", 0.0))
    if waypoint_check and not waypoint_check["passed"]:
        out.append(_diagnostic(
            "evader",
            "insufficient_waypoint_progress",
            "high",
            "The evader did not hit enough waypoints during the downtown chase.",
            metrics,
            "Increase waypoint-following stability, reduce wall pinning, or improve evader control training.",
            fields=("waypoints_hit", "closest_waypoint_distance", "final_waypoint_distance", "avg_evader_speed"),
        ))
    elif milestone_check and not milestone_check["passed"]:
        out.append(_diagnostic(
            "evader",
            "waypoint_milestone_gap",
            "medium",
            "The smoke gate passed, but the full downtown waypoint milestone is still short.",
            metrics,
            "Use this seed as an evader curriculum target and raise the required waypoint threshold gradually.",
            fields=("waypoints_hit", "closest_waypoint_distance", "final_waypoint_distance", "avg_evader_speed"),
        ))
    if speed_check and not speed_check["passed"]:
        out.append(_diagnostic(
            "evader",
            "speed_collapse",
            "high",
            "The evader's average speed fell below the chase-readiness threshold.",
            metrics,
            "Tune throttle recovery, collision recovery, or reward shaping for useful forward speed.",
            fields=("avg_evader_speed", "final_evader_speed", "closest_waypoint_distance"),
        ))
    if float(metrics.get("final_evader_speed", 0.0)) < 1.0 and float(metrics.get("final_waypoint_distance", 0.0)) > 18.0:
        out.append(_diagnostic(
            "evader",
            "stalled_before_waypoint",
            "medium",
            "The evader ended nearly stationary before reaching the active waypoint.",
            metrics,
            "Inspect collision recovery and local waypoint selection around the final position.",
            fields=("final_evader_speed", "final_waypoint_distance", "closest_waypoint_distance"),
        ))
    if max_impact > 18.0:
        out.append({
            "owner": "evader",
            "reason": "high_collision_energy",
            "severity": "medium",
            "message": "Downtown chase involved a large impact that may be causing route failure.",
            "evidence": {"max_impact": _number(max_impact)},
            "suggestion": "Penalize hard wall strikes or improve road-following recovery.",
        })


def _diagnose_roadblock(metrics: dict[str, Any], checks: list[dict[str, Any]], out: list[dict[str, Any]]) -> None:
    if not _passed(checks, "capture_events"):
        out.append(_diagnostic(
            "pursuer_team",
            "capture_not_confirmed",
            "high",
            "The roadblock did not produce a capture event.",
            metrics,
            "Tighten containment spacing, improve low-speed pin behavior, or extend capture hold time in training.",
            fields=("capture_events", "boxed_frames", "pinned_frames", "min_pursuer_distance"),
        ))
    if not _passed(checks, "boxed_frames"):
        out.append(_diagnostic(
            "pursuer_team",
            "weak_boxing",
            "high",
            "Pursuers did not sustain enough boxed frames around the evader.",
            metrics,
            "Improve formation assignment, lateral spacing, and role-specific roadblock training.",
            fields=("boxed_frames", "pinned_frames", "capture_events", "min_pursuer_distance"),
        ))
    if not _passed(checks, "capture_time"):
        out.append(_diagnostic(
            "pursuer_team",
            "slow_capture",
            "medium",
            "The roadblock captured too slowly for the acceptance threshold.",
            metrics,
            "Reward faster closure and reduce overshoot around pinned evaders.",
            fields=("first_capture_time", "boxed_frames", "pinned_frames"),
        ))


def _diagnose_spoof(
    metrics: dict[str, Any],
    replay_summary: dict[str, Any],
    checks: list[dict[str, Any]],
    out: list[dict[str, Any]],
) -> None:
    if not _passed(checks, "spoofed_radio_events"):
        out.append({
            "owner": "information_warfare",
            "reason": "spoof_delivery_missing",
            "severity": "high",
            "message": "The evader did not inject enough spoofed radio words.",
            "evidence": {"spoofed_radio_events": int(replay_summary.get("spoofed_radio_events", 0))},
            "suggestion": "Check jammer trigger, jamming budget, and spoof token action plumbing.",
        })
    if not _passed(checks, "deception_score"):
        out.append(_diagnostic(
            "information_warfare",
            "jam_not_deceptive",
            "high",
            "Spoof bursts were not counted as meaningful deception.",
            metrics,
            "Train the jammer against counterfactual pursuer deviation and target active pursuers.",
            fields=("deception_score", "max_active_spoofed_pursuers", "last_spoof_words"),
        ))
    if not _passed(checks, "confidence_drop"):
        out.append(_diagnostic(
            "information_warfare",
            "confidence_not_damaged",
            "medium",
            "The spoof burst did not reduce scanner confidence enough.",
            metrics,
            "Increase jammer payload pressure or make scanner confidence depend more strongly on spoof disruption.",
            fields=("confidence_drop", "min_confidence", "initial_confidence"),
        ))
    if not _passed(checks, "max_active_spoofed_pursuers"):
        out.append(_diagnostic(
            "information_warfare",
            "spoof_targets_inactive",
            "medium",
            "Spoofed commands did not affect enough pursuers.",
            metrics,
            "Prefer target masks for nearby pursuers and extend spoof-offset duration in curriculum probes.",
            fields=("max_active_spoofed_pursuers", "deception_score", "last_spoof_words"),
        ))


def _diagnose_cipher(metrics: dict[str, Any], checks: list[dict[str, Any]], out: list[dict[str, Any]]) -> None:
    if not _passed(checks, "cipher_rotations"):
        out.append(_diagnostic(
            "information_warfare",
            "cipher_rotation_missing",
            "high",
            "The scenario did not trigger a pursuer cipher rotation.",
            metrics,
            "Verify the comms epoch timer and scenario setup.",
            fields=("cipher_rotations", "sim_time"),
        ))
    if not _passed(checks, "confidence_drop"):
        out.append(_diagnostic(
            "information_warfare",
            "cipher_shift_not_visible",
            "medium",
            "Cipher rotation did not create a large enough confidence drop.",
            metrics,
            "Increase confidence damage from auth rotations or improve scanner uncertainty modeling.",
            fields=("confidence_drop", "min_confidence", "initial_confidence"),
        ))
    if not _passed(checks, "confidence_recovery"):
        out.append(_diagnostic(
            "information_warfare",
            "scanner_recovery_weak",
            "medium",
            "Scanner confidence did not recover after the cipher shift.",
            metrics,
            "Train decoder adaptation on post-rotation transcript windows.",
            fields=("confidence_recovery", "final_confidence", "min_confidence"),
        ))


def _diagnostic(
    owner: str,
    reason: str,
    severity: str,
    message: str,
    metrics: dict[str, Any],
    suggestion: str,
    *,
    fields: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "owner": owner,
        "reason": reason,
        "severity": severity,
        "message": message,
        "evidence": {field: _jsonable(metrics.get(field)) for field in fields if field in metrics},
        "suggestion": suggestion,
    }


def _aggregate_diagnostics(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {**diagnostic, "scenario": result["name"]}
        for result in results
        for diagnostic in result.get("diagnostics", [])
        if diagnostic.get("reason") != "scenario_nominal"
    ]
    rows.sort(key=lambda item: _severity_rank(item["severity"]), reverse=True)
    by_owner: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    for row in rows:
        by_owner[row["owner"]] = by_owner.get(row["owner"], 0) + 1
        by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1
    return {
        "count": len(rows),
        "high_count": sum(1 for row in rows if row["severity"] == "high"),
        "by_owner": by_owner,
        "by_reason": by_reason,
        "top": rows[:6],
    }


def _check_by_name(checks: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((check for check in checks if check["name"] == name), None)


def _passed(checks: list[dict[str, Any]], name: str) -> bool:
    check = _check_by_name(checks, name)
    return bool(check is None or check.get("passed"))


def _scenario_owner(name: str) -> str:
    if name == "downtown_chase":
        return "evader"
    if name == "roadblock":
        return "pursuer_team"
    return "information_warfare"


def _scenario_suggestion(name: str) -> str:
    if name == "downtown_chase":
        return "Improve evader waypoint driving, collision recovery, and useful speed."
    if name == "roadblock":
        return "Improve pursuer boxing, closure timing, and low-speed pin formation."
    if name == "spoof_burst":
        return "Improve jammer trigger timing, target selection, and spoof payload impact."
    return "Improve cipher rotation visibility and scanner adaptation."


def _severity_rank(severity: str) -> int:
    return {"info": 0, "low": 1, "medium": 2, "high": 3}.get(str(severity), 0)


def _score_scenario_row(row: dict[str, Any]) -> float:
    required_total = int(row.get("required_checks", 0))
    required_passed = int(row.get("required_checks_passed", 0))
    milestone_total = int(row.get("milestone_targets", 0))
    milestone_passed = int(row.get("milestone_targets_passed", 0))
    required_ratio = required_passed / max(1, required_total)
    if milestone_total <= 0:
        return float(required_ratio)
    milestone_ratio = milestone_passed / max(1, milestone_total)
    return float(0.78 * required_ratio + 0.22 * milestone_ratio)


def _record_path(record_dir: str | Path | None, name: str, seed: int) -> Path | None:
    if record_dir is None:
        return None
    root = Path(record_dir)
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{name}_seed{int(seed)}.jsonl"


def _pursuer_distances(sim: HeistSim) -> list[float]:
    ev = sim.evader.vehicle
    return [
        float(np.hypot(p.vehicle.x - ev.x, p.vehicle.y - ev.y))
        for p in sim.pursuers
    ]


def _token(word: str) -> int:
    try:
        return VOCABULARY.index(word)
    except ValueError:
        return 0


def _number(value: float | int) -> float | int:
    value_f = float(value)
    if value_f.is_integer():
        return int(value_f)
    return value_f


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float):
        return _number(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))
