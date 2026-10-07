"""Versioned JSONL replay recording and validation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .sim import HeistSim

REPLAY_VERSION = 1
FINGERPRINT_PRECISION = 6


def vehicle_frame(agent) -> dict[str, Any]:
    v = agent.vehicle
    return {
        "name": agent.name,
        "faction": agent.faction,
        "role": agent.role,
        "target": [float(agent.target[0]), float(agent.target[1])],
        "x": float(v.x),
        "y": float(v.y),
        "yaw": float(v.yaw),
        "vx": float(v.vx),
        "vy": float(v.vy),
        "speed": float(v.speed),
        "yaw_rate": float(v.r),
        "slip_angle": float(v.slip_angle),
        "wheel_grip": [float(x) for x in v.wheel_grip],
        "rpm": float(v.rpm),
        "gear": int(v.gear),
    }


def snapshot_to_frame(snapshot: dict, index: int) -> dict[str, Any]:
    agents = [vehicle_frame(agent) for agent in snapshot["agents"]]
    metrics = {
        "jamming_budget": int(snapshot["jamming_budget"]),
        "captures": int(snapshot["captures"]),
        "waypoints_hit": int(snapshot["waypoints_hit"]),
        "deception_score": float(snapshot["deception_score"]),
        "impact": float(snapshot["impact"]),
        "last_jam_time": float(snapshot["last_jam_time"]),
        "last_spoof_words": list(snapshot["last_spoof_words"]),
        "capture_event": bool(snapshot.get("capture_event", False)),
        "waypoint_event": bool(snapshot.get("waypoint_event", False)),
        "last_collision_impact": float(snapshot.get("last_collision_impact", 0.0)),
    }
    return {
        "type": "frame",
        "version": REPLAY_VERSION,
        "i": int(index),
        "time": float(snapshot["time"]),
        "agents": agents,
        "actions": _action_payload(snapshot.get("actions", {})),
        "waypoint": [float(snapshot["waypoint"][0]), float(snapshot["waypoint"][1])],
        "confidence": float(snapshot["confidence"]),
        "radio": [asdict(event) for event in snapshot["events"]],
        "predictions": {
            name: [float(pos[0]), float(pos[1])]
            for name, pos in snapshot["predictions"].items()
        },
        "metrics": metrics,
        "collisions": _collision_payload(snapshot.get("collisions", [])),
        "rewards": _numeric_mapping(snapshot.get("rewards", {})),
        "reward_components": _numeric_mapping(snapshot.get("reward_components", {})),
        "camera": _camera_payload(snapshot, agents, metrics),
    }


class ReplayRecorder:
    def __init__(self, path: str | Path, seed: int, dt: float):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.dt = dt
        self.frames = 0
        self._fh = None

    def __enter__(self):
        self._fh = self.path.open("w", encoding="utf-8")
        self._write({"type": "meta", "version": REPLAY_VERSION, "seed": self.seed, "dt": self.dt})
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh:
            self._fh.close()

    def record(self, snapshot: dict):
        self._write(snapshot_to_frame(snapshot, self.frames))
        self.frames += 1

    def _write(self, payload: dict):
        if self._fh is None:
            raise RuntimeError("ReplayRecorder must be used as a context manager")
        self._fh.write(json.dumps(payload, separators=(",", ":")) + "\n")


def record_scripted_replay(path: str | Path, seed: int = 11, steps: int = 1200) -> dict[str, Any]:
    sim = HeistSim(seed=seed)
    with ReplayRecorder(path, seed=seed, dt=sim.sim.dt) as recorder:
        for _ in range(steps):
            sim.step()
            recorder.record(sim.snapshot())
    return {"frames": steps, "path": str(path)}


def load_replay(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def validate_replay(lines: list[dict[str, Any]]) -> None:
    if not lines:
        raise ValueError("empty replay")
    meta = lines[0]
    if meta.get("type") != "meta" or meta.get("version") != REPLAY_VERSION:
        raise ValueError("bad replay metadata")
    dt = float(meta.get("dt", 0.0))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("bad replay dt")
    if len(lines) < 2:
        raise ValueError("replay has no frames")
    previous_time = -np.inf
    for i, frame in enumerate(lines[1:]):
        if frame.get("type") != "frame":
            raise ValueError(f"line {i + 2}: expected frame")
        if int(frame.get("version", REPLAY_VERSION)) != REPLAY_VERSION:
            raise ValueError(f"line {i + 2}: bad frame version")
        if int(frame.get("i", -1)) != i:
            raise ValueError(f"line {i + 2}: bad frame index")
        for key in ("time", "agents", "radio", "waypoint", "confidence", "predictions", "metrics"):
            if key not in frame:
                raise ValueError(f"line {i + 2}: missing {key}")
        time_value = float(frame["time"])
        if not np.isfinite(time_value):
            raise ValueError(f"line {i + 2}: bad time")
        if time_value < previous_time:
            raise ValueError(f"line {i + 2}: non-monotonic time")
        previous_time = time_value
        if len(frame["agents"]) != 6:
            raise ValueError(f"line {i + 2}: expected 6 agents")
        if not np.isfinite(frame["confidence"]):
            raise ValueError(f"line {i + 2}: bad confidence")
        if len(frame["waypoint"]) != 2 or not all(np.isfinite(float(x)) for x in frame["waypoint"]):
            raise ValueError(f"line {i + 2}: bad waypoint")
        for agent in frame["agents"]:
            for key in ("name", "faction", "x", "y", "yaw", "vx", "vy", "speed", "slip_angle", "wheel_grip"):
                if key not in agent:
                    raise ValueError(f"line {i + 2}: agent missing {key}")
            for key in ("x", "y", "yaw", "vx", "vy", "speed", "slip_angle"):
                if not np.isfinite(float(agent[key])):
                    raise ValueError(f"line {i + 2}: bad agent {key}")
            if len(agent["wheel_grip"]) != 4 or not all(np.isfinite(float(x)) for x in agent["wheel_grip"]):
                raise ValueError(f"line {i + 2}: bad wheel grip")
        for name, prediction in frame["predictions"].items():
            if not isinstance(name, str) or len(prediction) != 2:
                raise ValueError(f"line {i + 2}: bad prediction")
            if not all(np.isfinite(float(x)) for x in prediction):
                raise ValueError(f"line {i + 2}: bad prediction")
        for event in frame["radio"]:
            for key in ("time", "speaker", "word", "spoofed"):
                if key not in event:
                    raise ValueError(f"line {i + 2}: radio missing {key}")
            if not np.isfinite(float(event["time"])):
                raise ValueError(f"line {i + 2}: bad radio time")
        if "actions" in frame:
            if not isinstance(frame["actions"], dict):
                raise ValueError(f"line {i + 2}: bad actions")
            for name, action in frame["actions"].items():
                if not isinstance(name, str) or not isinstance(action, dict):
                    raise ValueError(f"line {i + 2}: bad action payload")
                control = action.get("control", [])
                if len(control) != 4 or not all(np.isfinite(float(x)) for x in control):
                    raise ValueError(f"line {i + 2}: bad action control")
        if "collisions" in frame:
            if not isinstance(frame["collisions"], list):
                raise ValueError(f"line {i + 2}: bad collisions")
            for collision in frame["collisions"]:
                if not np.isfinite(float(collision.get("impact", 0.0))):
                    raise ValueError(f"line {i + 2}: bad collision impact")
        if "rewards" in frame and not isinstance(frame["rewards"], dict):
            raise ValueError(f"line {i + 2}: bad rewards")
        if "reward_components" in frame and not isinstance(frame["reward_components"], dict):
            raise ValueError(f"line {i + 2}: bad reward components")
        if "rewards" in frame:
            _validate_finite_payload(frame["rewards"], label=f"line {i + 2}: rewards")
        if "reward_components" in frame:
            _validate_finite_payload(frame["reward_components"], label=f"line {i + 2}: reward components")
        if "camera" in frame:
            camera = frame["camera"]
            if not isinstance(camera, dict):
                raise ValueError(f"line {i + 2}: bad camera")
            focus = camera.get("focus", [])
            if len(focus) != 2 or not all(np.isfinite(float(x)) for x in focus):
                raise ValueError(f"line {i + 2}: bad camera focus")


def summarize_replay(lines: list[dict[str, Any]], *, include_fingerprint: bool = True) -> dict[str, Any]:
    validate_replay(lines)
    frames = lines[1:]
    duration = frames[-1]["time"] - frames[0]["time"] if len(frames) > 1 else 0.0
    radio_events = sum(len(frame["radio"]) for frame in frames)
    spoofed_radio_events = sum(1 for frame in frames for event in frame["radio"] if event.get("spoofed"))
    unique_events = {
        _radio_event_key(event)
        for frame in frames
        for event in frame["radio"]
    }
    unique_spoofed_events = {
        _radio_event_key(event)
        for frame in frames
        for event in frame["radio"]
        if event.get("spoofed")
    }
    unique_radio_events = _unique_radio_events(frames)
    agents = len(frames[-1]["agents"]) if frames else 0
    final_metrics = frames[-1].get("metrics", {}) if frames else {}
    confidence_values = [float(frame.get("confidence", 0.0)) for frame in frames]
    evader_speeds = [
        float(agent.get("speed", 0.0))
        for frame in frames
        for agent in frame.get("agents", [])
        if agent.get("faction") == "evader"
    ]
    max_impact = max((float(frame.get("metrics", {}).get("impact", 0.0)) for frame in frames), default=0.0)
    action_frames = sum(1 for frame in frames if isinstance(frame.get("actions"), dict) and frame["actions"])
    camera_hint_frames = sum(1 for frame in frames if isinstance(frame.get("camera"), dict))
    collision_events = sum(len(frame.get("collisions", [])) for frame in frames)
    reward_frames = sum(
        1
        for frame in frames
        if isinstance(frame.get("rewards"), dict)
        and (frame.get("rewards") or frame.get("reward_components") is not None)
    )
    reward_stats = _reward_summary(frames)
    comms_stats = _comms_summary(frames, unique_radio_events, confidence_values)
    summary = {
        "frames": len(frames),
        "duration": duration,
        "schema_version": int(lines[0].get("version", REPLAY_VERSION)),
        "radio_events": radio_events,
        "spoofed_radio_events": spoofed_radio_events,
        "unique_radio_events": len(unique_events),
        "unique_spoofed_radio_events": len(unique_spoofed_events),
        "agents": agents,
        "captures": int(final_metrics.get("captures", 0)),
        "waypoints_hit": int(final_metrics.get("waypoints_hit", 0)),
        "deception_score": float(final_metrics.get("deception_score", 0.0)),
        "avg_confidence": sum(confidence_values) / max(1, len(confidence_values)),
        "avg_evader_speed": sum(evader_speeds) / max(1, len(evader_speeds)),
        "max_impact": max_impact,
        "action_frames": action_frames,
        "camera_hint_frames": camera_hint_frames,
        "collision_events": collision_events,
        "reward_frames": reward_frames,
        **reward_stats,
        **comms_stats,
    }
    if include_fingerprint:
        fingerprint = replay_fingerprint(lines)
        summary.update({
            "full_sha256": fingerprint["full_sha256"],
            "state_sha256": fingerprint["state_sha256"],
            "radio_sha256": fingerprint["radio_sha256"],
        })
    return summary


def _reward_summary(frames: list[dict[str, Any]]) -> dict[str, float]:
    evader_rewards: list[float] = []
    pursuer_rewards: list[float] = []
    decoder_accuracy: list[float] = []
    spoof_susceptibility: list[float] = []
    evader_information_reward: list[float] = []
    pursuer_auth_penalty: list[float] = []
    reward_agent_coverage: list[float] = []
    expected_agents = {"EVADER", "P1", "P2", "P3", "P4", "P5"}
    for frame in frames:
        rewards = frame.get("rewards", {})
        if isinstance(rewards, dict):
            reward_keys = {str(key) for key in rewards.keys()}
            if reward_keys:
                reward_agent_coverage.append(len(expected_agents.intersection(reward_keys)) / len(expected_agents))
            for name, value in rewards.items():
                if _is_evader_reward_key(name):
                    evader_rewards.append(float(value))
                elif _is_pursuer_reward_key(name):
                    pursuer_rewards.append(float(value))

        components = frame.get("reward_components", {})
        if not isinstance(components, dict):
            continue
        evader_row = components.get("EVADER") or components.get("evader_0") or {}
        if isinstance(evader_row, dict):
            _append_float(decoder_accuracy, evader_row, "decoder_accuracy")
            _append_float(spoof_susceptibility, evader_row, "spoof_susceptibility")
            _append_float(evader_information_reward, evader_row, "information_reward")
        for name, row in components.items():
            if _is_pursuer_reward_key(name) and isinstance(row, dict):
                _append_float(pursuer_auth_penalty, row, "auth_penalty")

    return {
        "total_evader_reward": float(sum(evader_rewards)),
        "total_pursuer_reward": float(sum(pursuer_rewards)),
        "avg_evader_reward": _mean(evader_rewards),
        "avg_pursuer_reward": _mean(pursuer_rewards),
        "avg_decoder_accuracy": _mean(decoder_accuracy),
        "avg_spoof_susceptibility": _mean(spoof_susceptibility),
        "avg_evader_information_reward": _mean(evader_information_reward),
        "avg_pursuer_auth_penalty": _mean(pursuer_auth_penalty),
        "max_pursuer_auth_penalty": max(pursuer_auth_penalty, default=0.0),
        "max_spoof_susceptibility": max(spoof_susceptibility, default=0.0),
        "reward_agent_coverage": _mean(reward_agent_coverage),
    }


def _append_float(out: list[float], row: dict[str, Any], key: str) -> None:
    if key not in row:
        return
    value = float(row[key])
    if np.isfinite(value):
        out.append(value)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))


def _is_evader_reward_key(name: Any) -> bool:
    return str(name) in {"EVADER", "evader_0"}


def _is_pursuer_reward_key(name: Any) -> bool:
    text = str(name)
    return text.startswith("P") or text.startswith("pursuer_")


def _comms_summary(
    frames: list[dict[str, Any]],
    unique_events: list[dict[str, Any]],
    confidence_values: list[float],
) -> dict[str, float | int]:
    all_words = [str(event.get("word", "")) for event in unique_events if event.get("word")]
    pursuer_words = [
        str(event.get("word", ""))
        for event in unique_events
        if str(event.get("speaker", "")).startswith("P") and not event.get("spoofed")
    ]
    spoof_words = [
        str(event.get("word", ""))
        for event in unique_events
        if event.get("spoofed")
    ]
    speakers = [str(event.get("speaker", "")) for event in unique_events if event.get("speaker")]
    cipher_times = sorted({
        round(float(event.get("time", 0.0)), 6)
        for event in unique_events
        if event.get("speaker") == "dispatch" and event.get("word") == "cipher"
    })
    jam_times = sorted({
        round(float(event.get("time", 0.0)), 6)
        for event in unique_events
        if event.get("spoofed")
    })
    jam_drops = [_confidence_drop_after(frames, time_value, window=2.0) for time_value in jam_times]
    cipher_drops = [_confidence_drop_after(frames, time_value, window=2.5) for time_value in cipher_times]
    cipher_recoveries = [_confidence_recovery_after(frames, time_value, window=6.0) for time_value in cipher_times]
    return {
        "radio_word_entropy": _entropy_bits(all_words),
        "radio_word_entropy_norm": _normalized_entropy(all_words),
        "pursuer_word_entropy": _entropy_bits(pursuer_words),
        "pursuer_word_entropy_norm": _normalized_entropy(pursuer_words),
        "spoof_word_entropy": _entropy_bits(spoof_words),
        "spoof_word_entropy_norm": _normalized_entropy(spoof_words),
        "speaker_entropy": _entropy_bits(speakers),
        "speaker_entropy_norm": _normalized_entropy(speakers),
        "unique_words": len(set(all_words)),
        "unique_pursuer_words": len(set(pursuer_words)),
        "unique_spoof_words": len(set(spoof_words)),
        "unique_speakers": len(set(speakers)),
        "radio_spoof_ratio": len(spoof_words) / max(1, len(all_words)),
        "jam_events": len(jam_times),
        "cipher_rotations": len(cipher_times),
        "avg_jam_confidence_drop": _mean(jam_drops),
        "max_jam_confidence_drop": max(jam_drops, default=0.0),
        "avg_cipher_confidence_drop": _mean(cipher_drops),
        "avg_cipher_confidence_recovery": _mean(cipher_recoveries),
        "confidence_min": min(confidence_values, default=0.0),
        "confidence_max": max(confidence_values, default=0.0),
        "confidence_range": max(confidence_values, default=0.0) - min(confidence_values, default=0.0),
    }


def _unique_radio_events(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[float, str, str, bool, str]] = set()
    events: list[dict[str, Any]] = []
    for frame in frames:
        for event in frame.get("radio", []):
            key = _radio_event_key(event)
            if key in seen:
                continue
            seen.add(key)
            events.append(event)
    return events


def _entropy_bits(values: list[str]) -> float:
    total = len(values)
    if total <= 0:
        return 0.0
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * float(np.log2(p))
    return float(entropy)


def _normalized_entropy(values: list[str]) -> float:
    unique = len(set(values))
    if unique <= 1:
        return 0.0
    return float(_entropy_bits(values) / np.log2(unique))


def _confidence_drop_after(frames: list[dict[str, Any]], time_value: float, *, window: float) -> float:
    before = _confidence_at_or_before(frames, time_value)
    if before is None:
        return 0.0
    values = [
        float(frame.get("confidence", 0.0))
        for frame in frames
        if time_value <= float(frame.get("time", 0.0)) <= time_value + window
    ]
    if not values:
        return 0.0
    return float(max(0.0, before - min(values)))


def _confidence_recovery_after(frames: list[dict[str, Any]], time_value: float, *, window: float) -> float:
    values = [
        float(frame.get("confidence", 0.0))
        for frame in frames
        if time_value <= float(frame.get("time", 0.0)) <= time_value + window
    ]
    if len(values) < 2:
        return 0.0
    return float(max(0.0, max(values) - min(values)))


def _confidence_at_or_before(frames: list[dict[str, Any]], time_value: float) -> float | None:
    previous = None
    for frame in frames:
        if float(frame.get("time", 0.0)) <= time_value:
            previous = float(frame.get("confidence", 0.0))
        else:
            break
    return previous


def _report_score(summary: dict[str, Any], diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    penalties = {"info": 0.0, "low": 0.04, "medium": 0.12, "high": 0.24}
    penalty = sum(penalties.get(str(row.get("severity", "info")), 0.0) for row in diagnostics)
    contract = min(
        float(summary.get("action_frames", 0)) / max(1.0, float(summary.get("frames", 0))),
        float(summary.get("camera_hint_frames", 0)) / max(1.0, float(summary.get("frames", 0))),
        float(summary.get("reward_frames", 0)) / max(1.0, float(summary.get("frames", 0))),
        float(summary.get("reward_agent_coverage", 0.0)),
    )
    evidence = 0.25 * min(1.0, float(summary.get("radio_word_entropy", 0.0)) / 3.5)
    evidence += 0.20 * min(1.0, float(summary.get("unique_radio_events", 0)) / 40.0)
    evidence += 0.20 * min(1.0, float(summary.get("avg_evader_speed", 0.0)) / 20.0)
    evidence += 0.20 * contract
    evidence += 0.15 * min(1.0, float(summary.get("confidence_range", 0.0)) / 0.35)
    overall = float(np.clip(evidence - penalty, 0.0, 1.0))
    diagnostic_count = len([row for row in diagnostics if row.get("severity") != "info"])
    has_high_diagnostic = any(row.get("severity") == "high" for row in diagnostics)
    return {
        "overall": overall,
        "evidence": float(evidence),
        "penalty": float(penalty),
        "contract": float(contract),
        "diagnostic_count": diagnostic_count,
        "status": "nominal" if diagnostic_count == 0 and not has_high_diagnostic else "review",
    }


def _report_diagnostics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    frames = max(1, int(summary.get("frames", 0)))
    if int(summary.get("action_frames", 0)) != frames:
        diagnostics.append(_report_diagnostic(
            "replay_contract",
            "high",
            "missing_action_frames",
            "Replay does not contain action evidence for every frame.",
            {"action_frames": summary.get("action_frames"), "frames": frames},
        ))
    if int(summary.get("camera_hint_frames", 0)) != frames:
        diagnostics.append(_report_diagnostic(
            "replay_contract",
            "medium",
            "missing_camera_hints",
            "Replay does not contain camera hints for every frame.",
            {"camera_hint_frames": summary.get("camera_hint_frames"), "frames": frames},
        ))
    if int(summary.get("reward_frames", 0)) != frames or float(summary.get("reward_agent_coverage", 0.0)) < 0.99:
        diagnostics.append(_report_diagnostic(
            "reward_trace",
            "high",
            "incomplete_reward_trace",
            "Reward evidence is incomplete or does not cover all six agents.",
            {
                "reward_frames": summary.get("reward_frames"),
                "frames": frames,
                "reward_agent_coverage": summary.get("reward_agent_coverage"),
            },
        ))
    if int(summary.get("unique_radio_events", 0)) <= 0:
        diagnostics.append(_report_diagnostic(
            "information_warfare",
            "high",
            "radio_silent",
            "Replay has no unique radio events.",
            {"unique_radio_events": summary.get("unique_radio_events")},
        ))
    elif float(summary.get("radio_word_entropy", 0.0)) < 1.0:
        diagnostics.append(_report_diagnostic(
            "information_warfare",
            "medium",
            "low_token_entropy",
            "Radio vocabulary diversity is low for a cryptographic-comms episode.",
            {"radio_word_entropy": summary.get("radio_word_entropy")},
        ))
    if int(summary.get("spoofed_radio_events", 0)) > 0 and int(summary.get("jam_events", 0)) <= 0:
        diagnostics.append(_report_diagnostic(
            "information_warfare",
            "medium",
            "spoof_events_not_grouped",
            "Spoofed words exist but no jam burst could be reconstructed from unique radio events.",
            {"spoofed_radio_events": summary.get("spoofed_radio_events"), "jam_events": summary.get("jam_events")},
        ))
    if int(summary.get("cipher_rotations", 0)) > 0 and float(summary.get("avg_cipher_confidence_recovery", 0.0)) <= 0.0:
        diagnostics.append(_report_diagnostic(
            "scanner",
            "low",
            "cipher_without_recovery",
            "Cipher rotation was observed without measurable confidence recovery in the replay window.",
            {
                "cipher_rotations": summary.get("cipher_rotations"),
                "avg_cipher_confidence_recovery": summary.get("avg_cipher_confidence_recovery"),
            },
        ))
    if float(summary.get("max_impact", 0.0)) > 55.0:
        diagnostics.append(_report_diagnostic(
            "physics",
            "medium",
            "extreme_impact",
            "Replay contains unusually high impact energy.",
            {"max_impact": summary.get("max_impact")},
        ))
    if not diagnostics:
        diagnostics.append(_report_diagnostic(
            "episode",
            "info",
            "nominal_replay_evidence",
            "Replay evidence is internally complete and reportable.",
            {},
        ))
    return diagnostics


def _report_diagnostic(
    area: str,
    severity: str,
    reason: str,
    message: str,
    evidence: dict[str, Any],
) -> dict[str, Any]:
    return {
        "area": area,
        "severity": severity,
        "reason": reason,
        "message": message,
        "evidence": _json_safe_value(evidence),
    }


def _notable_events(frames: list[dict[str, Any]], *, limit: int = 18) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen_radio: set[tuple[float, str, str, bool, str]] = set()
    for frame in frames:
        time_value = float(frame.get("time", 0.0))
        metrics = frame.get("metrics", {}) if isinstance(frame.get("metrics"), dict) else {}
        if metrics.get("waypoint_event"):
            events.append({"time": time_value, "kind": "waypoint", "label": "waypoint hit"})
        if metrics.get("capture_event"):
            events.append({"time": time_value, "kind": "capture", "label": "capture"})
        for collision in frame.get("collisions", []):
            impact = float(collision.get("impact", 0.0))
            if impact >= 12.0:
                events.append({
                    "time": time_value,
                    "kind": "impact",
                    "label": "major impact",
                    "impact": impact,
                })
        for event in frame.get("radio", []):
            key = _radio_event_key(event)
            if key in seen_radio:
                continue
            seen_radio.add(key)
            if event.get("spoofed"):
                events.append({
                    "time": float(event.get("time", time_value)),
                    "kind": "jam",
                    "label": "spoofed radio burst",
                    "speaker": str(event.get("speaker", "")),
                    "word": str(event.get("word", "")),
                })
            elif event.get("speaker") == "dispatch" and event.get("word") == "cipher":
                events.append({
                    "time": float(event.get("time", time_value)),
                    "kind": "cipher",
                    "label": "cipher rotation",
                })
    events.sort(key=lambda row: (float(row.get("time", 0.0)), str(row.get("kind", ""))))
    compact: list[dict[str, Any]] = []
    last_key: tuple[str, float] | None = None
    for event in events:
        key = (str(event.get("kind", "")), round(float(event.get("time", 0.0)), 2))
        if key == last_key:
            continue
        compact.append(event)
        last_key = key
        if len(compact) >= limit:
            break
    return compact


def replay_fingerprint(lines: list[dict[str, Any]], *, precision: int = FINGERPRINT_PRECISION) -> dict[str, Any]:
    """Return stable replay fingerprints independent of JSON whitespace."""
    validate_replay(lines)
    full_hasher = hashlib.sha256()
    state_hasher = hashlib.sha256()
    radio_hasher = hashlib.sha256()
    for payload in lines:
        full_hasher.update(_canonical_bytes(payload, precision=precision))
        full_hasher.update(b"\n")
    for frame in lines[1:]:
        state_hasher.update(_canonical_bytes(_state_fingerprint_payload(frame), precision=precision))
        state_hasher.update(b"\n")
        radio_hasher.update(_canonical_bytes(frame.get("radio", []), precision=precision))
        radio_hasher.update(b"\n")
    frames = lines[1:]
    return {
        "schema_version": int(lines[0].get("version", REPLAY_VERSION)),
        "precision": int(precision),
        "frames": len(frames),
        "start_time": float(frames[0]["time"]),
        "end_time": float(frames[-1]["time"]),
        "full_sha256": full_hasher.hexdigest(),
        "state_sha256": state_hasher.hexdigest(),
        "radio_sha256": radio_hasher.hexdigest(),
    }


def replay_fidelity_manifest(path: str | Path, lines: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    replay_path = Path(path)
    replay_lines = lines if lines is not None else load_replay(replay_path)
    validate_replay(replay_lines)
    summary = summarize_replay(replay_lines, include_fingerprint=False)
    fingerprint = replay_fingerprint(replay_lines)
    return {
        "version": 1,
        "replay": str(replay_path),
        "valid": True,
        "meta": replay_lines[0],
        "summary": summary,
        "fingerprint": fingerprint,
    }


def replay_diagnostic_report(
    path: str | Path | None,
    lines: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a compact research report from replay-only evidence."""
    replay_lines = lines if lines is not None else load_replay(path)  # type: ignore[arg-type]
    validate_replay(replay_lines)
    summary = summarize_replay(replay_lines, include_fingerprint=True)
    frames = replay_lines[1:]
    diagnostics = _report_diagnostics(summary)
    return {
        "version": 1,
        "kind": "replay_diagnostic_report",
        "replay": str(path) if path is not None else None,
        "meta": replay_lines[0],
        "summary": summary,
        "score": _report_score(summary, diagnostics),
        "sections": {
            "chase": {
                "frames": int(summary["frames"]),
                "duration": float(summary["duration"]),
                "captures": int(summary["captures"]),
                "waypoints_hit": int(summary["waypoints_hit"]),
                "avg_evader_speed": float(summary["avg_evader_speed"]),
                "max_impact": float(summary["max_impact"]),
                "collision_events": int(summary["collision_events"]),
            },
            "information_warfare": {
                "radio_events": int(summary["radio_events"]),
                "unique_radio_events": int(summary["unique_radio_events"]),
                "spoofed_radio_events": int(summary["spoofed_radio_events"]),
                "unique_spoofed_radio_events": int(summary["unique_spoofed_radio_events"]),
                "radio_word_entropy": float(summary["radio_word_entropy"]),
                "pursuer_word_entropy": float(summary["pursuer_word_entropy"]),
                "radio_spoof_ratio": float(summary["radio_spoof_ratio"]),
                "jam_events": int(summary["jam_events"]),
                "cipher_rotations": int(summary["cipher_rotations"]),
                "avg_jam_confidence_drop": float(summary["avg_jam_confidence_drop"]),
                "avg_cipher_confidence_drop": float(summary["avg_cipher_confidence_drop"]),
                "avg_cipher_confidence_recovery": float(summary["avg_cipher_confidence_recovery"]),
                "avg_confidence": float(summary["avg_confidence"]),
                "confidence_min": float(summary["confidence_min"]),
                "confidence_range": float(summary["confidence_range"]),
            },
            "reward_trace": {
                "reward_frames": int(summary["reward_frames"]),
                "reward_agent_coverage": float(summary["reward_agent_coverage"]),
                "total_evader_reward": float(summary["total_evader_reward"]),
                "total_pursuer_reward": float(summary["total_pursuer_reward"]),
                "avg_evader_reward": float(summary["avg_evader_reward"]),
                "avg_pursuer_reward": float(summary["avg_pursuer_reward"]),
                "avg_decoder_accuracy": float(summary["avg_decoder_accuracy"]),
                "avg_spoof_susceptibility": float(summary["avg_spoof_susceptibility"]),
                "avg_evader_information_reward": float(summary["avg_evader_information_reward"]),
                "avg_pursuer_auth_penalty": float(summary["avg_pursuer_auth_penalty"]),
            },
            "replay_contract": {
                "schema_version": int(summary["schema_version"]),
                "action_frames": int(summary["action_frames"]),
                "camera_hint_frames": int(summary["camera_hint_frames"]),
                "reward_frames": int(summary["reward_frames"]),
                "state_sha256": str(summary["state_sha256"]),
                "radio_sha256": str(summary["radio_sha256"]),
                "full_sha256": str(summary["full_sha256"]),
            },
        },
        "notable_events": _notable_events(frames),
        "diagnostics": diagnostics,
    }


def compare_replay_fingerprints(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    *,
    precision: int = FINGERPRINT_PRECISION,
) -> dict[str, Any]:
    left_fp = replay_fingerprint(left, precision=precision)
    right_fp = replay_fingerprint(right, precision=precision)
    return {
        "matched": left_fp["state_sha256"] == right_fp["state_sha256"]
        and left_fp["radio_sha256"] == right_fp["radio_sha256"],
        "left": left_fp,
        "right": right_fp,
        "state_matched": left_fp["state_sha256"] == right_fp["state_sha256"],
        "radio_matched": left_fp["radio_sha256"] == right_fp["radio_sha256"],
        "full_matched": left_fp["full_sha256"] == right_fp["full_sha256"],
    }


def _action_payload(actions: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(actions, dict):
        return {}
    payload: dict[str, Any] = {}
    for name, action in sorted(actions.items()):
        if not isinstance(action, dict):
            continue
        row: dict[str, Any] = {
            "source": str(action.get("source", "")),
            "control": _float_vector(action.get("control", [0.0, 0.0, 0.0, 0.0]), limit=4, pad_to=4),
        }
        if "requested_control" in action:
            row["requested_control"] = _float_vector(action["requested_control"], limit=4)
        if "tokens" in action:
            row["tokens"] = _int_vector(action["tokens"], limit=12)
        if "spoof_tokens" in action:
            row["spoof_tokens"] = _int_vector(action["spoof_tokens"], limit=12)
        if "target_mask" in action:
            row["target_mask"] = [1 if value else 0 for value in _int_vector(action["target_mask"], limit=5)]
        if "jam" in action:
            row["jam"] = bool(action["jam"])
        payload[str(name)] = row
    return payload


def _collision_payload(collisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not isinstance(collisions, list):
        return []
    payload = []
    for collision in collisions:
        if not isinstance(collision, dict):
            continue
        payload.append({
            "time": _finite_float(collision.get("time", 0.0)),
            "kind": str(collision.get("kind", "impact")),
            "impact": _finite_float(collision.get("impact", 0.0)),
            "last_collision_impact": _finite_float(collision.get("last_collision_impact", 0.0)),
            "camera_shake": _finite_float(collision.get("camera_shake", 0.0)),
        })
    return payload


def _numeric_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(mapping, dict):
        return {}
    return {str(key): _json_safe_value(value) for key, value in sorted(mapping.items())}


def _camera_payload(snapshot: dict[str, Any], agents: list[dict[str, Any]], metrics: dict[str, Any]) -> dict[str, Any]:
    evader = next((agent for agent in agents if agent.get("faction") == "evader"), agents[0] if agents else {})
    x = _finite_float(evader.get("x", 0.0))
    y = _finite_float(evader.get("y", 0.0))
    vx = _finite_float(evader.get("vx", 0.0))
    vy = _finite_float(evader.get("vy", 0.0))
    speed = max(0.0, _finite_float(evader.get("speed", 0.0)))
    waypoint = _float_vector(snapshot.get("waypoint", [x, y]), limit=2, pad_to=2)
    impact = max(
        _finite_float(metrics.get("impact", 0.0)),
        _finite_float(metrics.get("last_collision_impact", 0.0)),
    )
    shake = min(8.0, impact * 0.32)
    if metrics.get("capture_event"):
        event = "capture"
    elif metrics.get("waypoint_event"):
        event = "waypoint"
    elif shake > 0.0:
        event = "impact"
    else:
        event = "chase"
    return {
        "focus": [x, y],
        "lead": [x + vx * 0.35, y + vy * 0.35],
        "target": waypoint,
        "subject": str(evader.get("name", "EVADER")),
        "event": event,
        "shake": float(shake),
        "zoom": float(max(1.35, min(3.2, 2.65 - speed * 0.025))),
    }


def _state_fingerprint_payload(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        "i": frame["i"],
        "time": frame["time"],
        "agents": [
            {
                "name": agent.get("name"),
                "faction": agent.get("faction"),
                "role": agent.get("role", ""),
                "target": agent.get("target", []),
                "x": agent.get("x"),
                "y": agent.get("y"),
                "yaw": agent.get("yaw"),
                "vx": agent.get("vx"),
                "vy": agent.get("vy"),
                "speed": agent.get("speed"),
                "yaw_rate": agent.get("yaw_rate"),
                "slip_angle": agent.get("slip_angle"),
                "wheel_grip": agent.get("wheel_grip", []),
                "rpm": agent.get("rpm"),
                "gear": agent.get("gear"),
            }
            for agent in frame.get("agents", [])
        ],
        "waypoint": frame.get("waypoint", []),
        "confidence": frame.get("confidence"),
        "predictions": frame.get("predictions", {}),
        "metrics": frame.get("metrics", {}),
        "actions": frame.get("actions", {}),
        "collisions": frame.get("collisions", []),
        "rewards": frame.get("rewards", {}),
        "reward_components": frame.get("reward_components", {}),
        "camera": frame.get("camera", {}),
    }


def _canonical_bytes(payload: Any, *, precision: int) -> bytes:
    return json.dumps(
        _canonical_payload(payload, precision=precision),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _canonical_payload(payload: Any, *, precision: int) -> Any:
    if isinstance(payload, dict):
        return {str(key): _canonical_payload(value, precision=precision) for key, value in sorted(payload.items())}
    if isinstance(payload, (list, tuple)):
        return [_canonical_payload(value, precision=precision) for value in payload]
    if isinstance(payload, bool) or payload is None or isinstance(payload, str):
        return payload
    if isinstance(payload, (int, np.integer)):
        return int(payload)
    if isinstance(payload, (float, np.floating)):
        value = float(payload)
        if not np.isfinite(value):
            raise ValueError("cannot fingerprint non-finite replay value")
        return round(value, precision)
    return payload


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe_value(inner) for key, inner in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe_value(inner) for inner in value]
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return _finite_float(value)
    return str(value)


def _float_vector(values: Any, *, limit: int, pad_to: int | None = None) -> list[float]:
    arr = np.asarray(values, dtype=float).reshape(-1)
    out = [_finite_float(value) for value in arr[:limit]]
    if pad_to is not None:
        while len(out) < pad_to:
            out.append(0.0)
    return out


def _int_vector(values: Any, *, limit: int) -> list[int]:
    arr = np.asarray(values).reshape(-1)
    return [int(value) for value in arr[:limit]]


def _finite_float(value: Any) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError("replay payload contains a non-finite value")
    return out


def _validate_finite_payload(payload: Any, *, label: str) -> None:
    if isinstance(payload, dict):
        for value in payload.values():
            _validate_finite_payload(value, label=label)
        return
    if isinstance(payload, (list, tuple)):
        for value in payload:
            _validate_finite_payload(value, label=label)
        return
    if isinstance(payload, bool) or payload is None or isinstance(payload, str):
        return
    if isinstance(payload, (int, np.integer, float, np.floating)):
        value = float(payload)
        if not np.isfinite(value):
            raise ValueError(f"{label}: non-finite value")
        return
    raise ValueError(f"{label}: unsupported value {type(payload).__name__}")


def _radio_event_key(event: dict[str, Any]) -> tuple[float, str, str, bool, str]:
    return (
        round(float(event.get("time", 0.0)), 6),
        str(event.get("speaker", "")),
        str(event.get("word", "")),
        bool(event.get("spoofed", False)),
        str(event.get("meaning", "")),
    )
