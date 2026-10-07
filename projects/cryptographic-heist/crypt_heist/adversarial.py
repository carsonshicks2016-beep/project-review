"""Combined radio, scanner, and jammer evaluation loop."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .jamming import EvaderJammerController
from .radio import PursuerRadioController
from .replay import ReplayRecorder, load_replay, summarize_replay, validate_replay
from .scanner import ScannerDecoderRuntime, pursuer_xy
from .sim import HeistSim


@dataclass
class AdversarialEvalResult:
    """Serializable summary of a combined information-warfare episode."""

    radio_checkpoint: str
    scanner_checkpoint: str
    jammer_checkpoint: str
    replay: str | None
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "radio_checkpoint": self.radio_checkpoint,
            "scanner_checkpoint": self.scanner_checkpoint,
            "jammer_checkpoint": self.jammer_checkpoint,
            "replay": self.replay,
            "metrics": self.metrics,
        }


@dataclass
class AuthenticationEvalResult:
    """Paired jammed-vs-baseline result for authentication curriculum gates."""

    jammed: AdversarialEvalResult
    baseline: AdversarialEvalResult
    metrics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "jammed": self.jammed.to_dict(),
            "baseline": self.baseline.to_dict(),
            "metrics": self.metrics,
        }


class AdversarialStackController:
    """Combines learned radio tokens and learned jamming into one action dict."""

    def __init__(self, radio_checkpoint: str | Path, jammer_checkpoint: str | Path, enable_jamming: bool = True):
        self.radio = PursuerRadioController(radio_checkpoint)
        self.jammer = EvaderJammerController(jammer_checkpoint) if enable_jamming else None

    def action(self, sim: HeistSim) -> dict[str, dict[str, Any]]:
        jam_action = self.jammer.action(sim) if self.jammer is not None else None
        return merge_action_dicts(self.radio.action(sim), jam_action)


def merge_action_dicts(*action_dicts: dict[str, dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for actions in action_dicts:
        if not actions:
            continue
        for agent, payload in actions.items():
            existing = merged.setdefault(agent, {})
            existing.update(dict(payload))
    return merged


def run_adversarial_eval(
    *,
    radio_checkpoint: str | Path,
    scanner_checkpoint: str | Path,
    jammer_checkpoint: str | Path,
    seed: int = 11,
    steps: int = 1200,
    record: str | Path | None = None,
    start_jam_ready: bool = True,
    enable_jamming: bool = True,
) -> AdversarialEvalResult:
    """Run learned radio, scanner, and jammer together in the high-fidelity sim."""
    sim = HeistSim(seed=seed, reset_on_capture=True)
    if enable_jamming and start_jam_ready:
        sim.jam_cooldown = 0.0
    if not enable_jamming:
        sim.jam_cooldown = 1.0e9
        sim.channel.jamming_budget = 0

    controller = AdversarialStackController(radio_checkpoint, jammer_checkpoint, enable_jamming=enable_jamming)
    scanner = ScannerDecoderRuntime(scanner_checkpoint)

    confidence_values: list[float] = []
    decoder_errors: list[float] = []
    spoof_offsets: list[float] = []
    active_spoof_counts: list[int] = []
    jam_confidence_drops: list[float] = []
    unique_events: set[tuple[float, str, str, bool, str]] = set()
    unique_spoofed: set[tuple[float, str, str, bool, str]] = set()
    cipher_events: set[tuple[float, str]] = set()
    jam_bursts = 0

    replay_cm = ReplayRecorder(record, seed=seed, dt=sim.sim.dt) if record else nullcontext(None)
    with replay_cm as recorder:
        for _ in range(int(steps)):
            if not enable_jamming:
                sim.jam_cooldown = 1.0e9
                sim.channel.jamming_budget = 0
            pre_confidence = float(sim.channel.confidence)
            pre_deception = float(sim.deception_score)
            sim.step(actions=controller.action(sim))
            post_step_confidence = float(sim.channel.confidence)
            predictions = scanner.apply(sim)
            post_scanner_confidence = float(sim.channel.confidence)

            confidence_values.append(post_scanner_confidence)
            decoder_errors.append(_prediction_error(predictions, pursuer_xy(sim)))
            offsets = [float(np.linalg.norm(p.spoof_offset)) for p in sim.pursuers if p.spoof_timer > 0.0]
            spoof_offsets.extend(offsets)
            active_spoof_counts.append(len(offsets))
            _collect_radio_events(sim, unique_events, unique_spoofed, cipher_events)

            if sim.deception_score > pre_deception:
                jam_bursts += int(sim.deception_score - pre_deception)
                jam_confidence_drops.append(max(0.0, pre_confidence - min(post_step_confidence, post_scanner_confidence)))

            if recorder is not None:
                recorder.record(sim.snapshot())

    replay_summary: dict[str, Any] = {}
    if record:
        lines = load_replay(record)
        validate_replay(lines)
        replay_summary = summarize_replay(lines)

    avg_decoder_error = _mean(decoder_errors)
    metrics = {
        "seed": int(seed),
        "steps": int(steps),
        "frames": int(steps),
        "jamming_enabled": bool(enable_jamming),
        "unique_radio_events": len(unique_events),
        "unique_spoofed_events": len(unique_spoofed),
        "cipher_rotations": len(cipher_events),
        "jam_bursts": jam_bursts,
        "final_jamming_budget": int(sim.channel.jamming_budget),
        "deception_score": float(sim.deception_score),
        "captures": int(sim.captures),
        "waypoints_hit": int(sim.waypoints_hit),
        "avg_confidence": _mean(confidence_values),
        "min_confidence": min(confidence_values) if confidence_values else 0.0,
        "final_confidence": float(sim.channel.confidence),
        "avg_decoder_error": avg_decoder_error,
        "decoder_accuracy_proxy": float(np.clip(np.exp(-avg_decoder_error / 120.0), 0.0, 1.0)),
        "avg_spoof_offset": _mean(spoof_offsets),
        "max_active_spoofed_pursuers": max(active_spoof_counts) if active_spoof_counts else 0,
        "spoof_susceptibility_proxy": float(np.clip(_mean(spoof_offsets) / 85.0, 0.0, 1.0)),
        "avg_jam_confidence_drop": _mean(jam_confidence_drops),
        "replay_summary": replay_summary,
    }
    return AdversarialEvalResult(
        radio_checkpoint=str(radio_checkpoint),
        scanner_checkpoint=str(scanner_checkpoint),
        jammer_checkpoint=str(jammer_checkpoint),
        replay=str(record) if record else None,
        metrics=metrics,
    )


def run_authentication_curriculum_eval(
    *,
    radio_checkpoint: str | Path,
    scanner_checkpoint: str | Path,
    jammer_checkpoint: str | Path,
    seed: int = 11,
    steps: int = 1200,
    record_prefix: str | Path | None = "replays/auth_curriculum",
) -> AuthenticationEvalResult:
    """Run paired jammed/no-jam episodes and compute spoof susceptibility."""
    jammed_record = None
    baseline_record = None
    if record_prefix is not None:
        prefix = Path(record_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        jammed_record = prefix.with_name(f"{prefix.name}_jammed.jsonl")
        baseline_record = prefix.with_name(f"{prefix.name}_baseline.jsonl")

    jammed = run_adversarial_eval(
        radio_checkpoint=radio_checkpoint,
        scanner_checkpoint=scanner_checkpoint,
        jammer_checkpoint=jammer_checkpoint,
        seed=seed,
        steps=steps,
        record=jammed_record,
        enable_jamming=True,
    )
    baseline = run_adversarial_eval(
        radio_checkpoint=radio_checkpoint,
        scanner_checkpoint=scanner_checkpoint,
        jammer_checkpoint=jammer_checkpoint,
        seed=seed,
        steps=steps,
        record=baseline_record,
        start_jam_ready=False,
        enable_jamming=False,
    )
    trajectory = _trajectory_deviation(jammed_record, baseline_record) if jammed_record and baseline_record else {}
    jammed_metrics = jammed.metrics
    baseline_metrics = baseline.metrics
    confidence_damage = max(0.0, baseline_metrics["avg_confidence"] - jammed_metrics["avg_confidence"])
    final_confidence_damage = max(0.0, baseline_metrics["final_confidence"] - jammed_metrics["final_confidence"])
    decoder_error_delta = max(0.0, jammed_metrics["avg_decoder_error"] - baseline_metrics["avg_decoder_error"])
    deception_lift = max(0.0, jammed_metrics["deception_score"] - baseline_metrics["deception_score"])
    mean_deviation = float(trajectory.get("mean_pursuer_deviation", 0.0))
    spoof_susceptibility = float(np.clip(
        0.50 * (mean_deviation / 60.0)
        + 0.25 * confidence_damage
        + 0.15 * min(1.0, decoder_error_delta / 120.0)
        + 0.10 * min(1.0, deception_lift),
        0.0,
        1.0,
    ))
    metrics = {
        "seed": int(seed),
        "steps": int(steps),
        "jammed_replay": str(jammed_record) if jammed_record else None,
        "baseline_replay": str(baseline_record) if baseline_record else None,
        "confidence_damage": confidence_damage,
        "final_confidence_damage": final_confidence_damage,
        "decoder_error_delta": decoder_error_delta,
        "deception_lift": deception_lift,
        "spoof_event_lift": int(jammed_metrics["unique_spoofed_events"] - baseline_metrics["unique_spoofed_events"]),
        "spoof_susceptibility_counterfactual": spoof_susceptibility,
        "pursuer_auth_penalty_proxy": float(np.clip(
            jammed_metrics["decoder_accuracy_proxy"] + spoof_susceptibility,
            0.0,
            2.0,
        )),
        "evader_information_reward_proxy": float(deception_lift + confidence_damage + min(1.0, decoder_error_delta / 120.0)),
        "trajectory": trajectory,
    }
    return AuthenticationEvalResult(jammed=jammed, baseline=baseline, metrics=metrics)


def configure_jam_ready(sim: HeistSim) -> None:
    """Make the first learned jammer decision immediately visible in live demos."""
    sim.jam_cooldown = 0.0


def _prediction_error(predictions: dict[str, tuple[float, float]], actual: np.ndarray) -> float:
    if not predictions:
        return 0.0
    pred = np.asarray([predictions.get(f"P{i + 1}", (0.0, 0.0)) for i in range(5)], dtype=np.float32)
    return float(np.linalg.norm(pred - actual, axis=1).mean())


def _collect_radio_events(
    sim: HeistSim,
    unique_events: set[tuple[float, str, str, bool, str]],
    unique_spoofed: set[tuple[float, str, str, bool, str]],
    cipher_events: set[tuple[float, str]],
) -> None:
    for event in sim.channel.events:
        key = (round(float(event.time), 6), event.speaker, event.word, bool(event.spoofed), event.meaning)
        unique_events.add(key)
        if event.spoofed:
            unique_spoofed.add(key)
        if event.speaker == "dispatch" and event.word == "cipher":
            cipher_events.add((round(float(event.time), 6), event.word))


def _trajectory_deviation(jammed_record: str | Path, baseline_record: str | Path) -> dict[str, float]:
    jammed_trace = _pursuer_trace(jammed_record)
    baseline_trace = _pursuer_trace(baseline_record)
    frames = min(len(jammed_trace), len(baseline_trace))
    if frames == 0:
        return {
            "paired_frames": 0,
            "mean_pursuer_deviation": 0.0,
            "max_pursuer_deviation": 0.0,
            "final_pursuer_deviation": 0.0,
        }
    delta = np.linalg.norm(jammed_trace[:frames] - baseline_trace[:frames], axis=2)
    return {
        "paired_frames": int(frames),
        "mean_pursuer_deviation": float(delta.mean()),
        "max_pursuer_deviation": float(delta.max()),
        "final_pursuer_deviation": float(delta[-1].mean()),
    }


def _pursuer_trace(path: str | Path) -> np.ndarray:
    lines = load_replay(path)
    validate_replay(lines)
    frames = []
    for frame in lines[1:]:
        pursuers = [
            [float(agent["x"]), float(agent["y"])]
            for agent in frame.get("agents", [])
            if agent.get("faction") == "pursuer"
        ]
        if len(pursuers) == 5:
            frames.append(pursuers)
    return np.asarray(frames, dtype=np.float32)


def _mean(values: list[float]) -> float:
    return float(sum(values) / max(1, len(values)))
