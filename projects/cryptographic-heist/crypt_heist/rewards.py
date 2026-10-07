"""Reward components for the heist MARL environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .observations import AGENT_NAMES


@dataclass
class RewardWeights:
    """Tunable weights for the information-warfare terms."""

    decoder_accuracy_alpha: float = 0.08
    spoof_susceptibility_gamma: float = 0.50
    evader_spoof_reward: float = 0.35
    decoder_error_scale: float = 120.0


@dataclass
class RewardSnapshot:
    waypoint_dist: float
    nearest_pursuer: float
    close_pursuers: int
    capture_timer: float
    waypoints_hit: int
    captures: int
    deception_score: float
    impact: float
    evader_speed: float
    confidence: float
    pursuer_xy: np.ndarray
    decoder_prediction_xy: np.ndarray | None
    live_decoder_error: float
    spoof_susceptibility: float
    active_spoofed_pursuers: int

    @classmethod
    def from_sim(cls, sim) -> "RewardSnapshot":
        ev = sim.evader.vehicle
        wx, wy = sim.evader_planner.waypoint
        pursuer_xy = np.asarray([[p.vehicle.x, p.vehicle.y] for p in sim.pursuers], dtype=np.float32)
        dists = [float(np.hypot(float(x) - ev.x, float(y) - ev.y)) for x, y in pursuer_xy]
        pred_xy = _decoder_prediction_array(sim)
        live_error = float(np.linalg.norm(pred_xy - pursuer_xy, axis=1).mean()) if pred_xy is not None else 0.0
        active_offsets = [
            float(np.linalg.norm(p.spoof_offset))
            for p in sim.pursuers
            if p.spoof_timer > 0.0
        ]
        spoof_susceptibility = float(np.clip(np.mean(active_offsets) / 85.0, 0.0, 1.0)) if active_offsets else 0.0
        return cls(
            waypoint_dist=float(np.hypot(ev.x - wx, ev.y - wy)),
            nearest_pursuer=min(dists) if dists else 999.0,
            close_pursuers=sum(d < 80.0 for d in dists),
            capture_timer=float(sim.capture_timer),
            waypoints_hit=int(sim.waypoints_hit),
            captures=int(sim.captures),
            deception_score=float(sim.deception_score),
            impact=float(sim.impact_energy),
            evader_speed=float(ev.speed),
            confidence=float(sim.channel.confidence),
            pursuer_xy=pursuer_xy,
            decoder_prediction_xy=pred_xy,
            live_decoder_error=live_error,
            spoof_susceptibility=spoof_susceptibility,
            active_spoofed_pursuers=len(active_offsets),
        )


class RewardSystem:
    """Dense but interpretable reward model for early curriculum stages."""

    def __init__(
        self,
        *,
        weights: RewardWeights | None = None,
        prediction_horizon_steps: int = 8,
    ):
        self.weights = weights or RewardWeights()
        self.prediction_horizon_steps = max(0, int(prediction_horizon_steps))
        self._step_index = 0
        self._pending_predictions: list[tuple[int, np.ndarray]] = []
        self._last_decoder_error = 0.0
        self._last_decoder_accuracy = 0.0
        self._last_decoder_samples = 0

    def reset(self) -> None:
        self._step_index = 0
        self._pending_predictions.clear()
        self._last_decoder_error = 0.0
        self._last_decoder_accuracy = 0.0
        self._last_decoder_samples = 0

    def compute(self, before: RewardSnapshot, after: RewardSnapshot, done: bool) -> tuple[dict[str, float], dict]:
        waypoint_progress = before.waypoint_dist - after.waypoint_dist
        pursuit_progress = before.nearest_pursuer - after.nearest_pursuer
        waypoint_delta = after.waypoints_hit - before.waypoints_hit
        capture_delta = after.captures - before.captures
        deception_delta = after.deception_score - before.deception_score
        impact_penalty = min(after.impact, 30.0) * 0.025
        decoder_metrics = self._decoder_metrics(after)
        decoder_accuracy = decoder_metrics["decoder_accuracy"]
        spoof_susceptibility = after.spoof_susceptibility
        decoder_accuracy_penalty = self.weights.decoder_accuracy_alpha * decoder_accuracy
        spoof_susceptibility_penalty = self.weights.spoof_susceptibility_gamma * spoof_susceptibility
        evader_information_reward = (
            deception_delta * 1.5
            + self.weights.evader_spoof_reward * spoof_susceptibility
        )
        pursuer_auth_penalty = decoder_accuracy_penalty + spoof_susceptibility_penalty

        evader = (
            0.01
            + waypoint_progress * 0.025
            + waypoint_delta * 9.0
            + evader_information_reward
            + min(after.evader_speed / 80.0, 1.0) * 0.02
            - after.capture_timer * 0.25
            - impact_penalty
            - capture_delta * 25.0
        )
        if done and capture_delta > 0:
            evader -= 10.0

        team_pursuer = (
            pursuit_progress * 0.035
            + after.close_pursuers * 0.035
            + after.capture_timer * 0.16
            + capture_delta * 22.0
            - waypoint_delta * 4.0
            - deception_delta * 0.8
            - pursuer_auth_penalty
            - impact_penalty * 0.45
        )

        rewards = {"evader_0": float(evader)}
        for i in range(5):
            rewards[f"pursuer_{i}"] = float(team_pursuer)

        components = {
            "waypoint_progress": waypoint_progress,
            "pursuit_progress": pursuit_progress,
            "waypoint_delta": waypoint_delta,
            "capture_delta": capture_delta,
            "deception_delta": deception_delta,
            "impact_penalty": impact_penalty,
            "confidence": after.confidence,
            "decoder_error": decoder_metrics["decoder_error"],
            "decoder_accuracy": decoder_accuracy,
            "decoder_accuracy_samples": decoder_metrics["decoder_accuracy_samples"],
            "live_decoder_error": after.live_decoder_error,
            "decoder_accuracy_penalty": decoder_accuracy_penalty,
            "spoof_susceptibility": spoof_susceptibility,
            "active_spoofed_pursuers": after.active_spoofed_pursuers,
            "spoof_susceptibility_penalty": spoof_susceptibility_penalty,
            "pursuer_auth_penalty": pursuer_auth_penalty,
            "evader_information_reward": evader_information_reward,
        }
        return rewards, components

    def _decoder_metrics(self, after: RewardSnapshot) -> dict[str, Any]:
        if after.decoder_prediction_xy is not None:
            due_step = self._step_index + self.prediction_horizon_steps
            self._pending_predictions.append((due_step, after.decoder_prediction_xy.copy()))

        realized_errors: list[float] = []
        still_pending: list[tuple[int, np.ndarray]] = []
        for due_step, pred_xy in self._pending_predictions:
            if due_step <= self._step_index:
                realized_errors.append(float(np.linalg.norm(pred_xy - after.pursuer_xy, axis=1).mean()))
            else:
                still_pending.append((due_step, pred_xy))
        self._pending_predictions = still_pending[-256:]

        if realized_errors:
            self._last_decoder_error = float(np.mean(realized_errors))
            self._last_decoder_accuracy = float(np.clip(
                np.exp(-self._last_decoder_error / max(1.0, self.weights.decoder_error_scale)),
                0.0,
                1.0,
            ))
            self._last_decoder_samples = len(realized_errors)

        metrics = {
            "decoder_error": self._last_decoder_error,
            "decoder_accuracy": self._last_decoder_accuracy,
            "decoder_accuracy_samples": self._last_decoder_samples,
        }
        self._step_index += 1
        return metrics


def zero_rewards() -> dict[str, float]:
    return {agent: 0.0 for agent in AGENT_NAMES}


def _decoder_prediction_array(sim) -> np.ndarray | None:
    predictions = getattr(sim, "decoder_predictions", {}) or {}
    if not predictions:
        return None
    rows = []
    for i in range(5):
        key = f"P{i + 1}"
        if key not in predictions:
            return None
        x, y = predictions[key]
        rows.append([float(x), float(y)])
    return np.asarray(rows, dtype=np.float32)
