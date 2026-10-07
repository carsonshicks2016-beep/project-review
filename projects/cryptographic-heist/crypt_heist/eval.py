"""Evaluation helpers for scripted and learned policies."""

from __future__ import annotations

from dataclasses import dataclass

from .sim import HeistSim


@dataclass
class EvalResult:
    seeds: int
    steps: int
    captures: int
    waypoints: int
    deception: float
    avg_confidence: float

    def to_text(self) -> str:
        return (
            f"seeds={self.seeds} steps={self.steps} captures={self.captures} "
            f"waypoints={self.waypoints} deception={self.deception:.1f} "
            f"avg_confidence={self.avg_confidence:.3f}"
        )


def evaluate_scripted(num_seeds: int = 5, steps: int = 1800) -> EvalResult:
    captures = 0
    waypoints = 0
    deception = 0.0
    confidence = 0.0
    samples = 0
    for seed in range(num_seeds):
        sim = HeistSim(seed=seed)
        for _ in range(steps):
            sim.step()
            confidence += sim.channel.confidence
            samples += 1
        captures += sim.captures
        waypoints += sim.waypoints_hit
        deception += sim.deception_score
    return EvalResult(
        seeds=num_seeds,
        steps=steps,
        captures=captures,
        waypoints=waypoints,
        deception=deception,
        avg_confidence=confidence / max(1, samples),
    )

