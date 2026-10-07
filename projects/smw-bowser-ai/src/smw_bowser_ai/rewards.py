from __future__ import annotations

from dataclasses import dataclass

from .memory import level_cleared, likely_death, progress_scalar
from .protocol import BridgeObservation
from .route import RouteManifest


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    progress: float = 0.0
    survival: float = 0.0
    powerup: float = 0.0
    level_clear: float = 0.0
    death: float = 0.0
    starworld: float = 0.0
    no_progress: float = 0.0
    route_event: float = 0.0


class RewardModel:
    def __init__(self, route: RouteManifest):
        self.route = route
        self.best_progress = 0
        self.no_progress_frames = 0

    def score(
        self,
        previous: BridgeObservation | None,
        current: BridgeObservation,
        *,
        route_event: bool = False,
    ) -> RewardBreakdown:
        progress_reward = 0.0
        no_progress_penalty = 0.0
        current_progress = progress_scalar(current)
        if previous is not None:
            delta = current_progress - progress_scalar(previous)
            progress_reward = max(-0.05, min(1.5, delta / 32.0))

        if current_progress > self.best_progress:
            self.best_progress = current_progress
            self.no_progress_frames = 0
        else:
            self.no_progress_frames += 1
            if self.no_progress_frames > 300:
                no_progress_penalty = -0.25

        survival = 0.01
        powerup = 0.03 if int(current.ram.get("powerup", 0) or 0) >= 1 else 0.0
        clear = 25.0 if level_cleared(current) else 0.0
        death = -100.0 if likely_death(previous, current) else 0.0
        starworld = -250.0 if self.route.detect_starworld(current) else 0.0
        route = 10.0 if route_event else 0.0
        total = progress_reward + survival + powerup + clear + death + starworld + no_progress_penalty + route
        return RewardBreakdown(
            total=total,
            progress=progress_reward,
            survival=survival,
            powerup=powerup,
            level_clear=clear,
            death=death,
            starworld=starworld,
            no_progress=no_progress_penalty,
            route_event=route,
        )

