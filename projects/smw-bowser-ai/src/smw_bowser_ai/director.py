from __future__ import annotations

from dataclasses import dataclass

from .action_space import ActionSpace
from .memory import MemoryMap, likely_death
from .protocol import BridgeAction, BridgeObservation
from .route import RouteManifest


@dataclass(frozen=True)
class DirectorDecision:
    kind: str
    action: BridgeAction
    reason: str
    delegate_to_player: bool = False
    abort: bool = False


class DirectorFSM:
    """Owns every explicitly non-gameplay component and delegates playable frames."""

    def __init__(
        self,
        action_space: ActionSpace,
        memory_map: MemoryMap,
        route: RouteManifest,
        *,
        final_evaluation: bool = False,
    ):
        self.action_space = action_space
        self.memory_map = memory_map
        self.route = route
        self.final_evaluation = final_evaluation
        self.previous: BridgeObservation | None = None
        self.title_pulse_interval = 45
        self.overworld_pulse_interval = 30

    def decide(self, observation: BridgeObservation) -> DirectorDecision:
        mode = observation.mode
        if mode == "unknown":
            mode = self.memory_map.classify_mode(observation.ram)

        if self.route.detect_starworld(observation):
            return DirectorDecision(
                kind="abort",
                action=self.action_space.noop(note="starworld detected"),
                reason="forbidden Starworld state detected",
                abort=True,
            )

        if likely_death(self.previous, observation):
            reason = "death detected"
            action = self.action_space.noop(note=reason)
            self.previous = observation
            return DirectorDecision(kind="abort", action=action, reason=reason, abort=True)

        if mode == "level":
            self.previous = observation
            return DirectorDecision(
                kind="delegate",
                action=self.action_space.noop(note="delegate playable level"),
                reason="playable level mode",
                delegate_to_player=True,
            )

        if mode == "overworld":
            macro = "confirm" if observation.frame % self.overworld_pulse_interval == 0 else "idle"
            action = self.action_space.bridge_action(macro, note="director overworld navigation")
            self.previous = observation
            return DirectorDecision(kind="director", action=action, reason="overworld navigation")

        if mode == "title_or_menu":
            macro = "start" if observation.frame % self.title_pulse_interval == 0 else "idle"
            action = self.action_space.bridge_action(macro, note="director title/menu navigation")
            self.previous = observation
            return DirectorDecision(kind="director", action=action, reason="title/menu navigation")

        if mode in {"transition", "dialogue", "cutscene", "credits"}:
            macro = "confirm" if observation.frame % 20 == 0 else "idle"
            action = self.action_space.bridge_action(macro, note=f"director {mode}")
            self.previous = observation
            return DirectorDecision(kind="director", action=action, reason=f"{mode} handling")

        action = self.action_space.noop(note="unknown mode fallback")
        self.previous = observation
        return DirectorDecision(kind="director", action=action, reason=f"unknown mode {mode}")

