from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .action_space import ActionSpace
from .memory import progress_scalar
from .protocol import BridgeObservation


class PlayerPolicy(Protocol):
    def select_macro(self, observation: BridgeObservation) -> str:
        ...


@dataclass
class NoopPolicy:
    def select_macro(self, observation: BridgeObservation) -> str:
        return "idle"


@dataclass
class HeuristicPolicy:
    """A tiny non-AI policy for smoke tests and first bridge runs."""

    action_space: ActionSpace

    def select_macro(self, observation: BridgeObservation) -> str:
        player_state = observation.ram.get("player_state")
        y_speed = int(observation.ram.get("mario_y_speed", 0) or 0)
        x_speed = int(observation.ram.get("mario_x_speed", 0) or 0)
        if player_state in {0x0F, 0x10}:
            return "swim_right"
        if y_speed == 0 and x_speed < 4 and observation.frame % 90 == 0:
            return "run_jump_right"
        if observation.frame % 240 == 0 and progress_scalar(observation) < 128:
            return "jump_right"
        return "run_right"


class PPOPolicy:
    def __init__(self, model_path: str | Path):
        try:
            from stable_baselines3 import PPO  # type: ignore[import-not-found]
        except Exception as error:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "stable-baselines3 is required for PPOPolicy. Install with: pip install -e '.[rl]'"
            ) from error
        self.model = PPO.load(str(model_path))

    def select_macro(self, observation: BridgeObservation) -> str:  # pragma: no cover - needs RL env
        action, _state = self.model.predict(observation.raw, deterministic=True)
        if isinstance(action, (list, tuple)):
            return str(action[0])
        return str(action)


def make_policy(name: str, action_space: ActionSpace, *, model_path: str | Path | None = None) -> PlayerPolicy:
    if name == "noop":
        return NoopPolicy()
    if name == "heuristic":
        return HeuristicPolicy(action_space)
    if name == "ppo":
        if model_path is None:
            raise ValueError("--model-path is required for PPO policy")
        return PPOPolicy(model_path)
    raise ValueError(f"unknown policy: {name}")

