"""NicheTask: a declarative selective-regime config (ROADMAP Stage 2.5).

A niche task is fully specified by its (JSON-serializable) fields: terrain,
forward direction, time limit, reward weights (in subclasses), and escalation
parameters. Subclasses implement `reward` and `is_fallen`. The escalation hook
(`difficulty` + `escalate()`) is groundwork for the POET environment co-evolution
in Stage 8; `configure_model` is where terrain (slopes/obstacles) will be applied
in Stage 7. For now both are inert for the default flat terrain.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

_REGISTRY: dict[str, type] = {}


def register_task(cls):
    _REGISTRY[cls.__name__] = cls
    return cls


def task_from_dict(d: dict):
    """Reconstruct the right NicheTask subclass from a config dict."""
    return _REGISTRY[d["task"]].from_dict(d)


@dataclass
class NicheTask:
    name: str = "niche"
    terrain: str = "flat"          # flat | slope | obstacles (Stage 7)
    forward_axis: int = 0          # 0 = x, 1 = y
    max_steps: int = 1000
    difficulty: float = 0.0        # escalation level (0 = easiest)
    escalation_rate: float = 0.0   # how much escalate() bumps difficulty

    # --- interface (override in subclasses) --------------------------------
    def reward(self, **kw):
        raise NotImplementedError

    def is_fallen(self, **kw):
        raise NotImplementedError

    def truncated(self, elapsed: int) -> bool:
        return elapsed >= self.max_steps

    def configure_model(self, model):
        """Apply terrain / task setup to a compiled model (Stage 7).

        Called once when the env is built. Tasks that shape the world (slope via
        equivalent gravity, payload, gravity scaling) mutate `model` here as a
        function of `self.difficulty`. Default: no-op."""
        return model

    def reset_episode(self, rng):
        """Reset per-episode task state (e.g. an energy reservoir). Default: no-op."""
        return self

    def perturb(self, model, data, rng, elapsed: int):
        """Apply an external disturbance/load before each control step by writing
        `data.xfrc_applied` (random pushes, a held load, a headwind), scaled by
        `self.difficulty`. Default: no-op."""
        return None

    def escalate(self, steps: float = 1.0):
        """Ratchet difficulty (POET curriculum, Stage 8). Subclasses may map
        difficulty onto concrete parameters; the base just tracks it."""
        self.difficulty += self.escalation_rate * steps
        return self

    # --- declarative config ------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["task"] = type(self).__name__
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "NicheTask":
        return cls(**{k: v for k, v in d.items() if k != "task"})
