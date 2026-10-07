"""Mastery-gated curriculum: promote / demote on completion rate, never on time.

A timestep schedule marches a policy onto stages it cannot drive. Mastery
gates wait until the rolling window clears the bar.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class CurriculumConfig:
    window: int = 100
    promote_at: float = 0.75
    demote_at: float = 0.35
    max_tier: int = 5
    min_tier: int = 0


@dataclass
class CurriculumEvent:
    old_tier: int
    new_tier: int
    completion_rate: float
    reason: str  # "promote" | "demote"


class Curriculum:
    """Rolling-window stage-completion curriculum."""

    def __init__(
        self,
        *,
        tier: int = 0,
        config: CurriculumConfig | None = None,
    ) -> None:
        self.cfg = config or CurriculumConfig()
        self.tier = int(tier)
        self._outcomes: deque[float] = deque(maxlen=self.cfg.window)
        # After a demotion, require a full window refill before re-promoting so
        # a policy does not thrash between adjacent tiers.
        self._cooldown = 0

    def observe(self, finished_clean: bool) -> CurriculumEvent | None:
        """Record one episode outcome; maybe promote or demote."""
        self._outcomes.append(1.0 if finished_clean else 0.0)
        if self._cooldown > 0:
            self._cooldown -= 1
            return None
        if len(self._outcomes) < self.cfg.window:
            return None
        rate = float(sum(self._outcomes) / len(self._outcomes))
        if rate >= self.cfg.promote_at and self.tier < self.cfg.max_tier:
            old = self.tier
            self.tier += 1
            self._outcomes.clear()
            self._cooldown = self.cfg.window
            return CurriculumEvent(old, self.tier, rate, "promote")
        if rate <= self.cfg.demote_at and self.tier > self.cfg.min_tier:
            old = self.tier
            self.tier -= 1
            self._outcomes.clear()
            self._cooldown = self.cfg.window
            return CurriculumEvent(old, self.tier, rate, "demote")
        return None

    @property
    def completion_rate(self) -> float | None:
        if not self._outcomes:
            return None
        return float(sum(self._outcomes) / len(self._outcomes))

    def state_dict(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "outcomes": list(self._outcomes),
            "cooldown": self._cooldown,
            "window": self.cfg.window,
            "promote_at": self.cfg.promote_at,
            "demote_at": self.cfg.demote_at,
            "max_tier": self.cfg.max_tier,
            "min_tier": self.cfg.min_tier,
        }

    def load_state_dict(self, data: dict[str, Any]) -> None:
        self.tier = int(data["tier"])
        self.cfg = CurriculumConfig(
            window=int(data.get("window", self.cfg.window)),
            promote_at=float(data.get("promote_at", self.cfg.promote_at)),
            demote_at=float(data.get("demote_at", self.cfg.demote_at)),
            max_tier=int(data.get("max_tier", self.cfg.max_tier)),
            min_tier=int(data.get("min_tier", self.cfg.min_tier)),
        )
        self._outcomes = deque(
            (float(x) for x in data.get("outcomes", [])),
            maxlen=self.cfg.window,
        )
        self._cooldown = int(data.get("cooldown", 0))
