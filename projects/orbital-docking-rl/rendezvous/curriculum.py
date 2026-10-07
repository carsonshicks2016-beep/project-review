"""Curriculum callback: promote every worker once the fleet is reliable."""

from __future__ import annotations

from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from .config import CURRICULUM


class CurriculumCallback(BaseCallback):
    """Advance the spawn envelope when the recent success rate clears the bar.

    Success is measured over a sliding window of finished episodes, and the
    window is cleared on promotion so the next stage is judged on its own.
    """

    def __init__(self, window: int = 200, min_episodes: int = 120, verbose: int = 1):
        super().__init__(verbose)
        self.stage_idx = 0
        self.window = window
        self.min_episodes = min_episodes
        self.results: deque[float] = deque(maxlen=window)
        self.history: list[tuple[int, str, float]] = []

    def _on_training_start(self) -> None:
        self._apply(self.stage_idx)

    def _apply(self, idx: int) -> None:
        self.training_env.env_method("set_stage", CURRICULUM[idx])

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.results.append(float(info.get("is_success", False)))

        if len(self.results) >= self.min_episodes:
            rate = sum(self.results) / len(self.results)
            stage = CURRICULUM[self.stage_idx]
            self.logger.record("curriculum/success_rate", rate)
            self.logger.record("curriculum/stage", self.stage_idx)
            if rate >= stage.promote_at and self.stage_idx < len(CURRICULUM) - 1:
                self.stage_idx += 1
                self._apply(self.stage_idx)
                self.results.clear()
                self.history.append((self.num_timesteps, CURRICULUM[self.stage_idx].name, rate))
                if self.verbose:
                    print(
                        f"[curriculum] {self.num_timesteps:>9,} steps | "
                        f"{stage.name} solved at {rate:.0%} -> {CURRICULUM[self.stage_idx].name}"
                    )
        return True
