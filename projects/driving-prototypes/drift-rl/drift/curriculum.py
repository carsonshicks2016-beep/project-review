"""Curriculum callback: add pillars and time as the fleet learns to link tricks."""

from __future__ import annotations

from collections import deque

from stable_baselines3.common.callbacks import BaseCallback

from drift.config import CURRICULUM


class CurriculumCallback(BaseCallback):
    """Promote on normalised score rate over a sliding window of episodes.

    Scoring on a *rate* rather than a total keeps the bar meaningful as later
    stages get longer. A patience limit promotes anyway if a stage stops
    improving, so a badly calibrated bar cannot pin training on rung one.
    """

    def __init__(self, window: int = 200, min_episodes: int = 120,
                 patience_steps: int = 1_200_000, verbose: int = 1):
        super().__init__(verbose)
        self.stage_idx = 0
        self.window = window
        self.min_episodes = min_episodes
        self.patience_steps = patience_steps
        self.results: deque[float] = deque(maxlen=window)
        self.history: list[tuple[int, str, float]] = []
        self.stage_started = 0

    def _on_training_start(self) -> None:
        self._apply(self.stage_idx)

    def _apply(self, idx: int) -> None:
        self.training_env.env_method("set_stage", CURRICULUM[idx])

    def _promote(self, rate: float, why: str) -> None:
        old = CURRICULUM[self.stage_idx].name
        self.stage_idx += 1
        self._apply(self.stage_idx)
        self.results.clear()
        self.stage_started = self.num_timesteps
        self.history.append((self.num_timesteps, CURRICULUM[self.stage_idx].name, rate))
        if self.verbose:
            print(f"[curriculum] {self.num_timesteps:>9,} steps | {old} {why} "
                  f"(norm score {rate:.2f}) -> {CURRICULUM[self.stage_idx].name}")

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.results.append(float(info.get("norm_score", 0.0)))

        last = self.stage_idx >= len(CURRICULUM) - 1
        if len(self.results) >= self.min_episodes and not last:
            rate = sum(self.results) / len(self.results)
            stage = CURRICULUM[self.stage_idx]
            self.logger.record("curriculum/norm_score", rate)
            self.logger.record("curriculum/stage", self.stage_idx)
            if rate >= stage.promote_at:
                self._promote(rate, "cleared")
            elif self.num_timesteps - self.stage_started > self.patience_steps:
                self._promote(rate, "timed out")
        return True
