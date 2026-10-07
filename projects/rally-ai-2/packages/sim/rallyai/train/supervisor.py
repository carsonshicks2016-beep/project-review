"""Automated training supervisor — collapse detection and auto-rollback.

Ported from Supra AI 2's ``supervise_fable5.py``, adapted for RallyAI's
point-to-point architecture. The supervisor hooks into the training loop
after each PPO update and periodically runs a quick deterministic eval
probe. If the policy collapses, it rolls back to the last HoF best
checkpoint. If it plateaus, it flags the event for the operator.

The supervisor must **never** crash the training loop. Every public
method wraps its work in ``try / except`` and logs errors without
re-raising.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from rallyai.train.ppo import PPOUpdateStats, Trainer


@dataclass
class Decision:
    """A single supervisor decision, logged to ``decisions.jsonl``."""

    kind: str  # 'rollback', 'plateau_reseed', 'probe_ok', 'probe_warning'
    timesteps: int
    wall_t: float
    detail: str
    probe_completion: float | None = None
    probe_best: float | None = None


class Supervisor:
    """Monitors training and intervenes on collapse or plateau.

    Call ``after_update`` after each PPO update. The supervisor decides
    internally when to run a probe (every ``probe_interval`` timesteps).
    """

    def __init__(
        self,
        *,
        trainer: Trainer,
        probe_interval: int = 50_000,
        ntfy_topic: str | None = None,
        plateau_probes: int = 5,
        collapse_threshold: float = 0.5,
    ) -> None:
        self._trainer = trainer
        self._probe_interval = int(probe_interval)
        self._plateau_probes = int(plateau_probes)
        self._collapse_threshold = float(collapse_threshold)
        self._last_probe_ts = 0
        self._probe_history: list[float] = []
        self._best_completion = 0.0

        # Notifier — gracefully degrade if the module hasn't been imported.
        try:
            from rallyai.train.notify import Notifier
            self._notifier = Notifier(ntfy_topic, run_id=trainer.run_id)
        except Exception:
            self._notifier = None

        # Decision log file, written to the run's output directory.
        self._decisions_path = Path(trainer.out_dir) / "decisions.jsonl"
        self._decisions_fh = self._decisions_path.open("a", encoding="utf-8")

    # ------------------------------------------------------------------ #
    # Public hook
    # ------------------------------------------------------------------ #

    def after_update(
        self,
        timesteps: int,
        window_stats: dict[str, Any],
        ppo_stats: PPOUpdateStats,
    ) -> list[Decision]:
        """Called after each PPO update. Returns any decisions made."""
        decisions: list[Decision] = []
        try:
            if timesteps - self._last_probe_ts < self._probe_interval:
                return decisions

            self._last_probe_ts = timesteps
            completion = self._run_probe()
            self._probe_history.append(completion)
            if completion > self._best_completion:
                self._best_completion = completion

            now = time.time()

            # --- collapse detection ----------------------------------- #
            if (
                len(self._probe_history) > 1
                and self._best_completion > 0.0
                and completion < self._best_completion * self._collapse_threshold
            ):
                d = Decision(
                    kind="rollback",
                    timesteps=timesteps,
                    wall_t=now,
                    detail=(
                        f"Collapse: probe completion {completion:.1%} "
                        f"< {self._collapse_threshold:.0%} of best "
                        f"({self._best_completion:.1%}). "
                        f"Rolling back to HoF best."
                    ),
                    probe_completion=completion,
                    probe_best=self._best_completion,
                )
                decisions.append(d)
                self._log_decision(d)
                self._notify(
                    f"Collapse at t={timesteps}: "
                    f"{completion:.1%} vs best {self._best_completion:.1%}",
                    priority="high",
                )
                self._rollback()
                # Reset history after rollback.
                self._probe_history.clear()
                self._best_completion = 0.0
                return decisions

            # --- plateau detection ------------------------------------ #
            if len(self._probe_history) >= self._plateau_probes:
                recent = self._probe_history[-self._plateau_probes:]
                if max(recent) <= self._best_completion and all(
                    r <= self._probe_history[-self._plateau_probes - 1]
                    if len(self._probe_history) > self._plateau_probes
                    else True
                    for r in recent
                ):
                    # No improvement across the plateau window.
                    d = Decision(
                        kind="plateau_reseed",
                        timesteps=timesteps,
                        wall_t=now,
                        detail=(
                            f"Plateau: no improvement over last "
                            f"{self._plateau_probes} probes. "
                            f"Best={self._best_completion:.1%}, "
                            f"latest={completion:.1%}."
                        ),
                        probe_completion=completion,
                        probe_best=self._best_completion,
                    )
                    decisions.append(d)
                    self._log_decision(d)
                    self._notify(
                        f"Plateau at t={timesteps}: "
                        f"best={self._best_completion:.1%}",
                    )
                    # Reset the window so we don't re-fire every update.
                    self._probe_history.clear()
                    return decisions

            # --- normal probe ----------------------------------------- #
            d = Decision(
                kind="probe_ok",
                timesteps=timesteps,
                wall_t=now,
                detail=f"Probe OK: completion={completion:.1%}",
                probe_completion=completion,
                probe_best=self._best_completion,
            )
            decisions.append(d)
            self._log_decision(d)

        except Exception as exc:
            print(
                f"[supervisor] error in after_update (non-fatal): {exc}",
                flush=True,
            )
        return decisions

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _run_probe(self) -> float:
        """Quick deterministic eval on 4 held-out seeds at the current tier."""
        from rallyai.train.evaluate import (
            ObsMeanActor,
            evaluate,
            held_out_seeds,
        )

        # Deep-copy the normaliser so we don't mutate the live one.
        norm_copy = copy.deepcopy(self._trainer.normaliser)
        norm_copy.freeze()

        actor = ObsMeanActor(
            self._trainer.policy, norm_copy, name="supervisor_probe"
        )
        seeds = held_out_seeds(4)
        result = evaluate(
            seeds=seeds,
            tier=self._trainer.tier,
            actor=actor,
            require_held_out=True,
        )
        return float(result.get("completion_rate", 0.0))

    def _rollback(self) -> None:
        """Load the HoF best checkpoint into the trainer's live state."""
        best_path = self._trainer.hof.path_for("best")
        if not best_path.exists():
            print(
                "[supervisor] no HoF best checkpoint available for rollback",
                flush=True,
            )
            return
        from rallyai.train.checkpoint import load_checkpoint

        payload = load_checkpoint(best_path)
        self._trainer.policy.load_state_dict(payload["state_dict"])
        if payload.get("opt") is not None:
            self._trainer.ppo.opt.load_state_dict(payload["opt"])
        self._trainer.normaliser.load_state_dict(payload["normaliser"])
        self._trainer.policy.clamp_log_std_()
        self._notify(f"Rolled back to {best_path.name}", priority="high")

    def _log_decision(self, d: Decision) -> None:
        """Append a decision line to ``decisions.jsonl``."""
        try:
            self._decisions_fh.write(
                json.dumps(asdict(d), separators=(",", ":")) + "\n"
            )
            self._decisions_fh.flush()
        except Exception as exc:
            print(f"[supervisor] decision log write error: {exc}", flush=True)

    def _notify(self, message: str, *, priority: str = "default") -> None:
        if self._notifier is not None:
            try:
                self._notifier.send("Supervisor", message, priority=priority)
            except Exception:
                pass

    def close(self) -> None:
        """Close the decision log file handle."""
        try:
            self._decisions_fh.close()
        except Exception:
            pass
