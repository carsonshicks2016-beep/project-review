"""Observation and return normalisation for PPO.

A policy restored without its observation normaliser is a different policy.
``(mean, var, count)`` travels in the same checkpoint file as the weights —
never a sidecar.

Welford statistics update only while ``frozen`` is false (rollout collection).
Evaluation freezes the normaliser so scores do not depend on eval order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

# Default gym observation size from SensorSpec().size (84).
DEFAULT_OBS_DIM = 84
OBS_CLIP = 10.0
EPS = 1e-8
MIN_VAR = 1e-12

# Append-only block layout matching sense/suite.py with default SensorSpec.
# Used for the "no block is degenerate" measurement; not a second source of size.
BLOCK_SLICES: dict[str, slice] = {
    "vision": slice(0, 18),       # 2 * 9 beams
    "proprio": slice(18, 43),     # 25
    "lookahead": slice(43, 65),   # 4 + 3 * 6
    "terrain": slice(65, 84),     # 7 + 2 * 6
}


@dataclass
class NormaliserState:
    """Serialisable payload — travels next to policy weights."""

    mean: np.ndarray
    var: np.ndarray
    count: float
    clip: float
    ret_mean: float = 0.0
    ret_var: float = 1.0
    ret_count: float = 1e-4

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.astype(np.float64).tolist(),
            "var": self.var.astype(np.float64).tolist(),
            "count": float(self.count),
            "clip": float(self.clip),
            "ret_mean": float(self.ret_mean),
            "ret_var": float(self.ret_var),
            "ret_count": float(self.ret_count),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormaliserState:
        return cls(
            mean=np.asarray(data["mean"], dtype=np.float64),
            var=np.asarray(data["var"], dtype=np.float64),
            count=float(data["count"]),
            clip=float(data.get("clip", OBS_CLIP)),
            ret_mean=float(data.get("ret_mean", 0.0)),
            ret_var=float(data.get("ret_var", 1.0)),
            ret_count=float(data.get("ret_count", 1e-4)),
        )


class ObservationNormaliser:
    """Per-dimension Welford mean/variance with optional freeze for eval."""

    def __init__(self, dim: int = DEFAULT_OBS_DIM, *, clip: float = OBS_CLIP) -> None:
        if dim < 1:
            raise ValueError(f"dim must be >= 1, got {dim}")
        self.dim = int(dim)
        self.clip = float(clip)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.var = np.ones(self.dim, dtype=np.float64)
        self.count = 1e-4
        # Return-scale tracker for the value target / vf_coef sizing.
        self.ret_mean = 0.0
        self.ret_var = 1.0
        self.ret_count = 1e-4
        self.frozen = False

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    def update(self, x: np.ndarray) -> bool:
        """Update running stats from a batch of observations. No-op when frozen."""
        if self.frozen:
            return False
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        if x.shape[-1] != self.dim:
            raise ValueError(f"obs dim {x.shape[-1]} != normaliser dim {self.dim}")
        # Drop non-finite rows so one corrupt env cannot poison the stats.
        x = x[np.all(np.isfinite(x), axis=1)]
        if x.shape[0] == 0:
            return False
        self._welford_batch(x)
        return True

    def _welford_batch(self, x: np.ndarray) -> None:
        b_mean = x.mean(axis=0)
        b_var = x.var(axis=0)
        b_n = float(x.shape[0])
        delta = b_mean - self.mean
        tot = self.count + b_n
        self.mean = self.mean + delta * (b_n / tot)
        m_a = self.var * self.count
        m_b = b_var * b_n
        self.var = np.maximum(
            (m_a + m_b + (delta ** 2) * self.count * b_n / tot) / tot,
            MIN_VAR,
        )
        self.count = tot

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Standardise and clip. Does not update statistics."""
        x = np.asarray(x, dtype=np.float64)
        out = (x - self.mean) / np.sqrt(self.var + EPS)
        return np.clip(out, -self.clip, self.clip).astype(np.float32)

    # Alias used by some call sites / Supra port muscle memory.
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.normalize(x)

    def update_returns(self, returns: np.ndarray | float) -> bool:
        """Track episode-return scale for value-loss / vf_coef sizing."""
        if self.frozen:
            return False
        r = np.asarray(returns, dtype=np.float64).reshape(-1)
        r = r[np.isfinite(r)]
        if r.size == 0:
            return False
        b_mean = float(r.mean())
        b_var = float(r.var()) if r.size > 1 else 0.0
        b_n = float(r.size)
        delta = b_mean - self.ret_mean
        tot = self.ret_count + b_n
        self.ret_mean += delta * (b_n / tot)
        m_a = self.ret_var * self.ret_count
        m_b = b_var * b_n
        self.ret_var = max(
            (m_a + m_b + (delta ** 2) * self.ret_count * b_n / tot) / tot,
            MIN_VAR,
        )
        self.ret_count = tot
        return True

    def normalize_returns(self, returns: np.ndarray | float) -> np.ndarray:
        r = np.asarray(returns, dtype=np.float64)
        out = (r - self.ret_mean) / np.sqrt(self.ret_var + EPS)
        return out.astype(np.float32)

    def state_dict(self) -> NormaliserState:
        return NormaliserState(
            mean=self.mean.copy(),
            var=self.var.copy(),
            count=float(self.count),
            clip=float(self.clip),
            ret_mean=float(self.ret_mean),
            ret_var=float(self.ret_var),
            ret_count=float(self.ret_count),
        )

    def load_state_dict(self, state: NormaliserState | dict[str, Any]) -> None:
        if isinstance(state, dict):
            state = NormaliserState.from_dict(state)
        if state.mean.shape != (self.dim,):
            raise ValueError(
                f"normaliser mean shape {state.mean.shape} != ({self.dim},)"
            )
        self.mean = np.asarray(state.mean, dtype=np.float64).copy()
        self.var = np.asarray(state.var, dtype=np.float64).copy()
        self.count = float(state.count)
        self.clip = float(state.clip)
        self.ret_mean = float(state.ret_mean)
        self.ret_var = float(state.ret_var)
        self.ret_count = float(state.ret_count)

    def to_dict(self) -> dict[str, Any]:
        return self.state_dict().to_dict()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ObservationNormaliser:
        mean = np.asarray(data["mean"], dtype=np.float64)
        obj = cls(dim=int(mean.shape[0]), clip=float(data.get("clip", OBS_CLIP)))
        obj.load_state_dict(data)
        return obj

    def block_stats(self) -> dict[str, dict[str, float]]:
        """Mean/variance summary per observation block (for the B3 measure)."""
        out: dict[str, dict[str, float]] = {}
        for name, sl in BLOCK_SLICES.items():
            if sl.stop > self.dim:
                continue
            out[name] = {
                "mean_abs": float(np.abs(self.mean[sl]).mean()),
                "var_mean": float(self.var[sl].mean()),
                "var_min": float(self.var[sl].min()),
                "var_max": float(self.var[sl].max()),
            }
        return out

    def recommended_vf_coef(self, *, base: float = 0.5, target_return_std: float = 1.0) -> float:
        """Scale ``vf_coef`` so value loss stays near unit-return magnitude.

        Episode returns here sit in the low thousands. Stock PPO ``vf_coef=0.5``
        assumes unit-scale returns; without this (or return normalisation) the
        value term dominates the clipped surrogate. Phase C can either
        normalise returns via ``normalize_returns`` or multiply ``vf_coef`` by
        this factor — same idea, one dial.
        """
        ret_std = float(np.sqrt(self.ret_var + EPS))
        if ret_std < 1e-3:
            return float(base)
        # Value targets ~ return_std; shrink vf_coef so value_loss * vf_coef
        # stays comparable to a unit-scale run.
        scale = (target_return_std / ret_std) ** 2
        return float(base * scale)

    def return_scale_note(self) -> dict[str, float]:
        """Compact guidance payload for metrics / checkpoint metadata."""
        return {
            "ret_mean": float(self.ret_mean),
            "ret_std": float(np.sqrt(self.ret_var + EPS)),
            "ret_count": float(self.ret_count),
            "recommended_vf_coef": self.recommended_vf_coef(),
        }
