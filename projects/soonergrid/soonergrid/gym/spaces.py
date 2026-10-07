"""
Observation and Action Space Definitions.

Provides Gymnasium-compatible space objects when the ``gymnasium`` package
is available, and lightweight pure-Python fallback spaces otherwise.

This design ensures the environment can be imported and used for
integration testing without requiring gymnasium as a hard dependency,
while still being fully compatible with standard RL frameworks when
gymnasium is installed.

Space Specifications:
    Observation Space: Box(low=0.0, high=1.0, shape=(36,), dtype=float32)
    Action Space:      Box(low=[-1, 0], high=[1, 1], shape=(2,), dtype=float32)
"""

from __future__ import annotations

import random
from typing import Any, List, Optional, Tuple, Sequence

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import gymnasium
    from gymnasium.spaces import Box as GymBox
    HAS_GYMNASIUM = True
except ImportError:
    gymnasium = None  # type: ignore[assignment]
    HAS_GYMNASIUM = False


class FallbackBox:
    """
    Lightweight pure-Python replacement for gymnasium.spaces.Box.

    Supports the minimal API needed by PettingZoo compatibility checks:
    - .shape, .low, .high, .dtype
    - .sample() -> list of floats
    - .contains(x) -> bool
    - .seed(n) for reproducibility

    This is used when gymnasium is not installed, allowing the environment
    to be tested and used with custom training loops that don't require
    the full Gymnasium dependency.
    """

    def __init__(
        self,
        low: Sequence[float],
        high: Sequence[float],
        shape: Optional[Tuple[int, ...]] = None,
        dtype: str = "float32",
    ):
        if shape is not None:
            self.shape = shape
            if isinstance(low, (int, float)):
                self.low = [float(low)] * shape[0]
            else:
                self.low = [float(x) for x in low]
            if isinstance(high, (int, float)):
                self.high = [float(high)] * shape[0]
            else:
                self.high = [float(x) for x in high]
        else:
            self.low = [float(x) for x in low]
            self.high = [float(x) for x in high]
            self.shape = (len(self.low),)

        self.dtype = dtype
        self._rng = random.Random()

    def sample(self) -> List[float]:
        """Uniform random sample within [low, high]."""
        return [
            self._rng.uniform(lo, hi)
            for lo, hi in zip(self.low, self.high)
        ]

    def contains(self, x: Any) -> bool:
        """Check if x is within bounds."""
        if hasattr(x, '__len__'):
            if len(x) != self.shape[0]:
                return False
            return all(
                lo - 1e-6 <= float(xi) <= hi + 1e-6
                for xi, lo, hi in zip(x, self.low, self.high)
            )
        return False

    def seed(self, seed: int) -> None:
        """Set random seed for reproducible sampling."""
        self._rng = random.Random(seed)

    def __repr__(self) -> str:
        return f"FallbackBox(shape={self.shape}, low={self.low[:3]}..., high={self.high[:3]}...)"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, (FallbackBox,)):
            return False
        return self.shape == other.shape and self.low == other.low and self.high == other.high


def make_observation_space(obs_dim: int) -> Any:
    """
    Creates the observation space for a single agent.

    Shape: (obs_dim,) with all values in [0, 1].
    Uses gymnasium.spaces.Box if available, otherwise FallbackBox.

    Args:
        obs_dim: Dimensionality of observation vector.

    Returns:
        A space object with .sample(), .contains(), .shape attributes.
    """
    if HAS_GYMNASIUM and HAS_NUMPY:
        return GymBox(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )
    return FallbackBox(
        low=[0.0] * obs_dim,
        high=[1.0] * obs_dim,
        shape=(obs_dim,),
    )


def make_action_space(action_dim: int) -> Any:
    """
    Creates the action space for a single agent.

    Shape: (action_dim,)
    - action[0]: green split delta in [-1, 1]
    - action[1]: phase hold probability in [0, 1]

    Uses gymnasium.spaces.Box if available, otherwise FallbackBox.

    Args:
        action_dim: Dimensionality of action vector (expected: 2).

    Returns:
        A space object with .sample(), .contains(), .shape attributes.
    """
    low_vals = [-1.0, 0.0][:action_dim]
    high_vals = [1.0, 1.0][:action_dim]

    if HAS_GYMNASIUM and HAS_NUMPY:
        return GymBox(
            low=np.array(low_vals, dtype=np.float32),
            high=np.array(high_vals, dtype=np.float32),
            shape=(action_dim,),
            dtype=np.float32,
        )
    return FallbackBox(
        low=low_vals,
        high=high_vals,
        shape=(action_dim,),
    )
