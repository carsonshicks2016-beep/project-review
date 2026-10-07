"""Unit tests for MarketMakingEnv Gym interface and step dynamics."""

import pytest
import numpy as np
from src.env.market_making_env import MarketMakingEnv


def test_env_reset_and_step():
    env = MarketMakingEnv(episode_length=50, seed=123)
    obs, info = env.reset(seed=123)

    assert obs.shape == (16,)
    assert isinstance(info, dict)
    assert info["total_equity"] == 10_000.0

    # Step with zero offset action
    action = np.array([0.0, 0.0, 0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, next_info = env.step(action)

    assert next_obs.shape == (16,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
