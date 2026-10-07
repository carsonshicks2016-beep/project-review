"""Action-space compatibility helpers for PPO models."""

import numpy as np


def ppo_action_to_allocation(model, action) -> float:
    """Map a PPO action to a portfolio allocation, supporting old and new models."""
    val = float(np.squeeze(action))
    action_space = getattr(model, "action_space", None)
    low = None
    high = None
    if action_space is not None:
        try:
            low = float(np.squeeze(action_space.low))
            high = float(np.squeeze(action_space.high))
        except Exception:
            low = None
            high = None

    if low is not None and low < 0:
        return float(np.clip((val + 1.0) / 2.0, 0.0, 1.0))

    return float(np.clip(val, 0.0, 1.0))


def action_space_mode_for_model(model) -> str:
    """Return the TradingEnv action mode expected by a saved PPO model."""
    action_space = getattr(model, "action_space", None)
    if action_space is None:
        return "allocation"
    try:
        low = float(np.squeeze(action_space.low))
    except Exception:
        return "allocation"
    return "symmetric" if low < 0 else "allocation"
