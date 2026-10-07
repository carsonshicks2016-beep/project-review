"""Supra Drift — a from-scratch neuro-evolution / RL drifting simulator.

Four-wheel physics + procedural tracks + raycast/proprioceptive sensors + a
drivable PyGame app, a genetic-algorithm racer, and a PyTorch PPO agent that
trains both race and drift (mode-conditioned). See README.md / HANDOFF.md.
"""
from . import config, physics, sensors, track

__all__ = ["config", "physics", "sensors", "track"]
__version__ = "1.0.0"
