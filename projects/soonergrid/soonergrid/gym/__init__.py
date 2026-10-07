"""
SoonerGrid Gymnasium/PettingZoo Multi-Agent RL Environment.

Provides a PettingZoo-compatible Parallel Environment wrapper around the
ResilientTrafficSimulationEngine for multi-agent signal control optimization.

Key Components:
    - NormanTrafficEnv: PettingZoo ParallelEnv with 7 signal agents
    - SteppableTrafficEngine: Single-step simulation core extracted from the
      monolithic run_simulation() loop
    - Observation/Action space definitions with Gymnasium Box/Discrete spaces
"""

from soonergrid.gym.env import NormanTrafficEnv
from soonergrid.gym.steppable_engine import SteppableTrafficEngine

__all__ = [
    "NormanTrafficEnv",
    "SteppableTrafficEngine",
]
