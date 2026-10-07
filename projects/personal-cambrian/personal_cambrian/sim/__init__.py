"""Physics evaluation (ROADMAP Stage 2): wrap a compiled creature + a task as a
standard Gymnasium RL environment, so any creature can be scored by physically
performing a task. This is the first stage where creatures are *evaluated*.
"""
from .env import CreatureEnv
from .tasks import (
    NicheTask, LocomotionTask, TrackTask, IronZoneTask, BallisticsTask,
    EnduranceTask, ChaosGridTask, TerrainTask, NICHES, task_from_dict,
)
# Stage 9.1 MJX backend: always importable (mjx_available() guards actual use), but
# only loads jax lazily inside mjx_env, so the package works without the GPU extra.
from .mjx_env import mjx_available, MjxBatchEnv, dynamics_parity, grounded_qpos

__all__ = [
    "CreatureEnv", "NicheTask", "LocomotionTask", "TrackTask", "IronZoneTask",
    "BallisticsTask", "EnduranceTask", "ChaosGridTask", "TerrainTask",
    "NICHES", "task_from_dict",
    "mjx_available", "MjxBatchEnv", "dynamics_parity", "grounded_qpos",
]
