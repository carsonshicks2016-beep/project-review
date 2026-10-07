"""Niche tasks: persistent selective regimes a creature is evaluated against
(ROADMAP Stage 2.4 / 2.5). Stage 2 ships locomotion; Stage 7 adds the rest of the
gauntlet (iron zone, ballistics, endurance, terrain, ...).
"""
from .base import NicheTask, task_from_dict, register_task
from .locomotion import LocomotionTask
from .track import TrackTask
from .iron_zone import IronZoneTask
from .ballistics import BallisticsTask
from .endurance import EnduranceTask
from .chaos_grid import ChaosGridTask
from .terrain import TerrainTask

# the niche gauntlet (Stage 7.1), keyed by short name for selection in runs
NICHES = {
    "locomotion": LocomotionTask,
    "track": TrackTask,
    "iron_zone": IronZoneTask,
    "ballistics": BallisticsTask,
    "endurance": EnduranceTask,
    "chaos_grid": ChaosGridTask,
    "terrain": TerrainTask,
}

__all__ = [
    "NicheTask", "LocomotionTask", "TrackTask", "IronZoneTask", "BallisticsTask",
    "EnduranceTask", "ChaosGridTask", "TerrainTask", "NICHES",
    "task_from_dict", "register_task",
]
