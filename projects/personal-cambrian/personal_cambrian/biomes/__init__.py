"""Biome registry. The full PLAN.md gauntlet has 10 biomes; this prototype ships
the 4 with defensible closed-form models. The rest (Chaos Grid, Elasticity
Field, Skill Arena, Durability Trial, Recovery Chamber, Real-World Terrain) are
specified in PLAN.md and arrive with the MuJoCo substrate.
"""
from .base import Biome, BiomeResult
from .track import Track
from .iron_zone import IronZone
from .ballistics import Ballistics
from .endurance import Endurance

ALL_BIOMES: list[Biome] = [Track(), IronZone(), Ballistics(), Endurance()]


def evaluate_all(agent) -> dict[str, BiomeResult]:
    return {b.name: b.evaluate(agent) for b in ALL_BIOMES}
