"""Morphogenesis: expand a generative `Genome` into a concrete creature.

Stage 1.3 (this commit): `develop(genome) -> Morphology` — an engine-neutral body
(bodies, joints, sites, muscle routes), no MuJoCo yet. Stage 1.4 adds symmetry
expansion; Stage 1.5 compiles `Morphology` to a MuJoCo model.
"""
from .develop import (
    Morphology, Body, SiteInstance, MuscleRoute, Waypoint, EdgeInstance, develop,
)
from .budgets import Budgets, compute_budgets

__all__ = [
    "Morphology", "Body", "SiteInstance", "MuscleRoute", "Waypoint",
    "EdgeInstance", "develop",
    "Budgets", "compute_budgets",
]
