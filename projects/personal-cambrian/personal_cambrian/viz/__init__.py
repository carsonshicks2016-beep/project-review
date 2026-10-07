"""Visualization (ROADMAP Stage 10): see the creatures, their motion, and the
family tree of body plans.

10.1 ships the bpy-free `morphology_plan` (a JSON-serializable scene description
extracted from a developed Morphology) and the `blender_build` bpy builder that
turns it into Blender objects. The plan is deliberately decoupled from Blender so
the geometry extraction is unit-testable without bpy, and the same dict can be
shipped into a headless `blender --background --python` run or the Blender MCP.
"""
from .morphology_plan import (
    GeomSpec, MuscleSpec, ScenePlan, build_plan, plan_from_genome,
)
from .trajectory import Trajectory, record_trajectory
from .biomechanics import BioTrajectory, record_biomechanics
from .phylo_render import (
    tree_layout, innovation_markers, render_phylogeny, render_from_json,
    INNOVATION_COLORS,
)
from .compare import lineage_plan, lineage_genomes, lineage_plan_from_phylogeny

__all__ = [
    "GeomSpec", "MuscleSpec", "ScenePlan", "build_plan", "plan_from_genome",
    "Trajectory", "record_trajectory",
    "BioTrajectory", "record_biomechanics",
    "tree_layout", "innovation_markers", "render_phylogeny", "render_from_json",
    "INNOVATION_COLORS",
    "lineage_plan", "lineage_genomes", "lineage_plan_from_phylogeny",
]
