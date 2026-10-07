"""Generative developmental encoding (PLAN.md §7, ROADMAP Stage 1).

This package replaces the Stage-0 flat 14-gene vector (`personal_cambrian.genome`)
with a Sims-style **recursive body-graph** genome: a graph of part *types* plus
connection rules with recursion and symmetry, from which `morphogenesis` (Stage
1.3+) grows a concrete, simulatable creature. This is what lets evolution *invent*
structure (new limbs, muscles, kinetic chains) rather than only rescale a human.

Stage 1.1 (this commit) defines the data structures + shape validation only.
Compilation to MuJoCo lives in `morphogenesis/` (Stage 1.5).
"""
from .genome import (
    Vec3, Quat,
    Shape, JointType, SymmetryKind, SHAPE_DIMS,
    AttachmentSite, Joint, PartNode, ConnectionEdge, MuscleGene, Genome,
    GenomeValidationError, SPECIFIC_TENSION_N_PER_CM2,
)
from .mutate import (
    micro, macro, mutate, MutationRecord, MutationSchedule, MACRO_OPERATORS,
)
from .innovation import diff, innovations, structural_distance, InnovationReport

__all__ = [
    "Vec3", "Quat",
    "Shape", "JointType", "SymmetryKind", "SHAPE_DIMS",
    "AttachmentSite", "Joint", "PartNode", "ConnectionEdge", "MuscleGene", "Genome",
    "GenomeValidationError", "SPECIFIC_TENSION_N_PER_CM2",
    "micro", "macro", "mutate", "MutationRecord", "MutationSchedule", "MACRO_OPERATORS",
    "diff", "innovations", "structural_distance", "InnovationReport",
]
