"""GEOMETRIC VIABILITY GATE (ROADMAP Stage 6.2): reject broken bodies before sim.

Simulating a malformed creature wastes the (expensive) eval budget and often
just blows MuJoCo up (NaN QACC). This module rejects, cheaply and *before* any
rollout, bodies that are geometrically non-viable:

  * self-intersection -- two NON-adjacent parts occupy essentially the same
    space (the degenerate co-location that subtree-duplication mutations create
    and that detonates contact solvers). Measured as the closest non-adjacent
    centre-to-centre distance normalised by part size; a valid creature's limbs
    may *clip* (the seeds do, at the zero rest pose) but their centres stay well
    apart, so coincidence -- not mere overlap -- is the signal.
  * immobile joints -- a hinge/slide with a degenerate (zero-span) range can
    never move, so it is dead weight (ROM must be preserved).
  * dead muscles -- an actuator with ~zero moment arm about every DOF can never
    produce joint torque (it pulls straight through the joint axis), and a body
    with no actuators at all cannot act.

Diagnostics the spec also asks for -- a broadphase AABB-overlap count and the
MuJoCo rest-pose contact count -- are reported but, because the seeds legitimately
clip at rest, are NOT used as hard rejections.

[APPROX] Thresholds are calibrated so both hand-authored seeds pass with margin
(min centre/size: quad 1.14, biped 0.67; min moment arm 0.03-0.04 m) while
degenerate constructions fail (coincident parts 0.0; on-axis muscle 0.0 m).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import mujoco

from ..encoding.genome import Genome, Shape, JointType
from ..morphogenesis import develop
from ..morphogenesis.to_mujoco import compile_morphology

# --- thresholds [APPROX] ----------------------------------------------------
COINCIDENCE_RATIO = 0.3      # non-adjacent centres closer than this*size => stacked
DEGENERATE_RANGE = 1e-4      # rad (hinge) / m (slide): span below this => immobile
MOMENT_ARM_TOL = 1e-3        # m: geometric moment arm below this => dead muscle


def _bounding_radius(shape: Shape, dims: dict) -> float:
    """Radius of a sphere enclosing the (already world-scaled) geom."""
    if shape is Shape.SPHERE:
        return dims["radius"]
    if shape is Shape.CAPSULE:
        return dims["length"] / 2.0 + dims["radius"]
    if shape is Shape.BOX:
        return math.sqrt(dims["x"] ** 2 + dims["y"] ** 2 + dims["z"] ** 2)
    if shape is Shape.ELLIPSOID:
        return max(dims["x"], dims["y"], dims["z"])
    return 0.0


def _adjacency(morph) -> set:
    """Parent-child pairs (these are allowed to overlap at their shared joint)."""
    return {frozenset((b.id, b.parent_id))
            for b in morph.bodies if b.parent_id is not None}


@dataclass
class ViabilityReport:
    """Why a body is (not) viable, plus cheap geometric diagnostics."""
    viable: bool
    reasons: list = field(default_factory=list)
    # diagnostics
    n_bodies: int = 0
    n_actuators: int = 0
    n_aabb_overlaps: int = 0          # broadphase, non-adjacent
    n_rest_contacts: int = 0          # MuJoCo narrowphase at rest pose
    n_coincident_pairs: int = 0
    n_degenerate_joints: int = 0
    n_dead_muscles: int = 0
    min_center_ratio: float = float("inf")   # closest non-adjacent centre/size
    min_moment_arm: float = float("inf")     # m, smallest actuator moment arm

    def __bool__(self) -> bool:
        return self.viable

    def to_dict(self) -> dict:
        return {
            "viable": self.viable, "reasons": list(self.reasons),
            "n_bodies": self.n_bodies, "n_actuators": self.n_actuators,
            "n_aabb_overlaps": self.n_aabb_overlaps,
            "n_rest_contacts": self.n_rest_contacts,
            "n_coincident_pairs": self.n_coincident_pairs,
            "n_degenerate_joints": self.n_degenerate_joints,
            "n_dead_muscles": self.n_dead_muscles,
            "min_center_ratio": self.min_center_ratio,
            "min_moment_arm": self.min_moment_arm,
        }


def _geometry_checks(morph, report: ViabilityReport) -> None:
    """Pure-Python broadphase: AABB overlaps + coincident (stacked) parts."""
    adj = _adjacency(morph)
    bodies = morph.bodies
    rad = {b.id: _bounding_radius(b.shape, b.dims) for b in bodies}
    pos = {b.id: np.asarray(b.world_pos, float) for b in bodies}
    for i, bi in enumerate(bodies):
        for bj in bodies[i + 1:]:
            if frozenset((bi.id, bj.id)) in adj:
                continue
            ri, rj = rad[bi.id], rad[bj.id]
            delta = pos[bi.id] - pos[bj.id]
            # broadphase AABB (bounding-sphere boxes) overlap
            if np.all(np.abs(delta) <= (ri + rj)):
                report.n_aabb_overlaps += 1
            # coincident centres relative to the smaller part
            dist = float(np.linalg.norm(delta))
            ratio = dist / max(min(ri, rj), 1e-9)
            report.min_center_ratio = min(report.min_center_ratio, ratio)
            if ratio < COINCIDENCE_RATIO:
                report.n_coincident_pairs += 1


def _joint_checks(morph, report: ViabilityReport) -> None:
    """Each articulating joint must keep a non-degenerate range of motion."""
    for b in morph.bodies:
        j = b.joint
        if j is None or j.type is JointType.FIXED:
            continue
        if j.type is JointType.BALL:
            if max(abs(j.range[0]), abs(j.range[1])) < DEGENERATE_RANGE:
                report.n_degenerate_joints += 1
        else:  # HINGE / SLIDE
            if (j.range[1] - j.range[0]) < DEGENERATE_RANGE:
                report.n_degenerate_joints += 1


def _actuation_checks(model, report: ViabilityReport) -> None:
    """Every actuator must have a non-zero moment arm about some DOF."""
    report.n_actuators = int(model.nu)
    if model.nu == 0:
        return
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    dense = np.zeros((model.nu, model.nv))
    mujoco.mju_sparse2dense(dense, data.actuator_moment, data.moment_rownnz,
                            data.moment_rowadr, data.moment_colind)
    gear = np.abs(model.actuator_gear[:, 0])
    # geometric moment arm = |generalized-force moment| / |gear| (gear=Fmax)
    arms = np.abs(dense).max(axis=1) / np.maximum(gear, 1e-9)
    report.min_moment_arm = float(arms.min())
    report.n_dead_muscles = int((arms < MOMENT_ARM_TOL).sum())
    report.n_rest_contacts = int(data.ncon)


def check_viability(genome: Genome) -> ViabilityReport:
    """Reject geometrically non-viable bodies before they reach simulation.

    Returns a `ViabilityReport`; `bool(report)` is True iff the body is viable.
    """
    report = ViabilityReport(viable=True)

    # the genome must grow and compile at all
    try:
        morph = develop(genome)
    except Exception as e:  # noqa: BLE001
        report.viable = False
        report.reasons.append(f"develop_error: {type(e).__name__}")
        return report
    report.n_bodies = morph.body_count

    _geometry_checks(morph, report)
    _joint_checks(morph, report)

    try:
        model, _ = compile_morphology(morph, add_floor=False, self_collide=True)
    except Exception as e:  # noqa: BLE001
        report.viable = False
        report.reasons.append(f"compile_error: {type(e).__name__}")
        return report

    _actuation_checks(model, report)

    # --- collapse diagnostics into hard rejection reasons ------------------
    if report.n_coincident_pairs > 0:
        report.reasons.append("self_intersection")
    if report.n_degenerate_joints > 0:
        report.reasons.append("immobile_joint")
    if report.n_actuators == 0:
        report.reasons.append("no_actuators")
    elif report.n_dead_muscles > 0:
        report.reasons.append("zero_moment_arm")

    report.viable = not report.reasons
    return report
