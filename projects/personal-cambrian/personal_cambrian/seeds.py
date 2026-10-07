"""Hand-authored seed genomes (ROADMAP Stage 1.8).

`agent_zero()`  - a human-ish biped (the ancestral seed for evolution).
`quadruped()`   - a deliberately NON-human body plan (4 legs + tail), proving the
                  generative encoding produces more than rescaled humans.

Conventions used here (see encoding/genome.py): each part's geom is centered at
its body frame origin, and the body origin sits at the attachment joint. So distal
"next-joint" sites are placed at NEGATIVE local z to make limbs hang downward
(identity orientations => local axes align with world). Bilateral symmetry (plane
xz -> mirror y) turns one limb rule into a left/right pair. Muscles either live
fully inside a mirrored limb subtree (knee/ankle/elbow) or originate near the
body midline (hip), so they mirror without crossing the body.
"""
from __future__ import annotations

from .encoding.genome import (
    Shape, JointType, SymmetryKind, AttachmentSite as S, Joint, PartNode,
    ConnectionEdge, MuscleGene, Genome,
)


def agent_zero() -> Genome:
    """Human-ish biped: torso + head + bilateral arms (upper+fore) + bilateral
    legs (thigh+shank+foot). Actuated at hips, knees, ankles, elbows."""
    torso = PartNode(id="torso", shape=Shape.CAPSULE, dims={"radius": 0.12, "length": 0.40}, sites=[
        S("neck", pos=(0.0, 0.0, 0.24)),
        S("shoulder", pos=(0.0, 0.15, 0.12)),
        S("hip", pos=(0.0, 0.09, -0.22)),
        S("hipm", pos=(0.13, 0.0, -0.16)),        # hip-muscle origin (near midline)
    ])
    head = PartNode(id="head", shape=Shape.SPHERE, dims={"radius": 0.10})
    uparm = PartNode(id="uparm", shape=Shape.CAPSULE, dims={"radius": 0.045, "length": 0.26}, sites=[
        S("elbow", pos=(0.0, 0.0, -0.14)),
        S("elbm", pos=(0.05, 0.0, -0.10)),
    ])
    forearm = PartNode(id="forearm", shape=Shape.CAPSULE, dims={"radius": 0.04, "length": 0.24}, sites=[
        S("elbins", pos=(0.05, 0.0, 0.10)),
    ])
    thigh = PartNode(id="thigh", shape=Shape.CAPSULE, dims={"radius": 0.07, "length": 0.40}, sites=[
        S("knee", pos=(0.0, 0.0, -0.22)),
        S("hipins", pos=(0.06, 0.0, 0.12)),
        S("kneem", pos=(0.05, 0.0, -0.12)),
    ])
    shank = PartNode(id="shank", shape=Shape.CAPSULE, dims={"radius": 0.05, "length": 0.40}, sites=[
        S("ankle", pos=(0.0, 0.0, -0.22)),
        S("kneeins", pos=(0.05, 0.0, 0.12)),
        S("anklem", pos=(0.04, 0.0, -0.12)),
    ])
    foot = PartNode(id="foot", shape=Shape.BOX, dims={"x": 0.10, "y": 0.05, "z": 0.03}, sites=[
        S("ankins", pos=(0.04, 0.0, 0.0)),
    ])

    edges = [
        ConnectionEdge("torso", "head", site_idx=0, joint=Joint(JointType.FIXED)),
        ConnectionEdge("torso", "uparm", site_idx=1, symmetry=SymmetryKind.BILATERAL,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1.5, 1.5))),
        ConnectionEdge("uparm", "forearm", site_idx=0,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(0.0, 2.3))),
        ConnectionEdge("torso", "thigh", site_idx=2, symmetry=SymmetryKind.BILATERAL,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1.2, 1.6))),
        ConnectionEdge("thigh", "shank", site_idx=0,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-2.4, 0.0))),
        ConnectionEdge("shank", "foot", site_idx=0,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-0.8, 0.8))),
    ]
    muscles = [
        MuscleGene(id="hip", origin=("torso", 3), insertion=("thigh", 1), pcsa_cm2=40.0),
        MuscleGene(id="knee", origin=("thigh", 2), insertion=("shank", 1), pcsa_cm2=30.0),
        MuscleGene(id="ankle", origin=("shank", 2), insertion=("foot", 0), pcsa_cm2=18.0),
        MuscleGene(id="elbow", origin=("uparm", 1), insertion=("forearm", 0), pcsa_cm2=15.0),
    ]
    return Genome(root="torso", parts=[torso, head, uparm, forearm, thigh, shank, foot],
                  edges=edges, muscles=muscles, max_depth=10)


def quadruped() -> Genome:
    """Non-human: a horizontal spine with a head, a 4-segment tail, and four legs
    (front + back bilateral pairs, same leg type). Actuated at the knees."""
    spine = PartNode(id="spine", shape=Shape.BOX, dims={"x": 0.22, "y": 0.08, "z": 0.06}, sites=[
        S("neck", pos=(0.22, 0.0, 0.04)),
        S("fronthip", pos=(0.16, 0.08, -0.04)),
        S("backhip", pos=(-0.16, 0.08, -0.04)),
        S("tail", pos=(-0.22, 0.0, 0.02)),
    ])
    head = PartNode(id="qhead", shape=Shape.SPHERE, dims={"radius": 0.08})
    qthigh = PartNode(id="qthigh", shape=Shape.CAPSULE, dims={"radius": 0.04, "length": 0.20}, sites=[
        S("qknee", pos=(0.0, 0.0, -0.11)),
        S("qkneem", pos=(0.03, 0.0, -0.06)),
    ])
    qshank = PartNode(id="qshank", shape=Shape.CAPSULE, dims={"radius": 0.035, "length": 0.20}, sites=[
        S("qkneeins", pos=(0.03, 0.0, 0.06)),
    ])
    tailseg = PartNode(id="tailseg", shape=Shape.CAPSULE, dims={"radius": 0.03, "length": 0.12},
                       recursion_limit=5, sites=[S("tnext", pos=(-0.12, 0.0, 0.0))])

    edges = [
        ConnectionEdge("spine", "qhead", site_idx=0, joint=Joint(JointType.FIXED)),
        ConnectionEdge("spine", "qthigh", site_idx=1, symmetry=SymmetryKind.BILATERAL,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1.0, 1.0))),
        ConnectionEdge("spine", "qthigh", site_idx=2, symmetry=SymmetryKind.BILATERAL,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1.0, 1.0))),
        ConnectionEdge("qthigh", "qshank", site_idx=0,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-1.8, 0.0))),
        # first tail segment off the spine, then a self-edge chains 3 more
        # (a recursive chain must self-reference so the wrap site exists on the child)
        ConnectionEdge("spine", "tailseg", site_idx=3,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-0.5, 0.5))),
        ConnectionEdge("tailseg", "tailseg", site_idx=0, recursion_count=3,
                       joint=Joint(JointType.HINGE, axis=(0, 1, 0), range=(-0.5, 0.5))),
    ]
    muscles = [
        MuscleGene(id="qknee", origin=("qthigh", 1), insertion=("qshank", 0), pcsa_cm2=12.0),
    ]
    return Genome(root="spine", parts=[spine, head, qthigh, qshank, tailseg],
                  edges=edges, muscles=muscles, max_depth=10)


SEEDS = {"agent_zero": agent_zero, "quadruped": quadruped}
