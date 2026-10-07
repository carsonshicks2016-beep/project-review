"""Structural diff / innovation detector (ROADMAP Stage 4.4).

Compares a parent genome to a child and reports *what* changed structurally, as a
set of legible flags plus a graph-edit-distance scalar. It is record-free (works
purely from the two genomes), so it reflects the actual structural delta — useful
for the phylogeny's innovation timeline (4.5) and as a morphological distance for
speciation (Stage 7).

Flags:
    +limb / -limb        a part (body segment) was added / removed
    +muscle / -muscle    a muscle-tendon unit was added / removed
    +compartment         a muscle was split into >=2 co-located compartments
                         (increased independent control of one origin->insertion)
    topology_change      connectivity changed (edge recursion/symmetry, or a
                         muscle's routing/attachment) among shared parts
    dof_change           a joint's degrees of freedom changed (e.g. hinge<->ball)

Continuous (micro) perturbations produce NO flags and zero structural distance.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from .genome import Genome


def _edge_sig(e):
    """Structure-only edge signature (excludes continuous pos/quat/scale and the
    joint *type*, which is reported separately as dof_change)."""
    return (e.parent_part, e.child_part, e.site_idx, e.recursion_count,
            e.symmetry.value, e.symmetry_count)


def _muscle_route_sig(m):
    return (m.origin, m.insertion, tuple(m.route_sites))


@dataclass
class InnovationReport:
    flags: set = field(default_factory=set)
    parts_added: set = field(default_factory=set)
    parts_removed: set = field(default_factory=set)
    muscles_added: set = field(default_factory=set)
    muscles_removed: set = field(default_factory=set)
    distance: int = 0

    def __bool__(self) -> bool:
        return bool(self.flags)


def diff(parent: Genome, child: Genome) -> InnovationReport:
    pp = {p.id for p in parent.parts}
    cp = {p.id for p in child.parts}
    parts_added, parts_removed = cp - pp, pp - cp

    pm = {m.id for m in parent.muscles}
    cm = {m.id for m in child.muscles}
    musc_added, musc_removed = cm - pm, pm - cm

    flags: set = set()
    if parts_added:
        flags.add("+limb")
    if parts_removed:
        flags.add("-limb")
    if musc_added:
        flags.add("+muscle")
    if musc_removed:
        flags.add("-muscle")

    # +compartment: an (origin, insertion) pair gained a 2nd+ muscle
    pc = Counter((m.origin, m.insertion) for m in parent.muscles)
    cc = Counter((m.origin, m.insertion) for m in child.muscles)
    if any(n >= 2 and n > pc.get(pair, 0) for pair, n in cc.items()):
        flags.add("+compartment")

    # topology_change: edge structure or muscle routing differs among shared parts
    shared = pp & cp
    pe = Counter(_edge_sig(e) for e in parent.edges
                 if e.parent_part in shared and e.child_part in shared)
    ce = Counter(_edge_sig(e) for e in child.edges
                 if e.parent_part in shared and e.child_part in shared)
    edge_edits = sum((pe - ce).values()) + sum((ce - pe).values())

    pr = {m.id: _muscle_route_sig(m) for m in parent.muscles}
    cr = {m.id: _muscle_route_sig(m) for m in child.muscles}
    route_edits = sum(1 for mid in (pm & cm) if pr[mid] != cr[mid])
    if edge_edits or route_edits:
        flags.add("topology_change")

    # dof_change: a matched edge's joint type changed
    pj = {(e.parent_part, e.child_part, e.site_idx): e.joint.type for e in parent.edges}
    cj = {(e.parent_part, e.child_part, e.site_idx): e.joint.type for e in child.edges}
    dof_edits = sum(1 for k in (pj.keys() & cj.keys()) if pj[k] != cj[k])
    if dof_edits:
        flags.add("dof_change")

    distance = (len(parts_added) + len(parts_removed) + len(musc_added)
                + len(musc_removed) + edge_edits + route_edits + dof_edits)
    return InnovationReport(flags, parts_added, parts_removed, musc_added,
                            musc_removed, distance)


def innovations(parent: Genome, child: Genome) -> set:
    """The set of structural-innovation flags from parent to child."""
    return diff(parent, child).flags


def structural_distance(parent: Genome, child: Genome) -> int:
    """Graph-edit-distance-style count of structural edits (0 for micro-only /
    identical genomes). Symmetric, usable as a morphological distance."""
    return diff(parent, child).distance
