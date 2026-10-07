"""Genome mutation operators (ROADMAP Stage 4).

4.1 — `micro(genome, rng, sigma)`: perturb continuous parameters (segment sizes,
density, joint range/stiffness/damping, edge scale, muscle Hill params). Topology
is unchanged, so the actuator count and adjacency are preserved (a child's
controller warm-starts directly from the parent's). All values are clamped to
physical ranges, so the result always passes `validate_shape()` and compiles.

Macro-mutations (add/remove/reroute structure) are Stage 4.2.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .genome import (
    Shape, JointType, SymmetryKind, AttachmentSite, Joint, PartNode,
    ConnectionEdge, MuscleGene, Genome,
)

# floors for strictly-positive quantities
_MIN_DIM = 0.005
_MIN_DENSITY = 100.0
_MIN_PCSA = 0.5
_MIN_LEN = 0.01
_MIN_VMAX = 0.5
_MIN_SCALE = 0.05
_MAX_PENNATION = 1.30          # < pi/2


def _scale(v: float, rng, sigma: float, floor: float) -> float:
    """Multiplicative jitter for a strictly-positive quantity, floored."""
    return max(floor, float(v) * (1.0 + rng.normal(0.0, sigma)))


def micro(genome: Genome, rng: Optional[np.random.Generator] = None,
          sigma: float = 0.1) -> Genome:
    """Return a continuously-perturbed copy of `genome` (topology preserved)."""
    rng = rng or np.random.default_rng()
    g = Genome.from_json(genome.to_json())          # deep copy (re-runs coercion)

    for p in g.parts:
        p.dims = {k: _scale(v, rng, sigma, _MIN_DIM) for k, v in p.dims.items()}
        p.density = _scale(p.density, rng, sigma, _MIN_DENSITY)

    for e in g.edges:
        lo, hi = e.joint.range                      # additive (radians/metres)
        lo += rng.normal(0.0, sigma)
        hi += rng.normal(0.0, sigma)
        e.joint.range = (min(lo, hi), max(lo, hi))
        e.joint.stiffness = max(0.0, e.joint.stiffness + rng.normal(0.0, sigma))
        e.joint.damping = max(0.0, e.joint.damping + rng.normal(0.0, sigma))
        e.scale = _scale(e.scale, rng, sigma, _MIN_SCALE)

    for m in g.muscles:
        m.pcsa_cm2 = _scale(m.pcsa_cm2, rng, sigma, _MIN_PCSA)
        m.optimal_fiber_len = _scale(m.optimal_fiber_len, rng, sigma, _MIN_LEN)
        m.tendon_slack_len = _scale(m.tendon_slack_len, rng, sigma, _MIN_LEN)
        m.vmax = _scale(m.vmax, rng, sigma, _MIN_VMAX)
        m.activation_cost = _scale(m.activation_cost, rng, sigma, 0.1)
        m.pennation = float(np.clip(m.pennation + rng.normal(0.0, sigma), 0.0, _MAX_PENNATION))
        m.fiber_type = float(np.clip(m.fiber_type + rng.normal(0.0, sigma), 0.0, 1.0))

    return g


# =========================================================================== #
# 4.2 macro-mutations: structural edits (each keeps the genome developable).    #
# =========================================================================== #
@dataclass
class MutationRecord:
    kind: str
    detail: dict = field(default_factory=dict)
    ok: bool = True            # False => no valid target, returned an unchanged copy


def _copy(g: Genome) -> Genome:
    return Genome.from_json(g.to_json())


def _uid(existing: set, base: str) -> str:
    n = 0
    while f"{base}{n}" in existing:
        n += 1
    return f"{base}{n}"


def _norm(v) -> float:
    return float(sum(c * c for c in v)) ** 0.5


def _random_dims(shape: Shape, rng) -> dict:
    if shape is Shape.CAPSULE:
        return {"radius": float(rng.uniform(0.02, 0.06)), "length": float(rng.uniform(0.10, 0.30))}
    if shape is Shape.SPHERE:
        return {"radius": float(rng.uniform(0.03, 0.08))}
    return {k: float(rng.uniform(0.03, 0.10)) for k in ("x", "y", "z")}   # box / ellipsoid


def _rand_axis(rng):
    return [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)][int(rng.integers(3))]


def _joint_copy(j: Joint) -> Joint:
    return Joint(j.type, tuple(j.axis), tuple(j.range), j.stiffness, j.damping)


def _edge_copy(e: ConnectionEdge, parent=None, child=None) -> ConnectionEdge:
    return ConnectionEdge(parent or e.parent_part, child or e.child_part, e.site_idx,
                          pos=tuple(e.pos), quat=tuple(e.quat), scale=e.scale,
                          joint=_joint_copy(e.joint), recursion_count=e.recursion_count,
                          terminal_only=e.terminal_only, symmetry=e.symmetry,
                          symmetry_count=e.symmetry_count)


def _muscle_copy(m: MuscleGene, *, id, origin=None, insertion=None, route_sites=None,
                 pcsa=None) -> MuscleGene:
    return MuscleGene(id=id, origin=origin or m.origin, insertion=insertion or m.insertion,
                      route_sites=route_sites if route_sites is not None else list(m.route_sites),
                      pcsa_cm2=pcsa if pcsa is not None else m.pcsa_cm2,
                      optimal_fiber_len=m.optimal_fiber_len, tendon_slack_len=m.tendon_slack_len,
                      pennation=m.pennation, vmax=m.vmax, fiber_type=m.fiber_type,
                      activation_cost=m.activation_cost)


def _adjacency(g: Genome) -> dict:
    adj: dict[str, list[str]] = {}
    for e in g.edges:
        adj.setdefault(e.parent_part, []).append(e.child_part)
    return adj


def _prune_unreachable(g: Genome) -> None:
    adj = _adjacency(g)
    seen, stack = set(), [g.root]
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        stack.extend(adj.get(c, []))
    g.parts = [p for p in g.parts if p.id in seen]
    g.edges = [e for e in g.edges if e.parent_part in seen and e.child_part in seen]
    g.muscles = [m for m in g.muscles
                 if m.origin[0] in seen and m.insertion[0] in seen
                 and all(r[0] in seen for r in m.route_sites)]


def _sites_of(g: Genome):
    return [(p.id, si) for p in g.parts for si in range(len(p.sites))]


# --- structural operators -------------------------------------------------
def add_part(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    """Grow a new appendage: a new part type attached to an existing site."""
    g = _copy(g)
    cands = _sites_of(g)
    if not cands:
        return g, MutationRecord("add_part", ok=False)
    parent_id, site_idx = cands[int(rng.integers(len(cands)))]
    new_id = _uid({p.id for p in g.parts}, "part")
    shape = [Shape.CAPSULE, Shape.BOX, Shape.SPHERE, Shape.ELLIPSOID][int(rng.integers(4))]
    sites = [AttachmentSite("s0", pos=(0.0, 0.0, -0.10)),
             AttachmentSite("s1", pos=(0.05, 0.0, -0.05))]
    g.parts.append(PartNode(id=new_id, shape=shape, dims=_random_dims(shape, rng),
                            sites=sites, recursion_limit=2))
    g.edges.append(ConnectionEdge(parent_id, new_id, site_idx, pos=(0.0, 0.0, -0.05),
                                  joint=Joint(JointType.HINGE, axis=_rand_axis(rng),
                                              range=(-1.0, 1.0))))
    return g, MutationRecord("add_part", {"new_part": new_id, "parent": parent_id})


def delete_subtree(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    """Remove a part type and prune everything that becomes unreachable (mass refund)."""
    g = _copy(g)
    nonroot = [p.id for p in g.parts if p.id != g.root]
    if not nonroot:
        return g, MutationRecord("delete_subtree", ok=False)
    target = nonroot[int(rng.integers(len(nonroot)))]
    g.parts = [p for p in g.parts if p.id != target]
    g.edges = [e for e in g.edges if e.parent_part != target and e.child_part != target]
    g.muscles = [m for m in g.muscles
                 if m.origin[0] != target and m.insertion[0] != target
                 and all(r[0] != target for r in m.route_sites)]
    _prune_unreachable(g)
    return g, MutationRecord("delete_subtree", {"removed": target})


def duplicate_subtree(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    """Clone a part's subtree under fresh ids and attach it at another site."""
    g = _copy(g)
    nonroot = [p.id for p in g.parts if p.id != g.root]
    if not nonroot:
        return g, MutationRecord("duplicate_subtree", ok=False)
    root_t = nonroot[int(rng.integers(len(nonroot)))]
    adj = _adjacency(g)
    sub, stack = set(), [root_t]
    while stack:
        c = stack.pop()
        if c in sub:
            continue
        sub.add(c)
        stack.extend(adj.get(c, []))

    existing = {p.id for p in g.parts}
    remap = {}
    for old in sub:
        new = _uid(existing, old + "_d")
        existing.add(new)
        remap[old] = new
    byid = {p.id: p for p in g.parts}
    for old in sub:
        p = byid[old]
        g.parts.append(PartNode(id=remap[old], shape=p.shape, dims=dict(p.dims),
                                density=p.density, recursion_limit=p.recursion_limit,
                                sites=[AttachmentSite(s.name, s.pos, s.quat) for s in p.sites]))
    for e in list(g.edges):
        if e.parent_part in sub and e.child_part in sub:
            g.edges.append(_edge_copy(e, parent=remap[e.parent_part], child=remap[e.child_part]))
    for m in list(g.muscles):
        if (m.origin[0] in sub and m.insertion[0] in sub
                and all(r[0] in sub for r in m.route_sites)):
            g.muscles.append(_muscle_copy(
                m, id=_uid({mm.id for mm in g.muscles}, m.id + "_d"),
                origin=(remap[m.origin[0]], m.origin[1]),
                insertion=(remap[m.insertion[0]], m.insertion[1]),
                route_sites=[(remap[r[0]], r[1]) for r in m.route_sites]))
    # attach the duplicate root to a site on a non-duplicate part
    attach = [(p.id, si) for p in g.parts if p.id not in remap.values()
              for si in range(len(p.sites))]
    if attach:
        ap, asi = attach[int(rng.integers(len(attach)))]
        g.edges.append(ConnectionEdge(ap, remap[root_t], asi, pos=(0.0, 0.0, -0.05),
                                      joint=Joint(JointType.HINGE, axis=(0.0, 1.0, 0.0),
                                                  range=(-1.0, 1.0))))
    return g, MutationRecord("duplicate_subtree", {"source": root_t, "n_parts": len(sub)})


def change_recursion_count(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    g = _copy(g)
    if not g.edges:
        return g, MutationRecord("change_recursion_count", ok=False)
    e = g.edges[int(rng.integers(len(g.edges)))]
    # always a real change: 1 can only go up; otherwise +/-1
    e.recursion_count = 2 if e.recursion_count <= 1 else \
        e.recursion_count + (1 if rng.random() < 0.5 else -1)
    return g, MutationRecord("change_recursion_count", {"count": e.recursion_count})


def toggle_symmetry(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    g = _copy(g)
    if not g.edges:
        return g, MutationRecord("toggle_symmetry", ok=False)
    e = g.edges[int(rng.integers(len(g.edges)))]
    order = [SymmetryKind.NONE, SymmetryKind.BILATERAL, SymmetryKind.RADIAL]
    e.symmetry = order[(order.index(e.symmetry) + 1) % 3]
    if e.symmetry is SymmetryKind.RADIAL:
        e.symmetry_count = max(2, e.symmetry_count)
    return g, MutationRecord("toggle_symmetry", {"symmetry": e.symmetry.value})


def add_muscle(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    """Add a muscle across an articulated joint (origin on parent, insertion on child)."""
    g = _copy(g)
    byid = {p.id: p for p in g.parts}
    cand = [e for e in g.edges if e.joint.type is not JointType.FIXED
            and byid[e.parent_part].sites and byid[e.child_part].sites]
    if not cand:
        return g, MutationRecord("add_muscle", ok=False)
    e = cand[int(rng.integers(len(cand)))]
    oi = int(rng.integers(len(byid[e.parent_part].sites)))
    ii = int(rng.integers(len(byid[e.child_part].sites)))
    if (e.parent_part, oi) == (e.child_part, ii):           # self-edge, same site
        if len(byid[e.child_part].sites) < 2:
            return g, MutationRecord("add_muscle", ok=False)
        ii = (ii + 1) % len(byid[e.child_part].sites)
    mid = _uid({m.id for m in g.muscles}, "mus")
    g.muscles.append(MuscleGene(id=mid, origin=(e.parent_part, oi), insertion=(e.child_part, ii),
                                pcsa_cm2=float(rng.uniform(5.0, 25.0)),
                                fiber_type=float(rng.uniform(0.3, 0.7))))
    return g, MutationRecord("add_muscle", {"muscle": mid, "edge": (e.parent_part, e.child_part)})


def remove_muscle(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    g = _copy(g)
    if not g.muscles:
        return g, MutationRecord("remove_muscle", ok=False)
    m = g.muscles.pop(int(rng.integers(len(g.muscles))))
    return g, MutationRecord("remove_muscle", {"muscle": m.id})


def reroute_tendon(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    g = _copy(g)
    if not g.muscles:
        return g, MutationRecord("reroute_tendon", ok=False)
    sites = _sites_of(g)
    m = g.muscles[int(rng.integers(len(g.muscles)))]
    wp = sites[int(rng.integers(len(sites)))]
    m.route_sites = list(m.route_sites) + [(wp[0], int(wp[1]))]
    return g, MutationRecord("reroute_tendon", {"muscle": m.id, "added_route": list(wp)})


def split_muscle(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    """Split a muscle into two independently-controlled compartments (half PCSA each)."""
    g = _copy(g)
    if not g.muscles:
        return g, MutationRecord("split_muscle", ok=False)
    m = g.muscles.pop(int(rng.integers(len(g.muscles))))
    for tag in ("a", "b"):
        g.muscles.append(_muscle_copy(m, id=_uid({mm.id for mm in g.muscles}, m.id + tag),
                                      pcsa=m.pcsa_cm2 / 2.0))
    return g, MutationRecord("split_muscle", {"source": m.id})


def fuse_muscles(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    """Fuse two muscles into one (summed PCSA), reducing control independence."""
    g = _copy(g)
    if len(g.muscles) < 2:
        return g, MutationRecord("fuse_muscles", ok=False)
    i, j = (int(x) for x in rng.choice(len(g.muscles), size=2, replace=False))
    a, b = g.muscles[i], g.muscles[j]
    fused = _muscle_copy(a, id=_uid({m.id for m in g.muscles}, "fused"),
                         pcsa=a.pcsa_cm2 + b.pcsa_cm2)
    g.muscles = [m for k, m in enumerate(g.muscles) if k not in (i, j)] + [fused]
    return g, MutationRecord("fuse_muscles", {"fused": [a.id, b.id]})


def add_joint_dof(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    g = _copy(g)
    up = {JointType.FIXED: JointType.HINGE, JointType.HINGE: JointType.BALL,
          JointType.SLIDE: JointType.BALL}
    cand = [e for e in g.edges if e.joint.type in up]
    if not cand:
        return g, MutationRecord("add_joint_dof", ok=False)
    e = cand[int(rng.integers(len(cand)))]
    e.joint.type = up[e.joint.type]
    if e.joint.type is JointType.BALL:
        hi = max(abs(e.joint.range[0]), abs(e.joint.range[1]), 1.0)
        e.joint.range = (0.0, hi)
    else:                                                   # FIXED -> HINGE
        e.joint.range = (-1.0, 1.0)
        if _norm(e.joint.axis) == 0:
            e.joint.axis = (0.0, 1.0, 0.0)
    return g, MutationRecord("add_joint_dof", {"new_type": e.joint.type.value})


def remove_dof(g: Genome, rng) -> tuple[Genome, MutationRecord]:
    g = _copy(g)
    down = {JointType.BALL: JointType.HINGE, JointType.HINGE: JointType.FIXED,
            JointType.SLIDE: JointType.FIXED}
    cand = [e for e in g.edges if e.joint.type in down]
    if not cand:
        return g, MutationRecord("remove_dof", ok=False)
    e = cand[int(rng.integers(len(cand)))]
    e.joint.type = down[e.joint.type]
    if e.joint.type is JointType.HINGE:
        e.joint.range = (-1.0, 1.0)
        if _norm(e.joint.axis) == 0:
            e.joint.axis = (0.0, 1.0, 0.0)
    return g, MutationRecord("remove_dof", {"new_type": e.joint.type.value})


MACRO_OPERATORS = {fn.__name__: fn for fn in (
    add_part, delete_subtree, duplicate_subtree, change_recursion_count,
    toggle_symmetry, add_muscle, remove_muscle, reroute_tendon, split_muscle,
    fuse_muscles, add_joint_dof, remove_dof,
)}


def macro(genome: Genome, rng: Optional[np.random.Generator] = None
          ) -> tuple[Genome, MutationRecord]:
    """Apply one random *applicable* macro-mutation. Tries operators in random
    order until one reports ok; falls back to an unchanged copy if none apply."""
    rng = rng or np.random.default_rng()
    ops = list(MACRO_OPERATORS.values())
    for i in rng.permutation(len(ops)):
        g, rec = ops[int(i)](genome, rng)
        if rec.ok:
            return g, rec
    return _copy(genome), MutationRecord("noop", ok=False)


# =========================================================================== #
# 4.3 mutation scheduling: mostly micro, rare (optionally annealed) macro.      #
# =========================================================================== #
@dataclass
class MutationSchedule:
    """Controls innovation rate over deep time. Every mutation applies micro; a
    macro structural change is added with probability `rate(generation)`."""
    macro_rate: float = 0.2
    micro_sigma: float = 0.1
    anneal: str = "none"               # none | linear | exp
    final_macro_rate: float = 0.2      # target rate at `horizon` (for annealing)
    horizon: int = 1000

    def rate(self, generation: int = 0) -> float:
        if self.anneal == "none":
            return self.macro_rate
        t = float(np.clip(generation / max(self.horizon, 1), 0.0, 1.0))
        if self.anneal == "linear":
            return self.macro_rate + t * (self.final_macro_rate - self.macro_rate)
        if self.anneal == "exp":
            r0 = max(self.macro_rate, 1e-9)
            return float(r0 * (max(self.final_macro_rate, 1e-9) / r0) ** t)
        raise ValueError(f"unknown anneal '{self.anneal}'")

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MutationSchedule":
        return cls(**d)


def mutate(genome: Genome, rng: Optional[np.random.Generator] = None, *,
           generation: int = 0, schedule: Optional[MutationSchedule] = None
           ) -> tuple[Genome, list[MutationRecord]]:
    """One mutation event: micro-perturb, then (rarely) add a macro change.
    Returns (child, records) where records logs what was applied."""
    rng = rng or np.random.default_rng()
    schedule = schedule or MutationSchedule()
    g = micro(genome, rng, schedule.micro_sigma)
    records = [MutationRecord("micro", {"sigma": schedule.micro_sigma})]
    if rng.random() < schedule.rate(generation):
        g, rec = macro(g, rng)
        records.append(rec)
    return g, records

