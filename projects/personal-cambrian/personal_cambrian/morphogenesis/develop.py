"""Morphogenesis traversal (ROADMAP Stage 1.3).

`develop(genome)` walks the body-graph from the root and instantiates a concrete
`Morphology`: a list of `Body` instances (each with a world transform, a parent-
relative transform, scaled geometry, its articulating `Joint`, and resolved
`SiteInstance`s) plus `MuscleRoute`s with instance-level waypoints.

Key semantics (documented here so Stage 1.4/1.5/1.6 stay consistent):

* **Chaining.** An edge with `recursion_count = k` appends a chain of k child
  instances (child, grandchild, ...), each attached at the same `site_idx` on the
  previous instance's type. The edge that builds a chain is **not** re-applied
  while expanding the chain's own elements (otherwise a self-edge would explode);
  other edges on those elements expand normally.
* **Termination.** Growth stops when global `max_depth` is reached, or when a part
  type's `recursion_limit` (count of that type along the current root→node path)
  would be exceeded, or when the chaining site index is missing on the new parent.
* **`terminal_only` edges** fire only on a chain's terminus (its deepest element).
  The root is treated as a non-terminus, so terminal structures hang off chain
  tips, not off the base.
* **Transforms.** child_world = parent_world ∘ scale(site ∘ edge); a child's local
  offset is scaled by the parent's cumulative scale; geometry scales by the
  cumulative product of edge scales.
* **Muscles (1.3 resolution).** A `MuscleGene` referencing part *types* O and I is
  instantiated once per developed edge whose endpoints are an O-instance and an
  I-instance (i.e., per joint between them) — so chain muscles repeat per joint.
  Route waypoints that reference O or I resolve to those endpoints; routes through
  other types are deferred to Stage 1.6.

Symmetry (`bilateral`/`radial`) is intentionally **not** expanded here — that is
Stage 1.4. This stage instantiates only the primary (un-mirrored) child.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math
import numpy as np

from ..encoding.genome import Genome, PartNode, Joint, MuscleGene, Shape

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]


# --------------------------------------------------------------------------- #
# quaternion / vector helpers (w, x, y, z); inputs need not be unit            #
# --------------------------------------------------------------------------- #
def _qnorm(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _qrot(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by (normalized) quaternion q."""
    q = _qnorm(q)
    u = q[1:]
    uv = np.cross(u, v)
    uuv = np.cross(u, uv)
    return v + 2.0 * (q[0] * uv + uuv)


def _qconj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _t(a: np.ndarray) -> tuple:
    return tuple(float(x) for x in a)


# --- symmetry helpers ------------------------------------------------------ #
# The symmetry plane names the two in-plane axes; the *normal* is the third and
# is the axis a reflection negates (default "xz" -> normal y -> left/right).
_PLANE_NORMAL = {"xz": 1, "xy": 2, "yz": 0}


def _reflect_vec(v, ax: int) -> np.ndarray:
    """Reflect a position across the plane whose normal is axis `ax`."""
    out = np.array(v, dtype=float)
    out[ax] = -out[ax]
    return out


def _mirror_quat(q, ax: int) -> np.ndarray:
    """Mirror an orientation across the plane with normal axis `ax`.

    Conjugating a rotation by a coordinate-plane reflection (F R F) stays a proper
    rotation: keep w and the normal-axis vector component, negate the other two.
    """
    out = np.array(q, dtype=float)
    for i in (0, 1, 2):           # vector components live at quat indices 1..3
        if i != ax:
            out[i + 1] = -out[i + 1]
    return out


def _rotz(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2.0), 0.0, 0.0, math.sin(theta / 2.0)])


# --------------------------------------------------------------------------- #
# concrete instances                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class SiteInstance:
    name: str
    idx: int
    rel_pos: Vec3   # relative to its own body frame (scaled)
    rel_quat: Quat
    world_pos: Vec3
    world_quat: Quat


@dataclass
class Body:
    id: str                 # unique instance id, e.g. "limb#3"
    part_id: str            # the part *type* it instantiates
    shape: Shape
    dims: dict[str, float]  # geometry, already scaled
    density: float
    scale: float            # cumulative scale along the path
    depth: int
    parent_id: Optional[str]
    joint: Optional[Joint]  # the joint that connects it to its parent (None for root)
    rel_pos: Vec3           # body frame relative to parent body frame
    rel_quat: Quat
    world_pos: Vec3
    world_quat: Quat
    sites: list[SiteInstance] = field(default_factory=list)

    def site(self, idx: int) -> SiteInstance:
        return self.sites[idx]


@dataclass
class Waypoint:
    body_id: str
    site_idx: int
    world_pos: Vec3


@dataclass
class MuscleRoute:
    id: str           # instance id, e.g. "hip_flexor#1"
    gene_id: str
    waypoints: list[Waypoint]
    fmax: float
    pcsa_cm2: float
    optimal_fiber_len: float
    tendon_slack_len: float
    pennation: float
    vmax: float
    fiber_type: float
    activation_cost: float

    def rest_length(self) -> float:
        L = 0.0
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            L += float(np.linalg.norm(np.array(a.world_pos) - np.array(b.world_pos)))
        return L


@dataclass
class EdgeInstance:
    parent_id: str
    parent_type: str
    child_id: str
    child_type: str
    edge: object  # the ConnectionEdge that produced it


@dataclass
class Morphology:
    bodies: list[Body]
    muscles: list[MuscleRoute]
    root_id: str
    genome_hash: str
    edges: list[EdgeInstance] = field(default_factory=list)

    @property
    def body_count(self) -> int:
        return len(self.bodies)

    def get_body(self, body_id: str) -> Optional[Body]:
        for b in self.bodies:
            if b.id == body_id:
                return b
        return None

    def total_mass(self) -> float:
        return sum(_volume(b.shape, b.dims) * b.density for b in self.bodies)

    def summary(self) -> str:
        lines = [f"Morphology({self.body_count} bodies, {len(self.muscles)} muscles, "
                 f"mass={self.total_mass():.2f} kg, root={self.root_id})"]
        for b in self.bodies:
            p = "" if b.parent_id is None else f" <- {b.parent_id}"
            lines.append(f"  {b.id:12s} {b.part_id:10s} d{b.depth} "
                         f"pos=({b.world_pos[0]:+.3f},{b.world_pos[1]:+.3f},{b.world_pos[2]:+.3f}){p}")
        return "\n".join(lines)


def _volume(shape: Shape, dims: dict[str, float]) -> float:
    if shape is Shape.CAPSULE:  # cylinder + two hemispherical caps
        r, L = dims["radius"], dims["length"]
        return math.pi * r * r * L + (4.0 / 3.0) * math.pi * r ** 3
    if shape is Shape.BOX:      # dims are half-extents
        return 8.0 * dims["x"] * dims["y"] * dims["z"]
    if shape is Shape.SPHERE:
        return (4.0 / 3.0) * math.pi * dims["radius"] ** 3
    if shape is Shape.ELLIPSOID:
        return (4.0 / 3.0) * math.pi * dims["x"] * dims["y"] * dims["z"]
    raise ValueError(f"unknown shape {shape}")


# --------------------------------------------------------------------------- #
# the traversal                                                               #
# --------------------------------------------------------------------------- #
class _Builder:
    def __init__(self, genome: Genome):
        self.g = genome
        self.adj: dict[str, list] = {}
        for e in genome.edges:
            self.adj.setdefault(e.parent_part, []).append(e)
        self.bodies: list[Body] = []
        self.edges: list[EdgeInstance] = []
        self._n = 0

    def _new_id(self, part_id: str) -> str:
        self._n += 1
        return f"{part_id}#{self._n}"

    def build(self) -> Morphology:
        root_type = self.g.get_part(self.g.root)
        root = self._make_root(root_type)
        self.bodies.append(root)
        # root is treated as a non-terminus so terminal_only structures hang off
        # chain tips, not the base; created_edge=None means "expand every edge".
        self._expand(root_type, root, depth=0, path={root_type.id: 1},
                     is_terminus=False, created_edge=None)
        muscles = self._build_muscles()
        return Morphology(self.bodies, muscles, root_id=root.id,
                          genome_hash=self.g.hash(), edges=self.edges)

    def _expand(self, ptype: PartNode, pbody: Body, depth: int,
                path: dict[str, int], is_terminus: bool, created_edge) -> None:
        from ..encoding.genome import SymmetryKind
        for e in self.adj.get(ptype.id, []):
            if e is created_edge:           # don't re-build the chain that made us
                continue
            if e.terminal_only and not is_terminus:
                continue
            if e.symmetry is SymmetryKind.RADIAL:
                n = e.symmetry_count
                for k in range(n):
                    self._grow_chain(e, pbody, ptype, depth, path,
                                     rot_z=2.0 * math.pi * k / n)
            elif e.symmetry is SymmetryKind.BILATERAL:
                self._grow_chain(e, pbody, ptype, depth, path)          # copy 0
                start = len(self.bodies)
                self._grow_chain(e, pbody, ptype, depth, path)          # copy 1 (nominal)
                self._reflect_slice(start, pbody)                       # -> mirror it
            else:
                self._grow_chain(e, pbody, ptype, depth, path)

    def _grow_chain(self, edge, parent_body: Body, parent_type: PartNode,
                    depth: int, path: dict[str, int], rot_z: float = 0.0) -> None:
        chain: list[tuple[Body, PartNode, int, dict[str, int]]] = []
        pb, pt, d, pc = parent_body, parent_type, depth, dict(path)
        ctype = self.g.get_part(edge.child_part)
        for step in range(edge.recursion_count):
            if d + 1 > self.g.max_depth:
                break
            if pc.get(ctype.id, 0) + 1 > ctype.recursion_limit:
                break
            if not (0 <= edge.site_idx < len(pt.sites)):
                break  # cannot attach: chaining site missing on this parent type
            # rot_z (radial) only rotates the whole limb's placement at its base
            cbody = self._attach(pb, pt, edge, ctype, d + 1,
                                 rot_z=rot_z if step == 0 else 0.0)
            self.bodies.append(cbody)
            self.edges.append(EdgeInstance(pb.id, pt.id, cbody.id, ctype.id, edge))
            npc = dict(pc)
            npc[ctype.id] = npc.get(ctype.id, 0) + 1
            chain.append((cbody, ctype, d + 1, npc))
            pb, pt, d, pc = cbody, ctype, d + 1, npc
        # expand each chain element's *other* edges; only the tip is a terminus
        last = len(chain) - 1
        for i, (cb, ct, cd, cpath) in enumerate(chain):
            self._expand(ct, cb, cd, cpath, is_terminus=(i == last), created_edge=edge)

    def _make_root(self, root_type: PartNode) -> Body:
        wp = np.zeros(3)
        wq = np.array([1.0, 0.0, 0.0, 0.0])
        return Body(
            id=self._new_id(root_type.id), part_id=root_type.id, shape=root_type.shape,
            dims=dict(root_type.dims), density=root_type.density, scale=1.0, depth=0,
            parent_id=None, joint=None, rel_pos=(0.0, 0.0, 0.0),
            rel_quat=(1.0, 0.0, 0.0, 0.0), world_pos=_t(wp), world_quat=_t(wq),
            sites=_site_instances(wp, wq, 1.0, root_type),
        )

    def _attach(self, pbody: Body, ptype: PartNode, edge, ctype: PartNode,
                depth: int, rot_z: float = 0.0) -> Body:
        site = ptype.sites[edge.site_idx]
        site_pos = np.array(site.pos)
        site_quat = _qnorm(np.array(site.quat))
        edge_pos = np.array(edge.pos)
        edge_quat = _qnorm(np.array(edge.quat))

        # radial symmetry: rotate the child's placement about the site's z-axis
        if rot_z != 0.0:
            rz = _rotz(rot_z)
            edge_pos = _qrot(rz, edge_pos)
            edge_quat = _qnorm(_qmul(rz, edge_quat))

        # local (parent-frame) transform = site ∘ edge
        local_pos = site_pos + _qrot(site_quat, edge_pos)
        local_quat = _qnorm(_qmul(site_quat, edge_quat))

        pscale = pbody.scale
        rel_pos = pscale * local_pos
        pw_pos = np.array(pbody.world_pos)
        pw_quat = np.array(pbody.world_quat)
        world_pos = pw_pos + _qrot(pw_quat, pscale * local_pos)
        world_quat = _qnorm(_qmul(pw_quat, local_quat))

        cscale = pscale * edge.scale
        dims = {k: v * cscale for k, v in ctype.dims.items()}
        return Body(
            id=self._new_id(ctype.id), part_id=ctype.id, shape=ctype.shape, dims=dims,
            density=ctype.density, scale=cscale, depth=depth, parent_id=pbody.id,
            joint=edge.joint, rel_pos=_t(rel_pos), rel_quat=_t(local_quat),
            world_pos=_t(world_pos), world_quat=_t(world_quat),
            sites=_site_instances(world_pos, world_quat, cscale, ctype),
        )

    def _reflect_slice(self, start: int, base_parent: Body) -> None:
        """Reflect bodies[start:] across the genome symmetry plane, in place.

        Reflects world transforms (positions + orientations) and site transforms,
        flips joint axes, then recomputes parent-relative transforms from the
        reflected world. Bodies are in pre-order, so a parent inside the slice is
        already reflected by the time its child is processed; the subtree root's
        parent (`base_parent`) is outside the slice and stays put.
        """
        ax = _PLANE_NORMAL.get(self.g.symmetry_plane, 1)
        by_id = {b.id: b for b in self.bodies}

        for b in self.bodies[start:]:                       # reflect world frames
            b.world_pos = _t(_reflect_vec(b.world_pos, ax))
            b.world_quat = _t(_qnorm(_mirror_quat(b.world_quat, ax)))
            for s in b.sites:
                s.world_pos = _t(_reflect_vec(s.world_pos, ax))
                s.world_quat = _t(_qnorm(_mirror_quat(s.world_quat, ax)))
            if b.joint is not None:                          # flip joint axis
                b.joint = Joint(type=b.joint.type, axis=_t(_reflect_vec(b.joint.axis, ax)),
                                range=b.joint.range, stiffness=b.joint.stiffness,
                                damping=b.joint.damping)

        for b in self.bodies[start:]:                       # rel transforms <- world
            parent = by_id[b.parent_id]
            invq = _qconj(np.array(parent.world_quat))
            b.rel_pos = _t(_qrot(invq, np.array(b.world_pos) - np.array(parent.world_pos)))
            b.rel_quat = _t(_qnorm(_qmul(invq, np.array(b.world_quat))))
            binv = _qconj(np.array(b.world_quat))
            for s in b.sites:
                s.rel_pos = _t(_qrot(binv, np.array(s.world_pos) - np.array(b.world_pos)))
                s.rel_quat = _t(_qnorm(_qmul(binv, np.array(s.world_quat))))

    # -- muscles ------------------------------------------------------------
    def _build_muscles(self) -> list[MuscleRoute]:
        by_id = {b.id: b for b in self.bodies}
        routes: list[MuscleRoute] = []
        for gene in self.g.muscles:
            otype, oidx = gene.origin
            itype, iidx = gene.insertion
            k = 0
            for ei in self.edges:
                pair = {ei.parent_type, ei.child_type}
                if pair != {otype, itype} and not (otype == itype == ei.parent_type == ei.child_type):
                    continue
                # assign which endpoint carries origin vs insertion
                if otype == itype:
                    o_body, i_body = ei.parent_id, ei.child_id
                elif ei.parent_type == otype:
                    o_body, i_body = ei.parent_id, ei.child_id
                else:
                    o_body, i_body = ei.child_id, ei.parent_id

                refs: list[tuple[str, int]] = [(o_body, oidx)]
                for rtype, ridx in gene.route_sites:
                    if rtype == otype:
                        refs.append((o_body, ridx))
                    elif rtype == itype:
                        refs.append((i_body, ridx))
                    # routes through other types deferred to Stage 1.6
                refs.append((i_body, iidx))

                k += 1
                waypoints = [
                    Waypoint(bid, sidx, by_id[bid].site(sidx).world_pos)
                    for bid, sidx in refs
                ]
                routes.append(MuscleRoute(
                    id=f"{gene.id}#{k}", gene_id=gene.id, waypoints=waypoints,
                    fmax=gene.fmax, pcsa_cm2=gene.pcsa_cm2,
                    optimal_fiber_len=gene.optimal_fiber_len,
                    tendon_slack_len=gene.tendon_slack_len, pennation=gene.pennation,
                    vmax=gene.vmax, fiber_type=gene.fiber_type,
                    activation_cost=gene.activation_cost,
                ))
        return routes


def _site_instances(world_pos: np.ndarray, world_quat: np.ndarray, scale: float,
                    part: PartNode) -> list[SiteInstance]:
    out: list[SiteInstance] = []
    wq = _qnorm(world_quat)
    for idx, s in enumerate(part.sites):
        local = scale * np.array(s.pos)
        wpos = world_pos + _qrot(wq, local)
        wquat = _qnorm(_qmul(wq, np.array(s.quat)))
        out.append(SiteInstance(
            name=s.name, idx=idx, rel_pos=_t(local), rel_quat=_t(_qnorm(np.array(s.quat))),
            world_pos=_t(wpos), world_quat=_t(wquat),
        ))
    return out


def develop(genome: Genome) -> Morphology:
    """Expand a (shape-valid) genome into a concrete Morphology."""
    genome.assert_valid()
    return _Builder(genome).build()
