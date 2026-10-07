"""Recursive body-graph generative genome (ROADMAP Stage 1.1).

A `Genome` is a directed graph of **part types** (`PartNode`) connected by
**rules** (`ConnectionEdge`) that say "attach child-type C to a site on parent-
type P, with this transform + joint, repeated `recursion_count` times, possibly
mirrored/duplicated by `symmetry`." Muscles (`MuscleGene`) span sites on parts.

Morphogenesis (Stage 1.3) *expands* this compact graph into a concrete body:
- A self-referential edge (child_part == parent_part) grows a chain; each part
  type's `recursion_limit` and the genome-global `max_depth` terminate growth.
- `symmetry=BILATERAL` mirrors the attached subtree across `Genome.symmetry_plane`
  (left/right limbs from one rule); `symmetry=RADIAL` makes `symmetry_count`
  rotated copies.

Stage 1.1 only defines and *validates the shape* of the genome (no MuJoCo, no
expansion). `validate_shape()` returns a list of structural problems — empty means
well-formed. The morphogenesis stages rely on a genome having passed this check.

Coordinate conventions: positions are `Vec3` (x, y, z) in metres in the local
frame; orientations are unit quaternions `Quat` as (w, x, y, z).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import hashlib
import json
import math

# Bump when the serialized genome layout changes; recorded in to_dict() output.
SCHEMA_VERSION = 1

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (w, x, y, z)

# Specific tension used to derive a muscle's max isometric force from its PCSA.
# 25 N/cm^2 sits inside the established 20-35 N/cm^2 range [EVIDENCE]; kept
# consistent with the Stage-0 scaffold (biomes/iron_zone.py).
SPECIFIC_TENSION_N_PER_CM2 = 25.0


class GenomeValidationError(ValueError):
    """Raised by Genome.assert_valid() when the genome is structurally malformed."""


# --------------------------------------------------------------------------- #
# Enums                                                                        #
# --------------------------------------------------------------------------- #
class Shape(str, Enum):
    CAPSULE = "capsule"
    BOX = "box"
    SPHERE = "sphere"
    ELLIPSOID = "ellipsoid"


# Required dimension keys per shape (all in metres; box/ellipsoid are half-extents
# / radii along local x, y, z; capsule is radius + cylinder length along local z).
SHAPE_DIMS: dict[Shape, tuple[str, ...]] = {
    Shape.CAPSULE: ("radius", "length"),
    Shape.BOX: ("x", "y", "z"),
    Shape.SPHERE: ("radius",),
    Shape.ELLIPSOID: ("x", "y", "z"),
}


class JointType(str, Enum):
    HINGE = "hinge"   # 1 DOF rotation about `axis`
    BALL = "ball"     # 3 DOF rotation
    SLIDE = "slide"   # 1 DOF translation along `axis`
    FIXED = "fixed"   # rigid weld (no DOF)


class SymmetryKind(str, Enum):
    NONE = "none"
    BILATERAL = "bilateral"  # one mirrored copy across the genome symmetry plane
    RADIAL = "radial"        # `symmetry_count` copies rotated about the site axis


# Valid genome-level mirror planes for BILATERAL symmetry; the value names the
# plane the body is reflected across (so "xz" flips the y coordinate -> L/R).
SYMMETRY_PLANES = ("xy", "xz", "yz")


# --------------------------------------------------------------------------- #
# Leaf structures                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class AttachmentSite:
    """A named frame on a part where children attach and muscles anchor."""
    name: str
    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = (1.0, 0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        self.pos = _as_vec3(self.pos)
        self.quat = _as_quat(self.quat)


@dataclass
class Joint:
    """The articulation introduced by a connection edge."""
    type: JointType = JointType.HINGE
    axis: Vec3 = (0.0, 0.0, 1.0)
    range: tuple[float, float] = (-math.pi / 2, math.pi / 2)  # radians (rotational) / m (slide)
    stiffness: float = 0.0
    damping: float = 0.1

    def __post_init__(self) -> None:
        if isinstance(self.type, str):
            self.type = JointType(self.type)
        self.axis = _as_vec3(self.axis)
        self.range = (float(self.range[0]), float(self.range[1]))
        self.stiffness = float(self.stiffness)
        self.damping = float(self.damping)


@dataclass
class PartNode:
    """A reusable body-part *type*. May be instantiated many times by edges/recursion."""
    id: str
    shape: Shape = Shape.CAPSULE
    dims: dict[str, float] = field(default_factory=lambda: {"radius": 0.05, "length": 0.30})
    density: float = 1000.0  # kg/m^3 (~soft tissue)
    sites: list[AttachmentSite] = field(default_factory=list)
    recursion_limit: int = 1  # max self-recursive repeats of this type along a chain

    def __post_init__(self) -> None:
        if isinstance(self.shape, str):
            self.shape = Shape(self.shape)
        self.dims = {str(k): float(v) for k, v in self.dims.items()}
        self.density = float(self.density)
        self.recursion_limit = int(self.recursion_limit)


@dataclass
class ConnectionEdge:
    """A growth rule: attach `child_part` to `parent_part` at site `site_idx`."""
    parent_part: str
    child_part: str
    site_idx: int
    pos: Vec3 = (0.0, 0.0, 0.0)        # child offset relative to the parent's site frame
    quat: Quat = (1.0, 0.0, 0.0, 0.0)  # child orientation relative to the site frame
    scale: float = 1.0                 # uniform scale applied down the child subtree
    joint: Joint = field(default_factory=Joint)
    recursion_count: int = 1           # number of repeats of this edge (chain length)
    terminal_only: bool = False        # if True, only instantiate at the recursion terminus
    symmetry: SymmetryKind = SymmetryKind.NONE
    symmetry_count: int = 1            # number of copies for RADIAL symmetry

    def __post_init__(self) -> None:
        if isinstance(self.symmetry, str):
            self.symmetry = SymmetryKind(self.symmetry)
        self.site_idx = int(self.site_idx)
        self.pos = _as_vec3(self.pos)
        self.quat = _as_quat(self.quat)
        self.scale = float(self.scale)
        self.recursion_count = int(self.recursion_count)
        self.symmetry_count = int(self.symmetry_count)
        self.terminal_only = bool(self.terminal_only)


@dataclass
class MuscleGene:
    """An actuated musculotendon unit spanning two sites (optionally via routes).

    Hill-type parameters; `fmax` (max isometric force) is derived from PCSA so the
    genome stays physically grounded and minimal.
    """
    id: str
    origin: tuple[str, int]       # (part_id, site_idx)
    insertion: tuple[str, int]    # (part_id, site_idx)
    route_sites: list[tuple[str, int]] = field(default_factory=list)
    pcsa_cm2: float = 5.0
    optimal_fiber_len: float = 0.10   # m
    tendon_slack_len: float = 0.10    # m
    pennation: float = 0.0            # radians
    vmax: float = 10.0                # optimal-fiber-lengths / s
    fiber_type: float = 0.5           # 0 = slow-oxidative .. 1 = fast-glycolytic
    activation_cost: float = 1.0      # metabolic/control cost multiplier

    def __post_init__(self) -> None:
        # normalize references to plain (str, int) tuples
        self.origin = (str(self.origin[0]), int(self.origin[1]))
        self.insertion = (str(self.insertion[0]), int(self.insertion[1]))
        self.route_sites = [(str(p), int(i)) for p, i in self.route_sites]
        # float-coerce Hill params so int/float inputs hash identically
        self.pcsa_cm2 = float(self.pcsa_cm2)
        self.optimal_fiber_len = float(self.optimal_fiber_len)
        self.tendon_slack_len = float(self.tendon_slack_len)
        self.pennation = float(self.pennation)
        self.vmax = float(self.vmax)
        self.fiber_type = float(self.fiber_type)
        self.activation_cost = float(self.activation_cost)

    @property
    def fmax(self) -> float:
        """Max isometric force (N) from PCSA and specific tension."""
        return self.pcsa_cm2 * SPECIFIC_TENSION_N_PER_CM2


# --------------------------------------------------------------------------- #
# Genome                                                                       #
# --------------------------------------------------------------------------- #
@dataclass
class Genome:
    root: str
    parts: list[PartNode] = field(default_factory=list)
    edges: list[ConnectionEdge] = field(default_factory=list)
    muscles: list[MuscleGene] = field(default_factory=list)
    max_depth: int = 8           # global hard cap on morphogenesis depth
    symmetry_plane: str = "xz"   # mirror plane for BILATERAL symmetry

    # -- indexing -----------------------------------------------------------
    def get_part(self, part_id: str) -> Optional[PartNode]:
        for p in self.parts:
            if p.id == part_id:
                return p
        return None

    @property
    def part_ids(self) -> list[str]:
        return [p.id for p in self.parts]

    def unreachable_parts(self) -> list[str]:
        """Part types not reachable from `root` by following edges (diagnostic;
        not a structural error — mutation may transiently orphan a type)."""
        if self.get_part(self.root) is None:
            return [p.id for p in self.parts]
        adj: dict[str, list[str]] = {}
        for e in self.edges:
            adj.setdefault(e.parent_part, []).append(e.child_part)
        seen, stack = set(), [self.root]
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(adj.get(cur, []))
        return [p.id for p in self.parts if p.id not in seen]

    # -- validation ---------------------------------------------------------
    def validate_shape(self) -> list[str]:
        """Return a list of structural problems; empty list == well-formed.

        Catches: empty/duplicate parts, missing root, bad geometry/density,
        unknown part references and dangling site indices in edges and muscles,
        degenerate joints, out-of-range parameters, and bad quaternions/vectors.
        """
        problems: list[str] = []

        # --- parts ---------------------------------------------------------
        if not self.parts:
            problems.append("genome has no parts")
        ids = self.part_ids
        dupes = {i for i in ids if ids.count(i) > 1}
        if dupes:
            problems.append(f"duplicate part ids: {sorted(dupes)}")
        id_set = set(ids)
        if self.root not in id_set:
            problems.append(f"root '{self.root}' is not a defined part")

        for p in self.parts:
            required = SHAPE_DIMS[p.shape]
            if set(p.dims) != set(required):
                problems.append(
                    f"part '{p.id}' shape {p.shape.value} needs dims {required}, "
                    f"got {sorted(p.dims)}")
            else:
                for k in required:
                    if not (p.dims[k] > 0):
                        problems.append(f"part '{p.id}' dim '{k}' must be > 0 (got {p.dims[k]})")
            if not (p.density > 0):
                problems.append(f"part '{p.id}' density must be > 0 (got {p.density})")
            if p.recursion_limit < 1:
                problems.append(f"part '{p.id}' recursion_limit must be >= 1")
            site_names = [s.name for s in p.sites]
            if len(set(site_names)) != len(site_names):
                problems.append(f"part '{p.id}' has duplicate site names")
            for s in p.sites:
                problems += _check_vec3(f"part '{p.id}' site '{s.name}' pos", s.pos)
                problems += _check_quat(f"part '{p.id}' site '{s.name}' quat", s.quat)

        # --- edges ---------------------------------------------------------
        for n, e in enumerate(self.edges):
            tag = f"edge[{n}] {e.parent_part}->{e.child_part}"
            parent = self.get_part(e.parent_part)
            if parent is None:
                problems.append(f"{tag}: unknown parent part '{e.parent_part}'")
            if e.child_part not in id_set:
                problems.append(f"{tag}: unknown child part '{e.child_part}'")
            if parent is not None:
                if not (0 <= e.site_idx < len(parent.sites)):
                    problems.append(
                        f"{tag}: dangling site_idx {e.site_idx} "
                        f"(parent '{parent.id}' has {len(parent.sites)} sites)")
            if e.recursion_count < 1:
                problems.append(f"{tag}: recursion_count must be >= 1")
            if e.scale <= 0:
                problems.append(f"{tag}: scale must be > 0 (got {e.scale})")
            if e.symmetry is SymmetryKind.RADIAL and e.symmetry_count < 2:
                problems.append(f"{tag}: RADIAL symmetry needs symmetry_count >= 2")
            if e.symmetry_count < 1:
                problems.append(f"{tag}: symmetry_count must be >= 1")
            lo, hi = e.joint.range
            if lo > hi:
                problems.append(f"{tag}: joint range lo {lo} > hi {hi}")
            if e.joint.type in (JointType.HINGE, JointType.SLIDE):
                if _norm(e.joint.axis) == 0.0:
                    problems.append(f"{tag}: {e.joint.type.value} joint needs a non-zero axis")
            problems += _check_vec3(f"{tag} pos", e.pos)
            problems += _check_quat(f"{tag} quat", e.quat)

        # --- muscles -------------------------------------------------------
        mids = [m.id for m in self.muscles]
        mdupes = {i for i in mids if mids.count(i) > 1}
        if mdupes:
            problems.append(f"duplicate muscle ids: {sorted(mdupes)}")
        for m in self.muscles:
            problems += self._check_site_ref(f"muscle '{m.id}' origin", m.origin, id_set)
            problems += self._check_site_ref(f"muscle '{m.id}' insertion", m.insertion, id_set)
            for k, r in enumerate(m.route_sites):
                problems += self._check_site_ref(f"muscle '{m.id}' route[{k}]", r, id_set)
            if m.origin == m.insertion:
                problems.append(f"muscle '{m.id}' origin and insertion are the same site")
            if not (m.pcsa_cm2 > 0):
                problems.append(f"muscle '{m.id}' pcsa_cm2 must be > 0")
            if not (m.optimal_fiber_len > 0):
                problems.append(f"muscle '{m.id}' optimal_fiber_len must be > 0")
            if not (m.tendon_slack_len > 0):
                problems.append(f"muscle '{m.id}' tendon_slack_len must be > 0")
            if not (m.vmax > 0):
                problems.append(f"muscle '{m.id}' vmax must be > 0")
            if not (0.0 <= m.fiber_type <= 1.0):
                problems.append(f"muscle '{m.id}' fiber_type must be in [0,1] (got {m.fiber_type})")

        # --- genome-level --------------------------------------------------
        if self.max_depth < 1:
            problems.append("max_depth must be >= 1")
        if self.symmetry_plane not in SYMMETRY_PLANES:
            problems.append(f"symmetry_plane must be one of {SYMMETRY_PLANES} (got '{self.symmetry_plane}')")

        return problems

    def _check_site_ref(self, tag: str, ref: tuple[str, int], id_set: set[str]) -> list[str]:
        part_id, site_idx = ref
        if part_id not in id_set:
            return [f"{tag}: unknown part '{part_id}'"]
        part = self.get_part(part_id)
        if part is None or not (0 <= site_idx < len(part.sites)):
            n = 0 if part is None else len(part.sites)
            return [f"{tag}: dangling site_idx {site_idx} (part '{part_id}' has {n} sites)"]
        return []

    # -- serialization & hashing -------------------------------------------
    def to_dict(self) -> dict:
        """Plain-dict form (JSON-ready); enums become their string values."""
        return {
            "_schema": SCHEMA_VERSION,
            "root": self.root,
            "max_depth": self.max_depth,
            "symmetry_plane": self.symmetry_plane,
            "parts": [_part_to_dict(p) for p in self.parts],
            "edges": [_edge_to_dict(e) for e in self.edges],
            "muscles": [_muscle_to_dict(m) for m in self.muscles],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(
            root=d["root"],
            parts=[_part_from_dict(p) for p in d.get("parts", [])],
            edges=[_edge_from_dict(e) for e in d.get("edges", [])],
            muscles=[_muscle_from_dict(m) for m in d.get("muscles", [])],
            max_depth=int(d.get("max_depth", 8)),
            symmetry_plane=d.get("symmetry_plane", "xz"),
        )

    def canonical_json(self) -> str:
        """Deterministic, compact JSON (sorted keys) — the basis for the hash."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Human-readable JSON (sorted keys). Use canonical_json() for hashing."""
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "Genome":
        return cls.from_dict(json.loads(s))

    def hash(self) -> str:
        """SHA-256 of the canonical JSON. Equal genomes share a hash."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def is_valid_shape(self) -> bool:
        return not self.validate_shape()

    def assert_valid(self) -> "Genome":
        problems = self.validate_shape()
        if problems:
            raise GenomeValidationError(
                f"{len(problems)} problem(s):\n  - " + "\n  - ".join(problems))
        return self


# --------------------------------------------------------------------------- #
# small geometry validators                                                    #
# --------------------------------------------------------------------------- #
def _norm(v) -> float:
    return math.sqrt(sum(float(c) ** 2 for c in v)) if v is not None else 0.0


def _check_vec3(tag: str, v) -> list[str]:
    if not (isinstance(v, (tuple, list)) and len(v) == 3):
        return [f"{tag} must be a 3-vector"]
    if any(not math.isfinite(float(c)) for c in v):
        return [f"{tag} has non-finite components"]
    return []


def _check_quat(tag: str, q) -> list[str]:
    if not (isinstance(q, (tuple, list)) and len(q) == 4):
        return [f"{tag} must be a 4-quaternion (w,x,y,z)"]
    if any(not math.isfinite(float(c)) for c in q):
        return [f"{tag} has non-finite components"]
    if _norm(q) == 0.0:
        return [f"{tag} has zero norm"]
    return []


def _as_vec3(v) -> Vec3:
    """Coerce a 3-sequence to a tuple of floats (so code/JSON inputs compare equal)."""
    return (float(v[0]), float(v[1]), float(v[2]))


def _as_quat(q) -> Quat:
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


# --------------------------------------------------------------------------- #
# serialization converters (paired to_dict / from_dict, used by Genome)        #
# --------------------------------------------------------------------------- #
def _site_to_dict(s: AttachmentSite) -> dict:
    return {"name": s.name, "pos": list(s.pos), "quat": list(s.quat)}


def _site_from_dict(d: dict) -> AttachmentSite:
    return AttachmentSite(name=d["name"],
                          pos=tuple(d.get("pos", (0.0, 0.0, 0.0))),
                          quat=tuple(d.get("quat", (1.0, 0.0, 0.0, 0.0))))


def _joint_to_dict(j: Joint) -> dict:
    return {"type": j.type.value, "axis": list(j.axis), "range": list(j.range),
            "stiffness": j.stiffness, "damping": j.damping}


def _joint_from_dict(d: dict) -> Joint:
    return Joint(type=d.get("type", "hinge"),
                 axis=tuple(d.get("axis", (0.0, 0.0, 1.0))),
                 range=tuple(d.get("range", (-math.pi / 2, math.pi / 2))),
                 stiffness=d.get("stiffness", 0.0), damping=d.get("damping", 0.1))


def _part_to_dict(p: PartNode) -> dict:
    return {"id": p.id, "shape": p.shape.value, "dims": dict(p.dims),
            "density": p.density, "recursion_limit": p.recursion_limit,
            "sites": [_site_to_dict(s) for s in p.sites]}


def _part_from_dict(d: dict) -> PartNode:
    return PartNode(id=d["id"], shape=d.get("shape", "capsule"),
                    dims=dict(d.get("dims", {})), density=d.get("density", 1000.0),
                    sites=[_site_from_dict(s) for s in d.get("sites", [])],
                    recursion_limit=d.get("recursion_limit", 1))


def _edge_to_dict(e: ConnectionEdge) -> dict:
    return {"parent_part": e.parent_part, "child_part": e.child_part,
            "site_idx": e.site_idx, "pos": list(e.pos), "quat": list(e.quat),
            "scale": e.scale, "joint": _joint_to_dict(e.joint),
            "recursion_count": e.recursion_count, "terminal_only": e.terminal_only,
            "symmetry": e.symmetry.value, "symmetry_count": e.symmetry_count}


def _edge_from_dict(d: dict) -> ConnectionEdge:
    return ConnectionEdge(
        parent_part=d["parent_part"], child_part=d["child_part"],
        site_idx=d["site_idx"], pos=tuple(d.get("pos", (0.0, 0.0, 0.0))),
        quat=tuple(d.get("quat", (1.0, 0.0, 0.0, 0.0))), scale=d.get("scale", 1.0),
        joint=_joint_from_dict(d.get("joint", {})),
        recursion_count=d.get("recursion_count", 1),
        terminal_only=d.get("terminal_only", False),
        symmetry=d.get("symmetry", "none"), symmetry_count=d.get("symmetry_count", 1))


def _muscle_to_dict(m: MuscleGene) -> dict:
    return {"id": m.id, "origin": list(m.origin), "insertion": list(m.insertion),
            "route_sites": [list(r) for r in m.route_sites],
            "pcsa_cm2": m.pcsa_cm2, "optimal_fiber_len": m.optimal_fiber_len,
            "tendon_slack_len": m.tendon_slack_len, "pennation": m.pennation,
            "vmax": m.vmax, "fiber_type": m.fiber_type,
            "activation_cost": m.activation_cost}


def _muscle_from_dict(d: dict) -> MuscleGene:
    return MuscleGene(
        id=d["id"], origin=tuple(d["origin"]), insertion=tuple(d["insertion"]),
        route_sites=[tuple(r) for r in d.get("route_sites", [])],
        pcsa_cm2=d.get("pcsa_cm2", 5.0),
        optimal_fiber_len=d.get("optimal_fiber_len", 0.10),
        tendon_slack_len=d.get("tendon_slack_len", 0.10),
        pennation=d.get("pennation", 0.0), vmax=d.get("vmax", 10.0),
        fiber_type=d.get("fiber_type", 0.5), activation_cost=d.get("activation_cost", 1.0))
