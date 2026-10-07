"""MORPHOLOGY -> SCENE PLAN (ROADMAP Stage 10.1, bpy-free core).

A `ScenePlan` is a flat, JSON-serializable description of a creature's geometry --
one `GeomSpec` per body (shape + size + world transform) and one `MuscleSpec` per
muscle (the waypoint polyline). It is extracted straight from a developed
`Morphology` (the same world frames the MuJoCo compiler uses), with NO Blender
dependency, so:

  * the geometry extraction is unit-testable in plain Python (no bpy), and
  * the plan dict can be shipped into a Blender session (MCP or headless
    `blender --background --python`) where `blender_build` turns it into objects.

Geometry conventions match `encoding.genome` / `morphogenesis.to_mujoco`:
capsule = radius + cylinder `length` along local z; box = half-extents (x,y,z);
sphere = radius; ellipsoid = radii (x,y,z). Quaternions are MuJoCo/Blender order
(w, x, y, z).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from ..morphogenesis import develop


@dataclass
class GeomSpec:
    name: str                       # unique object name (the body instance id)
    shape: str                      # 'capsule' | 'box' | 'sphere' | 'ellipsoid'
    dims: dict                      # capsule {radius,length}; box/ellipsoid {x,y,z}; sphere {radius}
    pos: tuple                      # world position (x, y, z)
    quat: tuple                     # world orientation (w, x, y, z)
    depth: int                      # recursion depth (root = 0) -- drives colour
    part_id: str                    # the part *type* this body instantiates
    parent: Optional[str] = None    # parent body id (None for root)


@dataclass
class MuscleSpec:
    name: str
    points: list                    # [(x,y,z), ...] waypoint world positions
    fmax: float


@dataclass
class ScenePlan:
    name: str
    geoms: list = field(default_factory=list)
    muscles: list = field(default_factory=list)

    def bounds(self):
        """(min_xyz, max_xyz) over all geom centres -- for camera framing."""
        if not self.geoms:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        xs = [g.pos for g in self.geoms]
        lo = tuple(min(p[i] for p in xs) for i in range(3))
        hi = tuple(max(p[i] for p in xs) for i in range(3))
        return lo, hi

    def to_dict(self) -> dict:
        return {"name": self.name,
                "geoms": [asdict(g) for g in self.geoms],
                "muscles": [asdict(m) for m in self.muscles]}

    @classmethod
    def from_dict(cls, d: dict) -> "ScenePlan":
        return cls(name=d["name"],
                   geoms=[GeomSpec(**g) for g in d["geoms"]],
                   muscles=[MuscleSpec(**m) for m in d["muscles"]])


def build_plan(morphology, *, name: str = "creature") -> ScenePlan:
    """Extract a `ScenePlan` from a developed `Morphology`."""
    geoms = []
    for b in morphology.bodies:
        geoms.append(GeomSpec(
            name=b.id,
            shape=b.shape.value if hasattr(b.shape, "value") else str(b.shape),
            dims=dict(b.dims),
            pos=tuple(float(v) for v in b.world_pos),
            quat=tuple(float(v) for v in b.world_quat),
            depth=int(b.depth),
            part_id=b.part_id,
            parent=b.parent_id,
        ))
    muscles = []
    for m in morphology.muscles:
        muscles.append(MuscleSpec(
            name=m.id,
            points=[tuple(float(v) for v in w.world_pos) for w in m.waypoints],
            fmax=float(m.fmax),
        ))
    return ScenePlan(name=name, geoms=geoms, muscles=muscles)


def plan_from_genome(genome, *, name: str = "creature") -> ScenePlan:
    """Develop a genome and extract its scene plan."""
    return build_plan(develop(genome), name=name)
