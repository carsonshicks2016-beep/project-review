"""Compile an engine-neutral `Morphology` into a MuJoCo model (ROADMAP Stage 1.5).

Uses the procedural `mujoco.MjSpec` API (mujoco >= 3.2) rather than hand-writing
MJCF XML. Each `Body` becomes a nested `<body>` placed by its parent-relative
transform; its articulating `Joint` becomes a `<joint>` at the body origin (which
is the attachment point); geometry + density drive MuJoCo's automatic inertia;
attachment sites become `<site>`s (used by Stage 1.6 to route muscle tendons).

The root body gets a free joint by default so the creature can move in the world.
Muscles/tendons are added in Stage 1.6, not here.
"""
from __future__ import annotations

from typing import Optional
import mujoco

from ..encoding.genome import Shape, JointType
from .develop import Morphology

_GEOM = {
    Shape.CAPSULE: mujoco.mjtGeom.mjGEOM_CAPSULE,
    Shape.BOX: mujoco.mjtGeom.mjGEOM_BOX,
    Shape.SPHERE: mujoco.mjtGeom.mjGEOM_SPHERE,
    Shape.ELLIPSOID: mujoco.mjtGeom.mjGEOM_ELLIPSOID,
}
_JNT = {
    JointType.HINGE: mujoco.mjtJoint.mjJNT_HINGE,
    JointType.SLIDE: mujoco.mjtJoint.mjJNT_SLIDE,
    JointType.BALL: mujoco.mjtJoint.mjJNT_BALL,
}

# Collision groups (contype/conaffinity bitmasks). Body geoms are in BODY; the
# floor is in WORLD. Two geoms collide iff (contype1 & conaffinity2) is nonzero
# either way round. Body geoms always collide with the floor; whether they
# collide with each other is toggled by `self_collide`. Parent-child pairs are
# additionally removed with explicit excludes (they overlap at the joint).
_BODY = 1
_WORLD = 2


def _geom_size(shape: Shape, dims: dict[str, float]) -> list[float]:
    if shape is Shape.CAPSULE:           # [radius, half-cylinder-length, _]
        return [dims["radius"], dims["length"] / 2.0, 0.0]
    if shape is Shape.BOX:               # half-extents
        return [dims["x"], dims["y"], dims["z"]]
    if shape is Shape.SPHERE:
        return [dims["radius"], 0.0, 0.0]
    if shape is Shape.ELLIPSOID:         # radii
        return [dims["x"], dims["y"], dims["z"]]
    raise ValueError(f"unknown shape {shape}")


def build_spec(morph: Morphology, *, add_floor: bool = False, free_root: bool = True,
               self_collide: bool = True, exclude_parent_child: bool = True,
               add_muscles: bool = True, name: str = "creature") -> mujoco.MjSpec:
    """Build (but do not compile) an MjSpec from a Morphology."""
    spec = mujoco.MjSpec()
    spec.modelname = name
    try:
        spec.compiler.autolimits = True   # setting a joint range auto-enables limits
        spec.compiler.degree = False      # our joint ranges/axes are in RADIANS
    except AttributeError:                # pragma: no cover - older API
        pass

    if add_floor:
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 1.0]      # infinite plane, 1 m grid spacing
        floor.pos = [0.0, 0.0, 0.0]
        floor.contype = _WORLD
        floor.conaffinity = _BODY

    mjb: dict[str, object] = {}
    for b in morph.bodies:
        parent = spec.worldbody if b.parent_id is None else mjb[b.parent_id]
        body = parent.add_body()
        body.name = b.id
        body.pos = list(b.rel_pos)
        body.quat = list(b.rel_quat)
        mjb[b.id] = body

        # joint connecting this body to its parent
        if b.parent_id is None:
            if free_root:
                body.add_freejoint()
        elif b.joint is not None and b.joint.type is not JointType.FIXED:
            j = body.add_joint()
            j.name = f"{b.id}:j"
            j.type = _JNT[b.joint.type]
            j.axis = list(b.joint.axis)
            if b.joint.type in (JointType.HINGE, JointType.SLIDE):
                j.range = [b.joint.range[0], b.joint.range[1]]
            elif b.joint.type is JointType.BALL:
                hi = max(abs(b.joint.range[0]), abs(b.joint.range[1]))
                if hi > 0:
                    j.range = [0.0, hi]
            # stiffness/damping are per-DOF 3-vectors in MjSpec (ball = 3 DOF)
            j.stiffness = [b.joint.stiffness] * 3
            j.damping = [b.joint.damping] * 3
        # FIXED / no joint -> body is welded rigidly to its parent

        # geometry (drives mass + inertia via density)
        g = body.add_geom()
        g.name = f"{b.id}:g"
        g.type = _GEOM[b.shape]
        g.size = _geom_size(b.shape, b.dims)
        g.density = b.density
        # always collide with the floor (WORLD); collide with other bodies (BODY)
        # only when self_collide is on. Parent-child pairs are excluded below.
        g.contype = _BODY
        g.conaffinity = (_BODY | _WORLD) if self_collide else _WORLD

        # attachment sites (muscle anchors)
        for s in b.sites:
            st = body.add_site()
            st.name = f"{b.id}:{s.name}"
            st.pos = list(s.rel_pos)
            st.quat = list(s.rel_quat)

    # --- exclude parent-child contacts (connected bodies overlap at the joint);
    # non-adjacent body pairs still collide, which Stage 6 uses to detect self-
    # intersection, and the floor still collides with everything.
    if exclude_parent_child:
        for b in morph.bodies:
            if b.parent_id is not None:
                ex = spec.add_exclude()
                ex.bodyname1 = b.parent_id
                ex.bodyname2 = b.id

    # --- muscles: spatial tendon through the waypoint sites + force actuator
    # v1: a general force actuator on the tendon (gear = Fmax, ctrl in [-1,1]).
    # v2 (later) will use MuJoCo's Hill `muscle` actuator (pull-only, force-
    # length-velocity) once length ranges are calibrated. Per-joint moment arms
    # are verified in tests/test_actuation.py.
    if add_muscles:
        for route in morph.muscles:
            tendon = spec.add_tendon()
            tendon.name = route.id
            tendon.width = 0.004
            tendon.rgba = [0.75, 0.12, 0.12, 1.0]
            for wp in route.waypoints:
                body = morph.get_body(wp.body_id)
                tendon.wrap_site(f"{wp.body_id}:{body.sites[wp.site_idx].name}")
            act = spec.add_actuator()
            act.name = f"{route.id}:act"
            act.trntype = mujoco.mjtTrn.mjTRN_TENDON
            act.target = route.id
            act.gear = [route.fmax, 0.0, 0.0, 0.0, 0.0, 0.0]
            act.ctrlrange = [-1.0, 1.0]

    return spec


def compile_morphology(morph: Morphology, **opts):
    """Build + compile. Returns (mjModel, MjSpec)."""
    spec = build_spec(morph, **opts)
    model = spec.compile()
    return model, spec


def export_xml(morph: Morphology, path: Optional[str] = None, **opts) -> str:
    """Return the MJCF XML (and optionally write it to `path`) for inspection."""
    spec = build_spec(morph, **opts)
    spec.compile()
    xml = spec.to_xml()
    if path:
        with open(path, "w") as f:
            f.write(xml)
    return xml
