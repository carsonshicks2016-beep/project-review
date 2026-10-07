"""SCENE PLAN / TRAJECTORY -> BLENDER (ROADMAP Stage 10.1-10.2, bpy layer).

Self-contained `bpy` builder. Each body becomes a per-body EMPTY at its world
transform with the geometry (capsule = cylinder + two end-spheres, box / sphere /
ellipsoid = scaled primitives) parented underneath in LOCAL coordinates -- so a body
is ONE transformable unit, and an animation only has to keyframe the empties.

  * `build_from_plan(plan)`        -- static creature (Stage 10.1).
  * `animate_trajectory(traj)`     -- keyframe the empties from a recorded rollout and
                                      render a locomotion video (Stage 10.2).

Imports only bpy / mathutils / math, so the whole file ships into a Blender session
(MCP `execute_blender_code`, or `blender -b -P blender_build.py -- file.json [out]`).
Quaternions are (w, x, y, z); capsule cylinders run along local z; box dims are
half-extents; ellipsoid dims are radii.
"""
import json
import math
import os
import sys

try:
    import bpy
    import mathutils
    _HAS_BPY = True
except Exception:        # noqa: BLE001 -- importable (for source-shipping) without Blender
    _HAS_BPY = False

_PALETTE = [(0.85, 0.30, 0.25), (0.30, 0.55, 0.85), (0.40, 0.75, 0.40),
            (0.85, 0.70, 0.25), (0.65, 0.40, 0.80), (0.30, 0.75, 0.75)]


def _material(name, rgb):
    m = bpy.data.materials.get(name)
    if m is not None:
        return m
    m = bpy.data.materials.new(name)
    m.diffuse_color = (rgb[0], rgb[1], rgb[2], 1.0)          # viewport solid shading
    m.use_nodes = True                                      # rendered colour
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (rgb[0], rgb[1], rgb[2], 1.0)
        if "Roughness" in bsdf.inputs:
            bsdf.inputs["Roughness"].default_value = 0.6
    return m


def _quat(q):
    return mathutils.Quaternion((q[0], q[1], q[2], q[3]))


def _world_matrix(pos, quat):
    return mathutils.Matrix.Translation(pos) @ _quat(quat).to_matrix().to_4x4()


def _child(obj, name, local, mat, empty, coll):
    obj.name = name
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.parent = empty
    obj.matrix_parent_inverse = mathutils.Matrix.Identity(4)
    obj.matrix_local = local
    for c in list(obj.users_collection):
        c.objects.unlink(obj)
    coll.objects.link(obj)


def _add_body(g, coll):
    """Create a per-body empty (the keyframe target) + its geometry children."""
    shape, dims = g["shape"], g["dims"]
    mat = _material(f"mat_d{g['depth']}", _PALETTE[g["depth"] % len(_PALETTE)])
    empty = bpy.data.objects.new(g["name"], None)           # the body frame
    empty.rotation_mode = "QUATERNION"
    empty.matrix_world = _world_matrix(tuple(g["pos"]), tuple(g["quat"]))
    coll.objects.link(empty)
    I = mathutils.Matrix.Identity(4)
    Tr = mathutils.Matrix.Translation

    if shape == "capsule":
        r, L = float(dims["radius"]), float(dims["length"])
        bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=L)
        _child(bpy.context.active_object, g["name"] + ".body", I, mat, empty, coll)
        for i, z in enumerate((L / 2.0, -L / 2.0)):
            bpy.ops.mesh.primitive_uv_sphere_add(radius=r)
            _child(bpy.context.active_object, f"{g['name']}.cap{i}", Tr((0, 0, z)),
                   mat, empty, coll)
    elif shape == "box":
        bpy.ops.mesh.primitive_cube_add(size=2.0)
        _child(bpy.context.active_object, g["name"] + ".body",
               mathutils.Matrix.Diagonal((dims["x"], dims["y"], dims["z"], 1.0)),
               mat, empty, coll)
    elif shape == "sphere":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=float(dims["radius"]))
        _child(bpy.context.active_object, g["name"] + ".body", I, mat, empty, coll)
    elif shape == "ellipsoid":
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0)
        _child(bpy.context.active_object, g["name"] + ".body",
               mathutils.Matrix.Diagonal((dims["x"], dims["y"], dims["z"], 1.0)),
               mat, empty, coll)
    return empty


def _add_muscle(m, coll):
    pts = m["points"]
    if len(pts) < 2:
        return None
    curve = bpy.data.curves.new(m["name"], "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = max(0.004, min(0.02, 0.004 + m["fmax"] / 4.0e4))
    sp = curve.splines.new("POLY")
    sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts):
        sp.points[i].co = (p[0], p[1], p[2], 1.0)
    obj = bpy.data.objects.new(m["name"], curve)
    obj.data.materials.append(_material("mat_muscle", (0.9, 0.1, 0.1)))
    coll.objects.link(obj)
    return obj


def _bounds(geoms):
    ps = [g["pos"] for g in geoms] or [(0, 0, 0)]
    lo = [min(p[i] for p in ps) for i in range(3)]
    hi = [max(p[i] for p in ps) for i in range(3)]
    ctr = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
    rad = max(0.5, max(hi[i] - lo[i] for i in range(3)))
    return ctr, rad


def _add_ground(z=0.0):
    bpy.ops.mesh.primitive_plane_add(size=40.0, location=(0, 0, z))
    bpy.context.active_object.data.materials.append(_material("mat_ground", (0.18, 0.18, 0.2)))


def _add_light_and_camera(geoms, *, track_span=0.0):
    ps = [g["pos"] for g in geoms] or [(0, 0, 0)]
    lo = [min(p[i] for p in ps) for i in range(3)]
    hi = [max(p[i] for p in ps) for i in range(3)]
    ctr = [(lo[i] + hi[i]) / 2.0 for i in range(3)]
    span = [hi[i] - lo[i] for i in range(3)]
    rad = max(0.5, max(span), track_span)

    bpy.ops.object.light_add(type="SUN")
    bpy.context.active_object.data.energy = 4.0
    bpy.context.active_object.rotation_euler = (math.radians(50), 0.0, math.radians(40))

    if span[0] > 2.0 * max(span[1], span[2], 0.3):        # a wide row -> front-on view
        d = span[0] * 1.3 + 1.5                           # far enough that the whole row fits the FOV
        loc = (ctr[0], ctr[1] - d, ctr[2] + 0.3 * d)
    else:                                                # default 3/4 view
        d = rad * 2.4
        loc = (ctr[0] + d, ctr[1] - d, ctr[2] + 0.6 * d)
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.active_object
    cam.rotation_euler = (mathutils.Vector(ctr) - cam.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = cam
    return cam


def _clear():
    for o in list(bpy.data.objects):
        if o.type in {"MESH", "CURVE", "CAMERA", "LIGHT", "EMPTY"}:
            bpy.data.objects.remove(o, do_unlink=True)


def _build(plan, coll):
    empties = {}
    for g in plan["geoms"]:
        empties[g["name"]] = _add_body(g, coll)
    for m in plan.get("muscles", []):
        _add_muscle(m, coll)
    return empties


def _collection(name):
    coll = bpy.data.collections.get(name) or bpy.data.collections.new(name)
    if name not in {c.name for c in bpy.context.scene.collection.children}:
        bpy.context.scene.collection.children.link(coll)
    return coll


def build_from_plan(plan, *, clear=True, add_camera=True):
    """Build the static creature described by `plan` (a ScenePlan dict)."""
    if not _HAS_BPY:
        raise RuntimeError("build_from_plan must run inside Blender (no bpy).")
    if isinstance(plan, str):
        plan = json.loads(plan)
    if clear:
        _clear()
    coll = _collection(plan.get("name", "creature"))
    empties = _build(plan, coll)
    if add_camera:
        _add_light_and_camera(plan["geoms"])
    print(f"built '{plan.get('name')}': {len(plan['geoms'])} bodies, "
          f"{len(plan.get('muscles', []))} muscles")
    return empties


def animate_trajectory(traj, *, fps=30):
    """Build the creature from `traj['plan']` and keyframe each body's empty from the
    recorded per-frame world transforms `traj['frames']` (list of {body_id:[pos,quat]})."""
    if not _HAS_BPY:
        raise RuntimeError("animate_trajectory must run inside Blender (no bpy).")
    if isinstance(traj, str):
        traj = json.loads(traj)
    plan = traj["plan"]
    frames = traj["frames"]
    _clear()
    coll = _collection(plan.get("name", "creature"))
    empties = _build(plan, coll)

    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(frames)
    scene.render.fps = int(traj.get("fps", fps))
    for fi, fr in enumerate(frames, start=1):
        for bid, (pos, quat) in fr.items():
            e = empties.get(bid)
            if e is None:
                continue
            e.location = mathutils.Vector(pos)
            e.rotation_quaternion = _quat(quat)
            e.keyframe_insert("location", frame=fi)
            e.keyframe_insert("rotation_quaternion", frame=fi)

    # camera frames the whole travelled path (use all body positions across frames)
    span = 0.0
    xs = [p[0] for fr in frames for (p, _q) in fr.values()]
    ys = [p[1] for fr in frames for (p, _q) in fr.values()]
    if xs:
        span = max(max(xs) - min(xs), max(ys) - min(ys))
    _add_ground(z=0.0)
    _add_light_and_camera(plan["geoms"], track_span=span)
    print(f"animated '{plan.get('name')}': {len(frames)} frames @ {scene.render.fps}fps")
    return empties


def _cylinder(name, mat, coll):
    bpy.ops.mesh.primitive_cylinder_add(radius=1.0, depth=1.0)
    o = bpy.context.active_object
    o.name = name
    o.rotation_mode = "QUATERNION"
    o.data.materials.clear()
    o.data.materials.append(mat)
    for c in list(o.users_collection):
        c.objects.unlink(o)
    coll.objects.link(o)
    return o


def _segment_key(o, p0, p1, frame, thickness):
    """Pose a unit cylinder as a bar from p0->p1 with the given radius, and keyframe it."""
    a, b = mathutils.Vector(p0), mathutils.Vector(p1)
    d = b - a
    L = max(d.length, 1e-5)
    o.location = (a + b) / 2.0
    o.rotation_quaternion = d.to_track_quat("Z", "X")
    o.scale = (thickness, thickness, L)
    o.keyframe_insert("location", frame=frame)
    o.keyframe_insert("rotation_quaternion", frame=frame)
    o.keyframe_insert("scale", frame=frame)


def animate_biomechanics(bio, *, fps=30):
    """Build + keyframe the creature, overlaying each muscle as a RED bar (radius
    scaled by tension) along its line of action and a GREEN moment-arm lever from the
    joint to the muscle line. `bio` is a BioTrajectory dict (Stage 10.4)."""
    if not _HAS_BPY:
        raise RuntimeError("animate_biomechanics must run inside Blender (no bpy).")
    if isinstance(bio, str):
        bio = json.loads(bio)
    plan, frames = bio["plan"], bio["frames"]
    fmax = bio.get("force_max", 1.0) or 1.0
    _clear()
    coll = _collection(plan.get("name", "creature"))
    empties = _build(plan, coll)

    n_m = len(frames[0]["muscles"]) if frames else 0
    force_mat = _material("mat_force", (0.9, 0.12, 0.12))
    lever_mat = _material("mat_lever", (0.98, 0.85, 0.05))   # yellow: distinct from limbs
    force_objs = [_cylinder(f"force{i}", force_mat, coll) for i in range(n_m)]
    lever_objs = [_cylinder(f"lever{i}", lever_mat, coll) for i in range(n_m)]

    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 1, len(frames)
    scene.render.fps = int(bio.get("fps", fps))
    for fi, fr in enumerate(frames, start=1):
        for bid, (pos, quat) in fr["bodies"].items():
            e = empties.get(bid)
            if e is None:
                continue
            e.location = mathutils.Vector(pos)
            e.rotation_quaternion = _quat(quat)
            e.keyframe_insert("location", frame=fi)
            e.keyframe_insert("rotation_quaternion", frame=fi)
        for i, mu in enumerate(fr["muscles"]):
            th = 0.006 + 0.035 * (mu["force"] / fmax)        # radius ~ tension
            _segment_key(force_objs[i], mu["org"], mu["ins"], fi, th)
            if mu["anchor"] is not None and mu["moment_arm"] > 1e-4:
                _segment_key(lever_objs[i], mu["anchor"], mu["foot"], fi, 0.010)
            else:                                            # no joint / zero lever -> hide
                lever_objs[i].scale = (0.0, 0.0, 0.0)
                lever_objs[i].keyframe_insert("scale", frame=fi)

    span = 0.0
    xs = [p[0] for fr in frames for (p, _q) in fr["bodies"].values()]
    ys = [p[1] for fr in frames for (p, _q) in fr["bodies"].values()]
    if xs:
        span = max(max(xs) - min(xs), max(ys) - min(ys))
    _add_ground(z=0.0)
    _add_light_and_camera(plan["geoms"], track_span=span)
    print(f"biomechanics '{plan.get('name')}': {len(frames)} frames, {n_m} muscles")
    return empties


def render_to(path, *, resolution=(960, 720), samples=16):
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.filepath = path
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = samples
    bpy.ops.render.render(write_still=True)
    return path


def render_animation(frames_dir, *, resolution=(800, 600)):
    """Render the keyframed animation as a PNG sequence into `frames_dir` (assembled
    into a video by viz.video.stitch -- this Blender build has no ffmpeg)."""
    os.makedirs(frames_dir, exist_ok=True)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = os.path.join(frames_dir, "frame_")
    bpy.ops.render.render(animation=True)
    return frames_dir


if __name__ == "__main__":   # blender -b -P blender_build.py -- in.json [out.(png|mp4)]
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if argv:
        with open(argv[0]) as f:
            data = json.load(f)
        out = argv[1] if len(argv) > 1 else None
        if "frames" in data and data["frames"] and "muscles" in data["frames"][0]:
            animate_biomechanics(data)                     # biomechanics clip (Stage 10.4)
            if out:
                render_animation(out)
        elif "frames" in data:                             # locomotion replay (Stage 10.2)
            animate_trajectory(data)
            if out:
                render_animation(out)
        else:                                              # a static scene plan (Stage 10.1)
            build_from_plan(data)
            if out:
                render_to(out)
