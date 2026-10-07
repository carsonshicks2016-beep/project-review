"""
Procedural low-poly trackside props & buildings — PS1/PS2-era flat-shaded faux-3D,
the same recipe as the car renderer in carart.py: hand-placed low vertex counts,
ONE flat shade per face (dot(normal, light) quantized, no Gouraud), dark edge
outlines for that "polygon jitter" definition, and a 2.5D ground-plane projection.

Each prop is a tiny mesh in LOCAL coordinates:
    +x = the prop's facing direction (toward the road for roadside furniture)
    +y = lateral (its left)
    +z = up (height in metres)

`emit_faces()` rotates the mesh onto the object's world heading (hx, hy), projects
every vertex with the caller's `project_pt(wx, wy, wz)`, flat-shades each face, and
returns face dicts that slot straight into background.py's global painter's-sort
render list — so props depth-interleave correctly with the trees and streetlights.

Faces whose name starts with "lit_" are emissive (lit windows / vending panels):
they ignore the scene shading + ambient so they glow at dusk and night.
"""
from __future__ import annotations

import math
import numpy as np

# Prop/building types this module knows how to build & emit.
PROP_TYPES = frozenset({
    "guardrail", "traffic_cone", "tire_stack", "vending_machine",
    "road_sign", "billboard", "garage", "grandstand",
})

# Big props that cast a simple ground shadow blob (background draws it).
SHADOW_TYPES = frozenset({"vending_machine", "billboard", "garage", "grandstand", "tire_stack"})


def _shade(c, f):
    return (max(0, min(255, int(c[0] * f))),
            max(0, min(255, int(c[1] * f))),
            max(0, min(255, int(c[2] * f))))


# --------------------------------------------------------------------------- #
# Mesh-construction helpers (return (vertices, faces); faces are
# (name, [vertex_idx...], (nx, ny, nz) local-normal, (r, g, b) color).
# Bottom faces are omitted — never seen from a near-top-down camera.
# --------------------------------------------------------------------------- #
def _box(x0, x1, y0, y1, z0, z1, col, col_top=None, prefix="box"):
    v = [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),   # 0..3 bottom
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),   # 4..7 top
    ]
    ct = col_top if col_top is not None else col
    f = [
        (prefix + "_top", [4, 5, 6, 7], (0.0, 0.0, 1.0), ct),
        (prefix + "_xp", [1, 2, 6, 5], (1.0, 0.0, 0.0), col),
        (prefix + "_xn", [3, 0, 4, 7], (-1.0, 0.0, 0.0), col),
        (prefix + "_yp", [2, 3, 7, 6], (0.0, 1.0, 0.0), col),
        (prefix + "_yn", [0, 1, 5, 4], (0.0, -1.0, 0.0), col),
    ]
    return v, f


def _frustum_sq(s0, s1, z0, z1, col, prefix):
    """A square frustum (truncated pyramid) — the building block of a cone."""
    b = [(-s0, -s0, z0), (s0, -s0, z0), (s0, s0, z0), (-s0, s0, z0)]
    t = [(-s1, -s1, z1), (s1, -s1, z1), (s1, s1, z1), (-s1, s1, z1)]
    v = b + t
    f = [
        (prefix + "_yp", [2, 3, 7, 6], (0.0, 1.0, 0.45), col),
        (prefix + "_yn", [0, 1, 5, 4], (0.0, -1.0, 0.45), col),
        (prefix + "_xp", [1, 2, 6, 5], (1.0, 0.0, 0.45), col),
        (prefix + "_xn", [3, 0, 4, 7], (-1.0, 0.0, 0.45), col),
    ]
    return v, f


def _ngon_prism(n, r, z0, z1, col_side, col_top, prefix):
    """An n-sided vertical prism — low-poly stand-in for a cylinder (tire/drum)."""
    ang = [2.0 * math.pi * k / n for k in range(n)]
    bottom = [(r * math.cos(a), r * math.sin(a), z0) for a in ang]
    top = [(r * math.cos(a), r * math.sin(a), z1) for a in ang]
    v = bottom + top
    f = []
    for k in range(n):
        k2 = (k + 1) % n
        am = (ang[k] + ang[k2]) / 2.0
        f.append((f"{prefix}_s{k}", [k, k2, n + k2, n + k],
                  (math.cos(am), math.sin(am), 0.0), col_side))
    f.append((f"{prefix}_top", list(range(n, 2 * n)), (0.0, 0.0, 1.0), col_top))
    return v, f


def _combine(parts):
    """Merge several (verts, faces) meshes, offsetting face indices."""
    verts, faces = [], []
    for pv, pf in parts:
        off = len(verts)
        verts.extend(pv)
        for name, idx, normal, col in pf:
            faces.append((name, [i + off for i in idx], normal, col))
    return verts, faces


# --------------------------------------------------------------------------- #
# Per-prop mesh builders. `o` is the placed-object dict (carries color/size).
# --------------------------------------------------------------------------- #
def _m_guardrail(o):
    """Armco / W-beam guardrail: a galvanized rail on two posts, run along +x."""
    L = o.get("length", 6.0)
    steel = (188, 194, 203)
    post = (118, 124, 132)
    rail = _box(-L / 2, L / 2, -0.06, 0.06, 0.55, 0.95, steel, prefix="rail")
    p0 = _box(-L / 2 + 0.2, -L / 2 + 0.45, -0.08, 0.08, 0.0, 0.95, post, prefix="postA")
    p1 = _box(L / 2 - 0.45, L / 2 - 0.2, -0.08, 0.08, 0.0, 0.95, post, prefix="postB")
    return _combine([rail, p0, p1])


def _m_traffic_cone(o):
    orange = o.get("color", (236, 110, 28))
    white = (238, 238, 240)
    base = _box(-0.28, 0.28, -0.28, 0.28, 0.0, 0.05, (44, 44, 48), prefix="cbase")
    body = _frustum_sq(0.24, 0.17, 0.05, 0.40, orange, "ccol")
    band = _frustum_sq(0.17, 0.155, 0.40, 0.52, white, "cband")
    tip = _frustum_sq(0.155, 0.03, 0.52, 0.78, orange, "ctip")
    return _combine([base, body, band, tip])


def _m_tire_stack(o):
    rubber = (36, 36, 40)
    rim = (78, 78, 86)
    n = o.get("count", 4)
    parts = []
    for k in range(n):
        z0 = k * 0.38
        parts.append(_ngon_prism(7, 0.55, z0, z0 + 0.34, rubber, rim, f"tire{k}"))
    return _combine(parts)


def _m_vending_machine(o):
    body = o.get("color", (208, 44, 44))
    W, D, H = 1.0, 0.72, 1.85
    box = _box(-D / 2, D / 2, -W / 2, W / 2, 0.0, H, body, col_top=(58, 60, 68), prefix="vend")
    # Glowing product window on the road-facing (+x) side.
    panel = ("lit_vend_face", None, (1.0, 0.0, 0.0),
             _shade(body, 1.35) if max(body) < 200 else (255, 240, 210))
    lit = [(D / 2 + 0.01, -W / 2 + 0.12, 0.55),
           (D / 2 + 0.01, W / 2 - 0.12, 0.55),
           (D / 2 + 0.01, W / 2 - 0.12, H - 0.18),
           (D / 2 + 0.01, -W / 2 + 0.12, H - 0.18)]
    # Dark dispenser slot near the bottom.
    slot = [(D / 2 + 0.012, -W / 2 + 0.18, 0.16),
            (D / 2 + 0.012, W / 2 - 0.18, 0.16),
            (D / 2 + 0.012, W / 2 - 0.18, 0.40),
            (D / 2 + 0.012, -W / 2 + 0.18, 0.40)]
    v, f = box
    base = len(v)
    v.extend(lit)
    f.append(("lit_vend_face", [base, base + 1, base + 2, base + 3], (1.0, 0.0, 0.0), panel[3]))
    base = len(v)
    v.extend(slot)
    f.append(("vend_slot", [base, base + 1, base + 2, base + 3], (1.0, 0.0, 0.0), (20, 21, 26)))
    return v, f


def _m_road_sign(o):
    post = (120, 122, 130)
    face = o.get("color", (40, 96, 176))
    pole = _box(-0.05, 0.05, -0.05, 0.05, 0.0, 1.7, post, prefix="sgnpost")
    # Panel faces the road (+x), centered atop the post, ~0.75 m square.
    panel = _box(0.0, 0.06, -0.38, 0.38, 1.55, 2.25, face, prefix="sgn")
    return _combine([pole, panel])


def _m_billboard(o):
    post = (96, 98, 104)
    face = o.get("color", (220, 196, 64))
    accent = o.get("accent", (40, 44, 56))
    pL = _box(-0.1, 0.1, -1.7, -1.5, 0.0, 3.0, post, prefix="bbpL")
    pR = _box(-0.1, 0.1, 1.5, 1.7, 0.0, 3.0, post, prefix="bbpR")
    # Big panel facing +x.
    panel = _box(0.0, 0.12, -2.1, 2.1, 2.7, 4.5, face, prefix="bbface")
    # A simple darker stripe across the panel for graphic interest.
    stripe = [(0.13, -2.0, 3.35), (0.13, 2.0, 3.35), (0.13, 2.0, 3.85), (0.13, -2.0, 3.85)]
    v, f = _combine([pL, pR, panel])
    base = len(v)
    v.extend(stripe)
    f.append(("bb_stripe", [base, base + 1, base + 2, base + 3], (1.0, 0.0, 0.0), accent))
    return v, f


def _m_garage(o):
    """A low-poly pit hut: box walls + gable roof + a roll-up door facing the road."""
    wall = o.get("color", (176, 172, 162))
    roof = (94, 100, 116)
    D, W, H, rh = o.get("depth", 6.0), o.get("width", 7.0), 3.0, 1.4
    walls = _box(-D / 2, D / 2, -W / 2, W / 2, 0.0, H, wall, col_top=wall, prefix="gw")
    v, f = walls
    base = len(v)
    # Roof: ridge runs along +y at x=0; two slopes + two gable triangles.
    v.extend([
        (D / 2, -W / 2, H), (D / 2, W / 2, H),       # 0,1 front eave
        (-D / 2, -W / 2, H), (-D / 2, W / 2, H),     # 2,3 back eave
        (0.0, -W / 2, H + rh), (0.0, W / 2, H + rh),  # 4,5 ridge
    ])
    f += [
        ("groof_xp", [base + 0, base + 1, base + 5, base + 4], (0.7, 0.0, 0.7), roof),
        ("groof_xn", [base + 2, base + 3, base + 5, base + 4], (-0.7, 0.0, 0.7), _shade(roof, 0.88)),
        ("ggable_yn", [base + 0, base + 2, base + 4], (0.0, -1.0, 0.2), _shade(wall, 0.92)),
        ("ggable_yp", [base + 1, base + 3, base + 5], (0.0, 1.0, 0.2), _shade(wall, 0.92)),
    ]
    # Roll-up door (dark) on the +x facade.
    base = len(v)
    v.extend([(D / 2 + 0.02, -W * 0.32, 0.0), (D / 2 + 0.02, W * 0.32, 0.0),
              (D / 2 + 0.02, W * 0.32, 2.2), (D / 2 + 0.02, -W * 0.32, 2.2)])
    f.append(("gdoor", [base, base + 1, base + 2, base + 3], (1.0, 0.0, 0.0), (58, 62, 72)))
    # Two lit windows flanking the door (glow at night).
    for k, yc in enumerate((-W * 0.42, W * 0.42)):
        base = len(v)
        v.extend([(D / 2 + 0.02, yc - 0.25, 1.4), (D / 2 + 0.02, yc + 0.25, 1.4),
                  (D / 2 + 0.02, yc + 0.25, 2.2), (D / 2 + 0.02, yc - 0.25, 2.2)])
        f.append((f"lit_gwin{k}", [base, base + 1, base + 2, base + 3], (1.0, 0.0, 0.0), (255, 226, 150)))
    return v, f


def _m_grandstand(o):
    """A raked spectator stand: stepped tiers facing the road, dotted with a crowd."""
    struct = (138, 140, 148)
    seat = (92, 102, 124)
    W = o.get("width", 12.0)
    steps = 5
    rng = np.random.default_rng(o.get("seed", 1))
    parts = []
    crowd_v, crowd_f = [], []
    for k in range(steps):
        x0 = 0.6 * k
        x1 = x0 + 0.9
        z0 = 0.0
        z1 = 0.5 + 0.55 * k
        col = struct if k == 0 else seat
        parts.append(_box(x0, x1, -W / 2, W / 2, z0, z1, col, col_top=seat, prefix=f"gs{k}"))
        # Crowd specks sitting on this tier's top.
        if k >= 1:
            for _ in range(int(W * 0.55)):
                cy = float(rng.uniform(-W / 2 + 0.3, W / 2 - 0.3))
                cx = float(rng.uniform(x0 + 0.2, x1 - 0.2))
                ch = float(rng.uniform(0.35, 0.55))
                col_p = tuple(int(c) for c in rng.choice([
                    (210, 80, 70), (70, 120, 200), (230, 210, 90),
                    (90, 180, 110), (220, 220, 225), (200, 120, 60),
                ]))
                b = len(crowd_v)
                crowd_v.extend([(cx - 0.12, cy - 0.12, z1), (cx + 0.12, cy - 0.12, z1),
                                (cx, cy, z1 + ch)])
                crowd_f.append((f"person", [b, b + 1, b + 2], (1.0, 0.0, 0.3), col_p))
    parts.append((crowd_v, crowd_f))
    return _combine(parts)


_BUILDERS = {
    "guardrail": _m_guardrail,
    "traffic_cone": _m_traffic_cone,
    "tire_stack": _m_tire_stack,
    "vending_machine": _m_vending_machine,
    "road_sign": _m_road_sign,
    "billboard": _m_billboard,
    "garage": _m_garage,
    "grandstand": _m_grandstand,
}


# --------------------------------------------------------------------------- #
# Emission — mesh → world → screen, flat-shaded, as background-compatible dicts.
# --------------------------------------------------------------------------- #
def emit_faces(o, project_pt, cx, cy, H_cam, light_dir, ambient):
    """Return a list of face dicts {dist, poly_pts, color, f_name, outline_color}
    for one placed prop, ready to extend background.py's `faces_to_render`."""
    builder = _BUILDERS.get(o["type"])
    if builder is None:
        return []
    verts, faces = builder(o)

    hx, hy = o.get("hx", 1.0), o.get("hy", 0.0)
    ox, oy = o["x"], o["y"]
    ldx, ldy, ldz = light_dir

    # Local → world (yaw by heading), then project once per vertex.
    world = []
    screen = []
    for lx, ly, lz in verts:
        wx = ox + lx * hx - ly * hy
        wy = oy + lx * hy + ly * hx
        world.append((wx, wy, lz))
        screen.append(project_pt(wx, wy, lz))

    out = []
    for name, idx, (nx, ny, nz), col in faces:
        # Rotate the local normal to world frame (same yaw).
        nwx = nx * hx - ny * hy
        nwy = nx * hy + ny * hx
        nwz = nz

        if name.startswith("lit_"):
            # Emissive: ignore scene shading; glow through dusk/night.
            shaded = _shade(col, max(ambient, 0.9) + 0.1)
            outline = _shade(col, 0.78)
        else:
            dot = nwx * ldx + nwy * ldy + nwz * ldz
            shade = (0.55 + 0.45 * max(0.0, dot)) * ambient
            shaded = _shade(col, shade)
            outline = _shade(shaded, 0.7)

        # Painter's depth: face centroid distance to the camera eye.
        fx = sum(world[i][0] for i in idx) / len(idx)
        fy = sum(world[i][1] for i in idx) / len(idx)
        fz = sum(world[i][2] for i in idx) / len(idx)
        dist = math.sqrt((fx - cx) ** 2 + (fy - cy) ** 2 + (fz - H_cam) ** 2)

        out.append({
            "dist": dist,
            "poly_pts": [screen[i] for i in idx],
            "color": shaded,
            "f_name": "prop",
            "outline_color": outline,
        })
    return out
