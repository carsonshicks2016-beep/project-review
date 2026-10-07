#!/usr/bin/env python3
"""Build the clean-room Porsche 919 Hybrid Evo visual reconstruction.

The mesh is authored from published dimensions, measured section profiles, and
human study of real-world photography.  It contains no OEM CAD, scan,
third-party mesh, reference photograph, or previous Observatory geometry.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh


HERE = Path(__file__).resolve().parent
VEHICLES = HERE.parent
if str(VEHICLES) not in sys.path:
    sys.path.insert(0, str(VEHICLES))

from cleanroom_mesh import (  # noqa: E402
    add_anchor,
    add_local,
    airfoil,
    body_shell,
    bounds_extents,
    box,
    canopy_shell,
    cylinder,
    ellipsoid,
    mesh_triangle_count,
    oriented_cylinder,
    pbr,
    quad,
    radial_spoke,
    rim_ring,
    scene_nodes,
    shade_scene,
    tyre,
)


ASSET_ID = "porsche_919evo_cleanroom_v2"
ENTRY_GLB = "porsche_919evo_cleanroom_v2.glb"
ENTRY_TEXTURE = "porsche_919evo_tribute_atlas_v2.png"
LENGTH_M = 5.078
BASE_BODY_LENGTH_M = 4.650
WIDTH_M = 1.900
HEIGHT_M = 1.050
WHEELBASE_M = 2.957
FRONT_TRACK_M = 1.536
REAR_TRACK_M = 1.542
FRONT_AXLE_X_M = WHEELBASE_M * 0.5
REAR_AXLE_X_M = -WHEELBASE_M * 0.5
DEFAULT_OUTPUT = HERE / ENTRY_GLB


WHITE = (238, 240, 238, 255)
RED = (198, 18, 39, 255)
BLACK = (15, 17, 19, 255)
CARBON = (21, 24, 26, 255)
GLASS = (21, 37, 48, 218)
RUBBER = (11, 12, 13, 255)
RIM = (28, 30, 31, 255)
METAL = (134, 139, 141, 255)
BRAKE = (62, 64, 65, 255)
CALIPER = (213, 181, 36, 255)
HEADLIGHT = (223, 242, 255, 255)
TAIL = (232, 19, 26, 255)
AMBER = (255, 142, 29, 255)

# Porsche publishes Michelin 310/710-18 tyres for the 919 Hybrid Evo.  These
# values make the visual ground contract explicit in the geometry itself:
# 710 mm outside diameter, 310 mm section width and an 18 inch rim.
TYRE_OUTER_RADIUS_M = 0.355
TYRE_SECTION_WIDTH_M = 0.310
RIM_RADIUS_M = 0.2286
WHEEL_HUMP_CLEARANCE_M = 0.090


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def livery_atlas() -> tuple[Image.Image, dict[str, tuple[float, float, float, float]]]:
    size = 2048
    image = Image.new("RGB", (size, size), WHITE[:3])
    draw = ImageDraw.Draw(image)
    tiles_px = {
        "porsche_black": (0, 0, 1200, 300),
        "number": (1200, 0, 1700, 500),
        "chopard": (0, 300, 900, 560),
        "side": (0, 560, 1200, 1080),
        "wing": (1200, 500, 2048, 820),
        "sponsors": (1200, 820, 2048, 1120),
        "tribute": (0, 1080, 2048, 1500),
        "black_panel": (0, 1500, 2048, 2048),
    }

    def center_text(rect, text, font_size, fill, stroke=0, stroke_fill=WHITE[:3]):
        x0, y0, x1, y1 = rect
        draw.text(
            ((x0 + x1) * 0.5, (y0 + y1) * 0.5), text,
            font=_font(font_size), fill=fill, anchor="mm",
            stroke_width=stroke, stroke_fill=stroke_fill,
        )

    draw.rectangle(tiles_px["porsche_black"], fill=BLACK[:3])
    center_text(tiles_px["porsche_black"], "PORSCHE", 188, WHITE[:3])

    draw.rectangle(tiles_px["number"], fill=RED[:3])
    center_text(tiles_px["number"], "1", 330, WHITE[:3])

    draw.rectangle(tiles_px["chopard"], fill=WHITE[:3])
    center_text(tiles_px["chopard"], "CHOPARD", 126, BLACK[:3])

    draw.rectangle(tiles_px["side"], fill=WHITE[:3])
    draw.rectangle((0, 930, 1200, 1080), fill=RED[:3])
    center_text((20, 570, 1180, 790), "PORSCHE", 116, BLACK[:3])
    center_text((20, 770, 1180, 930), "919 tribute", 88, RED[:3])

    draw.rectangle(tiles_px["wing"], fill=BLACK[:3])
    center_text(tiles_px["wing"], "PORSCHE", 118, WHITE[:3])

    draw.rectangle(tiles_px["sponsors"], fill=WHITE[:3])
    center_text((1200, 820, 1620, 1120), "Mobil 1", 62, BLACK[:3])
    center_text((1620, 820, 2048, 970), "SCHAEFFLER", 49, BLACK[:3])
    center_text((1620, 970, 2048, 1120), "MICHELIN", 50, BLACK[:3])

    draw.rectangle(tiles_px["tribute"], fill=RED[:3])
    names = "PORSCHE MOTORSPORT 919 HYBRID EVO 2014 2015 2016 2017 LE MANS WEC TRIBUTE"
    for y in range(1100, 1490, 42):
        draw.text((24, y), names * 3, font=_font(24), fill=WHITE[:3])

    draw.rectangle(tiles_px["black_panel"], fill=BLACK[:3])
    center_text((40, 1560, 1000, 1830), "CHOPARD", 112, WHITE[:3])
    center_text((1040, 1560, 2000, 1830), "SCHAEFFLER", 96, WHITE[:3])
    center_text((40, 1830, 1000, 2020), "919 EVO", 84, RED[:3])
    center_text((1040, 1830, 2000, 2020), "HYBRID", 84, WHITE[:3])

    uv: dict[str, tuple[float, float, float, float]] = {}
    for name, (x0, y0, x1, y1) in tiles_px.items():
        uv[name] = (x0 / size, 1.0 - y1 / size, x1 / size, 1.0 - y0 / size)
    return image, uv


def _materials(atlas: Image.Image) -> dict[str, trimesh.visual.material.PBRMaterial]:
    return {
        "white": pbr("919_tribute_white", WHITE, roughness=0.27),
        "red": pbr("919_tribute_red", RED, roughness=0.28),
        "black": pbr("919_satin_black", BLACK, roughness=0.43),
        "carbon": pbr("919_carbon_aero", CARBON, metallic=0.28, roughness=0.38),
        "glass": pbr("919_canopy_glass", GLASS, metallic=0.08, roughness=0.07),
        "rubber": pbr("michelin_slick_rubber", RUBBER, roughness=0.9),
        "rim": pbr("bbs_forged_magnesium", RIM, metallic=0.76, roughness=0.25),
        "metal": pbr("919_machined_metal", METAL, metallic=0.84, roughness=0.22),
        "brake": pbr("919_carbon_brake_disc", BRAKE, metallic=0.1, roughness=0.74),
        "caliper": pbr("919_brake_caliper", CALIPER, metallic=0.22, roughness=0.34),
        "light": pbr("919_headlamp_lens", HEADLIGHT, roughness=0.07),
        "tail": pbr("919_rear_lamp", TAIL, roughness=0.14),
        "amber": pbr("919_side_marker", AMBER, roughness=0.18),
        "atlas": pbr("919_tribute_livery_atlas", (255, 255, 255, 255), roughness=0.27, image=atlas),
    }


def _joined_exterior(material: trimesh.visual.material.PBRMaterial) -> trimesh.Trimesh:
    """One continuous body sheet with wheel humps and open side skirt bays."""
    profile_x = np.asarray((-2.34, -2.10, -1.82, -1.48, -1.12, -0.68, 0.0, 0.68, 1.12, 1.48, 1.82, 2.10, 2.34), dtype=float)
    central_profile = np.asarray((0.28, 0.40, 0.46, 0.50, 0.55, 0.58, 0.60, 0.58, 0.55, 0.50, 0.46, 0.40, 0.28), dtype=float)
    x_values = np.linspace(-2.34, 2.34, 43)
    y_values = np.linspace(-0.95, 0.95, 23)

    def roof(x: float, y: float) -> float:
        base = float(np.interp(x, profile_x, central_profile))
        shoulder = 0.13 * (abs(y) / 0.95) ** 1.55
        value = base - shoulder
        for axle in (FRONT_AXLE_X_M, REAR_AXLE_X_M):
            wheel_x = math.exp(-((x - axle) / 0.43) ** 2)
            wheel_y = math.exp(-((abs(y) - 0.78) / 0.22) ** 2)
            value += 0.40 * wheel_x * wheel_y
        return min(0.885, value)

    vertices: list[tuple[float, float, float]] = []
    uv: list[tuple[float, float]] = []
    for xi, x in enumerate(x_values):
        for yi, y in enumerate(y_values):
            vertices.append((float(x), float(y), roof(float(x), float(y))))
            uv.append((xi / (len(x_values) - 1), yi / (len(y_values) - 1)))
    stride = len(y_values)
    faces: list[tuple[int, int, int]] = []
    for xi in range(len(x_values) - 1):
        for yi in range(len(y_values) - 1):
            a = xi * stride + yi
            b = a + 1
            c = a + stride
            d = c + 1
            faces.extend(((a, c, b), (b, c, d)))

    # Close the outer silhouette except through deliberately open wheel bays.
    for yi in (0, len(y_values) - 1):
        for xi in range(len(x_values) - 1):
            midpoint = (x_values[xi] + x_values[xi + 1]) * 0.5
            if any(abs(midpoint - axle) < 0.52 for axle in (FRONT_AXLE_X_M, REAR_AXLE_X_M)):
                continue
            top_a = xi * stride + yi
            top_b = (xi + 1) * stride + yi
            x_a, y_a, z_a = vertices[top_a]
            x_b, y_b, z_b = vertices[top_b]
            lower_a = len(vertices)
            vertices.append((x_a, y_a, max(0.14, z_a - 0.28)))
            uv.append((xi / (len(x_values) - 1), 0.0))
            lower_b = len(vertices)
            vertices.append((x_b, y_b, max(0.14, z_b - 0.28)))
            uv.append(((xi + 1) / (len(x_values) - 1), 0.0))
            if yi == 0:
                faces.extend(((top_a, lower_a, top_b), (top_b, lower_a, lower_b)))
            else:
                faces.extend(((top_a, top_b, lower_a), (top_b, lower_b, lower_a)))

    for xi, reverse in ((0, True), (len(x_values) - 1, False)):
        for yi in range(len(y_values) - 1):
            top_a = xi * stride + yi
            top_b = top_a + 1
            x_a, y_a, z_a = vertices[top_a]
            x_b, y_b, z_b = vertices[top_b]
            lower_a = len(vertices)
            vertices.append((x_a, y_a, max(0.14, z_a - 0.18)))
            uv.append((0.0, yi / (len(y_values) - 1)))
            lower_b = len(vertices)
            vertices.append((x_b, y_b, max(0.14, z_b - 0.18)))
            uv.append((0.0, (yi + 1) / (len(y_values) - 1)))
            faces.extend(((top_a, top_b, lower_a), (top_b, lower_b, lower_a)) if reverse else ((top_a, lower_a, top_b), (top_b, lower_a, lower_b)))
    mesh = trimesh.Trimesh(vertices=np.asarray(vertices), faces=np.asarray(faces), process=False)
    mesh.visual = trimesh.visual.TextureVisuals(uv=np.asarray(uv), material=material)
    return mesh


def _validate_visual_clearances() -> None:
    """Reject source dimensions that would reintroduce wheel/body clipping."""
    front_inner_tyre_face = FRONT_TRACK_M * 0.5 - TYRE_SECTION_WIDTH_M * 0.5
    rear_inner_tyre_face = REAR_TRACK_M * 0.5 - TYRE_SECTION_WIDTH_M * 0.5
    if 0.800 - TYRE_OUTER_RADIUS_M * 2.0 < WHEEL_HUMP_CLEARANCE_M:
        raise RuntimeError("919 exterior wheel humps must stay above the tyre crowns")
    if 0.600 >= min(front_inner_tyre_face, rear_inner_tyre_face) - 0.010:
        raise RuntimeError("919 undertray would occupy the native tyre envelope")
    if RIM_RADIUS_M >= TYRE_OUTER_RADIUS_M:
        raise RuntimeError("919 rim must remain inside the tyre outer radius")


def _add_wheel(
    scene: trimesh.Scene,
    corner: str,
    center: tuple[float, float, float],
    materials: dict[str, trimesh.visual.material.PBRMaterial],
) -> None:
    outer_radius = TYRE_OUTER_RADIUS_M
    rim_radius = RIM_RADIUS_M
    width = TYRE_SECTION_WIDTH_M
    outside = 1.0 if center[1] > 0 else -1.0
    pivot = center
    add_local(
        scene,
        f"wheel_{corner}",
        tyre(outer_radius, rim_radius, width, center, materials["rubber"], major_sections=72, minor_sections=18),
        pivot=pivot,
    )
    face_y = center[1] + outside * (width * 0.5 - 0.010)
    add_local(scene, f"wheel_{corner}__rim_barrel", rim_ring(rim_radius, rim_radius * 0.62, 0.046, (center[0], face_y, center[2]), materials["rim"]), pivot=pivot)
    add_local(scene, f"wheel_{corner}__centerlock", cylinder(0.046, 0.024, (center[0], face_y + outside * 0.006, center[2]), materials["metal"], axis="y", sections=32), pivot=pivot)
    for index in range(12):
        angle = index * math.tau / 12.0
        spoke = radial_spoke((center[0], face_y + outside * 0.012, center[2]), rim_radius * 0.85, angle, 0.020, 0.020, materials["rim"])
        add_local(scene, f"wheel_{corner}__spoke_{index:02d}", spoke, pivot=pivot)
    # These are native parts of the visual wheel package, rather than a
    # second runtime wheel kit layered over the GLB.
    add_local(scene, f"brake_disc_{corner}", cylinder(0.170, 0.020, (center[0], face_y - outside * 0.033, center[2]), materials["brake"], axis="y", sections=48), pivot=pivot)
    add_local(scene, f"brake_caliper_{corner}", box((0.075, 0.044, 0.19), (center[0] - 0.085, face_y - outside * 0.052, center[2]), materials["caliper"]))
    inboard_y = 0.53 if center[1] > 0 else -0.53
    add_local(scene, f"suspension_pivot_{corner}", ellipsoid((0.026, 0.026, 0.026), (center[0], inboard_y, center[2] + 0.06), materials["metal"], 1))
    for suffix, dz in (("upper", 0.13), ("lower", -0.11)):
        start = (center[0], inboard_y, center[2] + dz)
        end = (center[0], center[1] - outside * 0.12, center[2] + dz * 0.45)
        add_local(scene, f"suspension_arm_{corner}_{suffix}", oriented_cylinder(start, end, 0.012, materials["metal"], sections=10))


def build_scene(atlas: Image.Image, uv: dict[str, tuple[float, float, float, float]]) -> trimesh.Scene:
    m = _materials(atlas)
    scene = trimesh.Scene(base_frame="vehicle_origin")

    # The Evo's 5.078 m envelope is aero-to-aero; the underlying 919 body is
    # retained within the separately documented 4.650 m base length.
    # Keep the structural floor inside the inner tyre faces.  The fenders are
    # independent, open panels, so the body never has to occupy wheel volume.
    add_local(scene, "body_undertray", box((4.61, 1.20, 0.052), (-0.01, 0.0, 0.026), m["carbon"]))
    add_local(scene, "aero_front_splitter", airfoil((2.429, 0.0, 0.050), 0.220, 1.82, 0.028, m["carbon"], incidence_deg=1.0))

    add_local(scene, "body_lower_shell", _joined_exterior(m["white"]))
    # Retain the public node contract: these describe positions on the single
    # joined exterior rather than separate cosmetic body pieces.
    add_anchor(scene, "nose_center", (1.95, 0.0, 0.48))
    add_anchor(scene, "rear_engine_cover", (-1.55, 0.0, 0.53))

    front_x = FRONT_AXLE_X_M
    rear_x = REAR_AXLE_X_M

    # Compact bubble canopy, black roof, and the 919's stacked airbox.
    add_local(scene, "cockpit_canopy", canopy_shell([
        (0.73, 0.08, 0.48, 0.55),
        (0.50, 0.31, 0.48, 0.75),
        (0.16, 0.39, 0.48, 0.91),
        (-0.18, 0.39, 0.47, 0.975),
        (-0.50, 0.32, 0.46, 0.89),
        (-0.77, 0.08, 0.45, 0.58),
    ], m["glass"], lateral_segments=41))
    add_local(scene, "roof_shell", body_shell([
        (0.34, 0.26, 0.78, 0.82, 0.93, 1.7),
        (-0.08, 0.30, 0.82, 0.87, 1.005, 1.7),
        (-0.46, 0.22, 0.76, 0.82, 0.94, 1.6),
    ], m["black"], lateral_segments=25))
    add_local(scene, "roof_airbox_upper", body_shell([
        (-0.18, 0.10, 0.93, 0.95, 1.020, 1.5),
        (-0.38, 0.13, 0.94, 0.98, HEIGHT_M, 1.5),
        (-0.63, 0.075, 0.90, 0.92, 0.98, 1.5),
    ], m["white"], lateral_segments=17))
    add_local(scene, "roof_airbox_mouth", box((0.025, 0.18, 0.062), (-0.185, 0.0, 1.005), m["black"]))
    add_local(scene, "roof_airbox_lower_mouth", box((0.030, 0.15, 0.043), (-0.10, 0.0, 0.934), m["black"]))

    # Long shark fin and massive active rear wing identify the unrestricted Evo.
    add_local(scene, "shark_fin", quad([
        (-0.42, 0.0, 0.96), (-2.08, 0.0, 0.52), (-2.08, 0.0, 0.83), (-0.48, 0.0, 1.02),
    ], m["white"]))
    add_local(scene, "aero_rear_wing", airfoil((-2.438, 0.0, 0.875), 0.202, WIDTH_M, 0.078, m["carbon"], incidence_deg=-4.0))
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        add_local(scene, f"aero_rear_endplate_{side_name}", box((0.23, 0.025, 0.43), (-2.415, side * 0.9375, 0.77), m["carbon"]))
        add_local(scene, f"aero_rear_pylon_{side_name}", oriented_cylinder((-2.05, side * 0.49, 0.42), (-2.39, side * 0.58, 0.84), 0.020, m["carbon"], sections=12))
    add_local(scene, "aero_front_flap_left", airfoil((2.22, 0.53, 0.18), 0.48, 0.34, 0.032, m["carbon"], incidence_deg=5.0))
    add_local(scene, "aero_front_flap_right", airfoil((2.22, -0.53, 0.18), 0.48, 0.34, 0.032, m["carbon"], incidence_deg=5.0))

    # Nose tunnels, low round lamps, sidepod ducts and mirrors.
    add_local(scene, "nose_lower_blackout", body_shell([
        (2.385, 0.74, 0.075, 0.105, 0.22, 1.55),
        (2.22, 0.78, 0.075, 0.12, 0.27, 1.62),
        (1.98, 0.75, 0.075, 0.13, 0.30, 1.65),
    ], m["black"], lateral_segments=38))
    add_local(scene, "nose_center_grille", box((0.026, 0.40, 0.060), (2.405, 0.0, 0.175), m["metal"]))
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        add_local(scene, f"headlight_housing_{side_name}", cylinder(0.092, 0.032, (2.170, side * 0.73, 0.235), m["carbon"], axis="x", sections=48))
        add_local(scene, f"headlight_{side_name}", cylinder(0.060, 0.036, (2.193, side * 0.73, 0.235), m["glass"], axis="x", sections=40))
        add_local(scene, f"sidepod_intake_{side_name}", box((0.36, 0.028, 0.17), (-0.30, side * 0.70, 0.39), m["black"]))
        add_local(scene, f"mirror_stalk_{side_name}", oriented_cylinder((0.36, side * 0.60, 0.64), (0.44, side * 0.82, 0.72), 0.012, m["carbon"], sections=10))
        add_local(scene, f"mirror_{side_name}", ellipsoid((0.105, 0.055, 0.050), (0.45, side * 0.84, 0.73), m["white"], 2))
        add_local(scene, f"side_marker_{side_name}", box((0.12, 0.016, 0.030), (1.05, side * 0.940, 0.39), m["amber"]))

    wheel_centers = {
        "fl": (front_x, FRONT_TRACK_M * 0.5, 0.355),
        "fr": (front_x, -FRONT_TRACK_M * 0.5, 0.355),
        "rl": (rear_x, REAR_TRACK_M * 0.5, 0.355),
        "rr": (rear_x, -REAR_TRACK_M * 0.5, 0.355),
    }
    for corner, center in wheel_centers.items():
        _add_wheel(scene, corner, center, m)

    add_local(scene, "rear_light", box((0.030, 0.53, 0.045), (-2.305, 0.0, 0.34), m["tail"]))
    add_local(scene, "exhaust_rear", cylinder(0.060, 0.19, (-2.30, 0.0, 0.22), m["metal"], axis="x", sections=24))
    for side in (-1.0, 1.0):
        for index in range(4):
            add_local(scene, f"rear_diffuser_{'left' if side > 0 else 'right'}_{index}", box((0.68, 0.020, 0.105), (-2.00, side * (0.16 + index * 0.17), 0.105), m["carbon"]))

    # Project-authored recreation of the 2018 tribute graphics.
    add_local(scene, "livery_nose_number", quad([
        (1.55, -0.22, 0.485), (1.55, 0.22, 0.485), (1.02, 0.24, 0.555), (1.02, -0.24, 0.555),
    ], m["atlas"], uv_rect=uv["number"]))
    add_local(scene, "livery_windscreen_header", quad([
        (0.59, -0.30, 0.77), (0.59, 0.30, 0.77), (0.42, 0.32, 0.87), (0.42, -0.32, 0.87),
    ], m["atlas"], uv_rect=uv["porsche_black"]))
    add_local(scene, "livery_nose_chopard", quad([
        (0.94, -0.28, 0.56), (0.94, 0.28, 0.56), (0.48, 0.29, 0.60), (0.48, -0.29, 0.60),
    ], m["atlas"], uv_rect=uv["chopard"]))
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        # Stand the flank graphics off the sidepod. At the authored 0.718 the
        # plane sits inside the bodywork over part of its span, so the wordmark
        # rendered chopped into fragments; the body half-width is 0.95, so this
        # still sits well inboard of the fender line.
        y = side * 0.773
        side_points = [(0.32, y, 0.30), (-1.22, y, 0.30), (-1.22, y, 0.55), (0.32, y, 0.55)]
        if side < 0:
            side_points = [side_points[1], side_points[0], side_points[3], side_points[2]]
        add_local(scene, f"livery_side_tribute_{side_name}", quad(side_points, m["atlas"], uv_rect=uv["side"]))
        ribbon_y = side * 0.785
        ribbon_points = [(0.62, ribbon_y, 0.27), (-1.54, ribbon_y, 0.27), (-1.54, ribbon_y, 0.36), (0.62, ribbon_y, 0.36)]
        if side < 0:
            ribbon_points = [ribbon_points[1], ribbon_points[0], ribbon_points[3], ribbon_points[2]]
        add_local(scene, f"livery_tribute_ribbon_{side_name}", quad(ribbon_points, m["atlas"], uv_rect=uv["tribute"]))
    add_local(scene, "livery_rear_wing", quad([
        (-2.536, -0.79, 0.84), (-2.536, 0.79, 0.84), (-2.536, 0.79, 0.93), (-2.536, -0.79, 0.93),
    ], m["atlas"], uv_rect=uv["wing"], flip_u=True))

    for name, point in {
        "camera_anchor_chase": (-2.10, 0.0, 1.38),
        "camera_anchor_roof": (-0.12, 0.0, 1.14),
        "audio_anchor_rear_ice": (-0.82, 0.0, 0.53),
        "audio_anchor_turbo": (-1.15, 0.0, 0.56),
        "audio_anchor_exhaust": (-2.35, 0.0, 0.22),
        "audio_anchor_front_mgu": (front_x, 0.0, 0.33),
    }.items():
        add_anchor(scene, name, point)

    scene.metadata.update({
        "asset_id": ASSET_ID,
        "vehicle_id": "porsche_919evo",
        "coordinate_system": "x-forward_y-left_z-up_metres",
        "authorship": "clean-room authored section model",
        "reference_basis": "published dimensions plus photographic proportion study",
        "physics_authority": False,
    })
    return scene


def _png(image: Image.Image) -> bytes:
    payload = io.BytesIO()
    image.save(payload, format="PNG", compress_level=9, optimize=False)
    return payload.getvalue()


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _record(path: str, payload: bytes, **extra) -> dict:
    return {"path": path, "sha256": _sha(payload), "bytes": len(payload), **extra}


def _json(data: dict) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_outputs() -> dict[str, bytes]:
    _validate_visual_clearances()
    atlas, uv = livery_atlas()
    scene = build_scene(atlas, uv)
    triangles = mesh_triangle_count(scene)
    if triangles > 80_000:
        raise RuntimeError(f"Porsche triangle budget exceeded: {triangles}")
    extents = bounds_extents(scene)
    expected = np.asarray((LENGTH_M, WIDTH_M, HEIGHT_M), dtype=float)
    if not np.allclose(extents, expected, atol=0.004, rtol=0.0):
        raise RuntimeError(f"Porsche extents {extents.tolist()} differ from official Evo envelope {expected.tolist()}")
    # Crease-aware vertex normals; without these the GLB ships no NORMAL
    # attribute and glTF forces flat shading on every surface.
    shade_scene(scene)
    glb = bytes(scene.export(file_type="glb"))
    atlas_bytes = _png(atlas)
    canonical_nodes = {
        "body": ["body_lower_shell", "nose_center", "rear_engine_cover", "cockpit_canopy", "roof_airbox_upper"],
        "wheels": ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"],
        "suspension_pivots": ["suspension_pivot_fl", "suspension_pivot_fr", "suspension_pivot_rl", "suspension_pivot_rr"],
        "brakes": ["brake_disc_fl", "brake_disc_fr", "brake_disc_rl", "brake_disc_rr"],
        "active_aero": ["aero_front_flap_left", "aero_front_flap_right", "aero_rear_wing"],
        "lights": ["headlight_left", "headlight_right", "rear_light", "side_marker_left", "side_marker_right"],
        "exhaust": ["exhaust_rear"],
        "cameras": ["camera_anchor_chase", "camera_anchor_roof"],
        "audio": ["audio_anchor_rear_ice", "audio_anchor_turbo", "audio_anchor_exhaust", "audio_anchor_front_mgu"],
    }
    exported = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    missing = sorted(set(sum(canonical_nodes.values(), [])) - scene_nodes(exported))
    if missing:
        raise RuntimeError(f"Porsche exported GLB is missing named nodes: {missing}")
    generator_bytes = Path(__file__).read_bytes()
    manifest = {
        "schema": "supra-car-asset-manifest-v2",
        "asset_id": ASSET_ID,
        "vehicle_id": "porsche_919evo",
        "display_name": "Porsche 919 Hybrid Evo",
        "status": "project-authored-visual-reconstruction",
        "detail_level": "observatory-hero",
        "runtime_detail_kit": False,
        "faithful_geometry_eligible": False,
        "licence": {
            "use": "private-research",
            "copyright": "Original deterministic geometry and texture authored for this project",
            "redistribution": "Project-generated assets only; no official or third-party media is embedded",
        },
        "provenance": {
            "method": "Measured section lofts, one joined exterior with wheel humps and open side bays, native wheel assemblies, authored aero surfaces, mechanical details, and project-rendered typography.",
            "official_media_embedded": False,
            "oem_cad_claim": False,
            "scan_claim": False,
            "prior_observatory_geometry_used": False,
            "accuracy_boundary": "Recognizable visual reconstruction, not OEM CAD, surveyed geometry, faithful-v2 simulation, or certification evidence.",
            "official_dimension_reference": "https://newsroom.porsche.com/en/motorsports/porsche-919-hybrid-evo-top-5-series-technical-check-16834.html",
            "visual_reference_note": "Porsche and period track photography studied externally; no pixels embedded.",
        },
        "coordinate_system": {"units": "m", "handedness": "right", "forward_axis": "+x", "left_axis": "+y", "up_axis": "+z", "origin": "vehicle longitudinal center at ground plane"},
        "visual_ground_contact": {
            "schema": "supra-observatory-ground-contact-v1",
            "ground_anchor_offset_m": 0.008,
            "contact_tolerance_m": 0.012,
            "wheel_nodes": {
                "wheel_fl": [FRONT_AXLE_X_M, FRONT_TRACK_M * 0.5, 0.355],
                "wheel_fr": [FRONT_AXLE_X_M, -FRONT_TRACK_M * 0.5, 0.355],
                "wheel_rl": [REAR_AXLE_X_M, REAR_TRACK_M * 0.5, 0.355],
                "wheel_rr": [REAR_AXLE_X_M, -REAR_TRACK_M * 0.5, 0.355],
            },
            "wheel_radii_m": {"wheel_fl": 0.355, "wheel_fr": 0.355, "wheel_rl": 0.355, "wheel_rr": 0.355},
            "policy": "Visual contact calibration only; it never changes collision, physics, reward, observations, or policy inference.",
        },
        "canonical_dimensions_m": {"length": LENGTH_M, "base_body_length": BASE_BODY_LENGTH_M, "width": WIDTH_M, "height": HEIGHT_M, "wheelbase": WHEELBASE_M, "front_track": FRONT_TRACK_M, "rear_track": REAR_TRACK_M, "allowed_relative_error": 0.005},
        "budgets": {"triangle_limit": 80000, "triangle_count": triangles, "texture_max_px": 2048, "texture_dimensions_px": [2048, 2048]},
        "files": {
            "glb": _record(ENTRY_GLB, glb),
            "texture": _record(ENTRY_TEXTURE, atlas_bytes, width=2048, height=2048, role="project-authored tribute livery and decal atlas embedded in GLB"),
            "generator": {"path": "build_asset.py", "sha256": _sha(generator_bytes), "bytes": len(generator_bytes), "implementation": f"Python + trimesh {trimesh.__version__} + Pillow"},
        },
        "canonical_nodes": canonical_nodes,
        "articulation_points_m": {
            "wheel_fl": [FRONT_AXLE_X_M, FRONT_TRACK_M * 0.5, 0.355],
            "wheel_fr": [FRONT_AXLE_X_M, -FRONT_TRACK_M * 0.5, 0.355],
            "wheel_rl": [REAR_AXLE_X_M, REAR_TRACK_M * 0.5, 0.355],
            "wheel_rr": [REAR_AXLE_X_M, -REAR_TRACK_M * 0.5, 0.355],
            "aero_front_flap_left": [2.22, 0.53, 0.18],
            "aero_front_flap_right": [2.22, -0.53, 0.18],
            "aero_rear_wing": [-2.438, 0.0, 0.875],
        },
        "audio_source_points_m": {
            "rear_ice": [-0.82, 0.0, 0.53], "turbo": [-1.15, 0.0, 0.56],
            "exhaust_rear": [-2.35, 0.0, 0.22], "front_mgu": [FRONT_AXLE_X_M, 0.0, 0.33],
            "gearbox": [-1.64, 0.0, 0.39],
        },
        "binding_policy": {
            "visual": "Animate named nodes only from present telemetry; absent channels freeze at neutral.",
            "audio": "Emit source channels only from authoritative telemetry.",
            "collision": "The Fable collision footprint remains separate; this visual reconstruction is not collision or certification authority.",
        },
    }
    return {ENTRY_GLB: glb, ENTRY_TEXTURE: atlas_bytes, "asset_manifest.json": _json(manifest)}


def build_glb() -> bytes:
    return build_outputs()[ENTRY_GLB]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    for name, payload in outputs.items():
        path = args.output if name == ENTRY_GLB else args.output.parent / name
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                raise SystemExit(f"asset is not reproducible: {path}")
        else:
            path.write_bytes(payload)
    payload = outputs[ENTRY_GLB]
    print(f"{args.output}: {len(payload)} bytes sha256={_sha(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
