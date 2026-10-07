#!/usr/bin/env python3
"""Build the clean-room Mazda 787B #55 Renown visual reconstruction.

Geometry is authored from dimensional data and section profiles.  Real-world
photographs are proportion and detail references only; no photograph, CAD,
scan, third-party mesh, or old Observatory geometry is incorporated.
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
    arch_shell,
    body_shell,
    bounds_extents,
    box,
    canopy_shell,
    cylinder,
    ellipsoid,
    mesh_triangle_count,
    oriented_cylinder,
    paint,
    pbr,
    quad,
    radial_spoke,
    rim_ring,
    scene_nodes,
    tyre,
)


ASSET_ID = "mazda787b_renown_cleanroom_v2"
ENTRY_GLB = "mazda787b_renown_cleanroom_v2.glb"
ENTRY_TEXTURE = "mazda787b_renown_atlas_v2.png"
LENGTH_M = 4.782
WIDTH_M = 1.994
HEIGHT_M = 1.003
WHEELBASE_M = 2.662
FRONT_TRACK_M = 1.534
REAR_TRACK_M = 1.504
DEFAULT_OUTPUT = HERE / ENTRY_GLB


ORANGE = (222, 50, 11, 255)
GREEN = (4, 72, 45, 255)
CREAM = (246, 238, 202, 255)
BLACK = (13, 16, 17, 255)
CARBON = (22, 26, 27, 255)
GLASS = (20, 38, 46, 214)
RUBBER = (12, 13, 13, 255)
GOLD = (176, 134, 47, 255)
METAL = (137, 143, 142, 255)
BRAKE = (66, 64, 60, 255)
CALIPER = (191, 30, 24, 255)
HEADLIGHT = (226, 239, 224, 255)
RED = (230, 27, 20, 255)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def livery_atlas() -> tuple[Image.Image, dict[str, tuple[float, float, float, float]]]:
    size = 2048
    image = Image.new("RGB", (size, size), GREEN[:3])
    draw = ImageDraw.Draw(image)
    tiles_px = {
        "nose": (0, 0, 1024, 512),
        "number": (1024, 0, 1536, 512),
        "charge": (1536, 0, 2048, 512),
        "side": (0, 512, 1024, 1024),
        "mazda": (1024, 512, 2048, 768),
        "wing": (1024, 768, 2048, 1024),
        "small": (0, 1024, 2048, 1280),
        "argyle": (0, 1280, 2048, 2048),
    }

    def center_text(rect, text, font_size, fill, stroke=0, stroke_fill=BLACK[:3]):
        x0, y0, x1, y1 = rect
        draw.text(
            ((x0 + x1) * 0.5, (y0 + y1) * 0.5), text,
            font=_font(font_size), fill=fill, anchor="mm",
            stroke_width=stroke, stroke_fill=stroke_fill,
        )

    # Nose: the famous orange field, cream diamonds, and RENOWN word mark.
    nose = tiles_px["nose"]
    draw.rectangle(nose, fill=ORANGE[:3])
    for x in range(80, 980, 150):
        draw.polygon(((x, 44), (x + 62, 96), (x, 148), (x - 62, 96)), fill=CREAM[:3])
        draw.polygon(((x + 75, 120), (x + 137, 172), (x + 75, 224), (x + 13, 172)), fill=GREEN[:3])
    center_text((0, 150, 1024, 470), "RENOWN", 178, CREAM[:3], 10, BLACK[:3])

    draw.rectangle(tiles_px["number"], fill=ORANGE[:3])
    draw.ellipse((1070, 42, 1490, 462), fill=CREAM[:3], outline=BLACK[:3], width=18)
    center_text(tiles_px["number"], "55", 248, BLACK[:3])

    draw.rectangle(tiles_px["charge"], fill=GREEN[:3])
    center_text((1536, 12, 2048, 322), "CHARGE", 94, CREAM[:3], 4, BLACK[:3])
    center_text((1536, 282, 2048, 492), "CIBIE", 62, (240, 214, 47), 3, BLACK[:3])

    draw.rectangle(tiles_px["side"], fill=GREEN[:3])
    draw.polygon(((0, 512), (520, 512), (1024, 1024), (660, 1024)), fill=ORANGE[:3])
    for x in (560, 690, 820, 950):
        draw.polygon(((x, 542), (x + 55, 592), (x, 642), (x - 55, 592)), fill=CREAM[:3])
    center_text((30, 590, 810, 990), "RENOWN", 116, CREAM[:3], 7, BLACK[:3])

    draw.rectangle(tiles_px["mazda"], fill=ORANGE[:3])
    center_text(tiles_px["mazda"], "mazda", 128, BLACK[:3])

    draw.rectangle(tiles_px["wing"], fill=BLACK[:3])
    center_text(tiles_px["wing"], "MAZDASPEED", 104, CREAM[:3])

    draw.rectangle(tiles_px["small"], fill=GREEN[:3])
    center_text((0, 1024, 520, 1280), "DUNLOP", 82, CREAM[:3])
    center_text((520, 1024, 1000, 1280), "NGK", 82, CREAM[:3])
    center_text((1000, 1024, 1480, 1280), "BREMBO", 68, CREAM[:3])
    center_text((1480, 1024, 2048, 1280), "RAYS", 78, CREAM[:3])

    draw.rectangle(tiles_px["argyle"], fill=ORANGE[:3])
    for row, y in enumerate(range(1340, 2000, 168)):
        for x in range(-80 + (row % 2) * 84, 2120, 168):
            draw.polygon(((x, y), (x + 74, y + 74), (x, y + 148), (x - 74, y + 74)), fill=CREAM[:3])
            draw.line(((x - 74, y + 74), (x, y), (x + 74, y + 74), (x, y + 148), (x - 74, y + 74)), fill=BLACK[:3], width=8)

    # UVs use a bottom-left origin while Pillow uses top-left.
    uv: dict[str, tuple[float, float, float, float]] = {}
    for name, (x0, y0, x1, y1) in tiles_px.items():
        uv[name] = (x0 / size, 1.0 - y1 / size, x1 / size, 1.0 - y0 / size)
    return image, uv


def _materials(atlas: Image.Image) -> dict[str, trimesh.visual.material.PBRMaterial]:
    return {
        "orange": pbr("renown_orange_paint", ORANGE, roughness=0.27),
        "green": pbr("renown_green_paint", GREEN, roughness=0.29),
        "cream": pbr("renown_cream_paint", CREAM, roughness=0.3),
        "carbon": pbr("carbon_fibre_aero", CARBON, metallic=0.25, roughness=0.42),
        "black": pbr("satin_black_trim", BLACK, roughness=0.48),
        "glass": pbr("smoked_polycarbonate_glass", GLASS, metallic=0.08, roughness=0.08),
        "rubber": pbr("dunlop_slick_rubber", RUBBER, roughness=0.88),
        "gold": pbr("rays_magnesium_gold", GOLD, metallic=0.78, roughness=0.24),
        "metal": pbr("machined_metal", METAL, metallic=0.84, roughness=0.22),
        "brake": pbr("carbon_brake_disc", BRAKE, metallic=0.12, roughness=0.72),
        "caliper": pbr("brembo_caliper_red", CALIPER, metallic=0.22, roughness=0.34),
        "light": pbr("cibie_headlamp_lens", HEADLIGHT, roughness=0.08),
        "red": pbr("rear_lamp_red", RED, roughness=0.16),
        "atlas": pbr("renown_livery_atlas", (255, 255, 255, 255), roughness=0.28, image=atlas),
    }


def _add_wheel(
    scene: trimesh.Scene,
    corner: str,
    center: tuple[float, float, float],
    outer_radius: float,
    width: float,
    materials: dict[str, trimesh.visual.material.PBRMaterial],
) -> None:
    rim_radius = outer_radius * 0.62
    outside = 1.0 if center[1] > 0 else -1.0
    pivot = center
    add_local(scene, f"wheel_{corner}", tyre(outer_radius, rim_radius, width, center, materials["rubber"]), pivot=pivot)
    face_y = center[1] + outside * (width * 0.5 + 0.006)
    add_local(
        scene, f"wheel_{corner}__rim_barrel",
        rim_ring(rim_radius * 0.98, rim_radius * 0.66, 0.046, (center[0], face_y, center[2]), materials["gold"]),
        pivot=pivot,
    )
    add_local(
        scene, f"wheel_{corner}__hub",
        cylinder(0.052, 0.050, (center[0], face_y + outside * 0.006, center[2]), materials["metal"], axis="y", sections=32),
        pivot=pivot,
    )
    for index in range(10):
        angle = index * math.tau / 10.0
        spoke_center = (center[0], face_y + outside * 0.011, center[2])
        spoke = radial_spoke(spoke_center, rim_radius * 0.83, angle, 0.022, 0.026, materials["gold"])
        add_local(scene, f"wheel_{corner}__spoke_{index:02d}", spoke, pivot=pivot)
    add_local(
        scene, f"brake_disc_{corner}",
        cylinder(rim_radius * 0.72, 0.018, (center[0], face_y - outside * 0.035, center[2]), materials["brake"], axis="y", sections=48),
    )
    caliper_x = center[0] - outer_radius * 0.24
    add_local(
        scene, f"brake_caliper_{corner}",
        box((0.085, 0.045, 0.19), (caliper_x, face_y - outside * 0.052, center[2]), materials["caliper"]),
    )
    inboard_y = 0.54 if center[1] > 0 else -0.54
    add_local(scene, f"suspension_pivot_{corner}", ellipsoid((0.025, 0.025, 0.025), (center[0], inboard_y, center[2] + 0.05), materials["metal"], 1))
    for suffix, dz in (("upper", 0.12), ("lower", -0.10)):
        start = (center[0], inboard_y, center[2] + dz)
        end = (center[0], center[1] - outside * width * 0.44, center[2] + dz * 0.55)
        add_local(scene, f"suspension_arm_{corner}_{suffix}", oriented_cylinder(start, end, 0.012, materials["metal"], sections=10))


def build_scene(atlas: Image.Image, uv: dict[str, tuple[float, float, float, float]]) -> trimesh.Scene:
    m = _materials(atlas)
    scene = trimesh.Scene(base_frame="vehicle_origin")

    # Exact official envelope: x-forward, y-left, z-up, metres.
    add_local(scene, "body_undertray", box((4.54, 1.67, 0.050), (-0.01, 0.0, 0.025), m["carbon"]))
    add_local(scene, "aero_front_splitter", airfoil((2.301, 0.0, 0.050), 0.180, 1.84, 0.026, m["carbon"], incidence_deg=1.0))

    # Low green floor/body, then the orange central nose and engine spine.
    add_local(scene, "body_lower_shell", body_shell([
        (2.28, 0.30, 0.08, 0.14, 0.22, 1.45),
        (2.02, 0.50, 0.08, 0.18, 0.30, 1.55),
        (1.38, 0.58, 0.08, 0.21, 0.36, 1.62),
        (0.72, 0.61, 0.08, 0.23, 0.41, 1.68),
        (-0.20, 0.60, 0.08, 0.24, 0.43, 1.72),
        (-1.18, 0.61, 0.08, 0.22, 0.41, 1.62),
        (-1.86, 0.56, 0.08, 0.18, 0.34, 1.52),
        (-2.27, 0.42, 0.08, 0.13, 0.24, 1.42),
    ], m["green"], lateral_segments=38))
    add_local(scene, "body_orange_spine", body_shell([
        (2.24, 0.22, 0.25, 0.29, 0.34, 1.65),
        (1.72, 0.46, 0.26, 0.34, 0.48, 1.68),
        (0.92, 0.50, 0.27, 0.38, 0.58, 1.72),
        (0.16, 0.45, 0.28, 0.39, 0.61, 1.75),
        (-0.78, 0.47, 0.27, 0.37, 0.57, 1.68),
        (-1.56, 0.50, 0.23, 0.31, 0.48, 1.58),
        (-2.08, 0.38, 0.18, 0.24, 0.35, 1.48),
    ], m["orange"], lateral_segments=34))

    front_x = 1.335
    rear_x = front_x - WHEELBASE_M
    for axle, x, y_center, width, length, crown in (
        ("front", front_x, 0.714, 0.280, 1.22, 0.565),
        ("rear", rear_x, 0.700, 0.294, 1.34, 0.600),
    ):
        for side_name, side in (("left", 1.0), ("right", -1.0)):
            y = side * y_center
            add_local(
                scene, f"fender_{axle}_{side_name}",
                arch_shell((x, y, 0.0), length, width, 0.255, crown, m["green"], longitudinal_segments=48, arc_segments=26, end_scale=0.075),
            )
            for vent_index in range(4):
                vent_x = x - 0.31 + vent_index * 0.15
                add_local(scene, f"fender_vent_{axle}_{side_name}_{vent_index}", box((0.105, 0.024, 0.016), (vent_x, side * (abs(y) + width * 0.63), crown - 0.028), m["black"]))

    # Cockpit proportions follow the narrow, forward-biased Group-C canopy.
    add_local(scene, "cockpit_canopy", canopy_shell([
        (0.76, 0.08, 0.47, 0.54),
        (0.54, 0.30, 0.47, 0.73),
        (0.18, 0.39, 0.48, 0.90),
        (-0.16, 0.385, 0.47, 0.955),
        (-0.50, 0.31, 0.46, 0.87),
        (-0.78, 0.08, 0.45, 0.58),
    ], m["glass"], lateral_segments=41))
    add_local(scene, "windscreen_header", box((0.055, 0.69, 0.030), (0.52, 0.0, 0.735), m["orange"]))
    add_local(scene, "canopy_center_frame", box((1.20, 0.028, 0.025), (-0.02, 0.0, 0.955), m["black"]))
    add_local(scene, "roof_intake", body_shell([
        (-0.18, 0.06, 0.91, 0.92, 0.96, 1.5),
        (-0.39, 0.09, 0.91, 0.94, HEIGHT_M, 1.5),
        (-0.64, 0.04, 0.88, 0.90, 0.94, 1.5),
    ], m["orange"], lateral_segments=14))
    add_local(scene, "roof_intake_mouth", box((0.020, 0.12, 0.050), (-0.185, 0.0, 0.955), m["black"]))

    # Side cooling intakes and the distinctive blunt nose radiator opening.
    add_local(scene, "radiator_inlet", box((0.032, 0.42, 0.105), (2.370, 0.0, 0.205), m["black"]))
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        add_local(scene, f"side_intake_{side_name}", box((0.44, 0.026, 0.18), (-0.63, side * 0.735, 0.38), m["black"]))
        add_local(scene, f"mirror_stalk_{side_name}", oriented_cylinder((0.42, side * 0.64, 0.61), (0.50, side * 0.84, 0.72), 0.014, m["carbon"], sections=10))
        add_local(scene, f"mirror_{side_name}", ellipsoid((0.12, 0.065, 0.055), (0.51, side * 0.86, 0.73), m["orange"], 2))

    # Four Cibié lamps beneath low, smoked rectangular covers.  The original
    # car's lamps sit in the fender deck rather than projecting from the nose.
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        cover_points = [
            (1.94, side * 0.59, 0.382), (1.94, side * 0.81, 0.382),
            (1.62, side * 0.84, 0.472), (1.62, side * 0.60, 0.472),
        ]
        if side < 0:
            cover_points = [cover_points[1], cover_points[0], cover_points[3], cover_points[2]]
        add_local(scene, f"headlight_cover_{side_name}", quad(cover_points, m["glass"]))
        for inner, y_offset in (("inner", -side * 0.052), ("outer", side * 0.052)):
            lamp = ellipsoid((0.066, 0.038, 0.024), (1.79, side * 0.715 + y_offset, 0.482), m["light"], 2)
            name = f"headlight_{side_name}" if inner == "inner" else f"headlight_{side_name}_outer"
            add_local(scene, name, lamp)

    # Accurate wheel/tire sizes and tracks from Mazda's published specification.
    wheel_data = {
        "fl": (front_x, FRONT_TRACK_M * 0.5, 0.320, 0.320, 0.300),
        "fr": (front_x, -FRONT_TRACK_M * 0.5, 0.320, 0.320, 0.300),
        "rl": (rear_x, REAR_TRACK_M * 0.5, 0.355, 0.355, 0.355),
        "rr": (rear_x, -REAR_TRACK_M * 0.5, 0.355, 0.355, 0.355),
    }
    for corner, (x, y, z, radius, width) in wheel_data.items():
        _add_wheel(scene, corner, (x, y, z), radius, width, m)

    # Rear wing, endplates, tail lamps, diffuser, and twin exhaust outlets.
    add_local(scene, "aero_rear_wing", airfoil((-2.30, 0.0, 0.835), 0.182, WIDTH_M, 0.072, m["carbon"], incidence_deg=-3.0))
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        add_local(scene, f"aero_rear_endplate_{side_name}", box((0.20, 0.022, 0.34), (-2.29, side * (WIDTH_M * 0.5 - 0.011), 0.72), m["carbon"]))
        add_local(scene, f"aero_rear_pylon_{side_name}", oriented_cylinder((-2.08, side * 0.47, 0.43), (-2.28, side * 0.52, 0.80), 0.018, m["carbon"], sections=12))
    add_local(scene, "rear_grille", box((0.030, 1.18, 0.20), (-2.31, 0.0, 0.28), m["black"]))
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        for index, y in enumerate((0.42, 0.28)):
            add_local(scene, f"rear_light_{side_name}_{index}", cylinder(0.048, 0.026, (-2.337, side * y, 0.34), m["red"], axis="x", sections=32))
    add_local(scene, "exhaust_left", cylinder(0.052, 0.18, (-2.25, 0.18, 0.19), m["metal"], axis="x", sections=24))
    add_local(scene, "exhaust_right", cylinder(0.052, 0.18, (-2.25, -0.18, 0.19), m["metal"], axis="x", sections=24))
    for side in (-1.0, 1.0):
        for index in range(4):
            add_local(scene, f"rear_diffuser_{'left' if side > 0 else 'right'}_{index}", box((0.62, 0.018, 0.10), (-2.01, side * (0.18 + index * 0.16), 0.105), m["carbon"]))

    # Paint graphics sit on authored surfaces; the photo references themselves
    # are never packed into the asset.
    add_local(scene, "livery_nose_renown", quad([
        (2.13, -0.37, 0.355), (2.13, 0.37, 0.355), (0.82, 0.40, 0.585), (0.82, -0.40, 0.585),
    ], m["atlas"], uv_rect=uv["nose"]))
    add_local(scene, "livery_nose_number", quad([
        (1.28, -0.18, 0.545), (1.28, 0.18, 0.545), (0.90, 0.18, 0.610), (0.90, -0.18, 0.610),
    ], m["atlas"], uv_rect=uv["number"]))
    for side_name, side in (("left", 1.0), ("right", -1.0)):
        y = side * 0.746
        points = [(0.72, y, 0.31), (-0.62, y, 0.31), (-0.62, y, 0.52), (0.72, y, 0.52)]
        if side < 0:
            points = [points[1], points[0], points[3], points[2]]
        add_local(scene, f"livery_side_renown_{side_name}", quad(points, m["atlas"], uv_rect=uv["side"]))
        number_y = side * 0.758
        number_points = [(0.10, number_y, 0.36), (-0.38, number_y, 0.36), (-0.38, number_y, 0.63), (0.10, number_y, 0.63)]
        if side < 0:
            number_points = [number_points[1], number_points[0], number_points[3], number_points[2]]
        add_local(scene, f"livery_side_number_{side_name}", quad(number_points, m["atlas"], uv_rect=uv["number"]))
    add_local(scene, "livery_wing_wordmark", quad([
        (-2.389, -0.73, 0.79), (-2.389, 0.73, 0.79), (-2.389, 0.73, 0.89), (-2.389, -0.73, 0.89),
    ], m["atlas"], uv_rect=uv["wing"], flip_u=True))

    for name, point in {
        "camera_anchor_chase": (-2.03, 0.0, 1.36),
        "camera_anchor_roof": (-0.08, 0.0, 1.08),
        "audio_anchor_engine": (-0.88, 0.0, 0.47),
        "audio_anchor_exhaust": (-2.36, 0.0, 0.19),
    }.items():
        add_anchor(scene, name, point)

    scene.metadata.update({
        "asset_id": ASSET_ID,
        "vehicle_id": "mazda787b",
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
    atlas, uv = livery_atlas()
    scene = build_scene(atlas, uv)
    triangles = mesh_triangle_count(scene)
    if triangles > 80_000:
        raise RuntimeError(f"Mazda triangle budget exceeded: {triangles}")
    extents = bounds_extents(scene)
    expected = np.asarray((LENGTH_M, WIDTH_M, HEIGHT_M), dtype=float)
    if not np.allclose(extents, expected, atol=0.003, rtol=0.0):
        raise RuntimeError(f"Mazda extents {extents.tolist()} differ from official envelope {expected.tolist()}")
    glb = bytes(scene.export(file_type="glb"))
    atlas_bytes = _png(atlas)
    named_nodes = {
        "body": ["body_lower_shell", "body_orange_spine", "cockpit_canopy", "roof_intake"],
        "wheels": ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"],
        "suspension_pivots": ["suspension_pivot_fl", "suspension_pivot_fr", "suspension_pivot_rl", "suspension_pivot_rr"],
        "brakes": ["brake_disc_fl", "brake_disc_fr", "brake_disc_rl", "brake_disc_rr"],
        "aero": ["aero_front_splitter", "aero_rear_wing"],
        "lights": ["headlight_left", "headlight_right", "rear_light_left_0", "rear_light_right_0"],
        "exhaust": ["exhaust_left", "exhaust_right"],
        "cameras": ["camera_anchor_chase", "camera_anchor_roof"],
        "audio": ["audio_anchor_engine", "audio_anchor_exhaust"],
    }
    exported = trimesh.load(io.BytesIO(glb), file_type="glb", force="scene")
    missing = sorted(set(sum(named_nodes.values(), [])) - scene_nodes(exported))
    if missing:
        raise RuntimeError(f"Mazda exported GLB is missing named nodes: {missing}")
    provenance = {
        "schema": "supra-observatory-provenance-v2",
        "asset_id": ASSET_ID,
        "authorship": "Original deterministic clean-room mesh and livery authored for this project.",
        "method": "Measured section lofts, open wheel-arch shells, authored aero surfaces, wheels, lamps, cockpit, and decals.",
        "references": {
            "dimensions": "https://www.mazda.com/en/experience/mspr/motorsports/lemans/mazda787b/",
            "visual_primary": "Mazda 787B #55 museum and period photography studied externally; no pixels embedded.",
        },
        "exclusions": {
            "oem_cad_or_scan": False,
            "official_or_third_party_media_embedded": False,
            "prior_viewer_geometry_or_texture_used": False,
            "network_required": False,
        },
        "classification": "project-authored-visual-reconstruction",
        "accuracy_boundary": "Recognizable visual reconstruction, not OEM CAD, surveyed geometry, or collision authority.",
    }
    provenance_bytes = _json(provenance)
    generator_bytes = Path(__file__).read_bytes()
    manifest = {
        "schema": "supra-observatory-vehicle-v2",
        "asset_id": ASSET_ID,
        "vehicle_id": "mazda787b",
        "display_name": "Mazda 787B #55 Renown",
        "entry_glb": ENTRY_GLB,
        "classification": "project-authored-visual-reconstruction",
        "detail_level": "observatory-hero",
        "runtime_detail_kit": False,
        "generator": {
            "path": "assets/vehicles/mazda787b/build_asset.py",
            "sha256": _sha(generator_bytes),
            "implementation": f"Python + trimesh {trimesh.__version__} + Pillow",
            "deterministic": True,
            "network_required": False,
        },
        "coordinate_system": {"units": "m", "handedness": "right", "forward_axis": "+x", "left_axis": "+y", "up_axis": "+z", "origin": "vehicle center at ground plane"},
        "dimensions_m": {"length": LENGTH_M, "width": WIDTH_M, "height": HEIGHT_M, "wheelbase": WHEELBASE_M, "front_track": FRONT_TRACK_M, "rear_track": REAR_TRACK_M},
        "budgets": {"triangle_limit": 80000, "triangle_count": triangles, "texture_max_px": 2048, "texture_dimensions_px": [2048, 2048]},
        "files": {
            "glb": _record(ENTRY_GLB, glb),
            "texture": _record(ENTRY_TEXTURE, atlas_bytes, width=2048, height=2048, role="project-authored livery and decal atlas embedded in GLB"),
            "provenance": _record("provenance.json", provenance_bytes),
        },
        "named_nodes": named_nodes,
        "binding_policy": "Named nodes accept visual telemetry only; geometry never feeds physics, reward, observations, or collision.",
    }
    return {
        ENTRY_GLB: glb,
        ENTRY_TEXTURE: atlas_bytes,
        "provenance.json": provenance_bytes,
        "asset_manifest.json": _json(manifest),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    for name, payload in outputs.items():
        path = HERE / name
        if args.check:
            if not path.is_file() or path.read_bytes() != payload:
                raise SystemExit(f"asset is not reproducible: {path}")
        else:
            path.write_bytes(payload)
    glb = outputs[ENTRY_GLB]
    print(f"{HERE / ENTRY_GLB}: {len(glb)} bytes sha256={_sha(glb)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
