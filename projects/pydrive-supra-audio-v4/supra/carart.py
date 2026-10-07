"""
Procedural 3D car rendering — retro PS1/PS2-style flat-shaded 3D models for Toyota Supra,
Mazda RX-7, and Nissan Skyline.

The models tilt dynamically based on physics telemetry (pitch/roll under weight transfer)
and project in 2.5D space. Painter's depth-sorting avoids face overlap anomalies.
"""
from __future__ import annotations

import numpy as np
import pygame

from .lightfx import draw_dual_headlight_beam

# Signature colors per chassis (realistic)
CAR_COLORS = {
    "supra":   (198, 32, 38),     # Renaissance Red
    "rx7":     (232, 233, 236),   # Crystal White
    "skyline": (33, 94, 173),     # Bayside Blue
    "lr4":     (190, 176, 142),   # Aruba / Nara sand tan
    "f150":    (22, 23, 25),      # Black XLT
    "mazda787b": (240, 100, 20),  # Renown Orange
}

def _rack_box(vertices, faces, x0, x1, y0, y1, z0, z1, col, prefix):
    """Append a small 3D box (roof-rack rail/crossbar) to the mesh in place."""
    b = len(vertices)
    vertices += [
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
    ]
    faces += [
        (prefix + "_top", [b + 4, b + 5, b + 6, b + 7], (0.0, 0.0, 1.0), col),
        (prefix + "_yp", [b + 2, b + 3, b + 7, b + 6], (0.0, 1.0, 0.2), col),
        (prefix + "_yn", [b + 0, b + 1, b + 5, b + 4], (0.0, -1.0, 0.2), col),
        (prefix + "_xp", [b + 1, b + 2, b + 6, b + 5], (1.0, 0.0, 0.2), col),
        (prefix + "_xn", [b + 3, b + 0, b + 4, b + 7], (-1.0, 0.0, 0.2), col),
    ]

def _shade(c, f):
    return (max(0, min(255, int(c[0] * f))),
            max(0, min(255, int(c[1] * f))),
            max(0, min(255, int(c[2] * f))))

# --- Base 3D Face List ---
# Format: (face_name, vertex_indices, body_normal, color_type)
# Centerline vertices: 0..7
# Left side vertices: 8..19
# Right side vertices: 20..31
_BASE_FACES = [
    # --- Hood / Front upper deck ---
    ("hood_left", [1, 2, 11, 9], (0.3, 0.1, 0.95), "body"),
    ("hood_right", [1, 21, 23, 2], (0.3, -0.1, 0.95), "body"),

    # --- Windshield ---
    ("windshield_left", [2, 3, 12, 11], (0.6, 0.2, 0.78), "glass"),
    ("windshield_right", [2, 23, 24, 3], (0.6, -0.2, 0.78), "glass"),

    # --- Roof ---
    ("roof_left", [3, 4, 13, 12], (0.0, 0.0, 1.0), "body"),
    ("roof_right", [3, 24, 25, 4], (0.0, 0.0, 1.0), "body"),

    # --- Rear Glass ---
    ("rear_glass_left", [4, 5, 14, 13], (-0.6, 0.2, 0.78), "glass"),
    ("rear_glass_right", [4, 25, 26, 5], (-0.6, -0.2, 0.78), "glass"),

    # --- Trunk ---
    ("trunk_left", [5, 6, 16, 14], (-0.1, 0.1, 0.99), "body"),
    ("trunk_right", [5, 26, 28, 6], (-0.1, -0.1, 0.99), "body"),

    # --- Front Bumper ---
    ("bumper_front_left", [0, 1, 9, 8], (0.95, 0.3, 0.0), "body"),
    ("bumper_front_right", [0, 20, 21, 1], (0.95, -0.3, 0.0), "body"),

    # --- Rear Bumper ---
    ("bumper_rear_left", [6, 7, 17, 16], (-0.95, 0.3, 0.0), "body"),
    ("bumper_rear_right", [6, 28, 29, 7], (-0.95, -0.3, 0.0), "body"),

    # --- Sides / Fenders ---
    ("side_glass_left", [11, 12, 13, 14], (0.0, 0.9, 0.44), "glass"),
    ("side_glass_right", [23, 26, 25, 24], (0.0, -0.9, 0.44), "glass"),

    ("fender_top_left", [9, 11, 10], (0.2, 0.2, 0.96), "body"),
    ("fender_top_right", [21, 22, 23], (0.2, -0.2, 0.96), "body"),

    ("fender_side_left", [8, 9, 10, 18], (0.3, 0.95, 0.0), "body"),
    ("fender_side_right", [20, 30, 22, 21], (0.3, -0.95, 0.0), "body"),

    ("door_upper_left", [10, 11, 14, 15], (0.0, 1.0, 0.0), "body"),
    ("door_upper_right", [22, 27, 26, 23], (0.0, -1.0, 0.0), "body"),

    ("door_lower_left", [18, 10, 15, 19], (0.0, 1.0, 0.0), "body"),
    ("door_lower_right", [30, 31, 27, 22], (0.0, -1.0, 0.0), "body"),

    ("haunch_side_left", [19, 15, 16, 17], (-0.3, 0.95, 0.0), "body"),
    ("haunch_side_right", [31, 29, 28, 27], (-0.3, -0.95, 0.0), "body"),
]

def get_car_mesh(name: str, HL: float, HW: float):
    """Returns (vertices, faces) for a low-poly 3D chassis model."""
    if name == "rx7":
        vertices = [
            # Centerline (0..7)
            (1.0 * HL, 0.0, 0.20),
            (0.95 * HL, 0.0, 0.35),
            (0.30 * HL, 0.0, 0.65),
            (-0.05 * HL, 0.0, 1.10),
            (-0.45 * HL, 0.0, 1.00),
            (-0.65 * HL, 0.0, 0.72),
            (-0.96 * HL, 0.0, 0.68),
            (-0.92 * HL, 0.0, 0.25),

            # Left side (8..19)
            (0.90 * HL, 0.35 * HW, 0.20),
            (0.85 * HL, 0.40 * HW, 0.35),
            (0.50 * HL, 0.90 * HW, 0.55),
            (0.25 * HL, 0.70 * HW, 0.65),
            (-0.05 * HL, 0.45 * HW, 1.10),
            (-0.45 * HL, 0.45 * HW, 1.00),
            (-0.60 * HL, 0.70 * HW, 0.72),
            (-0.55 * HL, 0.95 * HW, 0.72),
            (-0.90 * HL, 0.65 * HW, 0.68),
            (-0.85 * HL, 0.50 * HW, 0.25),
            (0.35 * HL, 0.85 * HW, 0.15),
            (-0.30 * HL, 0.90 * HW, 0.15),
        ]

        # Mirror left side to right side (20..31)
        for idx in range(8, 20):
            x, y, z = vertices[idx]
            vertices.append((x, -y, z))

        # RX-7 curved rear wing (pedestal style)
        vertices.append((-0.90 * HL, 0.50 * HW, 0.72)) # 32: L_wing_front
        vertices.append((-0.96 * HL, 0.50 * HW, 0.85)) # 33: L_wing_rear
        vertices.append((-0.90 * HL, -0.50 * HW, 0.72)) # 34: R_wing_front
        vertices.append((-0.96 * HL, -0.50 * HW, 0.85)) # 35: R_wing_rear
        vertices.append((-0.92 * HL, 0.0, 0.82)) # 36: C_wing_front
        vertices.append((-0.98 * HL, 0.0, 0.82)) # 37: C_wing_rear

        faces = list(_BASE_FACES) + [
            ("spoiler_left_leg", [16, 32, 33], (0.0, 1.0, 0.0), "body"),
            ("spoiler_right_leg", [28, 35, 34], (0.0, -1.0, 0.0), "body"),
            ("spoiler_blade_top", [32, 33, 37, 36], (0.0, 0.0, 1.0), "body"),
            ("spoiler_blade_top2", [36, 37, 35, 34], (0.0, 0.0, 1.0), "body"),
        ]

    elif name == "mazda787b":
        # ------------------------------------------------------------------
        # Mazda 787B — Group C prototype, rebuilt from reference:
        #   * cab-forward bubble canopy on a slab-wide, table-flat body
        #   * long kamm tail with a full-width deck
        #   * Renown "Charge" livery: orange/green QUADRANTS that swap
        #     halfway down the car, split by a white spine stripe
        #   * broad low rear wing on twin pylons with white endplates
        #   * deep skirts, rear arch spats, covered headlights (detail pass)
        # Perimeter contract for shadow/LOD: 0,8,18,19,17,7 / 0,7,29,31,30,20.
        # ------------------------------------------------------------------
        vertices = [
            # Centerline (0..7)
            (1.03 * HL, 0.0, 0.03),   # 0: splitter tip (lower, longer nose)
            (0.88 * HL, 0.0, 0.15),   # 1: nose crown
            (0.34 * HL, 0.0, 0.33),   # 2: screen base (windscreen far forward)
            (0.04 * HL, 0.0, 0.87),   # 3: canopy peak (cab-forward bubble)
            (-0.24 * HL, 0.0, 0.80),  # 4: canopy trailing edge
            (-0.38 * HL, 0.0, 0.43),  # 5: engine deck start
            (-1.01 * HL, 0.0, 0.38),  # 6: tail deck end (long flat kamm tail)
            (-1.03 * HL, 0.0, 0.10),  # 7: tail floor / diffuser

            # Left side (8..19)
            (1.01 * HL, 1.04 * HW, 0.04),   # 8: splitter corner
            (0.86 * HL, 0.98 * HW, 0.16),   # 9: nose shoulder (full width)
            (0.56 * HL, 1.08 * HW, 0.40),   # 10: front arch crown
            (0.30 * HL, 0.60 * HW, 0.35),   # 11: screen base outer
            (0.04 * HL, 0.30 * HW, 0.79),   # 12: canopy side front (narrow)
            (-0.24 * HL, 0.30 * HW, 0.73),  # 13: canopy side rear
            (-0.40 * HL, 0.62 * HW, 0.41),  # 14: deck shoulder
            (-0.60 * HL, 1.08 * HW, 0.43),  # 15: rear arch crown (spat line)
            (-1.00 * HL, 1.02 * HW, 0.36),  # 16: tail corner
            (-1.02 * HL, 1.04 * HW, 0.09),  # 17: tail floor corner
            (0.44 * HL, 1.08 * HW, 0.04),   # 18: front skirt
            (-0.28 * HL, 1.12 * HW, 0.04),  # 19: rear skirt (spat, widest)
        ]

        # Mirror left side to right side (20..31)
        for idx in range(8, 20):
            x, y, z = vertices[idx]
            vertices.append((x, -y, z))

        vertices += [
            # Rear wing (32..43): wide low blade, white endplates, twin pylons
            (-0.86 * HL, 0.97 * HW, 0.42),  # 32: L plate lower front
            (-0.86 * HL, 0.97 * HW, 0.88),  # 33: L plate upper front
            (-1.12 * HL, 0.97 * HW, 0.92),  # 34: L plate upper rear
            (-1.12 * HL, 0.97 * HW, 0.50),  # 35: L plate lower rear
            (-0.86 * HL, -0.97 * HW, 0.42), # 36: R plate lower front
            (-0.86 * HL, -0.97 * HW, 0.88), # 37: R plate upper front
            (-1.12 * HL, -0.97 * HW, 0.92), # 38: R plate upper rear
            (-1.12 * HL, -0.97 * HW, 0.50), # 39: R plate lower rear
            (-0.82 * HL, 0.34 * HW, 0.40),  # 40: L pylon base
            (-0.90 * HL, 0.34 * HW, 0.86),  # 41: L pylon top
            (-0.82 * HL, -0.34 * HW, 0.40), # 42: R pylon base
            (-0.90 * HL, -0.34 * HW, 0.86), # 43: R pylon top

            # White spine stripe (44..53): hood run + deck run, both ±0.07 HW
            (1.00 * HL, 0.07 * HW, 0.055),  # 44: L stripe, near tip
            (0.87 * HL, 0.07 * HW, 0.152),  # 45: L stripe, nose crown
            (0.33 * HL, 0.07 * HW, 0.332),  # 46: L stripe, screen base
            (1.00 * HL, -0.07 * HW, 0.055), # 47: R stripe, near tip
            (0.87 * HL, -0.07 * HW, 0.152), # 48: R stripe, nose crown
            (0.33 * HL, -0.07 * HW, 0.332), # 49: R stripe, screen base
            (-0.39 * HL, 0.07 * HW, 0.428), # 50: L stripe, deck start
            (-1.005 * HL, 0.07 * HW, 0.378),# 51: L stripe, tail end
            (-0.39 * HL, -0.07 * HW, 0.428),# 52: R stripe, deck start
            (-1.005 * HL, -0.07 * HW, 0.378),# 53: R stripe, tail end

            # Deck intake scoops (54..57 L, 58..61 R): dark boxes aft of canopy
            (-0.46 * HL, 0.42 * HW, 0.43),  # 54
            (-0.46 * HL, 0.26 * HW, 0.43),  # 55
            (-0.58 * HL, 0.26 * HW, 0.51),  # 56
            (-0.58 * HL, 0.42 * HW, 0.51),  # 57
            (-0.46 * HL, -0.42 * HW, 0.43), # 58
            (-0.46 * HL, -0.26 * HW, 0.43), # 59
            (-0.58 * HL, -0.26 * HW, 0.51), # 60
            (-0.58 * HL, -0.42 * HW, 0.51), # 61
        ]

        c_green = (0, 132, 62)          # Renown green
        c_orange = (243, 88, 18)        # Charge orange
        c_carbon = (14, 16, 19)
        c_white = (240, 240, 230)
        c_dark = (24, 26, 28)

        # Quadrant livery: front-left/rear-right ORANGE, front-right/rear-left
        # GREEN, white spine between them — the swap happens at the cockpit.
        faces = [
            # nose + hood
            ("stripe_hood", [44, 45, 46, 49, 48, 47], (0.25, 0.0, 0.97), c_white),
            ("hood_left", [45, 46, 11, 9], (0.28, 0.10, 0.95), c_orange),
            ("hood_right", [48, 21, 23, 49], (0.28, -0.10, 0.95), c_green),
            ("hood_tip_left", [44, 45, 9, 8], (0.55, 0.15, 0.82), c_orange),
            ("hood_tip_right", [47, 20, 21, 48], (0.55, -0.15, 0.82), c_green),
            ("bumper_front_left", [0, 44, 8], (0.92, 0.25, 0.30), c_orange),
            ("bumper_front_right", [0, 20, 47], (0.92, -0.25, 0.30), c_green),

            # glasshouse: wraparound screen + narrow bubble canopy
            ("windshield_left", [2, 3, 12, 11], (0.55, 0.22, 0.80), "glass"),
            ("windshield_right", [2, 23, 24, 3], (0.55, -0.22, 0.80), "glass"),
            ("canopy_left", [3, 4, 13, 12], (0.05, 0.30, 0.95), "glass"),
            ("canopy_right", [3, 24, 25, 4], (0.05, -0.30, 0.95), "glass"),
            ("screen_base", [11, 2, 23, 49, 46], (0.30, 0.0, 0.95), c_white),
            ("side_glass_left", [11, 12, 13, 14], (0.0, 0.92, 0.40), "glass"),
            ("side_glass_right", [23, 26, 25, 24], (0.0, -0.92, 0.40), "glass"),
            ("rear_glass_left", [4, 5, 14, 13], (-0.55, 0.22, 0.80), "glass"),
            ("rear_glass_right", [4, 25, 26, 5], (-0.55, -0.22, 0.80), "glass"),

            # engine deck: quadrants SWAP here (left green / right orange)
            ("stripe_deck", [5, 50, 51, 53, 52], (-0.08, 0.0, 0.99), c_white),
            ("engine_deck_left", [50, 51, 16, 14], (-0.08, 0.10, 0.99), c_green),
            ("engine_deck_right", [52, 26, 28, 53], (-0.08, -0.10, 0.99), c_orange),
            ("scoop_left", [54, 55, 56, 57], (0.55, 0.0, 0.83), c_carbon),
            ("scoop_right", [58, 61, 60, 59], (0.55, 0.0, 0.83), c_carbon),

            # kamm tail
            ("bumper_rear_left", [6, 7, 17, 16], (-0.96, 0.20, 0.0), c_dark),
            ("bumper_rear_right", [6, 28, 29, 7], (-0.96, -0.20, 0.0), c_dark),

            # flanks: arches + doors + spat haunches (quadrant colours)
            ("fender_top_left", [9, 11, 10], (0.18, 0.25, 0.95), c_orange),
            ("fender_top_right", [21, 22, 23], (0.18, -0.25, 0.95), c_green),
            ("fender_side_left", [8, 9, 10, 18], (0.25, 0.95, 0.0), c_orange),
            ("fender_side_right", [20, 30, 22, 21], (0.25, -0.95, 0.0), c_green),
            ("door_upper_left", [10, 11, 14, 15], (0.05, 0.98, 0.15), c_orange),
            ("door_upper_right", [22, 27, 26, 23], (0.05, -0.98, 0.15), c_green),
            ("door_lower_left", [18, 10, 15, 19], (0.0, 1.0, 0.0), c_green),
            ("door_lower_right", [30, 31, 27, 22], (0.0, -1.0, 0.0), c_orange),
            ("haunch_side_left", [19, 15, 16, 17], (-0.25, 0.95, 0.0), c_green),
            ("haunch_side_right", [31, 29, 28, 27], (-0.25, -0.95, 0.0), c_orange),

            # rear wing
            ("wing_L_plate", [32, 33, 34, 35], (0.0, 1.0, 0.0), c_white),
            ("wing_R_plate", [36, 39, 38, 37], (0.0, -1.0, 0.0), c_white),
            ("wing_blade_top", [33, 34, 38, 37], (0.0, 0.0, 1.0), c_carbon),
            ("wing_blade_rear", [34, 35, 39, 38], (-0.85, 0.0, 0.25), c_carbon),
            ("wing_pylon_left", [40, 41, 33, 32], (0.0, 1.0, 0.0), c_carbon),
            ("wing_pylon_right", [42, 36, 37, 43], (0.0, -1.0, 0.0), c_carbon),
        ]

    elif name == "skyline":
        vertices = [
            # Centerline (0..7)
            (1.0 * HL, 0.0, 0.15),
            (1.0 * HL, 0.0, 0.55),
            (0.32 * HL, 0.0, 0.78),
            (0.12 * HL, 0.0, 1.28),
            (-0.42 * HL, 0.0, 1.26),
            (-0.56 * HL, 0.0, 0.78),
            (-0.96 * HL, 0.0, 0.70),
            (-1.0 * HL, 0.0, 0.15),

            # Left side (8..19)
            (1.0 * HL, 0.48 * HW, 0.15),
            (1.0 * HL, 0.48 * HW, 0.55),
            (0.60 * HL, 0.94 * HW, 0.70),
            (0.25 * HL, 0.72 * HW, 0.78),
            (0.12 * HL, 0.48 * HW, 1.28),
            (-0.42 * HL, 0.48 * HW, 1.26),
            (-0.54 * HL, 0.72 * HW, 0.78),
            (-0.40 * HL, 0.96 * HW, 0.75),
            (-0.96 * HL, 0.60 * HW, 0.70),
            (-0.98 * HL, 0.55 * HW, 0.15),
            (0.40 * HL, 0.94 * HW, 0.15),
            (-0.20 * HL, 0.82 * HW, 0.15),
        ]

        # Mirror left side to right side (20..31)
        for idx in range(8, 20):
            x, y, z = vertices[idx]
            vertices.append((x, -y, z))

        # Skyline tall pedestal wing (exaggerated height)
        vertices.append((-0.80 * HL, 0.88 * HW, 1.15)) # 32: L wing front
        vertices.append((-0.94 * HL, 0.88 * HW, 1.15)) # 33: L wing back
        vertices.append((-0.80 * HL, -0.88 * HW, 1.15)) # 34: R wing front
        vertices.append((-0.94 * HL, -0.88 * HW, 1.15)) # 35: R wing back
        vertices.append((-0.78 * HL, 0.54 * HW, 0.70)) # 36: L pedestal front base
        vertices.append((-0.88 * HL, 0.54 * HW, 1.15)) # 37: L pedestal front top
        vertices.append((-0.78 * HL, -0.54 * HW, 0.70)) # 38: R pedestal front base
        vertices.append((-0.88 * HL, -0.54 * HW, 1.15)) # 39: R pedestal front top
        vertices.append((-0.78 * HL, 0.90 * HW, 1.22)) # 40: L endplate top front
        vertices.append((-0.98 * HL, 0.90 * HW, 1.22)) # 41: L endplate top back
        vertices.append((-0.98 * HL, 0.90 * HW, 1.08)) # 42: L endplate bottom back
        vertices.append((-0.78 * HL, 0.90 * HW, 1.08)) # 43: L endplate bottom front
        vertices.append((-0.78 * HL, -0.90 * HW, 1.22)) # 44: R endplate top front
        vertices.append((-0.98 * HL, -0.90 * HW, 1.22)) # 45: R endplate top back
        vertices.append((-0.98 * HL, -0.90 * HW, 1.08)) # 46: R endplate bottom back
        vertices.append((-0.78 * HL, -0.90 * HW, 1.08)) # 47: R endplate bottom front
        vertices.append((-0.82 * HL, 0.54 * HW, 0.70)) # 48: L pedestal back base
        vertices.append((-0.92 * HL, 0.54 * HW, 1.15)) # 49: L pedestal back top
        vertices.append((-0.82 * HL, -0.54 * HW, 0.70)) # 50: R pedestal back base
        vertices.append((-0.92 * HL, -0.54 * HW, 1.15)) # 51: R pedestal back top

        faces = list(_BASE_FACES) + [
            ("wing_blade", [32, 33, 35, 34], (0.0, 0.0, 1.0), "body"),
            ("pedestal_left", [36, 37, 49, 48], (0.0, 1.0, 0.0), (20, 20, 25)),
            ("pedestal_right", [38, 39, 51, 50], (0.0, -1.0, 0.0), (20, 20, 25)),
            ("endplate_left", [40, 41, 42, 43], (0.0, 1.0, 0.0), "body"),
            ("endplate_right", [44, 45, 46, 47], (0.0, -1.0, 0.0), "body"),
        ]

    elif name == "lr4":
        # Tall, boxy, upright SUV: short hood, near-vertical glass, long high flat
        # roof, near-vertical tailgate. Roof sits ~1.65 (vs ~1.2 for the sports
        # cars) so it towers in the 2.5D view.
        # Adjusted with a stepped roof profile (rear is higher than front).
        vertices = [
            # Centerline (0..7)
            (1.0 * HL, 0.0, 0.20),     # 0 front bumper
            (0.98 * HL, 0.0, 0.64),    # 1 tall upright grille / hood front
            (0.52 * HL, 0.0, 0.82),    # 2 windshield base (short hood)
            (0.36 * HL, 0.0, 1.62),    # 3 roof front
            (-0.72 * HL, 0.0, 1.68),   # 4 roof rear (raised slightly for stepped roof)
            (-0.84 * HL, 0.0, 0.84),   # 5 tailgate top (near vertical)
            (-0.95 * HL, 0.0, 0.70),   # 6 lower tailgate
            (-1.0 * HL, 0.0, 0.20),    # 7 rear bumper

            # Left side (8..19)
            (0.99 * HL, 0.88 * HW, 0.20),   # 8  front bumper corner
            (0.98 * HL, 0.88 * HW, 0.64),   # 9  hood corner
            (0.60 * HL, 1.00 * HW, 0.74),   # 10 front fender top (widest)
            (0.50 * HL, 0.82 * HW, 0.82),   # 11 A-pillar base
            (0.36 * HL, 0.66 * HW, 1.58),   # 12 A-pillar top (roof front side)
            (-0.72 * HL, 0.66 * HW, 1.64),  # 13 D-pillar top (roof rear side, stepped up to 1.64)
            (-0.84 * HL, 0.82 * HW, 0.84),  # 14 D-pillar base (tailgate side)
            (-0.60 * HL, 1.00 * HW, 0.82),  # 15 rear haunch (widest)
            (-0.94 * HL, 0.72 * HW, 0.70),  # 16 rear corner
            (-0.97 * HL, 0.54 * HW, 0.20),  # 17 rear bumper corner
            (0.52 * HL, 0.98 * HW, 0.20),   # 18 door sill front (low)
            (-0.42 * HL, 0.94 * HW, 0.20),  # 19 door sill rear (low)
        ]

        # Mirror left side to right side (20..31)
        for idx in range(8, 20):
            x, y, z = vertices[idx]
            vertices.append((x, -y, z))

        # Extra vertices for stepped roof and C-pillar details (starts at index 32)
        vertices += [
            (-0.15 * HL, 0.0, 1.62),         # 32 centerline roof step front bottom
            (-0.25 * HL, 0.0, 1.68),         # 33 centerline roof step rear top
            (-0.15 * HL, 0.82 * HW, 0.84),   # 34 left C-pillar front base
            (-0.15 * HL, 0.66 * HW, 1.58),   # 35 left C-pillar front top (front roof height)
            (-0.25 * HL, 0.82 * HW, 0.84),   # 36 left C-pillar rear base
            (-0.25 * HL, 0.66 * HW, 1.64),   # 37 left C-pillar rear top (rear roof height)
        ]

        # Mirror extra left side vertices to right side (indices 38..41)
        for idx in [34, 35, 36, 37]:
            x, y, z = vertices[idx]
            vertices.append((x, -y, z))

        # Filter base faces to exclude roof and side glass which we custom-split for the LR4
        base_to_keep = [
            "hood_left", "hood_right",
            "windshield_left", "windshield_right",
            "rear_glass_left", "rear_glass_right",
            "trunk_left", "trunk_right",
            "bumper_front_left", "bumper_front_right",
            "bumper_rear_left", "bumper_rear_right",
            "fender_top_left", "fender_top_right",
            "fender_side_left", "fender_side_right",
            "door_upper_left", "door_upper_right",
            "door_lower_left", "door_lower_right",
            "haunch_side_left", "haunch_side_right",
        ]
        faces = [f for f in _BASE_FACES if f[0] in base_to_keep]

        # Append custom LR4 split faces
        faces += [
            # Front Roof (before step)
            ("roof_front_left", [3, 32, 35, 12], (0.0, 0.0, 1.0), "body"),
            ("roof_front_right", [3, 24, 39, 32], (0.0, 0.0, 1.0), "body"),

            # Roof Step (angled transition)
            ("roof_step_left", [32, 33, 37, 35], (1.0, 0.0, 0.0), "body"),
            ("roof_step_right", [32, 39, 41, 33], (1.0, 0.0, 0.0), "body"),

            # Rear Roof (after step)
            ("roof_rear_left", [33, 4, 13, 37], (0.0, 0.0, 1.0), "body"),
            ("roof_rear_right", [33, 41, 25, 4], (0.0, 0.0, 1.0), "body"),

            # Side Glass & Pillars
            ("side_glass_front_left", [11, 12, 35, 34], (0.0, 1.0, 0.0), "glass"),
            ("side_glass_front_right", [23, 38, 39, 24], (0.0, -1.0, 0.0), "glass"),

            ("c_pillar_left", [34, 35, 37, 36], (0.0, 1.0, 0.0), "body"),
            ("c_pillar_right", [38, 40, 41, 39], (0.0, -1.0, 0.0), "body"),

            ("side_glass_rear_left", [36, 37, 13, 14], (0.0, 1.0, 0.0), "glass"),
            ("side_glass_rear_right", [40, 26, 25, 41], (0.0, -1.0, 0.0), "glass"),
        ]

        # Roof rack: two longitudinal rails + three crossbars, deep black.
        rack = (15, 15, 18)
        # Left Rail Segments (Front & Rear mounting heights)
        _rack_box(vertices, faces, 0.34 * HL, -0.15 * HL, 0.40 * HW, 0.56 * HW,
                  1.62, 1.70, rack, "rack_railL_front")
        _rack_box(vertices, faces, -0.15 * HL, -0.70 * HL, 0.40 * HW, 0.56 * HW,
                  1.68, 1.76, rack, "rack_railL_rear")
        # Right Rail Segments (Front & Rear mounting heights)
        _rack_box(vertices, faces, 0.34 * HL, -0.15 * HL, -0.56 * HW, -0.40 * HW,
                  1.62, 1.70, rack, "rack_railR_front")
        _rack_box(vertices, faces, -0.15 * HL, -0.70 * HL, -0.56 * HW, -0.40 * HW,
                  1.68, 1.76, rack, "rack_railR_rear")

        # Crossbars positioned to sit perfectly on rails
        _rack_box(vertices, faces, 0.27 * HL, 0.17 * HL, -0.54 * HW, 0.54 * HW,
                  1.70, 1.76, rack, "rack_cross0")
        _rack_box(vertices, faces, -0.13 * HL, -0.23 * HL, -0.54 * HW, 0.54 * HW,
                  1.76, 1.82, rack, "rack_cross1")
        _rack_box(vertices, faces, -0.53 * HL, -0.63 * HL, -0.54 * HW, 0.54 * HW,
                  1.76, 1.82, rack, "rack_cross2")

    elif name == "f150":
        # Long, tall pickup: blunt front clip, upright cab, open bed, high beltline.
        vertices = [
            # Centerline (0..7)
            (1.0 * HL, 0.0, 0.28),     # 0 front bumper
            (0.995 * HL, 0.0, 0.78),   # 1 upright grille / hood front
            (0.36 * HL, 0.0, 0.86),    # 2 windshield base
            (0.29 * HL, 0.0, 1.56),    # 3 roof front
            (-0.12 * HL, 0.0, 1.56),   # 4 roof rear
            (-0.24 * HL, 0.0, 0.94),   # 5 rear cab glass base
            (-0.97 * HL, 0.0, 0.78),   # 6 tailgate / bed rail top
            (-1.0 * HL, 0.0, 0.28),    # 7 rear bumper

            # Left side (8..19)
            (0.995 * HL, 0.90 * HW, 0.28),  # 8 front bumper corner
            (0.995 * HL, 0.90 * HW, 0.78),  # 9 hood corner
            (0.58 * HL, 1.04 * HW, 0.80),   # 10 front fender top
            (0.37 * HL, 0.88 * HW, 0.86),   # 11 A-pillar base
            (0.29 * HL, 0.70 * HW, 1.54),   # 12 A-pillar top
            (-0.12 * HL, 0.70 * HW, 1.54),  # 13 C-pillar top
            (-0.24 * HL, 0.88 * HW, 0.94),  # 14 rear cab / bed front
            (-0.72 * HL, 1.04 * HW, 0.84),  # 15 bed side top
            (-0.98 * HL, 0.96 * HW, 0.78),  # 16 tailgate corner
            (-1.0 * HL, 0.86 * HW, 0.28),   # 17 rear bumper corner
            (0.42 * HL, 1.02 * HW, 0.28),   # 18 rocker front
            (-0.96 * HL, 1.02 * HW, 0.28),  # 19 bed rocker rear
        ]

        for idx in range(8, 20):
            x, y, z = vertices[idx]
            vertices.append((x, -y, z))

        faces = list(_BASE_FACES)

        b = len(vertices)
        vertices += [
            (-0.28 * HL, 0.78 * HW, 0.84),   # bed opening front left
            (-0.96 * HL, 0.82 * HW, 0.76),   # bed opening rear left
            (-0.96 * HL, -0.82 * HW, 0.76),  # bed opening rear right
            (-0.28 * HL, -0.78 * HW, 0.84),  # bed opening front right
            (-0.38 * HL, 0.60 * HW, 0.46),   # bed floor front left
            (-0.90 * HL, 0.64 * HW, 0.38),   # bed floor rear left
            (-0.90 * HL, -0.64 * HW, 0.38),  # bed floor rear right
            (-0.38 * HL, -0.60 * HW, 0.46),  # bed floor front right
        ]
        bed_dark = (10, 11, 12)
        faces += [
            ("bed_lip", [b + 0, b + 1, b + 2, b + 3], (0.0, 0.0, 1.0), (18, 19, 21)),
            ("bed_floor", [b + 4, b + 5, b + 6, b + 7], (0.0, 0.0, 1.0), bed_dark),
            ("bed_front_wall", [b + 0, b + 3, b + 7, b + 4], (1.0, 0.0, 0.2), bed_dark),
            ("bed_left_wall", [b + 0, b + 4, b + 5, b + 1], (0.0, 1.0, 0.2), bed_dark),
            ("bed_right_wall", [b + 3, b + 2, b + 6, b + 7], (0.0, -1.0, 0.2), bed_dark),
            ("bed_tail_inner", [b + 1, b + 5, b + 6, b + 2], (-1.0, 0.0, 0.2), bed_dark),
        ]

    else:  # supra
        vertices = [
            # Centerline (0..7)
            (1.0 * HL, 0.0, 0.15),
            (0.95 * HL, 0.0, 0.48),
            (0.34 * HL, 0.0, 0.72),
            (0.12 * HL, 0.0, 1.22),
            (-0.35 * HL, 0.0, 1.20),
            (-0.55 * HL, 0.0, 0.72),
            (-0.92 * HL, 0.0, 0.70),
            (-1.0 * HL, 0.0, 0.15),

            # Left side (8..19)
            (0.89 * HL, 0.45 * HW, 0.15),
            (0.85 * HL, 0.45 * HW, 0.48),
            (0.62 * HL, 0.93 * HW, 0.65),
            (0.27 * HL, 0.68 * HW, 0.72),
            (0.12 * HL, 0.42 * HW, 1.22),
            (-0.35 * HL, 0.42 * HW, 1.20),
            (-0.55 * HL, 0.68 * HW, 0.72),
            (-0.42 * HL, 1.00 * HW, 0.74),
            (-0.88 * HL, 0.66 * HW, 0.70),
            (-0.95 * HL, 0.50 * HW, 0.15),
            (0.46 * HL, 0.92 * HW, 0.15),
            (-0.22 * HL, 0.83 * HW, 0.15),
        ]

        # Mirror left side to right side (20..31)
        for idx in range(8, 20):
            x, y, z = vertices[idx]
            vertices.append((x, -y, z))

        # Supra integrated ducktail lip (exaggerated height)
        vertices.append((-0.96 * HL, 0.0, 0.86)) # 32: C_duck_tip
        vertices.append((-0.92 * HL, 0.66 * HW, 0.86)) # 33: L_duck_tip
        vertices.append((-0.92 * HL, -0.66 * HW, 0.86)) # 34: R_duck_tip

        faces = list(_BASE_FACES) + [
            ("ducktail_left", [6, 16, 33, 32], (-0.9, 0.1, 0.4), "body"),
            ("ducktail_right", [6, 32, 34, 28], (-0.9, -0.1, 0.4), "body"),
        ]

    return vertices, faces

def add_wheels_to_mesh(vertices: list, faces: list, spec, veh) -> tuple[list, list]:
    """Generates 3D boxy wheels that steer and tilt with the car body."""
    steer = getattr(veh, "steer_angle", 0.0)
    wy = spec.half_track + 0.14
    wl = spec.wheel_radius * 2.2
    ww = 0.32
    wr = spec.wheel_radius
    top = 2 * wr
    if getattr(spec, "name", "") == "mazda787b":
        # Group C: keep the wheels tucked under the closed bodywork, but let the
        # outer sidewall sit proud of the skirt so side views do not read as spats.
        wy = spec.half_track + 0.20
        ww = 0.24
        wl = spec.wheel_radius * 2.05
        top = 0.50

    wheel_configs = [
        (spec.a, wy, steer, 1, "wheel_fl"),
        (spec.a, -wy, steer, -1, "wheel_fr"),
        (-spec.b, wy, 0.0, 1, "wheel_rl"),
        (-spec.b, -wy, 0.0, -1, "wheel_rr"),
    ]

    for cx, cyl, delta, sgn, w_name in wheel_configs:
        idx_start = len(vertices)

        # 8 corners of the 3D box wheel relative to wheel hub
        local_pts = [
            ( wl/2,  sgn * ww/2, top),    # 0: front top outer
            ( wl/2, -sgn * ww/2, top),    # 1: front top inner
            (-wl/2, -sgn * ww/2, top),    # 2: back top inner
            (-wl/2,  sgn * ww/2, top),    # 3: back top outer
            ( wl/2,  sgn * ww/2, 0.05),   # 4: front bot outer
            ( wl/2, -sgn * ww/2, 0.05),   # 5: front bot inner
            (-wl/2, -sgn * ww/2, 0.05),   # 6: back bot inner
            (-wl/2,  sgn * ww/2, 0.05),   # 7: back bot outer
        ]

        cs, sn = np.cos(delta), np.sin(delta)
        for lx, ly, lz in local_pts:
            # Rotate by steering angle
            rx = lx * cs - ly * sn
            ry = lx * sn + ly * cs
            # Translate to wheel position in body coordinates
            vertices.append((cx + rx, cyl + ry, lz))

        # Wheel faces and local normals
        local_faces = [
            ("top", [0, 1, 2, 3], (0, 0, 1), "wheel_tread"),
            ("outer", [0, 3, 7, 4], (0, sgn, 0), "wheel_rim"),
            ("inner", [1, 2, 6, 5], (0, -sgn, 0), "wheel_inner"),
            ("front", [0, 1, 5, 4], (1, 0, 0), "wheel_tread"),
            ("back", [3, 2, 6, 7], (-1, 0, 0), "wheel_tread"),
        ]

        for f_name, f_indices, (nx_loc, ny_loc, nz_loc), color_type in local_faces:
            # Rotate normal by steer delta to body frame
            nx_b = nx_loc * cs - ny_loc * sn
            ny_b = nx_loc * sn + ny_loc * cs
            nz_b = nz_loc

            global_indices = [idx_start + i for i in f_indices]
            faces.append((f"{w_name}_{f_name}", global_indices, (nx_b, ny_b, nz_b), color_type))

    return vertices, faces

def draw_car(screen, to_screen, scale, veh, spec, brake: float = 0.0, color=None, headlights: bool = False):
    name = getattr(spec, "name", "supra")
    base = color or CAR_COLORS.get(name, (198, 32, 38))
    b = float(np.clip(brake, 0, 1))

    HL = (spec.wheelbase + 1.7) / 2.0
    HW = (spec.track_width + 0.36) / 2.0
    yaw = veh.yaw

    on_screen_len = 2 * HL * scale
    detail = on_screen_len > 26

    # --- Low-Detail Fallback Mode ---
    if not detail:
        vertices, _ = get_car_mesh(name, HL, HW)
        cy, sy = np.cos(yaw), np.sin(yaw)
        poly_pts = []
        left_pts = []
        right_pts = []
        # Draw the flat bottom shape silhouette
        for idx in [0, 8, 18, 19, 17, 7, 29, 31, 30, 20]:
            lx, ly, _ = vertices[idx]
            wx = veh.x + lx * cy - ly * sy
            wy = veh.y + lx * sy + ly * cy
            poly_pts.append(to_screen(wx, wy))
        for idx in [0, 8, 18, 19, 17, 7]:
            lx, ly, _ = vertices[idx]
            wx = veh.x + lx * cy - ly * sy
            wy = veh.y + lx * sy + ly * cy
            left_pts.append(to_screen(wx, wy))
        for idx in [0, 7, 29, 31, 30, 20]:
            lx, ly, _ = vertices[idx]
            wx = veh.x + lx * cy - ly * sy
            wy = veh.y + lx * sy + ly * cy
            right_pts.append(to_screen(wx, wy))

        soft_sh = [(px + 4, py + 5) for px, py in poly_pts]
        hard_sh = [(px + 2, py + 3) for px, py in poly_pts]
        pygame.draw.polygon(screen, (24, 25, 29), soft_sh)
        pygame.draw.polygon(screen, (10, 11, 14), hard_sh)
        if name == "mazda787b":
            pygame.draw.polygon(screen, (246, 92, 24), left_pts)
            pygame.draw.polygon(screen, (0, 145, 70), right_pts)
            pygame.draw.line(screen, (238, 239, 226), left_pts[2], right_pts[-2], max(1, int(scale * 0.10)))
        else:
            pygame.draw.polygon(screen, base, poly_pts)
        pygame.draw.polygon(screen, _shade(base, 0.55), poly_pts, 1)
        return

    # --- 3D Mesh Construction ---
    vertices, faces = get_car_mesh(name, HL, HW)
    vertices, faces = add_wheels_to_mesh(vertices, faces, spec, veh)

    # --- Body Roll & Pitch Transformations (Suspension Dynamics) ---
    ax_val = np.clip(veh.ax, -20.0, 20.0)
    ay_val = np.clip(veh.ay, -20.0, 20.0)
    roll_gain = getattr(spec, "body_roll_gain", 1.0)   # tall SUVs lean harder
    tilt_pitch = -ax_val * 0.003 * roll_gain
    tilt_roll = -ay_val * 0.003 * roll_gain

    cy, sy = np.cos(yaw), np.sin(yaw)

    # Transform all vertices from body coordinates to world and project to screen
    vertices_world = []
    vertices_screen = []
    for lx, ly, lz in vertices:
        # 1. Apply shear/tilt
        tx = lx + tilt_pitch * lz
        ty = ly + tilt_roll * lz
        tz = lz

        # 2. Rotate to world coordinates
        wx = veh.x + tx * cy - ty * sy
        wy = veh.y + tx * sy + ty * cy
        vertices_world.append((wx, wy, tz))

        # 3. Project to screen (2.5D height offset)
        sx, sy_pt = to_screen(wx, wy)
        sy_projected = sy_pt - tz * 0.45 * scale
        vertices_screen.append((int(sx), int(sy_projected)))

    # --- Helper to project coordinates dynamically for details ---
    def S(lx, ly, lz):
        tx = lx + tilt_pitch * lz
        ty = ly + tilt_roll * lz
        wx = veh.x + tx * cy - ty * sy
        wy = veh.y + tx * sy + ty * cy
        sx, sy_pt = to_screen(wx, wy)
        return (int(sx), int(sy_pt - lz * 0.45 * scale))

    # --- Dynamic 3D Shadow ---
    shadow_pts = []
    # Project bottom perimeter loop on the ground (lz = 0)
    for idx in [0, 8, 18, 19, 17, 7, 29, 31, 30, 20]:
        lx, ly, _ = vertices[idx]
        wx = veh.x + lx * cy - ly * sy
        wy = veh.y + lx * sy + ly * cy
        sx, sy_pt = to_screen(wx, wy)
        shadow_pts.append((sx + 3, sy_pt + 4))
    pygame.draw.polygon(screen, (25, 26, 30), [(x + 3, y + 3) for x, y in shadow_pts])
    pygame.draw.polygon(screen, (10, 11, 14), shadow_pts)

    # --- Headlight Beams (Dusk and Night only) ---
    if headlights:
        if name in ("lr4", "f150"):
            beam_x, beam_y = 0.985, 0.58
            beam_len, beam_half_width = 18.0, 5.4
            beam_intensity = 0.70
        else:
            beam_x, beam_y = 0.9, 0.3
            beam_len, beam_half_width = 18.0, 4.4
            beam_intensity = 0.66
        left_src = S(beam_x * HL, beam_y * HW, 0.45)
        right_src = S(beam_x * HL, -beam_y * HW, 0.45)
        far_center = S(beam_x * HL + beam_len, 0.0, 0.0)
        far_edge = S(beam_x * HL + beam_len, beam_half_width, 0.0)
        far_half_width_px = float(np.hypot(far_edge[0] - far_center[0],
                                           far_edge[1] - far_center[1]))
        draw_dual_headlight_beam(
            pygame, screen, left_src, right_src, far_center, far_half_width_px,
            intensity=beam_intensity,
            source_radius_px=max(2.0, scale * 0.055),
            strips=44,
            blur_scale=0.5,
        )

    # --- Dynamic Flat Shading Setup ---
    # Direction TO light source in world coordinates (High, North-East)
    lx_to, ly_to, lz_to = 0.3, -0.4, 0.86

    # --- Depth Sort & Shading Resolution ---
    faces_to_render = []
    for f_name, f_indices, body_normal, color_type in faces:
        # Compute depth for Painter's algorithm (average world-y)
        avg_wy = np.mean([vertices_world[idx][1] for idx in f_indices])

        # Rotate body-frame normal to world frame
        nx_b, ny_b, nz_b = body_normal
        nx_w = nx_b * cy - ny_b * sy
        ny_w = nx_b * sy + ny_b * cy
        nz_w = nz_b

        # Dot product with light direction
        dot = nx_w * lx_to + ny_w * ly_to + nz_w * lz_to
        shade_factor = 0.55 + 0.45 * max(0.0, dot)

        # Resolve face base colors
        if color_type == "body":
            face_color = _shade(base, shade_factor)
        elif color_type == "glass":
            face_color = _shade((40, 50, 65), shade_factor)
        elif color_type == "wheel_tread":
            face_color = _shade((32, 32, 36), shade_factor)
        elif color_type == "wheel_rim":
            face_color = _shade((150, 150, 160), shade_factor)
        elif color_type == "wheel_inner":
            face_color = _shade((20, 20, 22), shade_factor)
        elif isinstance(color_type, tuple):
            face_color = _shade(color_type, shade_factor)
        else:
            face_color = _shade(base, shade_factor)

        poly_pts = [vertices_screen[idx] for idx in f_indices]
        faces_to_render.append((avg_wy, f_name, poly_pts, face_color))

    # Sort faces from north to south (smallest world-y first)
    faces_to_render.sort(key=lambda x: x[0])

    # --- Render Sorted Faces and Overlay Details ---
    for avg_wy, f_name, poly_pts, face_color in faces_to_render:
        pygame.draw.polygon(screen, face_color, poly_pts)

        # Outline edges slightly for PS1-style jitter/definition
        lw = max(1, int(scale * 0.025))
        pygame.draw.polygon(screen, _shade(face_color, 0.72), poly_pts, lw)

        if name in ("lr4", "f150") and f_name.startswith("wheel_") and f_name.endswith("_outer"):
            cx = (poly_pts[0][0] + poly_pts[1][0] + poly_pts[2][0] + poly_pts[3][0]) / 4.0
            c_y = (poly_pts[0][1] + poly_pts[1][1] + poly_pts[2][1] + poly_pts[3][1]) / 4.0
            hx = (poly_pts[0][0] - poly_pts[1][0] + poly_pts[3][0] - poly_pts[2][0]) / 4.0
            hy = (poly_pts[0][1] - poly_pts[1][1] + poly_pts[3][1] - poly_pts[2][1]) / 4.0
            vx = (poly_pts[0][0] - poly_pts[3][0] + poly_pts[1][0] - poly_pts[2][0]) / 4.0
            vy = (poly_pts[0][1] - poly_pts[3][1] + poly_pts[1][1] - poly_pts[2][1]) / 4.0
            rim_r = 0.90
            hub_r = 0.25
            rim_poly = []
            for k in range(16):
                ang = k * (2.0 * np.pi / 16.0)
                ca, sa = np.cos(ang), np.sin(ang)
                rim_poly.append((int(cx + ca * rim_r * hx + sa * rim_r * vx),
                                 int(c_y + ca * rim_r * hy + sa * rim_r * vy)))
            pygame.draw.polygon(screen, (176, 174, 164), rim_poly)
            pygame.draw.polygon(screen, (74, 76, 76), rim_poly, 1)
            inner_poly = []
            for k in range(16):
                ang = k * (2.0 * np.pi / 16.0)
                ca, sa = np.cos(ang), np.sin(ang)
                inner_poly.append((int(cx + ca * 0.68 * hx + sa * 0.68 * vx),
                                   int(c_y + ca * 0.68 * hy + sa * 0.68 * vy)))
            pygame.draw.polygon(screen, (40, 42, 44), inner_poly)
            spoke_w = max(1, int(scale * 0.030))
            is_front = "_fl_" in f_name or "_fr_" in f_name
            phase = (getattr(veh, "steer_angle", 0.0) * 0.9) if is_front else 0.0
            spoke_count = 6 if name == "f150" else 10
            for k in range(spoke_count):
                ang = phase + k * (2.0 * np.pi / spoke_count) - np.pi / 2.0
                ca, sa = np.cos(ang), np.sin(ang)
                shx = cx + ca * hub_r * hx + sa * hub_r * vx
                shy = c_y + ca * hub_r * hy + sa * hub_r * vy
                ehx = cx + ca * rim_r * 0.92 * hx + sa * rim_r * 0.92 * vx
                ehy = c_y + ca * rim_r * 0.92 * hy + sa * rim_r * 0.92 * vy
                pygame.draw.line(screen, (202, 199, 187), (int(shx), int(shy)), (int(ehx), int(ehy)), spoke_w)
            hub_poly = []
            for k in range(12):
                ang = k * (2.0 * np.pi / 12.0)
                ca, sa = np.cos(ang), np.sin(ang)
                hub_poly.append((int(cx + ca * hub_r * hx + sa * hub_r * vx),
                                 int(c_y + ca * hub_r * hy + sa * hub_r * vy)))
            pygame.draw.polygon(screen, (214, 211, 196), hub_poly)
            pygame.draw.polygon(screen, (80, 82, 85), hub_poly, 1)

        if name == "mazda787b" and f_name.startswith("wheel_") and f_name.endswith("_outer"):
            cx = (poly_pts[0][0] + poly_pts[1][0] + poly_pts[2][0] + poly_pts[3][0]) / 4.0
            c_y = (poly_pts[0][1] + poly_pts[1][1] + poly_pts[2][1] + poly_pts[3][1]) / 4.0
            hx = (poly_pts[0][0] - poly_pts[1][0] + poly_pts[3][0] - poly_pts[2][0]) / 4.0
            hy = (poly_pts[0][1] - poly_pts[1][1] + poly_pts[3][1] - poly_pts[2][1]) / 4.0
            vx = (poly_pts[0][0] - poly_pts[3][0] + poly_pts[1][0] - poly_pts[2][0]) / 4.0
            vy = (poly_pts[0][1] - poly_pts[3][1] + poly_pts[1][1] - poly_pts[2][1]) / 4.0
            brake_heat = 0.25 + 0.75 * b
            rim = []
            tire = []
            for k in range(16):
                ang = k * (2.0 * np.pi / 16.0)
                ca, sa = np.cos(ang), np.sin(ang)
                tire.append((int(cx + ca * 0.92 * hx + sa * 0.92 * vx),
                             int(c_y + ca * 0.92 * hy + sa * 0.92 * vy)))
                rim.append((int(cx + ca * 0.52 * hx + sa * 0.52 * vx),
                            int(c_y + ca * 0.52 * hy + sa * 0.52 * vy)))
            pygame.draw.polygon(screen, (12, 13, 15), tire)
            pygame.draw.polygon(screen, (150, 118, 78), rim)
            pygame.draw.polygon(screen, (30, 26, 22), rim, max(1, int(scale * 0.02)))
            if b > 0.05:
                glow_r = max(2, int(scale * 0.07 * brake_heat))
                pygame.draw.circle(screen, (255, int(80 + 90 * brake_heat), 28),
                                   (int(cx), int(c_y)), glow_r)

        # Draw details / decals relative to specific faces
        if name == "skyline":
            if f_name == "bumper_front_left":
                # Intercooler mesh
                g_pts = [S(1.0 * HL, 0.34 * HW, 0.15), S(1.0 * HL, -0.34 * HW, 0.15),
                         S(1.0 * HL, -0.34 * HW, 0.42), S(1.0 * HL, 0.34 * HW, 0.42)]
                pygame.draw.polygon(screen, (12, 13, 16), g_pts)
                pygame.draw.line(screen, (85, 90, 100), S(1.0 * HL, 0.22 * HW, 0.28), S(1.0 * HL, -0.22 * HW, 0.28), 2)
            elif f_name == "bumper_front_right":
                # Headlights
                hl_col = (255, 255, 200) if headlights else (248, 248, 235)
                hl_l = [S(1.0 * HL, 0.50 * HW, 0.42), S(1.0 * HL, 0.34 * HW, 0.42),
                        S(1.0 * HL, 0.34 * HW, 0.52), S(1.0 * HL, 0.50 * HW, 0.52)]
                pygame.draw.polygon(screen, hl_col, hl_l)
                hl_r = [S(1.0 * HL, -0.50 * HW, 0.42), S(1.0 * HL, -0.34 * HW, 0.42),
                        S(1.0 * HL, -0.34 * HW, 0.52), S(1.0 * HL, -0.50 * HW, 0.52)]
                pygame.draw.polygon(screen, hl_col, hl_r)
            elif f_name == "bumper_rear_left":
                # Quad round taillights left
                tail_col = (int(160 + 95 * b), int(20 + 20 * b), int(24 + 18 * b))
                pygame.draw.circle(screen, tail_col, S(-0.97 * HL, 0.44 * HW, 0.48), max(2, int(scale * 0.09)))
                pygame.draw.circle(screen, tail_col, S(-0.97 * HL, 0.22 * HW, 0.48), max(2, int(scale * 0.065)))
            elif f_name == "bumper_rear_right":
                # Quad round taillights right
                tail_col = (int(160 + 95 * b), int(20 + 20 * b), int(24 + 18 * b))
                pygame.draw.circle(screen, tail_col, S(-0.97 * HL, -0.44 * HW, 0.48), max(2, int(scale * 0.09)))
                pygame.draw.circle(screen, tail_col, S(-0.97 * HL, -0.22 * HW, 0.48), max(2, int(scale * 0.065)))

        elif name == "rx7":
            if f_name == "hood_left":
                popup_l = [S(0.78 * HL, 0.44 * HW, 0.40), S(0.66 * HL, 0.46 * HW, 0.42),
                           S(0.64 * HL, 0.18 * HW, 0.44), S(0.76 * HL, 0.16 * HW, 0.42)]
                popup_col = (255, 255, 200) if headlights else _shade(base, 0.82)
                pygame.draw.polygon(screen, popup_col, popup_l, 0 if headlights else max(1, int(scale * 0.02)))
            elif f_name == "hood_right":
                popup_r = [S(0.78 * HL, -0.44 * HW, 0.40), S(0.66 * HL, -0.46 * HW, 0.42),
                           S(0.64 * HL, -0.18 * HW, 0.44), S(0.76 * HL, -0.16 * HW, 0.42)]
                popup_col = (255, 255, 200) if headlights else _shade(base, 0.82)
                pygame.draw.polygon(screen, popup_col, popup_r, 0 if headlights else max(1, int(scale * 0.02)))
            elif f_name == "bumper_rear_left":
                bar_l = [S(-0.94 * HL, 0.55 * HW, 0.55), S(-0.96 * HL, 0.45 * HW, 0.45),
                         S(-0.96 * HL, 0.0, 0.45), S(-0.94 * HL, 0.0, 0.55)]
                pygame.draw.polygon(screen, (16, 17, 20), bar_l)
                tail_col = (int(160 + 95 * b), int(20 + 20 * b), int(24 + 18 * b))
                pygame.draw.circle(screen, tail_col, S(-0.95 * HL, 0.36 * HW, 0.50), max(2, int(scale * 0.07)))
                pygame.draw.circle(screen, tail_col, S(-0.95 * HL, 0.20 * HW, 0.50), max(2, int(scale * 0.07)))
            elif f_name == "bumper_rear_right":
                bar_r = [S(-0.94 * HL, 0.0, 0.55), S(-0.96 * HL, 0.0, 0.45),
                         S(-0.96 * HL, -0.45 * HW, 0.45), S(-0.94 * HL, -0.55 * HW, 0.55)]
                pygame.draw.polygon(screen, (16, 17, 20), bar_r)
                tail_col = (int(160 + 95 * b), int(20 + 20 * b), int(24 + 18 * b))
                pygame.draw.circle(screen, tail_col, S(-0.95 * HL, -0.36 * HW, 0.50), max(2, int(scale * 0.07)))
                pygame.draw.circle(screen, tail_col, S(-0.95 * HL, -0.20 * HW, 0.50), max(2, int(scale * 0.07)))

        elif name == "mazda787b":
            if f_name == "bumper_front_left":
                hl_col = (255, 255, 230) if headlights else (40, 42, 45)
                hl_l = [S(0.97 * HL, 0.34 * HW, 0.13), S(0.88 * HL, 0.80 * HW, 0.22),
                        S(0.80 * HL, 0.72 * HW, 0.24), S(0.91 * HL, 0.32 * HW, 0.17)]
                pygame.draw.polygon(screen, hl_col, hl_l)
                pygame.draw.polygon(screen, (8, 10, 13), hl_l, max(1, int(scale * 0.025)))
                pygame.draw.line(screen, (18, 20, 22), S(0.98 * HL, 0.88 * HW, 0.08),
                                 S(0.42 * HL, 1.07 * HW, 0.08), max(1, int(scale * 0.04)))
            elif f_name == "bumper_front_right":
                hl_col = (255, 255, 230) if headlights else (40, 42, 45)
                hl_r = [S(0.97 * HL, -0.34 * HW, 0.13), S(0.88 * HL, -0.80 * HW, 0.22),
                        S(0.80 * HL, -0.72 * HW, 0.24), S(0.91 * HL, -0.32 * HW, 0.17)]
                pygame.draw.polygon(screen, hl_col, hl_r)
                pygame.draw.polygon(screen, (8, 10, 13), hl_r, max(1, int(scale * 0.025)))
                pygame.draw.line(screen, (18, 20, 22), S(0.98 * HL, -0.88 * HW, 0.08),
                                 S(0.42 * HL, -1.07 * HW, 0.08), max(1, int(scale * 0.04)))
            elif f_name == "bumper_rear_left" or f_name == "bumper_rear_right":
                tail_col = (int(188 + 67 * b), int(18 + 58 * b), int(18 + 36 * b))
                y_pos = 0.50 * HW if f_name == "bumper_rear_left" else -0.50 * HW
                pygame.draw.circle(screen, (44, 8, 8), S(-1.00 * HL, y_pos, 0.28), max(4, int(scale * 0.13)))
                pygame.draw.circle(screen, tail_col, S(-1.01 * HL, y_pos, 0.28), max(3, int(scale * 0.085)))
                pygame.draw.circle(screen, (255, 90, 40), S(-1.00 * HL, 0.0, 0.26), max(2, int(scale * (0.045 + 0.05 * b))))
            elif f_name in ("canopy_left", "canopy_right"):
                pygame.draw.line(screen, (105, 150, 170), S(0.05 * HL, 0.02 * HW, 0.77),
                                 S(-0.20 * HL, 0.02 * HW, 0.68), max(1, int(scale * 0.035)))
            elif f_name == "engine_deck_right":
                roundel = S(-0.42 * HL, -0.32 * HW, 0.40)
                pygame.draw.circle(screen, (236, 238, 230), roundel, max(3, int(scale * 0.16)))
                pygame.draw.circle(screen, (25, 26, 28), roundel, max(1, int(scale * 0.10)), max(1, int(scale * 0.025)))
            elif f_name == "engine_deck_left":
                pygame.draw.line(screen, (28, 28, 30), S(-0.18 * HL, 0.62 * HW, 0.40),
                                 S(-0.76 * HL, 0.80 * HW, 0.36), max(1, int(scale * 0.05)))
            elif f_name == "door_lower_left":
                exh_l = [S(0.10 * HL, 1.05 * HW, 0.15), S(0.0 * HL, 1.05 * HW, 0.15),
                         S(0.0 * HL, 1.05 * HW, 0.20), S(0.10 * HL, 1.05 * HW, 0.20)]
                pygame.draw.polygon(screen, (10, 10, 10), exh_l)
                if getattr(veh, 'throttle', 0.0) > 0.6:
                    flame = [S(0.0 * HL, 1.06 * HW, 0.17), S(-0.15 * HL, 1.08 * HW, 0.17),
                             S(-0.05 * HL, 1.06 * HW, 0.19)]
                    pygame.draw.polygon(screen, (255, 120, 20), flame)
                    pygame.draw.polygon(screen, (255, 215, 80),
                                        [flame[0], S(-0.07 * HL, 1.075 * HW, 0.18), flame[2]])
            elif f_name == "door_lower_right":
                exh_r = [S(0.10 * HL, -1.05 * HW, 0.15), S(0.0 * HL, -1.05 * HW, 0.15),
                         S(0.0 * HL, -1.05 * HW, 0.20), S(0.10 * HL, -1.05 * HW, 0.20)]
                pygame.draw.polygon(screen, (10, 10, 10), exh_r)
                if getattr(veh, 'throttle', 0.0) > 0.6:
                    flame = [S(0.0 * HL, -1.06 * HW, 0.17), S(-0.15 * HL, -1.08 * HW, 0.17),
                             S(-0.05 * HL, -1.06 * HW, 0.19)]
                    pygame.draw.polygon(screen, (255, 120, 20), flame)
                    pygame.draw.polygon(screen, (255, 215, 80),
                                        [flame[0], S(-0.07 * HL, -1.075 * HW, 0.18), flame[2]])

        elif name == "supra":
            if f_name == "bumper_front_left":
                hl_col = (255, 255, 200) if headlights else (245, 246, 230)
                hl_l = [S(0.91 * HL, 0.15 * HW, 0.49), S(0.81 * HL, 0.46 * HW, 0.50),
                        S(0.73 * HL, 0.41 * HW, 0.51), S(0.82 * HL, 0.13 * HW, 0.50)]
                pygame.draw.polygon(screen, hl_col, hl_l)
            elif f_name == "bumper_front_right":
                hl_col = (255, 255, 200) if headlights else (245, 246, 230)
                hl_r = [S(0.91 * HL, -0.15 * HW, 0.49), S(0.81 * HL, -0.46 * HW, 0.50),
                        S(0.73 * HL, -0.41 * HW, 0.51), S(0.82 * HL, -0.13 * HW, 0.50)]
                pygame.draw.polygon(screen, hl_col, hl_r)
            elif f_name == "bumper_rear_left" or f_name == "bumper_rear_right":
                tail_col = (int(150 + 105 * b), int(20 + 20 * b), int(24 + 18 * b))
                bar = [S(-0.92 * HL, 0.54 * HW, 0.48), S(-0.94 * HL, 0.47 * HW, 0.46),
                       S(-0.94 * HL, -0.47 * HW, 0.46), S(-0.92 * HL, -0.54 * HW, 0.48)]
                pygame.draw.polygon(screen, tail_col, bar)
            elif f_name == "fender_side_left":
                vent_l = [S(0.52 * HL, 0.64 * HW, 0.45), S(0.42 * HL, 0.78 * HW, 0.42),
                          S(0.36 * HL, 0.74 * HW, 0.42), S(0.46 * HL, 0.60 * HW, 0.45)]
                pygame.draw.polygon(screen, (20, 21, 25), vent_l)
            elif f_name == "fender_side_right":
                vent_r = [S(0.52 * HL, -0.64 * HW, 0.45), S(0.42 * HL, -0.78 * HW, 0.42),
                          S(0.36 * HL, -0.74 * HW, 0.42), S(0.46 * HL, -0.60 * HW, 0.45)]
                pygame.draw.polygon(screen, (20, 21, 25), vent_r)

    # --- LR4 identity overlay ---
    # Draw these after the sorted body faces so the boxy grille, warm halogen
    # clusters, and vertical tail lamps stay readable at chase-camera scale.
    if name == "lr4":
        halogen = (255, 226, 156) if headlights else (232, 204, 150)
        bulb = (255, 244, 190) if headlights else (202, 178, 122)
        lens = (224, 214, 188)
        amber = (238, 142, 42)
        tail_col = (int(145 + 110 * b), int(18 + 30 * b), int(22 + 20 * b))

        # --- A/B/D Pillars (Floating Roof look) ---
        pillar_w = max(2, int(scale * 0.05))
        # A-pillars (front windshield sides)
        pygame.draw.line(screen, (20, 20, 22), S(0.50 * HL, 0.82 * HW, 0.82), S(0.36 * HL, 0.66 * HW, 1.58), pillar_w)
        pygame.draw.line(screen, (20, 20, 22), S(0.50 * HL, -0.82 * HW, 0.82), S(0.36 * HL, -0.66 * HW, 1.58), pillar_w)
        # B-pillars (middle window separator)
        pygame.draw.line(screen, (20, 20, 22), S(0.15 * HL, 0.82 * HW, 0.83), S(0.15 * HL, 0.66 * HW, 1.60), pillar_w)
        pygame.draw.line(screen, (20, 20, 22), S(0.15 * HL, -0.82 * HW, 0.83), S(0.15 * HL, -0.66 * HW, 1.60), pillar_w)
        # D-pillars (rear tailgate corners)
        pygame.draw.line(screen, (20, 20, 22), S(-0.84 * HL, 0.82 * HW, 0.84), S(-0.72 * HL, 0.66 * HW, 1.64), pillar_w)
        pygame.draw.line(screen, (20, 20, 22), S(-0.84 * HL, -0.82 * HW, 0.84), S(-0.72 * HL, -0.66 * HW, 1.64), pillar_w)

        # --- Front Fender Side Vents ---
        vent_w = max(1, int(scale * 0.015))
        # Left fender vent
        vent_l = [S(0.53 * HL, 0.995 * HW, 0.46), S(0.45 * HL, 0.995 * HW, 0.46),
                  S(0.45 * HL, 0.995 * HW, 0.62), S(0.53 * HL, 0.995 * HW, 0.62)]
        pygame.draw.polygon(screen, (20, 20, 22), vent_l)
        pygame.draw.polygon(screen, (140, 142, 146), vent_l, vent_w)
        # Right fender vent
        vent_r = [S(0.53 * HL, -0.995 * HW, 0.46), S(0.45 * HL, -0.995 * HW, 0.46),
                  S(0.45 * HL, -0.995 * HW, 0.62), S(0.53 * HL, -0.995 * HW, 0.62)]
        pygame.draw.polygon(screen, (20, 20, 22), vent_r)
        pygame.draw.polygon(screen, (140, 142, 146), vent_r, vent_w)

        # --- Hood Lettering ("LAND ROVER") ---
        for dy in (-0.12, -0.06, 0.0, 0.06, 0.12):
            pygame.draw.circle(screen, (215, 218, 220), S(0.98 * HL, dy * HW, 0.665), max(1, int(scale * 0.015)))

        # --- Front Grille: silver LR4 horizontal bars ---
        front_x = 0.985 * HL
        grille = [S(front_x, 0.34 * HW, 0.39), S(front_x, -0.34 * HW, 0.39),
                  S(front_x, -0.34 * HW, 0.66), S(front_x, 0.34 * HW, 0.66)]
        pygame.draw.polygon(screen, (18, 20, 21), grille)
        pygame.draw.polygon(screen, (194, 199, 198), grille, max(1, int(scale * 0.030)))
        for zz, width in ((0.45, 0.040), (0.52, 0.052), (0.59, 0.040)):
            pygame.draw.line(screen, (216, 221, 218), S(front_x, 0.30 * HW, zz),
                             S(front_x, -0.30 * HW, zz), max(1, int(scale * width)))
        badge = [S(front_x, 0.10 * HW, 0.50), S(front_x, -0.10 * HW, 0.50),
                 S(front_x, -0.10 * HW, 0.56), S(front_x, 0.10 * HW, 0.56)]
        pygame.draw.polygon(screen, (20, 80, 40), badge)
        pygame.draw.polygon(screen, (215, 219, 216), badge, 1)

        # --- Rectangular Halogen Headlamps ---
        def front_rect(yc, half_y, z0, z1):
            return [S(front_x, (yc - half_y) * HW, z0), S(front_x, (yc + half_y) * HW, z0),
                    S(front_x, (yc + half_y) * HW, z1), S(front_x, (yc - half_y) * HW, z1)]

        for side in (1.0, -1.0):
            lamp = front_rect(side * 0.67, 0.22, 0.36, 0.70)
            pygame.draw.polygon(screen, (20, 22, 24), lamp)
            pygame.draw.polygon(screen, (205, 209, 204), lamp, max(1, int(scale * 0.020)))
            lens_rect = front_rect(side * 0.66, 0.17, 0.40, 0.66)
            pygame.draw.polygon(screen, lens, lens_rect)
            pygame.draw.polygon(screen, (88, 92, 92), lens_rect, 1)

            low_beam = front_rect(side * 0.58, 0.065, 0.50, 0.61)
            high_beam = front_rect(side * 0.74, 0.055, 0.45, 0.55)
            pygame.draw.polygon(screen, (55, 56, 52), low_beam)
            pygame.draw.polygon(screen, halogen, front_rect(side * 0.58, 0.045, 0.52, 0.59))
            pygame.draw.polygon(screen, bulb, front_rect(side * 0.58, 0.020, 0.54, 0.57))
            pygame.draw.polygon(screen, (55, 56, 52), high_beam)
            pygame.draw.polygon(screen, halogen, front_rect(side * 0.74, 0.035, 0.47, 0.53))

            marker = front_rect(side * 0.85, 0.025, 0.40, 0.64)
            pygame.draw.polygon(screen, amber, marker)

        # --- Asymmetrical Tailgate Overlay & Stacked Tail Lamps ---
        rear_x = -0.985 * HL
        if b > 0.05:
            glow_w = max(4, int(scale * 0.24))
            glow_h = max(5, int(scale * 0.42))
            for gy_body in (0.76, -0.76):
                gx, gy = S(rear_x, gy_body * HW, 0.52)
                glow = pygame.Surface((glow_w, glow_h), pygame.SRCALPHA)
                pygame.draw.rect(glow, (255, 45, 45, int(75 * b)), glow.get_rect())
                screen.blit(glow, (gx - glow_w // 2, gy - glow_h // 2))

        # Asymmetrical window extension (black glass panel on left side)
        glass_ext = [S(rear_x, 0.82 * HW, 0.82), S(rear_x, -0.10 * HW, 0.82),
                     S(rear_x, -0.10 * HW, 0.74), S(rear_x, 0.82 * HW, 0.74)]
        pygame.draw.polygon(screen, (32, 40, 52), glass_ext)
        pygame.draw.polygon(screen, (20, 22, 24), glass_ext, 1)

        # License Plate (yellow NJ style) on the left side
        plate = [S(rear_x, 0.45 * HW, 0.44), S(rear_x, 0.15 * HW, 0.44),
                 S(rear_x, 0.15 * HW, 0.54), S(rear_x, 0.45 * HW, 0.54)]
        pygame.draw.polygon(screen, (240, 220, 130), plate)
        pygame.draw.polygon(screen, (20, 22, 24), plate, 1)
        pygame.draw.circle(screen, (20, 40, 20), S(rear_x, 0.30 * HW, 0.49), max(1, int(scale * 0.02)))

        # Tailgate Handle and Land Rover silver/green badge on the right
        handle = [S(rear_x, -0.05 * HW, 0.54), S(rear_x, -0.35 * HW, 0.54),
                  S(rear_x, -0.35 * HW, 0.60), S(rear_x, -0.05 * HW, 0.60)]
        pygame.draw.polygon(screen, (24, 25, 27), handle)

        badge_rear = [S(rear_x, -0.42 * HW, 0.55), S(rear_x, -0.48 * HW, 0.55),
                      S(rear_x, -0.48 * HW, 0.59), S(rear_x, -0.42 * HW, 0.59)]
        pygame.draw.polygon(screen, (20, 80, 40), badge_rear)

        # Rectangular stacked tail lamps.
        def rear_rect(yc, half_y, z0, z1):
            return [S(rear_x, (yc - half_y) * HW, z0), S(rear_x, (yc + half_y) * HW, z0),
                    S(rear_x, (yc + half_y) * HW, z1), S(rear_x, (yc - half_y) * HW, z1)]

        for side in (1.0, -1.0):
            lamp_bg = rear_rect(side * 0.76, 0.13, 0.24, 0.77)
            pygame.draw.polygon(screen, (15, 16, 18), lamp_bg)
            pygame.draw.polygon(screen, (70, 72, 74), lamp_bg, max(1, int(scale * 0.015)))

            for z0, z1, col in ((0.63, 0.75, tail_col),
                                (0.50, 0.61, amber),
                                (0.40, 0.49, (222, 222, 212)),
                                (0.27, 0.38, tail_col)):
                segment = rear_rect(side * 0.76, 0.095, z0, z1)
                pygame.draw.polygon(screen, col, segment)
                pygame.draw.polygon(screen, (38, 40, 42), segment, 1)

    # --- F-150 XLT identity overlay ---
    if name == "f150":
        chrome = (178, 182, 180)
        chrome_hi = (226, 229, 224)
        dark = (7, 8, 9)
        lens = (224, 226, 218) if headlights else (172, 176, 170)
        lamp_hot = (255, 242, 190) if headlights else (210, 202, 174)
        amber = (238, 142, 42)
        tail_col = (int(150 + 105 * b), int(18 + 28 * b), int(24 + 22 * b))
        front_x = 0.99 * HL
        rear_x = -0.985 * HL

        bed_top = [S(-0.28 * HL, 0.78 * HW, 0.84), S(-0.96 * HL, 0.82 * HW, 0.76),
                   S(-0.96 * HL, -0.82 * HW, 0.76), S(-0.28 * HL, -0.78 * HW, 0.84)]
        bed_floor = [S(-0.38 * HL, 0.60 * HW, 0.46), S(-0.90 * HL, 0.64 * HW, 0.38),
                     S(-0.90 * HL, -0.64 * HW, 0.38), S(-0.38 * HL, -0.60 * HW, 0.46)]
        pygame.draw.polygon(screen, (15, 16, 18), bed_top)
        pygame.draw.polygon(screen, (8, 9, 10), bed_floor)
        pygame.draw.polygon(screen, (54, 56, 58), bed_top, max(1, int(scale * 0.025)))
        for rib_y in (-0.34, 0.0, 0.34):
            pygame.draw.line(screen, (36, 38, 40), S(-0.40 * HL, rib_y * HW, 0.47),
                             S(-0.88 * HL, rib_y * HW, 0.39), max(1, int(scale * 0.018)))
        for ysgn in (1.0, -1.0):
            pygame.draw.line(screen, (78, 80, 82), S(-0.30 * HL, ysgn * 0.90 * HW, 0.86),
                             S(-0.96 * HL, ysgn * 0.90 * HW, 0.77), max(1, int(scale * 0.020)))

        line_w = max(1, int(scale * 0.025))
        for ysgn in (1.0, -1.0):
            pygame.draw.line(screen, dark, S(0.29 * HL, ysgn * 0.70 * HW, 1.48),
                             S(0.37 * HL, ysgn * 0.88 * HW, 0.86), line_w)
            pygame.draw.line(screen, dark, S(-0.12 * HL, ysgn * 0.70 * HW, 1.48),
                             S(-0.24 * HL, ysgn * 0.88 * HW, 0.94), line_w)
            pygame.draw.line(screen, (82, 84, 86), S(0.08 * HL, ysgn * 1.02 * HW, 0.30),
                             S(0.07 * HL, ysgn * 0.98 * HW, 0.85), line_w)
            pygame.draw.line(screen, (82, 84, 86), S(-0.28 * HL, ysgn * 1.02 * HW, 0.30),
                             S(-0.28 * HL, ysgn * 0.98 * HW, 0.86), line_w)
            pygame.draw.line(screen, (70, 72, 74), S(-0.30 * HL, ysgn * 1.02 * HW, 0.80),
                             S(-0.96 * HL, ysgn * 0.96 * HW, 0.74), line_w)
        rear_window = [S(-0.21 * HL, 0.46 * HW, 0.98), S(-0.21 * HL, -0.46 * HW, 0.98),
                       S(-0.13 * HL, -0.44 * HW, 1.44), S(-0.13 * HL, 0.44 * HW, 1.44)]
        pygame.draw.polygon(screen, (28, 36, 48), rear_window)
        pygame.draw.polygon(screen, dark, rear_window, 1)

        def front_panel(y0, y1, z0, z1):
            return [S(front_x, y0 * HW, z0), S(front_x, y1 * HW, z0),
                    S(front_x, y1 * HW, z1), S(front_x, y0 * HW, z1)]

        def side_marker(side):
            return [S(0.985 * HL, side * 0.935 * HW, 0.43),
                    S(0.80 * HL, side * 1.005 * HW, 0.43),
                    S(0.80 * HL, side * 1.005 * HW, 0.55),
                    S(0.985 * HL, side * 0.935 * HW, 0.55)]

        def side_panel(side, x0, x1, z0, z1, y=1.035):
            return [S(x0 * HL, side * y * HW, z0), S(x1 * HL, side * y * HW, z0),
                    S(x1 * HL, side * y * HW, z1), S(x0 * HL, side * y * HW, z1)]

        for side in (1.0, -1.0):
            running_board = side_panel(side, 0.38, -0.22, 0.17, 0.25, 1.08)
            pygame.draw.polygon(screen, (18, 19, 20), running_board)
            pygame.draw.polygon(screen, chrome, running_board, 1)

            fender_badge = side_panel(side, 0.48, 0.33, 0.55, 0.66, 1.055)
            pygame.draw.polygon(screen, chrome_hi, fender_badge)
            pygame.draw.polygon(screen, (42, 44, 46), side_panel(side, 0.45, 0.36, 0.58, 0.63, 1.057))
            pygame.draw.line(screen, (20, 72, 136), S(0.44 * HL, side * 1.058 * HW, 0.605),
                             S(0.37 * HL, side * 1.058 * HW, 0.605), 1)

            for x0, x1 in ((0.25, 0.14), (-0.05, -0.15)):
                handle = side_panel(side, x0, x1, 0.67, 0.72, 1.056)
                pygame.draw.polygon(screen, chrome, handle)
                pygame.draw.polygon(screen, dark, handle, 1)

            mirror = [S(0.37 * HL, side * 0.91 * HW, 1.06),
                      S(0.47 * HL, side * 1.12 * HW, 1.02),
                      S(0.47 * HL, side * 1.12 * HW, 0.88),
                      S(0.37 * HL, side * 0.94 * HW, 0.92)]
            pygame.draw.polygon(screen, (10, 11, 12), mirror)
            pygame.draw.polygon(screen, (54, 56, 58), mirror, 1)

            rear_wrap = side_panel(side, -0.86, -0.97, 0.48, 0.69, 1.025)
            pygame.draw.polygon(screen, tail_col, rear_wrap)
            pygame.draw.polygon(screen, (70, 20, 22), rear_wrap, 1)

        bumper = front_panel(0.94, -0.94, 0.22, 0.34)
        pygame.draw.polygon(screen, (18, 19, 21), bumper)
        pygame.draw.polygon(screen, chrome, bumper, max(1, int(scale * 0.015)))
        lower_slot = front_panel(0.48, -0.48, 0.25, 0.31)
        pygame.draw.polygon(screen, (5, 6, 7), lower_slot)

        grille = front_panel(0.50, -0.50, 0.34, 0.78)
        grille_pocket = front_panel(0.43, -0.43, 0.39, 0.73)
        pygame.draw.polygon(screen, chrome_hi, grille)
        pygame.draw.polygon(screen, (13, 14, 16), grille_pocket)
        pygame.draw.polygon(screen, (72, 76, 76), grille, max(1, int(scale * 0.012)))
        for y0, y1 in ((0.50, 0.44), (-0.44, -0.50)):
            pygame.draw.polygon(screen, _shade(chrome_hi, 0.94), front_panel(y0, y1, 0.36, 0.76))
        for z0, z1, shade in ((0.64, 0.72, 1.00), (0.51, 0.59, 0.92), (0.38, 0.46, 0.82)):
            bar = front_panel(0.42, -0.42, z0, z1)
            pygame.draw.polygon(screen, _shade(chrome_hi, shade), bar)
            pygame.draw.polygon(screen, (72, 76, 76), bar, 1)
        badge = front_panel(0.15, -0.15, 0.515, 0.585)
        pygame.draw.polygon(screen, (20, 72, 136), badge)
        pygame.draw.polygon(screen, chrome_hi, badge, 1)
        pygame.draw.line(screen, (238, 242, 238), S(front_x, 0.08 * HW, 0.55),
                         S(front_x, -0.08 * HW, 0.55), max(1, int(scale * 0.014)))

        amber_on = (255, 168, 48) if headlights else amber
        for side in (1.0, -1.0):
            inner_y = side * 0.48
            outer_y = side * 0.92
            lamp = front_panel(inner_y, outer_y, 0.35, 0.76)
            pygame.draw.polygon(screen, (12, 13, 15), lamp)
            pygame.draw.polygon(screen, chrome_hi, lamp, max(1, int(scale * 0.018)))

            lens_box = front_panel(side * 0.54, side * 0.83, 0.42, 0.68)
            pygame.draw.polygon(screen, lens, lens_box)
            pygame.draw.polygon(screen, (88, 92, 92), lens_box, 1)

            # F-series amber C-clamp: outboard vertical with top/bottom amber returns.
            c_vert = front_panel(side * 0.84, side * 0.92, 0.38, 0.73)
            c_top = front_panel(side * 0.55, side * 0.90, 0.67, 0.75)
            c_bot = front_panel(side * 0.55, side * 0.90, 0.35, 0.43)
            for clamp in (c_vert, c_top, c_bot):
                pygame.draw.polygon(screen, amber_on, clamp)
                pygame.draw.polygon(screen, (98, 56, 18), clamp, 1)

            marker = side_marker(side)
            pygame.draw.polygon(screen, amber_on, marker)
            pygame.draw.polygon(screen, (86, 48, 16), marker, 1)

            low_beam = front_panel(side * 0.56, side * 0.68, 0.50, 0.62)
            high_beam = front_panel(side * 0.70, side * 0.80, 0.48, 0.60)
            pygame.draw.polygon(screen, (42, 44, 43), low_beam)
            pygame.draw.polygon(screen, lamp_hot, front_panel(side * 0.58, side * 0.66, 0.52, 0.60))
            pygame.draw.polygon(screen, (42, 44, 43), high_beam)
            pygame.draw.polygon(screen, (236, 232, 210), front_panel(side * 0.72, side * 0.78, 0.50, 0.58))

        def rear_panel(y0, y1, z0, z1):
            return [S(rear_x, y0 * HW, z0), S(rear_x, y1 * HW, z0),
                    S(rear_x, y1 * HW, z1), S(rear_x, y0 * HW, z1)]

        rear_bumper = rear_panel(0.82, -0.82, 0.18, 0.32)
        pygame.draw.polygon(screen, chrome, rear_bumper)
        pygame.draw.polygon(screen, (78, 80, 82), rear_bumper, max(1, int(scale * 0.014)))
        for y0, y1 in ((0.78, 0.36), (-0.36, -0.78)):
            pygame.draw.polygon(screen, (18, 19, 20), rear_panel(y0, y1, 0.23, 0.29))
        pygame.draw.polygon(screen, dark, rear_panel(0.10, -0.10, 0.10, 0.20))

        tailgate = rear_panel(0.64, -0.64, 0.33, 0.76)
        pygame.draw.polygon(screen, (18, 19, 21), tailgate)
        pygame.draw.polygon(screen, (58, 60, 62), tailgate, max(1, int(scale * 0.018)))
        pygame.draw.polygon(screen, (11, 12, 14), rear_panel(0.48, -0.48, 0.42, 0.66), 1)
        pygame.draw.line(screen, (72, 74, 76), S(rear_x, 0.60 * HW, 0.72),
                         S(rear_x, -0.60 * HW, 0.72), max(1, int(scale * 0.014)))
        handle = rear_panel(0.15, -0.15, 0.61, 0.66)
        pygame.draw.polygon(screen, dark, handle)
        rear_badge = rear_panel(0.11, -0.11, 0.49, 0.55)
        pygame.draw.polygon(screen, (24, 74, 130), rear_badge)
        pygame.draw.polygon(screen, chrome_hi, rear_badge, 1)
        pygame.draw.line(screen, (238, 242, 238), S(rear_x, 0.07 * HW, 0.52),
                         S(rear_x, -0.07 * HW, 0.52), 1)
        for yy in (0.48, 0.41, 0.34, 0.27):
            pygame.draw.polygon(screen, chrome_hi, rear_panel(yy, yy - 0.045, 0.45, 0.54))
        plate = rear_panel(-0.24, -0.48, 0.36, 0.46)
        pygame.draw.polygon(screen, (226, 210, 132), plate)
        pygame.draw.polygon(screen, dark, plate, 1)

        if b > 0.05:
            glow_w = max(4, int(scale * 0.40))
            glow_h = max(5, int(scale * 0.62))
            for gy_body in (0.78, -0.78):
                gx, gy = S(rear_x, gy_body * HW, 0.55)
                glow = pygame.Surface((glow_w, glow_h), pygame.SRCALPHA)
                pygame.draw.ellipse(glow, (255, 45, 45, int(95 * b)), glow.get_rect())
                screen.blit(glow, (gx - glow_w // 2, gy - glow_h // 2))

        def rear_rect(yc, half_y, z0, z1):
            return [S(rear_x, (yc - half_y) * HW, z0), S(rear_x, (yc + half_y) * HW, z0),
                    S(rear_x, (yc + half_y) * HW, z1), S(rear_x, (yc - half_y) * HW, z1)]

        for side in (1.0, -1.0):
            lamp_bg = rear_rect(side * 0.78, 0.13, 0.28, 0.80)
            pygame.draw.polygon(screen, (14, 15, 17), lamp_bg)
            pygame.draw.polygon(screen, (70, 72, 74), lamp_bg, max(1, int(scale * 0.012)))
            for yc, hy, z0, z1, col in ((side * 0.78, 0.09, 0.64, 0.77, tail_col),
                                        (side * 0.87, 0.035, 0.36, 0.76, tail_col),
                                        (side * 0.78, 0.08, 0.49, 0.60, (225, 225, 212)),
                                        (side * 0.78, 0.09, 0.32, 0.45, tail_col)):
                seg_pts = rear_rect(yc, hy, z0, z1)
                pygame.draw.polygon(screen, col, seg_pts)
                pygame.draw.polygon(screen, (40, 42, 44), seg_pts, 1)

    # --- Brake Light Bloom Overlay ---
    if b > 0.05 and name not in ("lr4", "f150"):
        gx, gy = S(-0.94 * HL, 0, 0.48)
        glow = pygame.Surface((int(scale * 1.35), int(scale * 0.9)), pygame.SRCALPHA)
        pygame.draw.ellipse(glow, (255, 45, 45, int(110 * b)), glow.get_rect())
        screen.blit(glow, (gx - glow.get_width() // 2, gy - glow.get_height() // 2))
