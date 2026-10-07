#!/usr/bin/env python3
"""Generate paired thin-wing IB target-point clouds from a spanwise outline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

R_M = 0.0483
C_MEAN_M = 0.0183
F_HZ = 26.1
BODY_ANGLE = math.radians(39.8)
STROKE_PLANE_ANGLE = math.radians(15.0)
PHI_AMPLITUDE = 1.0
ALPHA_AMPLITUDE = 0.87


def read_outline(path: Path) -> list[tuple[float, float, float]]:
    rows = []
    with path.open(newline="") as f:
        for raw in f:
            if raw.startswith("#") or not raw.strip():
                continue
            fields = next(csv.reader([raw]))
            if fields[0] == "s_over_R":
                continue
            rows.append(tuple(map(float, fields)))
    if len(rows) < 3:
        raise ValueError("outline must contain at least three span stations")
    stations = [r[0] for r in rows]
    if stations[0] != 0.0 or stations[-1] != 1.0 or any(b <= a for a, b in zip(stations, stations[1:])):
        raise ValueError("s_over_R must strictly increase from 0 to 1")
    if any(te < le for _, le, te in rows):
        raise ValueError("x_te_shape must be greater than or equal to x_le_shape")
    return rows


def interp_outline(rows: list[tuple[float, float, float]], s: float) -> tuple[float, float]:
    for a, b in zip(rows, rows[1:]):
        if a[0] <= s <= b[0]:
            q = (s - a[0]) / (b[0] - a[0])
            return a[1] * (1 - q) + b[1] * q, a[2] * (1 - q) + b[2] * q
    return rows[-1][1], rows[-1][2]


def sample_wing(rows: list[tuple[float, float, float]], spacing_m: float) -> tuple[list[tuple[float, float, float]], float]:
    # Scale the planform isotropically in its plane so area/span equals reported
    # mean chord. This is an explicit model assumption, not a fitted lift term.
    span_breaks = [r[0] * R_M for r in rows]
    chord_shape = [r[2] - r[1] for r in rows]
    area_shape = sum((b - a) * (ca + cb) * 0.5 for a, b, ca, cb in zip(span_breaks, span_breaks[1:], chord_shape, chord_shape[1:]))
    mean_shape = area_shape / R_M
    scale = C_MEAN_M / mean_shape
    n_span = math.ceil(R_M / spacing_m)
    points = []
    for j in range(n_span + 1):
        s = min(j * spacing_m, R_M)
        le_shape, te_shape = interp_outline(rows, s / R_M)
        x_le = le_shape * scale
        x_te = te_shape * scale
        if x_te - x_le <= 1e-12:
            points.append((0.5 * (x_le + x_te), s, 0.0))
            continue
        n_chord = max(1, math.ceil((x_te - x_le) / spacing_m))
        for i in range(n_chord + 1):
            x = x_le + (x_te - x_le) * i / n_chord
            points.append((x, s, 0.0))
    return points, mean_shape


def write_vertex(path: Path, points: list[tuple[float, float, float]]) -> None:
    with path.open("w") as f:
        f.write(f"{len(points)}\n")
        for x, y, z in points:
            f.write(f"{x:.12e} {y:.12e} {z:.12e}\n")


def pose(point: tuple[float, float, float], side: int, phase: float = 0.0) -> tuple[float, float, float]:
    x, y, z = point
    phi = PHI_AMPLITUDE * math.cos(2 * math.pi * phase)
    alpha = ALPHA_AMPLITUDE * math.sin(2 * math.pi * phase)
    # Right-hand rotations: R_body(y,beta) R_stroke(x,chi) R_phi(x,s*phi) R_alpha(y,s*alpha).
    # The source figure fixes the angles but not an executable matrix convention;
    # this explicit convention is a modeling choice to be verified by a pose plot.
    def rotate_x(v, angle):
        a, b, c = v
        return a, b * math.cos(angle) - c * math.sin(angle), b * math.sin(angle) + c * math.cos(angle)

    def rotate_y(v, angle):
        a, b, c = v
        return a * math.cos(angle) + c * math.sin(angle), b, -a * math.sin(angle) + c * math.cos(angle)

    v = rotate_y((x, y, z), side * alpha)
    v = rotate_x(v, side * phi)
    v = rotate_x(v, STROKE_PLANE_ANGLE)
    return rotate_y(v, BODY_ANGLE)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outline", type=Path, default=Path(__file__).parents[1] / "reference_data" / "wing_outline.csv")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spacing-mm", type=float, default=0.75)
    args = parser.parse_args()
    if args.spacing_mm <= 0:
        parser.error("--spacing-mm must be positive")
    rows = read_outline(args.outline)
    one_wing, mean_shape = sample_wing(rows, args.spacing_mm * 1e-3)
    args.output.mkdir(parents=True, exist_ok=True)
    right = [pose(point, 1) for point in one_wing]
    left = [pose((x, -y, z), -1) for x, y, z in one_wing]
    write_vertex(args.output / "right_wing.vertex", right)
    write_vertex(args.output / "left_wing.vertex", left)
    manifest = {
        "geometry_status": "provisional_manual_raster_outline_not_pixel_registered_or_specimen_cad",
        "outline_file": str(args.outline),
        "outline_sha256": sha256(args.outline),
        "wing_length_m": R_M,
        "target_mean_chord_m": C_MEAN_M,
        "unscaled_outline_mean_chord_shape": mean_shape,
        "point_spacing_request_m": args.spacing_mm * 1e-3,
        "points_per_wing": len(one_wing),
        "right_vertex_sha256": sha256(args.output / "right_wing.vertex"),
        "left_vertex_sha256": sha256(args.output / "left_wing.vertex"),
        "coordinate_convention": "chord=x; right span=+y; left span=-y; hinge at origin; vertex points are transformed to the prescribed t=0 pose",
        "initial_pose_convention": "R_y(beta) R_x(chi) R_x(side*phi) R_y(side*alpha), phi=1 rad, alpha=0 rad at t=0",
        "model_assumptions": [
            "spanwise linear interpolation between tabulated outline stations",
            "isotropic planform scaling chosen so area divided by span equals published mean chord",
            "paired fore/hind wing treated as one rigid surface per side",
            "point cloud is a thin immersed surface; numerical spacing and target stiffness require sensitivity checks",
        ],
    }
    (args.output / "geometry_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
