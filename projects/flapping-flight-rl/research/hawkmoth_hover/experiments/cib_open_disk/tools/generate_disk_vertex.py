#!/usr/bin/env python3
"""Generate the point cloud used by the existing open-disk DG-IIM pilot.

The ordering and coordinates intentionally mirror the construction in
experiments/open_disk_dg_iim/src/open_disk.cpp. CIBMethod consumes marker
coordinates rather than the triangulated FE connectivity, so no connectivity
or surface quadrature is implied by this file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=0.5)
    parser.add_argument("--radial-cells", type=int, default=4)
    parser.add_argument("--azimuthal-cells", type=int, default=24)
    args = parser.parse_args()
    if args.radius <= 0 or args.radial_cells < 1 or args.azimuthal_cells < 6:
        parser.error("require radius > 0, radial cells >= 1, azimuthal cells >= 6")

    points = [(0.0, 0.0, 0.0)]
    for ir in range(1, args.radial_cells + 1):
        radius = args.radius * ir / args.radial_cells
        for j in range(args.azimuthal_cells):
            theta = 2.0 * math.pi * j / args.azimuthal_cells
            points.append((0.0, radius * math.cos(theta), radius * math.sin(theta)))

    body = str(len(points)) + "\n" + "".join(
        f"{x:.16e}\t{y:.16e}\t{z:.16e}\n" for x, y, z in points
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(body, encoding="ascii")
    manifest = {
        "format": "IBAMR IBStandardInitializer ASCII .vertex",
        "source_geometry": "open_disk_dg_iim/src/open_disk.cpp",
        "source_construction": "center point plus nr radial rings with nt azimuthal points",
        "radius_code_length": args.radius,
        "radial_cells": args.radial_cells,
        "azimuthal_cells": args.azimuthal_cells,
        "point_count": len(points),
        "triangle_connectivity_in_cib": False,
        "area_quadrature_in_cib": False,
        "anatomical_or_finite_thickness_claim": False,
        "vertex_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
