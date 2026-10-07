#!/usr/bin/env python3
"""Build a presentation-only Nordschleife ribbon for Watch 2.5D.

Samples the same runtime track the sim uses (named_track + hills config) so
centerline / width / road_z match Python truth. Output is static JSON under
watch25d/public/ — the client never invents elevation for the car; frames
still own pose via road_z_m.

Usage:
  PYTHONPATH=$PWD python3 watch25d/tools/build_track.py
  PYTHONPATH=$PWD python3 watch25d/tools/build_track.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "public" / "assets" / "track" / "nordschleife_ribbon.json"

# Chunkier than the 3 m source — PS1 touring-car ribbon, not a CAD mesh.
STRIDE = 3  # every 3rd sample ≈ 9 m
ROUND = 4


def _round_list(arr: np.ndarray) -> list:
    return np.round(arr, ROUND).tolist()


def build_payload() -> dict:
    # Import after path is usable when invoked as `PYTHONPATH=$PWD …`.
    from supra.track import configure_hills, named_track

    configure_hills(enabled=True, scale=1.0, force_flat=False)
    trk = named_track("nordschleife")

    idx = np.arange(0, len(trk.center), STRIDE, dtype=int)
    # Ensure the loop closes on the first sample.
    if idx[-1] != 0:
        idx = np.append(idx, 0)

    center = trk.center[idx]
    z = trk.z[idx]
    half = trk.half_width[idx]
    heading = np.arctan2(trk.tangent[idx, 1], trk.tangent[idx, 0])
    samples = np.column_stack([center[:, 0], center[:, 1], z])

    return {
        "schema": "watch25d-track-v1",
        "id": "nordschleife",
        "display_name": "Nürburgring Nordschleife",
        "width_m": float(trk.width),
        "length_m": float(trk.length),
        "sample_count": int(len(samples)),
        "source_stride": int(STRIDE),
        "source": {
            "track_asset": "supra/data/tracks/nordschleife.json",
            "runtime": "configure_hills(enabled=True, scale=1.0, force_flat=False); named_track('nordschleife')",
            "note": "Presentation ribbon only. Car height comes from frame road_z_m.",
        },
        "coordinate_system": {
            "forward_plane": "xy",
            "up_axis": "+z",
            "units": "m",
            "three_map": "(x,y,z)=(sim.x, sim.z, -sim.y)",
        },
        # Parallel arrays keep the file smaller than nested objects.
        "center_xyz": _round_list(samples),
        "half_width_m": _round_list(half),
        "heading_rad": _round_list(heading),
    }


def canonical_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def write_payload(payload: dict) -> Path:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_bytes(payload)
    OUT.write_bytes(raw)
    return OUT


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--check",
        action="store_true",
        help="Assert existing public asset matches a fresh build (byte-stable).",
    )
    args = ap.parse_args()

    payload = build_payload()
    fresh = canonical_bytes(payload)
    digest = hashlib.sha256(fresh).hexdigest()[:16]

    if args.check:
        if not OUT.is_file():
            print(f"FAIL missing {OUT}", file=sys.stderr)
            return 1
        existing = OUT.read_bytes()
        if existing != fresh:
            print(f"FAIL {OUT} drifted (expected sha256={digest}…)", file=sys.stderr)
            return 1
        print(f"OK {OUT.relative_to(REPO)} sha256={digest}… samples={payload['sample_count']}")
        return 0

    path = write_payload(payload)
    print(
        f"wrote {path.relative_to(REPO)} "
        f"samples={payload['sample_count']} width={payload['width_m']} "
        f"sha256={digest}…"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
