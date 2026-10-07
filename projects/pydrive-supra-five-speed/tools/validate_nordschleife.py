"""Validate the real-scale Nordschleife asset and terrain switch behavior."""
from __future__ import annotations

import sys

import numpy as np

from supra.track import configure_hills, named_track


def fail(msg: str) -> int:
    print(f"FAIL: {msg}")
    return 1


def main() -> int:
    configure_hills(enabled=False, force_flat=False)
    nord = named_track("nordschleife")
    err = abs(nord.length - 20_832.0)
    if err > 1.0:
        return fail(f"Nordschleife length {nord.length:.3f} m, err {err:.3f} m")
    if len(nord.center) < 6000:
        return fail(f"too few Nordschleife samples: {len(nord.center)}")
    for name, arr in {
        "center": nord.center,
        "z": nord.z,
        "grade": nord.grade,
        "curvature": nord.curvature,
    }.items():
        if not np.all(np.isfinite(arr)):
            return fail(f"{name} contains NaN/Inf")
    if not np.all(np.diff(nord.arc) > 0.0):
        return fail("arc is not strictly increasing")
    if float(np.max(nord.seg_len)) > 8.0:
        return fail(f"segment spike too large: {float(np.max(nord.seg_len)):.2f} m")
    if nord.elev_gain <= 250.0:
        return fail(f"elevation gain suspiciously low: {nord.elev_gain:.2f} m")
    max_grade = float(np.max(np.abs(nord.grade)))
    if max_grade > 0.20:
        return fail(f"road grade spike too large: {max_grade:.3f}")
    finite_launch = nord.launch_speed[np.isfinite(nord.launch_speed)]
    min_launch = float(np.min(finite_launch)) if len(finite_launch) else float("inf")
    if min_launch < 34.0:
        return fail(f"crest launch floor too low: {min_launch:.2f} m/s")
    if not nord.metadata.get("real_track"):
        return fail("missing real_track metadata")
    if nord._nearest_tree is None:
        return fail("KD-tree nearest lookup not active")

    akina = named_track("akina")
    if abs(akina.elev_gain) > 1e-6:
        return fail(f"Akina should default flat, got elev {akina.elev_gain}")

    configure_hills(enabled=False, force_flat=True)
    flat = named_track("nordschleife")
    if abs(flat.elev_gain) > 1e-6:
        return fail(f"--flat should flatten Nordschleife, got elev {flat.elev_gain}")

    configure_hills(enabled=True, force_flat=False)
    hilly_akina = named_track("akina")
    if hilly_akina.elev_gain <= 0.0:
        return fail("--hills should still enable procedural hills")

    print("Nordschleife validation OK")
    print(f"  length={nord.length:.3f} m samples={len(nord.center)} elev_gain={nord.elev_gain:.2f} m")
    print(f"  max_grade={max_grade:.3f} min_launch={min_launch:.2f} m/s")
    print(f"  elevation_validation={nord.metadata.get('elevation_validation')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
