"""Racing-line optimization over the real track geometry.

Turns the centerline + width into an optimized line and an ORACLE lap time,
judged by exactly the same physics core as the centerline reference
(`compute_speed_envelope_geometry`): same tyre/downforce model, same brake
authority, same active-aero drag states, same grade handling.

Method — iterated linearized minimum-curvature (the standard lap-sim
approach, e.g. Heilmeier et al. 2019):

  * the line is a lateral offset alpha(s) along the centerline's left normal,
    bounded so the car's certified collision footprint stays inside the
    track edges with a safety margin;
  * path curvature is linearized as kappa(alpha) ~ kappa_now + D2 (alpha -
    alpha_now) (D2 = periodic second difference by arc length), giving a
    bounded sparse least-squares problem per iteration
    (scipy.optimize.lsq_linear), with a small first-difference regularizer
    for smoothness;
  * after each solve the EXACT polyline curvature of the offset path is
    recomputed (non-uniform finite differences) and the linearization is
    repeated — 4-6 iterations converge on this track;
  * the oracle lap time is the envelope core run over the line's own
    (ds, curvature, grade) geometry.

Minimum-curvature is a geometry proxy for minimum time (within ~1% on
published comparisons); the honest claim exported here is "a concrete,
track-legal line whose physics-consistent profile yields T seconds", not a
proven global optimum.
"""
from __future__ import annotations

import hashlib
import json
import math
import time as _time

import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear

from .fable5 import compute_speed_envelope_geometry


def _polyline_curvature(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed curvature + segment lengths of a closed polyline (non-uniform)."""
    d = np.roll(P, -1, axis=0) - P
    ds = np.linalg.norm(d, axis=1)
    ds = np.maximum(ds, 1e-9)
    dsm = np.roll(ds, 1)                      # segment arriving at i
    # non-uniform central first/second derivatives wrt arc length
    Pp = (np.roll(P, -1, axis=0) - np.roll(P, 1, axis=0)) / (ds + dsm)[:, None]
    Ppp = 2.0 * ((np.roll(P, -1, axis=0) - P) / ds[:, None]
                 - (P - np.roll(P, 1, axis=0)) / dsm[:, None]) / (ds + dsm)[:, None]
    num = Pp[:, 0] * Ppp[:, 1] - Pp[:, 1] * Ppp[:, 0]
    den = np.power(np.maximum(Pp[:, 0] ** 2 + Pp[:, 1] ** 2, 1e-12), 1.5)
    return num / den, ds


def _second_diff_matrix(n: int, h: float) -> sparse.csr_matrix:
    """Periodic second-difference operator (1/h^2 scaling)."""
    main = np.full(n, -2.0)
    off = np.ones(n)
    D = sparse.diags([main, off, off], [0, 1, -1], (n, n), format="lil")
    D[0, n - 1] = 1.0
    D[n - 1, 0] = 1.0
    return (D.tocsr()) / (h * h)


def _first_diff_matrix(n: int, h: float) -> sparse.csr_matrix:
    off = np.ones(n)
    D = sparse.diags([-off, off], [0, 1], (n, n), format="lil")
    D[n - 1, 0] = 1.0
    return (D.tocsr()) / h


def optimize_raceline(trk, car, *, margin: float = 0.30, iters: int = 5,
                      smooth_reg: float = 0.02, verbose: bool = True) -> dict:
    """Optimize the line for `car` on `trk`; return geometry + oracle profile."""
    C = np.asarray(trk.center, dtype=float)
    n = len(C)
    ds0 = np.asarray(trk.seg_len, dtype=float)
    h = float(np.mean(ds0))
    grade = np.asarray(trk.grade, dtype=float)
    half_w = np.asarray(trk.half_width, dtype=float)

    # certified collision half-width (same rule as Vehicle.get_obb)
    car_half_w = (float(car.body_width) / 2.0
                  if getattr(car, "body_width", 0.0) > 0
                  else (float(car.track_width) + 0.36) / 2.0)
    bound = half_w - car_half_w - float(margin)
    if np.any(bound <= 0):
        raise ValueError("track too narrow for this car + margin")

    # left normal of the centerline
    T = (np.roll(C, -1, axis=0) - np.roll(C, 1, axis=0))
    T /= np.maximum(np.linalg.norm(T, axis=1), 1e-9)[:, None]
    N = np.stack([-T[:, 1], T[:, 0]], axis=1)

    D2 = _second_diff_matrix(n, h)
    D1 = _first_diff_matrix(n, h)
    A = sparse.vstack([D2, math.sqrt(smooth_reg) * D1]).tocsr()

    alpha = np.zeros(n)
    history = []
    for it in range(int(iters)):
        P = C + alpha[:, None] * N
        kappa, ds = _polyline_curvature(P)
        prof = compute_speed_envelope_geometry(car, ds, kappa, grade)
        history.append(round(prof["lap_time"], 3))
        if verbose:
            print(f"[raceline] iter {it}: lap {prof['lap_time']:.2f}s "
                  f"(max|alpha| {np.max(np.abs(alpha)):.2f} m)", flush=True)
        # linearize: kappa(beta) ~ kappa - D2 alpha + D2 beta -> LSQ in beta
        rhs = np.concatenate([-(kappa - D2 @ alpha), np.zeros(n)])
        sol = lsq_linear(A, rhs, bounds=(-bound, bound),
                         tol=1e-8, max_iter=60, verbose=0)
        beta = sol.x
        alpha = alpha + 0.75 * (beta - alpha)
        alpha = np.clip(alpha, -bound, bound)

    P = C + alpha[:, None] * N
    kappa, ds = _polyline_curvature(P)
    prof = compute_speed_envelope_geometry(car, ds, kappa, grade)
    history.append(round(prof["lap_time"], 3))

    # post-hoc legality: bound already reserves footprint half-width + margin;
    # the mid-body corner-cut sagitta (L^2 * kappa / 8) must fit in the margin
    half_len = (float(car.body_length) / 2.0
                if getattr(car, "body_length", 0.0) > 0
                else (float(car.wheelbase) + 1.7) / 2.0)
    sagitta = (2.0 * half_len) ** 2 * np.abs(kappa) / 8.0
    legal = bool(np.all(np.abs(alpha) <= bound + 1e-9)
                 and float(np.max(sagitta)) <= margin + 1e-6)

    return {
        "alpha": alpha,
        "points": P,
        "ds": ds,
        "curvature": kappa,
        "grade": grade,
        "profile": prof,
        "lap_time": float(prof["lap_time"]),
        "iters_history": history,
        "legal": legal,
        "max_sagitta_m": float(np.max(sagitta)),
        "bound_m": float(np.min(bound)),
        "params": {"margin": margin, "iters": iters, "smooth_reg": smooth_reg,
                   "car": car.name,
                   "drivetrain_version": getattr(car, "drivetrain_version", "?"),
                   "car_half_width_m": car_half_w},
    }


def raceline_manifest(res: dict, trk) -> dict:
    """JSON-safe provenance record (geometry arrays hashed, profile inline)."""
    geom = np.concatenate([res["alpha"], res["curvature"], res["ds"]])
    return {
        "schema": "raceline-oracle-v1",
        "created_unix": _time.time(),
        "track": {"length_m": float(trk.length), "points": int(len(trk.center)),
                  "half_width_m": float(np.min(trk.half_width))},
        "geometry_sha256": hashlib.sha256(
            np.ascontiguousarray(geom).tobytes()).hexdigest(),
        "oracle_lap_s": res["lap_time"],
        "iters_history": res["iters_history"],
        "legal": res["legal"],
        "max_sagitta_m": round(res["max_sagitta_m"], 4),
        "usable_offset_m": round(res["bound_m"], 3),
        "profile_params": res["profile"]["params"],
        "line_params": res["params"],
        "method": "iterated linearized minimum-curvature (bounded sparse LSQ) "
                  "+ envelope-core speed profile; min-curvature is a proxy for "
                  "min-time — this is a feasible line, not a proven optimum",
    }
