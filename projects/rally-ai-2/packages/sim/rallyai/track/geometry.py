"""Stage geometry: the corridor the car drives in and everything the agent can
see of it.

A ``Stage`` document (see ``shared/schemas/stage.schema.json``) is sparse and
human-editable. ``Track`` turns it into the dense, uniformly-sampled arrays the
physics and the sensors query thousands of times a second.

Three things make this fast enough to train against:

* the centerline is resampled onto a **uniform** ``ds`` grid, so "40 m ahead" is
  an index offset rather than a search;
* ``project`` starts from the caller's previous ``s`` and searches a small
  window, so the cost does not grow with stage length;
* ``raycast`` tests only the boundary segments and obstacles within beam range,
  gathered by arc length.

Coordinates are z-up: x east, y north, z up, yaw about +z with 0 facing +x.
Arc length ``s`` runs 0..length and **never wraps** — this is a point-to-point
stage, not a circuit. Look-ahead past the finish clamps to the finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal

import numpy as np

# Resolution of the dense internal grid. 1 m is far finer than the car can
# resolve at rally speeds (a 30 m/s car covers 1 m in 33 ms, one physics frame)
# and keeps a 2 km stage to a few thousand samples.
DS = 1.0

# Catmull-Rom samples per input segment before arc-length reparameterisation.
# The input centerline is sparse; interpolating it linearly would make curvature
# zero between knots and infinite at them, which the tyre model would feel as a
# kick every few metres.
SUBDIV = 8

# Smoothing windows, in metres. Both were set by measurement, not taste.
#
# Horizontal curvature needs none: inside a constant-radius corner the residual
# noise from the Catmull-Rom fit plus arc-length resample is 0.198% of the mean,
# and smoothing makes it *worse* (0.49% at a 2 m window) because the window
# averages across the curvature gradient at the corner's ease-in and ease-out.
# A 5 m window also flattened a 12 m hairpin to an apparent 14 m. The knob is
# kept, at zero, in case a generator ever produces rougher knots than the
# builder does.
CURV_SMOOTH_M = 0.0

# Vertical curvature needs a little smoothing — it is the second derivative of
# z — but far less than it once did, and this constant has been re-derived once
# already.
#
# It was 5 m, set against a measured jitter of 0.0046. That jitter turned out to
# be almost entirely the *builder* stepping grade between segments; once grade
# eased smoothly the noise at zero smoothing fell to 0.000275, seventeen times
# lower. Meanwhile a 5 m window spans 42% of a 26 m crest and was flattening
# short ones badly — a crest asked to launch the car at 20 m/s launched it at
# 23.5 instead.
#
# At 1 m the residual noise is 0.000214, under 1% of a real crest's vcurv, and
# crest fidelity is within 1.6%. The lesson worth keeping: this constant was
# justified by a measurement that a later fix invalidated.
VCURV_SMOOTH_M = 1.0

# Slack allowed when checking that surface segments cover the stage. Generous
# enough to absorb the difference between a declared length and a resampled one,
# far tighter than any real authoring mistake.
COVERAGE_TOL_M = 0.5

HitKind = Literal["edge", "obstacle"]

# Beam hit-type channel values. The agent is allowed to tell a road edge from a
# tree — a driver can — and the distinction matters enormously: one costs time,
# the other ends the episode.
HIT_EDGE = 0.0
HIT_OBSTACLE = 1.0


@dataclass(slots=True)
class TrackQuery:
    """Everything the car needs to know about the road beneath it."""

    s: float               # arc length along the centerline, metres
    lateral: float         # signed offset from centerline, +ve to the right of travel
    heading: float         # centerline tangent, radians about +z
    heading_error: float   # car yaw minus heading, wrapped to [-pi, pi]
    width: float           # full drivable width at s
    half_width: float
    camber: float          # bank angle, radians; +ve banks into a left-hander
    z: float               # road surface height beneath the car (camber-corrected)
    grade: float           # dz/ds, +ve uphill
    vcurv: float           # d2z/ds2, -ve is a crest
    curvature: float       # signed centerline curvature at s, 1/m
    mu: float              # peak friction of the surface at s
    surface: str
    on_track: bool         # inside the corridor
    progress: float        # 0..1 fraction of the stage completed


class Track:
    """Dense, queryable stage geometry built from a stage document."""

    def __init__(self, stage: dict[str, Any], ds: float = DS):
        self.stage = stage
        self.id: str = stage["id"]
        self.ds = float(ds)

        pts = stage["centerline"]
        knots = np.array([[p["x"], p["y"], p["z"], p["width"], p["camber"]] for p in pts],
                         dtype=np.float64)
        if len(knots) < 2:
            raise ValueError(f"stage {self.id}: centerline needs at least 2 points")

        self.declared_length = float(stage["length_m"])
        self._build_centerline(knots)
        self._build_surfaces(stage["surfaces"])
        self._build_obstacles(stage.get("obstacles", []))
        self._build_pace_notes(stage.get("pace_notes", []))

        self.start_s = float(np.clip(stage["start"]["s"], 0.0, self.length))
        self.start_heading = float(stage["start"]["heading"])
        # Clamped to the resampled length, not taken at face value. The declared
        # finish comes from the builder's integration; Track recovers its own
        # length from the spline it fits, and the two differ by a couple of
        # centimetres. Taking the declared value literally puts the finish line
        # a hair beyond the furthest s the car can ever project onto — the car
        # reaches 100% progress, never triggers the finish, and times out.
        self.finish_s = float(min(float(stage["finish"]["s"]), self.length))

    # ------------------------------------------------------------------ #
    # build
    # ------------------------------------------------------------------ #

    def _build_centerline(self, knots: np.ndarray) -> None:
        """Smooth the sparse knots, then resample onto a uniform arc-length grid."""
        fine = _catmull_rom(knots, SUBDIV)

        # True arc length of the smoothed curve, by chord accumulation.
        seg = np.linalg.norm(np.diff(fine[:, :2], axis=0), axis=1)
        s_fine = np.concatenate([[0.0], np.cumsum(seg)])
        length = float(s_fine[-1])

        n = max(2, round(length / self.ds) + 1)
        self.s = np.linspace(0.0, length, n)
        self.length = length
        # Actual spacing after fitting a whole number of samples to the length.
        self.ds = float(self.s[1] - self.s[0]) if n > 1 else self.ds

        self.x = np.interp(self.s, s_fine, fine[:, 0])
        self.y = np.interp(self.s, s_fine, fine[:, 1])
        self.z = np.interp(self.s, s_fine, fine[:, 2])
        self.width = np.interp(self.s, s_fine, fine[:, 3])
        self.camber = np.interp(self.s, s_fine, fine[:, 4])
        self.half_width = self.width * 0.5

        # Tangent, heading, curvature.
        dx = np.gradient(self.x, self.ds)
        dy = np.gradient(self.y, self.ds)
        self.heading = np.arctan2(dy, dx)
        unwrapped = np.unwrap(self.heading)
        curv = np.gradient(unwrapped, self.ds)
        self.curvature = _box_smooth(curv, round(CURV_SMOOTH_M / self.ds))

        # Vertical profile. grade is dz/ds; vcurv is its derivative, negative on
        # a crest — with its own speed, that is the agent's takeoff predictor.
        self.grade = np.gradient(self.z, self.ds)
        self.vcurv = _box_smooth(np.gradient(self.grade, self.ds),
                                 round(VCURV_SMOOTH_M / self.ds))

        # Right-hand normal in the ground plane: heading rotated -90 degrees.
        self.nx = np.sin(self.heading)
        self.ny = -np.cos(self.heading)

        # Corridor boundary polylines, used by raycast and collision.
        self.right_x = self.x + self.nx * self.half_width
        self.right_y = self.y + self.ny * self.half_width
        self.left_x = self.x - self.nx * self.half_width
        self.left_y = self.y - self.ny * self.half_width

    def _build_surfaces(self, surfaces: list[dict[str, Any]]) -> None:
        """Rasterise surface segments onto the dense grid for O(1) lookup.

        Gaps are validated against the *declared* segment bounds, then every
        sample is assigned by clamped lookup. Doing it in that order matters:
        the stage's declared ``length_m`` and the length Track recovers by
        resampling its own spline differ by a fraction of a millimetre, and
        rasterising by index would leave the final sample with no mu — a NaN
        straight into the tyre model. A real gap still raises.
        """
        if not surfaces:
            raise ValueError(f"stage {self.id}: no surface segments")

        segs = sorted(surfaces, key=lambda g: g["s_start"])
        for a, b in pairwise(segs):
            if abs(float(a["s_end"]) - float(b["s_start"])) > 1e-3:
                raise ValueError(
                    f"stage {self.id}: surfaces leave a gap between s={a['s_end']} "
                    f"and s={b['s_start']} — segments must tile with no gaps"
                )
        # Coverage is checked against the stage's DECLARED length, not the length
        # recovered by resampling. Those differ by a couple of centimetres — a
        # spline is fractionally longer than the knots it was fit through — and
        # comparing the two would reject every valid stage.
        if float(segs[0]["s_start"]) > COVERAGE_TOL_M:
            raise ValueError(f"stage {self.id}: surfaces do not start at s=0")
        if float(segs[-1]["s_end"]) < self.declared_length - COVERAGE_TOL_M:
            raise ValueError(
                f"stage {self.id}: surfaces end at s={segs[-1]['s_end']} but the "
                f"stage runs to {self.declared_length} — {self.declared_length - float(segs[-1]['s_end']):.1f} m uncovered"
            )

        self.surface_names = list(dict.fromkeys(g["type"] for g in segs))
        name_index = {n: i for i, n in enumerate(self.surface_names)}

        starts = np.array([float(g["s_start"]) for g in segs])
        which = np.clip(np.searchsorted(starts, self.s, side="right") - 1, 0, len(segs) - 1)

        self.mu = np.array([float(g["mu"]) for g in segs])[which]
        self.surface_idx = np.array([name_index[g["type"]] for g in segs],
                                    dtype=np.int16)[which]

    def _build_obstacles(self, obstacles: list[dict[str, Any]]) -> None:
        """Store obstacles sorted by arc length so beam-range gathering is a slice."""
        self.n_obstacles = len(obstacles)
        if not obstacles:
            self.obs_x = np.zeros(0)
            self.obs_y = np.zeros(0)
            self.obs_r = np.zeros(0)
            self.obs_s = np.zeros(0)
            return

        ox = np.array([o["x"] for o in obstacles], dtype=np.float64)
        oy = np.array([o["y"] for o in obstacles], dtype=np.float64)
        orad = np.array([o["radius"] for o in obstacles], dtype=np.float64)
        # Trust a stored s if the author supplied one; otherwise derive it.
        if all("s" in o for o in obstacles):
            os_ = np.array([o["s"] for o in obstacles], dtype=np.float64)
        else:
            os_ = np.array([self._nearest_s_global(px, py) for px, py in zip(ox, oy)])

        order = np.argsort(os_)
        self.obs_x, self.obs_y, self.obs_r, self.obs_s = (
            ox[order], oy[order], orad[order], os_[order],
        )

    def _build_pace_notes(self, notes: list[dict[str, Any]]) -> None:
        self.pace_notes = sorted(notes, key=lambda n: n["s"])
        self._pace_s = np.array([n["s"] for n in self.pace_notes], dtype=np.float64)

    # ------------------------------------------------------------------ #
    # projection — the hot path
    # ------------------------------------------------------------------ #

    def _nearest_s_global(self, x: float, y: float) -> float:
        """Nearest centerline arc length, searching the whole stage. Build-time
        only; ``project`` uses the windowed version."""
        i = int(np.argmin((self.x - x) ** 2 + (self.y - y) ** 2))
        return float(self.s[i])

    def project(self, x: float, y: float, yaw: float = 0.0,
                hint_s: float | None = None, window_m: float = 40.0) -> TrackQuery:
        """Locate the car on the stage.

        ``hint_s`` is the previous result. Supplying it turns a whole-stage
        search into a windowed one, which is what keeps step cost independent of
        stage length. Omit it only on reset or when the car may have teleported.
        """
        n = len(self.s)
        if hint_s is None:
            i = int(np.argmin((self.x - x) ** 2 + (self.y - y) ** 2))
        else:
            w = max(2, int(window_m / self.ds))
            c = int(np.clip(hint_s / self.ds, 0, n - 1))
            lo, hi = max(0, c - w), min(n, c + w + 1)
            i = lo + int(np.argmin((self.x[lo:hi] - x) ** 2 + (self.y[lo:hi] - y) ** 2))

        # Refine analytically onto the better of the two adjacent segments, so
        # accuracy is not limited by ds.
        s_ref, lat = self._refine(x, y, i)
        t = s_ref / self.ds
        i0 = int(np.clip(np.floor(t), 0, n - 2))
        f = float(np.clip(t - i0, 0.0, 1.0))

        heading = _lerp_angle(self.heading[i0], self.heading[i0 + 1], f)
        width = _lerp(self.width, i0, f)
        camber = _lerp(self.camber, i0, f)
        z_center = _lerp(self.z, i0, f)
        half = width * 0.5

        # Camber tilts the road across its width, so the surface under a car out
        # near the edge is not at the centerline height.
        z = z_center + lat * np.sin(camber)

        heading_err = _wrap(yaw - heading)

        return TrackQuery(
            s=s_ref,
            lateral=lat,
            heading=heading,
            heading_error=heading_err,
            width=width,
            half_width=half,
            camber=camber,
            z=z,
            grade=_lerp(self.grade, i0, f),
            vcurv=_lerp(self.vcurv, i0, f),
            curvature=_lerp(self.curvature, i0, f),
            mu=_lerp(self.mu, i0, f),
            surface=self.surface_names[self.surface_idx[i0]],
            on_track=abs(lat) <= half,
            progress=float(np.clip(s_ref / self.finish_s, 0.0, 1.0)),
        )

    def _refine(self, x: float, y: float, i: int) -> tuple[float, float]:
        """Project onto the segments either side of sample ``i``.

        Returns (arc length, signed lateral offset). Signed positive to the
        right of travel, matching the stage schema.
        """
        n = len(self.s)
        best_s = float(self.s[i])
        best_lat = 0.0
        best_d2 = np.inf

        for j in (i - 1, i):
            if j < 0 or j + 1 >= n:
                continue
            ax, ay = self.x[j], self.y[j]
            bx, by = self.x[j + 1], self.y[j + 1]
            ex, ey = bx - ax, by - ay
            elen2 = ex * ex + ey * ey
            if elen2 < 1e-12:
                continue
            t = ((x - ax) * ex + (y - ay) * ey) / elen2
            t = min(1.0, max(0.0, t))
            px, py = ax + t * ex, ay + t * ey
            dx, dy = x - px, y - py
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best_s = float(self.s[j] + t * self.ds)
                # Sign by the segment's right-hand normal.
                inv = 1.0 / np.sqrt(elen2)
                rx, ry = ey * inv, -ex * inv
                best_lat = float(dx * rx + dy * ry)

        return best_s, best_lat

    # ------------------------------------------------------------------ #
    # vision
    # ------------------------------------------------------------------ #

    def raycast(self, x: float, y: float, yaw: float, angles: np.ndarray,
                max_range: float, hint_s: float | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cast beams and return what each one hits.

        Returns ``(distances, hit_kinds, points)``:

        * ``distances`` — metres to the first hit, clamped to ``max_range``;
        * ``hit_kinds`` — ``HIT_EDGE`` or ``HIT_OBSTACLE`` per beam;
        * ``points`` — ``(n, 2)`` world hit positions, for the debug overlay.

        Corridor edges are included because the agent must see where the road
        runs out, even though crossing one is survivable. The kind channel keeps
        that distinct from a tree, which is not.
        """
        angles = np.asarray(angles, dtype=np.float64)
        wa = yaw + angles
        dx, dy = np.cos(wa), np.sin(wa)

        dists = np.full(len(angles), float(max_range))
        kinds = np.full(len(angles), HIT_EDGE)

        s_here = hint_s if hint_s is not None else self._nearest_s_global(x, y)
        self._ray_edges(x, y, dx, dy, max_range, s_here, dists)
        self._ray_obstacles(x, y, dx, dy, max_range, s_here, dists, kinds)

        points = np.stack([x + dx * dists, y + dy * dists], axis=1)
        return dists, kinds, points

    def _ray_edges(self, x, y, dx, dy, max_range, s_here, dists) -> None:
        """Vectorised ray-vs-segment against both corridor boundaries."""
        n = len(self.s)
        w = int(max_range / self.ds) + 2
        c = int(np.clip(s_here / self.ds, 0, n - 1))
        lo, hi = max(0, c - w), min(n - 1, c + w)
        if hi <= lo:
            return

        ax = np.concatenate([self.left_x[lo:hi], self.right_x[lo:hi]])
        ay = np.concatenate([self.left_y[lo:hi], self.right_y[lo:hi]])
        bx = np.concatenate([self.left_x[lo + 1:hi + 1], self.right_x[lo + 1:hi + 1]])
        by = np.concatenate([self.left_y[lo + 1:hi + 1], self.right_y[lo + 1:hi + 1]])

        ex, ey = bx - ax, by - ay
        apx, apy = ax - x, ay - y

        # (beams, segments)
        denom = np.outer(dx, ey) - np.outer(dy, ex)
        safe = np.where(np.abs(denom) > 1e-9, denom, 1.0)
        t = (apx * ey - apy * ex)[None, :] / safe          # along the ray
        u = (np.outer(dy, apx) - np.outer(dx, apy)) / safe  # along the segment

        valid = (np.abs(denom) > 1e-9) & (t >= 0) & (t <= max_range) & (u >= 0) & (u <= 1)
        hit = np.where(valid, t, np.inf).min(axis=1)
        np.minimum(dists, np.where(np.isfinite(hit), hit, max_range), out=dists)

    def _ray_obstacles(self, x, y, dx, dy, max_range, s_here, dists, kinds) -> None:
        """Vectorised ray-vs-circle against obstacles within beam range."""
        if self.n_obstacles == 0:
            return
        lo = np.searchsorted(self.obs_s, s_here - max_range - 5.0)
        hi = np.searchsorted(self.obs_s, s_here + max_range + 5.0)
        if hi <= lo:
            return

        qx, qy, qr = self.obs_x[lo:hi], self.obs_y[lo:hi], self.obs_r[lo:hi]
        mx, my = x - qx, y - qy                                # (k,)
        b = np.outer(dx, mx) + np.outer(dy, my)                # (beams, k)
        c = (mx * mx + my * my - qr * qr)[None, :]
        disc = b * b - c

        root = np.sqrt(np.maximum(disc, 0.0))
        t_near = -b - root
        t_far = -b + root
        # If the ray starts inside the circle the near root is negative; the far
        # root is the exit. Either way we want the first non-negative hit.
        t = np.where(t_near >= 0, t_near, t_far)
        valid = (disc >= 0) & (t >= 0) & (t <= max_range)

        hit = np.where(valid, t, np.inf).min(axis=1)
        closer = hit < dists
        dists[closer] = hit[closer]
        kinds[closer] = HIT_OBSTACLE

    # ------------------------------------------------------------------ #
    # look-ahead
    # ------------------------------------------------------------------ #

    def _ahead_idx(self, s: float, distances) -> np.ndarray:
        """Sample indices at s + each distance ahead, **clamped** at the finish.

        Clamped, not wrapped. Supra's circuit model wraps modulo lap length;
        doing that here would show the agent the start line as upcoming road.
        """
        targets = np.asarray(distances, dtype=np.float64) + s
        idx = np.rint(targets / self.ds).astype(np.int64)
        return np.clip(idx, 0, len(self.s) - 1)

    def lookahead_curvature(self, s: float, distances) -> np.ndarray:
        return self.curvature[self._ahead_idx(s, distances)]

    def lookahead_grade(self, s: float, distances) -> np.ndarray:
        return self.grade[self._ahead_idx(s, distances)]

    def lookahead_vcurv(self, s: float, distances) -> np.ndarray:
        return self.vcurv[self._ahead_idx(s, distances)]

    def lookahead_mu(self, s: float, distances) -> np.ndarray:
        return self.mu[self._ahead_idx(s, distances)]

    def lookahead_width(self, s: float, distances) -> np.ndarray:
        return self.width[self._ahead_idx(s, distances)]

    def pace_ahead(self, s: float, horizon: float = 60.0) -> dict[str, Any] | None:
        """The next pace note within ``horizon`` metres, or None."""
        if not self.pace_notes:
            return None
        i = int(np.searchsorted(self._pace_s, s, side="left"))
        if i >= len(self.pace_notes):
            return None
        note = self.pace_notes[i]
        return note if note["s"] - s <= horizon else None

    # ------------------------------------------------------------------ #
    # collision
    # ------------------------------------------------------------------ #

    def collide_obstacles(self, x: float, y: float, radius: float,
                          hint_s: float | None = None
                          ) -> tuple[np.ndarray | None, float]:
        """Deepest obstacle overlap as (outward normal, penetration depth)."""
        if self.n_obstacles == 0:
            return None, 0.0
        s_here = hint_s if hint_s is not None else self._nearest_s_global(x, y)
        reach = radius + float(self.obs_r.max()) + 1.0
        lo = np.searchsorted(self.obs_s, s_here - reach)
        hi = np.searchsorted(self.obs_s, s_here + reach)
        if hi <= lo:
            return None, 0.0

        dx = x - self.obs_x[lo:hi]
        dy = y - self.obs_y[lo:hi]
        dist = np.hypot(dx, dy)
        depth = (self.obs_r[lo:hi] + radius) - dist
        k = int(np.argmax(depth))
        if depth[k] <= 0:
            return None, 0.0

        d = dist[k]
        if d < 1e-9:
            return np.array([1.0, 0.0]), float(depth[k])
        return np.array([dx[k] / d, dy[k] / d]), float(depth[k])

    # ------------------------------------------------------------------ #
    # invariants
    # ------------------------------------------------------------------ #

    def obstacles_in_corridor(self, car_radius: float = 0.0) -> list[int]:
        """Indices of obstacles that intrude into the drivable corridor.

        The North Star's rule is that nothing which can end the episode may sit
        inside the corridor unless the agent can sense it. The agent *can* sense
        obstacles by raycast, so this is not a hard failure — but a generator
        dropping trees on the racing line is almost always a bug, and this is how
        you find out.
        """
        out: list[int] = []
        for k in range(self.n_obstacles):
            q = self.project(float(self.obs_x[k]), float(self.obs_y[k]),
                             hint_s=float(self.obs_s[k]))
            if abs(q.lateral) - self.obs_r[k] - car_radius < q.half_width:
                out.append(k)
        return out


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def _lerp(arr: np.ndarray, i: int, f: float) -> float:
    return float(arr[i] + (arr[i + 1] - arr[i]) * f)


def _lerp_angle(a: float, b: float, f: float) -> float:
    """Interpolate angles the short way round, so the seam at +/-pi is smooth."""
    return float(a + _wrap(b - a) * f)


def _box_smooth(v: np.ndarray, half_width: int) -> np.ndarray:
    """Centred moving average with edge-clamped padding."""
    if half_width < 1:
        return v
    k = 2 * half_width + 1
    padded = np.pad(v, half_width, mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(padded, kernel, mode="valid")


def _catmull_rom(knots: np.ndarray, subdiv: int, alpha: float = 0.5) -> np.ndarray:
    """**Centripetal** Catmull-Rom through ``knots`` (Barry-Goldman form).

    Centripetal (``alpha = 0.5``) rather than uniform parameterisation, because
    the generator emits knots at *non-uniform* spacing — dense through hairpins,
    sparse on straights. A uniformly-parameterised spline overshoots wherever
    that spacing changes abruptly, which showed up as 30 m corners measuring
    26 m: the curve bulged past its own knots on the way into the dense region.
    Centripetal parameterisation is the standard cure and provably admits no
    cusps or self-intersections.

    Parameterisation uses the spatial columns (x, y, z) only; width and camber
    ride along on the same parameter so they stay in step with position.
    """
    n = len(knots)
    if n == 2:
        t = np.linspace(0.0, 1.0, subdiv + 1)[:, None]
        return knots[0] + (knots[1] - knots[0]) * t

    # Reflect the endpoints so the curve passes through the first and last knot
    # with a sensible tangent instead of flattening.
    pad = np.vstack([2 * knots[0] - knots[1], knots, 2 * knots[-1] - knots[-2]])

    seg_len = np.linalg.norm(np.diff(pad[:, :3], axis=0), axis=1)
    knot_t = np.concatenate([[0.0], np.cumsum(np.maximum(seg_len, 1e-9) ** alpha)])

    p0, p1, p2, p3 = pad[0:n - 1], pad[1:n], pad[2:n + 1], pad[3:n + 2]
    t0, t1, t2, t3 = knot_t[0:n - 1], knot_t[1:n], knot_t[2:n + 1], knot_t[3:n + 2]

    u = np.linspace(0.0, 1.0, subdiv, endpoint=False)[None, :]
    t = t1[:, None] + u * (t2 - t1)[:, None]                      # (seg, subdiv)

    def blend(a, b, ta, tb):
        """Linear blend of two point sets across a parameter interval."""
        w = ((t - ta[:, None]) / np.maximum(tb - ta, 1e-12)[:, None])[:, :, None]
        return a * (1.0 - w) + b * w

    a1 = blend(p0[:, None, :], p1[:, None, :], t0, t1)
    a2 = blend(p1[:, None, :], p2[:, None, :], t1, t2)
    a3 = blend(p2[:, None, :], p3[:, None, :], t2, t3)
    b1 = blend(a1, a2, t0, t2)
    b2 = blend(a2, a3, t1, t3)
    c = blend(b1, b2, t1, t2)

    return np.vstack([c.reshape(-1, knots.shape[1]), knots[-1][None, :]])
