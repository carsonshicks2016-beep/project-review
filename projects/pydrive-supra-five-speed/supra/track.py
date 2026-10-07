"""
Procedural race tracks built from centripetal Catmull-Rom splines.

A Track is a closed loop of densely-sampled centreline points plus left/right
boundaries offset by the track width. It also answers the queries the driving
loop (and later the AI) needs:

  * nearest centreline point + arc-length progress,
  * signed lateral offset from the centreline,
  * local curvature.

The vision raycasts come in the next slice; for the drivable slice we only need
geometry + centreline tracking.
"""
from __future__ import annotations

import json
from pathlib import Path
import zlib

import numpy as np

G = 9.81   # gravity — elevation grade/launch-speed math (PHYSICS_3D_PLAN)


def _stable_seed(name: str) -> int:
    """A process-INDEPENDENT seed from a name. (Python's built-in hash() is
    salted per process via PYTHONHASHSEED, so `hash(name) % 9999` produced a
    DIFFERENT track every launch — a specialist trained on one 'akina' was then
    watched/evaluated on a different 'akina' it had never seen, and looked
    broken. CRC32 is stable across processes, so a named track is now the SAME
    layout everywhere: train, eval, watch, and thumbnail all agree.)"""
    return zlib.crc32(name.encode("utf-8")) % 9999


def _catmull_rom_centripetal(points: np.ndarray, samples_per_seg: int = 24,
                             alpha: float = 0.5) -> np.ndarray:
    """Closed centripetal Catmull-Rom spline through `points` (N,2)."""
    n = len(points)
    out = []
    for i in range(n):
        p0 = points[(i - 1) % n]
        p1 = points[i % n]
        p2 = points[(i + 1) % n]
        p3 = points[(i + 2) % n]

        def tj(ti, pi, pj):
            d = np.linalg.norm(pj - pi)
            return ti + max(d, 1e-6) ** alpha

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)

        for t in np.linspace(t1, t2, samples_per_seg, endpoint=False):
            a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
            b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
            c = (t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2
            out.append(c)
    return np.array(out)


def _grade_vcurv(z: np.ndarray, seg_len: np.ndarray):
    """Periodic central differences of a height profile along arc length.

    Returns (grade, vcurv). grade = dz/ds (+ = climbing). vcurv = d(grade)/ds:
    NEGATIVE at crests, POSITIVE in dips — the sign convention every formula
    in PHYSICS_3D_PLAN assumes. seg_len[i] is the arc distance i -> i+1, so
    the span of the central difference at i is seg_len[i-1] + seg_len[i].
    """
    span = np.roll(seg_len, 1) + seg_len
    grade = (np.roll(z, -1) - np.roll(z, 1)) / span
    vcurv = (np.roll(grade, -1) - np.roll(grade, 1)) / span
    return grade, vcurv


class Track:
    def __init__(self, centerline: np.ndarray, width: float = 12.0,
                 elevation: np.ndarray | None = None,
                 bank: np.ndarray | None = None,
                 metadata: dict | None = None,
                 landmarks: list[dict] | None = None,
                 sectors: list[dict] | None = None,
                 width_profile: np.ndarray | None = None):
        self.center = centerline                      # (M, 2)
        self.width = width
        self.half = width / 2.0
        self._elev_init = elevation
        self._bank_init = bank
        self._width_profile_init = width_profile
        self.metadata = dict(metadata or {})
        self.landmarks = list(landmarks or [])
        self.sectors = list(sectors or [])
        self.source_confidence = self.metadata.get("source_confidence")
        self.real_track = bool(self.metadata.get("real_track", False))
        self._build()

    # ------------------------------------------------------------------ #
    def _build(self):
        c = self.center
        # tangents (closed loop) -> unit normals (left-hand)
        nxt = np.roll(c, -1, axis=0)
        prv = np.roll(c, 1, axis=0)
        tang = nxt - prv
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9
        self.tangent = tang
        self.normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)   # left normal
        if self._width_profile_init is None:
            self.width_profile = np.full(len(c), float(self.width), dtype=float)
        else:
            self.width_profile = np.asarray(self._width_profile_init, dtype=float)
            if len(self.width_profile) != len(c):
                raise ValueError("width_profile must match the centerline length")
            self.width = float(np.mean(self.width_profile))
        self.half_width = self.width_profile / 2.0
        self.half = float(np.mean(self.half_width))
        self.left = c + self.normal * self.half_width[:, None]
        self.right = c - self.normal * self.half_width[:, None]
        # cumulative arc length
        seg = np.linalg.norm(np.diff(c, axis=0, append=c[:1]), axis=1)
        self.seg_len = seg
        self.arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.length = float(seg.sum())
        # signed curvature per point (1/R, + = left turn)
        self.curvature = self._curvature()
        # height field + banking (zeros unless provided — flat by default)
        self.set_elevation(self._elev_init, self._bank_init)
        self._nearest_tree = None
        if len(c) > 2048:
            try:
                from scipy.spatial import cKDTree
                self._nearest_tree = cKDTree(c)
            except Exception:
                self._nearest_tree = None

    def _curvature(self) -> np.ndarray:
        c = self.center
        p = np.roll(c, 1, axis=0)
        n = np.roll(c, -1, axis=0)
        d1 = c - p
        d2 = n - c
        cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        denom = (np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1)
                 * np.linalg.norm(n - p, axis=1)) + 1e-9
        return 2.0 * cross / denom

    def set_elevation(self, elevation: np.ndarray | None = None,
                      bank: np.ndarray | None = None):
        """Install (or clear) the height field + banking and rebuild every
        derived quantity. The single entry point for elevation — generators,
        the `--flat` lever, and synthetic test tracks all come through here."""
        M = len(self.center)
        z = (np.zeros(M) if elevation is None
             else np.asarray(elevation, dtype=float))
        b = (np.zeros(M) if bank is None
             else np.asarray(bank, dtype=float))
        if len(z) != M or len(b) != M:
            raise ValueError(
                f"elevation/bank must match the centerline: got "
                f"{len(z)}/{len(b)} values for {M} samples")
        self.z = z
        self.bank = b
        self.grade, self.vcurv = _grade_vcurv(z, self.seg_len)
        self.elev_gain = float(z.max() - z.min())
        # speed at which each crest stops holding the car down (inf = never)
        ls = np.full(M, np.inf)
        crest = self.vcurv < -1e-9
        ls[crest] = np.sqrt(G / -self.vcurv[crest])
        self.launch_speed = ls

    # ------------------------------------------------------------------ #
    def nearest(self, x: float, y: float) -> int:
        """Index of the closest centreline sample to a world point."""
        if self._nearest_tree is not None:
            _, i = self._nearest_tree.query([x, y], k=1)
            return int(i)
        d = (self.center[:, 0] - x) ** 2 + (self.center[:, 1] - y) ** 2
        return int(np.argmin(d))

    def frame(self, x: float, y: float) -> dict:
        """Centreline-relative info at a world point."""
        i = self.nearest(x, y)
        rel = np.array([x, y]) - self.center[i]
        lateral = float(np.dot(rel, self.normal[i]))   # signed, + = left of line
        half_width = float(self.half_width[i])
        return {
            "index": i,
            "progress": self.arc[i] / self.length,
            "arc": self.arc[i],
            "lateral": lateral,
            "off_track": abs(lateral) > half_width,
            "half_width": half_width,
            "width": float(self.width_profile[i]),
            "curvature": float(self.curvature[i]),
            "heading": float(np.arctan2(self.tangent[i, 1], self.tangent[i, 0])),
            "z": float(self.z[i]),
            "grade": float(self.grade[i]),
            "bank": float(self.bank[i]),
            "vcurv": float(self.vcurv[i]),
        }

    def start_pose(self) -> tuple[float, float, float]:
        """Start on the centreline at index 0, facing along the track."""
        return self.pose_at(0)

    def pose_at(self, index: int) -> tuple[float, float, float]:
        """Pose on the centreline at an arbitrary sample index, facing along the
        track. Used for 'exploring starts' so training can reset the car anywhere
        on the circuit, not just the start line."""
        i = int(index) % len(self.center)
        x, y = self.center[i]
        yaw = float(np.arctan2(self.tangent[i, 1], self.tangent[i, 0]))
        return float(x), float(y), yaw

    @property
    def min_radius(self) -> float:
        k = np.abs(self.curvature)
        k = k[k > 1e-6]
        return float(1.0 / k.max()) if len(k) else float("inf")

    # ------------------------------------------------------------------ #
    # vision: raycasts against the track walls
    # ------------------------------------------------------------------ #
    def raycast(self, x: float, y: float, yaw: float, angles: np.ndarray,
                max_range: float, window: int = 55):
        """Cast rays from (x, y) at world heading yaw + each angle, returning
        the distance to the nearest track wall per beam (capped at max_range)
        and the beam end points (for rendering).

        Only boundary segments within an index window around the car's nearest
        centreline point are tested — fast, and reused unchanged by the AI env.
        """
        M = len(self.center)
        i = self.nearest(x, y)
        idx = (i + np.arange(-window, window + 1)) % M
        nxt = (idx + 1) % M
        seg_a = np.concatenate([self.left[idx], self.right[idx]])
        seg_b = np.concatenate([self.left[nxt], self.right[nxt]])
        edge = seg_b - seg_a                       # (k, 2)
        P = np.array([x, y])
        ap = seg_a - P                             # (k, 2)

        # vectorised ray-segment test: (B beams) x (k segments) at once
        wa = yaw + np.asarray(angles)                  # (B,)
        dx, dy = np.cos(wa), np.sin(wa)                # (B,)
        denom = np.outer(dx, edge[:, 1]) - np.outer(dy, edge[:, 0])     # (B,k)
        apxe = ap[:, 0] * edge[:, 1] - ap[:, 1] * edge[:, 0]            # (k,)  ap x edge
        safe = np.where(np.abs(denom) > 1e-9, denom, 1.0)
        t = apxe[None, :] / safe                                        # (B,k) along ray
        u = (np.outer(dy, ap[:, 0]) - np.outer(dx, ap[:, 1])) / safe    # (B,k) along seg
        valid = (np.abs(denom) > 1e-9) & (t >= 0) & (t <= max_range) & (u >= 0) & (u <= 1)
        t = np.where(valid, t, np.inf)
        dists = t.min(axis=1)
        dists = np.where(np.isfinite(dists), dists, float(max_range))
        points = np.stack([x + dx * dists, y + dy * dists], axis=1)
        return dists, points

    # ------------------------------------------------------------------ #
    # look-ahead preview of the road geometry
    # ------------------------------------------------------------------ #
    def _lookahead_idx(self, arc: float, distances) -> np.ndarray:
        """Centreline sample indices at arc + each distance ahead (wrapped)."""
        targets = (arc + np.asarray(distances, dtype=float)) % self.length
        return np.searchsorted(self.arc, targets) % len(self.arc)

    def lookahead_curvature(self, arc: float, distances) -> np.ndarray:
        """Signed centreline curvature sampled at arc + each distance ahead."""
        return self.curvature[self._lookahead_idx(arc, distances)]

    def lookahead_grade(self, arc: float, distances) -> np.ndarray:
        """Road grade (dz/ds) sampled at arc + each distance ahead."""
        return self.grade[self._lookahead_idx(arc, distances)]

    def lookahead_vcurv(self, arc: float, distances) -> np.ndarray:
        """Vertical curvature (crest/dip, − = crest) at arc + each distance
        ahead — with own speed, the agent's takeoff predictor."""
        return self.vcurv[self._lookahead_idx(arc, distances)]


# --------------------------------------------------------------------------- #
# generators
# --------------------------------------------------------------------------- #
def oval(width: float = 14.0) -> Track:
    """A simple, forgiving test oval — good for first-drive sanity checks."""
    pts = np.array([
        [-120, -60], [0, -75], [120, -60],
        [150, 0], [120, 60], [0, 75], [-120, 60], [-150, 0],
    ], dtype=float)
    return Track(_catmull_rom_centripetal(pts, samples_per_seg=30), width=width)


def random_circuit(seed: int | None = None, width: float = 12.0,
                   n_points: int = 10, radius: float = 140.0,
                   min_radius: float = 28.0) -> Track:
    """Closed circuit from jittered control points on a ring, relaxed so the
    minimum corner radius stays drivable (>= ~min_radius metres)."""
    rng = np.random.default_rng(seed)
    angles = np.sort(rng.uniform(0, 2 * np.pi, n_points))
    radii = radius * rng.uniform(0.6, 1.15, n_points)
    pts = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)

    # a few rounds of Laplacian relaxation to smooth pinch points
    for _ in range(3):
        prv = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        pts = 0.5 * pts + 0.25 * (prv + nxt)

    track = Track(_catmull_rom_centripetal(pts, samples_per_seg=24), width=width)
    # if it's still too tight, inflate toward the centroid and rebuild
    tries = 0
    while track.min_radius < min_radius and tries < 6:
        centroid = pts.mean(axis=0)
        pts = centroid + (pts - centroid) * 1.12
        track = Track(_catmull_rom_centripetal(pts, samples_per_seg=24), width=width)
        tries += 1
    return track


def touge(seed: int | None = None, width: float = 9.0) -> Track:
    """Mountain-pass style loop: tighter, switchback-heavy hairpins.

    (Elevation profile is a later-slice concern; this lays down the 2D plan.)"""
    rng = np.random.default_rng(seed)
    base = np.array([
        [-40, -120], [40, -150], [90, -90], [50, -30], [100, 20],
        [60, 90], [-20, 130], [-90, 90], [-60, 20], [-110, -30], [-70, -80],
    ], dtype=float)
    base += rng.normal(0, 8, base.shape)
    return Track(_catmull_rom_centripetal(base, samples_per_seg=20), width=width)


# --------------------------------------------------------------------------- #
# unified, difficulty-parameterised generator
# --------------------------------------------------------------------------- #
# Each archetype maps difficulty 0..1 onto (easy, hard) ranges. `spacing` is
# metres of track per control point, so corner DENSITY scales with length (a long
# loop from few points is always gentle — density is what makes it technical).
STYLES = {
    "gp":        {"spacing": (175.0, 90.0), "jitter": (0.12, 0.34), "width": (15.0, 12.0),
                  "elong": 1.15, "minR": (45.0, 20.0)},   # flowing race circuit
    "technical": {"spacing": (110.0, 55.0), "jitter": (0.25, 0.52), "width": (12.0, 10.0),
                  "elong": 1.0,  "minR": (28.0, 12.0)},   # tight, multi-apex
    "speedway":  {"spacing": (280.0, 170.0), "jitter": (0.05, 0.18), "width": (18.0, 15.0),
                  "elong": 1.75, "minR": (75.0, 40.0)},   # fast, long straights
    "touge":     {"spacing": (95.0, 52.0),  "jitter": (0.34, 0.58), "width": (10.0, 8.5),
                  "elong": 1.0,  "minR": (20.0, 10.0)},   # narrow switchbacks
}


def _smooth(pts, k=1):
    for _ in range(k):
        prv = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        pts = 0.5 * pts + 0.25 * (prv + nxt)
    return pts


def _scale_to_length(pts, width, target_len, samples):
    """Uniformly scale control points about their centroid to hit a target
    perimeter (length and radius both scale linearly with the factor)."""
    t = Track(_catmull_rom_centripetal(pts, samples), width)
    s = target_len / t.length
    c = pts.mean(axis=0)
    return (pts - c) * s + c


def _ring_points(n, jitter, elong, rng):
    base = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ang = base + rng.uniform(-1, 1, n) * (np.pi / n) * 0.5    # ordered, irregular
    r = 1.0 + rng.uniform(-1, 1, n) * jitter
    x = np.cos(ang) * r * elong
    y = np.sin(ang) * r
    return np.stack([x, y], axis=1) * 100.0                   # rescaled to length later


# --------------------------------------------------------------------------- #
# elevation profiles (PHYSICS_3D_PLAN Stage 1)
# --------------------------------------------------------------------------- #
# Global hill override. Flat 2D terrain is the default again; hills were added
# for the 3D viewer experiment and are now explicit opt-in via run.py --hills or
# configure_hills(enabled=True). This keeps watch/train/diagnostics aligned.
_HILLS_ON = False
_HILL_SCALE = 1.0
_FORCE_FLAT = False


def configure_hills(enabled: bool = True, scale: float = 1.0,
                    force_flat: bool = False):
    """Process-wide hill override: `--flat` -> enabled=False kills elevation
    everywhere; `--hill-scale X` scales every profile (clamped 0..1.5)."""
    global _HILLS_ON, _HILL_SCALE, _FORCE_FLAT
    _HILLS_ON = bool(enabled)
    _HILL_SCALE = float(np.clip(scale, 0.0, 1.5))
    _FORCE_FLAT = bool(force_flat)


# Per-style hill character. `amp` = total height range in metres (easy -> hard).
# `grade_cap` = max |dz/ds|. `launch_floor` = the slowest speed (m/s) at which
# ANY crest on the track will launch the car (easy -> hard) — crests sharper
# than that get smoothed away, so jumps are possible but honest: hard touge
# has crests a committed driver WILL clear; a speedway never throws you.
HILLS = {
    "touge":     {"amp": (8.0, 26.0), "grade_cap": 0.14,
                  "launch_floor": (38.0, 26.0)},
    "technical": {"amp": (5.0, 14.0), "grade_cap": 0.10,
                  "launch_floor": (42.0, 32.0)},
    "gp":        {"amp": (4.0, 10.0), "grade_cap": 0.07,
                  "launch_floor": (55.0, 45.0)},
    # 120 m/s = 432 km/h: effectively "no jumpable crests"
    "speedway":  {"amp": (0.0, 3.0),  "grade_cap": 0.03,
                  "launch_floor": (120.0, 120.0)},
}


def _smooth1d_periodic(v: np.ndarray, passes: int = 1) -> np.ndarray:
    """k passes of the loop-periodic Laplacian (the 1D twin of _smooth)."""
    for _ in range(passes):
        v = 0.5 * v + 0.25 * (np.roll(v, 1) + np.roll(v, -1))
    return v


def _min_launch_speed_from_vcurv(vcurv: np.ndarray) -> float:
    crest = -np.asarray(vcurv, dtype=float)
    crest = crest[crest > 1e-9]
    if len(crest) == 0:
        return float("inf")
    return float(np.sqrt(G / float(np.max(crest))))


def _smooth_real_road_elevation(elevation: np.ndarray, seg_len: np.ndarray,
                                grade_cap: float = 0.18,
                                launch_floor: float = 36.0,
                                max_passes: int = 96):
    """Turn terrain samples into a plausible road surface.

    The Nordschleife asset samples DEM terrain along an OSM centreline. That is
    the right open-data source, but raw terrain pixels can include bridge edges,
    embankments, and interpolation spikes. The physics engine treats grade and
    vertical curvature literally, so a one-sample DEM glitch becomes a launch
    ramp. Smooth only enough to keep real elevation while rejecting impossible
    road grades and low-speed crests.
    """
    z = np.asarray(elevation, dtype=float).copy()
    raw_grade, raw_vcurv = _grade_vcurv(z, seg_len)
    raw_max_grade = float(np.max(np.abs(raw_grade))) if len(raw_grade) else 0.0
    raw_min_launch = _min_launch_speed_from_vcurv(raw_vcurv)

    passes = 0
    for passes in range(max_passes + 1):
        grade, vcurv = _grade_vcurv(z, seg_len)
        max_grade = float(np.max(np.abs(grade))) if len(grade) else 0.0
        min_launch = _min_launch_speed_from_vcurv(vcurv)
        if max_grade <= grade_cap and min_launch >= launch_floor:
            break
        if passes >= max_passes:
            break
        z = _smooth1d_periodic(z, 1)

    meta = {
        "road_elevation_filter": "periodic-laplacian-grade-launch-cap",
        "road_elevation_raw_max_grade": raw_max_grade,
        "road_elevation_raw_min_launch_speed_mps": raw_min_launch,
        "road_elevation_smooth_passes": int(passes),
        "road_elevation_grade_cap": float(grade_cap),
        "road_elevation_launch_floor_mps": float(launch_floor),
        "road_elevation_max_grade": max_grade,
        "road_elevation_min_launch_speed_mps": min_launch,
    }
    return z, meta


def _elevation_profile(trk: Track, style: str, difficulty: float,
                       rng: np.random.Generator,
                       hill_scale: float = 1.0) -> np.ndarray:
    """Loop-periodic height profile for an existing 2D track.

    z(s) = sum of low-order Fourier modes (periodic by construction), plus —
    on touge/technical — a curvature-correlated term: smoothed |curvature|,
    mean-removed and integrated along arc, so elevation accumulates through
    the twisty sections and hairpins sit near local crests/saddles like a
    real switchback stack. Then two constraint passes: scale to the style's
    grade cap, and smooth until every crest's launch speed respects the
    style's floor (see HILLS).
    """
    p = HILLS.get(style, HILLS["gp"])
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    hill_scale = float(np.clip(hill_scale, 0.0, 1.5))
    amp = (p["amp"][0] + (p["amp"][1] - p["amp"][0]) * difficulty) * hill_scale
    M = len(trk.center)
    if amp <= 1e-6:
        return np.zeros(M)

    s, L = trk.arc, trk.length
    base = np.zeros(M)
    for k in (1, 2, 3, 5, 8):
        a = rng.uniform(0.55, 1.0) / k
        phi = rng.uniform(0.0, 2.0 * np.pi)
        base += a * np.sin(2.0 * np.pi * k * s / L + phi)
    base /= max(np.ptp(base), 1e-9)

    if style in ("touge", "technical"):
        ksm = _smooth1d_periodic(np.abs(trk.curvature), 10)
        kbar = float(np.sum(ksm * trk.seg_len) / L)     # length-weighted mean
        cum = np.cumsum((ksm - kbar) * trk.seg_len)     # loop integral == 0
        cum -= cum.mean()
        if np.ptp(cum) > 1e-9:
            base = 0.75 * base + 0.25 * (cum / np.ptp(cum))
            base /= max(np.ptp(base), 1e-9)

    z = amp * base
    z -= z.mean()

    # constraint 1: style grade cap (uniform scale preserves the shape)
    grade, _ = _grade_vcurv(z, trk.seg_len)
    mg = float(np.max(np.abs(grade)))
    cap = p["grade_cap"]
    if mg > cap:
        z *= cap / mg

    # constraint 2: smooth until no crest launches below the style floor
    # (smoothing only ever reduces grades, so the cap stays honored)
    floor = (p["launch_floor"][0]
             + (p["launch_floor"][1] - p["launch_floor"][0]) * difficulty)
    for _ in range(120):
        _, vcurv = _grade_vcurv(z, trk.seg_len)
        worst = float(np.max(-vcurv))                   # sharpest crest
        if worst < 1e-9 or np.sqrt(G / worst) >= floor:
            break
        z = _smooth1d_periodic(z, 1)
    return z


def bank_corners(trk: Track, max_deg: float = 8.0,
                 k_ref: float = 0.02) -> np.ndarray:
    """Banking profile that leans the road INTO its corners.

    Data-path helper only — NOT wired into any generator yet (hills-v1 ships
    bank ≡ 0; PHYSICS_3D_PLAN Stage 7+ may enable it for gp/speedway).
    Sign follows the track convention (bank + = left edge higher): a LEFT
    turn (curvature +) banks with the right edge raised -> negative bank.
    Magnitude saturates at max_deg once |curvature| reaches k_ref.
    """
    k = _smooth1d_periodic(trk.curvature.copy(), 8)
    b = -np.clip(k / k_ref, -1.0, 1.0) * np.radians(max_deg)
    return _smooth1d_periodic(b, 4)


def make_track(difficulty: float = 0.5, style: str = "gp",
               length: float | None = None, seed: int | None = None,
               samples: int = 24, hills: bool = True,
               hill_scale: float = 1.0) -> Track:
    """Procedural track with a single difficulty knob (0 = easy .. 1 = hard).

    `difficulty` interpolates corner count, jitter, width, and the minimum-radius
    floor within the chosen archetype; `length` sets the perimeter (600-2000 m if
    omitted). This is the generator the PPO curriculum dials up over training.

    `hills=True` (the default since PHYSICS_3D_PLAN Stage 4) adds a
    style/difficulty-shaped elevation profile (see HILLS). The profile draws
    from the SAME rng stream, strictly after every layout draw — so a given
    seed produces the same 2D layout with hills on or off, and named tracks
    get deterministic hills. `--flat` / `--hill-scale` override globally via
    configure_hills().
    """
    difficulty = float(np.clip(difficulty, 0.0, 1.0))
    if style not in STYLES:
        raise ValueError(f"Unknown style '{style}'. Choose from {list(STYLES)}.")
    p = STYLES[style]
    rng = np.random.default_rng(seed)

    def lerp(pair):
        return pair[0] + (pair[1] - pair[0]) * difficulty

    jitter = lerp(p["jitter"])
    width = lerp(p["width"])
    minR_floor = lerp(p["minR"])
    if length is None:
        length = float(rng.uniform(600.0, 2000.0))
    # control-point count from length / spacing -> corner density scales with size
    n = int(np.clip(round(length / lerp(p["spacing"])), 5, 40))

    pts = _smooth(_ring_points(n, jitter, p["elong"], rng), 1)
    pts = _scale_to_length(pts, width, length, samples)
    trk = Track(_catmull_rom_centripetal(pts, samples), width)
    # enforce the radius floor (smooth + rescale keeps the target length)
    tries = 0
    while trk.min_radius < minR_floor and tries < 10:
        pts = _scale_to_length(_smooth(pts, 1), width, length, samples)
        trk = Track(_catmull_rom_centripetal(pts, samples), width)
        tries += 1
    trk.style = style
    trk.difficulty = difficulty
    if hills and _HILLS_ON and not _FORCE_FLAT and hill_scale * _HILL_SCALE > 0.0:
        trk.set_elevation(_elevation_profile(trk, style, difficulty, rng,
                                             hill_scale * _HILL_SCALE))
    return trk


# --------------------------------------------------------------------------- #
# curated named set (fixed seeds -> reproducible) for eval / showcase / driving
# --------------------------------------------------------------------------- #
NAMED = {
    "club":     ("gp", 0.25, 900),       # easy flowing intro circuit
    "national": ("gp", 0.55, 1500),      # mid flowing GP track
    "coast":    ("gp", 0.45, 1800),      # long sweeping circuit
    "sprint":   ("technical", 0.65, 750),   # short + busy
    "tech":     ("technical", 0.90, 1150),  # hard, multi-apex
    "oval2":    ("speedway", 0.30, 1700),   # fast speedway
    "akina":    ("touge", 0.80, 1250),      # mountain pass
    "pass":     ("touge", 1.00, 900),       # tightest switchbacks
    "endless":  ("gp", 0.50, 100000),       # 100km effectively infinite loop
}


def named_track(name: str) -> Track:
    if name in SPECIAL:
        return SPECIAL[name]()
    if name not in NAMED:
        raise ValueError(f"Unknown track '{name}'. Choose from {list(NAMED) + list(SPECIAL)}.")
    style, diff, length = NAMED[name]
    return make_track(diff, style, length, seed=_stable_seed(name))


def named_list():
    return list(NAMED) + list(SPECIAL)


def stadium(straight: float = 600.0, radius: float = 95.0, width: float = 16.0,
            spacing: float = 3.5, hills: bool = True, hill_scale: float = 1.0,
            hill_seed: int | None = None) -> Track:
    """A 'stadium' speedway: two genuinely long straights joined by constant-
    radius semicircle ends — built directly (not from jittered points) so the
    straights are dead straight and you can wind it out to top speed."""
    L = straight / 2.0
    ns = max(2, int(straight / spacing))
    na = max(4, int(np.pi * radius / spacing))
    pts = []
    for x in np.linspace(-L, L, ns, endpoint=False):
        pts.append((x, -radius))                                  # bottom straight
    for a in np.linspace(-np.pi / 2, np.pi / 2, na, endpoint=False):
        pts.append((L + radius * np.cos(a), radius * np.sin(a)))  # right end
    for x in np.linspace(L, -L, ns, endpoint=False):
        pts.append((x, radius))                                   # top straight
    for a in np.linspace(np.pi / 2, 1.5 * np.pi, na, endpoint=False):
        pts.append((-L + radius * np.cos(a), radius * np.sin(a)))  # left end
    t = Track(np.array(pts, dtype=float), width=width)
    t.style = "speedway"
    t.difficulty = 0.2
    if hills and _HILLS_ON and not _FORCE_FLAT and hill_scale * _HILL_SCALE > 0.0:
        rng = np.random.default_rng(hill_seed)
        t.set_elevation(_elevation_profile(t, "speedway", t.difficulty, rng,
                                           hill_scale * _HILL_SCALE))
    return t


# --------------------------------------------------------------------------- #
# hand-crafted DISTINCTIVE tracks — explicit waypoints (not the ring generator,
# which can only make convex blobs). Each is a fixed, deterministic layout with a
# real character: long straights, hairpins, snaking sides, non-convex shapes.
# --------------------------------------------------------------------------- #
def _handcrafted(points, width, smooth=1, samples=22, style="gp",
                 difficulty=0.5, hills=True, hill_scale=1.0,
                 hill_seed=None) -> Track:
    pts = np.array(points, dtype=float)
    if smooth:
        pts = _smooth(pts, smooth)
    trk = Track(_catmull_rom_centripetal(pts, samples), width=width)
    trk.style = style
    trk.difficulty = difficulty
    if hills and _HILLS_ON and not _FORCE_FLAT and hill_scale * _HILL_SCALE > 0.0:
        rng = np.random.default_rng(hill_seed)
        trk.set_elevation(_elevation_profile(trk, style, difficulty, rng,
                                             hill_scale * _HILL_SCALE))
    return trk


def _asset_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "data" / "tracks" / f"{name}.json"


def _real_track_from_asset(name: str) -> Track:
    path = _asset_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"real track asset missing: {path}. Rebuild with "
            f"`PYTHONPATH=$PWD python3 tools/build_nordschleife.py`."
        )
    data = json.loads(path.read_text())
    center = np.asarray(data["center"], dtype=float)
    elev = np.asarray(data["elevation"], dtype=float)
    meta = dict(data.get("metadata", {}))
    if _FORCE_FLAT or _HILL_SCALE <= 0.0:
        elev = None
    elif abs(_HILL_SCALE - 1.0) > 1e-9:
        # Preserve absolute altitude reference but scale gradients around mean.
        elev = elev.mean() + (elev - elev.mean()) * _HILL_SCALE
    if elev is not None:
        seg = np.linalg.norm(np.diff(center, axis=0, append=center[:1]), axis=1)
        elev, road_meta = _smooth_real_road_elevation(elev, seg)
        meta.update(road_meta)
    meta.update(
        asset=str(path),
        official_length_m=float(data.get("official_length_m", 0.0)),
        generated_spacing_m=float(data.get("generated_spacing_m", 0.0)),
        width_source=data.get("width_source"),
        display_name=data.get("display_name", name),
    )
    trk = Track(
        center,
        width=float(data.get("width_m", 11.0)),
        elevation=elev,
        metadata=meta,
        landmarks=data.get("landmarks", []),
        sectors=data.get("sectors", []),
    )
    trk.style = "real"
    trk.difficulty = 1.0
    return trk


def nordschleife() -> Track:
    """Full 20.832 km Nuerburgring Nordschleife.

    Unlike procedural hills, this track carries source DEM elevation by default
    because elevation is part of the real circuit. `--flat` / force_flat still
    clears it for compatibility.
    """
    return _real_track_from_asset("nordschleife")


def circuit() -> Track:
    """Road course: genuine long straights into a wide hairpin complex."""
    return _handcrafted(
        [(-210, -95), (-80, -103), (50, -104), (165, -94), (225, -45), (228, 30),
         (165, 62), (55, 54), (-60, 66), (-160, 60), (-232, 82), (-272, 8),
         (-232, -62)], width=13.0, smooth=0, style="gp", difficulty=0.5,
        hill_seed=_stable_seed("circuit"))


def crescent() -> Track:
    """A sweeping C / banana — a long curved run with a concave inner side."""
    out = [(205 * np.cos(t), 205 * np.sin(t))
           for t in np.linspace(np.radians(-128), np.radians(128), 9)]
    inn = [(108 * np.cos(t), 108 * np.sin(t))
           for t in np.linspace(np.radians(128), np.radians(-128), 6)]
    return _handcrafted(out + inn, width=12.0, smooth=1, style="gp",
                        difficulty=0.55, hill_seed=_stable_seed("crescent"))


def esses() -> Track:
    """A pinched 'gourd' loop with snaking long sides — flowing S-curves."""
    n, hh, ww, amp, freq = 8, 195.0, 108.0, 27.0, 2.0
    rt = [(ww + amp * np.sin(freq * np.pi * i / (n - 1)), -hh + 2 * hh * i / (n - 1))
          for i in range(n)]
    lt = [(-ww + amp * np.sin(freq * np.pi * i / (n - 1)), hh - 2 * hh * i / (n - 1))
          for i in range(n)]
    return _handcrafted(rt + lt, width=11.5, smooth=1, style="technical",
                        difficulty=0.6, hill_seed=_stable_seed("esses"))


def hook() -> Track:
    """A boot / hook shape with an inward notch — non-convex, technical."""
    return _handcrafted(
        [(-175, -120), (60, -132), (155, -100), (152, -18), (38, -30), (18, 42),
         (92, 72), (60, 155), (-42, 162), (-122, 122), (-162, 40)],
        width=12.0, smooth=1, style="technical", difficulty=0.6,
        hill_seed=_stable_seed("hook"))


def ridge() -> Track:
    """The canonical jump track (PHYSICS_3D_PLAN): hard touge with TWO
    authored jumpable crests — one launching off the flattest straight, one
    cresting into a bend. Crest sites are picked from the layout's curvature,
    the rolling base profile is locally calmed around them, then each gets a
    Gaussian crest sized for its target launch speed via
    v_launch = sqrt(G·w²/2h). Elevation is part of this track's DESIGN, so it
    carries hills even while the global default is off (`--flat` can still
    kill it once Stage 4 lands)."""
    seed = _stable_seed("ridge")
    trk = make_track(0.85, "touge", 1100, seed=seed, hills=False)  # layout only
    if _FORCE_FLAT or not _HILLS_ON or _HILL_SCALE <= 0.0:     # --flat flattens ridge too
        return trk
    rng = np.random.default_rng(seed + 1)
    # gentle rolling base: generated at LOW difficulty (launch floor ~38 m/s)
    # so the two authored crests are unambiguously THE jumps on this track
    z = _elevation_profile(trk, "touge", 0.4, rng, hill_scale=0.7)

    s, L, M = trk.arc, trk.length, len(trk.center)
    ds = L / M
    ksm = _smooth1d_periodic(np.abs(trk.curvature), 12)

    def win_mean(v, half_m):
        """Periodic boxcar mean over a +/- half_m arc window."""
        n = max(1, int(round(half_m / ds)))
        k = np.ones(2 * n + 1) / (2 * n + 1)
        return np.convolve(np.tile(v, 3), k, mode="same")[len(v):2 * len(v)]

    # site A — straight launch: center of the flattest-curvature stretch
    site_a = int(np.argmin(win_mean(ksm, 35.0)))
    # site B — crest into a bend: calm here, curvature rising 25-70 m ahead
    ahead = np.roll(win_mean(ksm, 22.5), -int(round(47.5 / ds)))
    here = win_mean(ksm, 15.0)
    score = ahead - 1.5 * here
    d_a = np.abs(s - s[site_a])
    d_a = np.minimum(d_a, L - d_a)                  # periodic arc distance
    score[d_a < 140.0] = -np.inf                    # keep the jumps apart
    site_b = int(np.argmax(score))

    # calm the base in both neighborhoods first, then add both crests
    z_calm = _smooth1d_periodic(z.copy(), 40)
    sites = [(site_a, 20.0, 28.0), (site_b, 20.0, 31.0)]   # (idx, w, v_launch)
    for idx, w, _v in sites:
        d = s - s[idx]
        d = (d + L / 2.0) % L - L / 2.0             # wrapped offset
        mask = np.exp(-(d / (2.5 * w)) ** 2)
        z = z * (1.0 - mask) + z_calm * mask
    for idx, w, v in sites:
        d = s - s[idx]
        d = (d + L / 2.0) % L - L / 2.0
        h = G * w * w / (2.0 * v * v)               # bump height for v_launch
        z += h * np.exp(-(d / w) ** 2)
    z -= z.mean()
    # base + bump grades can stack past the style cap at the bump skirts; a
    # uniform rescale restores it (launch speeds rise ~1/sqrt(scale), which
    # the designed band absorbs — Gate 1 verifies the final numbers)
    grade, _ = _grade_vcurv(z, trk.seg_len)
    mg = float(np.max(np.abs(grade)))
    cap = HILLS["touge"]["grade_cap"]
    if mg > cap:
        z *= cap / mg
    trk.set_elevation(z * _HILL_SCALE)          # --hill-scale shifts launch speeds
    return trk


# fixed tracks built by a dedicated generator (not make_track)
SPECIAL = {
    "nordschleife": nordschleife,
    "nurburgring": nordschleife,
    "nuerburgring": nordschleife,
    "speedbowl":  lambda: stadium(straight=620.0, radius=95.0, width=16.0,
                                  hill_seed=_stable_seed("speedbowl")),
    "superspeed": lambda: stadium(straight=820.0, radius=120.0, width=18.0,
                                  hill_seed=_stable_seed("superspeed")),
    "circuit":    circuit,
    "crescent":   crescent,
    "esses":      esses,
    "hook":       hook,
    "ridge":      ridge,
}


def curriculum_track(difficulty: float, seed: int | None = None) -> Track:
    """A training track at the requested difficulty, with archetype variety that
    shifts toward harder styles as difficulty rises.

    Hills ramp WITH the curriculum (PHYSICS_3D_PLAN Stage 7): rookies learn on
    gentle slopes (scale 0.4), full-character hills arrive with the hard
    corners (1.0). The stadium pool stays near-flat by style anyway."""
    rng = np.random.default_rng(seed)
    hs = 0.4 + 0.6 * float(np.clip(difficulty, 0.0, 1.0))
    if difficulty < 0.4:
        pool = ["gp", "speedway", "stadium", "gp"]
    elif difficulty < 0.75:
        pool = ["gp", "technical", "speedway", "stadium"]
    else:
        pool = ["technical", "touge", "gp", "stadium"]
    style = str(rng.choice(pool))
    if style == "stadium":
        # real long-straight tracks so the policy learns top-speed -> braking
        # zones (otherwise the named 'speedbowl' is out-of-distribution).
        straight = float(rng.uniform(220.0, 580.0))
        radius = float(rng.uniform(45.0, 105.0))
        width = float(rng.uniform(11.0, 16.0))
        return stadium(straight, radius, width, hill_scale=hs,
                       hill_seed=int(rng.integers(1_000_000_000)))
    length = float(rng.uniform(600.0, 2000.0))
    return make_track(difficulty, style, length,
                      seed=int(rng.integers(1_000_000_000)), hill_scale=hs)


def drift_curriculum_track(difficulty: float, seed: int | None = None) -> Track:
    """Drift-tuned curriculum: WIDE, flowing tracks early (where a big slip angle
    can actually be held AND the lap completed), narrowing toward tight touge as
    difficulty rises. Tracks are a notch wider/gentler than the race curriculum at
    the same difficulty, because sustaining a drift needs road width — that's what
    lets the policy learn 'drift-and-complete' before facing the brutal hairpins.

    The hill ramp is GENTLER than the race curriculum (0.25 -> 0.75 of full
    scale): slopes punish slides — the slide must be learnable before the
    mountain shows up (PHYSICS_3D_PLAN Stage 7)."""
    rng = np.random.default_rng(seed)
    hs = 0.25 + 0.5 * float(np.clip(difficulty, 0.0, 1.0))
    if difficulty < 0.35:
        pool = ["gp", "speedway", "gp", "stadium"]          # wide + flowing
    elif difficulty < 0.7:
        pool = ["gp", "gp", "technical", "stadium"]         # mixed
    else:
        pool = ["technical", "touge", "gp"]                 # tight
    style = str(rng.choice(pool))
    if style == "stadium":
        straight = float(rng.uniform(220.0, 560.0))
        radius = float(rng.uniform(55.0, 110.0))
        width = float(rng.uniform(13.0, 17.0))
        return stadium(straight, radius, width, hill_scale=hs,
                       hill_seed=int(rng.integers(1_000_000_000)))
    length = float(rng.uniform(600.0, 1800.0))
    # bias the effective difficulty down a touch -> wider lane + gentler radius,
    # so a held slide stays on the road while the policy is still learning.
    d = float(np.clip(difficulty * 0.85, 0.0, 1.0))
    return make_track(d, style, length,
                      seed=int(rng.integers(1_000_000_000)), hill_scale=hs)
