"""
Procedural race track generation + sensing.

Supports two track families:
  * **Circuit** (loop): closed-loop tracks via Catmull-Rom splines around
    randomised control points laid on a circle.
  * **Touge** (point-to-point): descending mountain-pass roads built from
    switchback hairpin control points with elevation that varies along
    the centreline.

Exposed for both families:
  * left/right edges (the walls),
  * arc-length progress measurement (for fitness),
  * raycast "vision" beams (the brain's eyes),
  * a curvature-based difficulty rating per point + look-ahead,
  * elevation / grade arrays (height in metres, slope rise/run).
"""

from __future__ import annotations
import math
import numpy as np

from .config import TrackSpec, SensorSpec


def _catmull_rom(p0, p1, p2, p3, n, alpha=0.5):
    """Sample n points on the CENTRIPETAL Catmull-Rom segment p1->p2.

    Centripetal parameterisation (alpha=0.5) is guaranteed not to overshoot
    into cusps or self-intersections, so we can use strong radius variation
    (real corners + flowing shapes) without the spline folding back on itself.
    """
    def knot(ti, pi, pj):
        return ti + (np.linalg.norm(pj - pi) + 1e-6) ** alpha
    t0 = 0.0
    t1 = knot(t0, p0, p1)
    t2 = knot(t1, p1, p2)
    t3 = knot(t2, p2, p3)
    t = np.linspace(t1, t2, n, endpoint=False).reshape(-1, 1)
    A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
    A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
    A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
    B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
    B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
    return (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2


# --------------------------------------------------------------------------
# Track "kinds": procedural difficulty tiers + named circuit layouts.
# --------------------------------------------------------------------------
# Procedural kinds -> (control points, base radius m, radial jitter m)
KIND_PARAMS = {
    # big, flowing layouts: fast sweepers to wind it out + real corners.
    "easy":      dict(n=8,  base=400, jitter=130),
    "loop":      dict(n=13, base=370, jitter=180),
    "technical": dict(n=18, base=340, jitter=215),
    "complex":   dict(n=24, base=315, jitter=245),
    # touge: special switchback generator (n/base/jitter unused)
    "touge":      dict(n=0, base=0, jitter=0),
    "touge_hard": dict(n=0, base=0, jitter=0),
}

# Named circuits as star-shaped radius profiles r(theta) -> distinct, repeatable
# layouts (guaranteed simple closed loops, no self-intersection).
_CIRCUIT_SCALE = 360.0
_CIRCUITS = {
    "speedway":   lambda a: 1.00 + 0.55 * math.cos(2 * a),                       # fast oval-ish
    "club":       lambda a: 0.78 + 0.30 * math.sin(3 * a) + 0.18 * math.cos(5 * a),  # tight/technical
    "grand_prix": lambda a: 1.00 + 0.30 * math.sin(a) + 0.22 * math.sin(2 * a + 0.8)
                            + 0.12 * math.cos(4 * a),                            # long mixed
}

# Curriculum tiers: each stage widens the pool the trainer samples from.
STAGES = [
    ["easy"],
    ["easy", "loop"],
    ["loop", "technical"],
    ["loop", "technical", "complex"],
    ["technical", "complex", "circuit:speedway", "circuit:club", "circuit:grand_prix"],
    ["touge", "touge_hard", "complex", "circuit:grand_prix"],
]


def stage_pool(stage):
    return STAGES[max(0, min(stage, len(STAGES) - 1))]


def sample_kind(rng, stage):
    pool = stage_pool(stage)
    return pool[int(rng.integers(len(pool)))]


def ga_stage_for_gen(generation):
    """Simple generation-based curriculum for the genetic algorithm."""
    for thresh, stage in ((8, 0), (24, 1), (50, 2), (90, 3), (140, 4)):
        if generation < thresh:
            return stage
    return 5


def _circuit_ctrl(name, n=30):
    f = _CIRCUITS[name]
    out = []
    for k in range(n):
        a = 2 * math.pi * k / n
        r = _CIRCUIT_SCALE * max(0.4, f(a))
        out.append((math.cos(a) * r, math.sin(a) * r))
    return np.array(out)


def _touge_ctrl(spec: TrackSpec, kind: str, rng):
    """Generate switchback-hairpin control points for a touge mountain pass.

    Returns (ctrl_xy, ctrl_elev, half_width, is_loop).
    ctrl_xy   – (M, 2) array of 2-D control point positions
    ctrl_elev – (M,)   array of elevation at each control point
    half_width – track half-width in metres
    is_loop   – always False for touge
    """
    hard = kind == "touge_hard"
    n_hairpins = (spec.touge_hairpins + 3) if hard else spec.touge_hairpins
    total_descent = spec.touge_descent
    hw = 5.5 if hard else spec.touge_half_width

    # Mountain road parameters
    run_length = rng.uniform(130, 200) if not hard else rng.uniform(100, 160)
    zigzag_width = rng.uniform(180, 320)  # lateral extent of the switchbacks
    hairpin_radius = rng.uniform(35, 55) if not hard else rng.uniform(28, 45)

    start_elev = total_descent  # start at the top
    pts_xy, pts_elev = [], []

    # Lay down the control points: start -> (hairpin -> connector) x N -> end
    for i in range(n_hairpins + 1):
        frac = i / max(n_hairpins, 1)
        elev = start_elev - frac * total_descent

        # Alternate left-right for the zigzag
        sign = 1.0 if (i % 2 == 0) else -1.0
        base_x = sign * zigzag_width * 0.5
        base_y = -frac * (run_length * n_hairpins)  # descending in -y

        if i == 0:
            # Start: straight entry section
            pts_xy.append((base_x, base_y + run_length * 0.5))
            pts_elev.append(elev + total_descent * 0.02)
            pts_xy.append((base_x, base_y))
            pts_elev.append(elev)
        elif i == n_hairpins:
            # End: straight exit section
            pts_xy.append((base_x, base_y))
            pts_elev.append(elev)
            pts_xy.append((base_x, base_y - run_length * 0.5))
            pts_elev.append(elev - total_descent * 0.02)
        else:
            # Hairpin turn: approach -> apex -> exit
            # Add subtle curvature jitter
            dx_jitter = rng.uniform(-25, 25)
            dy_jitter = rng.uniform(-15, 15)

            # Approach control point (sweeping in from the connecting section)
            mid_x = (base_x + (-sign) * zigzag_width * 0.5) * 0.3 + dx_jitter
            mid_y = base_y + run_length * 0.25 + dy_jitter
            mid_elev = elev + (total_descent / n_hairpins) * 0.3
            pts_xy.append((mid_x, mid_y))
            pts_elev.append(mid_elev)

            # Apex of hairpin
            apex_x = base_x + sign * hairpin_radius * rng.uniform(0.6, 1.2)
            apex_y = base_y + rng.uniform(-10, 10)
            pts_xy.append((apex_x, apex_y))
            pts_elev.append(elev)

            # Exit control point
            exit_x = (base_x + (-sign) * zigzag_width * 0.5) * 0.3 - dx_jitter
            exit_y = base_y - run_length * 0.25 + dy_jitter
            exit_elev = elev - (total_descent / n_hairpins) * 0.3
            pts_xy.append((exit_x, exit_y))
            pts_elev.append(exit_elev)

    ctrl_xy = np.array(pts_xy, dtype=np.float64)
    ctrl_elev = np.array(pts_elev, dtype=np.float64)
    return ctrl_xy, ctrl_elev, hw, False


def make_track(spec: TrackSpec, kind="loop", seed=None, half_width=None):
    """Build a Track of the given kind ('easy'/'loop'/'technical'/'complex',
    'circuit:<name>', 'touge', or 'touge_hard')."""
    if kind in ("touge", "touge_hard"):
        rng = np.random.default_rng(
            seed if seed is not None else np.random.randint(1, 1_000_000))
        ctrl_xy, ctrl_elev, hw, is_loop = _touge_ctrl(spec, kind, rng)
        return Track(spec, seed=seed, kind=kind, control_points=ctrl_xy,
                     half_width=hw, elevation_ctrl=ctrl_elev, is_loop=is_loop)
    if kind.startswith("circuit:"):
        ctrl = _circuit_ctrl(kind.split(":", 1)[1])
        return Track(spec, seed=seed, kind=kind, control_points=ctrl, half_width=half_width)
    return Track(spec, seed=seed, kind=kind, half_width=half_width)


class Track:
    def __init__(self, spec: TrackSpec, seed: int | None = None, kind="loop",
                 control_points=None, half_width=None,
                 elevation_ctrl=None, is_loop=None):
        self.spec = spec
        self.kind = kind
        self.half_width = float(half_width if half_width is not None else spec.half_width)
        self.seed = seed if seed is not None else np.random.randint(1, 1_000_000)
        rng = np.random.default_rng(self.seed)

        # Determine loop vs point-to-point
        if is_loop is not None:
            self.is_loop = is_loop
        else:
            self.is_loop = True  # default: all legacy kinds are loops

        # --- control points: explicit (circuits/touge) or generated (procedural) ---
        if control_points is not None:
            ctrl = np.asarray(control_points, dtype=np.float64)
        else:
            p = KIND_PARAMS.get(kind, KIND_PARAMS["loop"])
            n = p["n"]
            angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
            radii = p["base"] + rng.uniform(-p["jitter"], p["jitter"], size=n)
            ctrl = np.stack([np.cos(angles) * radii, np.sin(angles) * radii], axis=1)
        n = len(ctrl)

        # --- spline the centreline ---
        per_seg = max(2, spec.samples // n)
        pts = []
        if self.is_loop:
            for i in range(n):
                p0, p1 = ctrl[(i - 1) % n], ctrl[i]
                p2, p3 = ctrl[(i + 1) % n], ctrl[(i + 2) % n]
                pts.append(_catmull_rom(p0, p1, p2, p3, per_seg))
        else:
            # Open spline: phantom points at start/end by mirroring
            for i in range(n - 1):
                p0 = ctrl[i - 1] if i > 0 else 2 * ctrl[0] - ctrl[1]
                p1 = ctrl[i]
                p2 = ctrl[i + 1]
                p3 = ctrl[i + 2] if i + 2 < n else 2 * ctrl[-1] - ctrl[-2]
                pts.append(_catmull_rom(p0, p1, p2, p3, per_seg))
            # Include the very last point
            pts.append(ctrl[-1:].copy())
        self.center = np.concatenate(pts, axis=0)
        self.N = len(self.center)

        # Round off ONLY the too-tight corners to a drivable minimum radius.
        # This leaves fast/flowing sections untouched (unlike global smoothing),
        # so we keep strong variation yet guarantee every corner is drivable and
        # that the inner edge of the (wide) track never folds through itself.
        min_r = max(self.spec.min_corner_radius, self.half_width * 1.7)
        c = self.center.copy()
        for _ in range(60):
            if self.is_loop:
                nx = np.roll(c, -1, 0); pv = np.roll(c, 1, 0)
            else:
                nx = np.empty_like(c); pv = np.empty_like(c)
                nx[:-1] = c[1:]; nx[-1] = c[-1]
                pv[1:] = c[:-1]; pv[0] = c[0]
            seg = np.linalg.norm(nx - c, axis=1) + 1e-6
            head = np.arctan2((nx - pv)[:, 1], (nx - pv)[:, 0])
            if self.is_loop:
                dh = np.angle(np.exp(1j * (np.roll(head, -1) - head)))
            else:
                head_next = np.empty_like(head)
                head_next[:-1] = head[1:]; head_next[-1] = head[-1]
                dh = np.angle(np.exp(1j * (head_next - head)))
            sharp = np.abs(dh) / seg > 1.0 / min_r
            if not sharp.any():
                break
            m = sharp.copy()
            for _ in range(2):                          # widen the affected band
                if self.is_loop:
                    m = m | np.roll(m, 1) | np.roll(m, -1)
                else:
                    m_left = np.empty_like(m); m_left[1:] = m[:-1]; m_left[0] = False
                    m_right = np.empty_like(m); m_right[:-1] = m[1:]; m_right[-1] = False
                    m = m | m_left | m_right
            sm = 0.5 * c + 0.25 * (nx + pv)
            # Don't smooth endpoints for non-loop tracks
            if not self.is_loop:
                m[0] = False; m[-1] = False
            c[m] = sm[m]
        self.center = c

        # --- tangents, normals, edges ---
        if self.is_loop:
            nxt = np.roll(self.center, -1, axis=0)
            prv = np.roll(self.center, 1, axis=0)
        else:
            nxt = np.empty_like(self.center); prv = np.empty_like(self.center)
            nxt[:-1] = self.center[1:]; nxt[-1] = self.center[-1]
            prv[1:] = self.center[:-1]; prv[0] = self.center[0]
        tang = nxt - prv
        tang /= (np.linalg.norm(tang, axis=1, keepdims=True) + 1e-9)
        self.tangent = tang
        self.normal = np.stack([-tang[:, 1], tang[:, 0]], axis=1)  # left-hand normal
        hw = self.half_width
        self.left = self.center + self.normal * hw
        self.right = self.center - self.normal * hw

        # --- arc length ---
        if self.is_loop:
            seg = np.linalg.norm(np.roll(self.center, -1, axis=0) - self.center, axis=1)
        else:
            seg = np.empty(self.N)
            seg[:-1] = np.linalg.norm(self.center[1:] - self.center[:-1], axis=1)
            seg[-1] = seg[-2] if self.N > 1 else 1.0
        self.seg_len = seg
        self.arc = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
        self.length = float(np.sum(seg))

        # --- elevation & grade ---
        if elevation_ctrl is not None:
            # Interpolate elevation from control-point elevations to centreline
            # samples using arc-length parameterisation.
            ctrl_elev = np.asarray(elevation_ctrl, dtype=np.float64)
            # Compute arc-length of each control point by nearest-index lookup
            ctrl_arcs = np.zeros(len(ctrl_elev))
            for ci in range(len(ctrl)):
                d = ((self.center[:, 0] - ctrl[ci, 0]) ** 2 +
                     (self.center[:, 1] - ctrl[ci, 1]) ** 2)
                nearest = int(np.argmin(d))
                ctrl_arcs[ci] = self.arc[nearest]
            # Ensure monotonicity for interpolation
            for ci in range(1, len(ctrl_arcs)):
                if ctrl_arcs[ci] <= ctrl_arcs[ci - 1]:
                    ctrl_arcs[ci] = ctrl_arcs[ci - 1] + 1.0
            self.elevation = np.interp(self.arc, ctrl_arcs, ctrl_elev)
        else:
            self.elevation = np.zeros(self.N, dtype=np.float64)

        # grade = rise / run at each point
        self.grade = np.zeros(self.N, dtype=np.float64)
        if self.is_loop:
            elev_next = np.roll(self.elevation, -1)
        else:
            elev_next = np.empty(self.N)
            elev_next[:-1] = self.elevation[1:]
            elev_next[-1] = self.elevation[-1]
        self.grade = (elev_next - self.elevation) / (self.seg_len + 1e-6)
        # Clamp grade to physically plausible range
        self.grade = np.clip(self.grade, -0.25, 0.25)

        # --- curvature / difficulty (0..1) ---
        heading = np.arctan2(tang[:, 1], tang[:, 0])
        if self.is_loop:
            dh = np.angle(np.exp(1j * (np.roll(heading, -1) - heading)))
        else:
            head_next = np.empty_like(heading)
            head_next[:-1] = heading[1:]; head_next[-1] = heading[-1]
            dh = np.angle(np.exp(1j * (head_next - heading)))
        k = np.ones(7) / 7.0
        curv = np.abs(dh) / (seg + 1e-6)              # 1/radius (unsigned)
        if self.is_loop:
            curv = np.convolve(np.concatenate([curv[-3:], curv, curv[:3]]), k, "same")[3:-3]
        else:
            curv = np.convolve(np.pad(curv, 3, mode="edge"), k, "same")[3:-3]
        self.curvature = curv
        self.difficulty = np.clip(curv / (1.0 / spec.min_corner_radius), 0.0, 1.0)
        # signed curvature (+ = bends left, - = bends right) for the look-ahead
        # "track preview" the policy uses to plan braking points and apexes.
        signed = dh / (seg + 1e-6)
        if self.is_loop:
            signed = np.convolve(np.concatenate([signed[-3:], signed, signed[:3]]),
                                 k, "same")[3:-3]
        else:
            signed = np.convolve(np.pad(signed, 3, mode="edge"), k, "same")[3:-3]
        self.signed_curvature = signed

        # precomputed segment endpoints for raycasting
        if self.is_loop:
            self._lA, self._lB = self.left, np.roll(self.left, -1, axis=0)
            self._rA, self._rB = self.right, np.roll(self.right, -1, axis=0)
        else:
            # Open track: don't wrap the last segment back to the first
            self._lA, self._lB = self.left[:-1], self.left[1:]
            self._rA, self._rB = self.right[:-1], self.right[1:]

    # ------------------------------------------------------------------ #
    def grade_at(self, idx):
        """Road gradient at centreline index `idx`.
        Positive = uphill, negative = downhill."""
        return float(self.grade[idx % self.N])

    def gutter_zone(self, idx, lateral):
        """Return True if `lateral` is within 0.5 m of the inner edge on a
        corner with difficulty > 0.4.

        The *inner* edge is the mountain-side edge:
          - LEFT turn (positive signed curvature)  -> inner edge is RIGHT  (negative lateral)
          - RIGHT turn (negative signed curvature) -> inner edge is LEFT   (positive lateral)
        """
        if self.difficulty[idx % self.N] <= 0.4:
            return False
        sc = self.signed_curvature[idx % self.N]
        if abs(sc) < 1e-6:
            return False
        hw = self.half_width
        if sc > 0:  # left turn -> inner edge is right side
            # Right edge is at lateral = -hw
            return lateral < -(hw - 0.5)
        else:  # right turn -> inner edge is left side
            # Left edge is at lateral = +hw
            return lateral > (hw - 0.5)

    # ------------------------------------------------------------------ #
    def start_pose(self):
        """(x, y, yaw) at the start/finish line, pointing down the track."""
        h = math.atan2(self.tangent[0, 1], self.tangent[0, 0])
        return float(self.center[0, 0]), float(self.center[0, 1]), h

    def index_at_arc(self, s):
        """Centreline index nearest to arc-length `s` (metres).
        Wraps for loop tracks; clamps for non-loop."""
        if self.is_loop:
            s = s % self.length
        else:
            s = max(0.0, min(s, self.length))
        return int(np.clip(
            np.searchsorted(self.arc, s, side="right") - 1, 0, self.N - 1))

    def nearest_index(self, x, y, hint=None, window=60):
        if hint is None:
            d = (self.center[:, 0] - x) ** 2 + (self.center[:, 1] - y) ** 2
            return int(np.argmin(d))
        if self.is_loop:
            idx = (hint + np.arange(-window, window)) % self.N
        else:
            lo = max(0, hint - window)
            hi = min(self.N, hint + window)
            idx = np.arange(lo, hi)
        pts = self.center[idx]
        d = (pts[:, 0] - x) ** 2 + (pts[:, 1] - y) ** 2
        return int(idx[np.argmin(d)])

    def progress(self, x, y, hint=None):
        """Return (arc_distance, nearest_idx, signed_lateral_offset).
        For loop tracks, arc_distance wraps. For touge, it is absolute."""
        i = self.nearest_index(x, y, hint)
        to = np.array([x - self.center[i, 0], y - self.center[i, 1]])
        along = float(np.dot(to, self.tangent[i]))
        lateral = float(np.dot(to, self.normal[i]))          # +left, -right
        arc_s = self.arc[i] + along
        if self.is_loop:
            return arc_s, i, lateral
        else:
            # Non-loop: absolute distance, no wrapping
            return max(0.0, arc_s), i, lateral

    def off_track(self, lateral):
        return abs(lateral) > self.half_width

    def upcoming_difficulty(self, idx, lookahead=70):
        if self.is_loop:
            idxs = (idx + np.arange(0, lookahead)) % self.N
        else:
            idxs = np.clip(idx + np.arange(0, lookahead), 0, self.N - 1)
        return float(np.mean(self.difficulty[idxs]))

    def curvature_preview(self, idx, dists, scale):
        """Signed centre-line curvature (+left/-right) sampled at arc-distances
        `dists` (m) ahead of centreline index `idx`, scaled so a minimum-radius
        corner reads ~1.0 and clipped to [-1, 1].  This is the policy's
        look-ahead: it can see the shape of the road before it arrives."""
        s0 = self.arc[idx]
        out = np.empty(len(dists), dtype=np.float64)
        for j, d in enumerate(dists):
            out[j] = self.signed_curvature[self.index_at_arc(s0 + d)]
        return np.clip(out * scale, -1.0, 1.0)

    # ------------------------------------------------------------------ #
    def cast_rays(self, x, y, yaw, sensors: SensorSpec, hint=None):
        """
        Return normalised free-distance per beam in [0, 1] (1 = nothing within
        range).  Beams fan out across +/- ray_spread around the heading.
        """
        i = self.nearest_index(x, y, hint)
        w = 90
        if self.is_loop:
            idx = (i + np.arange(-w, w)) % self.N
        else:
            n_segs = len(self._lA)  # N-1 for non-loop
            lo = max(0, i - w)
            hi = min(n_segs, i + w)
            idx = np.arange(lo, hi)
        A = np.concatenate([self._lA[idx], self._rA[idx]], axis=0)
        B = np.concatenate([self._lB[idx], self._rB[idx]], axis=0)
        E = B - A
        O = np.array([x, y])
        AO = A - O

        n = sensors.n_rays
        offsets = np.linspace(-sensors.ray_spread, sensors.ray_spread, n)
        out = np.ones(n, dtype=np.float64)
        for j, off in enumerate(offsets):
            ang = yaw + off
            d = np.array([math.cos(ang), math.sin(ang)])
            denom = d[0] * E[:, 1] - d[1] * E[:, 0]
            ok = np.abs(denom) > 1e-9
            t = np.full(len(E), np.inf)
            u = np.full(len(E), np.inf)
            t[ok] = (AO[ok, 0] * E[ok, 1] - AO[ok, 1] * E[ok, 0]) / denom[ok]
            u[ok] = (AO[ok, 0] * d[1] - AO[ok, 1] * d[0]) / denom[ok]
            valid = ok & (t >= 0) & (u >= 0) & (u <= 1) & (t <= sensors.ray_max_range)
            if np.any(valid):
                out[j] = float(np.min(t[valid])) / sensors.ray_max_range
        return out, offsets, i
