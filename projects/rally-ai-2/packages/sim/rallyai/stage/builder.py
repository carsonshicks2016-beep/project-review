"""Turn a road *profile* into a stage document.

A stage is described as curvature and elevation against arc length, then
integrated into a centerline:

    heading(s) = integral of curvature
    x(s), y(s) = integral of (cos, sin) heading
    z(s)       = given directly

Describing roads this way rather than as a sequence of points is what makes
corner *intent* expressible — a hairpin is a constant-curvature segment with a
known radius, not a stretch of polyline that happens to bend. The procedural
generator composes archetypes into a profile and calls the same builder the
hand-authored fixtures use, so both go through one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal

import numpy as np

# Arc length between emitted centerline knots. Sparse enough that a stage file
# stays readable, dense enough that the Catmull-Rom fit in Track reproduces the
# intended curvature.
EMIT_DS = 4.0

# Integration step. Much finer than the emitted knots so the shape is accurate
# before it is sampled down.
INTEGRATE_DS = 0.25

# Length constants for the exponential approach of grade and width toward a
# segment's target. Roads change gradient and widen over tens of metres, not
# instantly, and a discontinuity in either is something the agent can sense.
GRADE_EASE_M = 18.0
WIDTH_EASE_M = 12.0

Direction = Literal["left", "right"]


@dataclass
class Profile:
    """A road as arrays against arc length. All arrays share ``s``'s length."""

    s: np.ndarray
    curvature: np.ndarray          # 1/m, +ve turns left
    z: np.ndarray                  # metres
    width: np.ndarray              # full drivable width, metres
    camber: np.ndarray             # radians, +ve banks into a left-hander
    surface: list[str] = field(default_factory=list)   # per sample


def severity_for_radius(radius: float) -> int:
    """Rally pace-note severity from corner radius: 1 is a hairpin, 6 is flat.

    These are the numbers a co-driver would call, so they are deliberately
    coarse — the agent gets the same resolution of information a driver does.
    """
    for limit, sev in ((15.0, 1), (25.0, 2), (40.0, 3), (70.0, 4), (120.0, 5)):
        if radius < limit:
            return sev
    return 6


def integrate(profile: Profile, start_xy: tuple[float, float] = (0.0, 0.0),
              start_heading: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate curvature into a centerline. Returns (x, y, heading)."""
    ds = float(profile.s[1] - profile.s[0])
    heading = start_heading + np.concatenate([[0.0], np.cumsum(profile.curvature[:-1] * ds)])
    # Trapezoidal position integration — plain rectangles drift noticeably over
    # a kilometre of curvature.
    cx = np.cos(heading)
    cy = np.sin(heading)
    x = start_xy[0] + np.concatenate([[0.0], np.cumsum(0.5 * (cx[1:] + cx[:-1]) * ds)])
    y = start_xy[1] + np.concatenate([[0.0], np.cumsum(0.5 * (cy[1:] + cy[:-1]) * ds)])
    return x, y, heading


def build_stage(
    profile: Profile,
    *,
    stage_id: str,
    name: str,
    seed: int,
    tier: int,
    generator_version: int,
    authored: bool = False,
    obstacles: list[dict[str, Any]] | None = None,
    surface_mu: dict[str, float] | None = None,
    emit_ds: float = EMIT_DS,
) -> dict[str, Any]:
    """Integrate a profile and emit a stage document.

    The result still needs ``contracts.write_json`` to quantise, hash and
    validate it — this function does not stamp, so callers cannot accidentally
    produce a hashed-but-invalid document.
    """
    surface_mu = surface_mu or DEFAULT_MU
    x, y, heading = integrate(profile)
    length = float(profile.s[-1])

    knot_s = adaptive_knots(profile, emit_ds)

    centerline = [
        {
            "s": float(ks),
            "x": float(np.interp(ks, profile.s, x)),
            "y": float(np.interp(ks, profile.s, y)),
            "z": float(np.interp(ks, profile.s, profile.z)),
            "width": float(np.interp(ks, profile.s, profile.width)),
            "camber": float(np.interp(ks, profile.s, profile.camber)),
        }
        for ks in knot_s
    ]

    surfaces = _surface_segments(profile, surface_mu, length)
    pace_notes = derive_pace_notes(profile)

    return {
        "schema_version": 1,
        "id": stage_id,
        "name": name,
        "generator": {
            "version": generator_version,
            "seed": seed,
            "tier": tier,
            "authored": authored,
        },
        "length_m": length,
        "centerline": centerline,
        "surfaces": surfaces,
        "pace_notes": pace_notes,
        "obstacles": obstacles or [],
        "start": {"s": 0.0, "heading": float(heading[0])},
        "finish": {"s": length},
    }


DEFAULT_MU = {"gravel": 0.72, "tarmac": 0.98, "snow": 0.36, "mud": 0.55}

# Largest heading change permitted between two emitted centerline knots.
# Knots are what survive into the stage file; Track refits a Catmull-Rom spline
# through them, and that fit overshoots when consecutive knots subtend a wide
# angle. At a fixed 4 m spacing a 10 m hairpin puts 23 degrees between knots and
# the refit came back 14% tighter than the road that was described — the file
# was not the road. 6 degrees keeps the round trip within a percent.
MAX_KNOT_ANGLE = np.radians(6.0)

# Never emit knots closer than this; below it the file grows without the
# geometry improving.
MIN_KNOT_DS = 0.75


def adaptive_knots(profile: Profile, emit_ds: float = EMIT_DS) -> np.ndarray:
    """Arc lengths at which to emit centerline knots.

    Spacing is whichever is tighter: ``emit_ds``, or the distance over which the
    heading turns by ``MAX_KNOT_ANGLE``. Straights therefore stay sparse and
    readable while hairpins get the density they need to survive the round trip
    through the file.
    """
    s = profile.s
    ds = float(s[1] - s[0])
    k = np.abs(profile.curvature)

    knots = [0.0]
    arc = 0.0
    angle = 0.0
    for i in range(1, len(s)):
        arc += ds
        angle += k[i] * ds
        if arc >= MIN_KNOT_DS and (arc >= emit_ds or angle >= MAX_KNOT_ANGLE):
            knots.append(float(s[i]))
            arc = 0.0
            angle = 0.0
    if knots[-1] < s[-1] - 1e-9:
        knots.append(float(s[-1]))
    return np.array(knots)


def _surface_segments(profile: Profile, mu: dict[str, float],
                      length: float) -> list[dict[str, Any]]:
    """Collapse the per-sample surface array into contiguous segments."""
    names = profile.surface or ["gravel"] * len(profile.s)
    segments: list[dict[str, Any]] = []
    start = 0
    for i in range(1, len(names) + 1):
        if i == len(names) or names[i] != names[start]:
            kind = names[start]
            segments.append({
                "s_start": float(profile.s[start]),
                "s_end": float(profile.s[i - 1]) if i < len(names) else length,
                "type": kind,
                "mu": float(mu[kind]),
            })
            start = i
    # Tile exactly: the schema requires full coverage with no gaps, and Track
    # raises if any sample is left uncovered.
    segments[0]["s_start"] = 0.0
    segments[-1]["s_end"] = length
    for a, b in pairwise(segments):
        b["s_start"] = a["s_end"]
    return segments


def derive_pace_notes(profile: Profile, min_gap_m: float = 25.0) -> list[dict[str, Any]]:
    """Derive co-driver calls from the geometry.

    Notes come from the road itself, exactly as a co-driver's would. This is the
    only look-ahead the agent gets beyond its own senses, so it must not encode
    anything a recce crew could not have written down.
    """
    s, k = profile.s, profile.curvature
    notes: list[dict[str, Any]] = []

    # Corner entries: where |curvature| rises through the "this is a corner"
    # threshold. 1/120 m is the flat-out boundary in severity_for_radius.
    thresh = 1.0 / 120.0
    is_corner = np.abs(k) > thresh
    edges = np.flatnonzero(np.diff(is_corner.astype(np.int8)) == 1) + 1

    for i in edges:
        j = i
        while j < len(k) and abs(k[j]) > thresh:
            j += 1
        span = k[i:j]
        if len(span) == 0:
            continue
        peak = float(np.max(np.abs(span)))
        radius = 1.0 / max(peak, 1e-6)
        direction = "left" if span[int(np.argmax(np.abs(span)))] > 0 else "right"
        sev = severity_for_radius(radius)

        modifiers: list[str] = []
        length_m = float(s[min(j, len(s) - 1)] - s[i])
        if length_m > 80.0:
            modifiers.append("long")
        elif length_m < 25.0:
            modifiers.append("short")
        # Tightens or opens, judged on the back half against the front half.
        mid = len(span) // 2
        if mid > 0:
            front, back = np.mean(np.abs(span[:mid])), np.mean(np.abs(span[mid:]))
            if back > front * 1.3:
                modifiers.append("tightens")
            elif front > back * 1.3:
                modifiers.append("opens")

        notes.append({
            "s": float(s[i]),
            "dir": direction,
            "severity": sev,
            "modifiers": modifiers,
            "text": " ".join([direction, str(sev)] + modifiers),
        })

    # Crests: local minima of vertical curvature sharp enough to unload the car.
    vcurv = np.gradient(np.gradient(profile.z, s), s)
    for i in _local_minima(vcurv, threshold=-0.004):
        notes.append({
            "s": float(s[i]),
            "dir": "crest",
            "severity": 6,
            "modifiers": [],
            "text": "crest",
        })

    notes.sort(key=lambda n: n["s"])
    # A co-driver does not call two things a metre apart.
    pruned: list[dict[str, Any]] = []
    for note in notes:
        if pruned and note["s"] - pruned[-1]["s"] < min_gap_m:
            continue
        pruned.append(note)
    return pruned


def _local_minima(v: np.ndarray, threshold: float) -> list[int]:
    out: list[int] = []
    for i in range(1, len(v) - 1):
        if v[i] < threshold and v[i] <= v[i - 1] and v[i] < v[i + 1]:
            out.append(i)
    return out


# --------------------------------------------------------------------------- #
# profile construction helpers — the archetype vocabulary in embryo
# --------------------------------------------------------------------------- #

class ProfileBuilder:
    """Accumulate road segments, then bake into a ``Profile``.

    Curvature transitions are cosine-eased over ``ease`` metres rather than
    stepped, because a step in curvature is a step in required steering angle,
    which no real road has and no tyre model enjoys.
    """

    def __init__(self, width: float = 8.0, surface: str = "gravel",
                 ds: float = INTEGRATE_DS):
        self.ds = ds
        self._default_width = width
        self._default_surface = surface
        self._k: list[float] = []
        self._z: list[float] = []
        self._w: list[float] = []
        self._c: list[float] = []
        self._surf: list[str] = []
        self._z_now = 0.0
        self._grade_now = 0.0
        self._width_now = width

    def _extend(self, n: int, curvature: float, grade: float, width: float,
                camber: float, surface: str) -> None:
        """Append ``n`` samples, easing grade and width toward their targets.

        Grade and width approach exponentially rather than stepping. A step in
        grade is a spike in vertical curvature, and vcurv is the agent's
        takeoff predictor — a discontinuity at every segment boundary would read
        as a phantom crest at every segment boundary. A step in width would
        likewise jump the corridor edge sideways under the raycast.
        """
        k_grade = 1.0 - np.exp(-self.ds / GRADE_EASE_M)
        k_width = 1.0 - np.exp(-self.ds / WIDTH_EASE_M)
        for _ in range(n):
            self._grade_now += (grade - self._grade_now) * k_grade
            self._width_now += (width - self._width_now) * k_width
            self._k.append(curvature)
            self._z_now += self._grade_now * self.ds
            self._z.append(self._z_now)
            self._w.append(self._width_now)
            self._c.append(camber)
            self._surf.append(surface)

    def straight(self, length: float, *, grade: float = 0.0,
                 width: float | None = None, surface: str | None = None) -> ProfileBuilder:
        self._extend(int(length / self.ds), 0.0, grade,
                     width or self._default_width, 0.0,
                     surface or self._default_surface)
        return self

    def corner(self, radius: float, angle_deg: float, direction: Direction, *,
               ease: float = 12.0, grade: float = 0.0, camber: float | None = None,
               width: float | None = None, surface: str | None = None) -> ProfileBuilder:
        """A constant-radius corner with eased entry and exit.

        ``camber`` defaults to banking into the turn, proportional to how tight
        it is. Pass a negative value for off-camber, which is where a stage
        starts to hurt.
        """
        sign = 1.0 if direction == "left" else -1.0
        k_peak = sign / radius
        arc = abs(np.radians(angle_deg)) * radius
        if camber is None:
            camber = sign * min(0.10, 1.2 / radius)

        # A transition cannot be longer than the corner it transitions into. A
        # 12 m hairpin through 90 degrees is only 18.8 m of arc, so the default
        # 12 m ease would consume it twice over and the corner would never hold
        # its radius — it measured 14 m instead of 12. Cap the ease so at least
        # 30% of the arc sits at the requested curvature.
        ease = min(ease, arc * 0.35)

        w = width or self._default_width
        surf = surface or self._default_surface

        n_ease = max(1, int(ease / self.ds))
        n_hold = max(1, int(max(0.0, arc - 2 * ease) / self.ds))

        for i in range(n_ease):
            f = 0.5 - 0.5 * np.cos(np.pi * i / n_ease)
            self._extend(1, k_peak * f, grade, w, camber * f, surf)
        self._extend(n_hold, k_peak, grade, w, camber, surf)
        for i in range(n_ease):
            f = 0.5 + 0.5 * np.cos(np.pi * i / n_ease)
            self._extend(1, k_peak * f, grade, w, camber * f, surf)
        return self

    def crest(self, length: float, height: float, *, curvature: float = 0.0,
              width: float | None = None, surface: str | None = None) -> ProfileBuilder:
        """A rise and fall sharp enough to unload — or launch — the car.

        Shaped ``sin^2`` rather than ``sin`` so the slope is zero at both ends
        and the bump joins the road it sits on smoothly. A plain ``sin`` bump
        starts and ends with non-zero gradient, which is a grade discontinuity
        placed exactly where the car is about to leave the ground.

        The crest rides on top of whatever gradient is already in effect, so a
        crest partway up a climb still climbs.
        """
        n = max(1, int(length / self.ds))
        w = width or self._default_width
        surf = surface or self._default_surface
        k_width = 1.0 - np.exp(-self.ds / WIDTH_EASE_M)

        for i in range(n):
            f = i / max(1, n - 1)
            bump = height * np.sin(np.pi * f) ** 2
            prev_bump = height * np.sin(np.pi * max(0.0, (i - 1) / max(1, n - 1))) ** 2
            self._width_now += (w - self._width_now) * k_width
            self._k.append(curvature)
            # Underlying gradient continues underneath the bump.
            self._z_now += self._grade_now * self.ds + (bump - prev_bump)
            self._z.append(self._z_now)
            self._w.append(self._width_now)
            self._c.append(0.0)
            self._surf.append(surf)
        return self

    def build(self) -> Profile:
        n = len(self._k)
        return Profile(
            s=np.arange(n) * self.ds,
            curvature=np.array(self._k),
            z=np.array(self._z),
            width=np.array(self._w),
            camber=np.array(self._c),
            surface=self._surf,
        )
