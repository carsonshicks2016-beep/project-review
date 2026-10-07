"""The agent's senses.

Four groups, in a fixed order, all normalised to roughly [-1, 1]:

    vision       2 * n_beams   raycast distance + what the beam hit
    proprio      25            what the car feels through the chassis
    lookahead    4 + 3 * n     the road ahead, read at speed-scaled distances
    terrain      7 + 2 * n     slope, air, height, and the crest preview

Two rules govern this file.

**The agent may only use what a driver could have.** No global stage array, no
privileged distance-to-finish beyond what a co-driver's notes give. Everything
here is either felt through the car, seen down the road, or called by the
co-driver.

**Anything that can kill the agent must be visible here.** If a way to die is
added to the environment, the sense that sees it coming is added in the same
commit. v1 shipped stages with trees on the racing line against an eighteen-
dimensional observation that could not see them; the policy was being asked to
avoid something it had no organ for.

Blocks are **appended, never inserted**, so an index that meant something last
week still means it today. Changing this layout invalidates every checkpoint and
every normaliser — do it deliberately, in one commit, not by accretion.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rallyai.physics.bridge import RallyCar
from rallyai.track import HIT_OBSTACLE, Track, TrackQuery

# Normalisation references. These set what "1.0" means to the network, so they
# should be near the top of the car's real range rather than exact maxima.
V_REF = 60.0        # m/s  — the car tops out near 57
A_REF = 20.0        # m/s^2
R_REF = 3.0         # rad/s yaw rate
CURV_REF = 0.05     # 1/m  (a 20 m radius reads 1.0)
MU_REF = 1.0
GRADE_REF = 0.20
VCURV_REF = 0.02
VZ_REF = 12.0       # m/s vertical
HEIGHT_REF = 3.0    # m above the road
WIDTH_REF = 12.0    # m


@dataclass(frozen=True)
class SensorSpec:
    """Sensor geometry. Part of the observation contract: changing any of these
    changes the observation size or meaning."""

    n_beams: int = 9
    beam_spread_deg: float = 75.0
    beam_range: float = 60.0

    # Look-ahead is measured in SECONDS, not metres, then converted using the
    # car's own speed. A driver reads further down the road the faster they go;
    # a fixed 40 m preview is a long way at 20 km/h and no warning at all at
    # 180 km/h. The floor keeps the preview meaningful when nearly stopped.
    lookahead_times: tuple[float, ...] = (0.5, 1.0, 1.5, 2.5, 3.5, 5.0)
    lookahead_min_m: float = 8.0

    # Beyond this, a pace note is not yet the driver's problem.
    pace_horizon_m: float = 120.0

    @property
    def n_look(self) -> int:
        return len(self.lookahead_times)

    @property
    def size(self) -> int:
        return 2 * self.n_beams + 25 + (4 + 3 * self.n_look) + (7 + 2 * self.n_look)

    @property
    def beam_angles(self) -> np.ndarray:
        """Beam offsets from the car's heading, left to right."""
        return np.radians(np.linspace(self.beam_spread_deg, -self.beam_spread_deg,
                                      self.n_beams))


@dataclass
class Observation:
    """The observation vector, plus everything needed to draw what the agent saw.

    The debug fields exist so TRAIN can render the agent's own rays and
    look-ahead points over the road. Watching what it sees while it learns is
    the point of the mode, and reconstructing that from the obs vector after the
    fact is guesswork.
    """

    vector: np.ndarray
    beam_distances: np.ndarray
    beam_points: np.ndarray
    beam_kinds: np.ndarray
    lookahead_s: np.ndarray                 # arc lengths previewed this step
    lookahead_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pace_note: dict | None = None


class SensorSuite:
    """Assembles observations. Stateless: same car and track give same answer."""

    def __init__(self, spec: SensorSpec | None = None):
        self.spec = spec or SensorSpec()
        self._angles = self.spec.beam_angles

    @property
    def size(self) -> int:
        return self.spec.size

    def observe(self, car: RallyCar, track: Track, query: TrackQuery) -> Observation:
        s = self.spec
        v = car.vehicle

        # ---------------- vision ----------------
        dists, kinds, points = track.raycast(
            v.x, v.y, v.yaw, self._angles, s.beam_range, hint_s=query.s
        )
        vision = np.concatenate([
            dists / s.beam_range,                        # 1.0 = clear to the horizon
            np.where(kinds == HIT_OBSTACLE, 1.0, 0.0),   # solid, or just the verge
        ])

        # ---------------- proprioception ----------------
        spec_car = car.spec
        mg_corner = spec_car.mass * 9.81 / 4.0
        n_gears = len(spec_car.gear_ratios)
        proprio = np.array([
            v.speed / V_REF,
            v.vx / V_REF,
            v.vy / V_REF,
            v.r / R_REF,
            v.ax / A_REF,
            v.ay / A_REF,
            v.slip_angle / (np.pi / 2),
            v.steer_angle / max(v.max_steer_angle, 1e-6),
            v.rpm / spec_car.redline_rpm,
            (v.gear - 1) / max(1, n_gears - 1),
            v.boost,
            *np.clip(v.wheel_grip, 0.0, 2.0),
            *(v.Fz / mg_corner),
            *np.clip(v.wheel_sr, -1.5, 1.5),
            query.lateral / max(query.half_width, 1e-6),
            query.heading_error / np.pi,
        ], dtype=np.float64)

        # ---------------- look-ahead ----------------
        # Distances scale with speed, so the preview is always the same number of
        # seconds down the road.
        look_m = np.maximum(np.array(s.lookahead_times) * v.speed, s.lookahead_min_m)
        look_s = np.minimum(query.s + look_m, track.finish_s)
        offsets = look_s - query.s

        note = track.pace_ahead(query.s, horizon=s.pace_horizon_m)
        lookahead = np.concatenate([
            np.clip(track.lookahead_curvature(query.s, offsets) / CURV_REF, -1.5, 1.5),
            track.lookahead_mu(query.s, offsets) / MU_REF,
            track.lookahead_width(query.s, offsets) / WIDTH_REF,
            _encode_pace(note, query.s, s.pace_horizon_m),
        ])

        # ---------------- terrain and air ----------------
        terrain = np.concatenate([
            [
                np.clip(v.grade_body / GRADE_REF, -2.0, 2.0),
                np.clip(v.bank_body / GRADE_REF, -2.0, 2.0),
                np.clip(v.pitch / 0.5, -2.0, 2.0),
                np.clip(v.roll / 0.5, -2.0, 2.0),
                np.clip(v.vz / VZ_REF, -2.0, 2.0),
                # Height above the road. Without it the agent knows it is in the
                # air but not how far off the ground, which is most of what
                # timing a landing depends on.
                np.clip(max(0.0, v.z - v.road_z) / HEIGHT_REF, 0.0, 2.0),
                1.0 if v.airborne else 0.0,
            ],
            np.clip(track.lookahead_grade(query.s, offsets) / GRADE_REF, -2.0, 2.0),
            np.clip(track.lookahead_vcurv(query.s, offsets) / VCURV_REF, -2.0, 2.0),
        ])

        vector = np.concatenate([vision, proprio, lookahead, terrain]).astype(np.float32)
        # A NaN here poisons the policy silently and is miserable to trace back
        # from a diverged run. Fail where it happened instead.
        if not np.all(np.isfinite(vector)):
            bad = np.flatnonzero(~np.isfinite(vector))
            raise FloatingPointError(
                f"non-finite observation at indices {bad.tolist()} "
                f"(s={query.s:.1f}, speed={v.speed:.1f}, airborne={v.airborne})"
            )

        return Observation(
            vector=vector,
            beam_distances=dists,
            beam_points=points,
            beam_kinds=kinds,
            lookahead_s=look_s,
            lookahead_points=_lookahead_points(track, look_s),
            pace_note=note,
        )


def _encode_pace(note: dict | None, s_now: float, horizon: float) -> np.ndarray:
    """A co-driver's call as four numbers.

    Direction is signed rather than one-hot so "how hard, which way" stays a
    smooth quantity; hazard flags crest and jump calls, which are about the
    vertical rather than the horizontal.
    """
    if note is None:
        return np.zeros(4)
    direction = {"left": 1.0, "right": -1.0}.get(note["dir"], 0.0)
    hazard = 1.0 if note["dir"] in ("crest", "jump", "caution") else 0.0
    # severity 1 (hairpin) -> 1.0 urgency, severity 6 (flat) -> 0.0
    urgency = (6.0 - float(note["severity"])) / 5.0
    distance = np.clip((note["s"] - s_now) / horizon, 0.0, 1.0)
    return np.array([direction, urgency, distance, hazard])


def _lookahead_points(track: Track, look_s: np.ndarray) -> np.ndarray:
    """World positions of the previewed points, for the debug overlay."""
    idx = np.clip(np.rint(look_s / track.ds).astype(np.int64), 0, len(track.s) - 1)
    return np.stack([track.x[idx], track.y[idx], track.z[idx]], axis=1)
