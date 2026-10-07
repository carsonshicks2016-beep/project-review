"""Procedural stage generation.

A stage is composed from a **vocabulary of corner archetypes** with plausible
sequencing — a hairpin, an ess sequence, a long fifth-gear sweep, a crest into a
braking zone — not a random walk that happens to bend. The difference matters:
a random walk produces roads with no intent, where nothing is *set up* by what
precedes it, and a policy trained on them learns to react rather than to read.

Difficulty is **one scalar**. Length, width, friction, corner severity,
elevation aggression and obstacle proximity all move together with the tier, so
the curriculum has one dial rather than six that can disagree.

Determinism
-----------
Same version + same seed + same tier must give a byte-identical file on any
machine. Two things protect that:

* only ``rng.random()`` and ``rng.integers()`` are used. NumPy guarantees the
  PCG64 bit stream across versions but explicitly does *not* guarantee that
  distribution methods like ``normal()`` return identical floats, so those are
  avoided entirely and every distribution is built by hand from uniforms.
* coordinates are quantised on write (see ``contracts``), so a last-bit
  difference between CPUs cannot change a stage's identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from rallyai.stage.builder import ProfileBuilder, build_stage
from rallyai.track import Track

GENERATOR_VERSION = 1

# How strongly the gradient is pulled back toward level between features.
# 1.0 is a free random walk (which drifts and stays drifted); lower reverts
# harder. Set by measuring net elevation across seeds — see _walk_grade.
GRADE_REVERSION = 0.55

GRAVITY = 9.81


def crest_height_for_takeoff(length_m: float, takeoff_mps: float) -> float:
    """Height of a ``sin^2`` crest that launches the car at ``takeoff_mps``.

    The car leaves the ground when the centripetal acceleration needed to follow
    the road exceeds gravity: ``v^2 * |vcurv| > g``. For ``z = h sin^2(pi s/L)``
    the peak vertical curvature is ``2 h pi^2 / L^2``, so

        h = L^2 * g / (2 pi^2 * v^2)

    A tier that wants the car airborne at 55 km/h asks for a low takeoff speed
    and gets a taller crest; a tier that wants crests to be mostly scenery asks
    for a high one.
    """
    return float(length_m ** 2 * GRAVITY / (2.0 * np.pi ** 2 * max(takeoff_mps, 1.0) ** 2))


# --------------------------------------------------------------------------- #
# difficulty
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TierParams:
    """Everything the difficulty scalar controls."""

    name: str
    length_m: tuple[float, float]
    width_m: tuple[float, float]
    min_radius: float               # tightest corner permitted
    max_radius: float               # the fastest sweeper on offer
    max_grade: float                # steepest sustained climb or descent
    # Crests are specified by the speed at which they launch the car, not by
    # height. Height alone is meaningless — a 3 m rise over 100 m is scenery and
    # over 20 m is a ramp. Takeoff is what the tier should actually control, and
    # deriving height from it also bounds the gradient: specifying height
    # directly gave 61% slopes at the top tier.
    crest_takeoff_mps: tuple[float, float]
    crest_length_m: tuple[float, float]
    crest_chance: float             # per eligible feature
    # Weighted, not a flat set. A uniform pool let the top tier draw tarmac
    # (mu 0.98) as often as snow (mu 0.36), which made tier 5 EASIER than
    # tier 4 in measurement — difficulty stopped being one scalar. Higher
    # tiers lean on low grip.
    surfaces: tuple[tuple[str, float], ...]
    surface_change_chance: float
    off_camber_chance: float
    tree_clearance: tuple[float, float]   # metres beyond the corridor edge
    tree_spacing: float


TIERS: tuple[TierParams, ...] = (
    TierParams(
        name="Wide Gravel Intro",
        length_m=(380.0, 520.0), width_m=(10.0, 12.0),
        min_radius=35.0, max_radius=160.0, max_grade=0.03,
        crest_takeoff_mps=(26.0, 34.0), crest_length_m=(30.0, 46.0), crest_chance=0.10,
        surfaces=(("gravel", 1.0),), surface_change_chance=0.0,
        off_camber_chance=0.0, tree_clearance=(4.0, 7.0), tree_spacing=16.0,
    ),
    TierParams(
        name="Forest Medium",
        length_m=(550.0, 780.0), width_m=(8.5, 10.5),
        min_radius=25.0, max_radius=150.0, max_grade=0.05,
        crest_takeoff_mps=(23.0, 31.0), crest_length_m=(30.0, 48.0), crest_chance=0.18,
        surfaces=(("gravel", 1.0),), surface_change_chance=0.10,
        off_camber_chance=0.08, tree_clearance=(3.0, 5.5), tree_spacing=13.0,
    ),
    TierParams(
        name="Tight Gravel",
        length_m=(750.0, 1050.0), width_m=(7.0, 9.0),
        min_radius=16.0, max_radius=130.0, max_grade=0.07,
        crest_takeoff_mps=(20.0, 28.0), crest_length_m=(28.0, 48.0), crest_chance=0.25,
        surfaces=(("gravel", 0.78), ("tarmac", 0.22)), surface_change_chance=0.20,
        off_camber_chance=0.15, tree_clearance=(2.5, 4.5), tree_spacing=11.0,
    ),
    TierParams(
        name="Mixed Surface",
        length_m=(1000.0, 1400.0), width_m=(6.5, 8.5),
        min_radius=13.0, max_radius=120.0, max_grade=0.09,
        crest_takeoff_mps=(18.0, 25.0), crest_length_m=(26.0, 46.0), crest_chance=0.30,
        surfaces=(("gravel", 0.52), ("mud", 0.30), ("tarmac", 0.18)), surface_change_chance=0.28,
        off_camber_chance=0.22, tree_clearance=(2.0, 4.0), tree_spacing=10.0,
    ),
    TierParams(
        name="Snow Hazard",
        length_m=(1250.0, 1750.0), width_m=(6.0, 8.0),
        min_radius=11.0, max_radius=110.0, max_grade=0.11,
        crest_takeoff_mps=(16.0, 23.0), crest_length_m=(26.0, 44.0), crest_chance=0.34,
        surfaces=(("snow", 0.52), ("gravel", 0.34), ("mud", 0.14)), surface_change_chance=0.32,
        off_camber_chance=0.28, tree_clearance=(1.8, 3.5), tree_spacing=9.0,
    ),
    TierParams(
        name="Frontier",
        length_m=(1600.0, 2300.0), width_m=(5.5, 7.5),
        min_radius=9.5, max_radius=100.0, max_grade=0.13,
        crest_takeoff_mps=(14.0, 21.0), crest_length_m=(24.0, 42.0), crest_chance=0.38,
        surfaces=(("snow", 0.58), ("mud", 0.26), ("gravel", 0.16)), surface_change_chance=0.36,
        off_camber_chance=0.34, tree_clearance=(1.5, 3.0), tree_spacing=8.0,
    ),
)

N_TIERS = len(TIERS)


def tier_params(tier: int) -> TierParams:
    return TIERS[int(np.clip(tier, 0, N_TIERS - 1))]


# --------------------------------------------------------------------------- #
# sequencing
# --------------------------------------------------------------------------- #

# What plausibly follows what. A hairpin wants a straight in front of it to
# brake down and one behind it to accelerate out of; an ess flows into another
# direction change; a plunge sets up a corner at the bottom. These weights are
# what turn a bag of corners into a road that reads like it was surveyed.
TRANSITIONS: dict[str, dict[str, float]] = {
    "start":         {"straight": 1.00},
    "straight":      {"sweeper": 0.24, "hairpin": 0.16, "ess": 0.18,
                      "tightening": 0.14, "crest_braking": 0.14, "chicane": 0.14},
    "sweeper":       {"straight": 0.34, "ess": 0.18, "tightening": 0.16,
                      "hairpin": 0.10, "plunge": 0.12, "crest_braking": 0.10},
    "hairpin":       {"straight": 0.62, "sweeper": 0.24, "plunge": 0.14},
    "ess":           {"straight": 0.38, "sweeper": 0.24, "hairpin": 0.14,
                      "crest_braking": 0.14, "tightening": 0.10},
    "tightening":    {"straight": 0.44, "sweeper": 0.20, "ess": 0.20, "hairpin": 0.16},
    "crest_braking": {"straight": 0.36, "sweeper": 0.30, "ess": 0.22, "hairpin": 0.12},
    "plunge":        {"straight": 0.26, "hairpin": 0.30, "sweeper": 0.30, "ess": 0.14},
    "chicane":       {"straight": 0.46, "sweeper": 0.30, "ess": 0.24},
}


# --------------------------------------------------------------------------- #
# rng helpers — uniforms only, for cross-version stability
# --------------------------------------------------------------------------- #

def _uni(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(lo + (hi - lo) * rng.random())


def _chance(rng: np.random.Generator, p: float) -> bool:
    return bool(rng.random() < p)


def _pick(rng: np.random.Generator, weights: dict[str, float]) -> str:
    keys = sorted(weights)                       # sorted: dict order must not matter
    cum = np.cumsum([weights[k] for k in keys])
    r = float(rng.random()) * float(cum[-1])
    return keys[int(np.searchsorted(cum, r))]


# --------------------------------------------------------------------------- #
# the generator
# --------------------------------------------------------------------------- #

class StageGenerator:
    """Composes archetypes into a stage profile."""

    def __init__(self, seed: int, tier: int):
        self.seed = int(seed)
        self.tier = int(np.clip(tier, 0, N_TIERS - 1))
        self.p = tier_params(self.tier)
        self.rng = np.random.default_rng(self.seed)
        self.base_width = _uni(self.rng, *self.p.width_m)
        self.surface = self._pick_surface(None)
        self.grade = 0.0
        self.last_dir = "left" if _chance(self.rng, 0.5) else "right"

    # -- surface ---------------------------------------------------------- #

    def _pick_surface(self, avoid: str | None) -> str:
        weights = {n: w for n, w in self.p.surfaces if n != avoid}
        if not weights:
            weights = dict(self.p.surfaces)
        return _pick(self.rng, weights)

    def _maybe_change_surface(self) -> None:
        if _chance(self.rng, self.p.surface_change_chance):
            self.surface = self._pick_surface(self.surface)

    # -- elevation -------------------------------------------------------- #

    def _walk_grade(self) -> float:
        """Slow, mean-reverting walk in gradient, bounded by the tier.

        Real elevation, not decorative sine hills: the road climbs for a while,
        then descends for a while, and the builder eases between the two.

        The reversion term matters. A plain bounded random walk has no pull back
        toward level, so it can sit near the limit for most of a stage — that
        produced 190 m of climb over 2 km, a sustained 10% gradient that would
        have the car power-limited for the whole run. Reverting keeps the local
        steepness while bounding the net.
        """
        step = _uni(self.rng, -0.45, 0.45) * self.p.max_grade
        self.grade = float(np.clip(self.grade * GRADE_REVERSION + step,
                                   -self.p.max_grade, self.p.max_grade))
        return self.grade

    # -- direction -------------------------------------------------------- #

    def _next_dir(self, alternate_bias: float = 0.62) -> str:
        """Corner direction, biased to alternate.

        A road that turns the same way repeatedly spirals, and a spiral either
        crosses itself or becomes a racetrack. Alternating keeps it a stage.
        """
        if _chance(self.rng, alternate_bias):
            self.last_dir = "right" if self.last_dir == "left" else "left"
        return self.last_dir

    def _radius(self, tight: float, loose: float) -> float:
        """A radius between two fractions of the tier's range."""
        lo = self.p.min_radius + (self.p.max_radius - self.p.min_radius) * tight
        hi = self.p.min_radius + (self.p.max_radius - self.p.min_radius) * loose
        return _uni(self.rng, lo, hi)

    def _camber(self, direction: str, radius: float) -> float | None:
        """Banked into the turn, unless this tier is being unkind."""
        if _chance(self.rng, self.p.off_camber_chance):
            return -_uni(self.rng, 0.02, 0.06)
        return None      # builder's default: bank into the turn

    def _width(self, scale: float = 1.0) -> float:
        return self.base_width * scale * _uni(self.rng, 0.92, 1.08)

    # ------------------------------------------------------------------ #
    # the archetype vocabulary
    # ------------------------------------------------------------------ #

    def straight(self, b: ProfileBuilder) -> None:
        length = _uni(self.rng, 45.0, 190.0)
        b.straight(length, grade=self._walk_grade(),
                   width=self._width(1.05), surface=self.surface)
        if _chance(self.rng, self.p.crest_chance):
            self._crest(b)

    def _crest(self, b: ProfileBuilder, takeoff_scale: float = 1.0) -> None:
        """A crest sized to launch the car at this tier's takeoff speed."""
        length = _uni(self.rng, *self.p.crest_length_m)
        takeoff = _uni(self.rng, *self.p.crest_takeoff_mps) * takeoff_scale
        b.crest(length, crest_height_for_takeoff(length, takeoff),
                width=self._width(1.0), surface=self.surface)

    def sweeper(self, b: ProfileBuilder) -> None:
        """A long, fast, committed corner. The fifth-gear one."""
        r = self._radius(0.55, 1.0)
        d = self._next_dir(alternate_bias=0.5)
        b.corner(r, _uni(self.rng, 45.0, 110.0), d, ease=_uni(self.rng, 14.0, 26.0),
                 grade=self._walk_grade(), camber=self._camber(d, r),
                 width=self._width(1.0), surface=self.surface)

    def hairpin(self, b: ProfileBuilder) -> None:
        """Slowest corner on the stage. Widened slightly on the way in, because
        real hairpins open out where the cars cut them."""
        r = self._radius(0.0, 0.10)
        d = self._next_dir()
        b.straight(_uni(self.rng, 25.0, 55.0), grade=self.grade,
                   width=self._width(1.12), surface=self.surface)
        b.corner(r, _uni(self.rng, 135.0, 180.0), d, ease=_uni(self.rng, 6.0, 11.0),
                 grade=self._walk_grade(), camber=self._camber(d, r),
                 width=self._width(1.05), surface=self.surface)

    def ess(self, b: ProfileBuilder) -> None:
        """Alternating medium corners with short links — flow, not stop-start."""
        n = int(self.rng.integers(2, 5))
        for _ in range(n):
            r = self._radius(0.18, 0.5)
            d = self._next_dir(alternate_bias=0.95)
            b.corner(r, _uni(self.rng, 40.0, 80.0), d, ease=_uni(self.rng, 8.0, 15.0),
                     grade=self.grade, camber=self._camber(d, r),
                     width=self._width(0.98), surface=self.surface)
            b.straight(_uni(self.rng, 8.0, 24.0), grade=self.grade,
                       width=self._width(1.0), surface=self.surface)

    def tightening(self, b: ProfileBuilder) -> None:
        """Opens innocently, then closes. The corner that catches people out."""
        d = self._next_dir()
        r0 = self._radius(0.35, 0.65)
        r1 = max(self.p.min_radius, r0 * _uni(self.rng, 0.35, 0.55))
        for r, ang in ((r0, _uni(self.rng, 35.0, 55.0)), (r1, _uni(self.rng, 50.0, 90.0))):
            b.corner(r, ang, d, ease=_uni(self.rng, 8.0, 16.0),
                     grade=self.grade, camber=self._camber(d, r),
                     width=self._width(1.0), surface=self.surface)

    def crest_braking(self, b: ProfileBuilder) -> None:
        """Over a crest, and the corner is waiting on the other side.

        The reason the terrain block previews vcurv at speed-scaled distances:
        the agent has to see this coming while it is still on the ground.
        """
        b.straight(_uni(self.rng, 30.0, 70.0), grade=self._walk_grade(),
                   width=self._width(1.0), surface=self.surface)
        # Slightly sharper than a plain crest: this one is meant to unsettle the
        # car right before it has to brake.
        self._crest(b, takeoff_scale=0.9)
        b.straight(_uni(self.rng, 12.0, 30.0), grade=self.grade,
                   width=self._width(1.0), surface=self.surface)
        r = self._radius(0.05, 0.3)
        d = self._next_dir()
        b.corner(r, _uni(self.rng, 60.0, 110.0), d, ease=_uni(self.rng, 7.0, 13.0),
                 grade=self.grade, camber=self._camber(d, r),
                 width=self._width(0.98), surface=self.surface)

    def plunge(self, b: ProfileBuilder) -> None:
        """Steep descent, then a corner at the bottom with the car light."""
        self.grade = -abs(self.p.max_grade) * _uni(self.rng, 0.7, 1.0)
        b.straight(_uni(self.rng, 50.0, 110.0), grade=self.grade,
                   width=self._width(1.0), surface=self.surface)
        r = self._radius(0.05, 0.35)
        d = self._next_dir()
        b.corner(r, _uni(self.rng, 55.0, 100.0), d, ease=_uni(self.rng, 8.0, 14.0),
                 grade=self.grade * 0.5, camber=self._camber(d, r),
                 width=self._width(1.0), surface=self.surface)
        self.grade *= 0.3

    def chicane(self, b: ProfileBuilder) -> None:
        """A quick flick and back. Barely slows a committed driver, punishes a
        greedy one."""
        d = self._next_dir()
        r = self._radius(0.12, 0.35)
        ang = _uni(self.rng, 30.0, 55.0)
        b.corner(r, ang, d, ease=6.0, grade=self.grade,
                 width=self._width(1.0), surface=self.surface)
        other = "right" if d == "left" else "left"
        self.last_dir = other
        b.corner(r * _uni(self.rng, 0.85, 1.15), ang, other, ease=6.0,
                 grade=self.grade, width=self._width(1.0), surface=self.surface)

    # ------------------------------------------------------------------ #
    # composition
    # ------------------------------------------------------------------ #

    def build_profile(self):
        b = ProfileBuilder(width=self.base_width, surface=self.surface)
        target = _uni(self.rng, *self.p.length_m)

        def _s() -> float:
            # Profile.s = arange(n) * ds, so length is (n - 1) * ds.
            n = len(b._k)
            return 0.0 if n == 0 else float((n - 1) * b.ds)

        sectors: list[dict[str, Any]] = []

        # Start with a clean run so the car is up to speed before the first
        # real feature, and so the start line is not inside a corner.
        s0 = _s()
        b.straight(_uni(self.rng, 60.0, 110.0), width=self.base_width,
                   surface=self.surface)
        sectors.append({"name": "start", "s_start": s0, "s_end": _s()})

        feature = "start"
        used: list[str] = []
        while len(b._k) * b.ds < target:
            feature = _pick(self.rng, TRANSITIONS[feature])
            self._maybe_change_surface()
            s_a = _s()
            getattr(self, feature)(b)
            sectors.append({"name": feature, "s_start": s_a, "s_end": _s()})
            used.append(feature)

        # Finish on a straight: a finish line inside a hairpin is a lottery.
        s_a = _s()
        b.straight(_uni(self.rng, 55.0, 95.0), grade=0.0,
                   width=self.base_width * 1.1, surface=self.surface)
        sectors.append({"name": "finish", "s_start": s_a, "s_end": _s()})
        self.features = used
        self.sectors = sectors
        return b.build()


# --------------------------------------------------------------------------- #
# obstacles
# --------------------------------------------------------------------------- #

def _place_trees(stage: dict[str, Any], rng: np.random.Generator,
                 params: TierParams) -> list[dict[str, Any]]:
    """Line the corridor, always outside it.

    The hard rule from §5: nothing that can terminate the agent sits inside the
    drivable corridor unless the agent can sense it. Placement goes through
    ``Track`` so trees follow the *actual* corridor edge including width
    changes, which makes the invariant hold by construction rather than by
    inspection. ``obstacles_in_corridor`` then checks it, and a test asserts it.
    """
    track = Track(stage)
    trees: list[dict[str, Any]] = []
    s = params.tree_spacing
    while s < track.length - params.tree_spacing:
        i = int(np.clip(round(s / track.ds), 0, len(track.s) - 1))
        for side in (-1.0, 1.0):
            if not _chance(rng, 0.82):
                continue
            radius = _uni(rng, 0.28, 0.55)
            clearance = _uni(rng, *params.tree_clearance)
            offset = side * (track.half_width[i] + clearance + radius)
            trees.append({
                "kind": "tree",
                "x": float(track.x[i] + track.nx[i] * offset),
                "y": float(track.y[i] + track.ny[i] * offset),
                "z": float(track.z[i]),
                "radius": radius,
                "height": _uni(rng, 5.0, 12.0),
                "s": float(track.s[i]),
                "lateral": float(offset),
            })
        s += params.tree_spacing * _uni(rng, 0.7, 1.4)
    return trees


# --------------------------------------------------------------------------- #
# self-intersection
# --------------------------------------------------------------------------- #

def _self_intersects(stage: dict[str, Any], min_separation: float) -> bool:
    """True if the road passes too close to a distant part of itself.

    Composing corners can fold a stage back over its own path. Real rally stages
    do not cross themselves, and a crossing would put one section's trees in
    another section's road — the exact invisible-killer failure the obstacle
    rule exists to prevent.
    """
    pts = np.array([[p["x"], p["y"]] for p in stage["centerline"]])
    s = np.array([p["s"] for p in stage["centerline"]])
    if len(pts) < 3:
        return False
    d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    far_apart = np.abs(s[:, None] - s[None, :]) > 4.0 * min_separation
    return bool((d[far_apart] < min_separation).any())


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

MAX_ATTEMPTS = 12


def generate(seed: int, tier: int = 0) -> dict[str, Any]:
    """Generate a stage. Deterministic in ``(GENERATOR_VERSION, seed, tier)``.

    Re-rolls on a self-intersecting layout, deriving each retry seed from the
    original so the retry sequence is itself reproducible.
    """
    params = tier_params(tier)
    last: dict[str, Any] | None = None

    for attempt in range(MAX_ATTEMPTS):
        # Derived deterministically; the same request always retries identically.
        attempt_seed = (int(seed) * 1_000_003 + attempt * 7_919) % (2**63)
        gen = StageGenerator(attempt_seed, tier)
        profile = gen.build_profile()

        stage = build_stage(
            profile,
            stage_id=f"gen_{int(seed)}_t{gen.tier}",
            name=f"{params.name} · seed {int(seed)}",
            seed=int(seed),
            tier=gen.tier,
            generator_version=GENERATOR_VERSION,
        )
        stage["meta"] = {
            "features": gen.features,
            "sectors": gen.sectors,
            "attempt": attempt,
        }
        last = stage

        min_sep = float(np.max([p["width"] for p in stage["centerline"]])) \
            + 2.0 * params.tree_clearance[1] + 6.0
        if not _self_intersects(stage, min_sep):
            stage["obstacles"] = _place_trees(
                stage, np.random.default_rng(attempt_seed ^ 0x5EED), params)
            return stage

    # Every attempt folded over itself. Return the last one without trees rather
    # than raising: a stage with no obstacles is degraded but drivable, and a
    # generator that can fail is a training run that can die at 3 a.m.
    assert last is not None
    last["obstacles"] = []
    last["meta"]["self_intersecting"] = True
    return last
