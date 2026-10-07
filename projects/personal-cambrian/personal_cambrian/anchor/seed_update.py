"""SEED-METRIC UPDATE (ROADMAP Stage 11.3).

Maps anchored `Metric`s onto the Agent-Zero seed's PRIORS and scales the seed genome
so its developed body reflects the measured values:

  * a MEASURED metric -> a `measured` prior with a NARROW confidence interval;
  * an unmeasured parameter -> a `prior` with an explicit POPULATION DEFAULT and a WIDE
    interval -- labelled a prior, never claimed as the person's value (so nothing is
    invented; unknowns stay priors).

`apply_geometry` then rescales the seed (uniform dimension scale for height, uniform
density scale for mass) so `develop(anchored).height ≈ measured height` and
`total_mass ≈ measured weight`. Strength / physiology priors are carried for the
Stage-11.4 near-human calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..encoding.genome import Genome, Shape
from ..morphogenesis import develop
from .metric import MetricSet


def standing_height(genome: Genome) -> float:
    """Geometric standing height of the developed creature: the world-z extent of the
    bodies including their sizes (a faithful stand-in for human standing height)."""
    m = develop(genome)
    zlo, zhi = float("inf"), float("-inf")
    for b in m.bodies:
        z = float(b.world_pos[2])
        d = b.dims
        if b.shape is Shape.CAPSULE:
            half = float(d["radius"]) + 0.5 * float(d["length"])
        elif b.shape in (Shape.BOX, Shape.ELLIPSOID):
            half = max(float(d["x"]), float(d["y"]), float(d["z"]))
        else:
            half = float(d["radius"])
        zlo, zhi = min(zlo, z - half), max(zhi, z + half)
    return max(zhi - zlo, 1e-6)

# population reference priors for parameters with no measurement (clearly priors,
# not personal data). Units: 1RM in lb, hr in bpm, hrv in ms.
_POP_DEFAULTS = {
    "bench_press_1rm": (135.0, "lb"), "squat_1rm": (185.0, "lb"),
    "deadlift_1rm": (225.0, "lb"), "overhead_press_1rm": (95.0, "lb"),
    "resting_hr": (65.0, "bpm"), "hrv_rmssd": (45.0, "ms"),
}


@dataclass(frozen=True)
class SeedPrior:
    name: str
    value: float
    ci: tuple                       # (lo, hi)
    status: str                     # "measured" | "prior"
    source: str
    unit: str = ""

    @property
    def ci_width(self) -> float:
        return self.ci[1] - self.ci[0]

    def to_dict(self) -> dict:
        return {"name": self.name, "value": self.value, "ci": list(self.ci),
                "status": self.status, "source": self.source, "unit": self.unit}


def _prior(name, metric, *, default, unit, rel_tol=0.02, default_rel=0.35) -> SeedPrior:
    """A measured prior (narrow CI) if `metric` is known, else a population prior
    (wide CI) around `default`."""
    if metric is not None and metric.known:
        ci = metric.ci or (metric.value * (1 - rel_tol), metric.value * (1 + rel_tol))
        return SeedPrior(name, metric.value, tuple(ci), "measured",
                         metric.source, metric.unit or unit)
    return SeedPrior(name, default, (default * (1 - default_rel), default * (1 + default_rel)),
                     "prior", "population-default", unit)


def build_priors(metrics: MetricSet, seed_genome: Genome) -> dict:
    """Build the seed's prior set from a MetricSet. Geometry defaults come from the
    seed itself; strength/physiology defaults from population references."""
    seed_h = standing_height(seed_genome)
    seed_m = develop(seed_genome).total_mass()
    priors = {
        "height_m": _prior("height_m", metrics.get("height_m"), default=seed_h,
                           unit="m", rel_tol=0.01, default_rel=0.30),
        "body_mass_kg": _prior("body_mass_kg", metrics.get("weight_kg"), default=seed_m,
                               unit="kg", rel_tol=0.01, default_rel=0.30),
    }
    for name, (dflt, unit) in _POP_DEFAULTS.items():
        priors[name] = _prior(name, metrics.get(name), default=dflt, unit=unit)
    return priors


def _scaled(genome: Genome, *, dim_scale: float = 1.0, density_scale: float = 1.0) -> Genome:
    """Uniformly rescale the whole creature: part sizes, ATTACHMENT-SITE positions, and
    EDGE offsets all by `dim_scale` (so both body sizes and skeleton layout scale), and
    densities by `density_scale`."""
    g = Genome.from_json(genome.to_json())
    for p in g.parts:
        p.dims = {k: v * dim_scale for k, v in p.dims.items()}
        p.density = p.density * density_scale
        for s in p.sites:
            s.pos = tuple(c * dim_scale for c in s.pos)
    for e in g.edges:
        e.pos = tuple(c * dim_scale for c in e.pos)
    return g


def apply_geometry(seed_genome: Genome, priors: dict) -> Genome:
    """Return the seed rescaled so its developed height & mass match the geometry priors.
    Uniform dimension scale sets standing height; uniform density scale then sets mass."""
    seed_h = standing_height(seed_genome)
    dim_scale = priors["height_m"].value / seed_h if seed_h > 0 else 1.0
    g = _scaled(seed_genome, dim_scale=dim_scale)
    m1 = develop(g).total_mass()
    density_scale = priors["body_mass_kg"].value / m1 if m1 > 0 else 1.0
    return _scaled(g, density_scale=density_scale)


def anchored_seed(metrics: MetricSet, seed_genome: Genome):
    """Convenience: build priors + return (anchored_genome, priors)."""
    priors = build_priors(metrics, seed_genome)
    return apply_geometry(seed_genome, priors), priors
