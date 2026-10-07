"""GENOTYPE: slow-changing structural parameters and the morphology mutation ops.

This is the low-dimensional (14-gene) prototype genome. Each gene carries three
nested realism envelopes:
    ha   HUMAN-ACHIEVABLE  reachable by the user through training/nutrition/rehab
    hp   HUMAN-POSSIBLE    rare but documented human anatomy/physiology
    op   OPEN-EVOLUTION    nonhuman / speculative ranges

PLAN.md describes the full developmental encoding (rules that grow muscles,
duplicate heads, route tendons) which the real engine needs so that mutations
produce coherent anatomy. Here we approximate it with bounded per-gene mutation
plus an anatomical-viability check (see realism.py / budgets.py). Discrete
topology mutations (duplicate/split/fuse/new-muscle) are stubbed in
`topology_mutations()` and flagged OPEN-EVOLUTION.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import numpy as np


@dataclass(frozen=True)
class GeneSpec:
    name: str
    unit: str
    prior: float          # population-prior default (status=unknown until measured)
    ha: tuple[float, float]
    hp: tuple[float, float]
    op: tuple[float, float]
    note: str = ""


# 14-gene prototype. Values are population priors, NOT the user's measurements.
GENE_SPECS: list[GeneSpec] = [
    GeneSpec("height_m",            "m",       1.78, (1.74, 1.82), (1.50, 2.10), (1.20, 2.60), "structural scale"),
    GeneSpec("leg_length_frac",     "frac",    0.48, (0.46, 0.50), (0.44, 0.53), (0.35, 0.62), "leg / height"),
    GeneSpec("lever_advantage",     "ratio",   1.00, (0.95, 1.05), (0.85, 1.20), (0.60, 1.80), "internal moment-arm factor"),
    GeneSpec("pcsa_capacity",       "cm^2",    520., (400., 720.), (350., 900.), (200., 2000.), "lower-body PCSA ceiling"),
    GeneSpec("fast_fiber_frac",     "frac",    0.50, (0.40, 0.62), (0.20, 0.80), (0.05, 0.97), "type-II fraction"),
    GeneSpec("tendon_stiffness",    "kN/m",    200., (150., 280.), (100., 400.), (40., 900.),  "series elastic"),
    GeneSpec("tendon_elastic_ret",  "frac",    0.35, (0.25, 0.45), (0.15, 0.60), (0.05, 0.90), "SSC energy reuse"),
    GeneSpec("vo2max_ceiling",      "ml/kg/min", 50., (44., 62.),  (35., 85.),  (25., 130.),  "aerobic structural ceiling"),
    GeneSpec("anaerobic_W",         "kJ",      22., (16., 28.),   (10., 35.),  (5., 80.),    "W' anaerobic capacity"),
    GeneSpec("heat_dissipation",    "W/m^2",   90., (75., 110.),  (60., 140.), (40., 400.),  "max sustainable heat flux"),
    GeneSpec("bone_load_capacity",  "ratio",   1.00, (0.90, 1.15), (0.75, 1.40), (0.50, 3.00), "skeletal/connective strength"),
    GeneSpec("neural_capacity",     "ratio",   1.00, (0.90, 1.15), (0.70, 1.40), (0.40, 4.00), "control-complexity budget"),
    GeneSpec("recovery_capacity",   "ratio",   1.00, (0.85, 1.20), (0.65, 1.45), (0.40, 3.00), "adaptation/recovery rate"),
    GeneSpec("tendon_length_frac",  "frac",    0.30, (0.26, 0.34), (0.20, 0.42), (0.10, 0.60), "tendon / fascicle ratio"),
]
GENE_INDEX = {g.name: i for i, g in enumerate(GENE_SPECS)}
N_GENES = len(GENE_SPECS)


class Genotype:
    """A point in gene space plus convenience accessors."""

    def __init__(self, values: dict[str, float] | np.ndarray | None = None):
        if values is None:
            self.x = np.array([g.prior for g in GENE_SPECS], dtype=float)
        elif isinstance(values, dict):
            self.x = np.array([values.get(g.name, g.prior) for g in GENE_SPECS], dtype=float)
        else:
            self.x = np.asarray(values, dtype=float).copy()

    # dict-style access by gene name
    def __getitem__(self, name: str) -> float:
        return float(self.x[GENE_INDEX[name]])

    def __setitem__(self, name: str, v: float) -> None:
        self.x[GENE_INDEX[name]] = v

    def as_dict(self) -> dict[str, float]:
        return {g.name: float(self.x[i]) for i, g in enumerate(GENE_SPECS)}

    def copy(self) -> "Genotype":
        return Genotype(self.x.copy())


# --- mutation --------------------------------------------------------------

def _envelope(spec: GeneSpec, level: str) -> tuple[float, float]:
    return {"ha": spec.ha, "hp": spec.hp, "op": spec.op}[level]


def mutate(g: Genotype, level: str = "ha", sigma_frac: float = 0.06,
           rng: np.random.Generator | None = None,
           clamp: bool = True) -> Genotype:
    """Gaussian mutation, scaled to each gene's envelope width and clamped to
    the chosen realism envelope. `level` in {"ha","hp","op"}.
    """
    rng = rng or np.random.default_rng()
    child = g.copy()
    for i, spec in enumerate(GENE_SPECS):
        lo, hi = _envelope(spec, level)
        step = rng.normal(0.0, sigma_frac * (hi - lo))
        v = child.x[i] + step
        if clamp:
            v = float(np.clip(v, lo, hi))
        child.x[i] = v
    return child


def topology_mutations() -> list[str]:
    """Discrete musculoskeletal-topology operators. [SCI-FI for this prototype]

    The real engine implements these against the developmental encoding so that
    a 'new muscle head' attaches to valid skeletal territory, produces a moment
    about a joint, and pays mass/metabolic/control costs (see PLAN.md ->
    Evolvable Musculoskeletal Topology). Here they are catalogued only.
    """
    return [
        "duplicate_muscle_head",     # OPEN-EVOLUTION
        "split_into_compartments",   # OPEN-EVOLUTION (independent neural control)
        "fuse_redundant_muscles",    # OPEN-EVOLUTION
        "reroute_tendon_path",       # OPEN-EVOLUTION
        "evolve_new_stabilizer",     # OPEN-EVOLUTION
        "lose_unused_muscle",        # OPEN-EVOLUTION (mass refund)
    ]


def random_genotype(level: str = "ha", rng: np.random.Generator | None = None) -> Genotype:
    rng = rng or np.random.default_rng()
    g = Genotype()
    for i, spec in enumerate(GENE_SPECS):
        lo, hi = _envelope(spec, level)
        g.x[i] = rng.uniform(lo, hi)
    return g
