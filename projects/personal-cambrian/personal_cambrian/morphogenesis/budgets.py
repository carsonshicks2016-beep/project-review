"""BIOLOGICAL BUDGETS (ROADMAP Stage 6.1): every structure must be paid for.

A creature cannot get better for free. This module reads the *instantiated*
`Morphology` (the bodies that morphogenesis actually grew and the muscle routes
it actually placed) and charges each structure its real physical cost:

  * mass           -- geom volume x density (rigid tissue) + muscle volume x density
  * surface area   -- per-shape, drives heat dissipation
  * metabolic power -- resting (mass) + active (muscle volume x activation cost)
  * heat           -- production (metabolic) vs dissipation (surface area)
  * neural load    -- #independently-controlled actuators, weighted by fibre speed,
                      vs CNS capacity (allometric, ~ mass^(2/3))
  * recovery       -- load/damage proxy (active fast-twitch mass) vs capacity

Costs are returned per-structure AND aggregated, with non-negative *overflow*
terms (production - capacity) that Stage 6.5 folds into the fitness as a penalty.
Adding a limb or a muscle therefore *necessarily* raises mass / metabolic / neural
load -- improvement has to out-earn its bill.

[APPROX] Coefficients are order-of-magnitude physiological, not calibrated. The
point is monotonic, mechanically-coherent pressure, not clinical accuracy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..encoding.genome import Genome, Shape
from .develop import Morphology, _volume

# --- physical constants -----------------------------------------------------
MUSCLE_DENSITY = 1060.0          # kg/m^3 [EVIDENCE] skeletal muscle ~1.06 g/cm^3
RESTING_W_PER_KG = 1.2          # [APPROX] basal metabolic power per kg lean mass
ACTIVE_W_PER_KG_MUSCLE = 12.0   # [APPROX] peak active muscle power draw per kg
DISSIP_W_PER_M2 = 90.0          # [APPROX] convective+radiative skin loss per m^2
NEURAL_CAP_COEF = 4.0           # [APPROX] CNS control channels ~ coef * mass^(2/3)
CM2_TO_M2 = 1.0e-4


def _surface_area(shape: Shape, dims: dict) -> float:
    """Outer surface area (m^2). dims are already world-scaled by morphogenesis."""
    if shape is Shape.CAPSULE:        # cylinder side + two hemispherical caps
        r, L = dims["radius"], dims["length"]
        return 2.0 * math.pi * r * L + 4.0 * math.pi * r * r
    if shape is Shape.BOX:            # x,y,z are half-extents
        x, y, z = dims["x"], dims["y"], dims["z"]
        return 8.0 * (x * y + y * z + x * z)
    if shape is Shape.SPHERE:
        r = dims["radius"]
        return 4.0 * math.pi * r * r
    if shape is Shape.ELLIPSOID:      # Thomsen approximation, p = 1.6075
        a, b, c = dims["x"], dims["y"], dims["z"]
        p = 1.6075
        return 4.0 * math.pi * (((a ** p * b ** p + a ** p * c ** p
                                  + b ** p * c ** p) / 3.0) ** (1.0 / p))
    return 0.0


def _muscle_volume_m3(pcsa_cm2: float, fiber_len_m: float) -> float:
    """Physiological volume = PCSA x optimal fibre length."""
    return (pcsa_cm2 * CM2_TO_M2) * fiber_len_m


@dataclass
class Budgets:
    """Per-structure + aggregate biological cost of one morphology."""
    # aggregate masses
    structural_mass_kg: float
    muscle_mass_kg: float
    total_mass_kg: float
    surface_area_m2: float
    # power
    resting_metabolic_W: float
    active_metabolic_W: float
    metabolic_W: float                 # resting + active
    heat_production_W: float
    heat_dissipation_W: float
    # control
    neural_load: float
    neural_capacity: float
    # recovery
    recovery_cost: float
    recovery_capacity: float
    # per-structure breakdowns (id -> cost dict)
    per_body: dict = field(default_factory=dict)
    per_muscle: dict = field(default_factory=dict)

    # --- overflows (>= 0): what the body cannot pay for ---------------------
    @property
    def heat_overflow(self) -> float:
        return max(0.0, self.heat_production_W - self.heat_dissipation_W)

    @property
    def neural_overflow(self) -> float:
        return max(0.0, self.neural_load - self.neural_capacity)

    @property
    def recovery_overflow(self) -> float:
        return max(0.0, self.recovery_cost - self.recovery_capacity)

    @property
    def overflows(self) -> dict:
        return {"heat": self.heat_overflow,
                "neural": self.neural_overflow,
                "recovery": self.recovery_overflow}

    @property
    def total_overflow(self) -> float:
        return self.heat_overflow + self.neural_overflow + self.recovery_overflow

    def penalty(self, weights: Optional[dict] = None) -> float:
        """Fitness penalty (>= 0) from normalized overflows (Stage 6.5 wiring).

        Each overflow is normalized by the corresponding capacity so the terms
        are comparable, then weighted (default 1.0 each)."""
        w = weights or {}
        heat = self.heat_overflow / max(self.heat_dissipation_W, 1.0)
        neural = self.neural_overflow / max(self.neural_capacity, 1.0)
        recovery = self.recovery_overflow / max(self.recovery_capacity, 1.0)
        return (w.get("heat", 1.0) * heat
                + w.get("neural", 1.0) * neural
                + w.get("recovery", 1.0) * recovery)

    def to_dict(self) -> dict:
        return {
            "structural_mass_kg": self.structural_mass_kg,
            "muscle_mass_kg": self.muscle_mass_kg,
            "total_mass_kg": self.total_mass_kg,
            "surface_area_m2": self.surface_area_m2,
            "resting_metabolic_W": self.resting_metabolic_W,
            "active_metabolic_W": self.active_metabolic_W,
            "metabolic_W": self.metabolic_W,
            "heat_production_W": self.heat_production_W,
            "heat_dissipation_W": self.heat_dissipation_W,
            "neural_load": self.neural_load,
            "neural_capacity": self.neural_capacity,
            "recovery_cost": self.recovery_cost,
            "recovery_capacity": self.recovery_capacity,
            "overflows": self.overflows,
            "total_overflow": self.total_overflow,
        }


def compute_budgets(morphology: Morphology,
                    genome: Optional[Genome] = None) -> Budgets:
    """Charge every structure in `morphology` its real biological cost.

    `genome` is accepted for API symmetry / future gene-level costs; all current
    costs are read from the instantiated morphology so they reflect what actually
    grew (recursion, symmetry, mutation) rather than the genotype's intent.
    """
    # --- mass + surface area, per body ------------------------------------
    structural_mass = 0.0
    surface_area = 0.0
    per_body = {}
    for b in morphology.bodies:
        vol = _volume(b.shape, b.dims)
        mass = vol * b.density
        area = _surface_area(b.shape, b.dims)
        structural_mass += mass
        surface_area += area
        per_body[b.id] = {"mass_kg": mass, "volume_m3": vol,
                          "surface_area_m2": area}

    # --- muscle mass + active metabolic + neural load, per muscle ---------
    muscle_mass = 0.0
    active_metabolic = 0.0
    neural_load = 0.0
    recovery_cost = 0.0
    per_muscle = {}
    for m in morphology.muscles:
        vol = _muscle_volume_m3(m.pcsa_cm2, m.optimal_fiber_len)
        mass = vol * MUSCLE_DENSITY
        # active power scales with muscle mass and its activation cost
        power = ACTIVE_W_PER_KG_MUSCLE * mass * m.activation_cost
        # one independent control channel; fast fibre is costlier to coordinate
        chan = 1.0 + 0.5 * m.fiber_type
        # recovery: active mass weighted toward fast-glycolytic (fatigue/damage)
        rec = mass * (0.5 + m.fiber_type)
        muscle_mass += mass
        active_metabolic += power
        neural_load += chan
        recovery_cost += rec
        per_muscle[m.id] = {"mass_kg": mass, "volume_m3": vol,
                            "active_metabolic_W": power, "neural_channels": chan,
                            "recovery_cost": rec}

    total_mass = structural_mass + muscle_mass

    # --- aggregate power / capacities -------------------------------------
    resting_metabolic = RESTING_W_PER_KG * total_mass
    metabolic = resting_metabolic + active_metabolic
    # essentially all metabolic power degrades to heat
    heat_production = metabolic
    heat_dissipation = DISSIP_W_PER_M2 * surface_area
    # CNS allometry: control capacity grows sub-linearly with body size
    neural_capacity = NEURAL_CAP_COEF * (total_mass ** (2.0 / 3.0))
    # recovery capacity scales with (oxidative) mass available to repair
    recovery_capacity = 0.6 * total_mass

    return Budgets(
        structural_mass_kg=structural_mass,
        muscle_mass_kg=muscle_mass,
        total_mass_kg=total_mass,
        surface_area_m2=surface_area,
        resting_metabolic_W=resting_metabolic,
        active_metabolic_W=active_metabolic,
        metabolic_W=metabolic,
        heat_production_W=heat_production,
        heat_dissipation_W=heat_dissipation,
        neural_load=neural_load,
        neural_capacity=neural_capacity,
        recovery_cost=recovery_cost,
        recovery_capacity=recovery_capacity,
        per_body=per_body,
        per_muscle=per_muscle,
    )
