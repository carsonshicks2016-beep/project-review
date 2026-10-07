"""BIOLOGICAL BUDGETS: prevent cost-free improvement.

Every body must pay for what it has. We compute total mass, resting+active
metabolic demand, peak heat production vs. dissipation, oxygen delivery vs.
demand, recovery cost, and neural-control complexity. Violations return a
non-negative overflow used as a fitness penalty.

[APPROX] Coefficients are order-of-magnitude physiological, not calibrated.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .genome import Genotype
from .phenotype import Phenotype


@dataclass
class BudgetReport:
    body_mass_kg: float
    bsa_m2: float                 # body surface area
    resting_kcal: float
    peak_heat_W: float
    heat_dissip_W: float
    o2_demand: float
    o2_supply: float
    recovery_cost: float
    neural_load: float
    neural_capacity: float
    violations: dict[str, float]  # name -> overflow (>=0)

    @property
    def total_overflow(self) -> float:
        return float(sum(self.violations.values()))


def evaluate_budgets(g: Genotype, p: Phenotype) -> BudgetReport:
    height = g["height_m"]

    # --- mass: non-muscle lean (bone/organ/water) from height, + muscle, + fat
    # [APPROX] frame ~ 9.5*h^2 (≈30 kg at 1.78 m); total = lean/(1-fat_frac)
    frame = 9.5 * height ** 2                       # non-muscle lean (kg)
    lean = frame + p.muscle_mass_kg
    body_mass = lean / (1.0 - p.body_fat_frac)

    # Du Bois body-surface-area [EVIDENCE] (approx, mass in kg, height in cm)
    bsa = 0.007184 * (body_mass ** 0.425) * ((height * 100) ** 0.725)

    # --- metabolic demand ---------------------------------------------------
    # resting kcal ~ proportional to lean mass [APPROX]
    resting_kcal = 370 + 21.6 * lean

    # peak heat at max effort scales with active muscle mass & anaerobic output
    peak_heat_W = 2.6 * p.muscle_mass_kg * (0.5 + p.anaerobic_fitness) + 40.0
    heat_dissip_W = g["heat_dissipation"] * bsa

    # --- oxygen delivery vs demand at VO2max --------------------------------
    vo2_supply = g["vo2max_ceiling"] * p.aerobic_fitness     # ml/kg/min realized
    # bigger, faster-twitch muscle costs more O2 to drive aerobically
    vo2_demand = 22.0 + 0.12 * p.muscle_mass_kg + 8.0 * g["fast_fiber_frac"]

    # --- recovery cost vs capacity -----------------------------------------
    recovery_cost = p.recovery_debt + 0.5 * p.tissue_damage
    recovery_capacity = g["recovery_capacity"]

    # --- neural control complexity -----------------------------------------
    # more independently-controlled, faster muscle costs control bandwidth
    neural_load = 0.5 + 0.4 * g["fast_fiber_frac"] + 0.3 * (p.muscle_mass_kg / 40.0)
    neural_capacity = g["neural_capacity"]

    violations = {
        "heat":     max(0.0, peak_heat_W - heat_dissip_W) / max(heat_dissip_W, 1.0),
        "oxygen":   max(0.0, vo2_demand - vo2_supply) / max(vo2_supply, 1.0),
        "recovery": max(0.0, recovery_cost - recovery_capacity),
        "neural":   max(0.0, neural_load - neural_capacity),
    }

    return BudgetReport(
        body_mass_kg=body_mass, bsa_m2=bsa, resting_kcal=resting_kcal,
        peak_heat_W=peak_heat_W, heat_dissip_W=heat_dissip_W,
        o2_demand=vo2_demand, o2_supply=vo2_supply,
        recovery_cost=recovery_cost, neural_load=neural_load,
        neural_capacity=neural_capacity, violations=violations,
    )
