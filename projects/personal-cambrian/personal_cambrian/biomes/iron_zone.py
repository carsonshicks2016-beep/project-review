"""IRON ZONE — maximal force / static torque.

[APPROX] Whole-body max external load is modeled from muscle force capacity:
    F_muscle = sigma * PCSA_eff * activation
    joint_torque = F_muscle * internal_moment_arm
    external_load ∝ joint_torque / external_lever(limb length)
Specific tension sigma ≈ 25 N/cm^2 is within the [EVIDENCE] 20-35 N/cm^2 range.
Strength is NOT driven by VO2max — the limiter here is contractile area, neural
drive, and leverage.
"""
from __future__ import annotations

import numpy as np
from .base import Biome, BiomeResult

# Whole-body max external load via allometric scaling of muscle mass.
# [APPROX] 1RM scales ~ muscle^0.67 (cross-section ∝ mass^2/3) modulated by
# leverage, neural drive, and technique. STRENGTH_K calibrated so a balanced
# Agent Zero squats ~90 kg and a strength specialist ~190-210 kg.
STRENGTH_K = 13.0


class IronZone(Biome):
    name = "iron_zone"
    unit = "kg(1RM-equiv)"

    def evaluate(self, agent) -> BiomeResult:
        g, p = agent.genotype, agent.phenotype
        skill = p.skill.get("iron_zone", 0.8)

        load_kg = (STRENGTH_K * p.muscle_mass_kg ** 0.67
                   * g["lever_advantage"] * p.strength_coeff * skill)

        # injury: spine/connective load vs bone+connective capacity; tendon stress
        injury = max(0.0, (load_kg / (150.0 * g["bone_load_capacity"])) - 0.6)
        injury += 0.4 * p.tissue_damage

        return BiomeResult(
            self.name, score=float(load_kg), unit=self.unit, injury=float(injury),
            detail={"joint_torque_Nm": float(load_kg * 9.81 * 0.4),
                    "strength_to_mass": float(load_kg / max(agent.body_mass, 1.0))},
        )
