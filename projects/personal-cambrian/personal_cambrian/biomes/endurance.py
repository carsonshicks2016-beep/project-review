"""ENDURANCE CORRIDOR — sustainable aerobic velocity.

[APPROX] Velocity at VO2max from realized aerobic power and running economy:
    v_at_vo2max = (VO2max_realized - VO2rest) / O2_cost_per_m
Running economy (O2 cost per metre) worsens with body mass and high fast-fiber
fraction, improves with elastic tendons. Sustainable race pace is a fraction of
v_at_vo2max (critical-velocity view). Here VO2max IS the right primary limiter —
unlike the Track biome.
"""
from __future__ import annotations

import numpy as np
from .base import Biome, BiomeResult


class Endurance(Biome):
    name = "endurance"
    unit = "m/s(sustain)"

    def evaluate(self, agent) -> BiomeResult:
        g, p = agent.genotype, agent.phenotype
        skill = p.skill.get("endurance", 0.8)
        mass = agent.body_mass

        vo2_realized = g["vo2max_ceiling"] * p.aerobic_fitness     # ml/kg/min
        vo2_net = max(vo2_realized - 7.0, 1.0)                     # minus resting

        # running economy: O2 cost per metre (ml/kg/m). lower is better.
        economy = 0.18 * (1.0 + 0.015 * (mass - 70.0)) \
            * (1.0 + 0.25 * g["fast_fiber_frac"]) \
            * (1.0 - 0.15 * g["tendon_elastic_ret"])
        economy = max(economy, 0.10)

        v_at_vo2max = vo2_net / 60.0 / economy                    # m/s
        # sustainable fraction (critical velocity) improves with aerobic fitness
        cv_frac = 0.80 + 0.08 * p.aerobic_fitness
        v_sustain = v_at_vo2max * cv_frac * (0.9 + 0.1 * skill)

        # repetitive-load injury is low in a single eval; durability trial scales it
        injury = 0.2 * p.tissue_damage + max(0.0, p.recovery_debt - 0.5)

        return BiomeResult(
            self.name, score=float(v_sustain), unit=self.unit, injury=float(injury),
            detail={"vo2_realized": float(vo2_realized), "economy_ml_kg_m": float(economy),
                    "v_at_vo2max": float(v_at_vo2max)},
        )
