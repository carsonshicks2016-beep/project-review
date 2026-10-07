"""BALLISTICS ARENA — explosive output (vertical jump as the prototype task).

[APPROX] Work-energy / impulse model:
    takeoff_v = sqrt(2 * (F_net/m) * push_distance) + elastic SSC contribution
    jump_height = takeoff_v^2 / (2 g)
Elastic contribution rewards tendon stiffness * elastic return (stretch-
shortening cycle). Patellar-tendon load drives injury risk.
"""
from __future__ import annotations

import numpy as np
from .base import Biome, BiomeResult

G = 9.81
BALL_K = 240.0  # calibrated so a balanced Agent Zero jumps ~0.40-0.45 m


class Ballistics(Biome):
    name = "ballistics"
    unit = "m(jump)"

    def evaluate(self, agent) -> BiomeResult:
        g, p = agent.genotype, agent.phenotype
        skill = p.skill.get("ballistics", 0.8)
        mass = agent.body_mass

        # net upward force = lower-body force - bodyweight
        leg_force = BALL_K * p.muscle_mass_kg ** 0.7 * p.strength_coeff * \
            (0.6 + 0.4 * g["fast_fiber_frac"]) * skill              # N [APPROX]
        f_net = max(leg_force - mass * G, 0.0)

        push = g["height_m"] * g["leg_length_frac"] * 0.45 * (0.7 + 0.3 * p.mobility)  # m
        v_concentric = np.sqrt(2.0 * (f_net / mass) * push)

        # stretch-shortening elastic bonus
        elastic = g["tendon_elastic_ret"] * np.tanh(g["tendon_stiffness"] / 250.0)
        v_takeoff = v_concentric * (1.0 + 0.25 * elastic)

        height = v_takeoff ** 2 / (2 * G)

        # patellar tendon load vs capacity
        tendon_stress = (leg_force / 2200.0) / (g["bone_load_capacity"] * (0.5 + g["tendon_stiffness"] / 400.0))
        injury = max(0.0, tendon_stress - 0.7) + 0.3 * p.tissue_damage

        return BiomeResult(
            self.name, score=float(height), unit=self.unit, injury=float(injury),
            detail={"v_takeoff_m_s": float(v_takeoff), "f_net_N": float(f_net),
                    "elastic_bonus": float(elastic)},
        )
