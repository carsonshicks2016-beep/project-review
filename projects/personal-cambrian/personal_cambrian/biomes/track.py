"""TRACK — acceleration and maximum velocity.

[APPROX] Sprint uses a Samozino/Morin-style force-velocity-power view:
    F0  max horizontal force per kg  (limited by strength relative to mass)
    V0  max theoretical velocity     (limited by contraction velocity:
        fast-fiber fraction, fascicle/tendon geometry, leg length)
    v_max ≈ V0 * efficiency;  accel time-constant from F0.
VO2max is deliberately NOT a limiter of v_max here — it limits repeatability,
not top speed. Hamstring strain risk rises with v_max and relative weakness.
"""
from __future__ import annotations

import numpy as np
from .base import Biome, BiomeResult


class Track(Biome):
    name = "track"
    unit = "m/s(vmax)"

    def evaluate(self, agent) -> BiomeResult:
        g, p = agent.genotype, agent.phenotype
        skill = p.skill.get("track", 0.8)
        mass = agent.body_mass

        # F0: horizontal force per kg from lower-body strength relative to mass
        strength_force = g["pcsa_capacity"] * p.strength_coeff * (0.5 + 0.5 * p.anaerobic_fitness)
        F0 = (strength_force / mass) * 0.012                       # N/kg-ish [APPROX]

        # V0: contraction velocity * leg length * fiber/tendon geometry
        leg = g["height_m"] * g["leg_length_frac"]
        fiber_speed = 0.6 + 0.8 * g["fast_fiber_frac"]
        tendon_geom = 0.8 + 0.6 * g["tendon_length_frac"]
        V0 = 10.5 * fiber_speed * tendon_geom * (leg / 0.9)        # m/s [APPROX]

        v_max = V0 * (0.62 + 0.30 * skill) * (0.85 + 0.15 * np.tanh(F0))
        accel = F0 * 3.0                                            # accel index

        # hamstring strain risk ~ high speed with low relative strength
        rel_strength = (strength_force / mass)
        injury = max(0.0, (v_max / 11.0) * (1.3 - np.tanh(rel_strength)) * (0.7 + g["fast_fiber_frac"]) - 0.4)

        return BiomeResult(
            self.name, score=float(v_max), unit=self.unit, injury=float(injury),
            detail={"F0_N_per_kg": float(F0), "V0_m_s": float(V0),
                    "accel_index": float(accel)},
        )
