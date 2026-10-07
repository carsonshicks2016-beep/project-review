#!/usr/bin/env python3
"""Calibration harness: print Agent Zero + hand-built archetype bodies so the
analytic biome constants can be sanity-checked against plausible human ranges.

Target plausibility bands (untrained/recreational -> trained, [APPROX]):
    body mass   72-86 kg     squat 1RM   70-220 kg
    v_max       7-11 m/s     vert jump   0.35-0.80 m
    sustain v   2.4-5.0 m/s  budget overflow ~0 for viable bodies
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.genome import Genotype
from personal_cambrian.phenotype import TrainingProgram
from personal_cambrian.agent import Agent
from personal_cambrian.biomes import evaluate_all


def row(label, geno, prog):
    a = Agent(geno, prog)
    b = evaluate_all(a)
    s2m = b["iron_zone"].detail["strength_to_mass"]
    print(f"{label:16s} mass={a.body_mass:5.1f}kg muscle={a.phenotype.muscle_mass_kg:4.1f} "
          f"| squat={b['iron_zone'].score:5.1f} s2m={s2m:4.2f} "
          f"vmax={b['track'].score:4.1f} jump={b['ballistics'].score:4.2f} "
          f"sustain={b['endurance'].score:4.2f} "
          f"| inj({b['track'].injury:.2f},{b['iron_zone'].injury:.2f},"
          f"{b['ballistics'].injury:.2f}) budovr={a.budgets.total_overflow:.2f}")


def main():
    bal = TrainingProgram(0.25, 0.25, 0.25, 0.25, 0.6, 0.6)
    row("AgentZero", Genotype(), bal)
    row("Strength", Genotype({"pcsa_capacity": 720, "lever_advantage": 1.05,
                              "bone_load_capacity": 1.15}),
        TrainingProgram(1.0, 0.1, 0.3, 0.0, 0.8, 0.75))
    row("Sprint", Genotype({"fast_fiber_frac": 0.62, "tendon_length_frac": 0.34,
                            "tendon_elastic_ret": 0.45, "leg_length_frac": 0.50,
                            "height_m": 1.82}),
        TrainingProgram(0.3, 1.0, 0.6, 0.0, 0.7, 0.7))
    row("Jumper", Genotype({"tendon_elastic_ret": 0.45, "tendon_stiffness": 280,
                            "fast_fiber_frac": 0.60, "pcsa_capacity": 640}),
        TrainingProgram(0.4, 0.4, 1.0, 0.0, 0.7, 0.7))
    row("Endurance", Genotype({"vo2max_ceiling": 62, "fast_fiber_frac": 0.40,
                               "tendon_elastic_ret": 0.45}),
        TrainingProgram(0.1, 0.2, 0.1, 1.0, 0.85, 0.6))
    row("Monster(OE)", Genotype({"pcsa_capacity": 1600, "fast_fiber_frac": 0.95,
                                 "heat_dissipation": 50, "recovery_capacity": 0.5}),
        TrainingProgram(1.0, 0.5, 0.5, 0.0, 1.0, 1.0))


if __name__ == "__main__":
    main()
