"""PHENOTYPE: trainable, time-dependent state, plus an evolvable training program.

We collapse a longitudinal training block into a closed-form saturating
adaptation model so the prototype stays cheap. The real engine replaces this
with a day-by-day adaptation loop (fitness-fatigue / damage-repair). The
training program itself is a small evolvable genome (TRAINING-PROGRAM EVOLUTION
in PLAN.md): focus weights + volume + intensity.

[APPROX] All curves below are physically-motivated saturating responses, not
fitted models. They are monotonic in the expected direction (validated in
tests/test_validation.py) but their coefficients are not calibrated to data.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .genome import Genotype


@dataclass
class TrainingProgram:
    """Evolvable developmental process. Focus weights are normalized internally."""
    strength: float = 0.25
    speed: float = 0.25
    power: float = 0.25
    endurance: float = 0.25
    volume: float = 0.6      # 0..1 overall training stress
    intensity: float = 0.6   # 0..1 average load relative to capacity

    def focus(self) -> dict[str, float]:
        w = np.array([self.strength, self.speed, self.power, self.endurance], float)
        w = np.clip(w, 0, None)
        s = w.sum() or 1.0
        w = w / s
        return {"strength": w[0], "speed": w[1], "power": w[2], "endurance": w[3]}

    def as_vector(self) -> np.ndarray:
        return np.array([self.strength, self.speed, self.power, self.endurance,
                         self.volume, self.intensity], float)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "TrainingProgram":
        return cls(*[float(x) for x in v])


@dataclass
class Phenotype:
    muscle_mass_kg: float          # trained contractile mass (lower-body-weighted)
    strength_coeff: float          # neural drive / voluntary activation (0..1)
    aerobic_fitness: float         # realized fraction of vo2max ceiling (0..1)
    anaerobic_fitness: float       # realized fraction of W' ceiling (0..1)
    body_fat_frac: float
    mobility: float                # 0..1
    tissue_damage: float           # accumulated, 0..1 (chronic)
    recovery_debt: float           # 0..1
    training_age: float            # years-equivalent of adaptation
    skill: dict[str, float] = field(default_factory=dict)  # per-biome technique 0..1


def _sat(x: float, k: float = 3.0) -> float:
    """Saturating 0..1 response with diminishing returns."""
    return 1.0 - np.exp(-k * max(0.0, x))


def adapt(g: Genotype, prog: TrainingProgram, weeks: float = 16.0) -> Phenotype:
    """Map (genotype, training program) -> trained phenotype.

    Hypertrophy, neural drive, aerobic and anaerobic capacity each saturate
    toward their genetic ceilings as a function of the matched training focus,
    volume, and time. Excess intensity*volume beyond connective capacity
    accumulates tissue damage and recovery debt (the cost of adaptation).
    """
    f = prog.focus()
    dose = prog.volume * (weeks / 16.0)               # normalized training dose
    rec = g["recovery_capacity"]

    # effective dose is gated by recovery capacity (overreaching wastes stimulus)
    eff = dose * (0.6 + 0.4 * np.tanh(rec))

    # --- hypertrophy: total skeletal muscle mass toward PCSA-implied ceiling -
    # [APPROX] anchored so a 1.78 m prior body carries ~30 kg muscle, trainable
    # roughly 0.85x-1.30x with matched stimulus; scales with height^2 and PCSA.
    pcsa_factor = g["pcsa_capacity"] / 520.0
    hyper_drive = (f["strength"] * 0.7 + f["power"] * 0.3)
    hyper = _sat(eff * (0.3 + hyper_drive))           # 0..1 hypertrophy progress
    muscle_mass = 30.0 * (g["height_m"] / 1.78) ** 2 \
        * pcsa_factor ** 0.6 * (0.85 + 0.45 * hyper)  # total skeletal muscle, kg

    # --- neural drive: trained voluntary activation ------------------------
    strength_coeff = 0.6 + 0.4 * _sat(eff * (f["strength"] + 0.5 * f["power"]) * g["neural_capacity"])

    # --- aerobic & anaerobic realized fractions ----------------------------
    aerobic = _sat(eff * (0.3 + f["endurance"]) * 1.4)
    anaerobic = _sat(eff * (0.3 + f["speed"] + f["power"]))

    # --- body composition: fat falls with volume, floored ------------------
    body_fat = float(np.clip(0.22 - 0.10 * _sat(dose), 0.06, 0.30))

    # --- mobility: improves with focused work, bounded by structure --------
    mobility = float(np.clip(0.5 + 0.4 * _sat(eff), 0.2, 0.98))

    # --- cost of adaptation: damage + recovery debt ------------------------
    overload = max(0.0, prog.intensity * prog.volume - 0.55 * g["bone_load_capacity"])
    tissue_damage = float(np.clip(overload * 1.5 / rec, 0.0, 1.0))
    recovery_debt = float(np.clip((prog.volume * prog.intensity) / rec - 0.4, 0.0, 1.0))

    training_age = weeks / 52.0

    # --- controller / skill per biome (motor learning) ---------------------
    # technique efficiency rises with matched focus, neural capacity, and time
    base = 0.78
    learn = 0.20 * _sat(training_age * 2.0) * np.tanh(g["neural_capacity"])
    skill = {
        "track":     base + learn * (0.4 + f["speed"]),
        "iron_zone": base + learn * (0.4 + f["strength"]),
        "ballistics":base + learn * (0.4 + f["power"]),
        "endurance": base + learn * (0.4 + f["endurance"]),
    }
    skill = {k: float(np.clip(v, 0.7, 1.0)) for k, v in skill.items()}

    return Phenotype(
        muscle_mass_kg=muscle_mass,
        strength_coeff=float(strength_coeff),
        aerobic_fitness=float(aerobic),
        anaerobic_fitness=float(anaerobic),
        body_fat_frac=body_fat,
        mobility=mobility,
        tissue_damage=tissue_damage,
        recovery_debt=recovery_debt,
        training_age=training_age,
        skill=skill,
    )
