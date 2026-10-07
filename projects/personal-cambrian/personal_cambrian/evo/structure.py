"""STRUCTURAL-FAILURE CHECK (ROADMAP Stage 6.3): reject bodies that would
instantly fail under their own predicted peak load.

A muscle can pull with `Fmax = PCSA x specific_tension`. That force is borne by
the bones it anchors to and by its own tendon. If a tiny limb anchors a giant
muscle, the bone stress (force / load-bearing cross-section) exceeds what the
tissue can carry and the attachment tears out the instant the muscle fires --
no point simulating it.

  * bone / attachment stress -- peak anchored force / minimum transverse
    cross-section of the part, vs a density-scaled tissue strength (denser
    tissue = stronger, Carter-Hayes sigma ~ rho^2). This is the discriminating
    check: it catches under-built attachments carrying huge muscles.
  * tendon stress -- Fmax / a nominal tendon cross-section, vs tendon ultimate
    tensile stress. Catches muscles too strong for any plausible tendon.

[APPROX] Strengths and the nominal tendon size are order-of-magnitude
physiological. Peak per-part load is summed over anchored muscles (conservative
"worst instant" assumption). Stresses are reported per-part / per-muscle so
Stage 6.5 can both reject and explain.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..encoding.genome import Genome, Shape
from ..morphogenesis import develop
from ..morphogenesis.develop import Morphology

# --- material constants [APPROX] -------------------------------------------
MPA = 1.0e6
SIGMA_BONE_MPA = 150.0          # ultimate stress of dense (cortical) tissue
REF_DENSITY = 1900.0            # kg/m^3 reference for that strength (~cortical bone)
SIGMA_TENDON_MPA = 100.0        # tendon ultimate tensile stress
NOMINAL_TENDON_RADIUS = 0.003   # m: tendons modeled at a nominal ~6 mm cross-section


def _min_cross_section(shape: Shape, dims: dict) -> float:
    """Smallest transverse cross-sectional area (m^2) -- bones fail at the thinnest."""
    if shape in (Shape.CAPSULE, Shape.SPHERE):
        return math.pi * dims["radius"] ** 2
    if shape is Shape.BOX:               # half-extents -> full face areas
        x, y, z = dims["x"], dims["y"], dims["z"]
        return 4.0 * min(x * y, x * z, y * z)
    if shape is Shape.ELLIPSOID:
        a, b, c = dims["x"], dims["y"], dims["z"]
        return math.pi * min(a * b, a * c, b * c)
    return 0.0


def _bone_capacity_mpa(density: float) -> float:
    """Tissue strength scales with apparent density squared (Carter-Hayes)."""
    return SIGMA_BONE_MPA * (density / REF_DENSITY) ** 2


@dataclass
class StructuralReport:
    """Predicted peak-load soundness of a body."""
    sound: bool
    failures: list = field(default_factory=list)
    n_bone_overloads: int = 0
    n_tendon_overloads: int = 0
    max_bone_stress_ratio: float = 0.0      # stress / capacity, worst part
    max_tendon_stress_ratio: float = 0.0    # stress / capacity, worst muscle
    per_part: dict = field(default_factory=dict)
    per_muscle: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.sound

    def to_dict(self) -> dict:
        return {
            "sound": self.sound, "failures": list(self.failures),
            "n_bone_overloads": self.n_bone_overloads,
            "n_tendon_overloads": self.n_tendon_overloads,
            "max_bone_stress_ratio": self.max_bone_stress_ratio,
            "max_tendon_stress_ratio": self.max_tendon_stress_ratio,
        }


def _structure_from_morphology(morph: Morphology) -> StructuralReport:
    report = StructuralReport(sound=True)
    tendon_csa = math.pi * NOMINAL_TENDON_RADIUS ** 2

    # accumulate peak anchored force on each part (every wrap body bears the force)
    part_load: dict[str, float] = {}
    for route in morph.muscles:
        # tendon stress for this muscle
        t_stress = (route.fmax / tendon_csa) / MPA
        t_cap = SIGMA_TENDON_MPA
        t_ratio = t_stress / t_cap
        report.per_muscle[route.id] = {
            "gene_id": route.gene_id, "fmax_N": route.fmax,
            "tendon_stress_mpa": t_stress, "tendon_capacity_mpa": t_cap,
            "ratio": t_ratio,
        }
        report.max_tendon_stress_ratio = max(report.max_tendon_stress_ratio, t_ratio)
        if t_ratio > 1.0:
            report.n_tendon_overloads += 1
        for wp in route.waypoints:
            part_load[wp.body_id] = part_load.get(wp.body_id, 0.0) + route.fmax

    # bone / attachment stress for each loaded part
    for bid, load in part_load.items():
        b = morph.get_body(bid)
        csa = _min_cross_section(b.shape, b.dims)
        stress = (load / max(csa, 1e-12)) / MPA
        cap = _bone_capacity_mpa(b.density)
        ratio = stress / max(cap, 1e-12)
        report.per_part[bid] = {
            "part_id": b.part_id, "load_N": load, "csa_m2": csa,
            "bone_stress_mpa": stress, "bone_capacity_mpa": cap, "ratio": ratio,
        }
        report.max_bone_stress_ratio = max(report.max_bone_stress_ratio, ratio)
        if ratio > 1.0:
            report.n_bone_overloads += 1

    if report.n_bone_overloads > 0:
        report.failures.append("bone_overload")
    if report.n_tendon_overloads > 0:
        report.failures.append("tendon_overload")
    report.sound = not report.failures
    return report


def check_structure(genome: Genome) -> StructuralReport:
    """Reject bodies that would instantly fail under predicted peak muscle load.

    Returns a `StructuralReport`; `bool(report)` is True iff the body is sound.
    """
    try:
        morph = develop(genome)
    except Exception as e:  # noqa: BLE001
        return StructuralReport(sound=False, failures=[f"develop_error: {type(e).__name__}"])
    return _structure_from_morphology(morph)
