#!/usr/bin/env python3
"""Export the declared first-harmonic hover kinematics and derived scales."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

F_HZ = 26.1
R_M = 0.0483
C_MEAN_M = 0.0183
NU_M2_S = 1.5e-5
RHO_KG_M3 = 1.23
PHI_AMP_RAD = 1.0  # half of the reported 2 rad peak-to-peak excursion
ALPHA_AMP_RAD = 0.87  # provisional digitization of Nakata & Liu Fig. 1(d)
THETA_RAD = 0.0


def sample(phase: float) -> tuple[float, float, float]:
    """Return phi, alpha, theta at phase in cycles using equation 2.1."""
    angle = 2.0 * math.pi * phase
    return (
        PHI_AMP_RAD * math.cos(angle),
        ALPHA_AMP_RAD * math.sin(angle),
        THETA_RAD,
    )


def scales() -> dict[str, float]:
    # Use the same U=2 Phi f R convention stated by the primary paper, where
    # Phi=2 rad is the total stroke excursion. This yields Re~6.15e3 from the
    # rounded published inputs, close to the paper's approximate 6.3e3.
    stroke_excursion_rad = 2.0
    u_ref = 2.0 * stroke_excursion_rad * F_HZ * R_M
    return {
        "period_s": 1.0 / F_HZ,
        "reference_velocity_m_s": u_ref,
        "reynolds_number": u_ref * C_MEAN_M / NU_M2_S,
        "reduced_frequency_k_c_over_2R": math.pi * F_HZ * C_MEAN_M / u_ref,
        "wingtip_radius_m": R_M,
        "mean_chord_m": C_MEAN_M,
        "air_density_kg_m3": RHO_KG_M3,
        "kinematic_viscosity_m2_s": NU_M2_S,
        "phi_amplitude_rad": PHI_AMP_RAD,
        "alpha_amplitude_rad": ALPHA_AMP_RAD,
        "alpha_amplitude_source": "provisional raster digitization; see provenance.yaml",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--samples", type=int, default=257, help="Samples over one cycle, including both endpoints")
    args = parser.parse_args()
    if args.samples < 9:
        parser.error("--samples must be at least 9")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "kinematics_phase.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["phase_cycles", "time_s", "phi_rad", "alpha_rad", "theta_rad"])
        for i in range(args.samples):
            phase = i / (args.samples - 1)
            phi, alpha, theta = sample(phase)
            writer.writerow([phase, phase / F_HZ, phi, alpha, theta])
    (args.output / "derived_scales.json").write_text(json.dumps(scales(), indent=2) + "\n")


if __name__ == "__main__":
    main()
