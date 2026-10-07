#!/usr/bin/env python3
"""Prepare immutable Stage A immersed-boundary verification runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

from generate_geometry import read_outline, sample_wing, write_vertex, sha256, pose

ROOT = Path(__file__).parents[1]
TEMPLATE = ROOT / "case" / "ibamr" / "input3d"
OUTLINE = ROOT / "reference_data" / "wing_outline.csv"

CASES = {
    "stationary": {"mode": "stationary", "velocity": (0.0, 0.0, 0.0), "duration": 0.00005},
    # Same uniform initial flow as matched translation, but the wing remains
    # fixed. This isolates relative motion from the moving-target update.
    "stationary_uniform": {"mode": "stationary", "velocity": (10.0, 0.0, 0.0), "duration": 0.0004},
    # The fluid and surface translate together. Uniform velocity is an exact
    # zero-relative-flow solution and moves the immersed mesh across AMR boxes.
    "matched_translation": {"mode": "translation", "velocity": (10.0, 0.0, 0.0), "duration": 0.0004},
    "zero_amplitude": {"mode": "hover", "velocity": (0.0, 0.0, 0.0), "duration": 0.00005},
}

RESOLUTIONS = (("coarse", 3), ("medium", 4), ("fine", 5))
TIMESTEPS = (("dt_coarse", 8192), ("dt_fine", 16384))
TARGET_FORCE_TIME_FRACTION = 0.5


def substitute(text: str, pattern: str, value: str) -> str:
    result, count = re.subn(pattern, value, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected one input assignment matching {pattern!r}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "stage_a")
    parser.add_argument("--case", choices=(*CASES, "all"), default="all")
    parser.add_argument("--spatial", choices=tuple(item[0] for item in RESOLUTIONS) + ("all",), default="all")
    parser.add_argument("--timestep", choices=tuple(item[0] for item in TIMESTEPS) + ("all",), default="all")
    parser.add_argument("--duration-s", type=float, help="override duration for a one-step diagnostic smoke")
    parser.add_argument("--target-stiffness", type=float, default=15.0,
                        help="uniform IB target stiffness in N/m per Lagrangian point")
    parser.add_argument("--target-damping", type=float, default=0.0,
                        help="uniform IB target damping in kg/s per Lagrangian point; moving-target force references are corrected to use relative velocity")
    args = parser.parse_args()
    if args.duration_s is not None and args.duration_s <= 0.0:
        parser.error("--duration-s must be positive")
    if args.target_stiffness < 0.0:
        parser.error("--target-stiffness must be nonnegative")
    if args.target_damping < 0.0:
        parser.error("--target-damping must be nonnegative")
    if args.target_damping > 0.0 and args.target_stiffness == 0.0:
        parser.error("nonzero target damping requires positive target stiffness for the relative-velocity reference shift")
    selected = list(CASES) if args.case == "all" else [args.case]
    selected_resolutions = [item for item in RESOLUTIONS if args.spatial in ("all", item[0])]
    selected_timesteps = [item for item in TIMESTEPS if args.timestep in ("all", item[0])]
    outline = read_outline(OUTLINE)
    prepared = []
    for case_name in selected:
        motion = CASES[case_name]
        for spatial_name, levels in selected_resolutions:
            for time_name, steps_per_period in selected_timesteps:
                duration = args.duration_s if args.duration_s is not None else motion["duration"]
                run_id = f"{case_name}_{spatial_name}_{time_name}"
                run_dir = args.output_root / run_id
                if run_dir.exists():
                    raise FileExistsError(f"refusing to overwrite verification run: {run_dir}")
                run_dir.mkdir(parents=True)
                deck = TEMPLATE.read_text()
                assignments = (
                    (r"^MAX_LEVELS = \d+$", f"MAX_LEVELS = {levels}"),
                    (r"^STEPS_PER_PERIOD = \d+$", f"STEPS_PER_PERIOD = {steps_per_period}"),
                    (r"^KINEMATICS_SCALE = [^\n]+$", "KINEMATICS_SCALE = 0.0"),
                    (r'^MOTION_MODE = [^\n]+$', f'MOTION_MODE = "{motion["mode"]}"'),
                    (r"^TRANSLATION_VELOCITY = [^\n]+$", "TRANSLATION_VELOCITY = " + ",".join(map(str, motion["velocity"]))),
                    (r'^   function_0 = "[^\n]+"$', f'   function_0 = "{motion["velocity"][0]}"'),
                    (r'^   function_1 = "[^\n]+"$', f'   function_1 = "{motion["velocity"][1]}"'),
                    (r'^   function_2 = "[^\n]+"$', f'   function_2 = "{motion["velocity"][2]}"'),
                    (r"^ENABLE_VERIFICATION_DIAGNOSTICS = [^\n]+$", "ENABLE_VERIFICATION_DIAGNOSTICS = TRUE"),
                    (r"^ENABLE_IB_FORCING = [^\n]+$", "ENABLE_IB_FORCING = TRUE"),
                    (r"^ENABLE_FLOW_STATE_DIAGNOSTICS = [^\n]+$", "ENABLE_FLOW_STATE_DIAGNOSTICS = FALSE"),
                    (r"^TARGET_FORCE_TIME_FRACTION = [^\n]+$",
                     f"TARGET_FORCE_TIME_FRACTION = {TARGET_FORCE_TIME_FRACTION:.1f}"),
                    (r"^TARGET_STIFFNESS = [^\n]+$", f"TARGET_STIFFNESS = {args.target_stiffness:.12g}"),
                    (r"^TARGET_DAMPING = [^\n]+$", f"TARGET_DAMPING = {args.target_damping:.12g}"),
                    (r"^END_CYCLES = [^\n]+$", "END_CYCLES = 1"),
                    (r"^END_TIME = [^\n]+$", f"END_TIME = {duration:.12g}"),
                )
                for pattern, value in assignments:
                    deck = substitute(deck, pattern, value)
                # The two adjacent boxes tile the complete fluid domain. Their
                # internal-plane fluxes cancel when summed, leaving a global
                # momentum balance with only physical-domain boundary terms.
                half_xy = 2.0 * 0.0483
                half_z = 8.0 * 0.0483
                for sid, y_bounds in ((0, (0.0, half_xy)), (1, (-half_xy, 0.0))):
                    pattern = rf"ControlVolume_{sid} \{{.*?\}}"
                    value = (f"ControlVolume_{sid} {{\n"
                             f"   lower_left_corner = {-half_xy:.8g},{y_bounds[0]:.8g},{-half_z:.8g}\n"
                             f"   upper_right_corner = {half_xy:.8g},{y_bounds[1]:.8g},{half_z:.8g}\n"
                             "}")
                    deck, count = re.subn(pattern, value, deck, count=1, flags=re.DOTALL)
                    if count != 1:
                        raise ValueError(f"missing ControlVolume_{sid} input block")
                input_path = run_dir / "input3d"
                input_path.write_text(deck)
                dx = 0.0483 / (16 * 2 ** (levels - 1))
                points, _ = sample_wing(outline, dx)
                right = [pose(point, 1) for point in points]
                left = [pose((x, -y, z), -1) for x, y, z in points]
                write_vertex(run_dir / "right_wing.vertex", right)
                write_vertex(run_dir / "left_wing.vertex", left)
                manifest = {
                    "run_id": run_id,
                    "verification_case": case_name,
                    "motion_mode": motion["mode"],
                    "translation_velocity_m_s": motion["velocity"],
                    "duration_s": duration,
                    "spatial_level": spatial_name,
                    "max_levels": levels,
                    "timestep_level": time_name,
                    "dt_max_steps_per_period": steps_per_period,
                    "nominal_finest_dx_m": dx,
                    "target_stiffness_N_m_per_point": args.target_stiffness,
                    "target_damping_kg_s_per_point": args.target_damping,
                    "target_force_time_fraction": TARGET_FORCE_TIME_FRACTION,
                    "kernel": "IB_4",
                    "ibamr_version": "0.19.0",
                    "input_sha256": sha256(input_path),
                    "outline_sha256": sha256(OUTLINE),
                    "right_wing_sha256": sha256(run_dir / "right_wing.vertex"),
                    "left_wing_sha256": sha256(run_dir / "left_wing.vertex"),
                    "launch_status": "prepared_not_run",
                    "interpretation": "numerical verification only; not moth performance or validation",
                }
                (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                prepared.append({"run_id": run_id, "run_dir": str(run_dir), "case": case_name,
                                 "spatial": spatial_name, "time": time_name})
    matrix_path = args.output_root / "prepared_matrix.csv"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    with matrix_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("run_id", "run_dir", "case", "spatial", "time"))
        writer.writeheader()
        writer.writerows(prepared)
    print(json.dumps({"prepared": len(prepared), "matrix": str(matrix_path)}, indent=2))


if __name__ == "__main__":
    main()
