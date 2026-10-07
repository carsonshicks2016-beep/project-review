"""Code-path checks for input preparation and acceptance-gate calculations.

These tests use synthetic histories and do not verify Navier–Stokes physics.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest

import numpy as np

TOOLS = Path(__file__).parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

import kinematics  # noqa: E402
import generate_geometry  # noqa: E402
import summarize_momentum_budget  # noqa: E402
import summarize_stage_a  # noqa: E402
import summarize_diagnosis  # noqa: E402
import summarize_force_isolation  # noqa: E402

spec = importlib.util.spec_from_file_location("force_gates", TOOLS / "force_gates.py")
force_gates = importlib.util.module_from_spec(spec)
assert spec and spec.loader
sys.modules[spec.name] = force_gates
spec.loader.exec_module(force_gates)


def write_history(path: Path, varying_amplitude: bool) -> None:
    frequency = 26.1
    period = 1 / frequency
    times = np.linspace(0, 4 * period, 2049)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "Fx_N", "Fy_N", "Fz_N", "power_W"])
        for t in times:
            cycle = min(3, int(t / period))
            amp = 1.0 + 0.1 * cycle if varying_amplitude else 1.0
            phase = 2 * math.pi * frequency * t
            writer.writerow([t, 0.002 * amp * math.cos(phase), 0.0, -0.03 + 0.01 * amp * math.sin(phase), 0.004 + 0.001 * math.sin(phase)])


class KinematicsTests(unittest.TestCase):
    def test_declared_harmonic_phase(self):
        self.assertEqual(kinematics.sample(0.0), (1.0, 0.0, 0.0))
        phi, alpha, theta = kinematics.sample(0.25)
        self.assertAlmostEqual(phi, 0.0, places=14)
        self.assertAlmostEqual(alpha, 0.87)
        self.assertEqual(theta, 0.0)

    def test_reference_scales_reproduce_rounded_source_values(self):
        derived = kinematics.scales()
        self.assertAlmostEqual(derived["reynolds_number"], 6151.8744, places=3)
        self.assertAlmostEqual(derived["reduced_frequency_k_c_over_2R"], 0.3, delta=0.01)


class GeometryTests(unittest.TestCase):
    def test_outline_has_expected_spanwise_area_scaling_and_stable_points(self):
        outline_path = TOOLS.parent / "reference_data" / "wing_outline.csv"
        outline = generate_geometry.read_outline(outline_path)
        points_a, mean_shape_a = generate_geometry.sample_wing(outline, 0.00075)
        points_b, mean_shape_b = generate_geometry.sample_wing(outline, 0.00075)
        self.assertEqual(points_a, points_b)
        self.assertEqual(mean_shape_a, mean_shape_b)
        self.assertEqual(len(points_a), 1673)
        self.assertGreater(mean_shape_a, 0.0)
        self.assertLess(mean_shape_a, 2.0)


class ForceGateTests(unittest.TestCase):
    def test_periodic_synthetic_history_passes_code_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "periodic.csv"
            write_history(path, varying_amplitude=False)
            result = force_gates.periodicity(path, 26.1, None, 256)
            self.assertEqual(result["gate"], "PASS")

    def test_nonperiodic_synthetic_history_fails_code_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "nonperiodic.csv"
            write_history(path, varying_amplitude=True)
            result = force_gates.periodicity(path, 26.1, None, 256)
            self.assertEqual(result["gate"], "FAIL")


class VerificationPreparationTests(unittest.TestCase):
    def test_stationary_matrix_is_immutable_and_uses_composite_cv_boxes(self):
        with tempfile.TemporaryDirectory() as td:
            output_root = Path(td) / "stage_a"
            subprocess.run(
                [sys.executable, str(TOOLS / "prepare_verification.py"), "--case", "stationary",
                 "--output-root", str(output_root)],
                check=True,
                capture_output=True,
                text=True,
            )
            with (output_root / "prepared_matrix.csv").open(newline="") as f:
                matrix = list(csv.DictReader(f))
            self.assertEqual(len(matrix), 6)
            case = output_root / matrix[0]["run_id"]
            deck = (case / "input3d").read_text()
            self.assertIn('MOTION_MODE = "stationary"', deck)
            self.assertIn("ENABLE_VERIFICATION_DIAGNOSTICS = TRUE", deck)
            self.assertIn("TARGET_FORCE_TIME_FRACTION = 0.5", deck)
            self.assertIn("lower_left_corner = -0.0966,0,-0.3864", deck)
            self.assertIn("upper_right_corner = 0.0966,0.0966,0.3864", deck)
            manifest = json.loads((case / "run_manifest.json").read_text())
            self.assertEqual(manifest["kernel"], "IB_4")
            self.assertEqual(manifest["target_force_time_fraction"], 0.5)
            self.assertEqual(manifest["launch_status"], "prepared_not_run")
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [sys.executable, str(TOOLS / "prepare_verification.py"), "--case", "stationary",
                     "--output-root", str(output_root)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_verification_stiffness_is_explicit_and_hashed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "stiffness-study"
            subprocess.run(
                [sys.executable, str(TOOLS / "prepare_verification.py"), "--case", "matched_translation",
                 "--output-root", str(output_root), "--duration-s", "1e-6", "--target-stiffness", "75"],
                check=True,
                capture_output=True,
                text=True,
            )
            case = output_root / "matched_translation_coarse_dt_coarse"
            manifest = json.loads((case / "run_manifest.json").read_text())
            self.assertEqual(manifest["target_stiffness_N_m_per_point"], 75.0)
            self.assertEqual(manifest["target_damping_kg_s_per_point"], 0.0)
            self.assertEqual(manifest["input_sha256"], generate_geometry.sha256(case / "input3d"))
            self.assertIn("TARGET_STIFFNESS = 75", (case / "input3d").read_text())
            self.assertIn("TARGET_DAMPING = 0", (case / "input3d").read_text())

    def test_target_damping_is_explicit_for_fixed_and_moving_target_cases(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "damping-study"
            subprocess.run(
                [sys.executable, str(TOOLS / "prepare_verification.py"), "--case", "stationary_uniform",
                 "--spatial", "coarse", "--timestep", "dt_coarse", "--duration-s", "1e-6",
                 "--target-stiffness", "1", "--target-damping", "0.0002002563909327",
                 "--output-root", str(output_root)],
                check=True, capture_output=True, text=True,
            )
            case = output_root / "stationary_uniform_coarse_dt_coarse"
            manifest = json.loads((case / "run_manifest.json").read_text())
            self.assertAlmostEqual(manifest["target_damping_kg_s_per_point"], 0.0002002563909327)
            self.assertIn("uniform_target_damping = TARGET_DAMPING", (case / "input3d").read_text())
            moving_root = Path(temporary_directory) / "moving-relative-damping"
            subprocess.run(
                [sys.executable, str(TOOLS / "prepare_verification.py"), "--case", "matched_translation",
                 "--spatial", "coarse", "--timestep", "dt_coarse", "--duration-s", "1e-6",
                 "--target-stiffness", "1", "--target-damping", "0.0002",
                 "--output-root", str(moving_root)],
                check=True, capture_output=True, text=True,
            )
            moving = moving_root / "matched_translation_coarse_dt_coarse"
            moving_manifest = json.loads((moving / "run_manifest.json").read_text())
            self.assertEqual(moving_manifest["target_damping_kg_s_per_point"], 0.0002)
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [sys.executable, str(TOOLS / "prepare_verification.py"), "--case", "stationary_uniform",
                     "--spatial", "coarse", "--timestep", "dt_coarse", "--duration-s", "1e-6",
                     "--target-stiffness", "0", "--target-damping", "0.0002",
                     "--output-root", str(Path(temporary_directory) / "invalid-zero-spring")],
                    check=True, capture_output=True, text=True,
                )

    def test_relative_target_damping_uses_reference_velocity_shift(self):
        source = (TOOLS.parent / "src" / "main.cpp").read_text()
        self.assertIn("prescribed_target_velocity_affine", source)
        self.assertIn("shift_target_references_for_relative_damping", source)
        self.assertIn("eta*(U_fluid-U_target)", source)

    def test_force_isolation_control_disables_target_and_standard_forcing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "force-isolation"
            subprocess.run(
                [sys.executable, str(TOOLS / "prepare_force_isolation.py"),
                 "--output-root", str(output_root), "--duration-s", "1e-6"],
                check=True, capture_output=True, text=True,
            )
            case = output_root / "prepared" / "stationary_uniform_coarse_dt_coarse"
            deck = (case / "input3d").read_text()
            manifest = json.loads((case / "run_manifest.json").read_text())
            self.assertIn("ENABLE_IB_FORCING = FALSE", deck)
            self.assertIn("ENABLE_FLOW_STATE_DIAGNOSTICS = TRUE", deck)
            self.assertIn("TARGET_STIFFNESS = 0", deck)
            self.assertEqual(manifest["verification_case"], "stationary_uniform_force_disabled")
            self.assertFalse(manifest["ib_forcing_enabled"])
            self.assertEqual(manifest["input_sha256"], generate_geometry.sha256(case / "input3d"))

    def test_force_isolation_can_disable_fine_amr_levels_and_records_geometry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "single-level-force-isolation"
            subprocess.run(
                [sys.executable, str(TOOLS / "prepare_force_isolation.py"),
                 "--output-root", str(output_root), "--duration-s", "1e-6", "--amr-levels", "1"],
                check=True, capture_output=True, text=True,
            )
            case = output_root / "prepared" / "stationary_uniform_coarse_dt_coarse"
            deck = (case / "input3d").read_text()
            manifest = json.loads((case / "run_manifest.json").read_text())
            study = json.loads((output_root / "force_isolation_manifest.json").read_text())
            self.assertIn("MAX_LEVELS = 1", deck)
            self.assertEqual(manifest["max_levels"], 1)
            self.assertEqual(manifest["nominal_finest_dx_m"], 0.0483 / 16)
            self.assertIn("dense Lagrangian geometry retained", manifest["amr_note"])
            self.assertEqual(manifest["input_sha256"], generate_geometry.sha256(case / "input3d"))
            self.assertEqual(study["amr_levels"], 1)
            self.assertIn("MAX_LEVELS=1", study["single_intended_change"])

    def test_diagnosis_study_prepares_hashed_factor_controls_without_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory) / "diagnosis"
            subprocess.run(
                [sys.executable, str(TOOLS / "prepare_diagnosis.py"), "--output-root", str(output_root),
                 "--duration-s", "1e-6"],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output_root / "diagnosis_manifest.json").read_text())
            self.assertEqual(manifest["status"], "PREPARED_NOT_RUN")
            self.assertEqual([item["name"] for item in manifest["scenario_inputs"]],
                             ["stationary", "zero_amplitude", "stationary_uniform", "matched_translation"])
            self.assertEqual(manifest["grid"], "coarse Stage A hierarchy only; no convergence claim")
            self.assertTrue((output_root / "diagnosis_manifest.sha256").is_file())
            summary = summarize_diagnosis.summarize(output_root)
            self.assertEqual(summary["status"], "DIAGNOSTIC_ONLY")
            self.assertEqual(summary["cases_completed"], 0)
            self.assertEqual(summary["stage_a_gate"], "FAILED / UNVALIDATED")
            moving = output_root / "matched_translation" / "matched_translation_coarse_dt_coarse"
            moving_manifest = json.loads((moving / "run_manifest.json").read_text())
            self.assertEqual(moving_manifest["input_sha256"], generate_geometry.sha256(moving / "input3d"))
            with self.assertRaises(subprocess.CalledProcessError):
                subprocess.run(
                    [sys.executable, str(TOOLS / "prepare_diagnosis.py"), "--output-root", str(output_root),
                     "--duration-s", "1e-6"],
                    check=True,
                    capture_output=True,
                    text=True,
                )


class MomentumBudgetTests(unittest.TestCase):
    def test_outer_boundary_terms_are_separate_and_refinement_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "momentum_budget.csv"
            path.write_text(
                "time_s,dt_s,dP_fluid_dt_x_N,dP_fluid_dt_y_N,dP_fluid_dt_z_N,"
                "force_on_body_x_N,force_on_body_y_N,force_on_body_z_N,"
                "pressure_traction_x_N,pressure_traction_y_N,pressure_traction_z_N,"
                "advective_momentum_flux_x_N,advective_momentum_flux_y_N,advective_momentum_flux_z_N,"
                "viscous_traction_x_N,viscous_traction_y_N,viscous_traction_z_N,"
                "signed_residual_x_N,signed_residual_y_N,signed_residual_z_N,normalized_residual,refined_physical_boundary\n"
                "1,0.1,-3,0,0,5,0,0,1,0,0,1,0,0,1,0,0,5,0,0,0,0\n"
                "2,0.1,-3,0,0,5,0,0,1,0,0,1,0,0,1,0,0,5,0,0,0,1\n"
            )
            report = summarize_momentum_budget.summarize(path)
            step = report["steps"][0]
            self.assertEqual(step["dP_fluid_dt_N"], [-3.0, 0.0, 0.0])
            self.assertEqual(step["hydrodynamic_force_on_body_N"], [5.0, 0.0, 0.0])
            self.assertEqual(step["pressure_traction_N"], [1.0, 0.0, 0.0])
            self.assertEqual(step["advective_momentum_flux_N"], [1.0, 0.0, 0.0])
            self.assertEqual(step["viscous_traction_N"], [1.0, 0.0, 0.0])
            self.assertTrue(report["outer_terms_independently_integrated"])
            self.assertEqual(report["gate"], "MEASURED_DIAGNOSTIC_WITH_AMR_BOUNDARY")
            self.assertFalse(report["steps"][0]["outer_boundary_refined"])
            self.assertTrue(report["steps"][1]["outer_boundary_refined"])


class ForceIsolationSummaryTests(unittest.TestCase):
    def test_force_persistence_is_detected_with_force_vector_disabled(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            case = root / "prepared" / "stationary_uniform_coarse_dt_coarse"
            baseline = root / "baseline"
            case.mkdir(parents=True)
            baseline.mkdir()

            force_fields = ["time_s", "dt_s", "target_error_rms_m", "target_error_max_m",
                            "no_slip_velocity_rms_over_Uref", "no_slip_velocity_max_over_Uref",
                            "divergence_rms_s_inv", "divergence_max_s_inv",
                            "divergence_surface_band_rms_s_inv", "divergence_surface_band_max_s_inv",
                            "Fx_N", "Fy_N", "Fz_N"]
            budget_fields = ["time_s", "dt_s", "normalized_residual"] + [
                f"{prefix}_{axis}_N" for prefix in (
                    "dP_fluid_dt", "force_on_body", "pressure_traction",
                    "advective_momentum_flux", "viscous_traction") for axis in "xyz"
            ]
            for directory, fx in ((case, 99.5), (baseline, 100.0)):
                values = {key: 0.0 for key in force_fields}
                values.update({"time_s": 1e-6, "dt_s": 1e-6, "Fx_N": fx})
                with (directory / "force_history.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=force_fields)
                    writer.writeheader()
                    writer.writerow(values)
            budget = {key: 0.0 for key in budget_fields}
            budget["normalized_residual"] = 1e-6
            with (case / "momentum_budget.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=budget_fields)
                writer.writeheader()
                writer.writerow(budget)
            (case / "resource_record.json").write_text(json.dumps({"status": "COMPLETED"}))
            for name in ("solver.stdout.log", "solver.stderr.log", "hawkmoth_hover.log"):
                (case / name).write_text("stokes solve residual norm = 0.1\n")
            result = summarize_force_isolation.summarize(root, baseline)
            self.assertTrue(result["force_persists_without_ib_forcing"])
            self.assertAlmostEqual(result["relative_x_force_change"], 0.005)
            self.assertEqual(result["stage_a_gate"], "FAILED / UNVALIDATED")

    def test_legacy_history_is_incomplete_instead_of_false_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory)
            (run_dir / "run_manifest.json").write_text(json.dumps({"run_id": "legacy", "nominal_finest_dx_m": 0.1}))
            (run_dir / "resource_record.json").write_text(json.dumps({"status": "COMPLETED"}))
            (run_dir / "force_history.csv").write_text(
                "time_s,dt_s,target_error_rms_m,target_error_max_m,no_slip_velocity_rms_over_Uref,"
                "no_slip_velocity_max_over_Uref,divergence_rms_s_inv,divergence_max_s_inv,power_W,Fx_N,Fy_N,Fz_N\n"
                "1,0.1,0,0,0,0,0,0,0,0,0,0\n"
            )
            self.assertEqual(summarize_stage_a.run_summary(run_dir)["gate"], "INCOMPLETE_FIELDS")


if __name__ == "__main__":
    unittest.main()
