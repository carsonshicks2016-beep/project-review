"""
Unit tests for Clohessy-Wiltshire dynamics simulation.
"""

import unittest
import numpy as np
from satellite_rl.dynamics import CWDynamics, OrbitParams, SpacecraftParams, G0


class TestCWDynamics(unittest.TestCase):

    def setUp(self):
        self.orbit = OrbitParams(altitude_m=400_000.0)
        self.sc = SpacecraftParams(dry_mass_kg=450.0, propellant_mass_kg=50.0, max_thrust_n=10.0, isp_s=220.0)
        self.dyn = CWDynamics(self.orbit, self.sc)

    def test_rk4_matches_analytical_stm(self):
        """Zero thrust RK4 integration should match analytical CW State Transition Matrix."""
        # Initial relative state: 100m radial, 200m along-track, 50m cross-track, small velocities
        s0 = np.array([100.0, 200.0, 50.0, 0.05, -0.1, 0.02], dtype=np.float64)
        zero_thrust = np.zeros(3, dtype=np.float64)

        dt = 1.0  # 1 second step
        steps = 300  # 5 minutes
        s_rk4 = s0.copy()
        fuel = self.sc.propellant_mass_kg

        for _ in range(steps):
            s_rk4, fuel, dv = self.dyn.step_rk4(s_rk4, zero_thrust, fuel, dt)

        # Compare with analytical STM for total duration T = 300s
        phi = self.dyn.state_transition_matrix(dt * steps)
        s_stm = phi @ s0

        # Positional error after 300s should be less than 1 mm
        pos_error = np.linalg.norm(s_rk4[:3] - s_stm[:3])
        self.assertLess(pos_error, 1e-3, f"Position discrepancy {pos_error} m is too large")

    def test_periodic_orbit_drift_cancellation(self):
        """
        When vy0 = -2 * omega * x0, there is zero secular drift along V-bar (closed relative ellipse).
        """
        x0 = 50.0
        vy0 = -2.0 * self.dyn.omega * x0
        s0 = np.array([x0, 0.0, 0.0, 0.0, vy0, 0.0], dtype=np.float64)

        # After one full orbital period, state should return to s0
        T = self.orbit.period
        phi = self.dyn.state_transition_matrix(T)
        s_end = phi @ s0

        diff = np.linalg.norm(s_end - s0)
        self.assertLess(diff, 1e-4, f"Periodic orbit drift difference {diff} m is too large")

    def test_thrust_and_fuel_consumption(self):
        """Thrust burns fuel according to rocket equation / mass flow dm = F / (Isp * g0) * dt."""
        s0 = np.zeros(6, dtype=np.float64)
        thrust = np.array([10.0, 0.0, 0.0], dtype=np.float64)  # 10 N in X
        fuel0 = 10.0  # kg
        dt = 5.0  # seconds

        s1, fuel1, dv = self.dyn.step_rk4(s0, thrust, fuel0, dt)

        expected_dm = (10.0 / (self.sc.isp_s * G0)) * dt
        actual_dm = fuel0 - fuel1
        self.assertAlmostEqual(actual_dm, expected_dm, places=5)
        self.assertGreater(s1[3], 0.0)  # vx should increase


if __name__ == "__main__":
    unittest.main()
