"""
Unit tests for vehicle dynamics and Pacejka tire model.
"""

import unittest
import numpy as np
from drift_rl.dynamics import VehicleDynamics, VehicleParams, VehicleState


class TestVehicleDynamics(unittest.TestCase):

    def setUp(self):
        self.params = VehicleParams()
        self.dyn = VehicleDynamics(self.params)

    def test_straight_line_acceleration(self):
        """Applying positive throttle accelerates the car forward in vx."""
        state = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=2.0, vy=0.0, yaw_rate=0.0, steer=0.0)
        dt = 0.05
        # 10 steps of full throttle
        for _ in range(10):
            state = self.dyn.step_rk4(state, steer=0.0, throttle_brake=1.0, handbrake=0.0, dt=dt)

        self.assertGreater(state.vx, 2.0, "Car should accelerate forward")
        self.assertAlmostEqual(state.vy, 0.0, places=2, msg="Lateral velocity should stay near zero on straight")
        self.assertAlmostEqual(state.yaw, 0.0, places=2, msg="Heading should stay straight")

    def test_handbrake_induces_slip(self):
        """Applying handbrake while turning should increase slip angle and yaw rate."""
        # Car moving forward at 15 m/s
        state_no_hb = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=15.0, vy=0.0, yaw_rate=0.0, steer=0.0)
        state_hb = VehicleState(x=0.0, y=0.0, yaw=0.0, vx=15.0, vy=0.0, yaw_rate=0.0, steer=0.0)

        dt = 0.05
        # Turn with steer = 0.6
        for _ in range(15):
            state_no_hb = self.dyn.step_rk4(state_no_hb, steer=0.6, throttle_brake=0.5, handbrake=0.0, dt=dt)
            state_hb = self.dyn.step_rk4(state_hb, steer=0.6, throttle_brake=0.5, handbrake=1.0, dt=dt)

        # With handbrake, rear breaks loose, producing higher body slip angle
        self.assertGreater(
            abs(state_hb.slip_angle_deg),
            abs(state_no_hb.slip_angle_deg),
            "Handbrake must induce larger body slip angle (oversteer)"
        )

    def test_pacejka_lateral_force_saturation(self):
        """Lateral force should saturate as slip angle increases (Pacejka curve)."""
        fz = 5000.0  # N
        angles = np.radians(np.linspace(0, 30, 50))
        forces = [abs(self.dyn.pacejka_lateral_force(a, fz)) for a in angles]

        # Max force should not exceed mu * fz
        max_force = max(forces)
        self.assertLessEqual(max_force, self.params.tire_mu_peak * fz * 1.05)

        # Curve should peak around 6 to 15 degrees then plateau or slightly drop
        peak_idx = int(np.argmax(forces))
        peak_angle_deg = np.degrees(angles[peak_idx])
        self.assertTrue(5.0 <= peak_angle_deg <= 16.0, f"Peak slip angle {peak_angle_deg} should be realistic")

    def test_car_corners_geometry(self):
        """Test bounding box corner coordinates."""
        state = VehicleState(x=10.0, y=20.0, yaw=0.0)
        corners = self.dyn.get_car_corners(state)
        self.assertEqual(corners.shape, (4, 2))
        # Center should be near (10, 20)
        center = np.mean(corners, axis=0)
        self.assertAlmostEqual(center[1], 20.0, places=2)


if __name__ == "__main__":
    unittest.main()
