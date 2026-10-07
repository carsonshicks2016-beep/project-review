"""
Unit tests for SatelliteDockingEnv Gymnasium compliance and docking logic.
"""

import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from satellite_rl.env import SatelliteDockingEnv


class TestSatelliteDockingEnv(unittest.TestCase):

    def setUp(self):
        self.env = SatelliteDockingEnv(stage="docking", max_steps=100)

    def test_gym_check_env(self):
        """Standard Gymnasium check_env validation."""
        check_env(self.env.unwrapped)

    def test_reset_and_observation_shape(self):
        obs, info = self.env.reset(seed=42)
        self.assertEqual(obs.shape, (10,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertIn("initial_distance", info)
        self.assertIn("propellant_kg", info)

    def test_step_consumes_propellant(self):
        obs, _ = self.env.reset(seed=42)
        initial_fuel = self.env.propellant

        action = np.array([1.0, 0.5, -0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertLess(self.env.propellant, initial_fuel)
        self.assertGreater(self.env.total_delta_v, 0.0)
        self.assertEqual(obs.shape, (10,))

    def test_successful_soft_docking(self):
        """Simulate perfect soft arrival at docking port."""
        self.env.reset(seed=42)
        # Manually set state to 0.15m along Y with safe closing speed -0.02 m/s
        self.env.state = np.array([0.0, 0.15, 0.0, 0.0, -0.02, 0.0], dtype=np.float64)
        self.env.prev_distance = 0.15

        # Small gentle thrust or zero thrust
        action = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertTrue(terminated)
        self.assertTrue(info["success"])
        self.assertFalse(info["collision"])
        self.assertGreater(reward, 100.0)

    def test_hard_impact_collision(self):
        """Simulate high speed arrival at docking port (crash)."""
        self.env.reset(seed=42)
        # Position 0.15m along Y but slamming at 0.5 m/s
        self.env.state = np.array([0.0, 0.15, 0.0, 0.0, -0.5, 0.0], dtype=np.float64)
        self.env.prev_distance = 0.15

        action = np.zeros(3, dtype=np.float32)
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.assertTrue(terminated)
        self.assertFalse(info["success"])
        self.assertTrue(info["collision"])
        self.assertLess(reward, 0.0)

    def test_curriculum_scaling(self):
        """Verify spawn distances scale with curriculum level."""
        self.env.set_curriculum_level(0.0)
        _, info_lvl0 = self.env.reset(seed=10)

        self.env.set_curriculum_level(1.0)
        _, info_lvl1 = self.env.reset(seed=10)

        self.assertGreater(info_lvl1["initial_distance"], info_lvl0["initial_distance"])


if __name__ == "__main__":
    unittest.main()
