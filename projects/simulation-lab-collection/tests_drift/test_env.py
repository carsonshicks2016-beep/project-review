"""
Unit tests for DriftGymkhanaEnv Gymnasium environment.
"""

import unittest
import numpy as np
from gymnasium.utils.env_checker import check_env
from drift_rl.env import DriftGymkhanaEnv


class TestDriftEnv(unittest.TestCase):

    def test_gym_api_compliance(self):
        """Verify environment conforms to Gymnasium specifications."""
        env = DriftGymkhanaEnv(track_name="touge")
        # gymnasium check_env validates reset, step, shapes, and types
        check_env(env.unwrapped, skip_render_check=True)
        env.close()

    def test_reset_and_step_shapes(self):
        """Test observation and action shapes and bounds."""
        env = DriftGymkhanaEnv(track_name="touge")
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (35,))
        self.assertFalse(np.isnan(obs).any(), "Obs contains NaN")
        self.assertIn("score", info)
        self.assertIn("multiplier", info)

        # Test step with continuous actions
        action = np.array([0.5, 0.8, 0.0], dtype=np.float32)
        next_obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(next_obs.shape, (35,))
        self.assertFalse(np.isnan(next_obs).any(), "Next obs contains NaN")
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        env.close()

    def test_all_tracks_initialization(self):
        """Verify all track presets (touge, gymkhana, stadium) load without error."""
        for track_name in ["touge", "gymkhana", "stadium"]:
            env = DriftGymkhanaEnv(track_name=track_name)
            obs, _ = env.reset()
            self.assertEqual(obs.shape, (35,))
            # Run 10 random steps
            for _ in range(10):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                if terminated or truncated:
                    break
            env.close()


if __name__ == "__main__":
    unittest.main()
