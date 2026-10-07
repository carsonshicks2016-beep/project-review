"""
Unit tests for humanoid parkour pipeline, neural architecture, and simulation.
"""

import unittest
import gymnasium as gym
import torch
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from humanoid_parkour.train import Agent, layer_init
from humanoid_parkour.envs.wrappers import make_env, RunningMeanStd, NormalizeObservation


class TestPipeline(unittest.TestCase):

    def setUp(self):
        self.env_id = "Walker2d-v5"
        self.num_envs = 2
        self.envs = gym.vector.SyncVectorEnv([make_env(self.env_id, i, False, "test") for i in range(self.num_envs)])
        self.agent = Agent(self.envs)

    def tearDown(self):
        self.envs.close()

    def test_agent_forward_pass(self):
        obs, _ = self.envs.reset(seed=42)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        action, logprob, entropy, value = self.agent.get_action_and_value(obs_t)

        self.assertEqual(action.shape, (self.num_envs, 6))
        self.assertEqual(logprob.shape, (self.num_envs,))
        self.assertEqual(entropy.shape, (self.num_envs,))
        self.assertEqual(value.shape, (self.num_envs, 1))

    def test_deterministic_actions(self):
        obs, _ = self.envs.reset(seed=42)
        obs_t = torch.as_tensor(obs, dtype=torch.float32)

        action1, _, _, _ = self.agent.get_action_and_value(obs_t, deterministic=True)
        action2, _, _, _ = self.agent.get_action_and_value(obs_t, deterministic=True)

        self.assertTrue(torch.allclose(action1, action2))

    def test_running_mean_std(self):
        rms = RunningMeanStd(shape=(4,))
        data = np.random.randn(100, 4) + 5.0
        rms.update(data)
        self.assertTrue(np.allclose(rms.mean, 5.0, atol=0.5))


if __name__ == "__main__":
    unittest.main()
