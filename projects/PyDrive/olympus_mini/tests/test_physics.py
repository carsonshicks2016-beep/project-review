import torch
import unittest
import os
import numpy as np
import mujoco
from olympus_mini.envs import make_env, SprintEnv, HurdleEnv, VaultEnv
from olympus_mini.policy import PPO, ActorCritic

class TestOlympusPhysics(unittest.TestCase):
    def test_xml_models_load(self):
        models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
        for xml_name in ["athlete.xml", "hurdle_track.xml", "vault_track.xml"]:
            path = os.path.join(models_dir, xml_name)
            self.assertTrue(os.path.exists(path), f"File {xml_name} missing")
            m = mujoco.MjModel.from_xml_path(path)
            d = mujoco.MjData(m)
            mujoco.mj_step(m, d)
            self.assertGreater(m.nq, 15)
            self.assertEqual(m.nu, 16)

    def test_sprint_env(self):
        env = SprintEnv(frame_skip=5, max_steps=50)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (74,))
        obs, rew, term, trunc, info = env.step(np.zeros(16))
        self.assertIn("vx", info)
        self.assertIn("dist", info)
        self.assertFalse(np.isnan(rew))
        env.close()

    def test_hurdle_env(self):
        env = HurdleEnv(frame_skip=5, max_steps=50)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (74,))
        obs, rew, term, trunc, info = env.step(np.zeros(16))
        self.assertIn("hurdles_cleared", info)
        self.assertFalse(np.isnan(rew))
        env.close()

    def test_vault_env(self):
        env = VaultEnv(frame_skip=5, max_steps=50)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (74,))
        obs, rew, term, trunc, info = env.step(np.zeros(16))
        self.assertIn("crossbar_cleared", info)
        self.assertFalse(np.isnan(rew))
        env.close()

    def test_actor_critic_symmetry(self):
        ac = ActorCritic(obs_dim=74, act_dim=16)
        dummy_action = np.ones((1, 16), dtype=np.float32)
        mirrored_act = ac.mirror_action(torch.as_tensor(dummy_action))
        self.assertEqual(mirrored_act.shape, (1, 16))
        
        dummy_obs = np.ones((1, 74), dtype=np.float32)
        mirrored_obs = ac.mirror_obs(torch.as_tensor(dummy_obs))
        self.assertEqual(mirrored_obs.shape, (1, 74))

    def test_sprint_posture_gating(self):
        env = SprintEnv(frame_skip=5, max_steps=20)
        obs, info = env.reset()
        # Force low z to simulate falling
        env.data.qpos[2] = 0.50
        obs, rew, term, trunc, info = env.step(np.zeros(16))
        self.assertTrue(term)
        self.assertEqual(info["posture"], 0.0)
        env.close()

    def test_ppo_step_and_update(self):
        env = SprintEnv(frame_skip=5, max_steps=20)
        ppo = PPO(obs_dim=74, act_dim=16, device="cpu")
        obs, _ = env.reset()
        act, logp, val = ppo.select_action(obs)
        self.assertEqual(act.shape, (16,))
        self.assertIsInstance(val, float)
        env.close()

if __name__ == "__main__":
    unittest.main()
