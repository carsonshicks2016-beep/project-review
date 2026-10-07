"""Targeted regressions for physical gaps, grounded success and honest evaluation."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.parkour_env import ParkourHumanoidEnv, TERRAIN_GEOM_GROUP
from envs import terrain
from eval import evaluate, make_evaluation_env, checkpoint_context


class ZeroPolicy:
    def predict(self, observation, deterministic):
        assert deterministic
        return np.zeros((len(observation), 17)), None


class PhysicsTests(unittest.TestCase):
    def setUp(self):
        self.env = ParkourHumanoidEnv()
        self.env.reset(seed=42)

    def tearDown(self):
        self.env.close()

    def place(self, x, z, y=0):
        qpos = self.env.init_qpos.copy()
        qpos[:3] = [x, y, z]
        self.env.set_state(qpos, np.zeros(self.env.model.nv))

    def test_gap_is_empty_and_valid_airborne_crossing_does_not_fail(self):
        for seed in range(100):
            self.env.reset(seed=seed)
            gaps = [s for s in self.env._segments if s.kind == "gap"]
            if gaps:
                break
        self.assertTrue(gaps)
        gap = gaps[0]
        x = (gap.x_start + gap.x_end) / 2
        self.place(x, gap.height + 1.40)
        hit = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(self.env.model, self.env.data,
            np.array([x, 0., gap.height + 1.4]), np.array([0., 0., -1.]),
            TERRAIN_GEOM_GROUP, 1, -1, hit)
        self.assertEqual(int(hit[0]), self.env._death_geom_id)
        self.assertGreater(distance, 2.0)
        self.assertTrue(self.env.is_healthy)
        self.assertFalse(self.env._supported_feet())
        self.place(x, gap.height + 0.4)
        self.assertFalse(self.env.is_healthy)

    def test_success_requires_healthy_supported_landing_and_is_terminal(self):
        finish = self.env._segments[-1].x_start + 1.0
        self.place(finish + 0.2, self.env._segments[-1].height + 1.4)
        with patch.object(self.env, "_supported_feet", return_value=set()):
            self.assertFalse(self.env._course_progress(True)[1])
        with patch.object(self.env, "_supported_feet", return_value=set(self.env._feet_ids)):
            self.assertFalse(self.env._course_progress(False)[1])
            with patch.object(self.env, "do_simulation"):
                _, _, terminated, truncated, info = self.env.step(np.zeros(17))
            self.assertTrue(info["success"])
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertFalse(info["failed"])
            self.assertEqual(info["completion_rate"], 1)
            self.assertEqual(info["progress_fraction"], 1)
            self.assertEqual(self.env._course_progress(True)[0], 0)  # no repeated bonus
            self.assertEqual(info["obstacles_total"], sum(s.kind != "flat" for s in self.env._segments))
            self.assertEqual(info["segments_cleared"], info["obstacles_total"])

    def test_timeout_is_truncation_not_success(self):
        self.env.max_episode_steps = 1
        _, reward, terminated, truncated, info = self.env.step(np.zeros(17))
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertFalse(info["success"])
        self.assertAlmostEqual(reward, sum(v for k, v in info.items() if k.startswith("reward_")))

    def test_level_change_only_changes_next_course(self):
        self.env.set_difficulty(4)
        self.assertEqual(self.env.step(np.zeros(17))[4]["curriculum_level"], 1)
        self.assertEqual(self.env.reset(seed=42)[1]["curriculum_level"], 4)

    def test_seed_reproduces_geometry_and_observation(self):
        a, _ = self.env.reset(seed=91)
        xml_a = self.env._current_xml
        b, _ = self.env.reset(seed=91)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(xml_a, self.env._current_xml)
        self.assertEqual(a.shape, (366,))
        self.assertEqual(self.env.action_space.shape, (17,))

    def test_flat_curriculum_and_rehearsal_are_explicit(self):
        segments, _ = terrain.generate_course(np.random.default_rng(5), difficulty=0)
        self.assertTrue(all(s.kind == "flat" for s in segments))
        self.env.set_difficulty(5)
        self.env.replay_prob = 1.0
        _, info = self.env.reset(seed=91)
        self.assertEqual(info["curriculum_level"], 5)
        self.assertLess(info["effective_difficulty"], 5)
        self.assertGreaterEqual(info["effective_difficulty"], 1)

    def test_exported_scene_compiles_and_matches_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "scene.xml"
            manifest = self.env.export_scene(path)
            compiled = mujoco.MjModel.from_xml_path(str(path))
            self.assertEqual(compiled.ngeom, self.env.model.ngeom)
            self.assertEqual(manifest["env_version"], "parkour-v2")
            self.assertEqual(json.loads(path.with_suffix(".json").read_text())["segments"][0]["kind"], "flat")


class EvaluationTests(unittest.TestCase):
    def test_evaluation_uses_kwargs_determinism_and_frozen_stats(self):
        vec = DummyVecEnv([lambda: ParkourHumanoidEnv()])
        norm = VecNormalize(vec)
        norm.reset()
        try:
            kwargs = dict(ctrl_cost_weight=0.5, max_episode_steps=3)
            first = evaluate(ZeroPolicy(), norm, "ParkourHumanoid", [51000], difficulty=1,
                             env_kwargs=kwargs)
            second = evaluate(ZeroPolicy(), norm, "ParkourHumanoid", [51000], difficulty=1,
                              env_kwargs=kwargs)
            self.assertEqual(first, second)
            self.assertEqual(first["environment_contract"]["reward_weights"]["control"], 0.5)
            self.assertEqual(first["episodes"][0]["length"], 3)
            self.assertTrue(first["episodes"][0]["truncated"])
            self.assertEqual(first["success_rate"], 0)
            self.assertEqual(first["timeout_rate"], 1)
        finally:
            norm.close()

    def test_empty_seed_list_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(None, None, "ParkourHumanoid", [])

    def test_evaluation_disables_rehearsal(self):
        env = make_evaluation_env("ParkourHumanoid", {"replay_prob": 1}, difficulty=4)
        try:
            info = env.reset(seed=42)[1]
            self.assertEqual(info["effective_difficulty"], 4)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
