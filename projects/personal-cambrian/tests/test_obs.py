"""Stage 2.2 acceptance: the observation builder produces correctly-dimensioned,
finite, deterministic flat + structured observations for both seeds.

Documented layout (per ObservationBuilder):
    flat_dim = 2*n_1dof_joints + 3(gravity) + 3(lin vel) + 3(ang vel)
               + n_bodies(floor-contact flags) + n_actuators(prev action)
    structured = (n_actuators, 11)

Runs:  python3 tests/test_obs.py   (requires mujoco + gymnasium)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.sim import CreatureEnv
from personal_cambrian.sim.obs import ObservationBuilder


def test_flat_dim_matches_documented_formula():
    for fn in (agent_zero, quadruped):
        env = CreatureEnv(fn())
        b = env.obs_builder
        expected = 2 * b.n_jnt1 + 3 + 3 + 3 + b.n_bodies + b.nu
        assert b.flat_dim == expected
        obs, _ = env.reset(seed=0)
        assert obs.shape == (b.flat_dim,)
        assert env.observation_space.shape == (b.flat_dim,)
        env.close()


def test_structured_shape():
    for fn, nu in ((agent_zero, 8), (quadruped, 4)):
        env = CreatureEnv(fn())
        env.reset(seed=0)
        s = env.structured_obs()
        assert s.shape == (nu, env.obs_builder.node_dim) == (nu, 11)
        assert env.obs_builder.n_nodes == nu
        env.close()


def test_observations_are_finite():
    env = CreatureEnv(agent_zero())
    env.reset(seed=1)
    for _ in range(50):
        env.step(env.action_space.sample())
        assert np.all(np.isfinite(env._get_obs()))
        assert np.all(np.isfinite(env.structured_obs()))
    env.close()


def test_observations_are_deterministic():
    e1, e2 = CreatureEnv(agent_zero()), CreatureEnv(agent_zero())
    e1.reset(seed=7); e2.reset(seed=7)
    for _ in range(10):
        a = e1.action_space.sample()
        e1.step(a); e2.step(a)
        assert np.allclose(e1._get_obs(), e2._get_obs())
        assert np.allclose(e1.structured_obs(), e2.structured_obs())
    e1.close(); e2.close()


def test_prev_action_appears_in_flat_obs():
    env = CreatureEnv(agent_zero())
    env.reset(seed=0)
    a = np.ones(env.action_space.shape, dtype=np.float32)
    obs, *_ = env.step(a)
    nu = env.obs_builder.nu
    assert np.allclose(obs[-nu:], 1.0)          # last nu entries are prev_action
    env.close()


def test_floor_contact_flags_fire_when_resting():
    # let the creature settle on the floor; at least one body should be in contact
    env = CreatureEnv(agent_zero())
    env.reset(seed=0)
    for _ in range(400):
        env.step(np.zeros(env.action_space.shape, np.float32))
    b = env.obs_builder
    contacts = b._floor_contacts(env.data)
    assert contacts.shape == (b.n_bodies,)
    assert contacts.sum() >= 1                   # something is touching the floor
    env.close()


def test_spanned_joints_precomputed_for_each_actuator():
    # every muscle's tendon spans at least one joint (else it could not actuate)
    env = CreatureEnv(agent_zero())
    for span in env.obs_builder.act_span:
        assert span.size >= 1
    env.close()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
