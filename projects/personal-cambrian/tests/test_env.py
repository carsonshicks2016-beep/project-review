"""Stage 2.1 acceptance: CreatureEnv is a valid Gymnasium environment.

Checks: it passes gymnasium.utils.env_checker.check_env; reset/step return the
right shapes and types; seeded resets are deterministic; a random rollout stays
finite; and both seeds (biped + quadruped) wrap successfully.

Runs:  python3 tests/test_env.py   (requires mujoco + gymnasium)
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from gymnasium.utils.env_checker import check_env

from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask


def test_env_checker_passes():
    env = CreatureEnv(agent_zero())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")          # ignore unbounded-obs-space warning
        check_env(env, skip_render_check=True)
    env.close()


def test_reset_and_step_contract():
    env = CreatureEnv(agent_zero())
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert isinstance(info, dict)
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    assert env.observation_space.contains(obs)
    assert isinstance(r, float)
    assert isinstance(term, bool) and isinstance(trunc, bool)
    assert {"forward_vel", "x", "root_z", "up_proj", "reward_terms"} <= set(info)
    env.close()


def test_action_space_matches_actuators():
    env = CreatureEnv(agent_zero())
    assert env.action_space.shape == (env.model.nu,) == (8,)
    env.close()


def test_seeded_reset_is_deterministic():
    e1, e2 = CreatureEnv(agent_zero()), CreatureEnv(agent_zero())
    o1, _ = e1.reset(seed=42)
    o2, _ = e2.reset(seed=42)
    assert np.allclose(o1, o2)
    # same actions -> same trajectory
    for _ in range(20):
        a = e1.action_space.sample()
        s1 = e1.step(a)
        s2 = e2.step(a)
        assert np.allclose(s1[0], s2[0])
    e1.close(); e2.close()


def test_random_rollout_stays_finite():
    env = CreatureEnv(agent_zero(), task=LocomotionTask(max_steps=300))
    env.reset(seed=1)
    steps = 0
    for _ in range(300):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        steps += 1
        if term or trunc:
            break
    assert steps > 0
    env.close()


def test_truncates_at_max_steps_when_upright_enough():
    # zero action: the biped may topple (terminate) or survive to truncation; either
    # way the episode must END cleanly within max_steps without error.
    env = CreatureEnv(agent_zero(), task=LocomotionTask(max_steps=50))
    env.reset(seed=2)
    done = False
    for _ in range(50):
        _, _, term, trunc, _ = env.step(np.zeros(env.action_space.shape, np.float32))
        if term or trunc:
            done = True
            break
    assert done
    env.close()


def test_quadruped_wraps_too():
    env = CreatureEnv(quadruped())
    obs, _ = env.reset(seed=0)
    assert env.action_space.shape == (4,)
    env.step(env.action_space.sample())
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
