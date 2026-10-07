"""Stage 2.3 acceptance: actions drive actuators within range; out-of-range is
clipped.

Checks: in-range actions map into each actuator's control range; out-of-range
actions saturate at the range bounds; zero maps to the range midpoint; the mapping
is monotonic (linear); NaN actions are handled; and after stepping with extreme
actions, data.ctrl stays within model.actuator_ctrlrange.

Runs:  python3 tests/test_action.py   (requires mujoco + gymnasium)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.sim import CreatureEnv


def _env(fn):
    e = CreatureEnv(fn())
    e.reset(seed=0)
    return e


def test_in_range_actions_map_into_ctrlrange():
    for fn in (agent_zero, quadruped):
        env = _env(fn)
        lo, hi = env._ctrl_lo, env._ctrl_hi
        rng = np.random.default_rng(0)
        for _ in range(20):
            a = rng.uniform(-1, 1, size=env.action_space.shape)
            ctrl = env._map_action(a)
            assert np.all(ctrl >= lo - 1e-9) and np.all(ctrl <= hi + 1e-9)
        env.close()


def test_out_of_range_is_clipped_to_bounds():
    env = _env(agent_zero)
    hi = env._map_action(np.full(env.action_space.shape, 10.0))
    lo = env._map_action(np.full(env.action_space.shape, -10.0))
    assert np.allclose(hi, env._ctrl_hi)
    assert np.allclose(lo, env._ctrl_lo)
    env.close()


def test_zero_action_is_range_midpoint():
    env = _env(agent_zero)
    ctrl = env._map_action(np.zeros(env.action_space.shape))
    mid = 0.5 * (env._ctrl_lo + env._ctrl_hi)
    assert np.allclose(ctrl, mid)
    env.close()


def test_mapping_is_monotonic():
    env = _env(agent_zero)
    c1 = env._map_action(np.full(env.action_space.shape, -0.5))
    c2 = env._map_action(np.full(env.action_space.shape, 0.5))
    assert np.all(c2 >= c1 - 1e-12)
    env.close()


def test_nan_action_is_handled():
    env = _env(agent_zero)
    a = np.zeros(env.action_space.shape)
    a[0] = np.nan
    ctrl = env._map_action(a)
    assert np.all(np.isfinite(ctrl))
    # NaN -> 0 -> midpoint for that actuator
    assert np.isclose(ctrl[0], 0.5 * (env._ctrl_lo[0] + env._ctrl_hi[0]))
    env.close()


def test_step_keeps_ctrl_in_range():
    env = _env(agent_zero)
    for sign in (+50.0, -50.0):
        env.step(np.full(env.action_space.shape, sign, dtype=np.float32))
        assert np.all(env.data.ctrl >= env.model.actuator_ctrlrange[:, 0] - 1e-9)
        assert np.all(env.data.ctrl <= env.model.actuator_ctrlrange[:, 1] + 1e-9)
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
