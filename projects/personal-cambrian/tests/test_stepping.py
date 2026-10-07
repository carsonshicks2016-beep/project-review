"""Stage 2.6 acceptance: the control-rate contract is fixed and deterministic.

Checks: control_dt = n_substeps * physics_dt; each env.step advances sim time by
exactly control_dt (frame-skip = n_substeps); a different n_substeps changes the
control rate; and identical (seed, action-sequence) yields a bit-identical full
trajectory (qpos, qvel, rewards) — while different seeds diverge.

Runs:  python3 tests/test_stepping.py   (requires mujoco + gymnasium)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import agent_zero
from personal_cambrian.sim import CreatureEnv


def _rollout(env, actions):
    """Run a fixed action sequence, returning per-step (qpos, qvel, reward)."""
    qpos, qvel, rews = [], [], []
    for a in actions:
        _, r, term, trunc, _ = env.step(a)
        qpos.append(env.data.qpos.copy())
        qvel.append(env.data.qvel.copy())
        rews.append(r)
        if term or trunc:
            break
    return np.array(qpos), np.array(qvel), np.array(rews)


def test_control_dt_formula():
    env = CreatureEnv(agent_zero(), n_substeps=5)
    assert env.control_dt == 5 * env.physics_dt
    assert np.isclose(env.control_hz, 1.0 / env.control_dt)
    env.close()


def test_step_advances_time_by_control_dt():
    env = CreatureEnv(agent_zero(), n_substeps=5)
    env.reset(seed=0)
    t0 = float(env.data.time)
    for k in range(1, 11):
        env.step(np.zeros(env.action_space.shape, np.float32))
        assert np.isclose(float(env.data.time) - t0, k * env.control_dt, atol=1e-9)
    env.close()


def test_n_substeps_changes_control_rate():
    e5 = CreatureEnv(agent_zero(), n_substeps=5)
    e10 = CreatureEnv(agent_zero(), n_substeps=10)
    assert np.isclose(e10.control_dt, 2.0 * e5.control_dt)
    e5.close(); e10.close()


def test_identical_seed_and_actions_give_identical_trajectory():
    e1, e2 = CreatureEnv(agent_zero()), CreatureEnv(agent_zero())
    e1.reset(seed=11); e2.reset(seed=11)
    rng = np.random.default_rng(0)
    actions = [rng.uniform(-1, 1, size=e1.action_space.shape).astype(np.float32)
               for _ in range(60)]
    q1, v1, r1 = _rollout(e1, actions)
    q2, v2, r2 = _rollout(e2, actions)
    assert np.array_equal(q1, q2)
    assert np.array_equal(v1, v2)
    assert np.array_equal(r1, r2)
    e1.close(); e2.close()


def test_different_seeds_diverge():
    e1, e2 = CreatureEnv(agent_zero()), CreatureEnv(agent_zero())
    e1.reset(seed=1); e2.reset(seed=2)
    rng = np.random.default_rng(0)
    actions = [rng.uniform(-1, 1, size=e1.action_space.shape).astype(np.float32)
               for _ in range(40)]
    q1, _, _ = _rollout(e1, actions)
    q2, _, _ = _rollout(e2, actions)
    assert not np.array_equal(q1, q2)
    e1.close(); e2.close()


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
