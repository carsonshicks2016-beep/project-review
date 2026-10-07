"""Stage 2.4 acceptance: the locomotion reward is finite, deterministic, and a
forward-moving rollout scores higher than a still one; falls terminate.

Mixes pure unit tests of LocomotionTask (reward math, fall logic) with env-level
integration tests (determinism, forward > still, fall termination).

Runs:  python3 tests/test_reward.py   (requires mujoco + gymnasium)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import agent_zero
from personal_cambrian.sim import CreatureEnv, LocomotionTask


# --- pure reward-function unit tests --------------------------------------- #
def test_reward_increases_with_forward_velocity():
    t = LocomotionTask()
    z = np.zeros(8)
    r_slow, _ = t.reward(forward_vel=0.0, action=z, prev_action=z, dt=0.01)
    r_fast, _ = t.reward(forward_vel=2.0, action=z, prev_action=z, dt=0.01)
    assert r_fast > r_slow


def test_reward_components_signed_correctly():
    t = LocomotionTask()
    a = np.ones(8)
    prev = np.zeros(8)
    r, c = t.reward(forward_vel=1.0, action=a, prev_action=prev, dt=0.01)
    assert c["forward"] > 0 and c["alive"] > 0
    assert c["energy"] <= 0 and c["smooth"] <= 0     # costs are negative
    assert np.isfinite(r)


def test_reward_handles_nan_inputs():
    t = LocomotionTask()
    r, _ = t.reward(forward_vel=np.nan, action=[np.nan], prev_action=[0.0], dt=0.01)
    assert np.isfinite(r)


def test_is_fallen_low_and_tipped():
    t = LocomotionTask(fall_fraction=0.5, upright_min=0.3)
    assert t.is_fallen(root_z=0.2, stand_height=1.0, up_proj=1.0)[0] is True   # low
    assert t.is_fallen(root_z=1.0, stand_height=1.0, up_proj=0.0)[1] == "tipped"
    assert t.is_fallen(root_z=1.0, stand_height=1.0, up_proj=1.0)[0] is False  # upright


# --- env-level integration tests ------------------------------------------- #
def test_env_reward_finite_on_random_rollout():
    env = CreatureEnv(agent_zero())
    env.reset(seed=1)
    for _ in range(200):
        _, r, term, trunc, info = env.step(env.action_space.sample())
        assert np.isfinite(r)
        assert set(info["reward_terms"]) == {"forward", "energy", "smooth", "alive"}
        if term or trunc:
            break
    env.close()


def test_env_reward_is_deterministic():
    e1, e2 = CreatureEnv(agent_zero()), CreatureEnv(agent_zero())
    e1.reset(seed=5); e2.reset(seed=5)
    for _ in range(30):
        a = e1.action_space.sample()
        r1 = e1.step(a)[1]
        r2 = e2.step(a)[1]
        assert r1 == r2
    e1.close(); e2.close()


def test_forward_motion_scores_higher_than_still():
    def cumulative_reward(forward_speed, steps=15):
        env = CreatureEnv(agent_zero(), task=LocomotionTask(max_steps=10_000))
        env.reset(seed=0)
        env.data.qvel[0] = forward_speed       # give the root forward momentum
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        env._prev_x = float(env.data.qpos[0])
        total = 0.0
        zero = np.zeros(env.action_space.shape, np.float32)
        for _ in range(steps):
            _, r, term, _, _ = env.step(zero)
            total += r
            if term:
                break
        env.close()
        return total

    moving = cumulative_reward(3.0)
    still = cumulative_reward(0.0)
    assert moving > still


def test_tipped_creature_terminates():
    import mujoco
    env = CreatureEnv(agent_zero())
    env.reset(seed=0)
    # rotate the root ~90 deg about x -> torso on its side -> low up_proj
    env.data.qpos[3:7] = [0.7071, 0.7071, 0.0, 0.0]
    mujoco.mj_forward(env.model, env.data)
    env._stand_height = 1.0                    # ensure 'low' doesn't pre-empt 'tipped'
    _, _, term, _, info = env.step(np.zeros(env.action_space.shape, np.float32))
    assert term and info["fall_reason"] in ("tipped", "low")
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
