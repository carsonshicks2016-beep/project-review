"""Stage 2.7 acceptance: an open-loop CPG gait produces measurable forward motion
that the reward rewards — proving the evaluation stack is non-degenerate before RL.

The quadruped is the vehicle: stable open-loop locomotion is achievable for it
(and effectively impossible for a biped without feedback). We assert the tuned
gait clearly beats the zero-action baseline in forward progress and earns positive
forward reward, deterministically.

Runs:  python3 tests/test_openloop.py   (requires mujoco + gymnasium)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.sim.openloop import open_loop_rollout, zero_action_rollout

GAIT = dict(freq=1.5, amp=1.0, phase="ramp", steps=400, seed=0)


def _quad_env():
    return CreatureEnv(quadruped(), task=LocomotionTask(max_steps=400))


def test_quadruped_moves_forward_under_open_loop():
    r = open_loop_rollout(_quad_env(), **GAIT)
    assert r["net_x"] > 0.05                  # measurable forward progress
    assert r["forward_reward"] > 0.0          # the task rewarded that progress


def test_gait_beats_zero_action():
    gait = open_loop_rollout(_quad_env(), **GAIT)
    still = zero_action_rollout(_quad_env(), steps=400, seed=0)
    assert gait["net_x"] > still["net_x"] + 0.1   # clearly better than doing nothing


def test_open_loop_is_deterministic():
    a = open_loop_rollout(_quad_env(), **GAIT)
    b = open_loop_rollout(_quad_env(), **GAIT)
    assert a["net_x"] == b["net_x"]
    assert a["forward_reward"] == b["forward_reward"]


def test_control_input_is_non_degenerate_for_biped():
    # the biped won't walk open-loop, but control must still change the outcome
    # (the reward channel responds to input) vs standing still.
    env = CreatureEnv(agent_zero(), task=LocomotionTask(max_steps=200))
    gait = open_loop_rollout(env, freq=2.5, amp=1.0, phase="alt", steps=200, seed=0)
    still = zero_action_rollout(CreatureEnv(agent_zero(), task=LocomotionTask(max_steps=200)),
                                steps=200, seed=0)
    assert abs(gait["net_x"] - still["net_x"]) > 1e-3   # input demonstrably matters


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
