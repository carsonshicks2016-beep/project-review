"""Stage 2.5 acceptance: niche tasks are declarative — fully specified by a
(JSON-serializable) config dict, reconstructible, with an escalation hook.

Runs:  python3 tests/test_task_config.py   (requires mujoco + gymnasium)
"""
import os
import sys
import json
import dataclasses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import agent_zero
from personal_cambrian.sim import CreatureEnv
from personal_cambrian.sim.tasks import NicheTask, LocomotionTask, task_from_dict


def test_locomotion_is_a_nichetask():
    assert isinstance(LocomotionTask(), NicheTask)


def test_config_roundtrip_and_dispatch():
    t = LocomotionTask(forward_weight=2.0, max_steps=321, terrain="flat",
                       forward_axis=1, alive_bonus=0.25)
    d = t.to_dict()
    assert d["task"] == "LocomotionTask"
    # JSON-serializable (the declarative config representation)
    d2 = json.loads(json.dumps(d))
    back = task_from_dict(d2)
    assert isinstance(back, LocomotionTask)
    assert back == t                      # dataclass equality over all fields


def test_all_fields_are_config():
    t = LocomotionTask()
    names = {f.name for f in dataclasses.fields(t)}
    assert {"name", "terrain", "forward_axis", "max_steps", "difficulty",
            "escalation_rate", "forward_weight", "ctrl_cost", "smooth_cost",
            "alive_bonus", "fall_fraction", "upright_min"} <= names


def test_escalation_bumps_difficulty():
    t = LocomotionTask(escalation_rate=0.5)
    assert t.difficulty == 0.0
    t.escalate()
    t.escalate(2.0)
    assert t.difficulty == 0.5 * 1.0 + 0.5 * 2.0      # 1.5


def test_truncation_uses_config_max_steps():
    t = LocomotionTask(max_steps=10)
    assert t.truncated(9) is False
    assert t.truncated(10) is True


def test_env_runs_from_config_built_task():
    cfg = {"task": "LocomotionTask", "name": "locomotion", "terrain": "flat",
           "forward_axis": 0, "max_steps": 25, "difficulty": 0.0,
           "escalation_rate": 0.0, "forward_weight": 1.5, "ctrl_cost": 1e-3,
           "smooth_cost": 1e-3, "alive_bonus": 0.2, "fall_fraction": 0.5,
           "upright_min": 0.3}
    env = CreatureEnv(agent_zero(), task=task_from_dict(cfg))
    env.reset(seed=0)
    done = False
    for _ in range(25):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r)
        if term or trunc:
            done = True
            break
    assert done
    env.close()


def test_reward_weights_take_effect_from_config():
    z = np.zeros(8)
    a = LocomotionTask(forward_weight=1.0).reward(
        forward_vel=2.0, action=z, prev_action=z, dt=0.01)[0]
    b = LocomotionTask(forward_weight=3.0).reward(
        forward_vel=2.0, action=z, prev_action=z, dt=0.01)[0]
    assert b > a                          # higher forward weight -> higher reward


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
