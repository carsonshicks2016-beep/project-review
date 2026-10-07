"""Stage 3.4 acceptance (mechanism): a child's controller is faithfully warm-started
from its parent's, across morphologies.

Faithful copy + cross-morphology transfer are the deterministic, CI-fast core; the
"starts above random + reaches competence in far fewer steps than from scratch"
measurement is scripts/warmstart_demo.py (it needs real training).

Runs:  python3 tests/test_warmstart.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import (
    PPOConfig, ModularPolicy, policy_for_env, warm_start, warm_started_make_agent,
)
from personal_cambrian.control.ppo import train


def _env(fn):
    return CreatureEnv(fn(), task=LocomotionTask(max_steps=40), obs_mode="structured")


def test_warm_start_copies_weights_and_reproduces_parent():
    parent = policy_for_env(_env(quadruped))
    child = warm_start(parent, _env(quadruped))            # same morphology
    for k in parent.state_dict():
        assert torch.allclose(parent.state_dict()[k], child.state_dict()[k])
    obs, _ = _env(quadruped).reset(seed=0)
    # identical weights + identical morphology -> identical deterministic policy
    assert np.allclose(parent.act(obs, deterministic=True),
                       child.act(obs, deterministic=True))


def test_warm_start_across_morphology():
    parent = policy_for_env(_env(quadruped))               # 4 actuators
    biped_env = _env(agent_zero)                           # 8 actuators
    child = warm_start(parent, biped_env)
    a = child.act(biped_env.reset(seed=0)[0], deterministic=True)
    assert a.shape == (8,) and np.all(np.isfinite(a))
    for k in parent.state_dict():                          # shared weights transferred
        assert torch.allclose(parent.state_dict()[k], child.state_dict()[k])
    biped_env.close()


def test_warm_started_factory_loads_parent():
    parent = policy_for_env(_env(quadruped))
    ma = warm_started_make_agent(lambda s: _env(quadruped), parent)
    child = ma(None, None)
    obs, _ = _env(quadruped).reset(seed=0)
    assert np.allclose(parent.act(obs, True), child.act(obs, True))   # starts AS the parent


def test_warm_started_child_trains():
    parent = policy_for_env(_env(quadruped))
    cme = lambda s: CreatureEnv(quadruped(), task=LocomotionTask(max_steps=40),
                                obs_mode="structured")
    ma = warm_started_make_agent(cme, parent)
    cfg = PPOConfig(total_timesteps=256, n_envs=2, n_steps=64, n_minibatches=2)
    agent, hist = train(cme, cfg, make_agent=ma)
    assert isinstance(agent, ModularPolicy) and len(hist) >= 1
    assert np.isfinite(hist[-1]["mean_return"])


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
