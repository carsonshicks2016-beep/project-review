"""Stage 3.3 acceptance: a single morphology-aware modular policy drives ANY body
plan with shared weights.

Headline check: the parameters are identical in shape across creatures with
different actuator counts, so one trained weight set (a 4-actuator quadruped's)
loads straight into an 8-actuator biped policy and both produce valid per-actuator
actions. Plus: forward/value shapes, determinism, and that it trains under the
existing PPO loop. ("matches/beats the MLP" is demonstrated in scripts/train_ppo.py
--policy modular.)

Runs:  python3 tests/test_modular.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from personal_cambrian.seeds import agent_zero, quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import PPOConfig, ModularPolicy, policy_for_env
from personal_cambrian.control.ppo import train
from personal_cambrian.control.modular import make_modular_setup


def _env(fn):
    return CreatureEnv(fn(), task=LocomotionTask(max_steps=40), obs_mode="structured")


def test_structured_env_obs_shape():
    env = _env(quadruped)
    b = env.obs_builder
    obs, _ = env.reset(seed=0)
    assert obs.shape == (b.n_nodes * b.node_dim,)        # flattened structured matrix
    assert env.action_space.shape == (b.n_nodes,)
    env.close()


def test_one_weight_set_drives_both_seeds():
    quad_env, biped_env = _env(quadruped), _env(agent_zero)
    quad_pi = policy_for_env(quad_env)                   # 4 actuators
    biped_pi = policy_for_env(biped_env)                 # 8 actuators

    # parameters are morphology-independent: identical names + shapes
    qs, bs = quad_pi.state_dict(), biped_pi.state_dict()
    assert qs.keys() == bs.keys()
    assert all(qs[k].shape == bs[k].shape for k in qs)

    # the SAME trained weights control both creatures
    biped_pi.load_state_dict(quad_pi.state_dict())
    qa = quad_pi.act(quad_env.reset(seed=0)[0], deterministic=True)
    ba = biped_pi.act(biped_env.reset(seed=0)[0], deterministic=True)
    assert qa.shape == (4,) and ba.shape == (8,)
    assert np.all(np.isfinite(qa)) and np.all(np.isfinite(ba))
    quad_env.close(); biped_env.close()


def test_param_count_independent_of_morphology():
    n_quad = sum(p.numel() for p in policy_for_env(_env(quadruped)).parameters())
    n_biped = sum(p.numel() for p in policy_for_env(_env(agent_zero)).parameters())
    assert n_quad == n_biped


def test_adjacency_not_in_state_dict():
    pi = policy_for_env(_env(quadruped))
    assert not any("adj" in k for k in pi.state_dict())  # adjacency excluded -> transferable


def test_forward_shapes_and_determinism():
    env = _env(quadruped)
    pi = policy_for_env(env)
    obs, _ = env.reset(seed=0)
    x = torch.as_tensor(np.stack([obs, obs]), dtype=torch.float32)   # batch of 2
    action, logp, ent, value = pi.get_action_and_value(x)
    assert action.shape == (2, env.obs_builder.n_nodes)
    assert logp.shape == (2,) and value.shape == (2,)
    assert np.array_equal(pi.act(obs, deterministic=True), pi.act(obs, deterministic=True))
    env.close()


def test_modular_policy_trains_under_ppo():
    make_env, make_agent = make_modular_setup("quadruped", ep_steps=40, hidden=32)
    cfg = PPOConfig(total_timesteps=256, n_envs=2, n_steps=64, n_minibatches=2, hidden=32)
    agent, history = train(make_env, cfg, make_agent=make_agent)
    assert isinstance(agent, ModularPolicy)
    assert len(history) >= 1 and np.isfinite(history[-1]["mean_return"])


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
