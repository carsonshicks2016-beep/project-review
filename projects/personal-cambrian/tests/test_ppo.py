"""Stage 3.1 (fast machinery test): PPO trains briefly without error and the agent
saves/loads and acts deterministically.

This is the CI-budget smoke test (a few hundred steps); the actual "return rises +
moves past a distance threshold" demonstration is scripts/train_ppo.py (minutes).

Runs:  python3 tests/test_ppo.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from personal_cambrian.seeds import quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import ActorCritic, PPOConfig, evaluate
from personal_cambrian.control.ppo import train

TINY = PPOConfig(total_timesteps=256, n_envs=2, n_steps=64, n_minibatches=2,
                 hidden=32, seed=0)


def _make_env(seed):
    # short episodes so the tiny rollout completes episodes (finite mean_return)
    return CreatureEnv(quadruped(), task=LocomotionTask(max_steps=40))


def test_ppo_trains_and_logs_history():
    logs = []
    agent, history = train(_make_env, TINY, log_fn=logs.append)
    assert isinstance(agent, ActorCritic)
    assert len(history) >= 1
    assert len(logs) == len(history)
    assert all(np.isfinite(h["mean_return"]) for h in history)
    assert history[-1]["global_step"] >= TINY.total_timesteps - TINY.n_envs * TINY.n_steps


def test_agent_act_shapes_and_determinism():
    env = _make_env(0)
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], 32)
    obs, _ = env.reset(seed=0)
    a = agent.act(obs, deterministic=True)
    assert a.shape == env.action_space.shape
    assert np.array_equal(a, agent.act(obs, deterministic=True))   # deterministic
    env.close()


def test_agent_save_load_roundtrip():
    env = _make_env(0)
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], 32)
    obs, _ = env.reset(seed=0)
    before = agent.act(obs, deterministic=True)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "policy.pt")
        torch.save(agent.state_dict(), path)
        clone = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], 32)
        clone.load_state_dict(torch.load(path))
    assert np.allclose(before, clone.act(obs, deterministic=True))
    env.close()


def test_evaluate_returns_finite_metrics():
    env = _make_env(0)
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], 32)
    m = evaluate(agent, env, max_steps=40, seed=0)
    assert set(m) == {"distance", "return", "steps"}
    assert np.isfinite(m["distance"]) and np.isfinite(m["return"]) and m["steps"] >= 1
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
