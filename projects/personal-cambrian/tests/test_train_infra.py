"""Stage 3.2 acceptance: training is checkpointed, resumable, and eval is
reproducible.

Checks (tiny CI budget): run_training writes config/metrics/checkpoints; a
checkpoint round-trips to identical weights; resume continues global_step + iter
from the checkpoint (not from zero); periodic eval is recorded; and evaluate() is
deterministic for a fixed (agent, seed).

Runs:  python3 tests/test_train_infra.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys
import json
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.sim import CreatureEnv, LocomotionTask
from personal_cambrian.control import (
    PPOConfig, ActorCritic, evaluate, run_training, agent_from_checkpoint,
)

TINY = PPOConfig(total_timesteps=256, n_envs=2, n_steps=64, n_minibatches=2,
                 hidden=32, seed=0)                       # -> 2 iters, 128 steps/iter


def _make_env(seed):
    return CreatureEnv(quadruped(), task=LocomotionTask(max_steps=40))


def test_run_training_writes_artifacts():
    with tempfile.TemporaryDirectory() as d:
        agent, hist = run_training(_make_env, TINY, d, eval_every=1,
                                   checkpoint_every=1, verbose=False)
        for fname in ("config.json", "metrics.jsonl", "last.pt", "best.pt"):
            assert os.path.exists(os.path.join(d, fname)), fname
        # metrics.jsonl has one line per iteration
        with open(os.path.join(d, "metrics.jsonl")) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) == len(hist) >= 1


def test_checkpoint_roundtrips_to_identical_weights():
    with tempfile.TemporaryDirectory() as d:
        agent, _ = run_training(_make_env, TINY, d, eval_every=1,
                                checkpoint_every=1, verbose=False)
        clone, ck = agent_from_checkpoint(os.path.join(d, "last.pt"))
        env = _make_env(0)
        obs, _ = env.reset(seed=0)
        assert np.allclose(agent.act(obs, deterministic=True),
                           clone.act(obs, deterministic=True))
        env.close()


def test_resume_continues_step_and_iter():
    with tempfile.TemporaryDirectory() as d:
        run_training(_make_env, TINY, d, eval_every=1, checkpoint_every=1, verbose=False)
        ck = agent_from_checkpoint(os.path.join(d, "last.pt"))[1]
        step0, iter0 = ck["global_step"], ck["iter"]
        # resume for another budget
        _, hist2 = run_training(_make_env, TINY, d, eval_every=1, checkpoint_every=1,
                                resume_from=os.path.join(d, "last.pt"), verbose=False)
        assert hist2[0]["iter"] == iter0 + 1                 # continues numbering
        assert hist2[0]["global_step"] > step0               # continues step count


def test_periodic_eval_is_recorded():
    with tempfile.TemporaryDirectory() as d:
        _, hist = run_training(_make_env, TINY, d, eval_every=1, checkpoint_every=2,
                               verbose=False)
        assert all("eval_distance" in h and "eval_return" in h for h in hist)


def test_eval_is_reproducible():
    env = _make_env(0)
    agent = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], 32)
    a = evaluate(agent, env, max_steps=40, seed=3)
    b = evaluate(agent, env, max_steps=40, seed=3)
    assert a == b
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
