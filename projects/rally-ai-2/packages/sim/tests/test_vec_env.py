"""Vectorised environment tests.

The load-bearing property for SyncVecEnv is bit-identity with a bare RallyEnv
at n=1: if the batching layer silently changes the trajectory, every later
throughput and learning claim is measuring a different world.

AsyncVecEnv must revive dead/stalled workers and keep the run alive — a dead
worker costs a short rollout, not the night.
"""

from __future__ import annotations

import json
import signal

import numpy as np
import pytest

from rallyai.env import EnvConfig, RallyEnv
from rallyai.stage.fixtures import proving_ground
from rallyai.stage.generator import generate
from rallyai.train.vec_env import AsyncVecEnv, SyncVecEnv


def _fixed_stage_fn(stage: dict):
    def stage_fn(_seed: int) -> dict:
        return stage

    return stage_fn


def test_sync_n1_bit_identical_to_bare_env():
    """n=1 through the vec env must match a bare RallyEnv given the same seed
    and action sequence. This is the reference the async path has to equal."""
    stage = proving_ground()
    stage_fn = _fixed_stage_fn(stage)
    seed = 7
    actions = [
        np.array([0.15, 0.8, 0.0, 0.0], dtype=np.float32),
        np.array([-0.2, 0.6, 0.1, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    ] * 40

    bare = RallyEnv(stage_fn(seed), EnvConfig())
    bare_obs, _ = bare.reset(seed=seed)
    bare_traj = [bare_obs.copy()]
    for a in actions:
        o, r, term, trunc, info = bare.step(a)
        bare_traj.append((o.copy(), float(r), bool(term), bool(trunc),
                          info.get("termination"), float(info["s"])))
        if term or trunc:
            break

    vec = SyncVecEnv(1, stage_fn, env_config=EnvConfig())
    vec_obs = vec.reset(seed=seed)
    assert vec_obs.shape == (1, bare.observation_space.shape[0])
    assert np.array_equal(vec_obs[0], bare_obs)

    for step_i, a in enumerate(actions):
        o, r, term, trunc, infos = vec.step(a.reshape(1, 4))
        b_o, b_r, b_term, b_trunc, b_term_name, b_s = bare_traj[step_i + 1]
        if b_term or b_trunc:
            # Auto-reset returns a fresh obs; the ending obs is in info.
            assert term[0] == b_term and trunc[0] == b_trunc
            assert r[0] == pytest.approx(b_r)
            assert np.array_equal(infos[0]["final_observation"], b_o)
            assert infos[0].get("termination") == b_term_name
            break
        assert np.array_equal(o[0], b_o)
        assert float(r[0]) == pytest.approx(b_r)
        assert bool(term[0]) is b_term and bool(trunc[0]) is b_trunc
        assert float(infos[0]["s"]) == pytest.approx(b_s)
    else:
        # Trajectory did not end; every step matched above.
        pass

    vec.close()


def test_sync_auto_reset_puts_final_observation_in_info():
    """GAE truncation bootstrap needs the ending obs, not the post-reset one."""
    # Short timeout so we hit truncated without waiting 180 s.
    cfg = EnvConfig(max_time_s=0.2)  # 6 control steps at 30 Hz
    stage = proving_ground()
    vec = SyncVecEnv(1, _fixed_stage_fn(stage), env_config=cfg)
    obs0 = vec.reset(seed=1)
    done = False
    last_pre_reset = None
    for _ in range(30):
        o, _r, term, trunc, infos = vec.step(
            np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
        )
        if term[0] or trunc[0]:
            assert "final_observation" in infos[0]
            final = infos[0]["final_observation"]
            assert final.shape == obs0.shape[1:]
            # Obs array must be the NEW episode, not the ending frame.
            assert not np.array_equal(o[0], final)
            last_pre_reset = final
            done = True
            break
    assert done
    assert last_pre_reset is not None
    vec.close()


def test_sync_subenvs_use_independent_seed_streams():
    """Identical seeds across sub-envs would overfit one stage n ways."""
    vec = SyncVecEnv(4, lambda s: generate(s, 0))
    obs = vec.reset(seed=100)
    # Different stages (or at least different start observations) across envs.
    distinct = {obs[i].tobytes() for i in range(4)}
    assert len(distinct) == 4
    vec.close()


def test_sync_reset_draws_new_stage_for_training():
    """Training resets must call stage_fn again, not replay the same document."""
    calls: list[int] = []

    def counting_fn(seed: int) -> dict:
        calls.append(int(seed))
        return generate(seed, 0)

    vec = SyncVecEnv(1, counting_fn)
    # Construction bootstraps once with seed 0.
    assert calls == [0]
    vec.reset(seed=5)
    assert 5 in calls
    n_after_reset = len(calls)

    # Force a quick episode end and auto-reset.
    cfg_vec = SyncVecEnv(1, counting_fn, env_config=EnvConfig(max_time_s=0.15))
    cfg_vec.reset(seed=9)
    for _ in range(20):
        _, _, term, trunc, _ = cfg_vec.step(
            np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        )
        if term[0] or trunc[0]:
            break
    assert len(calls) > n_after_reset
    cfg_vec.close()
    vec.close()


def test_sync_set_tier_reaches_default_generator():
    vec = SyncVecEnv(1, tier=0)
    assert vec._tier == 0
    vec.set_tier(2)
    assert vec._tier == 2
    assert vec._stage_fn.state["tier"] == 2  # type: ignore[attr-defined]
    vec.close()


def test_sync_step_rejects_wrong_action_shape():
    vec = SyncVecEnv(2, _fixed_stage_fn(proving_ground()))
    vec.reset(seed=0)
    with pytest.raises(ValueError, match="actions shape"):
        vec.step(np.zeros((2, 3), dtype=np.float32))
    vec.close()


# --------------------------------------------------------------------------- #
# AsyncVecEnv
# --------------------------------------------------------------------------- #

def test_async_reset_and_step_shapes():
    env = AsyncVecEnv(2, envs_per_worker=1, tier=0, stall_timeout_s=60.0)
    try:
        obs = env.reset(seed=11)
        assert obs.shape == (2, 84)
        actions = np.zeros((2, 4), dtype=np.float32)
        actions[:, 1] = 0.7
        o, r, term, trunc, infos = env.step(actions)
        assert o.shape == (2, 84)
        assert r.shape == (2,)
        assert term.shape == (2,) and trunc.shape == (2,)
        assert len(infos) == 2
    finally:
        env.close()


def test_async_auto_reset_carries_final_observation():
    env = AsyncVecEnv(
        1,
        env_config=EnvConfig(max_time_s=0.2),
        tier=0,
        stall_timeout_s=60.0,
    )
    try:
        env.reset(seed=3)
        saw = False
        for _ in range(40):
            _o, _r, term, trunc, infos = env.step(
                np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
            )
            if term[0] or trunc[0]:
                assert "final_observation" in infos[0]
                assert infos[0]["final_observation"].shape == (84,)
                saw = True
                break
        assert saw
    finally:
        env.close()


def test_async_sigkill_revives_and_completes(tmp_path):
    """Killing a worker mid-run must log died/revived and finish the rollout."""
    metrics = tmp_path / "metrics.jsonl"
    env = AsyncVecEnv(
        2,
        tier=0,
        metrics_path=metrics,
        run_id="revive-test",
        stall_timeout_s=15.0,
    )
    try:
        env.reset(seed=21)
        pid = env.procs[0].pid
        assert pid is not None
        import os
        os.kill(pid, 0)  # confirm alive; raises if not
        # SIGKILL — the failure mode overnight runs actually see.
        os.kill(pid, signal.SIGKILL)
        env.procs[0].join(timeout=5.0)

        actions = np.zeros((env.n, 4), dtype=np.float32)
        actions[:, 1] = 0.6
        for _ in range(8):
            env.step(actions)
    finally:
        env.close()

    lines = [json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    worker_lines = [line for line in lines if line.get("kind") == "worker"]
    events = [line["worker"]["event"] for line in worker_lines]
    assert "died" in events, events
    assert "revived" in events, events
    assert all(line["run_id"] == "revive-test" for line in worker_lines)


def test_async_uses_spawn_start_method():
    env = AsyncVecEnv(1, stall_timeout_s=60.0)
    try:
        assert env._ctx.get_start_method() == "spawn"
    finally:
        env.close()
