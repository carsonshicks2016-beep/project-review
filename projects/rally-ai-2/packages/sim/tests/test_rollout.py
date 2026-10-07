"""C2 — GAE truncation bootstrap vs true termination."""

from __future__ import annotations

import numpy as np

from rallyai.train.rollout import RolloutBuffer, compute_gae_from_arrays


def _hand_gae_two_step(
    *,
    rewards,
    values,
    next_values_at_end,
    gamma,
    gae_lambda,
):
    """Analytic GAE for a single env, two steps, episode ends at t=1."""
    # t=1 (last): next_nonterminal=0, delta = r1 + gamma*V_next*1_if_not_term - V1
    # For terminated V_next effective contribution uses not_terminated mask.
    r0, r1 = rewards
    v0, v1 = values
    # Episode ends at step 1 → GAE recursion cuts after t=1.
    # At t=1: not_terminated depends on whether we treat as terminate.
    # next_values_at_end already encodes 0 (term) or V_final (trunc).
    # Using not_terminated=1 for trunc and 0 for term is folded into next_values
    # for term (0) — for trunc next_values is V_final with not_terminated=1.
    delta1 = r1 + gamma * next_values_at_end - v1
    a1 = delta1
    # t=0 continues into t=1 (not episode end at t=0)
    delta0 = r0 + gamma * v1 - v0
    a0 = delta0 + gamma * gae_lambda * a1
    return np.array([a0, a1], dtype=np.float32)


def test_terminated_vs_truncated_gae_differ():
    gamma = 0.995
    gae_lambda = 0.95
    rewards = np.array([[1.0], [2.0]], dtype=np.float32)
    values = np.array([[0.5], [0.8]], dtype=np.float32)
    last_values = np.array([9.9], dtype=np.float32)  # post-reset; must be ignored
    v_final = 3.0

    # Terminated: bootstrap 0
    term = np.array([[False], [True]])
    trunc = np.array([[False], [False]])
    boot = np.array([[0.0], [0.0]], dtype=np.float32)
    adv_term, ret_term = compute_gae_from_arrays(
        rewards, values, term, trunc, last_values, boot,
        gamma=gamma, gae_lambda=gae_lambda,
    )

    # Truncated: bootstrap V(final)
    term2 = np.array([[False], [False]])
    trunc2 = np.array([[False], [True]])
    boot2 = np.array([[0.0], [v_final]], dtype=np.float32)
    adv_trunc, ret_trunc = compute_gae_from_arrays(
        rewards, values, term2, trunc2, last_values, boot2,
        gamma=gamma, gae_lambda=gae_lambda,
    )

    assert not np.allclose(adv_term, adv_trunc)

    expected_term = _hand_gae_two_step(
        rewards=(1.0, 2.0),
        values=(0.5, 0.8),
        next_values_at_end=0.0,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    # For terminated, delta1 = r1 + gamma*0*not_term - v1, but our code uses
    # not_terminated=0 so delta1 = r1 - v1 (no gamma term).
    expected_term[1] = 2.0 - 0.8
    expected_term[0] = (1.0 + gamma * 0.8 - 0.5) + gamma * gae_lambda * expected_term[1]

    expected_trunc = _hand_gae_two_step(
        rewards=(1.0, 2.0),
        values=(0.5, 0.8),
        next_values_at_end=v_final,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )

    np.testing.assert_allclose(adv_term[:, 0], expected_term, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(adv_trunc[:, 0], expected_trunc, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ret_term, adv_term + values, rtol=1e-5)
    np.testing.assert_allclose(ret_trunc, adv_trunc + values, rtol=1e-5)


def test_rollout_buffer_shapes():
    buf = RolloutBuffer(n_steps=4, n_envs=2, obs_dim=84, act_dim=4)
    for t in range(4):
        buf.add(
            obs=np.zeros((2, 84), np.float32),
            actions=np.zeros((2, 4), np.float32),
            log_probs=np.zeros(2, np.float32),
            rewards=np.ones(2, np.float32),
            values=np.zeros(2, np.float32),
            terminated=np.zeros(2, bool),
            truncated=np.zeros(2, bool),
        )
    batch = buf.compute_gae(np.zeros(2, np.float32))
    assert batch.advantages.shape == (4, 2)
    assert batch.returns.shape == (4, 2)
