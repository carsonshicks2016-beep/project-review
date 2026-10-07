"""C1 — Gaussian actor-critic with asymmetric action squash."""

from __future__ import annotations

import time

import numpy as np
import torch

from rallyai.train.policy import (
    ACT_DIM,
    OBS_DIM,
    ActorCritic,
    squash_actions,
    unsquash_actions,
)


def test_forward_shapes():
    net = ActorCritic()
    obs = torch.randn(8, OBS_DIM)
    mean, value = net(obs)
    assert mean.shape == (8, ACT_DIM)
    assert value.shape == (8,)


def test_log_std_init_and_clamp():
    net = ActorCritic(init_log_std=-0.5)
    assert torch.allclose(net.log_std, torch.full((ACT_DIM,), -0.5))
    with torch.no_grad():
        net.log_std.fill_(5.0)
    net.clamp_log_std_()
    assert float(net.log_std.detach().max()) <= 0.0 + 1e-6
    with torch.no_grad():
        net.log_std.fill_(-10.0)
    net.clamp_log_std_()
    assert float(net.log_std.detach().min()) >= -2.2 - 1e-6


def test_asymmetric_action_bounds():
    net = ActorCritic()
    obs = torch.randn(64, OBS_DIM)
    out = net.act(obs)
    a = out.actions.detach().numpy()
    assert a.shape == (64, ACT_DIM)
    assert np.all(a[:, 0] >= -1.0 - 1e-5) and np.all(a[:, 0] <= 1.0 + 1e-5)
    assert np.all(a[:, 1:] >= -1e-5) and np.all(a[:, 1:] <= 1.0 + 1e-5)


def test_squash_roundtrip():
    z = torch.randn(32, ACT_DIM)
    a = squash_actions(z)
    z2 = unsquash_actions(a)
    a2 = squash_actions(z2)
    assert torch.allclose(a, a2, atol=1e-5)


def test_log_prob_matches_sampled_distribution():
    torch.manual_seed(0)
    net = ActorCritic()
    obs = torch.randn(16, OBS_DIM)
    out = net.act(obs)
    logp2, ent, value = net.evaluate(obs, out.actions)
    assert torch.allclose(out.log_prob, logp2, atol=1e-4)
    assert ent.shape == (16,)
    assert value.shape == (16,)


def test_ortho_init_policy_head_small():
    net = ActorCritic()
    # Policy head gain 0.01 → small weights; value head larger.
    assert float(net.mean_head.weight.detach().abs().mean()) < 0.05
    assert float(net.value_head.weight.detach().abs().mean()) > float(
        net.mean_head.weight.detach().abs().mean()
    )


def test_forward_timing_smoke():
    net = ActorCritic()
    net.eval()
    for batch in (8, 4096):
        obs = torch.randn(batch, OBS_DIM)
        t0 = time.perf_counter()
        with torch.no_grad():
            net.act(obs)
        elapsed = time.perf_counter() - t0
        # Sanity: should finish well under a second on CPU.
        assert elapsed < 5.0, f"batch {batch} took {elapsed:.3f}s"
