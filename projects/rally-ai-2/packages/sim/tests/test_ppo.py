"""C3 — PPO update determinism, KL rollback, diagnostics."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from rallyai.train.policy import LOG_STD_MIN, ActorCritic
from rallyai.train.ppo import PPO, PPOConfig


def _fake_batch(n: int = 4096, obs_dim: int = 84, act_dim: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = rng.standard_normal((n, obs_dim), dtype=np.float32)
    actions = rng.uniform(-1, 1, size=(n, act_dim)).astype(np.float32)
    actions[:, 1:] = (actions[:, 1:] + 1.0) * 0.5
    logp = rng.standard_normal(n).astype(np.float32) * 0.1
    adv = rng.standard_normal(n).astype(np.float32)
    ret = adv + rng.standard_normal(n).astype(np.float32)
    old_v = ret + rng.standard_normal(n).astype(np.float32) * 0.1
    return obs, actions, logp, adv, ret, old_v


def _seeded_ppo(seed: int, cfg: PPOConfig) -> PPO:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return PPO(ActorCritic(), config=cfg)


def test_ppo_update_deterministic():
    cfg = PPOConfig(epochs=2, minibatch=512, target_kl=10.0, max_grad_norm=0.5)
    batch = _fake_batch()

    ppo_a = _seeded_ppo(0, cfg)
    before = copy.deepcopy(ppo_a.policy.state_dict())
    torch.manual_seed(7)
    np.random.seed(7)
    stats_a = ppo_a.update(*batch)

    ppo_b = _seeded_ppo(0, cfg)
    for k, v in before.items():
        assert torch.allclose(ppo_b.policy.state_dict()[k], v)
    torch.manual_seed(7)
    np.random.seed(7)
    stats_b = ppo_b.update(*batch)

    assert abs(stats_a.policy_loss - stats_b.policy_loss) < 1e-5
    assert abs(stats_a.value_loss - stats_b.value_loss) < 1e-5
    assert abs(stats_a.entropy - stats_b.entropy) < 1e-5
    assert abs(stats_a.approx_kl - stats_b.approx_kl) < 1e-5
    assert abs(stats_a.clip_frac - stats_b.clip_frac) < 1e-5
    assert np.isfinite(stats_a.explained_var)
    assert stats_a.rejected is False


def test_kl_rollback_restores_params_and_opt():
    cfg = PPOConfig(epochs=4, minibatch=256, target_kl=1e-12, lr=1e-2)
    ppo = _seeded_ppo(0, cfg)
    before_net = copy.deepcopy(ppo.policy.state_dict())
    before_opt = copy.deepcopy(ppo.opt.state_dict())
    stats = ppo.update(*_fake_batch(n=1024, seed=3))
    assert stats.rejected is True
    for k, v in before_net.items():
        assert torch.allclose(ppo.policy.state_dict()[k], v)
    assert ppo.opt.state_dict()["state"].keys() == before_opt["state"].keys()


def test_grad_norm_recorded():
    ppo = _seeded_ppo(0, PPOConfig(epochs=1, minibatch=512, target_kl=10.0))
    stats = ppo.update(*_fake_batch(n=512))
    assert stats.grad_norm >= 0.0
    assert np.isfinite(stats.grad_norm)


def test_kl_early_stop_truncates_the_epoch_loop():
    """The mechanism: fewer minibatch steps applied, same batch and seed.

    Deliberately *not* asserting that early stopping converts a rejected update
    into an accepted one. On synthetic advantages a single minibatch drives
    approx_kl to ~4, so both arms roll back regardless and the assertion would
    pass or fail for reasons unrelated to the guard. Whether it recovers real
    updates is an empirical question for a training A/B, not a unit test.
    """
    batch = _fake_batch(n=1024, seed=3)
    kwargs = {"epochs": 8, "minibatch": 256, "target_kl": 1e-4, "lr": 1e-2}
    full = 8 * 4  # epochs * ceil(1024 / 256)

    strict = _seeded_ppo(0, PPOConfig(**kwargs))
    torch.manual_seed(7)
    np.random.seed(7)
    strict_stats = strict.update(*batch)

    early = _seeded_ppo(0, PPOConfig(**kwargs, kl_early_stop=True))
    torch.manual_seed(7)
    np.random.seed(7)
    early_stats = early.update(*batch)

    assert strict_stats.early_stopped is False
    assert strict_stats.n_minibatches == full

    # Exactly one: kl_last is measured at the top of each minibatch, so the
    # first step lands and the second is refused. Asserting `< full` instead
    # would also pass if the loop merely finished the current epoch first,
    # which is a different and much weaker guarantee — checked by breaking the
    # inner `break` on purpose and confirming this line fails.
    assert early_stats.early_stopped is True
    assert early_stats.n_minibatches == 1


def test_kl_early_stop_is_off_by_default():
    assert PPOConfig().kl_early_stop is False
    ppo = _seeded_ppo(0, PPOConfig(epochs=2, minibatch=512, target_kl=1e-9, lr=1e-2))
    stats = ppo.update(*_fake_batch(n=512))
    assert stats.early_stopped is False
    assert stats.n_minibatches == 2 * 1


def test_log_std_max_caps_each_dim_where_it_is_set():
    """A lower ceiling must actually bind — this is the foundation_01 lever."""
    policy = ActorCritic(log_std_max=(-0.7, -0.7, -0.7, -0.7))
    with torch.no_grad():
        policy.log_std.data.fill_(5.0)
    policy.clamp_log_std_()
    assert policy._bounded_log_std().tolist() == pytest.approx([-0.7] * 4)

    default = ActorCritic()
    with torch.no_grad():
        default.log_std.data.fill_(5.0)
    default.clamp_log_std_()
    assert default._bounded_log_std().tolist() == pytest.approx([0.0] * 4)


def test_log_std_reported_per_dim_and_within_bounds():
    ppo = _seeded_ppo(0, PPOConfig(epochs=1, minibatch=512, target_kl=10.0))
    stats = ppo.update(*_fake_batch(n=512))
    assert len(stats.log_std) == ppo.policy.act_dim
    assert all(np.isfinite(x) for x in stats.log_std)
    for x, cap in zip(stats.log_std, ppo.policy.log_std_max):
        assert LOG_STD_MIN <= x <= cap


def test_log_std_reports_a_dim_pinned_at_its_cap():
    """The foundation_01 failure mode, made visible from the stream alone.

    Three of four dims finished that run flat against ``log_std_max`` and the
    only way to see it was to open the checkpoints afterwards. Drive one dim
    onto its cap on purpose and assert the reported value shows it.
    """
    ppo = _seeded_ppo(0, PPOConfig(epochs=1, minibatch=512, target_kl=10.0))
    with torch.no_grad():
        ppo.policy.log_std.data[1] = 5.0  # far above the cap
    ppo.policy.clamp_log_std_()

    stats = ppo.update(*_fake_batch(n=512))

    cap = ppo.policy.log_std_max[1]
    assert stats.log_std[1] == cap, "a pinned dim must report as pinned"
    assert stats.log_std[0] < cap, "an unpinned dim must not read as pinned"


def test_log_std_reported_is_live_not_the_pre_update_snapshot():
    """An accepted update must report where log_std ended up, not where it began.

    Reporting the pre-update snapshot is the plausible wiring bug here, and it
    is invisible to any assertion that only checks bounds or length — a pinned
    dim reads pinned either way. Pin it down by requiring the reported value to
    have moved.
    """
    cfg = PPOConfig(epochs=4, minibatch=256, target_kl=10.0, lr=1e-2)
    ppo = _seeded_ppo(0, cfg)
    before = tuple(float(x) for x in ppo.policy._bounded_log_std().detach())

    stats = ppo.update(*_fake_batch(n=1024, seed=3))

    assert stats.rejected is False
    live = tuple(float(x) for x in ppo.policy._bounded_log_std().detach())
    assert stats.log_std == live
    assert stats.log_std != before, "reported log_std never moved off the snapshot"


def test_log_std_reported_after_rollback_is_the_restored_value():
    """A rejected update reports the exploration scale that is actually live."""
    cfg = PPOConfig(epochs=4, minibatch=256, target_kl=1e-12, lr=1e-2)
    ppo = _seeded_ppo(0, cfg)
    before = ppo.policy._bounded_log_std().detach().clone()
    stats = ppo.update(*_fake_batch(n=1024, seed=3))
    assert stats.rejected is True
    assert stats.log_std == tuple(float(x) for x in before)
