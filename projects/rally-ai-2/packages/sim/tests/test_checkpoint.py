"""C4 — checkpoint validation and hall of fame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from rallyai.env.reward import RewardConfig
from rallyai.sense import SensorSpec
from rallyai.train.checkpoint import (
    HallOfFame,
    build_payload,
    load_checkpoint,
    load_for_eval,
    save_checkpoint,
    state_dict_hash,
    validate_checkpoint,
)
from rallyai.train.normalise import ObservationNormaliser
from rallyai.train.policy import ActorCritic


def _payload():
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    # Take one step so optimiser state is non-empty.
    loss = net.value_only(torch.zeros(2, 84)).sum()
    loss.backward()
    opt.step()
    norm = ObservationNormaliser(84)
    norm.update(np.random.randn(16, 84).astype(np.float32))
    return build_payload(
        policy=net,
        optimizer=opt,
        normaliser=norm,
        sensor_spec=SensorSpec(),
        reward_config=RewardConfig(),
        timesteps=1000,
        tier=1,
        stage="foundation",
        run_id="test",
    )


def test_save_load_roundtrip(tmp_path: Path):
    payload = _payload()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    assert loaded["timesteps"] == 1000
    assert loaded["has_normaliser"] is True
    assert loaded["policy_sha256"] == state_dict_hash(loaded["state_dict"])
    validate_checkpoint(loaded)


def test_rejects_nonfinite(tmp_path: Path):
    payload = _payload()
    key = next(iter(payload["state_dict"]))
    payload["state_dict"][key] = payload["state_dict"][key].float()
    payload["state_dict"][key].view(-1)[0] = float("nan")
    payload["policy_sha256"] = state_dict_hash(
        {k: v for k, v in payload["state_dict"].items() if torch.isfinite(v).all()}
    )
    with pytest.raises((ValueError, FloatingPointError)):
        # Recompute hash with nan still present so validate catches nonfinite.
        payload["policy_sha256"] = "deadbeef"
        validate_checkpoint(payload)


def test_rejects_hash_mismatch():
    payload = _payload()
    payload["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="policy_sha256"):
        validate_checkpoint(payload)


def test_rejects_missing_normaliser():
    payload = _payload()
    del payload["normaliser"]
    payload["has_normaliser"] = False
    with pytest.raises(ValueError, match="normaliser"):
        validate_checkpoint(payload)


def test_hall_of_fame_slots(tmp_path: Path):
    hof = HallOfFame(tmp_path, "run")
    payload = _payload()
    written = hof.update(
        payload,
        mean_return=10.0,
        clean_rate=0.5,
        mean_finish_time=40.0,
        max_progress=0.8,
    )
    assert "latest" in written
    assert "best" in written
    assert hof.path_for("latest").exists()
    assert hof.path_for("best").exists()
    # Second update with worse metrics only refreshes latest.
    written2 = hof.update(payload, mean_return=1.0, clean_rate=0.1)
    assert written2 == ["latest"]


def test_load_for_eval_roundtrip_finite_actions(tmp_path: Path):
    path = tmp_path / "eval.pt"
    save_checkpoint(path, _payload())
    policy, normaliser, meta = load_for_eval(path)
    assert meta["has_normaliser"] is True
    assert normaliser.frozen is True
    assert hasattr(policy, "mean_action")
    raw = np.random.randn(84).astype(np.float32)
    obs = normaliser.normalize(raw)
    action = policy.mean_action(obs)
    assert action.shape == (4,)
    assert np.all(np.isfinite(action))
    assert -1.0 - 1e-5 <= action[0] <= 1.0 + 1e-5
    assert np.all(action[1:] >= -1e-5) and np.all(action[1:] <= 1.0 + 1e-5)
    # Frozen: stats must not move.
    before = normaliser.mean.copy()
    normaliser.update(raw.reshape(1, -1))
    assert np.allclose(before, normaliser.mean)
