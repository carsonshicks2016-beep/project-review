"""Supervised imitation utilities for the first trainable policy workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .observations import OBS_SIZE, build_observation
from .sim import HeistSim


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("Install torch to train imitation policies") from exc
    return torch, nn


def control_to_action(control: tuple[float, float, float, float]) -> np.ndarray:
    steer, throttle, brake, handbrake = control
    return np.asarray([
        steer,
        throttle - brake,
        handbrake * 2.0 - 1.0,
    ], dtype=np.float32)


class MLPPolicy(require_torch()[1].Module):
    def __init__(self, obs_dim: int = OBS_SIZE, hidden: int = 128, action_dim: int = 3):
        _, nn = require_torch()
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


@dataclass
class ImitationResult:
    checkpoint: str
    samples: int
    epochs: int
    final_loss: float


def collect_evader_dataset(seed: int = 11, steps: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    sim = HeistSim(seed=seed, reset_on_capture=True)
    obs_rows = []
    act_rows = []
    for _ in range(steps):
        obs = build_observation(sim, "evader_0")
        control = sim.evader_planner.control(sim.evader.vehicle, [p.vehicle for p in sim.pursuers])
        obs_rows.append(obs)
        act_rows.append(control_to_action(control))
        sim.step()
    return np.stack(obs_rows).astype(np.float32), np.stack(act_rows).astype(np.float32)


def collect_evader_dagger_dataset(
    model: MLPPolicy,
    *,
    seed: int = 11,
    steps: int = 1200,
    policy_fraction: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect expert labels on states visited by the learned policy."""
    torch, _ = require_torch()
    sim = HeistSim(seed=seed, reset_on_capture=True)
    rng = np.random.default_rng(seed)
    obs_rows = []
    act_rows = []
    model.eval()
    for _ in range(steps):
        obs = build_observation(sim, "evader_0")
        expert = sim.evader_planner.control(sim.evader.vehicle, [p.vehicle for p in sim.pursuers])
        obs_rows.append(obs)
        act_rows.append(control_to_action(expert))
        if rng.random() < policy_fraction:
            with torch.no_grad():
                pred = model(torch.as_tensor(obs[None, :], dtype=torch.float32))[0]
            action = np.clip(pred.detach().cpu().numpy().astype(np.float32), -1.0, 1.0)
            sim.step(actions={"evader_0": {"control": action}})
        else:
            sim.step()
    return np.stack(obs_rows).astype(np.float32), np.stack(act_rows).astype(np.float32)


def train_evader_imitation(
    out: str | Path,
    seed: int = 11,
    steps: int = 4000,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    hidden: int = 128,
    dagger_rounds: int = 0,
    dagger_steps: int = 1200,
    dagger_policy_fraction: float = 1.0,
) -> ImitationResult:
    torch, nn = require_torch()
    torch.manual_seed(seed)
    obs, actions = collect_evader_dataset(seed=seed, steps=steps)
    model = MLPPolicy(obs_dim=obs.shape[1], hidden=hidden, action_dim=actions.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    rng = np.random.default_rng(seed)
    final_loss = 0.0
    for round_idx in range(max(0, int(dagger_rounds)) + 1):
        x = torch.as_tensor(obs)
        y = torch.as_tensor(actions)
        model.train()
        for _ in range(epochs):
            order = rng.permutation(len(obs))
            for start in range(0, len(obs), batch_size):
                idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long)
                pred = model(x[idx])
                loss = loss_fn(pred, y[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                final_loss = float(loss.detach().cpu())
        if round_idx < dagger_rounds:
            dagger_obs, dagger_actions = collect_evader_dagger_dataset(
                model,
                seed=seed + round_idx,
                steps=dagger_steps,
                policy_fraction=dagger_policy_fraction,
            )
            obs = np.concatenate([obs, dagger_obs], axis=0)
            actions = np.concatenate([actions, dagger_actions], axis=0)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "evader_imitation",
        "obs_dim": obs.shape[1],
        "action_dim": actions.shape[1],
        "hidden": hidden,
        "seed": seed,
        "steps": steps,
        "epochs": epochs,
        "dagger_rounds": int(dagger_rounds),
        "dagger_steps": int(dagger_steps),
        "dagger_policy_fraction": float(dagger_policy_fraction),
        "samples": int(len(obs)),
        "state_dict": model.state_dict(),
        "final_loss": final_loss,
    }, out)
    return ImitationResult(str(out), len(obs), epochs, final_loss)


def load_mlp_policy(path: str | Path):
    torch, _ = require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = MLPPolicy(obs_dim=ckpt["obs_dim"], hidden=ckpt["hidden"], action_dim=ckpt["action_dim"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
