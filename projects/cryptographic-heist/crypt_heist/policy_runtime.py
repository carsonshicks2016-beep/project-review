"""Runtime helpers for loading checkpoints into scripted simulations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .imitation import MLPPolicy, require_torch
from .observations import build_observation
from .ppo import PPOActorCritic


class EvaderCheckpointController:
    def __init__(self, checkpoint: str | Path):
        self.torch, _ = require_torch()
        self.model, self.metadata = load_control_checkpoint(
            checkpoint,
            allowed_kinds={"evader_imitation", "evader_ppo"},
        )

    def action(self, sim) -> dict:
        control = _policy_control(self.model, self.torch, sim, "evader_0")
        return {
            "evader_0": {
                "control": np.clip(control, -1.0, 1.0),
                "jam": 0,
                "spoof_tokens": np.zeros(5, dtype=np.int64),
                "target_mask": np.zeros(5, dtype=np.int64),
            }
        }


class PursuerCheckpointController:
    def __init__(self, checkpoint: str | Path, agent_name: str | None = None):
        self.torch, _ = require_torch()
        self.model, self.metadata = load_control_checkpoint(
            checkpoint,
            allowed_kinds={"pursuer_ppo"},
        )
        self.agent_name = agent_name or self.metadata.get("agent_name", "pursuer_0")

    def action(self, sim) -> dict:
        control = _policy_control(self.model, self.torch, sim, self.agent_name)
        return {
            self.agent_name: {
                "control": np.clip(control, -1.0, 1.0),
            }
        }


class PursuerTeamCheckpointController:
    def __init__(self, checkpoint: str | Path, agent_names: list[str] | None = None):
        self.torch, _ = require_torch()
        self.model, self.metadata = load_control_checkpoint(
            checkpoint,
            allowed_kinds={"pursuer_team_ppo"},
        )
        self.agent_names = agent_names or self.metadata.get("agent_names", [f"pursuer_{i}" for i in range(5)])

    def action(self, sim) -> dict:
        return {
            name: {
                "control": np.clip(_policy_control(self.model, self.torch, sim, name), -1.0, 1.0),
            }
            for name in self.agent_names
        }


def load_evader_checkpoint(path: str | Path):
    return load_control_checkpoint(path, allowed_kinds={"evader_imitation", "evader_ppo"})


def load_control_checkpoint(path: str | Path, allowed_kinds: set[str] | None = None):
    torch, _ = require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    kind = ckpt.get("kind", "evader_imitation")
    if allowed_kinds is not None and kind not in allowed_kinds:
        allowed = ", ".join(sorted(allowed_kinds))
        raise ValueError(f"checkpoint kind {kind!r} is not valid here; expected one of: {allowed}")
    if kind in {"evader_ppo", "pursuer_ppo", "pursuer_team_ppo"}:
        model = PPOActorCritic(
            obs_dim=int(ckpt["obs_dim"]),
            hidden=int(ckpt["hidden"]),
            action_dim=int(ckpt["action_dim"]),
        )
    elif kind == "evader_imitation":
        model = MLPPolicy(
            obs_dim=int(ckpt["obs_dim"]),
            hidden=int(ckpt["hidden"]),
            action_dim=int(ckpt["action_dim"]),
        )
    else:
        raise ValueError(f"unsupported evader checkpoint kind: {kind}")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def _policy_control(model, torch, sim, agent_name: str) -> np.ndarray:
    obs = build_observation(sim, agent_name)
    with torch.no_grad():
        obs_t = torch.as_tensor(obs[None, :], dtype=torch.float32)
        if hasattr(model, "deterministic_action"):
            pred = model.deterministic_action(obs_t)[0]
        else:
            pred = model(obs_t)[0]
    return pred.detach().cpu().numpy().astype(np.float32)
