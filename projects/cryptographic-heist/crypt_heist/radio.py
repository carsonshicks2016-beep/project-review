"""Learned pursuer radio-token policy and runtime helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .comms import VOCABULARY, WORD_TO_TOKEN
from .imitation import require_torch
from .observations import OBS_SIZE, build_observation
from .sim import HeistSim

MESSAGE_LEN = 6
PURSUER_NAMES = [f"pursuer_{i}" for i in range(5)]


class RadioTokenPolicy(require_torch()[1].Module):
    """Predict a fixed-length radio token sequence from a pursuer observation."""

    def __init__(
        self,
        obs_dim: int = OBS_SIZE,
        vocab_size: int = len(VOCABULARY),
        message_len: int = MESSAGE_LEN,
        hidden: int = 128,
        agent_embed: int = 16,
    ):
        torch, nn = require_torch()
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.vocab_size = int(vocab_size)
        self.message_len = int(message_len)
        self.hidden = int(hidden)
        self.agent_embedding = nn.Embedding(5, agent_embed)
        self.net = nn.Sequential(
            nn.Linear(obs_dim + agent_embed, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, message_len * vocab_size),
        )

    def forward(self, obs, agent_id):
        emb = self.agent_embedding(agent_id)
        logits = self.net(self._torch_cat(obs, emb))
        return logits.reshape(obs.shape[0], self.message_len, self.vocab_size)

    def relaxed_tokens(self, obs, agent_id, tau: float = 0.9, hard: bool = True):
        import torch.nn.functional as functional

        if tau <= 0.0:
            raise ValueError("Gumbel-Softmax temperature tau must be positive")
        logits = self.forward(obs, agent_id)
        flat = logits.reshape(-1, self.vocab_size)
        relaxed = functional.gumbel_softmax(flat, tau=tau, hard=hard)
        return relaxed.reshape(obs.shape[0], self.message_len, self.vocab_size)

    @staticmethod
    def _torch_cat(obs, emb):
        torch, _ = require_torch()
        return torch.cat([obs, emb], dim=-1)


@dataclass
class RadioTrainingResult:
    checkpoint: str
    samples: int
    epochs: int
    final_loss: float
    token_accuracy: float
    stats: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "samples": self.samples,
            "epochs": self.epochs,
            "final_loss": self.final_loss,
            "token_accuracy": self.token_accuracy,
            "stats": self.stats,
        }


def collect_radio_dataset(seed: int = 11, steps: int = 3600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect observations and scripted radio payloads from pursuer broadcasts."""
    sim = HeistSim(seed=seed, reset_on_capture=True)
    obs_rows: list[np.ndarray] = []
    agent_rows: list[int] = []
    token_rows: list[np.ndarray] = []
    seen = len(sim.channel.events)
    for _ in range(int(steps)):
        pre_obs = {name: build_observation(sim, name) for name in PURSUER_NAMES}
        sim.step()
        new_events = sim.channel.events[seen:]
        seen = len(sim.channel.events)
        by_speaker: dict[str, list[int]] = {}
        for event in new_events:
            if not event.speaker.startswith("P") or event.spoofed:
                continue
            token = WORD_TO_TOKEN.get(event.word)
            if token is None:
                continue
            by_speaker.setdefault(event.speaker, []).append(token)
        for speaker, tokens in by_speaker.items():
            if len(tokens) < MESSAGE_LEN:
                continue
            idx = int(speaker[1:]) - 1
            agent_name = f"pursuer_{idx}"
            obs_rows.append(pre_obs[agent_name])
            agent_rows.append(idx)
            token_rows.append(np.asarray(tokens[:MESSAGE_LEN], dtype=np.int64))
    if not obs_rows:
        raise RuntimeError("radio dataset collection produced no broadcasts")
    return (
        np.stack(obs_rows).astype(np.float32),
        np.asarray(agent_rows, dtype=np.int64),
        np.stack(token_rows).astype(np.int64),
    )


def train_radio_policy(
    out: str | Path,
    *,
    seed: int = 11,
    steps: int = 3600,
    epochs: int = 12,
    batch_size: int = 128,
    lr: float = 3e-4,
    hidden: int = 128,
    device: str = "cpu",
) -> RadioTrainingResult:
    torch, nn = require_torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs, agent_ids, tokens = collect_radio_dataset(seed=seed, steps=steps)
    model = RadioTokenPolicy(hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    x = torch.as_tensor(obs, dtype=torch.float32, device=device)
    a = torch.as_tensor(agent_ids, dtype=torch.long, device=device)
    y = torch.as_tensor(tokens, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    final_loss = 0.0
    token_accuracy = 0.0
    stats: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(obs))
        losses = []
        correct = 0
        total = 0
        for start in range(0, len(obs), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], dtype=torch.long, device=device)
            logits = model(x[idx], a[idx])
            loss = loss_fn(logits.reshape(-1, len(VOCABULARY)), y[idx].reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            final_loss = float(loss.detach().cpu())
            losses.append(final_loss)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y[idx]).sum().detach().cpu())
            total += int(y[idx].numel())
        token_accuracy = correct / max(1, total)
        stats.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else final_loss,
            "token_accuracy": float(token_accuracy),
            "samples": int(len(obs)),
        })

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "kind": "pursuer_radio_policy",
        "obs_dim": OBS_SIZE,
        "vocab_size": len(VOCABULARY),
        "message_len": MESSAGE_LEN,
        "hidden": hidden,
        "seed": seed,
        "steps": steps,
        "epochs": epochs,
        "state_dict": model.state_dict(),
        "final_loss": final_loss,
        "token_accuracy": token_accuracy,
        "stats": stats,
    }, out)
    return RadioTrainingResult(
        checkpoint=str(out),
        samples=len(obs),
        epochs=epochs,
        final_loss=final_loss,
        token_accuracy=token_accuracy,
        stats=stats,
    )


def load_radio_policy(path: str | Path):
    torch, _ = require_torch()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if ckpt.get("kind") != "pursuer_radio_policy":
        raise ValueError(f"unsupported radio checkpoint kind: {ckpt.get('kind')!r}")
    model = RadioTokenPolicy(
        obs_dim=int(ckpt["obs_dim"]),
        vocab_size=int(ckpt["vocab_size"]),
        message_len=int(ckpt["message_len"]),
        hidden=int(ckpt["hidden"]),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


class PursuerRadioController:
    """Runtime that supplies learned radio tokens while scripted driving remains active."""

    def __init__(self, checkpoint: str | Path):
        self.model, self.metadata = load_radio_policy(checkpoint)
        self.torch, _ = require_torch()
        self.agent_names = PURSUER_NAMES[:]

    def action(self, sim: HeistSim) -> dict[str, dict[str, np.ndarray]]:
        obs = np.stack([build_observation(sim, name) for name in self.agent_names]).astype(np.float32)
        agent_ids = np.asarray([int(name.split("_")[1]) for name in self.agent_names], dtype=np.int64)
        with self.torch.no_grad():
            x = self.torch.as_tensor(obs, dtype=self.torch.float32)
            a = self.torch.as_tensor(agent_ids, dtype=self.torch.long)
            logits = self.model(x, a)
            tokens = logits.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)
        return {
            name: {"tokens": tokens[i]}
            for i, name in enumerate(self.agent_names)
        }
