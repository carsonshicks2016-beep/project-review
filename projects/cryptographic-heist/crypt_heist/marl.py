"""PyTorch policy modules for the differentiable comms layer."""

from __future__ import annotations


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise RuntimeError("Install torch to use crypt_heist.marl") from exc
    return torch, nn, F


class PursuerPolicy(require_torch()[1].Module):
    """Shared pursuer policy with recurrent memory and token output."""

    def __init__(self, obs_dim: int, vocab_size: int = 1000, hidden: int = 192, message_len: int = 6):
        torch, nn, _ = require_torch()
        super().__init__()
        self.nn = nn
        self.vocab_size = int(vocab_size)
        self.message_len = int(message_len)
        self.agent_embed = nn.Embedding(5, 16)
        self.in_proj = nn.Linear(obs_dim + 16, hidden)
        self.recurrent = nn.GRU(hidden, hidden, batch_first=True)
        self.control = nn.Linear(hidden, 3)
        self.words = nn.Linear(hidden, self.message_len * self.vocab_size)
        self.value = nn.Linear(hidden, 1)
        self.intent = nn.Linear(hidden, 5)

    def forward(self, obs_seq, agent_id=None, hidden=None, tau: float = 0.9):
        torch, _, F = require_torch()
        if tau <= 0.0:
            raise ValueError("Gumbel-Softmax temperature tau must be positive")
        if agent_id is None:
            agent_id = torch.zeros(obs_seq.shape[0], dtype=torch.long, device=obs_seq.device)
        emb = self.agent_embed(agent_id).unsqueeze(1).expand(-1, obs_seq.shape[1], -1)
        x = torch.tanh(self.in_proj(torch.cat([obs_seq, emb], dim=-1)))
        z, hidden = self.recurrent(x, hidden)
        h = z[:, -1]
        controls = torch.tanh(self.control(h))
        logits = self.words(h).reshape(obs_seq.shape[0], self.message_len, self.vocab_size)
        relaxed = F.gumbel_softmax(
            logits.reshape(-1, self.vocab_size),
            tau=tau,
            hard=True,
        ).reshape(obs_seq.shape[0], self.message_len, self.vocab_size)
        value = self.value(h).squeeze(-1)
        intent = self.intent(h)
        return {
            "control": controls,
            "relaxed_tokens": relaxed,
            "token_logits": logits,
            "token_ids": logits.argmax(dim=-1),
            "value": value,
            "intent_logits": intent,
            "hidden": hidden,
        }


class EvaderPolicy(require_torch()[1].Module):
    """Evader policy with driver, scanner decoder, and jammer heads."""

    def __init__(self, spatial_dim: int, comm_dim: int, hidden: int = 192):
        torch, nn, _ = require_torch()
        super().__init__()
        self.nn = nn
        self.spatial = nn.Sequential(nn.Linear(spatial_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.comms = nn.GRU(comm_dim, hidden, batch_first=True)
        self.driver = nn.Linear(hidden * 2, 3)
        self.decoder = nn.Linear(hidden, 10)
        self.jammer = nn.Linear(hidden * 2, 2)
        self.spoof_words = nn.Linear(hidden * 2, 5 * comm_dim)
        self.value_survival = nn.Linear(hidden * 2, 1)
        self.value_info = nn.Linear(hidden * 2, 1)

    def forward(self, spatial_obs, comm_seq, hidden=None):
        torch, _, _ = require_torch()
        s = self.spatial(spatial_obs)
        cseq, hidden = self.comms(comm_seq, hidden)
        c = cseq[:, -1]
        fused = torch.cat([s, c], dim=-1)
        controls = torch.tanh(self.driver(fused))
        future_xy = self.decoder(c).reshape(-1, 5, 2)
        jam_logits = self.jammer(fused)
        spoof_logits = self.spoof_words(fused).reshape(spatial_obs.shape[0], 5, comm_seq.shape[-1])
        return {
            "control": controls,
            "future_xy": future_xy,
            "jam_logits": jam_logits,
            "spoof_logits": spoof_logits,
            "value_survival": self.value_survival(fused).squeeze(-1),
            "value_info": self.value_info(fused).squeeze(-1),
            "hidden": hidden,
        }
