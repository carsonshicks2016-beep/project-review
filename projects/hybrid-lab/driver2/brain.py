"""LSTM brain for the genetic algorithm.

A small recurrent network that processes the fable-v2 observation one timestep
at a time and outputs driving actions.  The hidden state gives the agent
temporal context — it can anticipate braking zones, learn energy management,
and maintain coherent driving strategy across segment boundaries.

Architecture:
    obs (70) → Linear+tanh (48) → LSTM (48) → Linear+tanh (3)
    Total: ~22 K parameters

The genome is the flat concatenation of every learnable parameter, exactly
like supra.brain.MLP.get_genome() — so the GA's crossover, mutation, and
elitism work unchanged.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import BrainConfig


class LSTMBrain(nn.Module):
    """Recurrent driving policy for GA evolution."""

    def __init__(self, cfg: BrainConfig | None = None):
        super().__init__()
        cfg = cfg or BrainConfig()
        self.cfg = cfg

        self.proj = nn.Linear(cfg.obs_size, cfg.proj_size)
        self.lstm = nn.LSTM(cfg.proj_size, cfg.hidden_size, batch_first=True)
        self.head = nn.Linear(cfg.hidden_size, cfg.n_actions)

        # Seed the longitudinal output bias so gen-0 brains accelerate
        # instead of sitting still.  Same idea as supra.brain.MLP.bias_long.
        with torch.no_grad():
            if cfg.n_actions >= 2:
                self.head.bias[1] = cfg.bias_long

    # -------------------------------------------------------------- #
    # forward (single-step, no batching — called at control_hz)
    # -------------------------------------------------------------- #
    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """One-step inference.

        Args:
            obs:    (obs_size,) or (1, obs_size) float tensor.
            hidden: (h, c) each (1, 1, hidden_size).  None → zeros.

        Returns:
            actions: (n_actions,) in [-1, 1].
            hidden:  updated (h, c) tuple.
        """
        if hidden is None:
            hidden = self.reset_hidden(obs.device)

        x = torch.tanh(self.proj(obs))           # (proj_size,)
        x = x.view(1, 1, -1)                     # (1, 1, proj_size)
        out, hidden = self.lstm(x, hidden)        # out (1, 1, hidden_size)
        actions = torch.tanh(self.head(out.squeeze(0).squeeze(0)))  # (n_actions,)
        return actions, hidden

    def reset_hidden(
        self, device: torch.device | str = "cpu"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Zero-initialised LSTM hidden state."""
        h = torch.zeros(1, 1, self.cfg.hidden_size, device=device)
        c = torch.zeros(1, 1, self.cfg.hidden_size, device=device)
        return (h, c)

    # -------------------------------------------------------------- #
    # genome encode / decode (flat float32 vector for the GA)
    # -------------------------------------------------------------- #
    @property
    def genome_size(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def get_genome(self) -> np.ndarray:
        """Flatten all parameters into a 1-D float32 vector."""
        return (
            nn.utils.parameters_to_vector(self.parameters())
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    def set_genome(self, genome: np.ndarray) -> "LSTMBrain":
        """Load parameters from a flat genome vector (in-place)."""
        vec = torch.from_numpy(genome.astype(np.float32))
        nn.utils.vector_to_parameters(vec, self.parameters())
        return self

    @classmethod
    def from_genome(
        cls, genome: np.ndarray, cfg: BrainConfig | None = None
    ) -> "LSTMBrain":
        """Construct a new brain and fill it from a genome vector."""
        brain = cls(cfg)
        brain.set_genome(genome)
        return brain

    def clone(self) -> "LSTMBrain":
        """Deep copy via genome round-trip."""
        return self.from_genome(self.get_genome(), self.cfg)

    # -------------------------------------------------------------- #
    # numpy convenience (GA workers don't need autograd)
    # -------------------------------------------------------------- #
    @torch.no_grad()
    def act_numpy(
        self,
        obs_np: np.ndarray,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor]]:
        """Numpy-in, numpy-out inference for GA evaluation loops."""
        obs = torch.from_numpy(obs_np.astype(np.float32))
        actions, hidden = self.forward(obs, hidden)
        return actions.cpu().numpy(), hidden
