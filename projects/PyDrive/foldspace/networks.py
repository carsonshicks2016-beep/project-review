"""Gaussian actor-critic for Foldspace.

Continuous-only (every action is a per-joint fold-angle nudge), exposing the same
``act`` / ``evaluate`` interface as the Duel network so it can reuse ``duel.ppo``.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def _layer(in_f, out_f, gain=np.sqrt(2)):
    lin = nn.Linear(in_f, out_f)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.constant_(lin.bias, 0.0)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
        )
        self.mean = _layer(hidden, act_dim, gain=0.01)
        self.value = _layer(hidden, 1, gain=1.0)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

    def _dist(self, feat):
        mean = self.mean(feat)
        return Normal(mean, torch.exp(self.log_std).expand_as(mean))

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        feat = self.trunk(obs)
        dist = self._dist(feat)
        action = dist.mean if deterministic else dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.value(feat).squeeze(-1)
        return action, logp, value

    def evaluate(self, obs, action):
        feat = self.trunk(obs)
        dist = self._dist(feat)
        logp = dist.log_prob(action).sum(-1)
        ent = dist.entropy().sum(-1)
        value = self.value(feat).squeeze(-1)
        return logp, ent, value
