"""Actor-critic network for Duel.

A shared MLP trunk feeds three heads:
  * a Gaussian over the 4 continuous controls (thrust x/y, aim x/y),
  * a Bernoulli pair over the 2 binary controls (fire, dash),
  * a scalar state value.

Continuous actions are left unsquashed — the environment clamps thrust to the
unit disc and normalises the aim vector — which keeps the PPO log-probabilities
simple and well-behaved.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli

N_CONT = 4
N_BIN = 2
ACT_DIM = N_CONT + N_BIN


def _layer(in_f, out_f, gain=np.sqrt(2)):
    lin = nn.Linear(in_f, out_f)
    nn.init.orthogonal_(lin.weight, gain)
    nn.init.constant_(lin.bias, 0.0)
    return lin


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
        )
        self.mean = _layer(hidden, N_CONT, gain=0.01)
        self.logits = _layer(hidden, N_BIN, gain=0.01)
        self.value = _layer(hidden, 1, gain=1.0)
        self.log_std = nn.Parameter(torch.full((N_CONT,), -0.5))

    def _dists(self, feat):
        mean = self.mean(feat)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std), Bernoulli(logits=self.logits(feat))

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        """obs: (B, obs_dim) tensor -> action (B, 6), logp (B), value (B)."""
        feat = self.trunk(obs)
        ncont, nbin = self._dists(feat)
        if deterministic:
            cont = ncont.mean
            binr = (nbin.probs > 0.5).float()
        else:
            cont = ncont.sample()
            binr = nbin.sample()
        action = torch.cat([cont, binr], dim=-1)
        logp = ncont.log_prob(cont).sum(-1) + nbin.log_prob(binr).sum(-1)
        value = self.value(feat).squeeze(-1)
        return action, logp, value

    def evaluate(self, obs, action):
        """For PPO updates: recompute logp, entropy, value under current params."""
        feat = self.trunk(obs)
        ncont, nbin = self._dists(feat)
        cont, binr = action[..., :N_CONT], action[..., N_CONT:]
        logp = ncont.log_prob(cont).sum(-1) + nbin.log_prob(binr).sum(-1)
        ent = ncont.entropy().sum(-1) + nbin.entropy().sum(-1)
        value = self.value(feat).squeeze(-1)
        return logp, ent, value
