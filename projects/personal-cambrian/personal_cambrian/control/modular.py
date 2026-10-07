"""Morphology-aware modular policy (ROADMAP Stage 3.3).

A Shared Modular Policy in the spirit of Huang et al. 2020 / NerveNet: actuators
are graph nodes carrying the structured per-muscle observation (sim/obs.py), the
edges are the kinematic-tree actuator adjacency, and a small set of *shared*
modules (encoder, message, update, heads) are applied to every node. Because the
parameters do not depend on the number of nodes, ONE weight set drives any body
plan, and weights transfer across creatures (the Stage-3.4 prerequisite).

The adjacency is a plain attribute (deliberately NOT a registered buffer) so it
stays out of `state_dict` — letting a 4-actuator quadruped's weights load straight
into an 8-actuator biped policy.

Interface matches ActorCritic so the same PPO `train()` loop drives it: the
observation is the flattened structured matrix (shape n_nodes*node_dim), reshaped
internally to (B, n_nodes, node_dim).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from .ppo import _layer


class ModularPolicy(nn.Module):
    def __init__(self, node_dim: int, n_nodes: int, adjacency, hidden: int = 64,
                 rounds: int = 2):
        super().__init__()
        self.node_dim, self.n_nodes, self.rounds, self.hidden = node_dim, n_nodes, rounds, hidden
        A = np.asarray(adjacency, dtype=np.float32)
        A = A / A.sum(axis=1, keepdims=True)          # row-normalized (incl. self-loops)
        self.adj = torch.tensor(A)                    # plain attr -> not in state_dict

        self.encoder = nn.Sequential(_layer(node_dim, hidden), nn.Tanh(),
                                     _layer(hidden, hidden), nn.Tanh())
        self.msg = nn.Sequential(_layer(hidden, hidden), nn.Tanh())
        self.update = nn.Sequential(_layer(2 * hidden, hidden), nn.Tanh())
        self.mean_head = _layer(hidden, 1, std=0.01)
        self.value_head = nn.Sequential(_layer(hidden, hidden), nn.Tanh(),
                                        _layer(hidden, 1, std=1.0))
        self.actor_logstd = nn.Parameter(torch.full((1,), -0.5))

    def _embed(self, x):
        """x: (B, n_nodes*node_dim) -> node embeddings (B, n_nodes, hidden)."""
        b = x.shape[0]
        h = self.encoder(x.reshape(b, self.n_nodes, self.node_dim))
        adj = self.adj.to(h.dtype)
        for _ in range(self.rounds):
            agg = torch.einsum("nm,bmh->bnh", adj, self.msg(h))   # mean over neighbors
            h = h + self.update(torch.cat([h, agg], dim=-1))      # residual update
        return h

    def get_value(self, x):
        return self.value_head(self._embed(x).mean(dim=1)).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        h = self._embed(x)
        mean = self.mean_head(h).squeeze(-1)               # (B, n_nodes)
        std = torch.exp(self.actor_logstd).expand_as(mean)
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.value_head(h.mean(dim=1)).squeeze(-1)
        return action, logp, entropy, value

    @torch.no_grad()
    def act(self, obs_np, deterministic: bool = False) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs_np, np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        h = self._embed(x)
        mean = self.mean_head(h).squeeze(-1)
        a = mean if deterministic else Normal(mean, torch.exp(self.actor_logstd)).sample()
        return a.squeeze(0).cpu().numpy()


def policy_for_env(env, hidden: int = 64, rounds: int = 2) -> ModularPolicy:
    ob = env.obs_builder
    return ModularPolicy(ob.node_dim, ob.n_nodes, env.actuator_adjacency, hidden, rounds)


def make_modular_setup(creature: str, ep_steps: int = 500, hidden: int = 64,
                       rounds: int = 2) -> tuple[Callable, Callable]:
    """Return (make_env, make_agent) for training a creature with the modular
    policy. `make_env` builds a structured-observation CreatureEnv; `make_agent`
    builds a ModularPolicy carrying that creature's adjacency."""
    from ..sim import CreatureEnv, LocomotionTask
    from ..seeds import SEEDS

    def make_env(seed: int):
        return CreatureEnv(SEEDS[creature](), task=LocomotionTask(max_steps=ep_steps),
                           obs_mode="structured")

    sample = make_env(0)
    node_dim = sample.obs_builder.node_dim
    n_nodes = sample.obs_builder.n_nodes
    adj = sample.actuator_adjacency
    sample.close()

    def make_agent(obs_dim, act_dim):     # dims ignored; morphology fixes the shape
        return ModularPolicy(node_dim, n_nodes, adj, hidden, rounds)

    return make_env, make_agent
