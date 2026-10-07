"""Frozen-opponent pool for stable self-play.

Pure mirror self-play (current policy vs an exact copy of itself) trains fast but
can chase its own tail into rock-paper-scissors cycles.  Mixing in a few frozen
past snapshots as opponents keeps newly learned policies from forgetting how to
beat older strategies.
"""
from __future__ import annotations

import copy
import random
from collections import deque

import torch

from .networks import ActorCritic


class OpponentPool:
    def __init__(self, obs_dim: int, capacity: int = 12, device="cpu"):
        self.obs_dim = obs_dim
        self.device = device
        self.snaps = deque(maxlen=capacity)

    def add(self, agent: ActorCritic):
        self.snaps.append(copy.deepcopy(agent.state_dict()))

    def __len__(self):
        return len(self.snaps)

    def sample(self) -> ActorCritic:
        net = ActorCritic(self.obs_dim).to(self.device)
        net.load_state_dict(random.choice(self.snaps))
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        return net
