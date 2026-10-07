"""Gaussian actor-critic for continuous rally controls.

Action layout: ``[steer, throttle, brake, handbrake]`` with asymmetric bounds —
steer in ``[-1, 1]``, the rest in ``[0, 1]``. Samples live in an unbounded
Gaussian; a tanh-family squash maps them into the box. Log-probs include the
Jacobian correction so the PPO ratio matches the distribution that was sampled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

OBS_DIM = 84
ACT_DIM = 4
DEFAULT_HIDDEN = (256, 256)
INIT_LOG_STD = -0.5
LOG_STD_MIN = -2.2
LOG_STD_MAX = 0.0
# Per-dim caps: keep exploration useful without pure noise.
DEFAULT_LOG_STD_MAX = (0.0, 0.0, 0.0, 0.0)
EPS = 1e-6


def _orthogonal_linear(module: nn.Linear, gain: float) -> None:
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def squash_actions(pre_tanh: torch.Tensor) -> torch.Tensor:
    """Map unbounded samples into the asymmetric action box."""
    steer = torch.tanh(pre_tanh[..., 0:1])
    rest = 0.5 * (torch.tanh(pre_tanh[..., 1:]) + 1.0)
    return torch.cat([steer, rest], dim=-1)


def unsquash_actions(actions: torch.Tensor) -> torch.Tensor:
    """Invert ``squash_actions`` (clamped for numeric safety)."""
    steer = actions[..., 0:1].clamp(-1.0 + EPS, 1.0 - EPS)
    rest = actions[..., 1:].clamp(EPS, 1.0 - EPS)
    u = (2.0 * rest - 1.0).clamp(-1.0 + EPS, 1.0 - EPS)
    return torch.cat([torch.atanh(steer), torch.atanh(u)], dim=-1)


def squash_log_det(pre_tanh: torch.Tensor) -> torch.Tensor:
    """``log|det ∂a/∂z|`` for the asymmetric squash, summed over action dims."""
    # steer: a = tanh(z)  →  |da/dz| = 1 - tanh²(z)
    steer = torch.tanh(pre_tanh[..., 0:1])
    log_steer = torch.log((1.0 - steer.pow(2)).clamp_min(EPS))
    # rest: a = (tanh(z)+1)/2  →  |da/dz| = 0.5 (1 - tanh²(z))
    rest_u = torch.tanh(pre_tanh[..., 1:])
    log_rest = torch.log((0.5 * (1.0 - rest_u.pow(2))).clamp_min(EPS))
    return torch.cat([log_steer, log_rest], dim=-1).sum(dim=-1)


@dataclass
class ActResult:
    actions: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    pre_tanh: torch.Tensor


class ActorCritic(nn.Module):
    """Shared-trunk Gaussian actor-critic with state-independent ``log_std``."""

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        act_dim: int = ACT_DIM,
        hidden: tuple[int, ...] = DEFAULT_HIDDEN,
        *,
        init_log_std: float = INIT_LOG_STD,
        log_std_max: tuple[float, ...] | None = DEFAULT_LOG_STD_MAX,
    ) -> None:
        super().__init__()
        if act_dim < 1:
            raise ValueError(f"act_dim must be >= 1, got {act_dim}")
        self.obs_dim = int(obs_dim)
        self.act_dim = int(act_dim)

        layers: list[nn.Module] = []
        last = self.obs_dim
        for h in hidden:
            lin = nn.Linear(last, h)
            _orthogonal_linear(lin, gain=np.sqrt(2.0))
            layers += [lin, nn.Tanh()]
            last = h
        self.trunk = nn.Sequential(*layers)

        self.mean_head = nn.Linear(last, self.act_dim)
        _orthogonal_linear(self.mean_head, gain=0.01)
        # Mild forward bias so an untrained policy discovers motion.
        with torch.no_grad():
            if self.act_dim >= 2:
                self.mean_head.bias[1] = 0.5  # throttle toward ~0.6 after squash
            if self.act_dim >= 3:
                self.mean_head.bias[2] = -1.0  # brake off
            if self.act_dim >= 4:
                self.mean_head.bias[3] = -1.0  # handbrake off

        self.value_head = nn.Linear(last, 1)
        _orthogonal_linear(self.value_head, gain=1.0)

        self.log_std = nn.Parameter(torch.full((self.act_dim,), float(init_log_std)))
        caps = DEFAULT_LOG_STD_MAX if log_std_max is None else log_std_max
        if len(caps) != self.act_dim:
            raise ValueError(
                f"log_std_max has {len(caps)} entries for act_dim={self.act_dim}"
            )
        self.log_std_max = tuple(float(x) for x in caps)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        return self.mean_head(h), self.value_head(h).squeeze(-1)

    def _log_std_upper(self) -> torch.Tensor:
        return torch.as_tensor(
            self.log_std_max, dtype=self.log_std.dtype, device=self.log_std.device
        )

    @torch.no_grad()
    def clamp_log_std_(self) -> None:
        upper = self._log_std_upper()
        self.log_std.data.clamp_(min=LOG_STD_MIN)
        self.log_std.data.copy_(torch.minimum(self.log_std.data, upper))

    def _bounded_log_std(self) -> torch.Tensor:
        upper = self._log_std_upper()
        return torch.minimum(
            torch.maximum(self.log_std, torch.full_like(self.log_std, LOG_STD_MIN)),
            upper,
        )

    def _dist(self, mean: torch.Tensor) -> Normal:
        std = self._bounded_log_std().exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, obs: torch.Tensor, *, deterministic: bool = False) -> ActResult:
        mean, value = self.forward(obs)
        dist = self._dist(mean)
        pre = mean if deterministic else dist.rsample()
        actions = squash_actions(pre)
        log_prob = dist.log_prob(pre).sum(-1) - squash_log_det(pre)
        entropy = dist.entropy().sum(-1)
        return ActResult(
            actions=actions,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            pre_tanh=pre,
        )

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(log_prob, entropy, value)`` for squashed actions."""
        mean, value = self.forward(obs)
        dist = self._dist(mean)
        pre = unsquash_actions(actions)
        log_prob = dist.log_prob(pre).sum(-1) - squash_log_det(pre)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value

    def value_only(self, obs: torch.Tensor) -> torch.Tensor:
        _, value = self.forward(obs)
        return value

    @torch.no_grad()
    def mean_action(self, obs: np.ndarray) -> np.ndarray:
        """Deterministic squashed mean action for eval / live (never samples)."""
        x = np.asarray(obs, dtype=np.float32)
        single = x.ndim == 1
        if single:
            x = x[None, :]
        if x.shape[-1] != self.obs_dim:
            raise ValueError(f"obs dim {x.shape[-1]} != policy obs_dim {self.obs_dim}")
        t = torch.as_tensor(x, dtype=torch.float32, device=self.log_std.device)
        out = self.act(t, deterministic=True)
        actions = out.actions.detach().cpu().numpy().astype(np.float32)
        return actions[0] if single else actions
