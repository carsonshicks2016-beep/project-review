"""PPO core: generalized advantage estimation and the clipped-surrogate update.

Trajectories are laid out as (T, S) where S is the number of independent
agent-streams (each duel contributes up to two streams).  GAE runs along T per
stream; a per-sample ``train_mask`` lets us drop frozen-opponent streams that
should not contribute gradients.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 8192
    lr: float = 3e-4
    ent_coef: float = 0.004
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03


def compute_gae(rewards, values, dones, last_values, gamma, lam):
    """All inputs numpy (T,S) except last_values (S,).  Returns adv, returns (T,S)."""
    T, S = rewards.shape
    adv = np.zeros((T, S), np.float32)
    last = np.zeros(S, np.float32)
    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t]
        nextval = values[t + 1] if t + 1 < T else last_values
        delta = rewards[t] + gamma * nextval * nonterminal - values[t]
        last = delta + gamma * lam * nonterminal * last
        adv[t] = last
    return adv, adv + values


def ppo_update(agent, optimizer, batch, cfg: PPOConfig, device):
    """batch: dict of numpy arrays already flattened to (M, ...) for trainable samples."""
    obs = torch.as_tensor(batch["obs"], device=device)
    act = torch.as_tensor(batch["act"], device=device)
    old_logp = torch.as_tensor(batch["logp"], device=device)
    adv = torch.as_tensor(batch["adv"], device=device)
    ret = torch.as_tensor(batch["ret"], device=device)
    old_val = torch.as_tensor(batch["val"], device=device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    M = obs.shape[0]
    idx = np.arange(M)

    stats = {"pg": 0.0, "vf": 0.0, "ent": 0.0, "kl": 0.0, "clipfrac": 0.0, "n": 0}
    stop = False
    for _ in range(cfg.epochs):
        if stop:
            break
        np.random.shuffle(idx)
        for s in range(0, M, cfg.minibatch):
            mb = idx[s:s + cfg.minibatch]
            logp, ent, val = agent.evaluate(obs[mb], act[mb])
            ratio = torch.exp(logp - old_logp[mb])
            a = adv[mb]
            pg = -torch.min(ratio * a,
                            torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            # clipped value loss
            v_clip = old_val[mb] + (val - old_val[mb]).clamp(-cfg.clip, cfg.clip)
            vf = 0.5 * torch.max((val - ret[mb]) ** 2, (v_clip - ret[mb]) ** 2).mean()
            ent_loss = ent.mean()
            loss = pg + cfg.vf_coef * vf - cfg.ent_coef * ent_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                kl = (old_logp[mb] - logp).mean().item()
                clipped = (torch.abs(ratio - 1.0) > cfg.clip).float().mean().item()
            stats["pg"] += pg.item(); stats["vf"] += vf.item()
            stats["ent"] += ent_loss.item(); stats["kl"] += kl
            stats["clipfrac"] += clipped; stats["n"] += 1
            if cfg.target_kl is not None and kl > 1.5 * cfg.target_kl:
                stop = True
                break

    n = max(stats["n"], 1)
    return {k: (v / n if k != "n" else v) for k, v in stats.items()}
