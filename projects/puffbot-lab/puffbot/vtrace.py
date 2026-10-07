"""Off-policy corrected targets and the PPO loss for asynchronous workers.

Workers never wait for each other or for the learner, so a fragment can be a policy
version or two old by the time it is trained on. V-trace (Espeholt et al., 2018)
corrects the value targets and advantages for that lag with truncated importance
weights; the PPO clip then keeps each update close to the policy the batch started
from (the "decoupled" PPO of Hilton et al., 2021). Fragments older than the configured
lag are dropped by the learner rather than trusted.
"""
from __future__ import annotations

import torch


@torch.no_grad()
def vtrace(values, bootstrap, rewards, discounts, log_rho, lam: float, rho_clip: float = 1.0):
    """All tensors are [N, T] except bootstrap [N].

    discounts[t] is gamma**frames for that decision, or 0 where the game ended.
    Returns (vs [N,T], advantages [N,T])."""
    rho = torch.exp(log_rho)
    clipped_rho = torch.clamp(rho, max=rho_clip)
    cs = lam * torch.clamp(rho, max=1.0)
    next_values = torch.cat([values[:, 1:], bootstrap[:, None]], dim=1)
    deltas = clipped_rho * (rewards + discounts * next_values - values)
    acc = torch.zeros_like(bootstrap)
    out = torch.empty_like(values)
    for t in range(values.shape[1] - 1, -1, -1):
        acc = deltas[:, t] + discounts[:, t] * cs[:, t] * acc
        out[:, t] = acc
    vs = values + out
    next_vs = torch.cat([vs[:, 1:], bootstrap[:, None]], dim=1)
    advantages = clipped_rho * (rewards + discounts * next_vs - values)
    return vs, advantages


def ppo_loss(logits, values, actions, prox_logp, advantages, targets, clip, value_coef, entropy_coef,
             policy_coef=1.0):
    """Flat [B] tensors (logits [B, A]). Returns (loss, stats dict). policy_coef 0 trains
    the value estimate alone (critic warm-up)."""
    logp_all = torch.log_softmax(logits, dim=-1)
    logp = logp_all.gather(-1, actions[:, None]).squeeze(-1)
    ratio = torch.exp(logp - prox_logp)
    adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    surrogate = torch.min(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    policy_loss = -surrogate.mean()
    value_loss = 0.5 * (values - targets).pow(2).mean()
    probs = logp_all.exp()
    entropy = -(probs * logp_all).sum(-1).mean()
    loss = policy_coef * (policy_loss - entropy_coef * entropy) + value_coef * value_loss
    with torch.no_grad():
        clip_frac = ((ratio - 1).abs() > clip).float().mean()
        approx_kl = (prox_logp - logp).mean()
    return loss, dict(policy_loss=policy_loss.item(), value_loss=value_loss.item(),
                      entropy=entropy.item(), clip_frac=clip_frac.item(), approx_kl=approx_kl.item())
