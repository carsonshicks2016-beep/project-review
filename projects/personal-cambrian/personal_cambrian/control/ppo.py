"""Compact PPO for continuous control (ROADMAP Stage 3.1), CleanRL-style.

An MLP actor-critic (diagonal-Gaussian policy + value head) trained with clipped
PPO and GAE over a hand-rolled synchronous vectorized rollout (we avoid the
gymnasium vector autoreset, whose semantics shift across versions). The policy is
intentionally a thin, swappable module so Stage 3.3 can drop in a morphology-aware
graph network behind the same `train()` loop.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_checkpoint(path, agent, opt, global_step, it, cfg, obs_dim, act_dim):
    """Full checkpoint: weights + optimizer + progress + config + dims (for resume)."""
    torch.save({"agent": agent.state_dict(), "opt": opt.state_dict(),
                "global_step": int(global_step), "iter": int(it),
                "cfg": dict(cfg.__dict__), "obs_dim": int(obs_dim),
                "act_dim": int(act_dim)}, path)


def load_checkpoint(path):
    return torch.load(path, weights_only=False)


def agent_from_checkpoint(path):
    """Rebuild an ActorCritic with weights loaded from a checkpoint."""
    ck = load_checkpoint(path)
    agent = ActorCritic(ck["obs_dim"], ck["act_dim"], ck["cfg"]["hidden"])
    agent.load_state_dict(ck["agent"])
    return agent, ck


def _layer(in_f, out_f, std=np.sqrt(2), bias=0.0):
    lyr = nn.Linear(in_f, out_f)
    nn.init.orthogonal_(lyr.weight, std)
    nn.init.constant_(lyr.bias, bias)
    return lyr


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.critic = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, 1, std=1.0))
        self.actor_mean = nn.Sequential(
            _layer(obs_dim, hidden), nn.Tanh(),
            _layer(hidden, hidden), nn.Tanh(),
            _layer(hidden, act_dim, std=0.01))
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), -0.5))

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        mean = self.actor_mean(x)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return action, logp, entropy, self.critic(x).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_np, deterministic: bool = False) -> np.ndarray:
        x = torch.as_tensor(np.asarray(obs_np, np.float32))
        if x.ndim == 1:
            x = x.unsqueeze(0)
        mean = self.actor_mean(x)
        a = mean if deterministic else torch.distributions.Normal(
            mean, torch.exp(self.actor_logstd)).sample()
        return a.squeeze(0).cpu().numpy()


@dataclass
class PPOConfig:
    total_timesteps: int = 200_000
    n_envs: int = 8
    n_steps: int = 512
    n_epochs: int = 4
    n_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    hidden: int = 64
    seed: int = 0


def train(make_env: Callable[[int], object], cfg: PPOConfig = PPOConfig(),
          log_fn: Optional[Callable[[dict], None]] = None, verbose: bool = False,
          *, eval_every: int = 0, eval_seed: int = 0,
          make_eval_env: Optional[Callable[[int], object]] = None,
          checkpoint_dir: Optional[str] = None, checkpoint_every: int = 0,
          resume_from: Optional[str] = None,
          make_agent: Optional[Callable[[int, int], nn.Module]] = None):
    """Train an ActorCritic. `make_env(seed)` builds one CreatureEnv.

    Infra (Stage 3.2): periodic deterministic eval (`eval_every`), best/last
    checkpoints under `checkpoint_dir`, and `resume_from` to continue a run.
    `cfg.total_timesteps` is the number of *additional* steps this call runs.
    Returns (agent, history)."""
    set_seed(cfg.seed)
    envs = [make_env(cfg.seed + i) for i in range(cfg.n_envs)]
    obs_dim = envs[0].observation_space.shape[0]
    act_dim = envs[0].action_space.shape[0]
    agent = (make_agent(obs_dim, act_dim) if make_agent is not None
             else ActorCritic(obs_dim, act_dim, cfg.hidden))
    opt = torch.optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    start_it, global_step, best_metric = 0, 0, -float("inf")
    if resume_from is not None:
        ck = load_checkpoint(resume_from)
        agent.load_state_dict(ck["agent"])
        opt.load_state_dict(ck["opt"])
        global_step = int(ck["global_step"])
        start_it = int(ck["iter"]) + 1

    N, T = cfg.n_envs, cfg.n_steps
    obs = np.stack([envs[i].reset(seed=cfg.seed + i)[0] for i in range(N)])
    done = np.zeros(N, np.float32)
    ep_ret = np.zeros(N)
    recent = deque(maxlen=100)

    n_iters = max(1, cfg.total_timesteps // (N * T))
    history = []
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    for it in range(start_it, start_it + n_iters):
        b_obs = torch.zeros((T, N, obs_dim))
        b_act = torch.zeros((T, N, act_dim))
        b_logp = torch.zeros((T, N))
        b_val = torch.zeros((T, N))
        b_rew = torch.zeros((T, N))
        b_done = torch.zeros((T, N))

        for t in range(T):
            ot = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                act, logp, _, val = agent.get_action_and_value(ot)
            b_obs[t] = ot; b_act[t] = act; b_logp[t] = logp
            b_val[t] = val; b_done[t] = torch.as_tensor(done)

            a_np = act.cpu().numpy()
            for i in range(N):
                o, r, term, trunc, _ = envs[i].step(a_np[i].astype(np.float32))
                b_rew[t, i] = r
                ep_ret[i] += r
                d = term or trunc
                done[i] = float(d)
                if d:
                    recent.append(ep_ret[i]); ep_ret[i] = 0.0
                    o, _ = envs[i].reset()
                obs[i] = o
            global_step += N

        # bootstrap + GAE
        with torch.no_grad():
            next_val = agent.get_value(torch.as_tensor(obs, dtype=torch.float32))
        adv = torch.zeros((T, N))
        lastgae = torch.zeros(N)
        next_done = torch.as_tensor(done)
        for t in reversed(range(T)):
            nonterm = 1.0 - (next_done if t == T - 1 else b_done[t + 1])
            nextv = next_val if t == T - 1 else b_val[t + 1]
            delta = b_rew[t] + cfg.gamma * nextv * nonterm - b_val[t]
            adv[t] = lastgae = delta + cfg.gamma * cfg.gae_lambda * nonterm * lastgae
        ret = adv + b_val

        # flatten and optimize
        f_obs = b_obs.reshape(-1, obs_dim)
        f_act = b_act.reshape(-1, act_dim)
        f_logp = b_logp.reshape(-1)
        f_adv = adv.reshape(-1)
        f_ret = ret.reshape(-1)
        bs = T * N
        mb = bs // cfg.n_minibatches
        idx = np.arange(bs)
        for _ in range(cfg.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, bs, mb):
                mbi = idx[start:start + mb]
                _, newlogp, ent, newval = agent.get_action_and_value(f_obs[mbi], f_act[mbi])
                ratio = torch.exp(newlogp - f_logp[mbi])
                a_mb = f_adv[mbi]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio,
                               -a_mb * torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip)).mean()
                v_loss = 0.5 * ((newval - f_ret[mbi]) ** 2).mean()
                loss = pg - cfg.ent_coef * ent.mean() + cfg.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()

        mean_ret = float(np.mean(recent)) if recent else float("nan")
        rec = {"iter": it, "global_step": global_step, "mean_return": mean_ret}

        if eval_every and (it + 1) % eval_every == 0:
            ev_env = (make_eval_env or make_env)(eval_seed)
            m = evaluate(agent, ev_env, seed=eval_seed)
            ev_env.close()
            rec["eval_distance"], rec["eval_return"] = m["distance"], m["return"]
            if checkpoint_dir and m["distance"] > best_metric:
                best_metric = m["distance"]
                save_checkpoint(os.path.join(checkpoint_dir, "best.pt"),
                                agent, opt, global_step, it, cfg, obs_dim, act_dim)
                rec["best"] = True

        if checkpoint_dir and checkpoint_every and (it + 1) % checkpoint_every == 0:
            save_checkpoint(os.path.join(checkpoint_dir, "last.pt"),
                            agent, opt, global_step, it, cfg, obs_dim, act_dim)

        history.append(rec)
        if log_fn:
            log_fn(rec)
        if verbose:
            extra = f"  eval_dist {rec['eval_distance']:+.2f}" if "eval_distance" in rec else ""
            print(f"  iter {it:3d}  step {global_step:7d}  mean_return {mean_ret:8.2f}{extra}")

    if checkpoint_dir:
        save_checkpoint(os.path.join(checkpoint_dir, "last.pt"),
                        agent, opt, global_step, start_it + n_iters - 1,
                        cfg, obs_dim, act_dim)
    for e in envs:
        e.close()
    return agent, history


@torch.no_grad()
def evaluate(agent: ActorCritic, env, *, deterministic: bool = True,
             max_steps: int = 500, seed: int = 0) -> dict:
    """Run one episode; report forward distance, return, and steps survived."""
    obs, _ = env.reset(seed=seed)
    x0 = float(env.data.qpos[env.task.forward_axis])
    ret, steps = 0.0, 0
    for _ in range(max_steps):
        a = agent.act(obs, deterministic=deterministic)
        obs, r, term, trunc, _ = env.step(a.astype(np.float32))
        ret += r; steps += 1
        if term or trunc:
            break
    dist = float(env.data.qpos[env.task.forward_axis]) - x0
    return {"distance": dist, "return": ret, "steps": steps}
