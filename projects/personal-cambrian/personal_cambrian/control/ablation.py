"""Fine-tune-budget fairness ablation (ROADMAP Stage 3.5).

Body-vs-brain credit assignment: when evolution scores a mutated body, it must
fine-tune that body's (warm-started) controller far enough that the score
reflects the BODY, not an undertrained brain. Too small a budget and a good body
with a half-trained controller is wrongly culled.

This module measures a body's score as a function of fine-tune budget, finds the
"knee" where the score plateaus, and recommends a budget that puts >=90% of bodies
past their knee. Because warm-start (Stage 3.4) starts bodies near-competent, the
knee is early — which is exactly what makes per-body fine-tuning affordable.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .ppo import PPOConfig, train
from .modular import ModularPolicy
from .warmstart import warm_started_make_agent


def find_knee(steps, scores, frac: float = 0.9):
    """Smallest step at which the (running-max) score first reaches `frac` of its
    final value — the budget past which more training barely helps."""
    steps = np.asarray(steps)
    scores = np.asarray(scores, dtype=float)
    if steps.size == 0:
        return None
    rmax = np.maximum.accumulate(np.nan_to_num(scores, nan=-np.inf))
    if not np.isfinite(rmax[-1]) or rmax[-1] <= 0:
        return int(steps[-1])
    idx = int(np.argmax(rmax >= frac * rmax[-1]))
    return int(steps[idx])


def recommend_budget(knees, coverage: float = 0.9) -> float:
    """A fine-tune budget that puts at least `coverage` of bodies past their knee.

    Uses the 'higher' quantile so the result is an actual knee value that
    guarantees >= coverage fraction of bodies have knee <= budget."""
    ks = np.asarray([k for k in knees if k is not None], dtype=float)
    if ks.size == 0:
        return 0.0
    return float(np.quantile(ks, coverage, method="higher"))


def scratch_make_agent(child_make_env: Callable[[int], object], hidden: int = 64,
                       rounds: int = 2) -> Callable:
    """A `make_agent` that builds a fresh (random) modular policy for the child."""
    sample = child_make_env(0)
    ob = sample.obs_builder
    nd, nn, adj = ob.node_dim, ob.n_nodes, sample.actuator_adjacency
    sample.close()
    return lambda o, a: ModularPolicy(nd, nn, adj, hidden, rounds)


def budget_curve(child_make_env: Callable[[int], object], make_agent: Callable,
                 max_steps: int, n_envs: int = 8, seed: int = 0):
    """Fine-tune one body to `max_steps`, returning (steps, scores) where score is
    the (stochastic) training mean_return per iteration."""
    cfg = PPOConfig(total_timesteps=max_steps, n_envs=n_envs, seed=seed)
    _, hist = train(child_make_env, cfg, make_agent=make_agent)
    steps = np.array([h["global_step"] for h in hist])
    scores = np.array([h["mean_return"] for h in hist], dtype=float)
    return steps, scores


def warmstart_curve(child_make_env, parent, max_steps, n_envs=8, seed=0):
    return budget_curve(child_make_env, warm_started_make_agent(child_make_env, parent),
                        max_steps, n_envs, seed)
