"""Config-driven training orchestration (ROADMAP Stage 3.2).

Wraps the PPO loop with a run directory, JSONL metric logging, periodic
deterministic evaluation, best/last checkpoints, and resume — so training runs are
repeatable and restartable.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

from .ppo import PPOConfig, train
from ..sim import CreatureEnv, LocomotionTask
from ..seeds import SEEDS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_creature_env(creature: str, ep_steps: int = 500) -> Callable[[int], CreatureEnv]:
    """A `make_env(seed)` factory for a named seed creature."""
    def _make(seed: int) -> CreatureEnv:
        return CreatureEnv(SEEDS[creature](), task=LocomotionTask(max_steps=ep_steps))
    return _make


def run_training(make_env: Callable[[int], object], cfg: PPOConfig, run_dir: str, *,
                 eval_every: int = 10, eval_seed: int = 0, checkpoint_every: int = 20,
                 resume_from: Optional[str] = None, make_agent: Optional[Callable] = None,
                 verbose: bool = True):
    """Train with logging + checkpoints into `run_dir`. Returns (agent, history).
    `make_agent(obs_dim, act_dim)` overrides the default MLP (e.g. ModularPolicy)."""
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(dict(cfg.__dict__), f, indent=2)

    mf = open(os.path.join(run_dir, "metrics.jsonl"), "a" if resume_from else "w")

    def log(rec):
        mf.write(json.dumps(rec) + "\n")
        mf.flush()

    agent, history = train(
        make_env, cfg, log_fn=log, verbose=verbose,
        eval_every=eval_every, eval_seed=eval_seed, make_eval_env=make_env,
        checkpoint_dir=run_dir, checkpoint_every=checkpoint_every,
        resume_from=resume_from, make_agent=make_agent)
    mf.close()
    return agent, history


def new_run_dir(tag: str = "run") -> str:
    d = os.path.join(ROOT, "runs", time.strftime("%Y%m%d-%H%M%S") + f"-{tag}")
    os.makedirs(d, exist_ok=True)
    return d
