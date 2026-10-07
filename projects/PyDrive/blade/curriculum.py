"""Curriculum manager + greedy evaluation for BLADE.

The `Curriculum` tracks which stage is active and decides — from a greedy
evaluation — when the agent has mastered it well enough to graduate.  `evaluate`
runs the deterministic policy and measures stage-relevant competence: how much of
the episode the fighter stays upright (survive), how far it closes on the
opponent (reach), and how many hits it lands.
"""
from __future__ import annotations

import numpy as np
import torch


class Curriculum:
    def __init__(self, stages, start=0):
        self.stages = stages
        self.i = start
        self.iters_in_stage = 0

    @property
    def stage(self):
        return self.stages[self.i]

    @property
    def is_last(self):
        return self.i >= len(self.stages) - 1

    def tick(self, n=1):
        self.iters_in_stage += n

    def metric_value(self, ev):
        return ev.get(self.stage.promote_metric, 0.0)

    def promote_progress(self, ev):
        s = self.stage
        if s.promote_metric == "none":
            return 0.0
        return min(1.0, self.metric_value(ev) / max(s.promote_threshold, 1e-6))

    def should_promote(self, ev):
        s = self.stage
        if self.is_last or s.promote_metric == "none":
            return False
        return self.iters_in_stage >= s.min_iters and self.metric_value(ev) >= s.promote_threshold

    def advance(self):
        self.i += 1
        self.iters_in_stage = 0


@torch.no_grad()
def evaluate(agent, eval_env, device):
    """Run the greedy policy for one episode per env; return competence metrics."""
    obs = eval_env.reset()
    N, ms = eval_env.n, eval_env.max_steps
    alive = np.ones(N, bool)
    steps_alive = np.zeros(N)
    ret = np.zeros(N)
    hits = np.zeros(N)
    final_dist = np.full(N, eval_env.stage.spawn_dist, np.float32)
    for _ in range(ms):
        a, _, _ = agent.act(torch.as_tensor(obs.reshape(2 * N, -1), device=device), deterministic=True)
        obs, rew, done, info = eval_env.step(a.cpu().numpy().reshape(N, 2, -1), auto_reset=False)
        for e in range(N):
            if alive[e]:
                steps_alive[e] += 1
                ret[e] += rew[e, 0]
                final_dist[e] = info[e]["dist"]
                if not info[e]["upright"][0]:
                    alive[e] = False
            hits[e] += info[e]["hits"][0]
        if not alive.any():
            break
    init = max(eval_env.stage.spawn_dist, 1e-6)
    return {
        "survive": float((steps_alive / ms).mean()),
        "reach": float(np.clip(1.0 - final_dist.mean() / init, 0, 1)),
        "hits": float(hits.mean()),
        "ret": float(ret.sum() / max(steps_alive.sum(), 1)),
        "surv_steps": float(steps_alive.mean()),
    }
