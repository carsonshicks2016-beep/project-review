"""PAIRED OPEN-ENDED TRAILBLAZER (POET) -- ROADMAP Stage 8.3.

Novelty search (8.2) reaches for new *behaviors* inside a fixed set of niches.
POET (Wang et al. 2019) goes one level up: it co-evolves the NICHES THEMSELVES
alongside the agents that solve them, manufacturing an endless, auto-curated
curriculum. A creature can only become genuinely alien if the *pressures* keep
escalating and diversifying -- so the environment population must stay open-ended
too.

The loop maintains a set of (environment, agent) PAIRS and repeats:

  * OPTIMIZE  -- improve each agent in its own paired environment (a step of RL).
  * REPRODUCE -- environments whose agent has "graduated" spawn MUTATED children
    (harder / different niches). A child is admitted only if it passes
    **minimal-criterion coevolution (MCC)**: some current agent scores inside a
    [too-hard, too-easy] band on it -- solvable but not yet solved. Among the
    admissible children the most ENV-NOVEL ones are taken.
  * TRANSFER  -- periodically test every other agent on each environment; if a
    foreign agent (the "trailblazer") beats the incumbent, adopt it. This is how a
    skill discovered in one niche jump-starts another.
  * RETIRE    -- cap the active set, retiring the oldest pairs so the frontier
    keeps turning over instead of ossifying.

This module is the OPTIMIZER-AGNOSTIC orchestrator: environment, agent, and the
optimize/evaluate/mutate operations are injected as callbacks, so the same engine
drives both a fast deterministic toy (tests) and the real RL creature pipeline
(`make_creature_poet` + `scripts/poet_curriculum.py`). The Stage-8.3 done-when --
the active-env set turns over while difficulty rises, and agents transfer -- is a
property of THIS loop, logged in `POET.history`.
"""
from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class POETPair:
    """One (environment, agent) pair tracked by the POET population."""
    id: int
    env: Any
    agent: Any
    parent: Optional[int] = None
    created_at: int = 0
    score: float = float("-inf")     # last eval of `agent` in its own `env`
    best_score: float = float("-inf")  # best score this env ever reached (for ANNECS)


@dataclass
class POETConfig:
    max_active: int = 8              # cap on simultaneously active pairs
    reproduce_every: int = 3         # attempt environment reproduction every N steps
    transfer_every: int = 3          # attempt agent transfers every N steps
    n_children: int = 4              # candidate children per eligible parent
    max_admit: int = 1               # admit at most N new envs per reproduction round
    optimize_steps: int = 1          # inner-loop optimization steps per POET step
    # minimal-criterion band on the BEST current agent's score on a candidate env:
    # below mc_low = too hard (nobody can do it); above mc_high = too easy (solved).
    mc_low: float = -0.3
    mc_high: float = 0.3
    repro_threshold: float = 0.3     # a parent may spawn once its own score >= this
    transfer_margin: float = 1e-6    # min improvement to accept a transfer
    novelty_k: int = 3               # k-NN for environment novelty


class POET:
    def __init__(self, cfg: POETConfig, *,
                 optimize: Callable[[Any, Any, int], Any],
                 evaluate: Callable[[Any, Any], float],
                 env_mutate: Callable[[Any, np.random.Generator], Any],
                 env_descriptor: Callable[[Any], np.ndarray],
                 env_difficulty: Callable[[Any], float],
                 rng: Optional[np.random.Generator] = None,
                 copy_agent: Callable[[Any], Any] = _copy.deepcopy):
        self.cfg = cfg
        self._optimize = optimize
        self._evaluate = evaluate
        self._env_mutate = env_mutate
        self._env_descriptor = env_descriptor
        self._env_difficulty = env_difficulty
        self._copy_agent = copy_agent
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.pairs: list[POETPair] = []
        self.env_log: list[POETPair] = []         # every pair ever created (survives retirement)
        self.env_archive: list[np.ndarray] = []   # descriptors of every env ever admitted
        self.transfers: list[dict] = []           # (recipient, donor, gain) events
        self.history: list[dict] = []
        self._next_id = 0
        self.t = 0
        self.n_added = 0
        self.n_removed = 0

    # -- population management ----------------------------------------------
    def add_env(self, env, agent, parent: Optional[int] = None) -> POETPair:
        p = POETPair(id=self._next_id, env=env, agent=agent, parent=parent,
                     created_at=self.t)
        self._next_id += 1
        p.score = self._evaluate(agent, env)
        p.best_score = p.score
        self.pairs.append(p)
        self.env_log.append(p)
        self.env_archive.append(np.asarray(self._env_descriptor(env), dtype=float))
        self.n_added += 1
        return p

    def _env_novelty(self, env) -> float:
        v = np.asarray(self._env_descriptor(env), dtype=float)
        if not self.env_archive:
            return float("inf")
        d = np.sort([float(np.linalg.norm(v - a)) for a in self.env_archive])
        return float(d[: min(self.cfg.novelty_k, d.size)].mean())

    # -- the three POET operators -------------------------------------------
    def _reproduce(self) -> int:
        parents = [p for p in self.pairs if p.score >= self.cfg.repro_threshold]
        if not parents:
            return 0
        agents = [p.agent for p in self.pairs]
        cands = []
        for parent in parents:
            for _ in range(self.cfg.n_children):
                child = self._env_mutate(parent.env, self.rng)
                scores = [self._evaluate(a, child) for a in agents]
                bi = int(np.argmax(scores))
                # MCC: the best current agent must find the child solvable-but-unsolved
                if self.cfg.mc_low <= scores[bi] <= self.cfg.mc_high:
                    cands.append((self._env_novelty(child), child, agents[bi], parent.id))
        cands.sort(key=lambda c: -c[0])               # most ENV-novel first
        added = 0
        for _nov, env, agent, pid in cands[: self.cfg.max_admit]:
            self.add_env(env, self._copy_agent(agent), parent=pid)
            added += 1
        return added

    def _attempt_transfers(self) -> int:
        n = 0
        donors = [(p.id, p.agent) for p in self.pairs]
        for p in self.pairs:
            best_agent, best_score, best_donor = None, p.score, None
            for did, a in donors:
                if did == p.id:
                    continue
                s = self._evaluate(a, p.env)          # direct (zero-shot) transfer
                if s > best_score + self.cfg.transfer_margin:
                    best_agent, best_score, best_donor = a, s, did
            if best_agent is not None:
                gain = best_score - p.score
                p.agent = self._copy_agent(best_agent)
                p.score = best_score
                self.transfers.append({"t": self.t, "to": p.id, "from": best_donor,
                                       "gain": float(gain)})
                n += 1
        return n

    def _retire(self) -> int:
        if len(self.pairs) <= self.cfg.max_active:
            return 0
        # retire the OLDEST pairs so the frontier keeps turning over
        self.pairs.sort(key=lambda p: p.created_at)
        n_remove = len(self.pairs) - self.cfg.max_active
        self.pairs = self.pairs[n_remove:]
        self.n_removed += n_remove
        return n_remove

    # -- the loop ------------------------------------------------------------
    def step(self) -> dict:
        self.t += 1
        for p in self.pairs:                          # 1. inner optimization
            p.agent = self._optimize(p.agent, p.env, self.cfg.optimize_steps)
            p.score = self._evaluate(p.agent, p.env)
        n_transfer = (self._attempt_transfers()
                      if self.t % self.cfg.transfer_every == 0 else 0)
        n_add = self._reproduce() if self.t % self.cfg.reproduce_every == 0 else 0
        for p in self.pairs:                          # freeze each env's peak before retiring
            p.best_score = max(p.best_score, p.score)
        n_remove = self._retire()

        diffs = [self._env_difficulty(p.env) for p in self.pairs]
        rec = {
            "t": self.t, "n_active": len(self.pairs),
            "max_difficulty": max(diffs) if diffs else 0.0,
            "mean_difficulty": float(np.mean(diffs)) if diffs else 0.0,
            "added": n_add, "removed": n_remove, "transfers": n_transfer,
            "cum_added": self.n_added, "cum_removed": self.n_removed,
            "cum_transfers": len(self.transfers),
            "best_score": max((p.score for p in self.pairs), default=float("nan")),
        }
        self.history.append(rec)
        return rec

    def run(self, iterations: int, log_fn: Optional[Callable[[dict], None]] = None):
        for _ in range(iterations):
            rec = self.step()
            if log_fn:
                log_fn(rec)
        return self.history

    # -- reporting -----------------------------------------------------------
    def env_outcomes(self) -> list[dict]:
        """Per-environment record in CREATION order (survives retirement): descriptor,
        best score ever reached, difficulty, and creation time. Feeds the Stage-8.4
        ANNECS (accumulated novel-and-solved environments) curve."""
        return [{"id": p.id, "t": p.created_at,
                 "descriptor": np.asarray(self._env_descriptor(p.env), dtype=float),
                 "difficulty": float(self._env_difficulty(p.env)),
                 "best_score": float(p.best_score)}
                for p in self.env_log]

    @property
    def difficulty_rose(self) -> bool:
        """True if the frontier (max active difficulty) ended higher than it began."""
        if len(self.history) < 2:
            return False
        return self.history[-1]["max_difficulty"] > self.history[0]["max_difficulty"]

    @property
    def turned_over(self) -> bool:
        """True if the active set both grew and shed environments over the run."""
        return self.n_added > len(self.pairs) and self.n_removed > 0


# --- environment mutation over the niche gauntlet --------------------------
# (lazy sim imports keep the POET core / its toy tests free of the mujoco dep)
def _niche_type_id(task) -> int:
    from ..sim.tasks import NICHES
    for i, cls in enumerate(NICHES.values()):
        if type(task) is cls:
            return i
    return -1


def niche_descriptor(task) -> np.ndarray:
    """Environment descriptor for POET novelty: (difficulty, niche-type index)."""
    return np.array([float(task.difficulty), float(_niche_type_id(task))], dtype=float)


def mutate_niche(task, rng, *, harder_bias: float = 0.75, step: Optional[float] = None):
    """Mutate a NicheTask into a child environment. Mostly RATCHETS DIFFICULTY UP
    (steeper slope / stronger load / less energy via each niche's `configure_model`/
    `perturb`/`escalate` mapping), occasionally easing it, so the env population
    drifts toward harder niches while keeping some spread."""
    from ..sim.tasks import task_from_dict
    clone = task_from_dict(task.to_dict())
    s = step if step is not None else (clone.escalation_rate or 0.1)
    if rng.random() < harder_bias:
        clone.difficulty = max(0.0, clone.difficulty + s)
    else:
        clone.difficulty = max(0.0, clone.difficulty - 0.5 * s)
    return clone


def make_creature_poet(genome, cfg: POETConfig, *, base_task=None,
                       train_steps: int = 2_000, ep_steps: int = 200, hidden: int = 64,
                       n_envs: int = 4, n_steps: int = 256, seed: int = 0, rng=None):
    """Wire POET onto the real RL creature pipeline for a FIXED seed morphology:
    agents are controllers, environments are NicheTasks, optimization is warm-started
    PPO, evaluation is the niche's own return. Co-evolves an open-ended niche
    curriculum (the slow run lives in scripts/poet_curriculum.py)."""
    from ..sim import CreatureEnv
    from ..control import PPOConfig, evaluate as rl_eval, warm_started_make_agent, scratch_make_agent
    from ..control.ppo import train

    def _cme(task):
        return lambda s: CreatureEnv(genome, task=task, obs_mode="structured")

    def optimize(agent, task, steps):
        cme = _cme(task)
        ma = (warm_started_make_agent(cme, agent) if agent is not None
              else scratch_make_agent(cme, hidden))
        policy, _ = train(cme, PPOConfig(total_timesteps=train_steps, n_envs=n_envs,
                                         n_steps=n_steps, hidden=hidden, seed=seed),
                          make_agent=ma)
        return policy

    def evaluate(agent, task):
        if agent is None:                 # a seed pair before its first optimization
            return float("-inf")
        env = _cme(task)(0)
        r = rl_eval(agent, env, max_steps=ep_steps)["return"]
        env.close()
        return float(r)

    return POET(cfg, optimize=optimize, evaluate=evaluate,
                env_mutate=mutate_niche, env_descriptor=niche_descriptor,
                env_difficulty=lambda t: float(t.difficulty), rng=rng)
