"""SURROGATE-ASSISTED QUALITY-DIVERSITY (ROADMAP Stage 9.4).

Each QD evaluation is an expensive physics+RL simulation. Most mutated candidates
are duds; simulating them all wastes the budget. A SURROGATE -- a cheap model
mapping a genome's (sim-free) features to a PREDICTED fitness + an UNCERTAINTY --
lets the ask() step pre-screen a batch of candidates and SIMULATE ONLY the
promising-or-uncertain ones, skipping the obvious losers. Real evaluations then
feed back to improve the surrogate (active learning / surrogate-assisted MAP-Elites).

Unlike the rest of Stage 9, the payoff here is HARDWARE-INDEPENDENT: "sims saved" is
a count, so the done-when -- *match the no-surrogate QD-score with materially fewer
sims* -- is fully verifiable on CPU.

  * `genome_features` -- a fixed-length, simulation-free descriptor vector.
  * `Surrogate` -- a standardized k-NN regressor giving (mean, uncertainty); cheap,
    online-updatable, no extra deps.
  * `surrogate_assisted_qd` -- a MAP-Elites loop that, per iteration, mutates a batch
    of candidates, ranks them by a UCB acquisition (predicted fitness + beta x
    uncertainty), simulates only the top one, and updates the surrogate. With the
    surrogate OFF it degrades to plain random-mutation MAP-Elites (the ablation
    baseline).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Optional

import numpy as np

from .descriptors import morphology_descriptors, normalize


# --- simulation-free genome features ---------------------------------------
_FEATURE_KEYS = ("limb_count", "muscle_count", "mass", "height", "aspect", "symmetry")


def genome_features(genome, morph=None) -> np.ndarray:
    """A fixed-length, normalized, sim-free feature vector for a genome (the
    surrogate's input). Reuses the Stage-5.1 morphology descriptors."""
    d = morphology_descriptors(genome, morph)
    return np.array([normalize(k, d[k]) for k in _FEATURE_KEYS], dtype=float)


# --- the surrogate model (standardized k-NN regressor) ---------------------
class Surrogate:
    """Predicts fitness from features via distance-weighted k-NN, with an
    uncertainty estimate (distance to the training set + neighbor disagreement).
    Cheap and online: `update` appends new (features, fitness) pairs."""

    def __init__(self, k: int = 5):
        self.k = int(k)
        self._X: list = []
        self._y: list = []

    def __len__(self) -> int:
        return len(self._y)

    def update(self, feats, fitness) -> None:
        feats = np.atleast_2d(np.asarray(feats, dtype=float))
        fitness = np.atleast_1d(np.asarray(fitness, dtype=float))
        for x, y in zip(feats, fitness):
            self._X.append(x); self._y.append(float(y))

    def trained(self) -> bool:
        return len(self._y) >= self.k

    def _std(self):
        X = np.asarray(self._X)
        mu = X.mean(0)
        sd = X.std(0) + 1e-8
        return mu, sd

    def predict(self, feats):
        """Return (mean[N], uncertainty[N]). Falls back to global mean/large
        uncertainty before it has k points."""
        Q = np.atleast_2d(np.asarray(feats, dtype=float))
        if not self.trained():
            mu = float(np.mean(self._y)) if self._y else 0.0
            return np.full(len(Q), mu), np.ones(len(Q))
        X = np.asarray(self._X); y = np.asarray(self._y)
        mu, sd = self._std()
        Xn = (X - mu) / sd
        Qn = (Q - mu) / sd
        d = np.linalg.norm(Qn[:, None, :] - Xn[None, :, :], axis=-1)   # [N, M]
        nn = np.argsort(d, axis=1)[:, : self.k]                        # k nearest
        dk = np.take_along_axis(d, nn, axis=1)
        w = 1.0 / (dk + 1e-6)
        yk = y[nn]
        mean = (w * yk).sum(1) / w.sum(1)
        # uncertainty: closeness to training data + disagreement among neighbors
        unc = dk.mean(1) + yk.std(1)
        return mean, unc

    def acquisition(self, feats, beta: float = 1.0):
        """UCB: predicted fitness + beta * uncertainty (exploit + explore)."""
        mean, unc = self.predict(feats)
        return mean + beta * unc


# --- surrogate-assisted MAP-Elites loop ------------------------------------
@dataclass
class SurrogateQDConfig:
    sim_budget: int = 300            # total expensive evaluations allowed
    n_candidates: int = 8            # mutations proposed per iteration
    beta: float = 0.5                # UCB exploration weight
    warmup: int = 20                 # random sims before the surrogate is trusted
    k: int = 5
    use_surrogate: bool = True
    seed: int = 0


@dataclass
class SurrogateQDResult:
    grid: dict                       # cell -> (best_fitness, genome)
    n_sims: int
    qd_curve: list = field(default_factory=list)   # (n_sims, qd_score) over the run

    @property
    def qd_score(self) -> float:
        return float(sum(max(f, 0.0) for f, _ in self.grid.values()))

    @property
    def coverage(self) -> int:
        return len(self.grid)


def surrogate_assisted_qd(seeds, evaluate_fn: Callable[[object], tuple],
                          mutate_fn: Callable[[object, np.random.Generator], object],
                          features_fn: Callable[[object], np.ndarray],
                          cfg: SurrogateQDConfig = SurrogateQDConfig()) -> SurrogateQDResult:
    """Run surrogate-assisted MAP-Elites. `evaluate_fn(genome) -> (fitness, cell)` is
    the expensive sim; `cell` is the (hashable) descriptor bin. Each iteration proposes
    `n_candidates` mutations and -- once the surrogate is warmed up -- SIMULATES ONLY
    THE ONE with the best UCB acquisition (the rest are skipped), then updates the
    surrogate with the real result. With `use_surrogate=False` it simulates a single
    random mutation per iteration (the no-surrogate baseline)."""
    rng = np.random.default_rng(cfg.seed)
    surr = Surrogate(cfg.k)
    grid: dict = {}
    qd_curve: list = []
    n_sims = 0

    def _insert_and_train(genome):
        nonlocal n_sims
        fitness, cell = evaluate_fn(genome)
        cur = grid.get(cell)
        if cur is None or fitness > cur[0]:
            grid[cell] = (fitness, genome)
        surr.update(features_fn(genome), fitness)
        n_sims += 1
        qd_curve.append((n_sims, float(sum(max(f, 0.0) for f, _ in grid.values()))))

    for g in seeds:                                  # seed the archive + surrogate
        if n_sims < cfg.sim_budget:
            _insert_and_train(g)

    while n_sims < cfg.sim_budget:
        parent = grid[list(grid.keys())[rng.integers(len(grid))]][1] if grid else seeds[0]
        use = cfg.use_surrogate and n_sims >= cfg.warmup and surr.trained()
        if not use:
            _insert_and_train(mutate_fn(parent, rng))   # warmup / baseline: one random sim
            continue
        # propose a batch, simulate only the best-acquisition candidate
        cands = [mutate_fn(parent, rng) for _ in range(cfg.n_candidates)]
        feats = np.array([features_fn(c) for c in cands])
        acq = surr.acquisition(feats, cfg.beta)
        _insert_and_train(cands[int(np.argmax(acq))])

    return SurrogateQDResult(grid=grid, n_sims=n_sims, qd_curve=qd_curve)


def sims_to_reach(curve, target_qd: float) -> Optional[int]:
    """The number of sims at which a qd_curve first reaches `target_qd` (None if never)."""
    for n, q in curve:
        if q >= target_qd:
            return n
    return None
