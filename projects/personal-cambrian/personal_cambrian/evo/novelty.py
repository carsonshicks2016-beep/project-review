"""NOVELTY SEARCH + LOCAL COMPETITION (NSLC) -- ROADMAP Stage 8.2.

Fitness-only search is *deceptive*: it climbs the nearest reward gradient and
gets stuck, so lineages plateau (the open-endedness problem). Novelty search
(Lehman & Stanley 2008) instead rewards DOING SOMETHING NEW -- a creature scores
well if its behavior characterization (Stage 8.1) is far from everything seen so
far. **Local competition** (NSLC, Lehman & Stanley 2011) adds the quality half
back: among its behavioral NEIGHBORS, is it any good? The two objectives together
push the search outward (novelty) while still refining within each behavioral
neighborhood (competition) -- exactly what a MAP-Elites archive wants from its
emitter.

This module is the SELECTION machinery; it is wired into the Stage-5 QD loop as an
opt-in *novelty emitter* (`QDConfig.novelty`). The behavior vectors come from
`behavior.characterize`; distances use `behavior.bc_distance`.

  * `NoveltyArchive` -- a growing record of behaviors; scores novelty (mean k-NN
    distance) and local competition (fraction of behavioral neighbors out-scored).
  * `novelty_emitter` -- pick a parent from the current elites, biased by a
    set-normalized blend of novelty and local competition.
  * `behavior_cells` / `behavior_spread` -- dimension-agnostic measures of how much
    behavior space a population covers (the Stage-8.2 ablation metric).
"""
from __future__ import annotations

import numpy as np

from .behavior import BehaviorChar, bc_distance


def _vec(x) -> np.ndarray:
    return x.vector if isinstance(x, BehaviorChar) else np.asarray(x, dtype=float)


class NoveltyArchive:
    """A permanent record of explored behaviors for novelty / local-competition scoring.

    Novelty is measured against the WHOLE archive (not just the live population) so
    a behavior already abandoned still counts as "seen" -- this is what stops the
    search from rediscovering the same thing forever. `max_size` bounds memory by
    forgetting a random old entry (uniform, so the kept set stays representative).
    """

    def __init__(self, k: int = 5, *, threshold: float = 0.0, max_size: int = 4000,
                 seed: int = 0):
        self.k = int(k)
        self.threshold = float(threshold)
        self.max_size = int(max_size)
        self._rng = np.random.default_rng(seed)
        self.bcs: list[np.ndarray] = []
        self.fitnesses: list[float] = []

    def __len__(self) -> int:
        return len(self.bcs)

    # -- scoring ------------------------------------------------------------
    def _sorted_neighbors(self, bc) -> np.ndarray:
        """Indices of archive entries by ascending BC distance to `bc`."""
        v = _vec(bc)
        d = np.array([bc_distance(v, b) for b in self.bcs])
        return np.argsort(d), d

    def novelty(self, bc) -> float:
        """Mean distance to the k nearest archived behaviors (behavioral sparseness).
        Empty archive -> inf (anything is maximally novel against nothing)."""
        if not self.bcs:
            return float("inf")
        v = _vec(bc)
        d = np.sort([bc_distance(v, b) for b in self.bcs])
        return float(d[: min(self.k, d.size)].mean())

    def local_competition(self, bc, fitness: float) -> float:
        """Fraction of the k behavioral neighbors whose fitness `fitness` matches or
        beats, in [0, 1]. Empty archive -> 1.0 (nothing to lose to)."""
        if not self.bcs:
            return 1.0
        order, _ = self._sorted_neighbors(bc)
        idx = order[: min(self.k, len(order))]
        neigh = np.array([self.fitnesses[i] for i in idx], dtype=float)
        return float(np.mean(float(fitness) >= neigh)) if neigh.size else 1.0

    def nslc_score(self, bc, fitness: float, *, w_novelty: float = 1.0,
                   w_compete: float = 1.0) -> float:
        """Raw NSLC objective: w_novelty * novelty + w_compete * local_competition.
        (The emitter set-normalizes novelty before blending; this raw form is for
        inspection / tests.)"""
        nov = self.novelty(bc)
        nov = 0.0 if not np.isfinite(nov) else nov
        return w_novelty * nov + w_compete * self.local_competition(bc, fitness)

    # -- growth -------------------------------------------------------------
    def add(self, bc, fitness: float = 0.0) -> None:
        self.bcs.append(_vec(bc).copy())
        self.fitnesses.append(float(fitness))
        if len(self.bcs) > self.max_size:                  # forget a random old entry
            j = int(self._rng.integers(len(self.bcs) - 1))
            self.bcs.pop(j)
            self.fitnesses.pop(j)

    def consider(self, bc, fitness: float = 0.0) -> float:
        """Add `bc` iff it is novel enough (>= threshold), else just score it.
        Returns its novelty. The first entry is always admitted."""
        nov = self.novelty(bc)
        if not self.bcs or not np.isfinite(nov) or nov >= self.threshold:
            self.add(bc, fitness)
        return nov


# --- selection (the emitter) -----------------------------------------------
def proportional_choice(scores, rng) -> int:
    """Index sampled with probability proportional to non-negative `scores`
    (uniform if they sum to ~0)."""
    s = np.asarray(scores, dtype=float)
    s = np.clip(s, 0.0, None)
    tot = float(s.sum())
    p = s / tot if tot > 1e-12 else np.full(s.size, 1.0 / s.size)
    return int(rng.choice(s.size, p=p))


def nslc_scores(elites, archive: NoveltyArchive, *, w_novelty: float = 1.0,
                w_compete: float = 1.0, bc_of=lambda e: e.meta["bc"],
                fitness_of=lambda e: e.fitness) -> np.ndarray:
    """Per-elite NSLC selection scores with novelty MIN-MAX normalized across the
    candidate set, so novelty (a distance) and local competition ([0,1]) blend on a
    comparable scale regardless of the BC metric's absolute magnitude."""
    bcs = [bc_of(e) for e in elites]
    nov = np.array([archive.novelty(b) for b in bcs], dtype=float)
    nov[~np.isfinite(nov)] = np.nanmax(nov[np.isfinite(nov)]) if np.any(np.isfinite(nov)) else 1.0
    lo, hi = nov.min(), nov.max()
    nov_n = (nov - lo) / (hi - lo) if hi > lo else np.zeros_like(nov)
    lc = np.array([archive.local_competition(b, fitness_of(e))
                   for b, e in zip(bcs, elites)], dtype=float)
    return w_novelty * nov_n + w_compete * lc


def novelty_emitter(elites, archive: NoveltyArchive, rng, *, w_novelty: float = 1.0,
                    w_compete: float = 1.0, bc_of=lambda e: e.meta["bc"],
                    fitness_of=lambda e: e.fitness):
    """Pick one elite as a parent, biased toward novel + locally-competitive behavior.
    Falls back to uniform when no elite carries a BC."""
    cand = [e for e in elites if _has_bc(e, bc_of)]
    if not cand:
        return elites[int(rng.integers(len(elites)))] if elites else None
    scores = nslc_scores(cand, archive, w_novelty=w_novelty, w_compete=w_compete,
                         bc_of=bc_of, fitness_of=fitness_of)
    return cand[proportional_choice(scores + 1e-9, rng)]


def _has_bc(e, bc_of) -> bool:
    try:
        return bc_of(e) is not None
    except (KeyError, AttributeError, TypeError):
        return False


# --- exploration metrics (the ablation yardstick) --------------------------
def behavior_spread(bcs) -> float:
    """Mean pairwise BC distance of a set of behaviors -- how spread out, on average,
    the population is in behavior space. 0/1 behaviors -> 0."""
    v = [_vec(b) for b in bcs]
    n = len(v)
    if n < 2:
        return 0.0
    tot, cnt = 0.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            tot += bc_distance(v[i], v[j]); cnt += 1
    return tot / cnt


def behavior_cells(bcs, radius: float) -> int:
    """Dimension-agnostic behavioral coverage: greedily count behaviors that are
    mutually farther apart than `radius` (a packing count). More distinct cells =
    more of behavior space explored. Works in any BC dimensionality."""
    reps: list[np.ndarray] = []
    for b in bcs:
        v = _vec(b)
        if all(bc_distance(v, r) > radius for r in reps):
            reps.append(v)
    return len(reps)
