"""Stage 8.2 acceptance: novelty search + local competition (NSLC).

The "done-when" is *a novelty-on run explores more behavior space than a
novelty-off run*. The expensive proof on the real creature pipeline lives in
`scripts/novelty_ablation.py` (RL). Here we prove the SAME selection mechanism on
a fast, deterministic, RL-free deceptive domain -- the textbook novelty-search
demonstration: fitness has a single attractor that traps fitness-only search,
while novelty selection (driven by the real `NoveltyArchive`) spreads across the
space. We also unit-test the NSLC scoring/selection machinery on synthetic BCs
and a tiny end-to-end QD smoke (needs mujoco) that the novelty emitter wires in.

Runs:  python3 tests/test_novelty.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.evo.novelty import (
    NoveltyArchive, novelty_emitter, nslc_scores, proportional_choice,
    behavior_cells, behavior_spread,
)


# --- NoveltyArchive scoring (synthetic, deterministic) ---------------------
def test_novelty_is_sparseness():
    arch = NoveltyArchive(k=2)
    assert arch.novelty([0.0, 0.0]) == float("inf")     # empty -> maximally novel
    for p in ([0.0, 0.0], [0.1, 0.0], [0.2, 0.0]):
        arch.add(p, 0.0)
    near = arch.novelty([0.05, 0.0])                     # sits among the cluster
    far = arch.novelty([5.0, 0.0])                       # way out in empty space
    assert far > near > 0.0


def test_local_competition_counts_neighbors_beaten():
    arch = NoveltyArchive(k=3)
    assert arch.local_competition([0.0], 1.0) == 1.0     # empty -> beats nobody to lose to
    for x, f in [([0.0], 0.0), ([0.1], 1.0), ([0.2], 2.0)]:
        arch.add(x, f)
    # a high-fitness behavior among these neighbors beats all 3
    assert arch.local_competition([0.05], 9.0) == 1.0
    # a low-fitness one beats none
    assert arch.local_competition([0.05], -9.0) == 0.0
    # middling beats ~the worse third(s)
    mid = arch.local_competition([0.05], 1.0)
    assert 0.0 < mid <= 1.0


def test_nslc_score_blends_both_terms():
    arch = NoveltyArchive(k=2)
    for x, f in [([0.0], 0.0), ([0.1], 0.0)]:
        arch.add(x, f)
    only_nov = arch.nslc_score([9.0], -9.0, w_novelty=1.0, w_compete=0.0)
    only_lc = arch.nslc_score([0.05], 9.0, w_novelty=0.0, w_compete=1.0)
    assert only_nov > 0.0          # far away -> novelty term carries it
    assert only_lc == 1.0          # beats both neighbors -> full local competition


def test_archive_bounds_memory():
    arch = NoveltyArchive(k=3, max_size=10, seed=0)
    for i in range(50):
        arch.add([float(i)], float(i))
    assert len(arch) == 10


# --- selection helpers ------------------------------------------------------
def test_proportional_choice_follows_scores():
    rng = np.random.default_rng(0)
    counts = np.zeros(3)
    for _ in range(2000):
        counts[proportional_choice([0.0, 1.0, 9.0], rng)] += 1
    assert counts[2] > counts[1] > counts[0]
    assert counts[0] == 0                                # zero score -> never picked


class _E:                                                # minimal Elite stand-in
    def __init__(self, bc, fitness):
        self.meta = {"bc": bc}
        self.fitness = fitness


def test_novelty_emitter_prefers_the_outlier():
    arch = NoveltyArchive(k=3)
    for p in ([0.0, 0.0], [0.05, 0.0], [0.0, 0.05]):     # a tight cluster of seen behavior
        arch.add(p, 0.0)
    elites = [_E([0.0, 0.0], 0.0), _E([0.02, 0.0], 0.0), _E([9.0, 9.0], 0.0)]  # idx 2 = outlier
    rng = np.random.default_rng(0)
    picks = [elites.index(novelty_emitter(elites, arch, rng, w_compete=0.0)) for _ in range(300)]
    assert picks.count(2) > picks.count(0) + picks.count(1)   # the novel one dominates


# --- exploration metrics ----------------------------------------------------
def test_behavior_cells_and_spread():
    same = [[0.0, 0.0]] * 5
    assert behavior_cells(same, radius=0.01) == 1            # all coincident -> one cell
    assert behavior_spread(same) == 0.0
    spread_pts = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]
    assert behavior_cells(spread_pts, radius=0.05) == 4      # mutually far -> four cells
    assert behavior_spread(spread_pts) > 0.0


# --- the done-when: novelty explores more (RL-free deceptive domain) --------
def _deceptive_fitness(g):
    """Single attractor at (0.5, 0.05): fitness-only search collapses onto it."""
    return -float(np.hypot(g[0] - 0.5, g[1] - 0.05))


def _grid_coverage(points, bins=10):
    occ = {(min(int(p[0] * bins), bins - 1), min(int(p[1] * bins), bins - 1))
           for p in points}
    return len(occ) / (bins * bins)


def _toy_search(novelty_on, *, steps=400, pop_size=12, seed=0):
    """Steady-state EA on the unit square; behavior == genome. Parent/replacement
    selection is novelty-driven (real NoveltyArchive) or fitness-driven."""
    rng = np.random.default_rng(seed)
    arch = NoveltyArchive(k=5, seed=seed)
    pop = [rng.random(2) for _ in range(pop_size)]
    fit = [_deceptive_fitness(g) for g in pop]
    for g, f in zip(pop, fit):
        arch.add(g, f)
    visited = [g.copy() for g in pop]
    for _ in range(steps):
        if novelty_on:
            i = proportional_choice([arch.novelty(g) for g in pop], rng)
        else:
            i = int(np.argmax(fit))
        child = np.clip(pop[i] + rng.normal(0, 0.08, 2), 0.0, 1.0)
        cf = _deceptive_fitness(child)
        arch.add(child, cf)
        visited.append(child.copy())
        j = (int(np.argmin([arch.novelty(p) for p in pop])) if novelty_on
             else int(np.argmin(fit)))
        pop[j], fit[j] = child, cf
    return visited


def test_novelty_on_explores_more_than_off():
    cov_on = _grid_coverage(_toy_search(True, seed=0))
    cov_off = _grid_coverage(_toy_search(False, seed=0))
    print(f"  behavior-space coverage: novelty-on={cov_on:.0%}  novelty-off={cov_off:.0%}")
    assert cov_on > cov_off, f"novelty-on {cov_on:.0%} not > novelty-off {cov_off:.0%}"
    assert cov_on >= 1.5 * cov_off                          # and by a clear margin


# --- end-to-end QD wiring (needs mujoco) -----------------------------------
def test_qd_novelty_emitter_runs_and_logs_bc():
    from personal_cambrian.seeds import quadruped
    from personal_cambrian.evo import run_qd, QDConfig
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=6,
                   seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                   ep_steps=40, hidden=32, macro_rate=0.5, novelty=True, seed=0)
    logs = []
    archive, _ = run_qd([quadruped()], cfg, log_fn=logs.append)
    assert all("bc_cells" in r and "bc_spread" in r and "bc_archive" in r for r in logs)
    assert logs[-1]["bc_archive"] >= 1
    # every surviving elite carries a behavior characterization
    assert all("bc" in e.meta for e in archive.grid.values())


def test_qd_novelty_is_reproducible():
    from personal_cambrian.seeds import quadruped
    from personal_cambrian.evo import run_qd, QDConfig
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=6,
                   seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                   ep_steps=40, hidden=32, macro_rate=0.5, novelty=True, seed=0)
    a, _ = run_qd([quadruped()], cfg)
    b, _ = run_qd([quadruped()], cfg)
    assert a.coverage == b.coverage and a.qd_score == b.qd_score


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"PASS  {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL  {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    sys.exit(0 if passed == len(fns) else 1)
