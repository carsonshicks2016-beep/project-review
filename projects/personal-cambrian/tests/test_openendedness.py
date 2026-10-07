"""Stage 8.4 acceptance: open-endedness metrics.

The "done-when" is that the metrics are computed over a LONG run and a plateau (if
any) is identified and analyzed. We prove it RL-free: a long toy-POET run with
unbounded difficulty yields a SUSTAINED ANNECS curve (no terminal plateau), while
a difficulty-CAPPED run saturates and the plateau is detected -- so the metric
discriminates sustained innovation from a stalled curriculum. We also unit-test the
primitives (cumulative / windowed rate / plateau detection) and the adapters
(ANNECS, coverage growth, phylogeny innovation series) on deterministic inputs.

Runs:  python3 tests/test_openendedness.py
"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.evo.openendedness import (
    cumulative, windowed_rate, detect_plateau, annecs, annecs_curve,
    coverage_growth, innovation_series, analyze,
)
from personal_cambrian.evo.poet import POET, POETConfig


# --- generic primitives -----------------------------------------------------
def test_cumulative_and_windowed_rate():
    assert list(cumulative([1, 0, 2, 0, 1])) == [1, 1, 3, 3, 4]
    r = windowed_rate(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), window=2)
    assert r[-1] == 1.0                                   # constant slope of 1


def test_detect_plateau_on_saturating_curve():
    cum = np.concatenate([np.arange(0.0, 20.0, 2.0), np.full(30, 18.0)])  # rises then flat
    p = detect_plateau(cum, window=5, rel_threshold=0.1)
    assert p is not None and p >= 9 and p < cum.size      # onset is in the flat tail


def test_no_plateau_on_linear_curve():
    assert detect_plateau(np.arange(0.0, 50.0), window=5, rel_threshold=0.1) is None


def test_flat_curve_is_plateau_from_start():
    assert detect_plateau(np.zeros(10)) == 0


# --- ANNECS -----------------------------------------------------------------
def test_annecs_counts_only_novel_and_solved():
    outs = [
        {"descriptor": [0.0], "best_score": 1.0},   # solved + novel  -> +1
        {"descriptor": [0.05], "best_score": 1.0},  # solved but a near-duplicate -> no
        {"descriptor": [1.0], "best_score": -1.0},  # novel but UNSOLVED -> no
        {"descriptor": [2.0], "best_score": 1.0},   # solved + novel  -> +1
    ]
    c = annecs_curve(outs, solve_threshold=0.0, novelty_threshold=0.5)
    assert list(c) == [1, 1, 1, 2]
    assert annecs(outs, solve_threshold=0.0, novelty_threshold=0.5) == 2


# --- adapters ---------------------------------------------------------------
def test_coverage_growth_is_new_cells_per_iter():
    logs = [{"cells": 1}, {"cells": 3}, {"cells": 3}, {"cells": 6}]
    g = coverage_growth(logs, key="cells")
    assert list(g) == [1, 2, 0, 3]                        # new cells per iteration
    assert list(cumulative(g)) == [1, 3, 3, 6]


def test_innovation_series_from_phylogeny():
    from personal_cambrian.evo.phylogeny import Phylogeny, PhyloNode
    p = Phylogeny()
    p.nodes = {
        "a": PhyloNode("a", "h0", None, 0, innovations=[]),
        "b": PhyloNode("b", "h1", "a", 1, innovations=["+limb"]),
        "c": PhyloNode("c", "h2", "b", 2, innovations=["+muscle", "topology_change"]),
    }
    s = innovation_series(p)
    assert list(s) == [0, 1, 2]                            # in generation order
    assert cumulative(s)[-1] == 3


def test_analyze_reports_plateau_and_verdict():
    sustained = analyze(np.ones(40), window=5)             # one innovation every step
    assert sustained["sustained"] and sustained["plateau_at"] is None
    bursts = [1] * 10 + [0] * 40                           # innovates then stops
    stalled = analyze(bursts, window=5, rel_threshold=0.1)
    assert not stalled["sustained"] and stalled["plateau_at"] is not None


# --- the done-when: metrics over a long toy-POET run -----------------------
@dataclass
class _Env:
    target: float


@dataclass
class _Agent:
    skill: float


def _opt(a, e, steps):
    s = a.skill
    for _ in range(steps):
        s += 0.5 * max(0.0, (e.target + 0.5) - s)
    return _Agent(s)


def _ev(a, e):
    return a.skill - e.target


def _mut(cap=None):
    def f(e, rng, step=0.5):
        d = step if rng.random() < 0.85 else -0.25 * step
        t = max(0.0, e.target + d)
        return _Env(min(cap, t) if cap is not None else t)
    return f


def _toy_poet(mutate):
    cfg = POETConfig(max_active=5, reproduce_every=1, transfer_every=3, n_children=4,
                     max_admit=1, optimize_steps=2, mc_low=-0.3, mc_high=0.3,
                     repro_threshold=0.3, novelty_k=3)
    p = POET(cfg, optimize=_opt, evaluate=_ev, env_mutate=mutate,
             env_descriptor=lambda e: np.array([e.target]),
             env_difficulty=lambda e: e.target, rng=np.random.default_rng(0))
    p.add_env(_Env(0.0), _Agent(0.0))
    return p


def test_long_run_unbounded_is_sustained():
    p = _toy_poet(_mut(cap=None))
    p.run(150)
    curve = annecs_curve(p.env_outcomes(), solve_threshold=0.0,
                         novelty_threshold=0.25, n_steps=150)
    print(f"  unbounded: ANNECS={curve[-1]:.0f}, plateau="
          f"{detect_plateau(curve, window=10, rel_threshold=0.15)}")
    assert curve[-1] > 10                                  # many novel niches solved
    assert detect_plateau(curve, window=10, rel_threshold=0.15) is None  # still innovating


def test_long_run_capped_plateaus_and_is_detected():
    p = _toy_poet(_mut(cap=2.0))                           # difficulty cannot keep rising
    p.run(150)
    curve = annecs_curve(p.env_outcomes(), solve_threshold=0.0,
                         novelty_threshold=0.25, n_steps=150)
    onset = detect_plateau(curve, window=10, rel_threshold=0.15)
    print(f"  capped: ANNECS={curve[-1]:.0f}, plateau onset={onset}")
    assert curve[-1] <= 12                                 # only a few reachable niches
    assert onset is not None and onset < curve.size        # the stall is identified


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
