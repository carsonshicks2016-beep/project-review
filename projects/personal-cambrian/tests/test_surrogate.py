"""Stage 9.4 acceptance: surrogate-assisted quality-diversity.

Done-when: match the no-surrogate QD-score with MATERIALLY FEWER SIMS (ablation).
This payoff is hardware-independent (sims are a count), so it is verified directly:
on a controllable domain where fitness is predictable from genome features, the
surrogate-assisted run reaches the no-surrogate run's QD-score with far fewer
expensive evaluations -- and a higher QD-score at equal budget. We also unit-test
the surrogate (learns a function, uncertainty rises away from data) and the sim-free
`genome_features`.

Runs:  python3 tests/test_surrogate.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.evo.surrogate import (
    Surrogate, genome_features, surrogate_assisted_qd, SurrogateQDConfig, sims_to_reach,
)


# --- surrogate model --------------------------------------------------------
def test_surrogate_learns_a_function():
    surr = Surrogate(k=3)
    X = np.array([[a, b] for a in range(-3, 4) for b in range(-3, 4)], dtype=float)
    surr.update(X, X[:, 0] + X[:, 1])                 # y = x0 + x1
    assert surr.trained()
    pred, _ = surr.predict([[0.5, 0.5]])
    assert abs(pred[0] - 1.0) < 0.6                   # ~ 0.5 + 0.5


def test_uncertainty_rises_away_from_data():
    surr = Surrogate(k=3)
    X = np.array([[a, b] for a in range(-2, 3) for b in range(-2, 3)], dtype=float)
    surr.update(X, (X ** 2).sum(1))
    _, near = surr.predict([[0.0, 0.0]])
    _, far = surr.predict([[20.0, 20.0]])
    assert far[0] > near[0]


def test_untrained_surrogate_is_safe():
    surr = Surrogate(k=5)
    mean, unc = surr.predict([[1.0, 2.0]])            # nothing learned yet
    assert mean.shape == (1,) and unc[0] == 1.0


# --- sim-free genome features ----------------------------------------------
def test_genome_features_shape_and_change():
    from personal_cambrian.seeds import quadruped
    from personal_cambrian.encoding.mutate import duplicate_subtree
    f0 = genome_features(quadruped())
    assert f0.shape == (6,) and np.all(np.isfinite(f0)) and np.all((f0 >= 0) & (f0 <= 1))
    bigger, _ = duplicate_subtree(quadruped(), np.random.default_rng(0))
    assert not np.allclose(genome_features(bigger), f0)   # added structure shifts features


# --- the done-when: same QD-score with fewer sims --------------------------
_G = 8


def _cell(x):
    b = np.clip(((np.clip(x[:2], -2, 2) + 2) / 4 * _G).astype(int), 0, _G - 1)
    return (int(b[0]), int(b[1]))


def _evaluate(x):
    # within a (x0,x1) cell, fitness peaks when the HIDDEN dims x2,x3 ~ (0.5,0.5):
    # a learnable structure the surrogate can exploit to pick better candidates.
    fit = 2.0 * np.exp(-((x[2] - 0.5) ** 2 + (x[3] - 0.5) ** 2) / 0.3) - 0.05 * (x[0] ** 2 + x[1] ** 2)
    return float(fit), _cell(x)


def _mutate(x, rng):
    return np.clip(x + rng.normal(0, 0.4, 4), -2.0, 2.0)


def _seeds():
    rng = np.random.default_rng(0)
    return [rng.uniform(-2, 2, 4) for _ in range(4)]


def test_surrogate_matches_qd_with_fewer_sims():
    base = surrogate_assisted_qd(_seeds(), _evaluate, _mutate, lambda x: np.asarray(x, float),
                                 SurrogateQDConfig(sim_budget=300, use_surrogate=False, seed=0))
    surr = surrogate_assisted_qd(_seeds(), _evaluate, _mutate, lambda x: np.asarray(x, float),
                                 SurrogateQDConfig(sim_budget=300, use_surrogate=True,
                                                   n_candidates=8, beta=0.5, warmup=20, seed=0))
    reach = sims_to_reach(surr.qd_curve, base.qd_score)
    print(f"  baseline qd={base.qd_score:.2f} (300 sims) | surrogate qd={surr.qd_score:.2f} "
          f"| reached baseline qd in {reach} sims")
    assert surr.qd_score > 1.3 * base.qd_score        # much higher QD-score at equal budget
    assert reach is not None and reach < 0.8 * base.n_sims   # baseline qd with >=20% fewer sims
    assert surr.coverage >= base.coverage             # diversity not sacrificed


def test_surrogate_off_is_plain_map_elites():
    # use_surrogate=False must be deterministic and equal to a random-mutation baseline
    a = surrogate_assisted_qd(_seeds(), _evaluate, _mutate, lambda x: np.asarray(x, float),
                              SurrogateQDConfig(sim_budget=120, use_surrogate=False, seed=0))
    b = surrogate_assisted_qd(_seeds(), _evaluate, _mutate, lambda x: np.asarray(x, float),
                              SurrogateQDConfig(sim_budget=120, use_surrogate=False, seed=0))
    assert a.qd_score == b.qd_score and a.n_sims == 120


def test_sims_to_reach():
    curve = [(1, 0.0), (2, 1.0), (3, 2.5), (4, 5.0)]
    assert sims_to_reach(curve, 2.0) == 3
    assert sims_to_reach(curve, 99.0) is None


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
