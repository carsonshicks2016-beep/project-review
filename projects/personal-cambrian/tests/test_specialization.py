"""Stage 7.6 acceptance: per-niche specialization tradeoff matrix.

The diagonal-detection logic (the heart of "the matrix shows a clear specialist
diagonal") is tested deterministically on synthetic matrices. The RL matrix
BUILDER is exercised by a tiny smoke run (shape / finite / reproducible); the
real, budgeted diagonal is produced by scripts/specialization_matrix.py.

Runs:  python3 tests/test_specialization.py   (smoke needs mujoco + gymnasium + torch)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.evo import (
    tradeoff_matrix, column_normalize, specialist_diagonal_fraction,
    is_specialist_diagonal, plot_tradeoff_matrix,
)
from personal_cambrian.evo.metrics import has_matplotlib


# --- analysis on synthetic matrices (deterministic) ------------------------
def test_perfect_diagonal_is_detected():
    # each specialist is best on its own niche (column-wise), worse elsewhere
    M = np.array([[10.0, 1.0, 2.0],
                  [3.0, 12.0, 1.0],
                  [0.0, 2.0, 9.0]])
    assert specialist_diagonal_fraction(M) == 1.0
    assert is_specialist_diagonal(M)


def test_generalist_breaks_the_diagonal():
    # specialist 0 dominates every column -> no tradeoff, weak diagonal
    M = np.array([[10.0, 10.0, 10.0],
                  [1.0, 9.0, 1.0],
                  [1.0, 1.0, 9.0]])
    assert specialist_diagonal_fraction(M) < 1.0
    assert not is_specialist_diagonal(M, min_fraction=0.9)


def test_column_normalize_scales_each_niche():
    M = np.array([[0.0, 100.0],
                  [10.0, 0.0]])
    N = column_normalize(M)
    assert np.allclose(N.min(axis=0), 0.0)
    assert np.allclose(N.max(axis=0), 1.0)
    # a constant column collapses to 0 (no spurious winner)
    C = column_normalize(np.array([[5.0, 5.0], [5.0, 5.0]]))
    assert np.allclose(C, 0.0)


def test_diagonal_fraction_is_column_wise():
    # 2x2 where row 0 wins col 0, row 1 wins col 1 -> fraction 1.0
    M = np.array([[5.0, 1.0], [1.0, 5.0]])
    assert specialist_diagonal_fraction(M) == 1.0
    # swap so neither diagonal wins -> 0.0
    assert specialist_diagonal_fraction(np.array([[1.0, 5.0], [5.0, 1.0]])) == 0.0


def test_plot_renders():
    if not has_matplotlib():
        print("  (skipped: no matplotlib)")
        return
    M = np.array([[10.0, 1.0], [1.0, 8.0]])
    with tempfile.TemporaryDirectory() as d:
        p = plot_tradeoff_matrix(M, ["track", "ballistics"], os.path.join(d, "tm.png"))
        assert os.path.getsize(p) > 1000


# --- the RL builder: tiny smoke (no diagonal claim at this budget) ----------
def test_tradeoff_matrix_builder_smoke():
    niches = ["track", "ballistics"]
    kw = dict(train_steps=256, ep_steps=40, hidden=32, n_envs=2, n_steps=64, seed=0)
    M, names = tradeoff_matrix(quadruped(), niches, **kw)
    assert names == niches
    assert M.shape == (2, 2)
    assert np.all(np.isfinite(M))


def test_tradeoff_matrix_is_reproducible():
    niches = ["track", "ballistics"]
    kw = dict(train_steps=256, ep_steps=40, hidden=32, n_envs=2, n_steps=64, seed=0)
    M1, _ = tradeoff_matrix(quadruped(), niches, **kw)
    M2, _ = tradeoff_matrix(quadruped(), niches, **kw)
    assert np.array_equal(M1, M2)


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
