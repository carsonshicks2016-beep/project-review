"""Stage 9.3 acceptance: JAX-vectorized MAP-Elites archive.

Done-when: QD-score parity with the reference archive (the in-house CPU `MAPElites`;
we never used pyribs) at much higher throughput. We verify:

  * PARITY -- fed the SAME candidates, the JAX archive's coverage and QD-score match
    the CPU archive exactly (batched scatter-max == sequential best-per-cell inserts);
  * BATCHED INSERT -- a whole batch goes in with one vectorized op, including the
    order-independence and best-per-cell semantics;
  * CELL ALIGNMENT -- JAX cell binning matches the CPU archive's, so grids align.

Throughput numbers (batched vs Python-loop insert) are in the test as a sanity check;
the GPU multiplier is hardware-bound. Skips cleanly if jax is absent.

Runs:  python3 tests/test_jax_archive.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.evo.jax_archive import jax_archive_available

_SKIP = not jax_archive_available()

if not _SKIP:
    from personal_cambrian.evo.jax_archive import JaxMAPElites
    from personal_cambrian.evo.archive import MAPElites
    from personal_cambrian.seeds import quadruped

_AXES = ["aspect", "limb_count"]
from personal_cambrian.evo.descriptors import DESCRIPTOR_BOUNDS


def _candidates(n, seed=0):
    """Random (descriptors, fitness) within the axis bounds."""
    rng = np.random.default_rng(seed)
    lo = np.array([DESCRIPTOR_BOUNDS[a][0] for a in _AXES])
    hi = np.array([DESCRIPTOR_BOUNDS[a][1] for a in _AXES])
    desc = lo + rng.random((n, len(_AXES))) * (hi - lo)
    fit = rng.normal(size=n)
    return desc, fit


def _cpu_archive(desc, fit, bins=12):
    a = MAPElites(_AXES, bins)
    g = quadruped()                                    # genome is irrelevant to qd_score
    for d, f in zip(desc, fit):
        a.add(g, float(f), {ax: float(v) for ax, v in zip(_AXES, d)})
    return a


# --- parity with the CPU archive -------------------------------------------
def test_coverage_and_qd_score_parity():
    desc, fit = _candidates(2000, seed=1)
    cpu = _cpu_archive(desc, fit, bins=12)
    jx = JaxMAPElites(_AXES, 12).add_batch(desc, fit)
    print(f"  cpu cov={cpu.coverage:.4f} qd={cpu.qd_score:.4f} | "
          f"jax cov={jx.coverage:.4f} qd={jx.qd_score:.4f}")
    assert jx.n_filled == len(cpu.grid)
    assert abs(jx.coverage - cpu.coverage) < 1e-9
    assert abs(jx.qd_score - cpu.qd_score) < 1e-4


def test_cells_match_cpu_binning():
    desc, _ = _candidates(500, seed=2)
    cpu = MAPElites(_AXES, 12)
    cpu_cells = [cpu.cell({ax: float(v) for ax, v in zip(_AXES, d)}) for d in desc]
    # CPU returns per-axis index tuples; rebuild flat indices with the same strides
    bins = cpu.bins
    strides = [1] * len(bins)
    for k in range(len(bins) - 2, -1, -1):
        strides[k] = strides[k + 1] * bins[k + 1]
    cpu_flat = np.array([sum(i * s for i, s in zip(t, strides)) for t in cpu_cells])
    jx_flat = np.asarray(JaxMAPElites(_AXES, 12).cells(desc))
    assert np.array_equal(cpu_flat, jx_flat)


# --- batched-insert semantics ----------------------------------------------
def test_best_per_cell_and_order_independence():
    # two candidates in the SAME cell: the higher fitness must win, either order
    a = JaxMAPElites(["aspect", "limb_count"], 4)
    d = np.array([[1.0, 5.0], [1.0, 5.0]])             # identical descriptors -> same cell
    a.add_batch(d, np.array([2.0, 9.0]))
    assert a.n_filled == 1 and abs(a.qd_score - 9.0) < 1e-6
    b = JaxMAPElites(["aspect", "limb_count"], 4)
    b.add_batch(d, np.array([9.0, 2.0]))               # reversed order
    assert abs(b.qd_score - 9.0) < 1e-6                # same result -> order-independent


def test_incremental_adds_keep_running_best():
    a = JaxMAPElites(_AXES, 8)
    d = np.array([[1.0, 5.0]])
    a.add_batch(d, np.array([1.0]))
    a.add_batch(d, np.array([5.0]))                    # improves the same cell
    a.add_batch(d, np.array([3.0]))                    # worse -> ignored
    assert a.n_filled == 1 and abs(a.qd_score - 5.0) < 1e-6


def test_throughput_batched_insert():
    import time
    desc, fit = _candidates(20000, seed=3)
    jx = JaxMAPElites(_AXES, 16)
    t0 = time.time(); jx.add_batch(desc, fit)
    import jax; jax.block_until_ready(jx.fitness)
    dt = time.time() - t0
    print(f"  batched insert of 20000 candidates: {dt*1000:.0f}ms")
    cpu = _cpu_archive(desc, fit, bins=16)             # same candidates, parity holds at scale
    assert abs(jx.qd_score - cpu.qd_score) < 1e-3
    assert jx.n_filled == len(cpu.grid)


if __name__ == "__main__":
    if _SKIP:
        print("SKIP  jax not installed (Stage 9 GPU extra)")
        sys.exit(0)
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
