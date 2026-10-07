"""Stage 5.3 acceptance (fast): the MAP-Elites ask/tell loop runs end-to-end and
fills the archive. The real (longer) run is scripts/evolve.py.

Runs:  python3 tests/test_qd.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.evo import run_qd, QDConfig
from personal_cambrian.evo.phylogeny import Phylogeny
from personal_cambrian.evo.archive import MAPElites

TINY = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=6,
                seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                ep_steps=40, hidden=32, macro_rate=0.5, seed=0)


def test_qd_runs_and_fills_archive():
    archive, phylo = run_qd([quadruped()], TINY)
    assert isinstance(archive, MAPElites) and isinstance(phylo, Phylogeny)
    assert len(archive.grid) >= 1                 # at least the seed got placed
    assert np.isfinite(archive.qd_score)
    assert 0 < archive.coverage <= 1.0


def test_phylogeny_tracks_lineage():
    archive, phylo = run_qd([quadruped()], TINY)
    assert len(phylo.roots()) == 1                # one founder
    root = phylo.roots()[0]
    # the founder plus its evaluated offspring are recorded
    assert len(phylo.nodes) >= 1
    assert all(root in phylo.ancestry(nid) for nid in phylo.nodes)   # single tree


def test_archive_logs_progress():
    logs = []
    run_qd([quadruped()], TINY, log_fn=logs.append)
    assert len(logs) == TINY.iterations
    assert all("coverage" in r and "qd_score" in r and "cells" in r for r in logs)
    # coverage is monotonic non-decreasing (cells only ever added)
    covs = [r["coverage"] for r in logs]
    assert all(b >= a - 1e-12 for a, b in zip(covs, covs[1:]))


def test_elites_reference_phylogeny_nodes():
    archive, phylo = run_qd([quadruped()], TINY)
    for e in archive.elites():
        assert e.meta["node"] in phylo.nodes      # every elite maps to a lineage node


def test_qd_is_deterministic():
    a1, _ = run_qd([quadruped()], TINY)
    a2, _ = run_qd([quadruped()], TINY)
    assert len(a1.grid) == len(a2.grid)
    assert abs(a1.qd_score - a2.qd_score) < 1e-6


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
