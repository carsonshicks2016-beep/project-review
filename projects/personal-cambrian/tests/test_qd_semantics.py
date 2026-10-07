"""Stage 5.5 acceptance: the MAP-Elites guarantees, end to end.

(1) Specialist preservation: a creature that is the best in its own niche is never
    overwritten by stronger creatures in OTHER niches, nor by weaker ones in its
    own cell.
(2) Reproducibility: a fixed seed reproduces the archive bit-for-bit (filled cells,
    fitnesses, elite genomes, descriptors) and the phylogeny.
Plus: a run preserves distinct niches, and a run's archive round-trips to JSON.

Runs:  python3 tests/test_qd_semantics.py   (requires mujoco + gymnasium + torch)
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.evo import run_qd, QDConfig, MAPElites

TINY = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=6,
                seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                ep_steps=40, hidden=32, macro_rate=0.5, seed=0)


def test_weak_specialist_survives_strong_generalists():
    a = MAPElites(["aspect", "limb_count"], bins=10)
    spec = {"aspect": 5.0, "limb_count": 3}
    spec_cell = a.cell(spec)
    a.add(quadruped(), 0.05, spec)                       # weak, but alone in its niche
    for f in (5.0, 6.0, 7.0, 8.0):                       # strong creatures elsewhere
        a.add(quadruped(), f, {"aspect": 1.0, "limb_count": 11})
    assert spec_cell in a.grid and a.grid[spec_cell].fitness == 0.05
    assert a.add(quadruped(), 0.01, spec) is False       # worse in same cell -> rejected
    assert a.grid[spec_cell].fitness == 0.05
    assert a.best().fitness == 8.0                       # global best is the generalist...
    assert len(a.grid) == 2                              # ...but the specialist is kept


def test_qd_run_is_bit_reproducible():
    a1, p1 = run_qd([quadruped()], TINY)
    a2, p2 = run_qd([quadruped()], TINY)
    assert set(a1.grid) == set(a2.grid)
    for cell in a1.grid:
        e1, e2 = a1.grid[cell], a2.grid[cell]
        assert e1.fitness == e2.fitness
        assert e1.genome.hash() == e2.genome.hash()
        assert e1.descriptors == e2.descriptors
    assert a1.coverage == a2.coverage and a1.qd_score == a2.qd_score
    assert len(p1.nodes) == len(p2.nodes)
    assert ({n.genome_hash for n in p1.nodes.values()}
            == {n.genome_hash for n in p2.nodes.values()})


def test_run_preserves_distinct_niches():
    # two very different body plans must occupy different niches
    archive, _ = run_qd([quadruped(), agent_zero()], TINY)
    assert len(archive.grid) >= 2
    assert len(set(archive.grid)) == len(archive.grid)   # cells are distinct keys


def test_run_archive_roundtrips_to_json():
    archive, _ = run_qd([quadruped()], TINY)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "archive.json")
        archive.save(path)                               # meta holds only a node id -> JSON-safe
        loaded = MAPElites.load(path)
    assert set(loaded.grid) == set(archive.grid)
    for cell in archive.grid:
        assert loaded.grid[cell].fitness == archive.grid[cell].fitness
        assert loaded.grid[cell].genome.hash() == archive.grid[cell].genome.hash()


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
