"""Stage 5.2 acceptance: the MAP-Elites archive keeps the best creature per niche
(specialists preserved), computes coverage + QD-score, and round-trips to JSON.

Runs:  python3 tests/test_archive.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped, agent_zero
from personal_cambrian.evo import MAPElites


def _g():
    return quadruped()


def test_insert_and_replace_semantics():
    a = MAPElites(axes=["aspect", "speed"], bins=10)
    d = {"aspect": 1.0, "speed": 2.0}
    assert a.add(_g(), 1.0, d) is True            # empty cell -> inserted
    assert a.add(_g(), 0.5, d) is False           # same cell, worse -> rejected
    assert a.add(_g(), 2.0, d) is True            # same cell, better -> replaced
    assert len(a.grid) == 1
    assert a.best().fitness == 2.0


def test_specialist_in_other_cell_is_kept():
    a = MAPElites(axes=["aspect", "speed"], bins=10)
    a.add(_g(), 5.0, {"aspect": 1.0, "speed": 2.0})    # strong generalist
    a.add(_g(), 0.1, {"aspect": 5.0, "speed": -1.0})   # weak specialist, different niche
    assert len(a.grid) == 2                            # specialist NOT overwritten
    assert a.coverage == 2 / 100
    assert abs(a.qd_score - 5.1) < 1e-9


def test_distinct_descriptors_map_to_distinct_cells():
    a = MAPElites(axes=["aspect", "speed"], bins=16)
    c1 = a.cell({"aspect": 0.5, "speed": -2.0})
    c2 = a.cell({"aspect": 5.0, "speed": 5.0})
    assert c1 != c2


def test_sample_returns_an_elite():
    a = MAPElites(axes=["mass", "limb_count"], bins=8)
    assert a.sample(np.random.default_rng(0)) is None  # empty
    a.add(_g(), 1.0, {"mass": 20.0, "limb_count": 11})
    e = a.sample(np.random.default_rng(0))
    assert e is not None and e.fitness == 1.0


def test_three_dimensional_archive():
    a = MAPElites(axes=["aspect", "speed", "symmetry"], bins=[6, 6, 4])
    a.add(_g(), 1.0, {"aspect": 1.0, "speed": 0.0, "symmetry": 0.5})
    assert a.n_cells == 6 * 6 * 4 and len(a.grid) == 1


def test_negative_fitness_excluded_from_qd_score():
    a = MAPElites(axes=["aspect", "speed"], bins=10)
    a.add(_g(), -3.0, {"aspect": 1.0, "speed": 2.0})
    a.add(_g(), 4.0, {"aspect": 5.0, "speed": -1.0})
    assert a.qd_score == 4.0                       # max(f,0) summed


def test_ascii_map_renders_for_2d():
    a = MAPElites(axes=["aspect", "speed"], bins=8)
    a.add(_g(), 1.0, {"aspect": 1.0, "speed": 2.0})
    assert "@" in a.ascii_map()                     # the single elite is the max


def test_json_roundtrip():
    a = MAPElites(axes=["aspect", "speed"], bins=12)
    a.add(quadruped(), 1.5, {"aspect": 1.0, "speed": 2.0}, policy="best.pt")
    a.add(agent_zero(), 0.7, {"aspect": 3.0, "speed": -0.5})
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "archive.json")
        a.save(path)
        b = MAPElites.load(path)
    assert b.axes == a.axes and b.bins == a.bins
    assert len(b.grid) == len(a.grid)
    assert abs(b.qd_score - a.qd_score) < 1e-9
    # genome + meta survive
    for c, e in a.grid.items():
        assert b.grid[c].genome.hash() == e.genome.hash()
        assert b.grid[c].meta == e.meta


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
