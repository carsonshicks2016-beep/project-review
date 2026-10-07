"""Stage 10.5 acceptance: side-by-side lineage comparison (bpy-free core).

The comparison IMAGE is rendered by Blender (verified by running it); here we test the
bpy-free plan builder: that `lineage_plan` arrays N morphologies along an axis with
prefixed, non-colliding names and correct offsets, and that `lineage_genomes` pulls a
root->leaf lineage out of a phylogeny.

Runs:  python3 tests/test_compare.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.morphogenesis import develop
from personal_cambrian.viz import lineage_plan, lineage_genomes


def test_lineage_plan_offsets_and_prefixes():
    bc = develop(quadruped()).body_count
    plan = lineage_plan([quadruped(), quadruped()], spacing=2.0, axis=0)
    assert len(plan.geoms) == 2 * bc
    assert all(g.name.startswith(("g0_", "g1_")) for g in plan.geoms)
    # each body in creature 1 is shifted +2.0 in x from its counterpart in creature 0
    x0 = {g.name[3:]: g.pos[0] for g in plan.geoms if g.name.startswith("g0_")}
    for g in plan.geoms:
        if g.name.startswith("g1_"):
            assert abs((g.pos[0] - x0[g.name[3:]]) - 2.0) < 1e-6
    roots = [g for g in plan.geoms if g.parent is None]
    assert len(roots) == 2                                 # one root per creature
    # parents are prefixed consistently (no cross-creature links)
    for g in plan.geoms:
        if g.parent is not None:
            assert g.parent[:3] == g.name[:3]


def test_lineage_plan_auto_spacing_separates_creatures():
    plan = lineage_plan([quadruped(), quadruped(), quadruped()], axis=0)
    xs_per = []
    for i in range(3):
        xs = [g.pos[0] for g in plan.geoms if g.name.startswith(f"g{i}_")]
        xs_per.append(sum(xs) / len(xs))
    assert xs_per[0] < xs_per[1] < xs_per[2]               # creatures arrayed in order
    lo, hi = plan.bounds()
    assert hi[0] - lo[0] > 1.0                             # the row spans a real width


def _lineage_phylo():
    from personal_cambrian.evo.phylogeny import Phylogeny
    from personal_cambrian.encoding.mutate import duplicate_subtree
    rng = np.random.default_rng(0)
    p = Phylogeny()
    g0 = quadruped()
    r = p.add_root(g0, generation=0)
    g1, _ = duplicate_subtree(g0, rng); n1 = p.add_offspring(r, g1, generation=1)
    g2, _ = duplicate_subtree(g1, rng); n2 = p.add_offspring(n1, g2, generation=2)
    return p, [r, n1, n2]


def test_lineage_genomes_follows_ancestry():
    p, chain = _lineage_phylo()
    genomes = lineage_genomes(p, leaf=chain[-1])
    assert len(genomes) == 3                               # root -> n1 -> n2
    assert genomes[0].hash() == p.genome(chain[0]).hash()  # starts at the root
    assert genomes[-1].hash() == p.genome(chain[-1]).hash()


def test_lineage_genomes_default_picks_most_divergent():
    p, chain = _lineage_phylo()
    genomes = lineage_genomes(p)                           # default = most-divergent leaf
    assert len(genomes) >= 2 and genomes[0].hash() == p.genome(chain[0]).hash()


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
