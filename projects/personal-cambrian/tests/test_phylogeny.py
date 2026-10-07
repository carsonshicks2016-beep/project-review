"""Stage 4.5 acceptance: a creature's ancestry + innovation timeline are queryable,
and the lineage persists.

Builds a real multi-generation lineage with macro-mutations, then checks ancestry,
auto-classified innovations, the innovation timeline, first-appearance lookup,
descendants, and JSON round-trip.

Runs:  python3 tests/test_phylogeny.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.encoding.mutate import add_part, add_muscle, toggle_symmetry
from personal_cambrian.evo import Phylogeny


def _build():
    """root --add_part--> a --add_muscle--> b ; root --toggle_symmetry--> c"""
    rng = np.random.default_rng(1)
    phy = Phylogeny()
    g_root = quadruped()
    root = phy.add_root(g_root, fitness=0.0)

    g_a, ra = add_part(g_root, rng)
    a = phy.add_offspring(root, g_a, [ra], fitness=1.0)

    g_b, rb = add_muscle(g_a, rng)
    b = phy.add_offspring(a, g_b, [rb])

    g_c, rc = toggle_symmetry(g_root, rng)
    c = phy.add_offspring(root, g_c, [rc])
    return phy, dict(root=root, a=a, b=b, c=c)


def test_ancestry_chain():
    phy, n = _build()
    assert phy.ancestry(n["b"]) == [n["root"], n["a"], n["b"]]
    assert phy.ancestry(n["c"]) == [n["root"], n["c"]]
    assert phy.roots() == [n["root"]]
    assert phy.depth(n["b"]) == 2


def test_innovations_auto_classified():
    phy, n = _build()
    assert "+limb" in phy.nodes[n["a"]].innovations          # add_part
    assert "+muscle" in phy.nodes[n["b"]].innovations        # add_muscle
    assert "topology_change" in phy.nodes[n["c"]].innovations  # toggle_symmetry
    assert phy.nodes[n["a"]].distance >= 1
    assert phy.nodes[n["a"]].mutations == ["add_part"]


def test_innovation_timeline_and_first_appearance():
    phy, n = _build()
    tl = phy.innovation_timeline(n["b"])                     # along root->a->b
    flagged = {nid for nid, _, _ in tl}
    assert n["a"] in flagged and n["b"] in flagged           # both introduced novelty
    assert n["root"] not in flagged                          # root introduced none
    assert phy.first_appearance("+limb") == n["a"]
    assert phy.first_appearance("nonexistent") is None


def test_children_and_descendants():
    phy, n = _build()
    assert set(phy.children(n["root"])) == {n["a"], n["c"]}
    assert set(phy.descendants(n["root"])) == {n["a"], n["b"], n["c"]}
    assert phy.children(n["b"]) == []


def test_generation_increments():
    phy, n = _build()
    assert phy.nodes[n["root"]].generation == 0
    assert phy.nodes[n["a"]].generation == 1
    assert phy.nodes[n["b"]].generation == 2


def test_meta_stored():
    phy, n = _build()
    assert phy.nodes[n["a"]].meta["fitness"] == 1.0


def test_json_roundtrip_preserves_tree_and_innovations():
    phy, n = _build()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "phylo.json")
        phy.save(path)
        loaded = Phylogeny.load(path)
    assert loaded.ancestry(n["b"]) == [n["root"], n["a"], n["b"]]
    assert "+limb" in loaded.nodes[n["a"]].innovations
    assert set(loaded.children(n["root"])) == {n["a"], n["c"]}
    assert loaded.first_appearance("+muscle") == n["b"]


def test_newick_contains_all_nodes():
    phy, n = _build()
    nwk = phy.to_newick()
    assert nwk.endswith(";")
    for nid in n.values():
        assert nid in nwk


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
