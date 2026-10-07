"""Stage 10.3 acceptance: phylogeny tree render.

Done-when: a tree image shows lineages + the innovation timeline. We test the pure
layout/marker logic deterministically on a hand-built tree (so the render rests on
correct geometry) and that the render writes a PNG -- from both a live Phylogeny and
its exported JSON (Stage 7.4).

Runs:  python3 tests/test_phylo_render.py
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.evo.phylogeny import Phylogeny, PhyloNode
from personal_cambrian.evo.metrics import has_matplotlib
from personal_cambrian.viz.phylo_render import (
    tree_layout, innovation_markers, render_phylogeny, render_from_json,
)


def _tree():
    p = Phylogeny()
    p.nodes = {
        "r": PhyloNode("r", "h0", None, 0, innovations=[]),
        "a": PhyloNode("a", "h1", "r", 1, innovations=[]),
        "b": PhyloNode("b", "h2", "r", 1, innovations=["+limb"]),
        "c": PhyloNode("c", "h3", "b", 2, innovations=["+muscle", "topology_change"]),
    }
    p._children = {"r": ["a", "b"], "a": [], "b": ["c"], "c": []}
    return p


def test_layout_places_generations_and_centres_parents():
    pos = tree_layout(_tree())
    assert pos["r"][0] == 0 and pos["a"][0] == 1 and pos["c"][0] == 2   # x = generation
    assert pos["a"][1] != pos["c"][1]                                   # leaves on distinct rows
    assert pos["b"][1] == pos["c"][1]                                   # single-child centred on it
    assert pos["r"][1] == (pos["a"][1] + pos["b"][1]) / 2               # root centred over kids


def test_innovation_markers_at_first_appearance():
    marks = innovation_markers(_tree())
    flags = sorted(m[2] for m in marks)
    assert flags == ["+limb", "+muscle", "topology_change"]             # one per innovation
    pos = tree_layout(_tree())
    by_flag = {f: (x, y) for x, y, f in marks}
    assert by_flag["+limb"] == pos["b"]                                 # placed on the branch
    assert by_flag["+muscle"] == pos["c"]


def test_normalize_accepts_phylogeny_and_json():
    import json
    p = _tree()
    a = tree_layout(p)
    b = tree_layout(json.loads(json.dumps(p.to_dict())))               # via exported JSON
    assert a.keys() == b.keys()
    assert a["c"] == b["c"]


def test_render_writes_png_from_phylogeny():
    if not has_matplotlib():
        print("  (skipped: no matplotlib)")
        return
    with tempfile.TemporaryDirectory() as d:
        out = render_phylogeny(_tree(), os.path.join(d, "tree.png"), title="t")
        assert os.path.getsize(out) > 1500


def test_render_writes_png_from_json():
    if not has_matplotlib():
        print("  (skipped: no matplotlib)")
        return
    with tempfile.TemporaryDirectory() as d:
        jp = os.path.join(d, "phylo.json")
        import json
        with open(jp, "w") as f:
            json.dump(_tree().to_dict(), f)
        out = render_from_json(jp, os.path.join(d, "tree.png"))
        assert os.path.getsize(out) > 1500


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
