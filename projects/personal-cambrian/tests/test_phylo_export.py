"""Stage 7.4 acceptance: phylogeny persistence + innovation-annotated export.

The family tree exports to NHX Newick (read by FigTree / Dendroscope / ETE /
Archaeopteryx and the Stage-10 Blender) with innovation events annotated on
branches, plus a JSON record. We prove it is viewer-loadable without a tree-parser
dependency by showing the annotated tree is well-formed and that, once the NHX
comments and branch lengths are stripped, its topology is IDENTICAL to the
canonical Newick -- i.e. the annotations are additive and parseable.

Runs:  python3 tests/test_phylo_export.py   (requires mujoco for develop/hash)
"""
import copy
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from personal_cambrian.seeds import quadruped
from personal_cambrian.evo import Phylogeny


def _tree():
    """root -> a (+muscle) -> b (+muscle), with deterministic innovations."""
    phy = Phylogeny()
    base = quadruped()
    root = phy.add_root(base, fitness=0.5)

    a_genome = copy.deepcopy(base)
    m = copy.deepcopy(a_genome.muscles[0]); m.id = "extraA"
    a_genome.muscles.append(m)
    a = phy.add_offspring(root, a_genome, [], fitness=1.0)

    b_genome = copy.deepcopy(a_genome)
    m2 = copy.deepcopy(b_genome.muscles[0]); m2.id = "extraB"
    b_genome.muscles.append(m2)
    b = phy.add_offspring(a, b_genome, [], fitness=2.0)
    return phy, {"root": root, "a": a, "b": b}


def _strip(nwk: str) -> str:
    nwk = re.sub(r"\[&&NHX:[^\]]*\]", "", nwk)      # drop NHX comments
    return re.sub(r":[0-9]+(\.[0-9]+)?", "", nwk)   # drop branch lengths


def _nhx_fields(nwk: str, nid: str) -> dict:
    m = re.search(re.escape(nid) + r"(?::[0-9.]+)?\[&&NHX:([^\]]*)\]", nwk)
    assert m, f"no NHX block for {nid}"
    return dict(kv.split("=", 1) for kv in m.group(1).split(":"))


# --- back-compat: the plain export is unchanged ----------------------------
def test_plain_newick_still_topology_only():
    phy, n = _tree()
    nwk = phy.to_newick()
    assert nwk.endswith(";") and "[&&NHX" not in nwk and ":" not in nwk
    for nid in n.values():
        assert nid in nwk


# --- NHX export is well-formed and topology-preserving ---------------------
def test_nhx_strips_back_to_canonical_topology():
    phy, _ = _tree()
    assert _strip(phy.to_nhx()) == phy.to_newick()   # annotations are additive
    assert phy.to_nhx().count("[&&NHX:") == len(phy.nodes)


def test_branch_lengths_are_graph_edit_distance():
    phy, n = _tree()
    nwk = phy.to_nhx()
    for nid in n.values():
        m = re.search(re.escape(nid) + r":([0-9]+)\[", nwk)
        assert m and int(m.group(1)) == phy.nodes[nid].distance


def test_innovation_markers_on_branches():
    phy, n = _tree()
    nwk = phy.to_nhx()
    # a and b each added a muscle -> +muscle annotated on their branches
    assert "+muscle" in _nhx_fields(nwk, n["a"]).get("innov", "")
    assert "+muscle" in _nhx_fields(nwk, n["b"]).get("innov", "")
    # every node carries the basic fields a viewer reads
    for nid in n.values():
        f = _nhx_fields(nwk, nid)
        assert "gen" in f and "dist" in f and "hash" in f


# --- persistence ------------------------------------------------------------
def test_export_writes_both_files_and_json_roundtrips():
    phy, n = _tree()
    with tempfile.TemporaryDirectory() as d:
        paths = phy.export(os.path.join(d, "family"))
        assert os.path.getsize(paths["newick"]) > 0
        assert os.path.getsize(paths["json"]) > 0
        with open(paths["newick"]) as f:
            assert f.read().strip() == phy.to_nhx()
        loaded = Phylogeny.load(paths["json"])
    # structure + innovations survive the JSON round-trip
    assert set(loaded.nodes) == set(phy.nodes)
    assert "+muscle" in loaded.nodes[n["a"]].innovations
    assert loaded.ancestry(n["b"]) == phy.ancestry(n["b"])


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
