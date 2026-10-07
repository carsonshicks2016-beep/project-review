"""Stage 8.5 acceptance: deep-time experiment analysis.

The headline run's done-when is *a deep, branching tree with clearly non-human
lineages and an innovation timeline*. The analysis machinery that detects those
properties is tested deterministically on a hand-built REAL phylogeny (micro vs
macro lineages grown from the quadruped seed). We then check the new multi-niche
QD wiring and run a cheap real evolution end-to-end, asserting it yields a
branching tree with structural innovations and genuine divergence (full depth /
distance thresholds are reserved for the slow headline run in scripts/deep_time.py).

Runs:  python3 tests/test_deeptime.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from personal_cambrian.seeds import quadruped
from personal_cambrian.evo.deeptime import (
    tree_metrics, lineage_divergence, divergent_lineages, most_divergent,
    innovation_timeline, deep_time_report,
)


def _build_tree():
    """A real branching phylogeny: a micro (near-clone) lineage and a macro lineage
    that duplicates subtrees (adds limbs/muscles -> divergence + innovations)."""
    from personal_cambrian.evo.phylogeny import Phylogeny
    from personal_cambrian.encoding.mutate import duplicate_subtree, micro
    rng = np.random.default_rng(0)
    seed = quadruped()
    phylo = Phylogeny()
    root = phylo.add_root(seed, generation=0)
    a = micro(seed, rng, 0.1)                              # low-divergence twig
    na = phylo.add_offspring(root, a, generation=1)
    b, _ = duplicate_subtree(seed, rng)                    # macro: structural change
    nb = phylo.add_offspring(root, b, generation=1)
    b2, _ = duplicate_subtree(b, rng)
    nb2 = phylo.add_offspring(nb, b2, generation=2)
    b3, _ = duplicate_subtree(b2, rng)
    nb3 = phylo.add_offspring(nb2, b3, generation=3)
    return phylo, seed, {"root": root, "a": na, "b": nb, "b2": nb2, "b3": nb3}


# --- tree shape -------------------------------------------------------------
def test_tree_metrics_on_branching_tree():
    phylo, seed, ids = _build_tree()
    tm = tree_metrics(phylo)
    assert tm["n_creatures"] == 5
    assert tm["max_depth"] == 3                            # root -> b -> b2 -> b3
    assert tm["n_roots"] == 1
    assert tm["n_branch_points"] >= 1                      # root forks into a and b
    assert tm["n_leaves"] == 2                             # a and b3 are tips


# --- divergence (the non-human axis) ---------------------------------------
def test_divergence_grows_along_the_macro_lineage():
    phylo, seed, ids = _build_tree()
    div = lineage_divergence(phylo, seed_genome=seed)
    assert div[ids["root"]] == 0.0                         # root IS the seed
    assert div[ids["b3"]] > div[ids["b"]] > 0.0            # accumulates with duplications
    assert div[ids["b3"]] > div[ids["a"]]                  # macro lineage drifts farther


def test_divergent_lineages_and_most_divergent():
    phylo, seed, ids = _build_tree()
    a_dist = lineage_divergence(phylo, seed_genome=seed)[ids["a"]]
    nh = divergent_lineages(phylo, threshold=a_dist + 1e-6, seed_genome=seed)
    assert ids["b3"] in nh and ids["a"] not in nh          # tip filter, sorted by distance
    nid, d = most_divergent(phylo, seed_genome=seed)
    assert nid == ids["b3"] and d > 0.0


# --- innovation timeline ----------------------------------------------------
def test_innovation_timeline_orders_first_appearances():
    phylo, seed, ids = _build_tree()
    tl = innovation_timeline(phylo)
    assert len(tl) >= 1                                    # duplications introduced flags
    gens = [e["generation"] for e in tl]
    assert gens == sorted(gens)                            # earliest first
    assert all({"generation", "flag", "node"} <= set(e) for e in tl)
    assert len({e["flag"] for e in tl}) == len(tl)         # one entry per flag (first only)


# --- the done-when report ---------------------------------------------------
def test_deep_time_report_meets_done_when():
    phylo, seed, ids = _build_tree()
    rep = deep_time_report(phylo, seed_genome=seed, divergence_threshold=0.1, min_depth=3)
    dw = rep["done_when"]
    assert dw["deep"] and dw["branching"]
    assert dw["non_human_lineages"] and dw["innovation_timeline"]
    assert rep["met"] is True
    assert rep["most_divergent"]["node"] == ids["b3"]
    assert rep["n_non_human"] >= 1


def test_report_on_empty_and_single_node():
    from personal_cambrian.evo.phylogeny import Phylogeny
    empty = deep_time_report(Phylogeny())
    assert empty["met"] is False and empty["tree"]["n_creatures"] == 0
    p = Phylogeny()
    p.add_root(quadruped(), generation=0)
    one = deep_time_report(p, min_depth=0)
    assert not one["done_when"]["branching"]               # a lone root does not branch


# --- multi-niche QD wiring (needs mujoco) ----------------------------------
def test_qd_runs_under_a_chosen_niche():
    from personal_cambrian.evo import run_qd, QDConfig
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=4, niche="terrain",
                   seed_steps=256, fine_tune_steps=256, n_envs=2, n_steps=64,
                   ep_steps=40, hidden=32, macro_rate=0.5, seed=0)
    archive, phylo = run_qd([quadruped()], cfg)
    assert len(phylo.nodes) >= 1                           # ran end-to-end under 'terrain'


# --- a cheap real deep-time run (needs mujoco) -----------------------------
def test_deep_time_run_smoke():
    from personal_cambrian.evo import run_qd, QDConfig
    cfg = QDConfig(axes=["aspect", "limb_count"], bins=8, iterations=24,
                   niche="locomotion", seed_steps=256, fine_tune_steps=256,
                   n_envs=2, n_steps=64, ep_steps=40, hidden=32, macro_rate=0.7,
                   novelty=True, seed=0)
    archive, phylo = run_qd([quadruped()], cfg)
    rep = deep_time_report(phylo, divergence_threshold=0.05, min_depth=2)
    assert rep["tree"]["n_creatures"] >= 5
    assert rep["done_when"]["branching"]                   # the lineage tree forks
    assert rep["n_innovation_types"] >= 1                  # macro mutation produced novelty
    _, d = most_divergent(phylo)
    assert d > 0.0                                         # creatures drifted from the seed


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
