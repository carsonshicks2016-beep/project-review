"""DEEP-TIME EXPERIMENT ANALYSIS (ROADMAP Stage 8.5).

Stage 8.5 is the headline run: long open-ended evolution composing everything built
so far (generative encoding -> viability/budget gates -> warm-started control ->
MAP-Elites + novelty emitter + speciation, optionally under a POET-generated niche).
This module reads the RESULT of such a run -- a `Phylogeny` (and optional archive) --
and quantifies the Stage-8.5 done-when: *a deep, branching tree with clearly
non-human lineages and an innovation timeline.*

It answers four questions:

  * DEEP & BRANCHING?  -- `tree_metrics`: max lineage depth, branch points, leaves,
    mean branching factor. A deep run is not a single chain; it forks.
  * NON-HUMAN LINEAGES? -- `divergent_lineages`: tips whose morphological distance
    (Stage 7.2) from the founding seed exceeds a threshold -- creatures that have
    drifted far from the human ancestor, the whole point of the project.
  * INNOVATION TIMELINE? -- `innovation_timeline`: when each structural innovation
    (+limb / +muscle / topology_change / ...) FIRST appeared across the tree.
  * SUSTAINED? -- folds in the Stage-8.4 plateau verdict on the innovation series.

`deep_time_report` rolls these into one summary with a boolean done-when assessment.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .distance import morphological_distance
from .openendedness import innovation_series, cumulative, detect_plateau


# --- tree shape -------------------------------------------------------------
def tree_metrics(phylo) -> dict:
    """Shape of the lineage tree: size, depth, and how much it branches."""
    nodes = phylo.nodes
    n = len(nodes)
    if n == 0:
        return {"n_creatures": 0, "max_depth": 0, "n_roots": 0, "n_leaves": 0,
                "n_branch_points": 0, "mean_branching": 0.0}
    child_counts = [len(phylo.children(nid)) for nid in nodes]
    internal = [c for c in child_counts if c > 0]
    return {
        "n_creatures": n,
        "max_depth": max(phylo.depth(nid) for nid in nodes),
        "n_roots": len(phylo.roots()),
        "n_leaves": len(phylo.leaves()),
        "n_branch_points": sum(1 for c in child_counts if c >= 2),
        "mean_branching": float(np.mean(internal)) if internal else 0.0,
    }


# --- divergence from the founder (the "non-human" axis) --------------------
def _seed_genome(phylo, seed_genome):
    if seed_genome is not None:
        return seed_genome
    roots = phylo.roots()
    return phylo.genome(roots[0]) if roots else None


def lineage_divergence(phylo, *, seed_genome=None) -> dict:
    """Morphological distance (Stage 7.2, in [0,1]) of every node's genome from the
    founding seed. Nodes whose genome was not retained are skipped."""
    seed = _seed_genome(phylo, seed_genome)
    out = {}
    if seed is None:
        return out
    for nid in phylo.nodes:
        g = phylo.genome(nid)
        if g is not None:
            out[nid] = morphological_distance(seed, g)
    return out


def divergent_lineages(phylo, *, threshold: float = 0.3, seed_genome=None,
                       leaves_only: bool = True) -> list:
    """Node ids that have drifted FAR from the seed (distance > threshold) -- the
    "clearly non-human" lineages. Restricted to tree tips by default. Sorted by
    descending divergence."""
    div = lineage_divergence(phylo, seed_genome=seed_genome)
    pool = set(phylo.leaves()) if leaves_only else set(phylo.nodes)
    hits = [(nid, d) for nid, d in div.items() if nid in pool and d > threshold]
    hits.sort(key=lambda kd: -kd[1])
    return [nid for nid, _ in hits]


def most_divergent(phylo, *, seed_genome=None):
    """(node_id, distance) of the single most non-human creature, or (None, 0.0)."""
    div = lineage_divergence(phylo, seed_genome=seed_genome)
    if not div:
        return None, 0.0
    nid = max(div, key=div.get)
    return nid, float(div[nid])


# --- innovation timeline ----------------------------------------------------
def innovation_timeline(phylo) -> list:
    """First appearance of each structural innovation across the whole tree, ordered
    by generation: list of {generation, flag, node}. This is the run's macro-history
    of *what kind of novelty arose and when*."""
    first = {}
    for flag_node in ((f, nid) for nid, nd in phylo.nodes.items() for f in nd.innovations):
        flag, nid = flag_node
        gen = phylo.nodes[nid].generation
        if flag not in first or gen < first[flag][0]:
            first[flag] = (gen, nid)
    timeline = [{"generation": g, "flag": flag, "node": nid}
                for flag, (g, nid) in first.items()]
    timeline.sort(key=lambda e: (e["generation"], e["flag"]))
    return timeline


# --- the done-when report ---------------------------------------------------
def deep_time_report(phylo, *, seed_genome=None, divergence_threshold: float = 0.3,
                     min_depth: int = 5, plateau_window: int = 10) -> dict:
    """One-shot Stage-8.5 assessment. Combines tree shape, non-human lineages, the
    innovation timeline, and the Stage-8.4 plateau verdict, with a boolean done-when:
    a deep AND branching tree that produced at least one clearly non-human lineage and
    a non-empty innovation timeline."""
    tm = tree_metrics(phylo)
    timeline = innovation_timeline(phylo)
    nh = divergent_lineages(phylo, threshold=divergence_threshold, seed_genome=seed_genome)
    top_nid, top_d = most_divergent(phylo, seed_genome=seed_genome)
    innov_cum = cumulative(innovation_series(phylo))
    plateau = detect_plateau(innov_cum, window=plateau_window) if innov_cum.size else None

    deep = tm["max_depth"] >= min_depth
    branching = tm["n_branch_points"] >= 1
    non_human = len(nh) >= 1
    has_timeline = len(timeline) >= 1
    return {
        "tree": tm,
        "innovation_timeline": timeline,
        "n_innovation_types": len(timeline),
        "non_human_lineages": nh,
        "n_non_human": len(nh),
        "most_divergent": {"node": top_nid, "distance": top_d},
        "innovation_plateau_at": plateau,
        "innovation_sustained": plateau is None,
        "done_when": {
            "deep": deep, "branching": branching,
            "non_human_lineages": non_human, "innovation_timeline": has_timeline,
        },
        "met": bool(deep and branching and non_human and has_timeline),
    }
