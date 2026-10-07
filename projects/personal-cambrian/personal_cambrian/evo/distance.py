"""MORPHOLOGICAL DISTANCE METRIC (ROADMAP Stage 7.2): how different are two bodies?

Speciation (7.3) and lineage-divergence metrics (7.5) need a single number for
"how far apart are these two body plans". We combine two complementary views:

  * graph distance     -- a graph-edit-distance on the genome GRAPHS (parts +
    (topology)            muscles added/removed, routing/joint/symmetry changes).
                          This is the Stage-4.4 `structural_distance`. It captures
                          DISCRETE plan differences a descriptor vector misses
                          (e.g. a re-routed muscle), and is 0 for a continuous
                          (micro) tweak.
  * descriptor distance -- Euclidean distance in NORMALISED descriptor space
    (proportions)         (limb count, muscle count, mass, height, aspect,
                          symmetry; Stage 5.1). This captures continuous shape
                          change two topologically-identical bodies still have.

Both are mapped to ~[0, 1] and blended, so the combined distance is symmetric,
zero for identical bodies, and bounded -- a drop-in for clustering / matrices.
"""
from __future__ import annotations

import numpy as np

from ..encoding.genome import Genome
from ..encoding.innovation import structural_distance
from .descriptors import morphology_descriptors, normalize

# descriptors that describe the BODY (exclude behaviour + the seed-relative one)
_MORPH_KEYS = ("limb_count", "muscle_count", "mass", "height", "aspect", "symmetry")


def descriptor_vector(genome: Genome, morph=None) -> np.ndarray:
    """Normalised ([0,1] per axis) morphology descriptor vector."""
    d = morphology_descriptors(genome, morph)
    return np.array([normalize(k, d[k]) for k in _MORPH_KEYS], dtype=float)


def descriptor_distance(g1: Genome, g2: Genome) -> float:
    """Euclidean distance in normalised descriptor space, scaled to [0, 1]."""
    diff = descriptor_vector(g1) - descriptor_vector(g2)
    return float(np.linalg.norm(diff) / np.sqrt(len(_MORPH_KEYS)))


def graph_distance(g1: Genome, g2: Genome, *, normalize_by_size: bool = True) -> float:
    """Graph-edit distance between genome graphs (Stage 4.4 structural_distance).

    Normalised by the combined element count so it lands in ~[0, 1] (you cannot
    edit more elements than both bodies contain)."""
    raw = structural_distance(g1, g2)
    if not normalize_by_size:
        return float(raw)
    size = len(g1.parts) + len(g1.muscles) + len(g2.parts) + len(g2.muscles)
    return float(raw / max(size, 1))


def morphological_distance(g1: Genome, g2: Genome, *,
                           w_graph: float = 0.5, w_desc: float = 0.5) -> float:
    """Blended topology + proportion distance. Symmetric; 0 for identical bodies."""
    return w_graph * graph_distance(g1, g2) + w_desc * descriptor_distance(g1, g2)


def distance_matrix(genomes, *, w_graph: float = 0.5, w_desc: float = 0.5) -> np.ndarray:
    """Symmetric pairwise morphological-distance matrix (zero diagonal)."""
    n = len(genomes)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = morphological_distance(genomes[i], genomes[j],
                                       w_graph=w_graph, w_desc=w_desc)
            M[i, j] = M[j, i] = d
    return M
