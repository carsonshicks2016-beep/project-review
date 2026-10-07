"""LINEAGE-DIVERGENCE METRICS (ROADMAP Stage 7.5): do niches drive distinct shapes?

If different selective regimes really push body plans in different directions, the
populations evolving under different niches should pull APART in morphology space
over generations. We quantify that:

  * niche centroid       -- the mean normalised descriptor vector of a niche's
                            current population (its "average body").
  * inter-niche separation -- the mean pairwise distance between niche centroids;
                            a single scalar for "how spread out are the niches".
  * divergence trajectory -- that separation per generation; a RISING curve is the
                            signature of niches separating (vs. averaging back to
                            one plan).

These operate on plain {niche -> [genomes]} collections, so they work on a live
archive, a phylogeny slice, or a recorded run. A matplotlib-guarded plotter draws
the trajectory and a 2-D descriptor-space scatter of the centroids.
"""
from __future__ import annotations

import numpy as np

from .distance import descriptor_vector, _MORPH_KEYS
from .metrics import has_matplotlib

_N = len(_MORPH_KEYS)


def niche_centroid(genomes) -> np.ndarray:
    """Mean normalised descriptor vector of a niche's population (its average body)."""
    if not genomes:
        return np.zeros(_N)
    return np.mean([descriptor_vector(g) for g in genomes], axis=0)


def centroid_distance(c1, c2) -> float:
    """Distance between two centroids, scaled to [0, 1] like descriptor_distance."""
    return float(np.linalg.norm(np.asarray(c1) - np.asarray(c2)) / np.sqrt(_N))


def inter_niche_separation(niche_genomes: dict):
    """(centroids, mean_pairwise_distance) for a {niche -> [genomes]} snapshot."""
    centroids = {k: niche_centroid(v) for k, v in niche_genomes.items()}
    names = sorted(centroids)
    dists = [centroid_distance(centroids[names[i]], centroids[names[j]])
             for i in range(len(names)) for j in range(i + 1, len(names))]
    return centroids, (float(np.mean(dists)) if dists else 0.0)


def divergence_trajectory(generations) -> list:
    """Mean inter-niche separation per generation snapshot (a list of {niche->genomes})."""
    return [inter_niche_separation(snap)[1] for snap in generations]


def per_descriptor_spread(niche_genomes: dict) -> dict:
    """Std-dev across niche centroids per descriptor -- which axes the niches use
    to differ (e.g. one niche grows limbs, another grows mass)."""
    centroids = np.array([niche_centroid(v) for v in niche_genomes.values()])
    if len(centroids) < 2:
        return {k: 0.0 for k in _MORPH_KEYS}
    spread = centroids.std(axis=0)
    return {k: float(spread[i]) for i, k in enumerate(_MORPH_KEYS)}


# --- plotting (matplotlib-guarded) -----------------------------------------
def plot_divergence(trajectory, out: str) -> str:
    """Line plot: inter-niche separation vs generation (rising == diverging)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(trajectory)), trajectory, marker="o")
    ax.set_xlabel("generation")
    ax.set_ylabel("mean inter-niche distance")
    ax.set_title("Lineage divergence: niches separating in morphology space")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def plot_niche_space(niche_genomes: dict, out: str, *, axes=(0, 1)) -> str:
    """Scatter the per-niche centroids in two descriptor dimensions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    i, j = axes
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, genomes in niche_genomes.items():
        c = niche_centroid(genomes)
        ax.scatter(c[i], c[j], s=80)
        ax.annotate(name, (c[i], c[j]), fontsize=9)
    ax.set_xlabel(_MORPH_KEYS[i])
    ax.set_ylabel(_MORPH_KEYS[j])
    ax.set_title("Per-niche mean morphology")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
