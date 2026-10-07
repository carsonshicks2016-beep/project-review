"""SPECIATION / NICHING IN SELECTION (ROADMAP Stage 7.3).

MAP-Elites already niches by descriptor *cell*, but a dominant body plan that
happens to fill many cells still gets most of the reproduction, and can slowly
crowd out a rarer, divergent lineage. Speciation is an orthogonal protection:
cluster the current elites into SPECIES by morphological distance (Stage 7.2),
then give every species an equal share of reproduction regardless of how many
cells it occupies. A lineage with one weird body is then as likely to breed as a
lineage with twenty similar ones -- so divergent plans persist instead of
averaging back to one.

Clustering is greedy threshold ("compatibility") clustering, NEAT-style: each
genome joins the nearest existing species within `threshold`, else founds a new
one. It is deterministic given the input order, which keeps QD reproducible.
"""
from __future__ import annotations

from .distance import morphological_distance


def speciate(genomes, threshold: float, distance=morphological_distance):
    """Greedy threshold clustering. Returns (labels, representatives).

    `labels[i]` is the species index of `genomes[i]`; `representatives[k]` is the
    founding genome of species k. Deterministic for a fixed input order."""
    reps = []
    labels = []
    for g in genomes:
        best, best_d = None, float("inf")
        for i, rep in enumerate(reps):
            d = distance(g, rep)
            if d < best_d:
                best, best_d = i, d
        if best is not None and best_d <= threshold:
            labels.append(best)
        else:
            reps.append(g)
            labels.append(len(reps) - 1)
    return labels, reps


def count_species(genomes, threshold: float, distance=morphological_distance) -> int:
    """Number of species among `genomes` at the given compatibility threshold."""
    if not genomes:
        return 0
    return len(speciate(genomes, threshold, distance)[1])


def group_by_species(elites, threshold: float, distance=morphological_distance):
    """Group archive Elites (each has `.genome`) into species -> list of elites."""
    labels, _ = speciate([e.genome for e in elites], threshold, distance)
    groups: dict[int, list] = {}
    for elite, lab in zip(elites, labels):
        groups.setdefault(lab, []).append(elite)
    return groups


def sample_species_aware(elites, rng, threshold: float,
                         distance=morphological_distance):
    """Sample an elite giving every species equal probability (then uniform within).

    This is the niching: a one-member species is sampled as often as a many-member
    one, so rare/divergent lineages keep reproducing. Returns None if no elites."""
    if not elites:
        return None
    groups = group_by_species(elites, threshold, distance)
    species = sorted(groups)                       # deterministic order
    chosen = species[int(rng.integers(len(species)))]
    members = groups[chosen]
    return members[int(rng.integers(len(members)))]
