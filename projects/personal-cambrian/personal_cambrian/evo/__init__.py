"""Evolution engine bookkeeping (ROADMAP Stage 4.5+).

Currently: the `Phylogeny` — the lineage tree of body plans, recording each
parent->child mutation and the structural innovations it introduced. This is a
primary project output (the anatomical family tree) and feeds speciation (Stage
7) and the Blender viz (Stage 10). The MAP-Elites archive (Stage 5) will live
here too.
"""
from .phylogeny import Phylogeny, PhyloNode
from . import descriptors
from .descriptors import (
    morphology_descriptors, behavior_descriptors, distance_from, compute,
    normalize, normalized, DESCRIPTOR_BOUNDS,
)
from .archive import MAPElites, Elite
from .qd import run_qd, QDConfig, pre_sim_gate, budget_penalty
from .metrics import (
    archive_novelty, fitness_grid, save_metrics, load_metrics,
    plot_metrics, plot_archive, has_matplotlib,
)
from .viability import check_viability, ViabilityReport
from .structure import check_structure, StructuralReport
from .supply import check_supply, SupplyReport
from .distance import (
    morphological_distance, descriptor_distance, graph_distance,
    distance_matrix, descriptor_vector,
)
from .speciation import (
    speciate, count_species, group_by_species, sample_species_aware,
)
from .divergence import (
    niche_centroid, centroid_distance, inter_niche_separation,
    divergence_trajectory, per_descriptor_spread, plot_divergence, plot_niche_space,
)
from .specialization import (
    tradeoff_matrix, train_specialist, score_on, column_normalize,
    specialist_diagonal_fraction, is_specialist_diagonal, plot_tradeoff_matrix,
)
from .behavior import (
    BehaviorChar, characterize, bc_distance, bc_matrix, nearest, novelty,
    LABELS as BC_LABELS, DIM as BC_DIM,
)
from .novelty import (
    NoveltyArchive, novelty_emitter, nslc_scores, proportional_choice,
    behavior_cells, behavior_spread,
)
from .poet import (
    POET, POETConfig, POETPair, mutate_niche, niche_descriptor, make_creature_poet,
)
from .openendedness import (
    cumulative, windowed_rate, detect_plateau, annecs, annecs_curve,
    coverage_growth, innovation_series, analyze, plot_openendedness,
)
from .deeptime import (
    tree_metrics, lineage_divergence, divergent_lineages, most_divergent,
    innovation_timeline, deep_time_report,
)
from .jax_archive import JaxMAPElites, jax_archive_available
from .surrogate import (
    Surrogate, genome_features, surrogate_assisted_qd, SurrogateQDConfig,
    SurrogateQDResult, sims_to_reach,
)
from .benchmarks import (
    stage9_report, save_report, surrogate_sims_benchmark, jax_archive_benchmark,
)

__all__ = [
    "Phylogeny", "PhyloNode", "descriptors",
    "morphology_descriptors", "behavior_descriptors", "distance_from", "compute",
    "normalize", "normalized", "DESCRIPTOR_BOUNDS",
    "MAPElites", "Elite",
    "run_qd", "QDConfig", "pre_sim_gate", "budget_penalty",
    "archive_novelty", "fitness_grid", "save_metrics", "load_metrics",
    "plot_metrics", "plot_archive", "has_matplotlib",
    "check_viability", "ViabilityReport",
    "check_structure", "StructuralReport",
    "check_supply", "SupplyReport",
    "morphological_distance", "descriptor_distance", "graph_distance",
    "distance_matrix", "descriptor_vector",
    "speciate", "count_species", "group_by_species", "sample_species_aware",
    "niche_centroid", "centroid_distance", "inter_niche_separation",
    "divergence_trajectory", "per_descriptor_spread", "plot_divergence",
    "plot_niche_space",
    "tradeoff_matrix", "train_specialist", "score_on", "column_normalize",
    "specialist_diagonal_fraction", "is_specialist_diagonal", "plot_tradeoff_matrix",
    "BehaviorChar", "characterize", "bc_distance", "bc_matrix", "nearest", "novelty",
    "BC_LABELS", "BC_DIM",
    "NoveltyArchive", "novelty_emitter", "nslc_scores", "proportional_choice",
    "behavior_cells", "behavior_spread",
    "POET", "POETConfig", "POETPair", "mutate_niche", "niche_descriptor",
    "make_creature_poet",
    "cumulative", "windowed_rate", "detect_plateau", "annecs", "annecs_curve",
    "coverage_growth", "innovation_series", "analyze", "plot_openendedness",
    "tree_metrics", "lineage_divergence", "divergent_lineages", "most_divergent",
    "innovation_timeline", "deep_time_report",
    "JaxMAPElites", "jax_archive_available",
    "Surrogate", "genome_features", "surrogate_assisted_qd", "SurrogateQDConfig",
    "SurrogateQDResult", "sims_to_reach",
    "stage9_report", "save_report", "surrogate_sims_benchmark", "jax_archive_benchmark",
]
