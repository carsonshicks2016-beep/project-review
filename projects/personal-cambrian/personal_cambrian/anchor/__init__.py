"""Reality anchor (ROADMAP Stage 11, OPTIONAL/low-priority).

Keeps only the NEAR-HUMAN region of the evolutionary tree plausible when real data is
supplied -- ingesting measured metrics (Whoop, lifting app) with provenance, setting
the Agent-Zero seed from them, and (Stage 11.4) a Bayesian update bounded to the
human-achievable region. The deep-time / open-evolution branch is deliberately left
UNTOUCHED (Stage 11.5 asserts this).

Binding rule: never invent missing personal data -- absent quantities are `unknown`,
never fabricated (see `Metric`).
"""
from .metric import Metric, MetricSet, aggregate
from .whoop import parse_whoop, load_default
from .lifting import parse_lifting
from .seed_update import (
    SeedPrior, build_priors, apply_geometry, anchored_seed, standing_height,
)
from .calibrate import (
    LocalLinearTrend, calibrate_and_validate, lift_progression, HUMAN_BOUNDS,
)

__all__ = ["Metric", "MetricSet", "aggregate", "parse_whoop", "load_default",
           "parse_lifting", "SeedPrior", "build_priors", "apply_geometry",
           "anchored_seed", "standing_height",
           "LocalLinearTrend", "calibrate_and_validate", "lift_progression",
           "HUMAN_BOUNDS"]
