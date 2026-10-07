"""Realism classification + biological-implausibility distance.

A genotype is labeled by the *most extreme* envelope any of its genes occupies:
all genes inside HA -> HUMAN-ACHIEVABLE; any gene outside HA but inside HP ->
HUMAN-POSSIBLE; any gene outside HP -> OPEN-EVOLUTION. The implausibility
penalty grows with how far genes sit outside the HA envelope, so fitness can
allow exotic bodies while still accounting for their reduced realism.
"""
from __future__ import annotations

from enum import IntEnum
import numpy as np

from .genome import Genotype, GENE_SPECS


class Realism(IntEnum):
    HUMAN_ACHIEVABLE = 0
    HUMAN_POSSIBLE = 1
    OPEN_EVOLUTION = 2

    @property
    def label(self) -> str:
        return {0: "HUMAN-ACHIEVABLE", 1: "HUMAN-POSSIBLE", 2: "OPEN-EVOLUTION"}[int(self)]


def classify(g: Genotype) -> Realism:
    level = Realism.HUMAN_ACHIEVABLE
    for i, spec in enumerate(GENE_SPECS):
        v = g.x[i]
        if not (spec.ha[0] <= v <= spec.ha[1]):
            level = max(level, Realism.HUMAN_POSSIBLE)
        if not (spec.hp[0] <= v <= spec.hp[1]):
            level = max(level, Realism.OPEN_EVOLUTION)
    return level


def implausibility(g: Genotype) -> float:
    """Continuous distance outside the HA envelope, in summed envelope-widths.

    0.0 means fully human-achievable. Used directly as the biological-
    implausibility penalty term in fitness.
    """
    total = 0.0
    for i, spec in enumerate(GENE_SPECS):
        v = g.x[i]
        lo, hi = spec.ha
        width = hi - lo
        if v < lo:
            total += (lo - v) / width
        elif v > hi:
            total += (v - hi) / width
    return total


def distance_from(g: Genotype, ref: Genotype) -> float:
    """Normalized Euclidean distance (in HA-width units) — 'similarity to me'."""
    d = 0.0
    for i, spec in enumerate(GENE_SPECS):
        width = spec.ha[1] - spec.ha[0]
        d += ((g.x[i] - ref.x[i]) / width) ** 2
    return float(np.sqrt(d))
