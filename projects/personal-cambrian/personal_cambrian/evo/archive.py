"""MAP-Elites archive (ROADMAP Stage 5.2).

Quality-diversity: instead of optimizing one averaged reward, we keep the
highest-fitness creature in each *niche* of a descriptor grid. Specialists are
preserved even when they score poorly elsewhere (each cell keeps its own best).

An N-D grid over the normalized descriptor axes (5.1's DESCRIPTOR_BOUNDS). Each
cell holds an `Elite` (genome + fitness + descriptors + meta, where meta can carry
a controller-checkpoint path). Reports coverage and QD-score, supports sampling
(for the ask step in 5.3), a 2-D ASCII heatmap, and JSON persistence.

This is intentionally dependency-free; pyribs / QDax (GPU, CVT archives) is the
Stage-9 scale-up swap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..encoding.genome import Genome
from .descriptors import normalize, DESCRIPTOR_BOUNDS


@dataclass
class Elite:
    genome: Genome
    fitness: float
    descriptors: dict
    meta: dict = field(default_factory=dict)        # JSON-able (e.g. controller path)


class MAPElites:
    def __init__(self, axes: list[str], bins=16):
        for a in axes:
            if a not in DESCRIPTOR_BOUNDS:
                raise ValueError(f"unknown descriptor axis '{a}'")
        self.axes = list(axes)
        self.bins = [int(bins)] * len(axes) if isinstance(bins, int) else [int(b) for b in bins]
        self.grid: dict[tuple, Elite] = {}
        self.n_evals = 0
        self.n_improvements = 0

    # -- cell indexing ------------------------------------------------------
    def cell(self, descriptors: dict) -> tuple:
        idx = []
        for ax, b in zip(self.axes, self.bins):
            u = normalize(ax, descriptors[ax])             # [0,1]
            idx.append(min(int(u * b), b - 1))
        return tuple(idx)

    # -- core ---------------------------------------------------------------
    def add(self, genome: Genome, fitness: float, descriptors: dict, **meta) -> bool:
        """Insert if the cell is empty or this beats its incumbent. Returns True
        if the archive changed (the QD 'tell' step)."""
        self.n_evals += 1
        c = self.cell(descriptors)
        cur = self.grid.get(c)
        if cur is None or fitness > cur.fitness:
            self.grid[c] = Elite(genome, float(fitness), dict(descriptors), dict(meta))
            self.n_improvements += 1
            return True
        return False

    # -- statistics ---------------------------------------------------------
    @property
    def n_cells(self) -> int:
        return int(np.prod(self.bins))

    @property
    def coverage(self) -> float:
        return len(self.grid) / self.n_cells

    @property
    def qd_score(self) -> float:
        return float(sum(max(e.fitness, 0.0) for e in self.grid.values()))

    def best(self) -> Optional[Elite]:
        return max(self.grid.values(), key=lambda e: e.fitness) if self.grid else None

    def best_where(self, predicate) -> Optional[Elite]:
        elites = [e for e in self.grid.values() if predicate(e)]
        return max(elites, key=lambda e: e.fitness) if elites else None

    def elites(self) -> list:
        return list(self.grid.values())

    def sample(self, rng) -> Optional[Elite]:
        if not self.grid:
            return None
        keys = list(self.grid.keys())
        return self.grid[keys[int(rng.integers(len(keys)))]]

    # -- 2-D ascii heatmap (when exactly 2 axes) ----------------------------
    def ascii_map(self) -> str:
        if len(self.axes) != 2:
            return f"<{len(self.axes)}-D archive: {len(self.grid)}/{self.n_cells} cells>"
        bx, by = self.bins
        m = np.full((bx, by), np.nan)
        for (i, j), e in self.grid.items():
            m[i, j] = e.fitness
        finite = m[np.isfinite(m)]
        if finite.size == 0:
            return "(empty archive)"
        lo, hi = float(finite.min()), float(finite.max())
        ramp = " .:-=+*#%@"
        lines = []
        for j in range(by - 1, -1, -1):                     # axis-1 high on top
            row = []
            for i in range(bx):
                v = m[i, j]
                if not np.isfinite(v):
                    row.append(" ")
                else:
                    t = 1.0 if hi == lo else (v - lo) / (hi - lo)
                    row.append(ramp[min(int(t * (len(ramp) - 1)), len(ramp) - 1)])
            lines.append("".join(row))
        return "\n".join(lines)

    # -- persistence --------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "axes": self.axes, "bins": self.bins,
            "n_evals": self.n_evals, "n_improvements": self.n_improvements,
            "cells": [{"cell": list(c), "genome": e.genome.to_dict(),
                       "fitness": e.fitness, "descriptors": e.descriptors, "meta": e.meta}
                      for c, e in self.grid.items()],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MAPElites":
        a = cls(d["axes"], d["bins"])
        a.n_evals = d.get("n_evals", 0)
        a.n_improvements = d.get("n_improvements", 0)
        for c in d["cells"]:
            a.grid[tuple(c["cell"])] = Elite(Genome.from_dict(c["genome"]), c["fitness"],
                                             c["descriptors"], c.get("meta", {}))
        return a

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "MAPElites":
        with open(path) as f:
            return cls.from_dict(json.load(f))
