"""Minimal MAP-Elites archive (NumPy only).

A 2D grid over behavior descriptors; each cell keeps the highest-fitness elite
seen for that niche. This is the quality-diversity core: we are NOT optimizing a
single averaged reward, we are filling a map of *distinct* athletic solutions
and keeping the best individual in each niche. Swap for pyribs/QDax to scale to
more descriptor dimensions (see PLAN.md technology selections).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional
import numpy as np


@dataclass
class Elite:
    genome: Any          # (Genotype, TrainingProgram)
    fitness: float
    eval: Any            # EvalResult
    realism: int


class MapElites:
    def __init__(self, resolution: int = 24):
        self.res = resolution
        self.grid: dict[tuple[int, int], Elite] = {}
        self.n_evals = 0
        self.n_improvements = 0

    def _cell(self, descriptors: tuple[float, float]) -> tuple[int, int]:
        i = min(int(descriptors[0] * self.res), self.res - 1)
        j = min(int(descriptors[1] * self.res), self.res - 1)
        return (i, j)

    def add(self, genome: Any, fitness: float, eval_result: Any, realism: int) -> bool:
        self.n_evals += 1
        cell = self._cell(eval_result.descriptors)
        cur = self.grid.get(cell)
        if cur is None or fitness > cur.fitness:
            self.grid[cell] = Elite(genome, fitness, eval_result, realism)
            self.n_improvements += 1
            return True
        return False

    # --- archive statistics ------------------------------------------------
    @property
    def coverage(self) -> float:
        return len(self.grid) / (self.res * self.res)

    @property
    def qd_score(self) -> float:
        """Sum of elite fitnesses (clamped at 0) — the standard QD quality metric."""
        return float(sum(max(e.fitness, 0.0) for e in self.grid.values()))

    def best(self) -> Optional[Elite]:
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda e: e.fitness)

    def best_in_realism(self, realism: int) -> Optional[Elite]:
        elites = [e for e in self.grid.values() if e.realism == realism]
        return max(elites, key=lambda e: e.fitness) if elites else None

    def sample_elite(self, rng: np.random.Generator) -> Optional[Elite]:
        if not self.grid:
            return None
        keys = list(self.grid.keys())
        return self.grid[keys[rng.integers(len(keys))]]

    def fitness_map(self) -> np.ndarray:
        m = np.full((self.res, self.res), np.nan)
        for (i, j), e in self.grid.items():
            m[i, j] = e.fitness
        return m

    def ascii_map(self) -> str:
        """Coarse text heatmap of fitness (x=strength/mass, y=v_max)."""
        m = self.fitness_map()
        finite = m[np.isfinite(m)]
        if finite.size == 0:
            return "(empty archive)"
        lo, hi = float(finite.min()), float(finite.max())
        ramp = " .:-=+*#%@"
        lines = []
        for j in range(self.res - 1, -1, -1):           # high v_max on top
            row = []
            for i in range(self.res):
                v = m[i, j]
                if not np.isfinite(v):
                    row.append(" ")
                else:
                    t = 0 if hi == lo else (v - lo) / (hi - lo)
                    row.append(ramp[min(int(t * (len(ramp) - 1)), len(ramp) - 1)])
            lines.append("".join(row))
        return "\n".join(lines)
