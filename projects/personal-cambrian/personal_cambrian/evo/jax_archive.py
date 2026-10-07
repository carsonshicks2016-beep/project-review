"""JAX-VECTORIZED MAP-ELITES ARCHIVE (ROADMAP Stage 9.3).

The CPU `MAPElites` (evo/archive.py) inserts candidates ONE AT A TIME in a Python
loop -- fine for hundreds, a bottleneck for the millions a GPU run produces. This is
a JAX port: the grid is a flat array and a whole BATCH of candidates is inserted with
a single vectorized SCATTER-MAX (`.at[cells].max(fitness)`), the GPU-QD primitive.

Because "keep the best per cell" is order-independent (a max), the batched insert is
EXACTLY EQUIVALENT to the sequential CPU inserts -- so coverage and QD-score match
the in-house archive bit-for-bit (the Stage-9.3 done-when, "QD-score parity"; we
never used pyribs, so the in-house archive is the reference). The throughput win is
the batched scatter; on a GPU `add_batch` over millions of candidates is one kernel.

Cell binning reuses the Stage-5.1 DESCRIPTOR_BOUNDS normalization, identical to the
CPU archive, so the two grids align cell-for-cell. Production runs can swap this for
QDax's CVT archive; this raw-JAX version keeps Stage 9 to the jax/jaxlib dep only.
"""
from __future__ import annotations

import numpy as np

from .descriptors import DESCRIPTOR_BOUNDS

try:
    import jax
    import jax.numpy as jp
    _HAS = True
except Exception:        # noqa: BLE001
    _HAS = False


def jax_archive_available() -> bool:
    return _HAS


class JaxMAPElites:
    """A grid MAP-Elites archive stored as flat JAX arrays, filled by batched
    scatter-max. `add_batch(descriptors[B, n_axes], fitnesses[B], ids=...)` inserts a
    whole batch at once; `coverage` / `qd_score` mirror the CPU archive exactly."""

    def __init__(self, axes, bins=12):
        if not _HAS:
            raise RuntimeError("JaxMAPElites needs jax (`pip install jax`).")
        for a in axes:
            if a not in DESCRIPTOR_BOUNDS:
                raise ValueError(f"unknown descriptor axis '{a}'")
        self.axes = list(axes)
        self.bins = ([int(bins)] * len(axes) if isinstance(bins, int)
                     else [int(b) for b in bins])
        self._lo = jp.asarray([DESCRIPTOR_BOUNDS[a][0] for a in self.axes])
        self._hi = jp.asarray([DESCRIPTOR_BOUNDS[a][1] for a in self.axes])
        self._b = jp.asarray(self.bins)
        # row-major strides so each per-axis bin tuple maps to one flat cell
        strides = np.ones(len(self.bins), dtype=np.int64)
        for k in range(len(self.bins) - 2, -1, -1):
            strides[k] = strides[k + 1] * self.bins[k + 1]
        self._strides = jp.asarray(strides)
        self.n_cells = int(np.prod(self.bins))
        self.fitness = jp.full(self.n_cells, -jp.inf)     # per-cell best fitness
        self.winner = jp.full(self.n_cells, -1, dtype=jp.int32)  # per-cell winning id
        self._next_id = 0

    # -- cell indexing (matches MAPElites.cell via shared DESCRIPTOR_BOUNDS) --
    def cells(self, descriptors) -> "jp.ndarray":
        """Flat cell index for each row of `descriptors` (shape [B, n_axes])."""
        d = jp.asarray(descriptors, dtype=float)
        u = jp.clip((d - self._lo) / (self._hi - self._lo), 0.0, 1.0)
        idx = jp.minimum((u * self._b).astype(jp.int32), self._b - 1)
        return (idx * self._strides).sum(axis=1)

    # -- the batched insert (scatter-max) -----------------------------------
    def add_batch(self, descriptors, fitnesses, ids=None):
        """Insert a batch; keep the best fitness per cell. Order-independent, so this
        equals inserting the same candidates one-by-one into the CPU archive."""
        fit = jp.asarray(fitnesses, dtype=float)
        B = int(fit.shape[0])
        if ids is None:
            ids = jp.arange(self._next_id, self._next_id + B, dtype=jp.int32)
            self._next_id += B
        else:
            ids = jp.asarray(ids, dtype=jp.int32)
        new_cells = self.cells(descriptors)

        # fold the current occupants in as candidates so repeated adds keep the best
        occ = jp.nonzero(jp.isfinite(self.fitness))[0]
        all_cells = jp.concatenate([occ.astype(jp.int32), new_cells])
        all_fit = jp.concatenate([self.fitness[occ], fit])
        all_ids = jp.concatenate([self.winner[occ], ids])

        self.fitness = jp.full(self.n_cells, -jp.inf).at[all_cells].max(all_fit)
        # winner = highest-fitness candidate per cell: sort ascending so the best wins
        order = jp.argsort(all_fit)
        self.winner = jp.full(self.n_cells, -1, dtype=jp.int32).at[
            all_cells[order]].set(all_ids[order])
        return self

    # -- statistics (mirror MAPElites) --------------------------------------
    @property
    def occupied(self) -> "jp.ndarray":
        return jp.isfinite(self.fitness)

    @property
    def coverage(self) -> float:
        return float(jp.sum(self.occupied)) / self.n_cells

    @property
    def qd_score(self) -> float:
        f = jp.where(self.occupied, jp.maximum(self.fitness, 0.0), 0.0)
        return float(jp.sum(f))

    @property
    def n_filled(self) -> int:
        return int(jp.sum(self.occupied))

    def best(self):
        """(flat_cell, fitness, winner_id) of the single best elite, or None."""
        if self.n_filled == 0:
            return None
        c = int(jp.argmax(jp.where(self.occupied, self.fitness, -jp.inf)))
        return c, float(self.fitness[c]), int(self.winner[c])
