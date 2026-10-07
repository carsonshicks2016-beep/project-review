"""Bounded reproduction of the published FlyWire mirror, for anatomy only.

Algorithm: navis cb9a5915, flybrains 273333c8. No target-anchor fitting.
Morphops supplies the TPS coefficients and kernel used by navis itself.
"""
import numpy as np
from scipy.spatial.distance import cdist


class AnatomicalMirror:
    def __init__(self, source, target, x_sum, batch_size=500):
        import morphops
        # Compatibility fixes also used in the pinned navis implementation.
        if not hasattr(np, 'row_stack'):
            np.row_stack = np.vstack
        morphops.lmk_util.distance_matrix = cdist
        self.ops = morphops
        self.source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)
        if self.source.shape != target.shape or self.source.ndim != 2 or self.source.shape[1] != 3:
            raise ValueError('Expected matching N by 3 landmark arrays')
        if not np.isfinite(self.source).all() or not np.isfinite(target).all():
            raise ValueError('Nonfinite landmarks')
        if batch_size <= 0 or not np.isfinite(x_sum):
            raise ValueError('Invalid mirror configuration')
        self.x_sum = float(x_sum)
        self.batch_size = int(batch_size)
        self.W, self.A = morphops.tps_coefs(self.source, target)

    def warp_flipped(self, points):
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError('Expected finite N by 3 points')
        out = np.empty_like(points)
        for first in range(0, len(points), self.batch_size):
            part = points[first:first+self.batch_size]
            out[first:first+self.batch_size] = self.ops.P_matrix(part) @ self.A + self.ops.K_matrix(part, self.source) @ self.W
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite transformed coordinates')
        return out

    def transform(self, points):
        points = np.asarray(points, dtype=float).copy()
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError('Expected N by 3 points')
        points[:, 0] = self.x_sum - points[:, 0]
        return self.warp_flipped(points)
