"""Foldspace — a vectorized environment where a folding strip learns to become a shape.

The strip is a chain of K equal-length segments.  A policy controls a base
heading plus K-1 joint fold angles and nudges them each step; the strip creases
itself toward a target silhouette.  Reward is the per-step reduction in symmetric
Chamfer distance between the folded strip and the target outline — a telescoping
signal whose episode return equals (initial_chamfer - final_chamfer), so
maximising it maximises final shape fidelity.
"""
from __future__ import annotations

import numpy as np

from . import shapes

# --- constants ---
K = 28                       # segments
N_JOINT = K - 1
SEGLEN = 1.0
STRIP_LEN = K * SEGLEN
M = 96                       # chamfer sample points per outline
T_MAX = 80                   # folding steps per episode
ANG_RATE = 0.22              # max angle change per joint per step (rad)
TARGET_PERIM = STRIP_LEN     # scale targets so the strip can trace them exactly

SHAPE_NAMES = shapes.SHAPE_NAMES
N_SHAPE = len(SHAPE_NAMES)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _resample_open(P, n):
    """Arc-length resample an open polyline (P,2) to n points incl. both ends."""
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n)
    idx = np.clip(np.searchsorted(cum, s, "right") - 1, 0, len(seg) - 1)
    frac = (s - cum[idx]) / np.maximum(seg[idx], 1e-9)
    return P[idx] + frac[:, None] * (P[idx + 1] - P[idx])


def _build_targets():
    """Return (N_SHAPE, M, 2) centred target point clouds + (N_SHAPE, K) oracle angles."""
    tpts = np.zeros((N_SHAPE, M, 2), np.float32)
    oracle = np.zeros((N_SHAPE, K), np.float32)
    for i, name in enumerate(SHAPE_NAMES):
        dense = shapes.outline(name)
        length = np.linalg.norm(np.diff(dense, axis=0), axis=1).sum()
        dense = dense * (TARGET_PERIM / length)
        pts = _resample_open(dense, M)
        pts = pts - pts.mean(0)
        tpts[i] = pts
        # oracle: reconstruct the shape's discrete curvature
        verts = _resample_open(dense, K + 1)
        d = np.diff(verts, axis=0)
        h = np.arctan2(d[:, 1], d[:, 0])
        oracle[i, 0] = h[0]
        oracle[i, 1:] = _wrap(np.diff(h))
    return tpts, oracle


class VecFold:
    def __init__(self, n_envs: int, seed: int = 0):
        self.n = n_envs
        self.rng = np.random.default_rng(seed)
        self.tpts, self.oracle = _build_targets()

        # precomputed uniform resample indices (strip seg lengths are equal)
        s = np.linspace(0, STRIP_LEN, M)
        self.sidx = np.clip(np.floor(s / SEGLEN).astype(int), 0, K - 1)
        self.sfrac = (s - self.sidx * SEGLEN) / SEGLEN

        self.c = np.zeros((self.n, K), np.float32)
        self.tid = np.zeros(self.n, np.int64)
        self.t = np.zeros(self.n, np.int64)
        self.prev_cf = np.zeros(self.n, np.float32)
        self.cf0 = np.ones(self.n, np.float32)

        self.act_dim = K
        self.obs_dim = self._obs().shape[1]
        self._reset_idx(np.arange(self.n))

    # ---------------------------------------------------------------- geometry
    def _vertices(self, c):
        cs = np.cumsum(c[:, 1:], axis=1)
        h = np.concatenate([c[:, :1], c[:, :1] + cs], axis=1)          # (n,K)
        dx = SEGLEN * np.cos(h)
        dy = SEGLEN * np.sin(h)
        vx = np.concatenate([np.zeros((self.n, 1)), np.cumsum(dx, 1)], 1)
        vy = np.concatenate([np.zeros((self.n, 1)), np.cumsum(dy, 1)], 1)
        return np.stack([vx, vy], 2).astype(np.float32)               # (n,K+1,2)

    def _strip_points(self, verts):
        a = verts[:, self.sidx]                                       # (n,M,2)
        b = verts[:, self.sidx + 1]
        return a + self.sfrac[None, :, None] * (b - a)

    def _chamfer(self, strip):
        s = strip - strip.mean(1, keepdims=True)
        t = self.tpts[self.tid]                                       # already centred
        d = np.linalg.norm(s[:, :, None, :] - t[:, None, :, :], axis=3)  # (n,M,M)
        return (d.min(2).mean(1) + d.min(1).mean(1)).astype(np.float32)

    # ---------------------------------------------------------------- api
    def _reset_idx(self, idx):
        m = len(idx)
        if m == 0:
            return
        self.tid[idx] = self.rng.integers(0, N_SHAPE, m)
        self.c[idx] = 0.0
        self.t[idx] = 0
        # baseline chamfer of the (now straight) strip vs its new target
        full = self._chamfer(self._strip_points(self._vertices(self.c)))
        self.prev_cf[idx] = full[idx]
        self.cf0[idx] = np.maximum(full[idx], 1e-3)

    def reset(self):
        self._reset_idx(np.arange(self.n))
        return self._obs()

    def step(self, action, auto_reset=True):
        # The policy outputs a *target* fold configuration; the strip eases toward
        # it at a bounded rate so the whole thing physically folds over the episode
        # while each episode's return cleanly reflects one configuration's quality.
        target = np.clip(np.asarray(action, np.float32).reshape(self.n, K), -np.pi, np.pi)
        delta = _wrap(target - self.c)
        self.c = _wrap(self.c + np.clip(delta, -ANG_RATE, ANG_RATE))
        strip = self._strip_points(self._vertices(self.c))
        cf = self._chamfer(strip)
        rew = (self.prev_cf - cf)                                     # telescoping improvement
        self.prev_cf = cf
        self.t += 1
        done = self.t >= T_MAX
        info = {"chamfer": cf.copy(), "cf0": self.cf0.copy(),
                "score": 1.0 - cf / self.cf0}
        if auto_reset:
            di = np.where(done)[0]
            if len(di):
                self._reset_idx(di)
        return self._obs(), rew.astype(np.float32), done, info

    def set_targets(self, tids):
        """Force specific target ids (for recording); reset strips to straight."""
        self.tid[:] = np.asarray(tids, np.int64)
        self.c[:] = 0.0
        self.t[:] = 0
        full = self._chamfer(self._strip_points(self._vertices(self.c)))
        self.prev_cf[:] = full
        self.cf0[:] = np.maximum(full, 1e-3)
        return self._obs()

    def _obs(self):
        return np.concatenate([
            np.cos(self.c), np.sin(self.c),
            np.eye(N_SHAPE, dtype=np.float32)[self.tid],
            (self.prev_cf / STRIP_LEN)[:, None],
            (self.t / T_MAX)[:, None].astype(np.float32),
        ], 1).astype(np.float32)

    # ---------------------------------------------------------------- render
    def snapshot(self):
        verts = self._vertices(self.c)
        return {"verts": verts.copy(), "tid": self.tid.copy(),
                "chamfer": self.prev_cf.copy(), "score": (1.0 - self.prev_cf / self.cf0).copy()}

    @staticmethod
    def metadata():
        return {"K": K, "seglen": SEGLEN, "M": M, "t_max": T_MAX,
                "shapes": SHAPE_NAMES,
                "targets": _build_targets()[0].tolist()}
