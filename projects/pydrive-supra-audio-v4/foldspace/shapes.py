"""Target silhouettes for Foldspace.

Each generator returns a dense, ordered set of 2D points tracing a shape's
outline at an arbitrary scale.  The environment resamples these to a fixed point
count and normalises perimeter + centroid, so absolute scale here doesn't matter.
"""
from __future__ import annotations

import numpy as np

TAU = 2 * np.pi


def _circle(n=400):
    t = np.linspace(0, TAU, n, endpoint=False)
    return np.stack([np.cos(t), np.sin(t)], 1)


def _poly(verts, n=400):
    """Resample a closed polygon (list of corners) to n points by arc length."""
    v = np.asarray(verts, float)
    v = np.vstack([v, v[:1]])
    seg = np.linalg.norm(np.diff(v, axis=0), axis=1)
    cum = np.concatenate([[0], np.cumsum(seg)])
    s = np.linspace(0, cum[-1], n, endpoint=False)
    idx = np.clip(np.searchsorted(cum, s, "right") - 1, 0, len(seg) - 1)
    frac = (s - cum[idx]) / np.maximum(seg[idx], 1e-9)
    return v[idx] + frac[:, None] * (v[idx + 1] - v[idx])


def _square():
    return _poly([(-1, -1), (1, -1), (1, 1), (-1, 1)])


def _triangle():
    return _poly([(0, 1.15), (-1, -0.8), (1, -0.8)])


def _star(points=5, n=400):
    ang = np.linspace(0, TAU, points * 2, endpoint=False) + np.pi / 2
    rad = np.where(np.arange(points * 2) % 2 == 0, 1.0, 0.42)
    verts = np.stack([np.cos(ang) * rad, np.sin(ang) * rad], 1)
    return _poly(verts, n)


def _heart(n=400):
    t = np.linspace(0, TAU, n, endpoint=False)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return np.stack([x, y], 1)


def _spiral(turns=2.6, n=400):
    t = np.linspace(0, turns * TAU, n)
    r = np.linspace(0.12, 1.0, n)
    return np.stack([np.cos(t) * r, np.sin(t) * r], 1)


def _cross(t=0.38):
    return _poly([
        (-t, -1), (t, -1), (t, -t), (1, -t), (1, t), (t, t),
        (t, 1), (-t, 1), (-t, t), (-1, t), (-1, -t), (-t, -t),
    ])


def _crescent(n=400):
    t = np.linspace(0.55, TAU - 0.55, n // 2)
    outer = np.stack([np.cos(t), np.sin(t)], 1)
    inner = np.stack([0.45 + np.cos(t[::-1]) * 0.78, np.sin(t[::-1]) * 0.78], 1)
    return np.vstack([outer, inner])


def _moon_arc(n=400):  # an open wave, exercises non-closed outlines
    t = np.linspace(-1, 1, n)
    return np.stack([t * 1.3, np.sin(t * np.pi) * 0.7], 1)


GENERATORS = {
    "circle": _circle,
    "square": _square,
    "triangle": _triangle,
    "star": _star,
    "heart": _heart,
    "spiral": _spiral,
    "cross": _cross,
    "crescent": _crescent,
}
SHAPE_NAMES = list(GENERATORS.keys())


def outline(name) -> np.ndarray:
    return np.asarray(GENERATORS[name](), np.float64)
