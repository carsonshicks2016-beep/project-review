"""OPEN-ENDEDNESS METRICS (ROADMAP Stage 8.4).

A search is *open-ended* if it keeps producing genuinely new, harder-won things
instead of converging and idling. Stages 8.2-8.3 (novelty, POET) are supposed to
sustain that; this module MEASURES whether they do, and -- critically -- detects a
PLATEAU (innovation has stopped) versus SUSTAINED innovation.

Three complementary signals, all reduced to a monotone *accumulation curve* whose
slope is the innovation rate:

  * COVERAGE GROWTH -- new MAP-Elites cells filled per generation (QD logs). A
    finite archive saturates, so this is where a plateau is expected.
  * INNOVATION-EVENT RATE -- new structural innovations introduced per generation
    (the phylogeny's auto-classified +limb/+muscle/... flags).
  * ANNECS -- *accumulated number of novel environments created and solved* (POET).
    An environment counts only if it was SOLVED (some agent cleared it) AND is
    behaviorally NOVEL versus everything already counted, so trivial re-treads of
    the same niche don't inflate it. A rising ANNECS = open-endedness; a flat one =
    the curriculum stopped generating real new challenges.

The generic primitives (`cumulative`, `windowed_rate`, `detect_plateau`) work on
any event series; thin adapters pull the series out of QD logs, a `Phylogeny`, or a
`POET`. `analyze` rolls a curve up into a plateau verdict. Data side is
dependency-free; plotting is matplotlib-guarded.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


# --- generic accumulation / rate / plateau ---------------------------------
def cumulative(events) -> np.ndarray:
    """Running total of a per-step event count -> a monotone accumulation curve."""
    return np.cumsum(np.asarray(events, dtype=float)) if len(events) else np.array([])


def windowed_rate(cum, window: int = 5) -> np.ndarray:
    """Innovation rate = slope of the accumulation curve over a trailing `window`."""
    cum = np.asarray(cum, dtype=float)
    out = np.zeros_like(cum)
    for i in range(len(cum)):
        lo = max(0, i - window)
        span = i - lo
        out[i] = (cum[i] - cum[lo]) / span if span > 0 else 0.0
    return out


def detect_plateau(cum, *, window: int = 5, rel_threshold: float = 0.1) -> Optional[int]:
    """Index where innovation has TERMINALLY plateaued: the earliest step from which
    the windowed rate stays below `rel_threshold` x its peak for the rest of the run.
    Returns None if the curve is still innovating at the end (sustained), and 0 if it
    never innovated at all."""
    cum = np.asarray(cum, dtype=float)
    if cum.size < 2:
        return None
    rate = windowed_rate(cum, window)
    peak = float(rate.max())
    if peak <= 0.0:
        return 0
    below = rate < rel_threshold * peak
    for i in range(cum.size):
        if below[i:].all():
            return i
    return None


# --- ANNECS (accumulated novel-and-solved environments) --------------------
def annecs_curve(outcomes, *, solve_threshold: float, novelty_threshold: float,
                 n_steps: Optional[int] = None) -> np.ndarray:
    """Cumulative count of environments that were SOLVED (best_score >= solve_threshold)
    AND are NOVEL (descriptor farther than `novelty_threshold` from every already-counted
    environment). Novelty is judged in CREATION order so re-treads don't count.

    By default the curve is indexed per environment (one point per outcome). Pass
    `n_steps` to index it over TIME instead -- a length-`n_steps` curve keyed on each
    outcome's 't' (creation iteration) -- which gives the long flat tail a plateau
    detector needs when environment creation itself stalls.

    `outcomes`: list of dicts with 'descriptor', 'best_score' (and 't' if `n_steps`),
    e.g. `POET.env_outcomes()`."""
    counted: list[np.ndarray] = []

    def _counts(o) -> bool:
        d = np.asarray(o["descriptor"], dtype=float)
        if o["best_score"] >= solve_threshold and all(
                float(np.linalg.norm(d - c)) > novelty_threshold for c in counted):
            counted.append(d)
            return True
        return False

    if n_steps is None:
        return np.asarray([(_counts(o), len(counted))[1] for o in outcomes], dtype=float)

    contrib = np.zeros(int(n_steps) + 1, dtype=float)
    for o in sorted(outcomes, key=lambda o: o.get("t", 0)):
        if _counts(o):
            contrib[min(int(o.get("t", 0)), int(n_steps))] += 1.0
    return np.cumsum(contrib)


def annecs(outcomes, *, solve_threshold: float, novelty_threshold: float) -> int:
    """Final ANNECS score for a run."""
    c = annecs_curve(outcomes, solve_threshold=solve_threshold,
                     novelty_threshold=novelty_threshold)
    return int(c[-1]) if c.size else 0


# --- adapters: pull an event series from each source -----------------------
def coverage_growth(logs, key: str = "cells") -> np.ndarray:
    """New cells filled per iteration from QD logs (first-difference of the running
    cell count, clamped at >=0). `cumulative` of this recovers the count curve."""
    counts = np.array([float(r[key]) for r in logs], dtype=float)
    if counts.size == 0:
        return counts
    return np.clip(np.diff(counts, prepend=0.0), 0.0, None)


def innovation_series(phylogeny) -> np.ndarray:
    """Number of structural-innovation flags introduced per node, in generation order
    -- the innovation-event series of a lineage tree."""
    nodes = sorted(phylogeny.nodes.values(), key=lambda n: (n.generation, n.id))
    return np.array([len(n.innovations or []) for n in nodes], dtype=float)


# --- run-level summary ------------------------------------------------------
def analyze(events, *, window: int = 5, rel_threshold: float = 0.1) -> dict:
    """Summarize an event series: totals, current vs peak rate, plateau onset, and a
    'sustained' / 'plateaued' verdict."""
    cum = cumulative(events)
    if cum.size == 0:
        return {"total": 0, "n_steps": 0, "peak_rate": 0.0, "final_rate": 0.0,
                "plateau_at": 0, "sustained": False}
    rate = windowed_rate(cum, window)
    plateau = detect_plateau(cum, window=window, rel_threshold=rel_threshold)
    return {
        "total": float(cum[-1]),
        "n_steps": int(cum.size),
        "peak_rate": float(rate.max()),
        "final_rate": float(rate[-1]),
        "plateau_at": plateau,                       # None = still innovating
        "sustained": plateau is None,
    }


# --- plotting (matplotlib-guarded) -----------------------------------------
def plot_openendedness(curves: dict, out: str, *, plateaus: Optional[dict] = None) -> str:
    """Plot one or more named accumulation curves (e.g. {'coverage': ..., 'ANNECS': ...}),
    marking any detected plateau onset with a vertical line."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plateaus = plateaus or {}
    fig, ax = plt.subplots(figsize=(8, 5))
    for name, series in curves.items():
        cum = cumulative(series) if np.ndim(series) and not _is_monotone(series) else np.asarray(series, float)
        ax.plot(range(len(cum)), cum, label=name)
        p = plateaus.get(name)
        if p is not None:
            ax.axvline(p, ls="--", alpha=0.5)
    ax.set_xlabel("generation / iteration")
    ax.set_ylabel("accumulated innovations")
    ax.set_title("Open-endedness: accumulation curves (-- = plateau onset)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def _is_monotone(series) -> bool:
    s = np.asarray(series, dtype=float)
    return s.size > 0 and bool(np.all(np.diff(s) >= -1e-9))


def has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:        # noqa: BLE001
        return False
