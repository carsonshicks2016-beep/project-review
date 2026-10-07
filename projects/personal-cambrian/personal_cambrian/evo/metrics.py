"""QD progress metrics + snapshots (ROADMAP Stage 5.4).

Tracks the standard quality-diversity signals per generation — coverage,
QD-score, max fitness, and archive novelty (mean k-NN spread in descriptor space)
— plus periodic 2-D archive snapshots. Metrics serialize to JSONL so plots can be
regenerated from a finished run. Plotting uses matplotlib if available; the data
side is dependency-free.
"""
from __future__ import annotations

import json

import numpy as np

from .descriptors import normalize


def archive_novelty(archive, k: int = 5) -> float:
    """Mean distance to the k nearest neighbors in normalized descriptor space,
    averaged over elites — a diversity/novelty measure that rises as the archive
    spreads. 0 with fewer than two elites."""
    elites = archive.elites()
    if len(elites) < 2:
        return 0.0
    X = np.array([[normalize(ax, e.descriptors[ax]) for ax in archive.axes]
                  for e in elites])
    d = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    kk = min(k, len(elites) - 1)
    return float(np.sort(d, axis=1)[:, :kk].mean())


def fitness_grid(archive):
    """2-D fitness grid (NaN where empty); None unless the archive has 2 axes."""
    if len(archive.axes) != 2:
        return None
    bx, by = archive.bins
    g = np.full((bx, by), np.nan)
    for (i, j), e in archive.grid.items():
        g[i, j] = e.fitness
    return g


def grid_to_json(g) -> list:
    """2-D grid -> nested lists with None for empty cells (JSON-safe)."""
    return [[None if not np.isfinite(v) else float(v) for v in row] for row in g]


# --- persistence (JSONL, one record per generation) ------------------------
def save_metrics(logs, path: str) -> None:
    with open(path, "w") as f:
        for rec in logs:
            f.write(json.dumps(rec) + "\n")


def load_metrics(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


# --- plotting (matplotlib-guarded) -----------------------------------------
def has_matplotlib() -> bool:
    try:
        import matplotlib  # noqa: F401
        return True
    except Exception:       # noqa: BLE001
        return False


def plot_metrics(logs, out: str) -> str:
    """Regenerate the QD progress curves (coverage, QD-score, max fitness,
    novelty) from logs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    its = [r["iter"] for r in logs]
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    for ax, key in zip(axs.flat, ("coverage", "qd_score", "max_fitness", "novelty")):
        ax.plot(its, [r.get(key) for r in logs])
        ax.set_title(key)
        ax.set_xlabel("iteration")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out


def plot_archive(archive, out: str) -> str:
    """Heatmap of the 2-D fitness archive."""
    g = fitness_grid(archive)
    if g is None:
        raise ValueError("plot_archive requires a 2-D archive")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(g.T, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xlabel(archive.axes[0])
    ax.set_ylabel(archive.axes[1])
    ax.set_title(f"MAP-Elites archive  (coverage {archive.coverage:.0%})")
    fig.colorbar(im, ax=ax, label="fitness")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
