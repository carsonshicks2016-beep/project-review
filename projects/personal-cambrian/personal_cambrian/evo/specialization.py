"""PER-NICHE SPECIALIZATION VALIDATION (ROADMAP Stage 7.6).

The claim evolution is supposed to deliver: a creature specialised for niche X
beats the others *at X*, and pays for it by being worse elsewhere. We check it
with a TRADEOFF MATRIX -- train one specialist per niche, then cross-evaluate
every specialist on every niche:

    M[i, j] = return of the niche-i specialist evaluated on niche-j

A "clear specialist diagonal" means that for each niche j (each COLUMN) the best
specialist is the one trained on j -- i.e. `argmax_i M[i, j] == j`. Because niches
reward different things on very different scales, the diagonal is judged
column-wise (and `column_normalize` rescales columns for plotting / comparison).

Training is RL, so the matrix BUILDER lives here but the real (slow) run is the
`scripts/specialization_matrix.py` demo; the analysis helpers below are pure and
unit-tested on synthetic matrices.
"""
from __future__ import annotations

import numpy as np

from ..sim import CreatureEnv, NICHES
from ..control import PPOConfig, evaluate, scratch_make_agent
from ..control.ppo import train
from .metrics import has_matplotlib


# --- matrix construction (RL) ----------------------------------------------
def _env_fn(genome, niche_cls, ep_steps):
    return lambda s: CreatureEnv(genome, task=niche_cls(max_steps=ep_steps),
                                 obs_mode="structured")


def train_specialist(genome, niche_cls, *, steps: int, hidden: int = 64,
                     n_envs: int = 4, n_steps: int = 256, ep_steps: int = 200,
                     seed: int = 0):
    """Train one policy for `genome` under `niche_cls` and return it."""
    cme = _env_fn(genome, niche_cls, ep_steps)
    policy, _ = train(cme, PPOConfig(total_timesteps=steps, n_envs=n_envs,
                                     n_steps=n_steps, hidden=hidden, seed=seed),
                      make_agent=scratch_make_agent(cme, hidden))
    return policy


def score_on(policy, genome, niche_cls, *, ep_steps: int = 200, eval_seed: int = 0) -> float:
    """The niche's own cumulative reward for `policy` driving `genome` in it."""
    env = _env_fn(genome, niche_cls, ep_steps)(0)
    perf = evaluate(policy, env, max_steps=ep_steps, seed=eval_seed)
    env.close()
    return float(perf["return"])


def tradeoff_matrix(genome, niche_names, *, train_steps: int = 30_000,
                    ep_steps: int = 200, hidden: int = 64, n_envs: int = 4,
                    n_steps: int = 256, seed: int = 0, eval_seed: int = 0,
                    log_fn=None):
    """Train a specialist per niche, then cross-evaluate. Returns (matrix, names).

    M[i, j] = return of the niche-i specialist on niche-j (rows=specialist,
    cols=niche-evaluated-on)."""
    names = list(niche_names)
    policies = {}
    for n in names:
        policies[n] = train_specialist(genome, NICHES[n], steps=train_steps,
                                       hidden=hidden, n_envs=n_envs, n_steps=n_steps,
                                       ep_steps=ep_steps, seed=seed)
        if log_fn:
            log_fn(f"trained specialist: {n}")
    M = np.zeros((len(names), len(names)), dtype=float)
    for i, ni in enumerate(names):
        for j, nj in enumerate(names):
            M[i, j] = score_on(policies[ni], genome, NICHES[nj],
                               ep_steps=ep_steps, eval_seed=eval_seed)
    return M, names


# --- analysis (pure) --------------------------------------------------------
def column_normalize(matrix) -> np.ndarray:
    """Min-max each column to [0, 1] so niches with different reward scales compare.
    A constant column maps to 0."""
    M = np.asarray(matrix, dtype=float)
    lo = M.min(axis=0)
    hi = M.max(axis=0)
    rng = np.where(hi > lo, hi - lo, 1.0)
    return (M - lo) / rng


def specialist_diagonal_fraction(matrix) -> float:
    """Fraction of niches (columns) whose best specialist is the diagonal one."""
    M = np.asarray(matrix, dtype=float)
    if M.size == 0:
        return 0.0
    return float(np.mean([int(np.argmax(M[:, j]) == j) for j in range(M.shape[1])]))


def is_specialist_diagonal(matrix, *, min_fraction: float = 0.5) -> bool:
    """True if the matrix shows a clear specialist diagonal."""
    return specialist_diagonal_fraction(matrix) >= min_fraction


def plot_tradeoff_matrix(matrix, names, out: str) -> str:
    """Heatmap of the column-normalised tradeoff matrix (specialists x niches)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    M = column_normalize(matrix)
    fig, ax = plt.subplots(figsize=(1.4 * len(names) + 2, 1.4 * len(names) + 1))
    im = ax.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("evaluated on niche"); ax.set_ylabel("specialist trained on")
    for j in range(len(names)):                    # ring the column winner
        i = int(np.argmax(np.asarray(matrix)[:, j]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   edgecolor="red", lw=2))
    fig.colorbar(im, ax=ax, label="column-normalised return")
    ax.set_title("Per-niche specialization tradeoff (red = column winner)")
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    return out
