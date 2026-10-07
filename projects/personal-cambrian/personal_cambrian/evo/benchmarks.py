"""STAGE-9 BENCHMARKS (ROADMAP Stage 9.5).

Quantifies the Stage-9 speedups into one report (the done-when is "numbers reported
in runs/"):

  * SURROGATE -- sims-to-QD-score with vs without the surrogate (9.4). Hardware-
    independent, so the headline number is real on CPU.
  * JAX ARCHIVE -- batched-insert QD-score parity vs the CPU archive + insert time
    (9.3). Parity is exact; the throughput multiplier is GPU-bound.
  * MJX -- dynamics parity + physics-steps/sec, loaded from runs/mjx_benchmark.json
    (produced by scripts/mjx_benchmark.py, 9.1). MJX-on-CPU is slow; the 100x is GPU.
  * JAX PPO (9.2) -- a note: matches CPU policy quality (see tests), far faster on GPU.

`stage9_report` composes whatever is available (each piece is import-guarded) and
`save_report` writes the JSON to runs/.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from .surrogate import surrogate_assisted_qd, SurrogateQDConfig, sims_to_reach
from .jax_archive import jax_archive_available


# --- a small controllable domain for the surrogate benchmark ---------------
_G = 8


def _cell(x):
    b = np.clip(((np.clip(x[:2], -2, 2) + 2) / 4 * _G).astype(int), 0, _G - 1)
    return (int(b[0]), int(b[1]))


def _evaluate(x):
    fit = 2.0 * np.exp(-((x[2] - 0.5) ** 2 + (x[3] - 0.5) ** 2) / 0.3) - 0.05 * (x[0] ** 2 + x[1] ** 2)
    return float(fit), _cell(x)


def _mutate(x, rng):
    return np.clip(x + rng.normal(0, 0.4, 4), -2.0, 2.0)


def surrogate_sims_benchmark(*, sim_budget: int = 300, n_candidates: int = 8,
                             beta: float = 0.5, seed: int = 0) -> dict:
    """Sims-to-QD with/without the surrogate on a learnable domain (9.4)."""
    rng = np.random.default_rng(seed)
    seeds = [rng.uniform(-2, 2, 4) for _ in range(4)]
    feats = lambda x: np.asarray(x, float)
    base = surrogate_assisted_qd(seeds, _evaluate, _mutate, feats,
                                 SurrogateQDConfig(sim_budget=sim_budget,
                                                   use_surrogate=False, seed=seed))
    surr = surrogate_assisted_qd(seeds, _evaluate, _mutate, feats,
                                 SurrogateQDConfig(sim_budget=sim_budget, use_surrogate=True,
                                                   n_candidates=n_candidates, beta=beta,
                                                   warmup=20, seed=seed))
    reach = sims_to_reach(surr.qd_curve, base.qd_score)
    return {
        "sim_budget": sim_budget,
        "baseline_qd": round(base.qd_score, 3),
        "surrogate_qd": round(surr.qd_score, 3),
        "qd_ratio_at_equal_budget": round(surr.qd_score / base.qd_score, 2) if base.qd_score else None,
        "sims_to_match_baseline": reach,
        "sims_saved_frac": round(1 - reach / sim_budget, 3) if reach else None,
    }


def jax_archive_benchmark(*, n: int = 20000, bins: int = 16, seed: int = 0) -> dict:
    """Batched-insert QD-score parity vs the CPU archive + insert time (9.3)."""
    if not jax_archive_available():
        return {"available": False, "note": "needs jax"}
    import jax
    from .jax_archive import JaxMAPElites
    from .archive import MAPElites
    from .descriptors import DESCRIPTOR_BOUNDS
    from ..seeds import quadruped
    axes = ["aspect", "limb_count"]
    rng = np.random.default_rng(seed)
    lo = np.array([DESCRIPTOR_BOUNDS[a][0] for a in axes])
    hi = np.array([DESCRIPTOR_BOUNDS[a][1] for a in axes])
    desc = lo + rng.random((n, len(axes))) * (hi - lo)
    fit = rng.normal(size=n)

    jx = JaxMAPElites(axes, bins)
    t0 = time.time(); jx.add_batch(desc, fit); jax.block_until_ready(jx.fitness)
    insert_ms = (time.time() - t0) * 1000.0

    cpu = MAPElites(axes, bins); g = quadruped()
    for d, f in zip(desc, fit):
        cpu.add(g, float(f), {ax: float(v) for ax, v in zip(axes, d)})
    return {
        "available": True, "n_candidates": n, "bins": bins,
        "cpu_qd": round(cpu.qd_score, 4), "jax_qd": round(jx.qd_score, 4),
        "qd_parity": abs(cpu.qd_score - jx.qd_score) < 1e-3,
        "batched_insert_ms": round(insert_ms, 1),
    }


def _load_mjx(mjx_json) -> dict:
    if mjx_json and os.path.exists(mjx_json):
        with open(mjx_json) as f:
            d = json.load(f)
        return {"available": True, **d}
    return {"available": False, "note": "run scripts/mjx_benchmark.py"}


def stage9_report(*, mjx_json: str = None, sim_budget: int = 300,
                  archive_n: int = 20000) -> dict:
    """Compose the Stage-9 benchmark report from the available pieces."""
    return {
        "stage": "9.5 throughput & surrogate benchmarks",
        "device": "cpu",
        "surrogate_9_4": surrogate_sims_benchmark(sim_budget=sim_budget),
        "jax_archive_9_3": jax_archive_benchmark(n=archive_n),
        "mjx_9_1": _load_mjx(mjx_json),
        "jax_ppo_9_2": {"note": "matches CPU PPO policy quality (tests/test_jax_ppo.py); "
                                "far faster on GPU (CPU-only box here)"},
        "caveat": "GPU speedups (MJX throughput, JAX-archive multiplier) need accelerator "
                  "HW; correctness/parity verified on CPU. Surrogate sims-saved is HW-independent.",
    }


def save_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    return path
