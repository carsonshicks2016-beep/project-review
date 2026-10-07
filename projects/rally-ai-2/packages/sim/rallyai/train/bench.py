"""Throughput benchmark for vectorised environments.

    python -m rallyai.train.bench --workers N --steps M [--tier T]

Prints steps/s, wall time, and a per-phase breakdown. Numbers here are what
the README's Measured throughput row is regenerated from.
"""

from __future__ import annotations

import argparse
import time

import numpy as np


def _random_actions(n: int, act_dim: int, rng: np.random.Generator) -> np.ndarray:
    """Gentle drive inputs — measure physics throughput, not crash/reset churn.

    Full-lock random steer ends episodes in a few steps and spends the bench on
    ``generate`` + ``Track`` rebuilds, which understates the parallel step rate
    the trainer actually sees during competent rollouts.
    """
    actions = np.zeros((n, act_dim), dtype=np.float32)
    actions[:, 0] = rng.uniform(-0.25, 0.25, size=n)
    actions[:, 1] = rng.uniform(0.35, 0.7, size=n)
    actions[:, 2] = 0.0
    actions[:, 3] = 0.0
    return actions


def bench_sync(n_envs: int, steps: int, tier: int) -> dict[str, float]:
    from rallyai.train.vec_env import SyncVecEnv

    env = SyncVecEnv(n_envs, tier=tier)
    t0 = time.perf_counter()
    env.reset(seed=0)
    t_reset = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    t_step = 0.0
    done_steps = 0
    while done_steps < steps:
        actions = _random_actions(env.n, env.act_dim, rng)
        t1 = time.perf_counter()
        env.step(actions)
        t_step += time.perf_counter() - t1
        done_steps += env.n
    env.close()
    wall = t_reset + t_step
    return {
        "workers": 0.0,
        "n_envs": float(n_envs),
        "steps": float(done_steps),
        "wall_s": wall,
        "steps_per_s": done_steps / wall if wall > 0 else 0.0,
        "reset_s": t_reset,
        "env_step_s": t_step,
        "policy_forward_s": 0.0,
        "other_s": 0.0,
    }


def bench_async(n_workers: int, steps: int, tier: int,
                envs_per_worker: int = 1) -> dict[str, float]:
    from rallyai.train.vec_env import AsyncVecEnv

    env = AsyncVecEnv(n_workers, envs_per_worker=envs_per_worker, tier=tier)
    t0 = time.perf_counter()
    env.reset(seed=0)
    t_reset = time.perf_counter() - t0
    rng = np.random.default_rng(0)
    t_step = 0.0
    # Synthetic "policy forward" — zero-cost placeholder until C1's policy exists.
    # Keeping the bucket means the Phase C split can land without reshaping output.
    t_policy = 0.0
    done_steps = 0
    while done_steps < steps:
        t_p0 = time.perf_counter()
        actions = _random_actions(env.n, env.act_dim, rng)
        t_policy += time.perf_counter() - t_p0
        t1 = time.perf_counter()
        env.step(actions)
        t_step += time.perf_counter() - t1
        done_steps += env.n
    env.close()
    wall = t_reset + t_step + t_policy
    other = max(0.0, wall - t_reset - t_step - t_policy)
    return {
        "workers": float(n_workers),
        "n_envs": float(env.n),
        "steps": float(done_steps),
        "wall_s": wall,
        "steps_per_s": done_steps / wall if wall > 0 else 0.0,
        "reset_s": t_reset,
        "env_step_s": t_step,
        "policy_forward_s": t_policy,
        "other_s": other,
    }


def _print_report(result: dict[str, float], *, baseline_steps_per_s: float | None = None) -> None:
    print(f"workers          {int(result['workers'])}")
    print(f"n_envs           {int(result['n_envs'])}")
    print(f"steps            {int(result['steps'])}")
    print(f"wall_s           {result['wall_s']:.3f}")
    print(f"steps_per_s      {result['steps_per_s']:.1f}")
    if baseline_steps_per_s and baseline_steps_per_s > 0 and result["workers"] > 0:
        print(f"vs_sync1         {result['steps_per_s'] / baseline_steps_per_s:.2f}×")
    print("--- phase breakdown ---")
    print(f"reset_s          {result['reset_s']:.3f}")
    print(f"env_step_s       {result['env_step_s']:.3f}")
    print(f"policy_forward_s {result['policy_forward_s']:.3f}")
    print(f"other_s          {result['other_s']:.3f}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=1,
                   help="Async worker processes. 0 = SyncVecEnv with --envs.")
    p.add_argument("--envs", type=int, default=1,
                   help="Envs per worker (async) or total envs (sync when workers=0).")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--tier", type=int, default=0)
    p.add_argument("--compare-sync1", action="store_true",
                   help="Also bench SyncVecEnv(n=1) and print the async speedup.")
    args = p.parse_args(argv)

    baseline = None
    if args.compare_sync1 and args.workers > 0:
        baseline_result = bench_sync(1, min(args.steps, 4000), args.tier)
        baseline = baseline_result["steps_per_s"]
        print("--- sync/1 baseline ---")
        _print_report(baseline_result)
        print("--- async ---")

    if args.workers <= 0:
        result = bench_sync(args.envs, args.steps, args.tier)
    else:
        result = bench_async(args.workers, args.steps, args.tier,
                             envs_per_worker=args.envs)
    _print_report(result, baseline_steps_per_s=baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
