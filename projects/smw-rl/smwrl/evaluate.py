"""Measure how good a level policy really is, from a fresh start every time.

This is the number that matters for the route. Training reports a clear rate
that mixes in curriculum-seeded episodes, which start next to the goal and
clear trivially -- YoshiIsland2 once showed 25% that way while its unaided rate
was 0%.

    python -m smwrl.evaluate --level YoshiIsland1 --episodes 50
    python -m smwrl.evaluate --all

A route of N levels completes at roughly (per-level clear rate)^N, so 24 levels
at 90% each finishes only 8% of the time. Reliability is the goal, not a
personal best.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from smwrl.env import make_env
from smwrl.levels import LEVELS, ROUTE
from smwrl.policy import load_policy_runtime
from smwrl.wrappers import EpisodeConfig, RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints"


def evaluate(level: str, episodes: int = 50, deterministic: bool = False,
             max_steps: int = 1200, verbose: bool = False,
             checkpoint: str | Path | None = None, seed: int = 0) -> dict:
    if episodes <= 0:
        return {"level": level, "episodes": 0, "error": "episodes must be positive"}
    if checkpoint is None:
        best = CKPT / level / "best.zip"
        path = best if best.exists() else CKPT / level / "latest.zip"
    else:
        path = Path(checkpoint)
    if not path.exists():
        return {"level": level, "episodes": 0, "error": "no policy"}
    digest_before = hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        runtime = load_policy_runtime(path, expected_level=level)
    except (ValueError, OSError, EOFError, RuntimeError) as e:
        return {"level": level, "episodes": 0,
                "error": f"cannot load policy: {type(e).__name__}: {e}"}
    if runtime.model_sha256 != digest_before:
        return {"level": level, "episodes": 0,
                "error": "checkpoint changed while it was being loaded"}
    model = runtime.model
    env = make_env(level, RewardConfig(), EpisodeConfig(max_steps=max_steps),
                   monitor=False, obs_cfg=runtime.obs_config)

    outcomes: Counter[str] = Counter()
    progress, returns, clear_steps = [], [], []
    try:
        for ep in range(episodes):
            episode_seed = seed + ep
            # SMW itself is deterministic; these seed the reset no-op offset and
            # stochastic policy draw so candidate/incumbent comparisons use the
            # same reproducible suite.
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            stack = runtime.initial_stack(obs)
            total = 0.0
            while True:
                batch = runtime.batch(stack)
                a, _ = model.predict(batch, deterministic=deterministic)
                obs, r, term, trunc, info = env.step(int(np.asarray(a).flat[0]))
                stack = runtime.advance(stack, obs)
                total += r
                if term or trunc:
                    why = ("cleared" if info.get("level_cleared")
                           else "death" if info.get("death")
                           else "stuck" if info.get("stuck") else "timeout")
                    outcomes[why] += 1
                    progress.append(info.get("progress", 0))
                    returns.append(total)
                    if why == "cleared":
                        clear_steps.append(info["steps"])
                    if verbose:
                        print(f"  ep {ep + 1:>3}: {why:<8} progress {info.get('progress', 0):>5}"
                              f"  return {total:>8.1f}")
                    break
    finally:
        env.close()

    n = max(1, sum(outcomes.values()))
    digest_after = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest_after != digest_before:
        return {"level": level, "episodes": 0,
                "error": "checkpoint changed during evaluation"}
    return {
        "level": level,
        "episodes": sum(outcomes.values()),
        "clear_rate": outcomes["cleared"] / n,
        "mean_progress": float(np.mean(progress)) if progress else 0.0,
        "max_progress": int(max(progress)) if progress else 0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "median_clear_steps": int(np.median(clear_steps)) if clear_steps else None,
        "outcomes": dict(outcomes),
        "checkpoint": str(path),
        "model_sha256": digest_before,
        "seed": seed,
        "deterministic": deterministic,
        "max_steps": max_steps,
    }


def route_completion(rates: list[float]) -> float:
    p = 1.0
    for r in rates:
        p *= r
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=sorted(LEVELS))
    ap.add_argument("--all", action="store_true", help="every level with a policy")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--checkpoint", type=Path,
                    help="evaluate this exact checkpoint instead of best/latest")
    ap.add_argument("--json", metavar="OUT.json")
    args = ap.parse_args()

    if args.checkpoint is not None and args.all:
        raise SystemExit("--checkpoint may only be used with one --level")

    if args.all:
        # Include missing policies as explicit failures.  Filtering them out and
        # calling the surviving product "full-route" produced a dangerously
        # optimistic success report.
        levels = list(ROUTE)
    elif args.level:
        levels = [args.level]
    else:
        levels = ["YoshiIsland1"]
    if not levels:
        print("no trained policies found")
        return

    print(f"{'level':<20} {'clear':>7} {'mean prog':>10} {'max prog':>9} "
          f"{'clear steps':>12}  outcomes")
    results = []
    for lv in levels:
        r = evaluate(lv, args.episodes, args.deterministic, verbose=args.verbose,
                     checkpoint=args.checkpoint, seed=args.seed)
        results.append(r)
        if r.get("error"):
            print(f"{lv:<20} {'-':>7}  {r['error']}")
            continue
        steps = r["median_clear_steps"]
        print(f"{lv:<20} {r['clear_rate']:>6.1%} {r['mean_progress']:>10.0f} "
              f"{r['max_progress']:>9} {str(steps) if steps else '-':>12}  {r['outcomes']}")

    good = [r for r in results if not r.get("error")]
    failed = [r for r in results if r.get("error")]
    if args.all:
        rates = [0.0 if r.get("error") else r["clear_rate"] for r in results]
        print(f"\nisolated-state one-shot product: {route_completion(rates):.6%} "
              f"across {len(results)} listed specialists")
        print("This is not a continuous-game completion estimate; map transitions, "
              "required castles/bosses, shared lives, and credits are outside this list.")
        if failed:
            print(f"INCOMPLETE: {len(failed)} required listed policies are missing or invalid")

    if args.json:
        Path(args.json).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
