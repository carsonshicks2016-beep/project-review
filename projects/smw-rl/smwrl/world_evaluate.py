"""Measure the whole-game policy honestly, and promote a checkpoint only on evidence.

`world_train` writes `latest.zip` in place every 100k steps and nothing else. So
a policy that gets worse silently overwrites the one that was good: the run that
scored 0.49 levels per unaided episode and chained four levels was destroyed by
the weaker policies that followed it, with no way back. The per-level pipeline
has not had this problem since `autopilot` learned to promote on evidence; this
is the same discipline for the policy that spans the whole game.

    python -m smwrl.world_evaluate --episodes 20            # just measure
    python -m smwrl.world_evaluate --episodes 20 --promote  # measure, maybe promote

Every episode starts from the first playable state, never from a curriculum
state deeper in the archive, because levels chained from an unaided start is the
only number that means anything here. Training's own average mixes in episodes
that begin next to a goal.

Two safeguards are inherited from the per-level version, and both exist because
their absence is silently wrong rather than loudly broken: the candidate is
frozen to its own file before measurement so a mid-evaluation republish cannot
change what was measured, and promotion verifies the digest of what it is about
to copy against the digest of what was actually evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from smwrl.policy import copy_policy_metadata, load_policy_runtime, metadata_path
from smwrl.world_env import WorldEnv, WorldEpisodeConfig, curriculum_from_archive
from smwrl.wrappers import RewardConfig

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT / "checkpoints" / "world"
DEFAULT_ARCHIVE = ROOT / "checkpoints" / "world_archive.pkl"


def mean_lower(values, z: float = 1.96) -> float:
    """95% lower confidence bound on a mean of per-episode counts.

    Levels cleared per episode is a count, not a coin flip, so the Wilson bound
    `autopilot` uses for a clear rate does not apply. A normal approximation on
    the sample mean serves the same purpose: promote on evidence rather than on
    one lucky run. With a handful of episodes the bound is harsh, which is the
    intended behaviour -- a small sample should not be able to promote.
    """
    n = len(values)
    if n == 0:
        return 0.0
    mean = float(np.mean(values))
    if n == 1:
        return 0.0
    sem = float(np.std(values, ddof=1)) / (n ** 0.5)
    return max(0.0, mean - z * sem)


def evaluation_score(evaluation: dict) -> tuple[float, float, float]:
    """Compared lexicographically: evidence first, then raw mean, then distance."""
    return (float(evaluation.get("levels_lower", 0.0)),
            float(evaluation.get("mean_levels", 0.0)),
            float(evaluation.get("mean_progress", 0.0)))


def evaluate(checkpoint: str | Path, episodes: int = 20, deterministic: bool = False,
             archive: str | Path = DEFAULT_ARCHIVE, seed: int = 0,
             max_steps: int | None = None, verbose: bool = False) -> dict:
    if episodes <= 0:
        return {"episodes": 0, "error": "episodes must be positive"}
    path = Path(checkpoint)
    if not path.exists():
        return {"episodes": 0, "error": "no policy"}
    digest_before = hashlib.sha256(path.read_bytes()).hexdigest()

    try:
        # expected_level=None: the whole-game contract claims no single level.
        runtime = load_policy_runtime(path, device="cpu", expected_level=None)
    except (ValueError, OSError, EOFError, RuntimeError) as e:
        return {"episodes": 0, "error": f"cannot load policy: {type(e).__name__}: {e}"}
    if runtime.model_sha256 != digest_before:
        return {"episodes": 0, "error": "checkpoint changed while it was being loaded"}

    curriculum = curriculum_from_archive(archive)
    if not curriculum:
        return {"episodes": 0, "error": f"no usable archive at {archive}"}

    ecfg = WorldEpisodeConfig()
    if max_steps is not None:
        ecfg.max_steps = max_steps
    # curriculum_ratio=0.0: every episode restores index 0, the unaided start.
    env = WorldEnv(curriculum, RewardConfig(), ecfg, runtime.obs_config,
                   curriculum_ratio=0.0)
    model = runtime.model

    levels, progress, returns, events = [], [], [], []
    outcomes: dict[str, int] = {}
    try:
        for ep in range(episodes):
            episode_seed = seed + ep
            # SMW is deterministic; these seed the reset no-op offset and the
            # stochastic policy draw, so candidate and incumbent face the same
            # reproducible suite.
            random.seed(episode_seed)
            np.random.seed(episode_seed)
            torch.manual_seed(episode_seed)
            obs, _ = env.reset(seed=episode_seed)
            stack = runtime.initial_stack(obs)
            total = 0.0
            while True:
                a, _ = model.predict(runtime.batch(stack), deterministic=deterministic)
                obs, r, term, trunc, info = env.step(int(np.asarray(a).flat[0]))
                stack = runtime.advance(stack, obs)
                total += r
                if term or trunc:
                    why = ("transition_failed" if info.get("transition_failed")
                           else "death" if info.get("death")
                           else "stuck" if info.get("stuck") else "timeout")
                    outcomes[why] = outcomes.get(why, 0) + 1
                    levels.append(int(info.get("levels_cleared", 0)))
                    progress.append(int(info.get("progress", 0)))
                    events.append(int(info.get("events", 0)))
                    returns.append(total)
                    if verbose:
                        print(f"  ep {ep + 1:>3}: {why:<18} levels "
                              f"{info.get('levels_cleared', 0)}  progress "
                              f"{info.get('progress', 0):>6}  return {total:>9.1f}")
                    break
    finally:
        env.close()

    digest_after = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest_after != digest_before:
        return {"episodes": 0, "error": "checkpoint changed during evaluation"}
    return {
        "episodes": len(levels),
        "mean_levels": float(np.mean(levels)) if levels else 0.0,
        "levels_lower": mean_lower(levels),
        "best_levels": int(max(levels)) if levels else 0,
        "mean_progress": float(np.mean(progress)) if progress else 0.0,
        "max_events": int(max(events)) if events else 0,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "outcomes": outcomes,
        "checkpoint": str(path),
        "model_sha256": digest_before,
        "seed": seed,
        "deterministic": deterministic,
    }


def snapshot_latest(directory: Path = CKPT) -> Path | None:
    """Freeze the exact artifact the evaluation will measure.

    `world_train` republishes latest.zip every 100k steps, which at ~450 steps/s
    is often enough to land in the middle of a 20-episode evaluation.
    """
    latest = directory / "latest.zip"
    if not latest.exists():
        return None
    candidate = directory / "candidate_eval.zip"
    tmp = directory / "candidate_eval.tmp.zip"
    shutil.copyfile(latest, tmp)
    os.replace(tmp, candidate)
    copy_policy_metadata(latest, candidate)
    return candidate


def cleanup_candidate(candidate: Path | None) -> None:
    if candidate is None:
        return
    candidate.unlink(missing_ok=True)
    metadata_path(candidate).unlink(missing_ok=True)


def read_incumbent(directory: Path = CKPT) -> dict | None:
    path = directory / "best.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def promote_best(source: Path, evaluation: dict, incumbent: dict | None = None,
                 directory: Path = CKPT) -> bool:
    """Copy the evaluated candidate to best.zip if it beat the incumbent."""
    if evaluation.get("error") or int(evaluation.get("episodes", 0)) <= 0:
        return False
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    if evaluation.get("model_sha256") != source_hash:
        raise ValueError("evaluation digest does not match the promotion candidate")
    score = evaluation_score(evaluation)
    if incumbent is not None and not incumbent.get("error"):
        if score <= evaluation_score(incumbent):
            return False

    directory.mkdir(parents=True, exist_ok=True)
    best = directory / "best.zip"
    tmp_model = directory / "best.tmp.zip"
    shutil.copyfile(source, tmp_model)
    os.replace(tmp_model, best)
    copy_policy_metadata(source, best)
    payload = dict(evaluation)
    payload.update({"promoted": time.time(), "source": source.name,
                    "model_sha256": source_hash})
    tmp_json = directory / "best.tmp.json"
    tmp_json.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp_json, directory / "best.json")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path,
                    help="default: a frozen snapshot of checkpoints/world/latest.zip")
    ap.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--deterministic", action="store_true",
                    help="argmax instead of sampling; sampling is the default "
                         "because it is what training actually does")
    ap.add_argument("--promote", action="store_true",
                    help="replace best.zip if this candidate beats the incumbent")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    candidate = None
    if args.checkpoint is None:
        candidate = snapshot_latest()
        if candidate is None:
            raise SystemExit(f"no policy at {CKPT / 'latest.zip'}; "
                             "run smwrl.world_train first")
        target = candidate
    else:
        target = args.checkpoint

    try:
        result = evaluate(target, episodes=args.episodes,
                          deterministic=args.deterministic, archive=args.archive,
                          seed=args.seed, max_steps=args.max_steps,
                          verbose=not args.quiet)
        if result.get("error"):
            raise SystemExit(f"evaluation failed: {result['error']}")

        print(f"\nepisodes            {result['episodes']}")
        print(f"levels/episode      {result['mean_levels']:.2f}"
              f"   (95% lower bound {result['levels_lower']:.2f})")
        print(f"best in one episode {result['best_levels']}")
        print(f"mean progress       {result['mean_progress']:.0f} px")
        print(f"max game progress   {result['max_events']} event(s)")
        print(f"outcomes            {result['outcomes']}")

        if args.promote:
            incumbent = read_incumbent()
            promoted = promote_best(target, result, incumbent)
            if promoted:
                print(f"\npromoted -> {CKPT / 'best.zip'}")
            elif incumbent:
                print(f"\nnot promoted: incumbent scores "
                      f"{incumbent.get('levels_lower', 0):.2f} lower bound "
                      f"/ {incumbent.get('mean_levels', 0):.2f} mean")
            else:
                print("\nnot promoted")
    finally:
        cleanup_candidate(candidate)


if __name__ == "__main__":
    main()
