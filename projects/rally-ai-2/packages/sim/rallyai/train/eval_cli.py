"""Evaluate a checkpoint on held-out seeds and (optionally) export a WATCH replay.

Thin CLI over :func:`rallyai.train.evaluate.evaluate` so E1's periodic held-out
measurements are reproducible and land as JSON beside the checkpoint. The replay
path writes a self-contained record straight into the viewer's ``public/replays``
so WATCH can play a trained-policy run with ``?r=<name>``.

Determinism: the eval harness uses mean actions and a frozen normaliser (see
``rallyai.train.evaluate``). This module adds nothing that samples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rallyai.env import EnvConfig, RallyEnv
from rallyai.stage.generator import generate
from rallyai.train.evaluate import (
    evaluate,
    held_out_seeds,
    load_actor_from_checkpoint,
)


def _repo_root() -> Path:
    # train/eval_cli.py -> train -> rallyai -> sim -> packages -> repo
    return Path(__file__).resolve().parents[4]


def default_replay_dir() -> Path:
    return _repo_root() / "packages" / "viewer" / "public" / "replays"


def export_replay(
    checkpoint: str | Path,
    *,
    seed: int,
    tier: int,
    out_path: str | Path,
    max_time_s: float = 180.0,
) -> dict[str, Any]:
    """Run one deterministic episode and write a WATCH replay JSON.

    Returns a small summary (termination, time, finished, frame count, path).
    """
    actor, _meta = load_actor_from_checkpoint(checkpoint)
    stage = generate(int(seed), int(tier))
    env = RallyEnv(stage, EnvConfig(record_frames=True, max_time_s=max_time_s))
    obs, _ = env.reset(seed=int(seed))
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = actor.action(env, obs)
        obs, _r, terminated, truncated, info = env.step(action)

    doc = env.export_replay(source="agent")
    doc.setdefault("meta", {})
    doc["meta"].update(
        {
            "seed": int(seed),
            "tier": int(tier),
            "agent": "ppo",
            "checkpoint": str(checkpoint),
        }
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    return {
        "path": str(out),
        "seed": int(seed),
        "tier": int(tier),
        "termination": info.get("termination"),
        "finished": bool(info.get("finished")),
        "time_s": float(info.get("time_s", env.time_s)),
        "frames": len(doc.get("frames", [])),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rallyai-eval",
        description="Held-out evaluation + WATCH replay export for a checkpoint",
    )
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument(
        "--tiers",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Tiers to evaluate (default: 0 1).",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=20,
        help="Number of held-out seeds (7, 17, 27, ...).",
    )
    p.add_argument("--max-time-s", type=float, default=180.0)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Where eval JSON lands (default: beside the checkpoint).",
    )
    p.add_argument(
        "--replay-tier",
        type=int,
        default=None,
        help="If set, export one replay at this tier for WATCH.",
    )
    p.add_argument("--replay-seed", type=int, default=7)
    p.add_argument(
        "--replay-out",
        type=Path,
        default=None,
        help="Replay path (default: viewer/public/replays/<stem>_t<tier>_s<seed>.json).",
    )
    return p


def _fmt(x: float | None, nd: int = 3) -> str:
    return "null" if x is None else f"{x:.{nd}f}"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ckpt = args.checkpoint
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    seeds = held_out_seeds(int(args.n_seeds))

    print(f"eval checkpoint={ckpt}")
    print(f"held-out seeds (n={len(seeds)}): {seeds[:6]}{'...' if len(seeds) > 6 else ''}")

    for tier in args.tiers:
        out_path = None
        if args.out_dir is not None:
            out_path = Path(args.out_dir) / f"{ckpt.stem}.tier{tier}.eval.json"
        else:
            out_path = ckpt.with_name(f"{ckpt.name}.tier{tier}.eval.json")
        m = evaluate(
            checkpoint=ckpt,
            seeds=seeds,
            tier=int(tier),
            out_path=out_path,
            env_config=EnvConfig(max_time_s=float(args.max_time_s)),
        )
        terms = m["terminations"]
        term_str = " ".join(f"{k}={v}" for k, v in terms.items() if v)
        print(
            f"[tier {tier}] completion={_fmt(m['completion_rate'])} "
            f"clean={_fmt(m['clean_rate'])} "
            f"mean_time_s={_fmt(m['mean_time_s'], 2)} "
            f"time_vs_optimal={_fmt(m['time_vs_optimal'])} "
            f"| {term_str} -> {out_path}"
        )

    if args.replay_tier is not None:
        stem = ckpt.stem
        replay_out = args.replay_out
        if replay_out is None:
            replay_out = default_replay_dir() / (
                f"{stem}_t{args.replay_tier}_s{args.replay_seed}.json"
            )
        summary = export_replay(
            ckpt,
            seed=int(args.replay_seed),
            tier=int(args.replay_tier),
            out_path=replay_out,
            max_time_s=float(args.max_time_s),
        )
        print(
            f"[replay] seed={summary['seed']} tier={summary['tier']} "
            f"termination={summary['termination']} finished={summary['finished']} "
            f"time_s={_fmt(summary['time_s'], 2)} frames={summary['frames']} "
            f"-> {summary['path']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
