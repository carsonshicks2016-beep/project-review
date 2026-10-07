"""Train CLI: resumable PPO with SIGINT-safe checkpointing."""

from __future__ import annotations

import argparse
from pathlib import Path

from rallyai.train.ppo import PPOConfig, Trainer
from rallyai.train.stages import STAGES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rallyai-train", description="RallyAI PPO trainer")
    p.add_argument("--run-id", required=True, help="Stable id; metrics filename matches")
    p.add_argument("--stage", default="foundation", choices=STAGES)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--timesteps", type=int, default=20_000_000)
    p.add_argument("--tier-start", type=int, default=0)
    p.add_argument("--n-steps", type=int, default=512)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    p.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume")
    p.add_argument(
        "--sync",
        action="store_true",
        help="Use in-process SyncVecEnv (debug / single-process)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=PPOConfig.lr)
    p.add_argument("--ent-coef", type=float, default=PPOConfig.ent_coef)
    p.add_argument("--target-kl", type=float, default=PPOConfig.target_kl)
    p.add_argument(
        "--kl-early-stop",
        action="store_true",
        help="Stop the epoch loop at target_kl instead of discarding the update",
    )
    p.add_argument(
        "--log-std-max",
        type=float,
        default=None,
        help="Per-dim exploration ceiling (default 0.0 = sigma 1.0, which "
        "foundation_01 saturated on 3 of 4 dims)",
    )
    p.add_argument("--supervise", action="store_true", help="Enable automated supervisor (collapse detection, auto-rollback)")
    p.add_argument("--ntfy-topic", default=None, help="ntfy.sh topic for push notifications (disabled if unset)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = PPOConfig(
        n_envs=max(1, args.workers),
        n_steps=args.n_steps,
        epochs=args.epochs,
        minibatch=args.minibatch,
        total_timesteps=args.timesteps,
        seed=args.seed,
        device=args.device,
        lr=args.lr,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        kl_early_stop=args.kl_early_stop,
    )
    trainer = Trainer(
        run_id=args.run_id,
        stage=args.stage,
        workers=args.workers,
        timesteps=args.timesteps,
        tier_start=args.tier_start,
        config=cfg,
        out_dir=args.out_dir,
        resume=args.resume,
        sync=args.sync,
        supervise=args.supervise,
        ntfy_topic=args.ntfy_topic,
        log_std_max=args.log_std_max,
    )
    summary = trainer.train()
    print(
        f"done timesteps={summary.get('timesteps')} "
        f"sps={summary.get('steps_per_s')} "
        f"ckpt={summary.get('checkpoint')}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
