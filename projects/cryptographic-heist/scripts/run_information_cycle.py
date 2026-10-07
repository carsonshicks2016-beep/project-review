from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.cycle import run_information_cycle
from crypt_heist.league import parse_seed_list


def main():
    parser = argparse.ArgumentParser(description="Train, evaluate, and optionally promote the information stack.")
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--checkpoint-dir", default="checkpoints/cycles")
    parser.add_argument("--log-dir", default="logs/cycles")
    parser.add_argument("--scanner-steps", type=int, default=2400)
    parser.add_argument("--scanner-epochs", type=int, default=8)
    parser.add_argument("--scanner-batch-size", type=int, default=256)
    parser.add_argument("--scanner-horizon-steps", type=int, default=72)
    parser.add_argument("--scanner-window", type=int, default=24)
    parser.add_argument("--scanner-sample-every", type=int, default=4)
    parser.add_argument("--scanner-embed", type=int, default=32)
    parser.add_argument("--radio-steps", type=int, default=3600)
    parser.add_argument("--radio-epochs", type=int, default=12)
    parser.add_argument("--radio-batch-size", type=int, default=128)
    parser.add_argument("--jammer-steps", type=int, default=3600)
    parser.add_argument("--jammer-epochs", type=int, default=12)
    parser.add_argument("--jammer-batch-size", type=int, default=128)
    parser.add_argument("--jammer-mode", choices=["heuristic", "counterfactual"], default="heuristic")
    parser.add_argument("--jammer-sample-every", type=int, default=12)
    parser.add_argument("--jammer-horizon-steps", type=int, default=72)
    parser.add_argument("--jammer-positive-threshold", type=float, default=0.16)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--league-seeds", default="11,12,13")
    parser.add_argument("--league-steps", type=int, default=1200)
    parser.add_argument("--promotion-threshold", type=float, default=0.52)
    parser.add_argument("--improvement-margin", type=float, default=0.01)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--no-compare-active", action="store_true")
    parser.add_argument("--active-dir", default="models/active")
    args = parser.parse_args()

    result = run_information_cycle(
        name=args.name,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        scanner_steps=args.scanner_steps,
        scanner_epochs=args.scanner_epochs,
        scanner_batch_size=args.scanner_batch_size,
        scanner_horizon_steps=args.scanner_horizon_steps,
        scanner_window=args.scanner_window,
        scanner_sample_every=args.scanner_sample_every,
        scanner_embed=args.scanner_embed,
        radio_steps=args.radio_steps,
        radio_epochs=args.radio_epochs,
        radio_batch_size=args.radio_batch_size,
        jammer_steps=args.jammer_steps,
        jammer_epochs=args.jammer_epochs,
        jammer_batch_size=args.jammer_batch_size,
        jammer_mode=args.jammer_mode,
        jammer_sample_every=args.jammer_sample_every,
        jammer_horizon_steps=args.jammer_horizon_steps,
        jammer_positive_threshold=args.jammer_positive_threshold,
        hidden=args.hidden,
        lr=args.lr,
        device=args.device,
        league_seeds=parse_seed_list(args.league_seeds),
        league_steps=args.league_steps,
        promotion_threshold=args.promotion_threshold,
        improvement_margin=args.improvement_margin,
        promote=args.promote,
        compare_active=not args.no_compare_active,
        active_dir=args.active_dir,
    )
    scores = result.league.scores
    decision = result.promotion_decision or {}
    print(
        f"cycle name={result.name} manifest={result.manifest} "
        f"overall={scores['overall_score']:.3f} "
        f"pursuer={scores['pursuer_security_score']:.3f} "
        f"evader={scores['evader_pressure_score']:.3f} "
        f"promoted={str(result.league.promoted).lower()} "
        f"reason={decision.get('reason', 'unknown')}"
    )
    if result.league.active_manifest:
        print(f"active_manifest={result.league.active_manifest}")


if __name__ == "__main__":
    main()
