from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.league import evaluate_checkpoint_set, parse_seed_list


def main():
    parser = argparse.ArgumentParser(description="Score and optionally promote a checkpoint set using auth metrics.")
    parser.add_argument("--name", default="candidate")
    parser.add_argument("--radio", default="checkpoints/radio_policy.pt")
    parser.add_argument("--scanner", default="checkpoints/scanner_decoder.pt")
    parser.add_argument("--jammer", default="checkpoints/jammer_policy.pt")
    parser.add_argument("--seeds", default="11,12,13")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record-prefix", default="logs/checkpoint_league")
    parser.add_argument("--out", default="logs/checkpoint_league_manifest.json")
    parser.add_argument("--promotion-threshold", type=float, default=0.52)
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--active-dir", default="models/active")
    args = parser.parse_args()

    result = evaluate_checkpoint_set(
        name=args.name,
        radio=args.radio,
        scanner=args.scanner,
        jammer=args.jammer,
        seeds=parse_seed_list(args.seeds),
        steps=args.steps,
        record_prefix=args.record_prefix,
        out=args.out,
        promotion_threshold=args.promotion_threshold,
        promote=args.promote,
        active_dir=args.active_dir,
    )
    scores = result.scores
    aggregates = result.aggregates
    print(
        f"league name={args.name} seeds={','.join(str(seed) for seed in result.seeds)} "
        f"overall={scores['overall_score']:.3f} "
        f"pursuer={scores['pursuer_security_score']:.3f} "
        f"evader={scores['evader_pressure_score']:.3f} "
        f"balance={scores['adversarial_balance_score']:.3f} "
        f"susceptibility={aggregates['mean_spoof_susceptibility']:.3f} "
        f"promoted={str(result.promoted).lower()} manifest={args.out}"
    )
    if result.active_manifest:
        print(f"active_manifest={result.active_manifest}")


if __name__ == "__main__":
    main()
