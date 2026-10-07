from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.jamming import train_counterfactual_jammer_policy


def main():
    parser = argparse.ArgumentParser(description="Train an evader jammer from paired jam/no-jam rollouts.")
    parser.add_argument("--out", default="checkpoints/jammer_counterfactual.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--sample-every", type=int, default=12)
    parser.add_argument("--horizon-steps", type=int, default=72)
    parser.add_argument("--positive-threshold", type=float, default=0.16)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = train_counterfactual_jammer_policy(
        args.out,
        seed=args.seed,
        steps=args.steps,
        sample_every=args.sample_every,
        horizon_steps=args.horizon_steps,
        positive_threshold=args.positive_threshold,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden=args.hidden,
        lr=args.lr,
        device=args.device,
    )
    data = result.to_dict()
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"counterfactual jammer checkpoint={result.checkpoint} samples={result.samples} "
        f"epochs={result.epochs} loss={result.final_loss:.6f} "
        f"trigger_accuracy={result.trigger_accuracy:.3f} positive_rate={result.positive_rate:.3f} "
        f"mean_counterfactual_reward={result.mean_counterfactual_reward:.3f}"
    )


if __name__ == "__main__":
    main()
