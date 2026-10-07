from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.imitation import train_evader_imitation


def main():
    parser = argparse.ArgumentParser(description="Train the first evader imitation checkpoint.")
    parser.add_argument("--out", default="checkpoints/evader_imitation.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dagger-rounds", type=int, default=0)
    parser.add_argument("--dagger-steps", type=int, default=1200)
    parser.add_argument("--dagger-policy-fraction", type=float, default=1.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    result = train_evader_imitation(
        args.out,
        seed=args.seed,
        steps=args.steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        dagger_rounds=args.dagger_rounds,
        dagger_steps=args.dagger_steps,
        dagger_policy_fraction=args.dagger_policy_fraction,
    )
    print(
        f"checkpoint={result.checkpoint} samples={result.samples} "
        f"epochs={result.epochs} final_loss={result.final_loss:.6f}"
    )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result.__dict__, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
