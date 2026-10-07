from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.radio import train_radio_policy


def main():
    parser = argparse.ArgumentParser(description="Train a pursuer radio-token policy from scripted broadcasts.")
    parser.add_argument("--out", default="checkpoints/radio_policy.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = train_radio_policy(
        args.out,
        seed=args.seed,
        steps=args.steps,
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
        f"radio checkpoint={result.checkpoint} samples={result.samples} epochs={result.epochs} "
        f"loss={result.final_loss:.6f} token_accuracy={result.token_accuracy:.3f}"
    )


if __name__ == "__main__":
    main()
