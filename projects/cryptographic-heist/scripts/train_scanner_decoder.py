from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.scanner import train_scanner_decoder


def main():
    parser = argparse.ArgumentParser(description="Train the evader scanner decoder from radio tokens.")
    parser.add_argument("--out", default="checkpoints/scanner_decoder.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=2400)
    parser.add_argument("--horizon-steps", type=int, default=72)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--sample-every", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--embed", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = train_scanner_decoder(
        args.out,
        seed=args.seed,
        steps=args.steps,
        horizon_steps=args.horizon_steps,
        window=args.window,
        sample_every=args.sample_every,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embed=args.embed,
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
        f"scanner checkpoint={result.checkpoint} samples={result.samples} epochs={result.epochs} "
        f"horizon={result.horizon_steps} window={result.window} loss={result.final_loss:.6f}"
    )


if __name__ == "__main__":
    main()
