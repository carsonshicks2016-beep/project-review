from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401
from crypt_heist.eval import evaluate_scripted


def main():
    parser = argparse.ArgumentParser(description="Evaluate scripted heist baselines.")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    result = evaluate_scripted(num_seeds=args.seeds, steps=args.steps)
    print(result.to_text())
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result.__dict__, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
