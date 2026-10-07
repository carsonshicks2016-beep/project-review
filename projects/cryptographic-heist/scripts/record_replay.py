from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401
from crypt_heist.replay import record_scripted_replay


def main():
    parser = argparse.ArgumentParser(description="Record a scripted heist replay.")
    parser.add_argument("out", help="Output JSONL path.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    args = parser.parse_args()
    summary = record_scripted_replay(args.out, seed=args.seed, steps=args.steps)
    print(f"wrote {summary['frames']} frames to {args.out}")


if __name__ == "__main__":
    main()
