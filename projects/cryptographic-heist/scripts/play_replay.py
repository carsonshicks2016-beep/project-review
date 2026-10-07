from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.replay_player import run_replay


def main():
    parser = argparse.ArgumentParser(description="Play a JSONL heist replay.")
    parser.add_argument("replay")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run_replay(args.replay, speed=args.speed, max_frames=args.max_frames)


if __name__ == "__main__":
    main()

