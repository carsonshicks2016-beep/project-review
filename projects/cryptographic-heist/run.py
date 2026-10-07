from __future__ import annotations

import argparse

from crypt_heist.app import run


def main():
    parser = argparse.ArgumentParser(description="Run The Cryptographic Heist Engine prototype.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mute", action="store_true", help="Disable procedural soundtrack.")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    run(seed=args.seed, mute=args.mute, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
