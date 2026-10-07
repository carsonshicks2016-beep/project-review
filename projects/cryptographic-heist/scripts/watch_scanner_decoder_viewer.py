from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.app import run
from crypt_heist.scanner import ScannerDecoderRuntime


def main():
    parser = argparse.ArgumentParser(description="Open the live viewer with a learned scanner decoder.")
    parser.add_argument("--checkpoint", default="checkpoints/scanner_decoder.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    scanner = ScannerDecoderRuntime(args.checkpoint)
    run(
        seed=args.seed,
        mute=args.mute,
        max_frames=args.max_frames,
        scanner=scanner,
        title_suffix=f"scanner:{Path(args.checkpoint).name}",
    )


if __name__ == "__main__":
    main()
