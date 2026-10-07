from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.app import run
from crypt_heist.radio import PursuerRadioController


def main():
    parser = argparse.ArgumentParser(description="Open the live viewer with learned pursuer radio tokens.")
    parser.add_argument("--checkpoint", default="checkpoints/radio_policy.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    controller = PursuerRadioController(args.checkpoint)
    run(
        seed=args.seed,
        mute=args.mute,
        max_frames=args.max_frames,
        controller=controller,
        title_suffix=f"radio:{Path(args.checkpoint).name}",
    )


if __name__ == "__main__":
    main()
