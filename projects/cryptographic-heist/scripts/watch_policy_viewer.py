from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.app import run
from crypt_heist.policy_runtime import EvaderCheckpointController


def main():
    parser = argparse.ArgumentParser(description="Open the live viewer with a learned evader checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/evader_ppo.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--control-repeat", type=int, default=4)
    args = parser.parse_args()
    controller = EvaderCheckpointController(args.checkpoint)
    run(
        seed=args.seed,
        mute=args.mute,
        max_frames=args.max_frames,
        controller=controller,
        control_repeat=args.control_repeat,
        title_suffix=Path(args.checkpoint).name,
    )


if __name__ == "__main__":
    main()
