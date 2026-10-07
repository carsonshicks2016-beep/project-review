from __future__ import annotations

import argparse
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.adversarial import AdversarialStackController, configure_jam_ready
from crypt_heist.app import run
from crypt_heist.scanner import ScannerDecoderRuntime


def main():
    parser = argparse.ArgumentParser(description="Open the live viewer with learned radio, scanner, and jammer.")
    parser.add_argument("--radio", default="checkpoints/radio_policy.pt")
    parser.add_argument("--scanner", default="checkpoints/scanner_decoder.pt")
    parser.add_argument("--jammer", default="checkpoints/jammer_policy.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--mute", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    controller = AdversarialStackController(args.radio, args.jammer)
    scanner = ScannerDecoderRuntime(args.scanner)
    suffix = f"adversarial:{Path(args.radio).name}+{Path(args.scanner).name}+{Path(args.jammer).name}"
    run(
        seed=args.seed,
        mute=args.mute,
        max_frames=args.max_frames,
        controller=controller,
        scanner=scanner,
        control_repeat=4,
        title_suffix=suffix,
        configure_sim=configure_jam_ready,
    )


if __name__ == "__main__":
    main()
