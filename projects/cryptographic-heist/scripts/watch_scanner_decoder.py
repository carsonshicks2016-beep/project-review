from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.replay import ReplayRecorder
from crypt_heist.scanner import ScannerDecoderRuntime
from crypt_heist.sim import HeistSim


def main():
    parser = argparse.ArgumentParser(description="Run a scanner decoder checkpoint and optionally record a replay.")
    parser.add_argument("--checkpoint", default="checkpoints/scanner_decoder.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record", default=None)
    args = parser.parse_args()
    sim = HeistSim(seed=args.seed, reset_on_capture=True)
    scanner = ScannerDecoderRuntime(args.checkpoint)
    if args.record:
        with ReplayRecorder(args.record, seed=args.seed, dt=sim.sim.dt) as recorder:
            for _ in range(args.steps):
                sim.step()
                scanner.apply(sim)
                recorder.record(sim.snapshot())
    else:
        for _ in range(args.steps):
            sim.step()
            scanner.apply(sim)
    print(
        f"scanner={args.checkpoint} steps={args.steps} confidence={sim.channel.confidence:.3f} "
        f"captures={sim.captures} waypoints={sim.waypoints_hit}"
    )


if __name__ == "__main__":
    main()
