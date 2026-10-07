from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.jamming import EvaderJammerController
from crypt_heist.replay import ReplayRecorder
from crypt_heist.sim import HeistSim


def main():
    parser = argparse.ArgumentParser(description="Run learned evader jamming with scripted driving.")
    parser.add_argument("--checkpoint", default="checkpoints/jammer_policy.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record", default=None)
    args = parser.parse_args()
    sim = HeistSim(seed=args.seed, reset_on_capture=True)
    sim.jam_cooldown = 0.0
    controller = EvaderJammerController(args.checkpoint)
    if args.record:
        with ReplayRecorder(args.record, seed=args.seed, dt=sim.sim.dt) as recorder:
            for _ in range(args.steps):
                sim.step(actions=controller.action(sim))
                recorder.record(sim.snapshot())
    else:
        for _ in range(args.steps):
            sim.step(actions=controller.action(sim))
    print(
        f"jammer={args.checkpoint} steps={args.steps} spoofed={len([e for e in sim.channel.events if e.spoofed])} "
        f"budget={sim.channel.jamming_budget} deception={sim.deception_score:.1f} confidence={sim.channel.confidence:.3f}"
    )


if __name__ == "__main__":
    main()
