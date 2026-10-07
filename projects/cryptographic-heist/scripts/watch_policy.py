from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.policy_runtime import EvaderCheckpointController
from crypt_heist.replay import ReplayRecorder
from crypt_heist.sim import HeistSim


def main():
    parser = argparse.ArgumentParser(description="Run a learned evader checkpoint against scripted pursuers.")
    parser.add_argument("--checkpoint", default="checkpoints/evader_imitation.pt")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record", default=None)
    args = parser.parse_args()
    sim = HeistSim(seed=args.seed, reset_on_capture=True)
    controller = EvaderCheckpointController(args.checkpoint)
    if args.record:
        with ReplayRecorder(args.record, seed=args.seed, dt=sim.sim.dt) as recorder:
            for _ in range(args.steps):
                sim.step(actions=controller.action(sim))
                recorder.record(sim.snapshot())
    else:
        for _ in range(args.steps):
            sim.step(actions=controller.action(sim))
    print(
        f"checkpoint={args.checkpoint} steps={args.steps} captures={sim.captures} "
        f"waypoints={sim.waypoints_hit} deception={sim.deception_score:.1f}"
    )


if __name__ == "__main__":
    main()

