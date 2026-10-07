from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.policy_runtime import PursuerCheckpointController
from crypt_heist.replay import ReplayRecorder
from crypt_heist.sim import HeistSim


def main():
    parser = argparse.ArgumentParser(description="Run a learned pursuer checkpoint against the scripted chase.")
    parser.add_argument("--checkpoint", default="checkpoints/pursuer_ppo.pt")
    parser.add_argument("--agent", default=None, help="Override the checkpoint's pursuer agent, e.g. pursuer_0.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record", default=None)
    args = parser.parse_args()
    sim = HeistSim(seed=args.seed, reset_on_capture=True)
    controller = PursuerCheckpointController(args.checkpoint, agent_name=args.agent)
    if args.record:
        with ReplayRecorder(args.record, seed=args.seed, dt=sim.sim.dt) as recorder:
            for _ in range(args.steps):
                sim.step(actions=controller.action(sim))
                recorder.record(sim.snapshot())
    else:
        for _ in range(args.steps):
            sim.step(actions=controller.action(sim))
    print(
        f"checkpoint={args.checkpoint} agent={controller.agent_name} steps={args.steps} "
        f"captures={sim.captures} waypoints={sim.waypoints_hit} deception={sim.deception_score:.1f}"
    )


if __name__ == "__main__":
    main()
