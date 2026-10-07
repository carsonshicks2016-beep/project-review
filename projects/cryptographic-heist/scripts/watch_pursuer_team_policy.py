from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.policy_runtime import PursuerTeamCheckpointController
from crypt_heist.replay import ReplayRecorder
from crypt_heist.sim import HeistSim


def main():
    parser = argparse.ArgumentParser(description="Run a learned five-pursuer team checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/pursuer_team_ppo.pt")
    parser.add_argument("--agents", default=None, help="Optional comma-separated pursuer override.")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record", default=None)
    args = parser.parse_args()
    agent_names = [part.strip() for part in args.agents.split(",") if part.strip()] if args.agents else None
    sim = HeistSim(seed=args.seed, reset_on_capture=True)
    controller = PursuerTeamCheckpointController(args.checkpoint, agent_names=agent_names)
    if args.record:
        with ReplayRecorder(args.record, seed=args.seed, dt=sim.sim.dt) as recorder:
            for _ in range(args.steps):
                sim.step(actions=controller.action(sim))
                recorder.record(sim.snapshot())
    else:
        for _ in range(args.steps):
            sim.step(actions=controller.action(sim))
    print(
        f"checkpoint={args.checkpoint} agents={','.join(controller.agent_names)} steps={args.steps} "
        f"captures={sim.captures} waypoints={sim.waypoints_hit} deception={sim.deception_score:.1f}"
    )


if __name__ == "__main__":
    main()
