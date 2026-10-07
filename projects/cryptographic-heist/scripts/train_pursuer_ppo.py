from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.ppo import train_pursuer_ppo


def main():
    parser = argparse.ArgumentParser(description="Train one PPO pursuer driver against the scripted evader.")
    parser.add_argument("--out", default="checkpoints/pursuer_ppo.pt")
    parser.add_argument("--agent", default="pursuer_0")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--updates", type=int, default=8)
    parser.add_argument("--steps-per-update", type=int, default=512)
    parser.add_argument("--max-cycles", type=int, default=900)
    parser.add_argument("--control-repeat", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=128)
    parser.add_argument("--decoder-alpha", type=float, default=0.08)
    parser.add_argument("--spoof-gamma", type=float, default=0.50)
    parser.add_argument("--evader-spoof-reward", type=float, default=0.35)
    parser.add_argument("--prediction-horizon-steps", type=int, default=8)
    parser.add_argument("--counterfactual-interval", type=int, default=16)
    parser.add_argument("--counterfactual-horizon-steps", type=int, default=24)
    parser.add_argument("--counterfactual-evader-weight", type=float, default=0.20)
    parser.add_argument("--counterfactual-pursuer-weight", type=float, default=0.20)
    parser.add_argument("--opponent-evader-checkpoint", default=None)
    parser.add_argument("--opponent-pursuer-team-checkpoint", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    result = train_pursuer_ppo(
        args.out,
        agent_name=args.agent,
        seed=args.seed,
        updates=args.updates,
        steps_per_update=args.steps_per_update,
        max_cycles=args.max_cycles,
        control_repeat=args.control_repeat,
        hidden=args.hidden,
        lr=args.lr,
        train_epochs=args.train_epochs,
        minibatch_size=args.minibatch_size,
        decoder_accuracy_alpha=args.decoder_alpha,
        spoof_susceptibility_gamma=args.spoof_gamma,
        evader_spoof_reward=args.evader_spoof_reward,
        prediction_horizon_steps=args.prediction_horizon_steps,
        counterfactual_interval=args.counterfactual_interval,
        counterfactual_horizon_steps=args.counterfactual_horizon_steps,
        counterfactual_evader_weight=args.counterfactual_evader_weight,
        counterfactual_pursuer_weight=args.counterfactual_pursuer_weight,
        opponent_evader_checkpoint=args.opponent_evader_checkpoint,
        opponent_pursuer_team_checkpoint=args.opponent_pursuer_team_checkpoint,
        device=args.device,
    )
    data = result.to_dict()
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"pursuer ppo checkpoint={result.checkpoint} agent={args.agent} "
        f"updates={result.updates} steps={result.steps} episodes={result.episodes} "
        f"mean_return={result.final_mean_episode_return:.3f} loss={result.final_loss:.4f}"
    )
    if result.stats:
        last = result.stats[-1]
        print(
            f"last update: step_reward={last['mean_step_reward']:.4f} "
            f"waypoints={last['waypoints_hit']} captures={last['captures']} "
            f"confidence={last['confidence']:.3f} "
            f"decoder_acc={last['decoder_accuracy']:.3f} "
            f"auth_penalty={last['pursuer_auth_penalty']:.3f} "
            f"counterfactual={last['counterfactual_deception']:.3f}"
        )
        if last.get("opponent_count", 0):
            print(f"opponents={last['opponent_count']}")


if __name__ == "__main__":
    main()
