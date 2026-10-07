from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.control_league import evaluate_control_pair
from crypt_heist.league import parse_seed_list


def main():
    parser = argparse.ArgumentParser(description="Score a learned evader against a learned pursuer team.")
    parser.add_argument("--name", default="active_control")
    parser.add_argument("--evader", default="models/control_active/evader_ppo.pt")
    parser.add_argument("--pursuer-team", default="models/control_active/pursuer_team_ppo.pt")
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--steps", type=int, default=900)
    parser.add_argument("--record-prefix", default="replays/control_pair")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--out", default="logs/control_pair_manifest.json")
    args = parser.parse_args()

    result = evaluate_control_pair(
        name=args.name,
        evader=args.evader,
        pursuer_team=args.pursuer_team,
        seeds=parse_seed_list(args.seeds),
        steps=args.steps,
        record_prefix=None if args.no_record else args.record_prefix,
        out=args.out,
    )
    scores = result.scores
    aggregates = result.aggregates
    print(
        f"control_pair name={result.checkpoint_pair.name} "
        f"seeds={','.join(str(seed) for seed in result.seeds)} "
        f"overall={scores['overall_score']:.3f} "
        f"pursuer={scores['pursuer_control_score']:.3f} "
        f"evader={scores['evader_control_score']:.3f} "
        f"spectacle={scores['spectacle_score']:.3f} "
        f"balance={scores['adversarial_balance_score']:.3f} "
        f"captures={aggregates['mean_captures']:.2f} "
        f"waypoints={aggregates['mean_waypoints']:.2f} "
        f"manifest={result.manifest}"
    )


if __name__ == "__main__":
    main()
