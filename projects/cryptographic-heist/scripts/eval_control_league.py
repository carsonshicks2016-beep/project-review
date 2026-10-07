from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.control_league import evaluate_control_candidate_league
from crypt_heist.league import parse_seed_list


def main():
    parser = argparse.ArgumentParser(description="Score a control candidate against historical pool opponents.")
    parser.add_argument("--name", default="candidate_control")
    parser.add_argument("--evader", default="models/control_active/evader_ppo.pt")
    parser.add_argument("--pursuer-team", default="models/control_active/pursuer_team_ppo.pt")
    parser.add_argument("--pool-dir", default="models/control_pool")
    parser.add_argument("--seeds", default="11,12")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--max-opponents", type=int, default=3)
    parser.add_argument("--record-prefix", default="replays/control_league")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--out", default="logs/control_league_manifest.json")
    args = parser.parse_args()

    result = evaluate_control_candidate_league(
        name=args.name,
        evader=args.evader,
        pursuer_team=args.pursuer_team,
        pool_dir=args.pool_dir,
        seeds=parse_seed_list(args.seeds),
        steps=args.steps,
        max_opponents=args.max_opponents,
        record_prefix=None if args.no_record else args.record_prefix,
        out=args.out,
    )
    scores = result.scores
    aggregates = result.aggregates
    print(
        f"control_league name={result.name} seeds={','.join(str(seed) for seed in result.seeds)} "
        f"opponents={len(result.pool_opponents)} overall={scores['overall_score']:.3f} "
        f"self={scores['self_pair_score']:.3f} "
        f"evader_generalization={scores['evader_generalization_score']:.3f} "
        f"team_resilience={scores['team_resilience_score']:.3f} "
        f"worst_case={scores['worst_case_score']:.3f} "
        f"cross_matches={int(aggregates['cross_matches'])} manifest={result.manifest}"
    )


if __name__ == "__main__":
    main()
