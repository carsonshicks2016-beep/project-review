from __future__ import annotations

import argparse

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.league import parse_seed_list
from crypt_heist.selfplay import run_self_play_cycle


def main():
    parser = argparse.ArgumentParser(description="Run one alternating self-play cycle.")
    parser.add_argument("--name", default="selfplay")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--checkpoint-dir", default="checkpoints/selfplay")
    parser.add_argument("--log-dir", default="logs/selfplay")
    parser.add_argument("--active-control-dir", default="models/control_active")
    parser.add_argument("--control-pool-dir", default="models/control_pool")
    parser.add_argument("--active-info-dir", default="models/active")
    parser.add_argument("--evader-updates", type=int, default=4)
    parser.add_argument("--team-updates", type=int, default=4)
    parser.add_argument("--steps-per-update", type=int, default=384)
    parser.add_argument("--evader-init-checkpoint", default=None)
    parser.add_argument("--evader-validation-steps", type=int, default=0)
    parser.add_argument("--evader-validation-seed", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=900)
    parser.add_argument("--control-repeat", type=int, default=4)
    parser.add_argument("--train-epochs", type=int, default=4)
    parser.add_argument("--evader-minibatch-size", type=int, default=128)
    parser.add_argument("--team-minibatch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--counterfactual-interval", type=int, default=16)
    parser.add_argument("--counterfactual-horizon-steps", type=int, default=24)
    parser.add_argument("--counterfactual-evader-weight", type=float, default=0.20)
    parser.add_argument("--counterfactual-pursuer-weight", type=float, default=0.20)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-control-pool", action="store_true")
    parser.add_argument("--skip-control-eval", action="store_true")
    parser.add_argument("--control-eval-seeds", default=None)
    parser.add_argument("--control-eval-steps", type=int, default=600)
    parser.add_argument("--control-record-prefix", default="replays/selfplay")
    parser.add_argument("--no-control-replay", action="store_true")
    parser.add_argument("--control-pool-eval-opponents", type=int, default=3)
    parser.add_argument("--evaluate-control-scenarios", action="store_true")
    parser.add_argument("--control-scenario-seed", type=int, default=31)
    parser.add_argument("--control-scenario-steps", type=int, default=None)
    parser.add_argument("--control-scenario-record-dir", default="replays/control_acceptance")
    parser.add_argument("--control-scenario-threshold", type=float, default=None)
    parser.add_argument("--evader-control-scenario-threshold", type=float, default=None)
    parser.add_argument("--pursuer-team-control-scenario-threshold", type=float, default=None)
    parser.add_argument("--no-promote-control", action="store_true")
    parser.add_argument("--control-promotion-threshold", type=float, default=0.0)
    parser.add_argument("--control-improvement-margin", type=float, default=0.0)
    parser.add_argument("--evader-control-promotion-threshold", type=float, default=None)
    parser.add_argument("--evader-control-improvement-margin", type=float, default=None)
    parser.add_argument("--pursuer-team-control-promotion-threshold", type=float, default=None)
    parser.add_argument("--pursuer-team-control-improvement-margin", type=float, default=None)
    parser.add_argument("--skip-information", action="store_true")
    parser.add_argument("--scanner-steps", type=int, default=1200)
    parser.add_argument("--scanner-epochs", type=int, default=4)
    parser.add_argument("--scanner-batch-size", type=int, default=128)
    parser.add_argument("--scanner-horizon-steps", type=int, default=24)
    parser.add_argument("--scanner-window", type=int, default=16)
    parser.add_argument("--scanner-sample-every", type=int, default=4)
    parser.add_argument("--scanner-embed", type=int, default=24)
    parser.add_argument("--radio-steps", type=int, default=1800)
    parser.add_argument("--radio-epochs", type=int, default=6)
    parser.add_argument("--radio-batch-size", type=int, default=96)
    parser.add_argument("--jammer-steps", type=int, default=1800)
    parser.add_argument("--jammer-epochs", type=int, default=6)
    parser.add_argument("--jammer-batch-size", type=int, default=96)
    parser.add_argument("--jammer-mode", choices=["heuristic", "counterfactual"], default="heuristic")
    parser.add_argument("--jammer-sample-every", type=int, default=12)
    parser.add_argument("--jammer-horizon-steps", type=int, default=48)
    parser.add_argument("--jammer-positive-threshold", type=float, default=0.16)
    parser.add_argument("--league-seeds", default="11")
    parser.add_argument("--league-steps", type=int, default=600)
    parser.add_argument("--promotion-threshold", type=float, default=0.52)
    parser.add_argument("--improvement-margin", type=float, default=0.01)
    parser.add_argument("--no-promote-information", action="store_true")
    args = parser.parse_args()

    result = run_self_play_cycle(
        name=args.name,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        active_control_dir=args.active_control_dir,
        control_pool_dir=args.control_pool_dir,
        active_info_dir=args.active_info_dir,
        evader_updates=args.evader_updates,
        team_updates=args.team_updates,
        steps_per_update=args.steps_per_update,
        evader_init_checkpoint=args.evader_init_checkpoint,
        evader_validation_steps=args.evader_validation_steps,
        evader_validation_seed=args.evader_validation_seed,
        max_cycles=args.max_cycles,
        control_repeat=args.control_repeat,
        train_epochs=args.train_epochs,
        evader_minibatch_size=args.evader_minibatch_size,
        team_minibatch_size=args.team_minibatch_size,
        hidden=args.hidden,
        lr=args.lr,
        counterfactual_interval=args.counterfactual_interval,
        counterfactual_horizon_steps=args.counterfactual_horizon_steps,
        counterfactual_evader_weight=args.counterfactual_evader_weight,
        counterfactual_pursuer_weight=args.counterfactual_pursuer_weight,
        device=args.device,
        use_control_pool=not args.no_control_pool,
        evaluate_control=not args.skip_control_eval,
        control_eval_seeds=parse_seed_list(args.control_eval_seeds) if args.control_eval_seeds else None,
        control_eval_steps=args.control_eval_steps,
        control_record_prefix=None if args.no_control_replay else args.control_record_prefix,
        control_pool_eval_opponents=args.control_pool_eval_opponents,
        evaluate_control_scenarios=args.evaluate_control_scenarios,
        control_scenario_seed=args.control_scenario_seed,
        control_scenario_steps=args.control_scenario_steps,
        control_scenario_record_dir=args.control_scenario_record_dir,
        control_scenario_threshold=args.control_scenario_threshold,
        evader_control_scenario_threshold=args.evader_control_scenario_threshold,
        pursuer_team_control_scenario_threshold=args.pursuer_team_control_scenario_threshold,
        promote_control=not args.no_promote_control,
        control_promotion_threshold=args.control_promotion_threshold,
        control_improvement_margin=args.control_improvement_margin,
        evader_control_promotion_threshold=args.evader_control_promotion_threshold,
        evader_control_improvement_margin=args.evader_control_improvement_margin,
        pursuer_team_control_promotion_threshold=args.pursuer_team_control_promotion_threshold,
        pursuer_team_control_improvement_margin=args.pursuer_team_control_improvement_margin,
        train_information=not args.skip_information,
        scanner_steps=args.scanner_steps,
        scanner_epochs=args.scanner_epochs,
        scanner_batch_size=args.scanner_batch_size,
        scanner_horizon_steps=args.scanner_horizon_steps,
        scanner_window=args.scanner_window,
        scanner_sample_every=args.scanner_sample_every,
        scanner_embed=args.scanner_embed,
        radio_steps=args.radio_steps,
        radio_epochs=args.radio_epochs,
        radio_batch_size=args.radio_batch_size,
        jammer_steps=args.jammer_steps,
        jammer_epochs=args.jammer_epochs,
        jammer_batch_size=args.jammer_batch_size,
        jammer_mode=args.jammer_mode,
        jammer_sample_every=args.jammer_sample_every,
        jammer_horizon_steps=args.jammer_horizon_steps,
        jammer_positive_threshold=args.jammer_positive_threshold,
        league_seeds=parse_seed_list(args.league_seeds),
        league_steps=args.league_steps,
        promotion_threshold=args.promotion_threshold,
        improvement_margin=args.improvement_margin,
        promote_information=not args.no_promote_information,
    )
    evader_return = result.training["evader"]["final_mean_episode_return"]
    team_return = result.training["pursuer_team"]["final_mean_episode_return"]
    opponents = result.control_opponents
    evader_opp = opponents["evader_training"]["source"]
    team_opp = opponents["pursuer_team_training"]["source"]
    info = result.information_cycle
    control = result.control_evaluation
    control_text = "skipped"
    if control is not None:
        scores = control["scores"]
        control_text = (
            f"overall={scores['overall_score']:.3f} "
            f"spectacle={scores['spectacle_score']:.3f}"
        )
    decision = result.control_promotion_decision or {}
    promoted = str(bool(decision.get("promoted", False))).lower()
    evader_promoted = str(bool(decision.get("evader_promoted", False))).lower()
    team_promoted = str(bool(decision.get("pursuer_team_promoted", False))).lower()
    reason = decision.get("reason", "unknown")
    info_text = "skipped"
    if info is not None:
        scores = info["league"]["scores"]
        decision = info.get("promotion_decision") or {}
        info_text = f"overall={scores['overall_score']:.3f} reason={decision.get('reason', 'unknown')}"
    print(
        f"selfplay name={result.name} manifest={result.manifest} "
        f"evader_return={evader_return:.3f} team_return={team_return:.3f} "
        f"active_control={result.active_control.get('manifest')} "
        f"opponents=evader:{evader_opp},team:{team_opp} "
        f"control_eval={control_text} promoted={promoted} "
        f"evader_promoted={evader_promoted} team_promoted={team_promoted} reason={reason} "
        f"information={info_text}"
    )


if __name__ == "__main__":
    main()
