from __future__ import annotations

import argparse
import sys

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from . import _bootstrap  # noqa: F401

from crypt_heist.scenarios import SCENARIO_NAMES, ScenarioThresholds, run_acceptance_suite, run_checkpoint_acceptance_suite


def main():
    parser = argparse.ArgumentParser(description="Run replay-backed project acceptance scenarios.")
    parser.add_argument(
        "--scenarios",
        default=",".join(SCENARIO_NAMES),
        help="Comma-separated scenario ids. Choices: " + ", ".join(SCENARIO_NAMES),
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--steps", type=int, default=None, help="Override every scenario's default step count.")
    parser.add_argument("--record-dir", default="replays/acceptance")
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument("--out", default="logs/acceptance_scenarios.json")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when any required check fails.")
    parser.add_argument("--evader-checkpoint", default=None)
    parser.add_argument("--pursuer-team-checkpoint", default=None)

    parser.add_argument("--downtown-min-waypoints", type=int, default=0)
    parser.add_argument("--downtown-target-waypoints", type=int, default=3)
    parser.add_argument("--downtown-min-radio-events", type=int, default=24)
    parser.add_argument("--downtown-min-avg-speed", type=float, default=4.0)
    parser.add_argument("--roadblock-min-capture-events", type=int, default=1)
    parser.add_argument("--roadblock-min-boxed-frames", type=int, default=90)
    parser.add_argument("--roadblock-max-capture-time", type=float, default=2.5)
    parser.add_argument("--spoof-min-spoofed-events", type=int, default=5)
    parser.add_argument("--spoof-min-deception", type=float, default=1.0)
    parser.add_argument("--spoof-min-confidence-drop", type=float, default=0.08)
    parser.add_argument("--spoof-min-active-pursuers", type=int, default=1)
    parser.add_argument("--cipher-min-rotations", type=int, default=1)
    parser.add_argument("--cipher-min-confidence-drop", type=float, default=0.15)
    parser.add_argument("--cipher-min-recovery", type=float, default=0.02)
    args = parser.parse_args()

    thresholds = ScenarioThresholds(
        downtown_min_waypoints=args.downtown_min_waypoints,
        downtown_target_waypoints=args.downtown_target_waypoints,
        downtown_min_radio_events=args.downtown_min_radio_events,
        downtown_min_avg_speed=args.downtown_min_avg_speed,
        roadblock_min_capture_events=args.roadblock_min_capture_events,
        roadblock_min_boxed_frames=args.roadblock_min_boxed_frames,
        roadblock_max_capture_time=args.roadblock_max_capture_time,
        spoof_min_spoofed_events=args.spoof_min_spoofed_events,
        spoof_min_deception=args.spoof_min_deception,
        spoof_min_confidence_drop=args.spoof_min_confidence_drop,
        spoof_min_active_pursuers=args.spoof_min_active_pursuers,
        cipher_min_rotations=args.cipher_min_rotations,
        cipher_min_confidence_drop=args.cipher_min_confidence_drop,
        cipher_min_recovery=args.cipher_min_recovery,
    )
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    runner = run_checkpoint_acceptance_suite if args.evader_checkpoint or args.pursuer_team_checkpoint else run_acceptance_suite
    common = {
        "scenarios": scenarios,
        "seed": args.seed,
        "steps": args.steps,
        "record_dir": None if args.no_record else args.record_dir,
        "out": args.out,
        "thresholds": thresholds,
    }
    if runner is run_checkpoint_acceptance_suite:
        manifest = runner(
            evader_checkpoint=args.evader_checkpoint,
            pursuer_team_checkpoint=args.pursuer_team_checkpoint,
            **common,
        )
    else:
        manifest = runner(**common)

    status = "passed" if manifest["passed"] else "failed"
    print(
        f"acceptance {status} scenarios={len(manifest['scenarios'])} "
        f"required={manifest['required_checks_passed']}/{manifest['required_checks']} "
        f"milestones={manifest['milestone_targets_passed']}/{manifest['milestone_targets']} "
        f"manifest={manifest.get('manifest', args.out)}"
    )
    for scenario in manifest["scenarios"]:
        scenario_status = "pass" if scenario["passed"] else "fail"
        failed = [check["name"] for check in scenario["checks"] if check["required"] and not check["passed"]]
        replay = scenario.get("replay") or "not-recorded"
        print(
            f"  {scenario['name']}: {scenario_status} "
            f"checks={scenario['required_checks_passed']}/{scenario['required_checks']} "
            f"frames={scenario['metrics']['frames']} replay={replay}"
            + (f" failed={','.join(failed)}" if failed else "")
        )
        primary = scenario.get("primary_diagnostic") or {}
        if primary:
            print(
                f"    diagnostic={primary.get('severity', 'info')}:{primary.get('owner', 'unknown')}:"
                f"{primary.get('reason', 'unknown')} - {primary.get('message', '')}"
            )
    top = (manifest.get("diagnostics") or {}).get("top", [])
    if top:
        print("top_diagnostics=" + "; ".join(
            f"{row.get('scenario')}:{row.get('severity')}:{row.get('reason')}"
            for row in top[:4]
        ))

    if args.strict and not manifest["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
