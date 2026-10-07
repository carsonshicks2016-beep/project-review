from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .action_space import ActionSpace, ActionSpaceError
from .bridge import BridgeClient
from .calibration import calibrate_trace
from .config import PROJECT_ROOT, default_config_path
from .director import DirectorFSM
from .evaluator import build_champion_artifact, evaluate_trace
from .ga import GAConfig, MacroGA
from .memory import MemoryMap
from .policy import make_policy
from .protocol import BridgeObservation
from .rewards import RewardModel
from .route import RouteManifest
from .telemetry import JsonlTelemetryWriter, read_jsonl
from .trainer import TrainingConfig, train_ppo


def _load_core(args: argparse.Namespace) -> tuple[ActionSpace, MemoryMap, RouteManifest]:
    action_space = ActionSpace.from_file(args.action_space)
    memory_map = MemoryMap.from_file(args.memory_map)
    route = RouteManifest.from_file(args.route)
    return action_space, memory_map, route


def cmd_inspect_config(args: argparse.Namespace) -> int:
    action_space, memory_map, route = _load_core(args)
    print(
        json.dumps(
            {
                "project_root": str(PROJECT_ROOT),
                "action_space": {
                    "version": action_space.version,
                    "macros": len(action_space.macros),
                },
                "memory_map": {
                    "version": memory_map.version,
                    "domain": memory_map.domain,
                    "signals": len(memory_map.signals),
                    "required_signals": memory_map.required_signal_names(),
                },
                "route": {
                    "version": route.version,
                    "name": route.name,
                    "steps": len(route.steps),
                    "forbid_starworld": route.forbid_starworld,
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    action_space, memory_map, route = _load_core(args)
    if not action_space.bridge_action("idle").buttons == ():
        raise ActionSpaceError("idle macro must have no buttons")
    synthetic = BridgeObservation(
        frame=1,
        mode="level",
        ram={"game_mode": 20, "lives": 5, "screen_x": 0, "mario_x": 10, "powerup": 1},
    )
    director = DirectorFSM(action_space, memory_map, route)
    decision = director.decide(synthetic)
    print(
        json.dumps(
            {
                "ok": decision.delegate_to_player,
                "decision": decision.reason,
                "route": route.name,
                "macros": action_space.names(),
            },
            indent=2,
        )
    )
    return 0 if decision.delegate_to_player else 1


def cmd_serve(args: argparse.Namespace) -> int:
    action_space, memory_map, route = _load_core(args)
    director = DirectorFSM(action_space, memory_map, route, final_evaluation=args.final_evaluation)
    reward_model = RewardModel(route)
    player_policy = make_policy(args.policy, action_space, model_path=args.model_path)
    writer = JsonlTelemetryWriter(args.telemetry) if args.telemetry else None
    previous: BridgeObservation | None = None

    def handle(observation: BridgeObservation):
        nonlocal previous
        mode = observation.mode
        if mode == "unknown":
            mode = memory_map.classify_mode(observation.ram)
            observation = replace(observation, mode=mode)

        decision = director.decide(observation)
        if decision.delegate_to_player:
            macro = player_policy.select_macro(observation)
            try:
                action = action_space.bridge_action(macro, note="player policy")
            except ActionSpaceError:
                action = action_space.noop(note=f"unknown policy macro {macro}")
        else:
            action = decision.action

        reward = reward_model.score(previous, observation)
        if writer:
            writer.write_frame(
                observation,
                action,
                decision=decision.reason,
                reward=reward.total,
            )
        previous = observation
        return action

    bridge = BridgeClient(args.host, args.port, timeout_s=args.timeout)
    print(f"Listening for BizHawk Lua bridge on {args.host}:{args.port}")
    print("Press Ctrl-C to stop.")
    try:
        bridge.serve_forever(handle)
    except KeyboardInterrupt:
        print("\nStopping bridge.")
    finally:
        if writer:
            writer.close()
    return 0


def cmd_calibrate_log(args: argparse.Namespace) -> int:
    memory_map = MemoryMap.from_file(args.memory_map)
    report = calibrate_trace(args.trace, memory_map)
    print(
        json.dumps(
            {
                "ok": report.ok,
                "missing_required": report.missing_required,
                "signals": {
                    name: {
                        "seen": sig.seen,
                        "changed": sig.changed,
                        "min": sig.minimum,
                        "max": sig.maximum,
                        "samples": sig.samples,
                    }
                    for name, sig in report.signals.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report.ok else 1


def cmd_validate_trace(args: argparse.Namespace) -> int:
    route = RouteManifest.from_file(args.route)
    result = route.validate_trace(read_jsonl(args.trace))
    print(
        json.dumps(
            {
                "ok": result.ok,
                "required_steps_seen": sorted(result.required_steps_seen),
                "issues": [issue.__dict__ for issue in result.issues],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.ok else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    route = RouteManifest.from_file(args.route)
    result = evaluate_trace(args.trace, route)
    print(json.dumps(result.__dict__, indent=2, sort_keys=True))
    return 0 if result.accepted else 1


def cmd_evolve_ga(args: argparse.Namespace) -> int:
    action_space = ActionSpace.from_file(args.action_space)
    ga = MacroGA(
        action_space,
        GAConfig(
            population_size=args.population,
            genome_length=args.length,
            generations=args.generations,
            seed=args.seed,
        ),
    )
    if not args.dry_run:
        print(
            "GA needs an emulator-backed fitness function before real evolution. "
            "Run with --dry-run to test the scaffold.",
            file=sys.stderr,
        )
        return 2
    best = ga.evolve(lambda macros: float(macros.count("run_right") + macros.count("run_jump_right")))
    print(json.dumps({"fitness": best.fitness, "macros": list(best.macros)}, indent=2))
    return 0


def cmd_train_ppo(args: argparse.Namespace) -> int:
    train_ppo(
        TrainingConfig(
            total_timesteps=args.timesteps,
            seed=args.seed,
            output_dir=Path(args.output_dir),
        )
    )
    return 0


def cmd_build_champion(args: argparse.Namespace) -> int:
    target = build_champion_artifact(
        args.output_dir,
        policy_file=args.policy_file,
        route_file=args.route,
        memory_map_file=args.memory_map,
        action_space_file=args.action_space,
        rom_hash=args.rom_hash,
        emulator_version=args.emulator_version,
        evaluation_log=args.evaluation_log,
        lua_bridge_file=args.lua_bridge,
    )
    print(json.dumps({"champion_dir": str(target)}, indent=2))
    return 0


def add_core_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action-space", default=str(default_config_path("action_space.yaml")))
    parser.add_argument("--memory-map", default=str(default_config_path("memory_map.yaml")))
    parser.add_argument("--route", default=str(default_config_path("route_no_starworld_safety.yaml")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smw-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    inspect_config = sub.add_parser("inspect-config")
    add_core_config_args(inspect_config)
    inspect_config.set_defaults(func=cmd_inspect_config)

    self_test = sub.add_parser("self-test")
    add_core_config_args(self_test)
    self_test.set_defaults(func=cmd_self_test)

    serve = sub.add_parser("serve")
    add_core_config_args(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=55355)
    serve.add_argument("--timeout", type=float, default=5.0)
    serve.add_argument("--policy", choices=["noop", "heuristic", "ppo"], default="heuristic")
    serve.add_argument("--model-path")
    serve.add_argument("--telemetry")
    serve.add_argument("--final-evaluation", action="store_true")
    serve.set_defaults(func=cmd_serve)

    calibrate = sub.add_parser("calibrate-log")
    calibrate.add_argument("trace")
    calibrate.add_argument("--memory-map", default=str(default_config_path("memory_map.yaml")))
    calibrate.set_defaults(func=cmd_calibrate_log)

    validate = sub.add_parser("validate-trace")
    validate.add_argument("trace")
    validate.add_argument("--route", default=str(default_config_path("route_no_starworld_safety.yaml")))
    validate.set_defaults(func=cmd_validate_trace)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("trace")
    evaluate.add_argument("--route", default=str(default_config_path("route_no_starworld_safety.yaml")))
    evaluate.set_defaults(func=cmd_evaluate)

    evolve = sub.add_parser("evolve-ga")
    evolve.add_argument("--action-space", default=str(default_config_path("action_space.yaml")))
    evolve.add_argument("--population", type=int, default=24)
    evolve.add_argument("--length", type=int, default=60)
    evolve.add_argument("--generations", type=int, default=10)
    evolve.add_argument("--seed", type=int, default=0)
    evolve.add_argument("--dry-run", action="store_true")
    evolve.set_defaults(func=cmd_evolve_ga)

    ppo = sub.add_parser("train-ppo")
    ppo.add_argument("--timesteps", type=int, default=100_000)
    ppo.add_argument("--seed", type=int, default=0)
    ppo.add_argument("--output-dir", default="artifacts/training")
    ppo.set_defaults(func=cmd_train_ppo)

    champion = sub.add_parser("build-champion")
    champion.add_argument("--output-dir", required=True)
    champion.add_argument("--policy-file", required=True)
    champion.add_argument("--route", default=str(default_config_path("route_no_starworld_safety.yaml")))
    champion.add_argument("--memory-map", default=str(default_config_path("memory_map.yaml")))
    champion.add_argument("--action-space", default=str(default_config_path("action_space.yaml")))
    champion.add_argument("--rom-hash", default="")
    champion.add_argument("--emulator-version", default="")
    champion.add_argument("--evaluation-log")
    champion.add_argument("--lua-bridge", default=str(PROJECT_ROOT / "lua" / "smw_bridge.lua"))
    champion.set_defaults(func=cmd_build_champion)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

