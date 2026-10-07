"""Whole-project operational readiness evidence for the Cryptographic Heist Engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_COMMAND_IDS = (
    "dashboard",
    "viewer",
    "record_replay",
    "play_replay",
    "summarize_replay",
    "replay_report",
    "verify_replay_fidelity",
    "verify_physics",
    "verify_env",
    "verify_spectator",
    "verify_operational_readiness",
    "build_evidence_bundle",
    "verify_evidence_bundle",
    "eval_acceptance_scenarios",
    "eval_acceptance_full_targets",
    "eval_evader_checkpoint_acceptance",
    "eval_pursuer_team_checkpoint_acceptance",
    "eval_active_control_acceptance",
    "train_evader_ppo",
    "train_pursuer_team_ppo",
    "train_scanner_decoder",
    "train_radio_policy",
    "train_jammer_policy",
    "run_information_cycle",
    "run_self_play_cycle_scenario_gated",
    "watch_adversarial_stack",
)


EXPECTED_CHECKPOINTS = (
    "checkpoints/evader_ppo.pt",
    "checkpoints/pursuer_team_ppo.pt",
    "checkpoints/scanner_decoder.pt",
    "checkpoints/radio_policy.pt",
    "checkpoints/jammer_policy.pt",
    "models/control_active/evader_ppo.pt",
    "models/control_active/pursuer_team_ppo.pt",
    "models/active/scanner_decoder.pt",
    "models/active/radio_policy.pt",
    "models/active/jammer_policy.pt",
)


def run_operational_validation(
    *,
    root: str | Path = ROOT,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate all operational-v1 evidence into one versioned manifest."""
    root = Path(root)
    checks = [
        _check_acceptance(root),
        _check_learned_control(root),
        _check_marl_environment(root),
        _check_self_play(root),
        _check_information_stack(root),
        _check_replay_fidelity(root),
        _check_physics(root),
        _check_spectator(root),
        _check_command_center(root),
        _check_replay_schema(root),
        _check_differentiable_comms(),
    ]
    passed = all(check["passed"] for check in checks)
    manifest = {
        "version": 1,
        "passed": passed,
        "checks_passed": sum(int(check["passed"]) for check in checks),
        "checks": len(checks),
        "score": sum(int(check["passed"]) for check in checks) / max(1, len(checks)),
        "summary": (
            "Operational readiness evidence is complete."
            if passed
            else "Operational readiness evidence has failing or missing checks."
        ),
        "results": checks,
    }
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _check_acceptance(root: Path) -> dict[str, Any]:
    smoke = _read_json(root / "logs/acceptance_scenarios.json")
    strict = _read_json(root / "logs/acceptance_full_targets.json")
    smoke_ok = _acceptance_passed(smoke, min_required=16, min_milestones=1)
    strict_ok = _acceptance_passed(strict, min_required=16, min_milestones=1)
    return _check(
        "scenario_acceptance_gates",
        smoke_ok and strict_ok,
        metrics={
            "smoke": _acceptance_metrics(smoke),
            "full_targets": _acceptance_metrics(strict),
        },
        thresholds={
            "required_checks": 16,
            "milestone_targets": 1,
            "requires_smoke_pass": True,
            "requires_full_targets_pass": True,
        },
    )


def _check_learned_control(root: Path) -> dict[str, Any]:
    manifests = {
        "evader": root / "logs/evader_checkpoint_acceptance.json",
        "pursuer_team": root / "logs/pursuer_team_checkpoint_acceptance.json",
        "active_control": root / "logs/control_acceptance_active.json",
    }
    metrics = {name: _acceptance_metrics(_read_json(path)) for name, path in manifests.items()}
    checkpoint_state = _checkpoint_state(root)
    active = _read_json(root / "models/control_active/manifest.json")
    active_checkpoints = active.get("active_checkpoints", {}) if isinstance(active, dict) else {}
    active_files_exist = all((root / str(path)).exists() for path in active_checkpoints.values())
    passed = (
        all(row["passed"] for row in metrics.values())
        and active_files_exist
        and all(checkpoint_state[path]["exists"] for path in EXPECTED_CHECKPOINTS[:7])
    )
    return _check(
        "learned_control_stack",
        passed,
        metrics={
            "acceptance": metrics,
            "active_manifest": str(root / "models/control_active/manifest.json"),
            "active_files_exist": active_files_exist,
            "checkpoints": {path: checkpoint_state[path] for path in EXPECTED_CHECKPOINTS[:7]},
        },
        thresholds={"requires_evader_team_and_active_acceptance": True},
    )


def _check_self_play(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "logs/selfplay/selfplay_scenario_selfplay_manifest.json")
    active = _read_json(root / "models/control_active/manifest.json")
    promotion = manifest.get("control_promotion_decision", {}) if isinstance(manifest, dict) else {}
    active_promotion = active.get("promotion_decision", {}) if isinstance(active, dict) else {}
    active_checkpoints = active.get("active_checkpoints", {}) if isinstance(active, dict) else {}
    active_files_exist = all((root / str(path)).exists() for path in active_checkpoints.values())
    control_league = manifest.get("control_league", {}) if isinstance(manifest, dict) else {}
    scenario = control_league.get("scenario_acceptance", {}) if isinstance(control_league, dict) else {}
    scenario_score = scenario.get("score", {}) if isinstance(scenario, dict) else {}
    active_promotion_valid = bool(
        active_promotion.get("promoted")
        and active_promotion.get("scenario_acceptance_passed")
        and active_files_exist
    )
    latest_promotion_valid = bool(
        promotion.get("promoted")
        and promotion.get("scenario_acceptance_passed")
    )
    passed = bool(
        (latest_promotion_valid or active_promotion_valid)
        and scenario.get("passed")
        and int(scenario.get("required_checks_passed", 0)) >= int(scenario.get("required_checks", 1))
        and int(scenario.get("milestone_targets_passed", 0)) >= int(scenario.get("milestone_targets", 1))
    )
    return _check(
        "scenario_gated_self_play",
        passed,
        metrics={
            "promoted": bool(latest_promotion_valid or active_promotion_valid),
            "latest_candidate_promoted": bool(promotion.get("promoted")),
            "active_control_promoted": bool(active_promotion.get("promoted")),
            "active_files_exist": active_files_exist,
            "reason": promotion.get("reason"),
            "active_reason": active_promotion.get("reason"),
            "evader_promoted": bool(promotion.get("evader_promoted") or active_promotion.get("evader_promoted")),
            "pursuer_team_promoted": bool(
                promotion.get("pursuer_team_promoted") or active_promotion.get("pursuer_team_promoted")
            ),
            "candidate_score": float(promotion.get("candidate_score", 0.0) or 0.0),
            "scenario_passed": bool(scenario.get("passed")),
            "scenario_required_checks_passed": int(scenario.get("required_checks_passed", 0)),
            "scenario_required_checks": int(scenario.get("required_checks", 0)),
            "scenario_milestone_targets_passed": int(scenario.get("milestone_targets_passed", 0)),
            "scenario_milestone_targets": int(scenario.get("milestone_targets", 0)),
            "scenario_score": scenario_score,
        },
        thresholds={
            "requires_promoted_active_control_pair": True,
            "latest_candidate_may_be_rejected_when_not_improved": True,
            "requires_scenario_acceptance": True,
        },
    )


def _check_marl_environment(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "logs/env_validation.json")
    results = manifest.get("results", []) if isinstance(manifest, dict) else []
    by_name = {str(item.get("name")): item for item in results if isinstance(item, dict)}
    pettingzoo = by_name.get("pettingzoo_parallel_api", {})
    determinism = by_name.get("seeded_env_determinism", {})
    passed = bool(
        manifest.get("passed")
        and int(manifest.get("checks_passed", 0)) >= int(manifest.get("checks", 1))
        and pettingzoo.get("passed")
        and determinism.get("passed")
    ) if isinstance(manifest, dict) else False
    return _check(
        "marl_environment_contract",
        passed,
        metrics={
            "checks_passed": int(manifest.get("checks_passed", 0)) if isinstance(manifest, dict) else 0,
            "checks": int(manifest.get("checks", 0)) if isinstance(manifest, dict) else 0,
            "pettingzoo_parallel_api": bool(pettingzoo.get("passed")),
            "seeded_env_determinism": bool(determinism.get("passed")),
            "failed_checks": [str(item.get("name")) for item in results if not item.get("passed")],
        },
        thresholds={"requires_pettingzoo_parallel_api": True, "requires_seeded_determinism": True},
    )


def _check_information_stack(root: Path) -> dict[str, Any]:
    league = _read_json(root / "logs/checkpoint_league_manifest.json")
    active = _read_json(root / "models/active/manifest.json")
    league_metrics = _information_metrics(league)
    active_metrics = _information_metrics(active)
    checkpoint_state = _checkpoint_state(root)
    checkpoints_ok = all(checkpoint_state[path]["exists"] for path in EXPECTED_CHECKPOINTS[7:])
    passed = league_metrics["passed"] and active_metrics["passed"] and checkpoints_ok
    return _check(
        "learned_information_stack",
        passed,
        metrics={
            "checkpoint_league": league_metrics,
            "active_information": active_metrics,
            "checkpoints": {path: checkpoint_state[path] for path in EXPECTED_CHECKPOINTS[7:]},
        },
        thresholds={
            "min_overall_score": 0.52,
            "requires_deception_or_spoof_lift": True,
            "requires_active_stack": True,
        },
    )


def _check_replay_fidelity(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "logs/replay_fidelity.json")
    summary = manifest.get("summary", {}) if isinstance(manifest, dict) else {}
    fingerprint = manifest.get("fingerprint", {}) if isinstance(manifest, dict) else {}
    scripted = manifest.get("scripted_determinism", {}) if isinstance(manifest, dict) else {}
    replay = manifest.get("replay") if isinstance(manifest, dict) else None
    replay_exists = bool(replay and (root / str(replay)).exists())
    passed = bool(
        manifest.get("valid")
        and manifest.get("passed")
        and replay_exists
        and int(summary.get("frames", 0)) >= 1200
        and fingerprint.get("state_sha256")
        and fingerprint.get("radio_sha256")
        and scripted.get("matched")
    ) if isinstance(manifest, dict) else False
    return _check(
        "deterministic_replay_fidelity",
        passed,
        metrics={
            "replay": replay,
            "replay_exists": replay_exists,
            "frames": int(summary.get("frames", 0)) if isinstance(summary, dict) else 0,
            "state_sha256": fingerprint.get("state_sha256") if isinstance(fingerprint, dict) else None,
            "radio_sha256": fingerprint.get("radio_sha256") if isinstance(fingerprint, dict) else None,
            "scripted_determinism_matched": bool(scripted.get("matched")) if isinstance(scripted, dict) else False,
        },
        thresholds={"min_frames": 1200, "requires_state_and_radio_hash": True},
    )


def _check_physics(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "logs/physics_validation.json")
    results = manifest.get("results", []) if isinstance(manifest, dict) else []
    by_name = {row.get("name"): row for row in results if isinstance(row, dict)}
    asym = by_name.get("handbrake_asymmetry", {}).get("metrics", {})
    passed = bool(
        manifest.get("passed")
        and int(manifest.get("checks_passed", 0)) >= int(manifest.get("checks", 1))
        and float(asym.get("yaw_ratio", 0.0)) >= 1.08
        and float(asym.get("radius_ratio", 0.0)) >= 1.35
        and float(asym.get("speed_ratio", 1.0)) <= 0.65
    ) if isinstance(manifest, dict) else False
    return _check(
        "vehicle_physics_acceptance",
        passed,
        metrics={
            "checks_passed": int(manifest.get("checks_passed", 0)) if isinstance(manifest, dict) else 0,
            "checks": int(manifest.get("checks", 0)) if isinstance(manifest, dict) else 0,
            "yaw_ratio": float(asym.get("yaw_ratio", 0.0)),
            "radius_ratio": float(asym.get("radius_ratio", 0.0)),
            "speed_ratio": float(asym.get("speed_ratio", 0.0)),
        },
        thresholds={"min_yaw_ratio": 1.08, "min_radius_ratio": 1.35, "max_speed_ratio": 0.65},
    )


def _check_spectator(root: Path) -> dict[str, Any]:
    manifest = _read_json(root / "logs/spectator_validation.json")
    results = manifest.get("results", []) if isinstance(manifest, dict) else []
    by_name = {row.get("name"): row for row in results if isinstance(row, dict)}
    audio = by_name.get("audio_confidence_mapping", {}).get("metrics", {}).get("plans", [])
    links = by_name.get("dashboard_replay_deep_links", {}).get("metrics", {})
    payload = by_name.get("web_replay_payload_contract", {}).get("metrics", {})
    buckets = [str(row.get("bucket")) for row in audio if isinstance(row, dict)]
    valid_replays = int(links.get("valid_replays", 0) or 0)
    with_links = int(links.get("with_viewer_links", 0) or 0)
    passed = bool(
        manifest.get("passed")
        and int(manifest.get("checks_passed", 0)) >= int(manifest.get("checks", 1))
        and buckets == ["low", "mid", "high"]
        and valid_replays > 0
        and valid_replays == with_links
        and payload.get("has_predictions")
        and payload.get("has_radio")
    ) if isinstance(manifest, dict) else False
    return _check(
        "spectator_ui_audio",
        passed,
        metrics={
            "checks_passed": int(manifest.get("checks_passed", 0)) if isinstance(manifest, dict) else 0,
            "checks": int(manifest.get("checks", 0)) if isinstance(manifest, dict) else 0,
            "audio_buckets": buckets,
            "valid_replays": valid_replays,
            "viewer_links": with_links,
            "has_predictions": bool(payload.get("has_predictions")),
            "has_radio": bool(payload.get("has_radio")),
        },
        thresholds={"audio_buckets": ["low", "mid", "high"], "requires_all_replays_linked": True},
    )


def _check_command_center(root: Path) -> dict[str, Any]:
    commands_path = root / "configs/commands.json"
    commands = _read_json(commands_path)
    rows = commands.get("commands", []) if isinstance(commands, dict) else []
    ids = {str(row.get("id")) for row in rows if isinstance(row, dict)}
    missing = [command_id for command_id in REQUIRED_COMMAND_IDS if command_id not in ids]
    scenario = next((row for row in rows if row.get("id") == "run_self_play_cycle_scenario_gated"), {})
    scenario_args = str(scenario.get("default_args", ""))
    static_paths = [
        root / "command_center/static/index.html",
        root / "command_center/static/app.js",
        root / "command_center/static/style.css",
        root / "viewer3d/static/index.html",
        root / "viewer3d/static/app.js",
        root / "viewer3d/static/style.css",
    ]
    missing_static = [str(path.relative_to(root)) for path in static_paths if not path.exists()]
    passed = bool(
        not missing
        and not missing_static
        and "--evader-init-checkpoint checkpoints/evader_ppo.pt" in scenario_args
        and "--evader-validation-steps" in scenario_args
    )
    return _check(
        "dashboard_command_control",
        passed,
        metrics={
            "commands": len(ids),
            "missing_command_ids": missing,
            "missing_static_assets": missing_static,
            "scenario_gated_has_warm_start": "--evader-init-checkpoint checkpoints/evader_ppo.pt" in scenario_args,
            "scenario_gated_has_validation": "--evader-validation-steps" in scenario_args,
        },
        thresholds={"required_command_ids": list(REQUIRED_COMMAND_IDS), "requires_dashboard_and_web_assets": True},
    )


def _check_replay_schema(root: Path) -> dict[str, Any]:
    from .replay import load_replay, summarize_replay, validate_replay

    manifest = _read_json(root / "logs/control_acceptance_active.json")
    replays = []
    for scenario in manifest.get("scenarios", []) if isinstance(manifest, dict) else []:
        replay = scenario.get("replay")
        if replay:
            replays.append(str(replay))
    summaries = []
    failures = []
    for replay in replays:
        path = root / replay
        try:
            lines = load_replay(path)
            validate_replay(lines)
            summary = summarize_replay(lines, include_fingerprint=True)
            frames = lines[1:]
            first_frame = frames[0] if frames else {}
            required_frame_keys = ("actions", "collisions", "rewards", "reward_components", "camera")
            missing_frame_keys = [key for key in required_frame_keys if key not in first_frame]
            reward_keys = set((first_frame.get("rewards") or {}).keys())
            component_keys = set((first_frame.get("reward_components") or {}).keys())
            expected_agents = {"EVADER", "P1", "P2", "P3", "P4", "P5"}
            component_probe = (first_frame.get("reward_components") or {}).get("EVADER", {})
            reward_evidence_ok = bool(
                expected_agents.issubset(reward_keys)
                and expected_agents.issubset(component_keys)
                and "decoder_accuracy" in component_probe
                and "spoof_susceptibility" in component_probe
                and "information_reward" in component_probe
            )
            comms_evidence_ok = all(
                key in summary
                for key in (
                    "radio_word_entropy",
                    "pursuer_word_entropy",
                    "radio_spoof_ratio",
                    "jam_events",
                    "cipher_rotations",
                    "avg_jam_confidence_drop",
                    "avg_cipher_confidence_drop",
                    "avg_cipher_confidence_recovery",
                )
            )
            enhanced_schema_ok = bool(
                summary["frames"] > 0
                and not missing_frame_keys
                and summary["action_frames"] == summary["frames"]
                and summary["camera_hint_frames"] == summary["frames"]
                and summary["reward_frames"] == summary["frames"]
                and reward_evidence_ok
                and comms_evidence_ok
            )
            summaries.append({
                "replay": replay,
                "frames": summary["frames"],
                "radio_events": summary["radio_events"],
                "action_frames": summary["action_frames"],
                "camera_hint_frames": summary["camera_hint_frames"],
                "reward_frames": summary["reward_frames"],
                "collision_events": summary["collision_events"],
                "missing_frame_keys": missing_frame_keys,
                "reward_evidence_ok": reward_evidence_ok,
                "comms_evidence_ok": comms_evidence_ok,
                "enhanced_schema_ok": enhanced_schema_ok,
                "state_sha256": summary["state_sha256"],
                "radio_sha256": summary["radio_sha256"],
            })
            if not enhanced_schema_ok:
                failures.append({
                    "replay": replay,
                    "error": "active acceptance replay is missing enhanced action/reward/comms/camera evidence",
                })
        except Exception as exc:  # pragma: no cover - reported as manifest evidence
            failures.append({"replay": replay, "error": f"{type(exc).__name__}: {exc}"})
    passed = bool(replays and not failures and len(summaries) == len(replays))
    return _check(
        "versioned_replay_schema",
        passed,
        metrics={
            "expected_replays": replays,
            "validated_replays": len(summaries),
            "failures": failures,
            "summaries": summaries,
        },
        thresholds={
            "requires_all_active_acceptance_replays_valid": True,
            "requires_actions_collisions_rewards_and_camera": True,
            "requires_comms_authentication_analytics": True,
        },
    )


def _check_differentiable_comms() -> dict[str, Any]:
    try:
        import torch

        from .comms import VOCABULARY
        from .marl import PursuerPolicy
        from .observations import OBS_SIZE
        from .radio import MESSAGE_LEN, RadioTokenPolicy

        torch.manual_seed(7)
        pursuer = PursuerPolicy(OBS_SIZE, vocab_size=len(VOCABULARY), hidden=32, message_len=MESSAGE_LEN)
        obs_seq = torch.zeros(2, 3, OBS_SIZE)
        output = pursuer(obs_seq, agent_id=torch.tensor([0, 1]), tau=0.9)
        relaxed = output["relaxed_tokens"]
        weights = torch.linspace(0.0, 1.0, len(VOCABULARY))
        loss = (relaxed * weights).sum()
        loss.backward()
        pursuer_grad = float(pursuer.words.weight.grad.abs().sum())

        radio = RadioTokenPolicy(hidden=32)
        radio_obs = torch.zeros(2, OBS_SIZE)
        radio_ids = torch.tensor([0, 1])
        radio_relaxed = radio.relaxed_tokens(radio_obs, radio_ids, tau=0.9, hard=True)
        radio_loss = (radio_relaxed * weights).sum()
        radio_loss.backward()
        radio_grad = float(radio.net[-1].weight.grad.abs().sum())
        passed = bool(
            tuple(relaxed.shape) == (2, MESSAGE_LEN, len(VOCABULARY))
            and tuple(output["token_logits"].shape) == (2, MESSAGE_LEN, len(VOCABULARY))
            and torch.allclose(relaxed.sum(dim=-1), torch.ones(2, MESSAGE_LEN))
            and tuple(radio_relaxed.shape) == (2, MESSAGE_LEN, len(VOCABULARY))
            and torch.allclose(radio_relaxed.sum(dim=-1), torch.ones(2, MESSAGE_LEN))
            and pursuer_grad > 0.0
            and radio_grad > 0.0
        )
        metrics = {
            "message_len": MESSAGE_LEN,
            "vocab_size": len(VOCABULARY),
            "pursuer_relaxed_shape": list(relaxed.shape),
            "radio_relaxed_shape": list(radio_relaxed.shape),
            "pursuer_grad_abs_sum": pursuer_grad,
            "radio_grad_abs_sum": radio_grad,
        }
    except Exception as exc:  # pragma: no cover - reported as manifest evidence
        passed = False
        metrics = {"error": f"{type(exc).__name__}: {exc}"}
    return _check(
        "differentiable_english_token_comms",
        passed,
        metrics=metrics,
        thresholds={"message_len": 6, "requires_straight_through_gradients": True},
    )


def _acceptance_passed(manifest: dict[str, Any] | None, *, min_required: int, min_milestones: int) -> bool:
    if not isinstance(manifest, dict):
        return False
    return bool(
        manifest.get("passed")
        and int(manifest.get("required_checks_passed", 0)) >= min_required
        and int(manifest.get("required_checks", 0)) >= min_required
        and int(manifest.get("milestone_targets_passed", 0)) >= min_milestones
        and int(manifest.get("milestone_targets", 0)) >= min_milestones
    )


def _acceptance_metrics(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {"exists": False, "passed": False}
    return {
        "exists": True,
        "passed": bool(manifest.get("passed")),
        "required_checks_passed": int(manifest.get("required_checks_passed", 0)),
        "required_checks": int(manifest.get("required_checks", 0)),
        "milestone_targets_passed": int(manifest.get("milestone_targets_passed", 0)),
        "milestone_targets": int(manifest.get("milestone_targets", 0)),
        "diagnostic_count": int((manifest.get("diagnostics") or {}).get("count", 0)),
    }


def _information_metrics(manifest: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        return {"exists": False, "passed": False}
    scores = manifest.get("scores") or {}
    aggregates = manifest.get("aggregates") or {}
    overall = float(scores.get("overall_score", 0.0))
    deception_lift = float(aggregates.get("mean_deception_lift", 0.0))
    spoof_lift = float(aggregates.get("mean_spoof_event_lift", 0.0))
    passed = bool(overall >= 0.52 and (deception_lift > 0.0 or spoof_lift > 0.0))
    return {
        "exists": True,
        "passed": passed,
        "overall_score": overall,
        "pursuer_security_score": float(scores.get("pursuer_security_score", 0.0)),
        "evader_pressure_score": float(scores.get("evader_pressure_score", 0.0)),
        "adversarial_balance_score": float(scores.get("adversarial_balance_score", 0.0)),
        "mean_deception_lift": deception_lift,
        "mean_spoof_event_lift": spoof_lift,
        "mean_spoof_susceptibility": float(aggregates.get("mean_spoof_susceptibility", 0.0)),
    }


def _checkpoint_state(root: Path) -> dict[str, dict[str, Any]]:
    state = {}
    for rel in EXPECTED_CHECKPOINTS:
        path = root / rel
        state[rel] = {
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() else 0,
        }
    return state


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, passed: bool, *, metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "metrics": metrics,
        "thresholds": thresholds,
    }
