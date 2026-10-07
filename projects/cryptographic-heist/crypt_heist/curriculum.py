"""Named training stages and promotion gates."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path
from typing import Any


class CurriculumStage(str, Enum):
    SCRIPTED_SANDBOX = "scripted_sandbox"
    EVADER_WAYPOINTS = "evader_waypoints"
    SINGLE_PURSUER_CHASE = "single_pursuer_chase"
    MULTI_PURSUER_CONTAINMENT = "multi_pursuer_containment"
    LEARNED_PURSUER_COMMS = "learned_pursuer_comms"
    EVADER_SCANNER = "evader_scanner"
    ADVERSARIAL_JAMMING = "adversarial_jamming"
    AUTHENTICATED_SELF_PLAY = "authenticated_self_play"


@dataclass(frozen=True)
class StageGate:
    min_eval_episodes: int
    waypoint_rate: float = 0.0
    capture_rate: float = 0.0
    decoder_accuracy: float = 0.0
    deception_rate: float = 0.0


GATES = {
    CurriculumStage.SCRIPTED_SANDBOX: StageGate(8),
    CurriculumStage.EVADER_WAYPOINTS: StageGate(16, waypoint_rate=0.55),
    CurriculumStage.SINGLE_PURSUER_CHASE: StageGate(16, capture_rate=0.45),
    CurriculumStage.MULTI_PURSUER_CONTAINMENT: StageGate(24, capture_rate=0.62),
    CurriculumStage.LEARNED_PURSUER_COMMS: StageGate(24, capture_rate=0.62),
    CurriculumStage.EVADER_SCANNER: StageGate(24, decoder_accuracy=0.70),
    CurriculumStage.ADVERSARIAL_JAMMING: StageGate(32, deception_rate=0.35),
    CurriculumStage.AUTHENTICATED_SELF_PLAY: StageGate(64, capture_rate=0.55, deception_rate=0.30),
}


def ordered_stages() -> list[CurriculumStage]:
    return list(CurriculumStage)


def next_stage(stage: CurriculumStage) -> CurriculumStage:
    stages = ordered_stages()
    i = stages.index(stage)
    return stages[min(i + 1, len(stages) - 1)]


@dataclass(frozen=True)
class DiagnosticRoute:
    lane: str
    stage: CurriculumStage
    owner: str
    command_id: str
    default_args: str
    objective: str
    rationale: str
    followup_command_ids: tuple[str, ...]
    tags: tuple[str, ...] = field(default_factory=tuple)


ROUTE_BY_REASON: dict[str, DiagnosticRoute] = {
    "insufficient_waypoint_progress": DiagnosticRoute(
        lane="evader_waypoint_driver",
        stage=CurriculumStage.EVADER_WAYPOINTS,
        owner="evader",
        command_id="train_evader_ppo",
        default_args=(
            "--out checkpoints/evader_ppo.pt --init-checkpoint checkpoints/evader_imitation.pt "
            "--bootstrap-imitation-if-missing --bootstrap-imitation-steps 12000 --bootstrap-imitation-epochs 20 "
            "--bootstrap-imitation-dagger-rounds 5 --bootstrap-imitation-dagger-steps 1200 "
            "--validation-steps 1200 --updates 12 --steps-per-update 768 "
            "--counterfactual-interval 16 --counterfactual-horizon-steps 24"
        ),
        objective="Improve waypoint progress, recovery, and useful escape speed.",
        rationale="Downtown chase diagnostics show the evader is not reaching enough route targets.",
        followup_command_ids=("watch_evader_ppo_checkpoint", "eval_evader_checkpoint_acceptance", "eval_acceptance_scenarios"),
        tags=("evader", "ppo", "waypoints", "control"),
    ),
    "waypoint_milestone_gap": DiagnosticRoute(
        lane="evader_waypoint_driver",
        stage=CurriculumStage.EVADER_WAYPOINTS,
        owner="evader",
        command_id="train_evader_ppo",
        default_args=(
            "--out checkpoints/evader_ppo.pt --init-checkpoint checkpoints/evader_imitation.pt "
            "--bootstrap-imitation-if-missing --bootstrap-imitation-steps 12000 --bootstrap-imitation-epochs 20 "
            "--bootstrap-imitation-dagger-rounds 5 --bootstrap-imitation-dagger-steps 1200 "
            "--validation-steps 1200 --updates 10 --steps-per-update 768 "
            "--counterfactual-interval 16 --counterfactual-horizon-steps 24"
        ),
        objective="Close the gap between the smoke gate and the full three-waypoint milestone.",
        rationale="The scenario passes the early gate but still misses the original downtown waypoint target.",
        followup_command_ids=("watch_evader_ppo_checkpoint", "eval_evader_checkpoint_acceptance", "eval_acceptance_full_targets"),
        tags=("evader", "milestone", "waypoints"),
    ),
    "speed_collapse": DiagnosticRoute(
        lane="evader_waypoint_driver",
        stage=CurriculumStage.EVADER_WAYPOINTS,
        owner="evader",
        command_id="train_evader_ppo",
        default_args=(
            "--out checkpoints/evader_ppo.pt --init-checkpoint checkpoints/evader_imitation.pt "
            "--bootstrap-imitation-if-missing --bootstrap-imitation-steps 12000 --bootstrap-imitation-epochs 20 "
            "--bootstrap-imitation-dagger-rounds 5 --bootstrap-imitation-dagger-steps 1200 "
            "--validation-steps 1200 --updates 12 --steps-per-update 768"
        ),
        objective="Recover useful forward speed after turns, impacts, and route corrections.",
        rationale="The evader is surviving without enough speed to make the chase functional.",
        followup_command_ids=("watch_evader_ppo_checkpoint", "eval_evader_checkpoint_acceptance", "eval_acceptance_scenarios"),
        tags=("evader", "speed", "control"),
    ),
    "stalled_before_waypoint": DiagnosticRoute(
        lane="evader_waypoint_driver",
        stage=CurriculumStage.EVADER_WAYPOINTS,
        owner="evader",
        command_id="train_evader_ppo",
        default_args=(
            "--out checkpoints/evader_ppo.pt --init-checkpoint checkpoints/evader_imitation.pt "
            "--bootstrap-imitation-if-missing --bootstrap-imitation-steps 12000 --bootstrap-imitation-epochs 20 "
            "--bootstrap-imitation-dagger-rounds 5 --bootstrap-imitation-dagger-steps 1200 "
            "--validation-steps 1200 --updates 10 --steps-per-update 768"
        ),
        objective="Reduce final-position stalls near active waypoints.",
        rationale="The evader ends nearly stationary before reaching its next objective.",
        followup_command_ids=("watch_evader_ppo_checkpoint", "eval_evader_checkpoint_acceptance", "eval_acceptance_scenarios"),
        tags=("evader", "recovery", "waypoints"),
    ),
    "high_collision_energy": DiagnosticRoute(
        lane="evader_waypoint_driver",
        stage=CurriculumStage.EVADER_WAYPOINTS,
        owner="evader",
        command_id="train_evader_ppo",
        default_args=(
            "--out checkpoints/evader_ppo.pt --init-checkpoint checkpoints/evader_imitation.pt "
            "--bootstrap-imitation-if-missing --bootstrap-imitation-steps 12000 --bootstrap-imitation-epochs 20 "
            "--bootstrap-imitation-dagger-rounds 5 --bootstrap-imitation-dagger-steps 1200 "
            "--validation-steps 1200 --updates 10 --steps-per-update 768"
        ),
        objective="Reduce hard wall strikes while preserving drift-capable escape behavior.",
        rationale="Large impacts are probably degrading route completion and replay quality.",
        followup_command_ids=("watch_evader_ppo_checkpoint", "eval_evader_checkpoint_acceptance", "eval_acceptance_scenarios"),
        tags=("evader", "collision", "control"),
    ),
    "capture_not_confirmed": DiagnosticRoute(
        lane="pursuer_team_containment",
        stage=CurriculumStage.MULTI_PURSUER_CONTAINMENT,
        owner="pursuer_team",
        command_id="train_pursuer_team_ppo",
        default_args=(
            "--out checkpoints/pursuer_team_ppo.pt --updates 12 --steps-per-update 768 "
            "--counterfactual-interval 16 --counterfactual-horizon-steps 24"
        ),
        objective="Make roadblocks produce confirmed low-speed captures.",
        rationale="The team reaches the scenario but does not complete the capture condition.",
        followup_command_ids=("watch_pursuer_team_ppo_checkpoint", "eval_acceptance_scenarios"),
        tags=("pursuer", "team", "capture", "containment"),
    ),
    "weak_boxing": DiagnosticRoute(
        lane="pursuer_team_containment",
        stage=CurriculumStage.MULTI_PURSUER_CONTAINMENT,
        owner="pursuer_team",
        command_id="train_pursuer_team_ppo",
        default_args="--out checkpoints/pursuer_team_ppo.pt --updates 12 --steps-per-update 768",
        objective="Improve sustained boxing geometry around the evader.",
        rationale="Pursuers are not maintaining enough simultaneous pressure frames.",
        followup_command_ids=("watch_pursuer_team_ppo_checkpoint", "eval_acceptance_scenarios"),
        tags=("pursuer", "team", "boxing"),
    ),
    "slow_capture": DiagnosticRoute(
        lane="pursuer_team_containment",
        stage=CurriculumStage.MULTI_PURSUER_CONTAINMENT,
        owner="pursuer_team",
        command_id="train_pursuer_team_ppo",
        default_args="--out checkpoints/pursuer_team_ppo.pt --updates 10 --steps-per-update 768",
        objective="Shorten roadblock closure time without causing pileups.",
        rationale="The team captures eventually, but too slowly for the gate.",
        followup_command_ids=("watch_pursuer_team_ppo_checkpoint", "eval_acceptance_scenarios"),
        tags=("pursuer", "team", "roadblock"),
    ),
    "spoof_delivery_missing": DiagnosticRoute(
        lane="evader_jammer",
        stage=CurriculumStage.ADVERSARIAL_JAMMING,
        owner="information_warfare",
        command_id="train_jammer_policy",
        default_args="--out checkpoints/jammer_policy.pt --steps 4800 --epochs 14",
        objective="Restore reliable spoof-token delivery through the normal radio action path.",
        rationale="The scenario did not observe enough evader-injected radio words.",
        followup_command_ids=("watch_jammer_policy", "eval_acceptance_scenarios"),
        tags=("jammer", "jamming", "comms"),
    ),
    "jam_not_deceptive": DiagnosticRoute(
        lane="counterfactual_jammer",
        stage=CurriculumStage.ADVERSARIAL_JAMMING,
        owner="information_warfare",
        command_id="train_counterfactual_jammer_policy",
        default_args=(
            "--out checkpoints/jammer_counterfactual.pt --steps 1800 --sample-every 12 "
            "--horizon-steps 72 --epochs 14"
        ),
        objective="Train spoof bursts against measured pursuer deviation.",
        rationale="Jamming is firing, but it is not creating enough deception pressure.",
        followup_command_ids=("watch_jammer_policy", "eval_authentication_curriculum"),
        tags=("jammer", "counterfactual", "deception"),
    ),
    "spoof_targets_inactive": DiagnosticRoute(
        lane="counterfactual_jammer",
        stage=CurriculumStage.ADVERSARIAL_JAMMING,
        owner="information_warfare",
        command_id="train_counterfactual_jammer_policy",
        default_args="--out checkpoints/jammer_counterfactual.pt --steps 1800 --sample-every 12 --horizon-steps 72 --epochs 14",
        objective="Prefer spoof targets that are close enough to affect containment.",
        rationale="Spoof payloads are not staying active on enough relevant pursuers.",
        followup_command_ids=("watch_jammer_policy", "eval_authentication_curriculum"),
        tags=("jammer", "targets", "counterfactual"),
    ),
    "confidence_not_damaged": DiagnosticRoute(
        lane="scanner_decoder",
        stage=CurriculumStage.EVADER_SCANNER,
        owner="information_warfare",
        command_id="train_scanner_decoder",
        default_args="--out checkpoints/scanner_decoder.pt --steps 3600 --epochs 10",
        objective="Make scanner confidence respond clearly to spoof disruption.",
        rationale="Spoof bursts are not producing an obvious decoder confidence drop.",
        followup_command_ids=("watch_scanner_decoder", "eval_acceptance_scenarios"),
        tags=("scanner", "confidence", "decoder"),
    ),
    "cipher_rotation_missing": DiagnosticRoute(
        lane="radio_authentication_probe",
        stage=CurriculumStage.LEARNED_PURSUER_COMMS,
        owner="information_warfare",
        command_id="eval_acceptance_scenarios",
        default_args="--scenarios cipher_shift --seed 11 --record-dir replays/acceptance --out logs/cipher_probe.json",
        objective="Verify that the radio channel can rotate protocols during cipher-shift scenarios.",
        rationale="The acceptance scenario did not observe a cipher rotation event.",
        followup_command_ids=("watch_radio_policy", "run_information_cycle"),
        tags=("radio", "cipher", "acceptance"),
    ),
    "cipher_shift_not_visible": DiagnosticRoute(
        lane="scanner_decoder",
        stage=CurriculumStage.EVADER_SCANNER,
        owner="information_warfare",
        command_id="train_scanner_decoder",
        default_args="--out checkpoints/scanner_decoder.pt --steps 3600 --epochs 10",
        objective="Make cipher shifts visibly perturb scanner confidence.",
        rationale="Protocol rotation happened, but the decoder confidence did not move enough.",
        followup_command_ids=("watch_scanner_decoder", "eval_acceptance_scenarios"),
        tags=("scanner", "cipher", "confidence"),
    ),
    "scanner_recovery_weak": DiagnosticRoute(
        lane="scanner_decoder",
        stage=CurriculumStage.EVADER_SCANNER,
        owner="information_warfare",
        command_id="train_scanner_decoder",
        default_args="--out checkpoints/scanner_decoder.pt --steps 4200 --epochs 12",
        objective="Improve decoder adaptation after a protocol shift.",
        rationale="Confidence drops after cipher rotation but does not recover cleanly.",
        followup_command_ids=("watch_scanner_decoder", "eval_acceptance_scenarios"),
        tags=("scanner", "recovery", "cipher"),
    ),
}

OWNER_FALLBACK_ROUTE = {
    "evader": ROUTE_BY_REASON["insufficient_waypoint_progress"],
    "pursuer_team": ROUTE_BY_REASON["weak_boxing"],
    "information_warfare": ROUTE_BY_REASON["jam_not_deceptive"],
}

SEVERITY_WEIGHT = {"info": 0, "low": 12, "medium": 36, "high": 64}


def load_curriculum_manifest(path: str | Path) -> dict[str, Any]:
    """Load a scenario acceptance manifest from disk."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plan_curriculum_from_manifest(
    manifest: dict[str, Any],
    *,
    source: str | Path | None = None,
    max_actions: int = 5,
) -> dict[str, Any]:
    """Route scenario diagnostics to dashboard-launchable training actions."""
    diagnostics = _manifest_diagnostics(manifest)
    source_text = str(source) if source is not None else str(manifest.get("manifest", "in-memory"))
    if not diagnostics:
        return {
            "version": 1,
            "source_manifest": source_text,
            "status": "nominal",
            "diagnostic_count": 0,
            "summary": "All acceptance diagnostics are nominal. Keep the scenario gate active and raise thresholds gradually.",
            "actions": [
                {
                    "priority": 24,
                    "lane": "scenario_gated_self_play",
                    "stage": CurriculumStage.AUTHENTICATED_SELF_PLAY.value,
                    "owner": "system",
                    "command_id": "run_self_play_cycle_scenario_gated",
                    "default_args": (
                        "--name selfplay_scenario --seed 11 --evader-updates 4 --team-updates 4 "
                        "--steps-per-update 384 --evader-init-checkpoint checkpoints/evader_ppo.pt "
                        "--evader-validation-steps 1200 --evader-validation-seed 11 "
                        "--league-seeds 11,12 --evaluate-control-scenarios "
                        "--control-scenario-seed 31 --control-scenario-threshold 0.90 "
                        "--evader-control-scenario-threshold 0.45 "
                        "--pursuer-team-control-scenario-threshold 0.50 --promotion-threshold 0.52"
                    ),
                    "objective": "Keep learned controls moving through replay-backed promotion gates.",
                    "rationale": "No scenario diagnostics need targeted repair right now.",
                    "diagnostic_reasons": [],
                    "scenarios": [],
                    "evidence": {},
                    "followup_command_ids": ("eval_acceptance_full_targets",),
                    "tags": ("self-play", "scenario", "promotion"),
                }
            ],
        }

    grouped: dict[str, dict[str, Any]] = {}
    for diagnostic in diagnostics:
        route = _route_for_diagnostic(diagnostic)
        lane = route.lane
        severity = str(diagnostic.get("severity", "info"))
        scenario = str(diagnostic.get("scenario", "unknown"))
        reason = str(diagnostic.get("reason", "unknown"))
        priority = SEVERITY_WEIGHT.get(severity, 0) + int(diagnostic.get("count", 1))
        if reason in {"failed_required_check", "missed_milestone_target"}:
            priority += 8
        row = grouped.setdefault(
            lane,
            {
                "priority": 0,
                "lane": lane,
                "stage": route.stage.value,
                "owner": route.owner,
                "command_id": route.command_id,
                "default_args": route.default_args,
                "objective": route.objective,
                "rationale": route.rationale,
                "diagnostic_reasons": [],
                "scenarios": [],
                "evidence": {},
                "followup_command_ids": route.followup_command_ids,
                "tags": route.tags,
            },
        )
        row["priority"] += priority
        if reason not in row["diagnostic_reasons"]:
            row["diagnostic_reasons"].append(reason)
        if scenario not in row["scenarios"]:
            row["scenarios"].append(scenario)
        row["evidence"][reason] = _jsonable(diagnostic.get("evidence", {}))

    actions = sorted(grouped.values(), key=lambda item: item["priority"], reverse=True)
    actions = actions[: max(1, int(max_actions))]
    top = diagnostics[0]
    return {
        "version": 1,
        "source_manifest": source_text,
        "status": "needs_training",
        "diagnostic_count": len(diagnostics),
        "summary": (
            f"Top diagnostic is {top.get('scenario', 'unknown')}:"
            f"{top.get('severity', 'info')}:{top.get('reason', 'unknown')}."
        ),
        "actions": actions,
    }


def write_curriculum_plan(plan: dict[str, Any], path: str | Path) -> Path:
    """Write a curriculum plan JSON artifact and return its path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(plan), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _manifest_diagnostics(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    aggregate = manifest.get("diagnostics") or {}
    rows = [dict(row) for row in aggregate.get("top", []) if row.get("reason") != "scenario_nominal"]
    if rows:
        return sorted(rows, key=_diagnostic_sort_key, reverse=True)
    rows = []
    for scenario in manifest.get("scenarios", []):
        scenario_name = scenario.get("name", "unknown")
        for diagnostic in scenario.get("diagnostics", []):
            if diagnostic.get("reason") == "scenario_nominal":
                continue
            rows.append({**diagnostic, "scenario": scenario_name})
    return sorted(rows, key=_diagnostic_sort_key, reverse=True)


def _route_for_diagnostic(diagnostic: dict[str, Any]) -> DiagnosticRoute:
    reason = str(diagnostic.get("reason", ""))
    if reason in ROUTE_BY_REASON:
        return ROUTE_BY_REASON[reason]
    owner = str(diagnostic.get("owner", "information_warfare"))
    return OWNER_FALLBACK_ROUTE.get(owner, OWNER_FALLBACK_ROUTE["information_warfare"])


def _diagnostic_sort_key(diagnostic: dict[str, Any]) -> tuple[int, int]:
    severity = str(diagnostic.get("severity", "info"))
    return (SEVERITY_WEIGHT.get(severity, 0), int(diagnostic.get("count", 1)))


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value
