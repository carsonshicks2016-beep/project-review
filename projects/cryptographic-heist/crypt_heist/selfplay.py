"""Self-play orchestration for control and information checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from .control_pool import (
    ControlOpponentSelection,
    ControlPoolEntry,
    archive_control_pair,
    load_control_pool,
    select_control_opponent,
)
from .control_league import (
    ControlCandidateLeagueResult,
    ControlPairEvalResult,
    evaluate_control_candidate_league,
    evaluate_control_pair,
)
from .cycle import InformationCycleResult, run_information_cycle
from .ppo import train_evader_ppo, train_pursuer_team_ppo

SELF_PLAY_VERSION = 1


@dataclass
class ControlCheckpointSet:
    evader: str | None = None
    pursuer_team: str | None = None
    manifest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "evader": self.evader,
            "pursuer_team": self.pursuer_team,
            "manifest": self.manifest,
        }


@dataclass
class SelfPlayCycleResult:
    name: str
    seed: int
    checkpoints: dict[str, str]
    training: dict[str, Any]
    control_opponents: dict[str, Any]
    active_control: dict[str, Any]
    manifest: str
    control_evaluation: dict[str, Any] | None = None
    control_league: dict[str, Any] | None = None
    control_promotion_decision: dict[str, Any] | None = None
    control_pool_entry: dict[str, Any] | None = None
    information_cycle: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SELF_PLAY_VERSION,
            "name": self.name,
            "seed": self.seed,
            "checkpoints": self.checkpoints,
            "training": self.training,
            "control_opponents": self.control_opponents,
            "active_control": self.active_control,
            "control_evaluation": self.control_evaluation,
            "control_league": self.control_league,
            "control_promotion_decision": self.control_promotion_decision,
            "control_pool_entry": self.control_pool_entry,
            "information_cycle": self.information_cycle,
            "manifest": self.manifest,
        }


def run_self_play_cycle(
    *,
    name: str = "selfplay",
    seed: int = 11,
    checkpoint_dir: str | Path = "checkpoints/selfplay",
    log_dir: str | Path = "logs/selfplay",
    active_control_dir: str | Path = "models/control_active",
    control_pool_dir: str | Path = "models/control_pool",
    active_info_dir: str | Path = "models/active",
    evader_updates: int = 4,
    team_updates: int = 4,
    steps_per_update: int = 384,
    evader_init_checkpoint: str | Path | None = None,
    evader_validation_steps: int = 0,
    evader_validation_seed: int | None = None,
    max_cycles: int = 900,
    control_repeat: int = 4,
    train_epochs: int = 4,
    evader_minibatch_size: int = 128,
    team_minibatch_size: int = 256,
    hidden: int = 128,
    lr: float = 3e-4,
    counterfactual_interval: int = 16,
    counterfactual_horizon_steps: int = 24,
    counterfactual_evader_weight: float = 0.20,
    counterfactual_pursuer_weight: float = 0.20,
    device: str = "cpu",
    use_control_pool: bool = True,
    evaluate_control: bool = True,
    control_eval_seeds: list[int] | None = None,
    control_eval_steps: int = 600,
    control_record_prefix: str | Path | None = "replays/selfplay",
    control_pool_eval_opponents: int = 3,
    evaluate_control_scenarios: bool = False,
    control_scenario_seed: int = 31,
    control_scenario_steps: int | None = None,
    control_scenario_record_dir: str | Path | None = "replays/control_acceptance",
    control_scenario_threshold: float | None = None,
    evader_control_scenario_threshold: float | None = None,
    pursuer_team_control_scenario_threshold: float | None = None,
    promote_control: bool = True,
    control_promotion_threshold: float = 0.0,
    control_improvement_margin: float = 0.0,
    evader_control_promotion_threshold: float | None = None,
    evader_control_improvement_margin: float | None = None,
    pursuer_team_control_promotion_threshold: float | None = None,
    pursuer_team_control_improvement_margin: float | None = None,
    train_information: bool = True,
    scanner_steps: int = 1200,
    scanner_epochs: int = 4,
    scanner_batch_size: int = 128,
    scanner_horizon_steps: int = 24,
    scanner_window: int = 16,
    scanner_sample_every: int = 4,
    scanner_embed: int = 24,
    radio_steps: int = 1800,
    radio_epochs: int = 6,
    radio_batch_size: int = 96,
    jammer_steps: int = 1800,
    jammer_epochs: int = 6,
    jammer_batch_size: int = 96,
    jammer_mode: str = "heuristic",
    jammer_sample_every: int = 12,
    jammer_horizon_steps: int = 48,
    jammer_positive_threshold: float = 0.16,
    league_seeds: list[int] | None = None,
    league_steps: int = 600,
    promotion_threshold: float = 0.52,
    improvement_margin: float = 0.01,
    promote_information: bool = True,
) -> SelfPlayCycleResult:
    """Run one alternating self-play cycle for driving and information policies."""
    safe_name = _safe_name(name)
    checkpoint_root = Path(checkpoint_dir)
    log_root = Path(log_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    active_before = load_active_control_set(active_control_dir)
    pool_before = load_control_pool(control_pool_dir) if use_control_pool else []
    evader_training_opponent = _select_control_opponent(
        use_control_pool=use_control_pool,
        pool_dir=control_pool_dir,
        kind="pursuer_team",
        seed=seed,
        fallback=active_before.pursuer_team,
        fallback_label="active_pursuer_team",
    )
    evader_path = checkpoint_root / f"{safe_name}_evader_ppo.pt"
    team_path = checkpoint_root / f"{safe_name}_pursuer_team_ppo.pt"

    evader_result = train_evader_ppo(
        evader_path,
        seed=seed,
        updates=evader_updates,
        steps_per_update=steps_per_update,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        train_epochs=train_epochs,
        minibatch_size=evader_minibatch_size,
        hidden=hidden,
        lr=lr,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
        opponent_pursuer_team_checkpoint=evader_training_opponent.checkpoint,
        init_checkpoint=evader_init_checkpoint,
        validation_steps=evader_validation_steps,
        validation_seed=evader_validation_seed,
        device=device,
    )
    team_training_opponent = _select_control_opponent(
        use_control_pool=use_control_pool,
        pool_dir=control_pool_dir,
        kind="evader",
        seed=seed + 1,
        fallback=evader_path,
        fallback_label="candidate_evader",
    )
    team_result = train_pursuer_team_ppo(
        team_path,
        seed=seed + 1,
        updates=team_updates,
        steps_per_update=steps_per_update,
        max_cycles=max_cycles,
        control_repeat=control_repeat,
        train_epochs=train_epochs,
        minibatch_size=team_minibatch_size,
        hidden=hidden,
        lr=lr,
        counterfactual_interval=counterfactual_interval,
        counterfactual_horizon_steps=counterfactual_horizon_steps,
        counterfactual_evader_weight=counterfactual_evader_weight,
        counterfactual_pursuer_weight=counterfactual_pursuer_weight,
        opponent_evader_checkpoint=team_training_opponent.checkpoint,
        device=device,
    )
    control_result: ControlPairEvalResult | None = None
    control_league: ControlCandidateLeagueResult | None = None
    if evaluate_control:
        control_result = evaluate_control_pair(
            name=safe_name,
            evader=evader_path,
            pursuer_team=team_path,
            seeds=control_eval_seeds or league_seeds or [seed],
            steps=control_eval_steps,
            record_prefix=control_record_prefix,
            out=log_root / f"{safe_name}_control_pair_manifest.json",
        )
        control_league = evaluate_control_candidate_league(
            name=safe_name,
            evader=evader_path,
            pursuer_team=team_path,
            pool_dir=control_pool_dir,
            seeds=control_eval_seeds or league_seeds or [seed],
            steps=control_eval_steps,
            max_opponents=control_pool_eval_opponents,
            record_prefix=control_record_prefix,
            out=log_root / f"{safe_name}_control_league_manifest.json",
            self_pair=control_result.to_dict(),
            evaluate_scenarios=evaluate_control_scenarios,
            scenario_seed=control_scenario_seed,
            scenario_steps=control_scenario_steps,
            scenario_record_dir=control_scenario_record_dir,
            scenario_out=log_root / f"{safe_name}_control_acceptance_manifest.json" if evaluate_control_scenarios else None,
        )

    manifest_path = log_root / f"{safe_name}_selfplay_manifest.json"
    info_result: InformationCycleResult | None = None
    if train_information:
        info_result = run_information_cycle(
            name=f"{safe_name}_info",
            seed=seed + 2,
            checkpoint_dir=checkpoint_root / "information",
            log_dir=log_root / "information",
            scanner_steps=scanner_steps,
            scanner_epochs=scanner_epochs,
            scanner_batch_size=scanner_batch_size,
            scanner_horizon_steps=scanner_horizon_steps,
            scanner_window=scanner_window,
            scanner_sample_every=scanner_sample_every,
            scanner_embed=scanner_embed,
            radio_steps=radio_steps,
            radio_epochs=radio_epochs,
            radio_batch_size=radio_batch_size,
            jammer_steps=jammer_steps,
            jammer_epochs=jammer_epochs,
            jammer_batch_size=jammer_batch_size,
            jammer_mode=jammer_mode,
            jammer_sample_every=jammer_sample_every,
            jammer_horizon_steps=jammer_horizon_steps,
            jammer_positive_threshold=jammer_positive_threshold,
            hidden=hidden,
            lr=lr,
            device=device,
            league_seeds=league_seeds or [seed],
            league_steps=league_steps,
            promotion_threshold=promotion_threshold,
            improvement_margin=improvement_margin,
            promote=promote_information,
            compare_active=True,
            active_dir=active_info_dir,
        )

    promotion_decision = _control_promotion_decision(
        promote_control=promote_control,
        evaluate_control=evaluate_control,
        active_before=active_before,
        pool_before=pool_before,
        control_result=control_result,
        control_league=control_league,
        threshold=control_promotion_threshold,
        improvement_margin=control_improvement_margin,
        evader_threshold=evader_control_promotion_threshold,
        evader_margin=evader_control_improvement_margin,
        team_threshold=pursuer_team_control_promotion_threshold,
        team_margin=pursuer_team_control_improvement_margin,
        scenario_threshold=control_scenario_threshold,
        evader_scenario_threshold=evader_control_scenario_threshold,
        team_scenario_threshold=pursuer_team_control_scenario_threshold,
    )
    if promotion_decision["promoted"]:
        promote_control_checkpoints(
            evader=evader_path if promotion_decision.get("evader_promoted") else None,
            pursuer_team=team_path if promotion_decision.get("pursuer_team_promoted") else None,
            active_dir=active_control_dir,
            cycle_name=safe_name,
            seed=seed,
            promotion_decision=promotion_decision,
        )

    pool_entry: ControlPoolEntry | None = None
    active_after = load_active_control_set(active_control_dir)
    if use_control_pool and promotion_decision["promoted"] and active_after.evader and active_after.pursuer_team:
        pool_entry = archive_control_pair(
            name=safe_name,
            seed=seed,
            evader=active_after.evader,
            pursuer_team=active_after.pursuer_team,
            pool_dir=control_pool_dir,
            source_manifest=manifest_path,
            control_evaluation=(
                control_league.to_dict()
                if control_league is not None
                else control_result.to_dict() if control_result is not None else None
            ),
        )

    result = SelfPlayCycleResult(
        name=safe_name,
        seed=int(seed),
        checkpoints={
            "evader": str(evader_path),
            "pursuer_team": str(team_path),
        },
        training={
            "evader": evader_result.to_dict(),
            "pursuer_team": team_result.to_dict(),
        },
        control_opponents={
            "evader_training": evader_training_opponent.to_dict(),
            "pursuer_team_training": team_training_opponent.to_dict(),
        },
        active_control=active_after.to_dict(),
        control_evaluation=control_result.to_dict() if control_result is not None else None,
        control_league=control_league.to_dict() if control_league is not None else None,
        control_promotion_decision=promotion_decision,
        control_pool_entry=pool_entry.to_dict() if pool_entry is not None else None,
        information_cycle=info_result.to_dict() if info_result is not None else None,
        manifest=str(manifest_path),
    )
    manifest_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def load_active_control_set(active_dir: str | Path = "models/control_active") -> ControlCheckpointSet:
    manifest = Path(active_dir) / "manifest.json"
    if not manifest.exists():
        return ControlCheckpointSet()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    active = data.get("active_checkpoints", {})
    return ControlCheckpointSet(
        evader=active.get("evader"),
        pursuer_team=active.get("pursuer_team"),
        manifest=str(manifest),
    )


def promote_control_checkpoints(
    *,
    evader: str | Path | None,
    pursuer_team: str | Path | None,
    active_dir: str | Path = "models/control_active",
    cycle_name: str = "selfplay",
    seed: int = 11,
    promotion_decision: dict[str, Any] | None = None,
) -> Path:
    active = Path(active_dir)
    active.mkdir(parents=True, exist_ok=True)
    paths = {
        "evader": active / "evader_ppo.pt",
        "pursuer_team": active / "pursuer_team_ppo.pt",
    }
    if evader is not None:
        shutil.copy2(evader, paths["evader"])
    if pursuer_team is not None:
        shutil.copy2(pursuer_team, paths["pursuer_team"])
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        raise ValueError(f"cannot promote partial control pair; missing active checkpoints: {missing}")
    manifest = active / "manifest.json"
    payload = {
        "version": SELF_PLAY_VERSION,
        "cycle_name": cycle_name,
        "seed": int(seed),
        "active_checkpoints": {key: str(path) for key, path in paths.items()},
        "active_manifest": str(manifest),
        "promotion_decision": promotion_decision or {},
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name).strip())
    return safe or "selfplay"


def _select_control_opponent(
    *,
    use_control_pool: bool,
    pool_dir: str | Path,
    kind: str,
    seed: int,
    fallback: str | Path | None,
    fallback_label: str,
) -> ControlOpponentSelection:
    if not use_control_pool:
        if fallback is None:
            return ControlOpponentSelection(kind=kind, checkpoint=None, source="none")
        return ControlOpponentSelection(kind=kind, checkpoint=str(fallback), source="fallback", label=fallback_label)
    return select_control_opponent(
        pool_dir=pool_dir,
        kind=kind,
        seed=seed,
        fallback=fallback,
        fallback_label=fallback_label,
        mix_fallback=True,
    )


def _control_promotion_decision(
    *,
    promote_control: bool,
    evaluate_control: bool,
    active_before: ControlCheckpointSet,
    pool_before: list[ControlPoolEntry],
    control_result: ControlPairEvalResult | None,
    control_league: ControlCandidateLeagueResult | None,
    threshold: float,
    improvement_margin: float,
    evader_threshold: float | None = None,
    evader_margin: float | None = None,
    team_threshold: float | None = None,
    team_margin: float | None = None,
    scenario_threshold: float | None = None,
    evader_scenario_threshold: float | None = None,
    team_scenario_threshold: float | None = None,
) -> dict[str, Any]:
    base_threshold = float(threshold)
    base_margin = float(improvement_margin)
    evader_threshold_value = base_threshold if evader_threshold is None else float(evader_threshold)
    team_threshold_value = base_threshold if team_threshold is None else float(team_threshold)
    evader_margin_value = base_margin if evader_margin is None else float(evader_margin)
    team_margin_value = base_margin if team_margin is None else float(team_margin)
    scenario_threshold_value = None if scenario_threshold is None else float(scenario_threshold)
    evader_scenario_threshold_value = (
        scenario_threshold_value if evader_scenario_threshold is None else float(evader_scenario_threshold)
    )
    team_scenario_threshold_value = (
        scenario_threshold_value if team_scenario_threshold is None else float(team_scenario_threshold)
    )
    if not promote_control:
        return {
            "promoted": False,
            "evader_promoted": False,
            "pursuer_team_promoted": False,
            "reason": "control_promotion_disabled",
            "candidate_score": 0.0,
            "evader_candidate_score": 0.0,
            "pursuer_team_candidate_score": 0.0,
            "incumbent_score": None,
            "evader_incumbent_score": None,
            "pursuer_team_incumbent_score": None,
            "threshold": base_threshold,
            "improvement_margin": base_margin,
            "evader_threshold": evader_threshold_value,
            "evader_improvement_margin": evader_margin_value,
            "pursuer_team_threshold": team_threshold_value,
            "pursuer_team_improvement_margin": team_margin_value,
            "scenario_threshold": scenario_threshold_value,
            "scenario_acceptance_score": None,
            "evader_scenario_threshold": evader_scenario_threshold_value,
            "evader_scenario_acceptance_score": None,
            "pursuer_team_scenario_threshold": team_scenario_threshold_value,
            "pursuer_team_scenario_acceptance_score": None,
        }
    if not evaluate_control:
        return {
            "promoted": True,
            "evader_promoted": True,
            "pursuer_team_promoted": True,
            "reason": "control_evaluation_skipped",
            "candidate_score": None,
            "evader_candidate_score": None,
            "pursuer_team_candidate_score": None,
            "incumbent_score": _best_pool_score(pool_before),
            "evader_incumbent_score": _best_pool_metric(pool_before, "evader_generalization_score", "evader_control_score"),
            "pursuer_team_incumbent_score": _best_pool_metric(pool_before, "team_resilience_score", "pursuer_control_score"),
            "threshold": base_threshold,
            "improvement_margin": base_margin,
            "evader_threshold": evader_threshold_value,
            "evader_improvement_margin": evader_margin_value,
            "pursuer_team_threshold": team_threshold_value,
            "pursuer_team_improvement_margin": team_margin_value,
            "scenario_threshold": scenario_threshold_value,
            "scenario_acceptance_score": None,
            "evader_scenario_threshold": evader_scenario_threshold_value,
            "evader_scenario_acceptance_score": None,
            "pursuer_team_scenario_threshold": team_scenario_threshold_value,
            "pursuer_team_scenario_acceptance_score": None,
        }
    score_source = control_league.to_dict() if control_league is not None else (
        control_result.to_dict() if control_result is not None else {}
    )
    scores = score_source.get("scores", {})
    candidate_score = float(scores.get("overall_score", 0.0))
    evader_score = float(scores.get("evader_generalization_score", scores.get("evader_control_score", candidate_score)))
    team_score = float(scores.get("team_resilience_score", scores.get("pursuer_control_score", candidate_score)))
    scenario_evidence = _scenario_acceptance_evidence(score_source)
    scenario_score = scores.get("scenario_acceptance_score")
    scenario_score_value = None if scenario_score is None else float(scenario_score)
    evader_scenario_score = scores.get("scenario_evader_acceptance_score", scenario_score)
    team_scenario_score = scores.get("scenario_team_acceptance_score", scenario_score)
    evader_scenario_score_value = None if evader_scenario_score is None else float(evader_scenario_score)
    team_scenario_score_value = None if team_scenario_score is None else float(team_scenario_score)
    incumbent_score = _best_pool_score(pool_before)
    evader_incumbent = _best_pool_metric(pool_before, "evader_generalization_score", "evader_control_score")
    team_incumbent = _best_pool_metric(pool_before, "team_resilience_score", "pursuer_control_score")
    if scenario_evidence.get("scenario_acceptance_present") and not scenario_evidence.get("scenario_acceptance_passed"):
        return {
            "promoted": False,
            "evader_promoted": False,
            "pursuer_team_promoted": False,
            "reason": "scenario_acceptance_failed",
            "candidate_score": candidate_score,
            "evader_candidate_score": evader_score,
            "pursuer_team_candidate_score": team_score,
            "scenario_acceptance_score": scenario_score_value,
            "scenario_threshold": scenario_threshold_value,
            "incumbent_score": incumbent_score,
            "evader_scenario_threshold": evader_scenario_threshold_value,
            "evader_scenario_acceptance_score": evader_scenario_score_value,
            "pursuer_team_scenario_threshold": team_scenario_threshold_value,
            "pursuer_team_scenario_acceptance_score": team_scenario_score_value,
            "evader_incumbent_score": evader_incumbent,
            "pursuer_team_incumbent_score": team_incumbent,
            "threshold": base_threshold,
            "improvement_margin": base_margin,
            "evader_threshold": evader_threshold_value,
            "evader_improvement_margin": evader_margin_value,
            "pursuer_team_threshold": team_threshold_value,
            "pursuer_team_improvement_margin": team_margin_value,
            **scenario_evidence,
        }
    if scenario_threshold_value is not None and (
        scenario_score_value is None or scenario_score_value < scenario_threshold_value
    ):
        return {
            "promoted": False,
            "evader_promoted": False,
            "pursuer_team_promoted": False,
            "reason": "below_scenario_acceptance_threshold",
            "candidate_score": candidate_score,
            "evader_candidate_score": evader_score,
            "pursuer_team_candidate_score": team_score,
            "scenario_acceptance_score": scenario_score_value,
            "scenario_threshold": scenario_threshold_value,
            "incumbent_score": incumbent_score,
            "evader_scenario_threshold": evader_scenario_threshold_value,
            "evader_scenario_acceptance_score": evader_scenario_score_value,
            "pursuer_team_scenario_threshold": team_scenario_threshold_value,
            "pursuer_team_scenario_acceptance_score": team_scenario_score_value,
            "evader_incumbent_score": evader_incumbent,
            "pursuer_team_incumbent_score": team_incumbent,
            "threshold": base_threshold,
            "improvement_margin": base_margin,
            "evader_threshold": evader_threshold_value,
            "evader_improvement_margin": evader_margin_value,
            "pursuer_team_threshold": team_threshold_value,
            "pursuer_team_improvement_margin": team_margin_value,
            **scenario_evidence,
        }
    if not active_before.evader or not active_before.pursuer_team:
        evader_scenario_ready = _scenario_component_ready(evader_scenario_score_value, evader_scenario_threshold_value)
        team_scenario_ready = _scenario_component_ready(team_scenario_score_value, team_scenario_threshold_value)
        evader_ready = bool(active_before.evader or (evader_score >= evader_threshold_value and evader_scenario_ready))
        team_ready = bool(active_before.pursuer_team or (team_score >= team_threshold_value and team_scenario_ready))
        can_complete_pair = evader_ready and team_ready
        evader_promoted = bool(can_complete_pair and not active_before.evader)
        team_promoted = bool(can_complete_pair and not active_before.pursuer_team)
        promoted = evader_promoted or team_promoted
        if promoted:
            reason = "first_active_control_pair"
        elif not evader_scenario_ready or not team_scenario_ready:
            reason = "below_component_scenario_threshold"
        else:
            reason = "below_component_threshold"
        return {
            "promoted": promoted,
            "evader_promoted": evader_promoted,
            "pursuer_team_promoted": team_promoted,
            "reason": reason,
            "candidate_score": candidate_score,
            "evader_candidate_score": evader_score,
            "pursuer_team_candidate_score": team_score,
            "incumbent_score": incumbent_score,
            "evader_incumbent_score": evader_incumbent,
            "pursuer_team_incumbent_score": team_incumbent,
            "threshold": base_threshold,
            "improvement_margin": base_margin,
            "evader_threshold": evader_threshold_value,
            "evader_improvement_margin": evader_margin_value,
            "pursuer_team_threshold": team_threshold_value,
            "pursuer_team_improvement_margin": team_margin_value,
            "scenario_threshold": scenario_threshold_value,
            "scenario_acceptance_score": scenario_score_value,
            "evader_scenario_threshold": evader_scenario_threshold_value,
            "evader_scenario_acceptance_score": evader_scenario_score_value,
            "pursuer_team_scenario_threshold": team_scenario_threshold_value,
            "pursuer_team_scenario_acceptance_score": team_scenario_score_value,
            **scenario_evidence,
        }
    overall_required = max(base_threshold, (incumbent_score if incumbent_score is not None else 0.0) + base_margin)
    evader_required = max(evader_threshold_value, (evader_incumbent if evader_incumbent is not None else 0.0) + evader_margin_value)
    team_required = max(team_threshold_value, (team_incumbent if team_incumbent is not None else 0.0) + team_margin_value)
    evader_scenario_ready = _scenario_component_ready(evader_scenario_score_value, evader_scenario_threshold_value)
    team_scenario_ready = _scenario_component_ready(team_scenario_score_value, team_scenario_threshold_value)
    evader_promoted = evader_score >= evader_required and evader_scenario_ready
    team_promoted = team_score >= team_required and team_scenario_ready
    promoted = evader_promoted or team_promoted
    if evader_promoted and team_promoted:
        reason = "both_control_components_improved"
    elif evader_promoted:
        reason = "evader_improved_over_pool"
    elif team_promoted:
        reason = "pursuer_team_improved_over_pool"
    elif not evader_scenario_ready or not team_scenario_ready:
        reason = "below_component_scenario_threshold"
    else:
        reason = "no_control_component_improved"
    return {
        "promoted": promoted,
        "evader_promoted": evader_promoted,
        "pursuer_team_promoted": team_promoted,
        "reason": reason,
        "candidate_score": candidate_score,
        "evader_candidate_score": evader_score,
        "pursuer_team_candidate_score": team_score,
        "incumbent_score": incumbent_score,
        "evader_incumbent_score": evader_incumbent,
        "pursuer_team_incumbent_score": team_incumbent,
        "required_score": overall_required,
        "evader_required_score": evader_required,
        "pursuer_team_required_score": team_required,
        "threshold": base_threshold,
        "improvement_margin": base_margin,
        "evader_threshold": evader_threshold_value,
        "evader_improvement_margin": evader_margin_value,
        "pursuer_team_threshold": team_threshold_value,
        "pursuer_team_improvement_margin": team_margin_value,
        "scenario_threshold": scenario_threshold_value,
        "scenario_acceptance_score": scenario_score_value,
        "evader_scenario_threshold": evader_scenario_threshold_value,
        "evader_scenario_acceptance_score": evader_scenario_score_value,
        "pursuer_team_scenario_threshold": team_scenario_threshold_value,
        "pursuer_team_scenario_acceptance_score": team_scenario_score_value,
        **scenario_evidence,
    }


def _scenario_component_ready(score: float | None, threshold: float | None) -> bool:
    if threshold is None:
        return True
    return score is not None and score >= threshold


def _scenario_acceptance_evidence(score_source: dict[str, Any]) -> dict[str, Any]:
    scenario = score_source.get("scenario_acceptance")
    if not isinstance(scenario, dict):
        return {"scenario_acceptance_present": False}
    score = scenario.get("score") or {}
    return {
        "scenario_acceptance_present": True,
        "scenario_acceptance_passed": bool(scenario.get("passed")),
        "scenario_required_score": float(score.get("required_score", 0.0)),
        "scenario_milestone_score": float(score.get("milestone_score", 0.0)),
        "scenario_required_checks_passed": int(
            scenario.get("required_checks_passed", score.get("required_checks_passed", 0))
        ),
        "scenario_required_checks": int(
            scenario.get("required_checks", score.get("required_checks", 0))
        ),
        "scenario_milestone_targets_passed": int(
            scenario.get("milestone_targets_passed", score.get("milestone_targets_passed", 0))
        ),
        "scenario_milestone_targets": int(
            scenario.get("milestone_targets", score.get("milestone_targets", 0))
        ),
    }


def _best_pool_score(entries: list[ControlPoolEntry]) -> float | None:
    return _best_pool_metric(entries, "overall_score")


def _best_pool_metric(entries: list[ControlPoolEntry], *keys: str) -> float | None:
    scores = []
    for entry in entries:
        evaluation = entry.control_evaluation or {}
        row = evaluation.get("scores", {})
        score = None
        for key in keys:
            if key in row:
                score = row[key]
                break
        if score is not None:
            scores.append(float(score))
    if not scores:
        return None
    return max(scores)
