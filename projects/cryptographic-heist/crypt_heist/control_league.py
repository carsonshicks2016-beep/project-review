"""Replay-backed evaluation for learned evader and pursuer-team checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .control_pool import ControlPoolEntry, load_control_pool
from .policy_runtime import EvaderCheckpointController, PursuerTeamCheckpointController
from .replay import ReplayRecorder, load_replay, summarize_replay, validate_replay
from .scenarios import ScenarioThresholds, run_control_acceptance_suite
from .sim import HeistSim

CONTROL_LEAGUE_VERSION = 1


@dataclass
class ControlCheckpointPair:
    name: str
    evader: str
    pursuer_team: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "evader": self.evader,
            "pursuer_team": self.pursuer_team,
        }


@dataclass
class ControlPairEvalResult:
    checkpoint_pair: ControlCheckpointPair
    seeds: list[int]
    steps: int
    scores: dict[str, float]
    aggregates: dict[str, float]
    per_seed: list[dict[str, Any]]
    manifest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CONTROL_LEAGUE_VERSION,
            "checkpoint_pair": self.checkpoint_pair.to_dict(),
            "seeds": self.seeds,
            "steps": self.steps,
            "scores": self.scores,
            "aggregates": self.aggregates,
            "per_seed": self.per_seed,
            "manifest": self.manifest,
        }


@dataclass
class ControlCandidateLeagueResult:
    name: str
    candidate: ControlCheckpointPair
    seeds: list[int]
    steps: int
    pool_opponents: list[dict[str, Any]]
    self_pair: dict[str, Any]
    cross_play: list[dict[str, Any]]
    aggregates: dict[str, float]
    scores: dict[str, float]
    scenario_acceptance: dict[str, Any] | None = None
    manifest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CONTROL_LEAGUE_VERSION,
            "name": self.name,
            "candidate": self.candidate.to_dict(),
            "seeds": self.seeds,
            "steps": self.steps,
            "pool_opponents": self.pool_opponents,
            "self_pair": self.self_pair,
            "cross_play": self.cross_play,
            "aggregates": self.aggregates,
            "scores": self.scores,
            "scenario_acceptance": self.scenario_acceptance,
            "manifest": self.manifest,
        }


def evaluate_control_pair(
    *,
    name: str,
    evader: str | Path,
    pursuer_team: str | Path,
    seeds: list[int],
    steps: int = 1200,
    record_prefix: str | Path | None = "replays/control_pair",
    out: str | Path | None = "logs/control_pair_manifest.json",
) -> ControlPairEvalResult:
    """Run a learned evader and learned pursuer team together and score the chase."""
    if not seeds:
        raise ValueError("at least one seed is required")
    pair = ControlCheckpointPair(name=name, evader=str(evader), pursuer_team=str(pursuer_team))
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        record = _seed_record_path(record_prefix, name, int(seed))
        per_seed.append(run_control_pair_eval(
            evader_checkpoint=evader,
            pursuer_team_checkpoint=pursuer_team,
            seed=int(seed),
            steps=steps,
            record=record,
        ))

    aggregates = _aggregate_control_metrics(per_seed)
    scores = _score_control_aggregates(aggregates)
    result = ControlPairEvalResult(
        checkpoint_pair=pair,
        seeds=[int(seed) for seed in seeds],
        steps=int(steps),
        scores=scores,
        aggregates=aggregates,
        per_seed=per_seed,
    )
    if out is not None:
        result.manifest = str(write_control_manifest(result, out))
    return result


def evaluate_control_candidate_league(
    *,
    name: str,
    evader: str | Path,
    pursuer_team: str | Path,
    pool_dir: str | Path = "models/control_pool",
    seeds: list[int],
    steps: int = 600,
    max_opponents: int = 3,
    record_prefix: str | Path | None = "replays/control_league",
    out: str | Path | None = "logs/control_league_manifest.json",
    self_pair: dict[str, Any] | None = None,
    evaluate_scenarios: bool = False,
    scenario_seed: int = 31,
    scenario_steps: int | None = None,
    scenario_record_dir: str | Path | None = "replays/control_acceptance",
    scenario_out: str | Path | None = None,
    scenario_names: list[str] | tuple[str, ...] | None = None,
    scenario_thresholds: ScenarioThresholds | None = None,
) -> ControlCandidateLeagueResult:
    """Evaluate a candidate pair against recent historical control generations."""
    if not seeds:
        raise ValueError("at least one seed is required")
    candidate = ControlCheckpointPair(name=name, evader=str(evader), pursuer_team=str(pursuer_team))
    opponents = _selected_pool_entries(pool_dir, max_opponents=max_opponents)
    if self_pair is None:
        self_pair = evaluate_control_pair(
            name=f"{name}_self",
            evader=evader,
            pursuer_team=pursuer_team,
            seeds=seeds,
            steps=steps,
            record_prefix=_sub_prefix(record_prefix, "self"),
            out=None,
        ).to_dict()

    cross_play: list[dict[str, Any]] = []
    for entry in opponents:
        cross_play.append(_evaluate_cross_match(
            name=name,
            role="candidate_evader_vs_pool_team",
            opponent=entry,
            evader_checkpoint=evader,
            pursuer_team_checkpoint=entry.pursuer_team,
            seeds=seeds,
            steps=steps,
            record_prefix=record_prefix,
        ))
        cross_play.append(_evaluate_cross_match(
            name=name,
            role="pool_evader_vs_candidate_team",
            opponent=entry,
            evader_checkpoint=entry.evader,
            pursuer_team_checkpoint=pursuer_team,
            seeds=seeds,
            steps=steps,
            record_prefix=record_prefix,
        ))

    aggregates = _aggregate_candidate_league(self_pair, cross_play)
    scenario_acceptance = None
    if evaluate_scenarios:
        scenario_acceptance = run_control_acceptance_suite(
            evader_checkpoint=evader,
            pursuer_team_checkpoint=pursuer_team,
            scenarios=scenario_names,
            seed=scenario_seed,
            steps=scenario_steps,
            record_dir=scenario_record_dir,
            out=scenario_out,
            thresholds=scenario_thresholds,
        )
        aggregates["scenario_acceptance"] = float(scenario_acceptance["score"]["overall_score"])
        aggregates["scenario_required_acceptance"] = float(scenario_acceptance["score"]["required_score"])
        aggregates["scenario_milestone_acceptance"] = float(scenario_acceptance["score"]["milestone_score"])
        aggregates["scenario_evader_acceptance"] = float(scenario_acceptance["score"]["evader_control_score"])
        aggregates["scenario_team_acceptance"] = float(scenario_acceptance["score"]["pursuer_team_control_score"])
        aggregates["scenario_information_acceptance"] = float(scenario_acceptance["score"]["information_warfare_score"])
    scores = _score_candidate_league(aggregates)
    result = ControlCandidateLeagueResult(
        name=name,
        candidate=candidate,
        seeds=[int(seed) for seed in seeds],
        steps=int(steps),
        pool_opponents=[entry.to_dict() for entry in opponents],
        self_pair=self_pair,
        cross_play=cross_play,
        aggregates=aggregates,
        scores=scores,
        scenario_acceptance=scenario_acceptance,
    )
    if out is not None:
        result.manifest = str(write_control_candidate_league_manifest(result, out))
    return result


def run_control_pair_eval(
    *,
    evader_checkpoint: str | Path,
    pursuer_team_checkpoint: str | Path,
    seed: int = 11,
    steps: int = 1200,
    record: str | Path | None = None,
) -> dict[str, Any]:
    """Run one seed of learned-vs-learned control and optionally record a replay."""
    sim = HeistSim(seed=seed, reset_on_capture=True)
    evader = EvaderCheckpointController(evader_checkpoint)
    pursuers = PursuerTeamCheckpointController(pursuer_team_checkpoint)

    confidence_total = 0.0
    speed_total = 0.0
    max_impact = 0.0

    if record is not None:
        with ReplayRecorder(record, seed=seed, dt=sim.sim.dt) as recorder:
            for _ in range(int(steps)):
                _step_pair(sim, evader, pursuers)
                confidence_total += sim.channel.confidence
                speed_total += sim.evader.vehicle.speed
                max_impact = max(max_impact, sim.impact_energy)
                recorder.record(sim.snapshot())
    else:
        for _ in range(int(steps)):
            _step_pair(sim, evader, pursuers)
            confidence_total += sim.channel.confidence
            speed_total += sim.evader.vehicle.speed
            max_impact = max(max_impact, sim.impact_energy)

    metrics = {
        "captures": int(sim.captures),
        "waypoints_hit": int(sim.waypoints_hit),
        "deception_score": float(sim.deception_score),
        "avg_confidence": float(confidence_total / max(1, int(steps))),
        "avg_evader_speed": float(speed_total / max(1, int(steps))),
        "max_impact": float(max_impact),
        "final_jamming_budget": int(sim.channel.jamming_budget),
        "radio_events": int(len(sim.channel.events)),
        "spoofed_radio_events": int(sum(1 for event in sim.channel.events if event.spoofed)),
    }
    replay_summary = None
    if record is not None:
        replay = load_replay(record)
        validate_replay(replay)
        replay_summary = summarize_replay(replay)
        metrics.update({
            "radio_events": int(replay_summary["radio_events"]),
            "spoofed_radio_events": int(replay_summary["spoofed_radio_events"]),
            "unique_radio_events": int(replay_summary["unique_radio_events"]),
            "unique_spoofed_radio_events": int(replay_summary["unique_spoofed_radio_events"]),
            "max_impact": float(replay_summary["max_impact"]),
        })
    return {
        "seed": int(seed),
        "steps": int(steps),
        "record": str(record) if record is not None else None,
        "metrics": metrics,
        "replay_summary": replay_summary,
    }


def write_control_manifest(result: ControlPairEvalResult, out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_control_candidate_league_manifest(result: ControlCandidateLeagueResult, out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _step_pair(sim: HeistSim, evader: EvaderCheckpointController, pursuers: PursuerTeamCheckpointController):
    sim.step(actions=_merge_action_dicts(evader.action(sim), pursuers.action(sim)))


def _merge_action_dicts(*action_dicts: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for actions in action_dicts:
        if not actions:
            continue
        for agent, payload in actions.items():
            existing = merged.setdefault(agent, {})
            if isinstance(existing, dict) and isinstance(payload, dict):
                existing.update(dict(payload))
            else:
                merged[agent] = payload
    return merged


def _seed_record_path(record_prefix: str | Path | None, name: str, seed: int) -> Path | None:
    if record_prefix is None:
        return None
    prefix = Path(record_prefix)
    safe_name = _safe_name(name)
    return prefix.with_name(f"{prefix.name}_{safe_name}_seed{int(seed)}.jsonl")


def _aggregate_control_metrics(per_seed: list[dict[str, Any]]) -> dict[str, float]:
    rows = [row["metrics"] for row in per_seed]
    steps = [float(row["steps"]) for row in per_seed]
    mean_steps = _mean(steps)
    expected_events = max(1.0, mean_steps / 600.0)
    return {
        "mean_captures": _mean(row["captures"] for row in rows),
        "mean_waypoints": _mean(row["waypoints_hit"] for row in rows),
        "mean_deception": _mean(row["deception_score"] for row in rows),
        "mean_confidence": _mean(row["avg_confidence"] for row in rows),
        "mean_evader_speed": _mean(row["avg_evader_speed"] for row in rows),
        "mean_max_impact": _mean(row["max_impact"] for row in rows),
        "mean_radio_events": _mean(row["radio_events"] for row in rows),
        "mean_spoofed_radio_events": _mean(row["spoofed_radio_events"] for row in rows),
        "capture_pressure": float(np.clip(_mean(row["captures"] for row in rows) / expected_events, 0.0, 1.0)),
        "waypoint_pressure": float(np.clip(_mean(row["waypoints_hit"] for row in rows) / expected_events, 0.0, 1.0)),
    }


def _selected_pool_entries(pool_dir: str | Path, *, max_opponents: int) -> list[ControlPoolEntry]:
    entries = load_control_pool(pool_dir)
    if max_opponents <= 0:
        return []
    return entries[-int(max_opponents):]


def _evaluate_cross_match(
    *,
    name: str,
    role: str,
    opponent: ControlPoolEntry,
    evader_checkpoint: str | Path,
    pursuer_team_checkpoint: str | Path,
    seeds: list[int],
    steps: int,
    record_prefix: str | Path | None,
) -> dict[str, Any]:
    per_seed = []
    for seed in seeds:
        record = _cross_record_path(record_prefix, name, role, opponent.id, int(seed))
        per_seed.append(run_control_pair_eval(
            evader_checkpoint=evader_checkpoint,
            pursuer_team_checkpoint=pursuer_team_checkpoint,
            seed=int(seed),
            steps=int(steps),
            record=record,
        ))
    aggregates = _aggregate_control_metrics(per_seed)
    scores = _score_control_aggregates(aggregates)
    return {
        "role": role,
        "opponent": opponent.to_dict(),
        "seeds": [int(seed) for seed in seeds],
        "steps": int(steps),
        "aggregates": aggregates,
        "scores": scores,
        "per_seed": per_seed,
    }


def _aggregate_candidate_league(self_pair: dict[str, Any], cross_play: list[dict[str, Any]]) -> dict[str, float]:
    self_scores = self_pair.get("scores", {})
    candidate_evader_rows = [
        row["scores"]
        for row in cross_play
        if row.get("role") == "candidate_evader_vs_pool_team"
    ]
    candidate_team_rows = [
        row["scores"]
        for row in cross_play
        if row.get("role") == "pool_evader_vs_candidate_team"
    ]
    evader_generalization = _mean_score(candidate_evader_rows, "evader_control_score", self_scores.get("evader_control_score", 0.0))
    team_resilience = _mean_score(candidate_team_rows, "pursuer_control_score", self_scores.get("pursuer_control_score", 0.0))
    cross_balance = _mean_score(
        [row["scores"] for row in cross_play],
        "adversarial_balance_score",
        self_scores.get("adversarial_balance_score", 0.0),
    )
    self_overall = float(self_scores.get("overall_score", 0.0))
    self_spectacle = float(self_scores.get("spectacle_score", 0.0))
    worst_case = min(self_overall, evader_generalization, team_resilience)
    return {
        "self_overall": self_overall,
        "self_pursuer_control": float(self_scores.get("pursuer_control_score", 0.0)),
        "self_evader_control": float(self_scores.get("evader_control_score", 0.0)),
        "self_spectacle": self_spectacle,
        "evader_generalization": evader_generalization,
        "team_resilience": team_resilience,
        "cross_balance": cross_balance,
        "worst_case": float(worst_case),
        "pool_opponents": float(len({row['opponent']['id'] for row in cross_play})),
        "cross_matches": float(len(cross_play)),
    }


def _score_candidate_league(aggregates: dict[str, float]) -> dict[str, float]:
    self_overall = aggregates["self_overall"]
    evader_generalization = aggregates["evader_generalization"]
    team_resilience = aggregates["team_resilience"]
    cross_balance = aggregates["cross_balance"]
    worst_case = aggregates["worst_case"]
    pool_depth_score = float(np.clip(aggregates["pool_opponents"] / 3.0, 0.0, 1.0))
    base_overall = float(np.clip(
        0.30 * self_overall
        + 0.24 * evader_generalization
        + 0.24 * team_resilience
        + 0.14 * cross_balance
        + 0.08 * worst_case,
        0.0,
        1.0,
    ))
    scenario_score = aggregates.get("scenario_acceptance")
    overall = base_overall
    if scenario_score is not None:
        overall = float(np.clip(0.82 * base_overall + 0.18 * float(scenario_score), 0.0, 1.0))
    scores = {
        "overall_score": overall,
        "base_overall_score": base_overall,
        "self_pair_score": self_overall,
        "evader_generalization_score": float(np.clip(evader_generalization, 0.0, 1.0)),
        "team_resilience_score": float(np.clip(team_resilience, 0.0, 1.0)),
        "cross_balance_score": float(np.clip(cross_balance, 0.0, 1.0)),
        "worst_case_score": float(np.clip(worst_case, 0.0, 1.0)),
        "pool_depth_score": pool_depth_score,
    }
    if scenario_score is not None:
        scores["scenario_acceptance_score"] = float(np.clip(float(scenario_score), 0.0, 1.0))
        scores["scenario_required_acceptance_score"] = float(np.clip(
            aggregates.get("scenario_required_acceptance", 0.0), 0.0, 1.0
        ))
        scores["scenario_milestone_acceptance_score"] = float(np.clip(
            aggregates.get("scenario_milestone_acceptance", 0.0), 0.0, 1.0
        ))
        scores["scenario_evader_acceptance_score"] = float(np.clip(
            aggregates.get("scenario_evader_acceptance", 0.0), 0.0, 1.0
        ))
        scores["scenario_team_acceptance_score"] = float(np.clip(
            aggregates.get("scenario_team_acceptance", 0.0), 0.0, 1.0
        ))
        scores["scenario_information_acceptance_score"] = float(np.clip(
            aggregates.get("scenario_information_acceptance", 0.0), 0.0, 1.0
        ))
    return scores


def _score_control_aggregates(aggregates: dict[str, float]) -> dict[str, float]:
    capture_pressure = aggregates["capture_pressure"]
    waypoint_pressure = aggregates["waypoint_pressure"]
    speed_score = float(np.clip(aggregates["mean_evader_speed"] / 34.0, 0.0, 1.0))
    radio_score = float(np.clip(aggregates["mean_radio_events"] / 120.0, 0.0, 1.0))
    impact_score = float(np.clip(aggregates["mean_max_impact"] / 14.0, 0.0, 1.0))
    deception_score = float(np.clip(aggregates["mean_deception"] / 4.0, 0.0, 1.0))

    pursuer_control = float(np.clip(
        0.72 * capture_pressure + 0.18 * (1.0 - waypoint_pressure) + 0.10 * radio_score,
        0.0,
        1.0,
    ))
    evader_control = float(np.clip(
        0.58 * waypoint_pressure + 0.24 * speed_score + 0.18 * deception_score,
        0.0,
        1.0,
    ))
    spectacle = float(np.clip(0.50 * speed_score + 0.25 * impact_score + 0.25 * radio_score, 0.0, 1.0))
    balance = float(np.clip(1.0 - abs(pursuer_control - evader_control), 0.0, 1.0))
    overall = float(np.clip(
        0.32 * pursuer_control + 0.32 * evader_control + 0.24 * spectacle + 0.12 * balance,
        0.0,
        1.0,
    ))
    return {
        "overall_score": overall,
        "pursuer_control_score": pursuer_control,
        "evader_control_score": evader_control,
        "spectacle_score": spectacle,
        "adversarial_balance_score": balance,
    }


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name).strip())
    return safe or "control_pair"


def _sub_prefix(record_prefix: str | Path | None, suffix: str) -> str | Path | None:
    if record_prefix is None:
        return None
    prefix = Path(record_prefix)
    return prefix.with_name(f"{prefix.name}_{suffix}")


def _cross_record_path(
    record_prefix: str | Path | None,
    name: str,
    role: str,
    opponent_id: str,
    seed: int,
) -> Path | None:
    if record_prefix is None:
        return None
    prefix = Path(record_prefix)
    safe_name = _safe_name(name)
    safe_role = _safe_name(role)
    safe_opp = _safe_name(opponent_id)
    return prefix.with_name(f"{prefix.name}_{safe_name}_{safe_role}_{safe_opp}_seed{int(seed)}.jsonl")


def _mean_score(rows: list[dict[str, float]], key: str, fallback: float) -> float:
    if not rows:
        return float(fallback)
    return _mean(row.get(key, fallback) for row in rows)


def _mean(values) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / max(1, len(values)))
