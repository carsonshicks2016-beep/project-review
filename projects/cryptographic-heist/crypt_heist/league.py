"""Checkpoint league scoring and promotion gates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from .adversarial import run_authentication_curriculum_eval

LEAGUE_VERSION = 1


@dataclass
class CheckpointSet:
    name: str
    radio: str
    scanner: str
    jammer: str

    def to_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "radio": self.radio,
            "scanner": self.scanner,
            "jammer": self.jammer,
        }


@dataclass
class LeagueEvalResult:
    checkpoint_set: CheckpointSet
    seeds: list[int]
    steps: int
    scores: dict[str, float]
    aggregates: dict[str, float]
    per_seed: list[dict[str, Any]]
    promoted: bool
    active_manifest: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": LEAGUE_VERSION,
            "checkpoint_set": self.checkpoint_set.to_dict(),
            "seeds": self.seeds,
            "steps": self.steps,
            "scores": self.scores,
            "aggregates": self.aggregates,
            "per_seed": self.per_seed,
            "promoted": self.promoted,
            "active_manifest": self.active_manifest,
        }


def parse_seed_list(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def evaluate_checkpoint_set(
    *,
    name: str,
    radio: str | Path,
    scanner: str | Path,
    jammer: str | Path,
    seeds: list[int],
    steps: int = 1200,
    record_prefix: str | Path | None = "logs/checkpoint_league",
    out: str | Path | None = "logs/checkpoint_league_manifest.json",
    promotion_threshold: float = 0.52,
    promote: bool = False,
    active_dir: str | Path = "models/active",
) -> LeagueEvalResult:
    """Evaluate and optionally promote one radio/scanner/jammer checkpoint set."""
    checkpoint_set = CheckpointSet(
        name=name,
        radio=str(radio),
        scanner=str(scanner),
        jammer=str(jammer),
    )
    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        prefix = _seed_record_prefix(record_prefix, name, seed)
        result = run_authentication_curriculum_eval(
            radio_checkpoint=radio,
            scanner_checkpoint=scanner,
            jammer_checkpoint=jammer,
            seed=int(seed),
            steps=int(steps),
            record_prefix=prefix,
        )
        per_seed.append(result.to_dict())

    aggregates = _aggregate_metrics(per_seed)
    scores = _score_aggregates(aggregates)
    promoted = bool(promote and scores["overall_score"] >= float(promotion_threshold))
    active_manifest = None
    league_result = LeagueEvalResult(
        checkpoint_set=checkpoint_set,
        seeds=[int(seed) for seed in seeds],
        steps=int(steps),
        scores=scores,
        aggregates=aggregates,
        per_seed=per_seed,
        promoted=promoted,
    )
    if promoted:
        active_manifest = str(promote_checkpoint_set(league_result, active_dir=active_dir))
        league_result.active_manifest = active_manifest
    if out is not None:
        write_league_manifest(league_result, out)
    return league_result


def promote_checkpoint_set(result: LeagueEvalResult, active_dir: str | Path = "models/active") -> Path:
    """Copy a promoted checkpoint trio into an active model directory."""
    active = Path(active_dir)
    active.mkdir(parents=True, exist_ok=True)
    paths = {
        "radio": active / "radio_policy.pt",
        "scanner": active / "scanner_decoder.pt",
        "jammer": active / "jammer_policy.pt",
    }
    source = result.checkpoint_set
    shutil.copy2(source.radio, paths["radio"])
    shutil.copy2(source.scanner, paths["scanner"])
    shutil.copy2(source.jammer, paths["jammer"])
    manifest = active / "manifest.json"
    payload = result.to_dict()
    payload["active_checkpoints"] = {key: str(path) for key, path in paths.items()}
    payload["active_manifest"] = str(manifest)
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def load_active_checkpoint_set(active_dir: str | Path = "models/active") -> CheckpointSet | None:
    """Load the current active radio/scanner/jammer stack, if one exists."""
    manifest = Path(active_dir) / "manifest.json"
    if not manifest.exists():
        return None
    data = json.loads(manifest.read_text(encoding="utf-8"))
    active = data.get("active_checkpoints", {})
    radio = active.get("radio")
    scanner = active.get("scanner")
    jammer = active.get("jammer")
    if not (radio and scanner and jammer):
        raise ValueError(f"active manifest is missing checkpoint paths: {manifest}")
    name = data.get("checkpoint_set", {}).get("name", "active")
    return CheckpointSet(name=f"active_{name}", radio=str(radio), scanner=str(scanner), jammer=str(jammer))


def write_league_manifest(result: LeagueEvalResult, out: str | Path) -> Path:
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _seed_record_prefix(record_prefix: str | Path | None, name: str, seed: int) -> str | Path | None:
    if record_prefix is None:
        return None
    prefix = Path(record_prefix)
    safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
    return prefix.with_name(f"{prefix.name}_{safe_name}_seed{int(seed)}")


def _aggregate_metrics(per_seed: list[dict[str, Any]]) -> dict[str, float]:
    rows = [row["metrics"] for row in per_seed]
    jammed = [row["jammed"]["metrics"] for row in per_seed]
    baseline = [row["baseline"]["metrics"] for row in per_seed]
    return {
        "mean_spoof_susceptibility": _mean(row["spoof_susceptibility_counterfactual"] for row in rows),
        "mean_confidence_damage": _mean(row["confidence_damage"] for row in rows),
        "mean_final_confidence_damage": _mean(row["final_confidence_damage"] for row in rows),
        "mean_decoder_error_delta": _mean(row["decoder_error_delta"] for row in rows),
        "mean_deception_lift": _mean(row["deception_lift"] for row in rows),
        "mean_spoof_event_lift": _mean(row["spoof_event_lift"] for row in rows),
        "mean_pursuer_deviation": _mean(row["trajectory"].get("mean_pursuer_deviation", 0.0) for row in rows),
        "mean_pursuer_auth_penalty": _mean(row["pursuer_auth_penalty_proxy"] for row in rows),
        "mean_evader_information_reward": _mean(row["evader_information_reward_proxy"] for row in rows),
        "mean_jammed_decoder_accuracy": _mean(row["decoder_accuracy_proxy"] for row in jammed),
        "mean_baseline_decoder_accuracy": _mean(row["decoder_accuracy_proxy"] for row in baseline),
        "mean_jammed_confidence": _mean(row["avg_confidence"] for row in jammed),
        "mean_baseline_confidence": _mean(row["avg_confidence"] for row in baseline),
    }


def _score_aggregates(aggregates: dict[str, float]) -> dict[str, float]:
    pursuer_security = float(np.clip(
        1.0
        - 0.45 * aggregates["mean_spoof_susceptibility"]
        - 0.35 * aggregates["mean_jammed_decoder_accuracy"]
        - 0.20 * aggregates["mean_confidence_damage"],
        0.0,
        1.0,
    ))
    evader_pressure = float(np.clip(
        0.45 * min(1.0, aggregates["mean_deception_lift"])
        + 0.25 * min(1.0, aggregates["mean_decoder_error_delta"] / 120.0)
        + 0.20 * min(1.0, aggregates["mean_confidence_damage"])
        + 0.10 * min(1.0, aggregates["mean_spoof_event_lift"] / 5.0),
        0.0,
        1.0,
    ))
    balance = float(np.clip(1.0 - abs(pursuer_security - evader_pressure), 0.0, 1.0))
    overall = float(np.clip(
        0.42 * pursuer_security
        + 0.38 * evader_pressure
        + 0.20 * balance,
        0.0,
        1.0,
    ))
    return {
        "overall_score": overall,
        "pursuer_security_score": pursuer_security,
        "evader_pressure_score": evader_pressure,
        "adversarial_balance_score": balance,
    }


def _mean(values) -> float:
    values = [float(value) for value in values]
    return float(sum(values) / max(1, len(values)))
