"""Automated train-evaluate-promote cycles for the information-warfare stack."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .jamming import train_counterfactual_jammer_policy, train_jammer_policy
from .league import (
    LeagueEvalResult,
    evaluate_checkpoint_set,
    load_active_checkpoint_set,
    promote_checkpoint_set,
    write_league_manifest,
)
from .radio import train_radio_policy
from .scanner import DEFAULT_HORIZON_STEPS, DEFAULT_WINDOW, train_scanner_decoder

CYCLE_VERSION = 1


@dataclass
class InformationCycleResult:
    name: str
    seed: int
    checkpoints: dict[str, str]
    training: dict[str, Any]
    league: LeagueEvalResult
    manifest: str
    incumbent: dict[str, Any] | None = None
    promotion_decision: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CYCLE_VERSION,
            "name": self.name,
            "seed": self.seed,
            "checkpoints": self.checkpoints,
            "training": self.training,
            "league": self.league.to_dict(),
            "manifest": self.manifest,
            "incumbent": self.incumbent,
            "promotion_decision": self.promotion_decision,
        }


def run_information_cycle(
    *,
    name: str = "candidate",
    seed: int = 11,
    checkpoint_dir: str | Path = "checkpoints/cycles",
    log_dir: str | Path = "logs/cycles",
    scanner_steps: int = 2400,
    scanner_epochs: int = 8,
    scanner_batch_size: int = 256,
    scanner_horizon_steps: int = DEFAULT_HORIZON_STEPS,
    scanner_window: int = DEFAULT_WINDOW,
    scanner_sample_every: int = 4,
    scanner_embed: int = 32,
    radio_steps: int = 3600,
    radio_epochs: int = 12,
    radio_batch_size: int = 128,
    jammer_steps: int = 3600,
    jammer_epochs: int = 12,
    jammer_batch_size: int = 128,
    jammer_mode: str = "heuristic",
    jammer_sample_every: int = 12,
    jammer_horizon_steps: int = 72,
    jammer_positive_threshold: float = 0.16,
    hidden: int = 128,
    lr: float = 3e-4,
    device: str = "cpu",
    league_seeds: list[int] | None = None,
    league_steps: int = 1200,
    promotion_threshold: float = 0.52,
    improvement_margin: float = 0.01,
    promote: bool = False,
    compare_active: bool = True,
    active_dir: str | Path = "models/active",
) -> InformationCycleResult:
    """Train scanner/radio/jammer checkpoints, score them, and optionally promote."""
    league_seeds = [int(seed)] if league_seeds is None else [int(value) for value in league_seeds]
    safe_name = _safe_name(name)
    checkpoint_root = Path(checkpoint_dir)
    log_root = Path(log_dir)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    scanner_path = checkpoint_root / f"{safe_name}_scanner_decoder.pt"
    radio_path = checkpoint_root / f"{safe_name}_radio_policy.pt"
    jammer_path = checkpoint_root / f"{safe_name}_jammer_policy.pt"

    scanner_result = train_scanner_decoder(
        scanner_path,
        seed=seed,
        steps=scanner_steps,
        horizon_steps=scanner_horizon_steps,
        window=scanner_window,
        sample_every=scanner_sample_every,
        epochs=scanner_epochs,
        batch_size=scanner_batch_size,
        lr=lr,
        embed=scanner_embed,
        hidden=hidden,
        device=device,
    )
    radio_result = train_radio_policy(
        radio_path,
        seed=seed,
        steps=radio_steps,
        epochs=radio_epochs,
        batch_size=radio_batch_size,
        lr=lr,
        hidden=hidden,
        device=device,
    )
    if jammer_mode == "counterfactual":
        jammer_result = train_counterfactual_jammer_policy(
            jammer_path,
            seed=seed,
            steps=jammer_steps,
            sample_every=jammer_sample_every,
            horizon_steps=jammer_horizon_steps,
            positive_threshold=jammer_positive_threshold,
            epochs=jammer_epochs,
            batch_size=jammer_batch_size,
            lr=lr,
            hidden=hidden,
            device=device,
        )
    elif jammer_mode == "heuristic":
        jammer_result = train_jammer_policy(
            jammer_path,
            seed=seed,
            steps=jammer_steps,
            epochs=jammer_epochs,
            batch_size=jammer_batch_size,
            lr=lr,
            hidden=hidden,
            device=device,
        )
    else:
        raise ValueError(f"jammer_mode must be 'heuristic' or 'counterfactual', got {jammer_mode!r}")

    league = evaluate_checkpoint_set(
        name=safe_name,
        radio=radio_path,
        scanner=scanner_path,
        jammer=jammer_path,
        seeds=league_seeds,
        steps=league_steps,
        record_prefix=log_root / f"{safe_name}_league",
        out=None,
        promotion_threshold=promotion_threshold,
        promote=False,
        active_dir=active_dir,
    )
    incumbent_result = None
    incumbent_set = load_active_checkpoint_set(active_dir) if compare_active else None
    if incumbent_set is not None:
        incumbent_result = evaluate_checkpoint_set(
            name=incumbent_set.name,
            radio=incumbent_set.radio,
            scanner=incumbent_set.scanner,
            jammer=incumbent_set.jammer,
            seeds=league_seeds,
            steps=league_steps,
            record_prefix=log_root / f"{safe_name}_incumbent_league",
            out=log_root / f"{safe_name}_incumbent_league_manifest.json",
            promotion_threshold=promotion_threshold,
            promote=False,
            active_dir=active_dir,
        )

    decision = _promotion_decision(
        candidate=league,
        incumbent=incumbent_result,
        promote=promote,
        promotion_threshold=promotion_threshold,
        improvement_margin=improvement_margin,
        compare_active=compare_active,
    )
    league.promoted = bool(decision["promoted"])
    if league.promoted:
        league.active_manifest = str(promote_checkpoint_set(league, active_dir=active_dir))
    write_league_manifest(league, log_root / f"{safe_name}_league_manifest.json")

    manifest_path = log_root / f"{safe_name}_cycle_manifest.json"
    result = InformationCycleResult(
        name=safe_name,
        seed=int(seed),
        checkpoints={
            "scanner": str(scanner_path),
            "radio": str(radio_path),
            "jammer": str(jammer_path),
        },
        training={
            "scanner": scanner_result.to_dict(),
            "radio": radio_result.to_dict(),
            "jammer": jammer_result.to_dict(),
        },
        league=league,
        manifest=str(manifest_path),
        incumbent=incumbent_result.to_dict() if incumbent_result is not None else None,
        promotion_decision=decision,
    )
    manifest_path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name).strip())
    return safe or "candidate"


def _promotion_decision(
    *,
    candidate: LeagueEvalResult,
    incumbent: LeagueEvalResult | None,
    promote: bool,
    promotion_threshold: float,
    improvement_margin: float,
    compare_active: bool,
) -> dict[str, Any]:
    candidate_score = float(candidate.scores["overall_score"])
    incumbent_score = None if incumbent is None else float(incumbent.scores["overall_score"])
    improvement = None if incumbent_score is None else candidate_score - incumbent_score
    threshold_passed = candidate_score >= float(promotion_threshold)
    improvement_passed = True if incumbent_score is None else improvement >= float(improvement_margin)
    should_promote = bool(promote and threshold_passed and improvement_passed)
    if not promote:
        reason = "promotion_disabled"
    elif not threshold_passed:
        reason = "below_threshold"
    elif incumbent_score is None and compare_active:
        reason = "first_active_stack"
    elif incumbent_score is None:
        reason = "no_incumbent_comparison"
    elif improvement_passed:
        reason = "improved_over_incumbent"
    else:
        reason = "did_not_improve_over_incumbent"
    return {
        "promoted": should_promote,
        "reason": reason,
        "compare_active": bool(compare_active),
        "incumbent_found": incumbent_score is not None,
        "candidate_score": candidate_score,
        "incumbent_score": incumbent_score,
        "improvement": improvement,
        "promotion_threshold": float(promotion_threshold),
        "threshold_passed": threshold_passed,
        "improvement_margin": float(improvement_margin),
        "improvement_passed": improvement_passed,
    }
