#!/usr/bin/env python3
"""Hermetic gates for Fable edition identity, resolution, and repair."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import supra.fable5 as f5  # noqa: E402
from supra.fable_editions import (  # noqa: E402
    CheckpointRejected,
    FABLE_EDITIONS,
    UnsafeCheckpointPath,
    catalog_payload,
    classify_checkpoint,
    load_edition_manifest,
    registry_payload,
    repair_manifest_auto_blocks,
    require_playable_checkpoint,
    resolve_checkpoint,
)
from supra.ppo import ActorCritic, state_dict_hash  # noqa: E402


FAILED: list[str] = []


def gate(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}"
          + (f" ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def _write_checkpoint(root: Path, name: str, edition_id: str, *,
                      current: bool = True, stage: str = "foundation",
                      lap_time: float | None = None,
                      progress: float = 0.5,
                      car: str | None = None,
                      track: str = "nordschleife",
                      fable_pipeline: bool = True,
                      obs_dim: int | None = None) -> Path:
    edition = FABLE_EDITIONS[edition_id]
    actual_obs = edition.obs_dim if obs_dim is None else obs_dim
    sdim = actual_obs - 2
    net = ActorCritic(actual_obs, edition.act_dim, (128, 128))
    policy_hash = state_dict_hash(net.state_dict())
    evaluation = {
        "stage": stage,
        "metric": 100.0 if lap_time else progress,
        "lap_time": lap_time,
        "progress_frac": progress,
        "clean_chain": 16 if lap_time else 8,
        "clean_sectors": 16 if lap_time else 8,
        "sector_count": 16,
        "terminal_rate": 0.0 if lap_time else 0.25,
        "pace_ratio": 1.0,
        "evaluated_policy_sha256": policy_hash,
    }
    payload = {
        "state_dict": net.state_dict(),
        "fable_pipeline": fable_pipeline,
        "car": edition.car_id if car is None else car,
        "track": track,
        "track_profile": "nordschleife-full-20.832km",
        "mode": "race",
        "fable_stage": stage,
        "fable_drivetrain_version": edition.drivetrain,
        "obs_layout": edition.observation_layout,
        "sensor_pace_block": True,
        "sensor_hybrid_block": edition.hybrid_observations,
        "obs_dim": actual_obs,
        "sdim": sdim,
        "act_dim": edition.act_dim,
        "hidden": [128, 128],
        "norm_mean": np.zeros(sdim, dtype=np.float64),
        "norm_var": np.ones(sdim, dtype=np.float64),
        "norm_count": 10.0,
        "policy_sha256": policy_hash,
        "fable_eval_protocol": f5.EVAL_PROTOCOL_VERSION,
        "fable_code_fingerprint": (f5.CODE_FINGERPRINT if current
                                    else "0" * 64),
        "fable_eval": evaluation,
        "checkpoint_time_unix": 1_700_000_000.0,
    }
    path = root / name
    torch.save(payload, path)
    return path


def registry_contract() -> None:
    print("== registry contract ==")
    payload = registry_payload()
    gate("exact editions", [item["id"] for item in payload] == ["787b", "919"])
    required = {
        "id", "car_id", "label", "manifest", "canonical_champion",
        "checkpoint_prefix", "drivetrain", "observation_layout", "obs_dim",
        "hybrid_observations", "benchmark", "compatibility_class",
        "authority_warning",
    }
    gate("dashboard descriptor fields",
         all(required <= set(item) for item in payload))
    gate("787B contract",
         payload[0]["car_id"] == "mazda787b"
         and payload[0]["observation_layout"] == "fable-v1"
         and payload[0]["hybrid_observations"] is False
         and payload[0]["obs_dim"] == 68)
    gate("919 contract and authority boundary",
         payload[1]["car_id"] == "porsche_919evo"
         and payload[1]["observation_layout"] == "fable-v2"
         and payload[1]["hybrid_observations"] is True
         and payload[1]["obs_dim"] == 70
         and "not faithful-v2" in payload[1]["authority_warning"])


def checkpoint_contracts(root: Path) -> None:
    print("== checkpoint safety and classification ==")
    current_787 = _write_checkpoint(root, "fable5_current_best.pt", "787b",
                                    current=True, stage="fast", lap_time=360.0)
    historical_787 = _write_checkpoint(root, "fable5_historical_best.pt", "787b",
                                       current=False, stage="finish", lap_time=500.0)
    current_919 = _write_checkpoint(root, "fable5_919_current_best.pt", "919",
                                    current=True, progress=0.8)

    current = require_playable_checkpoint(root, "787b", current_787.name)
    historical = require_playable_checkpoint(root, "787b", historical_787.name)
    gate("current checkpoint exact hashes",
         current.classification == "current"
         and len(current.file_sha256 or "") == 64
         and current.policy_sha256 == current.stored_policy_sha256)
    gate("historical checkpoint permanently badged",
         historical.classification == "historical"
         and any("Historical Fable policy" in w for w in historical.warnings))
    gate("919 warning cannot disappear",
         any("not faithful-v2" in w
             for w in require_playable_checkpoint(root, "919", current_919.name).warnings))

    cross = classify_checkpoint(root, "787b", current_919.name)
    gate("cross-car checkpoint rejected",
         not cross.playable and "checkpoint car" in (cross.rejection or ""))
    wrong_track = _write_checkpoint(root, "fable5_wrong_track.pt", "787b",
                                    track="club")
    non_fable = _write_checkpoint(root, "fable5_non_fable.pt", "787b",
                                  fable_pipeline=False)
    wrong_obs = _write_checkpoint(root, "fable5_wrong_obs.pt", "787b", obs_dim=70)
    corrupt = root / "fable5_corrupt.pt"
    corrupt.write_bytes(b"not a torch checkpoint")
    gate("wrong-track checkpoint rejected",
         not classify_checkpoint(root, "787b", wrong_track.name).playable)
    gate("non-Fable checkpoint rejected",
         not classify_checkpoint(root, "787b", non_fable.name).playable)
    gate("incompatible observation rejected",
         not classify_checkpoint(root, "787b", wrong_obs.name).playable)
    gate("corrupt checkpoint rejected closed",
         not classify_checkpoint(root, "787b", corrupt.name).playable)
    catalog = catalog_payload(root)
    gate("catalog is edition-scoped and directly JSON serializable",
         [item["id"] for item in catalog["editions"]] == ["787b", "919"]
         and catalog["editions"][0]["resolved_checkpoint"]["edition"] == "787b"
         and catalog["editions"][1]["resolved_checkpoint"]["edition"] == "919"
         and bool(json.dumps(catalog)))

    for unsafe in ("../escape.pt", "/tmp/escape.pt", "folder/brain.pt",
                   "folder\\brain.pt", "brain.bin"):
        try:
            classify_checkpoint(root, "787b", unsafe)
        except UnsafeCheckpointPath:
            passed = True
        else:
            passed = False
        gate(f"unsafe path rejected: {unsafe}", passed)


def resolution_order() -> None:
    print("== resolver priority ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        historical = _write_checkpoint(
            root, "fable5_ring_history_best.pt", "787b", current=False,
            stage="finish", lap_time=500.0,
        )
        active = _write_checkpoint(
            root, "fable5_ring_active_best.pt", "787b", current=True,
            stage="fast", lap_time=420.0,
        )
        champion = _write_checkpoint(
            root, FABLE_EDITIONS["787b"].canonical_champion, "787b",
            current=True, stage="flow", lap_time=440.0,
        )
        ranked = _write_checkpoint(
            root, "fable5_ring_frontier_best.pt", "787b", current=True,
            stage="frontier", lap_time=410.0,
        )
        manifest = {"car": "mazda787b", "track": "nordschleife",
                    "current_stage": "fast",
                    "active_best_checkpoint": active.name}

        resolved = resolve_checkpoint(
            root, "787b", explicit_selection=historical.name, manifest=manifest
        )
        gate("explicit selection first",
             resolved.checkpoint.name == historical.name
             and resolved.reason == "explicit selection")
        resolved = resolve_checkpoint(root, "787b", manifest=manifest)
        gate("active-stage best before champion",
             resolved.checkpoint.name == active.name
             and resolved.reason == "valid active-stage best")

        resolved = resolve_checkpoint(root, "787b", manifest={})
        gate("canonical champion before ranked stage",
             resolved.checkpoint.name == champion.name
             and resolved.reason == "valid canonical champion")
        champion.unlink()
        resolved = resolve_checkpoint(root, "787b", manifest={})
        gate("current-protocol ranked stage best",
             resolved.checkpoint.name == ranked.name
             and resolved.reason == "current-protocol ranked stage best")
        active.unlink()
        ranked.unlink()
        resolved = resolve_checkpoint(root, "787b", manifest={})
        gate("historical fallback last and amber",
             resolved.checkpoint.name == historical.name
             and resolved.reason == "playable historical fallback"
             and resolved.checkpoint.classification == "historical")

        try:
            resolve_checkpoint(root, "787b", explicit_selection="../escape.pt")
        except CheckpointRejected:
            passed = True
        else:
            passed = False
        gate("explicit traversal fails closed", passed)


def manifest_repair() -> None:
    print("== manifest repair ==")
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        checkpoint = _write_checkpoint(
            root, "fable5_919_ring_foundation.pt", "919", current=True
        )
        auto = {
            "prefix": "fable5_919_ring",
            "inflight": {"stage": "foundation",
                         "checkpoint": checkpoint.name},
        }
        source_path = root / FABLE_EDITIONS["787b"].manifest
        destination_path = root / FABLE_EDITIONS["919"].manifest
        source_path.write_text(json.dumps({"auto": auto}, indent=2) + "\n")
        destination_path.write_text(json.dumps({"car": "porsche_919evo",
                                                "auto": None}, indent=2) + "\n")
        source_before = source_path.read_bytes()
        destination_before = destination_path.read_bytes()

        scoped_source = load_edition_manifest(root, "787b")
        gate("edition state hides a leaked cross-car auto block",
             scoped_source.get("auto") is None
             and bool(scoped_source.get("_identity_warnings")))

        dry = repair_manifest_auto_blocks(root, dry_run=True)
        gate("dry run reports without mutation",
             dry["would_change"] and not dry["changed"]
             and source_path.read_bytes() == source_before
             and destination_path.read_bytes() == destination_before
             and not (root / "runtime").exists())

        report = repair_manifest_auto_blocks(
            root, timestamp="20260718-000000"
        )
        repaired_source = json.loads(source_path.read_text())
        repaired_destination = json.loads(destination_path.read_text())
        backup = root / str(report["backup_dir"])
        gate("proven cross-car auto block moved",
             report["changed"] and repaired_source["auto"] is None
             and repaired_destination["auto"] == auto)
        gate("repaired auto state is visible only in its destination edition",
             load_edition_manifest(root, "787b").get("auto") is None
             and load_edition_manifest(root, "919").get("auto") == auto)
        gate("both affected manifests backed up exactly",
             (backup / source_path.name).read_bytes() == source_before
             and (backup / destination_path.name).read_bytes() == destination_before)
        second = repair_manifest_auto_blocks(root)
        gate("repair is idempotent",
             not second["changed"] and not second["would_change"]
             and len(list((root / "runtime/manifest-repair").iterdir())) == 1)

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        source_path = root / FABLE_EDITIONS["787b"].manifest
        destination_path = root / FABLE_EDITIONS["919"].manifest
        source_path.write_text(json.dumps({"auto": {"prefix": "unknown"}}) + "\n")
        destination_path.write_text(json.dumps({"auto": None}) + "\n")
        before = source_path.read_bytes()
        ambiguous = repair_manifest_auto_blocks(root)
        gate("ambiguous state remains byte-for-byte untouched",
             not ambiguous["changed"] and ambiguous["warnings"]
             and source_path.read_bytes() == before
             and not (root / "runtime").exists())


def auto_manifest_routing() -> None:
    print("== auto-ladder edition routing ==")

    def fake_stage(_stage: str, _iterations: int, **kwargs):
        callback = kwargs.get("on_stage_ready")
        if callback:
            callback("fable5_919_ring_foundation.pt", 0)
        return {"iters_used": 1, "metric": 0.0}

    with tempfile.TemporaryDirectory() as td:
        previous = Path.cwd()
        os.chdir(td)
        try:
            with patch.object(f5, "_train_one_stage", side_effect=fake_stage):
                f5.run_fable(
                    1,
                    stage="auto",
                    car="porsche_919evo",
                    out="fable5_919_ring.pt",
                )
            p919 = Path(FABLE_EDITIONS["919"].manifest)
            p787 = Path(FABLE_EDITIONS["787b"].manifest)
            state = json.loads(p919.read_text())
            mazda_manifest_created = p787.exists()
        finally:
            os.chdir(previous)
        gate("919 auto state writes only to the 919 manifest",
             state["auto"]["prefix"] == "fable5_919_ring"
             and not mazda_manifest_created)


def event_identity_stamp() -> None:
    print("== runtime event identity ==")
    with tempfile.TemporaryDirectory() as td:
        previous = Path.cwd()
        os.chdir(td)
        try:
            evaluator = object.__new__(f5.FableEvaluator)
            evaluator.pit = None
            evaluator.spec = SimpleNamespace(stage="foundation")
            evaluator.car = "porsche_919evo"
            evaluator.drivetrain_version = "porsche919evo-7spd-ring-v2"
            evaluator.checkpoint = "fable5_919_ring_foundation.pt"
            ppo = SimpleNamespace(updates=7)
            evaluator._append_event(
                ppo,
                {"evaluated_policy_sha256": "abc", "latest_trace": [1, 2]},
                {"decision": "continue"},
            )
            path = Path("runtime/fable5/foundation/events.jsonl")
            event = json.loads(path.read_text().splitlines()[0])
        finally:
            os.chdir(previous)
        gate("events carry top-level car and drivetrain",
             event["car"] == "porsche_919evo"
             and event["drivetrain_version"] == "porsche919evo-7spd-ring-v2")
        gate("large traces stay out of runtime event",
             "latest_trace" not in event["evaluation"])


def main() -> int:
    registry_contract()
    with tempfile.TemporaryDirectory() as td:
        checkpoint_contracts(Path(td))
    resolution_order()
    manifest_repair()
    auto_manifest_routing()
    event_identity_stamp()
    if FAILED:
        print(f"\nFAILED: {', '.join(FAILED)}")
        return 1
    print("\nAll Fable edition identity gates passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
