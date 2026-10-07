from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import default_config_path, sha256_file
from .route import RouteManifest
from .telemetry import read_jsonl


@dataclass(frozen=True)
class EvaluationResult:
    accepted: bool
    clean_boot_successes: int
    deaths: int
    starworld_entries: int
    bowser_defeated: bool
    issues: tuple[str, ...]


def evaluate_trace(path: str | Path, route: RouteManifest) -> EvaluationResult:
    events = list(read_jsonl(path))
    route_result = route.validate_trace(events)
    deaths = sum(1 for event in events if event.get("death_detected"))
    starworld = sum(1 for event in events if route.detect_starworld(event))
    bowser_defeated = any(event.get("bowser_defeated") is True for event in events)
    clean_boot_successes = sum(1 for event in events if event.get("clean_boot_bowser_clear") is True)
    issues = [issue.message for issue in route_result.issues]
    if deaths:
        issues.append(f"death events detected: {deaths}")
    if starworld:
        issues.append(f"Starworld events detected: {starworld}")
    if not bowser_defeated:
        issues.append("Bowser defeat marker missing")
    accepted = route_result.ok and deaths == 0 and starworld == 0 and bowser_defeated
    return EvaluationResult(
        accepted=accepted,
        clean_boot_successes=clean_boot_successes,
        deaths=deaths,
        starworld_entries=starworld,
        bowser_defeated=bowser_defeated,
        issues=tuple(issues),
    )


def build_champion_artifact(
    output_dir: str | Path,
    *,
    policy_file: str | Path,
    route_file: str | Path | None = None,
    memory_map_file: str | Path | None = None,
    action_space_file: str | Path | None = None,
    rom_hash: str = "",
    emulator_version: str = "",
    evaluation_log: str | Path | None = None,
    lua_bridge_file: str | Path | None = None,
) -> Path:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    route_path = Path(route_file or default_config_path("route_no_starworld_safety.yaml"))
    memory_path = Path(memory_map_file or default_config_path("memory_map.yaml"))
    action_path = Path(action_space_file or default_config_path("action_space.yaml"))

    shutil.copy2(policy_file, target / "policy.zip")
    shutil.copy2(route_path, target / "route.yaml")
    shutil.copy2(memory_path, target / "memory_map.yaml")
    shutil.copy2(action_path, target / "action_space.yaml")
    if evaluation_log is not None:
        shutil.copy2(evaluation_log, target / "success_report.jsonl")

    manifest = {
        "schema": "smw-bowser-ai-brain-v1",
        "policy_file": "policy.zip",
        "route_file": "route.yaml",
        "memory_map_file": "memory_map.yaml",
        "action_space_file": "action_space.yaml",
        "rom_hash": rom_hash,
        "emulator_version": emulator_version,
        "lua_bridge_sha256": sha256_file(lua_bridge_file) if lua_bridge_file else "",
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    (target / "brain_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (target / "rom_hash.txt").write_text(rom_hash + "\n", encoding="utf-8")
    (target / "emulator_version.txt").write_text(emulator_version + "\n", encoding="utf-8")
    return target

