"""Historical opponent pool for control self-play checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np

CONTROL_POOL_VERSION = 1


@dataclass
class ControlPoolEntry:
    id: str
    generation: int
    name: str
    seed: int
    evader: str
    pursuer_team: str
    source_manifest: str | None = None
    control_evaluation: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "generation": self.generation,
            "name": self.name,
            "seed": self.seed,
            "evader": self.evader,
            "pursuer_team": self.pursuer_team,
            "source_manifest": self.source_manifest,
            "control_evaluation": self.control_evaluation,
        }


@dataclass
class ControlOpponentSelection:
    kind: str
    checkpoint: str | None
    source: str
    label: str | None = None
    generation: int | None = None
    entry_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "checkpoint": self.checkpoint,
            "source": self.source,
            "label": self.label,
            "generation": self.generation,
            "entry_id": self.entry_id,
        }


def load_control_pool(pool_dir: str | Path = "models/control_pool") -> list[ControlPoolEntry]:
    """Load historical self-play checkpoint pairs from the pool manifest."""
    manifest = Path(pool_dir) / "manifest.json"
    if not manifest.exists():
        return []
    data = json.loads(manifest.read_text(encoding="utf-8"))
    entries = []
    for row in data.get("entries", []):
        entries.append(ControlPoolEntry(
            id=str(row["id"]),
            generation=int(row["generation"]),
            name=str(row.get("name", row["id"])),
            seed=int(row.get("seed", 0)),
            evader=str(row["evader"]),
            pursuer_team=str(row["pursuer_team"]),
            source_manifest=row.get("source_manifest"),
            control_evaluation=row.get("control_evaluation"),
        ))
    return entries


def archive_control_pair(
    *,
    name: str,
    seed: int,
    evader: str | Path,
    pursuer_team: str | Path,
    pool_dir: str | Path = "models/control_pool",
    source_manifest: str | Path | None = None,
    control_evaluation: dict[str, Any] | None = None,
) -> ControlPoolEntry:
    """Copy a promoted control pair into the historical opponent pool."""
    pool = Path(pool_dir)
    generation_root = pool / "generations"
    generation_root.mkdir(parents=True, exist_ok=True)
    entries = load_control_pool(pool)
    generation = max((entry.generation for entry in entries), default=0) + 1
    safe_name = _safe_name(name)
    entry_id = f"{generation:04d}_{safe_name}_seed{int(seed)}"
    dest = generation_root / entry_id
    dest.mkdir(parents=True, exist_ok=True)

    evader_dest = dest / "evader_ppo.pt"
    team_dest = dest / "pursuer_team_ppo.pt"
    shutil.copy2(evader, evader_dest)
    shutil.copy2(pursuer_team, team_dest)

    entry = ControlPoolEntry(
        id=entry_id,
        generation=generation,
        name=safe_name,
        seed=int(seed),
        evader=str(evader_dest),
        pursuer_team=str(team_dest),
        source_manifest=str(source_manifest) if source_manifest is not None else None,
        control_evaluation=_evaluation_summary(control_evaluation),
    )
    entries.append(entry)
    _write_pool_manifest(pool, entries)
    return entry


def select_control_opponent(
    *,
    pool_dir: str | Path = "models/control_pool",
    kind: str,
    seed: int = 11,
    fallback: str | Path | None = None,
    fallback_label: str = "fallback",
    mix_fallback: bool = True,
) -> ControlOpponentSelection:
    """Select one historical checkpoint of the requested kind for training."""
    if kind not in {"evader", "pursuer_team"}:
        raise ValueError("kind must be 'evader' or 'pursuer_team'")
    entries = load_control_pool(pool_dir)
    candidates: list[ControlOpponentSelection] = [
        ControlOpponentSelection(
            kind=kind,
            checkpoint=getattr(entry, kind),
            source="pool",
            label=entry.name,
            generation=entry.generation,
            entry_id=entry.id,
        )
        for entry in entries
    ]
    if fallback is not None and (mix_fallback or not candidates):
        candidates.append(ControlOpponentSelection(
            kind=kind,
            checkpoint=str(fallback),
            source="fallback",
            label=fallback_label,
        ))
    if not candidates:
        return ControlOpponentSelection(kind=kind, checkpoint=None, source="none")
    idx = int(np.random.default_rng(seed).integers(0, len(candidates)))
    return candidates[idx]


def summarize_control_pool(pool_dir: str | Path = "models/control_pool") -> dict[str, Any]:
    entries = load_control_pool(pool_dir)
    latest = entries[-1].to_dict() if entries else None
    best = _best_entry(entries)
    return {
        "version": CONTROL_POOL_VERSION,
        "pool_dir": str(pool_dir),
        "entries": len(entries),
        "latest": latest,
        "best": best.to_dict() if best is not None else None,
        "manifest": str(Path(pool_dir) / "manifest.json"),
    }


def _write_pool_manifest(pool: Path, entries: list[ControlPoolEntry]) -> Path:
    manifest = pool / "manifest.json"
    payload = {
        "version": CONTROL_POOL_VERSION,
        "entries": [entry.to_dict() for entry in entries],
    }
    manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _evaluation_summary(control_evaluation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not control_evaluation:
        return None
    return {
        "scores": control_evaluation.get("scores", {}),
        "aggregates": control_evaluation.get("aggregates", {}),
        "manifest": control_evaluation.get("manifest"),
        "seeds": control_evaluation.get("seeds", []),
        "steps": control_evaluation.get("steps", 0),
    }


def _best_entry(entries: list[ControlPoolEntry]) -> ControlPoolEntry | None:
    if not entries:
        return None

    def score(entry: ControlPoolEntry) -> float:
        evaluation = entry.control_evaluation or {}
        scores = evaluation.get("scores", {})
        return float(scores.get("overall_score", -1.0))

    return max(entries, key=score)


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name).strip())
    return safe or "control_pair"
