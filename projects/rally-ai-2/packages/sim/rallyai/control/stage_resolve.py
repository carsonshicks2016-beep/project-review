"""Resolve a stage document for live / drive sessions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rallyai import contracts
from rallyai.stage.fixtures import proving_ground
from rallyai.stage.generator import generate


def resolve_stage(
    *,
    seed: int,
    tier: int,
    procedural: bool = True,
    stage: str | None = None,
) -> dict[str, Any]:
    """Return a stage dict for ``RallyEnv``.

    Precedence:
    1. Explicit ``stage`` path that exists on disk
    2. Named fixture ``proving_ground``
    3. Procedural ``generate(seed, tier)`` when ``procedural``
    4. Fallback to ``proving_ground``
    """
    if stage:
        path = Path(stage).expanduser()
        if path.exists() and path.is_file():
            return contracts.read_json(path, kind="stage")
        if stage in ("proving_ground", "proving-ground"):
            return proving_ground()
        # Treat as a bare filename under packages/shared/stages if present.
        shared = (
            Path(__file__).resolve().parents[3]
            / "shared"
            / "stages"
            / f"{stage}.json"
        )
        if shared.exists():
            return contracts.read_json(shared, kind="stage")
        raise FileNotFoundError(f"unknown stage: {stage}")

    if procedural:
        return generate(int(seed), int(tier))
    return proving_ground()
