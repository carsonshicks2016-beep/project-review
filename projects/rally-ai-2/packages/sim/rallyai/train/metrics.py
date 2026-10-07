"""Append-only metrics JSONL, validated against the shared schema.

The file is the truth; Phase F's WebSocket merely replays these lines.
Flush after every write so a killed run leaves a readable log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

_SCHEMA_CACHE: dict[str, Any] | None = None
_VALIDATOR = None


def _schema_path() -> Path:
    # packages/sim/rallyai/train/metrics.py → packages/shared/schemas/
    here = Path(__file__).resolve()
    return here.parents[3] / "shared" / "schemas" / "metrics.schema.json"


def _load_validator():
    global _SCHEMA_CACHE, _VALIDATOR
    if _VALIDATOR is not None:
        return _VALIDATOR
    import jsonschema

    path = _schema_path()
    with path.open(encoding="utf-8") as fh:
        _SCHEMA_CACHE = json.load(fh)
    _VALIDATOR = jsonschema.Draft202012Validator(_SCHEMA_CACHE)
    return _VALIDATOR


def validate_metrics_line(line: dict[str, Any]) -> None:
    """Raise ``jsonschema.ValidationError`` if ``line`` is not schema-valid."""
    _load_validator().validate(line)


class MetricsWriter:
    """One JSONL file per ``run_id``."""

    KINDS = (
        "run_start",
        "update",
        "eval",
        "checkpoint",
        "curriculum",
        "worker",
        "run_end",
    )

    def __init__(self, path: str | Path, run_id: str) -> None:
        self.path = Path(path)
        self.run_id = str(run_id)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def write(self, kind: str, **fields: Any) -> dict[str, Any]:
        line: dict[str, Any] = {
            "schema_version": 1,
            "kind": kind,
            "run_id": self.run_id,
            "wall_t": float(fields.pop("wall_t", time.time())),
        }
        for key, value in fields.items():
            if value is not None:
                line[key] = value
        validate_metrics_line(line)
        self._fh.write(json.dumps(line, separators=(",", ":")) + "\n")
        self._fh.flush()
        return line

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> MetricsWriter:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
