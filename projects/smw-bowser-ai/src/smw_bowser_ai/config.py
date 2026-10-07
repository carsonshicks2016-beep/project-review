from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(RuntimeError):
    """Raised when a project manifest cannot be loaded."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load JSON-compatible YAML, with optional PyYAML fallback for richer YAML."""

    target = Path(path)
    text = target.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError as json_error:
        try:
            import yaml  # type: ignore[import-not-found]
        except Exception as import_error:  # pragma: no cover - depends on local env
            raise ConfigError(
                f"{target} is not JSON-compatible YAML and PyYAML is not installed"
            ) from import_error
        try:
            value = yaml.safe_load(text)
        except Exception as yaml_error:  # pragma: no cover - depends on PyYAML
            raise ConfigError(f"could not parse {target}: {yaml_error}") from json_error

    if not isinstance(value, dict):
        raise ConfigError(f"{target} must contain an object at the top level")
    return value


def default_config_path(name: str) -> Path:
    return CONFIG_DIR / name


def sha256_file(path: str | Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

