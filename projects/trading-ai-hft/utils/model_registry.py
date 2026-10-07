"""Model discovery and manifest helpers for deployable trading models."""

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ACTIVE_MODELS_DIR = MODELS_DIR / "active"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
ACTIVE_MANIFEST = ACTIVE_MODELS_DIR / "manifest.json"


def load_active_manifest() -> dict:
    if not ACTIVE_MANIFEST.exists():
        return {}
    with open(ACTIVE_MANIFEST, "r", encoding="utf-8") as f:
        return json.load(f)


def discover_model_paths(prefer_active: bool = True, include_neat: bool = True,
                         include_ppo: bool = True) -> list:
    """Return deployable model paths, preferring models/active when populated."""
    paths = []

    if prefer_active and ACTIVE_MODELS_DIR.exists():
        if include_neat:
            paths.extend(sorted(ACTIVE_MODELS_DIR.glob("*.pkl")))
        if include_ppo:
            paths.extend(sorted(ACTIVE_MODELS_DIR.glob("*.zip")))
        if paths:
            return [str(p) for p in paths]

    if include_neat and CHECKPOINTS_DIR.exists():
        paths.extend(sorted(CHECKPOINTS_DIR.glob("*.pkl")))
    if include_ppo and MODELS_DIR.exists():
        paths.extend(sorted(p for p in MODELS_DIR.glob("*.zip") if p.parent != ACTIVE_MODELS_DIR))

    return [str(p) for p in paths]


def write_active_manifest(manifest: dict):
    ACTIVE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ACTIVE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
