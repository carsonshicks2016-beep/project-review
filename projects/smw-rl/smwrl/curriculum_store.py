"""Fail-closed persistence for clear-proven reverse curricula."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

CURRICULUM_FORMAT = 1


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metadata_path(curriculum_path: str | Path) -> Path:
    path = Path(curriculum_path)
    return path.with_suffix(".meta.json")


def write_verified_curriculum(archive_path: str | Path,
                              curriculum_path: str | Path,
                              archive, entries: list[tuple[int, int, bytes]]) -> None:
    """Atomically bind a curriculum to the exact v2 archive that proved it."""
    archive_path = Path(archive_path)
    path = Path(curriculum_path)
    if int(getattr(archive, "route_format", 0)) != 2:
        raise ValueError("only route-format v2 archives may export a curriculum")
    if not entries:
        raise ValueError("archive has no clear-proven route; refusing to export a curriculum")
    positions = [(int(room), int(x)) for room, x, _ in entries]
    if any(b <= a for a, b in zip(positions, positions[1:])):
        raise ValueError("curriculum route is not strictly ordered start-to-goal")

    archive_bytes = archive_path.read_bytes()
    payload = pickle.dumps(entries)
    meta = {
        "format": CURRICULUM_FORMAT,
        "source": "go_explore_clear_proof",
        "route_format": 2,
        "archive_sha256": _sha256(archive_bytes),
        "payload_sha256": _sha256(payload),
        "stages": len(entries),
        "positions": positions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_tmp = path.with_suffix(path.suffix + ".tmp")
    meta_dest = metadata_path(path)
    meta_tmp = meta_dest.with_suffix(meta_dest.suffix + ".tmp")
    payload_tmp.write_bytes(payload)
    meta_tmp.write_text(json.dumps(meta, indent=2) + "\n")
    # If interrupted between replacements, hashes disagree and the loader
    # rejects the pair.  A torn/stale curriculum can never silently train.
    os.replace(payload_tmp, path)
    os.replace(meta_tmp, meta_dest)


def load_verified_curriculum(archive_path: str | Path,
                             curriculum_path: str | Path) -> list[tuple[int, int, bytes]]:
    """Load only a curriculum bound to the current clear-proven archive."""
    archive_path = Path(archive_path)
    path = Path(curriculum_path)
    meta_path = metadata_path(path)
    if not archive_path.exists():
        raise ValueError(f"missing source archive {archive_path}")
    if not path.exists() or not meta_path.exists():
        raise ValueError(
            "missing verified curriculum pair; rebuild with "
            "`python -m smwrl.explore --level LEVEL --resume`"
        )
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"invalid curriculum metadata {meta_path}: {e}") from e
    required = {
        "format": CURRICULUM_FORMAT,
        "source": "go_explore_clear_proof",
        "route_format": 2,
    }
    for key, expected in required.items():
        if meta.get(key) != expected:
            raise ValueError(f"unsupported curriculum metadata {key}={meta.get(key)!r}")

    archive_bytes = archive_path.read_bytes()
    payload = path.read_bytes()
    if meta.get("archive_sha256") != _sha256(archive_bytes):
        raise ValueError("curriculum was exported from a different archive revision")
    if meta.get("payload_sha256") != _sha256(payload):
        raise ValueError("curriculum payload hash does not match its metadata")
    try:
        entries = pickle.loads(payload)
    except Exception as e:
        raise ValueError(f"invalid curriculum payload: {type(e).__name__}: {e}") from e
    if not isinstance(entries, list) or len(entries) != meta.get("stages") or not entries:
        raise ValueError("curriculum stage count is invalid")
    if any(not isinstance(e, tuple) or len(e) != 3 or not isinstance(e[2], bytes)
           for e in entries):
        raise ValueError("curriculum must contain (room, x, compressed_state) tuples")
    positions = [(int(room), int(x)) for room, x, _ in entries]
    if positions != [tuple(p) for p in meta.get("positions", [])]:
        raise ValueError("curriculum positions do not match metadata")
    if any(b <= a for a, b in zip(positions, positions[1:])):
        raise ValueError("curriculum positions are not strictly forward")
    return entries
