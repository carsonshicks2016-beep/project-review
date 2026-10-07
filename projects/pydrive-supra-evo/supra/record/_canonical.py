"""Canonical serialization and hashing for faithful-record evidence.

The record package deliberately uses only standard-library primitives here so
an evidence bundle can be verified without importing the simulator, PyTorch,
MuJoCo, or a checkpoint deserializer.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
import hashlib
import hmac
import json
from pathlib import Path
from typing import Any


SHA256_HEX_LENGTH = 64


def canonicalize(value: Any) -> Any:
    """Return a JSON-compatible, deterministic representation of ``value``."""
    if is_dataclass(value):
        return {field.name: canonicalize(getattr(value, field.name))
                for field in fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): canonicalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [canonicalize(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"cannot canonicalize {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(canonicalize(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_object(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hmac_sha256(value: Any, key: bytes) -> str:
    if not key:
        raise ValueError("verifier key must not be empty")
    return hmac.new(key, canonical_json_bytes(value), hashlib.sha256).hexdigest()


def secure_digest_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)


def require_sha256(value: str, field_name: str) -> None:
    if len(value) != SHA256_HEX_LENGTH:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be hexadecimal") from exc

