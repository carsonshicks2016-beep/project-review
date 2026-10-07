"""The one implementation of the three contracts.

Loading, validating, quantising and hashing stage / replay / metrics documents
happens here and nowhere else. Two implementations of "the canonical form" is
two answers to "is this the same stage", which is how a replay ends up playing
against geometry the physics never stepped.

Schemas live in ``packages/shared/schemas`` and are read by both Python and the
viewer. This module is the Python half.
"""

from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

# Coordinates are quantised to this many decimal places before they are written
# or hashed. Rationale: the generator does float maths, and a different CPU or
# numpy build can differ in the last bit or two. Micrometre precision is far
# finer than anything the physics or the renderer can resolve, and it makes
# "seed N produces the same stage on any machine" true rather than aspirational.
QUANTIZE_DP = 6


def _repo_root() -> Path:
    # rallyai/contracts.py -> rallyai -> sim -> packages -> repo root
    return Path(__file__).resolve().parents[3]


def schema_dir() -> Path:
    return _repo_root() / "packages" / "shared" / "schemas"


@cache
def load_schema(name: str) -> dict[str, Any]:
    """Load a schema by filename stem, e.g. ``load_schema("stage")``."""
    path = schema_dir() / f"{name}.schema.json"
    return json.loads(path.read_text())


@cache
def _validator(name: str):
    # Imported lazily: jsonschema is a dev dependency. Training and inference
    # must not need it, but anything that writes a contract document should.
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    resources = []
    for path in schema_dir().glob("*.schema.json"):
        resources.append((path.name, Resource.from_contents(json.loads(path.read_text()))))
    registry = Registry().with_resources(resources)
    return Draft202012Validator(load_schema(name), registry=registry)


class ContractError(ValueError):
    """A document does not satisfy its schema, or violates an invariant the
    schema cannot express."""


def validate(doc: dict[str, Any], kind: str) -> None:
    """Raise ``ContractError`` if ``doc`` does not satisfy the ``kind`` schema.

    ``kind`` is one of ``stage``, ``replay``, ``metrics``.
    """
    errors = sorted(_validator(kind).iter_errors(doc), key=lambda e: list(e.absolute_path))
    if not errors:
        return
    lines = [f"{kind} document failed validation ({len(errors)} error(s)):"]
    for err in errors[:10]:
        where = "/".join(str(p) for p in err.absolute_path) or "<root>"
        lines.append(f"  at {where}: {err.message}")
    if len(errors) > 10:
        lines.append(f"  ... and {len(errors) - 10} more")
    raise ContractError("\n".join(lines))


def quantize(value: Any, dp: int = QUANTIZE_DP) -> Any:
    """Recursively round floats and coerce numpy scalars to Python builtins.

    Applied when a document is *written*, so the file on disk and the hash of it
    agree. Ints are left alone; rounding them would be lossy for seeds.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        rounded = round(value, dp)
        # Normalise -0.0 to 0.0 so two runs that differ only in zero sign hash
        # identically.
        return rounded + 0.0
    if isinstance(value, dict):
        return {k: quantize(v, dp) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [quantize(v, dp) for v in value]
    # numpy scalars and arrays, without importing numpy at module scope
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "shape", None) == ():
        return quantize(item(), dp)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return quantize(tolist(), dp)
    return value


def canonical_bytes(doc: dict[str, Any]) -> bytes:
    """The one canonical serialisation: sorted keys, no whitespace, UTF-8.

    NaN and infinity are rejected — they are not valid JSON, they break every
    downstream reader, and in a stage file they mean the generator produced
    geometry no car can drive.
    """
    return json.dumps(
        doc,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(doc: dict[str, Any]) -> str:
    """SHA-256 over the canonical serialisation, with ``content_hash`` removed.

    Removing the field first is what makes the hash self-consistent: a document
    can carry its own hash without the hash depending on itself.
    """
    without = {k: v for k, v in doc.items() if k != "content_hash"}
    return "sha256:" + hashlib.sha256(canonical_bytes(without)).hexdigest()


def stamp(doc: dict[str, Any]) -> dict[str, Any]:
    """Quantise, then stamp the resulting document with its own content hash."""
    out = quantize(doc)
    out["content_hash"] = content_hash(out)
    return out


def write_json(doc: dict[str, Any], path: str | Path, *, kind: str | None = None) -> Path:
    """Quantise, stamp, validate and write — the only sanctioned way to emit a
    contract document.

    Written pretty-printed (2-space indent) because §3 requires these files stay
    readable and diffable. The *hash* is over the canonical compact form, so
    formatting can change without changing identity.
    """
    out = stamp(doc)
    if kind:
        validate(out, kind)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    return path


def read_json(path: str | Path, *, kind: str | None = None, verify_hash: bool = True) -> dict[str, Any]:
    """Read a contract document, optionally validating and verifying its hash."""
    path = Path(path)
    doc = json.loads(path.read_text())
    if kind:
        validate(doc, kind)
    stored = doc.get("content_hash")
    if verify_hash and stored is not None:
        actual = content_hash(doc)
        if actual != stored:
            raise ContractError(
                f"{path.name}: content_hash mismatch — the file has been edited "
                f"since it was written.\n  stored: {stored}\n  actual: {actual}"
            )
    return doc
