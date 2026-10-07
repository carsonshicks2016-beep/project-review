#!/usr/bin/env python3
"""Static gates for the pinned faithful Linux authority runtime."""
from __future__ import annotations

import hashlib
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "containers" / "faithful" / "Dockerfile"
LOCK = ROOT / "requirements-faithful-lock.txt"
EXPECTED_BASE = (
    "python:3.12.10-slim-bookworm@"
    "sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db"
)
EXPECTED_PACKAGES = {
    "absl-py", "casadi", "cffi", "cryptography", "etils", "fsspec",
    "glfw", "jax", "jaxlib", "ml-dtypes", "mujoco", "numpy",
    "opt-einsum", "pycparser", "pyopengl", "scipy", "trimesh",
    "typing-extensions", "zipp",
}
FAILED: list[str] = []


def gate(name: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        FAILED.append(name)


def main() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    lock = LOCK.read_text(encoding="utf-8")
    matches = list(re.finditer(
        r"^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+) [\\]$", lock, re.MULTILINE
    ))
    parsed = {match.group(1).lower(): match.group(2) for match in matches}
    every_stanza_hashed = all(
        "--hash=sha256:" in lock[
            match.end():matches[index + 1].start() if index + 1 < len(matches) else len(lock)
        ]
        for index, match in enumerate(matches)
    )

    print("== pinned Linux authority runtime ==")
    gate("base image is exact digest and Linux amd64",
         "ARG AUTHORITY_PLATFORM=linux/amd64" in dockerfile
         and f"FROM --platform=${{AUTHORITY_PLATFORM}} {EXPECTED_BASE}" in dockerfile)
    gate("authority dependency set is complete and exactly pinned",
         set(parsed) == EXPECTED_PACKAGES and len(matches) == len(parsed))
    gate("every direct and transitive package is hash locked",
         every_stanza_hashed and "--require-hashes" in dockerfile)
    gate("JAX double precision is enabled", "JAX_ENABLE_X64=True" in dockerfile)
    gate("MuJoCo headless EGL is selected", "MUJOCO_GL=egl" in dockerfile)
    gate("container runs the solver/physics identity probe during build",
         "validate_faithful_runtime.py --write-identity" in dockerfile)
    gate("lock is content-addressable",
         hashlib.sha256(LOCK.read_bytes()).hexdigest()
         == "576f6637926d6dbb48c4fcf299c6f6d0733ec52125c2b69b5e9024bcb85c97c4")
    gate("Dockerfile contains no floating latest tag", ":latest" not in dockerfile)

    if FAILED:
        raise SystemExit("faithful runtime spec validation failed: " + ", ".join(FAILED))
    print("faithful runtime spec validation: PASS")


if __name__ == "__main__":
    main()
