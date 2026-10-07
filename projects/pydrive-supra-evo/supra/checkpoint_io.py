"""Constrained loaders for project-owned PyTorch checkpoint dictionaries.

PyTorch checkpoints are pickle containers.  ``weights_only=False`` can execute
arbitrary code before identity or hashes are inspected, which is unacceptable
for dashboard discovery and evidence tooling.  The historical Supra files also
contain NumPy arrays and ``TorchVersion`` metadata, so this module adds only
those inert value/container types to PyTorch's weights-only allowlist.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_torch_checkpoint_safe(path: str | Path, *, map_location: Any = "cpu") -> dict[str, Any]:
    import numpy as np
    import torch

    safe_types: list[type | Any] = [
        np._core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
        torch.torch_version.TorchVersion,
    ]
    # NumPy 2 represents concrete dtypes as separate inert classes. Historical
    # checkpoints may contain any of them depending on optimizer/normalizer
    # state, so allow the dtype classes—but no user functions or project code.
    safe_types.extend(
        value for name in dir(np.dtypes)
        if isinstance((value := getattr(np.dtypes, name)), type)
    )
    with torch.serialization.safe_globals(safe_types):
        payload = torch.load(
            Path(path), map_location=map_location, weights_only=True
        )
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint is not a dictionary: {path}")
    return payload


__all__ = ["load_torch_checkpoint_safe"]
