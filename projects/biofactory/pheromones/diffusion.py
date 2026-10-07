from __future__ import annotations

import numpy as np


def diffuse_and_decay(
    layer: np.ndarray,
    diffusion_rate: float,
    decay_rate: float,
    max_concentration: float,
    terrain_decay_multiplier: np.ndarray | None = None,
    terrain_diffusion_multiplier: np.ndarray | None = None,
) -> np.ndarray:
    neighbors = np.zeros_like(layer)
    counts = np.zeros_like(layer)

    neighbors[1:, :] += layer[:-1, :]
    counts[1:, :] += 1.0
    neighbors[:-1, :] += layer[1:, :]
    counts[:-1, :] += 1.0
    neighbors[:, 1:] += layer[:, :-1]
    counts[:, 1:] += 1.0
    neighbors[:, :-1] += layer[:, 1:]
    counts[:, :-1] += 1.0

    average = neighbors / np.maximum(counts, 1.0)
    if terrain_diffusion_multiplier is not None:
        local_diffusion = diffusion_rate * terrain_diffusion_multiplier
        updated = layer + local_diffusion * (average - layer)
    else:
        updated = layer + diffusion_rate * (average - layer)

    if terrain_decay_multiplier is not None:
        local_decay = np.clip(decay_rate * terrain_decay_multiplier, 0.0, 0.95)
        updated *= 1.0 - local_decay
    else:
        updated *= 1.0 - decay_rate

    np.clip(updated, 0.0, max_concentration, out=updated)
    return updated.astype(np.float32, copy=False)
