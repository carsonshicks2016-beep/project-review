"""Behavior + morphology descriptors (ROADMAP Stage 5.1).

These are the axes a MAP-Elites archive (5.2) is binned over: instead of one
averaged reward, we keep the best creature in each *niche* of descriptor space.

Morphology descriptors are cheap, deterministic functions of the genome/developed
body (no simulation): limb count, muscle count, mass, height, body-plan aspect
ratio, symmetry fraction, and morphological distance from a reference seed (the
Stage-4.4 graph-edit distance). The one behavior descriptor here, forward speed,
needs a deterministic rollout with a policy (more niches arrive with Stage 7).

`DESCRIPTOR_BOUNDS` give reference ranges so descriptors can be normalized to
[0,1] for archive binning.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..encoding.genome import Genome, SymmetryKind
from ..encoding.innovation import structural_distance
from ..morphogenesis import develop


# reference (lo, hi) ranges for normalization / archive binning
DESCRIPTOR_BOUNDS = {
    "limb_count": (1.0, 20.0),
    "muscle_count": (0.0, 12.0),
    "mass": (1.0, 120.0),
    "height": (0.1, 2.0),
    "aspect": (0.2, 6.0),
    "symmetry": (0.0, 1.0),
    "morph_distance": (0.0, 30.0),
    "speed": (-2.0, 6.0),
}


# --- morphology (deterministic, no simulation) ----------------------------
def morphology_descriptors(genome: Genome, morph=None) -> dict:
    m = morph if morph is not None else develop(genome)
    pos = np.array([b.world_pos for b in m.bodies], dtype=float)
    span = pos.max(axis=0) - pos.min(axis=0) if len(pos) else np.zeros(3)
    horiz = float(max(span[0], span[1]))
    vert = float(max(span[2], 1e-6))
    n_sym = sum(1 for e in genome.edges if e.symmetry is not SymmetryKind.NONE)
    return {
        "limb_count": float(m.body_count),
        "muscle_count": float(len(genome.muscles)),
        "mass": float(m.total_mass()),
        "height": vert,
        "aspect": float(horiz / vert),
        "symmetry": float(n_sym / len(genome.edges)) if genome.edges else 0.0,
    }


def distance_from(genome: Genome, seed_genome: Genome) -> float:
    """Morphological distance from a reference seed (graph-edit distance)."""
    return float(structural_distance(seed_genome, genome))


# --- behavior (needs a deterministic rollout) ------------------------------
def forward_speed(env, policy, *, eval_seed: int = 0, max_steps: int = 300) -> float:
    """Mean forward speed (m/s) of `policy` driving `env`, deterministic given seed."""
    obs, _ = env.reset(seed=eval_seed)
    x0 = float(env.data.qpos[env.task.forward_axis])
    steps = 0
    for _ in range(max_steps):
        obs, _, term, trunc, _ = env.step(policy.act(obs, deterministic=True).astype(np.float32))
        steps += 1
        if term or trunc:
            break
    dx = float(env.data.qpos[env.task.forward_axis]) - x0
    return dx / max(steps * env.control_dt, 1e-6)


def behavior_descriptors(env, policy, *, eval_seed: int = 0, max_steps: int = 300) -> dict:
    return {"speed": forward_speed(env, policy, eval_seed=eval_seed, max_steps=max_steps)}


# --- combined --------------------------------------------------------------
def compute(genome: Genome, *, seed_genome: Optional[Genome] = None, morph=None,
            env=None, policy=None, eval_seed: int = 0, max_steps: int = 300) -> dict:
    """All available descriptors for a creature. Morphology always; morph_distance
    if `seed_genome`; behavior if `env` + `policy`."""
    d = morphology_descriptors(genome, morph)
    if seed_genome is not None:
        d["morph_distance"] = distance_from(genome, seed_genome)
    if env is not None and policy is not None:
        d.update(behavior_descriptors(env, policy, eval_seed=eval_seed, max_steps=max_steps))
    return d


def normalize(name: str, value: float) -> float:
    """Map a descriptor to [0,1] using its reference bounds."""
    lo, hi = DESCRIPTOR_BOUNDS[name]
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0))


def normalized(descriptors: dict) -> dict:
    return {k: normalize(k, v) for k, v in descriptors.items() if k in DESCRIPTOR_BOUNDS}
