"""BEHAVIOR CHARACTERIZATION (ROADMAP Stage 8.1).

A *behavior characterization* (BC) is a short, fixed-length vector describing
WHAT A CREATURE DOES -- as opposed to the morphology descriptors (what it *is*)
in `descriptors.py`. Novelty search (Stage 8.2) drives exploration by rewarding
creatures whose BC is far from everything seen so far, so the BC must be:

  * FIXED-LENGTH and MORPHOLOGY-INDEPENDENT -- a 3-limb crawler and a 9-limb
    serpent must land in the *same* space, so nothing here may depend on the
    actuator/body count. Every per-actuator or per-body signal is collapsed to a
    scalar before it enters the vector.
  * AGGREGATE / NOISE-ROBUST -- built from episode statistics (mean speed, gait
    spectrum, contact duty, ...) so the SAME gait under different reset noise maps
    to NEARBY points while a categorically different behavior maps far away. That
    stability is exactly the Stage-8.1 "done-when" property.

The vector has six interpretable blocks (19 dims total):

    path (5)      net dx, net dy, path length, straightness, mean speed
    posture (3)   mean height, vertical oscillation, mean uprightness
    gait (5)      4 normalized power bands of the vertical-bob spectrum + dom freq
    contact (3)   mean ground contacts, contact duty fraction, contact variance
    effort (2)    mean |action|, temporal action oscillation
    survival (1)  fraction of the episode survived

Raw features are divided by reference SCALES so that an L2 distance over the
normalized vector weights every axis comparably (`bc_distance`). A deterministic
rollout (fixed `eval_seed`) makes the whole thing reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- fixed feature layout --------------------------------------------------
_GAIT_BINS = 4

LABELS = (
    "net_dx", "net_dy", "path_len", "straightness", "mean_speed",
    "mean_height", "height_osc", "mean_up",
    "gait_b0", "gait_b1", "gait_b2", "gait_b3", "dom_freq",
    "mean_contacts", "contact_duty", "contact_var",
    "action_mag", "action_osc",
    "alive_frac",
)

# reference magnitude of each feature; the normalized BC is raw / SCALES, chosen
# so a "typical" walker sits around O(1) on every axis and no single axis (e.g.
# path length) can dominate the distance. Fractions / already-[0,1] axes use 1.0.
SCALES = np.array([
    3.0, 3.0, 5.0, 1.0, 3.0,        # path
    1.0, 0.3, 1.0,                  # posture
    1.0, 1.0, 1.0, 1.0, 5.0,        # gait (bands sum~1; dom_freq in Hz)
    4.0, 1.0, 2.0,                  # contact
    1.0, 1.0,                       # effort
    1.0,                            # survival
], dtype=float)

DIM = len(LABELS)
assert DIM == len(SCALES)


@dataclass
class BehaviorChar:
    """A fixed-length behavior characterization with its human-readable parts."""
    vector: np.ndarray          # normalized, length DIM
    raw: dict                   # label -> raw (un-normalized) feature
    steps: int                  # episode length actually simulated

    def to_dict(self) -> dict:
        return {"vector": self.vector.tolist(), "raw": dict(self.raw),
                "steps": int(self.steps)}


# --- deterministic rollout -------------------------------------------------
def _rollout(env, policy, eval_seed: int, max_steps: int) -> dict:
    """Drive `env` with `policy` for one episode, collecting morphology-independent
    per-step signals. `policy` only needs `.act(obs, deterministic=True)`."""
    obs, _ = env.reset(seed=eval_seed)
    com0 = env.data.subtree_com[0].copy()
    com_xy, com_z, up, fwd, ncon, amag, actions = [], [], [], [], [], [], []
    steps = 0
    for _ in range(max_steps):
        a = np.asarray(policy.act(obs, deterministic=True), dtype=np.float32).reshape(-1)
        obs, _, term, trunc, info = env.step(a)
        steps += 1
        com = env.data.subtree_com[0]
        com_xy.append([float(com[0]), float(com[1])])
        com_z.append(float(info["com_z"]))
        up.append(float(info["up_proj"]))
        fwd.append(float(info["forward_vel"]))
        ncon.append(int(env.data.ncon))
        amag.append(float(np.mean(np.abs(a))) if a.size else 0.0)
        actions.append(a)
        if term or trunc:
            break
    return {
        "com0": com0,
        "com_xy": np.asarray(com_xy, dtype=float).reshape(-1, 2),
        "com_z": np.asarray(com_z, dtype=float),
        "up": np.asarray(up, dtype=float),
        "fwd": np.asarray(fwd, dtype=float),
        "ncon": np.asarray(ncon, dtype=float),
        "amag": np.asarray(amag, dtype=float),
        "actions": actions,
        "steps": steps,
        "dt": float(env.control_dt),
        "max_steps": int(max_steps),
    }


def _gait_spectrum(z: np.ndarray, dt: float) -> tuple:
    """Normalized power in `_GAIT_BINS` frequency bands of the (detrended) vertical
    bob signal, plus the dominant frequency (Hz). Captures gait rhythm in a
    fixed-length, morphology-independent way."""
    bands = np.zeros(_GAIT_BINS, dtype=float)
    if z.size < 2 * _GAIT_BINS:           # too short for a meaningful spectrum
        return bands, 0.0
    sig = z - z.mean()
    power = np.abs(np.fft.rfft(sig)) ** 2
    freqs = np.fft.rfftfreq(z.size, d=dt)
    power, freqs = power[1:], freqs[1:]   # drop DC
    total = float(power.sum())
    if total <= 0.0 or power.size == 0:
        return bands, 0.0
    for k, part in enumerate(np.array_split(power, _GAIT_BINS)):
        bands[k] = float(part.sum()) / total
    dom_freq = float(freqs[int(np.argmax(power))])
    return bands, dom_freq


def _features(roll: dict) -> dict:
    """Reduce the raw trajectory to the named raw features (pre-normalization)."""
    xy, z, up = roll["com_xy"], roll["com_z"], roll["up"]
    dt, steps = roll["dt"], roll["steps"]
    f = {k: 0.0 for k in LABELS}

    if xy.shape[0] >= 1:
        net = xy[-1] - roll["com0"][:2]
        f["net_dx"], f["net_dy"] = float(net[0]), float(net[1])
        net_disp = float(np.linalg.norm(net))
        path_len = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if xy.shape[0] > 1 else 0.0
        f["path_len"] = path_len
        f["straightness"] = net_disp / path_len if path_len > 1e-9 else 0.0
        f["mean_speed"] = net_disp / max(steps * dt, 1e-6)

    if z.size:
        f["mean_height"] = float(z.mean())
        f["height_osc"] = float(z.std())
    if up.size:
        f["mean_up"] = float(up.mean())

    bands, dom = _gait_spectrum(z, dt)
    for k in range(_GAIT_BINS):
        f[f"gait_b{k}"] = float(bands[k])
    f["dom_freq"] = dom

    ncon = roll["ncon"]
    if ncon.size:
        f["mean_contacts"] = float(ncon.mean())
        f["contact_duty"] = float(np.mean(ncon >= 1))
        f["contact_var"] = float(ncon.std())

    amag = roll["amag"]
    if amag.size:
        f["action_mag"] = float(amag.mean())
    acts = roll["actions"]
    if len(acts) > 1:
        A = np.asarray(acts, dtype=float)            # (T, nu)
        f["action_osc"] = float(A.std(axis=0).mean())  # per-actuator temporal std, averaged

    f["alive_frac"] = steps / max(roll["max_steps"], 1)
    return f


def characterize(env, policy, *, eval_seed: int = 0, max_steps: int = 300) -> BehaviorChar:
    """Deterministic behavior characterization of `policy` driving `env`.

    Returns a `BehaviorChar` whose `.vector` is the fixed-length (DIM) normalized
    BC. `policy` is anything with `act(obs, deterministic=True)`."""
    roll = _rollout(env, policy, eval_seed, max_steps)
    raw = _features(roll)
    vec = np.array([raw[name] for name in LABELS], dtype=float) / SCALES
    return BehaviorChar(vector=vec, raw=raw, steps=roll["steps"])


# --- distances over BC space ------------------------------------------------
def bc_distance(a, b) -> float:
    """RMS (per-dimension L2) distance between two BCs or raw vectors."""
    va = a.vector if isinstance(a, BehaviorChar) else np.asarray(a, dtype=float)
    vb = b.vector if isinstance(b, BehaviorChar) else np.asarray(b, dtype=float)
    return float(np.linalg.norm(va - vb) / np.sqrt(DIM))


def bc_matrix(bcs) -> np.ndarray:
    """Pairwise BC-distance matrix for a list of BehaviorChars / vectors."""
    n = len(bcs)
    M = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = bc_distance(bcs[i], bcs[j])
    return M


def nearest(query, others) -> int:
    """Index of the BC in `others` closest to `query` (its behavioral neighbor)."""
    return int(np.argmin([bc_distance(query, o) for o in others]))


def novelty(query, archive, *, k: int = 5) -> float:
    """Mean distance from `query` to its `k` nearest BCs in `archive` -- the
    sparseness score novelty search (Stage 8.2) will maximize. Empty archive -> inf."""
    if not len(archive):
        return float("inf")
    d = np.sort([bc_distance(query, o) for o in archive])
    return float(d[: min(k, d.size)].mean())
