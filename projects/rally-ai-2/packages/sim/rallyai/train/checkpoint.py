"""Checkpoint I/O and the hall of fame.

A checkpoint is one file: policy, optimiser, observation normaliser, layout
metadata, and a content hash. Validate on every load — a corrupted checkpoint
that loads silently costs a night of training.
"""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rallyai.env.reward import RewardConfig
from rallyai.sense import SensorSpec
from rallyai.train.normalise import ObservationNormaliser

HOF_SLOTS = ("latest", "best", "cleanest", "fastest", "furthest")
OBS_LAYOUT_VERSION = 1


def require_finite(name: str, value: Any) -> None:
    if torch.is_tensor(value):
        ok = bool(torch.isfinite(value).all())
    else:
        ok = bool(np.all(np.isfinite(np.asarray(value))))
    if not ok:
        raise FloatingPointError(f"non-finite {name}")


def state_dict_hash(state_dict: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        if not torch.is_tensor(tensor):
            raise ValueError(f"state_dict entry {name!r} is not a tensor")
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode("utf-8"))
        h.update(str(tuple(arr.shape)).encode("ascii"))
        h.update(str(arr.dtype).encode("ascii"))
        h.update(arr.tobytes())
    return h.hexdigest()


def _nested_finite(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_nested_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_nested_finite(v) for v in value)
    if isinstance(value, (float, np.floating)):
        return bool(np.isfinite(value))
    return True


@dataclass
class CheckpointPayload:
    """In-memory checkpoint before serialisation."""

    policy: dict[str, Any]
    optimizer: dict[str, Any] | None
    normaliser: dict[str, Any]
    sensor_spec: dict[str, Any]
    reward_config: dict[str, Any]
    timesteps: int
    tier: int
    obs_layout_version: int = OBS_LAYOUT_VERSION
    obs_dim: int = 84
    act_dim: int = 4
    stage: str = "foundation"
    run_id: str = ""
    curriculum: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def build_payload(
    *,
    policy: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    normaliser: ObservationNormaliser,
    sensor_spec: SensorSpec,
    reward_config: RewardConfig,
    timesteps: int,
    tier: int,
    stage: str = "foundation",
    run_id: str = "",
    curriculum: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    state = {k: v.detach().cpu() for k, v in policy.state_dict().items()}
    for name, tensor in state.items():
        require_finite(f"policy.{name}", tensor)
    payload: dict[str, Any] = {
        "state_dict": state,
        "opt": None if optimizer is None else optimizer.state_dict(),
        "normaliser": normaliser.to_dict(),
        "sensor_spec": asdict(sensor_spec),
        "reward_config": asdict(reward_config),
        "timesteps": int(timesteps),
        "tier": int(tier),
        "obs_layout_version": OBS_LAYOUT_VERSION,
        "obs_dim": int(normaliser.dim),
        "act_dim": int(getattr(policy, "act_dim", 4)),
        "stage": str(stage),
        "run_id": str(run_id),
        "curriculum": curriculum,
        "policy_sha256": state_dict_hash(state),
        "has_normaliser": True,
        "extra": extra or {},
    }
    return payload


def validate_checkpoint(payload: dict[str, Any], *, require_hash: bool = True) -> None:
    state = payload.get("state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("missing state_dict")
    if not _nested_finite(state):
        raise ValueError("non-finite state_dict")

    actual = state_dict_hash(state)
    stored = payload.get("policy_sha256")
    if require_hash and not stored:
        raise ValueError("missing policy_sha256")
    if stored and stored != actual:
        raise ValueError("policy_sha256 does not match state_dict")

    if not payload.get("has_normaliser", False) and "normaliser" not in payload:
        raise ValueError("checkpoint missing normaliser")
    norm = payload.get("normaliser")
    if not isinstance(norm, dict):
        raise ValueError("invalid normaliser")
    mean = np.asarray(norm.get("mean"), dtype=np.float64)
    var = np.asarray(norm.get("var"), dtype=np.float64)
    count = float(norm.get("count", 0.0))
    obs_dim = int(payload.get("obs_dim", mean.shape[0] if mean.ndim else 0))
    if (
        obs_dim <= 0
        or mean.shape != (obs_dim,)
        or var.shape != (obs_dim,)
        or not np.all(np.isfinite(mean))
        or not np.all(np.isfinite(var))
        or not np.all(var > 0.0)
        or not np.isfinite(count)
        or count <= 0.0
    ):
        raise ValueError("invalid observation normaliser")

    layout = int(payload.get("obs_layout_version", -1))
    if layout != OBS_LAYOUT_VERSION:
        raise ValueError(
            f"obs_layout_version {layout} != current {OBS_LAYOUT_VERSION}"
        )
    # Sensor layout size must match the live SensorSpec default size.
    expected_size = SensorSpec().size
    if obs_dim != expected_size:
        raise ValueError(f"obs_dim {obs_dim} != SensorSpec.size {expected_size}")

    if "opt" in payload and payload["opt"] is not None:
        if not _nested_finite(payload["opt"]):
            raise ValueError("non-finite optimizer state")


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> Path:
    validate_checkpoint(payload, require_hash=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return path


def load_checkpoint(path: str | Path, *, map_location: str | None = "cpu") -> dict[str, Any]:
    path = Path(path)
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path} is not a dict")
    validate_checkpoint(payload, require_hash=True)
    return payload


def load_for_eval(
    path: str | Path,
    *,
    map_location: str | None = "cpu",
    device: str | torch.device | None = None,
) -> tuple[Any, ObservationNormaliser, dict[str, Any]]:
    """Load a deterministic policy + frozen normaliser for eval / live.

    Validates finite weights, content hash, normaliser presence, and observation
    layout. Returns ``(policy, normaliser, meta)`` where ``policy`` exposes
    ``mean_action(obs)`` and ``meta["has_normaliser"]`` is True.
    """
    # Local import keeps the training hot path free of a hard policy cycle.
    from rallyai.train.policy import ActorCritic

    path = Path(path)
    payload = load_checkpoint(path, map_location=map_location)
    if not payload.get("has_normaliser", False) or "normaliser" not in payload:
        raise ValueError(
            "checkpoint missing normaliser — a policy without it is a different policy"
        )

    obs_dim = int(payload["obs_dim"])
    act_dim = int(payload.get("act_dim", 4))
    policy = ActorCritic(obs_dim=obs_dim, act_dim=act_dim)
    policy.load_state_dict(payload["state_dict"])
    policy.clamp_log_std_()
    policy.eval()
    if device is not None:
        policy = policy.to(device)

    normaliser = ObservationNormaliser.from_dict(payload["normaliser"])
    if normaliser.dim != obs_dim:
        raise ValueError(
            f"normaliser dim {normaliser.dim} != checkpoint obs_dim {obs_dim}"
        )
    normaliser.freeze()

    meta: dict[str, Any] = {
        "has_normaliser": True,
        "policy_sha256": payload["policy_sha256"],
        "timesteps": int(payload.get("timesteps", 0)),
        "tier": int(payload.get("tier", 0)),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "obs_layout_version": int(payload.get("obs_layout_version", OBS_LAYOUT_VERSION)),
        "stage": payload.get("stage"),
        "run_id": payload.get("run_id"),
        "path": str(path),
        "sensor_spec": payload.get("sensor_spec"),
        "reward_config": payload.get("reward_config"),
    }
    return policy, normaliser, meta


class HallOfFame:
    """Five named slots beside a run directory."""

    def __init__(self, directory: str | Path, run_id: str) -> None:
        self.directory = Path(directory)
        self.run_id = str(run_id)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metrics: dict[str, float] = {
            "best": float("-inf"),
            "cleanest": float("-inf"),
            "fastest": float("inf"),  # lower stage time is better
            "furthest": float("-inf"),
        }

    def path_for(self, slot: str) -> Path:
        if slot not in HOF_SLOTS:
            raise ValueError(f"unknown hall-of-fame slot {slot!r}")
        return self.directory / f"{self.run_id}_{slot}.pt"

    def save_slot(self, slot: str, payload: dict[str, Any]) -> Path:
        path = self.path_for(slot)
        save_checkpoint(path, payload)
        return path

    def update(
        self,
        payload: dict[str, Any],
        *,
        mean_return: float | None = None,
        clean_rate: float | None = None,
        mean_finish_time: float | None = None,
        max_progress: float | None = None,
    ) -> list[str]:
        """Write ``latest`` always; update specialty slots when improved.

        Returns the list of slots written.
        """
        written = ["latest"]
        self.save_slot("latest", payload)

        if mean_return is not None and mean_return > self.metrics["best"]:
            self.metrics["best"] = float(mean_return)
            self.save_slot("best", payload)
            written.append("best")

        if clean_rate is not None and clean_rate > self.metrics["cleanest"]:
            self.metrics["cleanest"] = float(clean_rate)
            self.save_slot("cleanest", payload)
            written.append("cleanest")

        if (
            mean_finish_time is not None
            and np.isfinite(mean_finish_time)
            and mean_finish_time < self.metrics["fastest"]
        ):
            self.metrics["fastest"] = float(mean_finish_time)
            self.save_slot("fastest", payload)
            written.append("fastest")

        if max_progress is not None and max_progress > self.metrics["furthest"]:
            self.metrics["furthest"] = float(max_progress)
            self.save_slot("furthest", payload)
            written.append("furthest")

        return written

    def copy_slot(self, src_slot: str, dst_slot: str) -> Path:
        src = self.path_for(src_slot)
        dst = self.path_for(dst_slot)
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        return dst
