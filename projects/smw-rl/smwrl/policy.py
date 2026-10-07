"""Load a policy together with the observation contract it was trained on.

Several command-line tools used to assume every checkpoint consumed four
colour frames (12 channels).  Older, still-useful checkpoints consume four
grayscale frames instead, so evaluating all levels crashed as soon as it met
one.  This module is the single compatibility boundary for every policy
consumer.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from smwrl.wrappers import ObsConfig
from smwrl.actions import ACTION_SCHEMA_SHA256, ACTION_SCHEMA_VERSION, N_ACTIONS

LEGACY_META_NAME = "policy_meta.json"
EMBEDDED_META_NAME = "smwrl_contract.json"


def metadata_path(checkpoint: str | Path) -> Path:
    """Return the contract sidecar belonging to one concrete model file."""
    path = Path(checkpoint)
    if path.suffix != ".zip":
        raise ValueError(f"policy checkpoint path must end in .zip, got {path}")
    return path.with_suffix(".meta.json")


def _existing_metadata_path(checkpoint: Path) -> Path | None:
    """Prefer checkpoint-scoped metadata, with one compatibility fallback.

    Early versions of the contract work wrote one ``policy_meta.json`` per
    directory.  That cannot safely describe ``latest.zip`` and ``best.zip`` at
    the same time, so all new writes are model-scoped while old sidecars remain
    readable.
    """
    scoped = metadata_path(checkpoint)
    if scoped.exists():
        return scoped
    legacy = checkpoint.parent / LEGACY_META_NAME
    return legacy if legacy.exists() else None


def _metadata_payload(obs_config: ObsConfig, frame_stack: int,
                      level: str | None) -> dict:
    return {
        "version": 1,
        "level": level,
        "action_schema": {
            "version": ACTION_SCHEMA_VERSION,
            "sha256": ACTION_SCHEMA_SHA256,
            "count": N_ACTIONS,
        },
        "observation": {
            "color": obs_config.color,
            "crop_top": obs_config.crop_top,
            "size": obs_config.size,
            "frame_stack": frame_stack,
        },
    }


def _read_metadata(checkpoint: Path,
                   checkpoint_bytes: bytes | None = None) -> tuple[dict, str] | None:
    try:
        source = io.BytesIO(checkpoint_bytes) if checkpoint_bytes is not None else checkpoint
        with zipfile.ZipFile(source) as archive:
            if EMBEDDED_META_NAME in archive.namelist():
                return json.loads(archive.read(EMBEDDED_META_NAME)), \
                    f"{checkpoint}!/{EMBEDDED_META_NAME}"
    except (zipfile.BadZipFile, OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"invalid embedded policy metadata in {checkpoint}: {e}") from e
    external = _existing_metadata_path(checkpoint)
    if external is None:
        return None
    try:
        return json.loads(external.read_text()), str(external)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"invalid policy metadata at {external}: {e}") from e


@dataclass(frozen=True)
class PolicyRuntime:
    model: object
    obs_config: ObsConfig
    frame_stack: int
    checkpoint: Path
    model_sha256: str = ""

    def initial_stack(self, obs: np.ndarray) -> np.ndarray:
        # VecFrameStack resets its history to zero and places the first frame in
        # the newest slot.  Repeating the first frame four times at evaluation
        # was an avoidable train/eval distribution shift.
        stack = np.zeros((self.frame_stack, *obs.shape), dtype=obs.dtype)
        stack[-1] = obs
        return stack

    def batch(self, stack: np.ndarray) -> np.ndarray:
        """Convert (time, H, W, channels) into SB3's (1, C, H, W)."""
        h, w = self.obs_config.size, self.obs_config.size
        return stack.transpose(0, 3, 1, 2).reshape(1, -1, h, w)

    @staticmethod
    def advance(stack: np.ndarray, obs: np.ndarray) -> np.ndarray:
        return np.concatenate([stack[1:], obs[None]], axis=0)


def _contract_from_model(model, checkpoint: Path, allow_legacy: bool = False,
                         expected_level: str | None = None,
                         checkpoint_bytes: bytes | None = None) -> tuple[ObsConfig, int]:
    shape = tuple(int(x) for x in model.observation_space.shape)
    if len(shape) != 3 or shape[1] != shape[2]:
        raise ValueError(f"unsupported policy observation shape {shape} in {checkpoint}")
    if getattr(model.action_space, "n", None) != N_ACTIONS:
        raise ValueError(
            f"policy action count {getattr(model.action_space, 'n', None)} does not "
            f"match runtime schema v{ACTION_SCHEMA_VERSION} ({N_ACTIONS})"
        )

    metadata = _read_metadata(checkpoint, checkpoint_bytes)
    if metadata is not None:
        data, metadata_source = metadata
        try:
            if type(data) is not dict:
                raise ValueError("contract root must be a JSON object")
            if type(data.get("version")) is not int or data["version"] != 1:
                raise ValueError(f"unsupported contract version {data.get('version')!r}")
            level = data.get("level")
            if level is not None and (type(level) is not str or not level):
                raise ValueError("contract level must be a non-empty string or null")
            if expected_level is not None and level != expected_level:
                raise ValueError(
                    f"contract belongs to {level!r}, expected {expected_level!r}"
                )
            action = data["action_schema"]
            if (type(action["version"]) is not int
                    or type(action["count"]) is not int
                    or type(action["sha256"]) is not str
                    or action["version"] != ACTION_SCHEMA_VERSION
                    or action["sha256"] != ACTION_SCHEMA_SHA256
                    or action["count"] != N_ACTIONS):
                raise ValueError("checkpoint action schema does not match this runtime")
            ob = data["observation"]
            if (type(ob["color"]) is not bool or type(ob["crop_top"]) is not int
                    or type(ob["size"]) is not int
                    or type(ob["frame_stack"]) is not int):
                raise ValueError("observation contract fields have invalid types")
            cfg = ObsConfig(color=ob["color"], crop_top=ob["crop_top"], size=ob["size"])
            stack = ob["frame_stack"]
            if cfg.crop_top < 0 or cfg.size <= 0 or stack <= 0:
                raise ValueError("observation contract dimensions must be positive")
            expected = cfg.channels * stack
            if expected != shape[0] or cfg.size != shape[1]:
                raise ValueError(
                    f"{metadata_source} describes {(expected, cfg.size, cfg.size)} but "
                    f"the checkpoint expects {shape}"
                )
            return cfg, stack
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"invalid policy metadata at {metadata_source}: {e}") from e

    if not allow_legacy:
        raise ValueError(
            f"missing checkpoint contract {metadata_path(checkpoint)}; refusing to "
            "guess an action mapping for a metadata-free model"
        )

    channels = shape[0]
    if channels % 3 == 0:
        # All colour checkpoints in this project were trained after the HUD crop
        # became the default.
        return ObsConfig(color=True, crop_top=32, size=shape[1]), channels // 3
    # Legacy checkpoints predate metadata and used grayscale with the full frame.
    return ObsConfig(color=False, crop_top=0, size=shape[1]), channels


def load_policy_runtime(checkpoint: str | Path, device: str = "cpu",
                        allow_legacy: bool = False,
                        expected_level: str | None = None) -> PolicyRuntime:
    from stable_baselines3 import PPO

    path = Path(checkpoint)
    checkpoint_bytes = path.read_bytes()
    model = PPO.load(io.BytesIO(checkpoint_bytes), device=device)
    cfg, stack = _contract_from_model(
        model, path, allow_legacy=allow_legacy, expected_level=expected_level,
        checkpoint_bytes=checkpoint_bytes,
    )
    return PolicyRuntime(
        model, cfg, stack, path, hashlib.sha256(checkpoint_bytes).hexdigest()
    )


def write_policy_metadata(checkpoint: str | Path, obs_config: ObsConfig,
                          frame_stack: int = 4, level: str | None = None) -> None:
    """Atomically record the observation contract for newly trained policies."""
    path = metadata_path(checkpoint)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _metadata_payload(obs_config, frame_stack, level)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def publish_policy(model, checkpoint: str | Path, obs_config: ObsConfig,
                   frame_stack: int = 4, level: str | None = None) -> None:
    """Publish one self-describing SB3 zip atomically."""
    destination = Path(checkpoint)
    metadata_path(destination)  # validate before any write
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + ".tmp.zip")
    model.save(tmp)
    payload = _metadata_payload(obs_config, frame_stack, level)
    with zipfile.ZipFile(tmp, "a", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(EMBEDDED_META_NAME, json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, destination)
    # Keep a human-readable sidecar too.  Loading prefers the contract embedded
    # in the atomically published model, so a sidecar write failure cannot make
    # the checkpoint ambiguous.
    try:
        write_policy_metadata(destination, obs_config, frame_stack, level)
    except OSError as e:
        warnings.warn(f"published embedded policy contract but could not update sidecar: {e}")


def copy_policy_metadata(source: str | Path, destination: str | Path) -> bool:
    """Copy a model's sidecar atomically; return False for metadata-free legacy models."""
    source_path = _existing_metadata_path(Path(source))
    if source_path is None:
        # A newly copied legacy model must never inherit a sidecar belonging to
        # the previous destination model.
        metadata_path(destination).unlink(missing_ok=True)
        return False
    destination_path = metadata_path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination_path.with_suffix(destination_path.suffix + ".tmp")
    tmp.write_bytes(source_path.read_bytes())
    os.replace(tmp, destination_path)
    return True
