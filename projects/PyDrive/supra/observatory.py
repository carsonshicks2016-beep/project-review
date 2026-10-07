"""Authoritative, headless playback for the Fable Five Observatory.

The browser is deliberately absent from this module.  A playback session owns
the Python environment, observation normalizer, deterministic policy and
RaceBox.  Callers may serialize the returned dictionaries, but may not supply
physics state or policy observations back to the session.

Only paths that have already passed the server-side catalogue resolver should
be wrapped in :class:`ValidatedCheckpoint`.  This module still re-checks the
checkpoint payload so a stale catalogue entry fails closed when it is loaded.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import copy
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
from pathlib import Path
import struct
import threading
import time
from typing import Any, Callable, Mapping
import uuid

import numpy as np
import torch
import torch.nn as nn

from .config import PPOSpec, SensorSpec, SimSpec, get_car
from .fable_editions import (
    CheckpointRecord,
    FABLE_EDITIONS,
    FABLE_STAGES as REGISTERED_FABLE_STAGES,
    FABLE_TRACK,
    FableEdition,
    get_edition,
    require_playable_checkpoint,
    resolve_checkpoint_path,
)
from .fable5 import (
    FableEnv,
    attach_envelope,
    stage_defaults,
)
from .ppo import PPO, state_dict_hash
from .sensors import Observation, SensorSuite
from .track import named_track


SUPPORTED_CARS = frozenset(edition.car_id for edition in FABLE_EDITIONS.values())
SUPPORTED_SPEEDS = (0.25, 0.5, 1.0, 2.0)
FABLE_STAGES = frozenset(REGISTERED_FABLE_STAGES)
ACTION_NAMES = ("steering", "longitudinal", "gear_offset")
PROPRIO_LABELS = (
    "vx", "vy", "speed", "yaw_rate", "ax", "ay", "slip_angle",
    "heading_error", "lateral", "steer", "rpm", "gear", "boost",
    "grip_FL", "grip_FR", "grip_RL", "grip_RR", "Fz_FL", "Fz_FR",
    "Fz_RL", "Fz_RR", "sr_FL", "sr_FR", "sr_RL", "sr_RR",
)


class ObservatoryError(RuntimeError):
    """Base exception for a playback service failure."""


class CheckpointCompatibilityError(ObservatoryError, ValueError):
    """The checkpoint cannot be reconstructed by this playback runtime."""


class SessionLimitError(ObservatoryError):
    """The global or per-browser Observatory session cap was reached."""


class SessionClosedError(ObservatoryError):
    """The requested session has already released its resources."""


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservatoryError(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise ObservatoryError(f"{name} is not finite")
    return result


def _json_safe(value: Any) -> Any:
    """Convert NumPy/Torch values and reject non-finite telemetry."""
    if isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise ObservatoryError("telemetry array contains a non-finite value")
        return value.tolist()
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().numpy())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ObservatoryError("telemetry contains a non-finite value")
        return value
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    raise ObservatoryError(f"telemetry contains unsupported type {type(value).__name__}")


def frame_sha256(frame: Mapping[str, Any]) -> str:
    """Stable hash used by deterministic playback acceptance tests."""
    payload = json.dumps(_json_safe(frame), sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def runtime_track_sha256(track) -> str:
    """Hash the processed simulation-truth road used by playback and assets."""
    digest = hashlib.sha256()
    digest.update(b"fable-observatory-track-v1\0")
    for name in ("center", "arc", "z", "half_width", "bank"):
        array = np.asarray(getattr(track, name), dtype="<f8")
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(array.shape).encode("ascii") + b"\0")
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


_VALIDATED_CHECKPOINT_FACTORY_KEY = object()


@dataclass(frozen=True, init=False)
class ValidatedCheckpoint:
    """An immutable checkpoint identity created from a playable registry record.

    The constructor is intentionally sealed.  Production callers must first use
    :func:`supra.fable_editions.require_playable_checkpoint` (or the resolver)
    and then call :meth:`from_record`.  This keeps strings, paths, mappings and
    hand-built lookalikes out of the playback boundary.
    """

    path: Path | str
    edition: str
    car: str
    checkpoint_id: str | None = None
    file_sha256: str | None = None
    policy_sha256: str | None = None
    classification: str = "current"
    compatibility_class: str = "fable-playback"
    warnings: tuple[str, ...] = ()
    _factory_key: object = field(init=False, repr=False, compare=False)

    def __init__(
        self,
        path: Path | str,
        edition: str,
        car: str,
        checkpoint_id: str | None = None,
        file_sha256: str | None = None,
        policy_sha256: str | None = None,
        classification: str = "current",
        compatibility_class: str = "fable-playback",
        warnings: tuple[str, ...] = (),
        *,
        _factory_key: object | None = None,
    ) -> None:
        if _factory_key is not _VALIDATED_CHECKPOINT_FACTORY_KEY:
            raise CheckpointCompatibilityError(
                "ValidatedCheckpoint must be created from a playable Fable "
                "edition registry record; arbitrary paths and mappings are rejected"
            )
        try:
            selected = get_edition(str(edition))
        except KeyError as exc:
            raise CheckpointCompatibilityError(str(exc)) from exc
        resolved = Path(path).expanduser().resolve()
        normalized_car = str(car).lower()
        normalized_classification = str(classification)
        if normalized_car != selected.car_id:
            raise CheckpointCompatibilityError(
                "validated checkpoint car does not match its registry edition"
            )
        if compatibility_class != selected.compatibility_class:
            raise CheckpointCompatibilityError(
                "validated checkpoint compatibility class does not match its registry edition"
            )
        if normalized_classification not in ("current", "historical"):
            raise CheckpointCompatibilityError(
                f"checkpoint classification {normalized_classification!r} is not playable"
            )
        object.__setattr__(self, "path", resolved)
        object.__setattr__(self, "edition", selected.id)
        object.__setattr__(self, "car", selected.car_id)
        object.__setattr__(self, "checkpoint_id", checkpoint_id or resolved.name)
        object.__setattr__(self, "file_sha256", file_sha256)
        object.__setattr__(self, "policy_sha256", policy_sha256)
        object.__setattr__(self, "classification", normalized_classification)
        object.__setattr__(self, "compatibility_class", selected.compatibility_class)
        object.__setattr__(self, "warnings", tuple(str(w) for w in warnings))
        object.__setattr__(self, "_factory_key", _factory_key)

    @classmethod
    def from_record(cls, checkpoint_root: str | Path, record: CheckpointRecord, *,
                    resolution_warnings=()) -> "ValidatedCheckpoint":
        """Bridge an authoritative ``fable_editions.CheckpointRecord``."""
        if not isinstance(record, CheckpointRecord):
            raise CheckpointCompatibilityError(
                "checkpoint identity must be a Fable edition registry record"
            )
        if not record.playable:
            raise CheckpointCompatibilityError("checkpoint record is not playable")
        if not record.file_sha256:
            raise CheckpointCompatibilityError(
                "playable registry record is missing its checkpoint content hash"
            )
        try:
            selected = get_edition(record.edition)
        except KeyError as exc:
            raise CheckpointCompatibilityError(str(exc)) from exc
        authoritative = require_playable_checkpoint(
            checkpoint_root, selected, record.name
        )
        if authoritative != record:
            raise CheckpointCompatibilityError(
                "checkpoint registry record is stale or was not produced by the "
                "current edition classifier"
            )
        record = authoritative
        expected = {
            "car": (record.car_id, selected.car_id),
            "drivetrain": (record.drivetrain, selected.drivetrain),
            "observation layout": (
                record.observation_layout, selected.observation_layout,
            ),
            "observation size": (record.obs_dim, selected.obs_dim),
            "sensor size": (record.sdim, selected.sdim),
            "action size": (record.act_dim, selected.act_dim),
            "compatibility class": (
                record.compatibility_class, selected.compatibility_class,
            ),
        }
        mismatched = [name for name, (actual, wanted) in expected.items()
                      if actual != wanted]
        if mismatched:
            raise CheckpointCompatibilityError(
                "playable registry record disagrees with its edition: "
                + ", ".join(mismatched)
            )
        name = str(record.name)
        path = resolve_checkpoint_path(checkpoint_root, name)
        warnings = (*tuple(record.warnings or ()),
                    *tuple(resolution_warnings or ()))
        return cls(
            path=path,
            edition=selected.id,
            car=selected.car_id,
            checkpoint_id=name,
            file_sha256=record.file_sha256,
            policy_sha256=record.policy_sha256,
            classification=record.classification,
            compatibility_class=selected.compatibility_class,
            warnings=warnings,
            _factory_key=_VALIDATED_CHECKPOINT_FACTORY_KEY,
        )


def _require_registry_checkpoint(value: Any) -> ValidatedCheckpoint:
    if (not isinstance(value, ValidatedCheckpoint)
            or value._factory_key is not _VALIDATED_CHECKPOINT_FACTORY_KEY):
        raise CheckpointCompatibilityError(
            "playback requires a registry-created ValidatedCheckpoint; "
            "arbitrary strings, paths and mappings are rejected"
        )
    return value


@dataclass(frozen=True)
class CheckpointIdentity:
    checkpoint_id: str
    filename: str
    edition: str
    car: str
    stage: str
    drivetrain: str
    observation_layout: str
    compatibility_class: str
    classification: str
    file_sha256: str
    policy_sha256: str
    runtime_contract_sha256: str
    warnings: tuple[str, ...]

    def public(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "filename": self.filename,
            "edition": self.edition,
            "car": self.car,
            "stage": self.stage,
            "drivetrain": self.drivetrain,
            "observation_layout": self.observation_layout,
            "compatibility_class": self.compatibility_class,
            "classification": self.classification,
            "file_sha256": self.file_sha256,
            "policy_sha256": self.policy_sha256,
            "runtime_contract_sha256": self.runtime_contract_sha256,
            "warnings": list(self.warnings),
        }


@dataclass
class PlaybackRuntime:
    """Freshly reconstructed state needed to run one checkpoint truthfully."""

    sensor_spec: SensorSpec
    ppo_spec: PPOSpec
    fable_spec: Any
    sim_spec: SimSpec
    track_hash: str
    env: FableEnv


def _copy_ppo_spec(meta: Mapping[str, Any], sensor_spec: SensorSpec) -> PPOSpec:
    """Rebuild the policy cadence/sensor contract without training randomness."""
    cfg = PPOSpec()
    raw = meta.get("ppo_config")
    if isinstance(raw, Mapping):
        allowed = set(PPOSpec.__dataclass_fields__)
        for key, value in raw.items():
            if key not in allowed:
                continue
            current = getattr(cfg, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(cfg, key, value)
    cfg.hidden = tuple(int(v) for v in meta["hidden"])
    cfg.control_hz = int((raw or {}).get("control_hz", cfg.control_hz))
    cfg.episode_seconds = float(
        (raw or {}).get("episode_seconds", meta.get("episode_seconds", cfg.episode_seconds))
    )
    cfg.random_start = False
    cfg.track_pool = 1
    cfg.n_envs = 1
    cfg.n_workers = 1
    cfg.sensor_lookahead_distances = tuple(sensor_spec.lookahead_distances)
    cfg.sensor_pace_block = bool(sensor_spec.pace_block)
    cfg.sensor_pace_distances = tuple(sensor_spec.pace_distances)
    cfg.sensor_hybrid_block = bool(sensor_spec.hybrid_block)
    if cfg.control_hz <= 0 or cfg.control_hz > 120:
        raise CheckpointCompatibilityError("invalid checkpoint policy cadence")
    return cfg


def _sensor_spec(meta: Mapping[str, Any]) -> SensorSpec:
    spec = SensorSpec()
    distances = meta.get("sensor_lookahead_distances")
    if distances:
        spec.lookahead_distances = tuple(float(v) for v in distances)
    spec.pace_block = bool(meta.get("sensor_pace_block"))
    pace = meta.get("sensor_pace_distances")
    if pace:
        spec.pace_distances = tuple(float(v) for v in pace)
    spec.hybrid_block = bool(meta.get("sensor_hybrid_block"))
    return spec


def _sha256_file(path: Path) -> str:
    """Return the exact byte identity of a registry-approved checkpoint."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_contract_sha256(edition: FableEdition, meta: Mapping[str, Any]) -> str:
    """Hash every checkpoint field that can change reconstructed playback.

    The policy hash intentionally does not appear here: identical weights can
    still require a different normalizer, stage envelope, RaceBox or episode
    setup.  The descriptor's file hash is the complete byte identity; this
    companion hash lets clients and follow-mode logs name the runtime contract
    that was reconstructed from those bytes.
    """
    ppo_config = meta.get("ppo_config")
    payload = {
        "schema": "fable-observatory-runtime-contract-v1",
        "edition": edition.id,
        "car": edition.car_id,
        "drivetrain": edition.drivetrain,
        "observation_layout": edition.observation_layout,
        "track": meta.get("track"),
        "stage": meta.get("fable_stage"),
        "fable_envelope_scale": meta.get("fable_envelope_scale"),
        "fable_shift_lo_frac": meta.get("fable_shift_lo_frac"),
        "sensor_lookahead_distances": meta.get("sensor_lookahead_distances"),
        "sensor_pace_distances": meta.get("sensor_pace_distances"),
        "sensor_pace_block": meta.get("sensor_pace_block"),
        "sensor_hybrid_block": meta.get("sensor_hybrid_block"),
        "obs_dim": meta.get("obs_dim"),
        "sdim": meta.get("sdim"),
        "act_dim": meta.get("act_dim"),
        "ppo_config": dict(ppo_config) if isinstance(ppo_config, Mapping) else {},
    }
    encoded = json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_contract(
    descriptor: ValidatedCheckpoint, meta: Mapping[str, Any]
) -> FableEdition:
    descriptor = _require_registry_checkpoint(descriptor)
    try:
        edition = get_edition(descriptor.edition)
    except KeyError as exc:
        raise CheckpointCompatibilityError(str(exc)) from exc
    car = str(meta.get("car") or "").lower()
    if descriptor.car != edition.car_id or car != edition.car_id:
        raise CheckpointCompatibilityError(
            f"checkpoint car {car!r} does not match registry car {edition.car_id!r}"
        )
    if meta.get("track") != FABLE_TRACK:
        raise CheckpointCompatibilityError("Observatory playback is Nordschleife-only")
    if meta.get("fable_pipeline") is not True:
        raise CheckpointCompatibilityError("checkpoint is not a Fable Five policy")
    if meta.get("mode") != "race":
        raise CheckpointCompatibilityError("Fable playback requires race-mode PPO")
    stage = str(meta.get("fable_stage") or "")
    if stage not in FABLE_STAGES:
        raise CheckpointCompatibilityError(f"invalid or missing Fable stage {stage!r}")
    if meta.get("obs_layout") != edition.observation_layout:
        raise CheckpointCompatibilityError(
            f"{edition.id} requires observation layout {edition.observation_layout}"
        )
    if not meta.get("sensor_pace_block"):
        raise CheckpointCompatibilityError("Fable pace observations are missing")
    if bool(meta.get("sensor_hybrid_block")) != edition.hybrid_observations:
        raise CheckpointCompatibilityError(
            "checkpoint hybrid observation block does not match the edition registry"
        )
    if edition.act_dim != len(ACTION_NAMES):
        raise CheckpointCompatibilityError(
            "edition registry action size is incompatible with Observatory RaceBox"
        )
    if int(meta.get("act_dim", 0)) != edition.act_dim:
        raise CheckpointCompatibilityError(
            f"checkpoint action size does not match registry size {edition.act_dim}"
        )
    if tuple(int(v) for v in meta.get("hidden", ())) != (128, 128):
        raise CheckpointCompatibilityError("Observatory requires the Fable 128x128 brain")
    spec = _sensor_spec(meta)
    reconstructed_sdim = SensorSuite(spec).obs_size
    if (int(meta.get("sdim", 0)) != edition.sdim
            or reconstructed_sdim != edition.sdim
            or int(meta.get("obs_dim", 0)) != edition.obs_dim
            or edition.obs_dim != edition.sdim + 2):
        raise CheckpointCompatibilityError(
            "checkpoint sensor/hybrid dimensions do not match the edition registry"
        )
    drivetrain = str(meta.get("fable_drivetrain_version") or "")
    runtime_drivetrain = str(
        getattr(get_car(edition.car_id), "drivetrain_version", "")
    )
    if runtime_drivetrain != edition.drivetrain:
        raise CheckpointCompatibilityError(
            "runtime car drivetrain has drifted from the Fable edition registry"
        )
    if not drivetrain or drivetrain != edition.drivetrain:
        raise CheckpointCompatibilityError(
            f"checkpoint drivetrain {drivetrain!r} cannot be reconstructed as "
            f"{edition.drivetrain!r}"
        )
    scale = float(meta.get("fable_envelope_scale", math.nan))
    shift = float(meta.get("fable_shift_lo_frac", math.nan))
    if not math.isfinite(scale) or not 0.4 <= scale <= 1.3:
        raise CheckpointCompatibilityError("invalid Fable envelope scale")
    if not math.isfinite(shift) or not 0.35 <= shift <= 0.95:
        raise CheckpointCompatibilityError("invalid Fable shift behavior")
    return edition


@dataclass
class ProbeSnapshot:
    action: np.ndarray
    normalized: np.ndarray
    observations: dict[str, Any]
    brain: dict[str, Any]


class BrainProbe:
    """Read-only policy instrumentation; it never holds training references."""

    def __init__(self, net, norm, meta: Mapping[str, Any], sensor_spec: SensorSpec,
                 *, control_hz: int, sensitivity_hz: float = 5.0,
                 prediction_seconds: float = 1.0,
                 prediction_mode: str = "realtime"):
        if prediction_mode not in ("realtime", "deterministic"):
            raise ValueError("prediction_mode must be realtime or deterministic")
        self.net = net
        self.norm = norm
        self.meta = dict(meta)
        self.sensor_spec = copy.deepcopy(sensor_spec)
        self.sdim = int(meta["sdim"])
        self.control_hz = int(control_hz)
        self.sensitivity_interval = max(1, int(round(control_hz / sensitivity_hz)))
        self.prediction_seconds = float(prediction_seconds)
        self.prediction_mode = prediction_mode
        self._cached_sensitivity: dict[str, Any] | None = None
        self._cached_path: list[dict[str, float]] = []
        self._cached_path_sequence: int | None = None
        self._feature_rows = self._build_feature_rows()
        self._prediction_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="observatory-brain-probe"
        )
        self._prediction_future: Future | None = None
        self._prediction_source_sequence: int | None = None
        self._prediction_error: str | None = None

    def normalize(self, full_observation: np.ndarray) -> np.ndarray:
        obs = np.asarray(full_observation, dtype=np.float32).reshape(-1)
        if obs.shape != (int(self.meta["obs_dim"]),):
            raise ObservatoryError(
                f"observation shape {obs.shape} does not match policy "
                f"({self.meta['obs_dim']},)"
            )
        sensors = self.norm.normalize(obs[:self.sdim])
        normalized = np.concatenate([sensors, obs[self.sdim:]]).astype(np.float32)
        if not np.all(np.isfinite(normalized)):
            raise ObservatoryError("normalized observation is not finite")
        return normalized

    def _build_feature_rows(self) -> list[dict[str, Any]]:
        s = self.sensor_spec
        rows: list[dict[str, Any]] = []

        def add(group: str, labels) -> None:
            for label in labels:
                rows.append({"index": len(rows), "group": group,
                             "label": str(label)})

        ray_angles = np.linspace(s.beam_spread_deg, -s.beam_spread_deg, s.n_beams)
        add("rays", (f"ray_{angle:+.1f}deg" for angle in ray_angles))
        add("proprioception", PROPRIO_LABELS)
        add("curvature", (f"curvature_{d:g}m" for d in s.lookahead_distances))
        if s.hill_block:
            add("hill_air", ("grade", "bank", "pitch", "vertical_speed",
                             "height_above_road", "airborne"))
            add("hill_air", (f"grade_{d:g}m" for d in s.lookahead_distances))
            add("hill_air", (f"crest_curvature_{d:g}m"
                             for d in s.lookahead_distances))
        if s.pace_block:
            add("pace", (f"vref_{d:g}m" for d in s.pace_distances))
            add("pace", ("speed_to_envelope",))
        if s.hybrid_block:
            add("hybrid", ("battery_soc", "signed_mgu_power"))
        if len(rows) != self.sdim:
            raise CheckpointCompatibilityError(
                f"feature schema has {len(rows)} sensors, checkpoint has {self.sdim}"
            )
        add("mode", ("race_mode", "drift_mode"))
        return rows

    def _forward(self, normalized: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], float, np.ndarray]:
        with torch.no_grad():
            value = torch.as_tensor(normalized, dtype=torch.float32).unsqueeze(0)
            layers: list[np.ndarray] = []
            for module in self.net.trunk:
                value = module(value)
                if isinstance(module, nn.Tanh):
                    layers.append(value.squeeze(0).detach().cpu().numpy().copy())
            mean = self.net.mean(value)
            critic = self.net.value(value).squeeze(-1)
            std = self.net._dist(mean).stddev
        if len(layers) != 2 or any(layer.shape != (128,) for layer in layers):
            raise CheckpointCompatibilityError("brain probe expected two 128-unit layers")
        return (
            mean.squeeze(0).detach().cpu().numpy().astype(np.float32),
            layers,
            float(critic.item()),
            std.squeeze(0).detach().cpu().numpy().astype(np.float32),
        )

    def _groups(self, raw: np.ndarray, normalized: np.ndarray,
                observation: Observation) -> dict[str, Any]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in self._feature_rows:
            idx = row["index"]
            grouped.setdefault(row["group"], []).append({
                "index": idx,
                "label": row["label"],
                "value": float(raw[idx]),
                "policy_value": float(normalized[idx]),
            })
        names = {
            "rays": "Rays",
            "proprioception": "Proprioception",
            "curvature": "Curvature",
            "hill_air": "Hill / Air",
            "pace": "Pace Envelope",
            "hybrid": "919 Hybrid",
            "mode": "Mode",
        }
        result = {
            key: {"name": names[key], "values": values}
            for key, values in grouped.items()
        }
        result["rays"]["raw_distances_m"] = _json_safe(observation.beams)
        result["curvature"]["raw_per_m"] = _json_safe(observation.lookahead)
        if "hill_air" in result:
            result["hill_air"]["raw_local"] = {
                str(label): float(value) for label, value in observation.hill_labels
            }
            result["hill_air"]["raw_grade_preview"] = _json_safe(
                observation.lookahead_grade
            )
            result["hill_air"]["raw_crest_preview"] = _json_safe(
                observation.lookahead_vcurv
            )
        return result

    def _sensitivities(self, normalized: np.ndarray, *, top_k: int = 8,
                       epsilon: float = 1e-3) -> dict[str, Any]:
        # Central finite differences are vectorized into one network batch.
        n = normalized.size
        plus = np.repeat(normalized[None, :], n, axis=0)
        minus = plus.copy()
        idx = np.arange(n)
        plus[idx, idx] += epsilon
        minus[idx, idx] -= epsilon
        batch = np.concatenate([plus, minus], axis=0)
        with torch.no_grad():
            means, _ = self.net(torch.as_tensor(batch, dtype=torch.float32))
        out = means.detach().cpu().numpy()
        deriv = (out[:n] - out[n:]) / (2.0 * epsilon)
        per_action: dict[str, list[dict[str, Any]]] = {}
        continuous_inputs = self.sdim  # do not perturb the categorical mode bits
        for action_i, action_name in enumerate(ACTION_NAMES):
            order = np.argsort(np.abs(deriv[:continuous_inputs, action_i]))[::-1][:top_k]
            per_action[action_name] = [
                {
                    "index": int(i),
                    "group": self._feature_rows[int(i)]["group"],
                    "label": self._feature_rows[int(i)]["label"],
                    "derivative": float(deriv[int(i), action_i]),
                    "magnitude": float(abs(deriv[int(i), action_i])),
                }
                for i in order
            ]
        return {
            "label": "Local finite-difference sensitivity, not intent",
            "basis": "derivative of policy mean with respect to normalized input",
            "epsilon": epsilon,
            "top": per_action,
        }

    @staticmethod
    def _detached_environment(env: FableEnv) -> FableEnv:
        # The track and static specs are immutable during stepping.  Sharing them
        # avoids copying Nordschleife arrays while every mutable simulation object
        # (vehicle, RaceBox, counters, observations and RNG) is detached.
        shared = (env.trk, env.fixed_track, env.sensors, env.fable,
                  env.spec, env.sim, env.ppo)
        memo = {id(item): item for item in shared if item is not None}
        return copy.deepcopy(env, memo)

    def _predict_from_clone(self, clone: FableEnv) -> list[dict[str, float]]:
        points = [{
            "x": float(clone.veh.x), "y": float(clone.veh.y),
            "z": float(clone.veh.z), "yaw": float(clone.veh.yaw),
            "t": 0.0,
        }]
        ticks = max(1, int(round(self.prediction_seconds * self.control_hz)))
        start_t = float(clone.t)
        full = np.concatenate(
            [clone.last_obs.vector, clone._mode_vec]
        ).astype(np.float32)
        for _ in range(ticks):
            normalized = self.normalize(full)
            action, _, _, _ = self._forward(normalized)
            full, _, terminated, truncated, _ = clone.step(action)
            points.append({
                "x": float(clone.veh.x), "y": float(clone.veh.y),
                "z": float(clone.veh.z), "yaw": float(clone.veh.yaw),
                "t": float(clone.t - start_t),
            })
            if terminated or truncated:
                break
        return points

    def predict_path(self, env: FableEnv) -> list[dict[str, float]]:
        """Synchronous probe used by tests and the first session snapshot."""
        return self._predict_from_clone(self._detached_environment(env))

    def _schedule_prediction(self, env: FableEnv, sequence: int) -> None:
        # Detach while the session lock is held.  The worker receives no
        # reference to live mutable physics state.
        if self._prediction_future is not None:
            return
        clone = self._detached_environment(env)
        self._prediction_future = self._prediction_executor.submit(
            self._predict_from_clone, clone
        )
        self._prediction_source_sequence = int(sequence)

    def _publish_completed_prediction(self) -> bool:
        """Publish a rollout only after it has completed; never wait for it."""
        future = self._prediction_future
        if future is None or not future.done():
            return False
        source_sequence = self._prediction_source_sequence
        self._prediction_future = None
        self._prediction_source_sequence = None
        try:
            self._cached_path = future.result()
            self._cached_path_sequence = source_sequence
            self._prediction_error = None
            return True
        except Exception as exc:
            # Prediction is observability-only; keep the last valid path and
            # surface the failure instead of interrupting authoritative ticks.
            self._prediction_error = type(exc).__name__
            return False

    def _prediction_metadata(self, sequence: int) -> dict[str, Any]:
        source = self._cached_path_sequence
        age_ticks = max(0, int(sequence) - source) if source is not None else None
        stale = bool(age_ticks is None or age_ticks > self.sensitivity_interval * 2)
        pending = self._prediction_future is not None
        if self._prediction_error is not None:
            status = "error"
        elif not self._cached_path:
            status = "unavailable"
        elif stale:
            status = "stale"
        elif pending:
            status = "pending"
        else:
            status = "fresh"
        return {
            "predicted_path_source_sequence": source,
            "predicted_path_age_ticks": age_ticks,
            "predicted_path_age_seconds": (
                None if age_ticks is None else age_ticks / self.control_hz
            ),
            "predicted_path_pending": pending,
            "predicted_path_stale": stale,
            "predicted_path_status": status,
            "predicted_path_error": self._prediction_error,
        }

    def invalidate_prediction(self) -> None:
        """Discard probe output tied to the previous simulation state.

        A reset replaces the vehicle and its observation state atomically.  A
        detached rollout still in flight describes the old episode, so it must
        never be published into the reset snapshot.  Cancellation is best
        effort: a running worker may finish, but clearing our reference means
        its result cannot be consumed by a later capture.
        """
        future = self._prediction_future
        self._prediction_future = None
        if future is not None:
            future.cancel()
        self._cached_path = []
        self._cached_path_sequence = None
        self._prediction_source_sequence = None
        self._prediction_error = None

    def capture(self, full_observation: np.ndarray, observation: Observation,
                env: FableEnv, sequence: int, *, force_expensive: bool = False) -> ProbeSnapshot:
        raw = np.asarray(full_observation, dtype=np.float32).reshape(-1)
        normalized = self.normalize(raw)
        mean, layers, critic, std = self._forward(normalized)
        if self.prediction_mode == "realtime":
            self._publish_completed_prediction()
        expensive = (force_expensive or self._cached_sensitivity is None
                     or sequence % self.sensitivity_interval == 0)
        if expensive:
            self._cached_sensitivity = self._sensitivities(normalized)
            if force_expensive and not self._cached_path:
                self._cached_path = self.predict_path(env)
                self._cached_path_sequence = int(sequence)
                self._prediction_error = None
            elif self.prediction_mode == "deterministic":
                self._cached_path = self.predict_path(env)
                self._cached_path_sequence = int(sequence)
                self._prediction_error = None
            else:
                # The projection is display-only.  Do not wait under the
                # simulation lock: publish a completed prior rollout and let
                # the browser see a bounded freshness indicator meanwhile.
                self._publish_completed_prediction()
                self._schedule_prediction(env, sequence)
        brain = {
            "hidden_layers": [_json_safe(layer) for layer in layers],
            "critic_value": critic,
            "policy_std": _json_safe(std),
            "output_means": _json_safe(mean),
            "action_names": list(ACTION_NAMES),
            "sensitivity": copy.deepcopy(self._cached_sensitivity),
            "predicted_path": copy.deepcopy(self._cached_path),
            "predicted_path_seconds": self.prediction_seconds,
            **self._prediction_metadata(sequence),
        }
        return ProbeSnapshot(
            action=mean,
            normalized=normalized,
            observations={
                "raw_vector": _json_safe(raw),
                "normalized_vector": _json_safe(normalized),
                "groups": self._groups(raw, normalized, observation),
            },
            brain=brain,
        )

    def close(self) -> None:
        future = self._prediction_future
        if future is not None:
            future.cancel()
        self._prediction_executor.shutdown(wait=False, cancel_futures=True)
        self._prediction_future = None
        self._prediction_source_sequence = None


@dataclass(frozen=True)
class AudioPCMChunk:
    sequence: int
    sim_time: float
    sample_rate: int
    frames: int
    channels: int
    pcm: bytes

    HEADER = struct.Struct("<4sHHIIQd")

    def wire_bytes(self) -> bytes:
        """32-byte little-endian header followed by interleaved signed-16 PCM."""
        return self.HEADER.pack(
            b"FOA1", 1, self.channels, self.sample_rate, self.frames,
            self.sequence, self.sim_time,
        ) + self.pcm


class ObservatoryAudioRenderer:
    """Scene-aware, telemetry-driven Observatory sound renderer.

    Vehicle voices share the proven procedural stems with ``viewer2`` while
    this renderer adds camera-relative propagation, early reflections and a
    deterministic Nordschleife atmosphere. It remains downstream of the live
    simulator and cannot modify physics or policy state.
    """

    sample_rate = 48_000
    chunk_frames = 1_024
    channels = 2

    def __init__(self, car: str, seed: int):
        from .audio_v4 import (
            AudioTelemetryAdapter,
            DCBlocker,
            ObservatoryAmbience,
            SoftLimiter,
            SpatialRendererV4,
            VIEWER_AUDIO_SEED,
        )

        self.car = car
        self.session_seed = int(seed)
        self.audio_seed = VIEWER_AUDIO_SEED
        self.perspective = 0
        self.source = "observatory-immersive-audio-v1"
        self.adapter = AudioTelemetryAdapter()
        self.renderer = SpatialRendererV4(
            self.sample_rate, 1, seed=self.audio_seed, car_names=[car]
        )
        self.ambience = ObservatoryAmbience(
            self.sample_rate, seed=self.audio_seed + self.session_seed + 31_000
        )
        self.output_dc = DCBlocker()
        self.output_limiter = SoftLimiter(self.sample_rate, threshold=0.76)
        self.browser_listener = (0.0, 0.0, 0.0, 0.0, 0.0)
        self.listener_target = self.browser_listener
        self.listener_z = 0.0
        self.listener_z_target = 0.0
        self.listener_received = False
        self.camera = "broadcast"
        self.weather = "dry"
        self.frame = None
        self.sequence = 0
        self.sim_time = 0.0
        self.cut = 0
        self._previous_time = 0.0
        self._previous_load = np.ones(4, dtype=np.float64)
        self._brake_temperature = np.full(4, 0.2, dtype=np.float64)
        self._previous_landing = 0.0
        self._previous_damage = 0.0
        self._previous_surface = 0
        self._lock = threading.RLock()

    def reset(self, vehicle) -> None:
        with self._lock:
            self.adapter.reset([vehicle])
            self.frame = self.adapter.frame(vehicle)
            self.renderer.reset([self.frame])
            self.ambience.reset()
            self.output_dc = type(self.output_dc)()
            self.output_limiter = type(self.output_limiter)(
                self.sample_rate, threshold=0.76
            )
            self.browser_listener = (
                float(vehicle.x), float(vehicle.y), 0.0, 0.0, float(vehicle.yaw)
            )
            self.listener_target = self.browser_listener
            self.listener_z = float(getattr(vehicle, "z", 0.0))
            self.listener_z_target = self.listener_z
            self.listener_received = False
            self._previous_time = 0.0
            self._previous_load.fill(1.0)
            self._brake_temperature.fill(0.2)
            self._previous_landing = 0.0
            self._previous_damage = 0.0
            self._previous_surface = 0
            self.cut += 1

    def update(self, vehicle, controls: Mapping[str, Any], *, sequence: int,
               sim_time: float, scene: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            scene = scene or {}
            throttle = float(controls.get("throttle", 0.0))
            brake = float(controls.get("brake", 0.0))
            clutch = float(controls.get("clutch", 1.0))
            frame = self.adapter.frame(
                vehicle, throttle=throttle, brake=brake, clutch=clutch,
            )
            # Vehicle vx/vy are body-frame values; spatial propagation consumes
            # world-frame velocity.
            c, s = math.cos(frame.yaw), math.sin(frame.yaw)
            dt = float(np.clip(float(sim_time) - self._previous_time, 1e-4, 0.25))
            loads = np.asarray(frame.wheel_load, dtype=np.float64)
            suspension_velocity = np.clip(
                (loads - self._previous_load) / dt * 0.055, -10.0, 10.0
            )
            suspension_travel = np.clip((loads - 1.0) * 0.16, -1.0, 1.0)
            heat = brake * min(1.0, frame.speed / 72.0) * dt
            self._brake_temperature += heat * np.array((0.82, 0.82, 0.58, 0.58))
            cooling = math.exp(-dt * (0.09 + min(frame.speed, 100.0) * 0.0022))
            self._brake_temperature = 0.2 + (self._brake_temperature - 0.2) * cooling
            self._brake_temperature = np.clip(self._brake_temperature, 0.2, 1.0)
            slip_ratio = np.asarray(getattr(vehicle, "wheel_sr", (0, 0, 0, 0)), dtype=float)
            slip_angle = np.tan(np.asarray(getattr(vehicle, "wheel_slip", (0, 0, 0, 0)), dtype=float))
            combined_slip = np.copysign(
                np.hypot(slip_ratio, slip_angle),
                np.where(np.abs(slip_ratio) > 0.02, slip_ratio, slip_angle),
            )
            off_track = bool(scene.get("off_track", False))
            surface = 2 if off_track else 0
            wheel_surface = (surface,) * 4
            contact = (0.0 if bool(getattr(vehicle, "airborne", False)) else 1.0,) * 4
            phase = str(controls.get("shift_phase", "engaged"))
            shift_phase = {"unload": 1.0, "reengage": 0.55}.get(phase, 0.0)
            requested = int(controls.get("requested_gear", frame.gear))
            actual = int(controls.get("actual_gear", frame.gear))
            shift_mismatch = float(np.clip(
                (requested - actual) / max(1, len(getattr(vehicle.spec, "gear_ratios", (1,)))),
                -1.0, 1.0,
            ))
            if controls.get("shift_rejected"):
                shift_mismatch = math.copysign(1.0, shift_mismatch or -1.0)
            damage = max(
                float(getattr(vehicle, "engine_damage", 0.0)),
                float(getattr(vehicle, "aero_damage", 0.0)),
                float(np.max(getattr(vehicle, "wheel_damage", (0.0,)))),
            )
            landing_force = (float(getattr(vehicle, "landing_g", 0.0))
                             if not getattr(vehicle, "airborne", False) else 0.0)
            events = replace(frame.events)
            if landing_force > 0.0 and self._previous_landing <= 0.0:
                events.landing += 1
            if damage > self._previous_damage + 1e-5:
                events.damage += 1
            if surface != self._previous_surface:
                events.surface += 1
            frame = replace(frame,
                            vx=frame.vx * c - frame.vy * s,
                            vy=frame.vx * s + frame.vy * c,
                            wheel_slip=tuple(float(v) for v in combined_slip),
                            wheel_surface=wheel_surface,
                            wheel_contact=contact,
                            brake_temperature=tuple(float(v) for v in self._brake_temperature),
                            suspension_travel=tuple(float(v) for v in suspension_travel),
                            suspension_velocity=tuple(float(v) for v in suspension_velocity),
                            surface=surface,
                            airborne=float(bool(getattr(vehicle, "airborne", False))),
                            landing_force=landing_force,
                            damage=float(np.clip(damage, 0.0, 1.0)),
                            shift_phase=shift_phase,
                            shift_mismatch=shift_mismatch,
                            torque_direction=float(np.clip(
                                throttle - brake - (0.22 if throttle < 0.05 else 0.0),
                                -1.0, 1.0,
                            )),
                            events=events)
            self.frame = frame
            self.sequence = int(sequence)
            self.sim_time = float(sim_time)
            self._previous_time = float(sim_time)
            self._previous_load = loads
            self._previous_landing = landing_force
            self._previous_damage = damage
            self._previous_surface = surface

    def set_listener(self, listener: Mapping[str, Any]) -> None:
        with self._lock:
            values = tuple(_finite_float(listener.get(key, 0.0), f"listener.{key}")
                           for key in ("x", "y", "vx", "vy", "yaw"))
            listener_z = _finite_float(listener.get("z", self.listener_z_target),
                                       "listener.z")
            requested_cut = self.cut
            if "cut" in listener:
                requested_cut = int(_finite_float(listener.get("cut"), "listener.cut"))
            snap_listener = not self.listener_received or requested_cut != self.cut
            self.listener_target = values
            self.listener_z_target = listener_z
            if snap_listener:
                # Camera cuts are already faded by SpatialRendererV4. Snapping
                # here prevents an acoustic fly-through between unrelated shots.
                self.browser_listener = values
                self.listener_z = listener_z
            camera = str(listener.get("camera", self.camera)).lower()
            if camera in ("broadcast", "chase", "chase-low", "roof", "trackside",
                          "trackside-long", "trackside-close", "apex", "drone",
                          "helicopter", "free"):
                self.camera = camera
            requested = str(listener.get("perspective", "external")).lower()
            self.perspective = 2 if requested in ("onboard", "roof", "cockpit") else 0
            weather = str(listener.get("weather", self.weather)).lower()
            if weather in ("dry", "dusk", "night"):
                self.weather = weather
            self.cut = requested_cut
            self.listener_received = True

    def render_chunk(self) -> AudioPCMChunk:
        with self._lock:
            if self.frame is None:
                raise ObservatoryError("audio renderer has no telemetry frame")
            # The browser samples its continuously moving camera rather than
            # streaming a pose every animation frame. Ease between those
            # samples at audio-block cadence so distance, pan, and propagation
            # delay never jump at the listener-message interval.
            block_seconds = self.chunk_frames / self.sample_rate
            alpha = 1.0 - math.exp(-block_seconds / 0.065)
            eased_listener = [
                current + (target - current) * alpha
                for current, target in zip(self.browser_listener,
                                           self.listener_target)
            ]
            yaw_delta = math.atan2(
                math.sin(self.listener_target[4] - self.browser_listener[4]),
                math.cos(self.listener_target[4] - self.browser_listener[4]),
            )
            eased_listener[4] = self.browser_listener[4] + yaw_delta * alpha
            self.browser_listener = tuple(eased_listener)
            self.listener_z += (self.listener_z_target - self.listener_z) * alpha
            listener = self.browser_listener
            if not self.listener_received:
                listener = (self.frame.x, self.frame.y, self.frame.vx,
                            self.frame.vy, self.frame.yaw)
            preset = 1 if self.camera in (
                "trackside", "trackside-long", "trackside-close", "apex"
            ) else 0
            frame = replace(self.frame, acoustic_preset=preset)
            car_audio = self.renderer.render(
                [frame], listener, self.chunk_frames,
                perspective=self.perspective, cut=self.cut, master=0.60,
                reflections=True,
            )
            ambience = self.ambience.render(
                self.chunk_frames, weather=self.weather, listener=listener,
                listener_z=self.listener_z, perspective=self.perspective,
                camera=self.camera,
            )
            audio = car_audio + ambience
            self.output_dc.process(audio)
            self.output_limiter.process(audio)
            if audio.shape != (self.chunk_frames, self.channels) \
                    or not np.all(np.isfinite(audio)):
                raise ObservatoryError("immersive Observatory audio renderer returned invalid PCM")
            pcm = np.clip(audio, -1.0, 1.0)
            pcm = np.rint(pcm * 32767.0).astype("<i2", copy=False).tobytes()
            return AudioPCMChunk(
                sequence=self.sequence,
                sim_time=self.sim_time,
                sample_rate=self.sample_rate,
                frames=self.chunk_frames,
                channels=self.channels,
                pcm=pcm,
            )

    def close(self) -> None:
        with self._lock:
            self.frame = None


class FablePlaybackSession:
    """One deterministic, authoritative Fable policy playback."""

    buffer_seconds = 120.0

    def __init__(
        self,
        checkpoint: ValidatedCheckpoint,
        *,
        mode: str = "replay",
        seed: int = 7,
        audio: bool = True,
        prediction_mode: str = "realtime",
        follow_resolver: Callable[[ValidatedCheckpoint],
                                  ValidatedCheckpoint | None] | None = None,
    ):
        if mode not in ("replay", "follow-active-best"):
            raise ValueError("mode must be 'replay' or 'follow-active-best'")
        self.mode = mode
        self.seed = int(seed)
        self.prediction_mode = prediction_mode
        self.follow_resolver = follow_resolver
        self._lock = threading.RLock()
        self._closed = False
        self._paused = False
        self._speed = 1.0
        self._playback_revision = 0
        self._sequence = 0
        self._clock = 0.0
        self._episode = 0
        self._boundary_pending: str | None = None
        self._pending_discontinuity: dict[str, Any] | None = None
        self._queued_follow: ValidatedCheckpoint | None = None
        self._scrub_sequence: int | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=256)
        self._last_info: dict[str, Any] = {}
        self._last_controls = self._neutral_controls()
        self._wheel_angles = np.zeros(4, dtype=np.float64)

        descriptor = _require_registry_checkpoint(checkpoint)
        loaded = self._load_checkpoint(descriptor)
        self.descriptor, self.identity, self.net, self.norm, self.meta = loaded
        self._install_runtime(self._build_runtime(self.identity, self.meta))
        self.probe = BrainProbe(
            self.net, self.norm, self.meta, self.sensor_spec,
            control_hz=self.ppo_spec.control_hz,
            prediction_mode=self.prediction_mode,
        )
        initial = np.concatenate(
            [self.env.last_obs.vector, self.env._mode_vec]
        ).astype(np.float32)
        self._decision = self.probe.capture(
            initial, self.env.last_obs, self.env, 0, force_expensive=True
        )
        max_frames = int(math.ceil(self.buffer_seconds * self.ppo_spec.control_hz)) + 4
        self._frames: deque[dict[str, Any]] = deque(maxlen=max_frames)
        self.audio = ObservatoryAudioRenderer(self.identity.car, self.seed) if audio else None
        if self.audio is not None:
            self.audio.reset(self.env.veh)
            self.audio.update(self.env.veh, self._last_controls,
                              sequence=0, sim_time=0.0,
                              scene=self.env.last_obs.frame)
        self._emit("session_started", checkpoint=self.identity.public(), seed=self.seed)

    def _build_runtime(
        self, identity: CheckpointIdentity, meta: Mapping[str, Any]
    ) -> PlaybackRuntime:
        """Reconstruct all checkpoint-owned simulation state before activation.

        This is intentionally shared by session startup and follow-mode swaps.
        A new policy never inherits a previous stage's Fable envelope, RaceBox,
        episode budget, or sensor reconstruction.
        """
        sensor_spec = _sensor_spec(meta)
        ppo_spec = _copy_ppo_spec(meta, sensor_spec)
        fable_spec = stage_defaults(identity.stage)
        fable_spec.envelope_scale = float(meta["fable_envelope_scale"])
        fable_spec.shift_lo_frac = float(meta["fable_shift_lo_frac"])
        fable_spec.episode_seconds = float(ppo_spec.episode_seconds)
        sim_spec = SimSpec()
        track = attach_envelope(named_track(FABLE_TRACK), identity.car)
        track_hash = runtime_track_sha256(track)
        env = FableEnv(
            mode="race",
            car=identity.car,
            sim=sim_spec,
            ppo=ppo_spec,
            fixed_track=track,
            fable_spec=fable_spec,
            rng_seed=self.seed,
            diagnostics=True,
        )
        # FableEnv already reconstructed its sensor suite from ppo_spec.  Check
        # it before the first action so a changed runtime cannot silently feed a
        # policy the wrong vector.
        if env.obs_dim != int(meta["obs_dim"]):
            raise CheckpointCompatibilityError(
                f"runtime observation size {env.obs_dim} does not match "
                f"checkpoint {meta['obs_dim']}"
            )
        return PlaybackRuntime(
            sensor_spec=sensor_spec,
            ppo_spec=ppo_spec,
            fable_spec=fable_spec,
            sim_spec=sim_spec,
            track_hash=track_hash,
            env=env,
        )

    def _install_runtime(self, runtime: PlaybackRuntime) -> None:
        """Make a fully validated, fresh runtime visible to this session."""
        self.sensor_spec = runtime.sensor_spec
        self.ppo_spec = runtime.ppo_spec
        self.fable_spec = runtime.fable_spec
        self.sim_spec = runtime.sim_spec
        self.track_hash = runtime.track_hash
        self.env = runtime.env

    @staticmethod
    def _neutral_controls() -> dict[str, Any]:
        return {
            "raw_action": [0.0, 0.0, 0.0],
            "clipped_action": [0.0, 0.0, 0.0],
            "steer": 0.0,
            "longitudinal": 0.0,
            "throttle": 0.0,
            "brake": 0.0,
            "handbrake": 0.0,
            "gear_offset": 0.0,
            "clutch": 0.0,
            "requested_gear": 1,
            "actual_gear": 1,
            "shift_up": False,
            "shift_down": False,
            "shift_phase": "engaged",
            "shift_rejected": "",
        }

    @staticmethod
    def _load_checkpoint(descriptor: ValidatedCheckpoint):
        descriptor = _require_registry_checkpoint(descriptor)
        if not descriptor.path.is_file():
            raise CheckpointCompatibilityError(
                f"validated checkpoint is no longer a file: {descriptor.path.name}"
            )
        expected_file_hash = str(descriptor.file_sha256 or "").lower()
        if len(expected_file_hash) != 64 or any(
            char not in "0123456789abcdef" for char in expected_file_hash
        ):
            raise CheckpointCompatibilityError(
                "validated checkpoint is missing a valid content hash"
            )
        try:
            before_file_hash = _sha256_file(descriptor.path)
        except OSError as exc:
            raise CheckpointCompatibilityError(
                f"checkpoint content cannot be read: {descriptor.path.name}"
            ) from exc
        if before_file_hash != expected_file_hash:
            raise CheckpointCompatibilityError(
                "checkpoint bytes changed after server-side catalogue validation"
            )
        try:
            net, norm, meta = PPO.load_policy(descriptor.path)
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"safe policy load failed for {descriptor.path.name}: {exc}"
            ) from exc
        try:
            after_file_hash = _sha256_file(descriptor.path)
        except OSError as exc:
            raise CheckpointCompatibilityError(
                f"checkpoint content changed while loading: {descriptor.path.name}"
            ) from exc
        if after_file_hash != expected_file_hash or after_file_hash != before_file_hash:
            raise CheckpointCompatibilityError(
                "checkpoint bytes changed while Observatory was loading it"
            )
        edition = _checkpoint_contract(descriptor, meta)
        actual_hash = state_dict_hash(net.state_dict())
        stored_hash = str(meta.get("policy_sha256") or "")
        if not stored_hash or stored_hash != actual_hash:
            raise CheckpointCompatibilityError("checkpoint policy hash is missing or invalid")
        if descriptor.policy_sha256 and descriptor.policy_sha256 != actual_hash:
            raise CheckpointCompatibilityError(
                "checkpoint changed after server-side catalogue validation"
            )
        warnings = list(descriptor.warnings)
        if descriptor.classification == "historical":
            warning = "Historical fallback: playable, but not current-protocol evidence."
            if not any("historical" in existing.lower() for existing in warnings):
                warnings.append(warning)
        if edition.authority_warning and edition.authority_warning not in warnings:
            warnings.append(edition.authority_warning)
        identity = CheckpointIdentity(
            checkpoint_id=descriptor.checkpoint_id or descriptor.path.name,
            filename=descriptor.path.name,
            edition=edition.id,
            car=edition.car_id,
            stage=str(meta["fable_stage"]),
            drivetrain=edition.drivetrain,
            observation_layout=edition.observation_layout,
            compatibility_class=edition.compatibility_class,
            classification=descriptor.classification,
            file_sha256=expected_file_hash,
            policy_sha256=actual_hash,
            runtime_contract_sha256=_runtime_contract_sha256(edition, meta),
            warnings=tuple(warnings),
        )
        return descriptor, identity, net, norm, dict(meta)

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def control_hz(self) -> int:
        return int(self.ppo_spec.control_hz)

    @property
    def wall_interval(self) -> float:
        """Scheduler interval; speed never changes simulator dt or policy Hz."""
        return 1.0 / (self.control_hz * self._speed)

    @property
    def sequence(self) -> int:
        return self._sequence

    @property
    def sim_time(self) -> float:
        return self._clock

    def _ensure_open(self) -> None:
        if self._closed:
            raise SessionClosedError("Observatory session is closed")

    def _emit(self, name: str, **payload: Any) -> None:
        self._events.append(_json_safe({
            "name": name,
            "sequence": self._sequence,
            "sim_time": self._clock,
            **payload,
        }))

    def drain_events(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events)
            self._events.clear()
            return events

    def _playback_state(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "speed": self._speed,
            "mode": self.mode,
            "scrubbing": self._scrub_sequence is not None,
            "revision": self._playback_revision,
        }

    def hello(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            return _json_safe({
                "protocol": "fable-observatory-v1",
                "mode": self.mode,
                "seed": self.seed,
                "checkpoint": self.identity.public(),
                # A reconnect can happen after a control request reaches the
                # server but before its state frame reaches the browser.  Put
                # the current control state in the handshake so a paused
                # session is always resumable after that race.
                "playback": self._playback_state(),
                "track": {
                    "id": FABLE_TRACK,
                    "hash": self.track_hash,
                    "length_m": float(self.env.trk.length),
                    "samples": int(len(self.env.trk.center)),
                    "truth_width_m": float(self.env.trk.half * 2.0),
                    "truth_bank_rad": [float(np.min(self.env.trk.bank)),
                                       float(np.max(self.env.trk.bank))],
                },
                "simulation": {
                    "physics_dt": self.sim_spec.dt,
                    "physics_hz": 1.0 / self.sim_spec.dt,
                    "policy_hz": self.control_hz,
                    "control_substeps": self.env.control_period,
                    "deterministic_inference": "act_mean",
                    "audio": self.audio is not None,
                    "audio_format": ({
                        "sample_rate": 48_000, "channels": 2,
                        "sample_type": "signed-int16-le", "chunk_frames": 1_024,
                        "wire_magic": "FOA1",
                        "source": self.audio.source,
                        "perspective": "camera-relative-dynamic",
                        "ambience": "nordschleife-dry-dusk-night",
                    } if self.audio is not None else None),
                },
                "controls": {
                    "actions": list(ACTION_NAMES),
                    "gear_semantics": "clipped policy output multiplied by 2, then RaceBox",
                    "speeds": list(SUPPORTED_SPEEDS),
                    "buffer_seconds": self.buffer_seconds,
                },
            })

    def _control_telemetry(self, raw_action: np.ndarray, gear_before: int) -> dict[str, Any]:
        raw = np.asarray(raw_action, dtype=np.float64).reshape(-1)
        if raw.shape != (3,) or not np.all(np.isfinite(raw)):
            raise ObservatoryError("policy emitted an invalid action")
        clipped = np.clip(raw, -1.0, 1.0)
        long = float(clipped[1])
        gear_after = int(self.env.veh.gear)
        return {
            "raw_action": raw.tolist(),
            "clipped_action": clipped.tolist(),
            "steer": float(clipped[0]),
            "longitudinal": long,
            "throttle": max(long, 0.0),
            "brake": max(-long, 0.0),
            "handbrake": 0.0,
            "gear_offset": float(clipped[2] * 2.0),
            "clutch": float(self.env.rbox.clutch),
            "requested_gear": int(self.env.rbox.requested_gear),
            "actual_gear": gear_after,
            "shift_up": bool(gear_after > gear_before),
            "shift_down": bool(gear_after < gear_before),
            "shift_phase": self.env.rbox.shift_phase,
            "shift_rejected": str(self.env.rbox.rejected_shift),
        }

    def _sensor_geometry(self, observation: Observation) -> dict[str, Any]:
        return {
            "beam_origin": [float(self.env.veh.x), float(self.env.veh.y),
                            float(self.env.veh.z)],
            "beam_angles_rad": _json_safe(observation.beam_angles),
            "beam_distances_m": _json_safe(observation.beams),
            "beam_points": _json_safe(observation.beam_points),
            "lookahead_curvature_per_m": _json_safe(observation.lookahead),
            "lookahead_grade": _json_safe(observation.lookahead_grade),
            "lookahead_crest_curvature": _json_safe(observation.lookahead_vcurv),
        }

    def _vehicle_telemetry(self, elapsed: float) -> dict[str, Any]:
        v = self.env.veh
        self._wheel_angles = np.remainder(
            self._wheel_angles + np.asarray(v.wheel_w, dtype=float) * elapsed,
            math.tau,
        )
        c, s = math.cos(v.yaw), math.sin(v.yaw)
        world_vx = v.vx * c - v.vy * s
        world_vy = v.vx * s + v.vy * c
        wheels = []
        ids = ("FL", "FR", "RL", "RR")
        for i, wheel_id in enumerate(ids):
            steer = v.steer_angle if i < 2 else getattr(v, "rear_steer_angle", 0.0)
            wheels.append({
                "id": wheel_id,
                "rotation_rad": float(self._wheel_angles[i]),
                "angular_velocity_rad_s": float(v.wheel_w[i]),
                "steer_rad": float(steer),
                "load_n": float(v.Fz[i]),
                "slip_ratio": float(v.wheel_sr[i]),
                "slip_angle_rad": float(v.wheel_slip[i]),
                "grip": float(v.wheel_grip[i]),
                "contact": not bool(v.airborne),
            })
        cap = max(float(getattr(v.spec, "hybrid_battery_kj", 0.0)), 1.0)
        mgu_cap = max(float(getattr(v.spec, "hybrid_mgu_power_w", 0.0)), 1.0)
        body_length = (float(v.spec.body_length)
                       if float(getattr(v.spec, "body_length", 0.0)) > 0.0
                       else float(v.spec.wheelbase + 1.7))
        body_width = (float(v.spec.body_width)
                      if float(getattr(v.spec, "body_width", 0.0)) > 0.0
                      else float(v.spec.track_width + 0.36))
        return {
            "pose": {
                "x": float(v.x), "y": float(v.y), "z": float(v.z),
                "yaw": float(v.yaw), "pitch": float(v.pitch), "roll": float(v.roll),
            },
            "dynamics": {
                "speed_mps": float(v.speed),
                "speed_kmh": float(v.speed * 3.6),
                "body_velocity_mps": [float(v.vx), float(v.vy), float(v.vz)],
                "world_velocity_mps": [float(world_vx), float(world_vy), float(v.vz)],
                "yaw_rate_rad_s": float(v.r),
                "acceleration_mps2": [float(v.ax), float(v.ay)],
                "slip_angle_rad": float(v.slip_angle),
                "airborne": bool(v.airborne),
                "landing_g": float(v.landing_g),
                "surface_grip": float(v.surface_grip),
            },
            "powertrain": {
                "rpm": float(v.rpm),
                "gear": int(v.gear),
                "boost": float(v.boost),
                "hybrid_soc_kj": float(getattr(v, "hybrid_soc_kj", 0.0)),
                "hybrid_soc": float(getattr(v, "hybrid_soc_kj", 0.0) / cap),
                "mgu_power_w": float(getattr(v, "mgu_power_w", 0.0)),
                "mgu_power_fraction": float(getattr(v, "mgu_power_w", 0.0) / mgu_cap),
                "active_aero_low_drag": float(getattr(v, "aero_lowdrag", 0.0)),
                "active_aero_engaged": bool(getattr(v, "aero_lowdrag", 0.0) > 0.5),
            },
            "geometry": {
                "collision_length_m": body_length,
                "collision_width_m": body_width,
                "wheelbase_m": float(v.spec.wheelbase),
                "track_width_m": float(v.spec.track_width),
                "footprint_world_xy": _json_safe(v.get_obb()),
            },
            "wheels": wheels,
        }

    def _track_telemetry(self, observation: Observation) -> dict[str, Any]:
        frame = observation.frame
        return {
            "id": FABLE_TRACK,
            "hash": self.track_hash,
            "arc_m": float(frame["arc"]),
            "progress": float(frame["progress"]),
            "lateral_m": float(frame["lateral"]),
            "half_width_m": float(frame.get("half_width", self.env.trk.half)),
            "off_track": bool(frame["off_track"]),
            "heading_rad": float(frame["heading"]),
            "curvature_per_m": float(frame["curvature"]),
            "grade": float(frame["grade"]),
            "bank": float(frame["bank"]),
            "road_z_m": float(frame["z"]),
            "vertical_curvature_per_m": float(frame["vcurv"]),
        }

    def _make_frame(self, elapsed: float, info: Mapping[str, Any],
                    terminated: bool, truncated: bool) -> dict[str, Any]:
        observation = self.env.last_obs
        discontinuity = copy.deepcopy(self._pending_discontinuity)
        frame = {
            "schema": "fable-observatory-frame-v1",
            "sequence": self._sequence,
            "sim_time": self._clock,
            "episode": self._episode,
            "episode_time": float(self.env.t),
            "checkpoint": self.identity.public(),
            "vehicle": self._vehicle_telemetry(elapsed),
            "controls": self._last_controls,
            "control_semantics": (
                "controls were applied over the interval ending at this frame; "
                "brain output means are the deterministic command for the next interval"
            ),
            "track": self._track_telemetry(observation),
            "fable": {
                "stage": self.identity.stage,
                "valid": bool(not getattr(self.env, "fable_invalid", False)),
                "progress_m": float(getattr(self.env, "fable_progress_m", 0.0)),
                "pace_ratio": float(info.get("fable_pace_ratio", 0.0)),
                "footprint_valid": bool(info.get(
                    "fable_footprint_valid",
                    not getattr(self.env, "fable_footprint_invalid", False),
                )),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "termination_reason": info.get("termination_reason"),
                "lap_completed_at": info.get("fable_lap_t"),
            },
            "observations": self._decision.observations,
            "brain": self._decision.brain,
            "sensor_geometry": self._sensor_geometry(observation),
            "playback": {
                **self._playback_state(),
                "discontinuity": discontinuity,
            },
        }
        safe = _json_safe(frame)
        # Force the exact serializer contract here rather than discovering a
        # NaN in a websocket implementation later.
        json.dumps(safe, allow_nan=False, separators=(",", ":"))
        if discontinuity is not None:
            self._pending_discontinuity = None
        return safe

    def _state_frame(self) -> dict[str, Any]:
        """Return the current authoritative state without advancing physics."""
        return self._make_frame(0.0, self._last_info, False, False)

    def _swap_policy(self, descriptor: ValidatedCheckpoint) -> bool:
        if descriptor.car != self.identity.car or descriptor.edition != self.identity.edition:
            raise CheckpointCompatibilityError("follow candidate crosses edition or car")
        loaded = self._load_checkpoint(descriptor)
        new_descriptor, identity, net, norm, meta = loaded
        # Equal weights are not an equal playback identity: normalizer,
        # stage, envelope, shift behavior and episode budget live outside the
        # state dict.  Only exact, registry-verified checkpoint bytes are a
        # no-op.
        if identity.file_sha256 == self.identity.file_sha256:
            return False
        runtime = self._build_runtime(identity, meta)
        if runtime.sensor_spec != self.sensor_spec:
            raise CheckpointCompatibilityError("follow candidate changes observation layout")
        if runtime.ppo_spec.control_hz != self.control_hz:
            raise CheckpointCompatibilityError("follow candidate changes policy cadence")
        if runtime.track_hash != self.track_hash:
            raise CheckpointCompatibilityError("follow candidate changes runtime track truth")
        old = self.identity
        old_probe = self.probe
        new_probe = BrainProbe(
            net, norm, meta, runtime.sensor_spec,
            control_hz=runtime.ppo_spec.control_hz,
            prediction_mode=self.prediction_mode,
        )
        self.descriptor = new_descriptor
        self.identity = identity
        self.net = net
        self.norm = norm
        self.meta = meta
        self._install_runtime(runtime)
        self.probe = new_probe
        old_probe.close()
        self._emit(
            "checkpoint_swap", previous=old.public(), current=identity.public(),
            runtime_rebuilt=True,
        )
        return True

    def queue_follow_candidate(self, checkpoint: ValidatedCheckpoint) -> None:
        with self._lock:
            self._ensure_open()
            if self.mode != "follow-active-best":
                raise ObservatoryError("immutable replay sessions cannot queue a swap")
            candidate = _require_registry_checkpoint(checkpoint)
            self._queued_follow = candidate
            self._emit("checkpoint_swap_queued", checkpoint_id=candidate.checkpoint_id)

    def _follow_at_boundary(self, reason: str) -> None:
        if self.mode != "follow-active-best":
            return
        candidate = self._queued_follow
        self._queued_follow = None
        if candidate is None and self.follow_resolver is not None:
            try:
                resolved = self.follow_resolver(self.descriptor)
                if resolved is not None:
                    candidate = _require_registry_checkpoint(resolved)
            except Exception as exc:
                self._emit("warning", code="follow_resolver_failed",
                           message=str(exc), boundary=reason)
                return
        if candidate is None:
            return
        try:
            self._swap_policy(candidate)
        except Exception as exc:
            self._emit("warning", code="follow_candidate_rejected",
                       message=str(exc), boundary=reason)

    def _reset_environment(self, reason: str) -> None:
        self._follow_at_boundary(reason)
        self.env.reset()
        self._episode += 1
        self._boundary_pending = None
        self._pending_discontinuity = {
            "kind": "episode_reset",
            "reason": reason,
            "episode": self._episode,
            "sequence": self._sequence,
        }
        self._wheel_angles.fill(0.0)
        self._last_controls = self._neutral_controls()
        self._last_info = {}
        self.probe.invalidate_prediction()
        current = np.concatenate(
            [self.env.last_obs.vector, self.env._mode_vec]
        ).astype(np.float32)
        self._decision = self.probe.capture(
            current, self.env.last_obs, self.env, self._sequence,
            force_expensive=True,
        )
        if self.audio is not None:
            self.audio.reset(self.env.veh)
            self.audio.update(self.env.veh, self._last_controls,
                              sequence=self._sequence, sim_time=self._clock,
                              scene=self.env.last_obs.frame)
        self._emit("reset", reason=reason, episode=self._episode,
                   playback=self._playback_state())

    def _advance_live(self) -> dict[str, Any]:
        if self._boundary_pending is not None:
            self._reset_environment(self._boundary_pending)
        action = self._decision.action.copy()
        gear_before = int(self.env.veh.gear)
        episode_before = float(self.env.t)
        full, _, terminated, truncated, info = self.env.step(action)
        elapsed = max(self.sim_spec.dt, float(self.env.t) - episode_before)
        self._clock += elapsed
        self._sequence += 1
        self._wheel_angles = np.asarray(self._wheel_angles, dtype=np.float64)
        self._last_controls = self._control_telemetry(action, gear_before)
        self._last_info = dict(info)
        self._decision = self.probe.capture(
            full, self.env.last_obs, self.env, self._sequence
        )
        frame = self._make_frame(elapsed, info, terminated, truncated)
        self._frames.append(frame)
        if self.audio is not None:
            self.audio.update(
                self.env.veh, self._last_controls,
                sequence=self._sequence, sim_time=self._clock,
                scene=self.env.last_obs.frame,
            )
        lap_completed = info.get("fable_lap_t") is not None
        if terminated or truncated:
            reason = str(info.get("termination_reason")
                         or ("termination" if terminated else "truncation"))
            self._boundary_pending = reason
            self._emit("episode_boundary", reason=reason,
                       terminated=bool(terminated), truncated=bool(truncated))
        elif lap_completed:
            self._follow_at_boundary("lap_complete")
        return frame

    def tick(self) -> dict[str, Any] | None:
        """Advance one policy tick.  A scheduler applies ``wall_interval``."""
        with self._lock:
            self._ensure_open()
            if self._paused or self._scrub_sequence is not None:
                return None
            return self._advance_live()

    def pause(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            self._paused = True
            self._playback_revision += 1
            self._emit("paused", playback=self._playback_state())
            return self._state_frame()

    def resume(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            self._paused = False
            self._scrub_sequence = None
            self._playback_revision += 1
            self._emit("resumed", playback=self._playback_state())
            return self._state_frame()

    def step_once(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            if self._scrub_sequence is not None:
                future = [f for f in self._frames
                          if int(f["sequence"]) > self._scrub_sequence]
                if future:
                    self._scrub_sequence = int(future[0]["sequence"])
                    result = copy.deepcopy(future[0])
                    result["playback"]["paused"] = True
                    result["playback"]["scrubbing"] = True
                    return result
                self._scrub_sequence = None
            frame = self._advance_live()
            self._emit("single_step")
            return frame

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            self._scrub_sequence = None
            self._playback_revision += 1
            self._reset_environment("client_reset")
            # Send an immediate state frame as the reset boundary.  The next
            # advancing frame still uses the exact policy interval, so global
            # simulation time remains monotonic.
            return self._state_frame()

    def set_speed(self, speed: float) -> float:
        value = float(speed)
        matched = next((allowed for allowed in SUPPORTED_SPEEDS
                        if math.isclose(value, allowed, rel_tol=0.0, abs_tol=1e-9)), None)
        if matched is None:
            raise ValueError(f"speed must be one of {SUPPORTED_SPEEDS}")
        with self._lock:
            self._ensure_open()
            self._speed = matched
            self._playback_revision += 1
            self._emit("speed_changed", speed=matched,
                       playback=self._playback_state())
            return matched

    def scrub(self, *, seconds_ago: float | None = None,
              sequence: int | None = None, sim_time: float | None = None) -> dict[str, Any]:
        with self._lock:
            self._ensure_open()
            if not self._frames:
                raise ObservatoryError("replay buffer is empty")
            frames = list(self._frames)
            if sequence is not None:
                target = min(frames, key=lambda f: abs(int(f["sequence"]) - int(sequence)))
            else:
                target_time = (float(sim_time) if sim_time is not None else
                               self._clock - max(0.0, float(seconds_ago or 0.0)))
                target = min(frames, key=lambda f: abs(float(f["sim_time"]) - target_time))
            self._paused = True
            self._scrub_sequence = int(target["sequence"])
            self._playback_revision += 1
            self._emit("scrubbed", target_sequence=self._scrub_sequence,
                       target_time=float(target["sim_time"]),
                       playback=self._playback_state())
            result = copy.deepcopy(target)
            result["playback"]["paused"] = True
            result["playback"]["scrubbing"] = True
            return result

    def set_listener(self, listener: Mapping[str, Any]) -> None:
        with self._lock:
            self._ensure_open()
            if self.audio is not None:
                self.audio.set_listener(listener)

    def render_audio_chunk(self) -> AudioPCMChunk:
        with self._lock:
            self._ensure_open()
            if self.audio is None:
                raise ObservatoryError("audio is disabled for this session")
            return self.audio.render_chunk()

    def handle_control(self, message: Mapping[str, Any]) -> dict[str, Any] | None:
        """Transport-neutral implementation of the websocket control contract."""
        command = str(message.get("type") or message.get("command") or "").lower()
        if command == "pause":
            return {"type": "frame", "frame": self.pause()}
        if command == "resume":
            return {"type": "frame", "frame": self.resume()}
        if command in ("step", "single_step"):
            return {"type": "frame", "frame": self.step_once()}
        if command == "reset":
            return {"type": "frame", "frame": self.reset()}
        if command == "speed":
            return {"type": "event", "event": "speed_changed",
                    "speed": self.set_speed(float(message.get("speed", 1.0)))}
        if command == "scrub":
            kwargs = {key: message[key] for key in
                      ("seconds_ago", "sequence", "sim_time") if key in message}
            return {"type": "frame", "frame": self.scrub(**kwargs)}
        if command in ("listener", "camera"):
            listener = message.get("listener") or message
            self.set_listener(listener)
            return None
        raise ValueError(f"unknown Observatory control {command!r}")

    def buffer_status(self) -> dict[str, Any]:
        with self._lock:
            if not self._frames:
                return {"frames": 0, "oldest_time": None, "newest_time": None}
            return {
                "frames": len(self._frames),
                "oldest_time": float(self._frames[0]["sim_time"]),
                "newest_time": float(self._frames[-1]["sim_time"]),
                "oldest_sequence": int(self._frames[0]["sequence"]),
                "newest_sequence": int(self._frames[-1]["sequence"]),
            }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            if self.audio is not None:
                self.audio.close()
            if self.probe is not None:
                self.probe.close()
            self._frames.clear()
            self._events.clear()
            self.follow_resolver = None
            self.audio = None
            self.env = None
            self.net = None
            self.norm = None
            self.probe = None
            self._decision = None
            self._closed = True


@dataclass
class _ManagedSession:
    session: FablePlaybackSession
    browser_id: str
    connections: int = 0
    cleanup_deadline: float | None = None
    cleanup_timer: threading.Timer | None = None
    created_monotonic: float = field(default_factory=time.monotonic)


class ObservatorySessionManager:
    """Four-session global cap, one session per browser and delayed cleanup."""

    def __init__(self, *, max_sessions: int = 4, per_browser: int = 1,
                 cleanup_seconds: float = 120.0):
        if max_sessions < 1 or per_browser < 1 or cleanup_seconds < 0:
            raise ValueError("invalid Observatory session limits")
        self.max_sessions = int(max_sessions)
        self.per_browser = int(per_browser)
        self.cleanup_seconds = float(cleanup_seconds)
        self._sessions: dict[str, _ManagedSession] = {}
        self._pending_browsers: set[str] = set()
        self._lock = threading.RLock()

    def create(
        self,
        browser_id: str,
        checkpoint: ValidatedCheckpoint,
        **kwargs: Any,
    ) -> tuple[str, FablePlaybackSession]:
        browser = str(browser_id).strip()
        if not browser or len(browser) > 256:
            raise ValueError("browser_id is required")
        checkpoint = _require_registry_checkpoint(checkpoint)
        self.reap_expired()
        with self._lock:
            owned = sum(1 for item in self._sessions.values()
                        if item.browser_id == browser)
            if owned + (1 if browser in self._pending_browsers else 0) >= self.per_browser:
                raise SessionLimitError("this browser already owns an Observatory session")
            if len(self._sessions) + len(self._pending_browsers) >= self.max_sessions:
                raise SessionLimitError("the Observatory global session limit is full")
            self._pending_browsers.add(browser)
        try:
            session = FablePlaybackSession(checkpoint, **kwargs)
        except Exception:
            with self._lock:
                self._pending_browsers.discard(browser)
            raise
        session_id = uuid.uuid4().hex
        with self._lock:
            self._pending_browsers.discard(browser)
            item = _ManagedSession(
                session, browser,
                cleanup_deadline=time.monotonic() + self.cleanup_seconds,
            )
            self._sessions[session_id] = item
            self._arm_cleanup(session_id, item)
        return session_id, session

    def _arm_cleanup(self, session_id: str, item: _ManagedSession) -> None:
        if item.cleanup_timer is not None:
            item.cleanup_timer.cancel()
        if item.cleanup_deadline is None:
            item.cleanup_timer = None
            return
        delay = max(0.0, item.cleanup_deadline - time.monotonic())
        timer = threading.Timer(delay, self._cleanup_callback, args=(session_id,))
        timer.daemon = True
        item.cleanup_timer = timer
        timer.start()

    def _cleanup_callback(self, session_id: str) -> None:
        session = None
        with self._lock:
            item = self._sessions.get(session_id)
            if item is None or item.connections != 0 or item.cleanup_deadline is None:
                return
            if item.cleanup_deadline > time.monotonic():
                self._arm_cleanup(session_id, item)
                return
            item = self._sessions.pop(session_id)
            item.cleanup_timer = None
            session = item.session
        if session is not None:
            session.close()

    def get(self, session_id: str) -> FablePlaybackSession:
        with self._lock:
            item = self._sessions.get(str(session_id))
            if item is None:
                raise KeyError("unknown Observatory session")
            return item.session

    def connect(self, session_id: str) -> FablePlaybackSession:
        with self._lock:
            item = self._sessions.get(str(session_id))
            if item is None:
                raise KeyError("unknown Observatory session")
            item.connections += 1
            item.cleanup_deadline = None
            if item.cleanup_timer is not None:
                item.cleanup_timer.cancel()
                item.cleanup_timer = None
            return item.session

    def disconnect(self, session_id: str) -> None:
        with self._lock:
            item = self._sessions.get(str(session_id))
            if item is None:
                return
            if item.connections <= 0:
                return
            item.connections -= 1
            if item.connections == 0:
                item.cleanup_deadline = time.monotonic() + self.cleanup_seconds
                self._arm_cleanup(str(session_id), item)

    def delete(self, session_id: str) -> bool:
        with self._lock:
            item = self._sessions.pop(str(session_id), None)
        if item is None:
            return False
        if item.cleanup_timer is not None:
            item.cleanup_timer.cancel()
        item.session.close()
        return True

    def reap_expired(self, *, now: float | None = None) -> list[str]:
        instant = time.monotonic() if now is None else float(now)
        with self._lock:
            expired = [session_id for session_id, item in self._sessions.items()
                       if item.connections == 0
                       and item.cleanup_deadline is not None
                       and item.cleanup_deadline <= instant]
        for session_id in expired:
            self.delete(session_id)
        return expired

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active": len(self._sessions),
                "maximum": self.max_sessions,
                "per_browser": self.per_browser,
                "cleanup_seconds": self.cleanup_seconds,
                "sessions": {
                    session_id: {
                        "browser_id": item.browser_id,
                        "connections": item.connections,
                        "cleanup_pending": item.cleanup_deadline is not None,
                        "checkpoint": item.session.identity.public(),
                    }
                    for session_id, item in self._sessions.items()
                },
            }

    def close_all(self) -> None:
        with self._lock:
            session_ids = list(self._sessions)
        for session_id in session_ids:
            self.delete(session_id)


__all__ = [
    "ACTION_NAMES",
    "AudioPCMChunk",
    "BrainProbe",
    "CheckpointCompatibilityError",
    "CheckpointIdentity",
    "FablePlaybackSession",
    "ObservatoryAudioRenderer",
    "ObservatoryError",
    "ObservatorySessionManager",
    "SUPPORTED_CARS",
    "SUPPORTED_SPEEDS",
    "SessionClosedError",
    "SessionLimitError",
    "ValidatedCheckpoint",
    "frame_sha256",
    "runtime_track_sha256",
]
