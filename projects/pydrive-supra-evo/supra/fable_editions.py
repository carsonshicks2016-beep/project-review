"""Authoritative edition and checkpoint identity for Fable Five.

The Observatory, Command Center, and maintenance tools consume this module
instead of carrying independent browser-side copies of the 787B/919 contract.
It deliberately treats checkpoint files as untrusted input: names are confined
to one checkpoint directory, payloads use the constrained loader, policy state
is shape-checked, and every piece of car/track/observation identity must match
the selected edition before a policy is playable.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from types import MappingProxyType
from typing import Any, Mapping


FABLE_STAGES = ("foundation", "flow", "finish", "fast", "frontier")
FABLE_TRACK = "nordschleife"
FABLE_TRACK_PROFILE = "nordschleife-full-20.832km"
CURRENT_EVAL_PROTOCOL = "fable5-eval-v3-footprint-envelope"
MAX_CHECKPOINT_BYTES = 32 * 1024 * 1024

# This 919 checkpoint contract was current immediately before the identity-only
# auto-manifest/event-routing repair.  That repair changes fable5.py's broad
# source hash but does not change physics, sensors, actions, policy inference,
# or evaluation.  Keeping this one reviewed hash current prevents a bookkeeping
# fix from falsely demoting the active 919 policy.  No historical 787B source
# hash receives that accommodation.
_919_IDENTITY_REPAIR_BASELINE = (
    "11627bb7ba8eabb631a2c91d2ab2988b11ba90eeb3dca0847fa907f843cdda8a"
)


@dataclass(frozen=True)
class FableBenchmark:
    key: str
    label: str
    seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {"key": self.key, "label": self.label,
                "seconds": self.seconds}


@dataclass(frozen=True)
class FableEdition:
    id: str
    car_id: str
    label: str
    manifest: str
    canonical_champion: str
    checkpoint_prefix: str
    latest_eval: str
    drivetrain: str
    observation_layout: str
    hybrid_observations: bool
    obs_dim: int
    sdim: int
    act_dim: int
    benchmark: FableBenchmark
    compatibility_class: str
    authority_warning: str | None = None
    accepted_source_fingerprints: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "car_id": self.car_id,
            "label": self.label,
            "manifest": self.manifest,
            "canonical_champion": self.canonical_champion,
            "checkpoint_prefix": self.checkpoint_prefix,
            "latest_eval": self.latest_eval,
            "drivetrain": self.drivetrain,
            "observation_layout": self.observation_layout,
            "hybrid_observations": self.hybrid_observations,
            "obs_dim": self.obs_dim,
            "sdim": self.sdim,
            "act_dim": self.act_dim,
            "benchmark": self.benchmark.as_dict(),
            "compatibility_class": self.compatibility_class,
            "authority_warning": self.authority_warning,
            "legacy_approximation": self.id == "919",
            "certification_eligible": False if self.id == "919" else None,
        }


_EDITIONS = {
    "787b": FableEdition(
        id="787b",
        car_id="mazda787b",
        label="Mazda 787B",
        manifest="fable5_ring_pipeline.json",
        canonical_champion="fable5_ring_best.pt",
        checkpoint_prefix="fable5_ring",
        latest_eval="fable5_ring_eval_latest.json",
        drivetrain="mazda787b-5spd-ring-v1",
        observation_layout="fable-v1",
        hybrid_observations=False,
        obs_dim=68,
        sdim=66,
        act_dim=3,
        benchmark=FableBenchmark(
            key="bellof_956_1983_qualifying",
            label="Bellof 1983 qualifying",
            seconds=371.13,
        ),
        compatibility_class="fable-v1-68x3-mazda787b-5spd-ring-v1",
    ),
    "919": FableEdition(
        id="919",
        car_id="porsche_919evo",
        label="Porsche 919 Evo",
        manifest="fable5_919_ring_pipeline.json",
        canonical_champion="fable5_919_ring_best.pt",
        checkpoint_prefix="fable5_919_ring",
        latest_eval="fable5_919_ring_eval_latest.json",
        drivetrain="porsche919evo-7spd-ring-v2",
        observation_layout="fable-v2",
        hybrid_observations=True,
        obs_dim=70,
        sdim=68,
        act_dim=3,
        benchmark=FableBenchmark(
            key="porsche_919_evo_2018_record",
            label="Porsche 919 Evo 2018 record",
            seconds=319.55,
        ),
        compatibility_class="fable-v2-70x3-porsche919evo-7spd-ring-v2",
        authority_warning=(
            "Legacy Fable approximation: this 919 is not faithful-v2 and "
            "must not be used as certification evidence."
        ),
        accepted_source_fingerprints=(_919_IDENTITY_REPAIR_BASELINE,),
    ),
}

FABLE_EDITIONS: Mapping[str, FableEdition] = MappingProxyType(_EDITIONS)
_ALIASES = {
    "787b": "787b",
    "mazda787b": "787b",
    "mazda_787b": "787b",
    "919": "919",
    "919evo": "919",
    "porsche919evo": "919",
    "porsche_919evo": "919",
}


class CheckpointRejected(ValueError):
    """A checkpoint cannot be exposed to policy inference."""


class UnsafeCheckpointPath(CheckpointRejected):
    """A requested checkpoint name escapes or bypasses the checkpoint root."""


def get_edition(value: str | FableEdition) -> FableEdition:
    if isinstance(value, FableEdition):
        return value
    key = _ALIASES.get(str(value or "").strip().lower())
    if key is None:
        raise KeyError(f"unknown Fable edition: {value!r}")
    return FABLE_EDITIONS[key]


def edition_for_car(car_id: str) -> FableEdition | None:
    key = _ALIASES.get(str(car_id or "").strip().lower())
    return FABLE_EDITIONS.get(key) if key else None


def registry_payload() -> list[dict[str, Any]]:
    """JSON-ready descriptors in stable dashboard order."""
    return [FABLE_EDITIONS[key].as_dict() for key in ("787b", "919")]


def resolve_checkpoint_path(checkpoint_root: str | Path,
                            checkpoint: str | os.PathLike[str]) -> Path:
    """Resolve one plain checkpoint filename without permitting traversal."""
    root = Path(checkpoint_root).expanduser().resolve()
    raw = os.fspath(checkpoint)
    if not isinstance(raw, str):
        raise UnsafeCheckpointPath("checkpoint name must be text")
    if (not raw or "\x00" in raw or "/" in raw or "\\" in raw
            or Path(raw).is_absolute() or Path(raw).name != raw
            or Path(raw).suffix.lower() != ".pt"):
        raise UnsafeCheckpointPath(
            "checkpoint must be a plain .pt filename inside the checkpoint directory"
        )
    candidate = root / raw
    try:
        resolved = candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise CheckpointRejected(f"checkpoint does not exist: {raw}") from exc
    if resolved.parent != root or not resolved.is_file():
        raise UnsafeCheckpointPath(
            "checkpoint must resolve to a regular file in the checkpoint directory"
        )
    return resolved


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_source_fingerprint() -> str:
    # Lazy import avoids coupling Fable training startup to Observatory code.
    from .fable5 import CODE_FINGERPRINT
    return str(CODE_FINGERPRINT)


def _accepted_fingerprints(edition: FableEdition) -> frozenset[str]:
    return frozenset((*edition.accepted_source_fingerprints,
                      _current_source_fingerprint()))


def _evaluation_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    ev = payload.get("fable_eval")
    if not isinstance(ev, Mapping):
        return {}
    fields = (
        "stage", "metric", "lap_time", "progress_frac", "distance_laps",
        "clean_chain", "clean_sectors", "sector_count", "terminal_rate",
        "pace_ratio", "superhuman", "evaluated_policy_sha256",
        "recommendation",
    )
    return {key: ev.get(key) for key in fields if key in ev}


@dataclass(frozen=True)
class CheckpointRecord:
    name: str
    edition: str
    classification: str
    playable: bool
    warnings: tuple[str, ...]
    rejection: str | None
    file_sha256: str | None
    file_bytes: int | None
    policy_sha256: str | None
    stored_policy_sha256: str | None
    car_id: str | None
    track: str | None
    track_profile: str | None
    stage: str | None
    drivetrain: str | None
    observation_layout: str | None
    obs_dim: int | None
    sdim: int | None
    act_dim: int | None
    hidden: tuple[int, ...]
    mode: str | None
    eval_protocol: str | None
    code_fingerprint: str | None
    compatibility_class: str
    checkpoint_time_unix: float | None
    evidence: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "edition": self.edition,
            "classification": self.classification,
            "playable": self.playable,
            "warnings": list(self.warnings),
            "rejection": self.rejection,
            "file_sha256": self.file_sha256,
            "file_bytes": self.file_bytes,
            "policy_sha256": self.policy_sha256,
            "stored_policy_sha256": self.stored_policy_sha256,
            "car_id": self.car_id,
            "track": self.track,
            "track_profile": self.track_profile,
            "stage": self.stage,
            "drivetrain": self.drivetrain,
            "observation_layout": self.observation_layout,
            "obs_dim": self.obs_dim,
            "sdim": self.sdim,
            "act_dim": self.act_dim,
            "hidden": list(self.hidden),
            "mode": self.mode,
            "eval_protocol": self.eval_protocol,
            "code_fingerprint": self.code_fingerprint,
            "compatibility_class": self.compatibility_class,
            "checkpoint_time_unix": self.checkpoint_time_unix,
            "evidence": dict(self.evidence),
        }


def _empty_record(name: str, edition: FableEdition, *, rejection: str,
                  file_hash: str | None = None,
                  file_bytes: int | None = None) -> CheckpointRecord:
    return CheckpointRecord(
        name=name,
        edition=edition.id,
        classification="incompatible",
        playable=False,
        warnings=(),
        rejection=rejection,
        file_sha256=file_hash,
        file_bytes=file_bytes,
        policy_sha256=None,
        stored_policy_sha256=None,
        car_id=None,
        track=None,
        track_profile=None,
        stage=None,
        drivetrain=None,
        observation_layout=None,
        obs_dim=None,
        sdim=None,
        act_dim=None,
        hidden=(),
        mode=None,
        eval_protocol=None,
        code_fingerprint=None,
        compatibility_class=edition.compatibility_class,
        checkpoint_time_unix=None,
        evidence={},
    )


def _as_int(payload: Mapping[str, Any], key: str) -> int | None:
    try:
        return int(payload.get(key))
    except (TypeError, ValueError):
        return None


def _as_float(payload: Mapping[str, Any], key: str) -> float | None:
    try:
        value = float(payload.get(key))
    except (TypeError, ValueError):
        return None
    return value if value == value and abs(value) != float("inf") else None


def _identity_errors(payload: Mapping[str, Any], edition: FableEdition,
                     hidden: tuple[int, ...]) -> list[str]:
    errors: list[str] = []
    checks = (
        (payload.get("fable_pipeline") is True, "not a Fable Five checkpoint"),
        (payload.get("car") == edition.car_id,
         f"checkpoint car is {payload.get('car')!r}, expected {edition.car_id!r}"),
        (payload.get("track") == FABLE_TRACK,
         f"checkpoint track is {payload.get('track')!r}, expected {FABLE_TRACK!r}"),
        (payload.get("track_profile") == FABLE_TRACK_PROFILE,
         "checkpoint does not use the full Nordschleife Fable profile"),
        (payload.get("mode") == "race", "checkpoint is not a race policy"),
        (payload.get("fable_stage") in FABLE_STAGES,
         f"invalid Fable stage: {payload.get('fable_stage')!r}"),
        (payload.get("fable_drivetrain_version") == edition.drivetrain,
         "checkpoint drivetrain does not match the selected edition"),
        (payload.get("obs_layout") == edition.observation_layout,
         "checkpoint observation layout does not match the selected edition"),
        (payload.get("sensor_pace_block") is True,
         "checkpoint does not contain the required Fable pace block"),
        (bool(payload.get("sensor_hybrid_block")) == edition.hybrid_observations,
         "checkpoint hybrid observation block does not match the selected edition"),
        (_as_int(payload, "obs_dim") == edition.obs_dim,
         f"checkpoint obs_dim does not match {edition.obs_dim}"),
        (_as_int(payload, "sdim") == edition.sdim,
         f"checkpoint sdim does not match {edition.sdim}"),
        (_as_int(payload, "act_dim") == edition.act_dim,
         f"checkpoint act_dim does not match {edition.act_dim}"),
        (hidden == (128, 128), "checkpoint hidden layout is not [128, 128]"),
    )
    errors.extend(message for ok, message in checks if not ok)
    return errors


def classify_checkpoint(checkpoint_root: str | Path,
                        edition: str | FableEdition,
                        checkpoint: str | os.PathLike[str]) -> CheckpointRecord:
    """Classify a checkpoint as current, historical, or incompatible.

    Unsafe names raise immediately.  A safely named but corrupt or mismatched
    file returns an ``incompatible`` record so catalogue endpoints can explain
    why it is unavailable without ever loading it into a live policy.
    """
    selected = get_edition(edition)
    path = resolve_checkpoint_path(checkpoint_root, checkpoint)
    name = path.name
    try:
        file_bytes = path.stat().st_size
    except OSError as exc:
        return _empty_record(name, selected,
                             rejection=f"cannot read checkpoint: {exc}")
    if file_bytes <= 0 or file_bytes > MAX_CHECKPOINT_BYTES:
        return _empty_record(
            name,
            selected,
            rejection=(f"checkpoint size {file_bytes} bytes is outside the "
                       f"safe 1..{MAX_CHECKPOINT_BYTES} byte envelope"),
            file_hash=None,
            file_bytes=file_bytes,
        )
    try:
        file_hash = _file_sha256(path)
    except OSError as exc:
        return _empty_record(name, selected,
                             rejection=f"cannot read checkpoint: {exc}",
                             file_bytes=file_bytes)

    try:
        from .checkpoint_io import load_torch_checkpoint_safe
        from .ppo import ActorCritic, state_dict_hash, validate_checkpoint_payload

        payload = load_torch_checkpoint_safe(path, map_location="cpu")
        validate_checkpoint_payload(payload, require_hash=False)
        actual_policy_hash = state_dict_hash(payload["state_dict"])
        hidden = tuple(int(x) for x in payload.get("hidden", ()))
        obs_dim = _as_int(payload, "obs_dim")
        act_dim = _as_int(payload, "act_dim")
        if obs_dim is None or act_dim is None or not hidden:
            raise ValueError("checkpoint architecture metadata is incomplete")
        # Strict state loading catches tensor-shape or key mismatches that the
        # metadata-only checks cannot detect.
        net = ActorCritic(
            obs_dim,
            act_dim,
            hidden,
            action_log_std_max=payload.get("action_log_std_max"),
        )
        net.load_state_dict(payload["state_dict"], strict=True)
    except Exception as exc:
        return _empty_record(
            name,
            selected,
            rejection=f"unsafe or corrupt checkpoint: {type(exc).__name__}: {exc}",
            file_hash=file_hash,
            file_bytes=file_bytes,
        )

    errors = _identity_errors(payload, selected, hidden)
    evidence = _evaluation_summary(payload)
    if errors:
        record = _empty_record(
            name,
            selected,
            rejection="; ".join(errors),
            file_hash=file_hash,
            file_bytes=file_bytes,
        )
        return replace(
            record,
            policy_sha256=actual_policy_hash,
            stored_policy_sha256=payload.get("policy_sha256"),
            car_id=payload.get("car"),
            track=payload.get("track"),
            track_profile=payload.get("track_profile"),
            stage=payload.get("fable_stage"),
            drivetrain=payload.get("fable_drivetrain_version"),
            observation_layout=payload.get("obs_layout"),
            obs_dim=_as_int(payload, "obs_dim"),
            sdim=_as_int(payload, "sdim"),
            act_dim=_as_int(payload, "act_dim"),
            hidden=hidden,
            mode=payload.get("mode"),
            eval_protocol=payload.get("fable_eval_protocol"),
            code_fingerprint=payload.get("fable_code_fingerprint"),
            checkpoint_time_unix=_as_float(payload, "checkpoint_time_unix"),
            evidence=evidence,
        )

    current_failures: list[str] = []
    if payload.get("policy_sha256") != actual_policy_hash:
        current_failures.append("checkpoint does not store its exact policy hash")
    if payload.get("fable_eval_protocol") != CURRENT_EVAL_PROTOCOL:
        current_failures.append("evaluation protocol is not current")
    if payload.get("fable_code_fingerprint") not in _accepted_fingerprints(selected):
        current_failures.append("source fingerprint is historical")
    if evidence.get("stage") != payload.get("fable_stage"):
        current_failures.append("stored evaluation stage does not match checkpoint stage")
    if evidence.get("evaluated_policy_sha256") != actual_policy_hash:
        current_failures.append("stored evaluation does not bind this exact policy")

    warnings: list[str] = []
    if current_failures:
        classification = "historical"
        warnings.append(
            "Historical Fable policy: playable for visualization only; it is "
            "not current-protocol or certification evidence."
        )
        warnings.extend(current_failures)
    else:
        classification = "current"
    if selected.authority_warning:
        warnings.append(selected.authority_warning)

    return CheckpointRecord(
        name=name,
        edition=selected.id,
        classification=classification,
        playable=True,
        warnings=tuple(warnings),
        rejection=None,
        file_sha256=file_hash,
        file_bytes=file_bytes,
        policy_sha256=actual_policy_hash,
        stored_policy_sha256=payload.get("policy_sha256"),
        car_id=payload.get("car"),
        track=payload.get("track"),
        track_profile=payload.get("track_profile"),
        stage=payload.get("fable_stage"),
        drivetrain=payload.get("fable_drivetrain_version"),
        observation_layout=payload.get("obs_layout"),
        obs_dim=_as_int(payload, "obs_dim"),
        sdim=_as_int(payload, "sdim"),
        act_dim=_as_int(payload, "act_dim"),
        hidden=hidden,
        mode=payload.get("mode"),
        eval_protocol=payload.get("fable_eval_protocol"),
        code_fingerprint=payload.get("fable_code_fingerprint"),
        compatibility_class=selected.compatibility_class,
        checkpoint_time_unix=_as_float(payload, "checkpoint_time_unix"),
        evidence=evidence,
    )


def require_playable_checkpoint(checkpoint_root: str | Path,
                                edition: str | FableEdition,
                                checkpoint: str | os.PathLike[str]
                                ) -> CheckpointRecord:
    record = classify_checkpoint(checkpoint_root, edition, checkpoint)
    if not record.playable:
        raise CheckpointRejected(
            f"checkpoint {record.name!r} rejected for edition {record.edition}: "
            f"{record.rejection or 'incompatible'}"
        )
    return record


def discover_checkpoints(checkpoint_root: str | Path,
                         edition: str | FableEdition,
                         *, include_incompatible: bool = True
                         ) -> list[CheckpointRecord]:
    """Inspect immediate ``fable5*.pt`` candidates without following trees."""
    root = Path(checkpoint_root).expanduser().resolve()
    records: list[CheckpointRecord] = []
    for path in sorted(root.glob("fable5*.pt"), key=lambda p: p.name):
        if not path.is_file():
            continue
        try:
            record = classify_checkpoint(root, edition, path.name)
        except CheckpointRejected as exc:
            record = _empty_record(path.name, get_edition(edition),
                                   rejection=str(exc))
        if include_incompatible or record.playable:
            records.append(record)
    order = {"current": 0, "historical": 1, "incompatible": 2}
    return sorted(records, key=lambda r: (order.get(r.classification, 9), r.name))


def load_edition_manifest(checkpoint_root: str | Path,
                          edition: str | FableEdition) -> dict[str, Any]:
    selected = get_edition(edition)
    root = Path(checkpoint_root).expanduser().resolve()
    path = root / selected.manifest
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}

    state = dict(payload)
    warnings: list[str] = []
    stamped_car = state.get("car")
    stamped_track = state.get("track")
    stamped_drivetrain = state.get("drivetrain_version")
    if stamped_car not in (None, selected.car_id):
        return {
            "_identity_warnings": [
                f"manifest car {stamped_car!r} does not match edition "
                f"{selected.car_id!r}; state hidden"
            ]
        }
    if stamped_track not in (None, FABLE_TRACK):
        return {
            "_identity_warnings": [
                f"manifest track {stamped_track!r} is not {FABLE_TRACK!r}; state hidden"
            ]
        }
    if stamped_drivetrain not in (None, selected.drivetrain):
        return {
            "_identity_warnings": [
                "manifest drivetrain does not match this edition; state hidden"
            ]
        }

    # Never surface cross-edition checkpoint pointers merely because a local
    # JSON file names them.  Each pointer must independently prove identity.
    for key in ("active_checkpoint", "active_best_checkpoint", "best_checkpoint"):
        checkpoint = state.get(key)
        if not checkpoint:
            continue
        try:
            require_playable_checkpoint(root, selected, str(checkpoint))
        except CheckpointRejected as exc:
            state[key] = None
            warnings.append(f"{key} hidden: {exc}")

    latest = state.get("latest_eval")
    if isinstance(latest, Mapping):
        latest_car = latest.get("car")
        latest_drivetrain = latest.get("drivetrain_version")
        if (latest_car not in (None, selected.car_id)
                or latest_drivetrain not in (None, selected.drivetrain)):
            state["latest_eval"] = None
            warnings.append("latest evaluation identity does not match; hidden")

    auto = state.get("auto")
    if auto:
        inflight = auto.get("inflight") if isinstance(auto, Mapping) else None
        checkpoint = inflight.get("checkpoint") if isinstance(inflight, Mapping) else None
        auto_proven = stamped_car == selected.car_id
        if checkpoint:
            try:
                require_playable_checkpoint(root, selected, str(checkpoint))
                auto_proven = True
            except CheckpointRejected as exc:
                warnings.append(f"auto state hidden: {exc}")
                auto_proven = False
        if not auto_proven:
            state["auto"] = None
            if not checkpoint:
                warnings.append("unproved edition identity for auto state; hidden")
    if warnings:
        state["_identity_warnings"] = warnings
    return state


def _rank_key(record: CheckpointRecord) -> tuple[Any, ...]:
    ev = record.evidence
    stage_rank = FABLE_STAGES.index(record.stage) if record.stage in FABLE_STAGES else -1
    try:
        lap = float(ev.get("lap_time")) if ev.get("lap_time") else None
    except (TypeError, ValueError):
        lap = None
    if lap is not None and lap > 0.0:
        return (
            1, -lap, int(ev.get("clean_sectors") or 0),
            -float(ev.get("terminal_rate") if ev.get("terminal_rate") is not None else 1.0),
            stage_rank, float(ev.get("pace_ratio") or 0.0),
            float(ev.get("metric") or -1e9), record.name,
        )
    return (
        0, float(ev.get("progress_frac") or ev.get("distance_laps") or 0.0),
        int(ev.get("clean_chain") or 0), int(ev.get("clean_sectors") or 0),
        -float(ev.get("terminal_rate") if ev.get("terminal_rate") is not None else 1.0),
        float(ev.get("pace_ratio") or 0.0), stage_rank,
        float(ev.get("metric") or -1e9), record.name,
    )


@dataclass(frozen=True)
class ResolvedCheckpoint:
    checkpoint: CheckpointRecord
    reason: str
    resolution_warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        payload = self.checkpoint.as_dict()
        payload["resolution_reason"] = self.reason
        payload["resolution_warnings"] = list(self.resolution_warnings)
        return payload


def resolve_checkpoint(checkpoint_root: str | Path,
                       edition: str | FableEdition,
                       *, explicit_selection: str | None = None,
                       manifest: Mapping[str, Any] | None = None
                       ) -> ResolvedCheckpoint:
    """Resolve the Observatory policy in the mandated, fail-closed order."""
    selected = get_edition(edition)
    root = Path(checkpoint_root).expanduser().resolve()
    resolution_warnings: list[str] = []

    if explicit_selection and explicit_selection != "active-best":
        record = require_playable_checkpoint(root, selected, explicit_selection)
        return ResolvedCheckpoint(record, "explicit selection")

    state = dict(manifest) if manifest is not None else load_edition_manifest(root, selected)
    manifest_identity_ok = True
    if state.get("car") not in (None, selected.car_id):
        manifest_identity_ok = False
        resolution_warnings.append(
            "edition manifest car stamp does not match; live pointers ignored"
        )
    if state.get("track") not in (None, FABLE_TRACK):
        manifest_identity_ok = False
        resolution_warnings.append(
            "edition manifest track stamp does not match; live pointers ignored"
        )

    if manifest_identity_ok:
        active = state.get("active_best_checkpoint")
        current_stage = state.get("current_stage")
        if active:
            try:
                record = require_playable_checkpoint(root, selected, str(active))
                if current_stage in FABLE_STAGES and record.stage == current_stage:
                    return ResolvedCheckpoint(
                        record, "valid active-stage best", tuple(resolution_warnings)
                    )
                resolution_warnings.append(
                    "active-best checkpoint stage does not match the manifest current stage"
                )
            except CheckpointRejected as exc:
                resolution_warnings.append(f"active-best ignored: {exc}")

    try:
        champion = require_playable_checkpoint(
            root, selected, selected.canonical_champion
        )
    except CheckpointRejected as exc:
        resolution_warnings.append(f"canonical champion unavailable: {exc}")
    else:
        return ResolvedCheckpoint(
            champion, "valid canonical champion", tuple(resolution_warnings)
        )

    records = discover_checkpoints(root, selected, include_incompatible=False)
    current_stage_bests = [
        record for record in records
        if record.classification == "current"
        and record.stage in FABLE_STAGES
        and record.name.endswith("_best.pt")
    ]
    if current_stage_bests:
        return ResolvedCheckpoint(
            max(current_stage_bests, key=_rank_key),
            "current-protocol ranked stage best",
            tuple(resolution_warnings),
        )

    historical = [record for record in records
                  if record.classification == "historical"]
    if historical:
        # Prefer durable stage-best files to transient latest/HOF copies when
        # a lineage supplies them; evidence ranking then selects among stages.
        historical_stage_bests = [record for record in historical
                                  if record.name.endswith("_best.pt")]
        if historical_stage_bests:
            historical = historical_stage_bests
        return ResolvedCheckpoint(
            max(historical, key=_rank_key),
            "playable historical fallback",
            tuple(resolution_warnings),
        )

    detail = "; ".join(resolution_warnings) or "no compatible Fable checkpoint found"
    raise CheckpointRejected(
        f"no playable checkpoint for Fable edition {selected.id}: {detail}"
    )


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp",
                                    dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _unique_backup_dir(root: Path, stamp: str) -> Path:
    base = root / "runtime" / "manifest-repair" / stamp
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}-{suffix}")
        suffix += 1
    return candidate


def repair_manifest_auto_blocks(checkpoint_root: str | Path, *,
                                dry_run: bool = False,
                                timestamp: str | None = None
                                ) -> dict[str, Any]:
    """Move provably cross-edition ``auto`` state, preserving ambiguity.

    The inflight checkpoint is the only accepted identity proof.  A missing,
    corrupt, path-unsafe, or incompatible checkpoint leaves its source
    manifest byte-for-byte untouched and produces a warning.
    """
    root = Path(checkpoint_root).expanduser().resolve()
    manifests: dict[str, dict[str, Any] | None] = {}
    paths: dict[str, Path] = {}
    warnings: list[str] = []
    for edition_id, edition in FABLE_EDITIONS.items():
        path = root / edition.manifest
        paths[edition_id] = path
        manifests[edition_id] = _read_manifest(path)
        if manifests[edition_id] is None:
            warnings.append(f"{edition.manifest}: invalid JSON; left untouched")

    candidates: list[tuple[str, str, dict[str, Any]]] = []
    for source_id, source in manifests.items():
        if not isinstance(source, dict):
            continue
        auto = source.get("auto")
        if not auto:
            continue
        if not isinstance(auto, dict):
            warnings.append(
                f"{paths[source_id].name}: auto block is not an object; left untouched"
            )
            continue
        inflight = auto.get("inflight")
        checkpoint = inflight.get("checkpoint") if isinstance(inflight, dict) else None
        if not checkpoint:
            warnings.append(
                f"{paths[source_id].name}: auto block has no inflight checkpoint; left untouched"
            )
            continue
        try:
            checkpoint_path = resolve_checkpoint_path(root, str(checkpoint))
            from .checkpoint_io import load_torch_checkpoint_safe
            from .ppo import validate_checkpoint_payload
            payload = load_torch_checkpoint_safe(checkpoint_path, map_location="cpu")
            validate_checkpoint_payload(payload, require_hash=False)
            destination = edition_for_car(str(payload.get("car") or ""))
            if destination is None:
                raise CheckpointRejected(
                    f"checkpoint car {payload.get('car')!r} is not a registered edition"
                )
            require_playable_checkpoint(root, destination, checkpoint_path.name)
        except Exception as exc:
            warnings.append(
                f"{paths[source_id].name}: cannot prove auto-block destination "
                f"from {checkpoint!r} ({type(exc).__name__}: {exc}); left untouched"
            )
            continue
        if destination.id != source_id:
            candidates.append((source_id, destination.id, auto))

    # More than one different source targeting the same occupied destination is
    # not safe to arbitrate automatically.
    blocked: set[tuple[str, str]] = set()
    for source_id, destination_id, auto in candidates:
        destination = manifests.get(destination_id)
        if not isinstance(destination, dict):
            warnings.append(
                f"{paths[destination_id].name}: invalid destination manifest; "
                f"auto block in {paths[source_id].name} left untouched"
            )
            blocked.add((source_id, destination_id))
            continue
        existing = destination.get("auto")
        if existing not in (None, auto):
            warnings.append(
                f"{paths[destination_id].name}: destination already has different "
                f"auto state; {paths[source_id].name} left untouched"
            )
            blocked.add((source_id, destination_id))

    for destination_id in FABLE_EDITIONS:
        incoming = [(s, d, a) for s, d, a in candidates
                    if d == destination_id and (s, d) not in blocked]
        if len(incoming) > 1:
            signatures = {json.dumps(item[2], sort_keys=True, default=str)
                          for item in incoming}
            if len(signatures) > 1:
                for source_id, dest_id, _ in incoming:
                    blocked.add((source_id, dest_id))
                warnings.append(
                    f"{paths[destination_id].name}: multiple different incoming "
                    "auto blocks; all sources left untouched"
                )

    moves = [item for item in candidates if (item[0], item[1]) not in blocked]
    report: dict[str, Any] = {
        "schema": "fable-manifest-repair-v1",
        "dry_run": bool(dry_run),
        "changed": False,
        "would_change": bool(moves),
        "backup_dir": None,
        "moves": [
            {
                "source": paths[source].name,
                "destination": paths[destination].name,
                "inflight_checkpoint": ((auto.get("inflight") or {}).get("checkpoint")),
            }
            for source, destination, auto in moves
        ],
        "warnings": warnings,
    }
    if dry_run or not moves:
        return report

    stamp = timestamp or time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    if not stamp or any(ch not in "0123456789T-_" for ch in stamp):
        raise ValueError("repair timestamp may contain only digits, T, hyphen, underscore")
    backup_dir = _unique_backup_dir(root, stamp)
    backup_dir.mkdir(parents=True, exist_ok=False)
    affected = {paths[source] for source, _, _ in moves}
    affected.update(paths[destination] for _, destination, _ in moves)
    for path in sorted(affected, key=lambda p: p.name):
        if path.is_file():
            shutil.copy2(path, backup_dir / path.name)

    # Destination first makes an interrupted operation recoverable as a safe
    # duplicate; a later run recognizes equal state and only clears the source.
    for source_id, destination_id, auto in moves:
        destination = dict(manifests[destination_id] or {})
        destination["auto"] = auto
        _atomic_write_json(paths[destination_id], destination)
        manifests[destination_id] = destination
    for source_id, _, _ in moves:
        source = dict(manifests[source_id] or {})
        source["auto"] = None
        _atomic_write_json(paths[source_id], source)
        manifests[source_id] = source

    report["changed"] = True
    report["backup_dir"] = str(backup_dir.relative_to(root))
    _atomic_write_json(backup_dir / "repair-report.json", report)
    return report


def edition_catalog(checkpoint_root: str | Path,
                    edition: str | FableEdition) -> dict[str, Any]:
    """Build one edition-scoped, JSON-ready Observatory catalogue record."""
    selected = get_edition(edition)
    manifest = load_edition_manifest(checkpoint_root, selected)
    checkpoints = discover_checkpoints(checkpoint_root, selected)
    try:
        resolved = resolve_checkpoint(
            checkpoint_root, selected, manifest=manifest
        ).as_dict()
        resolution_error = None
    except CheckpointRejected as exc:
        resolved = None
        resolution_error = str(exc)
    return {
        **selected.as_dict(),
        "state": manifest,
        "resolved_checkpoint": resolved,
        "resolution_error": resolution_error,
        "checkpoints": [record.as_dict() for record in checkpoints],
    }


def catalog_payload(checkpoint_root: str | Path) -> dict[str, Any]:
    """Complete server-side payload for ``GET /api/observatory/catalog``."""
    return {
        "schema": "fable-observatory-catalog-v1",
        "editions": [edition_catalog(checkpoint_root, edition_id)
                     for edition_id in ("787b", "919")],
    }


__all__ = [
    "CheckpointRecord",
    "CheckpointRejected",
    "CURRENT_EVAL_PROTOCOL",
    "FABLE_EDITIONS",
    "FABLE_STAGES",
    "FABLE_TRACK",
    "FABLE_TRACK_PROFILE",
    "MAX_CHECKPOINT_BYTES",
    "FableBenchmark",
    "FableEdition",
    "ResolvedCheckpoint",
    "UnsafeCheckpointPath",
    "catalog_payload",
    "classify_checkpoint",
    "discover_checkpoints",
    "edition_for_car",
    "edition_catalog",
    "get_edition",
    "load_edition_manifest",
    "registry_payload",
    "repair_manifest_auto_blocks",
    "require_playable_checkpoint",
    "resolve_checkpoint",
    "resolve_checkpoint_path",
]
