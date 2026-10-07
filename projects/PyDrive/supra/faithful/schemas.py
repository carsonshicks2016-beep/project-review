"""Immutable, evidence-aware interfaces for the faithful-v2 record program.

The types in this module deliberately separate *having a numerical value* from
*having enough evidence to make a fidelity claim*.  Public anchors are useful
for building an approximation, but the readiness evaluator remains fail-closed
until high-sensitivity inputs have bounded, licensed or calibrated evidence and
the certification track is backed by a verified survey.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum, IntEnum
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse


Vec3 = tuple[float, float, float]
WheelScalars = tuple[float | None, float | None, float | None, float | None]
WheelContacts = tuple[bool | None, bool | None, bool | None, bool | None]
WheelVectors = tuple[Vec3 | None, Vec3 | None, Vec3 | None, Vec3 | None]
NumericValue = float | tuple[float, ...]

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _finite(value: float, name: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")


def _unit_interval(value: float, name: str) -> None:
    _finite(value, name)
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")


def _actual_bool(value: Any, name: str) -> None:
    """Reject truthy strings/integers at evidence and control boundaries."""
    if type(value) is not bool:
        raise TypeError(f"{name} must be an actual bool")


def _vec3(value: Vec3, name: str) -> None:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    for index, component in enumerate(value):
        _finite(component, f"{name}[{index}]")


def _wheel_tuple(value: tuple[Any, ...], name: str) -> None:
    if len(value) != 4:
        raise ValueError(f"{name} must contain FL, FR, RL and RR values")


def _primitive(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {field.name: _primitive(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_primitive(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _primitive(item) for key, item in sorted(value.items())}
    return value


def canonical_json(value: Any) -> str:
    """Return a stable JSON representation suitable for hashing/evidence bundles."""
    return json.dumps(_primitive(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _materialized_artifact_matches(uri: str, expected_sha256: str) -> bool:
    """Verify a local materialized artifact; declarations and remote URLs do not pass."""
    parsed = urlparse(uri)
    if parsed.scheme == "file":
        path = Path(unquote(parsed.path))
    elif not parsed.scheme:
        path = Path(uri)
        if not path.is_absolute():
            return False
    else:
        return False
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == expected_sha256


class Confidence(str, Enum):
    OFFICIAL = "official"
    LICENSED_MEASURED = "licensed-measured"
    CALIBRATED = "calibrated"
    INFERRED = "inferred"


class DistributionKind(str, Enum):
    FIXED = "fixed"
    UNIFORM = "uniform"
    NORMAL = "normal"
    EMPIRICAL = "empirical"


@dataclass(frozen=True, slots=True)
class UncertaintyDistribution:
    """A compact uncertainty description; no silent zero-uncertainty defaults."""

    kind: DistributionKind
    lower: float | None = None
    upper: float | None = None
    sigma: float | None = None
    dataset_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in ("lower", "upper", "sigma"):
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("uncertainty lower bound exceeds upper bound")
        if self.sigma is not None and self.sigma < 0.0:
            raise ValueError("uncertainty sigma must be nonnegative")
        if self.kind is DistributionKind.FIXED and any(
            value is not None for value in (self.lower, self.upper, self.sigma)
        ):
            raise ValueError("fixed distributions cannot carry bounds or sigma")
        if self.kind is DistributionKind.UNIFORM and (self.lower is None or self.upper is None):
            raise ValueError("uniform distributions require lower and upper bounds")
        if self.kind is DistributionKind.NORMAL and self.sigma is None:
            raise ValueError("normal distributions require sigma")
        if self.kind is DistributionKind.EMPIRICAL:
            if self.dataset_sha256 is None or not _SHA256_RE.fullmatch(self.dataset_sha256):
                raise ValueError("empirical distributions require a dataset SHA-256")

    @property
    def bounded(self) -> bool:
        return self.kind is DistributionKind.FIXED or (
            self.lower is not None and self.upper is not None
        )


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    """Provenance for one physical parameter.

    ``source_sha256`` hashes the object described by ``digest_scope``.  Public
    defaults use ``citation-record`` digests and therefore cannot masquerade as
    captured source contents.  Licensed ingesters should use ``content``.
    """

    source_uri: str
    source_sha256: str
    digest_scope: str
    license_id: str
    confidence: Confidence
    calibration_dataset: str
    calibration_dataset_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.source_uri:
            raise ValueError("evidence source_uri is required")
        if not _SHA256_RE.fullmatch(self.source_sha256):
            raise ValueError("evidence source_sha256 must be 64 lowercase hex characters")
        if self.digest_scope not in {"content", "citation-record", "derived-record"}:
            raise ValueError("digest_scope must be content, citation-record or derived-record")
        if not self.license_id:
            raise ValueError("evidence license_id is required")
        if not self.calibration_dataset:
            raise ValueError("calibration_dataset is required; use 'not-applicable' explicitly")
        if self.calibration_dataset_sha256 is not None and not _SHA256_RE.fullmatch(
            self.calibration_dataset_sha256
        ):
            raise ValueError("calibration_dataset_sha256 must be a SHA-256 when supplied")

    @property
    def verified_content(self) -> bool:
        return (
            self.digest_scope == "content"
            and _materialized_artifact_matches(
                self.source_uri, self.source_sha256
            )
        )

    @property
    def calibration_materialized(self) -> bool:
        return (
            self.calibration_dataset_sha256 is not None
            and _materialized_artifact_matches(
                self.calibration_dataset,
                self.calibration_dataset_sha256,
            )
        )


@dataclass(frozen=True, slots=True)
class PhysicalParameter:
    name: str
    value: NumericValue
    unit: str
    distribution: UncertaintyDistribution
    evidence: EvidenceRef
    high_sensitivity: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        _actual_bool(self.high_sensitivity, "high_sensitivity")
        if not self.name or not self.unit:
            raise ValueError("physical parameters require a name and SI unit")
        values = self.value if isinstance(self.value, tuple) else (self.value,)
        if not values:
            raise ValueError(f"{self.name} cannot be an empty tuple")
        for index, value in enumerate(values):
            _finite(value, f"{self.name}[{index}]")

    def scalar(self) -> float:
        if isinstance(self.value, tuple):
            raise TypeError(f"{self.name} is vector-valued")
        return float(self.value)


@dataclass(frozen=True, slots=True)
class VehicleSpecV2:
    schema_version: str
    vehicle_id: str
    geometry: tuple[PhysicalParameter, ...]
    mass_properties: tuple[PhysicalParameter, ...]
    suspension: tuple[PhysicalParameter, ...]
    tyres: tuple[PhysicalParameter, ...]
    aero: tuple[PhysicalParameter, ...]
    ice: tuple[PhysicalParameter, ...]
    mgu: tuple[PhysicalParameter, ...]
    battery: tuple[PhysicalParameter, ...]
    gearbox: tuple[PhysicalParameter, ...]
    brakes: tuple[PhysicalParameter, ...]
    embedded_controllers: tuple[PhysicalParameter, ...]
    thermal_limits: tuple[PhysicalParameter, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "vehicle-spec-v2":
            raise ValueError("VehicleSpecV2 requires schema_version='vehicle-spec-v2'")
        if not self.vehicle_id:
            raise ValueError("vehicle_id is required")
        names = [parameter.name for _, parameter in self.iter_parameters()]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"vehicle parameter names must be unique: {duplicates}")

    def groups(self) -> tuple[tuple[str, tuple[PhysicalParameter, ...]], ...]:
        return tuple(
            (name, getattr(self, name))
            for name in (
                "geometry", "mass_properties", "suspension", "tyres", "aero", "ice",
                "mgu", "battery", "gearbox", "brakes", "embedded_controllers",
                "thermal_limits",
            )
        )

    def iter_parameters(self) -> Iterable[tuple[str, PhysicalParameter]]:
        for group_name, group in self.groups():
            for parameter in group:
                yield group_name, parameter

    def parameter(self, name: str) -> PhysicalParameter:
        for _, parameter in self.iter_parameters():
            if parameter.name == name:
                return parameter
        raise KeyError(name)

    @property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class TrackSampleV2:
    station_m: float
    center_m: Vec3
    left_boundary_m: Vec3
    right_boundary_m: Vec3
    surface_material: str
    friction_region: str

    def __post_init__(self) -> None:
        _finite(self.station_m, "station_m")
        if self.station_m < 0.0:
            raise ValueError("track station cannot be negative")
        _vec3(self.center_m, "center_m")
        _vec3(self.left_boundary_m, "left_boundary_m")
        _vec3(self.right_boundary_m, "right_boundary_m")
        if not self.surface_material or not self.friction_region:
            raise ValueError("track samples require material and friction-region labels")


@dataclass(frozen=True, slots=True)
class TimingPlaneV1:
    name: str
    point_m: Vec3
    normal: Vec3

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("timing plane name is required")
        _vec3(self.point_m, "point_m")
        _vec3(self.normal, "normal")
        magnitude = math.sqrt(sum(component * component for component in self.normal))
        if magnitude < 0.999 or magnitude > 1.001:
            raise ValueError("timing plane normal must be normalized")


@dataclass(frozen=True, slots=True)
class TrackSurveyValidationV1:
    protocol_version: str
    passed: bool
    artifact_uri: str
    artifact_sha256: str
    verifier_signature_sha256: str
    evidence: EvidenceRef
    lap_length_error_m: float
    max_boundary_error_m: float
    max_vertical_error_m: float
    max_bank_error_deg: float

    def __post_init__(self) -> None:
        _actual_bool(self.passed, "track survey validation passed")
        if self.protocol_version != "track-survey-validation-v1":
            raise ValueError("unexpected track-survey validation protocol")
        if not self.artifact_uri:
            raise ValueError("track survey validation artifact_uri is required")
        for name in ("artifact_sha256", "verifier_signature_sha256"):
            value = getattr(self, name)
            if not _SHA256_RE.fullmatch(value) or value == "0" * 64:
                raise ValueError(f"{name} must be a nonzero SHA-256")
        for name in (
            "lap_length_error_m", "max_boundary_error_m", "max_vertical_error_m",
            "max_bank_error_deg",
        ):
            value = getattr(self, name)
            _finite(value, name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class TrackSurfaceV2:
    schema_version: str
    track_id: str
    configuration: str
    lap_length: PhysicalParameter
    survey_evidence: EvidenceRef
    samples: tuple[TrackSampleV2, ...]
    timing_planes: tuple[TimingPlaneV1, ...]
    curbs_sha256: str | None
    barriers_sha256: str | None
    materials_sha256: str | None
    certification_eligible: bool
    approximation_reason: str | None = None
    survey_validation: TrackSurveyValidationV1 | None = None

    def __post_init__(self) -> None:
        _actual_bool(self.certification_eligible, "certification_eligible")
        if self.schema_version != "track-surface-v2":
            raise ValueError("TrackSurfaceV2 requires schema_version='track-surface-v2'")
        if not self.track_id or not self.configuration:
            raise ValueError("track_id and configuration are required")
        if self.lap_length.scalar() <= 0.0:
            raise ValueError("track lap length must be positive")
        stations = tuple(sample.station_m for sample in self.samples)
        if stations and any(b <= a for a, b in zip(stations, stations[1:])):
            raise ValueError("track sample stations must be strictly increasing")
        if self.certification_eligible:
            if len(self.samples) < 2:
                raise ValueError("certification tracks require surveyed samples")
            if not self.survey_evidence.verified_content:
                raise ValueError("certification tracks require content-hashed survey evidence")
            for digest in (self.curbs_sha256, self.barriers_sha256, self.materials_sha256):
                if digest is None or not _SHA256_RE.fullmatch(digest):
                    raise ValueError("certification tracks require curb/barrier/material hashes")
            if self.survey_validation is None:
                raise ValueError("certification tracks require a survey-validation artifact")
        elif not self.approximation_reason:
            raise ValueError("noncertification tracks require an approximation reason")

    @property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class PermittedDriverControlsV1:
    steering: bool = True
    accelerator: bool = True
    brake: bool = True
    shift_request: bool = True
    boost_request: bool = False
    drs_request: bool = False

    def __post_init__(self) -> None:
        for name in (
            "steering", "accelerator", "brake", "shift_request",
            "boost_request", "drs_request",
        ):
            _actual_bool(getattr(self, name), name)


class ValidationArtifactKind(str, Enum):
    COMPONENT = "component-fidelity"
    SPA = "spa-calibration"
    NORD_HOLDOUT = "nordschleife-holdout"
    NUMERICS = "numerics-and-twin-correlation"
    ENERGY = "full-lap-energy"


@dataclass(frozen=True, slots=True)
class ValidationMetricV1:
    name: str
    value: float
    unit: str

    def __post_init__(self) -> None:
        if not self.name or not self.unit:
            raise ValueError("validation metrics require name and unit")
        _finite(self.value, self.name)


@dataclass(frozen=True, slots=True)
class ValidationArtifactV1:
    kind: ValidationArtifactKind
    protocol_version: str
    passed: bool
    artifact_uri: str
    artifact_sha256: str
    verifier_signature_sha256: str
    source_evidence: EvidenceRef
    metrics: tuple[ValidationMetricV1, ...]

    def __post_init__(self) -> None:
        _actual_bool(self.passed, "validation artifact passed")
        if self.protocol_version != "physics-validation-v1":
            raise ValueError("unexpected physics-validation protocol")
        if not self.artifact_uri:
            raise ValueError("validation artifact_uri is required")
        for name in ("artifact_sha256", "verifier_signature_sha256"):
            value = getattr(self, name)
            if not _SHA256_RE.fullmatch(value) or value == "0" * 64:
                raise ValueError(f"{name} must be a nonzero SHA-256")
        names = [metric.name for metric in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("validation metric names must be unique")

    def metric_map(self) -> dict[str, float]:
        return {metric.name: metric.value for metric in self.metrics}


@dataclass(frozen=True, slots=True)
class PhysicsValidationEvidenceV1:
    component: ValidationArtifactV1
    spa: ValidationArtifactV1
    nord_holdout: ValidationArtifactV1
    numerics: ValidationArtifactV1
    energy: ValidationArtifactV1
    freeze_manifest_sha256: str
    parameters_frozen_before_holdout: bool

    def __post_init__(self) -> None:
        _actual_bool(
            self.parameters_frozen_before_holdout,
            "parameters_frozen_before_holdout",
        )
        if not _SHA256_RE.fullmatch(self.freeze_manifest_sha256) or self.freeze_manifest_sha256 == "0" * 64:
            raise ValueError("freeze_manifest_sha256 must be a nonzero SHA-256")
        expected = (
            (self.component, ValidationArtifactKind.COMPONENT),
            (self.spa, ValidationArtifactKind.SPA),
            (self.nord_holdout, ValidationArtifactKind.NORD_HOLDOUT),
            (self.numerics, ValidationArtifactKind.NUMERICS),
            (self.energy, ValidationArtifactKind.ENERGY),
        )
        for artifact, kind in expected:
            if artifact.kind is not kind:
                raise ValueError(f"physics validation field requires {kind.value}")


@dataclass(frozen=True, slots=True)
class RecordScenarioV1:
    schema_version: str
    scenario_id: str
    vehicle_spec_sha256: str
    track_surface_sha256: str
    target_lap_time_s: float
    air_temperature: PhysicalParameter
    track_temperature: PhysicalParameter
    air_pressure: PhysicalParameter
    relative_humidity: PhysicalParameter
    wind_speed: PhysicalParameter
    wind_direction: PhysicalParameter
    initial_fuel_mass: PhysicalParameter
    initial_battery_soc: PhysicalParameter
    initial_tyre_temperature: PhysicalParameter
    initial_brake_temperature: PhysicalParameter
    controls: PermittedDriverControlsV1
    timing_rule: str
    holdout_policy: str
    record_telemetry_opened: bool = False
    physics_validation: PhysicsValidationEvidenceV1 | None = None

    def __post_init__(self) -> None:
        _actual_bool(self.record_telemetry_opened, "record_telemetry_opened")
        if self.schema_version != "record-scenario-v1":
            raise ValueError("RecordScenarioV1 requires schema_version='record-scenario-v1'")
        if not self.scenario_id:
            raise ValueError("scenario_id is required")
        for name in ("vehicle_spec_sha256", "track_surface_sha256"):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256")
        _finite(self.target_lap_time_s, "target_lap_time_s")
        if self.target_lap_time_s <= 0.0:
            raise ValueError("target lap time must be positive")
        _unit_interval(self.initial_battery_soc.scalar(), "initial_battery_soc")
        if not self.timing_rule or not self.holdout_policy:
            raise ValueError("timing_rule and holdout_policy are required")

    @property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(frozen=True, slots=True)
class PhysicsIdentity:
    backend_name: str
    backend_version: str
    precision: str
    timestep_s: float
    vehicle_spec_sha256: str
    track_surface_sha256: str
    source_sha256: str
    dependency_sha256: str
    capability_label: str

    def __post_init__(self) -> None:
        if self.precision not in {"float64", "float32"}:
            raise ValueError("physics precision must be float64 or float32")
        _finite(self.timestep_s, "timestep_s")
        if self.timestep_s <= 0.0:
            raise ValueError("physics timestep must be positive")
        for name in (
            "vehicle_spec_sha256", "track_surface_sha256", "source_sha256",
            "dependency_sha256",
        ):
            if not _SHA256_RE.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be a SHA-256")
        if not self.capability_label:
            raise ValueError("capability_label is required")

    @property
    def sha256(self) -> str:
        return content_sha256(self)


class ShiftRequest(IntEnum):
    DOWNSHIFT = -1
    HOLD = 0
    UPSHIFT = 1


@dataclass(frozen=True, slots=True)
class DriverActionV2:
    schema_version: str = "driver-action-v2"
    steering: float = 0.0
    accelerator: float = 0.0
    brake: float = 0.0
    shift_request: ShiftRequest = ShiftRequest.HOLD
    boost_request: bool | None = None
    drs_request: bool | None = None

    def __post_init__(self) -> None:
        if self.schema_version != "driver-action-v2":
            raise ValueError("DriverActionV2 requires schema_version='driver-action-v2'")
        _finite(self.steering, "steering")
        if not -1.0 <= self.steering <= 1.0:
            raise ValueError("steering must be in [-1, 1]")
        _unit_interval(self.accelerator, "accelerator")
        _unit_interval(self.brake, "brake")
        if not isinstance(self.shift_request, ShiftRequest):
            raise TypeError("shift_request must be a ShiftRequest")
        for name in ("boost_request", "drs_request"):
            value = getattr(self, name)
            if value is not None:
                _actual_bool(value, name)

    def assert_permitted(self, controls: PermittedDriverControlsV1) -> None:
        if not controls.steering and self.steering != 0.0:
            raise PermissionError("steering is disabled for this scenario")
        if not controls.accelerator and self.accelerator != 0.0:
            raise PermissionError("accelerator is disabled for this scenario")
        if not controls.brake and self.brake != 0.0:
            raise PermissionError("brake is disabled for this scenario")
        if not controls.shift_request and self.shift_request is not ShiftRequest.HOLD:
            raise PermissionError("shift requests are disabled for this scenario")
        if not controls.boost_request and self.boost_request is not None:
            raise PermissionError("boost_request is disabled without licensed interface evidence")
        if not controls.drs_request and self.drs_request is not None:
            raise PermissionError("drs_request is disabled without licensed interface evidence")


@dataclass(frozen=True, slots=True)
class MapPreviewPointV1:
    distance_ahead_m: float
    center_offset_body_m: Vec3
    left_width_m: float
    right_width_m: float

    def __post_init__(self) -> None:
        _finite(self.distance_ahead_m, "distance_ahead_m")
        _vec3(self.center_offset_body_m, "center_offset_body_m")
        _finite(self.left_width_m, "left_width_m")
        _finite(self.right_width_m, "right_width_m")
        if self.distance_ahead_m < 0.0 or min(self.left_width_m, self.right_width_m) < 0.0:
            raise ValueError("map-preview distances and widths cannot be negative")


class ActiveAeroMode(str, Enum):
    UNKNOWN = "unknown"
    LOW_DRAG = "low-drag"
    HIGH_DOWNFORCE = "high-downforce"
    TRANSITION = "transition"


@dataclass(frozen=True, slots=True)
class DriverControllerStatusV1:
    """Closed actor-visible ECU status; arbitrary privileged labels are impossible."""

    traction_control_active: bool | None
    brake_by_wire_active: bool | None
    hybrid_derated: bool | None
    active_aero_mode: ActiveAeroMode
    fault_code_count: int

    def __post_init__(self) -> None:
        for name in (
            "traction_control_active", "brake_by_wire_active", "hybrid_derated"
        ):
            value = getattr(self, name)
            if value is not None and type(value) is not bool:
                raise TypeError(f"{name} must be bool or None")
        if not isinstance(self.active_aero_mode, ActiveAeroMode):
            raise TypeError("active_aero_mode must be an ActiveAeroMode")
        if type(self.fault_code_count) is not int:
            raise TypeError("fault_code_count must be an integer")
        if self.fault_code_count < 0:
            raise ValueError("fault_code_count cannot be negative")


@dataclass(frozen=True, slots=True)
class DriverObservationV2:
    """Only signals plausibly measurable onboard plus a static map preview."""

    schema_version: str
    time_s: float
    gps_position_m: Vec3
    velocity_body_mps: Vec3
    acceleration_body_mps2: Vec3
    angular_velocity_body_rad_s: Vec3
    wheel_speed_rad_s: WheelScalars
    steering_angle_rad: float
    accelerator_pedal: float
    brake_pedal: float
    gear: int
    engine_rpm: float
    battery_soc: float
    deployment_available_w: float
    temperatures_k: tuple[float, ...]
    controller_status: DriverControllerStatusV1
    static_map_preview: tuple[MapPreviewPointV1, ...]

    def __post_init__(self) -> None:
        if self.schema_version != "driver-observation-v2":
            raise ValueError("DriverObservationV2 requires schema_version='driver-observation-v2'")
        _finite(self.time_s, "time_s")
        for name in (
            "gps_position_m", "velocity_body_mps", "acceleration_body_mps2",
            "angular_velocity_body_rad_s",
        ):
            _vec3(getattr(self, name), name)
        _wheel_tuple(self.wheel_speed_rad_s, "wheel_speed_rad_s")
        for index, speed in enumerate(self.wheel_speed_rad_s):
            if speed is not None:
                _finite(speed, f"wheel_speed_rad_s[{index}]")
        for name in (
            "steering_angle_rad", "accelerator_pedal", "brake_pedal", "engine_rpm",
            "battery_soc", "deployment_available_w",
        ):
            _finite(getattr(self, name), name)
        _unit_interval(self.accelerator_pedal, "accelerator_pedal")
        _unit_interval(self.brake_pedal, "brake_pedal")
        _unit_interval(self.battery_soc, "battery_soc")
        if self.gear < 1:
            raise ValueError("gear must be a positive forward gear")
        if not isinstance(self.controller_status, DriverControllerStatusV1):
            raise TypeError("controller_status must use the closed DriverControllerStatusV1 schema")
        for index, temperature in enumerate(self.temperatures_k):
            _finite(temperature, f"temperatures_k[{index}]")


@dataclass(frozen=True, slots=True)
class TelemetryEvent:
    sequence: int
    time_s: float
    event_type: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence cannot be negative")
        _finite(self.time_s, "event time")
        if not self.event_type:
            raise ValueError("event_type is required")


@dataclass(frozen=True, slots=True)
class VehicleTelemetryV2:
    """Truth telemetry. Unavailable signals remain ``None`` and are never guessed."""

    schema_version: str
    time_s: float
    step_index: int
    position_m: Vec3
    orientation_rpy_rad: Vec3
    velocity_body_mps: Vec3
    acceleration_body_mps2: Vec3
    angular_velocity_body_rad_s: Vec3
    steering_angle_rad: float | None
    accelerator_pedal: float | None
    brake_pedal: float | None
    engine_rpm: float | None
    engine_torque_nm: float | None
    engine_power_w: float | None
    turbo_speed_rad_s: float | None
    turbo_boost_pa: float | None
    gear: int | None
    gearbox_input_speed_rad_s: float | None
    gearbox_output_speed_rad_s: float | None
    shift_state: str | None
    battery_soc: float | None
    battery_voltage_v: float | None
    battery_current_a: float | None
    battery_power_w: float | None
    front_mgu_torque_nm: float | None
    front_mgu_power_w: float | None
    regen_power_w: float | None
    front_aero_position: float | None
    rear_aero_position: float | None
    aero_downforce_n: float | None
    aero_drag_n: float | None
    suspension_travel_m: WheelScalars
    wheel_speed_rad_s: WheelScalars
    wheel_force_xyz_n: WheelVectors
    wheel_slip_ratio: WheelScalars
    wheel_slip_angle_rad: WheelScalars
    wheel_contact: WheelContacts
    brake_pressure_pa: WheelScalars
    brake_temperature_k: WheelScalars
    tyre_temperature_k: WheelScalars
    coolant_temperature_k: float | None
    battery_temperature_k: float | None
    fuel_mass_kg: float | None
    events: tuple[TelemetryEvent, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != "vehicle-telemetry-v2":
            raise ValueError("VehicleTelemetryV2 requires schema_version='vehicle-telemetry-v2'")
        _finite(self.time_s, "time_s")
        if self.step_index < 0:
            raise ValueError("step_index cannot be negative")
        for name in (
            "position_m", "orientation_rpy_rad", "velocity_body_mps",
            "acceleration_body_mps2", "angular_velocity_body_rad_s",
        ):
            _vec3(getattr(self, name), name)
        optional_scalars = (
            "steering_angle_rad", "accelerator_pedal", "brake_pedal", "engine_rpm",
            "engine_torque_nm", "engine_power_w", "turbo_speed_rad_s", "turbo_boost_pa",
            "gearbox_input_speed_rad_s", "gearbox_output_speed_rad_s", "battery_soc",
            "battery_voltage_v", "battery_current_a", "battery_power_w",
            "front_mgu_torque_nm", "front_mgu_power_w", "regen_power_w",
            "front_aero_position", "rear_aero_position", "aero_downforce_n", "aero_drag_n",
            "coolant_temperature_k", "battery_temperature_k", "fuel_mass_kg",
        )
        for name in optional_scalars:
            value = getattr(self, name)
            if value is not None:
                _finite(value, name)
        for name in (
            "suspension_travel_m", "wheel_speed_rad_s", "wheel_force_xyz_n",
            "wheel_slip_ratio", "wheel_slip_angle_rad", "wheel_contact",
            "brake_pressure_pa", "brake_temperature_k", "tyre_temperature_k",
        ):
            _wheel_tuple(getattr(self, name), name)
        for index, contact in enumerate(self.wheel_contact):
            if contact is not None:
                _actual_bool(contact, f"wheel_contact[{index}]")
        for name in (
            "suspension_travel_m", "wheel_speed_rad_s", "wheel_slip_ratio",
            "wheel_slip_angle_rad", "brake_pressure_pa", "brake_temperature_k",
            "tyre_temperature_k",
        ):
            for index, value in enumerate(getattr(self, name)):
                if value is not None:
                    _finite(value, f"{name}[{index}]")
        for index, force in enumerate(self.wheel_force_xyz_n):
            if force is not None:
                _vec3(force, f"wheel_force_xyz_n[{index}]")
        for name in ("accelerator_pedal", "brake_pedal", "battery_soc"):
            value = getattr(self, name)
            if value is not None:
                _unit_interval(value, name)
        for name in ("front_aero_position", "rear_aero_position"):
            value = getattr(self, name)
            if value is not None:
                _unit_interval(value, name)
        if self.regen_power_w is not None and self.regen_power_w < 0.0:
            raise ValueError("regen_power_w is a nonnegative recovered-power magnitude")
        if self.gear is not None and self.gear < 1:
            raise ValueError("gear must be positive when supplied")
        if any(later.sequence <= earlier.sequence for earlier, later in zip(self.events, self.events[1:])):
            raise ValueError("telemetry event sequences must be strictly increasing")


@dataclass(frozen=True, slots=True)
class EvidenceReadinessReport:
    capability_label: str
    physics_validated: bool
    faithful_claim_allowed: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    high_sensitivity_unbounded: tuple[str, ...]
    high_sensitivity_inferred: tuple[str, ...]


class EvidenceReadinessEvaluator:
    """Protocol-owned, fail-closed gate for the literal faithful label.

    Readiness never trusts caller-controlled ``high_sensitivity`` flags or a
    track's declarative ``certification_eligible`` boolean.  Exact parameter
    names, evidence classes, survey tolerances and validation artifacts are
    fixed by this protocol implementation.
    """

    REQUIRED_PARAMETERS = (
        ("geometry", frozenset({
            "body_length_m", "body_width_m", "body_height_m", "wheelbase_m",
            "front_track_m", "rear_track_m", "effective_wheel_radius_m",
        })),
        ("mass_properties", frozenset({
            "vehicle_mass_kg", "driver_ballast_mass_kg", "reference_fuel_mass_kg",
            "cg_x_from_front_axle_m", "cg_height_m", "yaw_inertia_kg_m2",
        })),
        ("suspension", frozenset({
            "front_heave_rate_n_m", "rear_heave_rate_n_m", "front_damping_n_s_m",
            "rear_damping_n_s_m", "pitch_link_gain",
        })),
        ("tyres", frozenset({
            "front_tyre_size_mm", "rear_tyre_size_mm", "tyre_peak_mu",
            "tyre_relaxation_length_m", "tyre_optimal_temperature_k",
        })),
        ("aero", frozenset({
            "evo_downforce_relative_to_wec", "evo_aero_efficiency_relative_to_wec",
            "closed_aero_cda_m2", "closed_aero_cla_m2", "aero_balance_front",
        })),
        ("ice", frozenset({
            "ice_max_power_w", "ice_redline_rpm", "ice_peak_torque_nm",
            "ice_thermal_efficiency", "fuel_lower_heating_value_j_kg",
        })),
        ("mgu", frozenset({
            "front_mgu_max_power_w", "front_mgu_peak_torque_nm",
            "front_mgu_drive_efficiency", "front_mgu_regen_efficiency",
            "front_mgu_regen_power_limit_w",
        })),
        ("battery", frozenset({
            "battery_nominal_voltage_v", "battery_usable_energy_j",
            "battery_max_discharge_power_w", "battery_max_charge_power_w",
            "battery_min_soc", "battery_max_soc",
        })),
        ("gearbox", frozenset({
            "forward_gear_count", "gear_ratios", "final_drive_ratio", "driveline_efficiency",
        })),
        ("brakes", frozenset({
            "maximum_service_brake_force_n", "brake_bias_front", "brake_optimal_temperature_k",
        })),
        ("embedded_controllers", frozenset({
            "traction_control_slip_target", "brake_by_wire_regen_blend",
            "active_aero_response_time_s",
        })),
        ("thermal_limits", frozenset({
            "battery_derate_temperature_k", "coolant_derate_temperature_k",
            "brake_derate_temperature_k",
        })),
    )
    PUBLIC_OFFICIAL_ANCHORS = frozenset({
        "body_length_m", "body_width_m", "body_height_m", "vehicle_mass_kg",
        "driver_ballast_mass_kg", "front_tyre_size_mm", "rear_tyre_size_mm",
        "evo_downforce_relative_to_wec", "evo_aero_efficiency_relative_to_wec",
        "ice_max_power_w", "ice_redline_rpm", "front_mgu_max_power_w",
        "battery_nominal_voltage_v", "forward_gear_count",
    })
    PROTOCOL_HIGH_SENSITIVITY = frozenset({
        "wheelbase_m", "front_track_m", "rear_track_m", "effective_wheel_radius_m",
        "reference_fuel_mass_kg", "cg_x_from_front_axle_m", "cg_height_m",
        "yaw_inertia_kg_m2", "front_heave_rate_n_m", "rear_heave_rate_n_m",
        "front_damping_n_s_m", "rear_damping_n_s_m", "pitch_link_gain",
        "tyre_peak_mu", "tyre_relaxation_length_m", "tyre_optimal_temperature_k",
        "closed_aero_cda_m2", "closed_aero_cla_m2", "aero_balance_front",
        "ice_peak_torque_nm", "ice_thermal_efficiency", "fuel_lower_heating_value_j_kg",
        "front_mgu_peak_torque_nm", "front_mgu_drive_efficiency",
        "front_mgu_regen_efficiency", "front_mgu_regen_power_limit_w",
        "battery_usable_energy_j", "battery_max_discharge_power_w",
        "battery_max_charge_power_w", "battery_min_soc", "battery_max_soc",
        "gear_ratios", "final_drive_ratio", "driveline_efficiency",
        "maximum_service_brake_force_n", "brake_bias_front", "brake_optimal_temperature_k",
        "traction_control_slip_target", "brake_by_wire_regen_blend",
        "active_aero_response_time_s", "battery_derate_temperature_k",
        "coolant_derate_temperature_k", "brake_derate_temperature_k",
    })
    MEASUREMENT_REQUIRED_GROUPS = frozenset({
        "mass_properties", "suspension", "tyres", "aero", "ice", "mgu", "battery",
        "gearbox", "brakes", "embedded_controllers", "thermal_limits",
    })
    # Signed package/freeze verification exists in supra.faithful.evidence, but
    # a freeze is not yet transformed into and bound against these physical
    # schemas. Caller-authored hashes or pass booleans remain insufficient.
    SIGNED_EVIDENCE_FREEZE_BINDING_IMPLEMENTED = False
    REQUIRED_VEHICLE_ID = "porsche-919-hybrid-evo-2018-faithful-v2"
    REQUIRED_TIMING_RULE = (
        "continuous T13-to-T13 flying lap with interpolated timing-plane crossing"
    )
    REQUIRED_HOLDOUT_POLICY = (
        "Calibrate with component and Spa data; freeze every parameter/hash before opening "
        "Nordschleife record telemetry."
    )
    SCENARIO_PARAMETERS = (
        ("air_temperature", "air_temperature_k", "K"),
        ("track_temperature", "track_temperature_k", "K"),
        ("air_pressure", "air_pressure_pa", "Pa"),
        ("relative_humidity", "relative_humidity", "1"),
        ("wind_speed", "wind_speed_mps", "m/s"),
        ("wind_direction", "wind_direction_rad", "rad"),
        ("initial_fuel_mass", "initial_fuel_mass_kg", "kg"),
        ("initial_battery_soc", "initial_battery_soc", "1"),
        ("initial_tyre_temperature", "initial_tyre_temperature_k", "K"),
        ("initial_brake_temperature", "initial_brake_temperature_k", "K"),
    )
    PARAMETER_SEMANTICS = {
        "body_length_m": ("m", 1), "body_width_m": ("m", 1),
        "body_height_m": ("m", 1), "wheelbase_m": ("m", 1),
        "front_track_m": ("m", 1), "rear_track_m": ("m", 1),
        "effective_wheel_radius_m": ("m", 1),
        "vehicle_mass_kg": ("kg", 1), "driver_ballast_mass_kg": ("kg", 1),
        "reference_fuel_mass_kg": ("kg", 1),
        "cg_x_from_front_axle_m": ("m", 1), "cg_height_m": ("m", 1),
        "yaw_inertia_kg_m2": ("kg*m^2", 1),
        "front_heave_rate_n_m": ("N/m", 1), "rear_heave_rate_n_m": ("N/m", 1),
        "front_damping_n_s_m": ("N*s/m", 1), "rear_damping_n_s_m": ("N*s/m", 1),
        "pitch_link_gain": ("1", 1),
        "front_tyre_size_mm": ("mm,mm,in", 3), "rear_tyre_size_mm": ("mm,mm,in", 3),
        "tyre_peak_mu": ("1", 1), "tyre_relaxation_length_m": ("m", 1),
        "tyre_optimal_temperature_k": ("K", 1),
        "evo_downforce_relative_to_wec": ("1", 1),
        "evo_aero_efficiency_relative_to_wec": ("1", 1),
        "closed_aero_cda_m2": ("m^2", 1), "closed_aero_cla_m2": ("m^2", 1),
        "aero_balance_front": ("1", 1),
        "ice_max_power_w": ("W", 1), "ice_redline_rpm": ("rpm", 1),
        "ice_peak_torque_nm": ("N*m", 1), "ice_thermal_efficiency": ("1", 1),
        "fuel_lower_heating_value_j_kg": ("J/kg", 1),
        "front_mgu_max_power_w": ("W", 1), "front_mgu_peak_torque_nm": ("N*m", 1),
        "front_mgu_drive_efficiency": ("1", 1),
        "front_mgu_regen_efficiency": ("1", 1),
        "front_mgu_regen_power_limit_w": ("W", 1),
        "battery_nominal_voltage_v": ("V", 1), "battery_usable_energy_j": ("J", 1),
        "battery_max_discharge_power_w": ("W", 1),
        "battery_max_charge_power_w": ("W", 1),
        "battery_min_soc": ("1", 1), "battery_max_soc": ("1", 1),
        "forward_gear_count": ("count", 1), "gear_ratios": ("1", 7),
        "final_drive_ratio": ("1", 1), "driveline_efficiency": ("1", 1),
        "maximum_service_brake_force_n": ("N", 1), "brake_bias_front": ("1", 1),
        "brake_optimal_temperature_k": ("K", 1),
        "traction_control_slip_target": ("1", 1),
        "brake_by_wire_regen_blend": ("1", 1),
        "active_aero_response_time_s": ("s", 1),
        "battery_derate_temperature_k": ("K", 1),
        "coolant_derate_temperature_k": ("K", 1),
        "brake_derate_temperature_k": ("K", 1),
    }
    OFFICIAL_ANCHOR_VALUES = {
        "body_length_m": 5.078, "body_width_m": 1.900,
        "body_height_m": 1.050, "vehicle_mass_kg": 849.0,
        "driver_ballast_mass_kg": 39.0,
        "front_tyre_size_mm": (310.0, 710.0, 18.0),
        "rear_tyre_size_mm": (310.0, 710.0, 18.0),
        "evo_downforce_relative_to_wec": 1.53,
        "evo_aero_efficiency_relative_to_wec": 1.66,
        "ice_max_power_w": 720.0 * 735.49875, "ice_redline_rpm": 9000.0,
        "front_mgu_max_power_w": 440.0 * 735.49875,
        "battery_nominal_voltage_v": 800.0, "forward_gear_count": 7.0,
    }
    VALIDATION_METRICS = (
        (ValidationArtifactKind.COMPONENT, (
            ("all_within_measurement_uncertainty", "min", 1.0),
        )),
        (ValidationArtifactKind.SPA, (
            ("lap_time_error_fraction", "max", 0.005),
            ("speed_error_fraction", "max", 0.01),
            ("lateral_accel_error_g", "max", 0.25),
        )),
        (ValidationArtifactKind.NORD_HOLDOUT, (
            ("lap_time_error_fraction", "max", 0.005),
            ("speed_trace_nrmse", "max", 0.05),
            ("peak_speed_error_fraction", "max", 0.01),
            ("average_speed_error_fraction", "max", 0.01),
        )),
        (ValidationArtifactKind.NUMERICS, (
            ("half_ms_lap_delta_s", "max", 0.1),
            ("mjx_force_error_fraction", "max", 0.01),
            ("mjx_speed_error_mps", "max", 0.5),
            ("mjx_position_error_m", "max", 0.5),
        )),
        (ValidationArtifactKind.ENERGY, (
            ("full_lap_residual_fraction", "max", 0.001),
        )),
    )

    @classmethod
    def _validate_artifact(cls, artifact: ValidationArtifactV1, blockers: list[str]) -> None:
        label = artifact.kind.value
        if not artifact.passed:
            blockers.append(f"{label} validation artifact did not pass")
        if not _materialized_artifact_matches(artifact.artifact_uri, artifact.artifact_sha256):
            blockers.append(f"{label} validation artifact is not materialized with its declared hash")
        if not artifact.source_evidence.verified_content:
            blockers.append(f"{label} validation source content is not captured and hashed")
        if artifact.source_evidence.confidence not in {
            Confidence.LICENSED_MEASURED, Confidence.CALIBRATED
        }:
            blockers.append(f"{label} validation lacks licensed/calibrated evidence")
        if not artifact.source_evidence.calibration_materialized:
            blockers.append(f"{label} validation calibration dataset is not materialized")
        requirements = dict(cls.VALIDATION_METRICS)[artifact.kind]
        actual = artifact.metric_map()
        for name, comparison, threshold in requirements:
            if name not in actual:
                blockers.append(f"{label} validation missing metric: {name}")
                continue
            value = actual[name]
            if comparison == "max" and value > threshold:
                blockers.append(f"{label} metric {name} exceeds {threshold}")
            if comparison == "min" and value < threshold:
                blockers.append(f"{label} metric {name} is below {threshold}")

    @classmethod
    def evaluate(
        cls,
        vehicle: VehicleSpecV2,
        track: TrackSurfaceV2,
        scenario: RecordScenarioV1,
    ) -> EvidenceReadinessReport:
        blockers: list[str] = []
        warnings: list[str] = []
        unbounded: list[str] = []
        inferred: list[str] = []

        if not cls.SIGNED_EVIDENCE_FREEZE_BINDING_IMPLEMENTED:
            blockers.append(
                "verified signed evidence freeze is not bound to the physical schemas"
            )
        if vehicle.vehicle_id != cls.REQUIRED_VEHICLE_ID:
            blockers.append(f"vehicle_id must be {cls.REQUIRED_VEHICLE_ID}")

        if scenario.vehicle_spec_sha256 != vehicle.sha256:
            blockers.append("scenario vehicle hash does not match VehicleSpecV2")
        if scenario.track_surface_sha256 != track.sha256:
            blockers.append("scenario track hash does not match TrackSurfaceV2")
        if not math.isclose(
            scenario.target_lap_time_s, 319.546, rel_tol=0.0, abs_tol=1e-12
        ):
            blockers.append("scenario benchmark must be exactly 319.546 seconds")
        if scenario.timing_rule != cls.REQUIRED_TIMING_RULE:
            blockers.append("scenario timing rule does not match the frozen T13 contract")
        if scenario.holdout_policy != cls.REQUIRED_HOLDOUT_POLICY:
            blockers.append("scenario holdout policy does not match the frozen protocol")
        for field_name, parameter_name, unit in cls.SCENARIO_PARAMETERS:
            parameter = getattr(scenario, field_name)
            if parameter.name != parameter_name or parameter.unit != unit:
                blockers.append(f"scenario parameter semantics mismatch: {field_name}")
            if not parameter.distribution.bounded:
                blockers.append(f"scenario parameter is unbounded: {parameter_name}")
            if parameter.evidence.confidence not in {
                Confidence.LICENSED_MEASURED, Confidence.CALIBRATED
            }:
                blockers.append(
                    f"scenario parameter requires measured/calibrated evidence: {parameter_name}"
                )
            if not parameter.evidence.verified_content:
                blockers.append(
                    f"scenario parameter evidence is not materialized: {parameter_name}"
                )
            if not parameter.evidence.calibration_materialized:
                blockers.append(
                    f"scenario calibration dataset is not materialized: {parameter_name}"
                )
        if not track.certification_eligible:
            blockers.append(f"track is training-only: {track.approximation_reason}")
        else:
            if not track.survey_evidence.verified_content:
                blockers.append("track survey source content is not captured and hashed")
            survey = track.survey_validation
            if survey is None:
                blockers.append("track survey validation artifact is absent")
            else:
                if not survey.passed:
                    blockers.append("track survey validation artifact did not pass")
                if not _materialized_artifact_matches(survey.artifact_uri, survey.artifact_sha256):
                    blockers.append("track survey validation artifact is not materialized with its declared hash")
                if not survey.evidence.verified_content:
                    blockers.append("track survey validation evidence is not content-hashed")
                if survey.evidence.confidence not in {
                    Confidence.LICENSED_MEASURED, Confidence.CALIBRATED
                }:
                    blockers.append("track survey validation lacks licensed/calibrated evidence")
                if not survey.evidence.calibration_materialized:
                    blockers.append("track survey calibration dataset is not materialized")
                limits = (
                    ("lap length", survey.lap_length_error_m, 0.1),
                    ("boundary", survey.max_boundary_error_m, 0.05),
                    ("vertical surface", survey.max_vertical_error_m, 0.01),
                    ("bank", survey.max_bank_error_deg, 0.1),
                )
                for label, value, limit in limits:
                    if value > limit:
                        blockers.append(f"track {label} validation exceeds {limit}")
            # The current schema carries the certification surface inline.
            # Require at least 1 m station resolution and full-lap coverage;
            # a two-point declaration cannot become a survey by setting a flag.
            lap_length = track.lap_length.scalar()
            if len(track.samples) < math.ceil(lap_length):
                blockers.append("track survey sample density is coarser than 1 m")
            elif track.samples:
                stations = (0.0,) + tuple(sample.station_m for sample in track.samples) + (lap_length,)
                if max(b - a for a, b in zip(stations, stations[1:])) > 1.0 + 1e-9:
                    blockers.append("track survey contains a station gap greater than 1 m")
            if len(track.timing_planes) != 1 or track.timing_planes[0].name != "T13-start-finish":
                blockers.append("track must contain exactly the frozen T13 start/finish plane")

        grouped = {name: {parameter.name: parameter for parameter in values} for name, values in vehicle.groups()}
        protocol_parameter_names = set()
        for group_name, required_names in cls.REQUIRED_PARAMETERS:
            available = grouped.get(group_name, {})
            protocol_parameter_names.update(required_names)
            for name in sorted(required_names - set(available)):
                blockers.append(f"missing protocol parameter {group_name}.{name}")

        for group, parameter in vehicle.iter_parameters():
            semantics = cls.PARAMETER_SEMANTICS.get(parameter.name)
            if semantics is not None:
                expected_unit, expected_arity = semantics
                actual_arity = len(parameter.value) if isinstance(parameter.value, tuple) else 1
                if parameter.unit != expected_unit or actual_arity != expected_arity:
                    blockers.append(f"parameter semantics mismatch: {parameter.name}")
            if parameter.name in cls.OFFICIAL_ANCHOR_VALUES:
                expected = cls.OFFICIAL_ANCHOR_VALUES[parameter.name]
                actual_values = parameter.value if isinstance(parameter.value, tuple) else (parameter.value,)
                expected_values = expected if isinstance(expected, tuple) else (expected,)
                if len(actual_values) != len(expected_values) or any(
                    not math.isclose(
                        float(actual), float(required), rel_tol=0.0, abs_tol=1e-9
                    )
                    for actual, required in zip(actual_values, expected_values)
                ):
                    blockers.append(f"official anchor drift: {parameter.name}")
            sensitivity_owned_by_protocol = (
                parameter.name in cls.PROTOCOL_HIGH_SENSITIVITY or parameter.high_sensitivity
            )
            if sensitivity_owned_by_protocol and not parameter.distribution.bounded:
                unbounded.append(parameter.name)
            if sensitivity_owned_by_protocol and parameter.evidence.confidence is Confidence.INFERRED:
                inferred.append(parameter.name)
            if (
                group in cls.MEASUREMENT_REQUIRED_GROUPS
                and parameter.name in cls.PROTOCOL_HIGH_SENSITIVITY
                and parameter.evidence.confidence not in {
                    Confidence.LICENSED_MEASURED, Confidence.CALIBRATED
                }
            ):
                blockers.append(
                    f"{parameter.name} requires licensed measurement or calibrated evidence"
                )
            if (
                group in cls.MEASUREMENT_REQUIRED_GROUPS
                and parameter.name in cls.PROTOCOL_HIGH_SENSITIVITY
                and not parameter.evidence.calibration_materialized
            ):
                blockers.append(
                    f"{parameter.name} calibration dataset is not materialized"
                )
            if parameter.name in protocol_parameter_names and not parameter.evidence.verified_content:
                blockers.append(f"{parameter.name} evidence content is not captured and hashed")

        blockers.extend(f"unbounded high-sensitivity parameter: {name}" for name in unbounded)
        blockers.extend(f"inferred high-sensitivity parameter: {name}" for name in inferred)
        validation = scenario.physics_validation
        if validation is None:
            blockers.append(
                "physics validation bundle absent: component, Spa, holdout, numerics and energy are required"
            )
            if scenario.record_telemetry_opened:
                blockers.append("Nordschleife holdout telemetry was opened without a frozen-parameter proof")
        else:
            if not scenario.record_telemetry_opened:
                blockers.append(
                    "Nordschleife holdout validation exists but record telemetry is not marked opened"
                )
            if not validation.parameters_frozen_before_holdout:
                blockers.append("parameters were not provably frozen before opening Nordschleife holdout")
            for artifact in (
                validation.component, validation.spa, validation.nord_holdout,
                validation.numerics, validation.energy,
            ):
                cls._validate_artifact(artifact, blockers)
        if scenario.controls.boost_request:
            blockers.append(
                "driver boost request is enabled without a trusted licensed interface-evidence role"
            )
        if scenario.controls.drs_request:
            blockers.append(
                "driver DRS request is enabled without a trusted licensed interface-evidence role"
            )

        blockers = sorted(set(blockers))
        if blockers:
            label = "telemetry-constrained approximation"
        else:
            label = "physics-validated faithful-v2"
        return EvidenceReadinessReport(
            capability_label=label,
            physics_validated=not blockers,
            faithful_claim_allowed=not blockers,
            blockers=tuple(blockers),
            warnings=tuple(sorted(set(warnings))),
            high_sensitivity_unbounded=tuple(sorted(set(unbounded))),
            high_sensitivity_inferred=tuple(sorted(set(inferred))),
        )


@dataclass(frozen=True, slots=True)
class BackendCapability:
    backend_name: str
    available: bool
    runnable: bool
    precision: str
    timestep_s: float
    vectorized: bool
    differentiable: bool
    certification_authority: bool
    faithful_claim_allowed: bool
    capability_label: str
    blockers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "available", "runnable", "vectorized", "differentiable",
            "certification_authority", "faithful_claim_allowed",
        ):
            _actual_bool(getattr(self, name), name)
        if self.precision not in {"float64", "float32"}:
            raise ValueError("backend precision must be float64 or float32")
        _finite(self.timestep_s, "timestep_s")
        if self.timestep_s <= 0.0:
            raise ValueError("backend timestep must be positive")
        if not self.backend_name or not self.capability_label:
            raise ValueError("backend_name and capability_label are required")
        object.__setattr__(self, "blockers", tuple(self.blockers))


@runtime_checkable
class PhysicsBackend(Protocol):
    @property
    def identity(self) -> PhysicsIdentity: ...

    @property
    def capability(self) -> BackendCapability: ...

    def reset(self, *, speed_mps: float = 0.0) -> VehicleTelemetryV2: ...

    def step(self, action: DriverActionV2) -> VehicleTelemetryV2: ...

    def snapshot(self) -> Any: ...


__all__ = [
    "ActiveAeroMode", "BackendCapability", "Confidence", "DistributionKind",
    "DriverActionV2", "DriverControllerStatusV1", "DriverObservationV2",
    "EvidenceReadinessEvaluator", "EvidenceReadinessReport",
    "EvidenceRef", "MapPreviewPointV1", "NumericValue", "PermittedDriverControlsV1",
    "PhysicalParameter", "PhysicsBackend", "PhysicsIdentity", "PhysicsValidationEvidenceV1",
    "RecordScenarioV1", "ShiftRequest", "TelemetryEvent", "TimingPlaneV1",
    "TrackSampleV2", "TrackSurfaceV2", "TrackSurveyValidationV1",
    "UncertaintyDistribution", "ValidationArtifactKind", "ValidationArtifactV1",
    "ValidationMetricV1", "Vec3", "VehicleSpecV2", "VehicleTelemetryV2",
    "canonical_json", "content_sha256",
]
