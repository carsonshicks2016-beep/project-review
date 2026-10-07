"""Versioned vehicle assets and telemetry-faithful A/V projections.

This module is intentionally separate from DSP and rendering.  It translates
authoritative ``VehicleTelemetryV2`` samples into explicit source states.  A
missing physical signal stays missing: throttle and engine speed are never used
to invent turbo boost, electrical deployment, regeneration, or aero motion.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

from .schemas import VehicleTelemetryV2


ASSET_MANIFEST_SCHEMA = "supra-car-asset-manifest-v1"
AV_PROJECTION_SCHEMA = "faithful-av-projection-v1"
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSET_ROOT = _PROJECT_ROOT / "assets" / "vehicles"

Vec3 = tuple[float, float, float]
Quat4 = tuple[float, float, float, float]
Quad = tuple[float, float, float, float]
BoolQuad = tuple[bool, bool, bool, bool]


class AssetManifestError(ValueError):
    """The asset manifest is malformed or internally inconsistent."""


class AssetIntegrityError(RuntimeError):
    """An on-disk asset differs from its immutable manifest."""


@dataclass(frozen=True, slots=True)
class CanonicalDimensions:
    length_m: float
    width_m: float
    height_m: float
    allowed_relative_error: float

    @property
    def extents_m(self) -> Vec3:
        return self.length_m, self.width_m, self.height_m


@dataclass(frozen=True, slots=True)
class CarAsset:
    asset_id: str
    vehicle_id: str
    status: str
    faithful_geometry_eligible: bool
    manifest_path: Path
    glb_path: Path
    generator_path: Path
    glb_sha256: str
    generator_sha256: str
    glb_bytes: int
    dimensions: CanonicalDimensions
    canonical_nodes: Mapping[str, tuple[str, ...]]
    articulation_points_m: Mapping[str, Vec3]
    audio_source_points_m: Mapping[str, Vec3]


@dataclass(frozen=True, slots=True)
class AssetVerification:
    asset_id: str
    glb_sha256: str
    generator_sha256: str
    extents_m: Vec3 | None
    geometry_count: int | None
    node_count: int | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_positive(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise AssetManifestError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise AssetManifestError(f"{label} must be finite and positive")
    return number


def _vec3(value: object, label: str) -> Vec3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise AssetManifestError(f"{label} must contain exactly three coordinates")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise AssetManifestError(f"{label} contains a non-finite coordinate")
    return result  # type: ignore[return-value]


def _member_path(package_root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise AssetManifestError(f"{label} must be a non-empty relative path")
    candidate = (package_root / relative).resolve()
    try:
        candidate.relative_to(package_root.resolve())
    except ValueError as exc:
        raise AssetManifestError(f"{label} escapes its asset package") from exc
    return candidate


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise AssetManifestError(f"{label} must be an object")
    return value


class CarAssetRegistry:
    """Discovers immutable per-vehicle assets without viewer-specific policy."""

    def __init__(self, asset_root: Path | str = DEFAULT_ASSET_ROOT):
        self.asset_root = Path(asset_root).resolve()
        self._by_vehicle: dict[str, CarAsset] = {}
        self._by_asset: dict[str, CarAsset] = {}

    @classmethod
    def discover(cls, asset_root: Path | str = DEFAULT_ASSET_ROOT) -> "CarAssetRegistry":
        registry = cls(asset_root)
        if not registry.asset_root.is_dir():
            return registry
        for manifest_path in sorted(registry.asset_root.glob("*/asset_manifest.json")):
            registry.register_manifest(manifest_path)
        return registry

    def register_manifest(self, manifest_path: Path | str) -> CarAsset:
        path = Path(manifest_path).resolve()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AssetManifestError(f"cannot read {path}: {exc}") from exc
        root = path.parent
        if raw.get("schema") != ASSET_MANIFEST_SCHEMA:
            raise AssetManifestError(f"{path}: unsupported schema {raw.get('schema')!r}")

        asset_id = str(raw.get("asset_id", "")).strip()
        vehicle_id = str(raw.get("vehicle_id", "")).strip()
        if not asset_id or not vehicle_id:
            raise AssetManifestError(f"{path}: asset_id and vehicle_id are required")
        if asset_id in self._by_asset:
            raise AssetManifestError(f"duplicate asset_id: {asset_id}")
        if vehicle_id in self._by_vehicle:
            raise AssetManifestError(f"duplicate vehicle_id: {vehicle_id}")

        files = _mapping(raw.get("files"), "files")
        glb = _mapping(files.get("glb"), "files.glb")
        generator = _mapping(files.get("generator"), "files.generator")
        dimensions = _mapping(raw.get("canonical_dimensions_m"), "canonical_dimensions_m")
        canonical_nodes_raw = _mapping(raw.get("canonical_nodes"), "canonical_nodes")
        articulation_raw = _mapping(raw.get("articulation_points_m"), "articulation_points_m")
        sources_raw = _mapping(raw.get("audio_source_points_m"), "audio_source_points_m")

        glb_hash = str(glb.get("sha256", "")).lower()
        generator_hash = str(generator.get("sha256", "")).lower()
        if len(glb_hash) != 64 or len(generator_hash) != 64:
            raise AssetManifestError("asset hashes must be 64-character SHA-256 hex digests")
        try:
            int(glb_hash, 16)
            int(generator_hash, 16)
        except ValueError as exc:
            raise AssetManifestError("asset hashes must be hexadecimal") from exc

        nodes: dict[str, tuple[str, ...]] = {}
        for group, names in canonical_nodes_raw.items():
            if not isinstance(names, Sequence) or isinstance(names, (str, bytes)):
                raise AssetManifestError(f"canonical_nodes.{group} must be an array")
            normalized = tuple(str(name).strip() for name in names)
            if not normalized or any(not name for name in normalized):
                raise AssetManifestError(f"canonical_nodes.{group} contains an empty node name")
            nodes[str(group)] = normalized

        geometry_eligible = raw.get("faithful_geometry_eligible", False)
        if type(geometry_eligible) is not bool:
            raise AssetManifestError("faithful_geometry_eligible must be a JSON boolean")

        asset = CarAsset(
            asset_id=asset_id,
            vehicle_id=vehicle_id,
            status=str(raw.get("status", "unknown")),
            faithful_geometry_eligible=geometry_eligible,
            manifest_path=path,
            glb_path=_member_path(root, glb.get("path"), "files.glb.path"),
            generator_path=_member_path(root, generator.get("path"), "files.generator.path"),
            glb_sha256=glb_hash,
            generator_sha256=generator_hash,
            glb_bytes=int(_finite_positive(glb.get("bytes"), "files.glb.bytes")),
            dimensions=CanonicalDimensions(
                length_m=_finite_positive(dimensions.get("length"), "canonical_dimensions_m.length"),
                width_m=_finite_positive(dimensions.get("width"), "canonical_dimensions_m.width"),
                height_m=_finite_positive(dimensions.get("height"), "canonical_dimensions_m.height"),
                allowed_relative_error=_finite_positive(
                    dimensions.get("allowed_relative_error"),
                    "canonical_dimensions_m.allowed_relative_error",
                ),
            ),
            canonical_nodes=MappingProxyType(nodes),
            articulation_points_m=MappingProxyType({
                str(name): _vec3(point, f"articulation_points_m.{name}")
                for name, point in articulation_raw.items()
            }),
            audio_source_points_m=MappingProxyType({
                str(name): _vec3(point, f"audio_source_points_m.{name}")
                for name, point in sources_raw.items()
            }),
        )
        self._by_vehicle[vehicle_id] = asset
        self._by_asset[asset_id] = asset
        return asset

    def for_vehicle(self, vehicle_id: str) -> CarAsset:
        try:
            return self._by_vehicle[vehicle_id]
        except KeyError as exc:
            raise KeyError(f"no registered car asset for {vehicle_id!r}") from exc

    def by_asset_id(self, asset_id: str) -> CarAsset:
        try:
            return self._by_asset[asset_id]
        except KeyError as exc:
            raise KeyError(f"unknown car asset {asset_id!r}") from exc

    def vehicles(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_vehicle))

    def verify(self, asset: CarAsset | str, *, inspect_geometry: bool = False) -> AssetVerification:
        selected = self.by_asset_id(asset) if isinstance(asset, str) else asset
        for member in (selected.glb_path, selected.generator_path):
            if not member.is_file():
                raise AssetIntegrityError(f"missing asset member: {member}")
        if selected.glb_path.stat().st_size != selected.glb_bytes:
            raise AssetIntegrityError(
                f"{selected.glb_path}: expected {selected.glb_bytes} bytes, got {selected.glb_path.stat().st_size}"
            )
        glb_hash = _sha256(selected.glb_path)
        generator_hash = _sha256(selected.generator_path)
        if glb_hash != selected.glb_sha256:
            raise AssetIntegrityError(f"GLB hash mismatch: expected {selected.glb_sha256}, got {glb_hash}")
        if generator_hash != selected.generator_sha256:
            raise AssetIntegrityError(
                f"generator hash mismatch: expected {selected.generator_sha256}, got {generator_hash}"
            )

        extents: Vec3 | None = None
        geometry_count: int | None = None
        node_count: int | None = None
        if inspect_geometry:
            try:
                import trimesh
            except ImportError as exc:  # pragma: no cover - optional deployment dependency
                raise AssetIntegrityError("geometry inspection requires trimesh") from exc
            scene = trimesh.load_scene(selected.glb_path)
            observed = tuple(float(item) for item in scene.extents)
            expected = selected.dimensions.extents_m
            for axis, actual, target in zip("xyz", observed, expected):
                error = abs(actual - target) / target
                if error > selected.dimensions.allowed_relative_error:
                    raise AssetIntegrityError(
                        f"{axis}-extent {actual:.9f} m differs from {target:.9f} m by {error:.4%}"
                    )
            present_nodes = set(scene.graph.nodes_geometry)
            required_nodes = {name for names in selected.canonical_nodes.values() for name in names}
            missing_nodes = sorted(required_nodes - present_nodes)
            if missing_nodes:
                raise AssetIntegrityError(f"GLB is missing canonical nodes: {', '.join(missing_nodes)}")
            for node_name, expected_pivot in selected.articulation_points_m.items():
                if node_name not in present_nodes:
                    raise AssetIntegrityError(f"GLB is missing articulation node: {node_name}")
                transform, _ = scene.graph.get(node_name)
                actual_pivot = tuple(float(transform[index][3]) for index in range(3))
                if any(
                    abs(actual - expected) > 1e-6
                    for actual, expected in zip(actual_pivot, expected_pivot)
                ):
                    raise AssetIntegrityError(
                        f"{node_name} pivot {actual_pivot} does not match manifest {expected_pivot}"
                    )
            extents = observed  # type: ignore[assignment]
            geometry_count = len(scene.geometry)
            node_count = len(present_nodes)

        return AssetVerification(
            asset_id=selected.asset_id,
            glb_sha256=glb_hash,
            generator_sha256=generator_hash,
            extents_m=extents,
            geometry_count=geometry_count,
            node_count=node_count,
        )


# The projection dataclasses and adapter below contain no DSP/render policy.
# None means "not measured in this telemetry sample", never a neutral estimate.


@dataclass(frozen=True, slots=True)
class WheelVisualState:
    angular_speed_rad_s: float | None
    suspension_travel_m: float | None
    brake_temperature_k: float | None
    in_contact: bool | None


@dataclass(frozen=True, slots=True)
class VisualStateV1:
    position_m: Vec3
    orientation_rpy_rad: Vec3
    orientation_xyzw: Quat4
    wheels: tuple[WheelVisualState, WheelVisualState, WheelVisualState, WheelVisualState]
    front_aero_position: float | None
    rear_aero_position: float | None
    headlight_on: bool | None
    rain_light_on: bool | None


@dataclass(frozen=True, slots=True)
class IceAudioState:
    crank_rpm: float
    crank_torque_nm: float
    crank_power_w: float | None
    accelerator_pedal: float | None


@dataclass(frozen=True, slots=True)
class TurboAudioState:
    shaft_speed_rad_s: float
    boost_pa: float


@dataclass(frozen=True, slots=True)
class MguAudioState:
    signed_torque_nm: float
    signed_mechanical_power_w: float
    regen_power_w: float
    battery_power_w: float | None


@dataclass(frozen=True, slots=True)
class GearboxAudioState:
    input_shaft_speed_rad_s: float
    output_shaft_speed_rad_s: float
    gear: int
    shift_state: str | None


@dataclass(frozen=True, slots=True)
class WheelAudioState:
    slip_ratio: float
    slip_angle_rad: float
    force_xyz_n: Vec3
    in_contact: bool
    brake_temperature_k: float | None


@dataclass(frozen=True, slots=True)
class AudioStateV1:
    position_m: Vec3
    linear_velocity_world_mps: Vec3
    ice: IceAudioState | None
    turbo: TurboAudioState | None
    front_mgu: MguAudioState | None
    gearbox: GearboxAudioState | None
    wheels: tuple[WheelAudioState | None, WheelAudioState | None, WheelAudioState | None, WheelAudioState | None]
    source_points_m: Mapping[str, Vec3]


@dataclass(frozen=True, slots=True)
class AudioVisualProjectionV1:
    schema: str
    sample_time_s: float
    vehicle_id: str
    asset_id: str
    visual: VisualStateV1
    audio: AudioStateV1
    unavailable_channels: tuple[str, ...]


def _present_float(value: object) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("telemetry contains a non-finite scalar")
    return result


def _optional_quad(value: object, label: str) -> tuple[float | None, float | None, float | None, float | None]:
    if value is None:
        return None, None, None, None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{label} must contain four values")
    return tuple(_present_float(item) for item in value)  # type: ignore[return-value]


def _required_vec(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{label} must contain {length} values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} contains a non-finite value")
    return result


def _optional_bool(value: object) -> bool | None:
    return None if value is None else bool(value)


def _optional_bool_quad(value: object, label: str) -> tuple[bool | None, bool | None, bool | None, bool | None]:
    if value is None:
        return None, None, None, None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError(f"{label} must contain four values")
    return tuple(None if item is None else bool(item) for item in value)  # type: ignore[return-value]


def _rpy_to_xyzw(rpy: Vec3) -> Quat4:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    )


def _body_to_world(vector: Vec3, rpy: Vec3) -> Vec3:
    """Apply the exact intrinsic XYZ/body-to-world Rz*Ry*Rx rotation."""
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    x, y, z = vector
    return (
        (cy * cp) * x + (cy * sp * sr - sy * cr) * y + (cy * sp * cr + sy * sr) * z,
        (sy * cp) * x + (sy * sp * sr + cy * cr) * y + (sy * sp * cr - cy * sr) * z,
        (-sp) * x + (cp * sr) * y + (cp * cr) * z,
    )


class TruthfulAvAdapter:
    """Project authoritative telemetry without synthesizing missing physics.

    An audio source is enabled only when every physical signal required by that
    source is present.  This deliberately does not emulate the legacy behavior
    that inferred turbo or MGU activity from throttle and engine speed.
    """

    def __init__(self, registry: CarAssetRegistry):
        self.registry = registry

    def project(
        self,
        telemetry: VehicleTelemetryV2,
        *,
        vehicle_id: str | None = None,
    ) -> AudioVisualProjectionV1:
        resolved_vehicle = vehicle_id or ""
        if not resolved_vehicle:
            raise ValueError("vehicle_id is required because VehicleTelemetryV2 is identity-neutral")
        asset = self.registry.for_vehicle(resolved_vehicle)
        unavailable: list[str] = []

        position = _required_vec(telemetry.position_m, 3, "position_m")
        orientation_rpy = _required_vec(telemetry.orientation_rpy_rad, 3, "orientation_rpy_rad")
        velocity_body = _required_vec(telemetry.velocity_body_mps, 3, "velocity_body_mps")
        sample_time = _present_float(telemetry.time_s)
        if sample_time is None:
            raise ValueError("time_s is required")
        orientation_xyzw = _rpy_to_xyzw(orientation_rpy)  # type: ignore[arg-type]
        velocity_world = _body_to_world(velocity_body, orientation_rpy)  # type: ignore[arg-type]

        wheel_speed = _optional_quad(telemetry.wheel_speed_rad_s, "wheel_speed_rad_s")
        suspension = _optional_quad(telemetry.suspension_travel_m, "suspension_travel_m")
        brake_temperature = _optional_quad(telemetry.brake_temperature_k, "brake_temperature_k")
        contacts = _optional_bool_quad(telemetry.wheel_contact, "wheel_contact")
        visual_wheels = tuple(
            WheelVisualState(wheel_speed[i], suspension[i], brake_temperature[i], contacts[i])
            for i in range(4)
        )

        front_aero = _present_float(telemetry.front_aero_position)
        rear_aero = _present_float(telemetry.rear_aero_position)
        if front_aero is None:
            unavailable.append("visual.active_aero.front_aero_position")
        if rear_aero is None:
            unavailable.append("visual.active_aero.rear_aero_position")

        visual = VisualStateV1(
            position_m=position,  # type: ignore[arg-type]
            orientation_rpy_rad=orientation_rpy,  # type: ignore[arg-type]
            orientation_xyzw=orientation_xyzw,
            wheels=visual_wheels,  # type: ignore[arg-type]
            front_aero_position=front_aero,
            rear_aero_position=rear_aero,
            # Lighting is intentionally unavailable in VehicleTelemetryV2.
            headlight_on=None,
            rain_light_on=None,
        )
        unavailable.extend(("visual.lights.headlight_on", "visual.lights.rain_light_on"))

        ice_rpm = _present_float(telemetry.engine_rpm)
        ice_torque = _present_float(telemetry.engine_torque_nm)
        ice: IceAudioState | None = None
        if ice_rpm is not None and ice_torque is not None:
            ice = IceAudioState(
                crank_rpm=ice_rpm,
                crank_torque_nm=ice_torque,
                crank_power_w=_present_float(telemetry.engine_power_w),
                accelerator_pedal=_present_float(telemetry.accelerator_pedal),
            )
        else:
            unavailable.append("audio.rear_ice")

        turbo_speed = _present_float(telemetry.turbo_speed_rad_s)
        turbo_boost = _present_float(telemetry.turbo_boost_pa)
        turbo: TurboAudioState | None = None
        if turbo_speed is not None and turbo_boost is not None:
            turbo = TurboAudioState(turbo_speed, turbo_boost)
        else:
            unavailable.append("audio.turbo")

        mgu_torque = _present_float(telemetry.front_mgu_torque_nm)
        mgu_power = _present_float(telemetry.front_mgu_power_w)
        front_mgu: MguAudioState | None = None
        if mgu_torque is not None and mgu_power is not None:
            # Sign convention is part of VehicleTelemetryV2: negative
            # mechanical power is regenerative operation.  This is a direct
            # projection of measured power, not a pedal-based estimate.
            regen_power = max(0.0, -mgu_power)
            explicit_regen = _present_float(telemetry.regen_power_w)
            if explicit_regen is not None and not math.isclose(
                explicit_regen, regen_power, rel_tol=1e-6, abs_tol=1e-3
            ):
                raise ValueError(
                    "regen_power_w conflicts with signed front-MGU mechanical power"
                )
            front_mgu = MguAudioState(
                mgu_torque,
                mgu_power,
                regen_power,
                _present_float(telemetry.battery_power_w),
            )
        else:
            unavailable.append("audio.front_mgu")

        gearbox_input = _present_float(telemetry.gearbox_input_speed_rad_s)
        gearbox_output = _present_float(telemetry.gearbox_output_speed_rad_s)
        gear_raw = telemetry.gear
        gearbox: GearboxAudioState | None = None
        if gearbox_input is not None and gearbox_output is not None and gear_raw is not None:
            gearbox = GearboxAudioState(
                input_shaft_speed_rad_s=gearbox_input,
                output_shaft_speed_rad_s=gearbox_output,
                gear=int(gear_raw),
                shift_state=None if telemetry.shift_state is None else str(telemetry.shift_state),
            )
        else:
            unavailable.append("audio.gearbox")

        slip_ratio = _optional_quad(telemetry.wheel_slip_ratio, "wheel_slip_ratio")
        slip_angle = _optional_quad(telemetry.wheel_slip_angle_rad, "wheel_slip_angle_rad")
        wheel_audio: list[WheelAudioState | None] = []
        for index in range(4):
            force_raw = telemetry.wheel_force_xyz_n[index]
            force = None if force_raw is None else _required_vec(force_raw, 3, f"wheel_force_xyz_n[{index}]")
            required = (slip_ratio[index], slip_angle[index], force, contacts[index])
            if any(item is None for item in required):
                wheel_audio.append(None)
                unavailable.append(f"audio.wheel_{index}")
            else:
                wheel_audio.append(WheelAudioState(
                    slip_ratio=float(slip_ratio[index]),
                    slip_angle_rad=float(slip_angle[index]),
                    force_xyz_n=force,  # type: ignore[arg-type]
                    in_contact=bool(contacts[index]),
                    brake_temperature_k=brake_temperature[index],
                ))

        audio = AudioStateV1(
            position_m=position,  # type: ignore[arg-type]
            linear_velocity_world_mps=velocity_world,
            ice=ice,
            turbo=turbo,
            front_mgu=front_mgu,
            gearbox=gearbox,
            wheels=tuple(wheel_audio),  # type: ignore[arg-type]
            source_points_m=asset.audio_source_points_m,
        )
        return AudioVisualProjectionV1(
            schema=AV_PROJECTION_SCHEMA,
            sample_time_s=sample_time,
            vehicle_id=resolved_vehicle,
            asset_id=asset.asset_id,
            visual=visual,
            audio=audio,
            unavailable_channels=tuple(sorted(set(unavailable))),
        )
