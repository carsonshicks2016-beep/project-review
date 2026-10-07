"""Physics backend implementations for faithful-v2.

``MuJoCoAuthority`` is intentionally a *runnable public-anchor baseline*: it
uses MuJoCo's double-precision 1 ms integrator, separate ICE/MGU/battery energy
states, and the driver-control contract, but its planar force model is not the
licensed 6-DOF certification model described by the record program.  Its
capability report can therefore never claim faithful certification.

``MJXTrainingTwin`` exposes the future vectorized backend boundary and reports
dependency/readiness truth.  It fails closed on reset/step until a cross-checked
MJX dynamics adapter is installed; it never silently falls back to different
physics while calling itself MJX.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import importlib.metadata
import math
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np

from .schemas import (
    ActiveAeroMode,
    BackendCapability,
    DriverActionV2,
    DriverControllerStatusV1,
    DriverObservationV2,
    EvidenceReadinessEvaluator,
    PhysicsIdentity,
    RecordScenarioV1,
    ShiftRequest,
    TelemetryEvent,
    TrackSurfaceV2,
    VehicleSpecV2,
    VehicleTelemetryV2,
    canonical_json,
    content_sha256,
)


FIXED_TIMESTEP_S = 0.001
G = 9.80665
AIR_DENSITY_KG_M3 = 1.204


def _source_bundle_sha256(mjcf_text: str) -> str:
    """Hash every source unit that defines the backend contract plus MJCF."""
    digest = hashlib.sha256()
    for name in ("schemas.py", "defaults.py", "backends.py"):
        path = Path(__file__).with_name(name)
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    digest.update(b"authority.mjcf\0")
    digest.update(mjcf_text.encode("utf-8"))
    return digest.hexdigest()


def _distribution_record(distribution_name: str) -> dict[str, str]:
    """Capture installed package metadata and its wheel RECORD when available."""
    try:
        distribution = importlib.metadata.distribution(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return {"version": "missing", "record_sha256": "missing"}
    record = distribution.read_text("RECORD") or "RECORD-unavailable"
    return {
        "version": distribution.version,
        "record_sha256": hashlib.sha256(record.encode("utf-8")).hexdigest(),
    }


def _module_file_sha256(module: Any) -> str:
    path_value = getattr(module, "__file__", None)
    if not path_value:
        return "module-file-unavailable"
    path = Path(path_value)
    if not path.is_file():
        return "module-file-unavailable"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dependency_sha256(modules: dict[str, Any]) -> str:
    manifest: dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": sys.implementation.name,
            "cache_tag": sys.implementation.cache_tag,
        },
        "platform": platform.platform(),
    }
    for name, module in sorted(modules.items()):
        distribution_name = "jax" if name == "jax" else name.split(".", 1)[0]
        manifest[name] = {
            "module_version": str(getattr(module, "__version__", "unknown")),
            "module_file_sha256": _module_file_sha256(module),
            "distribution": _distribution_record(distribution_name),
        }
    return hashlib.sha256(canonical_json(manifest).encode("utf-8")).hexdigest()


def _optional_version(module_name: str) -> tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # optional dependency boundary
        return False, f"{type(exc).__name__}: {exc}"
    return True, str(getattr(module, "__version__", "unknown"))


@dataclass(frozen=True, slots=True)
class EnergyLedgerV1:
    fuel_chemical_energy_j: float
    battery_energy_j: float
    kinetic_energy_j: float
    cumulative_aero_loss_j: float
    cumulative_rolling_loss_j: float
    cumulative_friction_brake_loss_j: float
    cumulative_conversion_loss_j: float
    cumulative_numerical_residual_j: float

    @property
    def stored_energy_j(self) -> float:
        return self.fuel_chemical_energy_j + self.battery_energy_j + self.kinetic_energy_j

    @property
    def accounted_loss_j(self) -> float:
        return (
            self.cumulative_aero_loss_j
            + self.cumulative_rolling_loss_j
            + self.cumulative_friction_brake_loss_j
            + self.cumulative_conversion_loss_j
        )


@dataclass(frozen=True, slots=True)
class AuthoritySnapshotV1:
    identity_sha256: str
    qpos: tuple[float, ...]
    qvel: tuple[float, ...]
    telemetry: VehicleTelemetryV2
    energy: EnergyLedgerV1
    state_sha256: str


@dataclass(frozen=True, slots=True)
class AuthorityModelMetadataV1:
    timestep_s: float
    nq: int
    nv: int
    vehicle_mass_kg: float
    mjcf_sha256: str
    integrity_sha256: str


@dataclass(frozen=True, slots=True)
class TwinSnapshotV1:
    identity_sha256: str
    initialized: bool
    detail: str


class MuJoCoAuthority:
    """Deterministic public-anchor MuJoCo baseline with explicit energy state."""

    def __init__(
        self,
        vehicle: VehicleSpecV2,
        track: TrackSurfaceV2,
        scenario: RecordScenarioV1,
        *,
        timestep_s: float = FIXED_TIMESTEP_S,
    ) -> None:
        if not math.isclose(timestep_s, FIXED_TIMESTEP_S, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("MuJoCoAuthority faithful-v2 baseline is fixed at exactly 0.001 s")
        try:
            import mujoco
        except Exception as exc:  # pragma: no cover - depends on target installation
            raise RuntimeError("MuJoCoAuthority requires the 'mujoco' package") from exc

        self._mujoco = mujoco
        self.vehicle = vehicle
        self.track = track
        self.scenario = scenario
        self.dt = FIXED_TIMESTEP_S
        self.readiness = EvidenceReadinessEvaluator.evaluate(vehicle, track, scenario)

        baseline_blockers = tuple(sorted(set(self.readiness.blockers + (
            "public-anchor baseline is planar, not the licensed 6-DOF certification model",
            "per-wheel suspension/contact, tyre, aero-map and thermal authority are not implemented",
            "MJX cross-backend correlation has not passed",
        ))))
        self._capability = BackendCapability(
            backend_name="mujoco-authority-baseline",
            available=True,
            runnable=True,
            precision="float64",
            timestep_s=self.dt,
            vectorized=False,
            differentiable=False,
            certification_authority=False,
            faithful_claim_allowed=False,
            capability_label="public-anchor MuJoCo baseline (noncertifying)",
            blockers=baseline_blockers,
        )
        self._identity = PhysicsIdentity(
            backend_name="mujoco-authority-baseline",
            backend_version="faithful-v2-baseline-1",
            precision="float64",
            timestep_s=self.dt,
            vehicle_spec_sha256=vehicle.sha256,
            track_surface_sha256=track.sha256,
            source_sha256="0" * 64,
            dependency_sha256=_dependency_sha256({"mujoco": mujoco, "numpy": np}),
            capability_label=self._capability.capability_label,
        )
        self._mjcf_text = self._mjcf()
        self._mjcf_sha256 = hashlib.sha256(self._mjcf_text.encode("utf-8")).hexdigest()
        self._identity = PhysicsIdentity(
            backend_name=self._identity.backend_name,
            backend_version=self._identity.backend_version,
            precision=self._identity.precision,
            timestep_s=self._identity.timestep_s,
            vehicle_spec_sha256=self._identity.vehicle_spec_sha256,
            track_surface_sha256=self._identity.track_surface_sha256,
            source_sha256=_source_bundle_sha256(self._mjcf_text),
            dependency_sha256=self._identity.dependency_sha256,
            capability_label=self._identity.capability_label,
        )
        self._model = mujoco.MjModel.from_xml_string(self._mjcf_text)
        self._data = mujoco.MjData(self._model)
        if not math.isclose(float(self._model.opt.timestep), self.dt, rel_tol=0.0, abs_tol=1e-15):
            raise RuntimeError("MuJoCo model did not preserve the 1 ms authority timestep")
        self._initialize_constants()
        self._model_integrity_sha256 = self._model_integrity()
        self.reset()

    @property
    def identity(self) -> PhysicsIdentity:
        return self._identity

    @property
    def capability(self) -> BackendCapability:
        return self._capability

    @property
    def model_metadata(self) -> AuthorityModelMetadataV1:
        """Immutable verification data; the mutable live ``MjModel`` is not exposed."""
        self._assert_model_integrity()
        return AuthorityModelMetadataV1(
            timestep_s=float(self._model.opt.timestep),
            nq=int(self._model.nq),
            nv=int(self._model.nv),
            vehicle_mass_kg=float(self._model.body_mass[1]),
            mjcf_sha256=self._mjcf_sha256,
            integrity_sha256=self._model_integrity_sha256,
        )

    def _model_integrity(self) -> str:
        # Hash every exposed compiled-model array/scalar plus solver options,
        # not merely mass and timestep. This catches runtime edits to contacts,
        # geometry, joints, actuators, limits, damping, friction, solver flags,
        # and derived constants before an evidence snapshot can be produced.
        digest = hashlib.sha256()
        digest.update(self._mjcf_sha256.encode("ascii"))

        def update_public_values(prefix: str, owner: object) -> None:
            for name in sorted(value for value in dir(owner) if not value.startswith("_")):
                try:
                    value = getattr(owner, name)
                except Exception:
                    continue
                label = f"{prefix}.{name}".encode("utf-8")
                if isinstance(value, np.ndarray):
                    array = np.ascontiguousarray(value)
                    digest.update(label)
                    digest.update(str(array.dtype).encode("ascii"))
                    digest.update(repr(array.shape).encode("ascii"))
                    digest.update(array.tobytes())
                elif isinstance(value, (bool, int, float, str, bytes)):
                    digest.update(label)
                    digest.update(repr(value).encode("utf-8"))

        update_public_values("model", self._model)
        update_public_values("model.opt", self._model.opt)
        update_public_values("model.vis", self._model.vis)
        update_public_values("model.stat", self._model.stat)
        return digest.hexdigest()

    def _assert_model_integrity(self) -> None:
        actual = self._model_integrity()
        if actual != self._model_integrity_sha256:
            raise RuntimeError(
                "MuJoCoAuthority model integrity changed after PhysicsIdentity was frozen"
            )

    def _value(self, name: str) -> float:
        return self.vehicle.parameter(name).scalar()

    def _mjcf(self) -> str:
        mass = (
            self._value("vehicle_mass_kg")
            + self._value("driver_ballast_mass_kg")
            + self.scenario.initial_fuel_mass.scalar()
        )
        half_length = self._value("body_length_m") / 2.0
        half_width = self._value("body_width_m") / 2.0
        half_height = self._value("body_height_m") / 2.0
        yaw_inertia = self._value("yaw_inertia_kg_m2")
        # Only yaw is free in this planar baseline.  The locked-axis values are
        # finite, triangle-valid placeholders and never exposed as measured data.
        roll_inertia = 0.75 * yaw_inertia
        pitch_inertia = 0.75 * yaw_inertia
        # Three planar generalized coordinates are deliberate: this is an
        # executable integration/energy baseline, not a pretend 6-DOF model.
        return f"""
<mujoco model="porsche_919evo_faithful_v2_public_baseline">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="{self.dt:.9f}" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <body name="vehicle" pos="0 0 0">
      <joint name="world_x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="world_y" type="slide" axis="0 1 0" damping="0"/>
      <joint name="yaw" type="hinge" axis="0 0 1" damping="0"/>
      <inertial pos="0 0 0" mass="{mass:.12g}"
                diaginertia="{roll_inertia:.12g} {pitch_inertia:.12g} {yaw_inertia:.12g}"/>
      <geom name="canonical_body" type="box"
            size="{half_length:.12g} {half_width:.12g} {half_height:.12g}"
            contype="0" conaffinity="0" density="0" rgba="0.9 0.1 0.1 1"/>
    </body>
  </worldbody>
</mujoco>
"""

    def _initialize_constants(self) -> None:
        self._mass = float(self._model.body_mass[1])
        self._yaw_inertia = self._value("yaw_inertia_kg_m2")
        self._wheelbase = self._value("wheelbase_m")
        self._wheel_radius = self._value("effective_wheel_radius_m")
        self._ice_max_power = self._value("ice_max_power_w")
        self._ice_peak_torque = self._value("ice_peak_torque_nm")
        self._ice_efficiency = self._value("ice_thermal_efficiency")
        self._fuel_lhv = self._value("fuel_lower_heating_value_j_kg")
        self._mgu_max_power = self._value("front_mgu_max_power_w")
        self._mgu_peak_torque = self._value("front_mgu_peak_torque_nm")
        self._mgu_drive_efficiency = self._value("front_mgu_drive_efficiency")
        self._mgu_regen_efficiency = self._value("front_mgu_regen_efficiency")
        self._regen_power_limit = self._value("front_mgu_regen_power_limit_w")
        self._battery_capacity = self._value("battery_usable_energy_j")
        self._battery_discharge_limit = self._value("battery_max_discharge_power_w")
        self._battery_charge_limit = self._value("battery_max_charge_power_w")
        self._battery_min_soc = self._value("battery_min_soc")
        self._battery_max_soc = self._value("battery_max_soc")
        self._battery_voltage = self._value("battery_nominal_voltage_v")
        ratios = self.vehicle.parameter("gear_ratios").value
        if not isinstance(ratios, tuple):
            raise TypeError("gear_ratios must be tuple-valued")
        self._gear_ratios = tuple(float(ratio) for ratio in ratios)
        self._final_drive = self._value("final_drive_ratio")
        self._driveline_efficiency = self._value("driveline_efficiency")
        self._redline_rpm = self._value("ice_redline_rpm")
        self._max_brake_force = self._value("maximum_service_brake_force_n")
        self._cda = self._value("closed_aero_cda_m2")
        self._peak_mu = self._value("tyre_peak_mu")

    def reset(self, *, speed_mps: float = 0.0) -> VehicleTelemetryV2:
        self._assert_model_integrity()
        if not math.isfinite(speed_mps) or speed_mps < 0.0:
            raise ValueError("reset speed_mps must be finite and nonnegative")
        self._mujoco.mj_resetData(self._model, self._data)
        self._data.qvel[0] = float(speed_mps)
        self._mujoco.mj_forward(self._model, self._data)
        self._step_index = 0
        self._gear = 1
        self._previous_shift_request = ShiftRequest.HOLD
        self._steering_angle = 0.0
        self._accelerator = 0.0
        self._brake = 0.0
        self._boost = 0.0
        self._battery_energy = self._battery_capacity * self.scenario.initial_battery_soc.scalar()
        self._fuel_mass = self.scenario.initial_fuel_mass.scalar()
        self._battery_temperature = self.scenario.air_temperature.scalar() + 12.0
        self._coolant_temperature = self.scenario.air_temperature.scalar() + 70.0
        brake_temperature = self.scenario.initial_brake_temperature.scalar()
        tyre_temperature = self.scenario.initial_tyre_temperature.scalar()
        self._brake_temperatures = np.full(4, brake_temperature, dtype=np.float64)
        self._tyre_temperatures = np.full(4, tyre_temperature, dtype=np.float64)
        self._events: list[TelemetryEvent] = []
        self._event_sequence = 0
        self._last_acceleration_world = np.zeros(2, dtype=np.float64)
        self._last_engine_rpm = self._engine_rpm(speed_mps)
        self._last_engine_torque = 0.0
        self._last_engine_power = 0.0
        self._last_mgu_torque = 0.0
        self._last_mgu_power = 0.0
        self._last_regen_power = 0.0
        self._last_battery_power = 0.0
        self._last_drag_force = 0.0
        self._last_drive_force = 0.0
        self._last_brake_force = 0.0
        self._last_shift_state = "steady"
        self._loss_aero = 0.0
        self._loss_rolling = 0.0
        self._loss_brake = 0.0
        self._loss_conversion = 0.0
        self._numerical_residual = 0.0
        self._previous_total_energy = self._stored_energy()
        return self._telemetry()

    def _engine_rpm(self, forward_speed: float) -> float:
        ratio = self._gear_ratios[self._gear - 1] * self._final_drive
        wheel_rpm = max(0.0, forward_speed) / max(self._wheel_radius, 1e-9) * 60.0 / (2.0 * math.pi)
        return float(np.clip(max(3000.0, wheel_rpm * ratio), 3000.0, self._redline_rpm))

    def _world_to_body(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        c, s = math.cos(yaw), math.sin(yaw)
        return c * x + s * y, -s * x + c * y

    def _body_to_world(self, x: float, y: float, yaw: float) -> tuple[float, float]:
        c, s = math.cos(yaw), math.sin(yaw)
        return c * x - s * y, s * x + c * y

    def _append_event(self, event_type: str, detail: str) -> None:
        self._event_sequence += 1
        self._events.append(
            TelemetryEvent(self._event_sequence, float(self._data.time), event_type, detail)
        )

    def step(self, action: DriverActionV2) -> VehicleTelemetryV2:
        self._assert_model_integrity()
        action.assert_permitted(self.scenario.controls)
        self._last_shift_state = "steady"
        if (
            action.shift_request is not ShiftRequest.HOLD
            and self._previous_shift_request is ShiftRequest.HOLD
        ):
            old_gear = self._gear
            if action.shift_request is ShiftRequest.UPSHIFT:
                self._gear = min(len(self._gear_ratios), self._gear + 1)
            else:
                self._gear = max(1, self._gear - 1)
            if self._gear != old_gear:
                self._last_shift_state = "shifted"
                self._append_event("gear-change", f"{old_gear}->{self._gear}")
        self._previous_shift_request = action.shift_request

        self._accelerator = action.accelerator
        self._brake = action.brake
        max_steer = math.radians(32.0)
        max_steer_step = math.radians(360.0) * self.dt
        target_steer = action.steering * max_steer
        self._steering_angle += float(np.clip(
            target_steer - self._steering_angle, -max_steer_step, max_steer_step
        ))

        yaw = float(self._data.qpos[2])
        old_velocity = np.asarray(self._data.qvel[:2], dtype=np.float64).copy()
        forward_speed, lateral_speed = self._world_to_body(old_velocity[0], old_velocity[1], yaw)
        speed = math.hypot(forward_speed, lateral_speed)
        engine_rpm = self._engine_rpm(max(forward_speed, 0.0))
        engine_omega = engine_rpm * 2.0 * math.pi / 60.0
        available_torque = min(self._ice_peak_torque, self._ice_max_power / max(engine_omega, 1.0))
        engine_torque = self._accelerator * available_torque
        ratio = self._gear_ratios[self._gear - 1] * self._final_drive
        ice_force = engine_torque * ratio * self._driveline_efficiency / self._wheel_radius
        ice_force = min(ice_force, self._ice_max_power / max(abs(forward_speed), 5.0))

        soc = self._battery_energy / self._battery_capacity
        deploy_available = soc > self._battery_min_soc + 1e-9
        # Brake demand selects regen; the baseline never reports simultaneous
        # deployment and regeneration through one front MGU.
        mgu_requested_power = (
            self._accelerator * min(self._mgu_max_power, self._battery_discharge_limit)
            if self._brake == 0.0 else 0.0
        )
        if not deploy_available:
            mgu_requested_power = 0.0
        mgu_force = min(
            mgu_requested_power * self._mgu_drive_efficiency / max(abs(forward_speed), 5.0),
            self._mgu_peak_torque / self._wheel_radius,
        )
        traction_limit = self._peak_mu * self._mass * G
        drive_total = ice_force + mgu_force
        if drive_total > traction_limit:
            scale = traction_limit / drive_total
            ice_force *= scale
            mgu_force *= scale
            drive_total = traction_limit

        desired_brake_force = self._brake * self._max_brake_force
        regen_force = 0.0
        if abs(forward_speed) > 1.0 and soc < self._battery_max_soc - 1e-9:
            regen_force = min(
                desired_brake_force,
                self._regen_power_limit / max(abs(forward_speed), 1.0),
                self._battery_charge_limit / max(self._mgu_regen_efficiency * abs(forward_speed), 1.0),
            )
        friction_brake_force = max(0.0, desired_brake_force - regen_force)
        drag_force = 0.5 * AIR_DENSITY_KG_M3 * self._cda * speed * speed
        rolling_force = 0.012 * self._mass * G if speed > 0.05 else 0.0
        resist_sign = 1.0 if forward_speed >= 0.0 else -1.0
        longitudinal_force = drive_total - resist_sign * (
            desired_brake_force + drag_force + rolling_force
        )
        if abs(forward_speed) < 0.03 and longitudinal_force < 0.0 and drive_total <= 0.0:
            longitudinal_force = 0.0

        target_yaw_rate = forward_speed * math.tan(self._steering_angle) / self._wheelbase
        yaw_rate = float(self._data.qvel[2])
        yaw_torque = self._yaw_inertia * (target_yaw_rate - yaw_rate) / 0.08
        yaw_torque = float(np.clip(yaw_torque, -25000.0, 25000.0))
        lateral_force = self._mass * (-lateral_speed / 0.12 + forward_speed * target_yaw_rate)
        lateral_force = float(np.clip(lateral_force, -traction_limit, traction_limit))
        force_world = self._body_to_world(longitudinal_force, lateral_force, yaw)
        self._data.qfrc_applied[:] = (force_world[0], force_world[1], yaw_torque)
        old_forward = forward_speed
        self._mujoco.mj_step(self._model, self._data)
        self._data.qfrc_applied[:] = 0.0
        self._step_index += 1

        new_yaw = float(self._data.qpos[2])
        new_velocity = np.asarray(self._data.qvel[:2], dtype=np.float64).copy()
        new_forward, _ = self._world_to_body(new_velocity[0], new_velocity[1], new_yaw)
        if new_forward < 0.0 and self._accelerator == 0.0:
            _, lateral = self._world_to_body(new_velocity[0], new_velocity[1], new_yaw)
            vx, vy = self._body_to_world(0.0, lateral, new_yaw)
            self._data.qvel[0:2] = (vx, vy)
            new_velocity = np.asarray(self._data.qvel[:2], dtype=np.float64).copy()
            new_forward = 0.0
        average_forward = max(0.0, 0.5 * (max(old_forward, 0.0) + max(new_forward, 0.0)))

        ice_wheel_power = max(0.0, ice_force * average_forward)
        mgu_wheel_power = max(0.0, mgu_force * average_forward)
        regen_mechanical_power = max(0.0, regen_force * average_forward)
        fuel_input_power = ice_wheel_power / max(self._driveline_efficiency * self._ice_efficiency, 1e-9)
        battery_discharge_power = mgu_wheel_power / max(self._mgu_drive_efficiency, 1e-9)
        battery_charge_power = regen_mechanical_power * self._mgu_regen_efficiency
        battery_power = battery_discharge_power - battery_charge_power

        previous_stored = self._previous_total_energy
        fuel_energy_used = fuel_input_power * self.dt
        self._fuel_mass = max(0.0, self._fuel_mass - fuel_energy_used / self._fuel_lhv)
        self._battery_energy = float(np.clip(
            self._battery_energy - battery_power * self.dt,
            self._battery_min_soc * self._battery_capacity,
            self._battery_max_soc * self._battery_capacity,
        ))
        aero_loss = drag_force * speed * self.dt
        rolling_loss = rolling_force * speed * self.dt
        friction_brake_loss = friction_brake_force * average_forward * self.dt
        conversion_loss = (
            max(0.0, fuel_input_power - ice_wheel_power)
            + max(0.0, battery_discharge_power - mgu_wheel_power)
            + max(0.0, regen_mechanical_power - battery_charge_power)
        ) * self.dt
        self._loss_aero += aero_loss
        self._loss_rolling += rolling_loss
        self._loss_brake += friction_brake_loss
        self._loss_conversion += conversion_loss
        new_stored = self._stored_energy()
        expected_loss = aero_loss + rolling_loss + friction_brake_loss + conversion_loss
        self._numerical_residual += (new_stored - previous_stored) + expected_loss
        self._previous_total_energy = new_stored

        self._boost += (self._accelerator - self._boost) * min(1.0, self.dt / 0.12)
        ambient = self.scenario.air_temperature.scalar()
        self._battery_temperature += (
            abs(battery_power) * 1.0e-6 - (self._battery_temperature - ambient) / 240.0
        ) * self.dt
        self._coolant_temperature += (
            fuel_input_power * 4.0e-7 - (self._coolant_temperature - ambient) / 180.0
        ) * self.dt
        self._brake_temperatures += (
            friction_brake_force * average_forward * 1.0e-6
            - (self._brake_temperatures - ambient) / 300.0
        ) * self.dt

        self._last_acceleration_world = (new_velocity - old_velocity) / self.dt
        self._last_engine_rpm = engine_rpm
        self._last_engine_torque = engine_torque
        self._last_engine_power = ice_wheel_power
        self._last_mgu_torque = (mgu_force - regen_force) * self._wheel_radius
        self._last_mgu_power = mgu_wheel_power - regen_mechanical_power
        # MGU/regen telemetry is mechanical-side.  Battery power remains the
        # electrical-side signed quantity after conversion losses.
        self._last_regen_power = regen_mechanical_power
        self._last_battery_power = battery_power
        self._last_drag_force = drag_force
        self._last_drive_force = drive_total
        self._last_brake_force = desired_brake_force
        return self._telemetry()

    def _kinetic_energy(self) -> float:
        translational = 0.5 * self._mass * float(np.dot(self._data.qvel[:2], self._data.qvel[:2]))
        rotational = 0.5 * self._yaw_inertia * float(self._data.qvel[2]) ** 2
        return translational + rotational

    def _stored_energy(self) -> float:
        return self._fuel_mass * self._fuel_lhv + self._battery_energy + self._kinetic_energy()

    def _energy_ledger(self) -> EnergyLedgerV1:
        return EnergyLedgerV1(
            fuel_chemical_energy_j=self._fuel_mass * self._fuel_lhv,
            battery_energy_j=self._battery_energy,
            kinetic_energy_j=self._kinetic_energy(),
            cumulative_aero_loss_j=self._loss_aero,
            cumulative_rolling_loss_j=self._loss_rolling,
            cumulative_friction_brake_loss_j=self._loss_brake,
            cumulative_conversion_loss_j=self._loss_conversion,
            cumulative_numerical_residual_j=self._numerical_residual,
        )

    def _telemetry(self) -> VehicleTelemetryV2:
        yaw = float(self._data.qpos[2])
        velocity_body = self._world_to_body(float(self._data.qvel[0]), float(self._data.qvel[1]), yaw)
        acceleration_body = self._world_to_body(
            float(self._last_acceleration_world[0]), float(self._last_acceleration_world[1]), yaw
        )
        wheel_speed = velocity_body[0] / self._wheel_radius
        soc = self._battery_energy / self._battery_capacity
        engine_omega = self._last_engine_rpm * 2.0 * math.pi / 60.0
        brake_temperature = tuple(float(value) for value in self._brake_temperatures)
        tyre_temperature = tuple(float(value) for value in self._tyre_temperatures)
        return VehicleTelemetryV2(
            schema_version="vehicle-telemetry-v2",
            time_s=float(self._data.time),
            step_index=self._step_index,
            position_m=(float(self._data.qpos[0]), float(self._data.qpos[1]), 0.0),
            orientation_rpy_rad=(0.0, 0.0, yaw),
            velocity_body_mps=(velocity_body[0], velocity_body[1], 0.0),
            acceleration_body_mps2=(acceleration_body[0], acceleration_body[1], 0.0),
            angular_velocity_body_rad_s=(0.0, 0.0, float(self._data.qvel[2])),
            steering_angle_rad=self._steering_angle,
            accelerator_pedal=self._accelerator,
            brake_pedal=self._brake,
            engine_rpm=self._last_engine_rpm,
            engine_torque_nm=self._last_engine_torque,
            engine_power_w=self._last_engine_power,
            turbo_speed_rad_s=8000.0 + 100000.0 * self._boost,
            turbo_boost_pa=180000.0 * self._boost,
            gear=self._gear,
            gearbox_input_speed_rad_s=None,
            gearbox_output_speed_rad_s=None,
            shift_state=self._last_shift_state,
            battery_soc=soc,
            battery_voltage_v=self._battery_voltage,
            battery_current_a=self._last_battery_power / self._battery_voltage,
            battery_power_w=self._last_battery_power,
            front_mgu_torque_nm=self._last_mgu_torque,
            front_mgu_power_w=self._last_mgu_power,
            regen_power_w=self._last_regen_power,
            front_aero_position=None,
            rear_aero_position=None,
            aero_downforce_n=None,
            aero_drag_n=self._last_drag_force,
            suspension_travel_m=(None, None, None, None),
            wheel_speed_rad_s=(None, None, None, None),
            wheel_force_xyz_n=(None, None, None, None),
            wheel_slip_ratio=(None, None, None, None),
            wheel_slip_angle_rad=(None, None, None, None),
            wheel_contact=(None, None, None, None),
            brake_pressure_pa=(None, None, None, None),
            brake_temperature_k=brake_temperature,
            tyre_temperature_k=tyre_temperature,
            coolant_temperature_k=self._coolant_temperature,
            battery_temperature_k=self._battery_temperature,
            fuel_mass_kg=self._fuel_mass,
            events=tuple(self._events),
        )

    def observe(self) -> DriverObservationV2:
        telemetry = self._telemetry()
        deployment_available = (
            min(self._mgu_max_power, self._battery_discharge_limit)
            if telemetry.battery_soc is not None and telemetry.battery_soc > self._battery_min_soc
            else 0.0
        )
        return DriverObservationV2(
            schema_version="driver-observation-v2",
            time_s=telemetry.time_s,
            gps_position_m=telemetry.position_m,
            velocity_body_mps=telemetry.velocity_body_mps,
            acceleration_body_mps2=telemetry.acceleration_body_mps2,
            angular_velocity_body_rad_s=telemetry.angular_velocity_body_rad_s,
            wheel_speed_rad_s=telemetry.wheel_speed_rad_s,
            steering_angle_rad=float(telemetry.steering_angle_rad or 0.0),
            accelerator_pedal=float(telemetry.accelerator_pedal or 0.0),
            brake_pedal=float(telemetry.brake_pedal or 0.0),
            gear=int(telemetry.gear or 1),
            engine_rpm=float(telemetry.engine_rpm or 0.0),
            battery_soc=float(telemetry.battery_soc or 0.0),
            deployment_available_w=deployment_available,
            temperatures_k=(
                float(telemetry.coolant_temperature_k or 0.0),
                float(telemetry.battery_temperature_k or 0.0),
                *(float(value or 0.0) for value in telemetry.tyre_temperature_k),
            ),
            controller_status=DriverControllerStatusV1(
                traction_control_active=None,
                brake_by_wire_active=None,
                hybrid_derated=None,
                active_aero_mode=ActiveAeroMode.UNKNOWN,
                fault_code_count=0,
            ),
            static_map_preview=(),
        )

    def snapshot(self) -> AuthoritySnapshotV1:
        self._assert_model_integrity()
        telemetry = self._telemetry()
        energy = self._energy_ledger()
        qpos = tuple(float(value) for value in self._data.qpos)
        qvel = tuple(float(value) for value in self._data.qvel)
        state_hash = content_sha256((self.identity.sha256, qpos, qvel, telemetry, energy))
        return AuthoritySnapshotV1(
            identity_sha256=self.identity.sha256,
            qpos=qpos,
            qvel=qvel,
            telemetry=telemetry,
            energy=energy,
            state_sha256=state_hash,
        )


class MJXTrainingTwin:
    """Fail-closed MJX training-twin boundary.

    Dependency availability is reported independently from dynamics readiness.
    The class intentionally refuses to run until the vectorized model shares the
    authority MJCF/parameter registry and passes correlation gates.
    """

    def __init__(
        self,
        vehicle: VehicleSpecV2,
        track: TrackSurfaceV2,
        scenario: RecordScenarioV1,
    ) -> None:
        self.vehicle = vehicle
        self.track = track
        self.scenario = scenario
        jax_available, jax_detail = _optional_version("jax")
        mjx_available, mjx_detail = _optional_version("mujoco.mjx")
        available = jax_available and mjx_available
        dependency_modules: dict[str, Any] = {"numpy": np}
        for dependency_name in ("jax", "mujoco"):
            try:
                dependency_modules[dependency_name] = importlib.import_module(dependency_name)
            except Exception:
                pass
        blockers = [
            "vectorized faithful-v2 dynamics adapter is not implemented",
            "MuJoCo/MJX standardized-maneuver correlation has not passed",
            "MJX is a training twin and can never issue a record certificate",
        ]
        if not available:
            blockers.append(f"dependency unavailable: jax={jax_detail}; mujoco.mjx={mjx_detail}")
        readiness = EvidenceReadinessEvaluator.evaluate(vehicle, track, scenario)
        blockers.extend(readiness.blockers)
        self._capability = BackendCapability(
            backend_name="mjx-training-twin",
            available=available,
            runnable=False,
            precision="float64",
            timestep_s=FIXED_TIMESTEP_S,
            vectorized=True,
            differentiable=True,
            certification_authority=False,
            faithful_claim_allowed=False,
            capability_label="MJX interface only (not correlated; noncertifying)",
            blockers=tuple(sorted(set(blockers))),
        )
        self._identity = PhysicsIdentity(
            backend_name="mjx-training-twin",
            backend_version="faithful-v2-interface-1",
            precision="float64",
            timestep_s=FIXED_TIMESTEP_S,
            vehicle_spec_sha256=vehicle.sha256,
            track_surface_sha256=track.sha256,
            source_sha256=_source_bundle_sha256("MJX interface only; no runnable MJCF"),
            dependency_sha256=_dependency_sha256(dependency_modules),
            capability_label=self._capability.capability_label,
        )

    @property
    def identity(self) -> PhysicsIdentity:
        return self._identity

    @property
    def capability(self) -> BackendCapability:
        return self._capability

    def reset(self, *, speed_mps: float = 0.0) -> VehicleTelemetryV2:
        raise RuntimeError(
            "MJXTrainingTwin is not runnable until the shared-model correlation gate passes"
        )

    def step(self, action: DriverActionV2) -> VehicleTelemetryV2:
        del action
        raise RuntimeError(
            "MJXTrainingTwin is not runnable until the shared-model correlation gate passes"
        )

    def snapshot(self) -> TwinSnapshotV1:
        return TwinSnapshotV1(
            identity_sha256=self.identity.sha256,
            initialized=False,
            detail="interface-only; no vectorized state exists",
        )


__all__ = [
    "AuthorityModelMetadataV1", "AuthoritySnapshotV1", "EnergyLedgerV1",
    "FIXED_TIMESTEP_S", "MJXTrainingTwin", "MuJoCoAuthority", "TwinSnapshotV1",
]
