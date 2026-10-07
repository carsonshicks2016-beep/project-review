#!/usr/bin/env python3
"""Validate faithful-v2 car assets and non-fabricating A/V projection."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supra.faithful.av import (  # noqa: E402
    AssetManifestError,
    AssetIntegrityError,
    CarAssetRegistry,
    TruthfulAvAdapter,
)
from supra.faithful.schemas import VehicleTelemetryV2  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def complete_telemetry(**overrides):
    values = dict(
        schema_version="vehicle-telemetry-v2",
        time_s=12.5,
        step_index=12500,
        position_m=(1.0, 2.0, 0.3),
        orientation_rpy_rad=(0.01, -0.02, 0.1),
        velocity_body_mps=(72.0, 0.2, -0.1),
        acceleration_body_mps2=(1.0, 0.3, -0.2),
        angular_velocity_body_rad_s=(0.01, 0.02, 0.03),
        steering_angle_rad=0.02,
        brake_pedal=0.0,
        wheel_speed_rad_s=(211.0, 212.0, 210.0, 211.0),
        suspension_travel_m=(0.01, 0.011, 0.009, 0.01),
        brake_temperature_k=(820.0, 818.0, 740.0, 739.0),
        brake_pressure_pa=(0.0, 0.0, 0.0, 0.0),
        tyre_temperature_k=(365.0, 365.0, 360.0, 360.0),
        wheel_contact=(True, True, True, True),
        wheel_force_xyz_n=((100.0, 200.0, 4800.0), (100.0, 200.0, 4750.0),
                           (80.0, 160.0, 5100.0), (80.0, 160.0, 5050.0)),
        front_aero_position=0.05,
        rear_aero_position=0.12,
        aero_downforce_n=22000.0,
        aero_drag_n=3800.0,
        engine_rpm=8100.0,
        engine_torque_nm=620.0,
        engine_power_w=525000.0,
        accelerator_pedal=0.92,
        turbo_speed_rad_s=118000.0,
        turbo_boost_pa=141000.0,
        front_mgu_torque_nm=210.0,
        front_mgu_power_w=310000.0,
        regen_power_w=0.0,
        battery_soc=0.71,
        battery_voltage_v=800.0,
        battery_current_a=400.0,
        battery_power_w=320000.0,
        battery_temperature_k=323.0,
        gearbox_input_speed_rad_s=848.0,
        gearbox_output_speed_rad_s=309.0,
        gear=6,
        shift_state="engaged",
        wheel_slip_ratio=(0.03, 0.031, 0.027, 0.026),
        wheel_slip_angle_rad=(0.04, 0.041, 0.035, 0.034),
        coolant_temperature_k=368.0,
        fuel_mass_kg=18.0,
        events=(),
    )
    values.update(overrides)
    return VehicleTelemetryV2(**values)


def validate_asset() -> CarAssetRegistry:
    registry = CarAssetRegistry.discover()
    require("porsche_919evo" in registry.vehicles(), "919 asset not discovered")
    asset = registry.for_vehicle("porsche_919evo")
    require(not asset.faithful_geometry_eligible, "procedural proxy must not claim faithful eligibility")
    report = registry.verify(asset, inspect_geometry=True)
    require(report.extents_m is not None, "geometry was not inspected")
    require(report.geometry_count == 46, f"unexpected geometry count: {report.geometry_count}")
    require(report.node_count == 46, f"unexpected node count: {report.node_count}")
    for actual, expected in zip(report.extents_m or (), (5.078, 1.900, 1.050)):
        require(abs(actual - expected) <= expected * 0.005, "canonical dimension outside 0.5%")
    # Browser mapping is canonical (x forward, y left, z up) -> Three
    # (x forward, y up, z right). This catches the easy-to-miss sideways-car
    # regression even though trimesh itself reports the source z-up extents.
    source_extents = report.extents_m or (0.0, 0.0, 0.0)
    browser_extents = (source_extents[0], source_extents[2], source_extents[1])
    for actual, expected in zip(browser_extents, (5.078, 1.050, 1.900)):
        require(abs(actual - expected) <= expected * 0.005,
                "browser-coordinate asset extent outside 0.5%")

    manifest = json.loads(asset.manifest_path.read_text(encoding="utf-8"))
    provenance = manifest["provenance"]
    require(provenance["official_media_embedded"] is False, "manifest embeds official media")
    require(provenance["oem_cad_claim"] is False, "procedural proxy claims OEM CAD")
    required_groups = {"wheels", "suspension_pivots", "brakes", "active_aero", "lights", "exhaust"}
    require(required_groups <= set(asset.canonical_nodes), "missing canonical node group")

    result = subprocess.run(
        [sys.executable, str(asset.generator_path), "--check", "--output", str(asset.glb_path)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    require(result.returncode == 0, f"asset generator is not deterministic: {result.stderr}")

    # Integrity errors must be fail-closed rather than silently loading a
    # modified mesh. Copying to a temp directory leaves the project untouched.
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "porsche_919evo"
        shutil.copytree(asset.manifest_path.parent, package)
        glb = package / asset.glb_path.name
        payload = bytearray(glb.read_bytes())
        payload[-1] ^= 0x01
        glb.write_bytes(payload)
        isolated = CarAssetRegistry(Path(tmp))
        copied = isolated.register_manifest(package / "asset_manifest.json")
        try:
            isolated.verify(copied)
        except AssetIntegrityError:
            pass
        else:
            raise AssertionError("tampered GLB passed asset integrity verification")

    # JSON strings are truthy in Python. A manifest must not turn the literal
    # string "false" into a certification-eligibility claim.
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "porsche_919evo"
        shutil.copytree(asset.manifest_path.parent, package)
        manifest_path = package / "asset_manifest.json"
        malformed = json.loads(manifest_path.read_text(encoding="utf-8"))
        malformed["faithful_geometry_eligible"] = "false"
        manifest_path.write_text(json.dumps(malformed), encoding="utf-8")
        try:
            CarAssetRegistry(Path(tmp)).register_manifest(manifest_path)
        except AssetManifestError:
            pass
        else:
            raise AssertionError("string geometry-eligibility flag was accepted")
    return registry


def validate_projection(registry: CarAssetRegistry) -> None:
    adapter = TruthfulAvAdapter(registry)
    projected = adapter.project(complete_telemetry(), vehicle_id="porsche_919evo")
    require(projected.audio.ice is not None, "measured ICE telemetry did not bind")
    require(projected.audio.turbo is not None, "measured turbo telemetry did not bind")
    require(projected.audio.front_mgu is not None, "measured MGU telemetry did not bind")
    require(projected.audio.front_mgu.regen_power_w == 0.0, "positive MGU power became regen")
    require(projected.audio.gearbox is not None, "measured gearbox telemetry did not bind")
    require(all(wheel is not None for wheel in projected.audio.wheels), "wheel sources did not bind")
    require(projected.visual.rear_aero_position == 0.12, "aero telemetry did not bind")

    # High throttle/RPM must not fabricate absent electrical, turbo, or aero
    # state. The unavailable list is an observable fail-closed contract.
    missing = adapter.project(complete_telemetry(
        front_mgu_torque_nm=None,
        front_mgu_power_w=None,
        regen_power_w=None,
        turbo_speed_rad_s=None,
        turbo_boost_pa=None,
        front_aero_position=None,
        rear_aero_position=None,
    ), vehicle_id="porsche_919evo")
    require(missing.audio.front_mgu is None, "adapter guessed missing MGU state")
    require(missing.audio.turbo is None, "adapter guessed missing turbo state")
    require(missing.visual.front_aero_position is None, "adapter guessed front-aero state")
    require(missing.visual.rear_aero_position is None, "adapter guessed rear-aero state")
    for channel in ("audio.front_mgu", "audio.turbo", "visual.active_aero.front_aero_position"):
        require(channel in missing.unavailable_channels, f"missing channel not reported: {channel}")

    regen = adapter.project(complete_telemetry(
        front_mgu_torque_nm=-180.0,
        front_mgu_power_w=-225000.0,
        regen_power_w=225000.0,
    ), vehicle_id="porsche_919evo")
    require(regen.audio.front_mgu is not None, "regen MGU source missing")
    require(regen.audio.front_mgu.regen_power_w == 225000.0, "signed MGU power was not projected")
    try:
        adapter.project(complete_telemetry(
            front_mgu_power_w=-225000.0,
            regen_power_w=100000.0,
        ), vehicle_id="porsche_919evo")
    except ValueError:
        pass
    else:
        raise AssertionError("inconsistent regen telemetry was accepted")


def main() -> int:
    registry = validate_asset()
    validate_projection(registry)
    print("faithful A/V validation: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
