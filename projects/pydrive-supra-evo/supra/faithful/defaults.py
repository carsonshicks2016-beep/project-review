"""Public-anchor defaults for the Porsche 919 Evo faithful-v2 interfaces.

These values are a runnable bootstrap, not a fidelity certificate.  Public
Porsche figures are tagged ``official`` while all unsourced maps, coefficients
and control laws are explicitly ``inferred``.  The readiness evaluator will
therefore block the faithful label until licensed measurements replace them.
"""
from __future__ import annotations

import hashlib

from .schemas import (
    Confidence,
    DistributionKind,
    EvidenceRef,
    PermittedDriverControlsV1,
    PhysicalParameter,
    RecordScenarioV1,
    TimingPlaneV1,
    TrackSurfaceV2,
    UncertaintyDistribution,
    VehicleSpecV2,
)


PORSCHE_TECHNICAL_URI = (
    "https://newsroom.porsche.com/en/motorsports/"
    "porsche-919-hybrid-evo-top-5-series-technical-check-16834.html"
)
PORSCHE_RECORD_URI = (
    "https://newsroom.porsche.com/en/motorsports/"
    "porsche-919-hybrid-evo-record-nuerburgring-nordschleife-"
    "5-minutes-19-seconds-55-timo-bernhard-15752.html"
)
NURBURGRING_RECORD_URI = "https://nuerburgring.de/info/nuerburgring/records?locale=en"
MICHELIN_CORRELATION_URI = (
    "https://simulation.michelin.com/canopy/technical-articles/"
    "f1-vs-porsche-919-evo-at-the-nordschleife"
)
INFERRED_URI = "urn:supra-ai:faithful-v2:public-anchor-bootstrap"


def _record_digest(uri: str, statement: str) -> str:
    payload = f"faithful-v2-citation-record\n{uri}\n{statement}\n".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _evidence(
    uri: str,
    statement: str,
    confidence: Confidence,
    *,
    license_id: str = "public-reference-private-research",
    calibration_dataset: str = "not-applicable",
) -> EvidenceRef:
    return EvidenceRef(
        source_uri=uri,
        source_sha256=_record_digest(uri, statement),
        digest_scope="citation-record" if confidence is Confidence.OFFICIAL else "derived-record",
        license_id=license_id,
        confidence=confidence,
        calibration_dataset=calibration_dataset,
    )


def _official(name: str, value: float | tuple[float, ...], unit: str, statement: str) -> PhysicalParameter:
    return PhysicalParameter(
        name=name,
        value=value,
        unit=unit,
        distribution=UncertaintyDistribution(DistributionKind.FIXED),
        evidence=_evidence(PORSCHE_TECHNICAL_URI, statement, Confidence.OFFICIAL),
        notes="Public official anchor; replace citation-record digest with captured source content.",
    )


def _inferred(
    name: str,
    value: float | tuple[float, ...],
    unit: str,
    lower: float | None,
    upper: float | None,
    *,
    high_sensitivity: bool = True,
    notes: str,
) -> PhysicalParameter:
    distribution = (
        UncertaintyDistribution(DistributionKind.UNIFORM, lower=lower, upper=upper)
        if lower is not None and upper is not None
        else UncertaintyDistribution(DistributionKind.NORMAL, sigma=abs(float(value)) * 0.2)
    )
    return PhysicalParameter(
        name=name,
        value=value,
        unit=unit,
        distribution=distribution,
        evidence=_evidence(
            INFERRED_URI,
            f"bootstrap inference for {name}",
            Confidence.INFERRED,
            license_id="internal-bootstrap-not-measurement",
            calibration_dataset="none-public-anchor-only",
        ),
        high_sensitivity=high_sensitivity,
        notes=notes,
    )


def official_919evo_vehicle_spec() -> VehicleSpecV2:
    """Return the immutable public-anchor 919 Evo bootstrap specification.

    The name describes the source of the anchors, not the readiness state.  Use
    :class:`EvidenceReadinessEvaluator` before displaying any fidelity label.
    """
    return VehicleSpecV2(
        schema_version="vehicle-spec-v2",
        vehicle_id="porsche-919-hybrid-evo-2018-faithful-v2",
        geometry=(
            _official("body_length_m", 5.078, "m", "Overall length is 5,078 mm."),
            _official("body_width_m", 1.900, "m", "Overall width is 1,900 mm."),
            _official("body_height_m", 1.050, "m", "Overall height is 1,050 mm."),
            _inferred("wheelbase_m", 2.80, "m", 2.60, 3.00, notes="Replace with homologation/K&C geometry."),
            _inferred("front_track_m", 1.65, "m", 1.55, 1.80, notes="Wheel-center track is not established by public anchors."),
            _inferred("rear_track_m", 1.62, "m", 1.50, 1.78, notes="Wheel-center track is not established by public anchors."),
            _inferred("effective_wheel_radius_m", 0.355, "m", 0.345, 0.365, notes="Derived from nominal 710 mm tyre diameter."),
        ),
        mass_properties=(
            _official("vehicle_mass_kg", 849.0, "kg", "Vehicle weight is 849 kg."),
            _official("driver_ballast_mass_kg", 39.0, "kg", "Record configuration carried 39 kg of ballast."),
            _inferred("reference_fuel_mass_kg", 8.0, "kg", 2.0, 18.0, notes="Exact record fuel load requires telemetry."),
            _inferred("cg_x_from_front_axle_m", 1.45, "m", 1.20, 1.70, notes="Requires measured mass properties."),
            _inferred("cg_height_m", 0.28, "m", 0.22, 0.36, notes="Requires measured mass properties."),
            _inferred("yaw_inertia_kg_m2", 1150.0, "kg*m^2", 850.0, 1500.0, notes="Requires measured inertia tensor."),
        ),
        suspension=(
            _inferred("front_heave_rate_n_m", 310000.0, "N/m", 180000.0, 500000.0, notes="Placeholder pushrod-equivalent rate."),
            _inferred("rear_heave_rate_n_m", 340000.0, "N/m", 200000.0, 540000.0, notes="Placeholder pushrod-equivalent rate."),
            _inferred("front_damping_n_s_m", 18000.0, "N*s/m", 9000.0, 30000.0, notes="Placeholder damper coefficient."),
            _inferred("rear_damping_n_s_m", 20000.0, "N*s/m", 10000.0, 32000.0, notes="Placeholder damper coefficient."),
            _inferred("pitch_link_gain", 1.0, "1", 0.0, 2.0, notes="Active pitch-link behavior requires controller maps."),
        ),
        tyres=(
            _official("front_tyre_size_mm", (310.0, 710.0, 18.0), "mm,mm,in", "Michelin front tyre is 310/710-18."),
            _official("rear_tyre_size_mm", (310.0, 710.0, 18.0), "mm,mm,in", "Michelin rear tyre is 310/710-18."),
            _inferred("tyre_peak_mu", 2.10, "1", 1.70, 2.45, notes="Must be replaced by licensed combined-slip tyre data."),
            _inferred("tyre_relaxation_length_m", 0.35, "m", 0.15, 0.70, notes="Requires licensed transient tyre measurements."),
            _inferred("tyre_optimal_temperature_k", 363.15, "K", 343.15, 383.15, notes="Requires compound-specific thermal data."),
        ),
        aero=(
            PhysicalParameter(
                name="evo_downforce_relative_to_wec",
                value=1.53,
                unit="1",
                distribution=UncertaintyDistribution(DistributionKind.FIXED),
                evidence=_evidence(PORSCHE_RECORD_URI, "Evo downforce increased by 53 percent.", Confidence.OFFICIAL),
            ),
            PhysicalParameter(
                name="evo_aero_efficiency_relative_to_wec",
                value=1.66,
                unit="1",
                distribution=UncertaintyDistribution(DistributionKind.FIXED),
                evidence=_evidence(PORSCHE_RECORD_URI, "Aerodynamic efficiency increased by 66 percent.", Confidence.OFFICIAL),
            ),
            _inferred("closed_aero_cda_m2", 1.00, "m^2", 0.65, 1.35, notes="Requires wind-tunnel/CFD map across ride state."),
            _inferred("closed_aero_cla_m2", 6.50, "m^2", 4.5, 8.5, notes="Requires wind-tunnel/CFD map across ride state."),
            _inferred("aero_balance_front", 0.45, "1", 0.38, 0.52, notes="Requires aero balance maps."),
        ),
        ice=(
            _official("ice_max_power_w", 720.0 * 735.49875, "W", "Rear combustion engine output is 720 PS."),
            _official("ice_redline_rpm", 9000.0, "rpm", "Combustion engine speed is approximately 9,000 rpm."),
            _inferred("ice_peak_torque_nm", 600.0, "N*m", 500.0, 700.0, notes="Full torque/boost map requires licensed dyno data."),
            _inferred("ice_thermal_efficiency", 0.43, "1", 0.36, 0.49, notes="Fuel map and transient efficiency are not public."),
            _inferred("fuel_lower_heating_value_j_kg", 43.0e6, "J/kg", 41.0e6, 44.0e6, high_sensitivity=False, notes="Fuel batch value must be recorded for certification."),
        ),
        mgu=(
            _official("front_mgu_max_power_w", 440.0 * 735.49875, "W", "Front-axle MGU output is 440 PS."),
            _inferred("front_mgu_peak_torque_nm", 650.0, "N*m", 450.0, 850.0, notes="Requires torque-speed map."),
            _inferred("front_mgu_drive_efficiency", 0.94, "1", 0.88, 0.98, notes="Requires measured efficiency map."),
            _inferred("front_mgu_regen_efficiency", 0.86, "1", 0.75, 0.94, notes="Requires measured regen map."),
            _inferred("front_mgu_regen_power_limit_w", 250000.0, "W", 120000.0, 360000.0, notes="Requires licensed KERS control and hardware limits."),
        ),
        battery=(
            _official("battery_nominal_voltage_v", 800.0, "V", "The 919 Hybrid uses an 800-volt architecture."),
            _inferred("battery_usable_energy_j", 8.0e6, "J", 3.0e6, 15.0e6, notes="Usable record-lap energy window is not public."),
            _inferred("battery_max_discharge_power_w", 360000.0, "W", 250000.0, 500000.0, notes="Requires cell and inverter limits."),
            _inferred("battery_max_charge_power_w", 300000.0, "W", 150000.0, 450000.0, notes="Requires cell and inverter limits."),
            _inferred("battery_min_soc", 0.10, "1", 0.05, 0.30, notes="Requires control strategy and cell limits."),
            _inferred("battery_max_soc", 0.95, "1", 0.80, 1.00, notes="Requires control strategy and cell limits."),
        ),
        gearbox=(
            _official("forward_gear_count", 7.0, "count", "Transmission has seven forward gears."),
            _inferred("gear_ratios", (3.09221, 2.31930, 1.77249, 1.36942, 1.10771, 0.92796, 0.82357), "1", 0.75, 3.30, notes="Current simulator optimization, not licensed Porsche ratios."),
            _inferred("final_drive_ratio", 3.51264, "1", 3.0, 4.0, notes="Current simulator optimization, not licensed Porsche final drive."),
            _inferred("driveline_efficiency", 0.96, "1", 0.90, 0.98, notes="Requires measured load-dependent losses."),
        ),
        brakes=(
            _inferred("maximum_service_brake_force_n", 26000.0, "N", 18000.0, 35000.0, notes="Requires pressure/torque and brake-by-wire maps."),
            _inferred("brake_bias_front", 0.56, "1", 0.48, 0.68, notes="Dynamic brake-by-wire allocation is not public."),
            _inferred("brake_optimal_temperature_k", 773.15, "K", 623.15, 973.15, notes="Requires carbon brake material data."),
        ),
        embedded_controllers=(
            _inferred("traction_control_slip_target", 0.10, "1", 0.04, 0.18, notes="Requires licensed ECU logic."),
            _inferred("brake_by_wire_regen_blend", 0.55, "1", 0.0, 1.0, notes="Requires licensed brake-by-wire logic."),
            _inferred("active_aero_response_time_s", 0.20, "s", 0.08, 0.50, notes="Requires actuator and controller maps."),
        ),
        thermal_limits=(
            _inferred("battery_derate_temperature_k", 333.15, "K", 318.15, 353.15, notes="Requires cell thermal limit data."),
            _inferred("coolant_derate_temperature_k", 393.15, "K", 378.15, 413.15, notes="Requires powertrain thermal limit data."),
            _inferred("brake_derate_temperature_k", 1173.15, "K", 1073.15, 1273.15, notes="Requires carbon brake fade data."),
        ),
    )


def training_fallback_track() -> TrackSurfaceV2:
    """Return the explicitly noncertifying 20.832 km training track identity."""
    statement = "Public full-lap record configuration length is 20.832 km."
    evidence = _evidence(NURBURGRING_RECORD_URI, statement, Confidence.OFFICIAL)
    return TrackSurfaceV2(
        schema_version="track-surface-v2",
        track_id="nordschleife-public-training-fallback",
        configuration="2018-T13-to-T13-flying-lap-approximation",
        lap_length=PhysicalParameter(
            name="lap_length_m",
            value=20832.0,
            unit="m",
            distribution=UncertaintyDistribution(DistributionKind.FIXED),
            evidence=evidence,
        ),
        survey_evidence=evidence,
        samples=(),
        timing_planes=(
            TimingPlaneV1("T13-start-finish", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ),
        curbs_sha256=None,
        barriers_sha256=None,
        materials_sha256=None,
        certification_eligible=False,
        approximation_reason=(
            "No licensed June-2018 laser survey, variable legal boundaries, curbs, "
            "barriers or material map has been ingested."
        ),
    )


def _scenario_parameter(
    name: str,
    value: float,
    unit: str,
    lower: float,
    upper: float,
    notes: str,
) -> PhysicalParameter:
    return _inferred(name, value, unit, lower, upper, high_sensitivity=False, notes=notes)


def default_record_scenario(
    vehicle: VehicleSpecV2 | None = None,
    track: TrackSurfaceV2 | None = None,
) -> RecordScenarioV1:
    vehicle = vehicle or official_919evo_vehicle_spec()
    track = track or training_fallback_track()
    return RecordScenarioV1(
        schema_version="record-scenario-v1",
        scenario_id="porsche-919evo-nordschleife-2018-public-bootstrap-v1",
        vehicle_spec_sha256=vehicle.sha256,
        track_surface_sha256=track.sha256,
        target_lap_time_s=319.546,
        air_temperature=_scenario_parameter("air_temperature_k", 293.15, "K", 278.15, 308.15, "Freeze from recorded weather."),
        track_temperature=_scenario_parameter("track_temperature_k", 298.15, "K", 283.15, 323.15, "Freeze from recorded weather."),
        air_pressure=_scenario_parameter("air_pressure_pa", 101325.0, "Pa", 95000.0, 104000.0, "Freeze from recorded weather."),
        relative_humidity=_scenario_parameter("relative_humidity", 0.50, "1", 0.20, 0.95, "Freeze from recorded weather."),
        wind_speed=_scenario_parameter("wind_speed_mps", 0.0, "m/s", 0.0, 15.0, "Freeze from recorded weather."),
        wind_direction=_scenario_parameter("wind_direction_rad", 0.0, "rad", -3.141592653589793, 3.141592653589793, "Freeze from recorded weather."),
        initial_fuel_mass=_scenario_parameter("initial_fuel_mass_kg", 8.0, "kg", 2.0, 18.0, "Requires record preparation data."),
        initial_battery_soc=_scenario_parameter("initial_battery_soc", 0.90, "1", 0.50, 0.95, "Requires record preparation data."),
        initial_tyre_temperature=_scenario_parameter("initial_tyre_temperature_k", 363.15, "K", 333.15, 383.15, "Requires warm-up telemetry."),
        initial_brake_temperature=_scenario_parameter("initial_brake_temperature_k", 673.15, "K", 473.15, 873.15, "Requires warm-up telemetry."),
        controls=PermittedDriverControlsV1(
            steering=True,
            accelerator=True,
            brake=True,
            shift_request=True,
            boost_request=False,
            drs_request=False,
        ),
        timing_rule="continuous T13-to-T13 flying lap with interpolated timing-plane crossing",
        holdout_policy=(
            "Calibrate with component and Spa data; freeze every parameter/hash before opening "
            "Nordschleife record telemetry."
        ),
        record_telemetry_opened=False,
    )


__all__ = [
    "INFERRED_URI", "MICHELIN_CORRELATION_URI", "NURBURGRING_RECORD_URI",
    "PORSCHE_RECORD_URI", "PORSCHE_TECHNICAL_URI", "default_record_scenario",
    "official_919evo_vehicle_spec", "training_fallback_track",
]
