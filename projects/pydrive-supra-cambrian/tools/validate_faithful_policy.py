#!/usr/bin/env python3
"""Regression gates for the faithful-v2 driver architecture and actor boundary."""
from __future__ import annotations

from dataclasses import fields, replace
import hashlib
from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from supra.faithful.policy import (  # noqa: E402
    DRIVER_DT_S,
    DriverObservationEncoderV2,
    DriverObservationLayoutV2,
    DriverPolicyRuntime50Hz,
    FaithfulRecurrentActorCritic,
    PolicyArchitectureV1,
    TrainingAuthorizationV1,
    audit_actor_boundary,
    behavioral_cloning_loss,
)
from supra.faithful.schemas import (  # noqa: E402
    ActiveAeroMode,
    DriverControllerStatusV1,
    DriverObservationV2,
    MapPreviewPointV1,
)


def gate(name: str, condition: bool) -> None:
    print(f"  [{'PASS' if condition else 'FAIL'}] {name}")
    if not condition:
        raise SystemExit(f"gate failed: {name}")


def observation(time_s: float = 0.0) -> DriverObservationV2:
    preview = tuple(
        MapPreviewPointV1(distance, (distance, 0.1 * index, 0.02 * index), 5.5, 5.4)
        for index, distance in enumerate((25.0, 75.0, 150.0))
    )
    return DriverObservationV2(
        schema_version="driver-observation-v2",
        time_s=time_s,
        gps_position_m=(100.0, 200.0, 320.0),
        velocity_body_mps=(70.0, 0.2, -0.1),
        acceleration_body_mps2=(1.1, 2.0, -0.2),
        angular_velocity_body_rad_s=(0.01, 0.02, 0.10),
        wheel_speed_rad_s=(198.0, 198.2, 197.8, 198.1),
        steering_angle_rad=0.04,
        accelerator_pedal=0.72,
        brake_pedal=0.0,
        gear=6,
        engine_rpm=7800.0,
        battery_soc=0.71,
        deployment_available_w=260_000.0,
        temperatures_k=(365.0, 810.0, 325.0),
        controller_status=DriverControllerStatusV1(
            traction_control_active=False,
            brake_by_wire_active=True,
            hybrid_derated=False,
            active_aero_mode=ActiveAeroMode.LOW_DRAG,
            fault_code_count=0,
        ),
        static_map_preview=preview,
    )


def main() -> None:
    controller_fields = tuple(field.name for field in fields(DriverControllerStatusV1))
    track_sha256 = hashlib.sha256(b"frozen 2018 track").hexdigest()
    preview_generator_sha256 = hashlib.sha256(
        b"approved static map preview generator"
    ).hexdigest()
    layout = DriverObservationLayoutV2(
        temperature_count=3,
        controller_fields=controller_fields,
        preview_distances_m=(25.0, 75.0, 150.0),
        track_surface_sha256=track_sha256,
        map_preview_generator_sha256=preview_generator_sha256,
    )
    encoder = DriverObservationEncoderV2(layout)
    encoded = encoder.encode(observation())
    gate("fixed observation layout encodes exact feature count",
         encoded.shape == (layout.feature_count,) and np.isfinite(encoded).all())
    audit = audit_actor_boundary(
        layout,
        expected_track_surface_sha256=track_sha256,
        approved_map_preview_generator_sha256=preview_generator_sha256,
    )
    gate("actor boundary audit passes closed onboard schema", audit.passed)
    gate("tyre-force truth is rejected from actor schema",
         "wheel_force_xyz_n" in audit.rejected_privileged_fields)
    try:
        DriverObservationLayoutV2(
            3, (*controller_fields, "hidden_friction"), (25.0, 75.0, 150.0),
            track_sha256, preview_generator_sha256,
        )
        # A layout can be constructed for audit/reporting but may not pass.
        leak_blocked = not audit_actor_boundary(DriverObservationLayoutV2(
            3, (*controller_fields, "hidden_friction"), (25.0, 75.0, 150.0),
            track_sha256, preview_generator_sha256,
        ), expected_track_surface_sha256=track_sha256,
            approved_map_preview_generator_sha256=preview_generator_sha256).passed
    except ValueError:
        leak_blocked = True
    gate("privileged controller label fails boundary audit", leak_blocked)
    wrong_map_audit = audit_actor_boundary(
        layout,
        expected_track_surface_sha256="a" * 64,
        approved_map_preview_generator_sha256=preview_generator_sha256,
    )
    gate("static preview is hash-bound to the run track", not wrong_map_audit.passed)
    try:
        DriverControllerStatusV1(
            traction_control_active="false",  # type: ignore[arg-type]
            brake_by_wire_active=True,
            hybrid_derated=False,
            active_aero_mode=ActiveAeroMode.LOW_DRAG,
            fault_code_count=0,
        )
        string_bool_rejected = False
    except TypeError:
        string_bool_rejected = True
    gate("truthy string cannot impersonate an ECU boolean", string_bool_rejected)

    torch.manual_seed(919)
    architecture = PolicyArchitectureV1(layout.feature_count, privileged_critic_dim=7)
    model = FaithfulRecurrentActorCritic(architecture).eval()
    gate("actor architecture is exactly 2x256 plus GRU128",
         tuple(layer.out_features for layer in model.actor_features if isinstance(layer, torch.nn.Linear))
         == (256, 256) and model.actor_gru.hidden_size == 128)
    actor_input = torch.from_numpy(encoded)[None, :]
    critic_a = torch.zeros(1, 7)
    critic_b = torch.full((1, 7), 1000.0)
    step_a = model.step(actor_input, privileged_observation=critic_a, deterministic=True)
    step_b = model.step(actor_input, privileged_observation=critic_b, deterministic=True)
    gate("privileged critic cannot change actor action",
         torch.equal(step_a.steering, step_b.steering)
         and torch.equal(step_a.accelerator, step_b.accelerator)
         and torch.equal(step_a.brake, step_b.brake)
         and torch.equal(step_a.shift_index, step_b.shift_index))
    gate("privileged state reaches only the value path",
         step_a.value is not None and step_b.value is not None
         and not torch.equal(step_a.value, step_b.value))

    runtime = DriverPolicyRuntime50Hz(model, encoder)
    action0 = runtime.act(observation(0.0))
    action1 = runtime.act(observation(DRIVER_DT_S))
    gate("runtime emits separate bounded pedals and categorical shift",
         -1 <= action0.steering <= 1
         and 0 <= action0.accelerator <= 1 and 0 <= action0.brake <= 1
         and int(action0.shift_request) in (-1, 0, 1))
    gate("undocumented boost and DRS actions remain disabled",
         action1.boost_request is None and action1.drs_request is None)
    try:
        runtime.act(observation(DRIVER_DT_S + 0.01))
        cadence_rejected = False
    except ValueError:
        cadence_rejected = True
    gate("driver inference loop enforces 50 Hz", cadence_rejected)

    model.train()
    batch = torch.randn(3, 4, layout.feature_count)
    loss = behavioral_cloning_loss(
        model,
        batch,
        torch.zeros(3, 4),
        torch.full((3, 4), 0.7),
        torch.full((3, 4), 0.1),
        torch.ones(3, 4, dtype=torch.long),
    )
    loss.backward()
    gate("oracle/MPC behavioral-cloning objective is differentiable",
         torch.isfinite(loss) and model.steering_mean.weight.grad is not None)

    blocked = TrainingAuthorizationV1(None, None, audit.sha256, False, False, True)
    try:
        blocked.assert_long_run_allowed()
        failed_closed = False
    except PermissionError:
        failed_closed = True
    gate("long-run training is blocked before physics and oracle gates", failed_closed)
    self_asserted = TrainingAuthorizationV1(
        "a" * 64, "b" * 64, audit.sha256, True, True, True
    )
    try:
        self_asserted.assert_long_run_allowed()
        self_assertion_blocked = False
    except PermissionError:
        self_assertion_blocked = True
    gate(
        "self-asserted hashes cannot open the absent training verifier",
        self_assertion_blocked,
    )

    print("faithful policy validation: PASS")


if __name__ == "__main__":
    main()
