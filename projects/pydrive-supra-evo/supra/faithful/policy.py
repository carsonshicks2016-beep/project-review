"""Driver-equivalent recurrent policy and privileged-critic boundary.

This module implements the frozen faithful-v2 network shape and mixed action
distribution.  It does not authorize training by itself: the long-run entry
point must first receive physics-validation and oracle-feasibility artifacts.
The actor accepts only a flattened :class:`DriverObservationV2`; privileged
state has a separate critic-only encoder and never enters the actor graph.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Beta, Categorical, Normal
import torch.nn.functional as F

from .schemas import (
    DriverActionV2,
    DriverControllerStatusV1,
    DriverObservationV2,
    ShiftRequest,
    VehicleTelemetryV2,
    content_sha256,
)


POLICY_ARCHITECTURE = "faithful-driver-2x256-gru128-mixed-v1"
DRIVER_HZ = 50.0
DRIVER_DT_S = 1.0 / DRIVER_HZ
INDEPENDENT_TRAINING_AUTHORIZATION_VERIFIER_IMPLEMENTED = False


class TrainingPhase(str, Enum):
    BEHAVIORAL_CLONING = "behavioral_cloning"
    DAGGER_RECOVERY = "dagger_recovery"
    RECURRENT_PPO = "recurrent_ppo"
    AUTHORITY_EVALUATION = "authority_evaluation"


@dataclass(frozen=True, slots=True)
class TrainingAuthorizationV1:
    physics_validation_sha256: str | None
    oracle_feasibility_sha256: str | None
    actor_boundary_audit_sha256: str | None
    physics_validated: bool
    oracle_feasible: bool
    actor_boundary_passed: bool

    def __post_init__(self) -> None:
        for name in (
            "physics_validation_sha256", "oracle_feasibility_sha256",
            "actor_boundary_audit_sha256",
        ):
            value = getattr(self, name)
            if value is not None and (
                len(value) != 64 or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256")
        for name in ("physics_validated", "oracle_feasible", "actor_boundary_passed"):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be bool")

    def assert_long_run_allowed(self) -> None:
        blockers: list[str] = []
        if not INDEPENDENT_TRAINING_AUTHORIZATION_VERIFIER_IMPLEMENTED:
            blockers.append(
                "independent materialized training-authorization verifier is not implemented"
            )
        if not self.physics_validated or not self.physics_validation_sha256:
            blockers.append("physics validation artifact is absent or failed")
        if not self.oracle_feasible or not self.oracle_feasibility_sha256:
            blockers.append("authoritative oracle feasibility artifact is absent or failed")
        if not self.actor_boundary_passed or not self.actor_boundary_audit_sha256:
            blockers.append("actor observation-boundary audit is absent or failed")
        if blockers:
            raise PermissionError("faithful-v2 training is blocked: " + "; ".join(blockers))


@dataclass(frozen=True, slots=True)
class DriverObservationLayoutV2:
    """Freeze every variable-length actor input into the run identity."""

    temperature_count: int
    controller_fields: tuple[str, ...]
    preview_distances_m: tuple[float, ...]
    track_surface_sha256: str
    map_preview_generator_sha256: str
    schema: str = "driver-observation-layout-v2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "controller_fields", tuple(self.controller_fields))
        object.__setattr__(self, "preview_distances_m", tuple(self.preview_distances_m))
        if self.temperature_count < 0:
            raise ValueError("temperature_count must be nonnegative")
        if len(set(self.controller_fields)) != len(self.controller_fields):
            raise ValueError("controller field names must be unique")
        protocol_fields = tuple(field.name for field in fields(DriverControllerStatusV1))
        if self.controller_fields != protocol_fields:
            raise ValueError(
                "controller_fields must exactly match DriverControllerStatusV1; "
                "caller-defined actor channels are forbidden"
            )
        if any(not math.isfinite(value) or value < 0 for value in self.preview_distances_m):
            raise ValueError("map preview distances must be finite and nonnegative")
        if any(b <= a for a, b in zip(self.preview_distances_m,
                                      self.preview_distances_m[1:])):
            raise ValueError("map preview distances must be strictly increasing")
        for name in ("track_surface_sha256", "map_preview_generator_sha256"):
            value = getattr(self, name)
            if (not isinstance(value, str) or len(value) != 64
                    or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError(f"{name} must be a lowercase SHA-256")

    @property
    def feature_count(self) -> int:
        # time + GPS + body velocity/accel/angular velocity + four wheel speeds
        # + steering/pedals/gear/rpm/SOC/deployment + temperatures + ECU flags
        # + each map point: distance, xyz offset, left width, right width.
        fixed = 1 + 3 + 3 + 3 + 3 + 4 + 1 + 2 + 1 + 1 + 1 + 1
        return (fixed + self.temperature_count + len(self.controller_fields)
                + 6 * len(self.preview_distances_m))

    @property
    def sha256(self) -> str:
        return content_sha256(self)


_ACTOR_OBSERVATION_FIELDS = frozenset({
    "schema_version", "time_s", "gps_position_m", "velocity_body_mps",
    "acceleration_body_mps2", "angular_velocity_body_rad_s",
    "wheel_speed_rad_s", "steering_angle_rad", "accelerator_pedal",
    "brake_pedal", "gear", "engine_rpm", "battery_soc",
    "deployment_available_w", "temperatures_k", "controller_status",
    "static_map_preview",
})


def _controller_values(status: object, expected: tuple[str, ...]) -> tuple[float, ...]:
    """Read only the protocol-owned status dataclass in its frozen field order."""
    if not isinstance(status, DriverControllerStatusV1):
        raise TypeError("controller_status must use the closed faithful-v2 status schema")
    available = tuple(field.name for field in fields(DriverControllerStatusV1))
    if available != expected:
        raise ValueError(
            f"controller status fields {available!r} do not match layout {expected!r}"
        )
    raw_values = tuple(getattr(status, name) for name in expected)
    values: list[float] = []
    for value in raw_values:
        if value is None:
            values.append(-1.0)  # explicit ECU-state unavailable sentinel
        elif isinstance(value, bool):
            values.append(1.0 if value else 0.0)
        elif isinstance(value, Enum):
            members = tuple(type(value))
            values.append(float(members.index(value)))
        else:
            values.append(float(value))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("controller status contains a non-finite value")
    return tuple(values)


class DriverObservationEncoderV2:
    def __init__(self, layout: DriverObservationLayoutV2):
        self.layout = layout

    def encode(self, observation: DriverObservationV2) -> np.ndarray:
        if not isinstance(observation, DriverObservationV2):
            if isinstance(observation, VehicleTelemetryV2):
                raise PermissionError("truth telemetry may not enter the driver actor")
            raise TypeError("actor input must be DriverObservationV2")
        if len(observation.temperatures_k) != self.layout.temperature_count:
            raise ValueError("temperature vector does not match frozen actor layout")
        if len(observation.static_map_preview) != len(self.layout.preview_distances_m):
            raise ValueError("map preview does not match frozen actor layout")

        values: list[float] = [float(observation.time_s)]
        for vector in (
            observation.gps_position_m,
            observation.velocity_body_mps,
            observation.acceleration_body_mps2,
            observation.angular_velocity_body_rad_s,
            observation.wheel_speed_rad_s,
        ):
            if any(value is None for value in vector):
                raise ValueError("required actor sensor channel is unavailable")
            values.extend(float(value) for value in vector)
        values.extend((
            float(observation.steering_angle_rad),
            float(observation.accelerator_pedal),
            float(observation.brake_pedal),
            float(observation.gear),
            float(observation.engine_rpm),
            float(observation.battery_soc),
            float(observation.deployment_available_w),
        ))
        values.extend(float(value) for value in observation.temperatures_k)
        values.extend(_controller_values(
            observation.controller_status, self.layout.controller_fields
        ))
        for expected_distance, point in zip(
            self.layout.preview_distances_m, observation.static_map_preview
        ):
            if not math.isclose(point.distance_ahead_m, expected_distance,
                                rel_tol=0.0, abs_tol=1e-6):
                raise ValueError("map preview distance does not match frozen layout")
            values.extend((
                float(point.distance_ahead_m),
                *(float(value) for value in point.center_offset_body_m),
                float(point.left_width_m),
                float(point.right_width_m),
            ))
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (self.layout.feature_count,):
            raise AssertionError(
                f"encoder produced {vector.size} values; expected {self.layout.feature_count}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("actor observation contains a non-finite value")
        return vector


@dataclass(frozen=True, slots=True)
class ActorBoundaryAuditV1:
    passed: bool
    observation_schema_sha256: str
    allowed_fields: tuple[str, ...]
    rejected_privileged_fields: tuple[str, ...]
    blockers: tuple[str, ...]

    @property
    def sha256(self) -> str:
        return content_sha256(self)


def audit_actor_boundary(
    layout: DriverObservationLayoutV2,
    *,
    expected_track_surface_sha256: str,
    approved_map_preview_generator_sha256: str,
) -> ActorBoundaryAuditV1:
    declared = frozenset(field.name for field in fields(DriverObservationV2))
    blockers: list[str] = []
    if declared != _ACTOR_OBSERVATION_FIELDS:
        blockers.append("DriverObservationV2 fields changed without boundary review")
    protocol_controller_fields = tuple(
        field.name for field in fields(DriverControllerStatusV1)
    )
    if layout.controller_fields != protocol_controller_fields:
        blockers.append("controller status layout differs from the closed protocol schema")
    if layout.track_surface_sha256 != expected_track_surface_sha256:
        blockers.append("static map preview is not bound to the run track surface")
    if layout.map_preview_generator_sha256 != approved_map_preview_generator_sha256:
        blockers.append("static map preview producer is not the approved frozen generator")
    telemetry_fields = frozenset(field.name for field in fields(VehicleTelemetryV2))
    privileged = telemetry_fields - declared
    forbidden_labels = {
        "hidden_friction", "tyre_force", "wheel_force", "future_state",
        "constraint_margin", "oracle_action", "simulator_truth",
    }
    leaked_labels = forbidden_labels.intersection(layout.controller_fields)
    if leaked_labels:
        blockers.append("privileged controller labels: " + ", ".join(sorted(leaked_labels)))
    return ActorBoundaryAuditV1(
        passed=not blockers,
        observation_schema_sha256=content_sha256({
            "observation_fields": sorted(declared), "layout": layout,
            "expected_track_surface_sha256": expected_track_surface_sha256,
            "approved_map_preview_generator_sha256": (
                approved_map_preview_generator_sha256
            ),
        }),
        allowed_fields=tuple(sorted(declared)),
        rejected_privileged_fields=tuple(sorted(privileged)),
        blockers=tuple(blockers),
    )


@dataclass(frozen=True, slots=True)
class PolicyArchitectureV1:
    actor_observation_dim: int
    privileged_critic_dim: int = 0
    feature_units: tuple[int, int] = (256, 256)
    gru_units: int = 128
    driver_hz: float = DRIVER_HZ
    architecture_id: str = POLICY_ARCHITECTURE

    def __post_init__(self) -> None:
        if self.actor_observation_dim <= 0 or self.privileged_critic_dim < 0:
            raise ValueError("policy dimensions must be valid")
        if self.feature_units != (256, 256) or self.gru_units != 128:
            raise ValueError("faithful-v2 architecture is frozen at 2x256 + GRU128")
        if self.driver_hz != DRIVER_HZ:
            raise ValueError("faithful-v2 driver loop is frozen at 50 Hz")

    @property
    def sha256(self) -> str:
        return content_sha256(self)


@dataclass(slots=True)
class MixedPolicyStep:
    steering: Tensor
    accelerator: Tensor
    brake: Tensor
    shift_index: Tensor
    steering_latent: Tensor
    log_probability: Tensor
    entropy: Tensor
    value: Tensor | None
    actor_hidden: Tensor
    critic_hidden: Tensor | None


class FaithfulRecurrentActorCritic(nn.Module):
    """2x256 actor encoder, GRU128 memory, and mixed driver action heads."""

    def __init__(self, architecture: PolicyArchitectureV1):
        super().__init__()
        self.architecture = architecture
        dim = architecture.actor_observation_dim
        self.actor_features = nn.Sequential(
            nn.Linear(dim, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
        )
        self.actor_gru = nn.GRU(256, 128, batch_first=True)
        self.steering_mean = nn.Linear(128, 1)
        self.steering_log_std = nn.Parameter(torch.full((1,), -1.0))
        self.pedal_beta_parameters = nn.Linear(128, 4)
        self.shift_logits = nn.Linear(128, 3)

        if architecture.privileged_critic_dim:
            self.critic_features = nn.Sequential(
                nn.Linear(architecture.privileged_critic_dim, 256), nn.SiLU(),
                nn.Linear(256, 256), nn.SiLU(),
            )
            self.critic_gru = nn.GRU(256, 128, batch_first=True)
        else:
            self.critic_features = None
            self.critic_gru = None
        self.value_head = nn.Linear(128, 1)

    def initial_actor_hidden(self, batch_size: int, *, device=None) -> Tensor:
        return torch.zeros(1, batch_size, 128, device=device)

    def initial_critic_hidden(self, batch_size: int, *, device=None) -> Tensor | None:
        if self.critic_gru is None:
            return None
        return torch.zeros(1, batch_size, 128, device=device)

    @staticmethod
    def _sequence(value: Tensor) -> tuple[Tensor, bool]:
        if value.ndim == 2:
            return value[:, None, :], True
        if value.ndim != 3:
            raise ValueError("policy inputs must be [batch,features] or [batch,time,features]")
        return value, False

    def actor_latent(self, actor_observation: Tensor,
                     hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        sequence, _ = self._sequence(actor_observation)
        if sequence.shape[-1] != self.architecture.actor_observation_dim:
            raise ValueError("actor observation dimension mismatch")
        features = self.actor_features(sequence)
        if hidden is None:
            hidden = self.initial_actor_hidden(sequence.shape[0], device=sequence.device)
        return self.actor_gru(features, hidden)

    def critic_value(self, actor_latent: Tensor, privileged_observation: Tensor | None,
                     hidden: Tensor | None = None) -> tuple[Tensor | None, Tensor | None]:
        if self.critic_gru is None:
            return self.value_head(actor_latent).squeeze(-1), None
        if privileged_observation is None:
            return None, hidden
        sequence, _ = self._sequence(privileged_observation)
        if sequence.shape[-1] != self.architecture.privileged_critic_dim:
            raise ValueError("privileged critic dimension mismatch")
        features = self.critic_features(sequence)
        if hidden is None:
            hidden = self.initial_critic_hidden(sequence.shape[0], device=sequence.device)
        latent, next_hidden = self.critic_gru(features, hidden)
        return self.value_head(latent).squeeze(-1), next_hidden

    def _distributions(self, latent: Tensor) -> tuple[Normal, Beta, Beta, Categorical]:
        mean = self.steering_mean(latent).squeeze(-1)
        log_std = self.steering_log_std.clamp(-5.0, 1.0).expand_as(mean)
        steering = Normal(mean, log_std.exp())
        pedal = F.softplus(self.pedal_beta_parameters(latent)) + 1.0
        accelerator = Beta(pedal[..., 0], pedal[..., 1])
        brake = Beta(pedal[..., 2], pedal[..., 3])
        shift = Categorical(logits=self.shift_logits(latent))
        return steering, accelerator, brake, shift

    def step(self, actor_observation: Tensor, *, actor_hidden: Tensor | None = None,
             privileged_observation: Tensor | None = None,
             critic_hidden: Tensor | None = None,
             deterministic: bool = False) -> MixedPolicyStep:
        latent, next_actor_hidden = self.actor_latent(actor_observation, actor_hidden)
        steering_dist, accelerator_dist, brake_dist, shift_dist = self._distributions(latent)
        if deterministic:
            steering_latent = steering_dist.mean
            accelerator = accelerator_dist.mean
            brake = brake_dist.mean
            shift = shift_dist.probs.argmax(dim=-1)
        else:
            steering_latent = steering_dist.rsample()
            accelerator = accelerator_dist.rsample()
            brake = brake_dist.rsample()
            shift = shift_dist.sample()
        steering = torch.tanh(steering_latent)
        correction = torch.log(1.0 - steering.square() + 1e-6)
        log_probability = (
            steering_dist.log_prob(steering_latent) - correction
            + accelerator_dist.log_prob(accelerator)
            + brake_dist.log_prob(brake)
            + shift_dist.log_prob(shift)
        )
        entropy = (
            steering_dist.entropy() + accelerator_dist.entropy()
            + brake_dist.entropy() + shift_dist.entropy()
        )
        value, next_critic_hidden = self.critic_value(
            latent, privileged_observation, critic_hidden
        )
        return MixedPolicyStep(
            steering, accelerator, brake, shift, steering_latent,
            log_probability, entropy, value,
            next_actor_hidden, next_critic_hidden,
        )

    def evaluate_actions(
        self,
        actor_observation: Tensor,
        steering_latent: Tensor,
        accelerator: Tensor,
        brake: Tensor,
        shift_index: Tensor,
        *,
        actor_hidden: Tensor | None = None,
        privileged_observation: Tensor | None = None,
        critic_hidden: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor, Tensor | None]:
        latent, next_actor_hidden = self.actor_latent(actor_observation, actor_hidden)
        steering_dist, accelerator_dist, brake_dist, shift_dist = self._distributions(latent)
        steering = torch.tanh(steering_latent)
        log_probability = (
            steering_dist.log_prob(steering_latent)
            - torch.log(1.0 - steering.square() + 1e-6)
            + accelerator_dist.log_prob(accelerator.clamp(1e-6, 1 - 1e-6))
            + brake_dist.log_prob(brake.clamp(1e-6, 1 - 1e-6))
            + shift_dist.log_prob(shift_index)
        )
        entropy = (
            steering_dist.entropy() + accelerator_dist.entropy()
            + brake_dist.entropy() + shift_dist.entropy()
        )
        value, next_critic_hidden = self.critic_value(
            latent, privileged_observation, critic_hidden
        )
        return log_probability, entropy, value, next_actor_hidden, next_critic_hidden


class DriverPolicyRuntime50Hz:
    """Inference wrapper that cannot accept simulator truth or extra controls."""

    def __init__(self, model: FaithfulRecurrentActorCritic,
                 encoder: DriverObservationEncoderV2, *, device: str = "cpu"):
        if model.architecture.actor_observation_dim != encoder.layout.feature_count:
            raise ValueError("model and actor observation layout do not match")
        self.model = model.to(device).eval()
        self.encoder = encoder
        self.device = torch.device(device)
        self.hidden: Tensor | None = None
        self.last_time_s: float | None = None

    def reset(self) -> None:
        self.hidden = None
        self.last_time_s = None

    @torch.no_grad()
    def act(self, observation: DriverObservationV2) -> DriverActionV2:
        if self.last_time_s is not None:
            interval = observation.time_s - self.last_time_s
            if not math.isclose(interval, DRIVER_DT_S, rel_tol=0.0, abs_tol=5e-4):
                raise ValueError(
                    f"driver observation interval {interval:.6f}s violates 50 Hz loop"
                )
        vector = self.encoder.encode(observation)
        tensor = torch.from_numpy(vector).to(self.device)[None, :]
        output = self.model.step(tensor, actor_hidden=self.hidden, deterministic=True)
        self.hidden = output.actor_hidden
        self.last_time_s = observation.time_s
        shift_index = int(output.shift_index[0, -1].item())
        shift = (ShiftRequest.DOWNSHIFT, ShiftRequest.HOLD, ShiftRequest.UPSHIFT)[shift_index]
        return DriverActionV2(
            steering=float(output.steering[0, -1].item()),
            accelerator=float(output.accelerator[0, -1].item()),
            brake=float(output.brake[0, -1].item()),
            shift_request=shift,
            boost_request=None,
            drs_request=None,
        )


def behavioral_cloning_loss(
    model: FaithfulRecurrentActorCritic,
    actor_observation: Tensor,
    target_steering: Tensor,
    target_accelerator: Tensor,
    target_brake: Tensor,
    target_shift_index: Tensor,
) -> Tensor:
    """Supervised oracle/MPC imitation objective used before DAgger/PPO."""
    latent, _ = model.actor_latent(actor_observation)
    steering, accelerator, brake, shift = model._distributions(latent)
    steering_target = target_steering.clamp(-0.999999, 0.999999)
    steering_latent = torch.atanh(steering_target)
    return -(
        steering.log_prob(steering_latent)
        - torch.log(1.0 - steering_target.square() + 1e-6)
        + accelerator.log_prob(target_accelerator.clamp(1e-6, 1 - 1e-6))
        + brake.log_prob(target_brake.clamp(1e-6, 1 - 1e-6))
        + shift.log_prob(target_shift_index)
    ).mean()


__all__ = [
    "ActorBoundaryAuditV1", "DRIVER_DT_S", "DRIVER_HZ",
    "DriverObservationEncoderV2", "DriverObservationLayoutV2",
    "DriverPolicyRuntime50Hz", "FaithfulRecurrentActorCritic",
    "MixedPolicyStep", "POLICY_ARCHITECTURE", "PolicyArchitectureV1",
    "TrainingAuthorizationV1", "TrainingPhase", "audit_actor_boundary",
    "behavioral_cloning_loss",
]
