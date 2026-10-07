"""Immutable lap plans and a fail-closed optional CasADi/IPOPT boundary."""
from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
import math

from ._canonical import canonicalize, require_sha256, sha256_object


LAP_PLAN_SCHEMA = "faithful-lap-plan-v1"


@dataclass(frozen=True, slots=True)
class LapPlanSample:
    time_s: float
    progress_m: float
    position_m: tuple[float, float, float]
    speed_mps: float
    steering: float
    accelerator: float
    brake: float
    gear: int
    mgu_power_w: float
    regen_power_w: float
    soc_fraction: float
    drs_fraction: float
    pitch_link_fraction: float
    constraint_margin: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", tuple(self.position_m))
        if len(self.position_m) != 3:
            raise ValueError("position_m must contain exactly x, y, and z")
        numeric = (
            self.time_s, self.progress_m, *self.position_m, self.speed_mps,
            self.steering, self.accelerator, self.brake, self.mgu_power_w,
            self.regen_power_w, self.soc_fraction, self.drs_fraction,
            self.pitch_link_fraction, self.constraint_margin,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("lap-plan samples must contain only finite values")
        if self.time_s < 0 or self.progress_m < 0 or self.speed_mps < 0:
            raise ValueError("time, progress and speed must be nonnegative")
        if not -1.0 <= self.steering <= 1.0:
            raise ValueError("steering must be in [-1, 1]")
        for name in ("accelerator", "brake", "soc_fraction", "drs_fraction",
                     "pitch_link_fraction"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if not 0 <= self.gear <= 7:
            raise ValueError("gear must be neutral (0) or one of seven forward gears")
        if self.mgu_power_w < 0 or self.regen_power_w < 0:
            raise ValueError("deployment and regeneration power are unsigned magnitudes")
        if self.mgu_power_w > 0 and self.regen_power_w > 0:
            raise ValueError("MGU deployment and regeneration cannot occur simultaneously")


@dataclass(frozen=True, slots=True)
class LapPlanV1:
    run_identity_sha256: str
    scenario_sha256: str
    physics_identity_sha256: str
    solver_name: str
    solver_version: str
    samples: tuple[LapPlanSample, ...]
    lap_length_m: float = 20_832.0
    distance_tolerance_m: float = 0.1
    schema: str = LAP_PLAN_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        if self.schema != LAP_PLAN_SCHEMA:
            raise ValueError(f"unsupported lap plan schema: {self.schema}")
        for name in ("run_identity_sha256", "scenario_sha256",
                     "physics_identity_sha256"):
            require_sha256(getattr(self, name), name)
        if not self.solver_name or not self.solver_version:
            raise ValueError("solver name and version are required")
        if not math.isfinite(self.lap_length_m) or self.lap_length_m <= 0:
            raise ValueError("lap_length_m must be finite and positive")
        if (not math.isfinite(self.distance_tolerance_m)
                or not 0 < self.distance_tolerance_m <= 0.1):
            raise ValueError("distance_tolerance_m must be in (0, 0.1]")
        if len(self.samples) < 2:
            raise ValueError("lap plan requires at least two samples")
        for previous, current in zip(self.samples, self.samples[1:]):
            if current.time_s <= previous.time_s:
                raise ValueError("lap plan time must be strictly increasing")
            if current.progress_m < previous.progress_m:
                raise ValueError("lap plan progress must be monotonic")
        planned_distance = self.samples[-1].progress_m - self.samples[0].progress_m
        if abs(planned_distance - self.lap_length_m) > self.distance_tolerance_m:
            raise ValueError("lap plan does not cover the complete official lap")

    @property
    def predicted_lap_time_s(self) -> float:
        return self.samples[-1].time_s - self.samples[0].time_s

    @property
    def sha256(self) -> str:
        return sha256_object(self)

    def to_dict(self) -> dict:
        return canonicalize(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "LapPlanV1":
        data = dict(payload)
        data["samples"] = tuple(LapPlanSample(**sample) for sample in data["samples"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class OracleRuntimeStatus:
    casadi_available: bool
    ipopt_declared: bool
    missing_evidence: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "missing_evidence", tuple(self.missing_evidence))

    @property
    def ready(self) -> bool:
        return self.casadi_available and self.ipopt_declared and not self.missing_evidence


class OraclePrerequisiteError(RuntimeError):
    pass


def oracle_runtime_status(required_evidence: tuple[str, ...],
                          available_evidence: tuple[str, ...]) -> OracleRuntimeStatus:
    """Inspect optional solver availability without importing the simulator.

    ``ipopt_declared`` is conservative: an installed CasADi package is not
    enough. The caller must explicitly record ``casadi-ipopt`` in its frozen
    dependency evidence after a real solver smoke test.
    """
    available = set(available_evidence)
    missing = tuple(sorted(set(required_evidence) - available))
    casadi_available = find_spec("casadi") is not None
    ipopt_declared = "casadi-ipopt" in available
    return OracleRuntimeStatus(casadi_available, ipopt_declared, missing)


def require_oracle_prerequisites(required_evidence: tuple[str, ...],
                                 available_evidence: tuple[str, ...]) -> None:
    status = oracle_runtime_status(required_evidence, available_evidence)
    blockers: list[str] = []
    if not status.casadi_available:
        blockers.append("CasADi dependency is not installed")
    if not status.ipopt_declared:
        blockers.append("IPOPT has not passed and entered dependency evidence")
    if status.missing_evidence:
        blockers.append("missing physical evidence: " + ", ".join(status.missing_evidence))
    if blockers:
        raise OraclePrerequisiteError("; ".join(blockers))
