"""Independent flying-lap continuity protocol and training stage gates."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .identity import TrainingStage


RECORD_TARGET_S = 319.546
OFFICIAL_LAP_LENGTH_M = 20_832.0
# The current trace contains only center pose, speed and a caller-provided
# odometer. It does not yet carry the surveyed-surface projection, four-tyre
# legal-contact footprint or independently derived timing-plane crossings
# required to call a lap legal. Kinematic checks still run, but certification
# remains closed until that authority verifier is implemented.
AUTHORITY_SURVEY_LEGALITY_VERIFIER_IMPLEMENTED = False


class LapEventKind(str, Enum):
    WARMUP_START = "warmup_start"
    START_LINE_CROSSING = "start_line_crossing"
    FINISH_LINE_CROSSING = "finish_line_crossing"
    RESET = "reset"
    TELEPORT = "teleport"
    COLLISION = "collision"
    ILLEGAL_CONTACT = "illegal_contact"


@dataclass(frozen=True, slots=True)
class ProtocolSample:
    time_s: float
    position_m: tuple[float, float, float]
    speed_mps: float
    cumulative_distance_m: float
    fuel_kg: float
    soc_fraction: float
    state_epoch: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "position_m", tuple(self.position_m))
        if len(self.position_m) != 3:
            raise ValueError("position_m must contain exactly x, y, and z")
        values = (self.time_s, *self.position_m, self.speed_mps,
                  self.cumulative_distance_m, self.fuel_kg, self.soc_fraction)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("protocol samples must be finite")
        if min(self.time_s, self.speed_mps, self.cumulative_distance_m,
               self.fuel_kg) < 0:
            raise ValueError("sample time, speed, distance and fuel must be nonnegative")
        if not 0 <= self.soc_fraction <= 1:
            raise ValueError("SOC must be in [0, 1]")
        if type(self.state_epoch) is not int:
            raise TypeError("state_epoch must be an actual integer")
        if self.state_epoch < 0:
            raise ValueError("state_epoch must be nonnegative")


@dataclass(frozen=True, slots=True)
class LapEvent:
    kind: LapEventKind
    time_s: float
    cumulative_distance_m: float
    timing_plane: str | None = None
    crossing_direction: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", LapEventKind(self.kind))
        if not math.isfinite(self.time_s) or not math.isfinite(self.cumulative_distance_m):
            raise ValueError("lap events must be finite")
        if self.time_s < 0 or self.cumulative_distance_m < 0:
            raise ValueError("lap event time and distance must be nonnegative")
        if type(self.crossing_direction) is not int:
            raise TypeError("crossing_direction must be an actual integer")
        if self.kind in (LapEventKind.START_LINE_CROSSING,
                          LapEventKind.FINISH_LINE_CROSSING):
            if self.timing_plane != "t13_start_finish" or self.crossing_direction != 1:
                raise ValueError("timing events require a forward T13 timing-plane crossing")


@dataclass(frozen=True, slots=True)
class FlyingLapTraceV1:
    samples: tuple[ProtocolSample, ...]
    events: tuple[LapEvent, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "samples", tuple(self.samples))
        object.__setattr__(self, "events", tuple(self.events))
        if len(self.samples) < 2:
            raise ValueError("a flying-lap trace requires at least two samples")
        if not self.events:
            raise ValueError("a flying-lap trace requires events")
        if any(b.time_s <= a.time_s for a, b in zip(self.samples, self.samples[1:])):
            raise ValueError("sample time must be strictly increasing")
        if any(b.time_s < a.time_s for a, b in zip(self.events, self.events[1:])):
            raise ValueError("event time must be monotonic")


@dataclass(frozen=True, slots=True)
class FlyingLapDecision:
    legal: bool
    lap_time_s: float | None
    beats_record: bool
    blockers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FlyingLapProtocolV1:
    target_lap_s: float = RECORD_TARGET_S
    official_lap_length_m: float = OFFICIAL_LAP_LENGTH_M
    distance_tolerance_m: float = 0.1
    minimum_warmup_time_s: float = 20.0
    minimum_warmup_distance_m: float = 1_000.0
    minimum_sample_interval_s: float = 0.0005
    maximum_sample_interval_s: float = 0.1
    kinematic_relative_tolerance: float = 0.10
    kinematic_absolute_tolerance_m: float = 0.25
    fuel_increase_tolerance_kg: float = 0.01

    def __post_init__(self) -> None:
        if abs(self.target_lap_s - RECORD_TARGET_S) > 1e-12:
            raise ValueError("faithful-v2 benchmark is fixed at 319.546 seconds")
        if abs(self.official_lap_length_m - OFFICIAL_LAP_LENGTH_M) > 1e-9:
            raise ValueError("faithful-v2 lap is fixed at 20.832 km")
        if not 0 < self.distance_tolerance_m <= 0.1:
            raise ValueError("distance tolerance must be in (0, 0.1] metres")
        if self.minimum_warmup_time_s < 20.0 or self.minimum_warmup_distance_m < 1_000.0:
            raise ValueError("warmup contract may be made stricter but not weaker")
        if not 0.0005 <= self.minimum_sample_interval_s <= 0.1:
            raise ValueError(
                "minimum sample interval must be in [0.0005, 0.1] seconds"
            )
        if not (self.minimum_sample_interval_s
                <= self.maximum_sample_interval_s <= 0.1):
            raise ValueError(
                "maximum sample interval must not exceed 0.1 seconds"
            )
        if not 0 <= self.kinematic_relative_tolerance <= 0.10:
            raise ValueError("kinematic relative tolerance must not exceed 10%")
        if not 0 <= self.kinematic_absolute_tolerance_m <= 0.25:
            raise ValueError(
                "kinematic absolute tolerance must not exceed 0.25 metres"
            )
        if not 0 <= self.fuel_increase_tolerance_kg <= 0.01:
            raise ValueError("fuel tolerance must be in [0, 0.01] kg")

    def _distance_at(self, trace: FlyingLapTraceV1,
                     time_s: float) -> float | None:
        """Interpolate the independently sampled odometer at an event time."""
        samples = trace.samples
        if time_s < samples[0].time_s or time_s > samples[-1].time_s:
            return None
        for previous, current in zip(samples, samples[1:]):
            if time_s == previous.time_s:
                return previous.cumulative_distance_m
            if previous.time_s < time_s <= current.time_s:
                if time_s == current.time_s:
                    return current.cumulative_distance_m
                fraction = ((time_s - previous.time_s)
                            / (current.time_s - previous.time_s))
                return (previous.cumulative_distance_m
                        + fraction * (current.cumulative_distance_m
                                      - previous.cumulative_distance_m))
        if time_s == samples[-1].time_s:
            return samples[-1].cumulative_distance_m
        return None

    def _window_samples(self, trace: FlyingLapTraceV1, start_s: float,
                        finish_s: float) -> tuple[ProtocolSample, ...]:
        """Return samples bracketing the complete certified time window."""
        before = [index for index, sample in enumerate(trace.samples)
                  if sample.time_s <= start_s]
        after = [index for index, sample in enumerate(trace.samples)
                 if sample.time_s >= finish_s]
        if not before or not after:
            return ()
        start_index = before[-1]
        finish_index = after[0]
        if finish_index <= start_index:
            return ()
        return trace.samples[start_index:finish_index + 1]

    def _kinematically_consistent(self, left_m: float,
                                  right_m: float) -> bool:
        tolerance = (self.kinematic_absolute_tolerance_m
                     + self.kinematic_relative_tolerance
                     * max(abs(left_m), abs(right_m)))
        return abs(left_m - right_m) <= tolerance

    def evaluate(self, trace: FlyingLapTraceV1) -> FlyingLapDecision:
        blockers: list[str] = []
        warmups = [event for event in trace.events
                   if event.kind is LapEventKind.WARMUP_START]
        starts = [event for event in trace.events
                  if event.kind is LapEventKind.START_LINE_CROSSING]
        if len(warmups) != 1:
            blockers.append("exactly one warmup_start event is required")
        if len(starts) != 1:
            blockers.append("exactly one start-line crossing is required")
        if blockers:
            return FlyingLapDecision(False, None, False, tuple(blockers))

        warmup = warmups[0]
        start = starts[0]
        finishes = [event for event in trace.events
                    if event.kind is LapEventKind.FINISH_LINE_CROSSING
                    and event.time_s > start.time_s]
        if len(finishes) != 1:
            blockers.append("exactly one post-start finish-line crossing is required")
            return FlyingLapDecision(False, None, False, tuple(blockers))
        finish = finishes[0]

        warmup_time = start.time_s - warmup.time_s
        warmup_distance = start.cumulative_distance_m - warmup.cumulative_distance_m
        if warmup_time < self.minimum_warmup_time_s:
            blockers.append("warmup/outlap duration is too short")
        if warmup_distance < self.minimum_warmup_distance_m:
            blockers.append("warmup/outlap distance is too short")
        lap_distance = finish.cumulative_distance_m - start.cumulative_distance_m
        if abs(lap_distance - self.official_lap_length_m) > self.distance_tolerance_m:
            blockers.append("timed distance does not match the official 20.832 km lap")

        # Timing-plane event odometers are assertions, not measurements. Bind
        # them to the independently sampled odometer so a caller cannot simply
        # type 20,832 into the finish event while the car remains stationary.
        for label, event in (("warmup", warmup), ("start", start),
                             ("finish", finish)):
            sampled_distance = self._distance_at(trace, event.time_s)
            if sampled_distance is None:
                blockers.append(f"{label} event is outside sampled telemetry")
            elif abs(sampled_distance - event.cumulative_distance_m) > self.distance_tolerance_m:
                blockers.append(
                    f"{label} event distance disagrees with sampled odometer"
                )

        forbidden = {
            LapEventKind.RESET, LapEventKind.TELEPORT, LapEventKind.COLLISION,
            LapEventKind.ILLEGAL_CONTACT,
        }
        for event in trace.events:
            if warmup.time_s <= event.time_s <= finish.time_s and event.kind in forbidden:
                blockers.append(f"forbidden event during continuous run: {event.kind.value}")

        active = self._window_samples(trace, warmup.time_s, finish.time_s)
        if not active:
            blockers.append("samples do not cover the complete warmup and timed lap")
        if active:
            epoch = active[0].state_epoch
            if any(sample.state_epoch != epoch for sample in active):
                blockers.append("state epoch changed, indicating reset or state replacement")
            total_positive_fuel_delta_kg = 0.0
            for previous, current in zip(active, active[1:]):
                if current.cumulative_distance_m < previous.cumulative_distance_m:
                    blockers.append("cumulative distance moved backward")
                    break
                dt = current.time_s - previous.time_s
                if dt < self.minimum_sample_interval_s - 1e-12:
                    blockers.append("sample cadence is faster than the certified bound")
                    break
                if dt > self.maximum_sample_interval_s + 1e-12:
                    blockers.append("sample gap exceeds the certified cadence bound")
                    break
                displacement = math.dist(previous.position_m, current.position_m)
                integrated_speed = 0.5 * (
                    previous.speed_mps + current.speed_mps
                ) * dt
                odometer_delta = (current.cumulative_distance_m
                                  - previous.cumulative_distance_m)
                if not self._kinematically_consistent(
                        displacement, integrated_speed):
                    blockers.append(
                        "pose displacement and integrated speed disagree "
                        "(stationary trace or unreported teleport)"
                    )
                    break
                if not self._kinematically_consistent(
                        odometer_delta, integrated_speed):
                    blockers.append(
                        "odometer delta and integrated speed disagree"
                    )
                    break
                if not self._kinematically_consistent(
                        displacement, odometer_delta):
                    blockers.append(
                        "pose displacement and odometer delta disagree"
                    )
                    break
                total_positive_fuel_delta_kg += max(
                    0.0, current.fuel_kg - previous.fuel_kg
                )
            if total_positive_fuel_delta_kg > self.fuel_increase_tolerance_kg:
                blockers.append(
                    "total fuel increase during the continuous run exceeds tolerance"
                )

        lap_time = finish.time_s - start.time_s
        if lap_time <= 0:
            blockers.append("lap time must be positive")
        if not AUTHORITY_SURVEY_LEGALITY_VERIFIER_IMPLEMENTED:
            blockers.append(
                "surveyed timing-plane, boundary and tyre-contact authority verifier is not implemented"
            )
        legal = not blockers
        return FlyingLapDecision(
            legal, lap_time if legal else None,
            bool(legal and lap_time < self.target_lap_s), tuple(blockers),
        )


@dataclass(frozen=True, slots=True)
class StageMetrics:
    stage: TrainingStage
    recovered_sectors: int = 0
    sector_count: int = 16
    valid_perturbed_chains: int = 0
    perturbed_chain_trials: int = 0
    consecutive_legal_full_laps: int = 0
    best_legal_lap_s: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", TrainingStage(self.stage))
        counts = (
            self.recovered_sectors, self.sector_count,
            self.valid_perturbed_chains, self.perturbed_chain_trials,
            self.consecutive_legal_full_laps,
        )
        if any(type(value) is not int for value in counts):
            raise TypeError("stage counts must be actual integers")
        if any(value < 0 for value in counts):
            raise ValueError("stage counts must be nonnegative")
        if self.recovered_sectors > self.sector_count:
            raise ValueError("recovered sectors cannot exceed sector_count")
        if self.valid_perturbed_chains > self.perturbed_chain_trials:
            raise ValueError("valid chains cannot exceed chain trials")
        if self.best_legal_lap_s is not None:
            if (type(self.best_legal_lap_s) not in (int, float)
                    or isinstance(self.best_legal_lap_s, bool)):
                raise TypeError("best legal lap must be an actual number, not bool")
            if (not math.isfinite(self.best_legal_lap_s)
                    or self.best_legal_lap_s <= 0):
                raise ValueError("best legal lap must be finite and positive")


@dataclass(frozen=True, slots=True)
class StageGateDecision:
    passed: bool
    disposition: str
    blockers: tuple[str, ...]


def evaluate_stage_gate(metrics: StageMetrics,
                        record_target_s: float = RECORD_TARGET_S) -> StageGateDecision:
    if abs(record_target_s - RECORD_TARGET_S) > 1e-12:
        raise ValueError("faithful-v2 frontier benchmark is fixed at 319.546 seconds")
    blockers: list[str] = []
    stage = metrics.stage
    if stage is TrainingStage.FOUNDATION:
        if metrics.sector_count != 16 or metrics.recovered_sectors < 16:
            blockers.append("all 16 sectors must recover from bounded perturbations")
        disposition = "advance_to_flow"
    elif stage is TrainingStage.FLOW:
        if metrics.perturbed_chain_trials < 100:
            blockers.append("at least 100 perturbed chain trials are required")
        elif metrics.valid_perturbed_chains / metrics.perturbed_chain_trials < 0.95:
            blockers.append("at least 95% of perturbed chains must remain legal")
        disposition = "advance_to_finish"
    elif stage is TrainingStage.FINISH:
        if metrics.consecutive_legal_full_laps < 5:
            blockers.append("five consecutive continuous legal laps are required")
        disposition = "advance_to_fast"
    elif stage is TrainingStage.FAST:
        if metrics.best_legal_lap_s is None or metrics.best_legal_lap_s >= 330.0:
            blockers.append("a legal nominal lap below 330 seconds is required")
        disposition = "advance_to_frontier"
    else:
        if metrics.best_legal_lap_s is None or metrics.best_legal_lap_s >= record_target_s:
            blockers.append(f"a legal candidate below {record_target_s:.3f} seconds is required")
        disposition = "transfer_to_independent_certification"
    return StageGateDecision(not blockers, disposition if not blockers else "hold",
                             tuple(blockers))
