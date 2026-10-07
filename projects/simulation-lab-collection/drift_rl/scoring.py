"""
Drift scoring engine, trick detection, and combo multiplier chaining.
Recognizes Scandinavian flicks, Manji transitions, backward entries,
wall taps, clipping zone hits, and 360 entries.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
from drift_rl.dynamics import VehicleState
from drift_rl.tracks import Track


@dataclass
class TrickEvent:
    """Represents a successfully recognized trick."""
    name: str                   # Trick name (e.g. 'MANJI TRANSITION', 'BACKWARD ENTRY')
    points: float               # Base points awarded
    multiplier_bonus: float     # Multiplier increment (e.g. +0.5x)
    timestamp: float            # Time in simulation seconds


@dataclass
class DriftScoreState:
    """Snapshot of current scoring state for telemetry and HUD."""
    total_score: float = 0.0
    active_combo_score: float = 0.0
    multiplier: float = 1.0
    is_drifting: bool = False
    drift_duration: float = 0.0
    grace_timer: float = 0.0
    manji_chains: int = 0
    recent_tricks: List[TrickEvent] = field(default_factory=list)


class DriftScorer:
    """
    Evaluates vehicle state each step, awards continuous drift points,
    detects arcade stunt tricks, and manages combo multipliers.
    """

    MIN_DRIFT_ANGLE_DEG: float = 14.0       # Minimum slip angle to count as drifting
    MIN_DRIFT_SPEED: float = 3.5            # Minimum speed (m/s) for valid drift
    MAX_MULTIPLIER: float = 10.0            # Cap on combo multiplier
    GRACE_PERIOD: float = 1.2               # Grace time (s) during drift transitions
    SPINOUT_ANGLE_DEG: float = 105.0        # Extreme angle causing spinout if speed is low

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_banked_score: float = 0.0
        self.active_combo_score: float = 0.0
        self.multiplier: float = 1.0
        self.is_drifting: bool = False
        self.drift_duration: float = 0.0
        self.grace_remaining: float = 0.0
        self.manji_chains: int = 0
        self.recent_tricks: List[TrickEvent] = []

        # Trick detection internal history
        self.last_drift_side: int = 0         # -1 for left slide, +1 for right slide
        self.side_duration: float = 0.0       # time spent sliding on current side
        self.steer_history: List[Tuple[float, float]] = [] # [(time, steer_angle)]
        self.sim_time: float = 0.0
        self.in_backward_entry: bool = False
        self.backward_entry_start_time: float = 0.0
        self.yaw_accumulator: float = 0.0     # for 360 detection
        self.last_yaw: Optional[float] = None

    def update(
        self,
        state: VehicleState,
        track: Track,
        car_corners: np.ndarray,
        dt: float = 0.02,
    ) -> Tuple[float, List[TrickEvent], bool]:
        """
        Update score and check for tricks.

        Returns:
            step_reward: reward earned during this step
            new_tricks: list of TrickEvent triggered this step
            spinout_detected: True if the car spun out / lost control
        """
        self.sim_time += dt
        new_tricks: List[TrickEvent] = []
        spinout_detected = False

        slip_deg = abs(state.slip_angle_deg)
        speed = state.speed
        steer = state.steer

        # Update steer history (keep last 0.8s)
        self.steer_history.append((self.sim_time, steer))
        while self.steer_history and self.sim_time - self.steer_history[0][0] > 0.8:
            self.steer_history.pop(0)

        # Track continuous yaw rotation for 360 entry detection
        if self.last_yaw is not None:
            dyaw = (state.yaw - self.last_yaw + np.pi) % (2 * np.pi) - np.pi
            self.yaw_accumulator += dyaw
            if abs(self.yaw_accumulator) >= 2 * np.pi * 0.90:
                if speed > 6.0 and slip_deg > 20.0:
                    trick = TrickEvent("360 DRIFT ENTRY", 4000.0, 1.0, self.sim_time)
                    new_tricks.append(trick)
                    self._apply_trick(trick)
                self.yaw_accumulator = 0.0
        self.last_yaw = state.yaw

        # 1. Check for Spinout / Loss of Control
        # If slip angle is massive and speed collapsed or spinning uncontrollably
        if slip_deg > self.SPINOUT_ANGLE_DEG and speed < 2.5:
            spinout_detected = True
            self._break_combo(penalty_factor=0.35)
            return -20.0, new_tricks, spinout_detected

        # 2. Check Drift Condition
        currently_sliding = (slip_deg >= self.MIN_DRIFT_ANGLE_DEG) and (speed >= self.MIN_DRIFT_SPEED)

        if currently_sliding:
            current_side = -1 if state.slip_angle < 0 else 1

            if not self.is_drifting:
                # Initiate Drift!
                self.is_drifting = True
                self.last_drift_side = current_side
                self.side_duration = 0.0

                # Check Scandinavian Flick / Feint Entry
                if self._check_scandinavian_flick(current_side):
                    trick = TrickEvent("SCANDINAVIAN FLICK", 1500.0, 0.5, self.sim_time)
                    new_tricks.append(trick)
                    self._apply_trick(trick)
            else:
                # Check for Manji transition (switching sides during active combo)
                if current_side != self.last_drift_side and self.side_duration > 0.4:
                    self.manji_chains += 1
                    trick = TrickEvent(
                        f"MANJI TRANSITION x{self.manji_chains}",
                        1000.0 * min(self.manji_chains, 5),
                        0.4,
                        self.sim_time,
                    )
                    new_tricks.append(trick)
                    self._apply_trick(trick)
                    self.last_drift_side = current_side
                    self.side_duration = 0.0
                elif current_side == self.last_drift_side:
                    self.side_duration += dt

            # Reset transition grace
            self.grace_remaining = self.GRACE_PERIOD
            self.drift_duration += dt

            # Multiplier builds up with sustained drift
            growth_rate = 0.4 * dt
            self.multiplier = min(self.MAX_MULTIPLIER, self.multiplier + growth_rate)

            # Continuous Step Drift Points
            # Points scale with speed and drift angle
            angle_factor = np.clip(slip_deg / 40.0, 0.5, 2.2)
            speed_factor = np.clip(speed / 12.0, 0.4, 2.0)
            base_step_pts = (speed_factor ** 1.2) * angle_factor * 120.0 * dt
            step_pts = base_step_pts * self.multiplier
            self.active_combo_score += step_pts

            # Check Backward Entry (extreme angle slide > 75 deg maintained at speed)
            if slip_deg > 75.0 and speed > 7.0 and not self.in_backward_entry:
                self.in_backward_entry = True
                self.backward_entry_start_time = self.sim_time
            elif self.in_backward_entry and slip_deg <= 55.0 and speed > 5.0:
                # Recovered from backward entry!
                if self.sim_time - self.backward_entry_start_time >= 0.35:
                    trick = TrickEvent("BACKWARD ENTRY", 3500.0, 1.2, self.sim_time)
                    new_tricks.append(trick)
                    self._apply_trick(trick)
                self.in_backward_entry = False

            # Check Wall Tap / Proximity Brush
            wall_tap = self._check_wall_tap(car_corners, track, speed)
            if wall_tap:
                new_tricks.append(wall_tap)
                self._apply_trick(wall_tap)

            # Check Clipping Zones
            clip_trick = self._check_clipping_zones(state, track)
            if clip_trick:
                new_tricks.append(clip_trick)
                self._apply_trick(clip_trick)

            step_reward = step_pts * 0.01

        else:
            # Not sliding this step
            if self.is_drifting:
                # Count down grace timer
                self.grace_remaining -= dt
                if self.grace_remaining <= 0.0:
                    # Dropped combo cleanly (banked)
                    self._bank_combo()
                    self.is_drifting = False
            step_reward = 0.0

        # Maintain recent tricks list
        self.recent_tricks = [
            t for t in self.recent_tricks + new_tricks if self.sim_time - t.timestamp < 3.0
        ]

        return step_reward, new_tricks, spinout_detected

    def _apply_trick(self, trick: TrickEvent):
        """Add trick bonus points and increase multiplier."""
        self.active_combo_score += trick.points * self.multiplier
        self.multiplier = min(self.MAX_MULTIPLIER, self.multiplier + trick.multiplier_bonus)

    def _check_scandinavian_flick(self, current_side: int) -> bool:
        """
        Scandinavian flick: Car steered strongly away from corner then quickly snapped
        steering in the opposite direction right before breaking into a slide.
        """
        if len(self.steer_history) < 5:
            return False

        current_steer = self.steer_history[-1][1]
        opposite_sign = -current_side

        # Look for opposite steering within past 0.6s
        had_opposite_steer = any(
            (s * opposite_sign) > np.radians(15.0) for t, s in self.steer_history[:-2]
        )
        has_snap = (current_steer * current_side) > np.radians(15.0)

        return had_opposite_steer and has_snap

    def _check_wall_tap(self, car_corners: np.ndarray, track: Track, speed: float) -> Optional[TrickEvent]:
        """Check if rear bumper brushes close to a barrier without hitting it."""
        if speed < 6.0:
            return None

        # Rear corners: index 2 (rear right) and 3 (rear left)
        rear_corners = car_corners[2:4]
        min_dist = float("inf")

        for corner in rear_corners:
            for w1, w2 in track.segments:
                dist = point_to_segment_distance(corner, w1, w2)
                if dist < min_dist:
                    min_dist = dist

        # Kiss zone: within 0.1m - 0.45m of barrier
        if 0.10 <= min_dist <= 0.45:
            # Check cooldown so we don't spam 50 times a second
            if not any(t.name == "WALL TAP" and self.sim_time - t.timestamp < 1.0 for t in self.recent_tricks):
                return TrickEvent("WALL TAP", 2500.0, 0.7, self.sim_time)
        return None

    def _check_clipping_zones(self, state: VehicleState, track: Track) -> Optional[TrickEvent]:
        """Check if vehicle passed through an active clipping zone."""
        pos = np.array([state.x, state.y])

        for zone in track.clipping_zones:
            if zone.hit:
                continue
            dist = float(np.hypot(pos[0] - zone.x, pos[1] - zone.y))
            if dist <= zone.radius:
                zone.hit = True
                name = f"CLIPPING ZONE ({zone.zone_type.upper()})"
                return TrickEvent(name, zone.score_value, 0.5, self.sim_time)
        return None

    def _bank_combo(self):
        """Successfully bank accumulated combo score."""
        self.total_banked_score += self.active_combo_score
        self.active_combo_score = 0.0
        self.multiplier = 1.0
        self.drift_duration = 0.0
        self.manji_chains = 0
        self.last_drift_side = 0

    def _break_combo(self, penalty_factor: float = 0.0):
        """Break combo due to spinout or crash, applying penalty to active score."""
        lost_points = self.active_combo_score * (1.0 - penalty_factor)
        self.total_banked_score += self.active_combo_score * penalty_factor
        self.active_combo_score = 0.0
        self.multiplier = 1.0
        self.is_drifting = False
        self.drift_duration = 0.0
        self.manji_chains = 0
        self.last_drift_side = 0

    def handle_collision(self):
        """Called when car hits a wall."""
        self._break_combo(penalty_factor=0.0)

    def get_state(self) -> DriftScoreState:
        return DriftScoreState(
            total_score=self.total_banked_score + self.active_combo_score,
            active_combo_score=self.active_combo_score,
            multiplier=self.multiplier,
            is_drifting=self.is_drifting,
            drift_duration=self.drift_duration,
            grace_timer=max(0.0, self.grace_remaining),
            manji_chains=self.manji_chains,
            recent_tricks=list(self.recent_tricks),
        )


def point_to_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Calculates perpendicular or vertex distance from point p to segment a-b."""
    ab = b - a
    ab_len_sq = np.dot(ab, ab)
    if ab_len_sq < 1e-9:
        return float(np.hypot(p[0] - a[0], p[1] - a[1]))

    t = np.clip(np.dot(p - a, ab) / ab_len_sq, 0.0, 1.0)
    proj = a + t * ab
    return float(np.hypot(p[0] - proj[0], p[1] - proj[1]))
