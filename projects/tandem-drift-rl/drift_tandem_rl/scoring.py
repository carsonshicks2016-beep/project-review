"""
Formula Drift & D1GP Tandem Judging and Scoring Engine.
Calculates real-time scoring metrics, proximity grading, angle synchronization,
and trick events for both Leader and Chaser vehicles.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np
from drift_tandem_rl.dynamics import VehicleState
from drift_tandem_rl.tracks import Track, ClippingZone


@dataclass
class TandemScoreState:
    """Tandem battle scoring state for a single car or joint heat."""
    total_score: float = 0.0
    line_score: float = 0.0
    angle_score: float = 0.0
    speed_score: float = 0.0
    proximity_score: float = 0.0     # For chaser: door-to-door closeness
    angle_sync_score: float = 0.0    # For chaser: matching leader drift angle
    combo_multiplier: float = 1.0
    drift_duration: float = 0.0
    is_drifting: bool = False
    in_proximity_zone: bool = False
    incidental_contacts: int = 0
    severe_crashes: int = 0


@dataclass
class TandemBattleResult:
    """End-of-heat battle decision."""
    leader_score: float = 0.0
    chaser_score: float = 0.0
    margin: float = 0.0
    winner: str = "TIE"             # 'LEADER', 'CHASER', 'OMT' (One More Time)
    notes: List[str] = field(default_factory=list)


class TandemJudge:
    """
    Evaluates Formula Drift / D1GP criteria for Lead and Chase vehicles.
    """

    def __init__(
        self,
        min_drift_angle_deg: float = 12.0,
        sweet_spot_dist_min: float = 1.2,
        sweet_spot_dist_max: float = 2.8,
    ):
        self.min_drift_rad = np.radians(min_drift_angle_deg)
        self.sweet_spot_min = sweet_spot_dist_min
        self.sweet_spot_max = sweet_spot_dist_max
        self.reset()

    def reset(self):
        self.lead_score = TandemScoreState()
        self.chase_score = TandemScoreState()
        self.proximity_history: List[float] = []
        self.angle_diff_history: List[float] = []

    def evaluate_step(
        self,
        leader: VehicleState,
        chaser: VehicleState,
        track: Track,
        lead_corners: np.ndarray,
        chase_corners: np.ndarray,
        is_collision: bool,
        collision_penetration: float,
        dt: float = 0.02,
    ) -> Tuple[float, float, dict]:
        """
        Calculates instantaneous rewards and updates tandem scores for both cars.

        Returns:
            r_lead: Leader step reward
            r_chase: Chaser step reward
            info: Telemetry and judging metadata
        """
        lead_slip = abs(leader.slip_angle)
        chase_slip = abs(chaser.slip_angle)
        lead_speed = leader.speed
        chase_speed = chaser.speed

        lead_drifting = (lead_slip >= self.min_drift_rad) and (lead_speed >= 4.0)
        chase_drifting = (chase_slip >= self.min_drift_rad) and (chase_speed >= 4.0)

        self.lead_score.is_drifting = lead_drifting
        self.chase_score.is_drifting = chase_drifting

        if lead_drifting:
            self.lead_score.drift_duration += dt
        else:
            self.lead_score.drift_duration = max(0.0, self.lead_score.drift_duration - 2.0 * dt)

        if chase_drifting:
            self.chase_score.drift_duration += dt
        else:
            self.chase_score.drift_duration = max(0.0, self.chase_score.drift_duration - 2.0 * dt)

        # -------------------------------------------------------------
        # 1. Spatial Proximity & Relative Geometry
        # -------------------------------------------------------------
        rel_pos = np.array([chaser.x - leader.x, chaser.y - leader.y])
        cg_dist = float(np.linalg.norm(rel_pos))
        # Hull clearance (approximate bumper-to-bumper or door-to-door gap)
        clearance_dist = max(0.0, cg_dist - 4.40)
        self.proximity_history.append(clearance_dist)

        # Sweet spot is 0.8m to 2.8m clearance (door-to-door tandem drift)
        prox_reward = 0.0
        in_prox_sweet_spot = False
        if clearance_dist < 0.8:
            # Extreme proximity / kissing doors
            prox_reward = 1.0
            in_prox_sweet_spot = True
        elif clearance_dist <= 2.8:
            # Ideal Formula Drift tandem chase proximity
            prox_reward = 1.0 - ((clearance_dist - 1.6) / 1.4) ** 2
            in_prox_sweet_spot = True
        elif clearance_dist <= 6.0:
            # Trailing gap
            prox_reward = 0.3 * (1.0 - (clearance_dist - 2.8) / 3.2)
        else:
            # Fell behind
            prox_reward = -0.15 * min(1.0, (clearance_dist - 6.0) / 8.0)

        self.chase_score.in_proximity_zone = in_prox_sweet_spot

        # -------------------------------------------------------------
        # 2. Angle Matching & Drift Synchronization
        # -------------------------------------------------------------
        # Difference between chaser drift angle and leader drift angle
        angle_diff = abs(chaser.slip_angle - leader.slip_angle)
        self.angle_diff_history.append(angle_diff)

        # Relative yaw alignment
        yaw_diff = (chaser.yaw - leader.yaw + np.pi) % (2.0 * np.pi) - np.pi
        yaw_alignment = np.cos(yaw_diff)  # 1.0 if facing same direction

        # Angle matching reward: only rewarded when both cars are actually drifting!
        angle_sync_reward = 0.0
        if lead_drifting and chase_drifting:
            # Synchronized slip angle
            sync_factor = np.exp(-1.5 * (angle_diff ** 2))
            angle_sync_reward = sync_factor * max(0.0, yaw_alignment)
        elif chase_drifting and not lead_drifting:
            angle_sync_reward = 0.1
        else:
            angle_sync_reward = 0.0

        # -------------------------------------------------------------
        # 3. Leader Base Scoring (Line, Angle, Speed)
        # -------------------------------------------------------------
        # Clipping zones evaluation
        lead_clip_bonus = 0.0
        for zone in track.clipping_zones:
            if not zone.hit_leader:
                d_zone = np.hypot(leader.x - zone.x, leader.y - zone.y)
                if d_zone < zone.radius and lead_drifting:
                    zone.hit_leader = True
                    lead_clip_bonus += 1.5

        lead_angle_reward = np.clip(lead_slip / np.radians(45.0), 0.0, 1.2) if lead_drifting else 0.0
        lead_speed_reward = np.clip(lead_speed / 20.0, 0.0, 1.0)

        # Leader rewards: smooth fast line, deep drift angle, hitting clips
        r_lead = (
            1.2 * lead_angle_reward
            + 0.8 * lead_speed_reward
            + lead_clip_bonus
            + (0.3 if lead_drifting else -0.1)
        )

        # -------------------------------------------------------------
        # 4. Chaser Base Scoring (Proximity, Angle Match, Wake Resilience)
        # -------------------------------------------------------------
        chase_angle_reward = np.clip(chase_slip / np.radians(45.0), 0.0, 1.2) if chase_drifting else 0.0
        chase_speed_reward = np.clip(chase_speed / 20.0, 0.0, 1.0)

        # Dirty air wake survival bonus: rewarded for maintaining control inside turbulent wake
        wake_resilience_bonus = 0.0
        if chaser.in_wake > 0.2 and chase_drifting:
            wake_resilience_bonus = 0.4 * chaser.in_wake

        r_chase = (
            2.0 * prox_reward
            + 1.5 * angle_sync_reward
            + 0.6 * chase_angle_reward
            + 0.4 * chase_speed_reward
            + wake_resilience_bonus
        )

        # -------------------------------------------------------------
        # 5. Collision & Contact Judging
        # -------------------------------------------------------------
        contact_type = "NONE"
        if is_collision:
            if collision_penetration < 0.25 and abs(chase_speed - lead_speed) < 4.0:
                # Minor door-to-door rub / parallel contact ("kissing doors")
                contact_type = "DOOR_RUB"
                self.chase_score.incidental_contacts += 1
                r_chase += 0.2  # Crowd favorite in Formula D!
                r_lead += 0.1
            else:
                # Heavy shunt / rear-ending / T-bone
                contact_type = "AT_FAULT_CRASH"
                self.chase_score.severe_crashes += 1
                r_chase -= 40.0
                r_lead -= 10.0

        # Update cumulative metrics
        self.lead_score.total_score += max(0.0, r_lead)
        self.chase_score.total_score += max(0.0, r_chase)
        self.chase_score.proximity_score += max(0.0, prox_reward)
        self.chase_score.angle_sync_score += max(0.0, angle_sync_reward)

        info = {
            "distance": clearance_dist,
            "cg_distance": cg_dist,
            "angle_diff_deg": float(np.degrees(angle_diff)),
            "in_prox_sweet_spot": in_prox_sweet_spot,
            "in_wake": float(chaser.in_wake),
            "wake_intensity": float(chaser.in_wake),
            "smoke_friction": float(chaser.smoke_exposure),
            "contact_type": contact_type,
            "lead_drifting": lead_drifting,
            "chase_drifting": chase_drifting,
            "lead_score": self.lead_score.total_score,
            "chase_score": self.chase_score.total_score,
        }

        return r_lead, r_chase, info

    def judge_heat_decision(self) -> TandemBattleResult:
        """Determines the winner of the heat based on Formula Drift criteria."""
        l_pts = self.lead_score.total_score
        c_pts = self.chase_score.total_score
        diff = c_pts - l_pts
        margin = abs(diff)

        notes = []
        if self.chase_score.severe_crashes > 0:
            winner = "LEADER"
            notes.append("Chaser incurred zero score due to at-fault collision.")
        elif len(self.proximity_history) > 0 and np.mean(self.proximity_history) <= 3.0 and c_pts >= l_pts * 0.9:
            winner = "CHASER"
            notes.append(f"Chaser dominated with aggressive proximity ({np.mean(self.proximity_history):.1f}m avg) and angle sync.")
        elif l_pts > c_pts * 1.25:
            winner = "LEADER"
            notes.append("Leader gapped the chaser with superior line speed.")
        else:
            winner = "TIE"
            notes.append("Dead heat - One More Time (OMT) required.")

        return TandemBattleResult(
            leader_score=l_pts,
            chaser_score=c_pts,
            margin=margin,
            winner=winner,
            notes=notes,
        )
