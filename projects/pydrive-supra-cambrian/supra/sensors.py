"""
The agent's senses.

`SensorSuite.observe(vehicle, track)` turns raw simulator state into the exact
observation an AI policy reads:

  * raycast vision   - 9 beams to the track walls (distance per beam),
  * proprioception   - 25 signals the car "feels" (speeds, g, slip, loads,
                       grip usage, driveline state, where it sits on track),
  * look-ahead       - signed centreline curvature at 6 distances down the road,
  * hill/air (18)    - road slope under the nose/side, body pitch, vertical
                       speed, height above road, airborne flag, plus grade and
                       crest-curvature previews at the SAME 6 distances — the
                       "jump detector": with own speed the policy can predict
                       takeoff, time landings, and brake before downhill bends.
                       APPENDED after the original 40 dims (PHYSICS_3D_PLAN
                       Stage 5), so the old indices never move.

Everything is normalised to roughly [-1, 1] for the network, while the raw
values (and beam end points) are kept alongside so the dashboard can render
exactly what the agent sees. Built now, in slice 2, so slices 3-5 (GA, PPO race,
PPO drift) just call `observe()` and read `.vector`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import SensorSpec
from .physics import Vehicle
from .track import Track

# normalisation references (roughly map each signal onto [-1, 1])
V_REF = 80.0        # m/s
A_REF = 20.0        # m/s^2
R_REF = 3.0         # rad/s yaw rate
CURV_REF = 0.04     # 1/m  (~ 1 / 25 m radius)


def _wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


@dataclass
class Observation:
    vector: np.ndarray                       # full normalised obs for the policy
    beams: np.ndarray                        # raw beam distances (m)
    beam_points: np.ndarray                  # (n, 2) world hit points (rendering)
    beam_angles: np.ndarray                  # beam offsets from heading (rad)
    proprio: np.ndarray                      # normalised proprioceptive block
    proprio_labels: list = field(default_factory=list)   # [(label, raw), ...]
    lookahead: np.ndarray = None             # raw signed curvature ahead (1/m)
    frame: dict = field(default_factory=dict)
    heading_error: float = 0.0               # car heading vs track tangent (rad)
    hill_labels: list = field(default_factory=list)  # [(label, raw), ...] hill/air
    lookahead_grade: np.ndarray = None       # raw dz/ds ahead (viz)
    lookahead_vcurv: np.ndarray = None       # raw crest/dip curvature ahead (viz)
    opponent_beams: np.ndarray = None        # min dist to opponent per beam (m)
    opponent_beam_type: np.ndarray = None    # 0=wall, 1=opponent per beam


class SensorSuite:
    def __init__(self, spec: SensorSpec | None = None):
        self.spec = spec or SensorSpec()
        s = self.spec
        self.beam_angles = np.radians(
            np.linspace(s.beam_spread_deg, -s.beam_spread_deg, s.n_beams))

    @property
    def obs_size(self) -> int:
        base = self.spec.n_beams + 25 + len(self.spec.lookahead_distances)
        if self.spec.hill_block:
            base += 6 + 2 * len(self.spec.lookahead_distances)   # 40 -> 58
        if self.spec.pace_block:
            base += len(self.spec.pace_distances) + 1            # fable-v1
        if self.spec.hybrid_block:
            base += 2                                            # fable-v2
        return base

    def observe(self, veh: Vehicle, trk: Track,
                opponents: list[Vehicle] | None = None) -> Observation:
        s = self.spec

        # --- vision ---
        beams, points = trk.raycast(veh.x, veh.y, veh.yaw, self.beam_angles,
                                    s.beam_range)
        
        # --- opponent ray overlay: test the same beams against opponent OBBs ---
        opp_beams = np.full_like(beams, s.beam_range)
        beam_type = np.zeros(len(beams))  # 0 = wall hit, 1 = opponent hit
        if opponents:
            wa = veh.yaw + self.beam_angles
            dx, dy = np.cos(wa), np.sin(wa)
            P = np.array([veh.x, veh.y])
            for opp in opponents:
                if opp is veh:
                    continue
                obb = opp.get_obb()
                # 4 edges of the OBB
                for ei in range(4):
                    ej = (ei + 1) % 4
                    seg_a = obb[ei]
                    seg_b = obb[ej]
                    edge = seg_b - seg_a
                    ap = seg_a - P
                    # ray-segment intersection per beam
                    denom = dx * edge[1] - dy * edge[0]
                    safe = np.where(np.abs(denom) > 1e-9, denom, 1.0)
                    t = (ap[0] * edge[1] - ap[1] * edge[0]) / safe
                    u = (dy * ap[0] - dx * ap[1]) / safe
                    valid = (np.abs(denom) > 1e-9) & (t >= 0) & (t <= s.beam_range) & (u >= 0) & (u <= 1)
                    t_hit = np.where(valid, t, np.inf)
                    closer = t_hit < opp_beams
                    opp_beams = np.where(closer, t_hit, opp_beams)
            # merge: if opponent is closer than wall, the beam reads the opponent
            opp_closer = opp_beams < beams
            beams = np.where(opp_closer, opp_beams, beams)
            beam_type = np.where(opp_closer, 1.0, 0.0)
            # update hit points for rendering
            points = np.stack([veh.x + dx * beams, veh.y + dy * beams], axis=1)
        
        beams_norm = beams / s.beam_range          # [0, 1], 1 = clear

        # --- where we are on the track ---
        fr = trk.frame(veh.x, veh.y)
        heading_err = _wrap(veh.yaw - fr["heading"])
        lateral_norm = np.clip(fr["lateral"] / trk.half, -1.5, 1.5)

        # --- look-ahead road preview ---
        look = trk.lookahead_curvature(fr["arc"], s.lookahead_distances)
        look_norm = np.clip(look / CURV_REF, -1.0, 1.0)

        # --- proprioception (25) ---
        n_gears = len(veh.spec.gear_ratios)
        mg = veh.spec.mass * 9.81
        labels = [
            ("vx", veh.vx),
            ("vy", veh.vy),
            ("speed", veh.speed),
            ("yaw_rate", veh.r),
            ("ax", veh.ax),
            ("ay", veh.ay),
            ("slip_angle", veh.slip_angle),
            ("heading_err", heading_err),
            ("lateral", fr["lateral"]),
            ("steer", veh.steer_angle),
            ("rpm", veh.rpm),
            ("gear", veh.gear),
            ("boost", veh.boost),
            ("grip_FL", veh.wheel_grip[0]),
            ("grip_FR", veh.wheel_grip[1]),
            ("grip_RL", veh.wheel_grip[2]),
            ("grip_RR", veh.wheel_grip[3]),
            ("Fz_FL", veh.Fz[0]),
            ("Fz_FR", veh.Fz[1]),
            ("Fz_RL", veh.Fz[2]),
            ("Fz_RR", veh.Fz[3]),
            ("sr_FL", veh.wheel_sr[0]),
            ("sr_FR", veh.wheel_sr[1]),
            ("sr_RL", veh.wheel_sr[2]),
            ("sr_RR", veh.wheel_sr[3]),
        ]
        proprio = np.array([
            veh.vx / V_REF,
            veh.vy / V_REF,
            veh.speed / V_REF,
            veh.r / R_REF,
            veh.ax / A_REF,
            veh.ay / A_REF,
            veh.slip_angle / (np.pi / 2),
            heading_err / np.pi,
            lateral_norm,
            veh.steer_angle / veh.max_steer_angle,
            veh.rpm / veh.spec.redline_rpm,
            (veh.gear - 1) / max(1, n_gears - 1),
            veh.boost,
            veh.wheel_grip[0], veh.wheel_grip[1],
            veh.wheel_grip[2], veh.wheel_grip[3],
            veh.Fz[0] / mg, veh.Fz[1] / mg, veh.Fz[2] / mg, veh.Fz[3] / mg,
            np.clip(veh.wheel_sr[0], -1.5, 1.5), np.clip(veh.wheel_sr[1], -1.5, 1.5),
            np.clip(veh.wheel_sr[2], -1.5, 1.5), np.clip(veh.wheel_sr[3], -1.5, 1.5),
        ], dtype=np.float32)

        # --- hill/air block (18) — appended, old indices stay put ---
        hill_labels = []
        look_g = look_v = None
        blocks = [beams_norm, proprio, look_norm]
        if s.hill_block:
            height = max(0.0, veh.z - veh.road_z)
            airborne = 1.0 if veh.airborne else 0.0
            hill_labels = [
                ("grade", veh.grade_body),
                ("bank", veh.bank_body),
                ("pitch", veh.pitch),
                ("vz", veh.vz),
                ("height", height),
                ("air", airborne),
            ]
            hill = np.array([
                np.clip(veh.grade_body / s.grade_ref, -2.0, 2.0),
                np.clip(veh.bank_body / s.bank_ref, -2.0, 2.0),
                np.clip(veh.pitch / s.pitch_ref, -2.0, 2.0),
                np.clip(veh.vz / s.vz_ref, -2.0, 2.0),
                np.clip(height / s.height_ref, 0.0, 2.0),
                airborne,
            ], dtype=np.float32)
            look_g = trk.lookahead_grade(fr["arc"], s.lookahead_distances)
            look_v = trk.lookahead_vcurv(fr["arc"], s.lookahead_distances)
            blocks += [
                hill,
                np.clip(look_g / s.grade_ref, -1.0, 1.0).astype(np.float32),
                np.clip(look_v / s.vcurv_ref, -1.0, 1.0).astype(np.float32),
            ]

        # --- multi-agent block (appended after hill/air) ---
        opp_labels = []
        if opponents and len(opponents) > 1:
            # beam type channel: tells the network if each beam hit wall (0) or car (1)
            blocks.append(beam_type.astype(np.float32))
            
            # nearest opponent relative state
            best_dist = float('inf')
            best_rel_vx = 0.0
            best_rel_vy = 0.0
            for opp in opponents:
                if opp is veh:
                    continue
                ddx = opp.x - veh.x
                ddy = opp.y - veh.y
                dist = np.hypot(ddx, ddy)
                if dist < best_dist:
                    best_dist = dist
                    # relative velocity in ego body frame
                    c, sn = np.cos(veh.yaw), np.sin(veh.yaw)
                    opp_wx = opp.vx * np.cos(opp.yaw) - opp.vy * np.sin(opp.yaw)
                    opp_wy = opp.vx * np.sin(opp.yaw) + opp.vy * np.cos(opp.yaw)
                    ego_wx = veh.vx * np.cos(veh.yaw) - veh.vy * np.sin(veh.yaw)
                    ego_wy = veh.vx * np.sin(veh.yaw) + veh.vy * np.cos(veh.yaw)
                    rel_wx, rel_wy = opp_wx - ego_wx, opp_wy - ego_wy
                    best_rel_vx = rel_wx * c + rel_wy * sn
                    best_rel_vy = -rel_wx * sn + rel_wy * c
            
            opp_state = np.array([
                np.clip(best_dist / s.beam_range, 0.0, 1.0),
                np.clip(best_rel_vx / V_REF, -1.0, 1.0),
                np.clip(best_rel_vy / V_REF, -1.0, 1.0),
            ], dtype=np.float32)
            blocks.append(opp_state)
            
            # own damage state (so the AI knows it's crippled)
            dmg_state = np.array([
                veh.engine_damage,
                veh.aero_damage,
                np.clip(abs(veh.steer_bias) / 0.3, 0.0, 1.0),
                1.0 if veh.turbo_broken else 0.0,
                1.0 if veh.drivetrain_broken else 0.0,
                float(np.max(veh.wheel_damage)),
            ], dtype=np.float32)
            blocks.append(dmg_state)

        # --- pace block (Fable Five): the physics-true speed envelope, here
        # and down the road, + the current speed/envelope ratio. Appended LAST
        # so every legacy index stays put. Zeros when the track carries no
        # envelope (so the layout stays valid on any track).
        if s.pace_block:
            vref = getattr(trk, "fable_vref", None)
            nd = len(s.pace_distances)
            if vref is None:
                blocks.append(np.zeros(nd + 1, dtype=np.float32))
            else:
                arcs = (fr["arc"] + np.asarray(s.pace_distances)) % trk.length
                vs = np.interp(arcs, trk.arc, vref, period=trk.length)
                ratio = veh.speed / max(float(vs[0]), 1.0)
                blocks.append(np.concatenate([
                    vs / s.pace_ref,
                    [np.clip(ratio, 0.0, 2.0) - 1.0],
                ]).astype(np.float32))

        # --- hybrid block (fable-v2): battery SOC + signed MGU power. Gives
        # the policy direct vision into its energy state — how much boost is
        # left, and whether the MGU is deploying (+) or regenerating (-) right
        # now. Appended last so every fable-v1 index stays put; zeros on a
        # non-hybrid car (layout stays valid everywhere).
        if s.hybrid_block:
            cap = max(float(getattr(veh.spec, "hybrid_battery_kj", 0.0)), 1.0)
            soc = float(getattr(veh, "hybrid_soc_kj", 0.0)) / cap
            peak = max(float(getattr(veh.spec, "hybrid_mgu_power_w", 0.0)), 1.0)
            deploy = float(getattr(veh, "mgu_power_w", 0.0)) / peak
            blocks.append(np.array([
                np.clip(soc, 0.0, 1.0),
                np.clip(deploy, -1.5, 1.5),
            ], dtype=np.float32))

        vector = np.concatenate(blocks).astype(np.float32)
        return Observation(
            vector=vector, beams=beams, beam_points=points,
            beam_angles=self.beam_angles, proprio=proprio, proprio_labels=labels,
            lookahead=look, frame=fr, heading_error=heading_err,
            hill_labels=hill_labels, lookahead_grade=look_g,
            lookahead_vcurv=look_v,
            opponent_beams=opp_beams, opponent_beam_type=beam_type,
        )
