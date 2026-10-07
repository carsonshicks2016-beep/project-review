"""
Drift scoring — the reward that turns the agent into a drifter.

Drifting is the opposite of grip racing: you deliberately exceed the tyres' limit
and *hold* a big slip angle. The score follows the classic formula

    reward = (speed * |sin(slip)| * K + bonuses)
             * W_soft_entry * W_speed_gate * sustain

with, per the project spec:
  * soft entry  - a cubic smoothstep so tiny wiggles aren't rewarded, full credit
                  only once the slide is real (~2.3 deg -> ~11.5 deg),
  * speed gate  - scales with speed up to ~12.5 m/s, with a slow-speed penalty so
                  it can't farm points doing static donuts,
  * initiation  - a nudge for breaking the rear loose at speed (before the angle),
  * sustain     - a multiplier that grows the longer a slide is held,
  * transitions - a bonus for flicking the other way within a short window (chains),
  * gutter      - a bonus for clipping the inner edge of a corner mid-slide.

`DriftScorer` is stateful (sustain / chain timers persist across steps) and is
reset each episode. Pure scoring — reads the vehicle + track frame.
"""
from __future__ import annotations

import numpy as np

from .config import DriftReward


class DriftScorer:
    def __init__(self, cfg: DriftReward | None = None):
        self.cfg = cfg or DriftReward()
        self.reset()

    def reset(self):
        self.sustain = 1.0
        self.last_sign = 0
        self.t_since_dir = 99.0
        self.chain = 0
        self.prev_ad = 0.0          # previous |slip| (deg) — for the steady-hold test
        self.prev_yaw = 0.0         # previous |yaw rate| (deg/s) — for the save test

    def step(self, veh, fr, half_width: float, dt: float):
        """Return (reward, info) for one control step of drifting."""
        c = self.cfg
        # touching grass breaks the drift combo — a long *continuous on-road*
        # drift is worth far more than choppy ones broken by excursions
        if fr["off_track"]:
            self.sustain = 1.0
            self.chain = 0
            self.last_sign = 0
        slip = veh.slip_angle
        ad = abs(np.degrees(slip))
        spd = veh.speed

        # soft entry (cubic smoothstep) + speed gate
        s = np.clip((ad - c.entry_lo_deg) / (c.entry_hi_deg - c.entry_lo_deg), 0.0, 1.0)
        w_soft = s * s * (3 - 2 * s)
        w_speed = float(np.clip(spd / c.speed_gate, 0.0, 1.0))

        # over-rotation falloff: reward a *controlled* angle, treat a spin as a fail
        # (raw sin(slip) peaks at 90deg, which would incentivise full spins).
        if ad <= c.peak_deg:
            w_rot = 1.0
        else:
            w_rot = max(0.0, 1.0 - (ad - c.peak_deg) / (c.spin_deg - c.peak_deg))
        # anti-spin yaw-RATE gate: a controlled drift holds a steady angle, a spin
        # is runaway yaw rate. Scale the reward down as yaw rate exceeds a control
        # budget, so a slide tipping into a spin stops paying BEFORE the angle
        # blows up (the angle-only falloff above reacted too late). Multiplicative
        # and bounded [0,1], so it can only suppress reward, never inflate it.
        yaw_deg_s = abs(np.degrees(veh.r))
        if yaw_deg_s <= c.spin_rate_deg:
            w_spin = 1.0
        else:
            w_spin = max(0.0, 1.0 - (yaw_deg_s - c.spin_rate_deg) / c.spin_rate_span)
        w_rot *= w_spin
        core = spd * abs(np.sin(slip)) * c.angle_speed * w_rot
        reward = core * w_soft * w_speed * self.sustain

        # initiation: rear wheelspin at speed, before the angle is established
        rear_spin = max(veh.wheel_sr[2], veh.wheel_sr[3])
        if spd > 6.0 and rear_spin > 0.15 and ad < c.entry_hi_deg:
            reward += c.initiation * w_speed

        # slow-speed donut penalty
        if spd < c.donut_speed and ad > 15.0:
            reward -= c.donut_penalty

        # chains / transitions + sustain
        self.t_since_dir += dt
        drifting = ad > c.drift_min_deg and spd > 5.0
        if drifting:
            sign = 1 if slip > 0 else -1
            if (self.last_sign != 0 and sign != self.last_sign
                    and self.t_since_dir < c.chain_window):
                self.chain += 1
                reward += c.transition_bonus * min(self.chain, 5)
                self.t_since_dir = 0.0
            elif self.last_sign == 0:
                self.t_since_dir = 0.0
            self.last_sign = sign
            self.sustain = min(self.sustain + dt * c.sustain_rate, c.sustain_max)
        else:
            self.sustain = max(1.0, self.sustain - dt * 2.0)
            self.chain = 0
            self.last_sign = 0

        # gutter: sliding while hugging the inner edge of a corner
        if drifting and abs(fr["curvature"]) > 5e-3:
            inside = 1 if fr["curvature"] > 0 else -1     # left turn -> inside is +lateral
            if np.sign(fr["lateral"]) == inside and abs(fr["lateral"]) > 0.5 * half_width:
                reward += c.gutter_bonus * w_soft

        # controlled-hold bonus: reward keeping a real, controlled angle STEADY
        # (small change in slip, yaw under control) within the drift band — i.e.
        # holding a drift THROUGH a corner instead of snapping toward max angle
        # then spinning. Directly targets "drift-AND-complete" over "max-angle".
        d_ad = abs(ad - self.prev_ad) / max(dt, 1e-3)
        self.prev_ad = ad
        if (drifting and c.entry_hi_deg <= ad <= c.peak_deg
                and d_ad < c.hold_rate_deg and yaw_deg_s <= c.spin_rate_deg):
            reward += c.hold_bonus * w_speed

        # slide RECOVERY ("the save"): over-rotating (big angle, high yaw rate) but
        # actively bringing the yaw rate back DOWN = catching the slide before it
        # spins. This is exactly the skill that turns "drifts then spins out" into
        # "rides the edge and completes" — so reward it, scaled by how decisively
        # the rotation is being arrested.
        d_yaw = (yaw_deg_s - self.prev_yaw) / max(dt, 1e-3)
        self.prev_yaw = yaw_deg_s
        if (drifting and ad > c.peak_deg * 0.75 and yaw_deg_s > c.recover_yaw_deg
                and d_yaw < 0):
            reward += c.recover_bonus * w_speed * min(1.0, -d_yaw / 120.0)

        reward *= c.scale
        info = {"slip_deg": ad, "sustain": self.sustain, "chain": self.chain,
                "drifting": drifting}
        return reward, info
