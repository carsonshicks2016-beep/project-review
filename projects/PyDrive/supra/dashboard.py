"""
Telemetry dashboard — a side instrument panel.

Renders, in one column:
  * tachometer arc + speed/gear + boost,
  * a G-force meter with a motion trail,
  * a four-corner tyre diagram (load + grip usage + lock/spin per wheel),
  * a drift-angle indicator,
  * the raycast sensor fan and the look-ahead curvature strip,
  * the live neural-network input vector as a bar field,
  * the driver/agent control inputs.

The point of the sensor fan + NN-input field is that you can literally watch
what the agent perceives. Pure rendering; reads an Observation from sensors.py.
"""
from __future__ import annotations

import numpy as np
import pygame

# palette
BG = (18, 19, 23)
PANEL = (28, 30, 36)
INK = (225, 228, 235)
DIM = (140, 145, 155)
ACCENT = (90, 170, 240)
GOOD = (90, 200, 110)
WARN = (235, 190, 70)
BAD = (230, 80, 70)


def _lerp(a, b, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def grip_color(g: float):
    """Green (slack) -> yellow (near limit) -> red (saturated/over)."""
    if g < 0.85:
        return _lerp(GOOD, WARN, g / 0.85)
    return _lerp(WARN, BAD, (g - 0.85) / 0.25)


class Dashboard:
    WIDTH = 360

    def __init__(self):
        self.f_big = pygame.font.SysFont("menlo,consolas,monospace", 40, bold=True)
        self.f_med = pygame.font.SysFont("menlo,consolas,monospace", 18, bold=True)
        self.f = pygame.font.SysFont("menlo,consolas,monospace", 14)
        self.f_sm = pygame.font.SysFont("menlo,consolas,monospace", 11)
        self.g_trail = []   # G-force history for the trail
        self._was_air = False   # landing-g flash bookkeeping
        self._land_flash = 0

    # ------------------------------------------------------------------ #
    def draw(self, screen, veh, obs, inputs):
        sw, sh = screen.get_size()
        x0 = sw - self.WIDTH
        panel = pygame.Surface((self.WIDTH, sh), pygame.SRCALPHA)
        panel.fill((*PANEL, 235))
        screen.blit(panel, (x0, 0))
        pygame.draw.line(screen, (60, 64, 72), (x0, 0), (x0, sh), 2)

        pad = 16
        x = x0 + pad
        w = self.WIDTH - 2 * pad
        y = 14
        y = self._header(screen, x, y, w, veh)
        y = self._tacho(screen, x, y, w, veh)
        y = self._gmeter_and_drift(screen, x, y, w, veh, obs)
        y = self._road(screen, x, y, w, veh)
        y = self._tyres(screen, x, y, w, veh)
        y = self._sensors(screen, x, y, w, obs)
        y = self._nn_inputs(screen, x, y, w, obs)
        self._controls(screen, x, sh - 92, w, inputs)

    # ------------------------------------------------------------------ #
    def _header(self, s, x, y, w, veh):
        t = veh.telemetry()
        s.blit(self.f_big.render(f"{t['speed_kmh']:3.0f}", True, INK), (x, y))
        s.blit(self.f.render("km/h", True, DIM), (x + 96, y + 24))
        s.blit(self.f_big.render(f"{veh.gear}", True, ACCENT), (x + w - 34, y))
        s.blit(self.f_sm.render("GEAR", True, DIM), (x + w - 44, y + 42))
        return y + 58

    def _tacho(self, s, x, y, w, veh):
        cx, cy, r = x + w // 2, y + 66, 60
        rl = veh.spec.redline_rpm
        f_rpm = veh.rpm / rl
        a0, a1 = np.radians(150), np.radians(-150 + 360)  # sweep 240 deg
        sweep = np.radians(240)
        start = np.radians(150)
        # ticks + redline zone
        for k in range(0, 21):
            fr = k / 20
            ang = start - sweep * fr
            col = BAD if fr > 0.85 else DIM
            r1 = r - (8 if k % 5 == 0 else 4)
            p1 = (cx + r * np.cos(ang), cy - r * np.sin(ang))
            p2 = (cx + r1 * np.cos(ang), cy - r1 * np.sin(ang))
            pygame.draw.line(s, col, p1, p2, 2)
        # needle
        ang = start - sweep * min(f_rpm, 1.05)
        col = BAD if f_rpm > 0.85 else ACCENT
        tip = (cx + (r - 10) * np.cos(ang), cy - (r - 10) * np.sin(ang))
        pygame.draw.line(s, col, (cx, cy), tip, 3)
        pygame.draw.circle(s, INK, (cx, cy), 4)
        s.blit(self.f_sm.render(f"{veh.rpm:.0f} rpm", True, DIM), (cx - 26, cy + 14))
        # boost bar under the tacho
        by = y + 132
        self._bar(s, x, by, w, 10, veh.boost, ACCENT, "BOOST")
        return by + 26

    def _gmeter_and_drift(self, s, x, y, w, veh, obs):
        # G-meter (left), drift-angle dial (right)
        gr = 52
        gcx, gcy = x + gr + 4, y + gr + 4
        pygame.draw.circle(s, BG, (gcx, gcy), gr)
        for rr in (gr // 2, gr):
            pygame.draw.circle(s, (60, 64, 72), (gcx, gcy), rr, 1)
        pygame.draw.line(s, (50, 54, 62), (gcx - gr, gcy), (gcx + gr, gcy))
        pygame.draw.line(s, (50, 54, 62), (gcx, gcy - gr), (gcx, gcy + gr))
        gx = gcx + int(np.clip(-veh.ay / 12.0, -1, 1) * gr)
        gy = gcy + int(np.clip(-veh.ax / 12.0, -1, 1) * gr)
        self.g_trail.append((gx, gy))
        if len(self.g_trail) > 18:
            self.g_trail = self.g_trail[-18:]
        for i, (tx, ty) in enumerate(self.g_trail):
            a = i / len(self.g_trail)
            pygame.draw.circle(s, _lerp((50, 54, 62), ACCENT, a), (tx, ty), 2)
        pygame.draw.circle(s, WARN, (gx, gy), 5)
        s.blit(self.f_sm.render("G", True, DIM), (gcx - 3, gcy - gr - 14))
        gtot = np.hypot(veh.ax, veh.ay) / 9.81
        s.blit(self.f_sm.render(f"{gtot:.2f}g", True, DIM), (gcx - 14, gcy + gr + 2))

        # drift-angle dial
        dcx, dcy = x + w - gr - 4, y + gr + 4
        pygame.draw.circle(s, BG, (dcx, dcy), gr)
        pygame.draw.circle(s, (60, 64, 72), (dcx, dcy), gr, 1)
        slip = veh.slip_angle
        # car heading points up; velocity vector rotated by slip
        for ang, col, ln in [(0, DIM, gr - 6), (-slip, WARN, gr - 4)]:
            tip = (dcx + ln * np.sin(ang), dcy - ln * np.cos(ang))
            pygame.draw.line(s, col, (dcx, dcy), tip, 3 if col == WARN else 2)
        s.blit(self.f_sm.render("DRIFT", True, DIM), (dcx - 16, dcy - gr - 14))
        sd = abs(np.degrees(slip))
        col = BAD if sd > 50 else WARN if sd > 12 else DIM
        s.blit(self.f_sm.render(f"{np.degrees(slip):+.0f}d", True, col),
               (dcx - 16, dcy + gr + 2))
        return y + 2 * gr + 24

    def _road(self, s, x, y, w, veh):
        """ROAD line (PHYSICS_3D_PLAN Stage 8): live grade/pitch/roll attitude,
        an AIR tag while flying, and a landing-g flash on touchdown — the 2D
        twin of the 3D viewer's AIR pill."""
        grade = getattr(veh, "grade_body", 0.0)
        air = bool(getattr(veh, "airborne", False))
        if air:
            self._land_flash = 0
        elif self._was_air:
            self._land_flash = 90          # ~1.5 s at 60 fps
        self._was_air = air

        s.blit(self.f_sm.render("ROAD", True, DIM), (x, y))
        txt = (f"grade {grade * 100:+5.1f}%   "
               f"pitch {np.degrees(getattr(veh, 'pitch', 0.0)):+5.1f}°   "
               f"roll {np.degrees(getattr(veh, 'roll', 0.0)):+5.1f}°")
        s.blit(self.f.render(txt, True, INK), (x, y + 13))
        if air:
            tag = f"AIR {getattr(veh, 'air_time', 0.0):.1f}s"
            s.blit(self.f_med.render(tag, True, WARN), (x + w - 84, y + 8))
        elif self._land_flash > 0:
            self._land_flash -= 1
            lg = getattr(veh, "landing_g", 0.0)
            if lg > 0.3:
                col = BAD if lg > 3.0 else WARN
                s.blit(self.f_med.render(f"▼{lg:.1f}g", True, col),
                       (x + w - 84, y + 8))
        return y + 34

    def _tyres(self, s, x, y, w, veh):
        s.blit(self.f.render("TYRE LOAD / GRIP", True, DIM), (x, y))
        y += 20
        bx, by, bw, bh = x + w // 2 - 34, y, 68, 96   # car body
        pygame.draw.rect(s, (44, 48, 56), (bx, by, bw, bh), border_radius=8)
        mg = veh.spec.mass * 9.81
        # wheel positions (FL,FR,RL,RR)
        corners = [(bx - 12, by + 6), (bx + bw - 6, by + 6),
                   (bx - 12, by + bh - 30), (bx + bw - 6, by + bh - 30)]
        names = ["FL", "FR", "RL", "RR"]
        for i, (wx, wy) in enumerate(corners):
            load = veh.Fz[i] / (mg * 0.5)            # ~1.0 at static-ish
            grip = veh.wheel_grip[i]
            col = grip_color(grip)
            wh = int(16 + 14 * np.clip(load, 0, 2))   # taller = more load
            pygame.draw.rect(s, col, (wx, wy, 18, wh), border_radius=3)
            pygame.draw.rect(s, (20, 22, 26), (wx, wy, 18, wh), 1, border_radius=3)
            s.blit(self.f_sm.render(names[i], True, DIM), (wx, wy - 12))
            # lock (blue) / spin (orange) tick from slip ratio
            sr = veh.wheel_sr[i]
            if abs(sr) > 0.06:
                tcol = (250, 150, 60) if sr > 0 else (90, 150, 250)
                pygame.draw.rect(s, tcol, (wx, wy + wh + 2, 18, 3))
        # numeric grip in the body
        s.blit(self.f_sm.render("grip%", True, DIM), (bx + 8, by + bh // 2 - 16))
        gavg = int(np.mean(veh.wheel_grip) * 100)
        s.blit(self.f_med.render(f"{gavg}", True, grip_color(np.mean(veh.wheel_grip))),
               (bx + 16, by + bh // 2 - 2))
        return y + bh + 16

    def _sensors(self, s, x, y, w, obs):
        s.blit(self.f.render("VISION  (9 beams)", True, DIM), (x, y))
        y += 18
        cx, cy = x + w // 2, y + 96
        maxlen = 92
        rng = obs.beams.max() if len(obs.beams) else 1.0
        for ang, dist in zip(obs.beam_angles, obs.beams):
            fr = dist / 70.0
            ln = maxlen * min(fr, 1.0)
            # beam points "up" the panel; +angle = left
            ex = cx - ln * np.sin(ang)
            ey = cy - ln * np.cos(ang)
            col = _lerp(BAD, GOOD, min(fr, 1.0))
            pygame.draw.line(s, col, (cx, cy), (ex, ey), 2)
            pygame.draw.circle(s, col, (int(ex), int(ey)), 3)
        # car marker
        pygame.draw.circle(s, INK, (cx, cy), 4)
        # look-ahead curvature strip
        ly = y + 104
        s.blit(self.f_sm.render("look-ahead curvature", True, DIM), (x, ly))
        ly += 14
        n = len(obs.lookahead)
        cw = w // n
        for i, k in enumerate(obs.lookahead):
            kn = float(np.clip(k / 0.04, -1, 1))
            cxr = x + i * cw + cw // 2
            h = int(abs(kn) * 16)
            col = ACCENT if kn >= 0 else WARN          # left vs right bend
            pygame.draw.rect(s, col, (cxr - cw // 2 + 3, ly + 16 - h, cw - 6, max(h, 2)))
            pygame.draw.line(s, DIM, (x, ly + 16), (x + w, ly + 16), 1)
        ly += 28

        # hill/air block (appended obs group — PHYSICS_3D_PLAN Stage 5):
        # look-ahead grade strip (up = climb ahead) + a live attitude line
        if getattr(obs, "lookahead_grade", None) is not None:
            s.blit(self.f_sm.render("look-ahead grade", True, DIM), (x, ly))
            ly += 14
            for i, g in enumerate(obs.lookahead_grade):
                gn = float(np.clip(g / 0.20, -1, 1))
                cxr = x + i * cw + cw // 2
                h = int(abs(gn) * 16)
                col = GOOD if gn >= 0 else WARN        # climb vs descent
                pygame.draw.rect(s, col, (cxr - cw // 2 + 3,
                                          ly + 16 - h if gn >= 0 else ly + 16,
                                          cw - 6, max(h, 2)))
            pygame.draw.line(s, DIM, (x, ly + 16), (x + w, ly + 16), 1)
            ly += 22
            hill = dict(obs.hill_labels)
            txt = (f"grade {hill.get('grade', 0.0) * 100:+.0f}%   "
                   f"pitch {np.degrees(hill.get('pitch', 0.0)):+.1f}°   "
                   f"vz {hill.get('vz', 0.0):+.1f}")
            s.blit(self.f_sm.render(txt, True, INK), (x, ly))
            if hill.get("air"):
                s.blit(self.f.render("AIR", True, WARN), (x + w - 34, ly - 2))
            ly += 16
        return ly + 6

    def _nn_inputs(self, s, x, y, w, obs):
        s.blit(self.f.render(f"NN INPUT  ({obs.vector.size})", True, DIM), (x, y))
        y += 16
        v = obs.vector
        n = v.size
        bw = max(2, w // n)
        mid = y + 22
        pygame.draw.line(s, (60, 64, 72), (x, mid), (x + bw * n, mid), 1)
        for i, val in enumerate(v):
            vv = float(np.clip(val, -1, 1))
            h = int(vv * 18)
            col = ACCENT if vv >= 0 else WARN
            bx = x + i * bw
            pygame.draw.rect(s, col, (bx, mid - h if h >= 0 else mid, bw - 1, abs(h) + 1))
        return mid + 26

    def _controls(self, s, x, y, w, inputs):
        s.blit(self.f.render("CONTROLS", True, DIM), (x, y))
        y += 18
        self._bar(s, x, y, w, 9, inputs["throttle"], GOOD, "THR"); y += 16
        self._bar(s, x, y, w, 9, inputs["brake"], BAD, "BRK"); y += 16
        self._bar(s, x, y, w, 9, inputs["handbrake"], WARN, "HND"); y += 16
        # steering: centred bar
        cy = y + 5
        pygame.draw.rect(s, (50, 54, 62), (x, y, w, 9))
        cxm = x + w // 2
        sv = float(np.clip(inputs["steer"], -1, 1))
        if sv >= 0:
            pygame.draw.rect(s, ACCENT, (cxm, y, int(w / 2 * sv), 9))
        else:
            pygame.draw.rect(s, ACCENT, (cxm + int(w / 2 * sv), y, int(-w / 2 * sv), 9))
        pygame.draw.line(s, INK, (cxm, y - 2), (cxm, y + 11), 1)
        s.blit(self.f_sm.render("STEER", True, DIM), (x, y + 10))

    # ------------------------------------------------------------------ #
    def _bar(self, s, x, y, w, h, frac, color, label):
        frac = float(np.clip(frac, 0, 1))
        pygame.draw.rect(s, (50, 54, 62), (x, y, w, h))
        pygame.draw.rect(s, color, (x, y, int(w * frac), h))
        s.blit(self.f_sm.render(label, True, DIM), (x + w - 38, y - 1))


# --------------------------------------------------------------------------- #
# world overlay: draw the beams from the car into the scene
# --------------------------------------------------------------------------- #
def draw_beams(screen, cam, veh, obs):
    """Draw raycast beams from the car to their hit points in world space."""
    origin = cam.to_screen(veh.x, veh.y)
    for dist, pt in zip(obs.beams, obs.beam_points):
        end = cam.to_screen(*pt)
        fr = dist / 70.0
        col = _lerp((230, 90, 80), (120, 210, 140), min(fr, 1.0))
        pygame.draw.line(screen, col, origin, end, 1)
        pygame.draw.circle(screen, col, end, 3)
