"""
The side telemetry panel -- "every single detail of the car, its brain,
physics, fitness and road difficulty".

Pure pygame drawing.  render() paints the right-hand strip of the window.
"""

from __future__ import annotations
import math
import numpy as np
import pygame

from .config import Config

WHITE = (235, 238, 244)
DIM = (150, 156, 168)
PANEL = (24, 27, 33)
CARD = (32, 36, 44)
GREEN = (90, 210, 120)
YELLOW = (240, 200, 70)
RED = (230, 70, 60)
BLUE = (90, 170, 240)
ORANGE = (240, 150, 50)


class Dashboard:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        r = cfg.render
        self.x0 = r.width - r.dash_width
        self.w = r.dash_width
        self.h = r.height
        self.f_big = pygame.font.SysFont("menlo,consolas,monospace", 30, bold=True)
        self.f_h = pygame.font.SysFont("menlo,consolas,monospace", 18, bold=True)
        self.f = pygame.font.SysFont("menlo,consolas,monospace", 13)
        self.f_s = pygame.font.SysFont("menlo,consolas,monospace", 11)
        self._map_bounds = None

    # ---- primitives --------------------------------------------------- #
    def _txt(self, s, x, y, color=WHITE, font=None, right=False):
        font = font or self.f
        surf = font.render(str(s), True, color)
        if right:
            x -= surf.get_width()
        self.screen.blit(surf, (x, y))
        return surf.get_height()

    def _bar(self, x, y, w, h, frac, color, label=None, value=None, bg=(48, 52, 62)):
        frac = max(0.0, min(1.2, frac))
        pygame.draw.rect(self.screen, bg, (x, y, w, h), border_radius=3)
        pygame.draw.rect(self.screen, color, (x, y, int(w * min(frac, 1.0)), h),
                         border_radius=3)
        if frac > 1.0:  # overload marker
            pygame.draw.rect(self.screen, RED, (x + w - 3, y, 3, h))
        if label:
            self._txt(label, x, y - 15, DIM, self.f_s)
        if value is not None:
            self._txt(value, x + w, y - 15, WHITE, self.f_s, right=True)

    def _card(self, y, h, title=None):
        pygame.draw.rect(self.screen, CARD, (self.x0 + 8, y, self.w - 16, h),
                         border_radius=6)
        if title:
            self._txt(title, self.x0 + 18, y + 6, DIM, self.f_s)
        return y + (24 if title else 10)

    # ---- main --------------------------------------------------------- #
    def render(self, screen, sim, evo):
        self.screen = screen
        cfg = self.cfg
        pygame.draw.rect(screen, PANEL, (self.x0, 0, self.w, self.h))
        car = sim.focus_car
        x = self.x0 + 18
        y = 12

        # header
        self._txt(f"SUPRA-AI  ·  {cfg.car.engine}", x, y, WHITE, self.f_h)
        y += 24
        self._txt(f"GEN {sim.generation:>3}   ALIVE {sim.alive_count}/{len(sim.cars)}"
                  f"   t {sim.gen_time:4.1f}s", x, y, DIM, self.f_s)
        y += 20
        if car is None:
            return
        v = car.vehicle

        # ---- speed / rpm / gear card ----
        yc = self._card(y, 116, "POWERTRAIN")
        speed_str = f"{v.speed * 3.6:.0f}"
        speed_surf = self.f_big.render(speed_str, True, WHITE)
        self.screen.blit(speed_surf, (x, yc))
        speed_w = speed_surf.get_width()
        
        # Place labels dynamically relative to the speed text width to avoid overlap
        self._txt(f"({v.speed * 2.237:.0f} mph)", x + speed_w + 8, yc + 2, DIM, self.f_s)
        self._txt("km/h", x + speed_w + 8, yc + 16, DIM, self.f_s)
        
        gear_lbl = "N" if v.vx < 0.3 and v.gear == 0 else str(v.gear + 1)
        gear_surf = self.f_big.render(gear_lbl, True, YELLOW)
        # Center gear digit and GEAR label at self.x0 + self.w - 44
        gear_center_x = self.x0 + self.w - 44
        self.screen.blit(gear_surf, (gear_center_x - gear_surf.get_width() // 2, yc - 4))
        gear_lbl_width = self.f_s.size("GEAR")[0]
        self._txt("GEAR", gear_center_x - gear_lbl_width // 2, yc + 30, DIM, self.f_s)
        yc += 44
        # tach with redline + shift lights
        rpm_frac = v.rpm / cfg.car.rev_limit_rpm
        red_frac = cfg.car.redline_rpm / cfg.car.rev_limit_rpm
        tcol = RED if v.rpm >= cfg.car.redline_rpm else (YELLOW if rpm_frac > 0.78 else GREEN)
        self._bar(x, yc, self.w - 60, 14, rpm_frac, tcol,
                  value=f"{v.rpm:5.0f} rpm")
        # redline tick
        rx = x + int((self.w - 60) * red_frac)
        pygame.draw.line(screen, RED, (rx, yc - 2), (rx, yc + 16), 2)
        # shift lights
        for i in range(6):
            on = rpm_frac > (0.70 + i * 0.045)
            col = (RED if i >= 4 else (YELLOW if i >= 2 else GREEN)) if on else (60, 64, 72)
            pygame.draw.circle(screen, col, (x + 10 + i * 16, yc + 32), 5)
        self._txt(f"{v.engine_power_kw * 1.341:.0f} hp", x + 120, yc + 26, WHITE, self.f_s)
        self._txt(f"{v.engine_torque:.0f} Nm", x + 200, yc + 26, WHITE, self.f_s)
        y = yc + 50

        # ---- turbo / boost ----
        yc = self._card(y, 60, "SEQUENTIAL TWIN-TURBO")
        boost_frac = v.boost / cfg.car.max_boost_bar
        self._bar(x, yc + 4, self.w - 60, 12, boost_frac, ORANGE,
                  value=f"{v.boost * 14.5:.1f} psi")
        self._txt("BOOST", x, yc - 11, DIM, self.f_s)
        y = yc + 50

        # ---- driver inputs (applied = after rate-limited actuators) ----
        yc = self._card(y, 92, "DRIVER INPUTS  (applied)")
        st = v.steer / cfg.car.max_steer_angle
        self._bar(x, yc + 2, self.w - 60, 9, v.throttle, GREEN, "THROTTLE")
        self._bar(x, yc + 22, self.w - 60, 9, v.brake, RED, "BRAKE")
        self._bar(x, yc + 42, self.w - 60, 9, getattr(car, "clutch", 1.0), ORANGE,
                  "CLUTCH (1=engaged)")
        # steering centred bar
        cxw = self.w - 60
        cx = x + cxw // 2
        pygame.draw.rect(screen, (48, 52, 62), (x, yc + 64, cxw, 9), border_radius=3)
        sw = int((cxw // 2) * st)
        pygame.draw.rect(screen, BLUE, (min(cx, cx + sw), yc + 64, abs(sw), 9),
                         border_radius=3)
        pygame.draw.line(screen, WHITE, (cx, yc + 62, ), (cx, yc + 75), 1)
        self._txt(f"STEER {math.degrees(v.steer):+.0f}°", x, yc + 52, DIM, self.f_s)
        up_col = GREEN if getattr(car, "shift_up", False) else (60, 64, 72)
        dn_col = YELLOW if getattr(car, "shift_down", False) else (60, 64, 72)
        self._txt("SHIFT", x + 150, yc + 52, DIM, self.f_s)
        self._txt("▲", x + 200, yc + 51, up_col, self.f_s)
        self._txt("▼", x + 218, yc + 51, dn_col, self.f_s)
        y = yc + 92

        # ---- chassis dynamics: G-meter + live 4-wheel weight + grip ----
        yc = self._card(y, 158, "CHASSIS  ·  WEIGHT  ·  TYRES")
        # G circle (left)
        gcx, gcy, gr = x + 38, yc + 52, 36
        pygame.draw.circle(screen, (48, 52, 62), (gcx, gcy), gr, 1)
        pygame.draw.circle(screen, (40, 44, 54), (gcx, gcy), gr // 2, 1)
        pygame.draw.line(screen, (60, 64, 74), (gcx - gr, gcy), (gcx + gr, gcy), 1)
        pygame.draw.line(screen, (60, 64, 74), (gcx, gcy - gr), (gcx, gcy + gr), 1)
        gx = gcx + int(np.clip(v.lat_g / 1.4, -1, 1) * gr)
        gy = gcy + int(np.clip(-v.long_g / 1.4, -1, 1) * gr)
        pygame.draw.line(screen, (90, 96, 110), (gcx, gcy), (gx, gy), 1)
        pygame.draw.circle(screen, YELLOW, (gx, gy), 4)
        self._txt(f"{math.hypot(v.lat_g, v.long_g):.2f}g", gcx - 16, yc + 92, WHITE, self.f_s)
        self._txt("G-FORCE", gcx - 22, yc - 2, DIM, self.f_s)

        # live 4-corner load diagram (middle) -- this is the "weight"
        self._wheels(v, x + 86, yc + 8, 60, 96)

        # grip + slip bars (right)
        bx = x + 168
        bw = (self.x0 + self.w - 18) - bx
        self._bar(bx, yc + 14, bw, 9, v.grip_f, self._gripcol(v.grip_f),
                  "FRONT GRIP", f"{v.grip_f * 100:.0f}%")
        self._bar(bx, yc + 40, bw, 9, v.grip_r, self._gripcol(v.grip_r),
                  "REAR GRIP", f"{v.grip_r * 100:.0f}%")
        total = v.load_f + v.load_r + 1e-6
        self._bar(bx, yc + 66, bw, 9, v.load_f / total, BLUE,
                  "LOAD F/R %", f"{v.load_f/total*100:.0f}/{v.load_r/total*100:.0f}")
        self._txt(f"roll {v.roll:+.1f}°  pitch {v.pitch:+.1f}°", bx, yc + 80, DIM, self.f_s)
        # slip / attitude
        beta = math.degrees(v.slip_angle)
        attitude = "DRIFT" if abs(beta) > 12 else (
            "OVERSTEER" if v.grip_r > v.grip_f + 0.05 else
            "UNDERSTEER" if v.grip_f > v.grip_r + 0.05 else "NEUTRAL")
        acol = RED if attitude == "DRIFT" else (ORANGE if "STEER" in attitude else GREEN)
        self._txt(f"SLIP {beta:+5.1f}°", x, yc + 116, DIM, self.f_s)
        self._txt(attitude, x + 110, yc + 116, acol, self.f_s)
        self._txt(f"steer-feel {v.aligning_torque:+6.0f} Nm", x + 230, yc + 116, DIM, self.f_s)
        y = yc + 158

        # ---- the brain ----
        yc = self._card(y, 150, "NEURAL NET  ·  inputs → outputs")
        self._txt("VISION (raycasts)", x, yc + 2, DIM, self.f_s)
        n_rays = cfg.sensors.n_rays
        n_extra = cfg.sensors.n_extra
        rays = car.inputs[:n_rays]
        # draw ray inputs as vertical bars (shifted down to prevent overlapping the text header)
        bw2 = 14
        for i, rv in enumerate(rays):
            h = int(36 * rv)
            col = GREEN if rv > 0.5 else (YELLOW if rv > 0.25 else RED)
            pygame.draw.rect(screen, col, (x + i * (bw2 + 2), yc + 56 - h, bw2, h))
        # extra (proprioceptive) inputs
        extras = car.inputs[n_rays:n_rays + n_extra]
        elbl = ["vx", "vy", "yaw", "slip", "rpm", "gear", "latG", "str", "sat"]
        # Space out columns to 40px to prevent horizontal overlaps and center text over needles
        col_spacing = 40
        for i, (lb, ev) in enumerate(zip(elbl, extras)):
            ex = x + i * col_spacing
            lbl_w = self.f_s.size(lb)[0]
            self._txt(lb, ex + 8 - lbl_w // 2, yc + 68, DIM, self.f_s)
            mid = yc + 96
            h = int(np.clip(ev, -1, 1) * 14)
            pygame.draw.line(screen, BLUE, (ex + 8, mid), (ex + 8, mid - h), 3)
            pygame.draw.line(screen, (70, 74, 84), (ex, mid), (ex + 16, mid), 1)
        # look-ahead "track preview": signed curvature ahead (left=blue, right=orange)
        # (slice to n_curve so a drift policy's extra drift-state inputs don't overflow)
        preview = car.inputs[n_rays + n_extra:n_rays + n_extra + cfg.sensors.n_curve]
        self._txt("TRACK PREVIEW  ·  curvature ahead →", x, yc + 110, DIM, self.f_s)
        for i, cv in enumerate(preview):
            ex = x + i * col_spacing
            mid = yc + 138
            h = int(np.clip(cv, -1, 1) * 12)
            col = BLUE if cv >= 0 else ORANGE
            pygame.draw.line(screen, (70, 74, 84), (ex, mid), (ex + 16, mid), 1)
            pygame.draw.line(screen, col, (ex + 8, mid), (ex + 8, mid - h), 3)
        y = yc + 150

        # ---- fitness + road difficulty + minimap ----
        yc = self._card(y, 150, "FITNESS  ·  ROAD")
        self._txt(f"fitness {car.fitness:8.0f}", x, yc, WHITE, self.f)
        self._txt(f"dist {car.distance:7.0f} m   laps {car.laps}", x, yc + 16, DIM, self.f_s)
        best = f"{evo.best_fitness:8.0f}" if evo.best_fitness > -1e8 else "      --"
        self._txt(f"best ever {best}", x, yc + 32, GREEN, self.f_s)
        # difficulty meter (narrower width of 230 to prevent horizontal overlap with minimap)
        self._bar(x, yc + 60, 230, 9, car.road_difficulty,
                  self._gripcol(1 - car.road_difficulty),
                  "ROAD DIFFICULTY (ahead)",
                  ["EASY", "MED", "HARD"][min(2, int(car.road_difficulty * 3))])
        # fitness sparkline (narrower width of 230 to keep consistent alignment)
        self._sparkline(x, yc + 88, 230 + 116, 28, evo.history, evo.avg_history)
        # minimap (fits cleanly on the right side, completely enclosed in the card)
        self._minimap(sim, car, self.x0 + self.w - 120, yc + 44, 104, 76)

    # ---- helpers ------------------------------------------------------ #
    def _gripcol(self, f):
        f = max(0.0, min(1.0, f))
        if f > 0.92:
            return RED
        if f > 0.75:
            return YELLOW
        return GREEN

    def _wheels(self, v, x, y, w, h):
        """Top-down 4-corner load diagram: circle size = load, colour = grip."""
        self._txt("WEIGHT", x - 2, y - 14, DIM, self.f_s)
        pygame.draw.rect(self.screen, (26, 29, 36), (x + w // 2 - 10, y + 6, 20, h - 12),
                         border_radius=5)
        self._txt("F", x + w // 2 - 3, y - 1, DIM, self.f_s)
        nominal = self.cfg.car.mass * 9.81 / 4.0
        corners = [(x, y + 6), (x + w, y + 6), (x, y + h - 6), (x + w, y + h - 6)]
        for i, (cx, cy) in enumerate(corners):
            load = v.wheel_load[i]
            rad = int(np.clip(load / nominal, 0.15, 2.2) * 7) + 2
            col = self._gripcol(v.wheel_grip[i])
            pygame.draw.circle(self.screen, col, (cx, cy), rad)
            pygame.draw.circle(self.screen, (12, 12, 14), (cx, cy), rad, 1)
            self._txt(f"{load/1000:.1f}", cx - 8, cy + (8 if i >= 2 else -18), DIM, self.f_s)

    def _sparkline(self, x, y, w, h, hist, avg):
        pygame.draw.rect(self.screen, (20, 22, 28), (x, y, w - 116, h), border_radius=3)
        if len(hist) < 2:
            return
        data = hist[-60:]
        lo, hi = min(data), max(max(data), 1)
        rng = (hi - lo) or 1
        pts = [(x + int(i / (len(data) - 1) * (w - 118)),
                y + h - 2 - int((d - lo) / rng * (h - 4)))
               for i, d in enumerate(data)]
        pygame.draw.lines(self.screen, GREEN, False, pts, 2)

    def _minimap(self, sim, focus, x, y, w, h):
        tr = sim.track
        c = tr.center
        if self._map_bounds is None or self._map_bounds[0] is not tr:
            xs, ys = c[:, 0], c[:, 1]
            self._map_bounds = (tr, xs.min(), xs.max(), ys.min(), ys.max())
        _, xmin, xmax, ymin, ymax = self._map_bounds
        sx = w / (xmax - xmin + 1e-6)
        sy = h / (ymax - ymin + 1e-6)
        s = min(sx, sy)

        def to_px(px, py):
            return (int(x + (px - xmin) * s), int(y + (py - ymin) * s))

        pts = [to_px(p[0], p[1]) for p in c[::6]]
        pygame.draw.lines(self.screen, (90, 96, 110), True, pts, 1)
        for car in sim.cars:
            if car.alive:
                cp = to_px(car.vehicle.x, car.vehicle.y)
                col = YELLOW if car is focus else car.color
                pygame.draw.circle(self.screen, col, cp, 3 if car is focus else 2)
