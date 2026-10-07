"""
Pygame front-end: renders the track + field of Supras with a chase camera,
tyre smoke and skid marks, drives the sound, and runs the evolution loop.

Also provides train_headless() for fast (no-render) evolution.
"""

from __future__ import annotations
import math
import os
from collections import deque

import numpy as np
import pygame

from .config import Config
from .simulation import Simulation
from .evolution import Evolution
from .dashboard import Dashboard
from .sound import SoundEngine

HELP = ("[SPACE] pause  [F] car  [A] follow  [C] camera  [V] slow-mo  [M] mute  "
        "[T] fast  [R] track  [+/-] zoom  [S] save  [ESC] quit")


class App:
    def __init__(self, cfg: Config, seed=None, save_path="best_supra.npz"):
        self.cfg = cfg
        self.save_path = save_path
        pygame.init()
        pygame.display.set_caption("Supra-AI · neuro-evolution driving sim")
        self.screen = pygame.display.set_mode((cfg.render.width, cfg.render.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 14)
        self.font_b = pygame.font.SysFont("menlo,consolas,monospace", 20, bold=True)

        self.evo = Evolution(cfg, seed=seed)
        self.sim = Simulation(cfg, seed=seed)
        self.sim.set_population(self.evo.brains, self.evo.colors())
        self.dash = Dashboard(cfg)
        self.sound = SoundEngine(cfg.car.idle_rpm, cfg.car.redline_rpm,
                                 cfg.car.max_boost_bar)

        self.view_w = cfg.render.width - cfg.render.dash_width
        self.view_h = cfg.render.height
        self.mpp = cfg.render.meters_per_pixel_default   # metres per pixel (zoom)
        self.cam = [0.0, 0.0]
        self.paused = False
        self.auto_follow = True
        self.muted = False
        self.fast = 1
        self.evolve_enabled = True     # off when watching a fixed (e.g. PPO) policy
        self.reload_ppo_path = None     # set to hot-reload a PPO policy as it trains
        self._last_reload = 0
        self.smoke = deque(maxlen=600)
        self.skid = deque(maxlen=2500)
        self.flames = deque(maxlen=200)     # exhaust backfire pops
        self._prev_focus = -1

        # ---- cinematic camera + auto slow-mo ----
        self.cam_angle = 0.0                       # view rotation (radians)
        self.user_mpp = self.mpp                   # the zoom the user picked
        self.CAM_MODES = ["CHASE", "CINEMATIC", "DRIFT CAM", "BROADCAST"]
        self.cam_mode = 0
        self._prev_mode = 0
        self.time_scale = 1.0                      # 1.0 normal, <1 slow-mo
        self.slowmo_auto = False
        self._slowmo_t = 0.0
        self._slowmo_cd = 0.0
        self._bcast = None                         # (track, anchor points)
        self._bcast_active = 0
        self._bcast_prev = -1

        # ---- Mk4 Supra top-view silhouette (metres, +x fwd, +y left) ----
        self.SUPRA_BODY = [
            (2.30, 0.0), (2.22, 0.34), (2.02, 0.60), (1.55, 0.78),
            (0.55, 0.86), (-0.65, 0.86), (-1.55, 0.80), (-1.95, 0.60),
            (-2.12, 0.34), (-2.12, -0.34), (-1.95, -0.60), (-1.55, -0.80),
            (-0.65, -0.86), (0.55, -0.86), (1.55, -0.78), (2.02, -0.60),
            (2.22, -0.34)]
        # tighter greenhouse (long-hood / cab-rearward coupe proportions)
        self.SUPRA_ROOF = [
            (0.55, 0.40), (0.05, 0.54), (-0.85, 0.52), (-1.12, 0.34),
            (-1.12, -0.34), (-0.85, -0.52), (0.05, -0.54), (0.55, -0.40)]
        self.SUPRA_WSHIELD = [(0.55, 0.40), (0.05, 0.50), (0.05, -0.50), (0.55, -0.40)]
        # the iconic tall rear wing
        self.SUPRA_WING = [(-1.95, 0.94), (-2.34, 0.94), (-2.34, -0.94), (-1.95, -0.94)]
        self.SUPRA_WING_STAY = [(-1.8, 0.18), (-2.2, 0.18), (-2.2, -0.18), (-1.8, -0.18)]
        self.SUPRA_TAIL = [
            (-1.92, 0.56), (-2.10, 0.30), (-2.10, -0.30), (-1.92, -0.56)]
        self.SUPRA_HL_L = [(2.22, 0.32), (2.04, 0.54), (1.82, 0.47), (2.0, 0.28)]
        self.SUPRA_HL_R = [(x, -y) for x, y in self.SUPRA_HL_L]

        # scale the silhouette to the selected car's footprint (Supra = reference)
        _sx, _sy = cfg.car.body_length / 4.51, cfg.car.body_width / 1.81
        def _scale(pts):
            return [(px * _sx, py * _sy) for px, py in pts]
        self.SUPRA_BODY = _scale(self.SUPRA_BODY)
        self.SUPRA_ROOF = _scale(self.SUPRA_ROOF)
        self.SUPRA_WSHIELD = _scale(self.SUPRA_WSHIELD)
        self.SUPRA_WING = _scale(self.SUPRA_WING)
        self.SUPRA_WING_STAY = _scale(self.SUPRA_WING_STAY)
        self.SUPRA_TAIL = _scale(self.SUPRA_TAIL)
        self.SUPRA_HL_L = _scale(self.SUPRA_HL_L)
        self.SUPRA_HL_R = _scale(self.SUPRA_HL_R)

    # ---- camera ------------------------------------------------------- #
    @staticmethod
    def _world_vel(car):
        v = car.vehicle
        c, s = math.cos(v.yaw), math.sin(v.yaw)
        return (v.vx * c - v.vy * s, v.vx * s + v.vy * c)

    def w2s(self, px, py):
        s = 1.0 / self.mpp
        dx, dy = px - self.cam[0], py - self.cam[1]
        if self.cam_angle:                     # rotate the world about the camera
            ca, sa = math.cos(self.cam_angle), math.sin(self.cam_angle)
            dx, dy = dx * ca - dy * sa, dx * sa + dy * ca
        sx = self.view_w / 2 + dx * s
        sy = self.view_h / 2 - dy * s
        return sx, sy

    def in_view(self, px, py, margin=80):
        sx, sy = self.w2s(px, py)
        return -margin < sx < self.view_w + margin and -margin < sy < self.view_h + margin

    # ---- drawing ------------------------------------------------------ #
    def _tf(self, pts, cx, cy, cos, sin):
        return [self.w2s(cx + bx * cos - by * sin, cy + bx * sin + by * cos)
                for bx, by in pts]

    def _draw_track(self):
        tr = self.sim.track
        R = self.cfg.render
        N = tr.N
        for i in range(0, N, 1):
            j = (i + 1) % N
            if not (self.in_view(*tr.center[i]) or self.in_view(*tr.center[j])):
                continue
            col = R.track_color if (i // 6) % 2 == 0 else R.track_color2
            pygame.draw.polygon(self.screen, col,
                                [self.w2s(*tr.left[i]), self.w2s(*tr.right[i]),
                                 self.w2s(*tr.right[j]), self.w2s(*tr.left[j])])
            # red/white kerbs on the corners
            if tr.difficulty[i] > 0.28:
                kc = R.kerb_color1 if (i // 3) % 2 == 0 else R.kerb_color2
                li, lj = tr.left[i] - tr.normal[i] * 1.1, tr.left[j] - tr.normal[j] * 1.1
                pygame.draw.polygon(self.screen, kc, [self.w2s(*tr.left[i]),
                                    self.w2s(*li), self.w2s(*lj), self.w2s(*tr.left[j])])
                ri, rj = tr.right[i] + tr.normal[i] * 1.1, tr.right[j] + tr.normal[j] * 1.1
                pygame.draw.polygon(self.screen, kc, [self.w2s(*tr.right[i]),
                                    self.w2s(*ri), self.w2s(*rj), self.w2s(*tr.right[j])])
        # skid marks
        for sx, sy in self.skid:
            if self.in_view(sx, sy):
                pygame.draw.circle(self.screen, (34, 35, 41), self.w2s(sx, sy), 2)
        # white edge lines
        for arr in (tr.left, tr.right):
            pts = [self.w2s(*arr[i]) for i in range(0, N, 2)]
            pygame.draw.lines(self.screen, R.track_edge, True, pts, 2)
        self._draw_startline(tr)

    def _draw_startline(self, tr):
        a = np.array(tr.left[0]); b = np.array(tr.right[0])
        for k in range(8):
            p0 = a + (b - a) * (k / 8.0)
            p1 = a + (b - a) * ((k + 1) / 8.0)
            fwd = tr.tangent[0] * 1.6
            col = (235, 238, 244) if k % 2 == 0 else (30, 30, 34)
            pygame.draw.polygon(self.screen, col, [self.w2s(*p0), self.w2s(*p1),
                                self.w2s(*(p1 + fwd)), self.w2s(*(p0 + fwd))])

    def _draw_car(self, car, focus):
        v = car.vehicle
        if not self.in_view(v.x, v.y):
            return
        R = self.cfg.render
        cos, sin = math.cos(v.yaw), math.sin(v.yaw)
        scx, scy = self.w2s(v.x, v.y)

        base = car.color if car.alive else tuple(c // 3 for c in car.color)
        col = R.focus_color if focus else base
        dark = tuple(max(0, c - 70) for c in col)

        # focus glow
        if focus and car.alive:
            rg = max(10, int(3.4 / self.mpp))
            s = pygame.Surface((rg * 2, rg * 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (255, 215, 60, 45), (rg, rg), rg)
            self.screen.blit(s, (int(scx) - rg, int(scy) - rg))

        # drop shadow (offset slightly in world space for depth)
        pygame.draw.polygon(self.screen, (10, 11, 14),
                            self._tf(self.SUPRA_BODY, v.x - 0.35, v.y - 0.35, cos, sin))

        # wheels (fronts steer) -- drawn first so the body overlaps the inner half
        lf, lr = self.cfg.car.lf, self.cfg.car.lr
        wp = [(0.38, 0.14), (0.38, -0.14), (-0.38, -0.14), (-0.38, 0.14)]
        for ax, ay, st in [(lf, 0.86, v.steer), (lf, -0.86, v.steer),
                           (-lr, 0.86, 0.0), (-lr, -0.86, 0.0)]:
            wcx = v.x + ax * cos - ay * sin
            wcy = v.y + ax * sin + ay * cos
            wc, ws = math.cos(v.yaw + st), math.sin(v.yaw + st)
            pygame.draw.polygon(self.screen, (24, 24, 28), self._tf(wp, wcx, wcy, wc, ws))

        body_pts = self._tf(self.SUPRA_BODY, v.x, v.y, cos, sin)
        # rear wing stay + wing (drawn under/behind the body tail)
        pygame.draw.polygon(self.screen, dark, self._tf(self.SUPRA_WING_STAY, v.x, v.y, cos, sin))
        # body + outline
        pygame.draw.polygon(self.screen, col, body_pts)
        pygame.draw.polygon(self.screen, dark, body_pts, 1)
        # taillight bar
        pygame.draw.polygon(self.screen, (185, 28, 26), self._tf(self.SUPRA_TAIL, v.x, v.y, cos, sin))
        # greenhouse: glass + lighter windshield
        pygame.draw.polygon(self.screen, (28, 31, 39), self._tf(self.SUPRA_ROOF, v.x, v.y, cos, sin))
        pygame.draw.polygon(self.screen, (70, 84, 104), self._tf(self.SUPRA_WSHIELD, v.x, v.y, cos, sin))
        # headlights
        for hl in (self.SUPRA_HL_L, self.SUPRA_HL_R):
            pygame.draw.polygon(self.screen, (245, 245, 215), self._tf(hl, v.x, v.y, cos, sin))
        # the iconic rear wing on top
        wing = self._tf(self.SUPRA_WING, v.x, v.y, cos, sin)
        pygame.draw.polygon(self.screen, col, wing)
        pygame.draw.polygon(self.screen, dark, wing, 1)

    def _emit_effects(self, car):
        v = car.vehicle
        if not car.alive:
            return
        cos, sin = math.cos(v.yaw), math.sin(v.yaw)
        lr = self.cfg.car.lr
        # --- tyre smoke + twin skid lines (when sliding) ---
        if v.speed > 5 and (v.wheelspin > 0.92 or v.grip_r > 0.96 or abs(v.slip_angle) > 0.25):
            for by in (0.78, -0.78):
                self.skid.append((v.x - lr * cos - by * sin, v.y - lr * sin + by * cos))
            by = 0.78 if np.random.random() < 0.5 else -0.78
            if len(self.smoke) < self.smoke.maxlen:
                self.smoke.append({
                    "x": v.x - lr * cos - by * sin, "y": v.y - lr * sin + by * cos,
                    "vx": np.random.uniform(-1.2, 1.2), "vy": np.random.uniform(-1.2, 1.2),
                    "life": 1.0, "size": np.random.uniform(0.30, 0.55)})
        # --- exhaust backfire: on a gearshift, or lifting off while on boost ---
        thr = car.controls[1]
        lift = getattr(car, "_bf_prev_thr", 0.0) > 0.55 and thr < 0.2
        car._bf_prev_thr = thr
        cd = max(0.0, getattr(car, "_bf_cd", 0.0) - self.cfg.sim.dt)
        if v.speed > 3 and cd <= 0 and (v.shift_flash > 0.5 or (lift and v.boost > 0.25)):
            # exhaust tip at the back-right of the car
            ex = v.x - 2.30 * cos + 0.45 * sin
            ey = v.y - 2.30 * sin - 0.45 * cos
            for _ in range(3):
                spread = np.random.uniform(-0.5, 0.5)
                spd = np.random.uniform(3.0, 7.0)
                self.flames.append({
                    "x": ex, "y": ey,
                    "vx": -cos * spd - sin * spread, "vy": -sin * spd + cos * spread,
                    "life": 1.0, "size": np.random.uniform(0.4, 0.7)})
            cd = 0.16
        car._bf_cd = cd

    def _update_smoke(self, dt):
        for p in self.smoke:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["life"] -= dt * 1.1
        while self.smoke and self.smoke[0]["life"] <= 0:
            self.smoke.popleft()
        for p in self.flames:                      # backfire pops decay fast & bright
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["life"] -= dt * 7.0
        while self.flames and self.flames[0]["life"] <= 0:
            self.flames.popleft()

    def _draw_smoke(self):
        for p in self.smoke:
            if p["life"] <= 0 or not self.in_view(p["x"], p["y"]):
                continue
            age = 1.0 - p["life"]                       # puffs billow out as they age
            a = int(130 * p["life"])
            r = max(1, int(p["size"] * (1.0 + age * 1.6) / self.mpp))
            s = pygame.Surface((r * 2 + 2, r * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(s, (205, 207, 212, a), (r + 1, r + 1), r)
            sx, sy = self.w2s(p["x"], p["y"])
            self.screen.blit(s, (int(sx) - r, int(sy) - r))   # centre on the world point

    def _draw_flames(self):
        for p in self.flames:
            if p["life"] <= 0 or not self.in_view(p["x"], p["y"]):
                continue
            life = p["life"]
            r = max(2, int(p["size"] / self.mpp))
            s = pygame.Surface((r * 4, r * 4), pygame.SRCALPHA)
            c = (r * 2, r * 2)
            pygame.draw.circle(s, (255, 110, 20, int(140 * life)), c, r * 2)   # orange glow
            pygame.draw.circle(s, (255, 200, 70, int(210 * life)), c, r)       # yellow
            pygame.draw.circle(s, (255, 255, 230, int(235 * life)), c, max(1, r // 2))  # white core
            sx, sy = self.w2s(p["x"], p["y"])
            self.screen.blit(s, (int(sx) - r * 2, int(sy) - r * 2))

    # ---- loop --------------------------------------------------------- #
    def _advance(self, dt):
        for _ in range(self.fast):
            done = self.sim.step(dt)
            if self.auto_follow:
                self.sim.focus_leader()
            for c in self.sim.cars:
                self._emit_effects(c)
            self._update_smoke(dt)
            if done:
                if self.evolve_enabled:
                    self._evolve()
                else:
                    self.sim.reset_episode()
                    self.smoke.clear(); self.skid.clear()
                break

    def _evolve(self):
        fits = [c.fitness for c in self.sim.cars]
        self.evo.next_generation(fits)
        self.sim.generation += 1
        # curriculum: rotate to a fresh track every few generations
        cur = self.cfg.curriculum
        if cur.enabled and self.sim.generation % cur.ga_rotate_generations == 0:
            from .track import sample_kind, ga_stage_for_gen
            kind = sample_kind(self.evo.rng, ga_stage_for_gen(self.sim.generation))
            self.sim.new_track(kind=kind)
        self.sim.set_population(self.evo.brains, self.evo.colors())
        self.smoke.clear(); self.skid.clear()

    def _events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                return False
            if e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                elif e.key == pygame.K_SPACE:
                    self.paused = not self.paused
                elif e.key == pygame.K_f:
                    self.auto_follow = False; self.sim.cycle_focus()
                elif e.key == pygame.K_a:
                    self.auto_follow = not self.auto_follow
                elif e.key == pygame.K_m:
                    self.muted = not self.muted
                elif e.key == pygame.K_t:
                    self.fast = 1 if self.fast > 1 else 8
                elif e.key == pygame.K_c:
                    self.cam_mode = (self.cam_mode + 1) % len(self.CAM_MODES)
                elif e.key == pygame.K_v:
                    self.slowmo_auto = not self.slowmo_auto
                elif e.key == pygame.K_r:
                    self.sim.new_track(); self.sim.reset_episode()
                    self.smoke.clear(); self.skid.clear()
                elif e.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.user_mpp = max(0.08, self.user_mpp * 0.8); self.mpp = self.user_mpp
                elif e.key == pygame.K_MINUS:
                    self.user_mpp = min(1.2, self.user_mpp * 1.25); self.mpp = self.user_mpp
                elif e.key == pygame.K_s:
                    if self.evo.best_genome is not None:
                        self.evo.best_genome.save(self.save_path)
                        print(f"saved best (fitness {self.evo.best_fitness:.0f}) "
                              f"-> {self.save_path}")
        return True

    def _broadcast_anchor(self, fc):
        """Pick a trackside vantage; cut to a new one (with hysteresis) as the
        car sweeps past."""
        tr = self.sim.track
        if self._bcast is None or self._bcast[0] is not tr:
            n = 10
            idxs = [int(k * tr.N / n) for k in range(n)]
            pts = [(tr.center[i, 0] + tr.normal[i, 0] * (tr.half_width + 26.0),
                    tr.center[i, 1] + tr.normal[i, 1] * (tr.half_width + 26.0))
                   for i in idxs]
            self._bcast = (tr, np.array(pts))
            self._bcast_active = 0
        pts = self._bcast[1]
        d = (pts[:, 0] - fc.vehicle.x) ** 2 + (pts[:, 1] - fc.vehicle.y) ** 2
        nearest = int(np.argmin(d))
        if nearest != self._bcast_active and d[nearest] < d[self._bcast_active] * 0.7:
            self._bcast_active = nearest        # cut to a clearly-closer vantage
        return float(pts[self._bcast_active, 0]), float(pts[self._bcast_active, 1])

    def _update_camera(self, fc, dt):
        if fc is None:
            return
        mode = self.CAM_MODES[self.cam_mode]
        vx, vy = self._world_vel(fc)
        speed = math.hypot(vx, vy)
        cut = False
        if mode == "BROADCAST":
            ax, ay = self._broadcast_anchor(fc)
            tx = fc.vehicle.x * 0.6 + ax * 0.4
            ty = fc.vehicle.y * 0.6 + ay * 0.4
            tangle, tmpp = 0.0, self.user_mpp * 1.5
            cut = self._bcast_active != self._bcast_prev
            self._bcast_prev = self._bcast_active
        else:
            self._bcast_prev = -1
            if mode == "CINEMATIC":
                lead = min(speed * 0.25, 12.0)
                tx = fc.vehicle.x + math.cos(fc.vehicle.yaw) * lead
                ty = fc.vehicle.y + math.sin(fc.vehicle.yaw) * lead
                tangle, tmpp = math.pi / 2 - fc.vehicle.yaw, self.user_mpp * 0.92
            elif mode == "DRIFT CAM":
                course = math.atan2(vy, vx) if speed > 1.0 else fc.vehicle.yaw
                lead = min(speed * 0.3, 14.0)
                tx = fc.vehicle.x + math.cos(course) * lead
                ty = fc.vehicle.y + math.sin(course) * lead
                tangle, tmpp = math.pi / 2 - course, self.user_mpp * 0.85
            else:  # CHASE
                tx, ty = fc.vehicle.x, fc.vehicle.y
                tangle, tmpp = 0.0, self.user_mpp

        if (self.sim.focus != self._prev_focus) or (self.cam_mode != self._prev_mode) or cut:
            self.cam[0], self.cam[1], self.cam_angle = tx, ty, tangle
            self._prev_focus, self._prev_mode = self.sim.focus, self.cam_mode
        else:
            self.cam[0] += (tx - self.cam[0]) * 0.12
            self.cam[1] += (ty - self.cam[1]) * 0.12
            da = (tangle - self.cam_angle + math.pi) % (2 * math.pi) - math.pi
            self.cam_angle += da * 0.10
        self.mpp += (tmpp - self.mpp) * 0.08

    def run(self):
        self.sound.start()
        running = True
        while running:
            dt = self.cfg.sim.dt
            running = self._events()
            # hot-reload a PPO policy as it trains (watch live training)
            if self.reload_ppo_path is not None:
                now = pygame.time.get_ticks()
                if now - self._last_reload > 4000:
                    self._last_reload = now
                    for c in self.sim.cars:
                        if hasattr(c.brain, "reload"):
                            c.brain.reload(self.reload_ppo_path)
            # auto slow-mo: briefly slow time on a big slide (drama)
            fc = self.sim.focus_car
            self._slowmo_cd = max(0.0, self._slowmo_cd - dt)
            if (self.slowmo_auto and fc is not None and fc.alive
                    and abs(fc.vehicle.slip_angle) > 0.5 and fc.vehicle.speed > 12.0
                    and self._slowmo_cd <= 0.0):
                self._slowmo_t, self._slowmo_cd = 0.7, 2.5
            self._slowmo_t = max(0.0, self._slowmo_t - dt)
            self.time_scale += ((0.35 if self._slowmo_t > 0 else 1.0) - self.time_scale) * 0.25
            if not self.paused:
                self._advance(dt * self.time_scale)
            self._update_camera(fc, dt)
            self.sound.update(self.sim.cars, self.sim.focus, self.cam,
                              self.view_w / 2 * self.mpp, muted=self.muted)

            # render
            self.screen.fill(self.cfg.render.grass_color)
            self._draw_track()
            self._draw_smoke()
            for c in self.sim.cars:
                self._draw_car(c, focus=(c is fc))
            self._draw_flames()
            # HUD
            self.screen.blit(self.font.render(HELP, True, (140, 146, 158)), (12, self.view_h - 24))
            self.screen.blit(self.font.render(f"track: {self.sim.track.kind}", True,
                             (140, 146, 158)), (12, self.view_h - 44))
            if self.paused:
                self.screen.blit(self.font_b.render("PAUSED", True, (240, 200, 70)), (12, 12))
            if self.fast > 1:
                self.screen.blit(self.font_b.render(f"FAST x{self.fast}", True, (90, 210, 120)),
                                 (12, 36))
            if self.muted:
                self.screen.blit(self.font.render("MUTED", True, (230, 70, 60)), (12, 64))
            self.screen.blit(self.font.render(
                f"cam: {self.CAM_MODES[self.cam_mode]}"
                + ("   slow-mo:ON" if self.slowmo_auto else ""),
                True, (140, 146, 158)), (12, self.view_h - 64))
            if self._slowmo_t > 0:
                self.screen.blit(self.font_b.render("◐ SLOW-MO", True,
                                 (120, 200, 255)), (12, 112))
            if self.reload_ppo_path is not None:
                self.screen.blit(self.font_b.render("● LIVE PPO TRAINING", True,
                                 (90, 210, 120)), (12, 88))
            self.dash.render(self.screen, self.sim, self.evo)
            pygame.display.flip()
            self.clock.tick(self.cfg.render.fps)

        self.sound.stop()
        pygame.quit()


# ---------------------------------------------------------------------- #
def train_headless(cfg: Config, generations: int, seed=None,
                   save_path="best_supra.npz", quiet=False):
    """Evolve with no rendering or sound -- as fast as the CPU allows.

    Rotates to a fresh (and progressively harder) track every few generations
    so brains learn to *drive*, not memorise one circuit."""
    from .track import sample_kind, ga_stage_for_gen
    evo = Evolution(cfg, seed=seed)
    sim = Simulation(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    cur = cfg.curriculum
    dt = cfg.sim.dt
    for gen in range(generations):
        if cur.enabled and gen % cur.ga_rotate_generations == 0:
            sim.new_track(kind=sample_kind(rng, ga_stage_for_gen(gen)))
        sim.set_population(evo.brains, evo.colors())
        while True:
            if sim.step(dt):
                break
        fits = [c.fitness for c in sim.cars]
        evo.next_generation(fits)
        sim.generation += 1
        if not quiet:
            print(f"gen {gen:4d}  best {evo.best_fitness:9.1f}  "
                  f"this-gen {max(fits):9.1f}  avg {np.mean(fits):8.1f}  "
                  f"track {sim.track.kind}")
    if evo.best_genome is not None:
        evo.best_genome.save(save_path)
        if not quiet:
            print(f"\nsaved best (fitness {evo.best_fitness:.0f}) -> {save_path}")
    return evo
