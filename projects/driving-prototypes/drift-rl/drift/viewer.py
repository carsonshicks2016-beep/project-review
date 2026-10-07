"""Pygame renderer: the car, its smoke, and the combo meter it is risking."""

from __future__ import annotations

import math
import os
import random

import pygame

BG = (16, 17, 22)
FLOOR = (28, 30, 38)
LINE = (52, 56, 68)
WALL = (86, 92, 110)
CAR = (232, 236, 244)
CAR_DARK = (150, 158, 176)
TYRE = (24, 24, 28)
PILLAR = (236, 84, 64)
PILLAR_BAND = (236, 84, 64, 40)
HOT = (255, 176, 64)
COOL = (96, 208, 232)
TEXT = (226, 230, 240)
DIM = (128, 136, 152)

TRICK_COLOR = {
    "SWITCHBACK": (120, 216, 255),
    "DONUT": (255, 190, 80),
    "CLIP": (255, 110, 90),
    "MANJI": (190, 140, 255),
    "WALL KISS": (140, 255, 170),
    "BIG ANGLE": (255, 240, 120),
}


class Camera:
    """Fixed on the arena by default - drift *lines* are the thing worth seeing.

    Chase mode is there for close inspection of a single slide.
    """

    def __init__(self, w: int, h: int, arena_radius: float, chase: bool = False) -> None:
        self.w, self.h = w, h
        self.chase = chase
        self.scale = min(w, h) / ((1.0 if chase else 2.12) * arena_radius) / (2.0 if chase else 1.0)
        self.scale = (min(w, h) / (0.9 * arena_radius)) if chase else (min(w, h) / (2.12 * arena_radius))
        self.cx, self.cy = 0.0, 0.0

    def follow(self, x: float, y: float, blend: float = 0.08) -> None:
        if not self.chase:
            return
        self.cx += (x - self.cx) * blend
        self.cy += (y - self.cy) * blend

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        return (int(self.w / 2 + (x - self.cx) * self.scale),
                int(self.h / 2 - (y - self.cy) * self.scale))

    def px(self, metres: float) -> int:
        return max(1, int(metres * self.scale))


class Smoke:
    """Cheap particle puffs off whichever tyre is doing the most sliding."""

    def __init__(self, limit: int = 420) -> None:
        self.parts: list[list[float]] = []
        self.limit = limit

    def emit(self, x: float, y: float, intensity: float) -> None:
        n = int(min(4, intensity * 4))
        for _ in range(n):
            self.parts.append([x + random.uniform(-.5, .5), y + random.uniform(-.5, .5),
                               random.uniform(-1.4, 1.4), random.uniform(-1.4, 1.4),
                               1.0, random.uniform(0.6, 1.5)])
        if len(self.parts) > self.limit:
            del self.parts[:len(self.parts) - self.limit]

    def update(self, dt: float) -> None:
        for p in self.parts:
            p[0] += p[2] * dt
            p[1] += p[3] * dt
            p[5] += 3.2 * dt
            p[4] -= dt * 0.75
        self.parts = [p for p in self.parts if p[4] > 0.0]

    def draw(self, surf: pygame.Surface, cam: Camera) -> None:
        for x, y, _vx, _vy, life, rad in self.parts:
            sx, sy = cam.to_screen(x, y)
            r = cam.px(rad)
            if -r < sx < cam.w + r and -r < sy < cam.h + r:
                a = int(90 * life)
                puff = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(puff, (200, 205, 215, a), (r, r), r)
                surf.blit(puff, (sx - r, sy - r))

    def clear(self) -> None:
        self.parts.clear()


class Popup:
    __slots__ = ("text", "color", "x", "y", "life", "row")

    def __init__(self, text: str, color, x: float, y: float, row: int = 0) -> None:
        self.text, self.color, self.x, self.y, self.life = text, color, x, y, 1.0
        self.row = row


class Viewer:
    def __init__(self, width: int = 1100, height: int = 760, headless: bool = False,
                 title: str = "drift-rl", chase: bool = False) -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        pygame.init()
        pygame.display.set_caption(title)
        self.w, self.h = width, height
        self.screen = (pygame.Surface((width, height)) if headless
                       else pygame.display.set_mode((width, height)))
        self.headless = headless
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("menlo,dejavusansmono,monospace", 15)
        self.big = pygame.font.SysFont("menlo,dejavusansmono,monospace", 30, bold=True)
        self.huge = pygame.font.SysFont("menlo,dejavusansmono,monospace", 46, bold=True)
        self.chase = chase
        self.cam = Camera(width, height, 70.0, chase)
        self.smoke = Smoke()
        self.popups: list[Popup] = []
        self._trail: list[tuple[float, float]] = []

    def reset(self, env) -> None:
        self.cam = Camera(self.w, self.h, env.arena.cfg.radius, self.chase)
        if self.chase:
            self.cam.cx, self.cam.cy = env.state.x, env.state.y
        self.smoke.clear()
        self.popups.clear()
        self._trail.clear()

    # -- world -------------------------------------------------------------
    def _draw_arena(self, env) -> None:
        cam, arena = self.cam, env.arena
        c = cam.to_screen(0.0, 0.0)
        pygame.draw.circle(self.screen, FLOOR, c, cam.px(arena.cfg.radius))
        for r in range(10, int(arena.cfg.radius), 10):
            pygame.draw.circle(self.screen, LINE, c, cam.px(r), 1)
        pygame.draw.circle(self.screen, WALL, c, cam.px(arena.cfg.radius), 4)

        blocked = env.keeper.combo.last_clip
        for i, (px, py) in enumerate(arena.pillars):
            sp = cam.to_screen(px, py)
            band = cam.px(arena.cfg.pillar_radius + arena.cfg.clip_band)
            ring = pygame.Surface((band * 2, band * 2), pygame.SRCALPHA)
            live = i != blocked
            pygame.draw.circle(ring, (*PILLAR, 46 if live else 14), (band, band), band)
            pygame.draw.circle(ring, (*PILLAR, 130 if live else 40), (band, band), band, 2)
            self.screen.blit(ring, (sp[0] - band, sp[1] - band))
            pygame.draw.circle(self.screen, PILLAR if live else (110, 60, 55), sp,
                               cam.px(arena.cfg.pillar_radius))

    def _draw_trail(self) -> None:
        if len(self._trail) < 2:
            return
        pts = [self.cam.to_screen(x, y) for x, y in self._trail]
        for i in range(1, len(pts)):
            a = i / len(pts)
            col = (int(58 + 150 * a), int(62 + 150 * a), int(74 + 150 * a))
            pygame.draw.line(self.screen, col, pts[i - 1], pts[i], 2 if a < .6 else 3)

    def _draw_car(self, env) -> None:
        cam, cfg, s = self.cam, env.car_cfg, env.state
        c, sn = math.cos(s.yaw), math.sin(s.yaw)

        def body(px: float, py: float) -> tuple[int, int]:
            return cam.to_screen(s.x + px * c - py * sn, s.y + px * sn + py * c)

        hl, hw = cfg.body_len / 2, cfg.body_wid / 2
        shell = [body(hl, hw), body(hl, -hw), body(-hl, -hw), body(-hl, hw)]
        glow = pygame.Surface((self.w, self.h), pygame.SRCALPHA)
        pygame.draw.circle(glow, (255, 176, 64, 26), cam.to_screen(s.x, s.y), cam.px(4.0))
        self.screen.blit(glow, (0, 0))
        pygame.draw.polygon(self.screen, CAR, shell)
        pygame.draw.polygon(self.screen, CAR_DARK, shell, 2)
        # A nose flash so the heading is unambiguous when it is fully sideways.
        pygame.draw.polygon(self.screen, HOT, [body(hl, hw * .8), body(hl, -hw * .8),
                                               body(hl - .55, -hw * .8), body(hl - .55, hw * .8)])

        tw = cfg.track_width / 2
        for ax, steer in ((cfg.a, s.steer), (-cfg.b, 0.0)):
            for side in (tw, -tw):
                wc = body(ax, side)
                wa = s.yaw + steer
                wx, wy = math.cos(wa) * 0.34, math.sin(wa) * 0.34
                p1 = (wc[0] - wx * cam.scale, wc[1] + wy * cam.scale)
                p2 = (wc[0] + wx * cam.scale, wc[1] - wy * cam.scale)
                pygame.draw.line(self.screen, TYRE, p1, p2, max(2, cam.px(0.24)))

        # Velocity vector: the gap between it and the nose *is* the drift angle.
        course = s.yaw + s.beta
        tip = cam.to_screen(s.x + math.cos(course) * s.speed * 0.55,
                            s.y + math.sin(course) * s.speed * 0.55)
        pygame.draw.line(self.screen, COOL, cam.to_screen(s.x, s.y), tip, 2)

    def _emit_smoke(self, env) -> None:
        cfg, s = env.car_cfg, env.state
        _af, ar = env.model.slip_angles(s)
        kappa = abs(env.model.slip_ratio(s))
        intensity = max(0.0, min(1.5, abs(ar) / 0.5 + kappa * 0.35 - 0.25))
        if intensity <= 0 or s.speed < 2.0:
            return
        c, sn = math.cos(s.yaw), math.sin(s.yaw)
        for side in (cfg.track_width / 2, -cfg.track_width / 2):
            px, py = -cfg.b, side
            self.smoke.emit(s.x + px * c - py * sn, s.y + px * sn + py * c, intensity)

    # -- hud ---------------------------------------------------------------
    def _bar(self, x: int, y: int, w: int, h: int, frac: float, col) -> None:
        pygame.draw.rect(self.screen, (38, 41, 50), (x, y, w, h), border_radius=3)
        f = max(0.0, min(1.0, frac))
        if f > 0:
            pygame.draw.rect(self.screen, col, (x, y, int(w * f), h), border_radius=3)

    def _draw_hud(self, env, label: str = "") -> None:
        k, c, s = env.keeper, env.keeper.combo, env.state
        sc = env.score_cfg
        _af, ar = env.model.slip_angles(s)

        panel = pygame.Surface((300, 214), pygame.SRCALPHA)
        panel.fill((12, 13, 17, 205))
        self.screen.blit(panel, (14, 14))
        x0 = 30

        self.screen.blit(self.huge.render(f"{k.banked:,.0f}", True, TEXT), (x0, 22))
        self.screen.blit(self.font.render("BANKED", True, DIM), (x0, 74))

        live = c.active or c.pending > 0
        col = HOT if live else DIM
        self.screen.blit(self.big.render(f"+{c.pending:,.0f}", True, col), (x0, 96))
        self.screen.blit(self.font.render(f"x{c.multiplier:.2f}", True, col), (x0 + 150, 104))
        self._bar(x0, 132, 256, 7, (c.multiplier - 1) / (sc.mult_max - 1), col)

        ang = math.degrees(abs(ar))
        self.screen.blit(self.font.render(f"ANGLE {ang:5.1f}d", True, TEXT), (x0, 150))
        self._bar(x0, 168, 110, 6, ang / 60.0,
                  HOT if ang >= math.degrees(sc.angle_min) else DIM)
        self.screen.blit(self.font.render(f"SPEED {s.speed * 3.6:5.1f}k", True, TEXT), (x0 + 140, 150))
        self._bar(x0 + 140, 168, 116, 6, s.speed / 30.0, COOL)

        if c.active:
            grace = 1.0 - min(1.0, c.since_drift / sc.grace)
            self.screen.blit(self.font.render(f"COMBO {c.duration:4.1f}s", True, DIM), (x0, 186))
            self._bar(x0 + 110, 190, 146, 5, grace, (120, 216, 255))
        elif label:
            self.screen.blit(self.font.render(label, True, DIM), (x0, 186))

        counts = [(n, v) for n, v in k.trick_counts.items() if v]
        for i, (n, v) in enumerate(counts):
            t = self.font.render(f"{n} x{v}", True, TRICK_COLOR.get(n, DIM))
            self.screen.blit(t, (self.w - 150, 20 + i * 20))
        if k.wipeouts:
            self.screen.blit(self.font.render(f"WIPEOUTS x{k.wipeouts}", True, (255, 100, 100)),
                             (self.w - 150, 24 + len(counts) * 20))
        if label:
            self.screen.blit(self.font.render(label, True, DIM), (16, self.h - 26))

    def _draw_popups(self, dt: float) -> None:
        for p in self.popups:
            p.life -= dt * 0.75
            p.y += dt * 5.0
        self.popups = [p for p in self.popups if p.life > 0]
        for p in self.popups:
            sx, sy = self.cam.to_screen(p.x, p.y)
            surf = self.big.render(p.text, True, p.color)
            surf.set_alpha(int(255 * min(1.0, p.life * 1.6)))
            self.screen.blit(surf, (sx - surf.get_width() // 2, sy - 30 - p.row * 32))

    # -- frame -------------------------------------------------------------
    def draw(self, env, label: str = "") -> pygame.Surface:
        dt = env.car_cfg.dt
        s = env.state
        self.cam.follow(s.x, s.y)
        self._trail.append((s.x, s.y))
        if len(self._trail) > 420:
            self._trail.pop(0)
        fresh = sum(1 for q in self.popups if q.life > 0.82)
        for j, e in enumerate(env.last_events):
            self.popups.append(Popup(e.name, TRICK_COLOR.get(e.name, TEXT), e.x, e.y,
                                     fresh + j))
        self._emit_smoke(env)
        self.smoke.update(dt)

        self.screen.fill(BG)
        self._draw_arena(env)
        self._draw_trail()
        self.smoke.draw(self.screen, self.cam)
        self._draw_car(env)
        self._draw_popups(dt)
        self._draw_hud(env, label)
        return self.screen

    def flip(self, fps: int = 25) -> bool:
        """Present the frame. Returns False if the user closed the window."""
        if self.headless:
            return True
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return False
        pygame.display.flip()
        self.clock.tick(fps)
        return True

    def frame_rgb(self) -> "list":
        import numpy as np
        return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))

    def close(self) -> None:
        pygame.quit()


_SHARED: Viewer | None = None


def render_frame(env):
    """Backs ``DriftEnv.render()`` for rgb_array capture."""
    global _SHARED
    if _SHARED is None:
        _SHARED = Viewer(headless=True)
        _SHARED.reset(env)
    _SHARED.draw(env)
    return _SHARED.frame_rgb()
