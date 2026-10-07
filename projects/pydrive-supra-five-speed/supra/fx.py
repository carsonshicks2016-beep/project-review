"""
Particle / visual-effects engine for the watch view (cosmetic only — it never
touches physics, the env, or training). World-space particles so they stay put
while the chase camera moves: tyre smoke, off-track dust, curb/wall sparks, and
exhaust backfire flames.

Render order is the caller's job; typically: heat-trail -> car -> fx.draw().
"""
from __future__ import annotations

import math
import random


# --------------------------------------------------------------------------- #
# soft radial sprites (built once, scaled per particle)
# --------------------------------------------------------------------------- #
_BASE = {}


def _soft_sprite(pygame, radius=40):
    """A white radial-gradient blob with per-pixel alpha (soft edges)."""
    key = ("soft", radius)
    if key in _BASE:
        return _BASE[key]
    d = radius * 2
    s = pygame.Surface((d, d), pygame.SRCALPHA)
    for r in range(radius, 0, -1):
        a = int(255 * (1.0 - r / radius) ** 1.7)
        pygame.draw.circle(s, (255, 255, 255, a), (radius, radius), r)
    _BASE[key] = s
    return s


class Particle:
    __slots__ = ("x", "y", "vx", "vy", "age", "life", "r0", "r1",
                 "col", "kind", "drag", "spin", "rot")

    def __init__(self, x, y, vx, vy, life, r0, r1, col, kind, drag):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.age, self.life = 0.0, life
        self.r0, self.r1 = r0, r1          # world-unit radius start/end
        self.col = col
        self.kind = kind                    # smoke|dust|spark|flame
        self.drag = drag
        self.rot = random.uniform(0, 360)
        self.spin = random.uniform(-60, 60)


class FX:
    MAX = 900

    def __init__(self):
        self.parts: list[Particle] = []

    # -- emitters ---------------------------------------------------------- #
    def _add(self, p):
        if len(self.parts) < self.MAX:
            self.parts.append(p)

    def emit_smoke(self, x, y, vx, vy, intensity):
        n = 1 + int(intensity * 2)
        for _ in range(n):
            sp = random.uniform(0.6, 1.4)
            g = random.randint(180, 235)
            self._add(Particle(
                x + random.uniform(-0.3, 0.3), y + random.uniform(-0.3, 0.3),
                vx * 0.35 + random.uniform(-1.2, 1.2),
                vy * 0.35 + random.uniform(-1.2, 1.2),
                life=random.uniform(0.7, 1.3) * (0.7 + intensity),
                r0=0.6 * sp, r1=(2.6 + 2.2 * intensity) * sp,
                col=(g, g, g), kind="smoke", drag=2.2))

    def emit_dust(self, x, y, vx, vy, intensity):
        for _ in range(1 + int(intensity * 2)):
            self._add(Particle(
                x + random.uniform(-0.3, 0.3), y + random.uniform(-0.3, 0.3),
                vx * 0.4 + random.uniform(-1.5, 1.5),
                vy * 0.4 + random.uniform(-1.5, 1.5),
                life=random.uniform(0.5, 0.9),
                r0=0.5, r1=2.0 + 1.6 * intensity,
                col=(150, 116, 78), kind="dust", drag=3.0))

    def emit_sparks(self, x, y, vx, vy, n=6):
        for _ in range(n):
            ang = random.uniform(0, math.tau)
            sp = random.uniform(6, 22)
            self._add(Particle(
                x, y,
                vx * 0.3 + math.cos(ang) * sp,
                vy * 0.3 + math.sin(ang) * sp,
                life=random.uniform(0.18, 0.42),
                r0=0.35, r1=0.05,
                col=(255, random.randint(170, 220), 90), kind="spark", drag=4.0))

    def emit_flame(self, x, y, vx, vy, intensity=1.0):
        for _ in range(2 + int(intensity * 3)):
            self._add(Particle(
                x + random.uniform(-0.2, 0.2), y + random.uniform(-0.2, 0.2),
                vx + random.uniform(-1.5, 1.5), vy + random.uniform(-1.5, 1.5),
                life=random.uniform(0.08, 0.18),
                r0=0.7 * intensity, r1=0.15,
                col=(255, random.randint(120, 190), 50), kind="flame", drag=3.0))
        # a wisp of smoke trailing the flame
        self.emit_smoke(x, y, vx, vy, 0.3)

    # -- sim --------------------------------------------------------------- #
    def update(self, dt):
        alive = []
        for p in self.parts:
            p.age += dt
            if p.age >= p.life:
                continue
            k = 1.0 / (1.0 + p.drag * dt)
            p.vx *= k
            p.vy *= k
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.rot += p.spin * dt
            alive.append(p)
        self.parts = alive

    # -- render ------------------------------------------------------------ #
    def draw(self, pygame, screen, to_screen, scale):
        soft = _soft_sprite(pygame)
        for p in self.parts:
            f = p.age / p.life
            r_world = p.r0 + (p.r1 - p.r0) * f
            px = max(2, int(r_world * scale * 2))
            sx, sy = to_screen(p.x, p.y)
            if sx < -px or sy < -px or sx > screen.get_width() + px or sy > screen.get_height() + px:
                continue
            if p.kind in ("smoke", "dust"):
                a = int(150 * (1.0 - f) ** 1.3)
                img = pygame.transform.smoothscale(soft, (px, px))
                img.fill((*p.col, a), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(img, (sx - px // 2, sy - px // 2))
            else:  # spark / flame -> additive glow
                a = int(255 * (1.0 - f))
                img = pygame.transform.smoothscale(soft, (px, px))
                img.fill((*p.col, a), special_flags=pygame.BLEND_RGBA_MULT)
                screen.blit(img, (sx - px // 2, sy - px // 2),
                            special_flags=pygame.BLEND_RGB_ADD)


# --------------------------------------------------------------------------- #
# heat-trail: skid marks coloured by how hard the car was sliding (drift "heat")
# --------------------------------------------------------------------------- #
def heat_color(heat):
    """0 = cold grey faint mark, 1 = glowing hot orange/white drift score."""
    heat = max(0.0, min(1.0, heat))
    if heat < 0.5:
        t = heat / 0.5                      # dark rubber -> orange
        r = int(28 + (235 - 28) * t)
        g = int(28 + (120 - 28) * t)
        b = int(30 + (40 - 30) * t)
    else:
        t = (heat - 0.5) / 0.5              # orange -> hot yellow/white
        r = 235 + int((255 - 235) * t)
        g = 120 + int((235 - 120) * t)
        b = 40 + int((180 - 40) * t)
    return (r, g, b)


def _blend(a, b, t):
    t = max(0.0, min(1.0, t))
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


# --------------------------------------------------------------------------- #
# persistent skid marks: rubber laid down by the rear tyres while sliding, in
# world space, coloured by slide "heat" and fading over a few seconds. Two
# strokes (rear-left / rear-right); a `lift()` breaks the stroke when grip
# returns so marks don't connect across non-drift gaps.
# --------------------------------------------------------------------------- #
class SkidTrail:
    MAX_PTS = 700          # per wheel; ~12 s of continuous sliding at 60 fps
    FADE = 6.0             # seconds for a mark to fade to nothing

    def __init__(self):
        self.strokes = {2: [], 3: []}       # rear-left (RL=2), rear-right (RR=3)

    def add(self, wheel, x, y, heat):
        s = self.strokes[wheel]
        s.append([x, y, max(0.0, min(1.0, heat)), 0.0])
        if len(s) > self.MAX_PTS:
            del s[:len(s) - self.MAX_PTS]

    def lift(self):
        """Break both strokes (grip returned / off the gas) so the next slide
        starts a fresh mark instead of a straight line across the gap."""
        for w in self.strokes:
            if self.strokes[w] and self.strokes[w][-1] is not None:
                self.strokes[w].append(None)

    def update(self, dt):
        for w, s in self.strokes.items():
            keep = []
            for p in s:
                if p is None:
                    keep.append(p)
                    continue
                p[3] += dt
                if p[3] < self.FADE:
                    keep.append(p)
            # drop a leading break
            while keep and keep[0] is None:
                keep.pop(0)
            self.strokes[w] = keep

    def draw(self, pygame, screen, to_screen, scale, bg=(24, 26, 30)):
        wmark = max(1, int(scale * 0.13))
        for s in self.strokes.values():
            prev = None
            for p in s:
                if p is None:
                    prev = None
                    continue
                if prev is not None:
                    fade = 1.0 - prev[3] / self.FADE          # 1 fresh -> 0 gone
                    col = _blend(bg, heat_color(prev[2]), 0.85 * fade)
                    pygame.draw.line(screen, col, to_screen(prev[0], prev[1]),
                                     to_screen(p[0], p[1]), wmark)
                prev = p


# =========================================================================== #
# V2 FX — used by supra/viewer2.py (projected 2.5D). Everything below draws
# through the V2 View projector (world x, y, z -> screen), unlike the v1
# classes above which use the flat to_screen. Presentation-only.
# =========================================================================== #
import numpy as np


class Skids:
    """Continuous, world-space twin tyre trails for V2.

    Marks are presentation-only and age independently from the vehicle. A
    `None` separator breaks a stroke whenever grip returns, the car goes
    airborne, or it teleports/reset, preventing long lines across the circuit.
    """

    LIFE = 22.0
    MAX_PER_WHEEL = 1200

    def __init__(self):
        # Entries: [x, y, z, strength, age] or None as a stroke separator.
        self.trails = {-1: [], 1: []}

    @staticmethod
    def _break(trail):
        if trail and trail[-1] is not None:
            trail.append(None)

    def update(self, veh, fr, dt, brake=0.0):
        slide = abs(math.degrees(veh.slip_angle))
        drifting = veh.speed > 7.0 and slide > 8.0
        lockup = veh.speed > 13.0 and brake > 0.72
        marking = (drifting or lockup) and not fr["off_track"] and not veh.airborne

        if marking:
            b = getattr(veh.spec, "b", 1.3)
            ht = getattr(veh.spec, "half_track", 0.8)
            cyw, syw = math.cos(veh.yaw), math.sin(veh.yaw)
            strength = max(min(1.0, (slide - 7.0) / 27.0),
                           min(0.78, max(0.0, (brake - 0.68) / 0.32)))
            for sgn in (1, -1):
                wx = veh.x - cyw * b - syw * sgn * ht
                wy = veh.y - syw * b + cyw * sgn * ht
                trail = self.trails[sgn]
                prev = next((p for p in reversed(trail) if p is not None), None)
                if prev is not None:
                    jump2 = (wx - prev[0]) ** 2 + (wy - prev[1]) ** 2
                    if jump2 > 36.0:
                        self._break(trail)
                    elif jump2 < 0.018:
                        continue
                trail.append([wx, wy, float(fr["z"]) + 0.025, strength, 0.0])
        else:
            for trail in self.trails.values():
                self._break(trail)

        for wheel, trail in self.trails.items():
            for mark in trail:
                if mark is not None:
                    mark[4] += dt
            keep = [m for m in trail if m is None or m[4] < self.LIFE]
            while keep and keep[0] is None:
                keep.pop(0)
            # Collapse repeated separators left behind by expired strokes.
            compact = []
            for mark in keep:
                if mark is None and (not compact or compact[-1] is None):
                    continue
                compact.append(mark)
            if len(compact) > self.MAX_PER_WHEEL:
                compact = compact[-self.MAX_PER_WHEEL:]
                while compact and compact[0] is None:
                    compact.pop(0)
            self.trails[wheel] = compact

    def draw(self, pygame, screen, view, mood):
        road = mood.get("road", (48, 50, 54))
        width = max(2, int(view.scale * 0.15))
        soft_width = width + max(1, int(view.scale * 0.07))
        for trail in self.trails.values():
            prev = None
            for mark in trail:
                if mark is None:
                    prev = None
                    continue
                if prev is not None:
                    if (abs(mark[0] - view.cx) < 230 and
                            abs(mark[1] - view.cy) < 230):
                        fade = max(0.0, 1.0 - prev[4] / self.LIFE)
                        strength = min(prev[3], mark[3]) * fade
                        if strength > 0.025:
                            a = view.project(prev[0], prev[1], prev[2])
                            b = view.project(mark[0], mark[1], mark[2])
                            # A soft charcoal edge plus a dark rubber core gives
                            # the line width without per-segment alpha surfaces.
                            edge = _blend(road, (18, 18, 20), strength * 0.46)
                            core = _blend(road, (8, 9, 10), strength * 0.82)
                            pygame.draw.line(screen, edge, a, b, soft_width)
                            pygame.draw.line(screen, core, a, b, width)
                prev = mark


class Puffs:
    """Layered V2 particles: smoke, dust, gravel, water spray and sparks."""

    def __init__(self):
        self.p = []
        self.rings = []                    # landing shock sheets on the surface

    def _add(self, x, y, z, vx, vy, vz, radius, life, kind, ground=None):
        if len(self.p) >= 430:
            return
        self.p.append(dict(x=float(x), y=float(y), z=float(z),
                           vx=float(vx), vy=float(vy), vz=float(vz),
                           r=float(radius), age=0.0, life=float(life), kind=kind,
                           ground=float(z if ground is None else ground), bounces=0))

    def burst(self, x, y, z, rng, n=8, kind="dust"):
        count = min(30, max(n, int(n * 1.55)))
        for _ in range(count):
            a = rng.uniform(0.0, math.tau)
            speed = rng.uniform(1.8, 7.5)
            k = kind if kind in ("dust", "spray") else "dust"
            self._add(x + rng.uniform(-0.8, 0.8), y + rng.uniform(-0.8, 0.8),
                      z + rng.uniform(0.08, 0.30), math.cos(a) * speed,
                      math.sin(a) * speed, rng.uniform(0.8, 3.8),
                      rng.uniform(0.55, 1.35), rng.uniform(0.65, 1.35), k, z)
        self.rings.append(dict(x=float(x), y=float(y), z=float(z) + 0.025,
                               age=0.0, life=0.48 if kind == "dust" else 0.34,
                               strength=min(1.0, n / 14.0), kind=kind))
        if len(self.rings) > 12:
            del self.rings[:-12]

    def spark(self, x, y, z, rng, vx=0.0, vy=0.0):
        for _ in range(int(rng.integers(2, 5))):
            self._add(x + rng.uniform(-0.14, 0.14), y + rng.uniform(-0.14, 0.14),
                      z + rng.uniform(0.08, 0.22), vx * 0.28 + rng.uniform(-4.5, 4.5),
                      vy * 0.28 + rng.uniform(-4.5, 4.5), rng.uniform(2.0, 7.0),
                      rng.uniform(0.10, 0.24), rng.uniform(0.22, 0.58), "spark", z)

    def update(self, veh, fr, dt, rng, wet=False):
        slide = abs(math.degrees(veh.slip_angle))
        cyw, syw = math.cos(veh.yaw), math.sin(veh.yaw)
        car_vx, car_vy = veh.speed * cyw, veh.speed * syw
        b = getattr(veh.spec, "b", 1.3)
        ht = getattr(veh.spec, "half_track", 0.8)
        rear_x, rear_y = veh.x - cyw * b, veh.y - syw * b

        if veh.speed > 7 and slide > 11 and not veh.airborne:
            density = 1 + int(min(2, (slide - 11) / 18.0))
            for _ in range(density):
                kind = "dust" if fr["off_track"] else ("spray" if wet else "smoke")
                self._add(rear_x + rng.uniform(-0.65, 0.65), rear_y + rng.uniform(-0.65, 0.65),
                          fr["z"] + rng.uniform(0.18, 0.36),
                          car_vx * rng.uniform(0.35, 0.62) + rng.uniform(-1.2, 1.2),
                          car_vy * rng.uniform(0.35, 0.62) + rng.uniform(-1.2, 1.2),
                          rng.uniform(1.0, 3.4), rng.uniform(0.55, 1.15),
                          rng.uniform(0.85, 1.65), kind, fr["z"])

        # Off-track driven wheels throw discrete stones ahead of a broader dust
        # plume. The particles are visual and never become collision objects.
        if fr["off_track"] and veh.speed > 9 and not veh.airborne:
            for _ in range(1 + int(veh.speed > 28)):
                side = -1 if rng.random() < 0.5 else 1
                wx, wy = rear_x - syw * side * ht, rear_y + cyw * side * ht
                self._add(wx, wy, fr["z"] + 0.08,
                          car_vx * rng.uniform(0.18, 0.48) + rng.uniform(-4.0, 4.0),
                          car_vy * rng.uniform(0.18, 0.48) + rng.uniform(-4.0, 4.0),
                          rng.uniform(2.5, 7.5), rng.uniform(0.08, 0.18),
                          rng.uniform(0.45, 1.0), "gravel", fr["z"])
            if rng.random() < 0.58:
                self._add(rear_x, rear_y, fr["z"] + 0.16, car_vx * 0.38, car_vy * 0.38,
                          rng.uniform(1.2, 2.8), rng.uniform(0.65, 1.25),
                          rng.uniform(0.75, 1.35), "dust", fr["z"])

        # Wet mode produces two narrow rooster tails even without a slide.
        if wet and veh.speed > 6 and not veh.airborne and not fr["off_track"]:
            spray_n = 1 + int(veh.speed > 32)
            for side in (-1, 1):
                for _ in range(spray_n):
                    wx, wy = rear_x - syw * side * ht, rear_y + cyw * side * ht
                    self._add(wx, wy, fr["z"] + 0.10,
                              car_vx * rng.uniform(0.42, 0.68) + rng.uniform(-0.8, 0.8),
                              car_vy * rng.uniform(0.42, 0.68) + rng.uniform(-0.8, 0.8),
                              rng.uniform(1.4, 3.8), rng.uniform(0.35, 0.78),
                              rng.uniform(0.48, 0.92), "spray", fr["z"])

        for q in self.p:
            q["x"] += q["vx"] * dt
            q["y"] += q["vy"] * dt
            q["z"] += q["vz"] * dt
            q["age"] += dt
            if q["kind"] in ("spark", "gravel"):
                q["vz"] -= 18.0 * dt
                if q["z"] <= q["ground"] and q["vz"] < 0.0:
                    q["z"] = q["ground"] + 0.025
                    if q["bounces"] < 1:
                        q["vz"] = -q["vz"] * (0.36 if q["kind"] == "spark" else 0.24)
                        q["vx"] *= 0.72
                        q["vy"] *= 0.72
                        q["bounces"] += 1
                    else:
                        q["vz"] = 0.0
                        q["vx"] *= 0.88
                        q["vy"] *= 0.88
            elif q["kind"] == "spray":
                q["vz"] -= 5.5 * dt
                q["vx"] *= max(0.0, 1.0 - 1.8 * dt)
                q["vy"] *= max(0.0, 1.0 - 1.8 * dt)
            else:
                q["vz"] += (1.35 if q["kind"] == "smoke" else 0.42) * dt
                q["vx"] *= max(0.0, 1.0 - 1.15 * dt)
                q["vy"] *= max(0.0, 1.0 - 1.15 * dt)
        self.p = [q for q in self.p if q["age"] < q["life"]]
        for ring in self.rings:
            ring["age"] += dt
        self.rings = [r for r in self.rings if r["age"] < r["life"]]

    def draw(self, pygame, screen, view):
        # Landing sheets stay glued to the road and expand rapidly outward.
        for ring in self.rings:
            f = 1.0 - ring["age"] / ring["life"]
            sx, sy = view.project(ring["x"], ring["y"], ring["z"])
            rad = max(2, int(view.scale * (0.8 + ring["age"] * 11.0)))
            surf = pygame.Surface((rad * 2 + 4, rad + 4), pygame.SRCALPHA)
            col = (184, 164, 122) if ring["kind"] == "dust" else (194, 214, 226)
            pygame.draw.ellipse(surf, (*col, int(105 * f * ring["strength"])),
                                surf.get_rect(), max(1, int(view.scale * 0.055)))
            screen.blit(surf, (sx - rad - 2, sy - rad * 0.5 - 2))

        for q in self.p:
            f = max(0.0, 1.0 - q["age"] / q["life"])
            sx, sy = view.project(q["x"], q["y"], q["z"])
            kind = q["kind"]
            if kind == "spark":
                tail = view.project(q["x"] - q["vx"] * 0.055,
                                    q["y"] - q["vy"] * 0.055,
                                    q["z"] - q["vz"] * 0.035)
                width = max(1, int(view.scale * q["r"] * 0.55))
                pygame.draw.line(screen, (255, 108, 34), tail, (sx, sy), width + 2)
                pygame.draw.line(screen, (255, 244, 185), tail, (sx, sy), width)
                continue
            if kind == "gravel":
                tail = view.project(q["x"] - q["vx"] * 0.025,
                                    q["y"] - q["vy"] * 0.025,
                                    q["z"] - q["vz"] * 0.018)
                col = (156, 132, 91) if q["z"] > q["ground"] + 0.05 else (92, 78, 58)
                pygame.draw.line(screen, col, tail, (sx, sy),
                                 max(1, int(view.scale * q["r"])))
                continue

            grow = (1.0 - f) * (2.8 if kind != "spray" else 1.25)
            rad = max(2, int((q["r"] + grow) * view.scale *
                             (0.33 if kind != "spray" else 0.27)))
            if kind == "smoke":
                col, alpha = (156, 158, 162), int(108 * f)
            elif kind == "spray":
                col, alpha = (182, 207, 220), int(124 * f)
            else:
                col, alpha = (132, 108, 72), int(108 * f)
            surf = pygame.Surface((rad * 2 + 2, rad * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(surf, (*col, alpha // 2), (rad + 1, rad + 1), rad)
            pygame.draw.circle(surf, (*col, alpha), (rad + 1, rad + 1),
                               max(1, int(rad * 0.58)))
            screen.blit(surf, (sx - rad - 1, sy - rad - 1))
            if kind == "spray" and q["age"] < q["life"] * 0.55:
                tail = view.project(q["x"] - q["vx"] * 0.045,
                                    q["y"] - q["vy"] * 0.045,
                                    q["ground"] + 0.04)
                pygame.draw.line(screen, (184, 208, 220), tail, (sx, sy), 1)


def draw_exhaust_flame(pygame, screen, view, veh, spec, flame_t, rng):
    """Downshift backfire: brief blue-orange tongues out of the tail pipes."""
    if flame_t <= 0.0:
        return
    HL = (spec.wheelbase + 1.7) / 2.0
    HW = (spec.track_width + 0.36) / 2.0
    cyw, syw = math.cos(veh.yaw), math.sin(veh.yaw)
    road_z = float(getattr(veh, "road_z", 0.0))
    height = max(0.0, float(getattr(veh, "z", 0.0)) - road_z)
    f = min(1.0, flame_t / 0.14)
    for sgn in (1, -1):
        # pipe exits low on the tail, inboard of the brake lights
        px = veh.x - cyw * HL * 1.02 - syw * sgn * HW * 0.30
        py = veh.y - syw * HL * 1.02 + cyw * sgn * HW * 0.30
        sx, sy = view.project(px, py, road_z + height + 0.14)
        r = max(2, int(view.scale * (0.10 + 0.16 * f) * random.uniform(0.8, 1.25)))
        tx = px - cyw * random.uniform(0.5, 1.4) * f
        ty = py - syw * random.uniform(0.5, 1.4) * f
        ex, ey = view.project(tx, ty, road_z + height + 0.12)
        pygame.draw.line(screen, (255, 140, 40), (int(sx), int(sy)),
                         (int(ex), int(ey)), max(2, r))
        pygame.draw.circle(screen, (255, 220, 150), (int(sx), int(sy)), r)
        pygame.draw.circle(screen, (150, 190, 255), (int(sx), int(sy)),
                           max(1, r // 2))


def draw_audio_reactive_exhaust(pygame, screen, view, veh, spec, reactive):
    """Engine-synchronous pipe glow, shift pressure rings and limiter cuts.

    ``reactive`` is the viewer's read-only telemetry coordinator. Keeping this
    renderer duck-typed avoids coupling the effects module to simulation state.
    """
    load = float(getattr(reactive, "load", 0.0))
    harmonic = float(getattr(reactive, "harmonic", 0.0))
    shift = float(getattr(reactive, "shift", 0.0))
    backfire = float(getattr(reactive, "backfire", 0.0))
    limiter = float(getattr(reactive, "limiter_tick", 0.0))
    if max(load, shift, backfire, limiter) < 0.015:
        return

    HL = (spec.wheelbase + 1.7) / 2.0
    HW = (spec.track_width + 0.36) / 2.0
    cyw, syw = math.cos(veh.yaw), math.sin(veh.yaw)
    road_z = float(getattr(veh, "road_z", 0.0))
    height = max(0.0, float(getattr(veh, "z", 0.0)) - road_z)
    pulse = max(0.0, harmonic) * load

    for sgn in (1, -1):
        px = veh.x - cyw * HL * 1.02 - syw * sgn * HW * 0.30
        py = veh.y - syw * HL * 1.02 + cyw * sgn * HW * 0.30
        sx, sy = view.project(px, py, road_z + height + 0.14)
        center = (int(sx), int(sy))

        # Idle/load breathing is intentionally tiny; it becomes readable only
        # as the same harmonic signal also starts moving the body shell.
        if pulse > 0.01:
            rr = max(1, int(view.scale * (0.035 + pulse * 0.075)))
            pygame.draw.circle(screen, (110, 158, 220), center, rr)
            pygame.draw.circle(screen, (255, 152, 62), center, max(1, rr // 2))

        # Shift envelope falls from one to zero, expanding the pressure front
        # away from the pipe while its brightness dissipates.
        if shift > 0.02:
            progress = 1.0 - shift
            rr = max(2, int(view.scale * (0.10 + progress * 0.54)))
            ring = pygame.Surface((rr * 2 + 4, rr * 2 + 4), pygame.SRCALPHA)
            alpha = int(150 * shift * shift)
            pygame.draw.circle(ring, (158, 205, 255, alpha),
                               (rr + 2, rr + 2), rr, max(1, int(1 + shift * 2)))
            screen.blit(ring, (center[0] - rr - 2, center[1] - rr - 2))

        # These use the coordinator's exact event envelopes: no unrelated
        # random flash can happen between the corresponding audio events.
        event_energy = max(backfire, limiter * 0.72)
        if event_energy > 0.025:
            rr = max(2, int(view.scale * (0.08 + event_energy * 0.19)))
            glow = pygame.Surface((rr * 4 + 4, rr * 4 + 4), pygame.SRCALPHA)
            gc = rr * 2 + 2
            pygame.draw.circle(glow, (255, 96, 32, int(44 * event_energy)),
                               (gc, gc), rr * 2)
            pygame.draw.circle(glow, (255, 235, 194, int(230 * event_energy)),
                               (gc, gc), rr)
            screen.blit(glow, (center[0] - gc, center[1] - gc),
                        special_flags=pygame.BLEND_RGB_ADD)


class Cinematic:
    """Prerendered atmosphere: drifting cloud shadows on the ground, a sun-side
    glow + corner vignette colour grade per mood, and a whisper of film grain.
    Everything heavy is built once; the per-frame cost is a few blits."""

    def __init__(self, pygame, W, H, rng):
        self.W, self.H = W, H
        self._grade = {}          # mood name -> (combined overlays, glow)

        # soft cloud-shadow blob (concentric alpha ellipses, built once)
        bw, bh = 520, 340
        blob = pygame.Surface((bw, bh), pygame.SRCALPHA)
        for k in range(9, 0, -1):
            a = int(3.2 * (10 - k))
            rw, rh = int(bw * k / 18), int(bh * k / 18)
            pygame.draw.ellipse(blob, (8, 10, 8, a),
                                (bw // 2 - rw, bh // 2 - rh, rw * 2, rh * 2))
        self.blob = blob
        # world-anchored clouds: offset, drift velocity, size factor
        self.clouds = [(rng.uniform(0, 640), rng.uniform(0, 640),
                        rng.uniform(1.6, 3.4), rng.uniform(-1.2, 1.2),
                        rng.uniform(0.8, 1.6)) for _ in range(4)]

        # film grain: three full-frame noise plates cycled per frame
        self.grain = []
        for _ in range(3):
            g = pygame.Surface((W, H), pygame.SRCALPHA)
            try:
                alpha = pygame.surfarray.pixels_alpha(g)
                alpha[:, :] = (rng.random((W, H)) * 9).astype(np.uint8)
                del alpha
            except Exception:
                pass
            self.grain.append(g)

    def _build_grade(self, pygame, mood):
        W, H = self.W, self.H
        w4, h4 = W // 4, H // 4                       # build small, scale up
        xx, yy = np.meshgrid(np.linspace(0, 1, w4), np.linspace(0, 1, h4),
                             indexing="ij")

        # vignette: darkness creeping in from the corners
        r = np.sqrt(((xx - 0.5) * 1.12) ** 2 + ((yy - 0.55) * 0.95) ** 2)
        vig_a = (np.clip((r - 0.52) / 0.46, 0.0, 1.0) ** 1.6 * 120).astype(np.uint8)
        vig = pygame.Surface((w4, h4), pygame.SRCALPHA)
        rgb = pygame.surfarray.pixels3d(vig)
        rgb[:, :, 0] = 4; rgb[:, :, 1] = 5; rgb[:, :, 2] = 9
        del rgb
        alpha = pygame.surfarray.pixels_alpha(vig)
        alpha[:, :] = vig_a
        del alpha
        vig = pygame.transform.smoothscale(vig, (W, H))

        # sun glow: additive warm wash from the light's screen corner
        if mood["name"] == "dusk":
            gc, strength = (66, 30, 8), 1.25
        elif mood["name"] == "night":
            gc, strength = (10, 16, 38), 0.9
        else:
            gc, strength = (34, 28, 12), 1.0
        d2 = ((xx - 0.24) ** 2 + (yy - 0.10) ** 2)
        g = np.clip(1.0 - d2 / 0.75, 0.0, 1.0) ** 2 * strength
        glow = pygame.Surface((w4, h4))
        rgb = pygame.surfarray.pixels3d(glow)
        rgb[:, :, 0] = (gc[0] * g).astype(np.uint8)
        rgb[:, :, 1] = (gc[1] * g).astype(np.uint8)
        rgb[:, :, 2] = (gc[2] * g).astype(np.uint8)
        del rgb
        glow = pygame.transform.smoothscale(glow, (W, H))

        # A light atmospheric veil reinforces the world-distance fog used by
        # road strips and props. It is strongest toward the far/top of frame
        # and clears around the car, so the foreground retains crisp contrast.
        fog = mood.get("fog", (110, 120, 105))
        haze_strength = 24 if mood["name"] == "day" else (21 if mood["name"] == "dusk" else 15)
        # A low-alpha gradient quantized to only a few dozen integer values
        # forms visible full-width bands. Use a long smoothstep feather plus
        # ordered sub-alpha dithering so each transition is distributed across
        # columns instead of landing on one horizontal scan line.
        haze_t = np.clip((0.78 - yy) / 0.78, 0.0, 1.0)
        haze_t = haze_t * haze_t * (3.0 - 2.0 * haze_t)
        ix, iy = np.indices((w4, h4))
        dither = (((ix * 17 + iy * 29) & 63) / 64.0) - 0.4921875
        haze_a = np.clip(haze_t * haze_strength + dither, 0.0, 255.0).astype(np.uint8)
        haze = pygame.Surface((w4, h4), pygame.SRCALPHA)
        rgb = pygame.surfarray.pixels3d(haze)
        rgb[:, :, 0] = fog[0]; rgb[:, :, 1] = fog[1]; rgb[:, :, 2] = fog[2]
        del rgb
        alpha = pygame.surfarray.pixels_alpha(haze)
        alpha[:, :] = haze_a
        del alpha
        haze = pygame.transform.smoothscale(haze, (W, H))
        # Merge ordinary-alpha layers once, but do it in premultiplied math.
        # SDL's integer blit can preserve wildly different hidden RGB values
        # when two nearly transparent layers meet; after scaling, those values
        # appeared as the horizontal line visible in fullscreen captures.
        def alpha_over(bottom, top):
            brgb = pygame.surfarray.array3d(bottom).astype(np.float32)
            trgb = pygame.surfarray.array3d(top).astype(np.float32)
            ba = pygame.surfarray.array_alpha(bottom).astype(np.float32) / 255.0
            ta = pygame.surfarray.array_alpha(top).astype(np.float32) / 255.0
            out_a = ta + ba * (1.0 - ta)
            premul = trgb * ta[:, :, None] + brgb * ba[:, :, None] * (1.0 - ta[:, :, None])
            out_rgb = np.zeros_like(premul)
            np.divide(premul, np.maximum(out_a[:, :, None], 1e-6), out=out_rgb,
                      where=out_a[:, :, None] > 1e-6)
            out = pygame.Surface((W, H), pygame.SRCALPHA)
            rgb = pygame.surfarray.pixels3d(out)
            rgb[:, :, :] = np.clip(out_rgb, 0, 255).astype(np.uint8)
            del rgb
            alpha = pygame.surfarray.pixels_alpha(out)
            alpha[:, :] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
            del alpha
            return out

        base = alpha_over(haze, vig)
        overlays = [alpha_over(base, grain) for grain in self.grain]
        return base, overlays, glow

    def draw_clouds(self, pygame, screen, view, t):
        span = 640.0
        f = view.scale / 15.0
        bw = max(2, int(self.blob.get_width() * f))
        bh = max(2, int(self.blob.get_height() * f))
        blob = pygame.transform.scale(self.blob, (bw, bh))
        for ox, oy, vx, vy, s in self.clouds:
            rx = ((ox + vx * t - view.cx) % span) - span / 2.0
            ry = ((oy + vy * t - view.cy) % span) - span / 2.0
            sx, sy = view.project(view.cx + rx, view.cy + ry, view.z0)
            w = int(bw * s); h = int(bh * s)
            if -w < sx < self.W + w and -h < sy < self.H + h:
                b = blob if s == 1.0 else pygame.transform.scale(self.blob, (w, h))
                screen.blit(b, (sx - w // 2, sy - h // 2))

    def draw_grade(self, pygame, screen, mood, frame_no, quality=2):
        key = mood["name"]
        if key not in self._grade:
            self._grade[key] = self._build_grade(pygame, mood)
        base, overlays, glow = self._grade[key]
        # Rescue quality removes only film grain; atmosphere and color grade
        # remain identical. Balanced quality advances grain less often.
        if quality <= 0:
            plate = base
        else:
            stride = 1 if quality >= 2 else 3
            plate = overlays[(frame_no // stride) % len(overlays)]
        screen.blit(plate, (0, 0))
        screen.blit(glow, (0, 0), special_flags=pygame.BLEND_ADD)


class SpeedTension:
    """High-speed visual tension: a tightening speed vignette plus faint edge
    motion streaks. Silent below ~52 m/s; a wide screen centre stays clear so
    the effect frames the car instead of hiding it."""

    def __init__(self, pygame, W, H, rng):
        self.W, self.H = W, H
        self.rng = rng
        w4, h4 = W // 4, H // 4
        xx, yy = np.meshgrid(np.linspace(0, 1, w4), np.linspace(0, 1, h4),
                             indexing="ij")
        r = np.sqrt(((xx - 0.5) * 1.30) ** 2 + ((yy - 0.5) * 1.02) ** 2)
        a = (np.clip((r - 0.42) / 0.52, 0.0, 1.0) ** 1.9 * 165).astype(np.uint8)
        vig = pygame.Surface((w4, h4), pygame.SRCALPHA)
        rgb = pygame.surfarray.pixels3d(vig)
        rgb[:, :, 0] = 3; rgb[:, :, 1] = 4; rgb[:, :, 2] = 7
        del rgb
        alpha = pygame.surfarray.pixels_alpha(vig)
        alpha[:, :] = a
        del alpha
        self.vig = pygame.transform.smoothscale(vig, (W, H))
        edge_h = max(48, int(H * 0.18))
        edge_w = max(64, int(W * 0.16))
        middle_h = H - edge_h * 2
        self.vig_parts = (
            (self.vig.subsurface((0, 0, W, edge_h)).copy(), (0, 0)),
            (self.vig.subsurface((0, H - edge_h, W, edge_h)).copy(),
             (0, H - edge_h)),
            (self.vig.subsurface((0, edge_h, edge_w, middle_h)).copy(),
             (0, edge_h)),
            (self.vig.subsurface((W - edge_w, edge_h, edge_w, middle_h)).copy(),
             (W - edge_w, edge_h)),
        )
        self.p = [[rng.uniform(0, W), rng.uniform(0, H)] for _ in range(20)]
        # Streak alpha only occupies the outer frame. Two narrow reusable
        # surfaces avoid allocating/blitting a transparent full-screen buffer.
        self.edge_w = max(96, int(W * 0.31))
        self.left = pygame.Surface((self.edge_w, H), pygame.SRCALPHA)
        self.right = pygame.Surface((self.edge_w, H), pygame.SRCALPHA)

    def draw(self, pygame, screen, view, veh, dt, quality=2):
        f = (veh.speed - 52.0) / 44.0
        if f <= 0.0:
            return
        f = min(1.0, f)
        if quality <= 0:
            return                         # protect cadence before geometry
        vig_alpha = int((150 if quality >= 2 else 92) * f)
        for part, pos in self.vig_parts:
            part.set_alpha(vig_alpha)
            screen.blit(part, pos)
        # edge streaks flowing down-screen with the world
        vy = veh.speed * view.scale * dt
        length = 10.0 + 52.0 * f
        self.left.fill((0, 0, 0, 0))
        self.right.fill((0, 0, 0, 0))
        count = len(self.p) if quality >= 2 else 12
        for q in self.p[:count]:
            q[1] += vy
            if q[1] - length > self.H:
                q[0] = self.rng.uniform(0, self.W)
                q[1] = -self.rng.uniform(0, self.H * 0.3)
            cx = abs(q[0] - self.W * 0.5) / (self.W * 0.5)
            if cx < 0.38:                  # keep a wide clear centre
                continue
            a = int((10 + 26 * f) * min(1.0, (cx - 0.38) / 0.30 + 0.3))
            if q[0] < self.W * 0.5:
                surf, local_x = self.left, q[0]
            else:
                surf, local_x = self.right, q[0] - (self.W - self.edge_w)
            pygame.draw.line(surf, (235, 240, 250, a),
                             (local_x, q[1] - length), (local_x, q[1]), 1)
        if count:
            screen.blit(self.left, (0, 0))
            screen.blit(self.right, (self.W - self.edge_w, 0))


class FlybyDopplerFX:
    """Camera-relative speed cues for fixed trackside broadcast shots.

    The leading wavefronts tighten while the car closes on the camera; after
    the radial velocity changes sign, the wake stretches behind it. A single
    expanding ring marks the closest-pass instant. The class only draws and
    consumes coordinator telemetry—it cannot affect sound or simulation.
    """

    def __init__(self, W, H):
        self.W, self.H = W, H

    def draw(self, pygame, screen, view, veh, reactive, active=False):
        strength = float(getattr(reactive, "doppler_strength", 0.0))
        pass_pulse = float(getattr(reactive, "flyby_pulse", 0.0))
        if not active or max(strength, pass_pulse) < 0.025 or veh.speed < 20.0:
            return

        z = float(getattr(veh, "road_z", 0.0)) + max(
            0.0, float(getattr(veh, "z", 0.0)) - float(getattr(veh, "road_z", 0.0))) + 0.48
        cx, cy = view.project(veh.x, veh.y, z)
        # Project a short world-space velocity probe. This automatically makes
        # streaks follow the car's actual direction in every fixed shot.
        probe = 2.2
        fx = veh.x + math.cos(veh.yaw) * probe
        fy = veh.y + math.sin(veh.yaw) * probe
        px, py = view.project(fx, fy, z)
        dx, dy = px - cx, py - cy
        mag = max(1e-5, math.hypot(dx, dy))
        ux, uy = dx / mag, dy / mag
        nx, ny = -uy, ux
        surf = pygame.Surface((self.W, self.H), pygame.SRCALPHA)

        closing = float(getattr(reactive, "closing_speed", 0.0))
        if closing > 0.0:
            # Approaching wavefronts bunch together and brighten as radial
            # closing velocity rises—the visual equivalent of the rising pitch.
            compression = min(1.0, closing / 72.0)
            spacing = 17.0 - compression * 10.5
            half_w = 12.0 + strength * 16.0
            for i in range(1, 7):
                along = 12.0 + i * spacing
                mx, my = cx + ux * along, cy + uy * along
                taper = 1.0 - i / 8.0
                alpha = int((20 + 82 * strength) * taper)
                pygame.draw.line(
                    surf, (176, 218, 255, alpha),
                    (mx - nx * half_w * taper, my - ny * half_w * taper),
                    (mx + nx * half_w * taper, my + ny * half_w * taper),
                    max(1, int(1 + strength * 1.5)))
        else:
            # Receding wake spacing and length expand with negative closing
            # speed. Three offset filaments keep the car itself unobscured.
            stretch = min(1.0, -closing / 72.0)
            length = 30.0 + 105.0 * strength + 38.0 * stretch
            for lane in (-1.0, 0.0, 1.0):
                off = lane * (5.0 + strength * 4.0)
                start = (cx + nx * off - ux * 8.0, cy + ny * off - uy * 8.0)
                end = (start[0] - ux * length, start[1] - uy * length)
                alpha = int((30 if lane else 50) + strength * (60 if lane else 90))
                pygame.draw.line(surf, (255, 190, 116, alpha), start, end,
                                 max(1, int(1 + strength * 2.0)))

        if pass_pulse > 0.025:
            progress = 1.0 - pass_pulse
            radius = int(16.0 + progress * 150.0)
            alpha = int(150 * pass_pulse * pass_pulse)
            pygame.draw.circle(surf, (225, 240, 255, alpha),
                               (int(cx), int(cy)), radius,
                               max(1, int(1 + pass_pulse * 3)))
            # A short transverse speed slash makes the instant read even if the
            # ring expands partly beyond a tight apex-camera frame.
            slash = 34.0 + pass_pulse * 82.0
            pygame.draw.line(surf, (255, 226, 184, int(alpha * 0.72)),
                             (cx - nx * slash, cy - ny * slash),
                             (cx + nx * slash, cy + ny * slash), 2)

        screen.blit(surf, (0, 0))
