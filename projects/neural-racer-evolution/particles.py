"""
Particle effects:
  Smoke  — grey puffs emitted from rear wheels while drifting
  Sparks — hot yellow sparks on car-to-car collisions and wall deaths
"""
import random
import math
import pygame


class _P:
    """Lightweight particle — uses __slots__ to keep memory lean."""
    __slots__ = ['x', 'y', 'vx', 'vy', 'life', 'max_life', 'r', 'cr', 'cg', 'cb']

    def __init__(self, x, y, vx, vy, life, r, color):
        self.x = x;   self.y = y
        self.vx = vx; self.vy = vy
        self.life = life; self.max_life = life
        self.r = r
        self.cr, self.cg, self.cb = color


class Particles:
    MAX = 1400   # hard cap on live particles

    def __init__(self):
        self.pool = []

    # ------------------------------------------------------------------
    # Emitters
    # ------------------------------------------------------------------

    def emit_smoke(self, x, y, car_vx=0.0, car_vy=0.0, count=2):
        """Grey puffs drifting outward from the rear wheels."""
        for _ in range(count):
            ang  = random.uniform(0, math.tau)
            spd  = random.uniform(5, 28)
            life = random.uniform(0.45, 1.1)
            r    = random.randint(4, 9)
            g_   = random.randint(130, 180)
            self.pool.append(_P(
                x + random.uniform(-3, 3),
                y + random.uniform(-3, 3),
                car_vx * 0.08 + math.cos(ang) * spd,
                car_vy * 0.08 + math.sin(ang) * spd,
                life, r, (g_, g_, g_)
            ))

    def emit_sparks(self, x, y, vx=0.0, vy=0.0, count=10):
        """Hot yellow-white sparks for collisions and wall deaths."""
        for _ in range(count):
            ang  = random.uniform(0, math.tau)
            spd  = random.uniform(60, 220)
            life = random.uniform(0.07, 0.32)
            r    = random.randint(1, 3)
            col  = (255, random.randint(140, 255), random.randint(0, 50))
            self.pool.append(_P(
                x, y,
                vx + math.cos(ang) * spd,
                vy + math.sin(ang) * spd,
                life, r, col
            ))

    # ------------------------------------------------------------------
    # Update / Draw
    # ------------------------------------------------------------------

    def update(self, dt):
        nxt = []
        for p in self.pool:
            p.life -= dt
            if p.life <= 0.0:
                continue
            p.x  += p.vx * dt
            p.y  += p.vy * dt
            p.vx *= 0.91   # air drag
            p.vy *= 0.91
            nxt.append(p)
        self.pool = nxt

        # Trim oldest particles if over cap
        if len(self.pool) > self.MAX:
            del self.pool[:len(self.pool) - self.MAX]

    def draw(self, surface, offset, screen_w, screen_h):
        if not self.pool:
            return
        ox, oy = offset
        M = 60   # screen margin — skip off-screen particles
        for p in self.pool:
            sx = int(p.x - ox)
            sy = int(p.y - oy)
            if not (-M < sx < screen_w + M and -M < sy < screen_h + M):
                continue
            t   = max(0.0, p.life / p.max_life)   # 1 = fresh, 0 = dying
            col = (int(p.cr * t), int(p.cg * t), int(p.cb * t))
            r   = max(1, int(p.r * (0.35 + 0.65 * t)))
            pygame.draw.circle(surface, col, (sx, sy), r)
