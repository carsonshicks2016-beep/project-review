"""Live VECTOR DUEL viewer — watch the policy duel while it trains.

Runs decoupled from training: it loads the checkpoint that ``train.py`` writes
and hot-reloads it whenever the file changes, so the fights on screen visibly
sharpen as new checkpoints land.  No impact on training throughput.

    # terminal 1
    python3 -m duel.train --save-every 5 --out duel/checkpoints/duel.pt
    # terminal 2
    python3 -m duel.watch --ckpt duel/checkpoints/duel.pt

Keys:  SPACE pause · D toggle deterministic · R force-reload · ESC quit
Grid:  --grid 6   shows six duels at once.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from collections import deque

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pygame
import torch

from . import env as E
from .env import VecDuel
from .networks import ActorCritic, ACT_DIM

# ---- palette (matches the web renderer) ----
BG = (5, 6, 14)
BG1 = (11, 16, 36)
GRID = (44, 58, 96)
EDGE = (120, 150, 230)
TEXT = (230, 236, 255)
DIM = (107, 118, 168)
PCOL = [(55, 232, 255), (255, 61, 127)]   # ARC cyan, HEX rose
NAMES = ["ARC", "HEX"]


def rot(ang, lx, ly):
    c, s = math.cos(ang), math.sin(ang)
    return lx * c - ly * s, lx * s + ly * c


def make_glow(color, size=128):
    """Soft radial glow built by additively stacking faint translucent discs."""
    surf = pygame.Surface((size, size), pygame.SRCALPHA)
    cx = size // 2
    steps = 28
    for i in range(steps, 0, -1):
        r = int(cx * i / steps)
        ring = pygame.Surface((size, size), pygame.SRCALPHA)
        pygame.draw.circle(ring, (*color, 7), (cx, cx), r)
        surf.blit(ring, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    return surf


class Watcher:
    def __init__(self, args):
        self.args = args
        self.n = max(1, args.grid)
        pygame.init()
        flags = 0
        self.W, self.H = args.width, args.height
        self.screen = pygame.display.set_mode((self.W, self.H), flags)
        pygame.display.set_caption("VECTOR DUEL — LIVE")
        self.font = pygame.font.SysFont("menlo,dejavusansmono,couriernew,monospace", 14)
        self.bigfont = pygame.font.SysFont("menlo,dejavusansmono,couriernew,monospace", 22, bold=True)
        self.glow = {c: make_glow(PCOL[c]) for c in (0, 1)}
        self.glow_w = make_glow((255, 255, 255))
        self.bg = self._make_bg()

        self.env = VecDuel(self.n, seed=int(time.time()) % 9999)
        self.obs = self.env.reset()
        self.agent = None
        self.ckpt_iter = 0
        self.ckpt_mtime = 0.0
        self.tally = [0, 0]
        self.reload_flash = 0.0
        self.deterministic = args.deterministic
        self.paused = False

        self.trails = {}        # (env,player) -> deque of world pos
        self.parts = []         # particles: [x,y,vx,vy,life,max,color,env]
        self.shake = np.zeros(self.n)
        self.prev_hp = self.env.health.copy()

    # ------------------------------------------------------------ layout
    def cells(self):
        cols = int(math.ceil(math.sqrt(self.n)))
        rows = int(math.ceil(self.n / cols))
        top = 54
        gap = 14
        cw = (self.W - gap * (cols + 1)) / cols
        ch = (self.H - top - gap * (rows + 1)) / rows
        out = []
        for e in range(self.n):
            r, c = divmod(e, cols)
            x = gap + c * (cw + gap)
            y = top + gap + r * (ch + gap)
            out.append((x, y, cw, ch))
        return out

    def _make_bg(self):
        bg = pygame.Surface((self.W, self.H))
        for y in range(self.H):
            t = y / self.H
            col = [int(BG[i] + (BG1[i] - BG[i]) * (1 - abs(t - 0.45) * 1.6)) for i in range(3)]
            col = [max(0, min(255, v)) for v in col]
            pygame.draw.line(bg, col, (0, y), (self.W, y))
        return bg

    # ------------------------------------------------------------ checkpoint
    def try_reload(self, force=False):
        path = self.args.ckpt
        if not os.path.exists(path):
            return
        try:
            mt = os.path.getmtime(path)
        except OSError:
            return
        if not force and mt == self.ckpt_mtime:
            return
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if self.agent is None:
                self.agent = ActorCritic(ck["obs_dim"])
            self.agent.load_state_dict(ck["state_dict"])
            self.agent.eval()
            self.ckpt_iter = ck.get("iter", 0)
            self.ckpt_mtime = mt
            self.reload_flash = 1.2
        except Exception:
            pass    # file may be mid-write; try again next tick

    # ------------------------------------------------------------ sim
    def step_sim(self):
        if self.agent is None:
            return
        with torch.no_grad():
            act, _, _ = self.agent.act(
                torch.as_tensor(self.obs.reshape(2 * self.n, -1)),
                deterministic=self.deterministic)
        self.obs, rew, done, info = self.env.step(act.numpy().reshape(self.n, 2, ACT_DIM))

        # hit / death particles + shake from health drops
        hp = self.env.health
        for e in range(self.n):
            for p in range(2):
                if hp[e, p] < self.prev_hp[e, p]:
                    x, y = self.env.pos[e, p]
                    big = hp[e, p] <= 0 < self.prev_hp[e, p]
                    self.shake[e] = 0.9 if big else 0.4
                    for _ in range(48 if big else 12):
                        a = np.random.rand() * 6.28
                        s = (6 if big else 3) + np.random.rand() * (22 if big else 9)
                        self.parts.append([x, y, math.cos(a) * s, math.sin(a) * s,
                                           0.5 + np.random.rand() * (0.8 if big else 0.3),
                                           1.3, p, e])
        # tally + reset trails for finished arenas
        for e in np.where(done)[0]:
            w = info["winner"][e]
            if w >= 0:
                self.tally[w] += 1
            for p in range(2):
                self.trails.pop((e, p), None)
        self.prev_hp = hp.copy()

        # trails
        for e in range(self.n):
            for p in range(2):
                k = (e, p)
                self.trails.setdefault(k, deque(maxlen=16)).append(tuple(self.env.pos[e, p]))

    def update_fx(self, dt):
        for q in self.parts:
            q[0] += q[2] * dt; q[1] += q[3] * dt
            q[2] *= 0.9; q[3] *= 0.9; q[4] -= dt
        self.parts = [q for q in self.parts if q[4] > 0]
        self.shake *= 0.86
        if self.reload_flash > 0:
            self.reload_flash -= dt

    # ------------------------------------------------------------ render
    def draw(self):
        self.screen.blit(self.bg, (0, 0))
        cells = self.cells()
        for e, cell in enumerate(cells):
            self.draw_arena(e, cell)
        self.draw_hud()
        pygame.display.flip()

    def _tf(self, cell):
        x, y, w, h = cell
        sc = min((w - 16) / E.ARENA_W, (h - 16) / E.ARENA_H)
        ox = x + (w - E.ARENA_W * sc) / 2
        oy = y + (h - E.ARENA_H * sc) / 2
        return sc, ox, oy

    def draw_arena(self, e, cell):
        sc, ox, oy = self._tf(cell)
        sh = self.shake[e] * 6
        if sh > 0.2:
            ox += (np.random.rand() - .5) * sh
            oy += (np.random.rand() - .5) * sh
        S = lambda wx, wy: (float(ox + wx * sc), float(oy + wy * sc))
        aw, ah = E.ARENA_W * sc, E.ARENA_H * sc

        # grid + border
        grid = pygame.Surface((int(aw), int(ah)), pygame.SRCALPHA)
        stp = max(8, int(1.5 * sc))
        for gx in range(0, int(aw), stp):
            pygame.draw.line(grid, (*GRID, 36), (gx, 0), (gx, ah))
        for gy in range(0, int(ah), stp):
            pygame.draw.line(grid, (*GRID, 36), (0, gy), (aw, gy))
        pygame.draw.line(grid, (*GRID, 60), (aw / 2, 0), (aw / 2, ah))
        self.screen.blit(grid, (ox, oy))
        pygame.draw.circle(self.screen, (50, 66, 110),
                           (int(ox + aw / 2), int(oy + ah / 2)), int(2.2 * sc), 1)
        pygame.draw.rect(self.screen, EDGE, (ox, oy, aw, ah), 1, border_radius=8)

        if self.agent is None:
            return

        under = pygame.Surface((self.W, self.H), pygame.SRCALPHA)   # additive, below bodies
        over = pygame.Surface((self.W, self.H), pygame.SRCALPHA)    # additive, above bodies

        # bullets (under)
        for i in range(E.MAX_BULLETS):
            if self.env.b_life[e, i] <= 0:
                continue
            c = int(self.env.b_owner[e, i])
            bx, by = self.env.b_pos[e, i]
            vx, vy = self.env.b_vel[e, i]
            px, py = S(bx, by)
            tx, ty = S(bx - vx * 0.05, by - vy * 0.05)
            self._blit_glow(under, self.glow_w, px, py, 13)
            pygame.draw.line(under, (*PCOL[c], 180), (tx, ty), (px, py), 2)
            pygame.draw.circle(under, (255, 255, 255), (int(px), int(py)),
                               max(2, int(E.BULLET_R * sc)))

        for p in range(2):
            self.draw_under(e, p, S, sc, under)
        self.screen.blit(under, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        for p in range(2):
            self.draw_body(e, p, S, sc)

        for p in range(2):
            self.draw_over(e, p, S, sc, over)
        for q in self.parts:
            if q[7] != e:
                continue
            px, py = S(q[0], q[1])
            k = q[4] / q[5]
            pygame.draw.circle(over, (*PCOL[q[6]], int(230 * k)), (int(px), int(py)),
                               max(1, int(2 * (0.4 + k))))
        self.screen.blit(over, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def _blit_glow(self, target, glow, x, y, px):
        """px: desired glow diameter in screen pixels."""
        w = max(2, int(px))
        g = pygame.transform.smoothscale(glow, (w, w))
        target.blit(g, (x - w / 2, y - w / 2), special_flags=pygame.BLEND_RGB_ADD)

    def draw_under(self, e, p, S, sc, under):
        ang = float(self.env.face[e, p])
        wx, wy = self.env.pos[e, p]
        cx, cy = S(wx, wy)
        r = E.AGENT_R * sc
        col = PCOL[p]
        dashing = self.env.dash_t[e, p] > 0

        tr = self.trails.get((e, p))
        if tr and len(tr) > 1:
            for i in range(1, len(tr)):
                a = int(90 * i / len(tr))
                x0, y0 = S(*tr[i - 1]); x1, y1 = S(*tr[i])
                pygame.draw.line(under, (*col, a), (x0, y0), (x1, y1), max(1, int(i / len(tr) * 3)))

        nx, ny = S(wx + math.cos(ang) * E.AGENT_R * 1.3, wy + math.sin(ang) * E.AGENT_R * 1.3)
        lx, ly = S(wx + math.cos(ang) * 9, wy + math.sin(ang) * 9)
        pygame.draw.line(under, (*col, 60), (nx, ny), (lx, ly), 1)

        self._blit_glow(under, self.glow[p], cx, cy, (3.2 if dashing else 2.3) * r)

    def draw_body(self, e, p, S, sc):
        ang = float(self.env.face[e, p])
        cx, cy = S(*self.env.pos[e, p])
        r = E.AGENT_R * sc
        hp = max(0.0, float(self.env.health[e, p]))
        col = PCOL[p]

        ring = pygame.Rect(0, 0, r * 3.8, r * 3.8); ring.center = (cx, cy)
        pygame.draw.arc(self.screen, (54, 64, 86), ring, 0, 6.283, 2)
        frac = hp / E.HEALTH
        if frac > 0:
            pygame.draw.arc(self.screen, col, ring, math.pi / 2 - frac * 2 * math.pi, math.pi / 2, 3)

        pts = []
        for lx2, ly2 in [(r * 1.6, 0), (-r * 0.95, r), (-r * 0.45, 0), (-r * 0.95, -r)]:
            dx, dy = rot(ang, lx2, ly2)
            pts.append((cx + dx, cy + dy))
        pygame.draw.polygon(self.screen, col, pts)
        pygame.draw.polygon(self.screen, (235, 245, 255), pts, 1)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(cx), int(cy)), max(1, int(r * 0.32)))

    def draw_over(self, e, p, S, sc, over):
        if self.env.invuln[e, p] <= 0:
            return
        cx, cy = S(*self.env.pos[e, p])
        r = E.AGENT_R * sc
        a = int(70 + 50 * math.sin(time.time() * 22))
        pygame.draw.circle(over, (*PCOL[p], max(0, a)), (int(cx), int(cy)), int(r * 2.5), 2)

    def draw_hud(self):
        # title
        self._text(self.bigfont, "VECTOR DUEL", 16, 12, TEXT)
        sub = "LIVE" if not self.paused else "PAUSED"
        self._text(self.font, sub, 188, 20, PCOL[0] if not self.paused else DIM)

        status = ("WAITING FOR CHECKPOINT…" if self.agent is None
                  else f"iter {self.ckpt_iter}")
        self._text(self.font, status, 16, 34, DIM)

        if self.reload_flash > 0 and self.agent is not None:
            self._text(self.font, "◆ RELOADED", 110, 34, PCOL[0])

        mode = "DET" if self.deterministic else "STOCH"
        tally = f"{NAMES[0]} {self.tally[0]} — {self.tally[1]} {NAMES[1]}"
        right = f"{tally}    [{mode}]    SPACE·D·R·ESC"
        surf = self.font.render(right, True, DIM)
        self.screen.blit(surf, (self.W - surf.get_width() - 16, 24))
        # tally color chips
        cw = surf.get_width()
        base = self.W - cw - 16
        self.screen.blit(self.font.render(f"{NAMES[0]} {self.tally[0]}", True, PCOL[0]), (base, 24))

    def _text(self, font, s, x, y, col):
        self.screen.blit(font.render(s, True, col), (x, y))

    # ------------------------------------------------------------ loop
    def run(self):
        clock = pygame.time.Clock()
        self.try_reload(force=True)
        # shot mode: warm up a few duels then save a frame
        if self.args.shot:
            for _ in range(self.args.frames):
                self.step_sim(); self.update_fx(E.DT)
            self.draw()
            pygame.image.save(self.screen, self.args.shot)
            print("saved", self.args.shot)
            return
        last_check = 0.0
        while True:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return
                if ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        return
                    if ev.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    if ev.key == pygame.K_d:
                        self.deterministic = not self.deterministic
                    if ev.key == pygame.K_r:
                        self.try_reload(force=True)
            if not self.paused:
                self.step_sim()
            self.update_fx(E.DT)
            if time.time() - last_check > 0.5:
                self.try_reload()
                last_check = time.time()
            self.draw()
            clock.tick(self.args.fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="duel/checkpoints/duel.pt")
    ap.add_argument("--grid", type=int, default=1, help="number of simultaneous duels")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1100)
    ap.add_argument("--height", type=int, default=680)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--shot", default=None, help="headless: render a frame to this PNG and exit")
    ap.add_argument("--frames", type=int, default=60, help="warmup frames before --shot")
    args = ap.parse_args()
    Watcher(args).run()
    pygame.quit()


if __name__ == "__main__":
    main()
