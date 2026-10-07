"""
Evolving Cars
=============
A genetic algorithm learns to drive a track you draw with the mouse.

Each car has 9 rangefinder sensors feeding a tiny MLP that outputs steering and
throttle. No gradients, no labels, no reward shaping beyond "get further round
the track" -- just selection, crossover and mutation across generations.

Controls
--------
  drawing phase   drag to paint a track | ENTER done | P procedural | L load saved
  running         SPACE pause      1/2/3/4  sim speed 1x 4x 16x 64x
                  N  end generation now     R  restart evolution
                  T  draw a new track       G  toggle sensor rays
                  C  toggle checkpoints     S  save best genome + track
                  ESC quit
"""

import json
import math
import os
import sys

import cv2
import numpy as np
import pygame

# ---------------------------------------------------------------- config ----

W, H = 1240, 780
FPS = 60

TRACK_RADIUS = 33          # half-width of the painted corridor, px
CP_SPACING = 46            # distance between checkpoints along the centerline
CP_RADIUS = TRACK_RADIUS + 8

POP = 140
ELITES = 5
TOURNAMENT = 5
MUT_RATE = 0.14
MUT_SIGMA = 0.28
SIGMA_MIN, SIGMA_MAX = 0.10, 0.60

N_RAYS = 9
RAY_SPREAD = math.radians(100)     # total fan angle
RAY_MAX = 170.0
RAY_STEP = 5.0
HIDDEN = 10
N_IN = N_RAYS + 1                  # rays + normalised speed
N_OUT = 2                          # steer, throttle

MAX_SPEED = 6.4
ACCEL = 0.42
DRAG = 0.955
TURN = 0.085                       # radians per frame at full speed
STALL_FRAMES = 90                  # die if no new checkpoint for this long
MAX_STEPS = 1500                   # hard cap on episode length
TARGET_LAPS = 3                    # closed tracks end once a car does this many

HERE = os.path.dirname(os.path.abspath(__file__))
TRACK_FILE = os.path.join(HERE, "track.json")
BEST_FILE = os.path.join(HERE, "best_genome.npz")

BG = (14, 17, 22)
TRACK_COL = (38, 44, 54)
EDGE_COL = (58, 66, 82)
CAR_COL = (86, 97, 118)
ELITE_COL = (163, 230, 53)
BEST_COL = (34, 211, 238)
TEXT = (196, 205, 220)
DIM = (110, 120, 138)


# ----------------------------------------------------------------- track ----

class Track:
    def __init__(self, center, closed):
        self.center = [tuple(p) for p in center]
        self.closed = closed
        self.mask, self.surface = self._bake()
        self.checkpoints = self._checkpoints()
        p0 = np.array(self.checkpoints[0], float)
        p1 = np.array(self.checkpoints[1 % len(self.checkpoints)], float)
        self.start = p0.copy()
        self.start_angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])

    def _bake(self):
        img = np.zeros((H, W), np.uint8)
        pts = np.array(self.center, np.int32)
        cv2.polylines(img, [pts], self.closed, 255, TRACK_RADIUS * 2)
        for p in self.center:
            cv2.circle(img, (int(p[0]), int(p[1])), TRACK_RADIUS, 255, -1)
        mask = img > 0

        edge = cv2.dilate(img, np.ones((9, 9), np.uint8)) > 0
        rgb = np.zeros((W, H, 3), np.uint8)
        rgb[edge.T] = EDGE_COL
        rgb[mask.T] = TRACK_COL
        return mask, pygame.surfarray.make_surface(rgb)

    def _checkpoints(self):
        pts, acc = [self.center[0]], 0.0
        for a, b in zip(self.center, self.center[1:]):
            acc += math.dist(a, b)
            if acc >= CP_SPACING:
                pts.append(b)
                acc = 0.0
        if self.closed and math.dist(pts[-1], pts[0]) < CP_SPACING * 0.6:
            pts.pop()
        return np.array(pts, float)

    def save(self, path=TRACK_FILE):
        with open(path, "w") as f:
            json.dump({"center": self.center, "closed": self.closed}, f)

    @staticmethod
    def load(path=TRACK_FILE):
        with open(path) as f:
            d = json.load(f)
        return Track(d["center"], d["closed"])


def procedural_track(rng):
    """A random closed loop built from a few harmonics on a circle."""
    t = np.linspace(0, 2 * math.pi, 420, endpoint=False)
    r = np.full_like(t, 1.0)
    for k in range(2, 8):
        r += rng.uniform(0.10, 0.26) * np.sin(k * t + rng.uniform(0, 2 * math.pi))
    r = np.clip(r, 0.34, None)
    x, y = r * np.cos(t), r * np.sin(t)
    pad = TRACK_RADIUS + 24
    x = pad + (x - x.min()) / (x.max() - x.min()) * (W - 2 * pad)
    y = pad + (y - y.min()) / (y.max() - y.min()) * (H - 2 * pad)
    return Track(list(zip(x.tolist(), y.tolist())), True)


# ------------------------------------------------------------ population ----

class Population:
    """POP independent MLPs stored as stacked arrays so every car steps at once."""

    def __init__(self, rng):
        self.rng = rng
        s = 1.0 / math.sqrt(N_IN)
        self.W1 = rng.normal(0, s, (POP, HIDDEN, N_IN))
        self.b1 = np.zeros((POP, HIDDEN))
        self.W2 = rng.normal(0, 1 / math.sqrt(HIDDEN), (POP, N_OUT, HIDDEN))
        self.b2 = np.zeros((POP, N_OUT))
        self.sigma = MUT_SIGMA
        self.elite_idx = np.array([], int)

    def act(self, rays, speed):
        x = np.concatenate([1.0 - rays / RAY_MAX, (speed / MAX_SPEED)[:, None]], 1)
        h = np.tanh(np.einsum("nhi,ni->nh", self.W1, x) + self.b1)
        return np.tanh(np.einsum("noh,nh->no", self.W2, h) + self.b2)

    def _child(self, a, b):
        out = []
        for pa, pb in zip(a, b):
            m = self.rng.random(pa.shape) < 0.5
            c = np.where(m, pa, pb)
            mut = self.rng.random(c.shape) < MUT_RATE
            c = c + mut * self.rng.normal(0, self.sigma, c.shape)
            out.append(c)
        return out

    def evolve(self, fitness, improved):
        # stagnation widens the search, progress narrows it
        self.sigma = float(np.clip(self.sigma * (0.94 if improved else 1.10),
                                   SIGMA_MIN, SIGMA_MAX))
        order = np.argsort(fitness)[::-1]
        params = (self.W1, self.b1, self.W2, self.b2)
        new = [np.empty_like(p) for p in params]

        for slot, src in enumerate(order[:ELITES]):
            for n, p in zip(new, params):
                n[slot] = p[src]

        for slot in range(ELITES, POP):
            # order[] is sorted best-first, so the winner is the smallest index
            pa, pb = (int(order[self.rng.integers(0, POP, TOURNAMENT).min()])
                      for _ in range(2))
            child = self._child([p[pa] for p in params], [p[pb] for p in params])
            for n, c in zip(new, child):
                n[slot] = c

        self.W1, self.b1, self.W2, self.b2 = new
        self.elite_idx = np.arange(ELITES)

    def save_best(self, idx, path=BEST_FILE):
        np.savez(path, W1=self.W1[idx], b1=self.b1[idx],
                 W2=self.W2[idx], b2=self.b2[idx])


# ------------------------------------------------------------ simulation ----

RAY_OFFSETS = np.linspace(-RAY_SPREAD / 2, RAY_SPREAD / 2, N_RAYS)
RAY_STEPS = int(RAY_MAX / RAY_STEP)


def cast_rays(mask, pos, ang):
    theta = ang[:, None] + RAY_OFFSETS[None, :]
    dx, dy = np.cos(theta), np.sin(theta)
    dist = np.full(theta.shape, RAY_MAX)
    live = np.ones(theta.shape, bool)
    for s in range(1, RAY_STEPS + 1):
        d = s * RAY_STEP
        xi = (pos[:, 0:1] + dx * d).astype(np.int32)
        yi = (pos[:, 1:2] + dy * d).astype(np.int32)
        inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        on = np.zeros(theta.shape, bool)
        on[inb] = mask[yi[inb], xi[inb]]
        hit = live & ~on
        dist[hit] = d
        live &= ~hit
        if not live.any():
            break
    return dist


class Episode:
    def __init__(self, track, pop):
        self.track, self.pop = track, pop
        self.cp = track.checkpoints
        self.ncp = len(self.cp)
        self.goal = self.ncp * TARGET_LAPS if track.closed else self.ncp
        self.pos = np.tile(track.start, (POP, 1))
        self.ang = np.full(POP, track.start_angle)
        self.speed = np.zeros(POP)
        self.alive = np.ones(POP, bool)
        self.prog = np.zeros(POP, int)
        self.stall = np.zeros(POP, int)
        self.steps_used = np.zeros(POP, int)
        self.frac = np.zeros(POP)
        self.rays = np.full((POP, N_RAYS), RAY_MAX)
        self.steps = 0

    def fitness(self):
        return 100.0 * self.prog + 100.0 * self.frac - 0.04 * self.steps_used

    def step(self):
        a = self.alive
        if not a.any():
            return
        self.rays[a] = cast_rays(self.track.mask, self.pos[a], self.ang[a])
        out = self.pop.act(self.rays, self.speed)[a]

        throttle = (out[:, 1] + 1.0) * 0.5
        sp = np.clip((self.speed[a] + throttle * ACCEL) * DRAG, 0.0, MAX_SPEED)
        ang = self.ang[a] + out[:, 0] * TURN * (sp / MAX_SPEED)
        pos = self.pos[a] + np.stack([np.cos(ang), np.sin(ang)], 1) * sp[:, None]
        self.speed[a], self.ang[a], self.pos[a] = sp, ang, pos

        target = self.cp[self.prog[a] % self.ncp]
        d = np.linalg.norm(pos - target, axis=1)
        self.frac[a] = np.clip(1.0 - d / (CP_SPACING * 2.2), 0.0, 1.0)
        got = d < CP_RADIUS
        idx = np.flatnonzero(a)
        self.prog[idx[got]] += 1
        self.stall[a] += 1
        self.stall[idx[got]] = 0
        self.steps_used[a] += 1

        xi, yi = pos[:, 0].astype(np.int32), pos[:, 1].astype(np.int32)
        inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        on = np.zeros(len(idx), bool)
        on[inb] = self.track.mask[yi[inb], xi[inb]]
        self.alive[idx] = on & (self.stall[a] < STALL_FRAMES) & (self.prog[a] < self.goal)
        self.steps += 1

    def done(self):
        return (not self.alive.any()) or self.steps >= MAX_STEPS


# --------------------------------------------------------------- drawing ----

def draw_phase(screen, font, rng):
    """Let the user paint a centerline; return a Track."""
    center, drawing, painting = [], True, False
    surf = pygame.Surface((W, H))
    clock = pygame.time.Clock()

    while drawing:
        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
                pygame.quit(); sys.exit()
            if e.type == pygame.MOUSEBUTTONDOWN:
                painting = True
            if e.type == pygame.MOUSEBUTTONUP:
                painting = False
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_p:
                    return procedural_track(rng)
                if e.key == pygame.K_l and os.path.exists(TRACK_FILE):
                    return Track.load()
                if e.key in (pygame.K_RETURN, pygame.K_KP_ENTER) and len(center) > 8:
                    closed = math.dist(center[0], center[-1]) < TRACK_RADIUS * 2.4
                    return Track(center, closed)
                if e.key == pygame.K_BACKSPACE:
                    center.clear()

        if painting:
            p = pygame.mouse.get_pos()
            if not center or math.dist(center[-1], p) >= 11:
                center.append(p)

        surf.fill(BG)
        if len(center) > 1:
            pygame.draw.lines(surf, TRACK_COL, False, center, TRACK_RADIUS * 2)
            for p in center:
                pygame.draw.circle(surf, TRACK_COL, p, TRACK_RADIUS)
            pygame.draw.lines(surf, (70, 80, 98), False, center, 2)
        if center:
            pygame.draw.circle(surf, BEST_COL, center[0], 9, 2)
            if math.dist(center[0], pygame.mouse.get_pos()) < TRACK_RADIUS * 2.4 and len(center) > 8:
                pygame.draw.circle(surf, BEST_COL, center[0], 22, 1)

        screen.blit(surf, (0, 0))
        lines = ["DRAW A TRACK", "",
                 "drag the mouse to paint a course",
                 "finish near the start circle to make it a loop",
                 "", "ENTER  start evolving      P  random track",
                 "BACKSPACE  clear" + ("      L  load saved track" if os.path.exists(TRACK_FILE) else "")]
        for i, t in enumerate(lines):
            screen.blit(font.render(t, True, TEXT if i == 0 else DIM), (26, 24 + i * 19))
        pygame.display.flip()
        clock.tick(FPS)


# ----------------------------------------------------------------- render ----

def draw_car(screen, pos, ang, col, size=7.0):
    c, s = math.cos(ang), math.sin(ang)
    pts = [(pos[0] + c * size * 1.6, pos[1] + s * size * 1.6),
           (pos[0] - c * size + -s * size * 0.75, pos[1] - s * size + c * size * 0.75),
           (pos[0] - c * size * 0.45, pos[1] - s * size * 0.45),
           (pos[0] - c * size + s * size * 0.75, pos[1] - s * size - c * size * 0.75)]
    pygame.draw.polygon(screen, col, pts)


def hud(screen, font, big, ep, gen, best_ever, hist, sim_mult, paused, sigma):
    fit = ep.fitness()
    alive = int(ep.alive.sum())
    cur = float(fit.max()) if len(fit) else 0.0
    laps = int(ep.prog.max()) // max(ep.ncp, 1)
    finished = int((ep.prog >= ep.goal).sum())

    panel = pygame.Surface((252, 192), pygame.SRCALPHA)
    panel.fill((10, 13, 18, 215))
    screen.blit(panel, (18, 18))

    screen.blit(big.render(f"GEN {gen}", True, BEST_COL), (34, 30))
    rows = [("alive", f"{alive}/{POP}"),
            ("checkpoints", f"{int(ep.prog.max())}/{ep.ncp}"),
            ("laps", f"{laps}/{TARGET_LAPS}" if ep.track.closed else str(laps)),
            ("finished", str(finished)),
            ("best now", f"{cur:,.0f}"),
            ("best ever", f"{best_ever:,.0f}"),
            ("mutation", f"{sigma:.2f}"),
            ("speed", ("PAUSED" if paused else f"{sim_mult}x"))]
    for i, (k, v) in enumerate(rows):
        y = 62 + i * 16
        screen.blit(font.render(k, True, DIM), (34, y))
        screen.blit(font.render(v, True, TEXT), (168, y))

    # fitness-per-generation sparkline
    if len(hist) > 1:
        gw, gh, gx, gy = 252, 78, 18, H - 96
        g = pygame.Surface((gw, gh), pygame.SRCALPHA)
        g.fill((10, 13, 18, 215))
        top = max(max(hist), 1.0)
        n = min(len(hist), gw - 20)
        pts = [(10 + i * (gw - 20) / max(n - 1, 1),
                gh - 12 - (hist[len(hist) - n + i] / top) * (gh - 24))
               for i in range(n)]
        pygame.draw.lines(g, BEST_COL, False, pts, 2)
        g.blit(font.render("best fitness / generation", True, DIM), (10, gh - 14))
        screen.blit(g, (gx, gy))

    tips = "SPACE pause   1-4 speed   N next gen   R restart   T new track   G rays   C checkpoints   S save"
    screen.blit(font.render(tips, True, DIM), (18, H - 16))


# ------------------------------------------------------------------- main ----

def main():
    pygame.init()
    pygame.display.set_caption("Evolving Cars")
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont("Menlo,Monaco,monospace", 13)
    big = pygame.font.SysFont("Menlo,Monaco,monospace", 22, bold=True)
    clock = pygame.time.Clock()
    rng = np.random.default_rng()

    track = draw_phase(screen, font, rng)
    pop = Population(rng)
    ep = Episode(track, pop)
    gen, best_ever, hist = 1, 0.0, []
    sim_mult, paused, show_rays, show_cp = 4, False, True, False

    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                pygame.quit(); return
            if e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    pygame.quit(); return
                if e.key == pygame.K_SPACE:
                    paused = not paused
                if e.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    sim_mult = [1, 4, 16, 64][e.key - pygame.K_1]
                if e.key == pygame.K_n:
                    ep.alive[:] = False
                if e.key == pygame.K_g:
                    show_rays = not show_rays
                if e.key == pygame.K_c:
                    show_cp = not show_cp
                if e.key == pygame.K_r:
                    pop, ep = Population(rng), None
                    ep = Episode(track, pop)
                    gen, best_ever, hist = 1, 0.0, []
                if e.key == pygame.K_t:
                    track = draw_phase(screen, font, rng)
                    pop = Population(rng)
                    ep = Episode(track, pop)
                    gen, best_ever, hist = 1, 0.0, []
                if e.key == pygame.K_s:
                    pop.save_best(int(np.argmax(ep.fitness())))
                    track.save()

        if not paused:
            for _ in range(sim_mult):
                ep.step()
                if ep.done():
                    fit = ep.fitness()
                    top = float(fit.max())
                    improved = top > best_ever + 1e-6
                    best_ever = max(best_ever, top)
                    hist.append(top)
                    pop.evolve(fit, improved)
                    ep = Episode(track, pop)
                    gen += 1
                    break

        screen.fill(BG)
        screen.blit(track.surface, (0, 0))

        if show_cp:
            for p in track.checkpoints:
                pygame.draw.circle(screen, (52, 60, 74), p.astype(int), 3)

        fit = ep.fitness()
        lead = int(np.argmax(np.where(ep.alive, fit, -1e9))) if ep.alive.any() else int(np.argmax(fit))

        elite = set(pop.elite_idx.tolist())
        for i in np.flatnonzero(ep.alive):
            if i == lead:
                continue
            draw_car(screen, ep.pos[i], ep.ang[i],
                     ELITE_COL if i in elite else CAR_COL, 6.0)

        if ep.alive[lead]:
            if show_rays:
                for k, off in enumerate(RAY_OFFSETS):
                    th = ep.ang[lead] + off
                    d = ep.rays[lead, k]
                    end = (ep.pos[lead, 0] + math.cos(th) * d, ep.pos[lead, 1] + math.sin(th) * d)
                    pygame.draw.line(screen, (34, 211, 238, 60), ep.pos[lead], end, 1)
                    pygame.draw.circle(screen, BEST_COL, (int(end[0]), int(end[1])), 2)
            nxt = track.checkpoints[ep.prog[lead] % ep.ncp]
            pygame.draw.circle(screen, (250, 204, 21), nxt.astype(int), 7, 1)
            draw_car(screen, ep.pos[lead], ep.ang[lead], BEST_COL, 8.0)

        hud(screen, font, big, ep, gen, best_ever, hist, sim_mult, paused, pop.sigma)
        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
