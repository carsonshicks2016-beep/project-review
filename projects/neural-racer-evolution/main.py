"""
Neural Racer — NEAT Evolution
Controls:
  F         toggle fast-forward (5x)
  SPACE     pause / unpause
  R         skip to next generation immediately
  D         toggle ray visualisation
  C         toggle camera follow (best car vs. free)
  N         generate a new random track
  +/-       increase / decrease simulation speed
  Q / ESC   quit
"""

import sys
import random
import pygame
import numpy as np

from track import Track
from car import Car, COLLISION_RADIUS
import car as _car_mod
from neat_ga import NEAT, InnovationTracker, Species
import hall_of_fame as hof
from replay import ReplaySession
from heatmap import Heatmap
from particles import Particles
from skidmarks import Skidmarks
from audio import SoundEngine
from crashes import CrashRecorder, play_replay
from sliders import Slider, SliderPanel
from commentary import Commentator
from spectator import Spectator
import neat_ga as _neat_ga_mod

# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------
SCREEN_W, SCREEN_H = 1280, 720
FPS                = 60
POPULATION         = 100
N_INPUTS           = 16   # 9 rays + fwd_vel + lat_vel + drift_flag + cp_dist + cp_angle + ang_vel + lap_prog
N_OUTPUTS          = 3    # steer, throttle, brake
MAX_GEN_SECONDS        = 120.0
TRACK_CHANGE_EVERY     = 5
ELIMINATION_INTERVAL   = 18.0   # seconds between eliminations
ELIMINATION_START      = 20.0   # wait this long before first cull
N_MULTI_TRACKS         = 3      # tracks evaluated per generation in tournament mode


def resolve_car_collisions(cars, particles=None, audio=None, sim_time=0.0):
    """
    Impulse-based circle collision between all pairs of alive cars.
    Equal mass assumed.  Cars pushed off-track will be killed by their own
    wall-collision check on the next update().
    Pass particles to emit sparks, audio to play impact thunks.
    """
    DIAM        = COLLISION_RADIUS * 2
    DIAM_SQ     = DIAM * DIAM
    RESTITUTION = 0.72   # bounciness: 0 = perfectly inelastic, 1 = elastic
    PUSH_FRAC   = 0.55   # how much of the overlap to correct per frame (stability)

    alive = [c for c in cars if c.alive]
    n = len(alive)

    for i in range(n):
        for j in range(i + 1, n):
            a, b = alive[i], alive[j]

            dx = a.x - b.x
            dy = a.y - b.y
            dist_sq = dx * dx + dy * dy

            if dist_sq >= DIAM_SQ or dist_sq < 1e-6:
                continue

            dist = dist_sq ** 0.5
            nx   = dx / dist   # collision normal pointing from b → a
            ny   = dy / dist

            # --- Impulse ---
            # Relative velocity along the normal
            rvx = a.vx - b.vx
            rvy = a.vy - b.vy
            rv_normal = rvx * nx + rvy * ny

            # Only resolve if cars are actually approaching each other
            if rv_normal >= 0:
                continue

            impulse = -(1.0 + RESTITUTION) * rv_normal / 2.0  # equal mass → /2

            a.vx += impulse * nx
            a.vy += impulse * ny
            b.vx -= impulse * nx
            b.vy -= impulse * ny

            if particles:
                particles.emit_sparks((a.x + b.x) / 2, (a.y + b.y) / 2,
                                      (a.vx + b.vx) * 0.1, (a.vy + b.vy) * 0.1,
                                      count=5)
            if audio:
                audio.play_thunk(sim_time)

            # --- Position correction (prevent sinking) ---
            overlap = (DIAM - dist) * PUSH_FRAC
            a.x += nx * overlap * 0.5
            a.y += ny * overlap * 0.5
            b.x -= nx * overlap * 0.5
            b.y -= ny * overlap * 0.5


def apply_drafting(cars):
    """
    Cars directly behind another (within a narrow cone and short distance)
    get reduced drag — the slipstream effect.
    The drafting car's draft_boost is set each frame; decays to 0 otherwise.
    """
    DRAFT_DIST  = 90.0    # max distance behind the leader to feel the draft
    DRAFT_CONE  = 0.82    # cos(35°) — how tight the cone is behind the leader
    DECAY       = 8.0     # how fast boost fades when out of cone (per second)
    dt          = 1 / 60  # approximate; good enough for a per-frame decay

    alive = [c for c in cars if c.alive]
    # Reset all boosts first
    for c in alive:
        c.draft_boost = max(0.0, c.draft_boost - DECAY * dt)

    for i in range(len(alive)):
        for j in range(len(alive)):
            if i == j:
                continue
            leader   = alive[i]
            follower = alive[j]

            # Vector from follower to leader
            dx = leader.x - follower.x
            dy = leader.y - follower.y
            dist = (dx*dx + dy*dy) ** 0.5
            if dist < 1.0 or dist > DRAFT_DIST:
                continue

            # Leader must be ahead of follower (dot with follower's heading)
            import math
            fx = math.cos(follower.angle)
            fy = math.sin(follower.angle)
            dot = (dx / dist) * fx + (dy / dist) * fy
            if dot < DRAFT_CONE:
                continue

            # Boost proportional to how close and how centred in the cone
            strength = (1.0 - dist / DRAFT_DIST) * ((dot - DRAFT_CONE) / (1.0 - DRAFT_CONE))
            follower.draft_boost = min(1.0, follower.draft_boost + strength)


def draw_network(surface, genome, rect, font):
    """
    Draw the NEAT network with LIVE activations.
    Nodes glow based on their actual activation value this frame.
    Edges light up proportional to how much signal they are carrying.

    Colour key:
      Input nodes  — blue  (dim→bright as activation rises)
      Output nodes — amber (dim→bright)
      Hidden nodes — green (dim→bright)
      Edges        — green for positive weight, red for negative;
                     brightness = |src_activation × weight|
    """
    rx, ry, rw, rh = rect

    bg = pygame.Surface((rw, rh), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 210))
    surface.blit(bg, (rx, ry))
    pygame.draw.rect(surface, (100, 100, 100), (rx, ry, rw, rh), 1)

    label = font.render("NET  (live)", True, (160, 160, 160))
    surface.blit(label, (rx + 4, ry + 3))

    nodes  = genome.nodes
    conns  = genome.connections
    vals   = getattr(genome, 'last_values', {})  # live activation values

    inputs  = sorted([n for n in nodes if nodes[n].type == 'input'])
    outputs = sorted([n for n in nodes if nodes[n].type == 'output'])
    hidden  = sorted([n for n in nodes if nodes[n].type == 'hidden'])

    pad = 14

    def node_pos(nid):
        if nid in inputs:
            idx = inputs.index(nid)
            x = rx + pad
            y = ry + pad + idx * (rh - 2*pad) / max(len(inputs) - 1, 1)
        elif nid in outputs:
            idx = outputs.index(nid)
            x = rx + rw - pad
            y = ry + pad + idx * (rh - 2*pad) / max(len(outputs) - 1, 1)
        else:
            idx = hidden.index(nid)
            x = rx + rw * 0.5 + (idx % 3 - 1) * 18
            y = ry + pad + idx * (rh - 2*pad) / max(len(hidden), 1)
        return int(x), int(y)

    def node_color(nid):
        """Base hue per layer, brightness driven by |activation|."""
        v     = float(vals.get(nid, 0.0))
        v     = max(-1.0, min(1.0, v))
        abv   = abs(v)
        # Minimum glow so cold nodes are still visible; max = full colour
        bright = 0.15 + 0.85 * abv
        if nid in inputs:
            base = (80, 140, 255)
        elif nid in outputs:
            base = (255, 170, 50)
        else:
            base = (80, 230, 130)
        return tuple(int(c * bright) for c in base)

    # Draw connections — brightness = how much signal is flowing right now
    for conn in conns.values():
        if not conn.enabled:
            continue
        if conn.in_node not in nodes or conn.out_node not in nodes:
            continue

        src_act = float(vals.get(conn.in_node, 0.0))
        signal  = abs(src_act * conn.weight)   # strength of contribution
        signal  = min(1.0, signal)

        # Dim floor so all edges are faintly visible even when idle
        intensity = int(20 + signal * 235)
        col   = (0, intensity, 0) if conn.weight > 0 else (intensity, 0, 0)
        thick = 1 if signal < 0.2 else (2 if signal < 0.6 else 3)
        pygame.draw.line(surface, col, node_pos(conn.in_node), node_pos(conn.out_node), thick)

    # Draw nodes on top of edges
    for nid in inputs + hidden + outputs:
        px, py = node_pos(nid)
        col = node_color(nid)
        pygame.draw.circle(surface, col, (px, py), 5)
        pygame.draw.circle(surface, (220, 220, 220), (px, py), 5, 1)

    # Label the three output nodes
    out_labels = ["STR", "THR", "BRK"]
    for i, nid in enumerate(outputs):
        px, py = node_pos(nid)
        v = float(vals.get(nid, 0.0))
        lbl = font.render(f"{out_labels[i]}:{v:+.2f}", True, (200, 200, 200))
        surface.blit(lbl, (px + 7, py - 6))


def lerp_color(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t),
    )


GRAPH_W = 290
GRAPH_H = 110


def draw_fitness_graph(surface, sim, rect, font):
    """
    Mini line chart over the last ~100 generations.
      gold   = best fitness
      green  = average fitness
      purple = curriculum difficulty (mapped to graph height)
    """
    history = sim.fitness_history
    if len(history) < 2:
        return
    rx, ry, rw, rh = rect

    bg = pygame.Surface((rw, rh), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 165))
    surface.blit(bg, (rx, ry))
    pygame.draw.rect(surface, (80, 80, 80), (rx, ry, rw, rh), 1)

    hdr = font.render("FITNESS  +  CURRICULUM", True, (110, 110, 110))
    surface.blit(hdr, (rx + 4, ry + 3))

    pad = 10
    top = ry + 18
    gh  = rh - 26
    gw  = rw - pad * 2
    gx  = rx + pad

    N      = min(len(history), 100)
    recent = history[-N:]
    best_v = [h[0] for h in recent]
    avg_v  = [h[1] for h in recent]
    max_v  = max(max(best_v), 1.0)

    def pt(i, v):
        sx = gx + int(i * gw / max(N - 1, 1))
        sy = top + gh - int(v / max_v * gh)
        return sx, sy

    for vals, col in [(avg_v, (55, 160, 65)), (best_v, (255, 200, 50))]:
        pts = [pt(i, v) for i, v in enumerate(vals)]
        if len(pts) > 1:
            pygame.draw.lines(surface, col, False, pts, 1)

    # Difficulty line (purple) on a 0..1 axis mapped to graph height
    diff_hist = getattr(sim, 'difficulty_history', [])
    if len(diff_hist) >= 2:
        d_recent = diff_hist[-N:]
        # Pad with the first value if shorter than fitness history
        if len(d_recent) < N:
            d_recent = [d_recent[0]] * (N - len(d_recent)) + d_recent
        d_pts = [(gx + int(i * gw / max(N - 1, 1)),
                  top + gh - int(d * gh))
                 for i, d in enumerate(d_recent)]
        pygame.draw.lines(surface, (180, 120, 230), False, d_pts, 1)

    # Latest values aligned to top-right of panel
    b_lbl = font.render(f"best {best_v[-1]:,.0f}", True, (255, 200, 50))
    a_lbl = font.render(f"avg  {avg_v[-1]:,.0f}",  True, (55, 180, 70))
    surface.blit(b_lbl, (rx + rw - b_lbl.get_width() - 4, ry + 4))
    surface.blit(a_lbl, (rx + rw - a_lbl.get_width() - 4, ry + 16))
    if diff_hist:
        d_lbl = font.render(f"diff {diff_hist[-1]:.2f}", True, (200, 140, 240))
        surface.blit(d_lbl, (rx + rw - d_lbl.get_width() - 4, ry + 28))


TREE_W = 290
TREE_H = 170


def draw_species_tree(surface, neat, rect, font, palette):
    """
    Phylogenetic tree of all species ever born.
    X-axis = generation, Y-axis = vertical slot.
    Lines connect parent species to their children; extinct lines fade.
    """
    rx, ry, rw, rh = rect

    bg = pygame.Surface((rw, rh), pygame.SRCALPHA)
    bg.fill((0, 0, 0, 175))
    surface.blit(bg, (rx, ry))
    pygame.draw.rect(surface, (80, 80, 80), (rx, ry, rw, rh), 1)

    hdr = font.render("SPECIES TREE", True, (130, 130, 130))
    surface.blit(hdr, (rx + 4, ry + 3))

    archive = neat.species_archive
    if not archive:
        return

    pad = 12
    gx, gy = rx + pad, ry + 20
    gw, gh = rw - 2 * pad, rh - 30

    cur_gen = max(neat.generation, 1)
    min_gen = max(0, cur_gen - 80)   # show last 80 gens of history
    span    = max(cur_gen - min_gen, 1)

    def gen_to_x(g):
        return gx + int((g - min_gen) / span * gw)

    visible = [s for s in archive
               if (s.death_gen is None or s.death_gen >= min_gen)]
    if not visible:
        return

    # Assign each species a y slot deterministically by id
    slots = len(visible)
    slot_h = max(2, gh // max(slots, 1))
    visible.sort(key=lambda s: s.birth_gen)
    y_of = {s.id: gy + i * slot_h + slot_h // 2 for i, s in enumerate(visible)}

    n_pal = len(palette)
    for s in visible:
        col_base = palette[s.id % n_pal]
        x1 = gen_to_x(max(s.birth_gen, min_gen))
        x2 = gen_to_x(s.death_gen if s.death_gen is not None else cur_gen)
        y  = y_of[s.id]

        alive = s.death_gen is None
        col   = col_base if alive else tuple(int(c * 0.35) for c in col_base)
        pygame.draw.line(surface, col, (x1, y), (x2, y), 2)

        # Parent connector
        if s.parent_id is not None and s.parent_id in y_of:
            py = y_of[s.parent_id]
            pygame.draw.line(surface, col, (x1, py), (x1, y), 1)

    # Footer
    live   = len(neat.species)
    extinct = sum(1 for s in archive if s.death_gen is not None)
    foot = font.render(f"live:{live}  extinct:{extinct}  total:{len(archive)}",
                       True, (160, 160, 160))
    surface.blit(foot, (rx + rw - foot.get_width() - 4, ry + rh - 14))


# ------------------------------------------------------------------
# Simulation state (mutable container avoids nonlocal headaches)
# ------------------------------------------------------------------
class Sim:
    def __init__(self):
        # Curriculum state — set BEFORE the first track so it uses the right difficulty
        self.curriculum_mode    = True
        self.smoothed_difficulty = 0.15      # start gentle for fresh runs
        self.difficulty_history  = []        # per-generation snapshot for the graph

        self.neat       = NEAT(POPULATION, N_INPUTS, N_OUTPUTS)
        self.track_seed = random.randint(0, 99999)
        self.track      = Track(seed=self.track_seed, difficulty=self.smoothed_difficulty)
        self.heatmap    = Heatmap(self.track.bbox)
        self.cars       = []
        self.generation = 0
        self.sim_time   = 0.0
        self.speed_mult = 1       # 1, 2, 5, 10
        self.paused     = False
        self.show_rays  = True
        self.follow_cam = True
        self.cam_x      = 0.0
        self.cam_y      = 0.0
        self.best_fitness_ever = 0.0
        self.best_gen_ever     = 0
        self.elimination_mode  = False
        self.next_elim_time    = ELIMINATION_START
        self.show_network      = True
        self.show_heatmap      = True

        self.particles       = Particles()
        self.skidmarks       = Skidmarks()
        self.fitness_history = []       # [(best, avg)] appended each generation
        self._smoke_accum    = 0.0      # rate-limiter for smoke emission

        # Ghost racer — best genome ever seen races silently alongside the pop.
        self.ghost_car    = None
        self.ghost_genome = None

        # Multi-track tournament mode
        self.multi_track_mode  = False
        self.multi_track_phase = 0
        self.multi_tracks      = []
        self.collision_physics = True   # toggle with K

        self.show_species_tree = False  # toggle with T
        self.fast_mode = False          # skip visual-only work (burst/headless)
        self.max_gen_seconds = MAX_GEN_SECONDS   # per-Sim, so burst can shorten it

        # Audio, crash cam, spectator
        self.audio          = SoundEngine()
        self.crash_recorder = CrashRecorder()
        self.commentator    = Commentator()
        self.spectator      = Spectator(self.commentator)

        # Hyperparameter sliders (toggle with Y)
        self.slider_panel = SliderPanel([
            Slider("weight mutate rate", 0.0, 1.0,
                   lambda:    _neat_ga_mod.Cfg.weight_mutate_rate,
                   lambda v:  setattr(_neat_ga_mod.Cfg, 'weight_mutate_rate', v)),
            Slider("add connection rate", 0.0, 0.20,
                   lambda:    _neat_ga_mod.Cfg.add_conn_rate,
                   lambda v:  setattr(_neat_ga_mod.Cfg, 'add_conn_rate', v),
                   fmt="{:.3f}"),
            Slider("add node rate", 0.0, 0.20,
                   lambda:    _neat_ga_mod.Cfg.add_node_rate,
                   lambda v:  setattr(_neat_ga_mod.Cfg, 'add_node_rate', v),
                   fmt="{:.3f}"),
            Slider("compat threshold", 0.5, 6.0,
                   lambda:    _neat_ga_mod.Cfg.compat_threshold,
                   lambda v:  setattr(_neat_ga_mod.Cfg, 'compat_threshold', v)),
            Slider("max stagnation", 5, 50,
                   lambda:    _neat_ga_mod.Cfg.max_stagnation,
                   lambda v:  setattr(_neat_ga_mod.Cfg, 'max_stagnation', int(v)),
                   fmt="{:.0f}"),
            Slider("survival ratio", 0.05, 0.6,
                   lambda:    _neat_ga_mod.Cfg.survival_ratio,
                   lambda v:  setattr(_neat_ga_mod.Cfg, 'survival_ratio', v)),
            # --- Vehicle / tire tuning ---
            Slider("tire grip (TIRE_D)", 0.5, 2.5,
                   lambda:    _car_mod.TIRE_D,
                   lambda v:  setattr(_car_mod, 'TIRE_D', v)),
            Slider("tire peak shape (TIRE_C)", 1.0, 2.0,
                   lambda:    _car_mod.TIRE_C,
                   lambda v:  setattr(_car_mod, 'TIRE_C', v)),
            Slider("CG height (weight xfer)", 1.0, 8.0,
                   lambda:    _car_mod.CG_HEIGHT,
                   lambda v:  setattr(_car_mod, 'CG_HEIGHT', v)),
            Slider("engine force", 150, 600,
                   lambda:    _car_mod.ENGINE_FORCE,
                   lambda v:  setattr(_car_mod, 'ENGINE_FORCE', v),
                   fmt="{:.0f}"),
        ])

        # Try to load previous hall-of-fame best
        saved_genome, saved_gen, saved_fit = hof.load_best()
        if saved_fit:
            self.best_fitness_ever = saved_fit
            self.best_gen_ever     = saved_gen
            if saved_genome is not None:
                self.ghost_genome = saved_genome
            print(f"Loaded previous best: gen {saved_gen}, fitness {saved_fit:.0f}")

        self._spawn_cars()

    def _spawn_cars(self):
        import math
        t  = self.track
        cl = t.centerline
        n  = len(cl)
        si = t._start_idx

        cars_per_row = 8
        row_spacing  = 32.0   # desired px between rows along the track
        usable_width = t.width * 0.62
        lat_spacing  = usable_width / (cars_per_row - 1)

        # How many centerline indices equal one row_spacing?
        # Sample a local average step size near the start.
        sample_len = min(40, n)
        avg_step = sum(
            math.hypot(cl[(si + k + 1) % n][0] - cl[(si + k) % n][0],
                       cl[(si + k + 1) % n][1] - cl[(si + k) % n][1])
            for k in range(sample_len)
        ) / sample_len
        idx_per_row = max(1, round(row_spacing / avg_step))

        self.cars.clear()
        for car_idx, genome in enumerate(self.neat.population):
            genome.fitness = 0.0
            row = car_idx // cars_per_row
            col = car_idx % cars_per_row

            # Walk backward along the centerline for this row
            cl_idx = (si - row * idx_per_row) % n

            px, py = cl[cl_idx]
            nx_pt, ny_pt = cl[(cl_idx + 1) % n]
            dx, dy = nx_pt - px, ny_pt - py
            ln = math.hypot(dx, dy) + 1e-9
            fx, fy = dx / ln, dy / ln          # forward along track
            lx, ly = -fy, fx                   # lateral (left)
            heading = math.atan2(fy, fx)

            stagger = lat_spacing * 0.5 if row % 2 == 1 else 0.0
            lat_off = (col - (cars_per_row - 1) / 2.0) * lat_spacing + stagger

            spawn_x = px + lx * lat_off
            spawn_y = py + ly * lat_off

            self.cars.append(Car(spawn_x, spawn_y, heading, genome))

        self._colour_cars()
        self.heatmap.reset()
        self.skidmarks.reset()
        if hasattr(self, 'crash_recorder'):
            self.crash_recorder.reset_generation()

        # Spawn ghost racer (best-ever genome) just ahead of the grid
        if self.ghost_genome is not None:
            import math as _m
            sx, sy, sa = self.track.get_start_pos()
            self.ghost_car = Car(sx, sy, sa, self.ghost_genome.copy())
            self.ghost_car.color = (255, 220, 60)   # gold
        else:
            self.ghost_car = None

        self.sim_time       = 0.0
        self.next_elim_time = ELIMINATION_START

    # Distinct hues for up to 20 species — cycling if more exist
    _SPECIES_PALETTE = [
        (255, 80,  80),   # red
        (80,  160, 255),  # blue
        (80,  255, 120),  # green
        (255, 200, 50),   # yellow
        (220, 80,  255),  # purple
        (80,  240, 240),  # cyan
        (255, 140, 40),   # orange
        (255, 80,  180),  # pink
        (160, 255, 80),   # lime
        (255, 255, 140),  # pale yellow
        (80,  180, 255),  # sky blue
        (200, 140, 255),  # lavender
        (255, 160, 120),  # peach
        (120, 255, 200),  # mint
        (255, 100, 100),  # salmon
        (100, 200, 160),  # teal
        (230, 230, 80),   # gold
        (180, 80,  255),  # violet
        (255, 200, 160),  # cream
        (80,  255, 160),  # aqua
    ]

    def _colour_cars(self):
        """
        Each NEAT species gets its own hue from the palette.
        Within a species, brightness scales with fitness rank so
        the best performer in each species glows brightest.
        """
        # Build a stable species-id → palette index mapping
        species_ids = sorted({c.genome.species_id for c in self.cars})
        sid_to_idx  = {sid: i for i, sid in enumerate(species_ids)}
        n_pal       = len(self._SPECIES_PALETTE)

        # Per-species fitness range for intra-species brightness
        from collections import defaultdict
        sp_fits = defaultdict(list)
        for c in self.cars:
            sp_fits[c.genome.species_id].append(c.fitness)

        for car in self.cars:
            sid      = car.genome.species_id
            base_col = self._SPECIES_PALETTE[sid_to_idx[sid] % n_pal]

            fits = sp_fits[sid]
            lo, hi = min(fits), max(fits)
            rank_t = (car.fitness - lo) / (hi - lo + 1e-9)
            # Dim dead cars; brightest = highest fitness in species
            bright = 0.35 + 0.65 * rank_t if car.alive else 0.18
            car.color = tuple(int(c * bright) for c in base_col)

    def alive_cars(self):
        return [c for c in self.cars if c.alive]

    def step(self, dt):
        import math as _m

        # Snapshot alive set so we can detect deaths this tick
        prev_alive = {id(c) for c in self.cars if c.alive}

        for car in self.cars:
            if car.alive:
                car.update(dt, self.track)

        # Ghost car drives itself (not part of evolution)
        if self.ghost_car and self.ghost_car.alive:
            self.ghost_car.update(dt, self.track)

        if self.collision_physics:
            # Skip particle/audio side effects in fast mode
            if self.fast_mode:
                resolve_car_collisions(self.cars)
            else:
                resolve_car_collisions(self.cars, self.particles, self.audio, self.sim_time)
        apply_drafting(self.cars)

        # Visual-only side effects — skipped during burst / headless
        if not self.fast_mode:
            # Death sparks + crunch sound
            for car in self.cars:
                if id(car) in prev_alive and not car.alive:
                    self.particles.emit_sparks(car.x, car.y,
                                               car.vx * 0.25, car.vy * 0.25, count=8)
                    self.audio.play_crunch(self.sim_time)

            # Drift smoke — rate-limited to ~25 emissions per sim-second
            self._smoke_accum += dt
            if self._smoke_accum >= 0.04:
                self._smoke_accum -= 0.04
                for car in self.cars:
                    if car.alive and car.is_drifting:
                        fx = _m.cos(car.angle)
                        fy = _m.sin(car.angle)
                        self.particles.emit_smoke(
                            car.x - fx * 11.0,
                            car.y - fy * 11.0,
                            car.vx, car.vy, count=1
                        )

            self.particles.update(dt)
            self.skidmarks.update(self.cars)
            self.crash_recorder.record(self.cars, self.generation)
            self.spectator.update(self.alive_cars(), self.sim_time)

            if self.show_heatmap:
                self.heatmap.record(self.cars)

        # Elimination mode — kill the slowest alive car every N seconds
        if self.elimination_mode and self.sim_time >= self.next_elim_time:
            alive = [c for c in self.cars if c.alive]
            if len(alive) > 1:
                slowest = min(alive, key=lambda c: c.fitness)
                slowest.alive = False
            self.next_elim_time += ELIMINATION_INTERVAL

        if int(self.sim_time * 4) % 8 == 0:
            self._colour_cars()

        self.sim_time += dt

        all_dead  = not any(c.alive for c in self.cars)
        timed_out = self.sim_time >= self.max_gen_seconds
        return all_dead or timed_out

    def end_generation(self):
        best_car = max(self.cars, key=lambda c: c.fitness)

        # Fitness history (for the graph panel)
        fits = [c.fitness for c in self.cars]
        self.fitness_history.append((best_car.fitness, float(np.mean(fits))))

        # Curriculum update — adjust smoothed difficulty based on recent fitness
        self._update_curriculum()
        self.difficulty_history.append(self.smoothed_difficulty)

        if best_car.fitness > self.best_fitness_ever:
            self.best_fitness_ever = best_car.fitness
            self.best_gen_ever     = self.generation
            self.ghost_genome      = best_car.genome.copy()   # new ghost champion
            path = hof.save(best_car.genome, self.generation, best_car.fitness)
            print(f"[Gen {self.generation:4d}] New all-time best! "
                  f"fitness={best_car.fitness:.0f}  laps={best_car.laps}  -> {path}")

        # Save top-3 every 5 generations
        if self.generation % 5 == 0:
            sorted_cars = sorted(self.cars, key=lambda c: c.fitness, reverse=True)
            for rank, car in enumerate(sorted_cars[:3], 1):
                hof.save(car.genome, self.generation, car.fitness, f"_top{rank}")

        stats = self.neat.stats()
        print(f"[Gen {self.generation:4d}] best={stats['best']:.0f}  "
              f"mean={stats['mean']:.0f}  species={stats['species']}  "
              f"nodes_best={stats['nodes_best']}")

        self.neat.evolve()
        self.generation += 1

        if self.multi_track_mode:
            self._start_multi_generation()
        else:
            if self.generation % TRACK_CHANGE_EVERY == 0:
                self.track_seed = random.randint(0, 99999)
                self.track = self._new_track(self.track_seed)
                self.heatmap = Heatmap(self.track.bbox)
                print(f"  Track rotated -> seed {self.track_seed}  "
                      f"difficulty {self.current_difficulty():.2f}")
            self._spawn_cars()

    # ------------------------------------------------------------------
    # Multi-track tournament helpers
    # ------------------------------------------------------------------

    def _start_multi_generation(self):
        """Generate N_MULTI_TRACKS fresh tracks for the new generation."""
        self.multi_tracks = [
            self._new_track() for _ in range(N_MULTI_TRACKS)
        ]
        self.multi_track_phase = 0
        self.track   = self.multi_tracks[0]
        self.heatmap = Heatmap(self.track.bbox)
        self._spawn_cars()
        seeds = [t.seed for t in self.multi_tracks]
        print(f"  Multi-track gen {self.generation}: seeds {seeds}")

    # Curriculum tuning constants
    CURRICULUM_TARGET_FIT = 8000.0   # mean fitness at which difficulty saturates to 1.0
    CURRICULUM_WINDOW     = 5        # rolling window (generations)
    CURRICULUM_RISE_RATE  = 0.15     # smoothing per gen when difficulty climbs
    CURRICULUM_FALL_RATE  = 0.35     # faster reaction when fitness crashes

    def current_difficulty(self):
        """Smoothed difficulty. 1.0 when curriculum is off."""
        if not self.curriculum_mode:
            return 1.0
        return float(max(0.0, min(1.0, self.smoothed_difficulty)))

    def _update_curriculum(self):
        """
        Update smoothed difficulty from a rolling-window mean fitness.
        Asymmetric — climbs slowly, drops quickly. Means a population
        that crashes (after a physics tweak or worse track) gets relief fast.
        """
        if not self.fitness_history:
            return
        recent = self.fitness_history[-self.CURRICULUM_WINDOW:]
        recent_mean = sum(h[1] for h in recent) / len(recent)
        target = max(0.0, min(1.0, recent_mean / self.CURRICULUM_TARGET_FIT))
        rate   = (self.CURRICULUM_FALL_RATE if target < self.smoothed_difficulty
                  else self.CURRICULUM_RISE_RATE)
        self.smoothed_difficulty += (target - self.smoothed_difficulty) * rate

    def _new_track(self, seed=None):
        """Track factory that honours curriculum mode."""
        return Track(seed=seed if seed is not None else random.randint(0, 99999),
                     difficulty=self.current_difficulty())

    def _advance_multi_track_phase(self):
        """Carry accumulated fitness forward and move cars to the next track."""
        # Snapshot fitness earned on this phase for each genome
        saved = {id(g): g.fitness for g in self.neat.population}

        self.multi_track_phase += 1
        self.track   = self.multi_tracks[self.multi_track_phase]
        self.heatmap = Heatmap(self.track.bbox)
        self._spawn_cars()

        # Apply saved fitness as a bonus so the next-phase fitness adds on top
        for car in self.cars:
            car.fitness_bonus  = saved.get(id(car.genome), 0.0)
            car.genome.fitness = car.fitness_bonus

        print(f"  Multi-track phase {self.multi_track_phase + 1}/{N_MULTI_TRACKS}")

    # ------------------------------------------------------------------
    # Save / Resume training state
    # ------------------------------------------------------------------

    STATE_PATH = "training_state.json"

    def save_state(self):
        import json
        innov_map_serial = {
            f"{k[0]},{k[1]}": v
            for k, v in self.neat.innov._map.items()
        }
        data = {
            "generation":        self.generation,
            "track_seed":        self.track_seed,
            "best_fitness_ever": self.best_fitness_ever,
            "best_gen_ever":     self.best_gen_ever,
            "neat_generation":   self.neat.generation,
            "neat_sid":          self.neat._sid,
            "innov_map":         innov_map_serial,
            "innov_counter":     self.neat.innov._counter,
            "population":        [g.to_dict() for g in self.neat.population],
            "species": [
                {
                    "id":           sp.id,
                    "representative": sp.representative.to_dict(),
                    "best_fitness": sp.best_fitness,
                    "stagnation":   sp.stagnation,
                    "age":          sp.age,
                }
                for sp in self.neat.species
            ],
        }
        with open(self.STATE_PATH, "w") as f:
            json.dump(data, f)
        print(f"[Gen {self.generation}] Training state saved -> {self.STATE_PATH}")

    def load_state(self):
        import json, os
        from brain import Genome
        if not os.path.exists(self.STATE_PATH):
            print("No saved training state found — start a fresh run first.")
            return

        with open(self.STATE_PATH) as f:
            data = json.load(f)

        self.generation        = data["generation"]
        self.track_seed        = data["track_seed"]
        self.best_fitness_ever = data["best_fitness_ever"]
        self.best_gen_ever     = data["best_gen_ever"]

        # Restore track (curriculum difficulty for the current best-ever)
        self.track   = self._new_track(self.track_seed)
        self.heatmap = Heatmap(self.track.bbox)

        # Restore innovation tracker
        tracker = InnovationTracker(start=data["innov_counter"])
        for k_str, v in data["innov_map"].items():
            a, b = map(int, k_str.split(","))
            tracker._map[(a, b)] = v
        self.neat.innov      = tracker
        self.neat.generation = data["neat_generation"]
        self.neat._sid       = data["neat_sid"]

        # Restore population
        self.neat.population = [Genome.from_dict(d) for d in data["population"]]

        # Restore species shells (representatives + metadata)
        self.neat.species = []
        for sp_data in data["species"]:
            rep = Genome.from_dict(sp_data["representative"])
            sp  = Species(sp_data["id"], rep)
            sp.best_fitness = sp_data["best_fitness"]
            sp.stagnation   = sp_data["stagnation"]
            sp.age          = sp_data["age"]
            self.neat.species.append(sp)

        # Re-speciate assigns every genome to a species
        self.neat._speciate()

        self._spawn_cars()
        print(f"[Gen {self.generation}] Training state loaded from {self.STATE_PATH}")

    def best_alive(self):
        alive = self.alive_cars()
        if alive:
            return max(alive, key=lambda c: c.fitness)
        return max(self.cars, key=lambda c: c.fitness)

    def update_camera(self, target_x, target_y, smooth=0.15):
        target_cam_x = target_x - SCREEN_W / 2
        target_cam_y = target_y - SCREEN_H / 2
        self.cam_x += (target_cam_x - self.cam_x) * smooth
        self.cam_y += (target_cam_y - self.cam_y) * smooth


# ------------------------------------------------------------------
# HUD rendering
# ------------------------------------------------------------------

MINIMAP_W = 260
MINIMAP_H = 200


NET_W, NET_H   = 240, 160   # neural net visualiser panel size


def draw_hud(surface, sim, font_sm, font_lg, fps_actual, offset):
    # ---- Left info panel ----
    panel = pygame.Surface((318, 382), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 170))
    surface.blit(panel, (8, 8))

    alive       = len(sim.alive_cars())
    best_car    = sim.best_alive()
    species_cnt = len(sim.neat.species)
    hof_files   = hof.list_all()

    n      = len(sim.track.centerline)
    px_len = sum(
        ((sim.track.centerline[(i+1) % n][0] - sim.track.centerline[i][0])**2 +
         (sim.track.centerline[(i+1) % n][1] - sim.track.centerline[i][1])**2) ** 0.5
        for i in range(0, n, 4)
    ) * 4
    km_per_lap = px_len / 2000.0

    elim_tag = " [ELIM]" if sim.elimination_mode else ""
    next_e   = max(0.0, sim.next_elim_time - sim.sim_time) if sim.elimination_mode else 0

    lines = [
        (f"Generation  : {sim.generation}", (180, 255, 180)),
        (f"Alive       : {alive}/{POPULATION}{elim_tag}", (180, 255, 180)),
        (f"Best fit    : {best_car.fitness:,.0f}", (180, 255, 180)),
        (f"Best laps   : {best_car.laps}", (180, 255, 180)),
        (f"Lap length  : {km_per_lap:.1f} km ({km_per_lap*0.621:.1f} mi)", (180, 255, 180)),
        (f"Best speed  : {best_car.speed:.0f} px/s  "
         f"draft:{best_car.draft_boost*100:.0f}%", (180, 255, 180)),
        (f"Species     : {species_cnt}", (180, 255, 180)),
        (f"All-time    : {sim.best_fitness_ever:,.0f} (gen {sim.best_gen_ever})", (255, 220, 100)),
        (f"Track seed  : {sim.track_seed}", (180, 180, 180)),
        (f"Saved brains: {len(hof_files)}", (180, 180, 180)),
        (f"Ghost laps  : {sim.ghost_car.laps if sim.ghost_car else '─  (no champion yet)'}", (220, 190, 60)),
        (f"Multi-track : {'ON  phase ' + str(sim.multi_track_phase + 1) + '/' + str(N_MULTI_TRACKS) if sim.multi_track_mode else 'OFF'}", (100, 200, 255)),
        (f"Curriculum  : {'ON  diff ' + format(sim.current_difficulty(), '.2f') if sim.curriculum_mode else 'OFF (full)'}", (120, 220, 160)),
        ("", (0, 0, 0)),
        (f"[F]{sim.speed_mult}x [SPC]{'RUN' if sim.paused else 'PAUSE'} [R]Skip [E]Elim{'✓' if sim.elimination_mode else ' '}", (120, 120, 120)),
        (f"[H]Replay [V]Net [D]Rays [B]Heat [K]Collide{'✓' if sim.collision_physics else '✗'} [M]Multi", (120, 120, 120)),
        (f"[T]Tree [U]Curric{'✓' if sim.curriculum_mode else '✗'} [X]Burst{BURST_GENS}g [O]Audio{'✓' if sim.audio.enabled else '✗'}", (120, 120, 120)),
        (f"[P]Crash [Y]Sliders [Z]Spectate{'✓' if sim.spectator.enabled else '✗'} [S]Save [L]Load [N]Track", (120, 120, 120)),
        (f"FPS: {fps_actual:.0f}" + (f"  next cull: {next_e:.0f}s" if sim.elimination_mode else ""),
         (120, 120, 120)),
    ]

    for i, (text, col) in enumerate(lines):
        surf = font_sm.render(text, True, col)
        surface.blit(surf, (16, 16 + i * 18))

    if sim.speed_mult > 1:
        ff = font_lg.render(f">> {sim.speed_mult}x", True, (255, 210, 50))
        surface.blit(ff, (SCREEN_W - MINIMAP_W - 10 - ff.get_width() - 8, 16))

    if sim.paused:
        pt = font_lg.render("PAUSED", True, (255, 80, 80))
        surface.blit(pt, (SCREEN_W // 2 - pt.get_width() // 2, SCREEN_H // 2 - 20))

    if sim.elimination_mode:
        et = font_lg.render("ELIMINATION", True, (255, 80, 60))
        surface.blit(et, (SCREEN_W // 2 - et.get_width() // 2, 14))

    # Progress bar
    bar_w    = SCREEN_W - 20
    progress = min(sim.sim_time / MAX_GEN_SECONDS, 1.0)
    pygame.draw.rect(surface, (50, 50, 50), (10, SCREEN_H - 14, bar_w, 6))
    pygame.draw.rect(surface, (70, 170, 70), (10, SCREEN_H - 14, int(bar_w * progress), 6))

    # ---- Neural net panel (above minimap) ----
    if sim.show_network and best_car.genome:
        nv_x = SCREEN_W - MINIMAP_W - 10
        nv_y = SCREEN_H - MINIMAP_H - NET_H - 28
        draw_network(surface, best_car.genome, (nv_x, nv_y, NET_W, NET_H), font_sm)

    # ---- Mini-map ----
    mm_x = SCREEN_W - MINIMAP_W - 10
    mm_y = SCREEN_H - MINIMAP_H - 20
    sim.track.draw_minimap(
        surface, (mm_x, mm_y, MINIMAP_W, MINIMAP_H),
        sim.cars, offset, SCREEN_W, SCREEN_H,
    )


def draw_replay_hud(surface, session, font_sm, font_lg, offset):
    """Separate HUD shown during hall-of-fame replay."""
    panel = pygame.Surface((318, 120), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 180))
    surface.blit(panel, (8, 8))

    best = session.best_alive()
    lines = [
        ("HALL OF FAME REPLAY", (255, 220, 60)),
        (f"Brains loaded : {session.loaded_count}", (180, 255, 180)),
        (f"Alive         : {session.alive_count}", (180, 255, 180)),
        (f"Best fit      : {best.fitness:,.0f}" if best else "No cars", (180, 255, 180)),
        ("[H] / [ESC]  return to training", (120, 120, 120)),
    ]
    for i, (text, col) in enumerate(lines):
        surf = font_sm.render(text, True, col)
        surface.blit(surf, (16, 16 + i * 18))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

BURST_GENS = 25   # generations to run per X-press


def burst_train(sim, screen, font_lg, font_sm, n_gens=BURST_GENS):
    """
    Run n_gens generations with no per-frame rendering, showing a
    progress overlay that updates mid-generation. ESC aborts early.
    """
    import time
    DT             = 1.0 / 60.0
    BURST_GEN_SEC  = 45.0         # short cap during burst — fast iteration
    MAX_STEPS      = int(BURST_GEN_SEC / DT)
    PUMP_EVERY     = 25           # sim steps between event pumps + UI refreshes

    t0        = time.time()
    gens_done = 0
    aborted   = False

    # Enter fast mode: skip particles, skidmarks, heatmap, crash recording, audio
    prev_fast      = sim.fast_mode
    prev_audio_on  = sim.audio.enabled
    prev_max_sec   = sim.max_gen_seconds
    sim.fast_mode      = True
    sim.audio.stop_all()
    sim.audio.enabled  = False
    sim.max_gen_seconds = BURST_GEN_SEC

    def draw_progress(sub_progress=0.0):
        screen.fill((20, 20, 26))
        title = font_lg.render("BURST TRAINING", True, (255, 200, 50))
        screen.blit(title,
                    (SCREEN_W // 2 - title.get_width() // 2, SCREEN_H // 2 - 130))

        info = font_lg.render(
            f"gen {gens_done}/{n_gens}   best_ever {sim.best_fitness_ever:,.0f}",
            True, (220, 220, 220))
        screen.blit(info,
                    (SCREEN_W // 2 - info.get_width() // 2, SCREEN_H // 2 - 70))

        # Main progress bar (generations)
        bar_w = 600
        bar_x = SCREEN_W // 2 - bar_w // 2
        bar_y = SCREEN_H // 2 - 20
        pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_w, 20))
        gen_fill = (gens_done + sub_progress) / max(n_gens, 1)
        pygame.draw.rect(screen, (70, 170, 70), (bar_x, bar_y, int(bar_w * gen_fill), 20))
        pygame.draw.rect(screen, (200, 200, 200), (bar_x, bar_y, bar_w, 20), 1)

        # Sub-progress bar (within current generation)
        sub_y = bar_y + 28
        pygame.draw.rect(screen, (40, 40, 40), (bar_x, sub_y, bar_w, 6))
        pygame.draw.rect(screen, (120, 180, 255), (bar_x, sub_y, int(bar_w * sub_progress), 6))

        elapsed   = time.time() - t0
        completed = gens_done + sub_progress
        rate      = completed / max(elapsed, 1e-6)
        eta       = (n_gens - completed) / max(rate, 1e-6) if rate > 0 else 0.0
        meta = font_sm.render(
            f"{rate:.2f} gen/s   elapsed {elapsed:.0f}s   ETA {eta:.0f}s   "
            f"alive {sum(1 for c in sim.cars if c.alive)}/{POPULATION}",
            True, (160, 160, 160))
        screen.blit(meta,
                    (SCREEN_W // 2 - meta.get_width() // 2, SCREEN_H // 2 + 24))

        hint = font_sm.render("[ESC] to abort", True, (140, 140, 140))
        screen.blit(hint,
                    (SCREEN_W // 2 - hint.get_width() // 2, SCREEN_H // 2 + 54))
        pygame.display.flip()

    def pump_events():
        nonlocal aborted
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                aborted = True

    draw_progress(0.0)

    while gens_done < n_gens and not aborted:
        # One generation, with periodic UI refreshes
        sim_done = False
        for step_i in range(MAX_STEPS):
            if sim.step(DT):
                sim_done = True
                break
            if step_i % PUMP_EVERY == 0:
                pump_events()
                if aborted:
                    break
                draw_progress(step_i / MAX_STEPS)

        if aborted:
            break

        if (sim.multi_track_mode and
                sim.multi_track_phase < N_MULTI_TRACKS - 1):
            sim._advance_multi_track_phase()
        else:
            sim.end_generation()
            gens_done += 1
            draw_progress(0.0)

    # Restore prior state
    sim.fast_mode       = prev_fast
    sim.audio.enabled   = prev_audio_on
    sim.max_gen_seconds = prev_max_sec

    elapsed = time.time() - t0
    tag = "aborted" if aborted else "complete"
    print(f"Burst {tag}: {gens_done} gens in {elapsed:.1f}s "
          f"({gens_done/max(elapsed,1):.2f} gen/s)  "
          f"best_ever={sim.best_fitness_ever:.0f}")


def headless_train(n_generations, save_every=25):
    """
    Pure training loop with no rendering. Massively faster than the visual mode.
    Uses a dummy SDL video driver so pygame surfaces still work for
    any code that touches them, but nothing is displayed.
    """
    import os, time
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    pygame.init()

    sim = Sim()
    sim.collision_physics = False   # cleaner training signal
    sim.fast_mode = True            # skip all visual side-effects
    sim.audio.enabled = False
    # Heavy fixed timestep — fast but stable enough for sim physics
    DT = 1.0 / 60.0
    MAX_STEPS = int(MAX_GEN_SECONDS / DT)

    print(f"Headless training: {n_generations} generations  "
          f"(pop {POPULATION}, save every {save_every})")
    t0 = time.time()

    for g in range(n_generations):
        for _ in range(MAX_STEPS):
            done = sim.step(DT)
            if done:
                break
        sim.end_generation()

        if (g + 1) % save_every == 0:
            sim.save_state()

        if (g + 1) % 10 == 0:
            elapsed = time.time() - t0
            rate = (g + 1) / elapsed
            print(f"  gen {sim.generation:4d}  best_ever={sim.best_fitness_ever:.0f}  "
                  f"species={len(sim.neat.species)}  "
                  f"{rate:.1f} gen/s")

    sim.save_state()
    elapsed = time.time() - t0
    print(f"Done. {n_generations} gens in {elapsed:.1f}s "
          f"({n_generations/max(elapsed,1):.1f} gen/s). "
          f"Best fitness ever: {sim.best_fitness_ever:.0f}")
    pygame.quit()


def main():
    # CLI: python main.py --headless 500
    if len(sys.argv) >= 2 and sys.argv[1] == "--headless":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
        headless_train(n)
        return

    pygame.init()
    screen   = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Neural Racer — NEAT Evolution")
    clock    = pygame.time.Clock()
    font_sm  = pygame.font.SysFont("monospace", 13)
    font_lg  = pygame.font.SysFont("monospace", 22, bold=True)

    sim           = Sim()
    speed_options = [1, 2, 5, 10]
    speed_idx     = 0
    replay        = None   # ReplaySession or None

    running = True
    while running:
        dt_real    = clock.tick(FPS) / 1000.0
        fps_actual = clock.get_fps()

        # ----------------------------------------------------------------
        # Events
        # ----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                # ESC exits replay if active, otherwise quits
                if event.key == pygame.K_ESCAPE:
                    if replay:
                        replay = None
                    else:
                        running = False

                elif event.key == pygame.K_q and not replay:
                    running = False

                # --- Replay toggle ---
                elif event.key == pygame.K_h:
                    if replay:
                        replay = None
                    else:
                        replay = ReplaySession(sim.track)
                        if not replay.cars:
                            print("No saved brains found in hall_of_fame/")
                            replay = None

                # --- Training controls (ignored while in replay) ---
                elif not replay:
                    if event.key == pygame.K_f:
                        speed_idx = (speed_idx + 1) % len(speed_options)
                        sim.speed_mult = speed_options[speed_idx]

                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
                        speed_idx = min(speed_idx + 1, len(speed_options) - 1)
                        sim.speed_mult = speed_options[speed_idx]

                    elif event.key == pygame.K_MINUS:
                        speed_idx = max(speed_idx - 1, 0)
                        sim.speed_mult = speed_options[speed_idx]

                    elif event.key == pygame.K_SPACE:
                        sim.paused = not sim.paused

                    elif event.key == pygame.K_r:
                        sim.sim_time = MAX_GEN_SECONDS + 999.0

                    elif event.key == pygame.K_d:
                        sim.show_rays = not sim.show_rays

                    elif event.key == pygame.K_v:
                        sim.show_network = not sim.show_network

                    elif event.key == pygame.K_e:
                        sim.elimination_mode = not sim.elimination_mode
                        sim.next_elim_time   = sim.sim_time + ELIMINATION_START
                        print(f"Elimination mode: {'ON' if sim.elimination_mode else 'OFF'}")

                    elif event.key == pygame.K_c:
                        sim.follow_cam = not sim.follow_cam

                    elif event.key == pygame.K_n:
                        sim.track_seed = random.randint(0, 99999)
                        sim.track = sim._new_track(sim.track_seed)
                        sim.heatmap = Heatmap(sim.track.bbox)
                        sim._spawn_cars()
                        if replay:
                            replay = ReplaySession(sim.track)
                        print(f"New track: seed {sim.track_seed}")

                    elif event.key == pygame.K_b:
                        sim.show_heatmap = not sim.show_heatmap

                    elif event.key == pygame.K_k:
                        sim.collision_physics = not sim.collision_physics
                        print(f"Collision physics: {'ON' if sim.collision_physics else 'OFF'}")

                    elif event.key == pygame.K_t:
                        sim.show_species_tree = not sim.show_species_tree

                    elif event.key == pygame.K_u:
                        sim.curriculum_mode = not sim.curriculum_mode
                        print(f"Curriculum learning: {'ON' if sim.curriculum_mode else 'OFF'}")

                    elif event.key == pygame.K_x:
                        # Burst-train N generations with no rendering
                        burst_train(sim, screen, font_lg, font_sm)

                    elif event.key == pygame.K_o:
                        on = sim.audio.toggle()
                        print(f"Audio: {'ON' if on else 'OFF'}")

                    elif event.key == pygame.K_p:
                        # Slow-mo replay of the most dramatic crash
                        hl = sim.crash_recorder.best_highlight()
                        if hl is None:
                            print("No crashes recorded yet.")
                        else:
                            sim.audio.stop_all()
                            play_replay(hl, screen, sim.track, font_sm, font_lg,
                                        SCREEN_W, SCREEN_H)

                    elif event.key == pygame.K_y:
                        sim.slider_panel.toggle()

                    elif event.key == pygame.K_z:
                        sim.spectator.toggle()
                        print(f"Spectator broadcast: {'ON' if sim.spectator.enabled else 'OFF'}")

                    elif event.key == pygame.K_m:
                        sim.multi_track_mode = not sim.multi_track_mode
                        if sim.multi_track_mode:
                            sim._start_multi_generation()
                        print(f"Multi-track tournament: {'ON' if sim.multi_track_mode else 'OFF'}")

                    elif event.key == pygame.K_s:
                        sim.save_state()

                    elif event.key == pygame.K_l:
                        sim.load_state()

            # Mouse events flow to the slider panel
            if (event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEBUTTONUP,
                               pygame.MOUSEMOTION) and not replay):
                sim.slider_panel.handle_event(event)

        # ----------------------------------------------------------------
        # Simulation / replay update
        # ----------------------------------------------------------------
        if replay:
            replay.update(dt_real)
            # Auto-restart replay when all cars are dead
            if replay.alive_count == 0:
                replay = ReplaySession(sim.track)
        elif not sim.paused:
            dt_step = dt_real / sim.speed_mult
            for _ in range(sim.speed_mult):
                done = sim.step(dt_step)
                if done:
                    if (sim.multi_track_mode and
                            sim.multi_track_phase < N_MULTI_TRACKS - 1):
                        sim._advance_multi_track_phase()
                    else:
                        sim.end_generation()
                    break

        # ----------------------------------------------------------------
        # Camera
        # ----------------------------------------------------------------
        if replay:
            best = replay.best_alive()
            if best:
                sim.update_camera(best.x, best.y)
        else:
            best = sim.best_alive()
            # Spectator overrides the normal camera target
            if sim.spectator.enabled and sim.spectator.current_car is not None:
                sc = sim.spectator.current_car
                sim.update_camera(sc.x, sc.y)
            elif sim.follow_cam:
                sim.update_camera(best.x, best.y)
        offset = (int(sim.cam_x), int(sim.cam_y))

        # ----------------------------------------------------------------
        # Draw
        # ----------------------------------------------------------------
        screen.fill((28, 28, 32))
        sim.track.draw(screen, offset)
        if sim.show_heatmap and not replay:
            sim.heatmap.draw(screen, offset, max_speed=540)

        if replay:
            replay.draw(screen, offset, font_sm)
            draw_replay_hud(screen, replay, font_sm, font_lg, offset)
        else:
            import math as _m2

            # Skid marks (on track surface, behind cars)
            sim.skidmarks.draw(screen, offset, SCREEN_W, SCREEN_H)

            # Dead cars — faint dots
            for car in sim.cars:
                if not car.alive:
                    pygame.draw.circle(screen, (65, 65, 65),
                                       (int(car.x - offset[0]), int(car.y - offset[1])), 3)

            # Ghost racer — drawn before live cars so it sits underneath
            if sim.ghost_car:
                gc = sim.ghost_car
                gc.draw(screen, offset, draw_rays=False)
                if gc.alive:
                    gsx = int(gc.x - offset[0])
                    gsy = int(gc.y - offset[1])
                    # Glowing ring to distinguish it from the population
                    pygame.draw.circle(screen, (255, 240, 90), (gsx, gsy), 13, 1)
                    lbl = font_sm.render("GHOST", True, (255, 220, 60))
                    screen.blit(lbl, (gsx - lbl.get_width() // 2, gsy - 26))

            # Alive cars — sorted so best renders on top
            alive_sorted = sorted(sim.alive_cars(), key=lambda c: c.fitness)
            for i, car in enumerate(alive_sorted):
                draw_r = sim.show_rays and (i == len(alive_sorted) - 1)
                car.draw(screen, offset, draw_rays=draw_r)

            # Slipstream cone — faint yellow cone behind the best car
            if best and best.draft_boost > 0.05:
                bx = int(best.x - offset[0])
                by = int(best.y - offset[1])
                angle = best.angle + _m2.pi
                for spread in (-0.45, 0.0, 0.45):
                    ea = angle + spread
                    ex = int(bx + _m2.cos(ea) * 80 * best.draft_boost)
                    ey = int(by + _m2.sin(ea) * 80 * best.draft_boost)
                    pygame.draw.line(screen, (180, 160, 40), (bx, by), (ex, ey), 1)

            # Particles on top of cars, under HUD
            sim.particles.draw(screen, offset, SCREEN_W, SCREEN_H)

            # Spectator highlight ring around the current car
            if sim.spectator.enabled and sim.spectator.current_car and sim.spectator.current_car.alive:
                sc = sim.spectator.current_car
                ssx = int(sc.x - offset[0])
                ssy = int(sc.y - offset[1])
                pygame.draw.circle(screen, (255, 220, 120), (ssx, ssy), 22, 2)
                pygame.draw.circle(screen, (255, 100, 100), (ssx, ssy), 28, 1)

            # Audio: in spectator mode, follow the spectated car instead of leader
            audio_target = sim.spectator.current_car if (
                sim.spectator.enabled and sim.spectator.current_car) else best
            if audio_target:
                sim.audio.update(audio_target.speed, audio_target.is_drifting)

            draw_hud(screen, sim, font_sm, font_lg, fps_actual, offset)

            # Slider panel (top-right, above the network panel)
            if sim.slider_panel.visible:
                sp_w, sp_h = 300, 380
                sp_x = SCREEN_W - sp_w - 10
                sp_y = 8
                sim.slider_panel.draw(screen, font_sm, (sp_x, sp_y, sp_w, sp_h))

            # Spectator commentary banner (always on top, when enabled)
            sim.spectator.draw(screen, font_sm, font_lg, SCREEN_W, SCREEN_H)

            # Fitness history graph (below info panel)
            gy0 = 8 + 382 + 6
            draw_fitness_graph(screen, sim,
                               (8, gy0, GRAPH_W, GRAPH_H), font_sm)

            # Species tree (below fitness graph) — toggle with T
            if sim.show_species_tree:
                draw_species_tree(screen, sim.neat,
                                  (8, gy0 + GRAPH_H + 6, TREE_W, TREE_H),
                                  font_sm, Sim._SPECIES_PALETTE)

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
