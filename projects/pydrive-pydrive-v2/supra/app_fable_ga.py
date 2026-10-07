"""
Live view of the Fable GA population learning the Nordschleife.

The whole 787B population launches from the start line each generation; most spin
off in the first corners, a few carry speed further, and over generations they
visibly get better — the same "watch evolution happen" view as the classic
`app_ga`, but driving the *same* `FableEnv` (fable-v1 obs, learned RaceBox gear)
that the PPO pipeline trains on. Breeding, the deterministic `[eval-ga]` champion
eval, and checkpointing are all delegated to `FableGA`, so the live and headless
runs share identical genetics and yardstick.

The on-screen fitness is each car's accumulated `FableReward` from a standing
line launch (one start, for a clear mass-launch picture). The heavier multi-start
fitness lives in the headless trainer; use `--fable-ga` (no `--live`) for the
strongest run, and this view to watch it happen.
"""
from __future__ import annotations

import numpy as np

from .fable_ga import (FableGA, GA_CHECKPOINT, GA_EVAL_LATEST,
                       ga_log_line, hyper_for_pop, write_eval_snapshot)
from .fable5 import SUPERHUMAN_LAP, stage_defaults

# per-frame budget of env control-steps (auto time-accel: as cars crash and
# fewer remain, each steps more per frame so a generation still finishes fast)
STEP_BUDGET = 90


def _norm_apply(x, mean, var, clip):
    return np.clip((x - mean) / np.sqrt(var + 1e-8), -clip, clip)


class _Car:
    __slots__ = ("env", "brain", "obs", "done", "fit", "progress_m", "trail",
                 "color", "index")

    def __init__(self, env, brain, color, index):
        self.env = env
        self.brain = brain
        self.color = color
        self.index = index
        self.reset()

    def reset(self):
        self.obs = self.env.reset_at(0, speed=0.0)   # standing line launch
        self.done = False
        self.fit = 0.0
        self.progress_m = 0.0
        self.trail = [(self.env.veh.x, self.env.veh.y)]

    def step(self, mean, var, clip):
        if self.done:
            return
        a = self.brain.forward(_norm_apply(self.obs, mean, var, clip))
        self.obs, r, term, trunc, info = self.env.step(a)
        self.fit += float(r)
        self.progress_m = max(self.progress_m, float(info.get("fable_progress_m", 0.0)))
        if len(self.trail) < 4000:
            self.trail.append((self.env.veh.x, self.env.veh.y))
        if term or trunc or (info.get("laps", 0.0) >= 1.0
                             and not info.get("fable_invalid", False)):
            self.done = True


def _palette(n):
    import colorsys
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i / max(1, n)) * 0.85, 0.62, 0.96)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def run(stage: str = "frontier", generations: int = 200, pop: int | None = None,
        seed: int = 0, checkpoint: str = GA_CHECKPOINT, resume: str | None = None,
        hidden=None, fitness_starts=None, fitness_seconds=None,
        snapshot: str = GA_EVAL_LATEST, max_frames: int | None = None):
    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame is required. Install: pip3 install pygame")
    from .carart import draw_car
    from .background import TrackEnvironmentBackground
    from .brain import MLP

    # a modest default population for the live view (rendering + Python physics);
    # the headless trainer runs a bigger pop. Live pop is capped for framerate.
    live_pop = pop or 28
    hyper = hyper_for_pop(live_pop, hidden=hidden, starts=fitness_starts,
                          fitness_seconds=fitness_seconds)
    spec = stage_defaults(stage)
    print(f"[fable-ga/live] building {live_pop}-car population on the "
          f"Nordschleife (stage {stage})...", flush=True)
    ga = FableGA(hyper, spec, seed=seed)
    if resume:
        import os
        if os.path.exists(resume):
            ck = FableGA.load_champion(resume)
            ga.warm_start(ck["genome"], norm_mean=ck.get("norm_mean"),
                          norm_var=ck.get("norm_var"), norm_clip=ck.get("norm_clip"),
                          fitness=ck["fitness"], metric=ck["metric"])
            print(f"[fable-ga/live] resumed {resume} (gen {ck['generation']})",
                  flush=True)
        else:
            print(f"[fable-ga/live] resume not found: {resume}", flush=True)

    trk = ga.track
    mean, var, clip = ga.norm.mean, ga.norm.var, ga.norm.clip
    colors = _palette(len(ga.genomes))

    def build_cars():
        cars = []
        for i, g in enumerate(ga.genomes):
            env = ga._build_env()
            brain = MLP.from_genome(g, ga.obs_dim, ga.h.hidden, 3,
                                    out_activation="fable", bias_long=ga.h.bias_long)
            cars.append(_Car(env, brain, colors[i % len(colors)], i))
        return cars

    cars = build_cars()

    pygame.init()
    W, H = 1480, 820
    flags = 0
    screen = pygame.display.set_mode((W, H), flags)
    pygame.display.set_caption(f"Fable GA — evolving the 787B on the Nordschleife")
    clock = pygame.time.Clock()
    f = pygame.font.SysFont("menlo,consolas,monospace", 15)
    fb = pygame.font.SysFont("menlo,consolas,monospace", 26, bold=True)

    PANEL = 330
    view_w = W - PANEL
    scale = 6.0
    ccx, ccy = trk.start_pose()[0:2]

    def to_screen(p, cx, cy):
        return (int((p[0] - cx) * scale + view_w / 2),
                int((p[1] - cy) * scale + H / 2))

    bg_layer = TrackEnvironmentBackground(trk, seed=7)
    import types
    bg_cam = types.SimpleNamespace(
        cx=float(ccx), cy=float(ccy), scale=scale, view_all=False,
        to_screen=lambda x, y: to_screen((x, y), bg_cam.cx, bg_cam.cy))

    last_eval = {}
    top_speed = 0.0
    frames = 0
    running = True
    while running:
        clock.tick(120)
        fast = False
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_SPACE:
                    fast = True

        alive = [c for c in cars if not c.done]
        if not alive:
            # generation complete -> breed, checkpoint, eval, relaunch
            fits = np.array([c.fit for c in cars], dtype=float)
            stats = ga.advance(fits)
            ga.save_champion(checkpoint)
            due = (ga.generation - 1) % ga.h.eval_every == 0
            if due:
                last_eval = ga.champion_eval()
                print("  " + ga_log_line(stage, ga.generation, last_eval), flush=True)
                write_eval_snapshot(ga, stats, last_eval, snapshot)
            if ga.generation >= generations:
                running = False
            else:
                colors[:] = _palette(len(ga.genomes))
                cars = build_cars()
            continue

        # ---- simulate ----
        if fast:
            for c in alive:
                for _ in range(600):
                    if c.done:
                        break
                    c.step(mean, var, clip)
        else:
            steps = int(np.clip(STEP_BUDGET // max(1, len(alive)), 1, 20))
            for _ in range(steps):
                for c in alive:
                    if not c.done:
                        c.step(mean, var, clip)
                        sp = c.env.veh.speed * 3.6
                        if sp > top_speed:
                            top_speed = sp

        leader = max(cars, key=lambda c: c.progress_m, default=None)
        if leader is not None:
            ccx += (leader.env.veh.x - ccx) * 0.1
            ccy += (leader.env.veh.y - ccy) * 0.1
        bg_cam.cx, bg_cam.cy = ccx, ccy

        # ---- draw ----
        bg_layer.draw_terrain(screen, bg_cam)
        left = [to_screen(p, ccx, ccy) for p in trk.left]
        right = [to_screen(p, ccx, ccy) for p in trk.right]
        centerline = [to_screen(p, ccx, ccy) for p in trk.center]
        tarmac = left + right[::-1]
        if len(tarmac) > 2:
            pygame.draw.polygon(screen, (44, 46, 52), tarmac)
        for i in range(0, len(centerline) - 1, 5):
            pygame.draw.line(screen, (90, 90, 60), centerline[i], centerline[i + 1], 1)
        pygame.draw.lines(screen, (120, 70, 70), True, left, 2)
        pygame.draw.lines(screen, (120, 70, 70), True, right, 2)
        pygame.draw.circle(screen, (90, 220, 120),
                           to_screen(trk.center[0], ccx, ccy), 5)

        to_xy = lambda x, y: to_screen((x, y), ccx, ccy)
        for c in cars:
            if c.done:
                continue
            if len(c.trail) > 1:
                tp = [to_screen(q, ccx, ccy) for q in c.trail[-14:]]
                if len(tp) > 1:
                    pygame.draw.lines(screen, c.color, False, tp, 1)
            draw_car(screen, to_xy, scale, c.env.veh, c.env.veh.spec, color=c.color)
            if c is leader:
                pygame.draw.circle(screen, (255, 255, 255),
                                   to_screen((c.env.veh.x, c.env.veh.y), ccx, ccy),
                                   8, 1)

        _panel(pygame, screen, f, fb, W, H, PANEL, ga, cars, last_eval,
               trk, leader, top_speed, stage)
        pygame.display.flip()

        frames += 1
        if max_frames and frames >= max_frames:
            running = False

    pygame.quit()


def _panel(pygame, screen, f, fb, W, H, PANEL, ga, cars, ev, trk, leader,
           top_speed, stage):
    x0 = W - PANEL
    pygame.draw.rect(screen, (28, 30, 36), (x0, 0, PANEL, H))
    pygame.draw.line(screen, (60, 64, 72), (x0, 0), (x0, H), 2)
    x = x0 + 18
    alive = sum(1 for c in cars if not c.done)
    best_prog = max((c.progress_m for c in cars), default=0.0)
    lead_spd = leader.env.veh.speed * 3.6 if (leader and not leader.done) else 0.0
    lap = (ev or {}).get("lap_time") or 0.0
    lap_s = f"{lap:.2f}s" if lap else "--"
    L = trk.length

    screen.blit(fb.render(f"GEN {ga.generation}", True, (235, 238, 245)), (x, 18))
    screen.blit(f.render(f"GA vs PPO — stage {stage}", True, (150, 160, 175)), (x, 48))
    y = 74
    rows = [
        (f"alive       {alive}/{len(cars)}", (200, 205, 212)),
        (f"lead prog   {best_prog / L * 100:5.1f}% of lap", (120, 210, 140)),
        (f"lead dist   {best_prog:6.0f} m", (150, 200, 150)),
        ("", None),
        (f"champ metric {ga.best_metric:7.3f}", (235, 200, 90)),
        (f"champ lap    {lap_s}", (235, 160, 90)),
        (f"Bellof       {SUPERHUMAN_LAP:.2f}s", (180, 150, 120)),
        (f"theoretical  {trk.fable_envelope['lap_time']:.1f}s", (150, 150, 150)),
        ("", None),
        (f"LEAD SPD    {lead_spd:3.0f} km/h", (110, 200, 230)),
        (f"TOP SPEED   {top_speed:3.0f} km/h", (210, 130, 230)),
        (f"sigma       {ga.sigma:.3f}", (150, 155, 165)),
    ]
    for txt, col in rows:
        if txt:
            screen.blit(f.render(txt, True, col), (x, y))
        y += 24

    # fitness history chart (best / mean accumulated reward per generation)
    y += 8
    screen.blit(f.render("fitness / generation", True, (140, 145, 155)), (x, y))
    y += 18
    cw, ch = PANEL - 36, 140
    pygame.draw.rect(screen, (18, 20, 24), (x, y, cw, ch))
    hist = ga.history
    if len(hist) >= 2:
        bests = np.array([h[1] for h in hist])
        means = np.array([h[2] for h in hist])
        lo = float(min(bests.min(), means.min()))
        hi = float(max(bests.max(), 1.0))
        rng = max(1e-6, hi - lo)

        def pts(arr):
            return [(int(x + i / (len(arr) - 1) * cw),
                     int(y + ch - (v - lo) / rng * ch)) for i, v in enumerate(arr)]
        pygame.draw.lines(screen, (235, 200, 90), False, pts(bests), 2)
        pygame.draw.lines(screen, (110, 160, 230), False, pts(means), 1)
    y += ch + 12
    screen.blit(f.render("best", True, (235, 200, 90)), (x, y))
    screen.blit(f.render("mean", True, (110, 160, 230)), (x + 70, y))
    y += 26
    for line in ["SPACE  fast-forward gen", "ESC    quit",
                 "saves champion each gen", "[eval-ga] every "
                 f"{ga.h.eval_every} gens"]:
        screen.blit(f.render(line, True, (130, 135, 145)), (x, y))
        y += 20


# --------------------------------------------------------------------------- #
# FOLLOW MODE — a visually cheap, READ-ONLY window onto a headless training run.
#
# Instead of running its own competing evolution, this drives the *current
# champion* the headless trainer is writing to `ga_787b_ring.npz`, plus a few
# mutated variants for a swarm feel. It polls the checkpoint and hot-swaps to the
# new champion whenever training banks a better one, so you literally watch the
# best brain improve. It NEVER writes the checkpoint/snapshot, so it can't
# collide with the trainer — run one headless job and as many followers as you
# like.
# --------------------------------------------------------------------------- #
def _brains_from_champion(genome, obs_dim, hidden, bias_long, n, rng,
                          sigma: float = 0.05, rate: float = 0.12):
    """Champion (index 0, unmutated) + n-1 mutated variants — a cheap visual
    proxy of the neighbourhood the GA is exploring around its best brain."""
    from .brain import MLP
    base = np.asarray(genome, dtype=float).ravel()
    genomes = [base.copy()]
    for _ in range(max(0, n - 1)):
        g = base.copy()
        mask = rng.random(g.size) < rate
        g += mask * rng.standard_normal(g.size) * sigma
        genomes.append(g)
    return [MLP.from_genome(g, obs_dim, hidden, 3, out_activation="fable",
                            bias_long=bias_long) for g in genomes]


_TRACK_NP = {}          # id(trk) -> (left, right, center) as float arrays (cached)


def _np_track(trk):
    key = id(trk)
    t = _TRACK_NP.get(key)
    if t is None:
        t = (np.asarray(trk.left, dtype=float), np.asarray(trk.right, dtype=float),
             np.asarray(trk.center, dtype=float))
        _TRACK_NP[key] = t
    return t


def _draw_world(pygame, screen, trk, cars, champion, to_screen, bg_layer, bg_cam,
                draw_car, ccx, ccy, view_w, H):
    bg_layer.draw_terrain(screen, bg_cam)
    left_a, right_a, center_a = _np_track(trk)
    scale = bg_cam.scale
    # The Ring is 20.8 km; at this zoom only ~100 m is on screen. Redrawing the
    # whole loop every frame (a multi-thousand-vertex filled polygon) was the
    # entire render cost. Draw only a window of track around the camera instead,
    # transformed with numpy (vectorised) rather than a Python per-point loop.
    d2 = (center_a[:, 0] - ccx) ** 2 + (center_a[:, 1] - ccy) ** 2
    i0 = int(d2.argmin())
    idx = np.arange(i0 - 220, i0 + 220)

    def proj(arr):
        seg = np.take(arr, idx, axis=0, mode="wrap")
        sx = ((seg[:, 0] - ccx) * scale + view_w / 2).astype(int)
        sy = ((seg[:, 1] - ccy) * scale + H / 2).astype(int)
        return list(zip(sx.tolist(), sy.tolist()))

    left = proj(left_a)
    right = proj(right_a)
    centerline = proj(center_a)
    tarmac = left + right[::-1]
    if len(tarmac) > 2:
        pygame.draw.polygon(screen, (44, 46, 52), tarmac)
    for i in range(0, len(centerline) - 1, 5):
        pygame.draw.line(screen, (90, 90, 60), centerline[i], centerline[i + 1], 1)
    if len(left) > 1:                      # open polyline (window, not the full loop)
        pygame.draw.lines(screen, (120, 70, 70), False, left, 2)
    if len(right) > 1:
        pygame.draw.lines(screen, (120, 70, 70), False, right, 2)
    s0 = to_screen(trk.center[0], ccx, ccy)   # start/finish dot, only if on screen
    if -40 < s0[0] < view_w + 40 and -40 < s0[1] < H + 40:
        pygame.draw.circle(screen, (90, 220, 120), s0, 5)
    to_xy = lambda x, y: to_screen((x, y), ccx, ccy)
    # variants first, champion last (drawn on top, highlighted)
    for c in cars:
        if c is champion or c.done:
            continue
        if len(c.trail) > 1:
            tp = [to_screen(q, ccx, ccy) for q in c.trail[-14:]]
            if len(tp) > 1:
                pygame.draw.lines(screen, c.color, False, tp, 1)
        draw_car(screen, to_xy, bg_cam.scale, c.env.veh, c.env.veh.spec, color=c.color)
    if champion is not None and not champion.done:
        if len(champion.trail) > 1:
            tp = [to_screen(q, ccx, ccy) for q in champion.trail[-20:]]
            if len(tp) > 1:
                pygame.draw.lines(screen, (255, 240, 180), False, tp, 2)
        draw_car(screen, to_xy, bg_cam.scale, champion.env.veh, champion.env.veh.spec,
                 color=(255, 255, 255))
        pygame.draw.circle(screen, (255, 240, 150),
                           to_screen((champion.env.veh.x, champion.env.veh.y), ccx, ccy), 9, 2)


def run_follow(checkpoint: str = GA_CHECKPOINT, pop: int | None = None,
               seed: int = 0, snapshot: str = GA_EVAL_LATEST,
               max_frames: int | None = None):
    """Cheap read-only viewer: drive the live training champion + variants,
    hot-reloading whenever the trainer banks a new best."""
    import os
    import time
    import json
    import types
    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame is required. Install: pip3 install pygame")
    from .carart import draw_car
    from .background import TrackEnvironmentBackground
    from .config import SimSpec
    from .fable5 import (FableEnv, RING_TRACK, attach_envelope, stage_defaults,
                         _configure_ppo)
    from .track import named_track
    from .ppo_env import RunningNorm

    swarm = pop or 8           # a cheap swarm — this is a viewer, not a trainer
    rng = np.random.default_rng(seed)

    pygame.init()
    W, H = 1480, 820
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Fable GA — following the live training champion")
    clock = pygame.time.Clock()
    f = pygame.font.SysFont("menlo,consolas,monospace", 15)
    fb = pygame.font.SysFont("menlo,consolas,monospace", 26, bold=True)
    PANEL = 330
    view_w = W - PANEL
    frames = 0

    def _load_ck():
        try:
            return FableGA.load_champion(checkpoint)
        except Exception:
            return None

    def _quit_requested():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                    ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                return True
        return False

    # wait for the training to bank its first champion
    ck = _load_ck() if os.path.exists(checkpoint) else None
    while ck is None:
        if _quit_requested():
            pygame.quit(); return
        screen.fill((16, 18, 22))
        screen.blit(fb.render("waiting for the training to bank a champion…",
                              True, (200, 205, 215)), (60, H // 2 - 20))
        screen.blit(f.render(f"following {os.path.basename(checkpoint)} — start a "
                             f"headless run, or it'll appear at the first eval",
                             True, (130, 135, 145)), (60, H // 2 + 16))
        pygame.display.flip(); clock.tick(20)
        frames += 1
        if max_frames and frames >= max_frames:
            pygame.quit(); return
        if os.path.exists(checkpoint):
            ck = _load_ck()

    # reconstruct the exact env + obs normaliser the champion was trained under
    car = ck["car"]; obs_dim = ck["obs_dim"]; hidden = ck["hidden"]
    bias_long = ck["bias_long"]
    spec = stage_defaults(ck["stage"])
    cfg = _configure_ppo(spec)
    cfg.random_start = False
    cfg.episode_seconds = 1e9                 # drive until crash/lap, then relaunch
    sim = SimSpec()
    trk = attach_envelope(named_track(RING_TRACK), car)
    norm = RunningNorm(obs_dim)
    if ck.get("norm_mean") is not None and ck.get("norm_var") is not None:
        norm.mean = np.asarray(ck["norm_mean"])
        norm.var = np.asarray(ck["norm_var"])
        norm.clip = float(ck.get("norm_clip", 5.0))

    envs = [FableEnv(mode="race", car=car, ppo=cfg, sim=sim, fixed_track=trk,
                     fable_spec=spec, rng_seed=0, diagnostics=False)
            for _ in range(swarm)]
    colors = [(255, 255, 255)] + _palette(max(1, swarm - 1))

    def seed_cars(genome):
        brains = _brains_from_champion(genome, obs_dim, hidden, bias_long, swarm, rng)
        return [_Car(envs[i], brains[i], colors[i % len(colors)], i)
                for i in range(swarm)]

    cars = seed_cars(ck["genome"])
    ck_mtime = os.path.getmtime(checkpoint)

    scale = 6.0
    ccx, ccy = trk.start_pose()[0:2]

    def to_screen(p, cx, cy):
        return (int((p[0] - cx) * scale + view_w / 2),
                int((p[1] - cy) * scale + H / 2))

    bg_layer = TrackEnvironmentBackground(trk, seed=7)
    bg_cam = types.SimpleNamespace(
        cx=float(ccx), cy=float(ccy), scale=scale, view_all=False,
        to_screen=lambda x, y: to_screen((x, y), bg_cam.cx, bg_cam.cy))

    def read_snapshot():
        try:
            with open(snapshot, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}

    snap = read_snapshot()
    last_poll = time.time()
    running = True
    while running:
        clock.tick(120)
        if _quit_requested():
            running = False

        # hot-reload the champion when the trainer banks a new best (read-only)
        now = time.time()
        if now - last_poll > 1.5:
            last_poll = now
            try:
                mt = os.path.getmtime(checkpoint)
            except OSError:
                mt = ck_mtime
            if mt != ck_mtime:
                ck_mtime = mt
                nk = _load_ck()
                if nk is not None and np.asarray(nk["genome"]).size == \
                        cars[0].brain.get_genome().size:
                    if nk.get("norm_mean") is not None:
                        norm.mean = np.asarray(nk["norm_mean"])
                        norm.var = np.asarray(nk["norm_var"])
                        norm.clip = float(nk.get("norm_clip", norm.clip))
                    ck = nk
                    cars = seed_cars(nk["genome"])
                snap = read_snapshot()

        # Relaunch anything that crashed/looped LAST frame (once — resetting a
        # dead car on every substep was a reset-storm when the champion is still
        # bad and cars die instantly). Then step everyone a few substeps. A viewer
        # wants smooth frames over sim throughput, so keep per-frame work modest.
        for c in cars:
            if c.done:
                c.reset()
        # Total FableEnv.step calls per frame is the real cost (~2 ms each: beam
        # sensors + physics). Cap it low — ~8/frame at 30 fps is ~real-time motion,
        # which is exactly right for a cruising viewer. This is the fps knob.
        steps = max(1, 8 // swarm)
        for _ in range(steps):
            for c in cars:
                if not c.done:
                    c.step(norm.mean, norm.var, norm.clip)

        champ = cars[0]
        leader = max(cars, key=lambda c: c.progress_m, default=champ)
        focus = champ if not champ.done else leader
        if focus is not None:
            ccx += (focus.env.veh.x - ccx) * 0.1
            ccy += (focus.env.veh.y - ccy) * 0.1
        bg_cam.cx, bg_cam.cy = ccx, ccy

        _draw_world(pygame, screen, trk, cars, champ, to_screen, bg_layer, bg_cam,
                    draw_car, ccx, ccy, view_w, H)
        _panel_follow(pygame, screen, f, fb, W, H, PANEL, ck, snap, cars, trk, checkpoint)
        pygame.display.flip()
        frames += 1
        if max_frames and frames >= max_frames:
            running = False
    pygame.quit()


def _panel_follow(pygame, screen, f, fb, W, H, PANEL, ck, snap, cars, trk, checkpoint):
    import os
    x0 = W - PANEL
    pygame.draw.rect(screen, (28, 30, 36), (x0, 0, PANEL, H))
    pygame.draw.line(screen, (60, 64, 72), (x0, 0), (x0, H), 2)
    x = x0 + 18
    ev = (snap or {}).get("eval", {}) or {}
    gen = snap.get("generation", ck.get("generation", 0))
    metric = snap.get("best_metric", ck.get("metric", 0.0))
    lap = ev.get("lap_time") or ck.get("lap_time") or 0.0
    lap_s = f"{lap:.2f}s" if lap else "--"
    champ = cars[0]
    champ_spd = champ.env.veh.speed * 3.6 if not champ.done else 0.0

    screen.blit(fb.render(f"GEN {gen}", True, (235, 238, 245)), (x, 18))
    screen.blit(f.render("following live training", True, (150, 200, 150)), (x, 48))
    y = 74
    rows = [
        (f"champion   {os.path.basename(checkpoint)}", (150, 155, 165)),
        (f"metric     {float(metric):7.3f}", (235, 200, 90)),
        (f"champ lap  {lap_s}", (235, 160, 90)),
        (f"clean      {ev.get('clean_sectors','--')}/{ev.get('sector_count',16)}", (120, 210, 140)),
        (f"pace       {(ev.get('pace_ratio') or 0)*100:.0f}% of envelope", (120, 200, 210)),
        (f"progress   {(ev.get('max_progress_m') or 0):.0f} m", (150, 200, 150)),
        ("", None),
        (f"Bellof     {SUPERHUMAN_LAP:.2f}s", (180, 150, 120)),
        (f"theoretical {trk.fable_envelope['lap_time']:.1f}s", (150, 150, 150)),
        (f"CHAMP SPD  {champ_spd:3.0f} km/h", (255, 255, 255)),
        (f"variants   {len(cars) - 1}", (150, 155, 165)),
    ]
    for txt, col in rows:
        if txt:
            screen.blit(f.render(txt, True, col), (x, y))
        y += 24

    # champion metric history (best metric per eval, from the snapshot)
    y += 8
    screen.blit(f.render("best fitness / generation", True, (140, 145, 155)), (x, y))
    y += 18
    cw, ch = PANEL - 36, 150
    pygame.draw.rect(screen, (18, 20, 24), (x, y, cw, ch))
    hist = (snap or {}).get("history", [])
    if len(hist) >= 2:
        bests = np.array([h[1] for h in hist])
        means = np.array([h[2] for h in hist])
        lo = float(min(bests.min(), means.min()))
        hi = float(max(bests.max(), 1.0))
        rng = max(1e-6, hi - lo)

        def pts(arr):
            return [(int(x + i / (len(arr) - 1) * cw),
                     int(y + ch - (v - lo) / rng * ch)) for i, v in enumerate(arr)]
        pygame.draw.lines(screen, (235, 200, 90), False, pts(bests), 2)
        pygame.draw.lines(screen, (110, 160, 230), False, pts(means), 1)
    y += ch + 14
    screen.blit(f.render("white car = current champion", True, (235, 235, 235)), (x, y)); y += 20
    screen.blit(f.render("colours = mutated variants", True, (150, 180, 210)), (x, y)); y += 20
    for line in ["read-only — never writes", "reloads on each new best",
                 "ESC  quit"]:
        screen.blit(f.render(line, True, (130, 135, 145)), (x, y)); y += 20
