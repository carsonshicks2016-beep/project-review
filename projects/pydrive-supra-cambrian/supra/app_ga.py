"""
Live genetic-algorithm view — watch a population learn to drive.

A fixed, zoomed-to-fit camera shows the whole track. Each generation, the full
population launches from the start line; most crash early, a few make it further,
and over generations they visibly get better. A side readout shows generation,
alive count, best/mean fitness, and a fitness-history chart. The champion of each
generation is saved so you can `--watch` it in the full cockpit view.

Time is accelerated automatically: as cars crash and fewer remain, the sim steps
faster per frame (and Space fast-forwards the rest of a generation).
"""
from __future__ import annotations

import numpy as np

from .config import EvoSpec, SimSpec
from .evolution import GA

COMPUTE_BUDGET = 150     # vehicle-steps per frame target (auto time-accel)


def _fit_camera(trk, view_w, view_h, pad=0.88):
    c = trk.center
    mn, mx = c.min(axis=0), c.max(axis=0)
    span = np.maximum(mx - mn, 1e-3)
    scale = min(view_w / span[0], view_h / span[1]) * pad
    cx, cy = (mn + mx) / 2
    return scale, cx, cy


def run(car: str = "supra", seeds=(7,), generations: int = 200,
        pop: int | None = None, checkpoint: str = "ga_champion.npz",
        track=None, track_name=None, resume=None):
    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame is required. Install: pip install pygame")
    from .carart import draw_car
    from .background import TrackEnvironmentBackground

    evo = EvoSpec()
    if pop:
        evo.pop_size = pop
    sim = SimSpec()
    ga = GA(evo, car=car, seeds=tuple(seeds),
            tracks=([track] if track is not None else None), track_name=track_name)
    if resume:
        import os
        if os.path.exists(resume):
            ck = GA.load_champion(resume)
            GA.check_champion(ck, ga.obs_size, resume)
            ga.warm_start(ck["genome"], ck["fitness"])
            print(f"resumed {resume} (gen {ck['generation']})")
        else:
            print(f"resume file not found: {resume}")
    trk = ga.tracks[0]

    pygame.init()
    W, H = 1480, 820
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption(f"Supra Drift — GA evolution [{car}]")
    clock = pygame.time.Clock()
    f = pygame.font.SysFont("menlo,consolas,monospace", 15)
    fb = pygame.font.SysFont("menlo,consolas,monospace", 26, bold=True)

    PANEL = 320
    view_w = W - PANEL
    
    # If the track is massive (like the Nordschleife), use a zoomed-in follow camera.
    # Otherwise, try to fit it on screen.
    if trk.length > 5000:
        scale = 6.0
        ccx, ccy = trk.start_pose()[0:2]
    else:
        scale, ccx, ccy = _fit_camera(trk, view_w, H)

    def to_screen(p, cx_cam, cy_cam):
        return (int((p[0] - cx_cam) * scale + view_w / 2),
                int((p[1] - cy_cam) * scale + H / 2))

    # Matching ground/terrain tint for visual consistency with the immersive
    # views. This is a whole-track overview, so we draw only the (cheap) terrain
    # floor — the per-prop detail lives in the drive/watch follow-cam views.
    bg_layer = TrackEnvironmentBackground(trk, seed=7)
    import types
    bg_cam = types.SimpleNamespace(
        cx=float(ccx), cy=float(ccy), scale=scale, view_all=(trk.length <= 5000),
        to_screen=lambda x, y: to_screen((x, y), bg_cam.cx, bg_cam.cy),
    )

    agents = ga.make_agents(ga.genomes, trk)
    gen_running = True
    last_stats = {"best": 0.0, "mean": 0.0}
    
    top_speed_kmh = 0.0
    fastest_lap_time = 1e9

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

        # ---- simulate ----
        if gen_running:
            alive = [a for a in agents if not a.done]
            if not alive:
                gen_running = False
            else:
                if fast:                                    # finish the gen now
                    for a in alive:
                        a.run()
                    gen_running = False
                else:
                    steps = int(np.clip(COMPUTE_BUDGET // len(alive), 2, 40))
                    for _ in range(steps):
                        for a in alive:
                            if not a.done:
                                a.step(sim.dt)
                                spd = a.veh.speed * 3.6
                                if spd > top_speed_kmh:
                                    top_speed_kmh = spd
                                if a.max_cum >= 1.0 and a.time < fastest_lap_time:
                                    fastest_lap_time = a.time
        else:
            # generation complete -> breed, checkpoint, relaunch
            fits = np.array([a.fitness for a in agents])
            last_stats = ga.advance(fits)
            ga.save_champion(checkpoint)
            if ga.generation >= generations:
                running = False
            else:
                agents = ga.make_agents(ga.genomes, trk)
                gen_running = True

        # leader = furthest along this generation
        leader = max(agents, key=lambda a: a.max_cum, default=None)
        if leader:
            # camera smoothing
            ccx += (leader.veh.x - ccx) * 0.1
            ccy += (leader.veh.y - ccy) * 0.1
            
        bg_cam.cx, bg_cam.cy = ccx, ccy

        # ---- draw ----
        bg_layer.draw_terrain(screen, bg_cam)
        
        # Only draw the track geometry around the camera if we are zoomed in, 
        # or all of it if we are zoomed out.
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
        pygame.draw.circle(screen, (90, 220, 120), to_screen(trk.center[0], ccx, ccy), 5)  # start

        to_xy = lambda x, y: to_screen((x, y), ccx, ccy)
        for a in agents:
            if a.done:
                continue
            # short trail
            if len(a.trail) > 1:
                tp = [to_screen(q, ccx, ccy) for q in a.trail[-12:]]
                if len(tp) > 1:
                    pygame.draw.lines(screen, a.color, False, tp, 1)
            draw_car(screen, to_xy, scale, a.veh, a.veh.spec, color=a.color)
            if a is leader:
                pygame.draw.circle(screen, (255, 255, 255),
                                   to_screen((a.veh.x, a.veh.y), ccx, ccy), 8, 1)

        _panel(pygame, screen, f, fb, W, H, PANEL, ga, agents, last_stats, trk, 
               leader, top_speed_kmh, fastest_lap_time)
        pygame.display.flip()

    pygame.quit()


def _panel(pygame, screen, f, fb, W, H, PANEL, ga, agents, stats, trk, leader, top_speed_kmh, fastest_lap_time):
    x0 = W - PANEL
    pygame.draw.rect(screen, (28, 30, 36), (x0, 0, PANEL, H))
    pygame.draw.line(screen, (60, 64, 72), (x0, 0), (x0, H), 2)
    x = x0 + 18
    alive = sum(1 for a in agents if not a.done)
    best_now = max((a.max_cum for a in agents), default=0.0)

    screen.blit(fb.render(f"GEN {ga.generation}", True, (235, 238, 245)), (x, 18))
    y = 58
    
    leader_spd = leader.veh.speed * 3.6 if (leader and not leader.done) else 0.0
    lap_str = f"{fastest_lap_time:.2f}s" if fastest_lap_time < 1e8 else "--"
    
    rows = [
        (f"alive      {alive}/{len(agents)}", (200, 205, 212)),
        (f"best lap   {best_now:.2f}", (120, 210, 140)),
        (f"champion   {ga.best_fitness/trk.length:.2f} laps", (235, 200, 90)),
        (f"last best  {stats.get('best', 0)/trk.length:.2f} laps", (180, 185, 195)),
        (f"last mean  {stats.get('mean', 0)/trk.length:.2f} laps", (150, 155, 165)),
        ("", (0,0,0)),
        (f"LEADER SPD {leader_spd:3.0f} km/h", (110, 200, 230)),
        (f"TOP SPEED  {top_speed_kmh:3.0f} km/h", (210, 130, 230)),
        (f"FAST LAP   {lap_str}", (235, 120, 120)),
    ]
    for txt, col in rows:
        if txt:
            screen.blit(f.render(txt, True, col), (x, y))
        y += 24

    # fitness history chart
    y += 12
    screen.blit(f.render("fitness / generation", True, (140, 145, 155)), (x, y)); y += 18
    cw, ch = PANEL - 36, 150
    pygame.draw.rect(screen, (18, 20, 24), (x, y, cw, ch))
    hist = ga.history
    if len(hist) >= 2:
        bests = np.array([h[1] for h in hist]) / trk.length
        means = np.array([h[2] for h in hist]) / trk.length
        top = max(bests.max(), 0.3)
        def pts(arr):
            return [(int(x + i / (len(arr) - 1) * cw),
                     int(y + ch - (v / top) * ch)) for i, v in enumerate(arr)]
        pygame.draw.lines(screen, (235, 200, 90), False, pts(bests), 2)
        pygame.draw.lines(screen, (110, 160, 230), False, pts(means), 1)
        screen.blit(f.render(f"{top:.1f}", True, (110, 112, 120)), (x + cw - 30, y + 2))
    y += ch + 14
    screen.blit(f.render("best", True, (235, 200, 90)), (x, y))
    screen.blit(f.render("mean", True, (110, 160, 230)), (x + 70, y)); y += 26

    for line in ["SPACE  finish gen", "ESC    quit",
                 "saves champion each gen"]:
        screen.blit(f.render(line, True, (130, 135, 145)), (x, y)); y += 20
