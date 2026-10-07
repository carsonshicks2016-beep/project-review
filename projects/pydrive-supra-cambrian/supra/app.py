"""
2D PyGame application — drive the car yourself.

This is the slice-1 deliverable: a window, a track, and a physically-simulated
car you steer with the keyboard. It exists to validate that the physics *feel*
right before any AI is bolted on. Cameras / particles / cinematic modes get
richer in later slices; for now it's a clean chase cam + a readable HUD.

Controls
  Up / W .......... throttle          Left/Right or A/D .. steer
  Down / S ........ brake             Space .............. handbrake
  Shift ........... clutch (hold to disengage)
  Q / E ........... manual down / up shift
  T ............... toggle auto gearbox (on by default)
  Tab ............. toggle telemetry dashboard
  B ............... toggle raycast vision beams
  M ............... mute / unmute engine audio
  R ............... reset to start    Esc ................ quit
"""
from __future__ import annotations

import numpy as np

from .config import CarSpec, SimSpec, get_car
from .physics import Controls, Vehicle
from . import track as track_mod


# --------------------------------------------------------------------------- #
# auto gearbox: keeps the car easy to drive with just throttle/brake/steer
# --------------------------------------------------------------------------- #
class AutoBox:
    def __init__(self, spec: CarSpec):
        self.spec = spec
        self.shift_cooldown = 0.0
        self.clutch = 0.0   # start disengaged: no creep at a standstill

    def update(self, veh: Vehicle, throttle: float, dt: float):
        """Return (clutch, shift_up, shift_down) for an automatic transmission."""
        self.shift_cooldown = max(0.0, self.shift_cooldown - dt)
        up = down = False
        rpm = veh.rpm
        if self.shift_cooldown == 0.0:
            if rpm > self.spec.redline_rpm * 0.95 and veh.gear < len(self.spec.gear_ratios):
                up = True
                self.shift_cooldown = 0.4
            elif rpm < 1500 and veh.gear > 1 and veh.speed > 1.0:
                down = True
                self.shift_cooldown = 0.4
        # clutch logic, mimicking an automatic: dip during a shift; otherwise
        # disengage whenever off-throttle and the engine is near idle (at a stop
        # OR coasting/braking down to idle revs) so there's no creep and no
        # idle-governor torque fighting the brakes in a low gear. Engaged
        # otherwise, so lift-off engine braking still works at speed.
        idle_thresh = self.spec.idle_rpm * 1.25
        if up or down:
            target = 0.2
        elif throttle < 0.05 and (veh.speed < 1.5 or rpm < idle_thresh):
            target = 0.0
        else:
            target = 1.0
        self.clutch += (target - self.clutch) * min(1.0, dt * 12.0)
        return self.clutch, up, down


# --------------------------------------------------------------------------- #
# rendering helpers
# --------------------------------------------------------------------------- #
class Camera:
    def __init__(self, w, h, scale=6.0, center=None):
        self.w, self.h = w, h
        self.base_scale = scale
        self.scale = scale
        self.cx = self.cy = 0.0
        self.center_x, self.center_y = center if center else (w / 2, h / 2)
        self.shake_x = self.shake_y = 0.0       # transient camera-shake offset (px)

    def follow(self, x, y, lerp=0.12):
        self.cx += (x - self.cx) * lerp
        self.cy += (y - self.cy) * lerp

    def dynamics(self, speed, slide, dt, enabled=True):
        """Cinematic feel: zoom OUT a touch at speed (see further ahead) and add
        a small shake while sliding hard. Disabled -> snap back to the static cam."""
        import numpy as np
        if enabled:
            f = float(np.clip(speed / 62.0, 0.0, 1.0))           # 0..1 by ~220 km/h
            target = self.base_scale * (1.0 - 0.22 * f)
            # gentle, SMOOTHED shake (ease toward a new random target instead of
            # snapping each frame) so it reads as a subtle wobble, not a jitter.
            amp = float(np.clip(slide, 0.0, 1.0)) * 1.3 + f * 0.25
            tx = float(np.random.uniform(-amp, amp))
            ty = float(np.random.uniform(-amp, amp))
            self.shake_x += (tx - self.shake_x) * 0.3
            self.shake_y += (ty - self.shake_y) * 0.3
        else:
            target = self.base_scale
            self.shake_x += (0.0 - self.shake_x) * 0.3
            self.shake_y += (0.0 - self.shake_y) * 0.3
        self.scale += (target - self.scale) * min(1.0, dt * 4.0)

    def to_screen(self, x, y):
        sx = (x - self.cx) * self.scale + self.center_x + self.shake_x
        sy = (y - self.cy) * self.scale + self.center_y + self.shake_y
        # NaN/inf guard: int(nan) raises and a runaway physics state would
        # crash the viewer mid-drive — park the point far off-screen instead
        # (pygame clips it) and keep rendering.
        if not (np.isfinite(sx) and np.isfinite(sy)):
            return -32768, -32768
        # clamp to pygame's safe 16-bit coordinate range (huge-but-finite
        # values overflow SDL's short-based rects the same way)
        return (int(max(-32768.0, min(32767.0, sx))),
                int(max(-32768.0, min(32767.0, sy))))


def _color_shift(color, amount):
    return tuple(max(0, min(255, int(c + amount))) for c in color)


def _mix_color(a, b, t):
    t = float(np.clip(t, 0.0, 1.0))
    return tuple(max(0, min(255, int(a[i] * (1.0 - t) + b[i] * t))) for i in range(3))


def draw_road(pygame, screen, cam, trk, c_tar, c_edge, c_line):
    """Draw the tarmac as per-segment quads culled to the viewport, plus its edges
    and dashed centerline. Long real tracks are windowed around the camera so the
    renderer does not project thousands of off-screen points every frame."""
    W, H = screen.get_size()
    M = len(trk.center)
    if M < 2:
        return

    def on(p):
        return -60 <= p[0] <= W + 60 and -60 <= p[1] <= H + 60

    if getattr(cam, "view_all", False):
        indices = range(M)
    else:
        avg_step = max(0.5, float(getattr(trk, "length", float(M))) / float(M))
        radius_world = (float(np.hypot(W, H)) * 0.55 / max(float(cam.scale), 0.05)
                        + float(getattr(trk, "width", 10.0)) * 3.0 + 24.0)
        window = max(36, int(radius_world / avg_step) + 10)
        if window * 2 + 2 >= M:
            indices = range(M)
        else:
            if hasattr(trk, "nearest"):
                center_idx = int(trk.nearest(cam.cx, cam.cy))
            else:
                diff = trk.center - [cam.cx, cam.cy]
                center_idx = int(np.argmin(diff[:, 0] ** 2 + diff[:, 1] ** 2))
            indices = [(center_idx + off) % M for off in range(-window, window + 1)]

    edge_w = max(1, min(4, int(cam.scale * 0.10)))
    center_w = max(1, min(3, int(cam.scale * 0.07)))
    curb_w = max(2, min(5, edge_w + 1))
    guard_col = _mix_color(c_edge, (225, 230, 230), 0.35)
    dark_edge = _mix_color(c_edge, (8, 9, 11), 0.35)
    curv_arr = getattr(trk, "curvature", None)
    if curv_arr is None:
        curv_arr = np.zeros(M)

    for i in indices:
        j = (i + 1) % M
        a = cam.to_screen(*trk.left[i])
        b = cam.to_screen(*trk.left[j])
        c = cam.to_screen(*trk.right[j])
        d = cam.to_screen(*trk.right[i])
        if not (on(a) or on(b) or on(c) or on(d)):
            continue
        band = ((i // 10) % 5) - 2
        curvature = float(curv_arr[i])
        surf_warmth = min(4.0, abs(curvature) * 75.0)
        tar_col = _color_shift(c_tar, band + surf_warmth)
        pygame.draw.polygon(screen, tar_col, (a, b, c, d))
        pygame.draw.line(screen, dark_edge, a, b, edge_w + 1)
        pygame.draw.line(screen, dark_edge, d, c, edge_w + 1)
        pygame.draw.line(screen, c_edge, a, b, edge_w)
        pygame.draw.line(screen, c_edge, d, c, edge_w)

        if abs(curvature) > 0.011 and i % 3 != 2:
            curb_col = (214, 42, 36) if (i // 3) % 2 == 0 else (236, 238, 226)
            if curvature > 0.0:
                pygame.draw.line(screen, curb_col, a, b, curb_w)
                if i % 9 == 0:
                    pygame.draw.line(screen, guard_col, d, c, max(1, edge_w))
            else:
                pygame.draw.line(screen, curb_col, d, c, curb_w)
                if i % 9 == 0:
                    pygame.draw.line(screen, guard_col, a, b, max(1, edge_w))

        if i % 18 < 8:
            sc_i = cam.to_screen(*trk.center[i])
            sc_j = cam.to_screen(*trk.center[j])
            pygame.draw.line(screen, _mix_color(c_line, tar_col, 0.25), sc_i, sc_j, center_w)


def run(car: str = "supra", track_name: str = "random", seed: int | None = 7,
        audio_on: bool = True, controller=None, track=None, title=None,
        max_frames=None, agent=None, start_speed: float = 0.0, opponents: list = None,
        racers: list = None, sensor_spec=None):
    """Drive the car. If `controller` is given (a callable taking the vehicle and
    the current observation, returning (steer, throttle, brake)), it drives
    instead of the keyboard — used to watch a trained brain. `track` overrides the
    generated one (e.g. to replay the exact training circuit)."""
    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame is required to drive. Install: pip install pygame")

    from .sensors import SensorSuite
    from .dashboard import Dashboard, draw_beams
    from .carart import draw_car as draw_car_art
    from .fx import FX, SkidTrail
    from .background import TrackEnvironmentBackground
    from . import aiviz

    pygame.init()
    W, H = 1480, 820
    DASH_W = Dashboard.WIDTH
    screen = pygame.display.set_mode((W, H), vsync=1)
    pygame.display.set_caption(title or f"Supra Drift — {car} [{track_name}]")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 16)
    small = pygame.font.SysFont("menlo,consolas,monospace", 12)

    sim = SimSpec()
    if racers:
        opponents = racers
        controller = None
        agent = None
        veh = opponents[0].veh
        spec = opponents[0].spec
    else:
        spec = get_car(car)
        veh = Vehicle(spec, sim)

    sensors = SensorSuite(sensor_spec)
    dash = Dashboard()
    from .sound import SpatialAudioMixer
    audio = SpatialAudioMixer()

    import sys
    old_hook = sys.excepthook
    def safe_audio_teardown(exc_type, exc_value, exc_traceback):
        import traceback
        with open("FATAL_CRASH.txt", "w") as f:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        try: audio.stop()
        except: pass
        old_hook(exc_type, exc_value, exc_traceback)
    sys.excepthook = safe_audio_teardown


    tracks = {
        "oval": lambda: track_mod.oval(),
        "random": lambda: track_mod.random_circuit(seed=seed),
        "touge": lambda: track_mod.touge(seed=seed),
    }
    trk = track if track is not None else tracks.get(track_name, tracks["random"])()
    bg_layer = TrackEnvironmentBackground(trk, seed=seed or 7)
    sx, sy, syaw = trk.start_pose()

    def reset_grid():
        if racers:
            nx, ny = np.cos(syaw + np.pi/2), np.sin(syaw + np.pi/2)
            for i, opp in enumerate(racers):
                offset = i * 6.0
                ox = sx - np.cos(syaw) * offset
                oy = sy - np.sin(syaw) * offset
                lat = ((i % 2) * 2 - 1) * 0.8 if i > 0 else 0
                ox += nx * lat
                oy += ny * lat
                opp.reset(ox, oy, syaw, speed=start_speed)
                opp.crashed = False
                opp.laps = 0
                opp.prev_frac = 0.0
                opp._last_act = (0.0, 0.0, 0.0, 0.0)
        else:
            veh.reset(sx, sy, syaw, speed=start_speed)
            if opponents:
                nx, ny = np.cos(syaw + np.pi/2), np.sin(syaw + np.pi/2)
                for i, opp in enumerate(opponents):
                    offset = (i + 1) * 6.0
                    ox = sx - np.cos(syaw) * offset
                    oy = sy - np.sin(syaw) * offset
                    lat = ((i % 2) * 2 - 1) * 0.8
                    ox += nx * lat
                    oy += ny * lat
                    opp.reset(ox, oy, syaw, speed=start_speed)
                    opp.crashed = False
                    opp.laps = 0
                    opp.prev_frac = 0.0
                    opp._last_act = (0.0, 0.0, 0.0, 0.0)

    reset_grid()

    cam = Camera(W, H, scale=13.0, center=((W - DASH_W) / 2, H / 2))
    autobox = AutoBox(spec)
    auto = True
    show_dash = True
    fx = FX()          # smoke / dust / sparks / backfire particles
    skids = SkidTrail()
    backfire_cd = 0.0
    pred_path = []
    has_ai = agent is not None
    popups = []        # floating drift-score text: [x, y, text, age, color]
    damage_popups = []  # floating damage text: [x, y, text, age]
    # live drift meter (viewer-side; mirrors the scorer's shape, just for show)
    d_active, d_t, d_score, d_combo, d_peak = False, 0.0, 0.0, 1, 0.0
    # the visual extras are toggleable (number keys; F shows the menu).
    fxon = {"flames": True, "beams": False, "path": False, "smoke": False,
            "skids": True, "dust": False, "sparks": False, "drift": False,
            "cam": True, "streaks": False, "graphics_v2": True}
    FX_MENU = [("1", "flames", "Backfire flames"), ("2", "beams", "Vision beams"),
               ("3", "path", "Predicted path"), ("4", "smoke", "Tyre smoke"),
               ("5", "skids", "Skid marks"), ("6", "dust", "Off-track dust"),
               ("7", "sparks", "Curb sparks"), ("8", "drift", "Drift meter"),
               ("9", "cam", "Dynamic camera"), ("0", "streaks", "Speed streaks"),
               ("V", "graphics_v2", "Graphics V2 Preset")]
    FX_NUMKEY = {k: name for k, name, _ in FX_MENU}
    show_fxmenu = False
    # day/dusk/night colour grade (G cycles): (name, bg, tarmac, line, edge, overlay)
    GRADES = [
        ("day",   (24, 26, 30), (46, 48, 54), (200, 200, 80), (210, 70, 70), None),
        ("dusk",  (38, 28, 30), (54, 46, 46), (235, 180, 90), (220, 90, 70), (255, 140, 60, 26)),
        ("night", (12, 14, 22), (32, 34, 44), (120, 140, 200), (90, 110, 200), (30, 40, 90, 40)),
    ]
    grade = 0
    obs = sensors.observe(veh, trk)
    # analog input state (keyboard is binary; ramp it for a believable feel)
    t_in = b_in = s_in = 0.0
    handbrake = 0.0
    clutch = 1.0
    down = False
    # a trained policy decides at its training rate (30 Hz), not every frame —
    # running a drift policy at 60 Hz makes it over-correct and spin out.
    ctrl_period = max(1, round(1.0 / (30 * sim.dt)))
    ctrl_ctr = 0
    cur_act = (0.0, 0.0, 0.0, 0.0)

    # Open the audio stream only NOW — after the track is built and the first
    # observation is computed — so the heavy level-load work can't stall the
    # synthesis callback into an underrun crackle. It fades in over ~80 ms.
    if audio_on:
        all_vehs = racers if racers else [veh] + ([o.veh for o in opponents] if opponents else [])
        audio.start(all_vehs)

    endless_leader = None
    endless_time = 0.0
    endless_won = False
    endless_winner_name = ""
    drive_time = 0.0
    lap_prev_frac = trk.frame(veh.x, veh.y)["progress"] if not racers else 0.0
    lap_count = 0
    lap_start_time = 0.0
    lap_invalid = False
    best_clean_lap = None
    sector_defs = getattr(trk, "sectors", []) or [
        {"name": f"S{i+1}", "start_arc": trk.length * i / 4.0,
         "end_arc": trk.length * (i + 1) / 4.0}
        for i in range(4)
    ]
    sector_splits = [None] * len(sector_defs)
    sector_idx = 0

    running = True
    accum = 0.0
    frame_no = 0
    # Decouple physics from render: each frame, step the 120 Hz sim to consume the
    # ACTUAL wall-clock time elapsed — so a heavy/slow frame lowers the FPS but
    # never slows the car (no more slow-motion under render load). MAX_CATCHUP
    # bounds a single frame (alt-tab / breakpoint / spiral-of-death guard), so
    # real-time holds down to ~1/MAX_CATCHUP = 5 FPS before any slow-mo.
    MAX_CATCHUP = 0.20
    MAX_SUBSTEPS = int(MAX_CATCHUP / sim.dt) + 2
    while running:
        frame_dt = clock.tick(sim.fps) / 1000.0
        accum += min(frame_dt, MAX_CATCHUP)
        frame_no += 1
        if max_frames and frame_no > max_frames:
            running = False

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    veh.reset(sx, sy, syaw, speed=start_speed)
                    audio.reset()
                    fx.parts.clear()
                    drive_time = 0.0
                    lap_prev_frac = trk.frame(veh.x, veh.y)["progress"]
                    lap_count = 0
                    lap_start_time = 0.0
                    lap_invalid = False
                    best_clean_lap = None
                    sector_splits = [None] * len(sector_defs)
                    sector_idx = 0
                elif ev.key == pygame.K_t:
                    auto = not auto
                elif ev.key == pygame.K_TAB:
                    show_dash = not show_dash
                elif ev.key == pygame.K_b:
                    fxon["beams"] = not fxon["beams"]
                elif ev.key == pygame.K_f:
                    show_fxmenu = not show_fxmenu
                elif ev.key == pygame.K_BACKQUOTE:  # master: all effects on/off
                    anyon = any(fxon.values())
                    for k in fxon:
                        if k == "path" and not has_ai:
                            continue
                        fxon[k] = not anyon
                elif ev.key in (pygame.K_g, pygame.K_n):    # cycle day / dusk / night mode
                    grade = (grade + 1) % len(GRADES)
                    bg_layer.time_of_day = GRADES[grade][0]
                elif ev.key == pygame.K_v:                  # toggle graphics v2 preset
                    fxon["graphics_v2"] = not fxon["graphics_v2"]
                elif ev.unicode.upper() in FX_NUMKEY:
                    name = FX_NUMKEY[ev.unicode.upper()]
                    if name != "path" or has_ai:
                        fxon[name] = not fxon[name]
                elif ev.key == pygame.K_m:
                    audio.toggle_mute()
                elif ev.key == pygame.K_e and not auto:
                    veh.shift_up()
                elif ev.key == pygame.K_q and not auto:
                    veh.shift_down()

        if not racers:
            if controller is None:
                keys = pygame.key.get_pressed()
                t_tgt = 1.0 if (keys[pygame.K_UP] or keys[pygame.K_w]) else 0.0
                b_tgt = 1.0 if (keys[pygame.K_DOWN] or keys[pygame.K_s]) else 0.0
                s_tgt = (1.0 if (keys[pygame.K_LEFT] or keys[pygame.K_a]) else 0.0) \
                    - (1.0 if (keys[pygame.K_RIGHT] or keys[pygame.K_d]) else 0.0)
                handbrake = 1.0 if keys[pygame.K_SPACE] else 0.0
                manual_clutch = 0.0 if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else 1.0

        # fixed-timestep physics
        steps = 0
        while accum >= sim.dt and steps < MAX_SUBSTEPS:
            if not racers:
                if controller is not None:
                    # query the policy at its training rate, hold the action between
                    if ctrl_ctr % ctrl_period == 0:
                        cobs = sensors.observe(veh, trk, opponents=[o.veh for o in (opponents or [])])
                        out = controller(veh, cobs)
                        cur_act = (float(out[0]), float(out[1]), float(out[2]),
                                   float(out[3]) if len(out) > 3 else 0.0)
                    ctrl_ctr += 1
                    s_in, t_in, b_in, handbrake = cur_act
                    clutch, up, down = autobox.update(veh, t_in, sim.dt)
                else:
                    # ramp pedals into analog positions (no digital on/off harshness)
                    t_in += np.clip(t_tgt - t_in, -1, 1) * min(1.0, sim.dt * 5.0)
                    b_in += np.clip(b_tgt - b_in, -1, 1) * min(1.0, sim.dt * 8.0)
                    s_in = float(s_tgt)   # steering passes straight through
                    if auto:
                        clutch, up, down = autobox.update(veh, t_in, sim.dt)
                    else:
                        clutch, up, down = manual_clutch, False, False

                # feed surface grip + road plane from the track (off-track =
                # slippery; grade/bank/vcurv = hills are real forces)
                fr = trk.frame(veh.x, veh.y)
                lap_invalid = lap_invalid or fr["off_track"]
                frac = fr["progress"]
                dlap = frac - lap_prev_frac
                if dlap < -0.5:
                    elapsed = drive_time - lap_start_time
                    if not lap_invalid and elapsed > 1.0:
                        best_clean_lap = elapsed if best_clean_lap is None else min(best_clean_lap, elapsed)
                    lap_count += 1
                    lap_start_time = drive_time
                    lap_invalid = False
                    sector_splits = [None] * len(sector_defs)
                    sector_idx = 0
                elif dlap > 0.5:
                    lap_invalid = True
                lap_prev_frac = frac
                while (sector_idx + 1 < len(sector_defs)
                       and fr["arc"] >= float(sector_defs[sector_idx + 1]["start_arc"])):
                    sector_splits[sector_idx] = drive_time - lap_start_time
                    sector_idx += 1
                veh.surface_grip = spec.offtrack_grip if fr["off_track"] else 1.0
                veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"],
                             fr["vcurv"])

                veh.step(Controls(steer=s_in, throttle=t_in, brake=b_in,
                                  clutch=clutch, handbrake=handbrake,
                                  shift_up=up, shift_down=down))

            if opponents:
                from .race_env import resolve_collisions
                active_opps = [o for o in opponents if not getattr(o, "crashed", False)]

                active_vehs = [o.veh for o in active_opps]
                if not racers:
                    active_vehs.insert(0, veh)

                for opp in active_opps:
                    ofr = trk.frame(opp.veh.x, opp.veh.y)
                    opp.veh.surface_grip = opp.spec.offtrack_grip if ofr["off_track"] else 1.0
                    opp.veh.set_road(ofr["grade"], ofr["bank"], ofr["heading"], ofr["z"], ofr["vcurv"])

                    if abs(ofr["lateral"]) > trk.half + 12.0:
                        opp.crashed = True
                        continue

                    if ctrl_ctr % ctrl_period == 0:
                        other_vehs = [v for v in active_vehs if v != opp.veh]
                        os, ot, ob, oh = opp.act(trk, opponents=other_vehs)
                        opp._last_act = (os, ot, ob, oh)
                    else:
                        os, ot, ob, oh = getattr(opp, "_last_act", (0.0, 0.0, 0.0, 0.0))
                    oc, ou, od = opp.box.update(opp.veh, ot, sim.dt)
                    opp.veh.step(Controls(steer=os, throttle=ot, brake=ob, clutch=oc,
                                          handbrake=oh, shift_up=ou, shift_down=od))

                    # Update progress for leader tracking
                    frac = ofr["progress"]
                    d = frac - opp.prev_frac
                    if d < -0.5: opp.laps += 1
                    elif d > 0.5: opp.laps -= 1
                    opp.prev_frac = frac
                    opp.progress = opp.laps + frac

                resolve_collisions(active_vehs, sim.dt)

                if racers and track_name == "endless" and not endless_won:
                    leader = max(racers, key=lambda o: getattr(o, "progress", 0.0))
                    if leader is endless_leader:
                        endless_time += sim.dt
                        if endless_time > 90.0:
                            endless_won = True
                            endless_winner_name = leader.name
                            endless_time = 0.0
                    else:
                        endless_leader = leader
                        endless_time = 0.0

            if endless_won:
                accum = 0.0
                break

            drive_time += sim.dt
            accum -= sim.dt
            import time; time.sleep(0)
            steps += 1

            # ---- FX: backfire flame on downshift or overrun (matches BOV audio)
            backfire_cd = max(0.0, backfire_cd - sim.dt)

            f_veh = racers[0].veh if racers else veh
            f_down = False if racers else down
            f_t_in = 0.0 if racers else t_in

            if (fxon["flames"] and backfire_cd == 0.0 and ((f_down and f_veh.speed > 5)
                    or (f_t_in < 0.1 and f_veh.rpm > 3800 and f_veh.speed > 8
                        and np.random.rand() < (0.12 if car == "mazda787b" else 0.04)))):
                hx, hy = np.cos(f_veh.yaw), np.sin(f_veh.yaw)
                exx, exy = f_veh.x - hx * (spec.b + 0.5), f_veh.y - hy * (spec.b + 0.5)
                fx.emit_flame(exx, exy, -hx * 4.0, -hy * 4.0,
                              intensity=1.0 if f_down else 0.7)
                backfire_cd = 0.18
        if steps >= MAX_SUBSTEPS:
            accum = 0.0          # CPU can't keep up: drop the backlog, don't spiral

        # auto-reset when a watched policy crashes off-track (else it drives off
        # to nowhere with nothing to stop it)
        if controller is not None:
            frc = trk.frame(veh.x, veh.y)
            if abs(frc["lateral"]) > trk.half + 12.0:
                veh.reset(sx, sy, syaw, speed=start_speed)
                audio.reset()
                autobox.__init__(spec)
                fx.parts.clear()
                ctrl_ctr = 0
                cur_act = (0.0, 0.0, 0.0, 0.0)

        # what the agent perceives this frame (vision + proprioception + preview)
        # what the agent perceives this frame (vision + proprioception + preview)
        if not racers:
            obs = sensors.observe(veh, trk)
        else:
            leader = max(racers, key=lambda o: getattr(o, "progress", 0.0))
            obs = sensors.observe(leader.veh, trk)
            dash_veh = leader.veh
            dash_spec = leader.spec

        # Follow race leader if in symmetric mode
        if racers:
            leader = max(racers, key=lambda o: getattr(o, "progress", 0.0))
            cam.follow(leader.veh.x, leader.veh.y)
        else:
            cam.follow(veh.x, veh.y)

        # Spatial audio update for all vehicles
        tracked_veh = leader.veh if racers else veh
        lvx = tracked_veh.speed * np.cos(tracked_veh.yaw)
        lvy = tracked_veh.speed * np.sin(tracked_veh.yaw)

        all_vehs = [o.veh for o in racers] if racers else [veh]
        if opponents and not racers:
            all_vehs.extend([o.veh for o in opponents])

        all_throttles = []
        all_brakes = []
        for v in all_vehs:
            if v is veh:
                all_throttles.append(t_in)
                all_brakes.append(b_in)
            else:
                o = next((opp for opp in (racers or opponents or []) if opp.veh is v), None)
                if o and hasattr(o, "_last_act"):
                    all_throttles.append(max(0.0, float(o._last_act[1])))
                    all_brakes.append(max(0.0, -float(o._last_act[1])))
                else:
                    all_throttles.append(0.0)
                    all_brakes.append(0.0)

        audio.update((cam.cx, cam.cy), (lvx, lvy), tracked_veh.yaw,
                     all_vehs, all_throttles, brakes=all_brakes,
                     perspective=1)

        fx.update(frame_dt)

        # ---------- drift visuals (smoke / skids / dust / sparks / meter) -------
        fr_now = trk.frame(veh.x, veh.y)
        off = fr_now["off_track"]
        rear_sr = max(abs(veh.wheel_sr[2]), abs(veh.wheel_sr[3]))
        slip_deg = abs(np.degrees(veh.slip_angle))
        slide = float(np.clip(max(slip_deg / 42.0, rear_sr / 0.8), 0.0, 1.0))
        sliding = veh.speed > 6.0 and (slip_deg > 9.0 or rear_sr > 0.30)
        cyaw, syw = np.cos(veh.yaw), np.sin(veh.yaw)
        rear = [(veh.x - spec.b * cyaw - ly * syw, veh.y - spec.b * syw + ly * cyaw)
                for ly in (spec.half_track, -spec.half_track)]
        if fxon["smoke"] and sliding and not off:
            for wx, wy in rear:
                fx.emit_smoke(wx, wy, 0.0, 0.0, intensity=slide)
        if fxon["skids"] and sliding and not off:
            skids.add(2, rear[0][0], rear[0][1], slide)
            skids.add(3, rear[1][0], rear[1][1], slide)
        else:
            skids.lift()
        if fxon["dust"] and off and veh.speed > 3.0:
            for wx, wy in rear:
                fx.emit_dust(wx, wy, 0.0, 0.0, intensity=min(1.0, veh.speed / 25.0))
        if (fxon["sparks"] and veh.speed > 8.0
                and trk.half - 0.6 < abs(fr_now["lateral"]) < trk.half + 1.0 and np.random.rand() < 0.5):
            ly = spec.half_track if fr_now["lateral"] > 0 else -spec.half_track
            fx.emit_sparks(veh.x - spec.b * cyaw - ly * syw,
                           veh.y - spec.b * syw + ly * cyaw, 0.0, 0.0, n=4)
        skids.update(frame_dt)
        # live drift meter -> floating score popup when a slide ends
        if sliding and not off:
            if not d_active:
                d_active, d_t, d_score, d_combo, d_peak = True, 0.0, 0.0, 1, slip_deg
            d_t += frame_dt
            d_combo = min(5, 1 + int(d_t / 1.1))
            d_score += slip_deg * veh.speed * 0.04 * d_combo * frame_dt
            d_peak = max(d_peak, slip_deg)
        else:
            if d_active and d_score > 60:
                popups.append([veh.x, veh.y, f"+{int(d_score)}  {int(d_peak)}°  x{d_combo}",
                               0.0, (255, 210, 90)])
            d_active, d_t, d_score, d_combo, d_peak = False, 0.0, 0.0, 1, 0.0
        for p in popups:
            p[3] += frame_dt
        popups[:] = [p for p in popups if p[3] < 1.6]
        cam.dynamics(veh.speed, slide, frame_dt, enabled=fxon["cam"])
        cam.center_y = H / 2

        # drain damage events from the vehicle into popups
        for msg in veh.damage_events:
            damage_popups.append([veh.x, veh.y, msg, 0.0])
        veh.damage_events.clear()
        for p in damage_popups:
            p[3] += frame_dt
        damage_popups[:] = [p for p in damage_popups if p[3] < 2.5]

        # predicted path: amortised across frames (a small step budget per frame)
        # so it never spikes a single frame -> no stutter. Draws the last
        # completed path, refreshed ~8x/sec.
        if fxon["path"] and agent is not None:
            pred_path = agent.step_prediction(veh, trk)

        # ---------------- draw ----------------
        _, c_bg, c_tar, c_line, c_edge, c_ovl = GRADES[grade]
        # 1. Background terrain (V2 only, otherwise default solid fill)
        if fxon["graphics_v2"]:
            bg_layer.draw_terrain(screen, cam)
        else:
            screen.fill(c_bg)

        # 2. Road geometry (flat tarmac with borders and dashed centerline),
        #    drawn per-segment and viewport-culled (see draw_road).
        draw_road(pygame, screen, cam, trk, c_tar, c_edge, c_line)

        # skid marks sit on the tarmac, UNDER the car
        if fxon["skids"]:
            skids.draw(pygame, screen, cam.to_screen, cam.scale, bg=c_tar)

        target_veh = dash_veh if racers else veh
        target_spec = dash_spec if racers else spec

        if fxon["beams"]:
            draw_beams(screen, cam, target_veh, obs)

        # predicted path: dotted ghost-line of where the policy thinks it's going
        if fxon["path"] and agent is not None and len(pred_path) > 1:
            aiviz.draw_path(pygame, screen, cam.to_screen, pred_path, agent.confidence())

        # 3. Background objects (V2 only: Pine trees & Streetlights)
        if fxon["graphics_v2"]:
            bg_layer.draw_objects(screen, cam, veh)

        if opponents:
            for opp in opponents:
                if opp is not veh: # Don't double draw the dummy veh
                    draw_car_art(screen, cam.to_screen, cam.scale, opp.veh, opp.spec, brake=getattr(opp, "_last_act", (0,0,0,0))[2], headlights=(bg_layer.time_of_day in ("dusk", "night")))

        if not racers:
            draw_car_art(screen, cam.to_screen, cam.scale, veh, spec, brake=b_in, headlights=(bg_layer.time_of_day in ("dusk", "night")))
        if fxon["flames"] or fxon["smoke"] or fxon["dust"] or fxon["sparks"]:
            fx.draw(pygame, screen, cam.to_screen, cam.scale)   # smoke/dust/sparks/flames

        view_w = W - (DASH_W if show_dash else 0)
        if fxon["streaks"]:
            pass # _draw_streaks(pygame, screen, veh, view_w, H)  # disabled by user request
        if fxon["drift"]:
            _draw_popups(pygame, screen, font, cam.to_screen, popups)
            if d_active and d_score > 30:
                _draw_drift_hud(pygame, screen, font, d_score, d_combo, d_peak, view_w)
        # draw damage popups (always visible if there are any)
        _draw_damage_popups(pygame, screen, font, cam.to_screen, damage_popups)
        if c_ovl:           # colour-grade tint over the world (HUD/dash draw on top)
            ov = pygame.Surface((view_w, H), pygame.SRCALPHA)
            ov.fill(c_ovl)
            screen.blit(ov, (0, 0))
        _draw_hud(pygame, screen, font, veh, obs, auto, show_dash, view_w, H,
                  lap_count=lap_count, lap_time=drive_time - lap_start_time,
                  best_clean_lap=best_clean_lap, lap_invalid=lap_invalid,
                  sector_splits=sector_splits, sector_defs=sector_defs)
        _draw_minimap(pygame, screen, trk, veh, H)
        if show_fxmenu:
            _draw_fxmenu(pygame, screen, small, FX_MENU, fxon, has_ai, H)
        if show_dash:
            dash.draw(screen, veh, obs,
                      {"throttle": t_in, "brake": b_in, "steer": s_in,
                       "clutch": clutch, "handbrake": handbrake})

        pygame.display.flip()

    audio.stop()
    pygame.quit()


def _draw_streaks(pygame, screen, veh, view_w, H):
    """Subtle speed streaks trailing the travel direction near the screen edges,
    fading in only at high speed. Kept sparse + faint so it never reads as busy."""
    spd = veh.speed
    if spd < 22.0:
        return
    f = min(1.0, (spd - 22.0) / 45.0)                  # 0 ~80 km/h -> 1 ~240 km/h
    wvx = veh.vx * np.cos(veh.yaw) - veh.vy * np.sin(veh.yaw)
    wvy = veh.vx * np.sin(veh.yaw) + veh.vy * np.cos(veh.yaw)
    n = np.hypot(wvx, wvy) or 1.0
    dx, dy = wvx / n, wvy / n
    surf = pygame.Surface((view_w, H), pygame.SRCALPHA)
    cx, cy = view_w / 2, H / 2
    L = 60 + 130 * f
    # stable-ish per-spot seed; mask to a non-negative 31-bit int (tracks with
    # negative world coords made this negative -> default_rng ValueError -> crash).
    rng = np.random.default_rng((int(veh.x * 7) ^ int(veh.y * 13)) & 0x7FFFFFFF)
    for _ in range(8):
        px = cx + rng.uniform(-1, 1) * view_w * 0.5
        py = cy + rng.uniform(-1, 1) * H * 0.5
        if abs(px - cx) < view_w * 0.18 and abs(py - cy) < H * 0.18:
            continue                                    # keep the centre clean
        a = int(46 * f)
        pygame.draw.line(surf, (210, 220, 235, a), (px, py),
                         (px - dx * L, py - dy * L), 2)
    screen.blit(surf, (0, 0))


def _draw_popups(pygame, screen, font, to_screen, popups):
    """Floating drift-score text that rises and fades where a slide finished."""
    for x, y, text, age, col in popups:
        a = max(0, int(255 * (1.0 - age / 1.6)))
        surf = font.render(text, True, col)
        surf.set_alpha(a)
        sx, sy = to_screen(x, y)
        screen.blit(surf, (sx - surf.get_width() // 2, sy - 24 - int(age * 46)))


def _draw_drift_hud(pygame, screen, font, score, combo, peak, view_w):
    """Live drift readout at the top while a slide is in progress."""
    txt = f"DRIFT  {int(score)}   {int(peak)}°   x{combo}"
    surf = font.render(txt, True, (255, 215, 120))
    w = surf.get_width()
    x = view_w // 2 - w // 2
    panel = pygame.Surface((w + 24, 30), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 120))
    screen.blit(panel, (x - 12, 14))
    screen.blit(surf, (x, 20))


def _fmt_lap_time(t):
    if t is None:
        return "--:--.---"
    m = int(t // 60)
    s = t - 60 * m
    return f"{m}:{s:06.3f}"


def _draw_hud(pygame, screen, font, veh, obs, auto, show_dash, view_w, H,
              lap_count=0, lap_time=0.0, best_clean_lap=None, lap_invalid=False,
              sector_splits=None, sector_defs=None):
    """Slim glanceable overlay on the track view (the detail lives in the dash)."""
    fr = obs.frame
    panel = pygame.Surface((230, 96), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 110))
    screen.blit(panel, (12, 12))
    # lap + position
    offcol = (235, 100, 90) if fr["off_track"] else (200, 205, 212)
    screen.blit(font.render(f"LAP {fr['progress']*100:5.1f}%", True, (200, 205, 212)), (22, 20))
    screen.blit(font.render(f"off-line {fr['lateral']:+5.1f} m", True, offcol), (22, 42))
    mode = "AUTO" if auto else "MANUAL"
    screen.blit(font.render(f"{mode}", True, (150, 170, 240)), (22, 64))
    inv = " INVALID" if lap_invalid else ""
    screen.blit(font.render(
        f"lap {lap_count + 1}  {_fmt_lap_time(lap_time)}{inv}",
        True, (255, 210, 105) if lap_invalid else (210, 230, 255)), (22, 86))
    screen.blit(font.render(
        f"best clean {_fmt_lap_time(best_clean_lap)}",
        True, (150, 220, 160)), (22, 108))
    if sector_splits:
        sx = 22
        sy = 130
        for i, val in enumerate(sector_splits[:4]):
            label = (sector_defs[i].get("name", f"S{i+1}") if sector_defs and i < len(sector_defs)
                     else f"S{i+1}")
            screen.blit(font.render(f"{label[-1]} {_fmt_lap_time(val)}", True, (165, 170, 185)),
                        (sx + i * 104, sy))
    if fr["off_track"]:
        screen.blit(font.render("OFF TRACK", True, (235, 100, 90)), (110, 64))
    # if the dashboard is hidden, surface the essentials here
    if not show_dash:
        t = veh.telemetry()
        screen.blit(font.render(
            f"{t['speed_kmh']:.0f} km/h   {t['rpm']:.0f} rpm   gear {t['gear']}"
            f"   slip {t['slip_angle_deg']:+.0f}d", True, (220, 224, 230)), (22, 154))
    screen.blit(font.render(
        "arrows=drive  space=handbrake  t=auto  tab=dash  f=fx  g=grade  m=mute  r=reset  esc=quit",
        True, (135, 140, 150)), (22, H - 26))


def _draw_minimap(pygame, screen, trk, veh, H, racers=None):
    mw, mh = 190, 140
    ox, oy = 16, H - mh - 16
    c = trk.center
    mins = c.min(axis=0)
    span = (c.max(axis=0) - mins)
    span[span < 1e-6] = 1.0
    sc = min(mw / span[0], mh / span[1]) * 0.9

    def m(p):
        q = (np.array(p) - mins) * sc
        return int(ox + 10 + q[0]), int(oy + 10 + (mh - 20 - q[1]))

    panel = pygame.Surface((mw, mh), pygame.SRCALPHA)
    panel.fill((0, 0, 0, 110))
    screen.blit(panel, (ox, oy))
    pts = [m(p) for p in c]
    pygame.draw.lines(screen, (150, 150, 160), True, pts, 1)

    if racers:
        leader = max(racers, key=lambda o: getattr(o, "progress", 0.0))
        for o in racers:
            if not getattr(o, "crashed", False):
                pygame.draw.circle(screen, (80, 240, 80) if o == leader else (240, 80, 80), m((o.veh.x, o.veh.y)), 4 if o == leader else 3)
    else:
        pygame.draw.circle(screen, (240, 80, 80), m((veh.x, veh.y)), 4)


def _draw_fxmenu(pygame, screen, small, menu, fxon, has_ai, H):
    """Toggle menu for the remaining visual extras (number keys flip them)."""
    PW = 212
    PH = len(menu) * 20 + 50
    x, y = 12, 118
    panel = pygame.Surface((PW, PH), pygame.SRCALPHA)
    panel.fill((10, 12, 16, 195))
    screen.blit(panel, (x, y))
    screen.blit(small.render("EFFECTS  —  F to hide", True, (150, 160, 175)),
                (x + 12, y + 8))
    for i, (k, name, label) in enumerate(menu):
        ry = y + 28 + i * 20
        disabled = name == "path" and not has_ai
        on = fxon[name]
        if disabled:
            col, state = (70, 74, 82), "n/a"
        elif on:
            col, state = (120, 210, 150), "ON"
        else:
            col, state = (120, 126, 138), "off"
        screen.blit(small.render(f"{k}  {label}", True, col), (x + 12, ry))
        st = small.render(state, True, col)
        screen.blit(st, (x + PW - 12 - st.get_width(), ry))
    screen.blit(small.render("V = all on/off   ·   G = day/dusk/night", True, (120, 126, 138)),
                (x + 12, y + 28 + len(menu) * 20))


def _draw_damage_popups(pygame, screen, font, to_screen, damage_popups):
    """Floating damage text bubbles that rise and fade at the crash location.
    Each entry: [x, y, text, age]."""
    for p in damage_popups:
        x, y, text, age = p[0], p[1], p[2], p[3]
        a = max(0, int(255 * (1.0 - age / 2.5)))
        # red-orange background pill
        surf = font.render(text, True, (255, 80, 60))
        surf.set_alpha(a)
        sx, sy = to_screen(x, y)
        # draw dark backing
        bw, bh = surf.get_width() + 16, surf.get_height() + 8
        backing = pygame.Surface((bw, bh), pygame.SRCALPHA)
        backing.fill((0, 0, 0, min(180, int(200 * (1.0 - age / 2.5)))))
        bx = sx - bw // 2
        by = sy - 40 - int(age * 50)
        screen.blit(backing, (bx, by))
        screen.blit(surf, (bx + 8, by + 4))
