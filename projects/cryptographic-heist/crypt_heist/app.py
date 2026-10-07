"""Pygame spectator UI for the heist prototype."""

from __future__ import annotations

import math

import numpy as np

from .sim import HeistSim


class Camera:
    def __init__(self, w: int, h: int):
        self.w = w
        self.h = h
        self.x = 0.0
        self.y = 0.0
        self.scale = 2.2

    def follow(self, points: list[tuple[float, float]], impact: float, dt: float):
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 120.0)
        target_scale = float(np.clip(min(self.w, self.h) / (span * 1.55), 1.15, 4.3))
        self.x += (cx - self.x) * min(1.0, dt * 4.0)
        self.y += (cy - self.y) * min(1.0, dt * 4.0)
        self.scale += (target_scale - self.scale) * min(1.0, dt * 3.0)
        shake = min(8.0, impact * 0.32)
        if shake > 0.1:
            self.x += float(np.random.uniform(-shake, shake)) / self.scale
            self.y += float(np.random.uniform(-shake, shake)) / self.scale

    def to_screen(self, x: float, y: float):
        return int((x - self.x) * self.scale + self.w / 2), int((y - self.y) * self.scale + self.h / 2)


class Smoke:
    def __init__(self):
        self.parts = []

    def emit(self, veh, color):
        slip = abs(float(veh.slip_angle))
        grip = float(np.max(veh.wheel_grip)) if len(veh.wheel_grip) else 0.0
        if slip < 0.10 and grip < 0.96:
            return
        cy, sy = math.cos(veh.yaw), math.sin(veh.yaw)
        back = np.array([veh.x - cy * 2.2, veh.y - sy * 2.2])
        for _ in range(2 if slip > 0.18 else 1):
            jitter = np.random.normal(0.0, 1.1, size=2)
            self.parts.append([back[0] + jitter[0], back[1] + jitter[1], 0.0, 0.0, 0.0, color])

    def update(self, dt):
        live = []
        for p in self.parts:
            p[4] += dt
            if p[4] < 1.0:
                live.append(p)
        self.parts = live[-420:]

    def draw(self, pygame, screen, cam):
        surf = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
        for x, y, _, _, age, color in self.parts:
            sx, sy = cam.to_screen(x, y)
            alpha = int(max(0, 70 * (1.0 - age)))
            radius = int((4.0 + 15.0 * age) * cam.scale / 2.4)
            pygame.draw.circle(surf, (*color, alpha), (sx, sy), max(2, radius))
        screen.blit(surf, (0, 0))


def draw_car(pygame, screen, cam: Camera, veh, color, label: str, font):
    length = max(4.3, veh.spec.wheelbase + 1.4)
    width = max(1.9, veh.spec.track_width + 0.35)
    pts = np.array([
        [length * 0.55, 0.0],
        [length * 0.30, width * 0.50],
        [-length * 0.50, width * 0.48],
        [-length * 0.58, 0.0],
        [-length * 0.50, -width * 0.48],
        [length * 0.30, -width * 0.50],
    ])
    c, s = math.cos(veh.yaw), math.sin(veh.yaw)
    rot = np.array([[c, -s], [s, c]])
    world = pts @ rot.T + np.array([veh.x, veh.y])
    screen_pts = [cam.to_screen(float(x), float(y)) for x, y in world]
    pygame.draw.polygon(screen, color, screen_pts)
    pygame.draw.lines(screen, (245, 245, 245), True, screen_pts, 1)
    nose = cam.to_screen(float(world[0, 0]), float(world[0, 1]))
    center = cam.to_screen(veh.x, veh.y)
    pygame.draw.line(screen, (255, 255, 255), center, nose, 2)
    txt = font.render(label, True, (235, 238, 245))
    screen.blit(txt, (center[0] + 7, center[1] - 9))


def draw_city(pygame, screen, cam: Camera, city):
    screen.fill((16, 18, 22))
    for b in city.buildings:
        p0 = cam.to_screen(b.x0, b.y0)
        p1 = cam.to_screen(b.x1, b.y1)
        rect = pygame.Rect(min(p0[0], p1[0]), min(p0[1], p1[1]), abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))
        pygame.draw.rect(screen, (42, 46, 52), rect)
        pygame.draw.rect(screen, (65, 72, 82), rect, 1)
    for x in city.vertical_roads:
        a = cam.to_screen(x, city.top)
        b = cam.to_screen(x, city.bottom)
        pygame.draw.line(screen, (80, 82, 86), a, b, max(1, int(city.road * cam.scale * 0.04)))
    for y in city.horizontal_roads:
        a = cam.to_screen(city.left, y)
        b = cam.to_screen(city.right, y)
        pygame.draw.line(screen, (80, 82, 86), a, b, max(1, int(city.road * cam.scale * 0.04)))


def draw_dashboard(pygame, screen, rect, snap, fonts):
    font, small = fonts
    x, y, w, h = rect
    pygame.draw.rect(screen, (9, 10, 13), rect)
    pygame.draw.line(screen, (70, 78, 90), (x, y), (x + w, y), 2)

    left_w = int(w * 0.47)
    mid_w = int(w * 0.28)
    pygame.draw.line(screen, (44, 51, 60), (x + left_w, y + 12), (x + left_w, y + h - 12), 1)
    pygame.draw.line(screen, (44, 51, 60), (x + left_w + mid_w, y + 12), (x + left_w + mid_w, y + h - 12), 1)

    screen.blit(font.render("RADIO TRANSCRIPT", True, (200, 210, 225)), (x + 18, y + 12))
    events = snap["events"][-36:]
    lines = []
    current = []
    last_speaker = None
    for ev in events:
        if ev.speaker != last_speaker or len(current) >= 6:
            if current:
                lines.append((last_speaker, current))
            current = []
            last_speaker = ev.speaker
        current.append(ev)
    if current:
        lines.append((last_speaker, current))
    draw_y = y + 38
    selected = lines[-8:]
    spoof_lines = [line for line in lines if any(ev.spoofed for ev in line[1])]
    if spoof_lines and not any(any(ev.spoofed for ev in line[1]) for line in selected):
        selected = [spoof_lines[-1]] + selected[:-1]
    for speaker, evs in selected:
        sx = x + 18
        label = small.render(f"{speaker:>7}:", True, (140, 154, 170))
        screen.blit(label, (sx, draw_y))
        sx += 78
        for ev in evs:
            color = (255, 116, 76) if ev.spoofed else (224, 230, 238)
            word = small.render(ev.word, True, color)
            screen.blit(word, (sx, draw_y))
            sx += word.get_width() + 8
        draw_y += 21

    mx = x + left_w + 18
    screen.blit(font.render("SCANNER DECRYPTOR", True, (200, 210, 225)), (mx, y + 12))
    py = y + 42
    for name, pred in list(snap["predictions"].items())[:5]:
        line = f"{name} future vector  x={pred[0]:6.1f}  y={pred[1]:6.1f}"
        screen.blit(small.render(line, True, (185, 220, 210)), (mx, py))
        py += 22

    rx = x + left_w + mid_w + 18
    screen.blit(font.render("CONFIDENCE", True, (200, 210, 225)), (rx, y + 12))
    conf = snap["confidence"]
    bw, bh = w - (rx - x) - 32, 24
    pygame.draw.rect(screen, (28, 32, 38), (rx, y + 48, bw, bh), border_radius=4)
    fill = int(bw * conf)
    col = (230, 80, 80) if conf < 0.3 else (224, 190, 78) if conf < 0.7 else (92, 208, 148)
    pygame.draw.rect(screen, col, (rx, y + 48, fill, bh), border_radius=4)
    screen.blit(font.render(f"{conf * 100:5.1f}%", True, (245, 248, 252)), (rx, y + 82))
    stats = [
        f"jamming tokens: {snap['jamming_budget']}",
        f"waypoints hit:  {snap['waypoints_hit']}",
        f"captures:       {snap['captures']}",
        f"deception:      {snap['deception_score']:.0f}",
    ]
    for i, line in enumerate(stats):
        screen.blit(small.render(line, True, (170, 182, 198)), (rx, y + 112 + i * 20))
    since_jam = snap["time"] - snap.get("last_jam_time", -999.0)
    if 0.0 <= since_jam < 4.5:
        words = " ".join(snap.get("last_spoof_words", [])[:3])
        alert = f"SPOOF {since_jam:3.1f}s  {words}"
        screen.blit(small.render(alert, True, (255, 116, 76)), (rx, y + h - 28))


def run(
    seed: int | None = 11,
    mute: bool = False,
    max_frames: int | None = None,
    controller=None,
    scanner=None,
    control_repeat: int = 4,
    title_suffix: str = "",
    configure_sim=None,
):
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required. Install with: pip3 install -r requirements.txt") from exc

    pygame.mixer.pre_init(44100, -16, 1, 512)
    pygame.init()
    from .sound import HarmonicTensionSoundtrack

    width, height = 1480, 900
    dash_h = int(height * 0.25)
    chase_h = height - dash_h
    screen = pygame.display.set_mode((width, height))
    caption = "The Cryptographic Heist Engine"
    if title_suffix:
        caption = f"{caption} - {title_suffix}"
    pygame.display.set_caption(caption)
    chase = pygame.Surface((width, chase_h))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 16)
    small = pygame.font.SysFont("menlo,consolas,monospace", 13)
    cam = Camera(width, chase_h)
    sim = HeistSim(seed=seed)
    if configure_sim is not None:
        configure_sim(sim)
    smoke = Smoke()
    audio = HarmonicTensionSoundtrack(pygame, enabled=not mute)
    paused = False
    frames = 0
    held_actions = None
    held_ticks = 0

    running = True
    while running:
        frames += 1
        dt_real = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    sim.reset()
                    if configure_sim is not None:
                        configure_sim(sim)
                    held_actions = None
                    held_ticks = 0
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_m:
                    audio.set_enabled(not audio.enabled)

        if not paused:
            for _ in range(2):
                if controller is None:
                    sim.step()
                else:
                    if held_ticks <= 0:
                        held_actions = controller.action(sim)
                        held_ticks = max(1, int(control_repeat))
                    sim.step(actions=held_actions)
                    held_ticks -= 1
                if scanner is not None:
                    scanner.apply(sim)
            smoke.update(dt_real)

        snap = sim.snapshot()
        points = [(a.vehicle.x, a.vehicle.y) for a in snap["agents"]]
        cam.follow(points, snap["impact"], dt_real)
        draw_city(pygame, chase, cam, sim.city)
        wx, wy = snap["waypoint"]
        pygame.draw.circle(chase, (245, 205, 80), cam.to_screen(wx, wy), 9, 2)
        for agent in snap["agents"]:
            color = (230, 62, 70) if agent.faction == "evader" else (74, 136, 245)
            smoke.emit(agent.vehicle, (180, 186, 192) if agent.faction == "evader" else (120, 145, 190))
            draw_car(pygame, chase, cam, agent.vehicle, color, agent.name, small)
            if agent.faction == "pursuer":
                pygame.draw.line(chase, (55, 95, 160), cam.to_screen(agent.vehicle.x, agent.vehicle.y), cam.to_screen(*agent.target), 1)
        smoke.draw(pygame, chase, cam)

        screen.blit(chase, (0, 0))
        draw_dashboard(pygame, screen, (0, chase_h, width, dash_h), snap, (font, small))
        if paused:
            label = font.render("PAUSED", True, (255, 240, 180))
            screen.blit(label, (18, 18))
        pygame.display.flip()
        audio.update(snap["confidence"], dt_real)
        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()
