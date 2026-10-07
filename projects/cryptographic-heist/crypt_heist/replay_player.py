"""Pygame replay playback from JSONL logs.

This renderer intentionally consumes replay frames only. It does not import or
step ``HeistSim``, which keeps the replay boundary honest for future cinematic
renderers.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .replay import load_replay, validate_replay


class ReplayCamera:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.x = 0.0
        self.y = 0.0
        self.scale = 2.0

    def follow(self, agents: list[dict], dt: float):
        xs = [a["x"] for a in agents]
        ys = [a["y"] for a in agents]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        span = max(max(xs) - min(xs), max(ys) - min(ys), 150.0)
        target_scale = float(np.clip(min(self.width, self.height) / (span * 1.6), 0.95, 4.0))
        self.x += (cx - self.x) * min(1.0, dt * 5.0)
        self.y += (cy - self.y) * min(1.0, dt * 5.0)
        self.scale += (target_scale - self.scale) * min(1.0, dt * 4.0)

    def to_screen(self, x: float, y: float):
        return int((x - self.x) * self.scale + self.width / 2), int((y - self.y) * self.scale + self.height / 2)


def run_replay(path: str | Path, speed: float = 1.0, max_frames: int | None = None):
    try:
        import pygame
    except ImportError as exc:
        raise SystemExit("pygame is required. Install with: pip3 install -r requirements.txt") from exc

    lines = load_replay(path)
    validate_replay(lines)
    meta = lines[0]
    frames = [line for line in lines[1:] if line.get("type") == "frame"]
    if not frames:
        raise SystemExit("replay has no frames")

    pygame.init()
    width, height = 1480, 900
    dash_h = int(height * 0.25)
    chase_h = height - dash_h
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Replay - {Path(path).name}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("menlo,consolas,monospace", 16)
    small = pygame.font.SysFont("menlo,consolas,monospace", 13)
    cam = ReplayCamera(width, chase_h)
    paused = False
    idx = 0
    dt = float(meta.get("dt", 1.0 / 120.0))

    running = True
    while running:
        elapsed = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_LEFT:
                    idx = max(0, idx - 120)
                elif event.key == pygame.K_RIGHT:
                    idx = min(len(frames) - 1, idx + 120)
                elif event.key == pygame.K_r:
                    idx = 0

        frame = frames[idx]
        cam.follow(frame["agents"], elapsed)
        _draw_chase(pygame, screen, cam, frame, chase_h, small)
        _draw_dashboard(pygame, screen, frame, font, small, chase_h, width, dash_h, idx, len(frames), speed)
        pygame.display.flip()

        if not paused:
            advance = max(1, int(round((elapsed / max(dt, 1e-6)) * speed)))
            idx = min(len(frames) - 1, idx + advance)
        if idx >= len(frames) - 1:
            idx = 0
        if max_frames is not None:
            max_frames -= 1
            if max_frames <= 0:
                running = False

    pygame.quit()


def _draw_chase(pygame, screen, cam: ReplayCamera, frame: dict, chase_h: int, font):
    chase = pygame.Surface((cam.width, chase_h))
    chase.fill((15, 18, 23))
    _draw_grid(pygame, chase, cam)
    wx, wy = frame["waypoint"]
    pygame.draw.circle(chase, (245, 205, 80), cam.to_screen(wx, wy), 9, 2)

    for agent in frame["agents"]:
        if agent["faction"] == "pursuer":
            tx, ty = agent.get("target", [agent["x"], agent["y"]])
            pygame.draw.line(chase, (50, 92, 150), cam.to_screen(agent["x"], agent["y"]), cam.to_screen(tx, ty), 1)
    for agent in frame["agents"]:
        color = (230, 62, 70) if agent["faction"] == "evader" else (74, 136, 245)
        _draw_car(pygame, chase, cam, agent, color, font)

    screen.blit(chase, (0, 0))


def _draw_grid(pygame, surf, cam: ReplayCamera):
    for x in np.arange(-600.0, 700.0, 104.0):
        pygame.draw.line(surf, (47, 55, 64), cam.to_screen(float(x), -500.0), cam.to_screen(float(x), 500.0), 1)
    for y in np.arange(-500.0, 600.0, 104.0):
        pygame.draw.line(surf, (47, 55, 64), cam.to_screen(-700.0, float(y)), cam.to_screen(700.0, float(y)), 1)


def _draw_car(pygame, surf, cam: ReplayCamera, agent: dict, color, font):
    length = 5.0 if agent["faction"] == "evader" else 5.8
    width = 2.1 if agent["faction"] == "evader" else 2.4
    pts = np.array([
        [length * 0.55, 0.0],
        [length * 0.18, width * 0.50],
        [-length * 0.50, width * 0.45],
        [-length * 0.56, -width * 0.45],
        [length * 0.18, -width * 0.50],
    ])
    c, s = math.cos(agent["yaw"]), math.sin(agent["yaw"])
    rot = np.array([[c, -s], [s, c]])
    world = pts @ rot.T + np.array([agent["x"], agent["y"]])
    points = [cam.to_screen(float(x), float(y)) for x, y in world]
    pygame.draw.polygon(surf, color, points)
    pygame.draw.lines(surf, (245, 248, 252), True, points, 1)
    label = font.render(agent["name"], True, (235, 238, 245))
    cx, cy = cam.to_screen(agent["x"], agent["y"])
    surf.blit(label, (cx + 7, cy - 8))


def _draw_dashboard(pygame, screen, frame: dict, font, small, y: int, width: int, dash_h: int, idx: int, total: int, speed: float):
    rect = (0, y, width, dash_h)
    pygame.draw.rect(screen, (9, 10, 13), rect)
    pygame.draw.line(screen, (70, 78, 90), (0, y), (width, y), 2)
    left_w = int(width * 0.47)
    mid_w = int(width * 0.28)
    pygame.draw.line(screen, (44, 51, 60), (left_w, y + 12), (left_w, y + dash_h - 12), 1)
    pygame.draw.line(screen, (44, 51, 60), (left_w + mid_w, y + 12), (left_w + mid_w, y + dash_h - 12), 1)

    screen.blit(font.render("REPLAY RADIO", True, (200, 210, 225)), (18, y + 12))
    radio = frame.get("radio", [])[-48:]
    rows = _radio_rows(radio)[-8:]
    yy = y + 38
    for speaker, events in rows:
        sx = 18
        screen.blit(small.render(f"{speaker:>7}:", True, (140, 154, 170)), (sx, yy))
        sx += 78
        for event in events:
            color = (255, 116, 76) if event.get("spoofed") else (224, 230, 238)
            word = small.render(event.get("word", ""), True, color)
            screen.blit(word, (sx, yy))
            sx += word.get_width() + 8
        yy += 21

    mx = left_w + 18
    screen.blit(font.render("SCANNER DECRYPTOR", True, (200, 210, 225)), (mx, y + 12))
    yy = y + 42
    for name, pred in list(frame.get("predictions", {}).items())[:5]:
        screen.blit(small.render(f"{name} future vector  x={pred[0]:6.1f}  y={pred[1]:6.1f}", True, (185, 220, 210)), (mx, yy))
        yy += 22

    rx = left_w + mid_w + 18
    screen.blit(font.render("REPLAY STATUS", True, (200, 210, 225)), (rx, y + 12))
    conf = float(frame.get("confidence", 0.0))
    bw, bh = width - rx - 32, 24
    pygame.draw.rect(screen, (28, 32, 38), (rx, y + 48, bw, bh), border_radius=4)
    pygame.draw.rect(screen, _conf_color(conf), (rx, y + 48, int(bw * conf), bh), border_radius=4)
    metrics = frame.get("metrics", {})
    lines = [
        f"time:       {frame['time']:.2f}s",
        f"frame:      {idx + 1}/{total}",
        f"speed:      {speed:.2f}x",
        f"confidence: {conf * 100:.1f}%",
        f"captures:   {metrics.get('captures', 0)}",
        f"waypoints:  {metrics.get('waypoints_hit', 0)}",
        f"deception:  {metrics.get('deception_score', 0):.0f}",
    ]
    for i, line in enumerate(lines):
        screen.blit(small.render(line, True, (170, 182, 198)), (rx, y + 82 + i * 18))


def _radio_rows(events: list[dict]) -> list[tuple[str, list[dict]]]:
    rows = []
    speaker = None
    current = []
    for event in events:
        if event.get("speaker") != speaker or len(current) >= 6:
            if current:
                rows.append((speaker, current))
            speaker = event.get("speaker")
            current = []
        current.append(event)
    if current:
        rows.append((speaker, current))
    return rows


def _conf_color(conf: float):
    if conf < 0.3:
        return (230, 80, 80)
    if conf < 0.7:
        return (224, 190, 78)
    return (92, 208, 148)

