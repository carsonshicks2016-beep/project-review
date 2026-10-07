from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from biofactory.config import DEFAULT_CONFIG
from biofactory.rendering.camera import Camera
from biofactory.rendering.overlays import RENDER_QUALITIES, VISUAL_MODES
from biofactory.rendering.renderer import Renderer
from biofactory.rendering.ui import Selection
from biofactory.simulation.engine import SimulationEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark BioFactory rendering modes.")
    parser.add_argument("--ants", type=int, default=500)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--warmup-ticks", type=int, default=None)
    parser.add_argument("--zoom", type=float, default=1.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=820)
    parser.add_argument("--render-only", action="store_true", help="measure renderer without stepping the simulation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pygame.init()
    screen = pygame.display.set_mode((args.width, args.height))
    config = replace(DEFAULT_CONFIG, ants=replace(DEFAULT_CONFIG.ants, count=args.ants))
    engine = SimulationEngine(config)
    renderer = Renderer(engine)
    warmup_ticks = args.warmup_ticks
    if warmup_ticks is None:
        warmup_ticks = 0 if args.render_only else 900
    if warmup_ticks > 0:
        engine.step(warmup_ticks)
    camera = Camera(
        config.world.nest_x,
        config.world.nest_y,
        args.width - config.render.ui_width,
        args.height,
        config.render,
        zoom=args.zoom,
    )

    print(f"ants={args.ants} frames={args.frames} zoom={args.zoom}")
    for quality in RENDER_QUALITIES:
        renderer.overlays.render_quality = quality
        for mode in VISUAL_MODES:
            renderer.overlays.visual_mode = mode
            renderer.render(screen, camera, paused=False, speed=1, selection=Selection())
            start = time.perf_counter()
            sim_total = 0.0
            for _ in range(args.frames):
                if not args.render_only:
                    sim_start = time.perf_counter()
                    engine.step(1)
                    sim_total += time.perf_counter() - sim_start
                    renderer.set_external_timing("sim", sim_total / args.frames)
                renderer.render(screen, camera, paused=False, speed=1, selection=Selection())
            elapsed = time.perf_counter() - start
            fps = args.frames / elapsed if elapsed > 0 else 0.0
            print(f"{quality:>8} {mode:>7}: {elapsed / args.frames * 1000:6.2f} ms/frame {fps:6.2f} fps")
    pygame.quit()


if __name__ == "__main__":
    main()
