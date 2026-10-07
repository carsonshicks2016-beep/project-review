from __future__ import annotations

if __package__ is None or __package__ == "":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from dataclasses import replace
from time import perf_counter

from biofactory.config import DEFAULT_CONFIG, SimulationConfig
from biofactory.pheromones.pheromone_config import PheromoneType
from biofactory.resources.resource_types import ResourceType
from biofactory.simulation.engine import SimulationEngine
from biofactory.simulation.time_control import TimeControl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BioFactory: Neural Ant Logistics Simulator")
    parser.add_argument("--headless", action="store_true", help="run without Pygame rendering")
    parser.add_argument("--ticks", type=int, default=1200, help="ticks for headless mode")
    parser.add_argument("--ants", type=int, default=DEFAULT_CONFIG.ants.count, help="ant count")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.world.seed, help="deterministic world seed")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> SimulationConfig:
    return replace(
        DEFAULT_CONFIG,
        ants=replace(DEFAULT_CONFIG.ants, count=args.ants),
        world=replace(DEFAULT_CONFIG.world, seed=args.seed),
    )


def run_headless(config: SimulationConfig, ticks: int) -> None:
    engine = SimulationEngine(config)
    while engine.tick < ticks:
        engine.step(min(100, ticks - engine.tick))
    snapshot = engine.snapshot()
    print("BioFactory headless run complete")
    print(f"ticks: {snapshot.tick}")
    print(f"population: {snapshot.population}")
    print(f"brood eggs/larvae/pupae: {engine.colony.eggs}/{engine.colony.larvae}/{engine.colony.pupae}")
    print(f"deliveries: {snapshot.deliveries}")
    print(f"leaves stored: {snapshot.leaves_stored:.2f}")
    print(f"water stored: {snapshot.water_stored:.2f}")
    print(f"protein stored: {snapshot.protein_stored:.2f}")
    print(f"waste stored: {snapshot.waste_stored:.2f}")
    print(f"food stored: {snapshot.food_stored:.2f}")
    print(f"leaves world: {snapshot.leaves_world:.2f}")
    print(f"water world: {snapshot.world_resources[ResourceType.WATER]:.2f}")
    print(f"protein world: {snapshot.world_resources[ResourceType.PROTEIN]:.2f}")
    print(f"waste world: {snapshot.world_resources[ResourceType.WASTE]:.2f}")
    print(f"demand: {snapshot.demand_ratio:.3f}")
    print(f"max food pheromone: {snapshot.max_food_pheromone:.2f}")
    print(f"congested cells: {snapshot.traffic.congested_cells}")


def run_visual(config: SimulationConfig) -> None:
    import pygame

    from biofactory.rendering.camera import Camera
    from biofactory.rendering.renderer import Renderer
    from biofactory.rendering.ui import Selection

    pygame.init()
    screen = pygame.display.set_mode((config.window.width, config.window.height))
    pygame.display.set_caption(config.window.title)
    clock = pygame.time.Clock()

    engine = SimulationEngine(config)
    renderer = Renderer(engine)
    viewport_width = config.window.width - config.render.ui_width
    camera = Camera(
        x=config.world.nest_x,
        y=config.world.nest_y,
        screen_width=viewport_width,
        screen_height=config.window.height,
        render_config=config.render,
        zoom=1.0,
    )
    time_control = TimeControl(config.speed_options)
    selection = Selection()
    running = True

    while running:
        dt_ms = clock.tick(config.window.fps)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                _handle_keydown(event.key, time_control, renderer, camera, config, pygame)
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx < viewport_width:
                    factor = 1.12 if event.y > 0 else 1.0 / 1.12
                    camera.zoom_at(factor, mx, my)
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if event.pos[0] < viewport_width:
                    wx, wy = camera.screen_to_world(*event.pos)
                    ant = engine.nearest_ant(wx, wy, max_distance=max(1.5, 4.0 / camera.zoom))
                    if ant is not None:
                        selection.ant_id = ant.ant_id
                        selection.chamber_id = None
                        selection.cell = None
                    elif (chamber := engine.colony.chamber_at(wx, wy)) is not None:
                        selection.ant_id = None
                        selection.chamber_id = chamber.chamber_id
                        selection.cell = None
                    elif engine.world.in_bounds(wx, wy):
                        selection.ant_id = None
                        selection.chamber_id = None
                        selection.cell = engine.world.cell(wx, wy)

        _handle_pan(camera, dt_ms, pygame)
        if time_control.paused:
            if time_control.consume_step_request():
                sim_start = perf_counter()
                engine.step(1)
                renderer.set_external_timing("sim", perf_counter() - sim_start)
        else:
            sim_start = perf_counter()
            engine.step(time_control.speed)
            renderer.set_external_timing("sim", perf_counter() - sim_start)

        renderer.render(screen, camera, time_control.paused, time_control.speed, selection)
        pygame.display.flip()

    pygame.quit()


def _handle_keydown(key: int, time_control: TimeControl, renderer: Renderer, camera: Camera, config: SimulationConfig, pygame) -> None:
    if key == pygame.K_SPACE:
        time_control.toggle_pause()
    elif key == pygame.K_PERIOD:
        time_control.request_step()
    elif key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5):
        index = key - pygame.K_1
        if 0 <= index < len(config.speed_options):
            time_control.speed_index = index
    elif key == pygame.K_f:
        renderer.overlays.toggle(PheromoneType.FOOD)
    elif key == pygame.K_i:
        renderer.overlays.toggle(PheromoneType.WATER)
    elif key == pygame.K_o:
        renderer.overlays.toggle(PheromoneType.PROTEIN)
    elif key == pygame.K_v:
        renderer.overlays.toggle(PheromoneType.WASTE)
    elif key == pygame.K_d:
        renderer.overlays.toggle(PheromoneType.DEMAND)
    elif key == pygame.K_t:
        renderer.overlays.toggle(PheromoneType.TRAFFIC)
    elif key == pygame.K_r:
        renderer.overlays.resources = not renderer.overlays.resources
    elif key == pygame.K_h:
        renderer.overlays.traffic_heatmap = not renderer.overlays.traffic_heatmap
    elif key == pygame.K_b:
        renderer.overlays.cycle_visual_mode()
    elif key == pygame.K_p:
        renderer.overlays.cycle_render_quality()
    elif key == pygame.K_c:
        camera.center_on(config.world.nest_x, config.world.nest_y)


def _handle_pan(camera: Camera, dt_ms: int, pygame) -> None:
    keys = pygame.key.get_pressed()
    speed = 34.0 * (dt_ms / 1000.0)
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        speed *= 2.5
    dx = 0.0
    dy = 0.0
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        dx -= speed
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        dx += speed
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        dy -= speed
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        dy += speed
    if dx or dy:
        camera.pan_world(dx, dy)


def main() -> None:
    args = parse_args()
    config = build_config(args)
    if args.headless:
        run_headless(config, args.ticks)
    else:
        run_visual(config)


if __name__ == "__main__":
    main()
