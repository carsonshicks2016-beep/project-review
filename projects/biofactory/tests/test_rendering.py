from __future__ import annotations

import os
import unittest
from dataclasses import replace

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from biofactory.config import DEFAULT_CONFIG
from biofactory.rendering.camera import Camera
from biofactory.rendering.renderer import Renderer
from biofactory.rendering.ui import Selection
from biofactory.simulation.engine import SimulationEngine


class RendererSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        pygame.init()
        cls.screen = pygame.display.set_mode((640, 420))

    @classmethod
    def tearDownClass(cls) -> None:
        pygame.quit()

    def test_visual_modes_render_at_multiple_zooms(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=120),
            world=replace(
                DEFAULT_CONFIG.world,
                width=90,
                height=60,
                nest_x=45.0,
                nest_y=30.0,
                leaf_patch_count=6,
            ),
        )
        engine = SimulationEngine(config)
        renderer = Renderer(engine)
        engine.step(80)

        for render_quality in ("fast", "balanced", "quality"):
            renderer.overlays.render_quality = render_quality
            for visual_mode in ("hybrid", "circuit", "natural"):
                renderer.overlays.visual_mode = visual_mode
                for zoom in (0.55, 1.0, 2.0):
                    camera = Camera(
                        x=config.world.nest_x,
                        y=config.world.nest_y,
                        screen_width=400,
                        screen_height=420,
                        render_config=config.render,
                        zoom=zoom,
                    )
                    renderer.render(self.screen, camera, paused=False, speed=1, selection=Selection())

    def test_renderer_handles_ten_thousand_ant_frame(self) -> None:
        config = replace(
            DEFAULT_CONFIG,
            ants=replace(DEFAULT_CONFIG.ants, count=10000),
            world=replace(
                DEFAULT_CONFIG.world,
                width=90,
                height=60,
                nest_x=45.0,
                nest_y=30.0,
                leaf_patch_count=4,
            ),
        )
        engine = SimulationEngine(config)
        renderer = Renderer(engine)
        camera = Camera(
            x=config.world.nest_x,
            y=config.world.nest_y,
            screen_width=400,
            screen_height=420,
            render_config=config.render,
            zoom=0.55,
        )
        renderer.render(self.screen, camera, paused=False, speed=1, selection=Selection())


if __name__ == "__main__":
    unittest.main()
