"""
3D front-end entry point for Supra-AI.

Provides `run_3d(app)` which takes an existing App instance (with its
simulation, evolution, sound engine, etc. already set up), closes the
existing 2D pygame display, creates a new OPENGL|DOUBLEBUF window with
the proper Core Profile attributes for macOS, and runs the render loop
using Renderer3D.

Usage from main.py or wherever the App is created::

    from supra.app3d import run_3d
    run_3d(app)          # replaces app.run()
"""

from __future__ import annotations
import math
import sys
import traceback

import numpy as np

try:
    import pygame
except ImportError:
    pygame = None

try:
    import moderngl
except ImportError:
    moderngl = None


def run_3d(app):
    """Replace the 2D rendering loop with a full 3D OpenGL loop.

    Parameters
    ----------
    app : supra.app.App
        An initialised App instance.  Its simulation, evolution, sound,
        and configuration are reused; only the display and render path
        change.
    """
    # ---- dependency checks ----
    if moderngl is None:
        print(
            "\n╔══════════════════════════════════════════════════════════════╗\n"
            "║  moderngl is required for 3D mode.                         ║\n"
            "║  Install it with:  pip install moderngl                    ║\n"
            "╚══════════════════════════════════════════════════════════════╝\n"
        )
        sys.exit(1)

    if pygame is None:
        print("ERROR: pygame is required.  pip install pygame")
        sys.exit(1)

    from .viewer3d import Renderer3D

    cfg = app.cfg
    width = cfg.render.width
    height = cfg.render.height

    # ---- tear down existing 2D display and reinit video subsystem ----
    try:
        pygame.display.quit()
    except Exception:
        pass
    pygame.display.init()

    # ---- set OpenGL Core Profile attributes (macOS CRITICAL) ----
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_FORWARD_COMPATIBLE_FLAG, 1)
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
    pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)

    # ---- create OPENGL window ----
    screen = pygame.display.set_mode(
        (width, height),
        pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("Supra-AI · 3D OpenGL")

    # ---- create renderer and build initial track mesh ----
    try:
        renderer = Renderer3D(cfg, width, height)
    except Exception as exc:
        print(f"\n[3D] Failed to create Renderer3D:\n{exc}")
        traceback.print_exc()
        sys.exit(1)

    renderer._current_track = app.sim.track
    renderer.set_track(app.sim.track)
    _track_id = id(app.sim.track)

    # ---- font for HUD overlay ----
    # We'll render HUD text to a pygame surface and blit via moderngl texture
    hud_font = pygame.font.SysFont("menlo,consolas,monospace", 14)
    hud_font_b = pygame.font.SysFont("menlo,consolas,monospace", 18, bold=True)

    # ---- HUD overlay texture + quad ----
    hud_tex = renderer.ctx.texture((width, height), 4)
    hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

    hud_prog = renderer.ctx.program(
        vertex_shader="""
        #version 330 core
        in vec2 in_pos;
        in vec2 in_uv;
        out vec2 v_uv;
        void main() {
            gl_Position = vec4(in_pos, 0.0, 1.0);
            v_uv = in_uv;
        }
        """,
        fragment_shader="""
        #version 330 core
        uniform sampler2D u_tex;
        in vec2 v_uv;
        out vec4 fragColor;
        void main() {
            vec4 c = texture(u_tex, v_uv);
            fragColor = c;
        }
        """
    )

    hud_verts = np.array([
        # pos        uv
        -1, -1,    0, 0,
         1, -1,    1, 0,
        -1,  1,    0, 1,
         1, -1,    1, 0,
         1,  1,    1, 1,
        -1,  1,    0, 1,
    ], dtype=np.float32)
    hud_vbo = renderer.ctx.buffer(hud_verts.tobytes())
    hud_vao = renderer.ctx.vertex_array(
        hud_prog, [(hud_vbo, '2f 2f', 'in_pos', 'in_uv')])

    # ---- main loop ----
    clock = pygame.time.Clock()
    running = True
    dt = cfg.sim.dt

    # Sound engine is already initialised in the App
    if hasattr(app, 'sound'):
        app.sound.start()

    while running:
        # ---- events (reuse app._events) ----
        running = app._events()

        # ---- advance simulation ----
        if not app.paused:
            app._advance(dt)

        # ---- detect track change ----
        if id(app.sim.track) != _track_id:
            renderer._current_track = app.sim.track
            renderer.set_track(app.sim.track)
            _track_id = id(app.sim.track)

        # ---- camera ----
        fc = app.sim.focus_car
        if fc is not None:
            # Smooth camera transition on focus change (no jarring snap)
            if app.sim.focus != app._prev_focus:
                app._prev_focus = app.sim.focus
                v = fc.vehicle
                renderer._cam_yaw = v.yaw  # snap yaw only
                # Ensure camera stays above the track
                elev = 0.0
                if renderer._current_track is not None:
                    elev = renderer._smooth_elev_at(
                        renderer._current_track, fc._idx, v.x, v.y)
                min_z = elev + 8.0  # always stay well above
                if renderer.cam_pos[2] < min_z:
                    renderer.cam_pos[2] = min_z
            renderer.update_camera(fc, dt)

        # ---- sound ----
        if hasattr(app, 'sound'):
            cam_x = float(renderer.cam_pos[0])
            cam_y = float(renderer.cam_pos[1])
            view_radius = 200.0
            app.sound.update(app.sim.cars, app.sim.focus,
                             [cam_x, cam_y], view_radius, muted=app.muted)

        # ---- render 3D scene ----
        skid = getattr(app, 'skid', None)
        renderer.render(app.sim, app.smoke, app.flames, skid=skid)

        # ---- HUD overlay ----
        _render_hud(renderer, hud_tex, hud_vao, hud_prog,
                    hud_font, hud_font_b, app, width, height)

        # ---- swap ----
        pygame.display.flip()
        clock.tick(cfg.render.fps)

    # ---- cleanup ----
    if hasattr(app, 'sound'):
        app.sound.stop()
    renderer.release()
    pygame.quit()


# ──────────────────────────────────────────────────────────────────────
# HUD overlay
# ──────────────────────────────────────────────────────────────────────

def _render_hud(renderer, hud_tex, hud_vao, hud_prog,
                font, font_b, app, width, height):
    """Render HUD text to a pygame surface, upload as texture, and blit
    over the 3D scene as a transparent overlay."""

    sim = app.sim
    fc = sim.focus_car

    # Create a transparent RGBA surface for the HUD
    hud_surf = pygame.Surface((width, height), pygame.SRCALPHA)

    # Semi-transparent background bar at bottom
    bar_h = 110
    bar = pygame.Surface((width, bar_h), pygame.SRCALPHA)
    bar.fill((0, 0, 0, 140))
    hud_surf.blit(bar, (0, height - bar_h))

    y_base = height - bar_h + 8
    x = 16

    # ---- Top-left info ----
    if app.paused:
        hud_surf.blit(font_b.render("PAUSED", True, (240, 200, 70)),
                      (12, 12))
    if app.fast > 1:
        hud_surf.blit(font_b.render(f"FAST x{app.fast}", True, (90, 210, 120)),
                      (12, 36))
    if app.muted:
        hud_surf.blit(font.render("MUTED", True, (230, 70, 60)), (12, 60))

    if hasattr(app, 'reload_ppo_path') and app.reload_ppo_path is not None:
        hud_surf.blit(font_b.render("● LIVE PPO TRAINING", True,
                      (90, 210, 120)), (12, 84))

    # ---- Bottom bar content ----
    # Track info
    track_str = f"Track: {sim.track.kind}"
    hud_surf.blit(font.render(track_str, True, (180, 185, 195)),
                  (x, y_base))

    gen_str = f"Gen: {sim.generation}   Alive: {sim.alive_count}/{len(sim.cars)}"
    hud_surf.blit(font.render(gen_str, True, (180, 185, 195)),
                  (x, y_base + 20))

    if fc is not None:
        v = fc.vehicle
        speed_kph = v.speed * 3.6
        rpm = v.rpm
        gear = v.gear + 1  # 1-indexed for display

        speed_str = f"Speed: {speed_kph:5.1f} km/h"
        rpm_str = f"RPM: {rpm:5.0f}  Gear: {gear}"
        fitness_str = f"Fitness: {fc.fitness:8.1f}  Dist: {fc.distance - fc.start_distance:7.1f}m"

        # Speed (large, right side)
        speed_render = font_b.render(f"{speed_kph:.0f}", True, (255, 255, 255))
        unit_render = font.render("km/h", True, (150, 155, 165))
        hud_surf.blit(speed_render, (width - 180, y_base))
        hud_surf.blit(unit_render, (width - 180 + speed_render.get_width() + 6,
                                    y_base + 4))

        # RPM bar
        rpm_frac = min(1.0, (rpm - 850) / (7000 - 850))
        bar_w = 200
        bar_x = width - 190
        bar_y = y_base + 30
        # Background
        pygame.draw.rect(hud_surf, (40, 42, 48), (bar_x, bar_y, bar_w, 12))
        # Fill
        if rpm_frac > 0.85:
            rpm_col = (230, 60, 50)  # redline
        elif rpm_frac > 0.7:
            rpm_col = (240, 180, 40)
        else:
            rpm_col = (60, 200, 120)
        pygame.draw.rect(hud_surf, rpm_col,
                         (bar_x, bar_y, int(bar_w * rpm_frac), 12))
        hud_surf.blit(font.render(f"G{gear}", True, (220, 225, 235)),
                      (bar_x + bar_w + 8, bar_y - 2))

        # Info text
        col = (170, 175, 185)
        hud_surf.blit(font.render(speed_str, True, col), (x, y_base + 40))
        hud_surf.blit(font.render(rpm_str, True, col), (x, y_base + 58))
        hud_surf.blit(font.render(fitness_str, True, col), (x, y_base + 76))

        # Boost indicator
        if hasattr(v, 'boost') and v.boost > 0.05:
            boost_str = f"BOOST: {v.boost:.2f} bar"
            hud_surf.blit(font.render(boost_str, True, (100, 200, 255)),
                          (x + 380, y_base + 40))

        # Drift indicator
        if hasattr(v, 'slip_angle') and abs(v.slip_angle) > 0.20:
            angle_deg = abs(math.degrees(v.slip_angle))
            drift_col = (255, 160, 40) if angle_deg < 30 else (255, 60, 40)
            drift_str = f"DRIFT {angle_deg:.0f}°"
            hud_surf.blit(font_b.render(drift_str, True, drift_col),
                          (width // 2 - 60, y_base - 30))

    # Help text
    help_text = ("[SPACE] pause  [F] next car  [A] auto-follow  "
                 "[M] mute  [T] fast  [R] new track  [+/-] zoom  [ESC] quit")
    hud_surf.blit(font.render(help_text, True, (120, 125, 140)),
                  (x, y_base + bar_h - 26))

    # ---- Upload to texture ----
    # Flip vertically for GL coords and convert to bytes
    hud_data = pygame.image.tostring(hud_surf, 'RGBA', True)
    hud_tex.write(hud_data)

    # ---- Draw HUD quad over the scene ----
    renderer.ctx.disable(moderngl.DEPTH_TEST)
    hud_tex.use(0)
    hud_prog['u_tex'].value = 0
    hud_vao.render()
    renderer.ctx.enable(moderngl.DEPTH_TEST)
