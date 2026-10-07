"""
Interactive real-time visualizer for high-speed drifting and trick chaining.
Includes tire smoke particle physics, persistent rubber skid marks,
arcade trick combo HUD, and dual Human vs AI playable modes.
"""

import argparse
import math
import os
import random
import sys
from typing import List, Tuple, Optional
import numpy as np
import pygame
from stable_baselines3 import PPO, SAC

from drift_rl.dynamics import VehicleState, VehicleDynamics, VehicleParams
from drift_rl.tracks import Track, get_track
from drift_rl.scoring import DriftScorer, TrickEvent
from drift_rl.env import DriftGymkhanaEnv


# Particle class for tire smoke
class SmokeParticle:
    def __init__(self, x: float, y: float, vx: float, vy: float, initial_radius: float = 3.0):
        self.x = x
        self.y = y
        self.vx = vx + random.uniform(-1.0, 1.0)
        self.vy = vy + random.uniform(-1.0, 1.0)
        self.radius = initial_radius
        self.alpha = random.uniform(160, 220)
        self.lifetime = random.uniform(0.6, 1.2)
        self.age = 0.0

    def update(self, dt: float) -> bool:
        self.age += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.radius += 12.0 * dt  # expands as it dissipates
        self.alpha = max(0.0, self.alpha - (220.0 / self.lifetime) * dt)
        return self.age < self.lifetime


# Floating text popup for trick notifications
class TrickPopup:
    def __init__(self, text: str, points: float, mult_bonus: float):
        self.text = f"{text}  +{int(points):,} (x{mult_bonus:+.1f})"
        self.age = 0.0
        self.lifetime = 2.2
        self.y_offset = 0.0

    def update(self, dt: float) -> bool:
        self.age += dt
        self.y_offset -= 25.0 * dt
        return self.age < self.lifetime


class DriftVisualizer:
    def __init__(
        self,
        track_name: str = "touge",
        model_path: Optional[str] = None,
        algo: str = "ppo",
        width: int = 1280,
        height: int = 720,
        headless: bool = False,
    ):
        self.width = width
        self.height = height
        self.headless = headless
        self.track_name = track_name

        pygame.init()
        pygame.font.init()
        if not self.headless:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Antigravity ML Drift Master | AI vs Human Drift Telemetry")
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = pygame.Surface((width, height))

        self.clock = pygame.time.Clock()
        self.font_large = pygame.font.SysFont("Arial", 32, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 22, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 16)

        self.env = DriftGymkhanaEnv(track_name=track_name)
        self.model = None
        if model_path and os.path.exists(model_path):
            print(f"Loading {algo.upper()} model: {model_path}")
            self.model = SAC.load(model_path) if algo.lower() == "sac" else PPO.load(model_path)
            self.mode = "AI"
        else:
            self.mode = "HUMAN"  # Default to human drive if no model loaded

        # Skid mark surface (persistent rubber on track)
        self.skid_surface = pygame.Surface((3000, 3000), pygame.SRCALPHA)
        self.skid_surface.fill((0, 0, 0, 0))
        self.world_offset = np.array([1500.0, 1500.0])  # Center of skid surface

        self.smoke_particles: List[SmokeParticle] = []
        self.popups: List[TrickPopup] = []
        self.camera_pos = np.array([0.0, 0.0])
        self.zoom = 12.0  # pixels per meter

        self.last_corners: Optional[np.ndarray] = None
        self.obs, self.info = self.env.reset()

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates (meters) to screen pixel coordinates."""
        dx = (wx - self.camera_pos[0]) * self.zoom
        dy = (wy - self.camera_pos[1]) * self.zoom
        # Inverted Y for standard screen coordinates
        sx = int(self.width * 0.5 + dx)
        sy = int(self.height * 0.5 - dy)
        return sx, sy

    def world_to_skid(self, wx: float, wy: float) -> Tuple[int, int]:
        """Convert world coordinates to skid mark texture pixels."""
        # 10 pixels per meter on skid surface
        scale = 10.0
        sx = int(self.world_offset[0] + wx * scale)
        sy = int(self.world_offset[1] - wy * scale)
        return sx, sy

    def switch_track(self, track_name: str):
        self.track_name = track_name
        self.env = DriftGymkhanaEnv(track_name=track_name)
        self.skid_surface.fill((0, 0, 0, 0))
        self.smoke_particles.clear()
        self.popups.clear()
        self.obs, self.info = self.env.reset()

    def run(self, max_frames: Optional[int] = None):
        running = True
        frame = 0

        while running:
            dt = 0.02  # 50 Hz sim rate

            # Handle Events & Keyboard Controls
            human_steer = 0.0
            human_throttle = 0.0
            human_handbrake = 0.0

            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_m:
                            self.mode = "HUMAN" if self.mode == "AI" else "AI"
                        elif event.key == pygame.K_r:
                            self.obs, self.info = self.env.reset()
                            self.skid_surface.fill((0, 0, 0, 0))
                        elif event.key == pygame.K_1:
                            self.switch_track("touge")
                        elif event.key == pygame.K_2:
                            self.switch_track("gymkhana")
                        elif event.key == pygame.K_3:
                            self.switch_track("stadium")
                        elif event.key == pygame.K_ESCAPE:
                            running = False

                # Human Driving Controls
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                    human_steer += 1.0
                if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                    human_steer -= 1.0
                if keys[pygame.K_UP] or keys[pygame.K_w]:
                    human_throttle += 1.0
                if keys[pygame.K_DOWN] or keys[pygame.K_s]:
                    human_throttle -= 1.0
                if keys[pygame.K_SPACE]:
                    human_handbrake = 1.0

            # Step Environment
            if self.mode == "AI" and self.model is not None:
                action, _ = self.model.predict(self.obs, deterministic=True)
            elif self.mode == "HUMAN":
                action = np.array([human_steer, human_throttle, human_handbrake], dtype=np.float32)
            else:
                action = np.array([0.0, 0.5, 0.0], dtype=np.float32)

            self.obs, reward, terminated, truncated, self.info = self.env.step(action)

            # Trigger Popups for Tricks
            for trick in self.env.scorer.recent_tricks:
                if self.env.scorer.sim_time - trick.timestamp < 0.03:
                    self.popups.append(TrickPopup(trick.name, trick.points, trick.multiplier_bonus))

            # Auto-reset on termination in interactive view
            if terminated or truncated:
                self.obs, self.info = self.env.reset()

            # Smooth Camera Follow
            target_cam = np.array([self.env.state.x, self.env.state.y])
            self.camera_pos += (target_cam - self.camera_pos) * 0.15

            # Particle & Skid Simulation
            self._update_effects(dt)

            # Render Scene
            self._render()

            frame += 1
            if max_frames and frame >= max_frames:
                break

            if not self.headless:
                self.clock.tick(50)

        pygame.quit()

    def _update_effects(self, dt: float):
        state = self.env.state
        corners = self.env.dyn.get_car_corners(state)
        speed = state.speed
        slip_deg = abs(state.slip_angle_deg)

        # Rear tire positions: corners[2] (rear right), corners[3] (rear left)
        rr = corners[2]
        rl = corners[3]

        # Draw rubber skid marks if sliding
        if slip_deg > 14.0 and speed > 2.5:
            scale = 10.0
            p_rr = self.world_to_skid(rr[0], rr[1])
            p_rl = self.world_to_skid(rl[0], rl[1])

            if self.last_corners is not None:
                prev_rr = self.world_to_skid(self.last_corners[2, 0], self.last_corners[2, 1])
                prev_rl = self.world_to_skid(self.last_corners[3, 0], self.last_corners[3, 1])
                alpha = int(min(200, slip_deg * 3.5))
                skid_color = (25, 25, 25, alpha)
                pygame.draw.line(self.skid_surface, skid_color, prev_rr, p_rr, 4)
                pygame.draw.line(self.skid_surface, skid_color, prev_rl, p_rl, 4)

            # Spawn tire smoke particles
            smoke_count = int(min(6, (slip_deg / 15.0)))
            for _ in range(smoke_count):
                pos = random.choice([rr, rl])
                self.smoke_particles.append(
                    SmokeParticle(pos[0], pos[1], -state.vx * 0.2, -state.vy * 0.2)
                )

        self.last_corners = corners.copy()

        # Update smoke particles
        self.smoke_particles = [p for p in self.smoke_particles if p.update(dt)]
        # Update popups
        self.popups = [pop for pop in self.popups if pop.update(dt)]

    def _render(self):
        # 1. Background Grass/Asphalt
        self.screen.fill((28, 30, 36))  # Dark asphalt background

        # 2. Render Skidmark Surface
        skid_scale = 10.0
        # Map skid surface into screen space
        sx, sy = self.world_to_screen(0.0, 0.0)
        # For simplicity, blit visible track lines and boundaries
        self._render_track()

        # 3. Render Smoke Particles
        for p in self.smoke_particles:
            px, py = self.world_to_screen(p.x, p.y)
            r = int(p.radius * (self.zoom / 10.0))
            if 0 <= px < self.width and 0 <= py < self.height and r > 1:
                smoke_surf = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                color = (220, 225, 235, int(p.alpha))
                pygame.draw.circle(smoke_surf, color, (r, r), r)
                self.screen.blit(smoke_surf, (px - r, py - r))

        # 4. Render Car
        self._render_car()

        # 5. Render Arcade HUD
        self._render_hud()
        if not self.headless:
            pygame.display.flip()

    def save_frame(self, filename: str):
        """Save current frame to disk."""
        try:
            from PIL import Image
            raw_str = pygame.image.tostring(self.screen, "RGB")
            img = Image.frombytes("RGB", (self.width, self.height), raw_str)
            img.save(filename)
        except Exception:
            # Fallback to BMP if PIL fails
            base, _ = os.path.splitext(filename)
            pygame.image.save(self.screen, f"{base}.bmp")

    def _render_track(self):
        track = self.env.track

        # Draw outer & inner walls
        def draw_poly_wire(poly, color, width=3):
            pts = [self.world_to_screen(p[0], p[1]) for p in poly]
            if len(pts) > 2:
                pygame.draw.polygon(self.screen, (40, 44, 52), pts)  # track fill
                pygame.draw.lines(self.screen, color, True, pts, width)

        draw_poly_wire(track.outer_boundary, (80, 90, 110), 4)
        draw_poly_wire(track.inner_boundary, (80, 90, 110), 4)

        # Centerline racing guide (faint dashed yellow)
        center_pts = [self.world_to_screen(p[0], p[1]) for p in track.centerline]
        if len(center_pts) > 2:
            pygame.draw.lines(self.screen, (70, 75, 85), True, center_pts, 1)

        # Obstacles
        for obs in track.obstacles:
            pts = [self.world_to_screen(p[0], p[1]) for p in obs]
            pygame.draw.polygon(self.screen, (160, 50, 50), pts)
            pygame.draw.lines(self.screen, (220, 80, 80), True, pts, 2)

        # Clipping Zones
        for zone in track.clipping_zones:
            zx, zy = self.world_to_screen(zone.x, zone.y)
            zr = int(zone.radius * self.zoom)
            color = (50, 220, 120) if zone.hit else (240, 180, 40)
            pygame.draw.circle(self.screen, color, (zx, zy), zr, 2)
            pygame.draw.circle(self.screen, color, (zx, zy), 4)

    def _render_car(self):
        state = self.env.state
        corners = self.env.dyn.get_car_corners(state)
        pts = [self.world_to_screen(c[0], c[1]) for c in corners]

        # Chassis Body: Sleek drift neon styling
        body_color = (0, 200, 255) if self.mode == "AI" else (255, 120, 0)
        pygame.draw.polygon(self.screen, body_color, pts)
        pygame.draw.lines(self.screen, (255, 255, 255), True, pts, 2)

        # Windshield / Roof cabin
        center = np.mean(corners, axis=0)
        cx, cy = self.world_to_screen(center[0], center[1])
        pygame.draw.circle(self.screen, (20, 25, 35), (cx, cy), int(1.1 * self.zoom))

        # Render Tires (4 wheels)
        p = self.env.params
        hw = p.width * 0.5
        wheel_len = 0.8
        wheel_wid = 0.35

        def draw_wheel(pos_local, angle):
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            corners = np.array([
                [wheel_len * 0.5, wheel_wid * 0.5],
                [wheel_len * 0.5, -wheel_wid * 0.5],
                [-wheel_len * 0.5, -wheel_wid * 0.5],
                [-wheel_len * 0.5, wheel_wid * 0.5],
            ])
            rot_w = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            w_pts = corners @ rot_w.T

            # Rotate to world
            cos_y = np.cos(state.yaw)
            sin_y = np.sin(state.yaw)
            rot_car = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
            wheel_center_world = (pos_local @ rot_car.T) + np.array([state.x, state.y])

            poly_world = w_pts + wheel_center_world
            poly_screen = [self.world_to_screen(pt[0], pt[1]) for pt in poly_world]
            pygame.draw.polygon(self.screen, (15, 15, 20), poly_screen)
            pygame.draw.lines(self.screen, (100, 100, 110), True, poly_screen, 1)

        # Front wheels (steered by state.steer)
        draw_wheel(np.array([p.lf, hw]), state.yaw + state.steer)
        draw_wheel(np.array([p.lf, -hw]), state.yaw + state.steer)
        # Rear wheels (fixed to chassis heading)
        draw_wheel(np.array([-p.lr, hw]), state.yaw)
        draw_wheel(np.array([-p.lr, -hw]), state.yaw)

        # Headlights & Taillights
        fl_s = pts[0]
        fr_s = pts[1]
        rr_s = pts[2]
        rl_s = pts[3]
        pygame.draw.circle(self.screen, (255, 255, 180), fl_s, 4)  # Front Left Headlight
        pygame.draw.circle(self.screen, (255, 255, 180), fr_s, 4)  # Front Right Headlight
        pygame.draw.circle(self.screen, (255, 40, 40), rr_s, 4)    # Rear Right Taillight
        pygame.draw.circle(self.screen, (255, 40, 40), rl_s, 4)    # Rear Left Taillight

    def _render_hud(self):
        score_st = self.env.scorer.get_state()
        state = self.env.state
        speed_kmh = state.speed * 3.6
        slip_deg = abs(state.slip_angle_deg)

        # Top Bar: Total Score & Active Combo
        score_text = self.font_large.render(f"DRIFT SCORE: {int(score_st.total_score):,}", True, (255, 255, 255))
        self.screen.blit(score_text, (20, 20))

        # Active combo multiplier
        mult_color = (255, 215, 0) if score_st.multiplier > 1.5 else (180, 180, 180)
        combo_str = f"COMBO: x{score_st.multiplier:.1f}"
        if score_st.active_combo_score > 0:
            combo_str += f" (+{int(score_st.active_combo_score):,})"
        combo_text = self.font_large.render(combo_str, True, mult_color)
        self.screen.blit(combo_text, (20, 60))

        # Mode Indicator (AI / HUMAN)
        mode_color = (0, 220, 255) if self.mode == "AI" else (255, 150, 30)
        mode_text = self.font_medium.render(f"MODE: [{self.mode}] (Press 'M' to switch, 'R' to reset)", True, mode_color)
        self.screen.blit(mode_text, (20, 105))

        track_text = self.font_small.render(f"Track: {self.track_name.upper()} (Keys 1-3 to switch track)", True, (160, 160, 170))
        self.screen.blit(track_text, (20, 135))

        # Bottom Left: Telemetry Gauges
        gauge_y = self.height - 110
        speed_text = self.font_medium.render(f"SPEED: {int(speed_kmh)} KM/H", True, (255, 255, 255))
        self.screen.blit(speed_text, (20, gauge_y))

        angle_color = (50, 255, 120) if slip_deg < 40 else (255, 200, 40) if slip_deg < 75 else (255, 60, 200)
        angle_text = self.font_medium.render(f"DRIFT ANGLE: {int(slip_deg)}°", True, angle_color)
        self.screen.blit(angle_text, (20, gauge_y + 30))

        if score_st.is_drifting:
            status_text = self.font_medium.render(">> DRIFTING <<", True, (255, 80, 40))
            self.screen.blit(status_text, (20, gauge_y + 60))

        # Floating Trick Banners (Animated Center Screen)
        popup_y = 180
        for popup in self.popups:
            alpha_ratio = max(0.0, 1.0 - (popup.age / popup.lifetime))
            pop_surf = self.font_large.render(popup.text, True, (255, 230, 80))
            pop_surf.set_alpha(int(alpha_ratio * 255))
            rect = pop_surf.get_rect(center=(self.width // 2, popup_y + int(popup.y_offset)))
            self.screen.blit(pop_surf, rect)
            popup_y += 38


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Drift Visualizer")
    parser.add_argument("--track", type=str, default="touge", choices=["touge", "gymkhana", "stadium"])
    parser.add_argument("--model", type=str, default=None, help="Path to trained model .zip")
    parser.add_argument("--algo", type=str, default="ppo", choices=["ppo", "sac"])
    parser.add_argument("--headless", action="store_true", help="Run in headless offscreen mode")
    parser.add_argument("--frames", type=int, default=None, help="Max frames to run")
    parser.add_argument("--save-screenshot", type=str, default=None, help="Save frame screenshot path")
    args = parser.parse_args()

    vis = DriftVisualizer(
        track_name=args.track,
        model_path=args.model,
        algo=args.algo,
        headless=args.headless,
    )
    vis.run(max_frames=args.frames)
    if args.save_screenshot:
        vis.save_frame(args.save_screenshot)
        print(f"Screenshot saved to {args.save_screenshot}")
