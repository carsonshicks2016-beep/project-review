"""
Interactive Real-Time Tandem Drift Battle Visualizer.
Features:
- Dual-car dynamic rendering (Neon Cyan Leader vs Sunburst Gold Chaser).
- Translucent aerodynamic wake cone streamlines & turbulent vortex flutter.
- Volumetric tire smoke particles with dissipation and persistent rubber skid marks.
- Live Formula Drift Tandem Telemetry HUD:
  - Door-to-Door Proximity Gauge (m clearance with sweet-spot highlight)
  - Angle Match % and Relative Slip Angle
  - Wake Turbulence & Dirty Air Downforce Loss Meter
  - Surface Friction Loss % (Molten rubber / smoke)
  - Live Heat Scores & Fault Notifications
- Modes: AI vs AI, Human Chaser vs AI Leader, Human Leader vs AI Chaser.
- Headless frame capture for high-resolution artifact generation.
"""

import argparse
import math
import os
import random
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import List, Tuple, Optional
import numpy as np
import pygame
import torch

from drift_tandem_rl.dynamics import VehicleState, VehicleParams, TandemDynamics
from drift_tandem_rl.tracks import Track, get_track
from drift_tandem_rl.scoring import TandemJudge
from drift_tandem_rl.env import TandemMultiAgentEnv
from drift_tandem_rl.mappo import MAPPO


class SmokeParticle:
    def __init__(self, x: float, y: float, vx: float, vy: float, initial_radius: float = 3.2):
        self.x = x
        self.y = y
        self.vx = vx + random.uniform(-1.0, 1.0)
        self.vy = vy + random.uniform(-1.0, 1.0)
        self.radius = initial_radius
        self.alpha = random.uniform(160, 230)
        self.lifetime = random.uniform(0.7, 1.4)
        self.age = 0.0

    def update(self, dt: float) -> bool:
        self.age += dt
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.radius += 10.0 * dt
        self.alpha = max(0.0, self.alpha - (220.0 / self.lifetime) * dt)
        return self.age < self.lifetime


class TandemVisualizer:
    def __init__(
        self,
        track_name: str = "touge",
        model_path: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        headless: bool = False,
        mode: str = "AI_VS_AI",
    ):
        self.width = width
        self.height = height
        self.headless = headless
        self.track_name = track_name
        self.mode = mode.upper()  # 'AI_VS_AI', 'HUMAN_CHASE', 'HUMAN_LEAD'

        pygame.init()
        pygame.font.init()

        if not self.headless:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("MAPPO Tandem Drift Battles | Wake Turbulence & Dirty Air Sim")
        else:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            self.screen = pygame.Surface((width, height))

        self.clock = pygame.time.Clock()
        self.font_title = pygame.font.SysFont("Arial", 26, bold=True)
        self.font_medium = pygame.font.SysFont("Arial", 18, bold=True)
        self.font_small = pygame.font.SysFont("Arial", 14)
        self.font_hud = pygame.font.SysFont("Courier", 15, bold=True)

        self.env = TandemMultiAgentEnv(track_name=track_name)
        self.mappo: Optional[MAPPO] = None

        if model_path and os.path.exists(model_path):
            print(f"Loading MAPPO model weights from {model_path}...")
            self.mappo = MAPPO(obs_dim=self.env.observation_space.shape[0], action_dim=3)
            self.mappo.load(model_path)
        else:
            # Initialize fresh MAPPO policy for self-play/demo
            self.mappo = MAPPO(obs_dim=self.env.observation_space.shape[0], action_dim=3)

        # Persistent rubber skid mark surface
        self.skid_surface = pygame.Surface((3200, 3200), pygame.SRCALPHA)
        self.skid_surface.fill((0, 0, 0, 0))
        self.world_offset = np.array([1600.0, 1600.0])

        self.smoke_particles: List[SmokeParticle] = []
        self.cam_x = 0.0
        self.cam_y = 0.0
        self.zoom = 12.0  # Pixels per meter

        # Human control commands
        self.human_steer = 0.0
        self.human_throttle = 0.0
        self.human_handbrake = 0.0

        self.obs, _ = self.env.reset()

    def world_to_screen(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = int(self.width * 0.5 + (wx - self.cam_x) * self.zoom)
        sy = int(self.height * 0.5 - (wy - self.cam_y) * self.zoom)
        return sx, sy

    def world_to_skid(self, wx: float, wy: float) -> Tuple[int, int]:
        sx = int(self.world_offset[0] + wx * self.zoom)
        sy = int(self.world_offset[1] - wy * self.zoom)
        return sx, sy

    def handle_input(self) -> bool:
        """Processes keyboard events. Returns False if user requested quit."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_r:
                    self.reset_battle()
                elif event.key == pygame.K_m:
                    # Cycle modes
                    modes = ["AI_VS_AI", "HUMAN_CHASE", "HUMAN_LEAD"]
                    curr_idx = modes.index(self.mode) if self.mode in modes else 0
                    self.mode = modes[(curr_idx + 1) % len(modes)]
                    print(f"Switched mode to: {self.mode}")

        keys = pygame.key.get_pressed()
        # Steer
        if keys[pygame.K_LEFT]:
            self.human_steer = max(-1.0, self.human_steer - 0.15)
        elif keys[pygame.K_RIGHT]:
            self.human_steer = min(1.0, self.human_steer + 0.15)
        else:
            self.human_steer *= 0.65

        # Throttle / Brake
        if keys[pygame.K_UP]:
            self.human_throttle = min(1.0, self.human_throttle + 0.20)
        elif keys[pygame.K_DOWN]:
            self.human_throttle = max(-1.0, self.human_throttle - 0.30)
        else:
            self.human_throttle *= 0.50

        # Handbrake
        self.human_handbrake = 1.0 if keys[pygame.K_SPACE] else 0.0

        return True

    def reset_battle(self):
        self.obs, _ = self.env.reset()
        self.smoke_particles.clear()

    def step(self):
        lead_obs = self.obs["leader"]
        chase_obs = self.obs["chaser"]

        # Default actions from MAPPO
        action_dict = self.mappo.select_action(lead_obs, chase_obs, deterministic=False)
        lead_act = action_dict["act_lead"]
        chase_act = action_dict["act_chase"]

        # Human override
        if self.mode == "HUMAN_CHASE":
            chase_act = np.array([self.human_steer, self.human_throttle, self.human_handbrake], dtype=np.float32)
        elif self.mode == "HUMAN_LEAD":
            lead_act = np.array([self.human_steer, self.human_throttle, self.human_handbrake], dtype=np.float32)

        # Autopilot assist for lead if MAPPO is un-trained
        if np.abs(lead_act).sum() < 0.1:
            lead_act = self._get_autopilot_lead()

        actions = {"leader": lead_act, "chaser": chase_act}
        self.obs, rewards, terminations, truncations, infos = self.env.step(actions)

        # Smooth camera tracking midpoint between both cars
        lead_s = self.env.leader_state
        chase_s = self.env.chaser_state
        mid_x = (lead_s.x + chase_s.x) * 0.5
        mid_y = (lead_s.y + chase_s.y) * 0.5
        self.cam_x += (mid_x - self.cam_x) * 0.12
        self.cam_y += (mid_y - self.cam_y) * 0.12

        # Spawn visual tire smoke
        self._spawn_car_smoke(lead_s)
        self._spawn_car_smoke(chase_s)

        # Leave skid marks on track
        self._draw_skid_marks(lead_s, (40, 40, 40, 110))
        self._draw_skid_marks(chase_s, (30, 30, 30, 140))

        # Auto-reset on terminal
        if terminations["leader"] or terminations["chaser"] or truncations["leader"]:
            self.reset_battle()

    def _get_autopilot_lead(self) -> np.ndarray:
        s = self.env.leader_state
        track = self.env.track
        curr_pos = np.array([s.x, s.y])
        targets = track.get_target_waypoints(curr_pos, count=2, step_stride=7)
        target = targets[0]

        rel = target - curr_pos
        cos_y = np.cos(s.yaw)
        sin_y = np.sin(s.yaw)
        by = -rel[0] * sin_y + rel[1] * cos_y

        steer_cmd = np.clip(by * 0.11, -1.0, 1.0)
        throttle_cmd = 0.88 if s.speed < 14.0 else 0.35
        handbrake_cmd = 0.85 if abs(by) > 4.5 and abs(s.slip_angle_deg) < 14.0 else 0.0
        return np.array([steer_cmd, throttle_cmd, handbrake_cmd], dtype=np.float32)

    def _spawn_car_smoke(self, car: VehicleState):
        if car.speed > 4.0 and abs(car.slip_angle_deg) > 13.0:
            corners = self.env.dyn.get_car_corners(car)
            # Rear wheels are corners 2 and 3
            rw1 = corners[2]
            rw2 = corners[3]
            for pt in [rw1, rw2]:
                if random.random() < 0.65:
                    self.smoke_particles.append(
                        SmokeParticle(float(pt[0]), float(pt[1]), car.velocity_world[0] * 0.3, car.velocity_world[1] * 0.3)
                    )

    def _draw_skid_marks(self, car: VehicleState, color_rgba):
        if car.speed > 5.0 and abs(car.slip_angle_deg) > 14.0:
            corners = self.env.dyn.get_car_corners(car)
            rw1 = corners[2]
            rw2 = corners[3]
            for pt in [rw1, rw2]:
                sx, sy = self.world_to_skid(pt[0], pt[1])
                pygame.draw.circle(self.skid_surface, color_rgba, (sx, sy), 3)

    def render(self):
        # 1. Asphalt Background
        self.screen.fill((34, 38, 44))

        # 2. Draw Track Boundaries & Centerline
        self._render_track()

        # 3. Blit Skid Marks
        skid_screen_pos = (
            int(self.width * 0.5 - (self.world_offset[0] + self.cam_x * self.zoom)),
            int(self.height * 0.5 - (self.world_offset[1] - self.cam_y * self.zoom)),
        )
        self.screen.blit(self.skid_surface, skid_screen_pos)

        # 4. Render Aerodynamic Wake Cone Streamlines
        self._render_wake_cone()

        # 5. Render Smoke Particles
        dt = 0.02
        surviving_smoke = []
        for p in self.smoke_particles:
            if p.update(dt):
                surviving_smoke.append(p)
                sx, sy = self.world_to_screen(p.x, p.y)
                if 0 <= sx < self.width and 0 <= sy < self.height:
                    s_surf = pygame.Surface((int(p.radius * 2), int(p.radius * 2)), pygame.SRCALPHA)
                    gray = 220
                    pygame.draw.circle(s_surf, (gray, gray, gray, int(p.alpha)), (int(p.radius), int(p.radius)), int(p.radius))
                    self.screen.blit(s_surf, (sx - int(p.radius), sy - int(p.radius)))
        self.smoke_particles = surviving_smoke

        # 6. Render Vehicles
        self._render_car(self.env.leader_state, color=(0, 235, 255), label="LEAD (CYBER)")
        self._render_car(self.env.chaser_state, color=(255, 175, 20), label="CHASE (GOLD)")

        # 7. Render Formula Drift Tandem Telemetry HUD
        self._render_hud()

        if not self.headless:
            pygame.display.flip()
            self.clock.tick(50)

    def _render_track(self):
        track = self.env.track
        # Inner & Outer boundary walls
        for seg in [track.inner_boundary, track.outer_boundary]:
            if len(seg) > 1:
                screen_pts = [self.world_to_screen(pt[0], pt[1]) for pt in seg]
                pygame.draw.lines(self.screen, (160, 160, 160), True, screen_pts, 3)

        # Centerline dashed
        c_pts = [self.world_to_screen(pt[0], pt[1]) for pt in track.centerline]
        for i in range(0, len(c_pts) - 1, 2):
            pygame.draw.line(self.screen, (75, 85, 95), c_pts[i], c_pts[i + 1], 2)

        # Clipping zones
        for zone in track.clipping_zones:
            zx, zy = self.world_to_screen(zone.x, zone.y)
            z_rad = int(zone.radius * self.zoom)
            color = (50, 220, 100) if zone.hit_leader else (255, 60, 60)
            pygame.draw.circle(self.screen, color, (zx, zy), z_rad, 2)
            lbl = self.font_small.render(f"CLIP ({zone.zone_type[:3].upper()})", True, color)
            self.screen.blit(lbl, (zx - 25, zy - 10))

    def _render_wake_cone(self):
        """Draws aerodynamic wake streamlines shedding from the leader into the follower."""
        lead = self.env.leader_state
        v_lead = lead.velocity_world
        speed = float(np.linalg.norm(v_lead))
        if speed < 2.5:
            return

        wake_dir = -v_lead / speed
        wake_norm = np.array([-wake_dir[1], wake_dir[0]])
        lead_pos = np.array([lead.x, lead.y])

        # 5 streamline ribbons
        p = self.env.dyn.params
        wake_len = p.wake_length
        streamer_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        for frac in np.linspace(-1.0, 1.0, 7):
            half_w_end = (p.width * 0.5) + wake_len * np.tan(p.wake_spread_angle)
            start_pt = lead_pos + wake_norm * (frac * p.width * 0.5)
            end_pt = lead_pos + wake_dir * wake_len + wake_norm * (frac * half_w_end)

            sx1, sy1 = self.world_to_screen(start_pt[0], start_pt[1])
            sx2, sy2 = self.world_to_screen(end_pt[0], end_pt[1])

            # Turbulent flutter when follower is in wake
            alpha = int(90 * (1.0 - abs(frac) * 0.6))
            color = (180, 220, 255, alpha) if self.env.chaser_state.in_wake < 0.2 else (255, 90, 90, alpha + 30)
            pygame.draw.line(streamer_surf, color, (sx1, sy1), (sx2, sy2), 2)

        self.screen.blit(streamer_surf, (0, 0))

    def _render_car(self, car: VehicleState, color: Tuple[int, int, int], label: str):
        corners = self.env.dyn.get_car_corners(car)
        screen_corners = [self.world_to_screen(c[0], c[1]) for c in corners]

        # Chassis body
        pygame.draw.polygon(self.screen, color, screen_corners)
        pygame.draw.polygon(self.screen, (255, 255, 255), screen_corners, 2)

        # Windshield
        c_mid = np.mean(screen_corners, axis=0)
        f_mid = (np.array(screen_corners[0]) + np.array(screen_corners[1])) * 0.5
        w_mid = c_mid + (f_mid - c_mid) * 0.45
        pygame.draw.circle(self.screen, (30, 30, 40), (int(w_mid[0]), int(w_mid[1])), 4)

        # Wheels / steer visualization
        # Front wheels turn with car.steer
        cos_y = np.cos(car.yaw + car.steer)
        sin_y = np.sin(car.yaw + car.steer)
        for fc in [screen_corners[0], screen_corners[1]]:
            w1 = (fc[0] - cos_y * 6, fc[1] + sin_y * 6)
            w2 = (fc[0] + cos_y * 6, fc[1] - sin_y * 6)
            pygame.draw.line(self.screen, (15, 15, 15), w1, w2, 4)

        # Label
        lbl = self.font_small.render(f"{label} {car.speed * 3.6:.0f}km/h | {car.slip_angle_deg:+.0f}°", True, color)
        self.screen.blit(lbl, (screen_corners[0][0] - 30, screen_corners[0][1] - 22))

    def _render_hud(self):
        lead = self.env.leader_state
        chase = self.env.chaser_state
        judge = self.env.judge

        # Clearance & angle sync
        clearance = judge.proximity_history[-1] if judge.proximity_history else 0.0
        angle_diff = judge.angle_diff_history[-1] if judge.angle_diff_history else 0.0
        angle_match_pct = max(0.0, 100.0 * (1.0 - angle_diff / np.radians(45.0)))

        # Header panel
        panel_rect = pygame.Rect(15, 15, 480, 210)
        panel_surf = pygame.Surface((panel_rect.width, panel_rect.height), pygame.SRCALPHA)
        panel_surf.fill((15, 20, 28, 225))
        pygame.draw.rect(panel_surf, (60, 130, 220), (0, 0, panel_rect.width, panel_rect.height), 2)
        self.screen.blit(panel_surf, (panel_rect.x, panel_rect.y))

        # Title
        t_surf = self.font_title.render("FORMULA DRIFT TANDEM BATTLE", True, (0, 235, 255))
        self.screen.blit(t_surf, (28, 24))

        mode_lbl = self.font_small.render(f"MODE: {self.mode}  (Press 'M' to switch, 'R' to reset)", True, (220, 220, 120))
        self.screen.blit(mode_lbl, (28, 56))

        # Proximity Gauge
        prox_color = (60, 255, 120) if (0.8 <= clearance <= 2.8) else ((255, 60, 60) if clearance > 5.0 else (255, 220, 40))
        p_text = f"DOOR-TO-DOOR GAP: {clearance:.2f} m"
        if clearance <= 0.8 and clearance > 0.0:
            p_text += " [KISSING DOORS!]"
        elif 0.8 <= clearance <= 2.8:
            p_text += " [SWEET SPOT]"
        else:
            p_text += " [TRAILING]"
        self.screen.blit(self.font_medium.render(p_text, True, prox_color), (28, 80))

        # Angle Match & Slip Angles
        a_text = f"ANGLE MATCH: {angle_match_pct:.0f}%  (Lead: {lead.slip_angle_deg:+.0f}° | Chase: {chase.slip_angle_deg:+.0f}°)"
        self.screen.blit(self.font_small.render(a_text, True, (230, 230, 230)), (28, 108))

        # Aerodynamic Wake Status
        wake_color = (255, 80, 80) if chase.in_wake > 0.25 else (140, 200, 255)
        w_text = f"DIRTY AIR INTENSITY: {chase.in_wake * 100:.0f}%  | FRONT DOWNFORCE LOSS: {chase.in_wake * 60:.0f}%"
        self.screen.blit(self.font_small.render(w_text, True, wake_color), (28, 130))

        # Surface smoke friction degradation
        f_text = f"TIRE SMOKE FRICTION DEGRADATION: {chase.smoke_exposure * 30:.1f}% LOSS"
        self.screen.blit(self.font_small.render(f_text, True, (255, 180, 60)), (28, 152))

        # Live Battle Scores
        s_lead = judge.lead_score.total_score
        s_chase = judge.chase_score.total_score
        scores_text = f"LEAD SCORE: {s_lead:.1f} pts  |  CHASE SCORE: {s_chase:.1f} pts"
        self.screen.blit(self.font_hud.render(scores_text, True, (255, 255, 255)), (28, 178))

    def run_interactive_loop(self):
        """Standard 60 FPS interactive Pygame loop."""
        running = True
        while running:
            running = self.handle_input()
            self.step()
            self.render()
        pygame.quit()

    def capture_frame(self, filename: str):
        """Renders one high-res frame and saves to disk as artifact."""
        self.step()
        self.render()
        try:
            import matplotlib.pyplot as plt
            arr = pygame.surfarray.array3d(self.screen)
            arr = np.transpose(arr, (1, 0, 2))
            plt.imsave(filename, arr)
            print(f"Captured telemetry screenshot to {filename}")
        except Exception as e:
            try:
                import cv2
                arr = pygame.surfarray.array3d(self.screen)
                arr = cv2.cvtColor(np.transpose(arr, (1, 0, 2)), cv2.COLOR_RGB2BGR)
                cv2.imwrite(filename, arr)
                print(f"Captured telemetry screenshot via OpenCV to {filename}")
            except Exception as e2:
                pygame.image.save(self.screen, filename.replace(".png", ".bmp"))
                print(f"Fallback saved as BMP to {filename.replace('.png', '.bmp')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tandem Drift Battle Visualizer")
    parser.add_argument("--track", type=str, default="touge", choices=["touge", "gymkhana"])
    parser.add_argument("--model", type=str, default=None, help="Path to MAPPO checkpoint (.pt)")
    parser.add_argument("--mode", type=str, default="AI_VS_AI", choices=["AI_VS_AI", "HUMAN_CHASE", "HUMAN_LEAD"])
    parser.add_argument("--headless", action="store_true", help="Run without X11/display")
    parser.add_argument("--screenshot", type=str, default=None, help="Save frame to image file")

    args = parser.parse_args()

    vis = TandemVisualizer(
        track_name=args.track,
        model_path=args.model,
        headless=args.headless or (args.screenshot is not None),
        mode=args.mode,
    )

    if args.screenshot:
        for _ in range(35):
            vis.step()
        vis.capture_frame(args.screenshot)
    else:
        vis.run_interactive_loop()
