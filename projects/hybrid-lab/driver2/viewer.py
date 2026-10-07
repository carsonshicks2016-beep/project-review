import argparse
import json
import os
import time
import math
import sys
import collections

import numpy as np
import pygame

from supra.track import named_track
from supra.config import get_car, SimSpec, SensorSpec
from supra.physics import Vehicle, Controls
from supra.app import AutoBox
from .config import Driver2Config, BrainConfig, GAConfig, CurriculumConfig, EvalConfig
from .brain import LSTMBrain
from .env import Driver2Env

def main(run_id=None):
    if run_id is None:
        parser = argparse.ArgumentParser()
        parser.add_argument("--run", type=str, required=True, help="Run ID")
        args = parser.parse_args()
        run_id = args.run

    run_dir = os.path.join("runtime/driver2", run_id)

    # We will need these files
    manifest_path = os.path.join(run_dir, "manifest.json")
    global generations_path, best_path
    generations_path = os.path.join(run_dir, "generations.jsonl")
    best_path = os.path.join(run_dir, "best.npz")

    global driver_cfg, env
    driver_cfg = Driver2Config()

    if not os.path.exists(manifest_path):
        print(f"Warning: manifest.json not found in {run_dir}, using default config.")
    else:
        with open(manifest_path, "r") as f:
            manifest_data = json.load(f)
        cfg_data = manifest_data.get("cfg", manifest_data)
        if "car" in cfg_data: driver_cfg.car = cfg_data["car"]
        if "track" in cfg_data: driver_cfg.track = cfg_data["track"]

    env = Driver2Env(driver_cfg)

    pygame.init()
    WIDTH, HEIGHT = 1280, 720
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Driver 2.0 Viewer - {run_id}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Courier", 16)
    large_font = pygame.font.SysFont("Courier", 24, bold=True)
    
    class ReplayRunner:
        def __init__(self, env: Driver2Env, brain: LSTMBrain):
            self.env = env
            self.brain = brain
            self.trk = env.track
            self.n_pts = len(self.trk.center) if hasattr(self.trk, "center") else self.trk.n_points
            self.start_idx = 0
            self.veh = Vehicle(env.spec, env.sim)
            
            sx, sy = self.trk.center[self.start_idx % self.n_pts]
            syaw = float(np.arctan2(self.trk.tangent[self.start_idx % self.n_pts, 1], self.trk.tangent[self.start_idx % self.n_pts, 0]))
            self.veh.reset(sx, sy, syaw, speed=50.0)
            
            self.box = AutoBox(env.spec)
            self.hidden = brain.reset_hidden()
            self.actions_np = np.zeros(env.cfg.brain.n_actions)
            self.t = 0.0
            self.step_count = 0
    
        def step(self):
            for _ in range(2):  # 60fps * 2 = 120hz physics
                fr = self.trk.frame(self.veh.x, self.veh.y)
                if self.step_count % self.env.control_period == 0:
                    obs = self.env.sensors.observe(self.veh, self.trk)
                    self.actions_np, self.hidden = self.brain.act_numpy(obs.vector, self.hidden)
                    
                steer = float(np.clip(self.actions_np[0], -1.0, 1.0))
                long_val = float(np.clip(self.actions_np[1], -1.0, 1.0))
                throttle = max(0.0, long_val)
                brake = max(0.0, -long_val)
                
                clutch, shift_up, shift_down = self.box.update(self.veh, throttle, self.env.dt)
                if self.env.cfg.brain.n_actions >= 3:
                    gear_offset = int(round(float(self.actions_np[2]) * 2.0))
                    target_gear = self.veh.gear + gear_offset
                    target_gear = max(1, min(len(self.env.spec.gear_ratios), target_gear))
                    shift_up = target_gear > self.veh.gear
                    shift_down = target_gear < self.veh.gear
                    
                self.veh.surface_grip = self.env.spec.offtrack_grip if fr["off_track"] else 1.0
                self.veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])
                self.veh.step(
                    Controls(steer=steer, throttle=throttle, brake=brake, clutch=clutch, shift_up=shift_up, shift_down=shift_down),
                    self.env.dt
                )
                self.t += self.env.dt
                self.step_count += 1
                
                # Restart segment if off track too long or stopped
                if fr["off_track"] or (self.t > 3.0 and self.veh.speed < 1.0):
                    self.__init__(self.env, self.brain)
                    break
    
    runner = None
    last_best_mtime = 0
    last_check_time = 0
    
    # Data for dashboard
    generations = []
    best_fitness = []
    mean_fitness = []
    worst_fitness = []
    
    def load_data():
        global last_best_mtime, runner, generations, best_fitness, mean_fitness, worst_fitness
        
        if os.path.exists(generations_path):
            gens = []
            bfs, mfs, wfs = [], [], []
            try:
                with open(generations_path, "r") as f:
                    for line in f:
                        if not line.strip(): continue
                        d = json.loads(line)
                        gens.append(d["generation"])
                        bfs.append(d["best_fitness"])
                        mfs.append(d["mean_fitness"])
                        wfs.append(d["worst_fitness"])
                generations = gens
                best_fitness = bfs
                mean_fitness = mfs
                worst_fitness = wfs
            except Exception:
                pass
    
        if os.path.exists(best_path):
            mtime = os.path.getmtime(best_path)
            if mtime > last_best_mtime:
                last_best_mtime = mtime
                try:
                    data = np.load(best_path)
                    genome = data["genome"]
                    brain = LSTMBrain(driver_cfg.brain)
                    brain.set_genome(genome)
                    runner = ReplayRunner(env, brain)
                    print(f"Loaded new brain from {best_path}")
                except Exception as e:
                    print(f"Error loading best.npz: {e}")
    
    load_data()
    
    # Track points for minimap
    tcenter = env.track.center
    min_x, max_x = np.min(tcenter[:,0]), np.max(tcenter[:,0])
    min_y, max_y = np.min(tcenter[:,1]), np.max(tcenter[:,1])
    w = max_x - min_x
    h = max_y - min_y
    scale = min(400/w, 600/h) if w>0 and h>0 else 1.0
    
    def draw_track_map(surface, rect):
        pygame.draw.rect(surface, (30, 30, 30), rect)
        pygame.draw.rect(surface, (100, 100, 100), rect, 2)
        
        cx, cy = rect.center
        pts = []
        for p in tcenter:
            px = cx + (p[0] - (min_x + w/2)) * scale
            py = cy - (p[1] - (min_y + h/2)) * scale  # flip y
            pts.append((px, py))
        
        if len(pts) > 1:
            pygame.draw.lines(surface, (150, 150, 150), True, pts, 2)
            
        if runner:
            vx = cx + (runner.veh.x - (min_x + w/2)) * scale
            vy = cy - (runner.veh.y - (min_y + h/2)) * scale
            pygame.draw.circle(surface, (255, 50, 50), (int(vx), int(vy)), 4)
    
    def draw_replay(surface, rect):
        pygame.draw.rect(surface, (20, 20, 20), rect)
        pygame.draw.rect(surface, (100, 100, 100), rect, 2)
        
        if not runner:
            text = font.render("No Replay Data", True, (200, 200, 200))
            surface.blit(text, text.get_rect(center=rect.center))
            return
            
        cx, cy = rect.center
        zoom = 5.0
        
        veh_x, veh_y = runner.veh.x, runner.veh.y
        veh_yaw = runner.veh.yaw
        
        # Draw track around car
        # We only draw points near the car to save time
        car_pos = np.array([veh_x, veh_y])
        dists = np.linalg.norm(tcenter - car_pos, axis=1)
        close_idx = np.where(dists < 100)[0]
        
        if len(close_idx) > 0:
            pts = []
            for i in close_idx:
                p = tcenter[i]
                dx = p[0] - veh_x
                dy = p[1] - veh_y
                
                # Rotate by -yaw so car faces up
                sn, cs = math.sin(-veh_yaw + math.pi/2), math.cos(-veh_yaw + math.pi/2)
                rx = dx * cs - dy * sn
                ry = dx * sn + dy * cs
                
                sx = cx + rx * zoom
                sy = cy - ry * zoom
                pts.append((sx, sy))
                
            for i in range(len(pts)-1):
                pygame.draw.line(surface, (150,150,150), pts[i], pts[i+1], 4)
                
        # Draw car
        car_w, car_h = 10, 20
        car_rect = pygame.Rect(0, 0, car_w, car_h)
        car_rect.center = (cx, cy)
        pygame.draw.rect(surface, (255, 50, 50), car_rect)
        
        # speed
        speed_text = font.render(f"Speed: {runner.veh.speed*3.6:.0f} km/h", True, (255, 255, 255))
        surface.blit(speed_text, (rect.x + 10, rect.y + 10))
    
    def draw_plots(surface, rect):
        pygame.draw.rect(surface, (15, 15, 15), rect)
        pygame.draw.rect(surface, (100, 100, 100), rect, 2)
        
        if not generations:
            return
            
        margin = 20
        plt_w = rect.width - margin*2
        plt_h = rect.height - margin*2
        
        max_gen = max(generations) if generations else 1
        max_fit = max(best_fitness) if best_fitness else 1
        min_fit = min(worst_fitness) if worst_fitness else 0
        
        fit_range = max(max_fit - min_fit, 0.1)
        
        def to_pt(g, f):
            x = rect.x + margin + (g / max_gen) * plt_w
            y = rect.y + rect.height - margin - ((f - min_fit) / fit_range) * plt_h
            return (x, y)
            
        b_pts = [to_pt(g, f) for g, f in zip(generations, best_fitness)]
        m_pts = [to_pt(g, f) for g, f in zip(generations, mean_fitness)]
        w_pts = [to_pt(g, f) for g, f in zip(generations, worst_fitness)]
        
        if len(b_pts) > 1:
            pygame.draw.lines(surface, (50, 255, 50), False, b_pts, 2)
            pygame.draw.lines(surface, (255, 255, 50), False, m_pts, 2)
            pygame.draw.lines(surface, (255, 50, 50), False, w_pts, 2)
            
        lbl = font.render(f"Best: {best_fitness[-1]:.2f}", True, (50, 255, 50))
        surface.blit(lbl, (rect.x + margin, rect.y + margin))
    
    def draw_status(surface, rect):
        pygame.draw.rect(surface, (40, 40, 40), rect)
        pygame.draw.rect(surface, (100, 100, 100), rect, 2)
        
        y = rect.y + 20
        x = rect.x + 20
        
        title = large_font.render(f"Run: {args.run}", True, (255, 255, 255))
        surface.blit(title, (x, y))
        y += 40
        
        gen_lbl = font.render(f"Generations: {generations[-1] if generations else 0}", True, (200, 200, 200))
        surface.blit(gen_lbl, (x, y))
        y += 30
        
        if runner:
            act = runner.actions_np
            act_lbl = font.render(f"Actions: St {act[0]:.2f} Lg {act[1]:.2f}", True, (200, 200, 200))
            surface.blit(act_lbl, (x, y))
    
    running = True
    while running:
        dt = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
        now = time.time()
        if now - last_check_time > 1.0:
            load_data()
            last_check_time = now
            
        if runner:
            runner.step()
            
        screen.fill((0, 0, 0))
        
        # layout
        # Left: Track map
        draw_track_map(screen, pygame.Rect(20, 20, 400, 480))
        # Bottom Left: Plot
        draw_plots(screen, pygame.Rect(20, 520, 400, 180))
        
        # Center/Right: Replay
        draw_replay(screen, pygame.Rect(440, 20, 520, 680))
        
        # Right: Status
        draw_status(screen, pygame.Rect(980, 20, 280, 680))
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
