from __future__ import annotations
import numpy as np
import random
from dataclasses import dataclass, field
from collections import defaultdict
import math

from supra.config import get_car, SimSpec, SensorSpec
from supra.physics import Controls, Vehicle
from supra.track import named_track
from supra.sensors import SensorSuite
from supra.app import AutoBox
from supra.fable5 import compute_speed_envelope_geometry

from .brain import LSTMBrain
from .config import Driver2Config, EvalConfig


@dataclass
class EvalResult:
    """Results from evaluating a driver2 brain on the Nordschleife."""
    lap_time: float | None
    clean_sectors: int
    sector_count: int
    clean_chain: int
    terminal_rate: float
    pace_ratio: float
    max_progress_m: float
    offtrack_seconds: float
    soc_trace: list[float]
    mean_speed: float
    theoretical_lap: float
    footprint_legal: bool
    terminal_reasons: dict = field(default_factory=dict)


class Driver2Evaluator:
    """Deterministic evaluator for Driver 2.0 LSTMBrain."""
    
    def __init__(self, cfg: Driver2Config | None = None):
        self.cfg = cfg or Driver2Config()
        
        # Load Track
        self.track = named_track(self.cfg.track)
        
        # Setup Specs
        self.car_spec = get_car(self.cfg.car)
        self.sim_spec = SimSpec(dt=self.cfg.dt)
        self.sensor_spec = SensorSpec()
        self.sensor_spec.hill_block = True
        self.sensor_spec.pace_block = True
        self.sensor_spec.hybrid_block = True
        
        # Instantiate environment objects
        self.veh = Vehicle(self.car_spec, self.sim_spec)
        self.sensors = SensorSuite(self.sensor_spec)
        self.autobox = AutoBox(self.car_spec)
        
        # Calculate speed envelope
        n = len(self.track.curvature)
        ds = np.zeros(n)
        for i in range(n):
            j = (i + 1) % n
            ds[i] = self.track.arc[j] - self.track.arc[i]
            if ds[i] <= 0:
                ds[i] += self.track.length
        grade = getattr(self.track, "grade", None)
        res = compute_speed_envelope_geometry(self.car_spec, ds, self.track.curvature, grade=grade)
        self.v_ref = res["v"]
        self.theoretical_lap = res["lap_time"]

        # Precompute 16 sector starts
        self.sector_count = 16
        self.sector_indices = np.linspace(0, len(self.track.arc) - 1, self.sector_count, dtype=int)
        
    def _run_episode(self, brain: LSTMBrain, start_idx: int, start_speed: float, max_time: float, 
                     track_footprint: bool = False, track_soc: bool = False) -> dict:
        """Runs a single episode (sector or flying lap)."""
        x = self.track.center[start_idx][0]
        y = self.track.center[start_idx][1]
        
        tx, ty = self.track.tangent[start_idx]
        yaw = math.atan2(ty, tx)
        
        self.veh.reset(x, y, yaw, speed=start_speed)
        hidden = brain.reset_hidden()
        
        sim_time = 0.0
        start_arc = self.track.arc[start_idx]
        
        control_steps = max(1, round(1.0 / (self.cfg.ga.control_hz * self.cfg.dt)))
        
        term_reason = ""
        terminal = False
        clean = True
        
        offtrack_time = 0.0
        stall_time = 0.0
        
        soc_trace = []
        last_soc_time = -1.0
        
        footprint_legal = True
        
        mean_speed_sum = 0.0
        steps = 0
        
        pace_ratio_sum = 0.0
        
        # Main loop
        step_idx = 0
        actions = np.zeros(3)
        while sim_time < max_time:
            # 1. Physics frame
            veh_x, veh_y = self.veh.x, self.veh.y
            frame = self.track.frame(veh_x, veh_y)
            
            curr_arc = frame['arc']
            
            # Progress checking (wrap around for lap completion)
            progress_m = curr_arc - start_arc
            if progress_m < -self.track.length * 0.5:
                progress_m += self.track.length
            elif progress_m > self.track.length * 0.5:
                progress_m -= self.track.length
                
            if progress_m >= self.track.length - 10.0: # Close enough to lap
                break
                
            # Set road conditions
            self.veh.set_road(
                grade=frame['grade'],
                bank=frame['bank'],
                heading=frame['heading'],
                z=frame['z'],
                vcurv=frame['vcurv']
            )
            self.veh.surface_grip = 1.0
            
            # Brain control step
            if step_idx % control_steps == 0:
                obs = self.sensors.observe(self.veh, self.track)
                actions, hidden = brain.act_numpy(obs.vector, hidden)
                
            steer = float(actions[0])
            longitudinal = float(actions[1])
            gear_offset = round(float(actions[2]) * 2.0)
            
            throttle = max(0.0, longitudinal)
            brake = max(0.0, -longitudinal)
            
            clutch, shift_up, shift_down = self.autobox.update(self.veh, throttle, self.cfg.dt)
            
            if gear_offset > 0:
                shift_up = True
            elif gear_offset < 0:
                shift_down = True
                
            controls = Controls(
                steer=steer,
                throttle=throttle,
                brake=brake,
                clutch=clutch,
                shift_up=shift_up,
                shift_down=shift_down
            )
            
            self.veh.step(controls, self.cfg.dt)
            
            # Logging and rules
            sim_time += self.cfg.dt
            step_idx += 1
            mean_speed_sum += self.veh.speed
            steps += 1
            
            # Pace ratio
            idx_curr = np.searchsorted(self.track.arc, curr_arc) % len(self.track.arc)
            v_ref = self.v_ref[idx_curr]
            if v_ref > 0:
                pace_ratio_sum += (self.veh.speed / v_ref)
            
            if track_soc and sim_time - last_soc_time >= 1.0:
                soc_trace.append(self.veh.hybrid_soc_kj)
                last_soc_time = sim_time
                
            if track_footprint:
                obb = self.veh.get_obb()
                for pt in obb:
                    f = self.track.frame(pt[0], pt[1])
                    if abs(f['lateral']) > f['half_width']:
                        footprint_legal = False
                        break
            
            # Termination rules
            if frame['off_track']:
                offtrack_time += self.cfg.dt
                clean = False
            else:
                offtrack_time = 0.0
                
            if offtrack_time > 2.0:
                terminal = True
                term_reason = "offtrack_timeout"
                break
                
            if abs(frame['lateral']) > frame['half_width'] + 20.0:
                terminal = True
                term_reason = "way_off_track"
                break
                
            if self.veh.speed < 1.0:
                stall_time += self.cfg.dt
            else:
                stall_time = 0.0
                
            if stall_time > 3.0:
                terminal = True
                term_reason = "stall"
                break
                
            # Check backwards
            speed_vec_x = self.veh.speed * math.cos(self.veh.yaw)
            speed_vec_y = self.veh.speed * math.sin(self.veh.yaw)
            dot = speed_vec_x * tx + speed_vec_y * ty
            if dot < -5.0:
                terminal = True
                term_reason = "backwards"
                break
                
        if not terminal and sim_time >= max_time:
            term_reason = "timeout"
            
        return {
            'time': sim_time,
            'progress_m': progress_m,
            'terminal': terminal,
            'term_reason': term_reason,
            'clean': clean,
            'footprint_legal': footprint_legal,
            'soc_trace': soc_trace,
            'mean_speed': mean_speed_sum / max(1, steps),
            'pace_ratio': pace_ratio_sum / max(1, steps),
            'offtrack_seconds': offtrack_time if not clean else 0.0
        }

    def evaluate(self, brain: LSTMBrain, n_laps: int = 1) -> EvalResult:
        """Evaluates the brain over 16 sectors and optionally full flying laps."""
        if hasattr(self.cfg, 'eval') and hasattr(self.cfg.eval, 'rng_seed'):
            random.seed(self.cfg.eval.rng_seed)
            np.random.seed(self.cfg.eval.rng_seed)
        
        # Phase 1: Sectors
        clean_sectors = 0
        terminals = 0
        term_reasons = defaultdict(int)
        pace_ratios = []
        
        chain = 0
        max_chain = 0
        
        sector_seconds = getattr(self.cfg.eval, 'sector_seconds', 30.0) if hasattr(self.cfg, 'eval') else 30.0
        
        for idx in self.sector_indices:
            start_speed = min(60.0, self.v_ref[idx] * 0.90)
            res = self._run_episode(brain, idx, start_speed, max_time=sector_seconds)
            
            if res['clean']:
                clean_sectors += 1
                chain += 1
                max_chain = max(max_chain, chain)
            else:
                chain = 0
                
            if res['terminal']:
                terminals += 1
                if res['term_reason']:
                    term_reasons[res['term_reason']] += 1
                    
            pace_ratios.append(res['pace_ratio'])
            
        terminal_rate = terminals / self.sector_count
        avg_pace = sum(pace_ratios) / max(1, len(pace_ratios))
        
        # Phase 2: Flying Lap
        best_lap_time = None
        best_progress = 0.0
        best_offtrack = 0.0
        best_soc = []
        best_mean_speed = 0.0
        best_footprint = True
        
        lap_budget = getattr(self.cfg.eval, 'lap_budget', 900.0) if hasattr(self.cfg, 'eval') else 900.0
        
        for _ in range(n_laps):
            start_idx = 0
            start_speed = min(60.0, self.v_ref[0] * 0.90)
            res = self._run_episode(brain, start_idx, start_speed, max_time=lap_budget, 
                                    track_footprint=True, track_soc=True)
            
            is_complete = res['progress_m'] >= self.track.length - 20.0
            
            if is_complete:
                if best_lap_time is None or res['time'] < best_lap_time:
                    best_lap_time = res['time']
                    best_progress = self.track.length
                    best_offtrack = res['offtrack_seconds']
                    best_soc = res['soc_trace']
                    best_mean_speed = res['mean_speed']
                    best_footprint = res['footprint_legal']
            else:
                if best_lap_time is None and res['progress_m'] > best_progress:
                    best_progress = res['progress_m']
                    best_offtrack = res['offtrack_seconds']
                    best_soc = res['soc_trace']
                    best_mean_speed = res['mean_speed']
                    best_footprint = res['footprint_legal']
                    
            if res['terminal'] and res['term_reason']:
                term_reasons[res['term_reason']] += 1
                
        return EvalResult(
            lap_time=best_lap_time,
            clean_sectors=clean_sectors,
            sector_count=self.sector_count,
            clean_chain=max_chain,
            terminal_rate=terminal_rate,
            pace_ratio=avg_pace,
            max_progress_m=best_progress,
            offtrack_seconds=best_offtrack,
            soc_trace=best_soc,
            mean_speed=best_mean_speed,
            theoretical_lap=self.theoretical_lap,
            footprint_legal=best_footprint,
            terminal_reasons=dict(term_reasons)
        )
