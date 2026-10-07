"""Driver 2.0 environment — single-segment episode runner.

Wraps the existing 919 Evo vehicle physics, Nordschleife track geometry,
fable-v2 sensor suite, and AutoBox gearbox into a lightweight evaluation
loop.  One episode = one segment of the curriculum.

The env is stateless between episodes: call evaluate_segment() with a brain
and a segment definition, get back a SegmentResult.
"""
from __future__ import annotations

import numpy as np
import torch

from supra.config import get_car, SimSpec, SensorSpec
from supra.physics import Controls, Vehicle
from supra.track import named_track
from supra.sensors import SensorSuite
from supra.app import AutoBox

from .brain import LSTMBrain
from .config import Driver2Config, GAConfig


class Driver2Env:
    """Evaluate an LSTM brain on a single track segment."""

    def __init__(self, cfg: Driver2Config | None = None):
        cfg = cfg or Driver2Config()
        self.cfg = cfg
        self.dt = cfg.dt

        # Vehicle + track (reuse existing supra infrastructure)
        self.spec = get_car(cfg.car)
        self.sim = SimSpec()
        self.track = named_track(cfg.track)

        # Fable-v2 sensor suite: base + hills + pace + hybrid
        sensor_cfg = SensorSpec()
        sensor_cfg.hill_block = True
        sensor_cfg.pace_block = True
        sensor_cfg.hybrid_block = True
        self.sensors = SensorSuite(sensor_cfg)

        # Verify obs size matches brain config
        assert self.sensors.obs_size == cfg.brain.obs_size, (
            f"Sensor obs_size {self.sensors.obs_size} != "
            f"brain obs_size {cfg.brain.obs_size}"
        )

        # Speed envelope for entry speeds and pace ratio
        self.v_ref = self._compute_speed_envelope()

        # Control period: brain runs at control_hz, physics at 1/dt Hz
        self.control_period = max(1, round(1.0 / (cfg.ga.control_hz * self.dt)))

    def _compute_speed_envelope(self) -> np.ndarray:
        """Compute the physics-true speed envelope for the track."""
        from supra.fable5 import compute_speed_envelope_geometry

        trk = self.track
        n = len(trk.curvature)
        ds = np.diff(trk.arc, append=trk.arc[0] + trk.length) - np.concatenate(
            [[0.0], np.diff(trk.arc)]
        )
        # Use segment lengths from the track's arc array
        ds = np.zeros(n)
        for i in range(n):
            j = (i + 1) % n
            ds[i] = trk.arc[j] - trk.arc[i]
            if ds[i] <= 0:
                ds[i] += trk.length
        grade = trk.grade if hasattr(trk, "grade") else None
        res = compute_speed_envelope_geometry(
            self.spec, ds, trk.curvature, grade=grade
        )
        return res["v"]

    # ------------------------------------------------------------------ #
    # Segment evaluation
    # ------------------------------------------------------------------ #
    def evaluate_segment(
        self,
        brain: LSTMBrain,
        start_idx: int,
        end_idx: int,
        entry_speed: float,
        time_budget: float,
        v_ref_segment: np.ndarray | None = None,
    ) -> "SegmentResult":
        """Run one episode: drive from start_idx to end_idx.

        Args:
            brain:       LSTM brain (weights already set).
            start_idx:   Track centerline index for segment start.
            end_idx:     Track centerline index for segment end.
            entry_speed: Vehicle speed at spawn (m/s).
            time_budget: Max episode duration (s).
            v_ref_segment: Reference speeds over the segment (for pace ratio).

        Returns:
            SegmentResult with fitness, progress, pace, clean/terminal status.
        """
        trk = self.track
        n_pts = len(trk.center) if hasattr(trk, "center") else trk.n_points

        # Spawn the vehicle at the segment start
        veh = Vehicle(self.spec, self.sim)
        sx, sy = trk.center[start_idx % n_pts]
        syaw = float(
            np.arctan2(trk.tangent[start_idx % n_pts, 1],
                       trk.tangent[start_idx % n_pts, 0])
        )
        veh.reset(sx, sy, syaw, speed=float(entry_speed))

        box = AutoBox(self.spec)
        hidden = brain.reset_hidden()

        # Segment arc-length boundaries
        arc_start = float(trk.arc[start_idx % n_pts])
        arc_end = float(trk.arc[end_idx % n_pts])
        if arc_end <= arc_start:
            arc_end += trk.length  # wrap

        seg_length = arc_end - arc_start

        # Tracking
        t = 0.0
        step_count = 0
        offtrack_time = 0.0
        total_speed = 0.0
        speed_samples = 0
        max_progress = 0.0
        terminal = False
        terminal_reason = None
        actions_np = np.zeros(self.cfg.brain.n_actions)

        # Stall / backwards detection
        stall_time = 0.0
        since_progress = 0.0
        best_progress = 0.0

        while t < time_budget:
            # Sensor observation
            fr = trk.frame(veh.x, veh.y)

            # Brain inference at control_hz
            if step_count % self.control_period == 0:
                obs = self.sensors.observe(veh, trk)
                actions_np, hidden = brain.act_numpy(obs.vector, hidden)

            # Map actions to Controls (fable layout: steer, long, gear_offset)
            steer = float(np.clip(actions_np[0], -1.0, 1.0))
            long_val = float(np.clip(actions_np[1], -1.0, 1.0))
            throttle = max(0.0, long_val)
            brake = max(0.0, -long_val)

            # Gear from AutoBox + brain offset
            clutch, shift_up, shift_down = box.update(veh, throttle, self.dt)
            if self.cfg.brain.n_actions >= 3:
                gear_offset = int(round(float(actions_np[2]) * 2.0))
                target_gear = veh.gear + gear_offset
                target_gear = max(1, min(len(self.spec.gear_ratios), target_gear))
                shift_up = target_gear > veh.gear
                shift_down = target_gear < veh.gear

            # Road plane + physics step
            veh.surface_grip = self.spec.offtrack_grip if fr["off_track"] else 1.0
            veh.set_road(
                fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"]
            )
            veh.step(
                Controls(
                    steer=steer,
                    throttle=throttle,
                    brake=brake,
                    clutch=clutch,
                    shift_up=shift_up,
                    shift_down=shift_down,
                ),
                self.dt,
            )

            t += self.dt
            step_count += 1

            # Track progress
            fr_post = trk.frame(veh.x, veh.y)
            cur_arc = float(fr_post["arc"])
            # Unwrap relative to segment start
            progress_arc = cur_arc - arc_start
            if progress_arc < -trk.length / 2:
                progress_arc += trk.length
            elif progress_arc > trk.length / 2:
                progress_arc -= trk.length
            progress_frac = max(0.0, progress_arc / seg_length)
            max_progress = max(max_progress, progress_frac)

            # Speed tracking
            total_speed += veh.speed
            speed_samples += 1

            # Off-track accumulation
            if fr_post["off_track"]:
                offtrack_time += self.dt

            # Termination checks
            # Off-track too long
            if offtrack_time > 2.0:
                terminal = True
                terminal_reason = "offtrack_timeout"
                break

            # Way off track
            if abs(fr_post["lateral"]) > fr_post.get("half_width", 10.0) + 6.0:
                terminal = True
                terminal_reason = "left_track"
                break

            # Stall detection (after 1s grace)
            if t > 1.0 and veh.speed < 0.8:
                stall_time += self.dt
            else:
                stall_time = 0.0
            if stall_time > 3.0:
                terminal = True
                terminal_reason = "stall"
                break

            # Backwards detection
            if progress_frac < max_progress - 0.10:
                terminal = True
                terminal_reason = "backwards"
                break

            # No progress timeout
            if progress_frac > best_progress + 0.001:
                best_progress = progress_frac
                since_progress = 0.0
            else:
                since_progress += self.dt
            if since_progress > 5.0:
                terminal = True
                terminal_reason = "no_progress"
                break

            # Segment complete
            if progress_frac >= 1.0:
                break

        # Compute results
        mean_speed = total_speed / max(speed_samples, 1)

        # Pace ratio: compare to mean reference speed over the segment
        if v_ref_segment is not None and len(v_ref_segment) > 0:
            mean_ref = float(np.mean(v_ref_segment))
        else:
            # Fallback: compute from v_ref over segment indices
            if end_idx > start_idx:
                idxs = np.arange(start_idx, end_idx) % n_pts
            else:
                idxs = np.concatenate([
                    np.arange(start_idx, n_pts),
                    np.arange(0, end_idx),
                ])
            mean_ref = float(np.mean(self.v_ref[idxs])) if len(idxs) > 0 else 50.0
        pace_ratio = mean_speed / max(mean_ref, 1.0)

        clean = offtrack_time == 0.0 and not terminal

        # Fitness: progress × pace × clean_bonus - terminal_penalty
        fitness = (
            min(max_progress, 1.0)
            * pace_ratio
            * (1.5 if clean else 1.0)
            - (0.5 if terminal else 0.0)
        )

        from .curriculum import SegmentResult

        return SegmentResult(
            segment_idx=0,  # caller sets this
            progress_frac=min(max_progress, 1.0),
            mean_speed=mean_speed,
            pace_ratio=pace_ratio,
            clean=clean,
            terminal=terminal,
            terminal_reason=terminal_reason or "",
            offtrack_seconds=offtrack_time,
            time_elapsed=t,
            fitness=fitness,
        )
