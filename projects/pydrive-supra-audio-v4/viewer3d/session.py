"""Realtime physics session for the standalone 3D browser viewer.

This module may share simulator primitives, but it must stay independent from
the 2D PyGame viewer/rendering modules.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time
import zlib

import numpy as np

from supra.config import CarSpec, SimSpec, get_car
from supra.physics import Controls, Vehicle
from supra import track as track_mod


def _stable_seed(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) % 9999


def resolve_track(name: str = "club", seed: int = 7, style: str | None = None,
                  difficulty: float = 0.5, length: float | None = None):
    """Resolve the same track families the rest of the simulator uses."""
    name = (name or "club").strip()
    if style:
        return track_mod.make_track(difficulty, style, length, seed=seed), {
            "name": f"{style}-{seed}",
            "kind": "generated",
            "style": style,
            "difficulty": difficulty,
            "seed": seed,
        }
    if name in track_mod.named_list():
        return track_mod.named_track(name), {
            "name": name,
            "kind": "named",
            "seed": _stable_seed(name),
        }
    if name == "oval":
        return track_mod.oval(), {"name": "oval", "kind": "legacy", "seed": seed}
    if name == "touge":
        return track_mod.touge(seed=seed), {"name": "touge", "kind": "legacy", "seed": seed}
    if name == "random":
        return track_mod.random_circuit(seed=seed), {"name": "random", "kind": "legacy", "seed": seed}
    return track_mod.named_track("club"), {"name": "club", "kind": "named", "seed": _stable_seed("club")}


class AutoBox3D:
    """3D-viewer-local automatic gearbox helper.

    This mirrors the behavior of the 2D app's AutoBox without importing the 2D
    app module.
    """
    def __init__(self, spec: CarSpec):
        self.spec = spec
        self.shift_cooldown = 0.0
        self.clutch = 0.0

    def update(self, veh: Vehicle, throttle: float, dt: float):
        self.shift_cooldown = max(0.0, self.shift_cooldown - dt)
        up = down = False
        rpm = veh.rpm
        if self.shift_cooldown == 0.0:
            if rpm > self.spec.redline_rpm * 0.95 and veh.gear < len(self.spec.gear_ratios):
                up = True
                self.shift_cooldown = 0.4
            elif rpm < 1500 and veh.gear > 1 and veh.speed > 1.0:
                down = True
                self.shift_cooldown = 0.4

        idle_thresh = self.spec.idle_rpm * 1.25
        if up or down:
            target = 0.2
        elif throttle < 0.05 and (veh.speed < 1.5 or rpm < idle_thresh):
            target = 0.0
        else:
            target = 1.0
        self.clutch += (target - self.clutch) * min(1.0, dt * 12.0)
        return self.clutch, up, down


@dataclass
class DriverInput:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    handbrake: float = 0.0
    clutch: float = 1.0
    shift_up: bool = False
    shift_down: bool = False
    auto: bool = True

    def update_from_payload(self, data: dict):
        self.steer = float(np.clip(data.get("steer", self.steer), -1.0, 1.0))
        self.throttle = float(np.clip(data.get("throttle", self.throttle), 0.0, 1.0))
        self.brake = float(np.clip(data.get("brake", self.brake), 0.0, 1.0))
        self.handbrake = float(np.clip(data.get("handbrake", self.handbrake), 0.0, 1.0))
        self.clutch = float(np.clip(data.get("clutch", self.clutch), 0.0, 1.0))
        self.auto = bool(data.get("auto", self.auto))
        self.shift_up = self.shift_up or bool(data.get("shiftUp", False))
        self.shift_down = self.shift_down or bool(data.get("shiftDown", False))


@dataclass
class SimSession:
    car: str = "supra"
    track_name: str = "club"
    seed: int = 7
    style: str | None = None
    difficulty: float = 0.5
    length: float | None = None
    spec: CarSpec = field(init=False)
    sim: SimSpec = field(init=False)
    veh: Vehicle = field(init=False)
    trk: object = field(init=False)
    track_meta: dict = field(init=False)
    gearbox: AutoBox3D = field(init=False)
    driver: DriverInput = field(default_factory=DriverInput)
    agent: object = field(default=None)
    sensors: object = field(default=None)
    ctrl_ctr: int = 0
    ctrl_period: int = 4
    cur_act: tuple = (0.0, 0.0, 0.0, 0.0)
    checkpoint: str | None = None
    t_in: float = 0.0
    b_in: float = 0.0
    wheel_spin: float = 0.0
    sim_time: float = 0.0
    _accum: float = 0.0
    _last_wall: float = field(default_factory=time.perf_counter)

    def __post_init__(self):
        self.configure(self.car, self.track_name, self.seed, self.style,
                       self.difficulty, self.length)

    def configure(self, car: str = "supra", track_name: str = "club", seed: int = 7,
                  style: str | None = None, difficulty: float = 0.5,
                  length: float | None = None, checkpoint: str | None = None):
        self.car = car or "supra"
        self.track_name = track_name or "club"
        self.seed = int(seed or 7)
        self.style = style
        self.difficulty = float(difficulty)
        self.length = length
        self.spec = get_car(self.car)
        self.sim = SimSpec()
        self.trk, self.track_meta = resolve_track(self.track_name, self.seed, self.style,
                                                  self.difficulty, self.length)
        self.veh = Vehicle(self.spec, self.sim)
        self.gearbox = AutoBox3D(self.spec)
        self.driver = DriverInput()
        self.checkpoint = checkpoint
        self.agent = None
        self.sensors = None
        self.agent_error = None
        self.agent_meta = None
        self._agent_path = None
        self._agent_mtime = 0.0
        self._agent_poll_wall = 0.0
        if checkpoint:
            try:
                self._load_agent(checkpoint)
                self.ctrl_period = max(1, round(1.0 / (30 * self.sim.dt)))
                self.ctrl_ctr = 0
                self.cur_act = (0.0, 0.0, 0.0, 0.0)
            except Exception as e:
                # surfaced to the client as a toast (pre-hills checkpoints land
                # here with the loud retrain message) — manual driving continues
                self.agent_error = str(e)
                print(f"Failed to load agent from {checkpoint}: {e}")
        self.reset()

    def _load_agent(self, checkpoint: str):
        """(Re)load a PPO policy through the loud-guarded loader. Raises on
        any incompatibility (e.g. pre-hills obs) — caller decides how to
        surface it."""
        import os
        from supra.aiviz import PolicyAgent
        from supra.ppo import PPO
        from supra.sensors import SensorSuite
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, os.path.basename(checkpoint))
        net, norm, d = PPO.load_policy(path)
        self.agent = PolicyAgent(net, norm, d, mode=d.get("mode", "drift"))
        if self.sensors is None:
            self.sensors = SensorSuite()
        self._agent_path = path
        self._agent_mtime = os.path.getmtime(path)
        self.agent_meta = {
            "name": os.path.basename(path),
            "mode": str(d.get("mode", "ppo")),
            "updates": int(d.get("updates", 0)),
            "metric": float(d.get("metric", 0.0) or 0.0),
        }

    def maybe_reload_agent(self, min_interval: float = 2.0) -> bool:
        """Hot-reload the policy when the checkpoint file changes on disk —
        this is what lets the viewer WATCH A TRAINING RUN LIVE: the trainer
        rewrites the rolling checkpoint every iteration (atomic rename), the
        session swaps the brain in mid-drive, the car keeps rolling. Returns
        True when a reload happened. Throttled by wall clock; load failures
        (e.g. mid-replace stat races) keep the old brain and retry later."""
        import os
        if not self.agent or not self._agent_path:
            return False
        now = time.perf_counter()
        if now - self._agent_poll_wall < min_interval:
            return False
        self._agent_poll_wall = now
        try:
            mtime = os.path.getmtime(self._agent_path)
            if mtime <= self._agent_mtime:
                return False
            self._load_agent(self._agent_path)
            return True
        except Exception:
            return False

    def reset(self):
        sx, sy, syaw = self.trk.start_pose()
        start_speed = 0.0
        if self.agent and getattr(self.agent, "mode", None) in ("drift", "hybrid"):
            k = abs(float(self.trk.curvature[0]))
            start_speed = (min(30.0, (11.0 / k) ** 0.5) * 0.85) if k > 1e-4 else 25.0
            
        self.veh.reset(sx, sy, syaw, speed=start_speed)
        self.gearbox = AutoBox3D(self.spec)
        self.t_in = 0.0
        self.b_in = 0.0
        self.wheel_spin = 0.0
        self.sim_time = 0.0
        self._accum = 0.0
        self._last_wall = time.perf_counter()
        if getattr(self, "agent", None):
            self.ctrl_ctr = 0
            self.cur_act = (0.0, 0.0, 0.0, 0.0)

    def tick_wall(self, max_catchup: float = 0.12):
        now = time.perf_counter()
        dt = min(max(0.0, now - self._last_wall), max_catchup)
        self._last_wall = now
        return self.step_elapsed(dt)

    def step_elapsed(self, elapsed: float):
        self._accum += elapsed
        steps = 0
        max_steps = int(0.16 / self.sim.dt) + 2
        while self._accum >= self.sim.dt and steps < max_steps:
            self._step_fixed(self.sim.dt)
            self._accum -= self.sim.dt
            steps += 1
        if steps >= max_steps:
            self._accum = 0.0
        return steps

    def _step_fixed(self, dt: float):
        if self.agent and self.sensors:
            if self.ctrl_ctr % self.ctrl_period == 0:
                cobs = self.sensors.observe(self.veh, self.trk)
                out = self.agent.act(self.veh, cobs)
                self.cur_act = (float(out[0]), float(out[1]), float(out[2]), float(out[3]) if len(out) > 3 else 0.0)
            self.ctrl_ctr += 1
            steer, self.t_in, self.b_in, handbrake = self.cur_act
            clutch, up, down = self.gearbox.update(self.veh, self.t_in, dt)
        else:
            self.t_in += float(np.clip(self.driver.throttle - self.t_in, -1.0, 1.0)) * min(1.0, dt * 5.0)
            self.b_in += float(np.clip(self.driver.brake - self.b_in, -1.0, 1.0)) * min(1.0, dt * 8.0)
            steer = self.driver.steer
            handbrake = self.driver.handbrake
            if self.driver.auto:
                clutch, up, down = self.gearbox.update(self.veh, self.t_in, dt)
            else:
                clutch = self.driver.clutch
                up, down = self.driver.shift_up, self.driver.shift_down

        fr = self.trk.frame(self.veh.x, self.veh.y)
        self.veh.surface_grip = self.spec.offtrack_grip if fr["off_track"] else 1.0
        self.veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"],
                          fr["vcurv"])
        self.veh.step(Controls(steer=steer, throttle=self.t_in, brake=self.b_in,
                               clutch=clutch, handbrake=handbrake,
                               shift_up=up, shift_down=down), dt=dt)
        if not self.agent:
            self.driver.shift_up = False
            self.driver.shift_down = False
        self.wheel_spin += float(np.mean(self.veh.wheel_w)) * dt
        self.sim_time += dt

        if self.agent and abs(fr["lateral"]) > self.trk.half + 12.0:
            self.reset()

    def track_payload(self, stride: int = 2) -> dict:
        idx = np.arange(0, len(self.trk.center), stride)
        if idx[-1] != len(self.trk.center) - 1:
            idx = np.append(idx, len(self.trk.center) - 1)
        left, right, center, bank = [], [], [], []
        for i in idx:
            z = float(self.trk.z[i])           # real, physics-authoritative
            left.append([float(self.trk.left[i, 0]), float(self.trk.left[i, 1]), z])
            right.append([float(self.trk.right[i, 0]), float(self.trk.right[i, 1]), z])
            center.append([float(self.trk.center[i, 0]), float(self.trk.center[i, 1]), z])
            bank.append(float(self.trk.bank[i]))
        return {
            "type": "track",
            "meta": {
                **self.track_meta,
                "length": float(self.trk.length),
                "width": float(self.trk.width),
                "car": self.car,
                "elevGain": float(self.trk.elev_gain),
            },
            "left": left,
            "right": right,
            "center": center,
            "bank": bank,
        }

    def state_payload(self) -> dict:
        fr = self.trk.frame(self.veh.x, self.veh.y)
        steer_out = self.cur_act[0] if self.agent else self.driver.steer
        hb_out = self.cur_act[3] if self.agent else self.driver.handbrake
        return {
            "type": "state",
            "time": self.sim_time,
            "car": self.car,
            "pose": {
                "x": self.veh.x,
                "y": self.veh.y,
                "z": self.veh.z,        # CAR z (not road z) — jumps render
                "yaw": self.veh.yaw,
                "pitch": self.veh.pitch,
                "roll": self.veh.roll,
            },
            "vehicle": {
                "speed": self.veh.speed,
                "speedKmh": self.veh.speed * 3.6,
                "rpm": self.veh.rpm,
                "gear": self.veh.gear,
                "steerAngle": self.veh.steer_angle,
                "wheelSpin": self.wheel_spin,
                "wheelRadius": self.spec.wheel_radius,
                "slipAngle": self.veh.slip_angle,
                "airborne": bool(self.veh.airborne),
                "airTime": float(self.veh.air_time),
                "landingG": float(self.veh.landing_g),
            },
            "wheels": {
                "slip": [float(v) for v in self.veh.wheel_slip],
                "slipRatio": [float(v) for v in self.veh.wheel_sr],
                "grip": [float(v) for v in self.veh.wheel_grip],
            },
            "controls": {
                "steer": steer_out,
                "throttle": self.t_in,
                "brake": self.b_in,
                "handbrake": hb_out,
                "auto": self.driver.auto,
            },
            "track": {
                "progress": fr["progress"],
                "arc": fr["arc"],
                "lateral": fr["lateral"],
                "offTrack": fr["off_track"],
                "curvature": fr["curvature"],
            },
            **({"agent": self.agent_meta}
               if getattr(self, "agent_meta", None) and self.agent else {}),
        }

