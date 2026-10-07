"""Fresh race/drift environment built on a frozen copy of the project's physics.

60 bounded sensor inputs; tanh actions = steering, signed throttle/brake,
signed handbrake (positive only). 120 Hz physics, 30 Hz control.
"""
import math
import numpy as np
from .config import Config
from .core.config import SimSpec, get_car
from .core.physics import Vehicle, Controls
from .core.sensors import SensorSuite
from .core.track import named_track, configure_hills


def wrap(x):
    return (x + math.pi) % (2 * math.pi) - math.pi


class Gearbox:
    def __init__(self, spec):
        self.spec, self.cooldown, self.clutch = spec, 0., 0.

    def update(self, car, throttle, dt):
        self.cooldown = max(0., self.cooldown - dt)
        up = self.cooldown == 0 and car.rpm > self.spec.redline_rpm * .95 and car.gear < len(self.spec.gear_ratios)
        down = self.cooldown == 0 and car.rpm < 1500 and car.gear > 1 and car.speed > 1
        if up or down:
            self.cooldown = .4
        target = .2 if up or down else (0. if throttle < .05 and (car.speed < 1.5 or car.rpm < self.spec.idle_rpm * 1.25) else 1.)
        self.clutch += (target - self.clutch) * min(1., dt * 12)
        return self.clutch, up, down


class Environment:
    obs_dim, act_dim = 60, 3

    def __init__(self, config=None, seed=None):
        self.cfg = config or Config()
        self.rng = np.random.default_rng(self.cfg.seed if seed is None else seed)
        configure_hills(enabled=self.cfg.hills)
        self.track = named_track(self.cfg.track)
        self.car = Vehicle(get_car(self.cfg.car), SimSpec())
        self.sensors = SensorSuite()
        self.style_scale = 1.
        self.reset()

    def reset(self, random_start=False, index=0, speed=0.):
        if random_start:
            index = int(self.rng.integers(len(self.track.center)))
            speed = float(self.rng.uniform(4, 20))
        x, y, yaw = self.track.pose_at(index)
        if random_start:
            yaw += float(self.rng.uniform(-.12, .12))
        self.car.reset(x, y, yaw, speed)
        self.box = Gearbox(self.car.spec)
        fr = self.track.frame(x, y)
        self.car.set_road(fr['grade'], fr['bank'], fr['heading'], fr['z'], fr['vcurv'])
        self.previous_progress = self._progress(fr)
        self.elapsed = self.progress = self.return_ = self.off_time = self.stall_time = 0.
        self.drift_steps = self.steps = self.off_steps = 0
        self.previous_action = np.zeros(3)
        self.parts = dict(progress=0., pace=0., style=0., penalty=0.)
        self.reason = ''
        return self.observe()

    def _progress(self, fr):
        # Project onto the local tangent: avoids quantized nearest-point rewards.
        i = fr['index']
        delta = np.array([self.car.x, self.car.y]) - self.track.center[i]
        return (float(self.track.arc[i]) + float(delta @ self.track.tangent[i])) / self.track.length

    def observe(self):
        self.observation = self.sensors.observe(self.car, self.track)
        goal = {'race': [1, 0], 'drift': [0, 1], 'hybrid': [1, 1]}[self.cfg.mode]
        obs = np.concatenate((self.observation.vector, goal)).astype(np.float32)
        if not np.isfinite(obs).all():
            raise FloatingPointError('Non-finite simulation observation.')
        return np.clip(obs, -5, 5)

    def step(self, action):
        a = np.asarray(action, dtype=float)
        if a.shape != (3,) or not np.isfinite(a).all():
            raise ValueError('Expected three finite control values.')
        a = np.clip(a, -1, 1)
        steer, longitudinal, hb = a
        throttle, brake = max(0., longitudinal), max(0., -longitudinal)
        parts = dict(progress=0., pace=0., style=0., penalty=-.002 * float(np.square(a-self.previous_action).sum()))
        terminated = False
        drift = False
        for _ in range(4):
            dt = 1 / 120
            fr = self.track.frame(self.car.x, self.car.y)
            self.car.surface_grip = self.car.spec.offtrack_grip if fr['off_track'] else 1.
            self.car.set_road(fr['grade'], fr['bank'], fr['heading'], fr['z'], fr['vcurv'])
            clutch, up, down = self.box.update(self.car, throttle, dt)
            self.car.step(Controls(steer=steer, throttle=throttle, brake=brake,
                                  handbrake=max(0., hb), clutch=clutch, shift_up=up, shift_down=down))
            fr = self.track.frame(self.car.x, self.car.y)
            p = self._progress(fr)
            dp = (p - self.previous_progress + .5) % 1 - .5
            self.previous_progress = p
            before = self.progress
            self.progress += dp
            self.elapsed += dt
            off = fr['off_track']
            self.off_time = self.off_time + dt if off else 0.
            self.stall_time = self.stall_time + dt if self.elapsed > 2 and self.car.speed < 1 else 0.
            slip = abs(math.degrees(self.car.slip_angle))
            corner = float(np.clip((abs(fr['curvature']) - .012) / .012, 0, 1))
            corner = corner * corner * (3-2*corner)
            entry = float(np.clip((slip-3)/13, 0, 1))
            entry = entry * entry * (3-2*entry)
            rotation = float(np.clip((90-slip)/40, 0, 1))
            spin = float(np.clip(1-(abs(math.degrees(self.car.r))-160)/140, 0, 1))
            on = not off and not self.car.airborne
            drift = bool(on and slip > 6 and slip < 90 and self.car.speed > 5 and spin > 0 and
                         (corner > 0 or self.cfg.mode == 'drift'))
            if on:
                heading = wrap(self.car.yaw-fr['heading'])
                parts['progress'] += dp * (60 if self.cfg.mode != 'drift' else 30)
                parts['pace'] += .06 * math.cos(heading) * self.car.speed / 80 + .008 * self.car.speed / 80
                straight = float(np.clip(1-abs(fr['curvature'])/.018, 0, 1))
                parts['pace'] += straight * max(0., math.cos(heading)) * (.06*(self.car.speed/82)**2+.02*throttle)
                if self.cfg.mode != 'race':
                    gate = corner if self.cfg.mode == 'hybrid' else 1.
                    parts['style'] += self.cfg.style_weight * self.style_scale * self.car.speed * abs(math.sin(self.car.slip_angle)) * entry * rotation * spin * gate * min(1., self.car.speed/12.5)
                if int(max(0, self.progress)) > int(max(0, before)):
                    parts['progress'] += 5
            else:
                parts['penalty'] -= .15 if off else .01
            if self.off_time > .8 or abs(fr['lateral']) > fr['half_width'] + 5:
                self.reason, terminated = 'Off track', True
            elif self.stall_time > 3:
                self.reason, terminated = 'Stalled', True
            elif self.progress < -.08:
                self.reason, terminated = 'Wrong direction', True
            if terminated:
                parts['penalty'] -= 3
                break
        self.previous_action = a
        self.steps += 1
        self.drift_steps += int(drift)
        self.off_steps += int(off)
        reward = sum(parts.values())
        self.return_ += reward
        self.parts = parts
        truncated = self.elapsed >= self.cfg.episode_seconds and not terminated
        if truncated:
            self.reason = 'Time limit'
        info = dict(return_=self.return_, progress=self.progress, drift=self.drift_steps/self.steps,
                    offtrack=self.off_steps/self.steps, seconds=self.elapsed, reason=self.reason,
                    speed=self.car.speed*3.6, reward_parts=parts)
        return self.observe(), float(reward), terminated, truncated, info

    def baseline_action(self):
        """Explicitly labelled reference driver, never passed off as trained PPO."""
        fr = self.track.frame(self.car.x, self.car.y)
        look = max(8., self.car.speed*.8)
        arc = (fr['arc'] + look) % self.track.length
        i = int(np.searchsorted(self.track.arc, arc)) % len(self.track.center)
        target = self.track.center[i]
        angle = wrap(math.atan2(target[1]-self.car.y, target[0]-self.car.x)-self.car.yaw)
        steer = math.atan2(2*self.car.spec.wheelbase*math.sin(angle), look) / self.car.max_steer_angle
        curvature = max(.001, float(np.max(np.abs(self.track.lookahead_curvature(fr['arc'], (8, 20, 40))))))
        target_speed = min(32., math.sqrt(5./curvature))
        return np.array([np.clip(steer, -1, 1), np.clip((target_speed-self.car.speed)*.18, -1, .8), -1.])

    def telemetry(self):
        c = self.car
        fr = self.observation.frame
        return dict(x=c.x, y=c.y, z=c.z, yaw=c.yaw, pitch=c.pitch, roll=c.roll,
                    speed=c.speed*3.6, rpm=c.rpm, gear=c.gear, slip=math.degrees(c.slip_angle),
                    steer=float(self.previous_action[0]), throttle=max(0., float(self.previous_action[1])),
                    brake=max(0., -float(self.previous_action[1])), handbrake=max(0., float(self.previous_action[2])),
                    elapsed=self.elapsed, progress=self.progress, reward=self.return_, parts=self.parts,
                    offtrack=bool(fr['off_track']), airborne=bool(c.airborne),
                    drift=self.drift_steps/max(1, self.steps), reason=self.reason,
                    beams=self.observation.beam_points.tolist())

    def geometry(self):
        t = self.track
        return dict(name=self.cfg.track, length=t.length, width=t.width,
                    center=t.center.tolist(), left=t.left.tolist(), right=t.right.tolist(),
                    elevation=t.z.tolist(), bank=t.bank.tolist())
