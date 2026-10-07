"""Mazda 787B procedural audio engine v4.

The runtime contract deliberately lives outside vehicle physics and learning
observations.  Viewers publish versioned snapshots; the audio subprocess owns
all DSP state.  No recording is loaded at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import multiprocessing as mp
import os
import time
from typing import Sequence

import numpy as np
from scipy.signal import lfilter

SCHEMA_VERSION = 5
SPEED_OF_SOUND = 343.0
MAX_AUDIBLE_RADIUS_M = 350.0
PREFERRED_BLOCK = 2048
HEADER = 18
CAR_STRIDE = 64
RATIOS_787B = (3.27451, 2.28028, 1.57271, 1.16654, 0.96318)
SURFACES = {"dry": 0, "wet": 1, "grass": 2, "dirt": 3, "kerb": 4, "gravel": 5}


def _f(value, default=0.0, lo=-1e9, hi=1e9):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return float(np.clip(value if np.isfinite(value) else default, lo, hi))


@dataclass
class AudioEvents:
    reset: int = 0
    shift: int = 0
    landing: int = 0
    collision: int = 0
    damage: int = 0
    surface: int = 0


@dataclass
class AudioFrame:
    """One neutral-default, versioned vehicle audio snapshot."""
    version: int = SCHEMA_VERSION
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    speed: float = 0.0
    rpm: float = 850.0
    throttle: float = 0.0
    brake: float = 0.0
    clutch: float = 1.0
    gear: int = 1
    ratio: float = RATIOS_787B[0]
    limiter: float = 0.0
    wheel_slip: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    wheel_load: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    surface: int = 0
    wetness: float = 0.0
    airborne: float = 0.0
    landing_force: float = 0.0
    damage: float = 0.0
    collision_impulse: float = 0.0
    shift_mismatch: float = 0.0
    torque_direction: float = 1.0
    kerb: float = 0.0
    engine_state: int = 2
    engine_temperature: float = 0.5
    wheel_surface: tuple[int, ...] = (0, 0, 0, 0)
    wheel_contact: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    brake_temperature: tuple[float, ...] = (0.2, 0.2, 0.2, 0.2)
    suspension_travel: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    suspension_velocity: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
    impact_corner: int = -1
    shift_phase: float = 0.0
    acoustic_preset: int = 0
    events: AudioEvents = field(default_factory=AudioEvents)


class AudioTelemetryAdapter:
    """Derives sound telemetry without touching physics or observations."""
    def __init__(self):
        self.events: dict[int, AudioEvents] = {}
        self.previous: dict[int, dict[str, float]] = {}

    def reset(self, vehicles: Sequence[object]):
        for v in vehicles:
            event = self.events.setdefault(id(v), AudioEvents())
            event.reset += 1
            self.previous.pop(id(v), None)

    def frame(self, vehicle, throttle=0.0, brake=None, clutch=None) -> AudioFrame:
        key = id(vehicle)
        ev = self.events.setdefault(key, AudioEvents())
        prev = self.previous.setdefault(key, {})
        spec = getattr(vehicle, "spec", None)
        ratios = getattr(spec, "gear_ratios", RATIOS_787B) or RATIOS_787B
        gear = int(_f(getattr(vehicle, "gear", 1), 1, 1, len(ratios)))
        speed = _f(getattr(vehicle, "speed", 0), 0, 0, 200)
        yaw = _f(getattr(vehicle, "yaw", 0))
        rpm = _f(getattr(vehicle, "rpm", 850), 850, 0, 12000)
        surface_name = str(getattr(vehicle, "_audio_surface", "dry")).lower()
        surface = SURFACES.get(surface_name, int(_f(getattr(vehicle, "_audio_surface_id", 0), 0, 0, 5)))
        damage = _f(getattr(vehicle, "damage", getattr(vehicle, "_audio_damage", 0)), 0, 0, 1)
        landing = _f(getattr(vehicle, "_audio_landing_force", 0), 0, 0, 8)
        collision = _f(getattr(vehicle, "_audio_collision_impulse", 0), 0, 0, 20)
        if prev:
            if gear != int(prev["gear"]): ev.shift += 1
            if landing > 0 and prev["landing"] <= 0: ev.landing += 1
            if collision > 0 and prev["collision"] <= 0: ev.collision += 1
            if damage > prev["damage"] + 1e-5: ev.damage += 1
            if surface != int(prev["surface"]): ev.surface += 1
        wheel_slip = list(getattr(vehicle, "wheel_sr", (0, 0, 0, 0)))[:4]
        wheel_slip += [0.0] * (4 - len(wheel_slip))
        loads = list(getattr(vehicle, "wheel_load", getattr(vehicle, "_audio_wheel_load", (1, 1, 1, 1))))[:4]
        loads += [1.0] * (4 - len(loads))
        wheel_surface = list(getattr(vehicle, "_audio_wheel_surface", (surface,) * 4))[:4]
        wheel_surface += [surface] * (4 - len(wheel_surface))
        contact = list(getattr(vehicle, "_audio_wheel_contact", (1, 1, 1, 1)))[:4]
        contact += [1.0] * (4 - len(contact))
        brake_temp = list(getattr(vehicle, "_audio_brake_temperature", (0.2,) * 4))[:4]
        brake_temp += [0.2] * (4 - len(brake_temp))
        susp_travel = list(getattr(vehicle, "_audio_suspension_travel", (0,) * 4))[:4]
        susp_travel += [0.0] * (4 - len(susp_travel))
        susp_velocity = list(getattr(vehicle, "_audio_suspension_velocity", (0,) * 4))[:4]
        susp_velocity += [0.0] * (4 - len(susp_velocity))
        brake = _f(getattr(vehicle, "_audio_brake", 0) if brake is None else brake, 0, 0, 1)
        clutch = _f(getattr(vehicle, "_audio_clutch", 1) if clutch is None else clutch, 1, 0, 1)
        redline = _f(getattr(spec, "redline_rpm", 9000), 9000, 1000, 14000)
        vx = _f(getattr(vehicle, "vx", speed * math.cos(yaw)))
        vy = _f(getattr(vehicle, "vy", speed * math.sin(yaw)))
        mismatch = _f(getattr(vehicle, "_audio_shift_mismatch", 0), 0, -1, 1)
        torque = _f(getattr(vehicle, "_audio_torque_direction", 2 * _f(throttle, 0, 0, 1) - 0.4), 0, -1, 1)
        frame = AudioFrame(
            x=_f(getattr(vehicle, "x", 0)), y=_f(getattr(vehicle, "y", 0)), yaw=yaw,
            vx=vx, vy=vy, speed=speed, rpm=rpm, throttle=_f(throttle, 0, 0, 1),
            brake=brake, clutch=clutch, gear=gear, ratio=_f(ratios[gear-1], RATIOS_787B[gear-1], .1, 8),
            limiter=float(bool(getattr(vehicle, "limiter", False)) or rpm >= redline),
            wheel_slip=tuple(_f(x, 0, -5, 5) for x in wheel_slip),
            wheel_load=tuple(_f(x, 1, 0, 5) for x in loads), surface=surface,
            wetness=_f(getattr(vehicle, "_audio_wetness", 0), 0, 0, 1),
            airborne=_f(getattr(vehicle, "airborne", getattr(vehicle, "_audio_airborne", 0)), 0, 0, 1),
            landing_force=landing, damage=damage, collision_impulse=collision,
            shift_mismatch=mismatch, torque_direction=torque,
            kerb=_f(getattr(vehicle, "_snd_kerb", 0), 0, 0, 1),
            engine_state=int(_f(getattr(vehicle, "_audio_engine_state", 2), 2, 0, 3)),
            engine_temperature=_f(getattr(vehicle, "_audio_engine_temperature", .5), .5, 0, 1),
            wheel_surface=tuple(int(_f(x, surface, 0, 5)) for x in wheel_surface),
            wheel_contact=tuple(_f(x, 1, 0, 1) for x in contact),
            brake_temperature=tuple(_f(x, .2, 0, 1) for x in brake_temp),
            suspension_travel=tuple(_f(x, 0, -1, 1) for x in susp_travel),
            suspension_velocity=tuple(_f(x, 0, -10, 10) for x in susp_velocity),
            impact_corner=int(_f(getattr(vehicle, "_audio_impact_corner", -1), -1, -1, 3)),
            shift_phase=_f(getattr(vehicle, "_audio_shift_phase", 0), 0, 0, 1),
            acoustic_preset=int(_f(getattr(vehicle, "_audio_acoustic_preset", 0), 0, 0, 2)),
            events=AudioEvents(**vars(ev)))
        prev.update(gear=gear, landing=landing, collision=collision, damage=damage, surface=surface)
        return frame


def _encode(frame: AudioFrame) -> list[float]:
    return [frame.x, frame.y, frame.yaw, frame.vx, frame.vy, frame.speed, frame.rpm,
            frame.throttle, frame.brake, frame.clutch, frame.gear, frame.ratio, frame.limiter,
            *frame.wheel_slip, *frame.wheel_load, frame.surface, frame.wetness, frame.airborne,
            frame.landing_force, frame.damage, frame.collision_impulse, frame.shift_mismatch,
            frame.torque_direction, frame.kerb, frame.events.reset, frame.events.shift,
            frame.events.landing, frame.events.collision, frame.events.damage, frame.events.surface,
            frame.engine_state, frame.engine_temperature, *frame.wheel_surface,
            *frame.wheel_contact, *frame.brake_temperature, *frame.suspension_travel,
            *frame.suspension_velocity, frame.impact_corner, frame.shift_phase,
            frame.acoustic_preset, 0, 0, 0]


def _decode(values) -> AudioFrame:
    v = list(values)
    return AudioFrame(x=v[0], y=v[1], yaw=v[2], vx=v[3], vy=v[4], speed=v[5], rpm=v[6],
        throttle=v[7], brake=v[8], clutch=v[9], gear=int(v[10]), ratio=v[11], limiter=v[12],
        wheel_slip=tuple(v[13:17]), wheel_load=tuple(v[17:21]), surface=int(v[21]), wetness=v[22],
        airborne=v[23], landing_force=v[24], damage=v[25], collision_impulse=v[26],
        shift_mismatch=v[27], torque_direction=v[28], kerb=v[29], engine_state=int(v[36]),
        engine_temperature=v[37], wheel_surface=tuple(map(int, v[38:42])),
        wheel_contact=tuple(v[42:46]), brake_temperature=tuple(v[46:50]),
        suspension_travel=tuple(v[50:54]), suspension_velocity=tuple(v[54:58]),
        impact_corner=int(v[58]), shift_phase=v[59], acoustic_preset=int(v[60]),
        events=AudioEvents(*map(int, v[30:36])))


class DCBlocker:
    def __init__(self, channels=2, pole=.995):
        self.pole, self.x1, self.y1 = pole, np.zeros(channels), np.zeros(channels)
    def process(self, x):
        for i in range(len(x)):
            current = x[i].copy()
            self.y1 = current - self.x1 + self.pole * self.y1
            self.x1 = current
            x[i] = self.y1
        return x


class SoftLimiter:
    def __init__(self, threshold=.50):
        self.threshold, self.gain, self.reduction_db = threshold, 1.0, 0.0
    def process(self, x):
        peak = float(np.max(np.abs(x), initial=0))
        target = min(1.0, self.threshold / max(peak, 1e-9))
        self.gain = min(self.gain + .004, target) if target > self.gain else .65*self.gain + .35*target
        self.reduction_db = -20*math.log10(max(self.gain, 1e-8))
        x *= self.gain
        np.tanh(x / .82, out=x); x *= .82
        np.clip(x, -.98, .98, out=x)
        return x


class FractionalDelay:
    """Phase-continuous propagation delay; distance motion creates Doppler."""
    def __init__(self, sample_rate, max_distance=MAX_AUDIBLE_RADIUS_M):
        self.sr = sample_rate
        self.buf = np.zeros(int(sample_rate * max_distance / SPEED_OF_SOUND) + 8192, np.float32)
        self.write = 0
        self.delay = 0.0
        self.initialized = False
    def reset(self, distance=0):
        self.buf.fill(0); self.write = 0; self.delay = distance / SPEED_OF_SOUND * self.sr
        self.initialized = distance > 0
    def process(self, source, distance):
        n = len(source); out = np.empty(n, np.float32)
        target = min(len(self.buf)-4, max(0.0, distance / SPEED_OF_SOUND * self.sr))
        if not self.initialized:
            self.delay = target
            self.initialized = True
        delays = np.linspace(self.delay, target, n, endpoint=False)
        size = len(self.buf)
        write_idx = (self.write + np.arange(n)) % size
        self.buf[write_idx] = source
        pos = (write_idx - delays) % size
        p0 = pos.astype(np.int64); frac = pos-p0
        out[:] = self.buf[p0]*(1-frac) + self.buf[(p0+1)%size]*frac
        self.write = (self.write+n) % size
        self.delay = target
        return out


class RadialDoppler:
    """Smoothed fallback used only when propagation delay is disabled."""
    def __init__(self):
        self.ratio=1.0; self.phase=0.0
    def reset(self): self.ratio=1.0; self.phase=0.0
    def process(self, source, radial_velocity):
        target=float(np.clip(SPEED_OF_SOUND/(SPEED_OF_SOUND+radial_velocity),.72,1.35))
        self.ratio += .12*(target-self.ratio)
        idx=self.phase+np.arange(len(source))*self.ratio
        out=np.interp(idx,np.arange(len(source)),source,left=source[0],right=source[-1]).astype(np.float32)
        self.phase=float((idx[-1]+self.ratio)-len(source))
        self.phase=float(np.clip(self.phase,-2,2))
        return out


class EarlyReflections:
    """Small stateful ground/venue reflection network."""
    def __init__(self,sample_rate):
        self.sr=sample_rate; self.size=max(4096,int(.09*sample_rate)); self.buf=np.zeros(self.size,np.float32); self.write=0
    def reset(self): self.buf.fill(0); self.write=0
    def process(self,source,preset):
        delays=((.006,.013),(.011,.027),(.018,.043))[min(max(int(preset),0),2)]
        gains=((.16,.07),(.22,.12),(.28,.16))[min(max(int(preset),0),2)]
        idx=(self.write+np.arange(len(source)))%self.size; self.buf[idx]=source
        out=np.zeros(len(source),np.float32)
        for delay,gain in zip(delays,gains): out += self.buf[(idx-int(delay*self.sr))%self.size]*gain
        self.write=(self.write+len(source))%self.size
        return out


class EngineSynthV4:
    """Deterministic, stem-producing R26B procedural model."""
    STEMS = ("exhaust", "intake", "mechanical", "accessories", "transmission",
             "tyre_fl", "tyre_fr", "tyre_rl", "tyre_rr", "brakes", "chassis", "wind")
    def __init__(self, sample_rate=48000, seed=78755):
        self.sr, self.rng = int(sample_rate), np.random.default_rng(seed)
        self.phase = np.zeros(12); self.clock = 0; self.prev = AudioFrame()
        self.seen = AudioEvents(); self.env = dict(shift=0., landing=0., collision=0., damage=0.)
        self.runner_state = 0
        self.lifecycle_env = 0.0
        self.brake_hot = np.zeros(4, dtype=bool)
    def reset(self, frame=None):
        self.phase.fill(0); self.env = {k:0. for k in self.env}; self.prev = frame or AudioFrame()
        self.runner_state = 0; self.lifecycle_env = 0.0; self.brake_hot.fill(False)
        self.seen = AudioEvents(**vars(self.prev.events))
    def synthesize_stems(self, frame: AudioFrame, n: int):
        t = np.arange(n, dtype=np.float64)/self.sr
        rpm = max(0., frame.rpm); effective_rpm=max(300.,rpm); fundamental = effective_rpm/60.*4.0
        cut = 1.0 if not frame.limiter else ((self.clock//max(1,int(.012*self.sr))) % 3 != 1)
        harmonics = np.zeros(n)
        rpm_f=np.clip(effective_rpm/9000,0,1); load=np.clip(frame.throttle,0,1)
        harmonic_curve=(1.0, .52+.18*load, .30+.20*rpm_f, .20+.15*load,
                        .12+.13*rpm_f, .07+.10*load)
        for h, amp in enumerate(harmonic_curve, 1):
            p = self.phase[0] + 2*np.pi*fundamental*h*t
            harmonics += amp*np.sin(p + .07*h*h)
        self.phase[0] = (self.phase[0] + 2*np.pi*fundamental*n/self.sr) % (2*np.pi)
        combustion_load = .18 + .82*load
        exhaust = np.tanh((harmonics+.16*harmonics**3)*(1.0+1.9*combustion_load))*(.10+.25*combustion_load)*cut
        if frame.throttle < .12 and rpm > 3000:
            miss=((self.clock+np.arange(n))//max(1,int(self.sr*.031)))%5==0
            exhaust += self.rng.standard_normal(n)*(.010+.014*frame.engine_temperature)*(rpm/9000)*miss
        # Three-state variable runners with hysteresis.
        if self.runner_state==0 and rpm>5700: self.runner_state=1
        elif self.runner_state==1 and rpm<5300: self.runner_state=0
        elif self.runner_state==1 and rpm>7900: self.runner_state=2
        elif self.runner_state==2 and rpm<7500: self.runner_state=1
        runner=(0.0,.48,1.0)[self.runner_state]
        formant = 520 + 1180*runner + 260*load
        p = self.phase[1] + 2*np.pi*formant*t
        intake = np.sin(p)*(frame.throttle**1.5)*(.035+.065*runner)
        self.phase[1] = (self.phase[1]+2*np.pi*formant*n/self.sr)%(2*np.pi)
        rotor_order=effective_rpm/60
        beat=.82+.18*np.sin(2*np.pi*3.7*t+self.phase[5])
        mechanical = (np.sin(2*np.pi*rotor_order*t+self.phase[2])*.017*beat
                      +np.sin(2*np.pi*rotor_order*4*t+self.phase[6])*.007
                      +self.rng.standard_normal(n)*.0045)*(effective_rpm/9000)
        self.phase[2] = (self.phase[2]+2*np.pi*(rpm/60)*n/self.sr)%(2*np.pi)
        accessories=(np.sin(2*np.pi*rotor_order*1.5*t+self.phase[7])*.004
                     +np.sin(2*np.pi*rotor_order*2.5*t+self.phase[8])*.003)*(effective_rpm/9000)
        shaft = effective_rpm/60; output = shaft/max(frame.ratio, .1); final_drive=4.357
        diff=output/final_drive; mesh_in=shaft*17; mesh_out=output*23
        transmission=(np.sin(2*np.pi*mesh_in*t+self.phase[3])*.018
                      +np.sin(2*np.pi*mesh_out*t+self.phase[4])*.014
                      +np.sin(2*np.pi*diff*37*t+self.phase[9])*.012)*(.35+.65*abs(frame.torque_direction))
        transmission += np.sin(2*np.pi*output*9*t+self.phase[4])*.020*max(0,-frame.torque_direction)
        self.phase[3]=(self.phase[3]+2*np.pi*mesh_in*n/self.sr)%(2*np.pi)
        self.phase[4]=(self.phase[4]+2*np.pi*mesh_out*n/self.sr)%(2*np.pi)
        self.phase[9]=(self.phase[9]+2*np.pi*diff*37*n/self.sr)%(2*np.pi)
        slips=np.asarray(frame.wheel_slip); loads=np.asarray(frame.wheel_load); contacts=np.asarray(frame.wheel_contact)
        surface_color=(1.0,.68,.48,.58,1.30,.82)
        tyres=[]; brakes=[]
        for w in range(4):
            slip=abs(slips[w]); load_w=np.clip(loads[w],0,2)*contacts[w]
            longitudinal=np.clip(slip-.10,0,2); lock=np.clip(-slips[w]-.20,0,1)*frame.brake
            lateral=np.clip(abs(frame.shift_mismatch)*.25+slip-.18,0,1)
            unloaded=max(0,.35-load_w)*min(1,frame.speed/20)
            color=surface_color[min(max(frame.wheel_surface[w],0),5)]*(1-.32*frame.wetness)
            noise=self.rng.standard_normal(n)
            tyre=noise*(.035*longitudinal+.026*lateral+.016*unloaded)*load_w*color
            if frame.wheel_surface[w]==4: tyre += np.sign(np.sin(2*np.pi*(35+frame.speed*1.8)*t))*.012*frame.kerb
            tyres.append(tyre.astype(np.float32))
            temp=frame.brake_temperature[w]
            if temp>.72: self.brake_hot[w]=True
            elif temp<.55: self.brake_hot[w]=False
            scrub=noise*.010*frame.brake*load_w*min(1,frame.speed/35)
            squeal=np.sin(2*np.pi*(1850+170*w+650*temp)*t+self.phase[10])*.012*frame.brake*self.brake_hot[w]
            modulation=.45+.55*np.sign(np.sin(2*np.pi*(8+frame.speed*.4)*t)) if lock>.02 else 1
            brakes.append((scrub+squeal*modulation).astype(np.float32))
        if frame.clutch < .82 and abs(frame.shift_mismatch)>.05:
            chatter=(np.sign(np.sin(2*np.pi*(95+shaft*.3)*t))*self.rng.standard_normal(n))
            transmission += chatter*.026*(1-frame.clutch)*abs(frame.shift_mismatch)
        shift_severity=(1-frame.clutch)*(.25+abs(frame.shift_mismatch))*(.4+.6*abs(frame.torque_direction))
        transmission += np.sin(2*np.pi*72*t)*frame.shift_phase*shift_severity*.035
        for name in self.env:
            gen=getattr(frame.events,name)
            if gen != getattr(self.seen,name):
                setattr(self.seen,name,gen); self.env[name]=1.0
        chassis=np.zeros(n)
        for name, freq, level in (("shift",78,.05),("landing",54,.12),("collision",93,.18),("damage",137,.08)):
            env=self.env[name]
            if env>1e-5:
                decay=np.exp(-np.arange(n)/(self.sr*(.04 if name=="shift" else .12)))
                chassis += np.sin(2*np.pi*freq*t)*env*decay*level
                self.env[name]=float(env*decay[-1])
        susp=np.asarray(frame.suspension_velocity); top=np.clip(np.abs(susp)-2,0,8)
        chassis += self.rng.standard_normal(n)*(.008*frame.kerb+.004*frame.damage+.003*np.mean(top))
        if frame.engine_state != self.prev.engine_state:
            self.lifecycle_env=1.0
        lifecycle=np.zeros(n)
        if self.lifecycle_env>1e-5:
            decay=np.exp(-np.arange(n)/(self.sr*.28)); state=frame.engine_state
            freq={0:42,1:115,2:310,3:58}[state]
            lifecycle=(np.sin(2*np.pi*freq*t)+.35*self.rng.standard_normal(n))*self.lifecycle_env*decay*.055
            self.lifecycle_env=float(self.lifecycle_env*decay[-1])
        wind=self.rng.standard_normal(n)*min(.09, frame.speed*.0007)
        self.prev=frame; self.clock += n
        tyre_sum=np.sum(tyres,axis=0); brake_sum=np.sum(brakes,axis=0)
        return dict(exhaust=exhaust.astype(np.float32), intake=intake.astype(np.float32),
            mechanical=mechanical.astype(np.float32), accessories=accessories.astype(np.float32),
            transmission=transmission.astype(np.float32), tyre_fl=tyres[0],tyre_fr=tyres[1],
            tyre_rl=tyres[2],tyre_rr=tyres[3],tyres=tyre_sum.astype(np.float32),
            brake_fl=brakes[0],brake_fr=brakes[1],brake_rl=brakes[2],brake_rr=brakes[3],
            brakes=brake_sum.astype(np.float32), chassis=(chassis+lifecycle).astype(np.float32), wind=wind.astype(np.float32))


class SpatialRendererV4:
    def __init__(self, sample_rate, cars, seed=78755, propagation=True):
        self.sr=sample_rate; self.propagation=propagation
        self.synths=[EngineSynthV4(sample_rate,seed+i) for i in range(cars)]
        self.delays=[[FractionalDelay(sample_rate) for _ in range(12)] for _ in range(cars)]
        self.fallback=[[RadialDoppler() for _ in range(12)] for _ in range(cars)]
        self.reflections=[EarlyReflections(sample_rate) for _ in range(cars)]
        self.dc=DCBlocker(); self.limiter=SoftLimiter(); self.last_cut=0; self.cut_fade=0
        self.air=np.zeros((cars,12)); self.last_peak=0
        self.previous_out=np.zeros((PREFERRED_BLOCK,2),np.float32)
    def reset(self, frames=None):
        for i,s in enumerate(self.synths):
            f=frames[i] if frames else None; s.reset(f)
            for d in self.delays[i]: d.reset()
            for d in self.fallback[i]: d.reset()
            self.reflections[i].reset()
        self.cut_fade=int(.1*self.sr)
    @staticmethod
    def _source_position(frame, longitudinal, lateral):
        c=math.cos(frame.yaw); s=math.sin(frame.yaw)
        return frame.x+c*longitudinal-s*lateral, frame.y+s*longitudinal+c*lateral
    def render(self, frames, listener, n, perspective=1, cut=0, master=.55):
        out=np.zeros((n,2),np.float32); lx,ly,lvx,lvy,lyaw=listener
        # Deterministic voice virtualization keeps dense 32-car fields within
        # their deadline while preserving the twelve acoustically nearest cars.
        voice_limit=5 if self.sr>=96000 else 12
        active=set(range(len(frames)))
        if len(frames)>voice_limit:
            active=set(sorted(range(len(frames)),key=lambda k:(frames[k].x-lx)**2+(frames[k].y-ly)**2)[:voice_limit])
        if cut != self.last_cut:
            self.last_cut=cut; self.cut_fade=int(.1*self.sr)
        for i,(s,f) in enumerate(zip(self.synths,frames)):
            if i not in active: continue
            stems=s.synthesize_stems(f,n)
            detailed=((len(frames)<=8 and self.sr<96000) or i<4)
            sources=[
              (stems["exhaust"],-2.15,0.0,"exhaust"),(stems["intake"],-.45,.35,"intake"),
              (stems["mechanical"],-.55,0,"mechanical"),(stems["accessories"],-.25,0,"mechanical"),
              (stems["transmission"],-1.15,0,"mechanical"),(stems["chassis"],0,0,"mechanical"),
              (stems["tyre_fl"]+stems["brake_fl"],1.28,-.78,"tyre"),
              (stems["tyre_fr"]+stems["brake_fr"],1.28,.78,"tyre"),
              (stems["tyre_rl"]+stems["brake_rl"],-1.22,-.78,"tyre"),
              (stems["tyre_rr"]+stems["brake_rr"],-1.22,.78,"tyre")]
            if not detailed:
                sources=[(stems["exhaust"]+stems["intake"],-1.25,0,"exhaust"),
                         (stems["mechanical"]+stems["accessories"]+stems["transmission"]+
                          stems["chassis"]+stems["tyres"]+stems["brakes"],-.35,0,"mechanical")]
            car_reflect=np.zeros(n,np.float32)
            for j,(bus,longitudinal,lateral,kind) in enumerate(sources):
                sx,sy=self._source_position(f,longitudinal,lateral); dx=sx-lx; dy=sy-ly; dist=math.hypot(dx,dy)
                if dist>MAX_AUDIBLE_RADIUS_M: continue
                rel=math.atan2(dy,dx)-lyaw; pan=float(np.clip(math.sin(rel),-1,1))
                pl=math.sqrt((1-pan)*.5); pr=math.sqrt((1+pan)*.5); atten=1/(1+(dist/13)**1.45)
                bearing=math.atan2(ly-sy,lx-sx)-f.yaw
                if kind=="exhaust": direct=.25+.75*max(0,math.cos(bearing-math.pi))
                elif kind=="intake": direct=.38+.62*max(0,math.cos(bearing-.35))
                elif kind=="tyre": direct=.68
                else: direct=.55
                if self.propagation: propagated=self.delays[i][j].process(bus,dist)
                else:
                    radial=((f.vx-lvx)*dx+(f.vy-lvy)*dy)/max(dist,.1)
                    propagated=self.fallback[i][j].process(bus,radial)
                alpha=max(.015,min(.7,120/(120+dist*dist)))
                propagated,state=lfilter((alpha,),(1.0,-(1.0-alpha)),propagated,zi=(self.air[i,j],))
                self.air[i,j]=state[0]; weighted=propagated*atten*direct
                out[:,0]+=weighted*pl; out[:,1]+=weighted*pr; car_reflect+=weighted
            reflected=self.reflections[i].process(car_reflect,f.acoustic_preset)
            out[:,0]+=reflected*.72; out[:,1]+=reflected*.72
            if perspective in (1,2): out += stems["wind"][:,None]*.55
        if self.cut_fade:
            m=min(n,self.cut_fade); theta=np.linspace(0,math.pi/2,m)
            old=self.previous_out[:m] if len(self.previous_out)>=m else np.zeros((m,2),np.float32)
            out[:m]=old*np.cos(theta)[:,None]+out[:m]*np.sin(theta)[:,None]; self.cut_fade=max(0,self.cut_fade-n)
        out*=master; self.dc.process(out); self.last_peak=float(np.max(np.abs(out),initial=0)); self.limiter.process(out)
        if len(self.previous_out)!=n: self.previous_out=np.empty((n,2),np.float32)
        self.previous_out[:]=out
        return out


def _read_snapshot(shared, cars):
    for retry in range(4):
        a=int(shared[3])
        if a&1: continue
        head=list(shared[:HEADER]); data=list(shared[HEADER:HEADER+cars*CAR_STRIDE])
        if a==int(shared[3]): return head,[_decode(data[i*CAR_STRIDE:(i+1)*CAR_STRIDE]) for i in range(cars)],retry
    return None,None,4


def _worker(shared, stats, cars):
    try:
        import sounddevice as sd
        requested=os.environ.get("SUPRA_AUDIO_DEVICE")
        device=(int(requested) if requested and requested.lstrip("-").isdigit()
                else (requested or None))
        dev=sd.query_devices(device, "output"); sr=int(round(dev["default_samplerate"]))
        renderer=SpatialRendererV4(sr,cars)
        initialized=False; started=time.perf_counter(); durations=np.zeros(256); duration_i=0; reset_requested=0.0
        def callback(outdata, frames, timing, status):
            nonlocal initialized,duration_i,reset_requested
            begin=time.perf_counter(); head,vehicles,retries=_read_snapshot(shared,cars); stats[9]+=retries
            if head is None:
                outdata.fill(0); stats[8]+=1; return
            if int(head[2])!=SCHEMA_VERSION or not int(head[4]): outdata.fill(0); return
            reset=int(head[15])
            if reset != int(stats[6]):
                reset_requested=time.perf_counter(); renderer.reset(vehicles); stats[6]=reset; shared[16]=reset
                stats[10]=(time.perf_counter()-reset_requested)*1000
            listener=(head[6],head[7],head[8],head[9],head[10])
            fade=min(1.,(time.perf_counter()-started)/.1)
            out=renderer.render(vehicles,listener,frames,int(head[12]),int(head[11]),head[1]*fade)
            outdata[:]=out; initialized=True
            elapsed=time.perf_counter()-begin; stats[1]=elapsed; stats[2]=max(stats[2]*.995,elapsed); stats[11]+=1
            durations[duration_i%len(durations)]=elapsed; duration_i+=1
            if duration_i%32==0:
                valid=durations[:min(duration_i,len(durations))]
                stats[12],stats[13],stats[14]=np.percentile(valid,(50,95,99))
            stats[3]=renderer.last_peak; stats[4]=renderer.limiter.reduction_db
            if status: stats[0]+=1
        kwargs=dict(device=device,channels=2,dtype="float32",samplerate=sr,latency="high",callback=callback)
        try: stream=sd.OutputStream(blocksize=PREFERRED_BLOCK,**kwargs)
        except Exception: stream=sd.OutputStream(**kwargs)
        with stream:
            while shared[0]>0: time.sleep(.05)
    except Exception as exc:
        stats[7]=1
        print(f"[audio-v4] worker failed: {type(exc).__name__}: {exc}", flush=True)


class SpatialAudioMixerV4:
    """Drop-in viewer API with seqlocked AudioFrame V4 shared memory."""
    def __init__(self,master=.55):
        self.master=master; self.muted=False; self.ok=False; self.process=None; self.shared_data=None
        self.stats_data=None; self.adapter=AudioTelemetryAdapter(); self.num_cars=0; self._cut=0; self._perspective=1
        self._vehicles=[]
    def start(self,racers=None):
        if not racers: return self
        self.stop(); vehicles=[r.veh if hasattr(r,"veh") else r for r in racers]; self.num_cars=len(vehicles)
        self._vehicles=vehicles
        self.shared_data=mp.Array('d',HEADER+self.num_cars*CAR_STRIDE,lock=False); self.stats_data=mp.Array('d',16,lock=False)
        self.stats_data[5]=-1.0  # measured by the offline warm-up allocation gate
        self.shared_data[0]=1; self.shared_data[1]=self.master; self.shared_data[2]=SCHEMA_VERSION; self.shared_data[3]=0
        self.shared_data[4]=1; self.shared_data[12]=1; self.adapter.reset(vehicles)
        self.update((vehicles[0].x,vehicles[0].y),(0,0),vehicles[0].yaw,vehicles,[0]*len(vehicles))
        self.process=mp.Process(target=_worker,args=(self.shared_data,self.stats_data,self.num_cars),daemon=True); self.process.start(); self.ok=True
        return self
    def update(self,listener_pos,listener_vel,listener_yaw,vehicles,throttles,brakes=None,clutches=None,perspective=None,camera_cut=False):
        if not self.ok or not self.shared_data or not self.process or not self.process.is_alive(): return
        if camera_cut: self._cut+=1
        self._vehicles=list(vehicles[:self.num_cars])
        if perspective is not None: self._perspective=int(perspective)
        seq=int(self.shared_data[3]); self.shared_data[3]=seq+1 if not seq&1 else seq+2
        self.shared_data[1]=0 if self.muted else self.master
        self.shared_data[6:11]=[_f(listener_pos[0]),_f(listener_pos[1]),_f(listener_vel[0]),_f(listener_vel[1]),_f(listener_yaw)]
        self.shared_data[11]=self._cut; self.shared_data[12]=self._perspective
        for i,v in enumerate(vehicles[:self.num_cars]):
            b=brakes[i] if brakes and i<len(brakes) else None; c=clutches[i] if clutches and i<len(clutches) else None
            frame=self.adapter.frame(v,throttles[i] if i<len(throttles) else 0,b,c); start=HEADER+i*CAR_STRIDE
            self.shared_data[start:start+CAR_STRIDE]=_encode(frame)
        self.shared_data[3]+=1
    def reset(self,vehicles=None):
        vehicles = list(vehicles) if vehicles is not None else self._vehicles
        if vehicles: self.adapter.reset(vehicles)
        if self.shared_data: self.shared_data[15]+=1
    @property
    def metrics(self):
        if not self.stats_data: return {}
        s=list(self.stats_data); return dict(underruns=int(s[0]),callback_seconds=s[1],callback_peak_seconds=s[2],
            peak=s[3],limiter_reduction_db=s[4],steady_heap_growth_bytes=int(s[5]),reset_ack=int(s[6]),
            failed=bool(s[7]),stale_frames=int(s[8]),seqlock_retries=int(s[9]),reset_latency_ms=s[10],
            callback_count=int(s[11]),callback_p50=s[12],callback_p95=s[13],callback_p99=s[14],process_restarts=int(s[15]))
    def toggle_mute(self): self.muted=not self.muted
    def stop(self):
        if self.shared_data: self.shared_data[0]=0
        if self.process:
            self.process.join(1)
            if self.process.is_alive(): self.process.terminate()
        self.ok=False; self.process=None


SpatialAudioMixer = SpatialAudioMixerV4
