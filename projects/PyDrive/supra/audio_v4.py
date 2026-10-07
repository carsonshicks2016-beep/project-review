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

TWO_PI = 2.0 * np.pi

SCHEMA_VERSION = 6   # v6: +boost, +mgu_power, +hybrid_soc telemetry
SPEED_OF_SOUND = 343.0
MAX_AUDIBLE_RADIUS_M = 350.0
PREFERRED_BLOCK = 2048
HEADER = 18
CAR_STRIDE = 64
VIEWER_AUDIO_SEED = 78755
VIEWER_CHASE_PERSPECTIVE = 1
RATIOS_787B = (3.27451, 2.28028, 1.57271, 1.16654, 0.96318)
SURFACES = {"dry": 0, "wet": 1, "grass": 2, "dirt": 3, "kerb": 4, "gravel": 5}


def _f(value, default=0.0, lo=-1e9, hi=1e9):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = default
    return float(np.clip(value if np.isfinite(value) else default, lo, hi))


def _quad(value, default):
    """Normalize scalar, short, or malformed wheel telemetry to four values."""
    if value is None:
        return [default] * 4
    if np.isscalar(value):
        return [value] * 4
    try:
        values = list(value)[:4]
    except (TypeError, ValueError):
        return [default] * 4
    values += [default] * (4 - len(values))
    return values


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
    # hybrid/turbo telemetry (defaults = legacy frame: full pack, no signal)
    boost: float = 0.0           # real turbo spool state (0..1)
    mgu_power: float = 0.0       # MGU power / cap: +1 full deploy, -1 full regen
    hybrid_soc: float = 1.0      # battery state of charge (0..1)
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
        wheel_slip = _quad(getattr(vehicle, "wheel_sr", None), 0.0)
        raw_load=getattr(vehicle,"wheel_load",None)
        if raw_load is None:
            raw_load=getattr(vehicle,"Fz",getattr(vehicle,"_audio_wheel_load",(1,1,1,1)))
            raw=_quad(raw_load,1.0)
            positive=[max(0,float(x)) for x in raw]; mean=max(1e-6,sum(positive)/max(1,len(positive)))
            raw_load=[x/mean for x in positive]
        loads = _quad(raw_load,1.0)
        wheel_surface = _quad(getattr(vehicle, "_audio_wheel_surface", None), surface)
        contact = _quad(getattr(vehicle, "_audio_wheel_contact", getattr(vehicle,"_contact",None)), 1.0)
        brake_temp = _quad(getattr(vehicle, "_audio_brake_temperature", None), 0.2)
        susp_travel = _quad(getattr(vehicle, "_audio_suspension_travel", None), 0.0)
        susp_velocity = _quad(getattr(vehicle, "_audio_suspension_velocity", None), 0.0)
        brake = _f(getattr(vehicle, "_audio_brake", 0) if brake is None else brake, 0, 0, 1)
        clutch = _f(getattr(vehicle, "_audio_clutch", 1) if clutch is None else clutch, 1, 0, 1)
        redline = _f(getattr(spec, "redline_rpm", 9000), 9000, 1000, 14000)
        explicit_limiter=getattr(vehicle,"limiter",None)
        if explicit_limiter is None:
            was_limited=bool(prev.get("limiter",False))
            limiter=(rpm >= redline) if not was_limited else (rpm >= redline-150)
        else:
            limiter=bool(explicit_limiter)
        vx = _f(getattr(vehicle, "vx", speed * math.cos(yaw)))
        vy = _f(getattr(vehicle, "vy", speed * math.sin(yaw)))
        boost = _f(getattr(vehicle, "boost", 0), 0, 0, 2)
        mgu_cap = _f(getattr(spec, "hybrid_mgu_power_w", 0), 0, 0, 2e6)
        mgu_norm = (_f(getattr(vehicle, "mgu_power_w", 0), 0, -2e6, 2e6) / mgu_cap
                    if mgu_cap > 0 else 0.0)
        batt = _f(getattr(spec, "hybrid_battery_kj", 0), 0, 0, 1e6)
        soc = (_f(getattr(vehicle, "hybrid_soc_kj", batt), batt, 0, 1e6) / batt
               if batt > 0 else 1.0)
        mismatch = _f(getattr(vehicle, "_audio_shift_mismatch", 0), 0, -1, 1)
        torque = _f(getattr(vehicle, "_audio_torque_direction", 2 * _f(throttle, 0, 0, 1) - 0.4), 0, -1, 1)
        frame = AudioFrame(
            x=_f(getattr(vehicle, "x", 0)), y=_f(getattr(vehicle, "y", 0)), yaw=yaw,
            vx=vx, vy=vy, speed=speed, rpm=rpm, throttle=_f(throttle, 0, 0, 1),
            brake=brake, clutch=clutch, gear=gear, ratio=_f(ratios[gear-1], RATIOS_787B[min(gear-1, len(RATIOS_787B)-1)], .1, 8),
            limiter=float(limiter),
            wheel_slip=tuple(_f(x, 0, -5, 5) for x in wheel_slip),
            wheel_load=tuple(_f(x, 1, 0, 5) for x in loads), surface=surface,
            wetness=_f(getattr(vehicle, "_audio_wetness", 0), 0, 0, 1),
            airborne=_f(getattr(vehicle, "airborne", getattr(vehicle, "_audio_airborne", 0)), 0, 0, 1),
            landing_force=landing, damage=damage, collision_impulse=collision,
            shift_mismatch=mismatch, torque_direction=torque,
            kerb=_f(getattr(vehicle, "_snd_kerb", 0), 0, 0, 1),
            boost=float(np.clip(boost, 0.0, 1.0)),
            mgu_power=float(np.clip(mgu_norm, -1.0, 1.0)),
            hybrid_soc=float(np.clip(soc, 0.0, 1.0)),
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
        prev.update(gear=gear, landing=landing, collision=collision, damage=damage, surface=surface,limiter=limiter)
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
            frame.acoustic_preset, frame.boost, frame.mgu_power, frame.hybrid_soc]


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
        boost=v[61], mgu_power=v[62], hybrid_soc=v[63],
        events=AudioEvents(*map(int, v[30:36])))


class DCBlocker:
    def __init__(self, channels=2, pole=.995):
        self.pole=pole; self.zi=np.zeros((1,channels))
    def process(self, x):
        x[:],self.zi=lfilter((1.0,-1.0),(1.0,-self.pole),x,axis=0,zi=self.zi)
        return x


class SoftLimiter:
    def __init__(self, sample_rate=48000, threshold=.68):
        self.threshold, self.gain, self.reduction_db = threshold, 1.0, 0.0
        self.env=0.0; self.attack=math.exp(-1/(sample_rate*.0015)); self.release=math.exp(-1/(sample_rate*.09))
    def process(self, x):
        peaks=np.max(np.abs(x),axis=1)
        if float(np.max(peaks,initial=0))<self.threshold and self.env<self.threshold and self.gain>.999:
            self.env=max(float(np.max(peaks,initial=0)),self.env*self.release)
            self.gain=1.0; self.reduction_db=0.0
            np.tanh(x/.82,out=x); x*=.82
            return x
        for i,peak in enumerate(peaks):
            c=self.attack if peak>self.env else self.release
            self.env=c*self.env+(1-c)*peak
            target=min(1.0,self.threshold/max(self.env,1e-9))
            gc=.92 if target<self.gain else .998
            self.gain=gc*self.gain+(1-gc)*target
            x[i]*=self.gain
        self.reduction_db = -20*math.log10(max(self.gain, 1e-8))
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
    """Small stateful ground/venue reflection network.

    Trackside/forest presets use denser, darker returns so external cameras
    hear a corridor rather than a dry open-field source. Reflections are a
    post-propagation bed only — never a second Doppler transform.
    """
    def __init__(self,sample_rate):
        self.sr=sample_rate; self.size=max(4096,int(.12*sample_rate)); self.buf=np.zeros(self.size,np.float32); self.write=0
        self.dark=0.0
    def reset(self): self.buf.fill(0); self.write=0; self.dark=0.0
    def process(self,source,preset):
        delays=((.0055,.012,.021),(.009,.019,.034),(.014,.028,.048))[min(max(int(preset),0),2)]
        gains=((.045,.016,.006),(.070,.028,.010),(.095,.040,.014))[min(max(int(preset),0),2)]
        idx=(self.write+np.arange(len(source)))%self.size; self.buf[idx]=source
        out=np.zeros(len(source),np.float32)
        for delay,gain in zip(delays,gains): out += self.buf[(idx-int(delay*self.sr))%self.size]*gain
        # One-pole darkening keeps the return under the bark instead of hissy.
        alpha=.22
        darkened,zi=lfilter((alpha,),(1.0,-(1.0-alpha)),out,zi=(self.dark,))
        self.dark=float(zi[0]); out=darkened
        self.write=(self.write+len(source))%self.size
        return out


class ObservatoryAmbience:
    """Deterministic Nordschleife atmosphere for the streamed 3D viewer.

    This is intentionally downstream of simulation. It reacts only to the
    selected visual preset and listener/camera motion, so no ambient state can
    enter physics, observations, rewards, or policy inference.
    """
    def __init__(self, sample_rate=48000, seed=99173):
        self.sr=int(sample_rate); self.rng=np.random.default_rng(seed)
        self.clock=0; self.wind_state=np.zeros(2); self.rustle_state=np.zeros(2)
        self.bird_env=0.0; self.bird_phase=0.0; self.bird_frequency=2400.0
        self.insect_phase=np.zeros(3); self.last_weather="dry"
    def reset(self):
        self.clock=0; self.wind_state.fill(0); self.rustle_state.fill(0)
        self.bird_env=0.0; self.bird_phase=0.0; self.insect_phase.fill(0)
    def _filtered_noise(self,n,alpha,state):
        noise=self.rng.standard_normal((n,2)); out=np.empty((n,2),np.float32)
        for channel in range(2):
            filtered,zi=lfilter((alpha,),(1.0,-(1.0-alpha)),noise[:,channel],zi=(state[channel],))
            state[channel]=zi[0]; out[:,channel]=filtered
        return out
    def render(self,n,*,weather="dry",listener=(0,0,0,0,0),listener_z=0.0,
               perspective=0,camera="broadcast"):
        weather=str(weather).lower() if weather in ("dry","dusk","night") else "dry"
        local=perspective in (1,2); speed=math.hypot(float(listener[2]),float(listener[3]))
        absolute=(self.clock+np.arange(n,dtype=np.float64))/self.sr
        gust=.62+.22*np.sin(TWO_PI*.071*absolute)+.16*np.sin(TWO_PI*.137*absolute+1.4)
        wind=self._filtered_noise(n,.012,self.wind_state)
        rustle_raw=self.rng.standard_normal((n,2)).astype(np.float32)
        rustle_low=self._filtered_noise(n,.08,self.rustle_state)
        rustle=rustle_raw-rustle_low
        weather_gain={"dry":.72,"dusk":.86,"night":.48}[weather]
        altitude_gain=1.0+min(1.0,max(0.0,float(listener_z)-3.0)/28.0)*.8
        camera_wind=min(.026,speed*.00024)*(1.0 if camera in ("drone","helicopter","free") else .55)
        bed=(wind*(.010*weather_gain*altitude_gain+camera_wind)
             +rustle*(.0018 if local else .0036)*weather_gain)
        bed*=gust[:,None]

        wildlife=np.zeros((n,2),np.float32)
        duration=n/self.sr
        if weather=="dry" and self.bird_env<1e-4 and self.rng.random()<duration*.34:
            self.bird_env=1.0; self.bird_frequency=float(self.rng.uniform(1900,3300))
        if self.bird_env>1e-5 and weather=="dry":
            decay=np.exp(-np.arange(n)/(self.sr*.19)); chirp=self.bird_frequency+680*np.sin(TWO_PI*3.1*absolute)
            angle=self.bird_phase+TWO_PI*np.cumsum(chirp)/self.sr; self.bird_phase=float(angle[-1]%TWO_PI)
            tone=np.sin(angle)*decay*self.bird_env*.006
            pan=.38*np.sin(TWO_PI*.043*absolute[0]+.7); wildlife[:,0]+=tone*math.sqrt((1-pan)*.5); wildlife[:,1]+=tone*math.sqrt((1+pan)*.5)
            self.bird_env=float(self.bird_env*decay[-1])
        if weather in ("dusk","night"):
            insect_level=.0048 if weather=="dusk" else .0062
            for index,frequency in enumerate((4180.0,4930.0,5660.0)):
                modulation=.28+.72*np.maximum(0,np.sin(TWO_PI*(5.1+index*.83)*absolute+index))
                angle=self.insect_phase[index]+TWO_PI*frequency*np.arange(1,n+1)/self.sr
                self.insect_phase[index]=float(angle[-1]%TWO_PI)
                tone=np.sin(angle)*modulation*insect_level/3
                wildlife[:,index%2]+=tone
        if local: wildlife*=.28; bed*=.44
        self.clock+=n; self.last_weather=weather
        return (bed+wildlife).astype(np.float32)


# Per-car acoustic voice. Identity-critical constants (fires_per_rev, redline,
# final_drive, runners, formant base) stay locked to the procedural profile.
# Timbre knobs below shape the from-scratch stem rewrite without changing those
# drivetrain/firing contracts.
VOICE_PROFILES = {
    "mazda787b": dict(
        fires_per_rev=4.0,      # 4-rotor R26B: one pulse per rotor per rev
        redline=9000.0,
        final_drive=4.357,
        runners=(5700.0, 5300.0, 7900.0, 7500.0),  # up1, down1, up2, down2
        formant=(520.0, 1180.0, 220.0),            # base, runner sweep, load
        texture=1.0,            # inter-rotor shimmer depth
        mgu_level=0.0,          # hybrid MGU whine (919)
        turbo_level=0.0,        # compressor whine + wastegate vent (919)
        bank_count=4,           # independent rotor voices
        bark=1.15,              # open-pipe upper-partial weight
        pipe_modes=(410.0, 980.0, 1680.0),
        pulse_sharp=0.55,
        piston=False,
    ),
    "porsche_919evo": dict(
        fires_per_rev=2.0,      # 2.0 L V4 four-stroke: two firings per rev
        redline=9000.0,
        final_drive=3.64479,
        runners=None,
        formant=(680.0, 0.0, 300.0),
        texture=0.28,
        mgu_level=1.0,
        turbo_level=1.0,
        bank_count=2,
        bark=1.12,
        pipe_modes=(620.0, 1280.0, 2450.0),
        pulse_sharp=0.36,
        piston=True,
        exhaust_drive=0.95,
        exhaust_level=0.30,
        rasp_level=0.055,
        turbo_whine=0.012,
        turbo_whoosh=0.032,
        wastegate_level=0.14,
        mgu_deploy=0.0065,
        mgu_regen=0.0075,
        valve_tick=1.45,
        mesh_gain=1.15,
    ),
}


class EngineSynthV4:
    """From-scratch procedural engine/vehicle stem model.

    Every stem is rebuilt: exhaust blowdown banks, induction, machinery,
    accessories (incl. 919 MGU), transmission mesh, per-wheel tyres/brakes,
    chassis events, and body wind. Voiced by VOICE_PROFILES; Doppler remains
    exclusively the spatial propagation-delay path.
    """
    STEMS = ("exhaust", "intake", "mechanical", "accessories", "transmission",
             "tyre_fl", "tyre_fr", "tyre_rl", "tyre_rr", "brakes", "chassis", "wind")

    def __init__(self, sample_rate=48000, seed=78755, car="mazda787b"):
        self.car = str(car)
        self.voice = VOICE_PROFILES.get(self.car, VOICE_PROFILES["mazda787b"])
        self.sr = int(sample_rate)
        self.rng = np.random.default_rng(seed)
        self.phase = np.zeros(20, dtype=np.float64)
        self.harmonic_phase = np.zeros(12, dtype=np.float64)
        self.bank_phase = np.zeros(4, dtype=np.float64)
        self.pipe_phase = np.zeros(3, dtype=np.float64)
        self.clock = 0
        self.prev = AudioFrame()
        self.limiter_gain = 1.0
        self.seen = AudioEvents()
        self.env = dict(shift=0., landing=0., collision=0., damage=0., surface=0.)
        self.runner_state = 0
        self.runner_formant = float(self.voice["formant"][0])
        self.intake_filter = np.zeros(2, dtype=np.float64)
        self.intake_noise_lp = np.zeros(1, dtype=np.float64)
        self.rasp_hp = np.zeros(1, dtype=np.float64)
        self.rasp_bp = np.zeros(1, dtype=np.float64)
        self.exhaust_body = np.zeros(1, dtype=np.float64)
        self.mech_tick_lp = np.zeros(1, dtype=np.float64)
        self.tyre_lp = np.zeros((4, 1), dtype=np.float64)
        self.wind_lp = np.zeros(1, dtype=np.float64)
        self.overrun_lp = np.zeros(1, dtype=np.float64)
        self.turbo_whoosh_lp = np.zeros(1, dtype=np.float64)
        self.mgu_noise_lp = np.zeros(1, dtype=np.float64)
        self.lifecycle_env = 0.0
        self.brake_hot = np.zeros(4, dtype=bool)
        self.boost_env = 0.0
        self.wastegate_env = 0.0
        self._regen_phase2 = 0.0
        self._regen_phase3 = 0.0
        self._overrun_env = 0.0
        self._mgu_amp = 0.0

    def reset(self, frame=None):
        self.phase.fill(0)
        self.harmonic_phase.fill(0)
        self.bank_phase.fill(0)
        self.pipe_phase.fill(0)
        self.limiter_gain = 1.0
        self.env = {k: 0. for k in self.env}
        self.prev = frame or AudioFrame()
        self.runner_state = 0
        self.runner_formant = float(self.voice["formant"][0])
        self.intake_filter.fill(0)
        self.intake_noise_lp.fill(0)
        self.rasp_hp.fill(0)
        self.rasp_bp.fill(0)
        self.exhaust_body.fill(0)
        self.mech_tick_lp.fill(0)
        self.tyre_lp.fill(0)
        self.wind_lp.fill(0)
        self.overrun_lp.fill(0)
        self.turbo_whoosh_lp.fill(0)
        self.mgu_noise_lp.fill(0)
        self.lifecycle_env = 0.0
        self.brake_hot.fill(False)
        self.boost_env = 0.0
        self.wastegate_env = 0.0
        self._regen_phase2 = 0.0
        self._regen_phase3 = 0.0
        self._overrun_env = 0.0
        self._mgu_amp = 0.0
        self.seen = AudioEvents(**vars(self.prev.events))

    def _osc(self, index, frequency, n):
        freq = np.asarray(frequency, dtype=np.float64)
        if freq.ndim == 0:
            freq = np.full(n, float(freq))
        elif freq.shape[0] != n:
            freq = np.broadcast_to(freq, (n,))
        angle = self.phase[index] + TWO_PI * np.cumsum(freq) / self.sr
        self.phase[index] = float(angle[-1] % TWO_PI)
        return np.sin(angle)

    def _one_pole(self, x, alpha, state):
        y, zi = lfilter((alpha,), (1.0, -(1.0 - alpha)), x, zi=(state[0],))
        state[0] = zi[0]
        return y

    def _bandpass_noise(self, n, hp_alpha, lp_alpha):
        raw = self.rng.standard_normal(n)
        hp, self.rasp_hp = lfilter((1.0, -1.0), (1.0, -(1.0 - hp_alpha)), raw, zi=self.rasp_hp)
        bp = self._one_pole(hp, lp_alpha, self.rasp_bp)
        return bp

    def _powertrain_919(self, frame, n, *, absolute_t, rpm, effective_rpm, rpm_line,
                        rpm_f, rpm_f_line, load, throttle_line, combustion,
                        fundamental, crank_line, cut, detail):
        """Porsche 919 Evo V4 turbo-hybrid powertrain.

        Full race-car voice: odd-heavy screamer exhaust, rising compressor spool,
        race wastegate vent, and MGU deploy/regen. Anti-trill rules only:
        no continuous >4 kHz pure sines, no full-pack MGU fallback, tones are
        noise-AM'd before they hit delay-Doppler.
        """
        voice = self.voice
        bark = float(voice.get("bark", 1.0))
        texture = float(voice.get("texture", 0.28))
        banks = int(voice.get("bank_count", 2)) if detail else 1
        harm_count = 8 if detail else 5

        # ---- exhaust: odd-heavy V4 screamer ----
        harm = np.zeros(n, dtype=np.float64)
        base_amps = [
            1.00,
            0.24 + 0.10 * load,
            0.52 + 0.22 * rpm_f * bark,   # H3
            0.18 + 0.10 * load,
            0.38 + 0.18 * rpm_f * bark,   # H5
            0.12 + 0.08 * load,
            0.22 + 0.14 * rpm_f * bark,   # H7
            0.08 + 0.06 * load,
        ]
        phase_color = (0.0, 0.21, -0.33, 0.55, -0.71, 0.94, 0.38, -0.62)
        amp_sum = 0.0
        for h, amp in enumerate(base_amps[:harm_count], 1):
            amp_sum += amp
            angle = self.harmonic_phase[h - 1] + TWO_PI * np.cumsum(fundamental * h) / self.sr
            color = 1.0 + (0.10 if h % 2 else -0.05) * (1.0 + 0.35 * texture)
            harm += amp * color * np.sin(angle + phase_color[h - 1])
            self.harmonic_phase[h - 1] = float(angle[-1] % TWO_PI)
        harm /= max(1e-9, amp_sum)

        shimmer = (1.0
                   + 0.012 * texture * np.sin(TWO_PI * 11.0 * absolute_t)
                   + 0.008 * texture * np.sin(TWO_PI * 23.0 * absolute_t + 0.9))
        boost_feel = 0.90 + 0.20 * float(self.boost_env)
        drive = float(voice.get("exhaust_drive", 0.95)) + 0.45 * combustion * bark
        tonal = np.tanh(harm * shimmer * drive * boost_feel) * (
            float(voice.get("exhaust_level", 0.30)) * (0.50 + 0.50 * combustion))

        fire_angle = self.bank_phase[0] + TWO_PI * np.cumsum(fundamental) / self.sr
        self.bank_phase[0] = float(fire_angle[-1] % TWO_PI)
        pulse = np.zeros(n, dtype=np.float64)
        for b in range(banks):
            ang = fire_angle + (TWO_PI * b) / max(1, banks)
            raw = np.sin(ang)
            pos = np.clip(raw, 0.0, 1.0)
            gate = pos * pos * (3.0 - 2.0 * pos) + 0.20 * np.clip(-raw, 0.0, 1.0)
            imbalance = 1.0 + 0.04 * texture * np.sin(ang * 0.5 + 0.9 * b)
            pulse += gate * imbalance
        pulse /= max(1, banks)
        pulse = 0.30 + 0.70 * np.clip(pulse, 0.0, 1.0)

        rasp = self._bandpass_noise(n, 0.20 + 0.12 * load, 0.46 - 0.10 * load)
        rasp *= pulse * float(voice.get("rasp_level", 0.055)) * combustion * (0.30 + 0.70 * rpm_f) * bark

        modes = voice.get("pipe_modes", (620.0, 1280.0, 2450.0))
        pipe = np.zeros(n, dtype=np.float64)
        mode_w = (0.008, 0.007, 0.0035)
        mode_n = (3 if detail else 1)
        for i, (mode_f, w) in enumerate(zip(modes[:mode_n], mode_w[:mode_n])):
            f_line = mode_f * (1.0 + 0.05 * throttle_line + 0.04 * float(self.boost_env))
            ang = self.pipe_phase[i] + TWO_PI * np.cumsum(f_line) / self.sr
            self.pipe_phase[i] = float(ang[-1] % TWO_PI)
            pipe += np.sin(ang) * w * pulse * combustion * (0.35 + 0.65 * rpm_f_line) * bark

        exhaust = (tonal + rasp + pipe) * cut
        exhaust = self._one_pole(exhaust, 0.42, self.exhaust_body)

        if detail and load < 0.12 and rpm > 3200:
            self._overrun_env = min(1.0, self._overrun_env + n / self.sr * 2.8)
        elif detail:
            self._overrun_env = max(0.0, self._overrun_env - n / self.sr * 4.0)
        if detail and self._overrun_env > 1e-4:
            miss = 0.2 + 0.8 * (0.5 + 0.5 * np.sin(TWO_PI * (22.0 + 55.0 * rpm_f) * absolute_t)) ** 2.8
            pop = self._one_pole(self.rng.standard_normal(n) * miss, 0.62, self.overrun_lp)
            exhaust = exhaust + pop * self._overrun_env * 0.018 * rpm_f

        # ---- intake + turbo spool + wastegate ----
        formant_base, _, formant_load = voice["formant"]
        formant_target = formant_base + formant_load * load + 160.0 * float(self.boost_env)
        self.runner_formant += 0.14 * (formant_target - self.runner_formant)
        formant = self.runner_formant
        intake_tone = self._osc(0, np.full(n, formant), n)
        intake_noise = self._one_pole(self.rng.standard_normal(n), 0.40, self.intake_noise_lp)
        radius = 0.86
        omega = TWO_PI * formant / self.sr
        resonant, self.intake_filter = lfilter(
            (1 - radius, 0, 0),
            (1, -2 * radius * np.cos(omega), radius * radius),
            intake_noise, zi=self.intake_filter)
        whoosh = self._one_pole(self.rng.standard_normal(n), 0.50, self.turbo_whoosh_lp)
        intake_gain = (throttle_line ** 1.2) * (0.018 + 0.034 * rpm_f_line)
        intake = (intake_tone * 0.045 + resonant * 0.22 + intake_noise * 0.12) * intake_gain
        intake = intake + whoosh * float(voice.get("turbo_whoosh", 0.032)) * (
            0.18 + 0.82 * throttle_line) * (0.30 + 0.70 * float(self.boost_env) + 0.25 * load)

        boost_at_lift = self.boost_env
        if (float(self.prev.throttle) - load) > 0.30 and boost_at_lift > 0.25:
            self.wastegate_env = max(self.wastegate_env, boost_at_lift)
        boost_start = float(self.boost_env)
        if frame.boost > 1e-4:
            self.boost_env = float(np.clip(frame.boost, 0.0, 1.0))
        else:
            target = 0.75 * load * np.clip((rpm_f - 0.18) / 0.50, 0.0, 1.0)
            tau = 0.32 if target > self.boost_env else 0.09
            self.boost_env += (target - self.boost_env) * min(1.0, (n / self.sr) / tau)
        boost_line = np.linspace(boost_start, self.boost_env, n)
        level = float(voice["turbo_level"])

        # Rising compressor spool — clear pitch climb, noise-AM'd, capped <4 kHz.
        compressor_air = self._one_pole(self.rng.standard_normal(n), 0.52, self.turbo_whoosh_lp)
        whine_f = 1100.0 + 2700.0 * boost_line
        whine = self._osc(10, whine_f, n)
        am = 0.35 + 0.65 * np.clip(np.abs(compressor_air) * 2.0, 0.0, 1.0)
        spool = level * boost_line * (0.25 + 0.75 * load)
        intake = intake + (
            compressor_air * 0.024
            + whine * float(voice.get("turbo_whine", 0.012)) * 2.2 * am
        ) * spool

        # Race wastegate vent on lift — audible PSSHH, not a sine whistle.
        if self.wastegate_env > 1e-4:
            vent_decay = np.exp(-np.arange(n) / (self.sr * 0.15))
            vent_env = self.wastegate_env * vent_decay
            vent_noise = np.diff(self.rng.standard_normal(n), prepend=0.0)
            vent_body = self._osc(11, 260.0 + 480.0 * vent_env, n)
            intake = intake + (
                0.80 * vent_noise
                + 0.28 * vent_body * np.clip(np.abs(vent_noise), 0.0, 1.0)
            ) * vent_env * float(voice.get("wastegate_level", 0.14)) * level * 1.45
            self.wastegate_env = float(self.wastegate_env * vent_decay[-1])

        # ---- mechanical ----
        beat = 0.97 + 0.03 * np.sin(TWO_PI * 2.9 * absolute_t)
        crank = self._osc(1, crank_line, n) * 0.010 * beat
        order2 = self._osc(2, crank_line * 2.0, n) * 0.0060
        tick = self._one_pole(self.rng.standard_normal(n), 0.60, self.mech_tick_lp)
        tick *= (0.0026 + 0.0050 * rpm_f) * float(voice.get("valve_tick", 1.45))
        cam = self._osc(13, crank_line * 2.0, n) * 0.0022 * rpm_f
        mechanical = (crank + order2 + tick + cam) * (0.50 + 0.50 * rpm_f)

        # ---- accessories + MGU (ledger-only; no full-pack fallback) ----
        accessories = (self._osc(3, crank_line * 1.5, n) * 0.0018
                       + self._osc(4, crank_line * 3.0, n) * 0.0012) * (0.45 + 0.55 * rpm_f)
        mgu = float(frame.mgu_power)
        mgu_level = float(voice["mgu_level"])
        target_amp = mgu_level * min(1.0, abs(mgu)) if abs(mgu) > 1e-4 else 0.0
        self._mgu_amp += (target_amp - self._mgu_amp) * min(1.0, (n / self.sr) / 0.05)
        strength = float(self._mgu_amp)
        if strength > 1e-4:
            grain = self._one_pole(self.rng.standard_normal(n), 0.50, self.mgu_noise_lp)
            if mgu >= 0:
                f_line = 1600.0 + 2400.0 * rpm_f_line   # stay under ~4 kHz
                tone = self._osc(9, f_line, n)
                accessories = accessories + (
                    grain * 0.0048
                    + tone * float(voice.get("mgu_deploy", 0.0065)) * (0.45 + 0.55 * np.abs(grain))
                ) * strength
            else:
                speed_start = float(self.prev.speed)
                speed_line = np.linspace(speed_start, frame.speed, n)
                howl_line = 850.0 + 1700.0 * np.clip(speed_line / 95.0, 0.0, 1.0)
                ang2 = self._regen_phase2 + TWO_PI * np.cumsum(howl_line) / self.sr
                self._regen_phase2 = float(ang2[-1] % TWO_PI)
                accessories = accessories + (
                    grain * 0.0052
                    + np.sin(ang2) * float(voice.get("mgu_regen", 0.0075)) * (0.45 + 0.55 * np.abs(grain))
                ) * strength

        # ---- transmission ----
        shaft_line = crank_line
        output_line = shaft_line / max(float(frame.ratio), 0.1)
        final_drive = float(voice["final_drive"])
        diff_line = output_line / final_drive
        mesh_g = float(voice.get("mesh_gain", 1.15))
        mesh_noise = self.rng.standard_normal(n) * 0.0030
        torque = float(frame.torque_direction)
        transmission = (
            self._osc(5, shaft_line * 19.0, n) * 0.0065
            + self._osc(6, output_line * 26.0, n) * 0.0045
            + self._osc(7, diff_line * 41.0, n) * 0.0030
            + mesh_noise
        ) * (0.22 + 0.58 * abs(torque)) * mesh_g
        transmission = transmission + self._osc(8, output_line * 11.0, n) * 0.0060 * max(0.0, -torque)
        if frame.clutch < 0.82 and abs(frame.shift_mismatch) > 0.05:
            chatter = (np.tanh(np.sin(TWO_PI * (105.0 + effective_rpm * 0.0045) * absolute_t) * 2.3)
                       * self.rng.standard_normal(n))
            transmission = transmission + chatter * 0.018 * (1.0 - frame.clutch) * abs(frame.shift_mismatch)
        shift_severity = ((1.0 - frame.clutch)
                         * (0.25 + abs(frame.shift_mismatch))
                         * (0.4 + 0.6 * abs(torque)))
        transmission = transmission + (
            np.sin(TWO_PI * 88.0 * absolute_t) * frame.shift_phase * shift_severity * 0.022)

        return exhaust, intake, mechanical, accessories, transmission

    def synthesize_stems(self, frame: AudioFrame, n: int, detail: bool = True):
        absolute_t = (self.clock + np.arange(n, dtype=np.float64)) / self.sr
        voice = self.voice
        redline = float(voice["redline"])
        rpm = max(0.0, float(frame.rpm))
        previous_rpm = max(300.0, float(self.prev.rpm))
        effective_rpm = max(300.0, rpm)
        rpm_line = np.linspace(previous_rpm, effective_rpm, n)
        rpm_f = float(np.clip(effective_rpm / redline, 0.0, 1.0))
        rpm_f_line = np.clip(rpm_line / redline, 0.0, 1.0)
        load = float(np.clip(frame.throttle, 0.0, 1.0))
        throttle_line = np.linspace(float(self.prev.throttle), load, n)
        combustion = 0.16 + 0.84 * throttle_line
        fires = float(voice["fires_per_rev"])
        fundamental = rpm_line / 60.0 * fires
        rotor_line = rpm_line / 60.0
        bark = float(voice.get("bark", 1.0))
        texture = float(voice.get("texture", 1.0))
        sharp = float(voice.get("pulse_sharp", 0.5))
        banks = int(voice.get("bank_count", 4)) if detail else 1
        piston = bool(voice.get("piston", False))
        harm_count = 6 if detail else 4
        crank_line = rotor_line

        # ---- limiter: soft cyclic ignition cuts (no hard silences) ----
        if frame.limiter:
            pattern = np.where(
                ((self.clock + np.arange(n)) // max(1, int(0.0085 * self.sr))) % 4 == 1,
                0.52, 1.0).astype(np.float64)
        else:
            pattern = np.ones(n, dtype=np.float64)
        alpha_lim = 0.07
        cut, zi = lfilter((alpha_lim,), (1.0, -(1.0 - alpha_lim)), pattern, zi=(float(self.limiter_gain),))
        self.limiter_gain = float(zi[0])

        if piston:
            exhaust, intake, mechanical, accessories, transmission = self._powertrain_919(
                frame, n, absolute_t=absolute_t, rpm=rpm, effective_rpm=effective_rpm,
                rpm_line=rpm_line, rpm_f=rpm_f, rpm_f_line=rpm_f_line, load=load,
                throttle_line=throttle_line, combustion=combustion, fundamental=fundamental,
                crank_line=crank_line, cut=cut, detail=detail)
        else:
            # ---- 787B R26B rotary exhaust / induction / machinery ----
            harm = np.zeros(n, dtype=np.float64)
            base_amps = [
                1.00,
                0.42 + 0.16 * load,
                0.30 + 0.18 * rpm_f * bark,
                0.20 + 0.14 * load * bark,
                0.14 + 0.12 * rpm_f * bark,
                0.10 + 0.10 * load * bark,
                0.07 + 0.08 * rpm_f * bark,
                0.05 + 0.06 * load,
                0.035 + 0.05 * rpm_f * bark,
                0.022 + 0.035 * load * bark,
            ]
            phase_color = (0.0, 0.37, -0.22, 0.81, -0.55, 1.13, 0.44, -0.91, 0.62, -0.18)
            amp_sum = 0.0
            for h, amp in enumerate(base_amps[:harm_count], 1):
                amp_sum += amp
                angle = self.harmonic_phase[h - 1] + TWO_PI * np.cumsum(fundamental * h) / self.sr
                color = (1.0 + (0.08 if (h % 2 == 0) else -0.03) * texture)
                harm += amp * color * np.sin(angle + phase_color[h - 1])
                self.harmonic_phase[h - 1] = float(angle[-1] % TWO_PI)
            harm /= max(1e-9, amp_sum)

            shimmer = (1.0
                       + 0.028 * texture * np.sin(TWO_PI * 17.3 * absolute_t)
                       + 0.016 * texture * np.sin(TWO_PI * 29.1 * absolute_t + 0.7)
                       + 0.012 * texture * np.sin(TWO_PI * 5.7 * absolute_t + 1.1))
            carrier = harm * shimmer
            drive = 0.85 + 0.55 * combustion * bark
            tonal = np.tanh(carrier * drive) * (0.10 + 0.28 * combustion)

            fire_angle = self.bank_phase[0] + TWO_PI * np.cumsum(fundamental) / self.sr
            self.bank_phase[0] = float(fire_angle[-1] % TWO_PI)
            pulse = np.zeros(n, dtype=np.float64)
            for b in range(banks):
                ang = fire_angle + (TWO_PI * b) / max(1, banks)
                raw = np.sin(ang)
                pos = np.clip(raw, 0.0, 1.0)
                gate = pos * pos * (3.0 - 2.0 * pos) + 0.22 * np.clip(-raw, 0.0, 1.0)
                imbalance = 1.0 + 0.07 * texture * np.sin(ang * 0.25 + 0.4 * b + 1.3)
                pulse += gate * imbalance
            pulse /= max(1, banks)
            pulse = 0.28 + 0.72 * np.clip(pulse, 0.0, 1.0)

            hp_a = 0.18 + 0.10 * load
            lp_a = 0.42 - 0.12 * load
            rasp = self._bandpass_noise(n, hp_a, lp_a)
            rasp *= pulse * (0.022 + 0.055 * combustion * (0.35 + 0.65 * rpm_f) * bark)

            modes = voice.get("pipe_modes", (410.0, 980.0, 1680.0))
            pipe = np.zeros(n, dtype=np.float64)
            mode_w = (0.006, 0.0035, 0.0018)
            mode_n = min(2, len(modes)) if detail else 1
            for i, (mode_f, w) in enumerate(zip(modes[:mode_n], mode_w[:mode_n])):
                f_line = mode_f * (1.0 + 0.04 * throttle_line)
                ang = self.pipe_phase[i] + TWO_PI * np.cumsum(f_line) / self.sr
                self.pipe_phase[i] = float(ang[-1] % TWO_PI)
                pipe += np.sin(ang) * w * pulse * combustion * (0.4 + 0.6 * rpm_f_line) * bark

            exhaust = (tonal + rasp + pipe) * cut
            exhaust = self._one_pole(exhaust, 0.38, self.exhaust_body)

            if detail and load < 0.14 and rpm > 2800:
                self._overrun_env = min(1.0, self._overrun_env + n / self.sr * 2.4)
            elif detail:
                self._overrun_env = max(0.0, self._overrun_env - n / self.sr * 3.5)
            if detail and self._overrun_env > 1e-4:
                miss_rate = 18.0 + 40.0 * rpm_f
                miss = 0.25 + 0.75 * (0.5 + 0.5 * np.sin(TWO_PI * miss_rate * absolute_t)) ** 3.0
                pop = self._one_pole(self.rng.standard_normal(n) * miss, 0.55, self.overrun_lp)
                exhaust = exhaust + pop * self._overrun_env * (0.012 + 0.018 * frame.engine_temperature) * rpm_f

            runners = voice["runners"]
            if runners is not None:
                up1, down1, up2, down2 = runners
                if self.runner_state == 0 and rpm > up1:
                    self.runner_state = 1
                elif self.runner_state == 1 and rpm < down1:
                    self.runner_state = 0
                elif self.runner_state == 1 and rpm > up2:
                    self.runner_state = 2
                elif self.runner_state == 2 and rpm < down2:
                    self.runner_state = 1
                runner = (0.0, 0.48, 1.0)[self.runner_state]
            else:
                self.runner_state = 0
                runner = 0.0
            formant_base, formant_runner, formant_load = voice["formant"]
            formant_target = formant_base + formant_runner * runner + formant_load * load
            self.runner_formant += 0.12 * (formant_target - self.runner_formant)
            formant = self.runner_formant
            intake_tone = self._osc(0, np.full(n, formant), n)
            intake_noise = self._one_pole(self.rng.standard_normal(n), 0.35 + 0.25 * runner, self.intake_noise_lp)
            radius = 0.91
            omega = TWO_PI * formant / self.sr
            resonant, self.intake_filter = lfilter(
                (1 - radius, 0, 0),
                (1, -2 * radius * np.cos(omega), radius * radius),
                intake_noise, zi=self.intake_filter)
            intake_gain = (throttle_line ** 1.35) * (0.018 + 0.040 * runner + 0.022 * rpm_f_line)
            intake = (intake_tone * 0.085 + resonant * 0.20 + intake_noise * 0.11) * intake_gain

            beat = 0.96 + 0.04 * np.sin(TWO_PI * 3.7 * absolute_t)
            crank = self._osc(1, rotor_line, n) * 0.012 * beat
            order4 = self._osc(2, rotor_line * 4.0, n) * 0.0038
            tick = self._one_pole(self.rng.standard_normal(n), 0.55, self.mech_tick_lp)
            tick *= (0.0020 + 0.0035 * rpm_f) * 0.85
            mechanical = (crank + order4 + tick) * (0.55 + 0.45 * rpm_f)

            accessories = (self._osc(3, rotor_line * 1.5, n) * 0.0020
                           + self._osc(4, rotor_line * 2.5, n) * 0.0014) * (0.5 + 0.5 * rpm_f)

            shaft_line = rotor_line
            output_line = shaft_line / max(float(frame.ratio), 0.1)
            final_drive = float(voice["final_drive"])
            diff_line = output_line / final_drive
            mesh_noise = self.rng.standard_normal(n) * 0.0028
            torque = float(frame.torque_direction)
            # Prefer lower mesh orders + noise; high pure mesh tones also trill under Doppler.
            transmission = (
                self._osc(5, shaft_line * 17.0, n) * 0.0060
                + mesh_noise
            ) * (0.22 + 0.58 * abs(torque))
            transmission = transmission + self._osc(8, output_line * 9.0, n) * 0.0060 * max(0.0, -torque)
            if frame.clutch < 0.82 and abs(frame.shift_mismatch) > 0.05:
                chatter = (np.tanh(np.sin(TWO_PI * (92.0 + effective_rpm * 0.004) * absolute_t) * 2.1)
                           * self.rng.standard_normal(n))
                transmission = transmission + chatter * 0.016 * (1.0 - frame.clutch) * abs(frame.shift_mismatch)
            shift_severity = ((1.0 - frame.clutch)
                             * (0.25 + abs(frame.shift_mismatch))
                             * (0.4 + 0.6 * abs(torque)))
            transmission = transmission + (
                np.sin(TWO_PI * 72.0 * absolute_t) * frame.shift_phase * shift_severity * 0.018)

        # ---- per-wheel tyres + brakes ----
        slips = np.asarray(frame.wheel_slip, dtype=np.float64)
        loads = np.asarray(frame.wheel_load, dtype=np.float64)
        contacts = np.asarray(frame.wheel_contact, dtype=np.float64)
        surface_color = (1.0, 0.70, 0.48, 0.58, 1.35, 0.84)
        speed_start = float(self.prev.speed)
        speed_line = np.linspace(speed_start, float(frame.speed), n)
        tyres = []
        brakes = []
        for w in range(4):
            slip = abs(float(slips[w]))
            load_w = float(np.clip(loads[w], 0.0, 2.0) * contacts[w])
            longitudinal = max(0.0, slip - 0.10)
            lock = max(0.0, -float(slips[w]) - 0.20) * float(frame.brake)
            lateral = max(0.0, slip - 0.18)
            unloaded = max(0.0, 0.35 - load_w) * min(1.0, float(frame.speed) / 20.0)
            color = surface_color[min(max(int(frame.wheel_surface[w]), 0), 5)] * (1.0 - 0.32 * frame.wetness)
            noise = self.rng.standard_normal(n)
            if detail:
                tyre = self._one_pole(noise, 0.28 + 0.12 * min(1.0, slip), self.tyre_lp[w])
            else:
                tyre = noise * 0.55
            tyre *= (0.038 * longitudinal + 0.028 * lateral + 0.015 * unloaded) * load_w * color
            if detail and int(frame.wheel_surface[w]) == 4:
                kerb = np.tanh(np.sin(TWO_PI * (34.0 + frame.speed * 1.85) * absolute_t) * 2.3)
                tyre = tyre + kerb * 0.007 * float(frame.kerb)
            tyres.append(tyre.astype(np.float32))

            temp = float(frame.brake_temperature[w])
            if temp > 0.72:
                self.brake_hot[w] = True
            elif temp < 0.55:
                self.brake_hot[w] = False
            scrub = noise * 0.011 * float(frame.brake) * load_w * min(1.0, float(frame.speed) / 35.0)
            if detail and self.brake_hot[w] and frame.brake > 0.05:
                squeal = np.sin(TWO_PI * (1480.0 + 110.0 * w + 460.0 * temp) * absolute_t)
                squeal *= 0.0050 * float(frame.brake)
                modulation = (0.55 + 0.45 * np.sin(TWO_PI * (8.0 + frame.speed * 0.4) * absolute_t)
                              if lock > 0.02 else 1.0)
                brakes.append((scrub + squeal * modulation).astype(np.float32))
            else:
                brakes.append(scrub.astype(np.float32))

        # ---- chassis events + suspension grit ----
        for name in self.env:
            gen = getattr(frame.events, name)
            if gen != getattr(self.seen, name):
                setattr(self.seen, name, gen)
                self.env[name] = 1.0
        chassis = np.zeros(n, dtype=np.float64)
        for name, freq, level in (
            ("shift", 78.0, 0.055),
            ("landing", 52.0, 0.135),
            ("collision", 96.0, 0.190),
            ("damage", 140.0, 0.085),
            ("surface", 118.0, 0.038),
        ):
            env = self.env[name]
            if env > 1e-5:
                decay = np.exp(-np.arange(n) / (self.sr * (0.04 if name == "shift" else 0.12)))
                if name in ("shift", "surface"):
                    severity = 1.0
                elif name == "landing":
                    severity = 0.55 + min(2.2, frame.landing_force / 3.0)
                elif name == "collision":
                    severity = 0.65 + min(2.0, frame.collision_impulse / 5.0)
                else:
                    severity = 0.7 + frame.damage
                body = np.sin(TWO_PI * freq * absolute_t)
                grit = self.rng.standard_normal(n) * 0.35
                chassis += (body + grit) * env * decay * level * 0.78 * severity
                self.env[name] = float(env * decay[-1])
        susp = np.asarray(frame.suspension_velocity, dtype=np.float64)
        top = np.clip(np.abs(susp) - 2.0, 0.0, 8.0)
        chassis += self.rng.standard_normal(n) * (
            0.008 * frame.kerb + 0.004 * frame.damage + 0.0035 * float(np.mean(top)))

        if frame.engine_state != self.prev.engine_state:
            self.lifecycle_env = 1.0
        lifecycle = np.zeros(n, dtype=np.float64)
        if self.lifecycle_env > 1e-5:
            decay = np.exp(-np.arange(n) / (self.sr * 0.28))
            freq = {0: 42.0, 1: 115.0, 2: 310.0, 3: 58.0}[int(frame.engine_state)]
            lifecycle = (np.sin(TWO_PI * freq * absolute_t)
                         + 0.35 * self.rng.standard_normal(n)) * self.lifecycle_env * decay * 0.045
            self.lifecycle_env = float(self.lifecycle_env * decay[-1])

        # ---- body wind (stem only; listener-local wind is spatial) ----
        wind = self._one_pole(self.rng.standard_normal(n), 0.08, self.wind_lp)
        wind = wind * np.minimum(0.10, speed_line * 0.00085)

        self.prev = frame
        self.clock += n
        tyre_sum = np.sum(tyres, axis=0)
        brake_sum = np.sum(brakes, axis=0)
        return dict(
            exhaust=exhaust.astype(np.float32),
            intake=np.asarray(intake, dtype=np.float32),
            mechanical=np.asarray(mechanical, dtype=np.float32),
            accessories=np.asarray(accessories, dtype=np.float32),
            transmission=np.asarray(transmission, dtype=np.float32),
            tyre_fl=tyres[0], tyre_fr=tyres[1], tyre_rl=tyres[2], tyre_rr=tyres[3],
            tyres=tyre_sum.astype(np.float32),
            brake_fl=brakes[0], brake_fr=brakes[1], brake_rl=brakes[2], brake_rr=brakes[3],
            brakes=brake_sum.astype(np.float32),
            chassis=(chassis + lifecycle).astype(np.float32),
            wind=np.asarray(wind, dtype=np.float32),
        )


class SpatialRendererV4:
    def __init__(self, sample_rate, cars, seed=VIEWER_AUDIO_SEED, propagation=True, car_names=None):
        self.sr=sample_rate; self.propagation=propagation
        names=list(car_names or [])
        self.synths=[EngineSynthV4(sample_rate,seed+i,
                                   car=(names[i] if i<len(names) else "mazda787b"))
                     for i in range(cars)]
        self.delays=[[FractionalDelay(sample_rate) for _ in range(12)] for _ in range(cars)]
        self.fallback=[[RadialDoppler() for _ in range(12)] for _ in range(cars)]
        self.reflections=[EarlyReflections(sample_rate) for _ in range(cars)]
        self.dc=DCBlocker(); self.limiter=SoftLimiter(sample_rate); self.last_cut=0; self.cut_fade=0
        self.air=np.zeros((cars,12)); self.last_peak=0
        self.source_gain=np.zeros((cars,12,2)); self.voice_gain=np.zeros(cars)
        self.previous_out=np.zeros((PREFERRED_BLOCK,2),np.float32)
        self.local_rng=np.random.default_rng(seed+9000); self.wind_state=0.0
        self.presence_lp=np.zeros(2,dtype=np.float64)
    def reset(self, frames=None):
        for i,s in enumerate(self.synths):
            f=frames[i] if frames else None; s.reset(f)
            for d in self.delays[i]: d.reset()
            for d in self.fallback[i]: d.reset()
            self.reflections[i].reset()
        self.cut_fade=int(.1*self.sr)
        self.presence_lp.fill(0)
    @staticmethod
    def _source_position(frame, longitudinal, lateral):
        c=math.cos(frame.yaw); s=math.sin(frame.yaw)
        return frame.x+c*longitudinal-s*lateral, frame.y+s*longitudinal+c*lateral
    def render(self, frames, listener, n, perspective=1, cut=0, master=.55,
               reflections=False):
        out=np.zeros((n,2),np.float32); lx,ly,lvx,lvy,lyaw=listener
        # Deterministic voice virtualization keeps dense 32-car fields within
        # their deadline while preserving the twelve acoustically nearest cars.
        voice_limit=4 if self.sr>=96000 else (7 if len(frames)>=16 else 9)
        active=set(range(len(frames)))
        if len(frames)>voice_limit:
            active=set(sorted(range(len(frames)),key=lambda k:(frames[k].x-lx)**2+(frames[k].y-ly)**2)[:voice_limit])
        local_perspective=perspective in (1,2)
        if cut != self.last_cut and not local_perspective:
            self.last_cut=cut; self.cut_fade=int(.1*self.sr)
            for ds in self.delays:
                for delay in ds: delay.initialized=False
        elif local_perspective:
            self.last_cut=cut; self.cut_fade=0
        for i,(s,f) in enumerate(zip(self.synths,frames)):
            selected=i in active; old_voice=self.voice_gain[i]; target_voice=1.0 if selected else 0.0
            step=1-math.exp(-n/(self.sr*.08)); new_voice=old_voice+(target_voice-old_voice)*step
            self.voice_gain[i]=new_voice
            if not selected and max(old_voice,new_voice)<.001: continue
            voice_ramp=np.linspace(old_voice,new_voice,n)
            detailed=(selected and self.sr<96000 and (len(frames)<=4 or i<4))
            stems=s.synthesize_stems(f,n,detail=detailed)
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
                if local_perspective:
                    pan=float(np.clip(lateral/1.8,-.45,.45))
                    # Slightly quieter onboard so cuts to trackside don't feel like mute.
                    atten={"exhaust":.62,"intake":.48,"mechanical":.40,"tyre":.32}[kind]
                    direct=1.0
                else:
                    rel=math.atan2(dy,dx)-lyaw; pan=float(np.clip(math.sin(rel),-1,1))
                    # Flatter scenic falloff: readable roar at 50–120 m without
                    # blowing up the closest pass (broadcast makeup curve).
                    atten=1/(1+(dist/26)**1.05)
                    bearing=math.atan2(ly-sy,lx-sx)-f.yaw
                    if kind=="exhaust": direct=.28+.72*max(0,math.cos(bearing-math.pi))
                    elif kind=="intake": direct=.40+.60*max(0,math.cos(bearing-.35))
                    elif kind=="tyre": direct=.70
                    else: direct=.58
                pl=math.sqrt((1-pan)*.5); pr=math.sqrt((1+pan)*.5)
                if local_perspective: propagated=bus
                elif self.propagation: propagated=self.delays[i][j].process(bus,dist)
                else:
                    radial=((f.vx-lvx)*dx+(f.vy-lvy)*dy)/max(dist,.1)
                    propagated=self.fallback[i][j].process(bus,radial)
                # Air absorption still darkens with distance, but keeps more bark.
                alpha=.72 if local_perspective else max(.045,min(.72,210/(210+dist*dist*0.55)))
                propagated,state=lfilter((alpha,),(1.0,-(1.0-alpha)),propagated,zi=(self.air[i,j],))
                self.air[i,j]=state[0]; weighted=propagated*atten*direct
                old_l,old_r=self.source_gain[i,j]
                gain_l=np.linspace(old_l,pl,n)*voice_ramp; gain_r=np.linspace(old_r,pr,n)*voice_ramp
                self.source_gain[i,j]=pl,pr
                out[:,0]+=weighted*gain_l; out[:,1]+=weighted*gain_r; car_reflect+=weighted*voice_ramp
            reflected=self.reflections[i].process(car_reflect,f.acoustic_preset)
            if reflections and not local_perspective:
                # Keep the forest return quiet — a wet reflection bed comb-filters
                # under moving chase/trackside/aerial cameras into a Jetsons trill.
                out[:,0]+=reflected*.48; out[:,1]+=reflected*.48
        if not local_perspective:
            # Mild external presence shelf — keeps midrange body on chase/trackside
            # without amplifying delay-Doppler warble into a whistle.
            alpha_p=.10
            for ch in range(2):
                low,zi=lfilter((alpha_p,),(1.0,-(1.0-alpha_p)),out[:,ch],zi=(self.presence_lp[ch],))
                self.presence_lp[ch]=zi[0]
                out[:,ch]+= (out[:,ch]-low)*0.18
        if perspective in (1,2):
            listener_speed=math.hypot(lvx,lvy); noise=self.local_rng.standard_normal(n)
            alpha=.035; wind=np.empty(n,np.float32)
            for k in range(n):
                self.wind_state += alpha*(noise[k]-self.wind_state); wind[k]=self.wind_state
            out += (wind*min(.035,listener_speed*.00034))[:,None]
        if self.cut_fade:
            total=max(1,int(.1*self.sr)); m=min(n,self.cut_fade); done=total-self.cut_fade
            theta=np.linspace(done/total*math.pi/2,(done+m)/total*math.pi/2,m,endpoint=False)
            out[:m]*=np.sin(theta)[:,None]; self.cut_fade=max(0,self.cut_fade-n)
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


def _worker(shared, stats, cars, car_names=None):
    try:
        import sounddevice as sd
        requested=os.environ.get("SUPRA_AUDIO_DEVICE")
        device=(int(requested) if requested and requested.lstrip("-").isdigit()
                else (requested or None))
        dev=sd.query_devices(device, "output"); sr=int(round(dev["default_samplerate"]))
        renderer=SpatialRendererV4(sr,cars,car_names=car_names)
        initialized=False; started=time.perf_counter(); durations=np.zeros(256); duration_i=0; reset_requested=0.0
        last_head=None; last_vehicles=None
        def callback(outdata, frames, timing, status):
            nonlocal initialized,duration_i,reset_requested,last_head,last_vehicles
            begin=time.perf_counter(); head,vehicles,retries=_read_snapshot(shared,cars); stats[9]+=retries
            if head is None:
                stats[8]+=1
                if last_head is None:
                    outdata.fill(0); return
                head,vehicles=last_head,last_vehicles
            else:
                last_head,last_vehicles=head,vehicles
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
        self._diag_last=0.0; self._diag_rpm=None
    def start(self,racers=None):
        if not racers: return self
        self.stop(); vehicles=[r.veh if hasattr(r,"veh") else r for r in racers]; self.num_cars=len(vehicles)
        self._vehicles=vehicles
        self.shared_data=mp.Array('d',HEADER+self.num_cars*CAR_STRIDE,lock=False); self.stats_data=mp.Array('d',16,lock=False)
        self.stats_data[5]=-1.0  # measured by the offline warm-up allocation gate
        self.shared_data[0]=1; self.shared_data[1]=self.master; self.shared_data[2]=SCHEMA_VERSION; self.shared_data[3]=0
        self.shared_data[4]=1; self.shared_data[12]=1; self.adapter.reset(vehicles)
        self.shared_data[6:11]=[vehicles[0].x,vehicles[0].y,0,0,vehicles[0].yaw]
        for i,v in enumerate(vehicles):
            frame=self.adapter.frame(v,0.0); start=HEADER+i*CAR_STRIDE
            self.shared_data[start:start+CAR_STRIDE]=_encode(frame)
        # Car identity crosses the process boundary once, at start: the worker
        # voices each synth from VOICE_PROFILES (default = validated 787B).
        car_names=[getattr(v.spec,"name","mazda787b") for v in vehicles]
        self.process=mp.Process(target=_worker,args=(self.shared_data,self.stats_data,self.num_cars,car_names),daemon=True); self.process.start(); self.ok=True
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
        if os.environ.get("SUPRA_AUDIO_DIAGNOSTICS") == "1" and vehicles:
            now=time.monotonic()
            rpm=float(getattr(vehicles[0],"rpm",0.0)); delta=0.0 if self._diag_rpm is None else rpm-self._diag_rpm
            if now-self._diag_last>=1.0:
                m=self.metrics
                print(f"[audio-v5 diag] rpm={rpm:.0f} drpm={delta:+.0f} gear={getattr(vehicles[0],'gear',0)} "
                      f"cut={self._cut} cb={m.get('callback_count',0)} underrun={m.get('underruns',0)} "
                      f"stale={m.get('stale_frames',0)} retry={m.get('seqlock_retries',0)} "
                      f"p99ms={m.get('callback_p99',0)*1000:.2f} gainred={m.get('limiter_reduction_db',0):.2f}",flush=True)
                self._diag_last=now
            self._diag_rpm=rpm
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
