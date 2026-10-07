"""
Real-time engine audio — synthesised from scratch, no samples. (v3)

A background `sounddevice` output stream runs a synthesis callback that builds
each audio buffer live from the car's telemetry, customized per car type:

  * Engine     - additive inline-six harmonics for Supra (2JZ/B58 note) and Skyline (RB26 note),
                 and rotary harmonics for RX-7 (2-rotor 13B note), distorted via waveshaping.
  * Turbo      - single or chorused twin-turbo whines.
  * Blow-off   - classic flutter for Supra, quick high-pitched sneeze for RX-7,
                 and overlapping dual release for Skyline.
  * Tyre       - squeal + noise scaled by tyre slide.
  * Backfire   - bassy pops for Supra, highly frequent firecracker snaps for RX-7,
                 and metallic snaps for Skyline.

Thread model: the sim thread calls `update(...)` to set target parameters and
fire events; the audio thread reads them in `_callback`. Continuous params are
ramped per-buffer to avoid zipper noise; phases/filters persist across buffers.
"""
from __future__ import annotations

import os
import threading

import numpy as np

SR = 44100

# ~46 ms buffer. The synth runs in a Python callback (needs the GIL), so a busy
# 120 Hz render thread can starve it -> underrun crackle. A larger block gives
# the callback more deadline slack (PortAudio queues several at latency="high"),
# which is the main defence against that contention. Per-buffer param smoothing
# below is tuned for this block size so the engine note stays responsive.
BLOCK = 2048
TWO_PI = 2.0 * np.pi
MAZDA_787B_GEAR_RATIOS = np.array(
    [3.27451, 2.28028, 1.57271, 1.16654, 0.96318], dtype=float)


def _clean_float(value, default, lo=None, hi=None):
    try:
        out = float(value)
    except (TypeError, ValueError):
        out = float(default)
    if not np.isfinite(out):
        out = float(default)
    if lo is not None or hi is not None:
        low = -np.inf if lo is None else float(lo)
        high = np.inf if hi is None else float(hi)
        out = float(np.clip(out, low, high))
    return out


class EngineSynth:
    def __init__(self):
        self.car_name = "supra"
        self.rng = np.random.default_rng()
        self.t_redline = 6800.0
        self.c_redline = 6800.0

        # --- target params (set by the sim thread) ---
        self.t_rpm = 850.0
        self.t_throttle = 0.0
        self.t_boost = 0.0
        self.t_screech = 0.0       # 0..1 slide intensity
        self.t_rpm_frac = 0.0
        self.t_speed = 0.0
        self.t_gear = 1
        self.t_num_gears = 6
        self.t_gear_ratio = float(MAZDA_787B_GEAR_RATIOS[0])

        # --- ramped (smoothed) current values used in the callback ---
        self.c_rpm = 850.0
        self.c_throttle = 0.0
        self.c_boost = 0.0
        self.c_screech = 0.0
        self.c_gear_ratio = self.t_gear_ratio

        # --- persistent oscillator / filter / envelope state ---
        self.eng_phase = 0.0
        self.eng_formant_phase = 0.0
        self.shift_phase = 0.0
        self.tur_phase = 0.0
        self.tur_phase2 = 0.0      # Skyline dual-turbo chorus
        self.squeal_phase = 0.0
        self.clock = 0            # running sample count (for LFOs)

        # BOV envelopes & phases
        self.bov_env = 0.0
        self.bov_env2 = 0.0        # Skyline secondary BOV
        self.bov_flutter_phase = 0.0
        self.bov_flutter_phase2 = 0.0
        self.bov_tone_phase = 0.0
        self.bov_tone_phase2 = 0.0

        self.bov_boost = 0.0           # boost snapshot when BOV fired
        self.bov_rpm_frac = 0.0        # rpm fraction snapshot for pitch/rate
        self.crackle_env = 0.0
        self.shift_env = 0.0
        self.shift_dir = 0.0
        self._crackle_cd = 0           # frames until another pop may fire (spacing)
        self._noise_tail = 0.0
        self.pop_hist = np.zeros(10)   # crackle filter state

        # environment / body layers (wind, road rumble, intake resonance)
        self.intake_phase = 0.0
        self.thump_phase = 0.0
        self._wind_tail = np.zeros(31)   # box-filter history for wind noise
        self._road_tail = np.zeros(63)   # box-filter history for road rumble
        self._rotor_beat_phase = 0.0     # slow inter-rotor beating (787B)

        # race behaviours: limiter, downshift blip, wheelspin, kerb strikes
        self.blip_env = 0.0              # downshift rev-blip pitch bump
        self.t_wspin = 0.0               # rear wheelspin roughness target
        self.c_wspin = 0.0
        self.t_kerb = 0.0                # kerb-strike intensity target
        self.c_kerb = 0.0
        self.kerb_phase = 0.0
        self._kerb_tail = np.zeros(7)

        # --- event edge-detection memory ---
        self._prev_throttle = 0.0
        self._prev_gear = 1
        self._suppress_gear_edge = True
        self._bov_armed = False     # set while on boost+throttle, fires on lift

    # ------------------------------------------------------------------ #
    def reset(self):
        """Resync to a freshly (re)spawned car. Resets active envelopes and state."""
        self._prev_gear = 1
        self._suppress_gear_edge = True
        self._prev_throttle = 0.0
        self._bov_armed = False
        self.crackle_env = 0.0
        self.bov_env = 0.0
        self.bov_env2 = 0.0
        self.shift_env = 0.0
        self.shift_dir = 0.0
        self.blip_env = 0.0
        self.t_wspin = self.c_wspin = 0.0
        self.t_kerb = self.c_kerb = 0.0
        self.t_gear = 1
        self.c_gear_ratio = self.t_gear_ratio
        self.t_screech = self.c_screech = 0.0

    def update(self, veh, throttle: float):
        """Push the latest car state from the sim thread (cheap, lock-light)."""
        speed = _clean_float(getattr(veh, "speed", 0.0), 0.0, 0.0)
        rpm = _clean_float(getattr(veh, "rpm", self.t_rpm), self.t_rpm, 100.0, 14000.0)
        boost = _clean_float(getattr(veh, "boost", 0.0), 0.0, 0.0, 2.0)
        throttle = _clean_float(throttle, 0.0, 0.0, 1.0)
        slip_angle = _clean_float(getattr(veh, "slip_angle", 0.0), 0.0)

        wheel_sr = getattr(veh, "wheel_sr", [0.0, 0.0, 0.0, 0.0])
        rear_l = _clean_float(wheel_sr[2] if len(wheel_sr) > 2 else 0.0, 0.0)
        rear_r = _clean_float(wheel_sr[3] if len(wheel_sr) > 3 else rear_l, rear_l)
        rear_slide = max(abs(rear_l), abs(rear_r))
        slip = abs(np.degrees(slip_angle)) / 35.0
        screech = float(np.clip(max(rear_slide - 0.15, slip - 0.2) * 1.3, 0.0, 1.0))
        if speed < 2.0:
            screech = 0.0

        spec = getattr(veh, "spec", None)
        redline = _clean_float(getattr(spec, "redline_rpm", self.t_redline), self.t_redline, 1000.0, 14000.0)
        gear_ratios = getattr(spec, "gear_ratios", MAZDA_787B_GEAR_RATIOS)
        if gear_ratios is None or len(gear_ratios) == 0:
            gear_ratios = MAZDA_787B_GEAR_RATIOS
        num_gears = max(1, len(gear_ratios))
        raw_gear = _clean_float(getattr(veh, "gear", self.t_gear), self.t_gear, 1, num_gears)
        gear = int(raw_gear)
        gear_ratio = _clean_float(gear_ratios[gear - 1], MAZDA_787B_GEAR_RATIOS[min(gear - 1, 5)], 0.2, 6.0)

        self.t_rpm = rpm
        self.t_throttle = throttle
        self.t_boost = boost
        self.t_rpm_frac = float(rpm / redline)
        self.t_screech = screech
        self.t_speed = speed
        self.t_gear = gear
        self.t_num_gears = num_gears
        self.t_gear_ratio = gear_ratio

        # rear wheelspin: rough, chewing power delivery under throttle
        if throttle > 0.4 and speed > 3.0:
            self.t_wspin = float(np.clip((rear_slide - 0.28) * 1.9, 0.0, 1.0))
        else:
            self.t_wspin = 0.0
        # kerb strike intensity (viewer sets veh._snd_kerb; absent -> 0)
        self.t_kerb = _clean_float(getattr(veh, "kerb", 0.0), 0.0, 0.0, 1.0)

        # event detection
        self.car_name = getattr(spec, "name", "supra")
        self.t_redline = redline

        if self._suppress_gear_edge:
            self._prev_gear = gear
            self._suppress_gear_edge = False

        if gear != self._prev_gear and speed > 5.0:
            self.shift_env = max(self.shift_env, 0.85)
            self.shift_dir = 1.0 if gear > self._prev_gear else -1.0
            if gear < self._prev_gear:
                # heel-toe rev blip: short pitch flare while the dogs engage
                self.blip_env = max(self.blip_env, 0.85)

        # turbo flutter: arm while on boost+throttle, fire when the driver lifts.
        if throttle > 0.55 and boost > 0.30:
            self._bov_armed = True
        if self._bov_armed and throttle < 0.25:
            self.bov_boost = float(np.clip(boost, 0.25, 1.0))
            self.bov_rpm_frac = float(np.clip(rpm / redline, 0.0, 1.0))
            self.bov_env = self.bov_boost
            self.bov_env2 = self.bov_boost # Fire secondary BOV for Skyline
            self._bov_armed = False

        # backfire / overrun crackle — spaced out by a cooldown so it pops occasionally
        self._crackle_cd = max(0, self._crackle_cd - 1)
        if self._crackle_cd == 0:
            if gear < self._prev_gear and speed > 5.0:
                # downshift bang
                self.crackle_env = max(self.crackle_env, 0.7)
                self._crackle_cd = 10
            elif throttle < 0.1 and rpm > 3500:
                # overrun pop: rotaries pop significantly more frequently
                prob = 0.08 if self.car_name == "mazda787b" else (0.12 if self.car_name == "rx7" else 0.05)
                if self.rng.random() < prob:
                    self.crackle_env = max(self.crackle_env, self.rng.uniform(0.25, 0.5))
                    self._crackle_cd = 18 if self.car_name == "mazda787b" else (6 if self.car_name == "rx7" else 14)
            elif rpm >= redline * 1.015 and throttle > 0.5:
                # banging on the limiter: rapid ignition-cut snaps
                if self.rng.random() < 0.30:
                    self.crackle_env = max(self.crackle_env, self.rng.uniform(0.30, 0.55))
                    self._crackle_cd = 4
        self._prev_throttle = throttle
        self._prev_gear = gear

    # ------------------------------------------------------------------ #
    def synthesize(self, frames, pitch_factor=1.0):
        n = frames
        idx = np.arange(n)
        tline = (self.clock + idx) / SR
        self.clock += n
        pitch_factor = _clean_float(pitch_factor, 1.0, 0.25, 4.0)

        # per-buffer smoothing toward targets. Rates are higher than they look
        # because buffers now arrive ~half as often (2048 vs 1024) — these keep
        # the same ~response time as before so rpm/throttle don't feel laggy.
        def ramp(cur, tgt, rate=0.25):
            return cur + (tgt - cur) * rate

        target_redline = _clean_float(self.t_redline, self.c_redline, 1000.0, 14000.0)
        self.c_redline = ramp(_clean_float(self.c_redline, target_redline, 1000.0, 14000.0), target_redline, 1.0)
        target_rpm = _clean_float(self.t_rpm, self.c_rpm, 100.0, 14000.0)
        rpm_start = _clean_float(self.c_rpm, target_rpm, 100.0, 14000.0)
        self.c_rpm = ramp(rpm_start, target_rpm, 0.50)
        rpm_line = np.linspace(rpm_start, self.c_rpm, n)

        self.c_throttle = ramp(_clean_float(self.c_throttle, self.t_throttle, 0.0, 1.0),
                               _clean_float(self.t_throttle, 0.0, 0.0, 1.0), 0.50)
        self.c_boost = ramp(_clean_float(self.c_boost, self.t_boost, 0.0, 2.0),
                            _clean_float(self.t_boost, 0.0, 0.0, 2.0), 0.50)
        self.c_screech = ramp(_clean_float(self.c_screech, self.t_screech, 0.0, 1.0),
                              _clean_float(self.t_screech, 0.0, 0.0, 1.0), 0.60)
        self.c_gear_ratio = ramp(_clean_float(self.c_gear_ratio, self.t_gear_ratio, 0.2, 6.0),
                                 _clean_float(self.t_gear_ratio, 1.0, 0.2, 6.0), 0.65)

        thr, boost = self.c_throttle, self.c_boost
        rpm_frac_line = np.clip(rpm_line / max(self.c_redline, 1000.0), 0.0, 1.15)
        rpm_frac = float(rpm_frac_line[-1])
        car = self.car_name

        # ---------------- engine ----------------
        f0_line = np.maximum(rpm_line, 200.0) / 60.0          # crank Hz (floor for audibility)
        # downshift rev-blip: brief heel-toe pitch flare over the telemetry rpm
        if self.blip_env > 1e-3 and self.car_name != "mazda787b":
            blip = self.blip_env * np.exp(-idx / (0.16 * SR))
            f0_line = f0_line * (1.0 + 0.07 * blip)
            self.blip_env *= float(np.exp(-n / (0.16 * SR)))
        phase = self.eng_phase + np.cumsum(TWO_PI * (f0_line * pitch_factor) / SR)
        self.eng_phase = float(phase[-1] % TWO_PI)
        eng = np.zeros(n) if not hasattr(self, '_eng') or len(self._eng) != n else self._eng
        eng.fill(0.0)
        self._eng = eng

        # Load chassis-specific engine harmonic structures
        if car == "mazda787b":
            # --- Specialized 4-Rotor Wankel DSP (R26B) ---
            # Four rotors fire every 90 degrees of eccentric-shaft rotation. Build the
            # voice from continuous harmonics of that firing order; no gated noise or
            # discontinuous pulses, which were what made the car tick instead of sing.
            fire_phase = (4.0 * phase) % TWO_PI
            powerband = np.clip((rpm_frac_line - 0.58) / 0.36, 0.0, 1.0)
            powerband = powerband * powerband * (3.0 - 2.0 * powerband)
            power = float(powerband[-1])

            # Load model: on-throttle pull vs closed-throttle overrun. A Group C car
            # with open peripheral-port pipes never goes quiet on the brakes — the
            # voice changes (rounder, gargling) instead of vanishing.
            overrun = float(np.clip((0.16 - thr) / 0.16, 0.0, 1.0)
                            * np.clip((rpm_frac - 0.22) / 0.30, 0.0, 1.0))

            harmonic_orders = np.array([1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 20], dtype=float)
            low_amps = np.array([0.92, 0.74, 0.56, 0.40, 0.28, 0.20, 0.12, 0.075, 0.045, 0.030, 0.020, 0.010])
            high_amps = np.array([0.40, 0.58, 0.78, 0.92, 0.82, 0.68, 0.46, 0.31, 0.20, 0.14, 0.10, 0.05])
            # Overrun tilts the spectrum back down: closed throttle kills the intake
            # scream but leaves the exhaust pulse train thumping.
            over_amps = np.array([1.00, 0.62, 0.70, 0.34, 0.40, 0.18, 0.14, 0.06, 0.04, 0.02, 0.012, 0.006])
            amps = low_amps[:, None] * (1.0 - powerband) + high_amps[:, None] * powerband
            amps = amps * (1.0 - overrun) + over_amps[:, None] * overrun
            rotor_tone = np.sum(amps * np.sin(harmonic_orders[:, None] * fire_phase), axis=0)
            rotor_tone /= np.maximum(np.sum(amps, axis=0), 1e-6)

            # Four rotors are never perfectly matched: give the crank orders a slow
            # beating pattern so the note breathes instead of sounding like a synth.
            beat_f = 1.4 + 5.2 * rpm_frac
            beat_phase = self._rotor_beat_phase + TWO_PI * beat_f * (idx + 1) / SR
            self._rotor_beat_phase = float(beat_phase[-1] % TWO_PI)
            imbalance = (0.10 * np.sin(phase + 0.3) + 0.055 * np.sin(2.0 * phase - 0.9)
                         + 0.030 * np.sin(3.0 * phase + 1.7)) * (0.6 + 0.4 * np.sin(beat_phase))

            # Telescopic intake / peripheral-port resonance: aim the formant at a
            # harmonic of the four-rotor firing frequency so the scream climbs with
            # RPM without introducing an unrelated metallic ring.
            fire_freq = max(4.0 * (self.c_rpm / 60.0), 1.0)
            formant_freq = 1300.0 + power * 3550.0
            hf = formant_freq / fire_freq
            k = int(np.floor(hf)); frac = hf - k
            k = int(np.clip(k, 2, 30))
            formant = ((1.0 - frac) * np.sin(k * fire_phase)
                       + frac * np.sin((k + 1) * fire_phase))
            port_breath = 0.76 + 0.24 * (0.5 + 0.5 * np.sin(fire_phase - 0.35)) ** 2.0
            throat = 0.28 * np.sin((k + 2) * fire_phase + 0.25) + 0.16 * np.sin((k + 4) * fire_phase - 0.6)
            formant = (formant + throat * powerband) * port_breath * (0.34 + 0.62 * powerband)
            formant *= (1.0 - 0.85 * overrun)   # intake scream dies with the throttle plate

            # Second, fixed body resonance: the carbon tub / exhaust collector rings
            # around ~640 Hz regardless of rpm. Snap it to the nearest firing harmonic
            # too, so it stays phase-locked and never sounds like a separate whistle.
            hb = 640.0 / fire_freq
            kb = int(np.clip(np.floor(hb), 1, 24)); fb = float(np.clip(hb - kb, 0.0, 1.0))
            body_res = ((1.0 - fb) * np.sin(kb * fire_phase + 0.5)
                        + fb * np.sin((kb + 1) * fire_phase + 0.5))
            body_res *= 0.20 + 0.16 * thr + 0.10 * overrun

            # Low-RPM rotary brap, shaped as tone instead of pulse-noise ticks.
            idle_weight = float(np.clip((0.42 - rpm_frac) / 0.32, 0.0, 1.0))
            brap = idle_weight * (0.26 * np.sin(fire_phase + 0.4)
                                  + 0.18 * np.sin(2.0 * fire_phase - 0.7)
                                  + 0.08 * np.sin(3.0 * fire_phase + 0.2))

            mech = 0.30 * np.sin(phase) + 0.15 * np.sin(2.0 * phase) + imbalance
            open_pipe = np.tanh(
                1.30 * np.sin(fire_phase - 0.18)
                + 0.74 * np.sin(2.0 * fire_phase + 0.42)
                + 0.36 * np.sin(3.0 * fire_phase - 0.55)
            )
            wide_scream = np.tanh((rotor_tone + 0.32 * formant + 0.52 * open_pipe) * (1.5 + 2.4 * powerband))
            raw_eng = (0.96 * rotor_tone + 0.38 * formant + body_res
                       + 0.52 * open_pipe * powerband + 0.38 * wide_scream * powerband
                       + 0.30 * open_pipe * overrun + brap + mech)

            turbulence = self.rng.standard_normal(n)
            turbulence = np.convolve(turbulence, np.ones(9) / 9.0, mode="same")
            exhaust_gate = (0.48 + 0.52 * (0.5 + 0.5 * np.sin(fire_phase - 0.4)) ** 0.75)
            raw_eng += turbulence * exhaust_gate * (0.04 + 0.13 * thr + 0.10 * overrun) * (0.20 + 0.80 * powerband + 0.6 * overrun)

            # Asymmetric straight-pipe overdrive — positive pressure peaks clip harder
            # than the negative side. DC is removed after the full mix. Overrun keeps
            # some drive so engine braking still barks.
            drive = 2.6 + 5.2 * thr + 1.7 * power + 1.4 * overrun
            eng = np.where(raw_eng > 0, np.tanh(raw_eng * drive),
                           np.tanh(raw_eng * drive * 0.6))

            # Inter-rotor and intake shimmer: audible life, not a separate flutter.
            warble = 1.0 + 0.026 * np.sin(TWO_PI * (5.0 + 5.0 * rpm_frac) * tline)
            warble += 0.014 * np.sin(TWO_PI * (9.0 + 3.0 * rpm_frac) * tline + 0.7)
            # Overrun gargle: slow rough amplitude chop from unburnt charge lighting
            # in the pipes.
            if overrun > 0.05:
                warble += overrun * 0.10 * np.sin(TWO_PI * 23.0 * tline + 1.3) \
                          * (0.5 + 0.5 * np.sin(TWO_PI * 7.0 * tline))
            eng *= warble

            # Straight-cut Group C gearbox sheen. Keep it under the exhaust voice:
            # audible gear color, but not a separate high-pitched squeal.
            gear = int(np.clip(self.t_gear, 1, max(1, self.t_num_gears)))
            gear_norm = (gear - 1) / max(1, self.t_num_gears - 1)
            ratio_norm = self.c_gear_ratio / float(MAZDA_787B_GEAR_RATIOS[0])
            mesh_teeth = 13.0 + 2.15 * gear
            gear_whine_f = f0_line * mesh_teeth * (0.72 + 0.28 * np.sqrt(max(ratio_norm, 0.05))) * pitch_factor
            gear_whine_f = np.clip(gear_whine_f, 220.0, 5200.0)
            gear_phase = self.eng_formant_phase + np.cumsum(TWO_PI * gear_whine_f / SR)
            self.eng_formant_phase = float(gear_phase[-1] % TWO_PI)
            gear_whine = np.sin(gear_phase) + 0.22 * np.sin(2.0 * gear_phase + 0.2)
            # Straight-cut gears whine loudest under load, but also sing on the
            # overrun when the wheels drive the box (coast whine).
            gear_load = (0.25 + 0.75 * max(thr, 0.85 * overrun)) * (0.35 + 0.65 * rpm_frac) * (1.05 - 0.20 * gear_norm)
            eng += gear_whine * (0.004 + 0.018 * gear_load)

            if self.shift_env > 1e-4:
                decay_t = 0.075
                env = self.shift_env * np.exp(-idx / (decay_t * SR))
                chirp_f = (1500.0 + 260.0 * gear + 650.0 * power) * (1.0 + 0.16 * self.shift_dir * env)
                shift_phase = self.shift_phase + np.cumsum(TWO_PI * (chirp_f * pitch_factor) / SR)
                self.shift_phase = float(shift_phase[-1] % TWO_PI)
                dogbox = (np.sin(shift_phase) + 0.25 * np.sin(2.0 * shift_phase)) * env
                dogbox += self.rng.standard_normal(n) * env * 0.20
                eng += dogbox * 0.036
                # Driveline thump: the dog engagement kicks the whole car — a short
                # low-frequency knock under the chirp sells the physical shift.
                thump_f = 62.0 if self.shift_dir > 0 else 74.0
                tph = self.thump_phase + np.cumsum(np.full(n, TWO_PI * thump_f / SR))
                self.thump_phase = float(tph[-1] % TWO_PI)
                eng += np.sin(tph) * env * env * 0.09
                self.shift_env *= float(np.exp(-n / (decay_t * SR)))

        else:
            # --- Standard Additive Piston Engine DSP ---
            if car == "rx7":
                # Peaky metallic rotary (1.5 and 4.5 order half-harmonics)
                orders = np.array([1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 6.0], dtype=float)
                amps = np.array([0.60, 0.45, 1.00, 0.75, 0.50, 0.35, 0.20], dtype=float)
                drive = 2.4 + 5.6 * thr  # High waveshaping distortion for metallic rotary buzz
            elif car == "skyline":
                # RB26 high-pitched mechanical scream (5th & 8th orders)
                orders = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 12.0], dtype=float)
                amps = np.array([0.35, 0.50, 1.00, 0.60, 0.80, 0.45, 0.30], dtype=float)
                drive = 1.6 + 4.2 * thr
            elif car == "lr4":
                # NA 5.0 V8 — deep burble dominated by the 4th order (8 cyl, 4 firings
                # per rev), low rumble, smoother (less waveshaping) than the JDM turbos.
                orders = np.array([1.0, 2.0, 4.0, 6.0, 8.0, 10.0], dtype=float)
                amps = np.array([0.55, 0.85, 1.00, 0.55, 0.35, 0.20], dtype=float)
                drive = 1.4 + 3.0 * thr
            elif car == "f150":
                # NA 5.0 Coyote V8 — deeper truck idle than the sports cars, but a
                # cleaner, harder upper-rpm bark than the LR4.
                orders = np.array([0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0], dtype=float)
                amps = np.array([0.30, 0.58, 0.82, 1.00, 0.58, 0.40, 0.18], dtype=float)
                drive = 1.5 + 3.6 * thr
            else:
                # Supra Inline-6 raspy harmonics
                orders = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0], dtype=float)
                amps = np.array([0.45, 0.55, 1.00, 0.55, 0.50, 0.34, 0.24], dtype=float)
                drive = 1.5 + 4.0 * thr

            for order, amp in zip(orders, amps):
                eng += amp * np.sin(order * phase)
            eng /= amps.sum()
            # V8 idle lope: uneven cadence at low rpm from the cross-plane firing gaps.
            if car in ("lr4", "f150"):
                lope = float(np.clip((0.30 - rpm_frac) / 0.30, 0.0, 1.0))
                if lope > 0.0:
                    eng *= 1.0 + 0.16 * lope * np.sin(0.5 * phase + 0.8)
            # Exhaust pressure pulses clip asymmetrically on every engine.
            eng = np.where(eng > 0, np.tanh(eng * drive), np.tanh(eng * drive * 0.72))

        # induction/combustion noise (richer/raspier on rotaries)
        if car == "mazda787b":
            noise_weight = 0.045 + 0.075 * thr   # tonal scream first; just enough
                                                 # combustion air to keep it alive
        elif car == "rx7":
            noise_weight = 0.16 + 0.30 * thr
        else:
            noise_weight = 0.12 + 0.25 * thr
        eng += noise_weight * self.rng.standard_normal(n) * (0.3 + 0.7 * rpm_frac) * 0.4

        eng_gain = (0.16 + 0.55 * thr) * (0.45 + 0.55 * rpm_frac)
        if car == "rx7":
            eng_gain *= 1.15
        elif car == "mazda787b":
            # Open pipes never go quiet: engine braking keeps most of the volume.
            over_floor = (0.16 + 0.40 * float(np.clip((0.16 - thr) / 0.16, 0.0, 1.0))
                          * float(np.clip((rpm_frac - 0.22) / 0.30, 0.0, 1.0)))
            eng_gain = max(eng_gain, over_floor * (0.45 + 0.55 * rpm_frac))
            eng_gain *= 1.45  # Group C loudness
        eng *= eng_gain

        # rev limiter: hard ignition-cut chop when the telemetry pins past
        # redline (the crackle path above adds the snaps between cuts)
        lim = np.clip((rpm_frac_line - 1.015) / 0.05, 0.0, 1.0)
        if float(lim[-1]) > 0.0 or float(lim[0]) > 0.0:
            gate = np.where(np.sin(TWO_PI * 13.5 * tline) > -0.2, 1.0, 0.28)
            eng *= 1.0 - lim * (1.0 - gate)

        # rear wheelspin: the power delivery chews instead of pulling clean
        self.c_wspin = ramp(_clean_float(self.c_wspin, self.t_wspin, 0.0, 1.0),
                            _clean_float(self.t_wspin, 0.0, 0.0, 1.0), 0.45)
        if self.c_wspin > 1e-3:
            chew = np.sin(TWO_PI * (26.0 + 34.0 * rpm_frac) * tline)
            eng *= 1.0 + 0.14 * self.c_wspin * chew

        # ---------------- turbo whine ----------------
        turbo = np.zeros(n, dtype=np.float32) if not hasattr(self, '_turbo') or len(self._turbo) != n else self._turbo
        self._turbo = turbo
        turbo.fill(0.0)

        if car in ("lr4", "f150", "mazda787b"):
            pass  # NA engines
        elif car == "rx7":
            # Buzzy high-frequency single spool sequential
            ftur = 2200.0 + boost * 5000.0 + rpm_frac * 2400.0
            tphase = self.tur_phase + np.cumsum(np.full(n, TWO_PI * (ftur * pitch_factor) / SR))
            self.tur_phase = float(tphase[-1] % TWO_PI)
            flutter = 1.0 + (0.5 if boost > 0.6 else 0.0) * np.sin(TWO_PI * 62.0 * tline)
            turbo = np.sin(tphase) * (boost ** 1.6) * 0.15 * flutter
        elif car == "skyline":
            # Chorused dual-turbo spool (two offset frequency whines)
            ftur1 = 1700.0 + boost * 4000.0 + rpm_frac * 2000.0
            ftur2 = 1740.0 + boost * 4070.0 + rpm_frac * 2030.0 # slight offset

            tphase1 = self.tur_phase + np.cumsum(np.full(n, TWO_PI * (ftur1 * pitch_factor) / SR))
            self.tur_phase = float(tphase1[-1] % TWO_PI)

            tphase2 = self.tur_phase2 + np.cumsum(np.full(n, TWO_PI * (ftur2 * pitch_factor) / SR))
            self.tur_phase2 = float(tphase2[-1] % TWO_PI)

            flutter = 1.0 + (0.4 if boost > 0.6 else 0.0) * np.sin(TWO_PI * 52.0 * tline)
            turbo = (np.sin(tphase1) + np.sin(tphase2)) * 0.5 * (boost ** 1.45) * 0.17 * flutter
        else:
            # Supra default single turbo spool
            ftur = 1800.0 + boost * 4200.0 + rpm_frac * 2200.0
            tphase = self.tur_phase + np.cumsum(np.full(n, TWO_PI * (ftur * pitch_factor) / SR))
            self.tur_phase = float(tphase[-1] % TWO_PI)
            flutter = 1.0 + (0.5 if boost > 0.6 else 0.0) * np.sin(TWO_PI * 58.0 * tline)
            turbo = np.sin(tphase) * (boost ** 1.5) * 0.14 * flutter

        # -------- blow-off / turbo flutter ("stu-tu-tu-tu") --------
        bov = np.zeros(n) if not hasattr(self, '_bov') or len(self._bov) != n else self._bov
        bov.fill(0.0)
        self._bov = bov
        if self.bov_env > 1e-3 and car not in ("lr4", "f150", "mazda787b"):
            bb = self.bov_boost                                  # boost snapshot
            br = self.bov_rpm_frac                               # RPM snapshot

            if car == "rx7":
                # Quick, sharp sequential bypass sneeze (very short decay)
                decay_t = 0.07 + 0.13 * bb
                env = self.bov_env * np.exp(-idx / (decay_t * SR))
                env_norm = env / max(env[0], 1e-6)

                fph = self.bov_flutter_phase + np.cumsum(np.full(n, TWO_PI * (18.0 * pitch_factor) / SR))
                self.bov_flutter_phase = float(fph[-1] % TWO_PI)
                pulse = (0.5 + 0.5 * np.sin(fph)) ** 1.2

                tone_base = 280.0 + 120.0 * bb + 80.0 * br
                tone_sweep = 120.0 * bb * env_norm
                tph = self.bov_tone_phase + np.cumsum(TWO_PI * ((tone_base + tone_sweep) * pitch_factor) / SR)
                self.bov_tone_phase = float(tph[-1] % TWO_PI)
                body = np.sin(tph)

                noise = self.rng.standard_normal(n)
                bright = np.diff(noise, prepend=self._noise_tail)
                self._noise_tail = noise[-1]

                bov = (0.40 * body + 0.60 * bright) * pulse * env * 1.35
                self.bov_env *= float(np.exp(-n / (decay_t * SR)))

            elif car == "skyline":
                # Overlapping double BOV release (two slightly offset envelopes/LFOs)
                # Primary valve
                decay_t1 = 0.16 + 0.32 * bb
                env1 = self.bov_env * np.exp(-idx / (decay_t1 * SR))
                env_norm1 = env1 / max(env1[0], 1e-6)

                frate1 = 6.0 + 8.0 * bb + 4.0 * br
                fph1 = self.bov_flutter_phase + np.cumsum(TWO_PI * ((frate1 + 6.0 * env_norm1) * pitch_factor) / SR)
                self.bov_flutter_phase = float(fph1[-1] % TWO_PI)
                pulse1 = (0.5 + 0.5 * np.sin(fph1)) ** 3.0

                tone_f1 = 120.0 + 140.0 * bb + 30.0 * br + 50.0 * bb * env_norm1
                tph1 = self.bov_tone_phase + np.cumsum(TWO_PI * (tone_f1 * pitch_factor) / SR)
                self.bov_tone_phase = float(tph1[-1] % TWO_PI)
                body1 = np.sin(tph1) + 0.4 * np.sin(2.0 * tph1)

                # Secondary valve (delayed, higher frequency)
                decay_t2 = 0.10 + 0.24 * bb
                env2 = self.bov_env2 * np.exp(-idx / (decay_t2 * SR))
                env_norm2 = env2 / max(env2[0], 1e-6)

                frate2 = 8.0 + 9.0 * bb + 5.0 * br
                fph2 = self.bov_flutter_phase2 + np.cumsum(TWO_PI * ((frate2 + 8.0 * env_norm2) * pitch_factor) / SR)
                self.bov_flutter_phase2 = float(fph2[-1] % TWO_PI)
                pulse2 = (0.5 + 0.5 * np.sin(fph2)) ** 4.5

                tone_f2 = 180.0 + 180.0 * bb + 50.0 * br + 70.0 * bb * env_norm2
                tph2 = self.bov_tone_phase2 + np.cumsum(TWO_PI * (tone_f2 * pitch_factor) / SR)
                self.bov_tone_phase2 = float(tph2[-1] % TWO_PI)
                body2 = np.sin(tph2)

                noise = self.rng.standard_normal(n)
                bright = np.diff(noise, prepend=self._noise_tail)
                self._noise_tail = noise[-1]

                bov1 = (0.55 * body1 + 0.45 * bright) * pulse1 * env1
                bov2 = (0.70 * body2 + 0.30 * bright) * pulse2 * env2
                bov = (bov1 + bov2) * 0.95

                self.bov_env *= float(np.exp(-n / (decay_t1 * SR)))
                self.bov_env2 *= float(np.exp(-n / (decay_t2 * SR)))

            else:
                # Supra classic compressor surge
                decay_t = 0.18 + 0.35 * bb
                env = self.bov_env * np.exp(-idx / (decay_t * SR))
                env_norm = env / max(env[0], 1e-6)

                frate = (5.0 + 7.0 * bb + 3.0 * br) + 10.0 * bb * env_norm
                fph = self.bov_flutter_phase + np.cumsum(TWO_PI * (frate * pitch_factor) / SR)
                self.bov_flutter_phase = float(fph[-1] % TWO_PI)

                pulse = (0.5 + 0.5 * np.sin(fph)) ** (1.5 + 4.5 * bb)

                tone_f = (90.0 + 160.0 * bb + 40.0 * br) + 60.0 * bb * env_norm
                tph = self.bov_tone_phase + np.cumsum(TWO_PI * (tone_f * pitch_factor) / SR)
                self.bov_tone_phase = float(tph[-1] % TWO_PI)

                body = (0.45 + 0.20 * bb) * np.sin(tph) + (0.55 - 0.20 * bb) * np.sin(0.5 * tph) + (0.10 + 0.15 * bb) * np.sin(2.0 * tph)

                noise = self.rng.standard_normal(n)
                bright = np.diff(noise, prepend=self._noise_tail)
                self._noise_tail = noise[-1]

                bov = ((0.50 + 0.55 * bb) * body + (0.55 - 0.30 * bb) * bright) * pulse * env * 1.15

                if bb > 0.55 and self.bov_env > 0.85:
                    crack_samples = min(int(0.008 * SR), n)
                    crack_env = np.zeros(n) if not hasattr(self, '_crack_env') or len(self._crack_env) != n else self._crack_env
                    crack_env.fill(0.0)
                    self._crack_env = crack_env
                    crack_env[:crack_samples] = np.exp(-np.arange(crack_samples) / (0.002 * SR))
                    bov += self.rng.standard_normal(n) * crack_env * 0.6 * bb

                self.bov_env *= float(np.exp(-n / (decay_t * SR)))

        # ---------------- tyre screech ----------------
        screech = np.zeros(n) if not hasattr(self, '_screech') or len(self._screech) != n else self._screech
        screech.fill(0.0)
        self._screech = screech
        if self.c_screech > 1e-3:
            sq_f = 1050.0 + self.t_speed * 6.0
            sphase = self.squeal_phase + np.cumsum(np.full(n, TWO_PI * (sq_f * pitch_factor) / SR))
            self.squeal_phase = float(sphase[-1] % TWO_PI)
            vib = 1.0 + 0.3 * np.sin(TWO_PI * 33.0 * tline)
            tone = np.sin(sphase) * vib
            screech = (0.6 * tone + 0.4 * self.rng.standard_normal(n)) * self.c_screech * 0.32

        # ---------------- backfire / crackle ----------------
        crackle = np.zeros(n) if not hasattr(self, '_crackle') or len(self._crackle) != n else self._crackle
        crackle.fill(0.0)
        self._crackle = crackle
        if self.crackle_env > 1e-3:
            decay_rate = 0.035 if car == "rx7" else 0.05
            env = self.crackle_env * np.exp(-idx / (decay_rate * SR))
            pop = self.rng.standard_normal(n)

            # Low-pass filter (RX-7 uses a shorter filter to sound sharper and less bassy)
            filter_size = 4 if car == "rx7" else 8
            pop_cat = np.concatenate((self.pop_hist[-(filter_size - 1):], pop))
            k = np.ones(filter_size) / float(filter_size)
            pop = np.convolve(pop_cat, k, mode="valid")
            self.pop_hist = pop_cat[-10:]

            crackle_gain = 0.45 if car == "rx7" else 0.32
            crackle = pop * env * crackle_gain
            self.crackle_env *= float(np.exp(-n / (decay_rate * SR)))

        # ---------------- environment: wind + road ----------------
        # Both derive from speed alone, so they need no new telemetry. Box-filtered
        # noise with a persistent history tail keeps the layers click-free across
        # buffer boundaries.
        env_snd = np.zeros(n) if not hasattr(self, '_env_snd') or len(self._env_snd) != n else self._env_snd
        env_snd.fill(0.0)
        self._env_snd = env_snd
        spd_norm = float(np.clip(self.t_speed / 95.0, 0.0, 1.15))
        if spd_norm > 0.02:
            # Wind: gets both louder AND brighter with speed (shorter box filter).
            wind_w = int(np.clip(26.0 - 20.0 * spd_norm, 5, 26))
            wn = self.rng.standard_normal(n)
            wn_cat = np.concatenate((self._wind_tail[-(wind_w - 1):], wn))
            wind = np.convolve(wn_cat, np.ones(wind_w) / wind_w, mode="valid")
            self._wind_tail = wn_cat[-31:]
            # Gusting so it isn't a flat hiss.
            gust = 1.0 + 0.22 * np.sin(TWO_PI * 1.7 * tline) * np.sin(TWO_PI * 0.43 * tline + 0.9)
            env_snd += wind * gust * (spd_norm ** 2) * 0.34

            # Road rumble: heavier filtering, scales more linearly, and roughens up
            # when the tyres are sliding.
            rn = self.rng.standard_normal(n)
            rn_cat = np.concatenate((self._road_tail[-62:], rn))
            road = np.convolve(rn_cat, np.ones(63) / 63.0, mode="valid")
            self._road_tail = rn_cat[-63:]
            env_snd += road * spd_norm * (0.30 + 0.35 * self.c_screech)

        # kerb strike: rhythmic thwack-thwack-thwack of the serrations. Pulse
        # rate follows speed; each pulse is a noise slap over a low body knock.
        self.c_kerb = ramp(_clean_float(self.c_kerb, self.t_kerb, 0.0, 1.0),
                           _clean_float(self.t_kerb, 0.0, 0.0, 1.0), 0.55)
        if self.c_kerb > 1e-3:
            pulse_f = 16.0 + self.t_speed * 0.42
            kph = self.kerb_phase + np.cumsum(np.full(n, TWO_PI * pulse_f / SR))
            self.kerb_phase = float(kph[-1] % TWO_PI)
            pulse = (0.5 + 0.5 * np.sin(kph)) ** 3.5
            kn = self.rng.standard_normal(n)
            kn_cat = np.concatenate((self._kerb_tail[-6:], kn))
            slap = np.convolve(kn_cat, np.ones(7) / 7.0, mode="valid")
            self._kerb_tail = kn_cat[-7:]
            knock = np.sin(TWO_PI * 82.0 * tline) * 0.5
            env_snd += (0.85 * slap + knock) * pulse * self.c_kerb \
                * (0.22 + 0.30 * spd_norm)

        mix = eng + turbo + bov + screech + crackle + env_snd
        mix = np.tanh(mix * 0.9)
        # DC blocker: the asymmetric waveshaping (esp. the 787b straight-pipe
        # overdrive) biases the buffer off-center, which pulls the speaker cone and
        # clicks whenever throttle/rpm shift the bias. The buffer spans many engine
        # periods, so its mean is an accurate, click-free DC estimate to remove.
        mix -= float(mix.mean())
        return mix.astype(np.float32)



import multiprocessing as mp
import threading
import time

def _audio_worker(shared_array, num_cars, car_names, redlines):
    import sounddevice as sd
    from .sound import EngineSynth
    import numpy as np

    # Build synths
    synths = [EngineSynth() for _ in range(num_cars)]

    # Mock vehicles to satisfy EngineSynth API
    class MockSpec: pass
    class MockVeh: pass

    vehs = []
    for i in range(num_cars):
        v = MockVeh()
        v.spec = MockSpec()
        v.spec.name = car_names[i]
        v.spec.redline_rpm = redlines[i]
        vehs.append(v)

    def callback(outdata, frames, time_info, status):
        if shared_array[0] == 0.0:
            outdata.fill(0.0)
            return
        if shared_array[0] == 2.0:
            for synth in synths:
                synth.reset()
            shared_array[0] = 1.0
            outdata.fill(0.0)
            return

        master = shared_array[1]
        lx = shared_array[2]
        ly = shared_array[3]
        lvx = shared_array[4]
        lvy = shared_array[5]
        lyaw = shared_array[6]

        c_states = []
        for i in range(num_cars):
            idx = 7 + i * 11
            v = vehs[i]
            v.x = shared_array[idx]
            v.y = shared_array[idx+1]
            v.yaw = shared_array[idx+2]
            v.speed = shared_array[idx+3]
            v.rpm = shared_array[idx+4]
            throttle = shared_array[idx+5]
            v.boost = shared_array[idx+6]
            v.gear = int(shared_array[idx+7])
            v.slip_angle = shared_array[idx+8]
            v.wheel_sr = [0.0, 0.0, shared_array[idx+9], shared_array[idx+9]]
            v.kerb = shared_array[idx+10]

            synths[i].update(v, throttle)
            v_vel = (v.speed * np.cos(v.yaw), v.speed * np.sin(v.yaw))
            c_states.append((v.x, v.y, v_vel[0], v_vel[1]))

        n = frames
        left_mix = np.zeros(n, dtype=np.float32)
        right_mix = np.zeros(n, dtype=np.float32)

        for i, s in enumerate(synths):
            cx, cy, cvx, cvy = c_states[i]

            dx = cx - lx
            dy = cy - ly
            dist = np.hypot(dx, dy)

            if dist < 0.1:
                pan_l = pan_r = 0.5
                doppler = 1.0
                vol = 1.0
            else:
                rel_x = dx * np.cos(-lyaw) - dy * np.sin(-lyaw)
                rel_y = dx * np.sin(-lyaw) + dy * np.cos(-lyaw)
                angle = np.arctan2(rel_y, rel_x)
                pan_r = (np.sin(angle) + 1.0) / 2.0
                pan_l = 1.0 - pan_r

                v_rel_x = cvx - lvx
                v_rel_y = cvy - lvy
                # Trackside flybys need audible pitch travel, but full-scale
                # real doppler is too much for this stylized camera scale.
                speed_of_sound = 520.0
                v_approach = -(dx * v_rel_x + dy * v_rel_y) / dist
                doppler = speed_of_sound / (speed_of_sound - v_approach)
                doppler = np.clip(doppler, 0.82, 1.22)

                vol = 1.0 / (1.0 + (dist / 12.0) ** 1.5)

            audio = s.synthesize(n, pitch_factor=doppler)

            left_mix += audio * pan_l * vol
            right_mix += audio * pan_r * vol

        # Master compression/clipping
        out = np.column_stack((left_mix, right_mix)) * master
        out = np.clip(out, -0.98, 0.98)
        outdata[:] = out.astype(np.float32)

    try:
        stream = sd.OutputStream(
            channels=2, dtype="float32", samplerate=SR, blocksize=BLOCK,
            latency="high", callback=callback)
        print(f"[audio process] online sr={SR} block={BLOCK} cars={num_cars}", flush=True)
        with stream:
            while shared_array[0] > 0.0:
                time.sleep(0.05)
    except Exception as e:
        print(f"[audio process] died: {e}")

class SpatialAudioMixer:
    def __init__(self, master=0.55):
        self.master = master
        self.muted = False
        self.ok = False
        self.process = None
        self.shared_data = None
        self._warned_dead = False

    def start(self, racers=None):
        if not racers:
            print("[audio] disabled (no vehicles to synthesize)")
            return self
        try:
            self.num_cars = len(racers)
            car_names = []
            redlines = []
            for r in racers:
                v = r.veh if hasattr(r, 'veh') else r
                car_names.append(getattr(v.spec, "name", "supra"))
                redlines.append(float(v.spec.redline_rpm))

            # [running, master, lx, ly, lvx, lvy, lyaw, ... (11 per car)]
            size = 7 + self.num_cars * 11
            self.shared_data = mp.Array('f', size)
            self.shared_data[0] = 1.0
            self.shared_data[1] = self.master

            self.process = mp.Process(target=_audio_worker, args=(self.shared_data, self.num_cars, car_names, redlines))
            self.process.start()
            self.ok = True
            time.sleep(0.05)
            self._refresh_ok()
            if self.ok:
                print(f"[audio] started pid={self.process.pid} cars={self.num_cars}", flush=True)
        except Exception as e:
            print(f"[audio] disabled ({e.__class__.__name__}: {e})")
            self.ok = False
        return self

    def _refresh_ok(self):
        if self.process and not self.process.is_alive() and self.ok:
            self.ok = False
            if not self._warned_dead:
                print(f"[audio] disabled (audio process exited {self.process.exitcode})")
                self._warned_dead = True
        return self.ok

    def toggle_mute(self):
        self.muted = not self.muted
        if self._refresh_ok() and self.shared_data:
            self.shared_data[1] = 0.0 if self.muted else self.master

    def update(self, listener_pos, listener_vel, listener_yaw, vehicles, throttles):
        if not self._refresh_ok() or not self.shared_data: return
        self.shared_data[2] = listener_pos[0]
        self.shared_data[3] = listener_pos[1]
        self.shared_data[4] = listener_vel[0]
        self.shared_data[5] = listener_vel[1]
        self.shared_data[6] = listener_yaw

        for i, veh in enumerate(vehicles):
            idx = 7 + i * 11
            self.shared_data[idx] = veh.x
            self.shared_data[idx+1] = veh.y
            self.shared_data[idx+2] = veh.yaw
            self.shared_data[idx+3] = veh.speed
            self.shared_data[idx+4] = veh.rpm
            self.shared_data[idx+5] = throttles[i]
            self.shared_data[idx+6] = veh.boost
            self.shared_data[idx+7] = float(veh.gear)
            self.shared_data[idx+8] = veh.slip_angle
            self.shared_data[idx+9] = float(max(abs(veh.wheel_sr[2]), abs(veh.wheel_sr[3])))
            # kerb-strike intensity: the viewer sets _snd_kerb; v1/app.py never
            # does, so this reads 0.0 there and the kerb layer stays silent
            self.shared_data[idx+10] = float(getattr(veh, "_snd_kerb", 0.0))

    def reset(self):
        if self._refresh_ok() and self.shared_data:
            self.shared_data[0] = 2.0

    def stop(self):
        if self.ok and self.shared_data and self.process:
            self.shared_data[0] = 0.0
            self.process.join(timeout=1.0)
            if self.process.is_alive():
                self.process.terminate()
        self.ok = False


# Audio V5 is the default. V3 remains as the one-release rollback.
if os.environ.get("SUPRA_AUDIO_ENGINE", "v5").lower() != "v3":
    from .audio_v4 import SpatialAudioMixerV4 as SpatialAudioMixer
