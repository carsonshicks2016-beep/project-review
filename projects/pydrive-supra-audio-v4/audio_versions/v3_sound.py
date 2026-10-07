"""
Real-time engine audio — synthesised from scratch, no samples.  (v3)

A background `sounddevice` output stream runs a synthesis callback that builds
each audio buffer live from the car's telemetry:

  * Engine     - additive inline-six harmonics of the crank frequency, waveshaped
                 for an aggressive/raspy 2JZ snarl that hardens with load.
  * Turbo      - rising whine that tracks boost + rpm, with wastegate flutter up top.
  * Blow-off   - BOOST-DEPENDENT compressor-surge flutter ("stu-tu-tu-tu").
                 Captures boost + RPM at the moment of release and scales decay,
                 flutter rate, pulse sharpness, pitch (with pressure-drop sweep),
                 harmonic mix, and tone/noise ratio accordingly.  Low boost =
                 breathy puff; full boost = violent bark with initial crack.
  * Tyre       - squeal + noise scaled by how hard the tyres are sliding.
  * Backfire   - exhaust crackle/pops on downshifts and on overrun.

Tuned cinematic (loud BOV, prominent crackle). Degrades gracefully: if there's
no audio device or sounddevice is missing, it silently no-ops so the sim runs on.

Thread model: the sim thread calls `update(...)` to set target parameters and
fire events; the audio thread reads them in `_callback`. Continuous params are
ramped per-buffer to avoid zipper noise; phases/filters persist across buffers.
"""
from __future__ import annotations

import threading

import numpy as np

SR = 44100
BLOCK = 512                      # ~11.6 ms latency
TWO_PI = 2.0 * np.pi

# inline-six additive harmonic orders of the crank frequency, with weights
# skewed toward higher orders for rasp (3rd order = the firing frequency).
ENGINE_ORDERS = np.array([1, 2, 3, 4, 6, 9, 12], dtype=float)
ENGINE_AMPS = np.array([0.45, 0.55, 1.00, 0.55, 0.50, 0.34, 0.24])


class EngineAudio:
    def __init__(self, master: float = 0.55):
        self.ok = False
        self.muted = False
        self.master = master
        self._stream = None
        self._lock = threading.Lock()

        # --- target params (set by the sim thread) ---
        self.t_rpm = 850.0
        self.t_throttle = 0.0
        self.t_boost = 0.0
        self.t_screech = 0.0       # 0..1 slide intensity
        self.t_rpm_frac = 0.0
        self.t_speed = 0.0

        # --- ramped (smoothed) current values used in the callback ---
        self.c_rpm = 850.0
        self.c_throttle = 0.0
        self.c_boost = 0.0
        self.c_screech = 0.0

        # --- persistent oscillator / filter / envelope state ---
        self.eng_phase = 0.0
        self.tur_phase = 0.0
        self.squeal_phase = 0.0
        self.clock = 0            # running sample count (for LFOs)
        self.bov_env = 0.0
        self.bov_flutter_phase = 0.0   # turbo-flutter pulse-train LFO
        self.bov_tone_phase = 0.0      # tonal "tu" body
        self.bov_boost = 0.0           # boost snapshot when BOV fired
        self.bov_rpm_frac = 0.0        # rpm fraction snapshot for pitch/rate
        self.crackle_env = 0.0
        self._noise_tail = 0.0

        # --- event edge-detection memory ---
        self._prev_throttle = 0.0
        self._prev_gear = 1
        self._bov_armed = False     # set while on boost+throttle, fires on lift

    # ------------------------------------------------------------------ #
    def start(self):
        """Open the audio stream; on any failure, stay silent but alive."""
        try:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=SR, blocksize=BLOCK, channels=2,
                dtype="float32", callback=self._callback, latency="low")
            self._stream.start()
            self.ok = True
        except Exception as e:           # no device, no portaudio, etc.
            print(f"[audio] disabled ({e.__class__.__name__}: {e})")
            self.ok = False
        return self

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
        self.ok = False

    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted

    # ------------------------------------------------------------------ #
    def update(self, veh, throttle: float):
        """Push the latest car state from the sim thread (cheap, lock-light)."""
        if not self.ok:
            return
        rear_slide = max(abs(veh.wheel_sr[2]), abs(veh.wheel_sr[3]))
        slip = abs(np.degrees(veh.slip_angle)) / 35.0
        screech = float(np.clip(max(rear_slide - 0.15, slip - 0.2) * 1.3, 0.0, 1.0))
        if veh.speed < 2.0:
            screech = 0.0

        self.t_rpm = float(veh.rpm)
        self.t_throttle = float(np.clip(throttle, 0, 1))
        self.t_boost = float(veh.boost)
        self.t_rpm_frac = float(veh.rpm / veh.spec.redline_rpm)
        self.t_screech = screech
        self.t_speed = float(veh.speed)

        # event detection
        with self._lock:
            # turbo flutter: arm while on boost+throttle, fire when the driver
            # lifts. Level-based (not a per-frame delta) so it survives the
            # analog throttle ramp instead of never crossing a threshold.
            if throttle > 0.55 and veh.boost > 0.30:
                self._bov_armed = True
            if self._bov_armed and throttle < 0.25:
                self.bov_boost = float(np.clip(veh.boost, 0.25, 1.0))
                self.bov_rpm_frac = float(np.clip(
                    veh.rpm / veh.spec.redline_rpm, 0.0, 1.0))
                self.bov_env = self.bov_boost
                self._bov_armed = False
            # backfire on downshift
            if veh.gear < self._prev_gear:
                self.crackle_env = max(self.crackle_env, 1.0)
            # overrun crackle: closed throttle, high revs -> random pops
            if throttle < 0.1 and veh.rpm > 3500 and np.random.rand() < 0.30:
                self.crackle_env = max(self.crackle_env, np.random.uniform(0.3, 0.8))
            self._prev_throttle = throttle
            self._prev_gear = veh.gear

    # ------------------------------------------------------------------ #
    def _callback(self, outdata, frames, time_info, status):
        n = frames
        idx = np.arange(n)
        tline = (self.clock + idx) / SR
        self.clock += n

        # per-buffer smoothing toward targets
        def ramp(cur, tgt, rate=0.25):
            return cur + (tgt - cur) * rate
        self.c_rpm = ramp(self.c_rpm, self.t_rpm, 0.30)
        self.c_throttle = ramp(self.c_throttle, self.t_throttle, 0.30)
        self.c_boost = ramp(self.c_boost, self.t_boost, 0.30)
        self.c_screech = ramp(self.c_screech, self.t_screech, 0.40)
        thr, boost = self.c_throttle, self.c_boost
        rpm_frac = np.clip(self.c_rpm / 7000.0, 0.0, 1.1)

        # ---------------- engine ----------------
        f0 = max(self.c_rpm, 200.0) / 60.0          # crank Hz (floor for audibility)
        phase = self.eng_phase + np.cumsum(np.full(n, TWO_PI * f0 / SR))
        self.eng_phase = float(phase[-1] % TWO_PI)
        eng = np.zeros(n)
        for order, amp in zip(ENGINE_ORDERS, ENGINE_AMPS):
            eng += amp * np.sin(order * phase)
        eng /= ENGINE_AMPS.sum()
        # raspy waveshaping: harder with throttle
        drive = 1.5 + 4.0 * thr
        eng = np.tanh(eng * drive)
        # induction/combustion noise, grows with load
        eng += (0.12 + 0.25 * thr) * np.random.randn(n) * (0.3 + 0.7 * rpm_frac) * 0.4
        eng_gain = (0.16 + 0.55 * thr) * (0.45 + 0.55 * rpm_frac)
        eng *= eng_gain

        # ---------------- turbo whine ----------------
        ftur = 1800.0 + boost * 4200.0 + rpm_frac * 2200.0
        tphase = self.tur_phase + np.cumsum(np.full(n, TWO_PI * ftur / SR))
        self.tur_phase = float(tphase[-1] % TWO_PI)
        flutter = 1.0 + (0.5 if boost > 0.6 else 0.0) * np.sin(TWO_PI * 58.0 * tline)
        turbo = np.sin(tphase) * (boost ** 1.5) * 0.14 * flutter

        # -------- blow-off / turbo flutter ("stu-tu-tu-tu") --------
        # Boost-dependent compressor surge.  Captures boost + RPM at release and
        # scales every synthesis parameter: low boost = breathy puff, full boost =
        # violent barking surge with initial crack.  RPM adds pitch + flutter edge.
        bov = np.zeros(n)
        if self.bov_env > 1e-3:
            bb = self.bov_boost                                  # boost snapshot
            br = self.bov_rpm_frac                               # RPM snapshot

            # --- envelope: more stored air = longer release ---
            decay_t = 0.18 + 0.35 * bb                           # 0.27s low → 0.53s full
            env = self.bov_env * np.exp(-idx / (decay_t * SR))
            env_norm = env / max(env[0], 1e-6)                   # 1→0 shape for sweeps

            # --- flutter rate: boost sets base, RPM adds edge, decelerates ---
            frate_base = 5.0 + 7.0 * bb + 3.0 * br              # 8Hz lazy → 15Hz snappy
            frate = frate_base + 10.0 * bb * env_norm            # fast start, slows down
            fph = self.bov_flutter_phase + np.cumsum(TWO_PI * frate / SR)
            self.bov_flutter_phase = float(fph[-1] % TWO_PI)

            # --- pulse sharpness: soft rolls at low boost, staccato at full ---
            sharpness = 1.5 + 4.5 * bb                          # 2.6 soft → 6.0 sharp
            pulse = (0.5 + 0.5 * np.sin(fph)) ** sharpness

            # --- tone pitch: boost+RPM base with pressure-drop sweep ---
            tone_base = 90.0 + 160.0 * bb + 40.0 * br           # 130Hz → 290Hz
            tone_sweep = 60.0 * bb * env_norm                   # downward glide
            tone_f_arr = tone_base + tone_sweep                  # per-sample pitch
            tph = self.bov_tone_phase + np.cumsum(TWO_PI * tone_f_arr / SR)
            self.bov_tone_phase = float(tph[-1] % TWO_PI)

            # --- body harmonics: high boost = brighter bark, low = softer woof ---
            fund  = 0.45 + 0.20 * bb                            # 0.50 → 0.65
            sub   = 0.55 - 0.20 * bb                            # 0.50 → 0.35
            harm2 = 0.10 + 0.15 * bb                            # 0.14 → 0.25
            body = fund * np.sin(tph) + sub * np.sin(0.5 * tph) + harm2 * np.sin(2.0 * tph)

            # --- air noise ---
            noise = np.random.randn(n)
            bright = np.diff(noise, prepend=self._noise_tail)
            self._noise_tail = noise[-1]

            # --- tone vs noise mix: hissy at low boost, tonal bark at full ---
            tone_mix  = 0.50 + 0.55 * bb                        # 0.75 → 1.05
            noise_mix = 0.55 - 0.30 * bb                        # 0.48 → 0.25
            bov = (tone_mix * body + noise_mix * bright) * pulse * env * 1.15

            # --- initial transient "crack" on high-boost releases ---
            if bb > 0.55 and self.bov_env > 0.85:
                crack_dur = int(0.008 * SR)                      # ~8 ms
                crack_samples = min(crack_dur, n)
                crack_env = np.zeros(n)
                crack_env[:crack_samples] = np.exp(
                    -np.arange(crack_samples) / (0.002 * SR))
                bov += np.random.randn(n) * crack_env * 0.6 * bb

            self.bov_env *= float(np.exp(-n / (decay_t * SR)))

        # ---------------- tyre screech ----------------
        screech = np.zeros(n)
        if self.c_screech > 1e-3:
            sq_f = 1050.0 + self.t_speed * 6.0
            sphase = self.squeal_phase + np.cumsum(np.full(n, TWO_PI * sq_f / SR))
            self.squeal_phase = float(sphase[-1] % TWO_PI)
            vib = 1.0 + 0.3 * np.sin(TWO_PI * 33.0 * tline)
            tone = np.sin(sphase) * vib
            screech = (0.6 * tone + 0.4 * np.random.randn(n)) * self.c_screech * 0.32

        # ---------------- backfire / crackle ----------------
        crackle = np.zeros(n)
        if self.crackle_env > 1e-3:
            env = self.crackle_env * np.exp(-idx / (0.05 * SR))
            pop = np.random.randn(n)
            # low-pass for a rumbly pop (short moving average)
            k = np.ones(8) / 8.0
            pop = np.convolve(pop, k, mode="same")
            crackle = pop * env * 0.9
            self.crackle_env *= float(np.exp(-n / (0.05 * SR)))

        mix = eng + turbo + bov + screech + crackle
        mix = np.tanh(mix * 0.9)
        if self.muted:
            mix *= 0.0
        else:
            mix *= self.master
        out = mix.astype(np.float32)
        outdata[:, 0] = out
        outdata[:, 1] = out
