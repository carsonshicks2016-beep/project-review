"""
Real-time, spatial engine + turbo audio.

Synthesised from scratch (no samples) so it tracks RPM exactly, and now
*positional*: the nearest few cars each get their own voice, mixed into stereo:

  * engine    -> additive harmonics of the crank frequency (inline-six 3rd/6th
                 firing character), amplitude follows rpm + load,
  * turbo     -> a whine whose pitch rises with shaft speed (rpm * boost),
  * blow-off  -> a filtered-noise "pshhh" on lift-off / gearshifts,
  * DOPPLER   -> every voice's pitch is shifted by its closing speed toward the
                 listener (the camera), so a car screams up in pitch as it
                 approaches and drops as it passes,
  * STEREO    -> each voice is panned left/right by its on-screen position and
                 attenuated by distance, so the field spreads across the field.

The listener is the camera; its velocity is taken from the focused car (the
camera rides with it).  If `sounddevice` or an output device isn't available it
silently no-ops.
"""

from __future__ import annotations
import math
import random
import numpy as np

try:
    import sounddevice as sd
    _HAVE_SD = True
except Exception:
    _HAVE_SD = False


class SoundEngine:
    SR = 44100
    BLOCK = 1024
    C_SOUND = 343.0          # m/s, speed of sound (for Doppler)
    MAX_RANGE = 95.0         # m: cars beyond this are silent
    MAX_VOICES = 6           # only the nearest N cars are sonified

    # crank-relative harmonic orders and weights (inline-6 character)
    ORDERS = np.array([1.0, 2.0, 3.0, 4.0, 6.0, 9.0, 12.0])
    WEIGHTS = np.array([0.18, 0.10, 0.55, 0.12, 0.45, 0.20, 0.10])

    def __init__(self, idle_rpm=850.0, redline=6800.0, max_boost=0.8):
        self.idle = idle_rpm
        self.redline = redline
        self.max_boost = max_boost
        self.enabled = _HAVE_SD
        self.muted = False
        self.master = 0.5
        self.voices = {}          # car index -> voice state dict
        self._stream = None

    # ------------------------------------------------------------------ #
    def _new_voice(self):
        return dict(
            # targets (written by update)
            rpm_t=self.idle, thr_t=0.0, boost_t=0.0, load_t=0.0,
            pan_t=0.0, gain_t=0.0, dopp_t=1.0,
            # smoothed (read/advanced in callback)
            c_rpm=self.idle, c_thr=0.0, c_boost=0.0, c_load=0.0,
            pan=0.0, gain=0.0, dopp=1.0,
            phase=0.0, tphase=0.0, bov=0.0, prev_thr=0.0,
            # tire screech
            slip_t=0.0, c_slip=0.0, screech_phase=0.0,
            # backfire / anti-lag crackle
            backfire_cd=0.0, backfire_env=0.0,
            # gear clunk
            clunk=0.0, clunk_phase=0.0,
            # turbo wastegate flutter
            flutter_phase=0.0)

    def _ensure(self, n):
        for i in range(n):
            if i not in self.voices:
                self.voices[i] = self._new_voice()

    @staticmethod
    def _world_vel(car):
        v = car.vehicle
        c, s = math.cos(v.yaw), math.sin(v.yaw)
        return (v.vx * c - v.vy * s, v.vx * s + v.vy * c)

    # ------------------------------------------------------------------ #
    def start(self):
        if not self.enabled:
            return
        try:
            self._stream = sd.OutputStream(
                samplerate=self.SR, channels=2, blocksize=self.BLOCK,
                dtype="float32", callback=self._callback)
            self._stream.start()
        except Exception:
            self.enabled = False

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None

    # ------------------------------------------------------------------ #
    def update(self, cars, focus, cam, half_w_m, muted=False):
        """Call once per frame with the full field.

        cars      : list of Car
        focus     : index of the focused car (the listener rides with it)
        cam       : (x, y) world position of the camera = listener
        half_w_m  : metres from screen centre to screen edge (for stereo pan)
        """
        self.muted = muted
        if not self.enabled or not cars:
            return
        self._ensure(len(cars))
        # silence everything by default; re-activate the nearest cars below
        for v in self.voices.values():
            v["gain_t"] = 0.0

        lx, ly = cam
        if 0 <= focus < len(cars):
            lvx, lvy = self._world_vel(cars[focus])
        else:
            lvx = lvy = 0.0

        cand = []
        for i, c in enumerate(cars):
            if not getattr(c, "alive", True):
                continue
            dx = c.vehicle.x - lx
            dy = c.vehicle.y - ly
            d = math.hypot(dx, dy)
            if d > self.MAX_RANGE and i != focus:
                continue
            cand.append((d, i, c, dx, dy))
        cand.sort(key=lambda t: t[0])

        half_w_m = max(1.0, half_w_m)
        for d, i, c, dx, dy in cand[:self.MAX_VOICES]:
            v = self.voices[i]
            vv = c.vehicle
            v["rpm_t"] = max(self.idle, vv.rpm)
            v["thr_t"] = c.controls[1]
            v["boost_t"] = max(0.0, vv.boost)
            v["load_t"] = min(1.0, max(vv.grip_r, vv.grip_f))
            # tire slip target for screech
            v["slip_t"] = min(1.0, max(vv.grip_r, vv.grip_f)) * min(vv.speed / 30.0, 1.0)
            # gear clunk trigger
            if vv.shift_flash > 0.5:
                v["clunk"] = 1.0
            # distance attenuation + stereo pan
            dd = max(0.6, d)
            v["gain_t"] = 1.0 / (1.0 + (dd / 16.0) ** 2)
            v["pan_t"] = float(np.clip(dx / half_w_m, -1.0, 1.0))
            # Doppler: closing speed of the source toward the listener
            svx, svy = self._world_vel(c)
            ux, uy = dx / dd, dy / dd
            v_radial = (svx - lvx) * ux + (svy - lvy) * uy   # + = receding
            v["dopp_t"] = float(np.clip(self.C_SOUND / (self.C_SOUND + v_radial),
                                        0.8, 1.25))
            # blow-off on lift while on boost, or on a gearshift
            lift = v["prev_thr"] > 0.55 and v["thr_t"] < 0.25
            if (lift and v["boost_t"] > 0.25) or vv.shift_flash > 0.5:
                v["bov"] = 1.0
            v["prev_thr"] = v["thr_t"]

    # ------------------------------------------------------------------ #
    def _callback(self, outdata, frames, time_info, status):
        if self.muted:
            outdata[:] = 0.0
            return
        sr = self.SR
        idx = np.arange(frames)
        L = np.zeros(frames)
        R = np.zeros(frames)

        for v in list(self.voices.values()):
            # smooth gain even when silent so voices fade cleanly, then skip
            if v["gain"] < 1e-4 and v["gain_t"] < 1e-4:
                v["gain"] += (v["gain_t"] - v["gain"]) * 0.2
                continue
            v["c_rpm"] += (v["rpm_t"] - v["c_rpm"]) * 0.45
            v["c_thr"] += (v["thr_t"] - v["c_thr"]) * 0.30
            v["c_boost"] += (v["boost_t"] - v["c_boost"]) * 0.30
            v["c_load"] += (v["load_t"] - v["c_load"]) * 0.30
            v["gain"] += (v["gain_t"] - v["gain"]) * 0.20
            v["pan"] += (v["pan_t"] - v["pan"]) * 0.20
            v["dopp"] += (v["dopp_t"] - v["dopp"]) * 0.30

            rpm_frac = float(np.clip(
                (v["c_rpm"] - self.idle) / (self.redline - self.idle + 1e-6), 0.0, 1.2))
            dopp = v["dopp"]

            # ---- engine: additive harmonics (Doppler-shifted) ----
            f_crank = v["c_rpm"] / 60.0 * dopp
            dphi = 2.0 * math.pi * f_crank / sr
            phase = v["phase"] + dphi * idx
            eng = np.zeros(frames)
            for order, w in zip(self.ORDERS, self.WEIGHTS):
                eng += w * np.sin(order * phase)
            eng += 0.12 * rpm_frac * np.sin(3.0 * phase) ** 3
            load = 0.30 + 0.70 * v["c_thr"]
            eng *= 0.22 * (0.5 + 0.9 * rpm_frac) * load
            v["phase"] = (v["phase"] + dphi * frames) % (2.0 * math.pi)

            # ---- turbo whine (Doppler-shifted) ----
            boost_frac = v["c_boost"] / (self.max_boost + 1e-6)
            f_turbo = (1800.0 + 7000.0 * boost_frac * (0.4 + 0.6 * rpm_frac)) * dopp
            dphi_t = 2.0 * math.pi * f_turbo / sr
            tphase = v["tphase"] + dphi_t * idx
            turbo = (0.06 * boost_frac) * np.sin(tphase)
            v["tphase"] = (v["tphase"] + dphi_t * frames) % (2.0 * math.pi)

            # ---- turbo wastegate flutter (25Hz AM above 80% boost) ----
            if boost_frac > 0.8:
                flutter_dphi = 2.0 * math.pi * 25.0 / sr
                flutter_ph = v["flutter_phase"] + flutter_dphi * idx
                turbo *= 1.0 + 0.6 * np.sin(flutter_ph)
                v["flutter_phase"] = (v["flutter_phase"] + flutter_dphi * frames) % (2.0 * math.pi)

            # ---- blow-off / flutter: decaying filtered-noise burst ----
            bov = np.zeros(frames)
            if v["bov"] > 0.001:
                env = v["bov"] * np.exp(-idx / (sr * 0.12))
                noise = np.convolve(np.random.randn(frames), np.ones(8) / 8.0, "same")
                bov = 0.5 * env * noise
                v["bov"] *= math.exp(-frames / (sr * 0.12))
                if v["bov"] < 0.01:
                    v["bov"] = 0.0

            # ---- backfire / anti-lag crackle ----
            backfire = np.zeros(frames)
            if v["c_rpm"] > 4000.0 and v["c_thr"] < 0.15 and v["c_boost"] > 0.1:
                bf_env = v["backfire_env"]
                bf_cd = v["backfire_cd"]
                pop_prob = (v["c_rpm"] / self.redline) * (0.5 + 0.5 * boost_frac)
                for fi in range(frames):
                    bf_cd -= 1.0 / sr
                    if bf_cd <= 0.0 and random.random() < pop_prob:
                        bf_env = 1.0
                        bf_cd = random.uniform(0.03, 0.15)
                    backfire[fi] = 0.35 * bf_env * (random.random() * 2.0 - 1.0)
                    bf_env *= math.exp(-1.0 / (sr * 0.008))
                v["backfire_env"] = bf_env
                v["backfire_cd"] = bf_cd

            # ---- tire screech ----
            screech = np.zeros(frames)
            v["c_slip"] += (v["slip_t"] - v["c_slip"]) * 0.25
            intensity = v["c_slip"]
            if intensity > 0.4:
                screech_dphi = 2.0 * math.pi * 3000.0 / sr
                screech_ph = v["screech_phase"] + screech_dphi * idx
                noise = np.random.randn(frames)
                screech = 0.15 * intensity * noise * np.abs(np.sin(screech_ph))
                v["screech_phase"] = (v["screech_phase"] + screech_dphi * frames) % (2.0 * math.pi)

            # ---- gear clunk ----
            clunk_sound = np.zeros(frames)
            if v["clunk"] > 0.001:
                clunk_dphi = 2.0 * math.pi * 120.0 / sr
                clunk_ph = v["clunk_phase"] + clunk_dphi * idx
                clunk_env = v["clunk"] * np.exp(-idx / (sr * 0.015))
                clunk_sound = 0.25 * clunk_env * np.sin(clunk_ph)
                v["clunk_phase"] = (v["clunk_phase"] + clunk_dphi * frames) % (2.0 * math.pi)
                v["clunk"] *= math.exp(-frames / (sr * 0.015))
                if v["clunk"] < 0.001:
                    v["clunk"] = 0.0

            mono = (eng + turbo + bov + backfire + screech + clunk_sound) * v["gain"]
            # equal-power stereo pan
            ang = (float(np.clip(v["pan"], -1.0, 1.0)) + 1.0) * (math.pi / 4.0)
            L += mono * math.cos(ang)
            R += mono * math.sin(ang)

        g = self.master * 1.4
        outdata[:, 0] = np.tanh(L * g).astype(np.float32)
        outdata[:, 1] = np.tanh(R * g).astype(np.float32)
