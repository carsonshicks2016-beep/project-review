"""
Procedural audio for Neural Racer.

Engine — two layered tones (low rumble + high whine) cross-faded by speed
Squeal — band-filtered noise loop, gated by drift state
Thunk  — short impact sound on car-car collisions
Crunch — heavier impact when a car dies (hits wall)

All samples are pre-generated once at startup, no real-time synthesis.
"""
import numpy as np
import pygame
import pygame.sndarray


SAMPLE_RATE = 22050


# ----------------------------------------------------------------------
# Waveform generators
# ----------------------------------------------------------------------

def _saw_harmonics(freq, duration, harmonics=8):
    n = int(duration * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    w = np.zeros(n, dtype=np.float32)
    for k in range(1, harmonics + 1):
        w += np.sin(2 * np.pi * freq * k * t) / k
    w /= np.max(np.abs(w)) + 1e-9
    return w


def _noise(duration):
    n = int(duration * SAMPLE_RATE)
    return np.random.uniform(-1, 1, n).astype(np.float32)


def _lowpass(wave, window):
    if window <= 1:
        return wave
    k = np.ones(window, dtype=np.float32) / window
    return np.convolve(wave, k, mode='same')


def _to_sound(wave, volume=1.0):
    """Mono float32 [-1, 1] → stereo pygame Sound."""
    wave = np.clip(wave * volume, -1.0, 1.0)
    int16 = (wave * 32767).astype(np.int16)
    stereo = np.stack([int16, int16], axis=-1)
    # Ensure C-contiguous for sndarray
    stereo = np.ascontiguousarray(stereo)
    return pygame.sndarray.make_sound(stereo)


# ----------------------------------------------------------------------
# Sound engine
# ----------------------------------------------------------------------

class SoundEngine:
    # Engine pitch ladder — N samples spanning idle → redline.
    # Adjacent tones must be close enough in pitch that crossfading them
    # sounds like a continuous pitch sweep, not a chord.
    # 24 tones over 1.8 octaves ≈ 0.9 semitones per step (sub-semitone).
    N_TONES  = 24
    FREQ_LO  = 80.0       # Hz at idle
    FREQ_HI  = 280.0      # Hz at top speed

    def __init__(self):
        self.enabled = True
        self.ok      = False
        try:
            pygame.mixer.quit()
        except pygame.error:
            pass
        try:
            pygame.mixer.pre_init(SAMPLE_RATE, -16, 2, 256)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(32)
            self.ok = True
        except pygame.error as e:
            print(f"Audio init failed: {e}")
            return

        # --- Engine pitch ladder ---
        # Clean tones, no subharmonic — adjacent tones already overlap during
        # crossfade and a subharmonic would just add another dissonant voice.
        self.engine_sounds   = []
        self.engine_channels = [None] * self.N_TONES
        for i in range(self.N_TONES):
            t = i / (self.N_TONES - 1)
            freq = self.FREQ_LO * (self.FREQ_HI / self.FREQ_LO) ** t
            # 4 harmonics gives engine-like timbre without being too buzzy
            wave = _saw_harmonics(freq, 1.2, harmonics=4)
            self.engine_sounds.append(_to_sound(wave))

        # --- Squeal: filtered noise loop ---
        squeal = _lowpass(_noise(0.8), 5) - _lowpass(_noise(0.8), 35)
        squeal /= np.max(np.abs(squeal)) + 1e-9
        self.squeal      = _to_sound(squeal * 0.5)
        self.squeal_ch   = None
        self._squealing  = False

        # --- Thunk: short noise burst, exponential decay ---
        n = int(0.18 * SAMPLE_RATE)
        t = np.random.uniform(-1, 1, n).astype(np.float32)
        t *= np.exp(-np.linspace(0, 5, n))
        t = _lowpass(t, 40)
        t /= np.max(np.abs(t)) + 1e-9
        self.thunk = _to_sound(t * 0.5)

        # --- Crunch: longer, heavier ---
        n = int(0.4 * SAMPLE_RATE)
        c = np.random.uniform(-1, 1, n).astype(np.float32)
        c *= np.exp(-np.linspace(0, 4, n))
        c = _lowpass(c, 20)
        c /= np.max(np.abs(c)) + 1e-9
        self.crunch = _to_sound(c * 0.7)

        # Rate limiters
        self._last_thunk  = 0.0
        self._last_crunch = 0.0

    # ------------------------------------------------------------------

    def _ensure_engine_playing(self):
        if not self.ok:
            return
        for i, snd in enumerate(self.engine_sounds):
            ch = self.engine_channels[i]
            if ch is None or not ch.get_busy():
                ch = snd.play(loops=-1)
                if ch:
                    ch.set_volume(0.0)
                self.engine_channels[i] = ch

    def update(self, leader_speed, leader_is_drifting, max_speed=540.0):
        """
        Per-frame update of continuous sounds.
        Engine pitch sweeps continuously with speed by crossfading the
        two ladder tones nearest to the current speed-position.
        """
        if not self.enabled or not self.ok:
            return

        self._ensure_engine_playing()

        # Speed → position in the tone ladder, with a slight idle bias so
        # low speeds still have a clear engine presence.
        speed_t  = max(0.0, min(1.0, leader_speed / max_speed))
        # Soft-curve: sqrt makes the lower half of the speed range
        # occupy more tone-ladder slots → more audible pitch detail.
        tone_pos = (speed_t ** 0.65) * (self.N_TONES - 1)

        # Sharper triangular envelope — half-width 0.6 instead of 1.0.
        # At any given moment only the 1 nearest tone is loud and the next
        # one is just starting to fade in. No simultaneous chord ringing.
        ENVELOPE_HALF = 0.6
        MAX_TONE_VOL  = 0.55
        for i, ch in enumerate(self.engine_channels):
            if ch is None:
                continue
            dist = abs(i - tone_pos)
            if dist >= ENVELOPE_HALF:
                vol = 0.0
            else:
                # Equal-power crossfade so total perceived loudness stays flat
                t   = 1.0 - (dist / ENVELOPE_HALF)
                vol = (t ** 0.5) * MAX_TONE_VOL
            ch.set_volume(vol)

        if leader_is_drifting:
            if not self._squealing:
                self.squeal_ch = self.squeal.play(loops=-1)
                self._squealing = True
            if self.squeal_ch:
                self.squeal_ch.set_volume(0.35)
        else:
            if self._squealing and self.squeal_ch:
                self.squeal_ch.stop()
                self._squealing = False

    def play_thunk(self, sim_time):
        if not self.enabled or not self.ok:
            return
        if sim_time - self._last_thunk < 0.08:
            return
        self.thunk.play()
        self._last_thunk = sim_time

    def play_crunch(self, sim_time):
        if not self.enabled or not self.ok:
            return
        if sim_time - self._last_crunch < 0.15:
            return
        self.crunch.play()
        self._last_crunch = sim_time

    def stop_all(self):
        for ch in self.engine_channels:
            if ch:
                ch.stop()
        self.engine_channels = [None] * self.N_TONES
        if self.squeal_ch:
            self.squeal_ch.stop()
        self.squeal_ch  = None
        self._squealing = False

    def toggle(self):
        self.enabled = not self.enabled
        if not self.enabled:
            self.stop_all()
        return self.enabled
