"""
Procedural Audio Synthesis Engine for PULSE-16 Drum Machine, Synth & Orchestra.
Generates pure mathematical waveforms:
- 808/909 Drums & Latin Percussion
- Orchestral Section (Timpani, Pizzicato Strings, Brass Horns, Tubular Bells, Orchestra Hit)
- FM Bass & Melodic Lead Synths
Zero external audio sample files required.
"""

import numpy as np

SAMPLE_RATE = 44100


def to_pcm16_stereo(mono_signal: np.ndarray) -> np.ndarray:
    """Converts a normalized float mono signal [-1.0, 1.0] to 16-bit stereo PCM array."""
    clamped = np.clip(mono_signal, -0.98, 0.98)
    pcm16 = (clamped * 32767).astype(np.int16)
    return np.column_stack((pcm16, pcm16))


def soft_clip(signal: np.ndarray, drive: float = 1.3) -> np.ndarray:
    """Applies warm analog-style soft clipping saturation."""
    return np.tanh(signal * drive) / np.tanh(drive)


def bandpass_noise(duration: float, low_f: float, high_f: float) -> np.ndarray:
    """Generates shaped white noise filtered between low_f and high_f using FFT."""
    n_samples = max(1, int(SAMPLE_RATE * duration))
    noise = np.random.uniform(-1.0, 1.0, n_samples)
    fft_spec = np.fft.rfft(noise)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / SAMPLE_RATE)
    
    band = (freqs >= low_f) & (freqs <= high_f)
    fft_spec[~band] *= 0.03
    
    filtered = np.fft.irfft(fft_spec, n_samples)
    peak = np.max(np.abs(filtered))
    if peak > 0:
        filtered /= peak
    return filtered


# ==========================================
# 1. DRUM CORE & PERCUSSION
# ==========================================

def generate_kick() -> np.ndarray:
    """TR-808 style Sub Kick: exponential pitch drop (160 Hz -> 42 Hz) with attack click."""
    duration = 0.45
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    freq_env = 42.0 + (160.0 - 42.0) * np.exp(-t * 32.0)
    phase = 2.0 * np.pi * np.cumsum(freq_env) / SAMPLE_RATE
    body = np.sin(phase) * np.exp(-t * 7.5)
    
    click_len = int(SAMPLE_RATE * 0.008)
    click_t = t[:click_len]
    click = np.sin(2.0 * np.pi * 1400.0 * click_t) * np.exp(-click_t * 600.0)
    body[:click_len] += click * 0.45
    return soft_clip(body, 1.4)


def generate_snare() -> np.ndarray:
    """Crisp Snare: dual acoustic body tone blended with filtered noise snap."""
    duration = 0.26
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    f_env = 130.0 + (190.0 - 130.0) * np.exp(-t * 22.0)
    body_phase = 2.0 * np.pi * np.cumsum(f_env) / SAMPLE_RATE
    body = np.sin(body_phase) * np.exp(-t * 16.0)
    
    noise = bandpass_noise(duration, 1200.0, 7000.0)
    noise_env = noise * np.exp(-t * 15.0)
    
    stick_len = int(SAMPLE_RATE * 0.004)
    stick = np.random.uniform(-1.0, 1.0, stick_len) * np.exp(-t[:stick_len] * 700.0)
    body[:stick_len] += stick * 0.4
    
    combined = body * 0.55 + noise_env * 0.75
    return soft_clip(combined, 1.2)


def generate_clap() -> np.ndarray:
    """808/909 Handclap: 3 pre-claps (flam) followed by a diffuse room tail."""
    duration = 0.28
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    sig = np.zeros_like(t)
    
    delays = [0.0, 0.011, 0.023]
    for delay in delays:
        idx = int(delay * SAMPLE_RATE)
        sub_len = len(t) - idx
        sub_t = t[:sub_len]
        burst_noise = bandpass_noise(len(sub_t) / SAMPLE_RATE, 800.0, 3200.0)
        burst = burst_noise * np.exp(-sub_t * 140.0)
        sig[idx:] += burst * 0.55
        
    tail_noise = bandpass_noise(duration, 900.0, 4000.0)
    tail = tail_noise * np.exp(-t * 16.0)
    sig += tail * 0.65
    return soft_clip(sig, 1.3)


def generate_rimshot() -> np.ndarray:
    """Crisp Rimshot / Side-stick: dual resonant frequencies + click."""
    duration = 0.08
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    f1 = np.sin(2.0 * np.pi * 480.0 * t) * np.exp(-t * 90.0)
    f2 = np.sin(2.0 * np.pi * 1850.0 * t) * np.exp(-t * 120.0)
    noise = np.random.uniform(-1.0, 1.0, len(t)) * np.exp(-t * 180.0)
    sig = (f1 * 0.5 + f2 * 0.5 + noise * 0.45) * 1.5
    return np.clip(sig, -1.0, 1.0)


def generate_hihat_closed() -> np.ndarray:
    """Tight Closed Hi-Hat: metallic harmonic cluster with fast 45ms decay."""
    duration = 0.055
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    metallic = (
        np.sin(2.0 * np.pi * 3200.0 * t) * 0.3 +
        np.sin(2.0 * np.pi * 5400.0 * t) * 0.4 +
        np.sin(2.0 * np.pi * 7800.0 * t) * 0.5 +
        np.sin(2.0 * np.pi * 10500.0 * t) * 0.4
    )
    noise = bandpass_noise(duration, 7000.0, 16000.0)
    env = np.exp(-t * 75.0)
    hat = (metallic * 0.4 + noise * 0.8) * env
    return np.clip(hat * 1.5, -1.0, 1.0)


def generate_hihat_open() -> np.ndarray:
    """Open Hi-Hat: 350ms sustained natural metallic acoustic decay."""
    duration = 0.38
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    metallic = (
        np.sin(2.0 * np.pi * 3150.0 * t) * 0.3 +
        np.sin(2.0 * np.pi * 5350.0 * t) * 0.4 +
        np.sin(2.0 * np.pi * 7600.0 * t) * 0.5 +
        np.sin(2.0 * np.pi * 10200.0 * t) * 0.4
    )
    noise = bandpass_noise(duration, 6500.0, 15000.0)
    env = np.exp(-t * 9.5)
    hat = (metallic * 0.4 + noise * 0.8) * env
    return np.clip(hat * 1.4, -1.0, 1.0)


def generate_shaker() -> np.ndarray:
    """Latin Shaker / Maraca: two-stage forward/back shake envelope."""
    duration = 0.09
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    noise = np.random.uniform(-1.0, 1.0, len(t))
    noise_hp = np.diff(np.diff(noise, prepend=0), prepend=0)
    att_len = int(SAMPLE_RATE * 0.015)
    env = np.ones_like(t)
    env[:att_len] = np.linspace(0.15, 1.0, att_len)
    env[att_len:] = np.exp(-np.linspace(0, 1, len(t) - att_len) * 14.0)
    sig = noise_hp * env * 1.5
    return np.clip(sig, -1.0, 1.0)


def generate_crash() -> np.ndarray:
    """Explosive Crash Cymbal: metallic cluster decaying over 1.2s."""
    duration = 1.2
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    metallic = (
        np.sin(2.0 * np.pi * 2800.0 * t) * 0.25 +
        np.sin(2.0 * np.pi * 4200.0 * t) * 0.30 +
        np.sin(2.0 * np.pi * 6700.0 * t) * 0.35 +
        np.sin(2.0 * np.pi * 9400.0 * t) * 0.25
    )
    noise = bandpass_noise(duration, 5500.0, 16000.0)
    env = np.exp(-t * 3.8)
    sig = (metallic * 0.35 + noise * 0.65) * env * 1.4
    return np.clip(sig, -1.0, 1.0)


def generate_conga() -> np.ndarray:
    """Resonant Conga: pitched skin resonance with sharp hand slap."""
    duration = 0.22
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    f_env = 220.0 + (320.0 - 220.0) * np.exp(-t * 40.0)
    phase = 2.0 * np.pi * np.cumsum(f_env) / SAMPLE_RATE
    body = np.sin(phase) * np.exp(-t * 16.0)
    slap_len = int(SAMPLE_RATE * 0.01)
    slap = np.sin(2.0 * np.pi * 920.0 * t[:slap_len]) * np.exp(-t[:slap_len] * 300.0)
    body[:slap_len] += slap * 0.6
    return soft_clip(body, 1.3)


def generate_tom() -> np.ndarray:
    """Resonant Mid Tom: pitch sweep (165 Hz -> 85 Hz) with wood strike."""
    duration = 0.32
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    f_env = 85.0 + (165.0 - 85.0) * np.exp(-t * 22.0)
    phase = 2.0 * np.pi * np.cumsum(f_env) / SAMPLE_RATE
    body = np.sin(phase) * np.exp(-t * 11.0)
    stick_len = int(SAMPLE_RATE * 0.006)
    body[:stick_len] += np.sin(2.0 * np.pi * 700.0 * t[:stick_len]) * 0.4
    return soft_clip(body, 1.3)


def generate_cowbell() -> np.ndarray:
    """808 Cowbell: dual detuned square waves at 587 Hz & 845 Hz."""
    duration = 0.24
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    s1 = np.sign(np.sin(2.0 * np.pi * 587.0 * t))
    s2 = np.sign(np.sin(2.0 * np.pi * 845.0 * t))
    combo = (s1 + s2) * 0.5 * np.exp(-t * 19.0)
    return soft_clip(combo * 0.8, 1.2)


# ==========================================
# 2. ORCHESTRAL SECTION
# ==========================================

def generate_timpani(freq_hz: float = 73.42) -> np.ndarray:
    """
    Orchestral Timpani (Kettle Drum):
    Physical membrane modal ratios (Bessel harmonics: 1.0, 1.59, 2.14, 2.30)
    with heavy felt mallet attack and thunderous hall resonance.
    """
    duration = 0.85
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Circular membrane modal partials
    m1 = np.sin(2.0 * np.pi * freq_hz * t) * np.exp(-t * 3.4)
    m2 = np.sin(2.0 * np.pi * (freq_hz * 1.59) * t) * np.exp(-t * 5.2) * 0.65
    m3 = np.sin(2.0 * np.pi * (freq_hz * 2.14) * t) * np.exp(-t * 7.5) * 0.45
    m4 = np.sin(2.0 * np.pi * (freq_hz * 2.30) * t) * np.exp(-t * 9.0) * 0.30
    
    # Felt mallet transient
    mallet_len = int(SAMPLE_RATE * 0.018)
    mallet_t = t[:mallet_len]
    mallet = np.sin(2.0 * np.pi * 340.0 * mallet_t) * np.exp(-mallet_t * 180.0)
    
    body = m1 + m2 + m3 + m4
    body[:mallet_len] += mallet * 0.55
    return soft_clip(body * 1.2, 1.3)


def generate_pizzicato_string(freq_hz: float, duration: float = 0.38) -> np.ndarray:
    """
    Orchestral Pizzicato Strings (Violin / Cello Pluck):
    Harmonic series with progressive higher-frequency damping and crisp fingertip pluck.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    sig = np.zeros_like(t)
    
    # Plucked string harmonics with frequency-dependent decay
    for h in range(1, 9):
        decay = 10.0 + h * 9.5
        sig += (1.0 / (h ** 0.85)) * np.sin(2.0 * np.pi * (freq_hz * h) * t) * np.exp(-t * decay)
        
    # Rosin/pluck friction transient
    pluck_len = int(SAMPLE_RATE * 0.006)
    pluck = np.random.uniform(-1.0, 1.0, pluck_len) * np.exp(-np.linspace(0, 1, pluck_len) * 8.0)
    sig[:pluck_len] += pluck * 0.45
    
    return soft_clip(sig * 1.35, 1.1)


def generate_brass_stab(freq_hz: float = 130.81, duration: float = 0.48) -> np.ndarray:
    """
    Cinematic Brass Ensemble Hit (French Horns & Trombones):
    Detuned sawtooth waves with dynamic brass swell envelope and lowpass filter sweep.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    
    # Detuned brass section
    saw1 = 2.0 * ((t * freq_hz) % 1.0) - 1.0
    saw2 = 2.0 * ((t * (freq_hz * 1.007)) % 1.0) - 1.0
    saw3 = 2.0 * ((t * (freq_hz * 0.993)) % 1.0) - 1.0
    brass_raw = (saw1 + saw2 + saw3) / 3.0
    
    # Brass swell envelope: 25ms attack swell, punchy sustained body, smooth decay
    att_samples = int(SAMPLE_RATE * 0.024)
    env = np.ones_like(t)
    env[:att_samples] = np.linspace(0.15, 1.0, att_samples)
    env[att_samples:] = np.exp(-np.linspace(0, 1, len(t) - att_samples) * 4.2)
    
    # Resonant filter sweep opening like a brass bell
    n = len(t)
    fft_spec = np.fft.rfft(brass_raw)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    fft_spec *= np.clip(1.0 - (freqs - 1900.0) / 2800.0, 0.04, 1.0)
    filtered = np.fft.irfft(fft_spec, n)
    
    return soft_clip(filtered * env * 1.7, 1.3)


def generate_tubular_bell(freq_hz: float = 440.0, duration: float = 1.1) -> np.ndarray:
    """
    Majestic Tubular Bells (Orchestral Chimes):
    Inharmonic cylindrical tube partials (1.0, 2.76, 5.40, 8.93) with long shimmering decay.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    partials = [1.0, 2.76, 5.40, 8.93]
    sig = np.zeros_like(t)
    
    for idx, p in enumerate(partials):
        decay = 2.8 + idx * 3.2
        sig += (1.0 / (idx + 1.0)) * np.sin(2.0 * np.pi * (freq_hz * p) * t) * np.exp(-t * decay)
        
    hammer_len = int(SAMPLE_RATE * 0.008)
    hammer = np.sin(2.0 * np.pi * 1300.0 * t[:hammer_len]) * np.exp(-t[:hammer_len] * 350.0)
    sig[:hammer_len] += hammer * 0.4
    
    return soft_clip(sig * 1.25, 1.1)


def generate_orchestra_hit(duration: float = 0.65) -> np.ndarray:
    """
    Iconic Full Orchestra Hit:
    Layered blast of Timpani sub-thud, French Horn brass stab, Pizzicato strings,
    and symphonic crash wash.
    """
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    n = len(t)
    
    timp = generate_timpani(65.41)[:n]
    brass = generate_brass_stab(130.81, duration)[:n]
    pizz = generate_pizzicato_string(392.0, duration)[:n]
    crash = generate_crash()[:n]
    
    combo = timp * 0.45 + brass * 0.50 + pizz * 0.35 + crash * 0.30
    return soft_clip(combo * 1.5, 1.4)


# ==========================================
# 3. SYNTHESIS BASS & MELODIC LEAD
# ==========================================

def generate_bass_note(freq_hz: float, duration: float = 0.38) -> np.ndarray:
    """FM Synth Bass: resonant low-pass filter sweep feel."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    mod_index = 1.8 * np.exp(-t * 9.0)
    modulator = np.sin(2.0 * np.pi * (freq_hz * 2.0) * t) * mod_index
    carrier = np.sin(2.0 * np.pi * freq_hz * t + modulator)
    sub = np.sin(2.0 * np.pi * freq_hz * t) * 0.5
    env = np.exp(-t * 4.8)
    sig = (carrier + sub) * env
    return soft_clip(sig, 1.3)


def generate_lead_note(freq_hz: float, wave_type: str = "saw", duration: float = 0.30) -> np.ndarray:
    """Melodic Lead Synth: saw/square wave with analog warmth filter."""
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    phase = (t * freq_hz) % 1.0
    sig = (2.0 * phase - 1.0) * 0.6
    
    n = len(sig)
    fft_spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
    lp_mask = np.clip(1.0 - (freqs - 2500.0) / 3000.0, 0.1, 1.0)
    fft_spec *= lp_mask
    filtered = np.fft.irfft(fft_spec, n)
    
    env = np.exp(-t * 6.5)
    return soft_clip(filtered * env * 1.3, 1.2)


# Pitches
BASS_NOTES = {
    "BASS_C": 65.41,    # C2
    "BASS_EB": 77.78,   # Eb2
    "BASS_F": 87.31,    # F2
    "BASS_G": 98.00,    # G2
}

LEAD_NOTES = {
    "LEAD_C3": 130.81,  # C3
    "LEAD_EB3": 155.56, # Eb3
    "LEAD_G3": 196.00,  # G3
    "LEAD_BB3": 233.08, # Bb3
}

PIZZ_NOTES = {
    "PIZZ_C": 261.63,   # C4
    "PIZZ_EB": 311.13,  # Eb4
    "PIZZ_G": 392.00,   # G4
}


def build_raw_instrument_dictionary() -> dict[str, np.ndarray]:
    """Generates and returns all raw normalized floating-point waveforms."""
    library = {
        # Core 808 Drums
        "KICK": generate_kick(),
        "SNARE": generate_snare(),
        "CLAP": generate_clap(),
        "RIMSHOT": generate_rimshot(),
        "HIHAT_CL": generate_hihat_closed(),
        "HIHAT_OP": generate_hihat_open(),
        "SHAKER": generate_shaker(),
        "CRASH": generate_crash(),
        "CONGA": generate_conga(),
        "TOM": generate_tom(),
        "COWBELL": generate_cowbell(),
        # Orchestral Section
        "TIMPANI": generate_timpani(73.42),
        "BRASS_STAB": generate_brass_stab(130.81),
        "TUBULAR_BELL": generate_tubular_bell(440.0),
        "ORCH_HIT": generate_orchestra_hit(),
    }
    
    # Pizzicato Strings
    for key, freq in PIZZ_NOTES.items():
        library[key] = generate_pizzicato_string(freq)
        
    # Bass synth notes
    for note_key, freq in BASS_NOTES.items():
        library[note_key] = generate_bass_note(freq)
        
    # Lead synth notes
    for note_key, freq in LEAD_NOTES.items():
        library[note_key] = generate_lead_note(freq, wave_type="saw")
        
    return library
