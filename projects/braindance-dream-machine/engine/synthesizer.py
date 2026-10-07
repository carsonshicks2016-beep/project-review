"""
Synthetic Audio Generator for Atmospheric Jungle & Braindance IDM
Creates starter audio clips for immediate testing of the RAVE pipeline.
"""

import numpy as np
import soundfile as sf
import os
from pathlib import Path

SR = 44100

def generate_sub_bass(duration: float, freq: float = 45.0) -> np.ndarray:
    """Generates deep 808 sub-bass with gentle saturation."""
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    # Fundamental + slight 2nd harmonic
    sub = np.sin(2 * np.pi * freq * t) + 0.2 * np.sin(2 * np.pi * freq * 2 * t)
    # Soft saturation
    sub = np.tanh(sub * 1.5)
    return sub

def generate_atmospheric_pad(duration: float) -> np.ndarray:
    """
    LTJ Bukem style lush minor 9th / 11th ambient pad.
    Rich saw/tri waves filtered with slow sinusoidal LFO.
    """
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    # Chord: D min9 (D3=146.8, F3=174.6, A3=220.0, C4=261.6, E4=329.6)
    freqs = [146.83, 174.61, 220.00, 261.63, 329.63, 440.00]
    signal = np.zeros_like(t)

    for i, f in enumerate(freqs):
        # Detuned oscillators for lush chorus
        detune = 1.0 + (i % 3 - 1) * 0.003
        osc1 = np.sin(2 * np.pi * f * t)
        osc2 = 0.5 * (2 * (t * f * detune - np.floor(t * f * detune + 0.5))) # tri/saw
        signal += (osc1 + osc2)

    # Apply slow LFO filter / volume sweep (0.2 Hz)
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.15 * t)
    signal = signal * lfo

    # Add deep sub bass underneath
    sub = generate_sub_bass(duration, 36.7) # D1
    signal = 0.7 * signal + 0.5 * sub

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.85
    return signal

def generate_braindance_breakbeat(duration: float, bpm: float = 165.0) -> np.ndarray:
    """
    Aphex Twin style sliced, micro-edited breakbeat with kicks, snappy snares,
    ghost notes, and glitch stutters.
    """
    num_samples = int(SR * duration)
    signal = np.zeros(num_samples)
    beat_sec = 60.0 / bpm
    sixteenth = beat_sec / 4.0

    # Create kick drum synthesis
    def synth_kick():
        k_dur = 0.18
        k_t = np.linspace(0, k_dur, int(SR * k_dur), endpoint=False)
        pitch_env = 130.0 * np.exp(-35.0 * k_t) + 45.0
        phase = 2 * np.pi * np.cumsum(pitch_env) / SR
        amp_env = np.exp(-18.0 * k_t)
        return np.sin(phase) * amp_env

    # Create snare synthesis (burst + noise)
    def synth_snare(pitch_mult=1.0):
        s_dur = 0.15
        s_t = np.linspace(0, s_dur, int(SR * s_dur), endpoint=False)
        body = np.sin(2 * np.pi * 200.0 * pitch_mult * s_t) * np.exp(-25.0 * s_t)
        noise = (np.random.rand(len(s_t)) * 2 - 1) * np.exp(-18.0 * s_t)
        return 0.5 * body + 0.5 * noise

    # Create hihat / ghost hit
    def synth_hihat():
        h_dur = 0.05
        h_t = np.linspace(0, h_dur, int(SR * h_dur), endpoint=False)
        noise = (np.random.rand(len(h_t)) * 2 - 1) * np.exp(-40.0 * h_t)
        return noise * 0.4

    # Sequence pattern: standard amen-like base with Aphex stutter variations
    kick = synth_kick()
    snare = synth_snare()
    hihat = synth_hihat()

    t_curr = 0.0
    step = 0
    while t_curr < duration - beat_sec:
        idx = int(t_curr * SR)
        pattern_pos = step % 16

        # Kick on 0, 10
        if pattern_pos in [0, 10]:
            end_idx = min(num_samples, idx + len(kick))
            signal[idx:end_idx] += kick[:end_idx - idx] * 0.9

        # Snare on 4, 12, with occasional ghost on 7 or 15
        if pattern_pos in [4, 12]:
            # Occasional Aphex micro-pitch modulation
            pitch_shift = 1.0 if step % 32 < 24 else 1.3
            sn = synth_snare(pitch_shift)
            end_idx = min(num_samples, idx + len(sn))
            signal[idx:end_idx] += sn[:end_idx - idx] * 0.8
        elif pattern_pos in [7, 15]:
            end_idx = min(num_samples, idx + len(hihat))
            signal[idx:end_idx] += hihat[:end_idx - idx] * 0.4

        # Hihat on every eighth / sixteenth
        if pattern_pos % 2 == 0:
            end_idx = min(num_samples, idx + len(hihat))
            signal[idx:end_idx] += hihat[:end_idx - idx] * 0.3

        # Aphex micro-stutter edit at bar turnaround (step 28-31)
        if step % 32 >= 28 and step % 2 == 0:
            stutter = synth_snare(1.5)[:int(SR * 0.03)]
            end_idx = min(num_samples, idx + len(stutter))
            signal[idx:end_idx] += stutter[:end_idx - idx] * 0.7

        t_curr += sixteenth
        step += 1

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.9
    return signal

def generate_acid_bassline(duration: float, bpm: float = 165.0) -> np.ndarray:
    """
    Aphex-style TB-303 acid sequence with resonant filter envelope & slides.
    """
    num_samples = int(SR * duration)
    signal = np.zeros(num_samples)
    beat_sec = 60.0 / bpm
    sixteenth = beat_sec / 4.0

    # Acid notes (minor pentatonic / dorian)
    notes = [110.0, 110.0, 130.8, 146.8, 164.8, 220.0, 110.0, 196.0]
    t_curr = 0.0
    step = 0

    while t_curr < duration - sixteenth:
        idx = int(t_curr * SR)
        freq = notes[step % len(notes)]
        n_dur = sixteenth * (1.5 if step % 4 == 3 else 0.8) # glide / staccato
        n_t = np.linspace(0, n_dur, int(SR * n_dur), endpoint=False)

        # Saw wave
        saw = 2 * (n_t * freq - np.floor(n_t * freq + 0.5))

        # Resonant filter modulation (approximate with harmonic ring)
        cutoff = 800.0 + 1200.0 * np.exp(-12.0 * n_t)
        res = np.sin(2 * np.pi * cutoff * n_t) * 0.3
        note_audio = (saw + res) * np.exp(-5.0 * n_t)

        end_idx = min(num_samples, idx + len(note_audio))
        signal[idx:end_idx] += note_audio[:end_idx - idx] * 0.7

        t_curr += sixteenth
        step += 1

    signal = signal / (np.max(np.abs(signal)) + 1e-6) * 0.8
    return signal

def generate_starter_pack(out_dir: str = "data/demo", duration: float = 12.0):
    """Generates 3 starter audio tracks for testing."""
    os.makedirs(out_dir, exist_ok=True)
    out_path = Path(out_dir)

    print("Synthesizing starter atmospheric pad (Bukem style)...")
    pad = generate_atmospheric_pad(duration)
    sf.write(out_path / "bukem_atmospheric_pad.wav", pad, SR)

    print("Synthesizing starter braindance breakbeats (Aphex style)...")
    breaks = generate_braindance_breakbeat(duration, bpm=165.0)
    sf.write(out_path / "aphex_braindance_breakbeat.wav", breaks, SR)

    print("Synthesizing starter acid bassline...")
    acid = generate_acid_bassline(duration, bpm=165.0)
    sf.write(out_path / "aphex_acid_sequence.wav", acid, SR)

    # Combined hybrid track
    print("Synthesizing hybrid jungle/IDM demo track...")
    hybrid = pad * 0.6 + breaks * 0.7 + acid * 0.4
    hybrid = hybrid / (np.max(np.abs(hybrid)) + 1e-6) * 0.9
    sf.write(out_path / "hybrid_aphex_bukem_demo.wav", hybrid, SR)

    print(f"Starter pack created in '{out_dir}'!")
    return [
        str(out_path / "bukem_atmospheric_pad.wav"),
        str(out_path / "aphex_braindance_breakbeat.wav"),
        str(out_path / "aphex_acid_sequence.wav"),
        str(out_path / "hybrid_aphex_bukem_demo.wav")
    ]

if __name__ == "__main__":
    generate_starter_pack()
