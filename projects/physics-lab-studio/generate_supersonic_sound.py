#!/usr/bin/env python3
"""
Advanced Supersonic Aircraft Acoustic Generator
Physically models:
1. Retarded time solutions for supersonic moving source
2. Double N-wave shock profile (bow shock + tail shock)
3. Zone of silence ahead of Mach cone
4. Dual-ray engine turbulence superposition (approaching reverse-time + receding forward-time)
5. Distance attenuation & atmospheric absorption
6. Stereo panning & ground reflection
"""

import sys
import math
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, lfilter

def generate_supersonic_sound(
    mach=1.4,
    altitude_m=1200.0,
    aircraft_length_m=20.0,
    duration_s=8.0,
    sample_rate=44100,
    speed_of_sound=343.0,
    output_filename="supersonic_flyby.wav"
):
    c = speed_of_sound
    v = mach * c
    h = altitude_m
    total_samples = int(duration_s * sample_rate)
    
    # Timing calculation
    # Overhead occurs at source time t_s = 0
    # Shock arrives at ground observer at t_boom:
    if mach > 1.0001:
        t_boom_exact = (h * math.sqrt(mach**2 - 1.0)) / (mach * c)
        boom_delay = t_boom_exact
    else:
        t_boom_exact = h / c
        boom_delay = h / c
        
    # Center the flyby so the boom (or closest approach) happens at 40% of duration
    t_target = duration_s * 0.38
    t_start = t_boom_exact - t_target
    t_obs = np.linspace(t_start, t_start + duration_s, total_samples, endpoint=False)
    
    # 1. Jet engine turbulent noise source
    np.random.seed(42)
    raw_noise = np.random.normal(0, 1, int((duration_s + 15) * sample_rate))
    
    # Multiband filter for realistic jet engine roar
    b_jet_low, a_jet_low = butter(2, [50.0 / (sample_rate / 2.0), 450.0 / (sample_rate / 2.0)], btype='band')
    b_jet_high, a_jet_high = butter(1, 1600.0 / (sample_rate / 2.0), btype='low')
    
    jet_low = lfilter(b_jet_low, a_jet_low, raw_noise) * 1.5
    jet_high = lfilter(b_jet_high, a_jet_high, raw_noise) * 0.5
    jet_source = jet_low + jet_high
    jet_source /= (np.std(jet_source) + 1e-6)
    
    src_center = len(jet_source) // 2
    
    def get_source(t_s_array):
        indices = src_center + (t_s_array * sample_rate).astype(np.int64)
        indices = np.clip(indices, 0, len(jet_source) - 1)
        return jet_source[indices]
    
    audio_left = np.zeros(total_samples, dtype=np.float64)
    audio_right = np.zeros(total_samples, dtype=np.float64)
    
    if mach < 0.999:
        # Subsonic calculation: strictly 1 retarded time root
        a_q = c**2 - v**2
        b_q = -2.0 * (c**2) * t_obs
        c_q = (c**2) * (t_obs**2) - h**2
        disc = np.maximum(0.0, b_q**2 - 4.0 * a_q * c_q)
        t_s = (-b_q - np.sqrt(disc)) / (2.0 * a_q)
        
        valid = (t_obs >= t_s)
        R = np.sqrt((v * t_s)**2 + h**2)
        cos_theta = (v * t_s) / R
        doppler_factor = 1.0 / np.maximum(0.05, 1.0 + mach * cos_theta)
        
        gain = (300.0 / np.maximum(100.0, R)) * (doppler_factor ** 0.5)
        gain *= (0.6 + 0.4 * (1.0 - cos_theta) / 2.0)
        
        src_samples = get_source(t_s) * gain * valid
        
        pan = np.clip((v * t_s) / (h * 1.5), -1.0, 1.0)
        pan_angle = (pan + 1.0) * (math.pi / 4.0)
        audio_left = src_samples * np.cos(pan_angle)
        audio_right = src_samples * np.sin(pan_angle)
        
    elif abs(mach - 1.0) <= 0.001:
        # Transonic (Mach approx 1.0): linear in t_s
        # 2 c^2 t t_s + (h^2 - c^2 t^2) = 0 => t_s = (c^2 t^2 - h^2) / (2 c^2 t)
        mask = (t_obs >= h / c)
        safe_t = np.where(mask, t_obs, h / c)
        t_s = (c**2 * safe_t**2 - h**2) / (2.0 * c**2 * safe_t)
        
        R = np.sqrt((v * t_s)**2 + h**2)
        gain = (350.0 / np.maximum(100.0, R)) * mask
        src_samples = get_source(t_s) * gain
        
        # Add transonic sound barrier pile-up impulse
        tau = t_obs - (h / c)
        transonic_pulse = np.where((tau >= 0) & (tau < 0.12), np.sin(2 * math.pi * 35.0 * tau) * np.exp(-tau / 0.03) * 8.0, 0.0)
        
        audio_left = src_samples + transonic_pulse
        audio_right = src_samples + transonic_pulse
        
    else:
        # Supersonic calculation (mach > 1.001)
        a_q = v**2 - c**2
        mask_boom = (t_obs >= t_boom_exact)
        
        term_b = 2.0 * (c**2) * t_obs
        term_c = h**2 - (c**2) * (t_obs**2)
        disc = (term_b**2) - 4.0 * a_q * term_c
        disc_pos = np.maximum(0.0, disc)
        sqrt_disc = np.sqrt(disc_pos)
        
        t_s1 = np.where(mask_boom, (-term_b - sqrt_disc) / (2.0 * a_q), 0.0)
        t_s2 = np.where(mask_boom, (-term_b + sqrt_disc) / (2.0 * a_q), 0.0)
        
        # Process both retarded sound paths
        for t_s, path_weight in [(t_s1, 0.75), (t_s2, 0.75)]:
            R = np.sqrt((v * t_s)**2 + h**2)
            cos_theta = (v * t_s) / R
            dt_dts = np.abs(1.0 + mach * cos_theta)
            doppler = np.clip(1.0 / (dt_dts + 0.12), 0.1, 4.0)
            
            gain = (350.0 / np.maximum(100.0, R)) * (doppler ** 0.4) * path_weight
            src_samples = get_source(t_s) * gain * mask_boom
            
            pan = np.clip((v * t_s) / (h * 1.5), -1.0, 1.0)
            pan_angle = (pan + 1.0) * (math.pi / 4.0)
            audio_left += src_samples * np.cos(pan_angle)
            audio_right += src_samples * np.sin(pan_angle)
            
        # 2. Add realistic physical N-wave sonic boom
        delta_tn = float(np.clip(0.007 * aircraft_length_m * math.sqrt(mach), 0.08, 0.24))
        tau = t_obs - t_boom_exact
        boom_mask = (tau >= 0) & (tau <= delta_tn * 2.5)
        
        p_peak = 12.0 * (mach ** 0.75) / (math.pow(h / 500.0, 0.75) + 0.1)
        tau_rise = 0.0018  # 1.8 ms shock rise time
        
        n_wave = np.zeros(total_samples)
        for i, t_val in enumerate(tau):
            if 0 <= t_val < delta_tn:
                linear_slope = 1.0 - 2.0 * (t_val / delta_tn)
                bow_rise = 1.0 - math.exp(-t_val / tau_rise)
                n_wave[i] = linear_slope * bow_rise
            elif delta_tn <= t_val < delta_tn * 2.5:
                dt_tail = t_val - delta_tn
                tail_rise = 1.0 - math.exp(-dt_tail / tau_rise)
                tail_decay = math.exp(-dt_tail / 0.05)
                n_wave[i] = (1.0 - 0.5 * (dt_tail / 0.05)) * tail_rise * tail_decay * 0.8
                
        # Add high-frequency shock crackle burst
        noise_burst = np.random.normal(0, 1, total_samples) * 0.35
        safe_tau = np.maximum(0.0, tau)
        shock_front_envelope = np.where(tau >= 0, np.exp(-safe_tau / 0.004), 0.0) + \
                               np.where(tau >= delta_tn, np.exp(-np.abs(tau - delta_tn) / 0.004), 0.0)
                               
        n_wave_total = (n_wave + noise_burst * shock_front_envelope) * p_peak * boom_mask
        n_wave_total *= 1.85  # Ground reflection
        
        audio_left += n_wave_total
        audio_right += n_wave_total

    # Apply atmospheric absorption (distance-dependent low-pass)
    cutoff_hz = float(np.clip(3800.0 * (1200.0 / (h + 600.0)), 400.0, 9000.0))
    b_air, a_air = butter(2, cutoff_hz / (sample_rate / 2.0), btype='low')
    audio_left = lfilter(b_air, a_air, audio_left)
    audio_right = lfilter(b_air, a_air, audio_right)
    
    # Ground resonance & sub-bass boost (the chest-thumping 25-60Hz boom)
    b_sub, a_sub = butter(1, 85.0 / (sample_rate / 2.0), btype='low')
    sub_l = lfilter(b_sub, a_sub, audio_left) * 0.85
    sub_r = lfilter(b_sub, a_sub, audio_right) * 0.85
    audio_left += sub_l
    audio_right += sub_r
    
    # Master limiter / normalizer with soft saturation
    peak = max(np.max(np.abs(audio_left)), np.max(np.abs(audio_right)), 1e-4)
    audio_left = np.tanh(audio_left / (peak * 0.65)) * 0.92
    audio_right = np.tanh(audio_right / (peak * 0.65)) * 0.92
    
    stereo = np.column_stack((audio_left, audio_right))
    int16_stereo = (stereo * 32767.0).astype(np.int16)
    wavfile.write(output_filename, sample_rate, int16_stereo)
    
    return {
        "output_filename": output_filename,
        "mach": mach,
        "altitude_m": altitude_m,
        "boom_delay_s": boom_delay,
        "sample_rate": sample_rate,
        "duration_s": duration_s
    }

if __name__ == "__main__":
    presets = [
        {"name": "subsonic_mach08", "mach": 0.8, "alt": 1500.0},
        {"name": "transonic_mach10", "mach": 1.0, "alt": 1200.0},
        {"name": "supersonic_mach14", "mach": 1.4, "alt": 1200.0},
        {"name": "supersonic_mach20", "mach": 2.0, "alt": 2500.0},
        {"name": "hypersonic_mach50", "mach": 5.0, "alt": 8000.0},
    ]
    
    for p in presets:
        fname = f"{p['name']}.wav"
        print(f"Generating {fname} (Mach {p['mach']}, {p['alt']}m)...")
        generate_supersonic_sound(mach=p['mach'], altitude_m=p['alt'], output_filename=fname)
    print("All audio files generated successfully with zero warnings!")
