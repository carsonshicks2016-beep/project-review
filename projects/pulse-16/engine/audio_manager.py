"""
Audio Manager for PULSE-16.
Handles pygame mixer initialization, polyphonic sound playback,
real-time waveform data feeding, master DSP effects, WAV mixdown,
and multi-track stem export (Drums, Orchestra, Synths).
"""

import os
import wave
import numpy as np
import pygame
from engine.synth import (
    SAMPLE_RATE,
    build_raw_instrument_dictionary,
    to_pcm16_stereo,
    soft_clip
)
from patterns.presets import TRACK_CATEGORIES


class AudioManager:
    """Manages audio playback, volume levels, master FX, visualizer feeds, and WAV stems export."""

    def __init__(self):
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=SAMPLE_RATE, size=-16, channels=2, buffer=512)
        pygame.mixer.set_num_channels(64)
        
        # Master DSP Effects
        self.fx_drive: float = 0.0      # 0.0 to 1.0
        self.fx_filter_lp: float = 1.0  # 1.0 to 0.05
        self.fx_delay: float = 0.0      # 0.0 to 0.8
        
        # Base signals
        self.raw_signals = build_raw_instrument_dictionary()
        
        # Add 25 Chromatic Piano / Synth Keyboard notes (C3 to C5)
        self.chromatic_keys = []
        semitones = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        base_freq = 130.8128  # C3
        for i in range(25):
            f = base_freq * (2.0 ** (i / 12.0))
            octave = 3 + (i // 12)
            note_name = f"{semitones[i % 12]}{octave}"
            self.chromatic_keys.append(note_name)
            
            # Procedural polyphonic analog pluck note
            dur = 0.42
            t = np.linspace(0, dur, int(SAMPLE_RATE * dur), False)
            phase = (t * f) % 1.0
            sig = (2.0 * phase - 1.0) * 0.55
            # Soft resonant lowpass rolloff
            n = len(sig)
            fft_s = np.fft.rfft(sig)
            fr = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
            fft_s *= np.clip(1.0 - (fr - 2800.0) / 3200.0, 0.05, 1.0)
            sig = np.fft.irfft(fft_s, n)
            env = np.exp(-t * 5.2)
            self.raw_signals[f"PIANO_{note_name}"] = soft_clip(sig * env * 1.3, 1.2)

        self.processed_signals: dict[str, np.ndarray] = {}
        self.sounds: dict[str, pygame.mixer.Sound] = {}
        self._rebuild_processed_sounds()
        
        # Per-instrument volumes (0.0 to 1.0)
        self.volumes: dict[str, float] = {k: 0.85 for k in self.sounds}
        self.volumes["KICK"] = 0.98
        self.volumes["BASS_C"] = 0.92
        self.volumes["CRASH"] = 0.70
        self.volumes["TIMPANI"] = 0.92
        self.volumes["ORCH_HIT"] = 0.88
        
        # Track mute / solo states
        self.muted: dict[str, bool] = {k: False for k in self.sounds}
        self.soloed: dict[str, bool] = {k: False for k in self.sounds}
        
        self.master_volume: float = 0.85
        
        # Visualizer buffer
        self.vis_buffer_len = 1024
        self.vis_buffer = np.zeros(self.vis_buffer_len, dtype=np.float32)
        self.recent_peak = 0.0

    def _apply_dsp_to_signal(self, sig: np.ndarray) -> np.ndarray:
        out = sig.copy()
        if self.fx_drive > 0.01:
            drive_gain = 1.0 + self.fx_drive * 3.5
            out = soft_clip(out * drive_gain, drive=1.0 + self.fx_drive * 1.5)
            
        if self.fx_filter_lp < 0.95:
            n = len(out)
            fft_spec = np.fft.rfft(out)
            freqs = np.fft.rfftfreq(n, 1.0 / SAMPLE_RATE)
            cutoff_hz = 500.0 + self.fx_filter_lp * 19500.0
            mask = np.clip(1.0 - (freqs - cutoff_hz) / (cutoff_hz * 0.4 + 1.0), 0.02, 1.0)
            fft_spec *= mask
            out = np.fft.irfft(fft_spec, n)
            
        return np.clip(out, -1.0, 1.0)

    def _rebuild_processed_sounds(self) -> None:
        for key, raw_mono in self.raw_signals.items():
            processed = self._apply_dsp_to_signal(raw_mono)
            self.processed_signals[key] = processed
            pcm_stereo = to_pcm16_stereo(processed)
            self.sounds[key] = pygame.sndarray.make_sound(pcm_stereo)

    def set_fx_drive(self, value: float) -> None:
        self.fx_drive = max(0.0, min(1.0, float(value)))
        self._rebuild_processed_sounds()

    def set_fx_filter(self, value: float) -> None:
        self.fx_filter_lp = max(0.05, min(1.0, float(value)))
        self._rebuild_processed_sounds()

    def set_fx_delay(self, value: float) -> None:
        self.fx_delay = max(0.0, min(0.8, float(value)))

    def toggle_mute(self, key: str) -> bool:
        self.muted[key] = not self.muted.get(key, False)
        return self.muted[key]

    def toggle_solo(self, key: str) -> bool:
        self.soloed[key] = not self.soloed.get(key, False)
        return self.soloed[key]

    def is_track_audible(self, key: str) -> bool:
        any_solo = any(self.soloed.values())
        if any_solo:
            return self.soloed.get(key, False) and not self.muted.get(key, False)
        return not self.muted.get(key, False)

    def play_sound(self, key: str, velocity: float = 1.0) -> None:
        if key not in self.sounds or not self.is_track_audible(key):
            return

        sound = self.sounds[key]
        vol = self.volumes.get(key, 0.85) * velocity * self.master_volume
        sound.set_volume(max(0.0, min(1.0, vol)))
        
        channel = pygame.mixer.find_channel()
        if channel:
            channel.play(sound)
        else:
            sound.play()
            
        raw = self.processed_signals.get(key, self.raw_signals[key])
        chunk_len = min(len(raw), self.vis_buffer_len)
        self.vis_buffer[:chunk_len] = raw[:chunk_len] * vol
        self.recent_peak = max(self.recent_peak, float(np.max(np.abs(raw[:chunk_len])) * vol))

    def play_piano_note(self, note_name: str, velocity: float = 1.0) -> None:
        """Plays a chromatic piano key (e.g. 'C3', 'Eb4')."""
        key_id = f"PIANO_{note_name}"
        self.play_sound(key_id, velocity)

    def decay_visualizer(self, dt: float) -> None:
        self.vis_buffer *= max(0.0, 1.0 - dt * 14.0)
        self.recent_peak = max(0.0, self.recent_peak - dt * 2.5)

    def get_visualizer_data(self) -> tuple[np.ndarray, float]:
        return self.vis_buffer, self.recent_peak

    def render_submix(
        self,
        grid: dict[str, list[bool]],
        bpm: float,
        swing: float,
        category_filter: str = "ALL",
        bars: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """Renders an audio submix buffer for a specific instrument category."""
        step_duration = (60.0 / bpm) / 4.0
        steps_per_bar = 16
        total_steps = steps_per_bar * bars
        total_time = total_steps * step_duration + 2.0
        total_samples = int(SAMPLE_RATE * total_time)
        
        sub_left = np.zeros(total_samples, dtype=np.float32)
        sub_right = np.zeros(total_samples, dtype=np.float32)
        
        step_starts = []
        for s in range(total_steps):
            bar_s = s % steps_per_bar
            sw = (swing * step_duration * 0.4) if (bar_s % 2 == 1) else 0.0
            step_starts.append(int(((s * step_duration) + sw) * SAMPLE_RATE))
            
        for track_name, steps in grid.items():
            if track_name not in self.processed_signals:
                continue
            cat = TRACK_CATEGORIES.get(track_name, "ALL")
            if category_filter != "ALL" and cat != category_filter:
                continue
                
            raw = self.processed_signals[track_name]
            track_vol = self.volumes.get(track_name, 0.85) * self.master_volume
            scaled = raw * track_vol
            sig_len = len(scaled)
            
            for s_idx, sample_start in enumerate(step_starts):
                bar_step = s_idx % steps_per_bar
                if steps[bar_step]:
                    end_idx = min(sample_start + sig_len, total_samples)
                    actual_len = end_idx - sample_start
                    if actual_len > 0:
                        sub_left[sample_start:end_idx] += scaled[:actual_len]
                        sub_right[sample_start:end_idx] += scaled[:actual_len]
                        
        # Delay on submix if enabled
        if self.fx_delay > 0.05 and category_filter != "DRUMS":
            delay_sec = step_duration * 3.0
            delay_samples = int(delay_sec * SAMPLE_RATE)
            if delay_samples < total_samples:
                wet = self.fx_delay * 0.55
                del_l = np.zeros_like(sub_left)
                del_r = np.zeros_like(sub_right)
                del_r[delay_samples:] += sub_left[:-delay_samples] * wet
                del_l[delay_samples * 2:] += del_r[delay_samples:-delay_samples] * 0.4
                sub_left += del_l
                sub_right += del_r

        peak = max(np.max(np.abs(sub_left)), np.max(np.abs(sub_right)), 1e-6)
        if peak > 0.98:
            sub_left = soft_clip(sub_left / peak * 1.05, 1.2)
            sub_right = soft_clip(sub_right / peak * 1.05, 1.2)
            
        return sub_left, sub_right

    def _write_wav_file(self, left: np.ndarray, right: np.ndarray, filepath: str) -> None:
        pcm_l = (np.clip(left, -0.98, 0.98) * 32767).astype(np.int16)
        pcm_r = (np.clip(right, -0.98, 0.98) * 32767).astype(np.int16)
        stereo = np.column_stack((pcm_l, pcm_r)).tobytes()
        with wave.open(filepath, "wb") as wf:
            wf.setnchannels(2)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(stereo)

    def export_pattern_to_wav(
        self,
        grid: dict[str, list[bool]],
        bpm: float,
        swing: float,
        filepath: str,
        bars: int = 2
    ) -> None:
        """Renders complete master mix loop to a stereo WAV file."""
        left, right = self.render_submix(grid, bpm, swing, "ALL", bars)
        self._write_wav_file(left, right, filepath)

    def export_stems(
        self,
        grid: dict[str, list[bool]],
        bpm: float,
        swing: float,
        output_dir: str,
        bars: int = 2
    ) -> list[str]:
        """
        Exports separate WAV stems for:
        1. Drums
        2. Orchestra
        3. Synth & Bass
        4. Full Master Mix
        """
        os.makedirs(output_dir, exist_ok=True)
        created_files = []
        
        categories = [
            ("drums_stem.wav", "DRUMS"),
            ("orchestra_stem.wav", "ORCHESTRA"),
            ("synth_stem.wav", "SYNTH"),
            ("full_master_mix.wav", "ALL"),
        ]
        
        for filename, cat in categories:
            l, r = self.render_submix(grid, bpm, swing, cat, bars)
            out_path = os.path.join(output_dir, filename)
            self._write_wav_file(l, r, out_path)
            created_files.append(out_path)
            
        return created_files
