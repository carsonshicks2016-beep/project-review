"""
Sequencer Engine for PULSE-16.
Features 4 Multi-Pattern Memory Banks (A, B, C, D), Song Mode Arranger timeline,
3-state velocity accents (Off -> Normal -> Accent), and MPC-style swing math.
"""

import json
import time
from typing import Optional
from patterns.presets import (
    TRACK_NAMES,
    PRESETS,
    empty_grid,
    get_preset_trap,
    generate_algorithmic_groove
)
from engine.audio_manager import AudioManager


class Sequencer:
    """Accurate 16-step sequencer with 4 pattern banks, song mode, accents, and swing."""

    def __init__(self, audio_manager: AudioManager):
        self.audio = audio_manager
        
        # 4 Pattern Banks: A, B, C, D
        # 0 = Off, 1 = Normal, 2 = Accent
        self.banks: dict[str, dict[str, list[int]]] = {
            "A": self._make_int_grid(empty_grid()),
            "B": self._make_int_grid(empty_grid()),
            "C": self._make_int_grid(empty_grid()),
            "D": self._make_int_grid(empty_grid()),
        }
        self.active_bank: str = "A"
        
        # Song Mode Arranger
        self.song_mode: bool = False
        self.song_chain: list[str] = ["A", "A", "B", "B", "C", "C", "D", "D"]
        self.song_chain_idx: int = 0
        self.bars_per_pattern: int = 1  # Number of 16-step cycles per song block
        self.current_bar_count: int = 0
        
        self.bpm: float = 120.0
        self.swing: float = 0.0
        
        self.is_playing: bool = False
        self.current_step: int = 0
        self.last_step_time: float = 0.0
        self.next_step_duration: float = self._calculate_step_duration(0)
        
        # Load default Cinematic Epic into Bank A
        self.load_preset("Cinematic Epic")

    def _make_int_grid(self, bool_grid: dict[str, list[bool]]) -> dict[str, list[int]]:
        return {k: [1 if v else 0 for v in steps] for k, steps in bool_grid.items()}

    @property
    def grid(self) -> dict[str, list[int]]:
        return self.banks[self.active_bank]

    def set_bank(self, bank_name: str) -> None:
        if bank_name in self.banks:
            self.active_bank = bank_name

    def toggle_song_mode(self) -> bool:
        self.song_mode = not self.song_mode
        if self.song_mode:
            self.song_chain_idx = 0
            self.current_bar_count = 0
            self.active_bank = self.song_chain[0]
        return self.song_mode

    def copy_current_bank_to(self, target_bank: str) -> None:
        if target_bank in self.banks:
            src = self.grid
            self.banks[target_bank] = {k: list(v) for k, v in src.items()}

    def _calculate_step_duration(self, step_idx: int) -> float:
        base_16th = (60.0 / self.bpm) / 4.0
        swing_amount = self.swing * 0.4
        if step_idx % 2 == 0:
            return base_16th * (1.0 + swing_amount)
        else:
            return base_16th * (1.0 - swing_amount)

    def play(self) -> None:
        if not self.is_playing:
            self.is_playing = True
            self.last_step_time = time.perf_counter()
            self.next_step_duration = self._calculate_step_duration(self.current_step)
            self._trigger_step(self.current_step)

    def pause(self) -> None:
        self.is_playing = False

    def stop(self) -> None:
        self.is_playing = False
        self.current_step = 0
        self.current_bar_count = 0
        if self.song_mode:
            self.song_chain_idx = 0
            self.active_bank = self.song_chain[0]

    def toggle_play(self) -> bool:
        if self.is_playing:
            self.pause()
        else:
            self.play()
        return self.is_playing

    def cycle_step(self, track: str, step: int) -> int:
        if track in self.grid and 0 <= step < 16:
            curr = self.grid[track][step]
            new_val = (curr + 1) % 3
            self.grid[track][step] = new_val
            
            if new_val == 1:
                self.audio.play_sound(track, velocity=0.85)
            elif new_val == 2:
                self.audio.play_sound(track, velocity=1.30)
                
            return new_val
        return 0

    def set_bpm(self, bpm: float) -> None:
        self.bpm = max(40.0, min(240.0, float(bpm)))
        self.next_step_duration = self._calculate_step_duration(self.current_step)

    def set_swing(self, swing: float) -> None:
        self.swing = max(0.0, min(0.5, float(swing)))
        self.next_step_duration = self._calculate_step_duration(self.current_step)

    def clear_active_bank(self) -> None:
        self.banks[self.active_bank] = self._make_int_grid(empty_grid())

    def load_preset(self, preset_name: str) -> bool:
        if preset_name in PRESETS:
            data = PRESETS[preset_name]()
            self.banks[self.active_bank] = self._make_int_grid(data["grid"])
            self.bpm = data["bpm"]
            self.swing = data["swing"]
            return True
        return False

    def generate_random_groove(self) -> str:
        data = generate_algorithmic_groove()
        int_grid = self._make_int_grid(data["grid"])
        for track, steps in int_grid.items():
            for s in range(16):
                if steps[s] == 1 and (s == 0 or s == 4 or s == 12):
                    if time.perf_counter() % 2 > 0.7:
                        steps[s] = 2
                        
        self.banks[self.active_bank] = int_grid
        self.bpm = data["bpm"]
        self.swing = data["swing"]
        return data["name"]

    def _trigger_step(self, step_idx: int) -> None:
        active_grid = self.grid
        for track in TRACK_NAMES:
            val = active_grid.get(track, [0] * 16)[step_idx]
            if val == 1:
                self.audio.play_sound(track, velocity=0.85)
            elif val == 2:
                self.audio.play_sound(track, velocity=1.30)

    def update(self) -> None:
        if not self.is_playing:
            return

        now = time.perf_counter()
        elapsed = now - self.last_step_time

        if elapsed >= self.next_step_duration:
            next_s = (self.current_step + 1) % 16
            
            # Step wrapped to downbeat step 0: bar cycle complete
            if next_s == 0:
                self.current_bar_count += 1
                if self.song_mode:
                    if self.current_bar_count >= self.bars_per_pattern:
                        self.current_bar_count = 0
                        self.song_chain_idx = (self.song_chain_idx + 1) % len(self.song_chain)
                        self.active_bank = self.song_chain[self.song_chain_idx]
                        
            self.current_step = next_s
            self._trigger_step(self.current_step)
            self.last_step_time = now
            self.next_step_duration = self._calculate_step_duration(self.current_step)

    def get_boolean_grid_for_export(self) -> dict[str, list[bool]]:
        return {k: [v > 0 for v in steps] for k, steps in self.grid.items()}
