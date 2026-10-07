"""
Dual-Mode Audio Visualizer for PULSE-16.
Supports Phosphor Oscilloscope and 16-Band Real-Time FFT Spectrum Analyzer
with floating peak-hold needles and stereo VU meters.
"""

import math
import numpy as np
import pygame
from ui.theme import PANEL_BG, PANEL_BORDER, TEXT_MUTED, TEXT_PRIMARY, TRACK_COLORS


class AudioVisualizer:
    """Renders real-time audio waveforms or 16-band FFT spectrum bars."""

    def __init__(self, rect: pygame.Rect):
        self.rect = rect
        self.surface = pygame.Surface((rect.width, rect.height))
        self.mode = "FFT"  # "OSC" or "FFT"
        self.meter_decay = 0.0
        
        # FFT peak-hold memory (16 bands)
        self.num_bands = 16
        self.band_levels = np.zeros(self.num_bands, dtype=np.float32)
        self.band_peaks = np.zeros(self.num_bands, dtype=np.float32)
        
        # Beat flash timer
        self.beat_flash = 0.0

    def toggle_mode(self) -> str:
        self.mode = "OSC" if self.mode == "FFT" else "FFT"
        return self.mode

    def trigger_downbeat_pulse(self) -> None:
        self.beat_flash = 1.0

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.toggle_mode()
                return True
        return False

    def draw(self, screen: pygame.Surface, audio_buffer: np.ndarray, peak: float) -> None:
        self.surface.fill((14, 16, 22))
        
        # Downbeat pulse glow on border
        if self.beat_flash > 0:
            border_c = (
                min(255, int(42 + self.beat_flash * 120)),
                min(255, int(48 + self.beat_flash * 180)),
                min(255, int(62 + self.beat_flash * 190)),
            )
            self.beat_flash = max(0.0, self.beat_flash - 0.08)
        else:
            border_c = PANEL_BORDER

        pygame.draw.rect(self.surface, border_c, (0, 0, self.rect.width, self.rect.height), 1, border_radius=6)
        
        # Mode badge top-left
        font = pygame.font.SysFont("Helvetica", 10, bold=True)
        badge_txt = f"[{self.mode} VISUALIZER]"
        b_surf = font.render(badge_txt, True, (0, 220, 200))
        self.surface.blit(b_surf, (8, 6))

        mid_y = self.rect.height // 2
        vis_w = self.rect.width - 42  # Reserve space for right VU meter
        
        if self.mode == "OSC":
            # Oscilloscope View
            pygame.draw.line(self.surface, (24, 28, 38), (6, mid_y), (vis_w, mid_y), 1)
            pygame.draw.line(self.surface, (24, 28, 38), (vis_w // 2, 6), (vis_w // 2, self.rect.height - 6), 1)
            
            n_points = min(len(audio_buffer), 128)
            if n_points > 1:
                step_x = (vis_w - 12) / (n_points - 1)
                points = []
                for i in range(n_points):
                    sample = audio_buffer[i * (len(audio_buffer) // n_points)]
                    if peak < 0.01:
                        t = pygame.time.get_ticks() / 500.0
                        sample = math.sin(t + i * 0.18) * 0.03
                    y = mid_y - int(sample * (mid_y - 8))
                    points.append((int(i * step_x + 8), max(6, min(self.rect.height - 6, y))))
                    
                if len(points) >= 2:
                    glow_color = (0, 160, 140) if peak > 0.08 else (0, 60, 60)
                    pygame.draw.lines(self.surface, glow_color, False, points, 3)
                    core_color = (0, 255, 220) if peak > 0.08 else (0, 150, 130)
                    pygame.draw.lines(self.surface, core_color, False, points, 1)

        else:
            # 16-Band Real-Time FFT Spectrum Analyzer
            n_fft = len(audio_buffer)
            if n_fft > 0 and peak > 0.02:
                fft_vals = np.abs(np.fft.rfft(audio_buffer))
                # Distribute into 16 logarithmically spaced bins
                band_edges = np.logspace(0, np.log10(len(fft_vals)), self.num_bands + 1, dtype=int)
                for b in range(self.num_bands):
                    start_i = max(0, band_edges[b])
                    end_i = min(len(fft_vals), band_edges[b + 1])
                    if end_i > start_i:
                        mag = float(np.mean(fft_vals[start_i:end_i])) * 0.4
                    else:
                        mag = 0.0
                    target = min(1.0, mag)
                    self.band_levels[b] = self.band_levels[b] * 0.5 + target * 0.5
            else:
                # Gentle ambient idle ripple
                t = pygame.time.get_ticks() / 400.0
                for b in range(self.num_bands):
                    idle_val = 0.05 + 0.04 * math.sin(t + b * 0.4)
                    self.band_levels[b] = max(idle_val, self.band_levels[b] * 0.82)
                    
            # Update peak needles with gravity drop
            for b in range(self.num_bands):
                if self.band_levels[b] > self.band_peaks[b]:
                    self.band_peaks[b] = self.band_levels[b]
                else:
                    self.band_peaks[b] = max(0.0, self.band_peaks[b] - 0.03)

            # Draw spectrum bars
            bar_area_w = vis_w - 14
            bar_w = (bar_area_w - (self.num_bands - 1) * 3) // self.num_bands
            bar_max_h = self.rect.height - 24
            
            for b in range(self.num_bands):
                bx = 8 + b * (bar_w + 3)
                bh = int(self.band_levels[b] * bar_max_h)
                by = self.rect.height - 8 - bh
                
                # Dynamic rainbow gradient per band
                ratio = b / self.num_bands
                if ratio < 0.25:
                    bar_color = (255, 75, 75)   # Sub Bass Red
                elif ratio < 0.5:
                    bar_color = (255, 175, 30)  # Low Mid Amber
                elif ratio < 0.75:
                    bar_color = (0, 240, 160)   # Mid Mint
                else:
                    bar_color = (0, 210, 255)   # Highs Cyan
                    
                if bh > 0:
                    pygame.draw.rect(self.surface, bar_color, (bx, by, bar_w, bh), border_radius=2)
                    
                # Floating peak cap needle
                peak_h = int(self.band_peaks[b] * bar_max_h)
                peak_y = max(6, self.rect.height - 8 - peak_h - 2)
                pygame.draw.rect(self.surface, (255, 255, 255), (bx, peak_y, bar_w, 2))

        # Stereo VU Peak Meter on the far right
        meter_x = self.rect.width - 28
        meter_h = self.rect.height - 16
        meter_w = 16
        
        self.meter_decay = max(peak, self.meter_decay * 0.88)
        level_height = int(self.meter_decay * meter_h)
        level_height = min(meter_h, max(0, level_height))
        
        num_segs = 10
        seg_h = (meter_h - (num_segs - 1) * 2) // num_segs
        for s in range(num_segs):
            seg_y = self.rect.height - 8 - (s + 1) * (seg_h + 2)
            seg_ratio = s / num_segs
            
            if seg_ratio > 0.8:
                lit_color = (255, 55, 55)   # Clip Red
                dim_color = (60, 20, 20)
            elif seg_ratio > 0.55:
                lit_color = (255, 185, 30)  # Amber
                dim_color = (60, 45, 15)
            else:
                lit_color = (0, 235, 140)   # Green
                dim_color = (15, 50, 35)
                
            is_active = (s * seg_h) < level_height
            color = lit_color if is_active else dim_color
            pygame.draw.rect(self.surface, color, (meter_x, seg_y, meter_w, seg_h), border_radius=2)

        screen.blit(self.surface, self.rect.topleft)
