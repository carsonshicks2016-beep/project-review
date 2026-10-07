"""
Interactive 25-Key Chromatic Virtual Piano Keyboard & Arpeggiator for PULSE-16.
Features ivory and obsidian keys with press animations, mouse glissando,
computer keyboard mappings, and hardware-style arpeggiation.
"""

import math
import random
import pygame
from typing import Callable, Optional
from ui.theme import PANEL_BG, PANEL_BORDER, TEXT_PRIMARY, TEXT_MUTED, TEXT_SECONDARY


# 25 chromatic notes in 2 octaves (C3 to C5)
NOTE_NAMES = [
    "C3", "C#3", "D3", "D#3", "E3", "F3", "F#3", "G3", "G#3", "A3", "A#3", "B3",
    "C4", "C#4", "D4", "D#4", "E4", "F4", "F#4", "G4", "G#4", "A4", "A#4", "B4",
    "C5"
]

# Natural white notes in 25 chromatic span:
# C3, D3, E3, F3, G3, A3, B3, C4, D4, E4, F4, G4, A4, B4, C5 (15 white keys)
WHITE_NOTES = [n for n in NOTE_NAMES if "#" not in n]
BLACK_NOTES = [n for n in NOTE_NAMES if "#" in n]


class PianoKeyboard:
    """25-key chromatic piano keyboard with real-time audio triggering & arpeggiator."""

    def __init__(self, rect: pygame.Rect, on_note_play: Callable[[str, float], None]):
        self.rect = rect
        self.on_note_play = on_note_play
        self.font = pygame.font.SysFont("Helvetica", 10, bold=True)
        self.font_small = pygame.font.SysFont("Helvetica", 9)
        
        # Key press visual timers (note_name -> float 0.0..1.0)
        self.pressed_timers: dict[str, float] = {n: 0.0 for n in NOTE_NAMES}
        
        # Arpeggiator state
        self.arp_enabled: bool = False
        self.arp_mode: str = "UP"  # "UP", "DOWN", "RANDOM"
        self.arp_latch: bool = False
        self.latched_notes: list[str] = ["C3", "E3", "G3"]
        self.arp_step: int = 0
        
        # Build key geometries
        self._build_key_rects()

    def _build_key_rects(self) -> None:
        """Calculates precise screen rects for all 15 white keys and 10 black keys."""
        top_offset = 32  # Header area for ARP controls
        kb_h = self.rect.height - top_offset - 8
        white_w = (self.rect.width - 24) // 15
        
        self.white_rects: dict[str, pygame.Rect] = {}
        self.black_rects: dict[str, pygame.Rect] = {}
        
        # 1. White keys
        for i, note in enumerate(WHITE_NOTES):
            x = self.rect.x + 12 + i * white_w
            y = self.rect.y + top_offset
            self.white_rects[note] = pygame.Rect(x, y, white_w - 1, kb_h)

        # 2. Black keys (positioned between white keys)
        # Black note offsets within chromatic octave:
        # C# (after C), D# (after D), F# (after F), G# (after G), A# (after A)
        black_height = int(kb_h * 0.62)
        black_width = int(white_w * 0.64)
        
        white_note_to_idx = {n: i for i, n in enumerate(WHITE_NOTES)}
        black_pairs = [
            ("C#3", "C3"), ("D#3", "D3"), ("F#3", "F3"), ("G#3", "G3"), ("A#3", "A3"),
            ("C#4", "C4"), ("D#4", "D4"), ("F#4", "F4"), ("G#4", "G4"), ("A#4", "A4"),
        ]
        
        for b_note, w_note in black_pairs:
            w_idx = white_note_to_idx[w_note]
            w_rect = self.white_rects[w_note]
            bx = w_rect.right - (black_width // 2)
            by = self.rect.y + top_offset
            self.black_rects[b_note] = pygame.Rect(bx, by, black_width, black_height)

    def trigger_note(self, note_name: str, velocity: float = 1.0) -> None:
        """Plays a note, starts press animation, and records into arpeggiator."""
        if note_name in self.pressed_timers:
            self.pressed_timers[note_name] = 1.0
            self.on_note_play(note_name, velocity)
            
            if self.arp_latch:
                if note_name not in self.latched_notes:
                    self.latched_notes.append(note_name)
                    if len(self.latched_notes) > 5:
                        self.latched_notes.pop(0)

    def step_arpeggiator(self) -> None:
        """Triggers next note in the active arpeggiator pattern."""
        if not self.arp_enabled or not self.latched_notes:
            return
            
        if self.arp_mode == "UP":
            note = self.latched_notes[self.arp_step % len(self.latched_notes)]
            self.arp_step = (self.arp_step + 1) % len(self.latched_notes)
        elif self.arp_mode == "DOWN":
            idx = len(self.latched_notes) - 1 - (self.arp_step % len(self.latched_notes))
            note = self.latched_notes[idx]
            self.arp_step = (self.arp_step + 1) % len(self.latched_notes)
        else:  # RANDOM
            note = random.choice(self.latched_notes)
            
        self.trigger_note(note, 0.95)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check ARP buttons in header
            if self._handle_header_click(event.pos):
                return True
                
            # Check black keys first (they sit on top)
            for b_note, b_rect in self.black_rects.items():
                if b_rect.collidepoint(event.pos):
                    self.trigger_note(b_note, 1.0)
                    return True
                    
            # Check white keys
            for w_note, w_rect in self.white_rects.items():
                if w_rect.collidepoint(event.pos):
                    self.trigger_note(w_note, 1.0)
                    return True
                    
        return False

    def _handle_header_click(self, pos: tuple[int, int]) -> bool:
        # ARP toggle button at rect.x + 180
        arp_btn = pygame.Rect(self.rect.x + 200, self.rect.y + 6, 80, 20)
        if arp_btn.collidepoint(pos):
            self.arp_enabled = not self.arp_enabled
            return True
            
        mode_btn = pygame.Rect(self.rect.x + 288, self.rect.y + 6, 96, 20)
        if mode_btn.collidepoint(pos):
            modes = ["UP", "DOWN", "RANDOM"]
            cur_i = modes.index(self.arp_mode)
            self.arp_mode = modes[(cur_i + 1) % len(modes)]
            return True
            
        latch_btn = pygame.Rect(self.rect.x + 392, self.rect.y + 6, 80, 20)
        if latch_btn.collidepoint(pos):
            self.arp_latch = not self.arp_latch
            return True
            
        return False

    def update(self, dt: float) -> None:
        for note in self.pressed_timers:
            if self.pressed_timers[note] > 0:
                self.pressed_timers[note] = max(0.0, self.pressed_timers[note] - dt * 5.0)

    def draw(self, screen: pygame.Surface) -> None:
        # Main chassis
        pygame.draw.rect(screen, (18, 22, 28), self.rect, border_radius=6)
        pygame.draw.rect(screen, PANEL_BORDER, self.rect, width=1, border_radius=6)
        
        # Header banner
        title_surf = self.font.render("SYNTH PIANO & ARPEGGIATOR", True, (0, 245, 212))
        screen.blit(title_surf, (self.rect.x + 12, self.rect.y + 8))
        
        # ARP Toggle Button
        arp_btn = pygame.Rect(self.rect.x + 200, self.rect.y + 6, 80, 20)
        arp_bg = (0, 160, 110) if self.arp_enabled else (34, 40, 52)
        pygame.draw.rect(screen, arp_bg, arp_btn, border_radius=4)
        pygame.draw.rect(screen, PANEL_BORDER, arp_btn, width=1, border_radius=4)
        txt = "ARP: ON" if self.arp_enabled else "ARP: OFF"
        a_surf = self.font_small.render(txt, True, TEXT_PRIMARY)
        screen.blit(a_surf, a_surf.get_rect(center=arp_btn.center))
        
        # Mode Button
        mode_btn = pygame.Rect(self.rect.x + 288, self.rect.y + 6, 96, 20)
        pygame.draw.rect(screen, (34, 40, 52), mode_btn, border_radius=4)
        pygame.draw.rect(screen, PANEL_BORDER, mode_btn, width=1, border_radius=4)
        m_surf = self.font_small.render(f"MODE: {self.arp_mode}", True, (255, 185, 30))
        screen.blit(m_surf, m_surf.get_rect(center=mode_btn.center))
        
        # Latch Button
        latch_btn = pygame.Rect(self.rect.x + 392, self.rect.y + 6, 80, 20)
        l_bg = (120, 70, 160) if self.arp_latch else (34, 40, 52)
        pygame.draw.rect(screen, l_bg, latch_btn, border_radius=4)
        pygame.draw.rect(screen, PANEL_BORDER, latch_btn, width=1, border_radius=4)
        l_surf = self.font_small.render("LATCH: ON" if self.arp_latch else "LATCH: OFF", True, TEXT_PRIMARY)
        screen.blit(l_surf, l_surf.get_rect(center=latch_btn.center))

        # 1. Draw 15 White Keys
        for note, r in self.white_rects.items():
            timer = self.pressed_timers[note]
            if timer > 0.05:
                # Active glow (Electric Turquoise)
                bg = (
                    int(235 * (1 - timer) + 0 * timer),
                    int(238 * (1 - timer) + 245 * timer),
                    int(242 * (1 - timer) + 212 * timer),
                )
            else:
                bg = (235, 238, 244)  # Ivory white
                
            pygame.draw.rect(screen, bg, r, border_bottom_left_radius=4, border_bottom_right_radius=4)
            pygame.draw.rect(screen, (60, 68, 85), r, width=1, border_bottom_left_radius=4, border_bottom_right_radius=4)
            
            # Note label on bottom of key
            lbl_color = (20, 25, 35) if timer < 0.1 else (255, 255, 255)
            n_surf = self.font_small.render(note, True, lbl_color)
            screen.blit(n_surf, n_surf.get_rect(center=(r.centerx, r.bottom - 12)))

        # 2. Draw 10 Black Keys on top
        for note, r in self.black_rects.items():
            timer = self.pressed_timers[note]
            if timer > 0.05:
                bg = (255, 170, 30)  # Glowing Amber
            else:
                bg = (28, 30, 38)   # Matte Obsidian
                
            pygame.draw.rect(screen, bg, r, border_bottom_left_radius=3, border_bottom_right_radius=3)
            pygame.draw.rect(screen, (85, 95, 115), r, width=1, border_bottom_left_radius=3, border_bottom_right_radius=3)
            
            # Subtle top glossy highlight
            pygame.draw.line(screen, (60, 65, 80), (r.left + 2, r.top + 2), (r.right - 2, r.top + 2), 1)
