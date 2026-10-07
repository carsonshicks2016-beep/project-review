"""
Interactive UI Components for PULSE-16.
Buttons, multi-state 16-step pads with velocity accents, Master FX knobs,
Bank selector buttons, and Category filter tabs.
"""

from typing import Callable, Optional
import pygame
from ui.theme import (
    PANEL_BG,
    PANEL_BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    STEP_INACTIVE_A,
    STEP_INACTIVE_B,
    STEP_PLAYHEAD_BG,
    TRACK_COLORS,
    COLOR_BANK_ACTIVE,
)


class Button:
    """General interactive push button with hover and active states."""

    def __init__(
        self,
        rect: pygame.Rect,
        text: str,
        font: pygame.font.Font,
        bg_color: tuple[int, int, int] = (38, 44, 56),
        hover_color: tuple[int, int, int] = (50, 58, 74),
        text_color: tuple[int, int, int] = TEXT_PRIMARY,
        border_radius: int = 5,
        on_click: Optional[Callable[[], None]] = None,
    ):
        self.rect = rect
        self.text = text
        self.font = font
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.text_color = text_color
        self.border_radius = border_radius
        self.on_click = on_click
        self.is_active = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True
        return False

    def draw(self, screen: pygame.Surface) -> None:
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        
        bg = self.hover_color if hovered else self.bg_color
        pygame.draw.rect(screen, bg, self.rect, border_radius=self.border_radius)
        
        border_c = (0, 230, 190) if self.is_active else PANEL_BORDER
        pygame.draw.rect(screen, border_c, self.rect, width=1, border_radius=self.border_radius)
        
        txt_surf = self.font.render(self.text, True, self.text_color)
        txt_rect = txt_surf.get_rect(center=self.rect.center)
        screen.blit(txt_surf, txt_rect)


class StepButton:
    """
    Interactive 16-step grid button with 3 velocity states:
    0 = Off, 1 = Normal Hit, 2 = ACCENT (Punchy Boost).
    """

    def __init__(self, rect: pygame.Rect, track_name: str, step_index: int):
        self.rect = rect
        self.track_name = track_name
        self.step_index = step_index
        beat_group = (step_index // 4) % 2
        self.base_color = STEP_INACTIVE_A if beat_group == 0 else STEP_INACTIVE_B
        self.accent_color = TRACK_COLORS.get(track_name, (0, 200, 255))

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                return True
        return False

    def draw(self, screen: pygame.Surface, state: int, is_playhead: bool) -> None:
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        
        # Determine background
        if state == 2:
            # Accent: Super intense glow
            r = min(255, self.accent_color[0] + 40)
            g = min(255, self.accent_color[1] + 40)
            b = min(255, self.accent_color[2] + 40)
            bg = (r, g, b)
        elif state == 1:
            # Normal hit
            bg = self.accent_color
        else:
            # Off
            if is_playhead:
                bg = STEP_PLAYHEAD_BG
            elif hovered:
                bg = (self.base_color[0] + 16, self.base_color[1] + 16, self.base_color[2] + 16)
            else:
                bg = self.base_color

        pygame.draw.rect(screen, bg, self.rect, border_radius=5)
        
        # Border
        if state > 0:
            border_c = (255, 255, 255) if (state == 2 or is_playhead) else (200, 220, 240)
            pygame.draw.rect(screen, border_c, self.rect, width=1, border_radius=5)
            
            if state == 2:
                # Accent double-pip LED
                p1 = pygame.Rect(self.rect.centerx - 6, self.rect.centery - 4, 4, 8)
                p2 = pygame.Rect(self.rect.centerx + 2, self.rect.centery - 4, 4, 8)
                pygame.draw.rect(screen, (255, 255, 255), p1, border_radius=1)
                pygame.draw.rect(screen, (255, 255, 255), p2, border_radius=1)
            else:
                # Normal single center pip
                p = pygame.Rect(self.rect.centerx - 3, self.rect.centery - 3, 6, 6)
                pygame.draw.rect(screen, (255, 255, 255), p, border_radius=2)
        else:
            border_c = (75, 88, 110) if is_playhead else PANEL_BORDER
            pygame.draw.rect(screen, border_c, self.rect, width=1, border_radius=5)


class TrackTriggerPad:
    """Left-hand track audition pad with hotkey indicator and visual flash."""

    def __init__(
        self,
        rect: pygame.Rect,
        track_name: str,
        label: str,
        keybind: str,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        on_trigger: Callable[[], None]
    ):
        self.rect = rect
        self.track_name = track_name
        self.label = label
        self.keybind = keybind
        self.font = font
        self.small_font = small_font
        self.on_trigger = on_trigger
        self.accent_color = TRACK_COLORS.get(track_name, (200, 200, 200))
        self.flash_timer = 0.0

    def trigger(self) -> None:
        self.flash_timer = 1.0
        self.on_trigger()

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.trigger()
                return True
        return False

    def update(self, dt: float) -> None:
        if self.flash_timer > 0:
            self.flash_timer = max(0.0, self.flash_timer - dt * 6.5)

    def draw(self, screen: pygame.Surface) -> None:
        mouse_pos = pygame.mouse.get_pos()
        hovered = self.rect.collidepoint(mouse_pos)
        
        if self.flash_timer > 0.05:
            ratio = self.flash_timer
            r = int(PANEL_BG[0] * (1 - ratio) + self.accent_color[0] * ratio * 0.75)
            g = int(PANEL_BG[1] * (1 - ratio) + self.accent_color[1] * ratio * 0.75)
            b = int(PANEL_BG[2] * (1 - ratio) + self.accent_color[2] * ratio * 0.75)
            bg = (min(255, r), min(255, g), min(255, b))
        elif hovered:
            bg = (36, 42, 54)
        else:
            bg = PANEL_BG

        pygame.draw.rect(screen, bg, self.rect, border_radius=5)
        
        # Color accent strip on left edge
        accent_strip = pygame.Rect(self.rect.x, self.rect.y, 4, self.rect.height)
        pygame.draw.rect(screen, self.accent_color, accent_strip, border_top_left_radius=5, border_bottom_left_radius=5)
        pygame.draw.rect(screen, PANEL_BORDER, self.rect, width=1, border_radius=5)

        txt_surf = self.font.render(self.label, True, TEXT_PRIMARY)
        screen.blit(txt_surf, (self.rect.x + 8, self.rect.y + 3))
        
        key_surf = self.small_font.render(f"[{self.keybind}]", True, self.accent_color)
        screen.blit(key_surf, (self.rect.x + 8, self.rect.y + 19))


class FxSlider:
    """Compact Master FX parameter slider with interactive drag and fill."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        value_getter: Callable[[], float],
        value_setter: Callable[[float], None],
        format_str: str,
        font: pygame.font.Font,
        small_font: pygame.font.Font,
        color: tuple[int, int, int] = (0, 230, 200)
    ):
        self.rect = rect
        self.label = label
        self.value_getter = value_getter
        self.value_setter = value_setter
        self.format_str = format_str
        self.font = font
        self.small_font = small_font
        self.color = color
        self.is_dragging = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.is_dragging = True
                self._update_val_from_mouse(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.is_dragging = False
        elif event.type == pygame.MOUSEMOTION and self.is_dragging:
            self._update_val_from_mouse(event.pos[0])
            return True
        return False

    def _update_val_from_mouse(self, mouse_x: int) -> None:
        rel = (mouse_x - (self.rect.x + 8)) / max(1, (self.rect.width - 16))
        clamped = max(0.0, min(1.0, rel))
        self.value_setter(clamped)

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, PANEL_BG, self.rect, border_radius=5)
        pygame.draw.rect(screen, PANEL_BORDER, self.rect, width=1, border_radius=5)
        
        # Label & value text
        lbl_surf = self.small_font.render(self.label, True, TEXT_MUTED)
        screen.blit(lbl_surf, (self.rect.x + 8, self.rect.y + 2))
        
        val_str = self.format_str.format(self.value_getter() * 100)
        val_surf = self.small_font.render(val_str, True, self.color)
        screen.blit(val_surf, (self.rect.right - val_surf.get_width() - 8, self.rect.y + 2))
        
        # Track rail & active fill
        bar_x = self.rect.x + 8
        bar_y = self.rect.y + 16
        bar_w = self.rect.width - 16
        bar_h = 6
        pygame.draw.rect(screen, (36, 42, 54), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        
        fill_w = int(self.value_getter() * bar_w)
        if fill_w > 0:
            pygame.draw.rect(screen, self.color, (bar_x, bar_y, fill_w, bar_h), border_radius=3)
            # Handle pip
            pygame.draw.circle(screen, (255, 255, 255), (bar_x + fill_w, bar_y + bar_h // 2), 4)


class ValueStepper:
    """Numeric adjuster with - and + buttons (BPM, Swing)."""

    def __init__(
        self,
        rect: pygame.Rect,
        label: str,
        value_getter: Callable[[], float],
        value_setter: Callable[[float], None],
        step_delta: float,
        format_str: str,
        font: pygame.font.Font,
        small_font: pygame.font.Font
    ):
        self.rect = rect
        self.label = label
        self.value_getter = value_getter
        self.value_setter = value_setter
        self.step_delta = step_delta
        self.format_str = format_str
        self.font = font
        self.small_font = small_font
        
        btn_w = 24
        btn_h = rect.height - 18
        self.btn_minus = pygame.Rect(rect.x + 6, rect.y + 14, btn_w, btn_h)
        self.btn_plus = pygame.Rect(rect.right - 6 - btn_w, rect.y + 14, btn_w, btn_h)

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.btn_minus.collidepoint(event.pos):
                self.value_setter(self.value_getter() - self.step_delta)
                return True
            elif self.btn_plus.collidepoint(event.pos):
                self.value_setter(self.value_getter() + self.step_delta)
                return True
        return False

    def draw(self, screen: pygame.Surface) -> None:
        pygame.draw.rect(screen, PANEL_BG, self.rect, border_radius=5)
        pygame.draw.rect(screen, PANEL_BORDER, self.rect, width=1, border_radius=5)
        
        lbl_surf = self.small_font.render(self.label, True, TEXT_MUTED)
        screen.blit(lbl_surf, (self.rect.x + 8, self.rect.y + 2))
        
        mouse_pos = pygame.mouse.get_pos()
        m_hover = self.btn_minus.collidepoint(mouse_pos)
        pygame.draw.rect(screen, (48, 56, 70) if m_hover else (34, 40, 52), self.btn_minus, border_radius=3)
        m_txt = self.font.render("-", True, TEXT_PRIMARY)
        screen.blit(m_txt, m_txt.get_rect(center=self.btn_minus.center))
        
        p_hover = self.btn_plus.collidepoint(mouse_pos)
        pygame.draw.rect(screen, (48, 56, 70) if p_hover else (34, 40, 52), self.btn_plus, border_radius=3)
        p_txt = self.font.render("+", True, TEXT_PRIMARY)
        screen.blit(p_txt, p_txt.get_rect(center=self.btn_plus.center))
        
        val_str = self.format_str.format(self.value_getter())
        val_surf = self.font.render(val_str, True, (0, 235, 210))
        val_rect = val_surf.get_rect(center=(self.rect.centerx, self.btn_minus.centery))
        screen.blit(val_surf, val_rect)
