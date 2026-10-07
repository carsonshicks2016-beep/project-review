"""
PULSE-16: Procedural Drum Machine, Synth & Orchestra Sequencer
Main application entry point with interactive Pygame UI, Song Mode Arranger,
25-Key Chromatic Virtual Piano Keyboard & Arpeggiator, and Multi-Track Stems Exporter.
"""

import os
import sys
import time
import pygame

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.audio_manager import AudioManager
from engine.sequencer import Sequencer
from patterns.presets import (
    TRACK_NAMES,
    TRACK_LABELS,
    TRACK_KEYBINDS,
    TRACK_CATEGORIES,
    PRESETS,
)
from ui.theme import (
    BG_MAIN,
    PANEL_BG,
    PANEL_BORDER,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TEXT_MUTED,
    COLOR_PLAY,
    COLOR_MUTE,
    COLOR_SOLO,
    COLOR_BANK_ACTIVE,
    WINDOW_WIDTH,
    WINDOW_HEIGHT,
)
from ui.components import (
    Button,
    StepButton,
    TrackTriggerPad,
    ValueStepper,
    FxSlider,
)
from ui.visualizer import AudioVisualizer
from ui.piano_keyboard import PianoKeyboard


class Pulse16App:
    """Main application manager for PULSE-16."""

    def __init__(self):
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("PULSE-16 | Procedural Studio DAW & Synthesizer")
        
        self.width = 1260
        self.height = 920
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        
        # Typography
        self.font_title = pygame.font.SysFont("Helvetica", 20, bold=True)
        self.font_main = pygame.font.SysFont("Helvetica", 13, bold=True)
        self.font_small = pygame.font.SysFont("Helvetica", 11)
        self.font_tiny = pygame.font.SysFont("Helvetica", 10)
        
        # Audio Engine & Sequencer
        self.audio = AudioManager()
        self.sequencer = Sequencer(self.audio)
        
        # Active Category Tab: "DRUMS", "ORCHESTRA", "SYNTH", "ALL"
        self.active_category = "ORCHESTRA"
        self.all_page = 0
        
        # Virtual Piano toggle
        self.show_piano = True
        
        # Toast Notifications
        self.toast_msg = "PULSE-16 Studio Ready! Virtual Piano, Arp & Song Mode loaded."
        self.toast_timer = 4.0
        
        # Layout initialization
        self._init_layout()
        self.running = True

    def _get_visible_tracks(self) -> list[str]:
        if self.active_category == "DRUMS":
            return [t for t in TRACK_NAMES if TRACK_CATEGORIES.get(t) == "DRUMS"]
        elif self.active_category == "ORCHESTRA":
            return [t for t in TRACK_NAMES if TRACK_CATEGORIES.get(t) == "ORCHESTRA"]
        elif self.active_category == "SYNTH":
            return [t for t in TRACK_NAMES if TRACK_CATEGORIES.get(t) == "SYNTH"]
        else:  # ALL
            start = self.all_page * 13
            return TRACK_NAMES[start:start + 13]

    def _init_layout(self) -> None:
        """Initializes all buttons, steppers, FX sliders, and grid pads."""
        # 1. Dual-Mode Visualizer (Top Right)
        vis_x = self.width - 290
        self.visualizer = AudioVisualizer(pygame.Rect(vis_x, 14, 274, 82))
        
        # 2. Top Transport & Steppers (Top Bar)
        y_top = 16
        self.btn_play = Button(
            pygame.Rect(170, y_top, 80, 38),
            "▶ PLAY",
            self.font_main,
            bg_color=(20, 85, 55),
            hover_color=(28, 115, 75),
            on_click=self._toggle_play
        )
        self.btn_stop = Button(
            pygame.Rect(255, y_top, 58, 38),
            "⏹ STOP",
            self.font_main,
            bg_color=(48, 38, 44),
            hover_color=(70, 52, 60),
            on_click=self._stop
        )
        self.stepper_bpm = ValueStepper(
            pygame.Rect(318, y_top - 4, 95, 46),
            "TEMPO",
            lambda: self.sequencer.bpm,
            lambda v: self.sequencer.set_bpm(v),
            step_delta=2.0,
            format_str="{:.0f} BPM",
            font=self.font_main,
            small_font=self.font_tiny
        )
        self.stepper_swing = ValueStepper(
            pygame.Rect(418, y_top - 4, 88, 46),
            "SWING",
            lambda: self.sequencer.swing * 100,
            lambda v: self.sequencer.set_swing(v / 100),
            step_delta=5.0,
            format_str="{:.0f}%",
            font=self.font_main,
            small_font=self.font_tiny
        )
        
        # 3. Master FX Rack Sliders
        fx_y = y_top - 4
        self.slider_drive = FxSlider(
            pygame.Rect(512, fx_y, 90, 46),
            "DRIVE",
            lambda: self.audio.fx_drive,
            lambda v: self.audio.set_fx_drive(v),
            format_str="{:.0f}%",
            font=self.font_main,
            small_font=self.font_tiny,
            color=(255, 110, 50)
        )
        self.slider_filter = FxSlider(
            pygame.Rect(607, fx_y, 90, 46),
            "LP FILTER",
            lambda: self.audio.fx_filter_lp,
            lambda v: self.audio.set_fx_filter(v),
            format_str="{:.0f}%",
            font=self.font_main,
            small_font=self.font_tiny,
            color=(0, 220, 255)
        )
        self.slider_delay = FxSlider(
            pygame.Rect(702, fx_y, 90, 46),
            "DELAY",
            lambda: self.audio.fx_delay,
            lambda v: self.audio.set_fx_delay(v),
            format_str="{:.0f}%",
            font=self.font_main,
            small_font=self.font_tiny,
            color=(180, 140, 255)
        )
        
        # 4. Action Buttons
        self.btn_random = Button(
            pygame.Rect(798, y_top, 82, 38),
            "🎲 RANDOM",
            self.font_main,
            bg_color=(56, 42, 85),
            hover_color=(80, 60, 120),
            on_click=self._generate_random
        )
        self.btn_clear = Button(
            pygame.Rect(885, y_top, 58, 38),
            "CLEAR",
            self.font_main,
            bg_color=(42, 46, 56),
            hover_color=(60, 66, 80),
            on_click=self._clear_grid
        )
        
        # 5. Pattern Bank Selector Buttons [A] [B] [C] [D]
        self.bank_buttons: list[Button] = []
        bx = 210
        by = 66
        for b_name in ["A", "B", "C", "D"]:
            b_btn = Button(
                pygame.Rect(bx, by, 34, 25),
                b_name,
                self.font_main,
                bg_color=(32, 38, 50),
                hover_color=(45, 54, 70),
                on_click=lambda n=b_name: self._switch_bank(n)
            )
            self.bank_buttons.append(b_btn)
            bx += 38
            
        # 6. Song Mode Toggle & Chain
        self.btn_song_mode = Button(
            pygame.Rect(bx + 8, by, 90, 25),
            "SONG: OFF",
            self.font_small,
            bg_color=(38, 44, 56),
            hover_color=(52, 60, 76),
            on_click=self._toggle_song_mode
        )
        
        # 7. Preset Buttons
        self.preset_buttons: list[Button] = []
        px = 530
        for p_name in PRESETS.keys():
            p_btn = Button(
                pygame.Rect(px, by, 84, 25),
                p_name,
                self.font_small,
                bg_color=(30, 36, 46),
                hover_color=(44, 52, 66),
                on_click=lambda n=p_name: self._load_preset(n)
            )
            self.preset_buttons.append(p_btn)
            px += 88
            
        # 8. Category Navigation Tabs
        self.cat_buttons: list[Button] = []
        cats = [
            ("ORCHESTRA", "ORCHESTRA (7)"),
            ("DRUMS", "DRUMS (11)"),
            ("SYNTH", "SYNTH & BASS (8)"),
            ("ALL", "ALL TRACKS (26)"),
        ]
        cx = 18
        cy = 108
        for cat_id, cat_lbl in cats:
            c_btn = Button(
                pygame.Rect(cx, cy, 134, 28),
                cat_lbl,
                self.font_small,
                bg_color=(28, 32, 42),
                hover_color=(40, 48, 62),
                on_click=lambda c=cat_id: self._set_category(c)
            )
            self.cat_buttons.append(c_btn)
            cx += 140
            
        # Piano Keyboard Toggle Button
        self.btn_toggle_piano = Button(
            pygame.Rect(cx + 8, cy, 134, 28),
            "PIANO & ARP",
            self.font_small,
            bg_color=(35, 50, 70),
            hover_color=(50, 70, 98),
            on_click=self._toggle_piano_view
        )
        
        # Export WAV and Export Stems Buttons
        self.btn_export = Button(
            pygame.Rect(self.width - 240, cy, 110, 28),
            "MASTER WAV",
            self.font_small,
            bg_color=(25, 75, 105),
            hover_color=(35, 105, 145),
            on_click=self._export_wav
        )
        self.btn_stems = Button(
            pygame.Rect(self.width - 124, cy, 108, 28),
            "STEMS WAV",
            self.font_small,
            bg_color=(75, 45, 95),
            hover_color=(105, 65, 135),
            on_click=self._export_stems
        )

        # 9. All Track Pads, Mutes, Solos, and Step Buttons
        self.pads: dict[str, TrackTriggerPad] = {}
        self.mute_btns: dict[str, Button] = {}
        self.solo_btns: dict[str, Button] = {}
        self.step_buttons: dict[str, list[StepButton]] = {}
        
        pad_w = 124
        mute_w = 25
        solo_w = 25
        step_w = 48
        step_h = 36
        gap = 5
        group_gap = 12
        start_steps_x = 18 + pad_w + mute_w + solo_w + 22

        for track in TRACK_NAMES:
            dummy_rect = pygame.Rect(18, 0, pad_w, step_h)
            pad = TrackTriggerPad(
                dummy_rect,
                track,
                TRACK_LABELS[track],
                TRACK_KEYBINDS[track],
                self.font_main,
                self.font_small,
                on_trigger=lambda t=track: self.audio.play_sound(t)
            )
            self.pads[track] = pad
            
            m_btn = Button(
                pygame.Rect(0, 0, mute_w, step_h),
                "M",
                self.font_main,
                on_click=lambda t=track: self.audio.toggle_mute(t)
            )
            self.mute_btns[track] = m_btn
            
            s_btn = Button(
                pygame.Rect(0, 0, solo_w, step_h),
                "S",
                self.font_main,
                on_click=lambda t=track: self.audio.toggle_solo(t)
            )
            self.solo_btns[track] = s_btn
            
            self.step_buttons[track] = []
            curr_x = start_steps_x
            for step_i in range(16):
                s_rect = pygame.Rect(curr_x, 0, step_w, step_h)
                s_btn = StepButton(s_rect, track, step_i)
                self.step_buttons[track].append(s_btn)
                curr_x += step_w + gap
                if (step_i + 1) % 4 == 0 and step_i < 15:
                    curr_x += group_gap

        # 10. Interactive 25-Key Virtual Piano Keyboard & Arpeggiator
        piano_h = 138
        piano_y = self.height - piano_h - 44
        self.piano_keyboard = PianoKeyboard(
            pygame.Rect(18, piano_y, self.width - 36, piano_h),
            on_note_play=lambda n, v: self.audio.play_piano_note(n, v)
        )

    def _toggle_piano_view(self) -> None:
        self.show_piano = not self.show_piano
        self.btn_toggle_piano.is_active = self.show_piano
        status = "OPEN" if self.show_piano else "MINIMIZED"
        self.show_toast(f"Virtual Piano Keyboard: {status}")

    def _toggle_song_mode(self) -> None:
        is_song = self.sequencer.toggle_song_mode()
        self.btn_song_mode.text = "SONG: ON" if is_song else "SONG: OFF"
        self.btn_song_mode.bg_color = (0, 140, 100) if is_song else (38, 44, 56)
        mode_str = "Song Mode ON (Auto-chains Banks A-A-B-B-C-C-D-D)" if is_song else "Looping single pattern bank"
        self.show_toast(mode_str)

    def _set_category(self, cat: str) -> None:
        self.active_category = cat
        self.show_toast(f"Displaying: {cat} tracks")

    def _switch_bank(self, bank: str) -> None:
        self.sequencer.set_bank(bank)
        self.show_toast(f"Switched to Pattern Bank {bank}")

    def _toggle_play(self) -> None:
        is_playing = self.sequencer.toggle_play()
        self.btn_play.text = "⏸ PAUSE" if is_playing else "▶ PLAY"
        self.btn_play.bg_color = (130, 95, 12) if is_playing else (20, 85, 55)

    def _stop(self) -> None:
        self.sequencer.stop()
        self.btn_play.text = "▶ PLAY"
        self.btn_play.bg_color = (20, 85, 55)
        self.show_toast("Sequencer stopped and rewound.")

    def _generate_random(self) -> None:
        name = self.sequencer.generate_random_groove()
        self.show_toast(f"Generated {name}! ({self.sequencer.bpm:.0f} BPM)")

    def _clear_grid(self) -> None:
        self.sequencer.clear_active_bank()
        self.show_toast(f"Cleared Pattern Bank {self.sequencer.active_bank}.")

    def _load_preset(self, name: str) -> None:
        self.sequencer.load_preset(name)
        self.show_toast(f"Loaded preset: {name}")

    def _export_wav(self) -> None:
        out_filename = "pulse16_master_loop.wav"
        export_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), out_filename)
        bool_grid = self.sequencer.get_boolean_grid_for_export()
        self.audio.export_pattern_to_wav(
            bool_grid,
            self.sequencer.bpm,
            self.sequencer.swing,
            export_path,
            bars=2
        )
        self.show_toast(f"Exported master loop to {out_filename}!")

    def _export_stems(self) -> None:
        stems_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stems")
        bool_grid = self.sequencer.get_boolean_grid_for_export()
        files = self.audio.export_stems(
            bool_grid,
            self.sequencer.bpm,
            self.sequencer.swing,
            stems_dir,
            bars=2
        )
        self.show_toast(f"Exported {len(files)} separate audio stems to stems/ folder!")

    def show_toast(self, message: str) -> None:
        self.toast_msg = message
        self.toast_timer = 3.5

    def handle_keyboard(self, event: pygame.event.Event) -> None:
        if event.type != pygame.KEYDOWN:
            return

        if event.key == pygame.K_SPACE:
            self._toggle_play()
        elif event.key == pygame.K_ESCAPE:
            self._stop()
        elif event.key == pygame.K_g:
            self._generate_random()
        elif event.key == pygame.K_v and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.visualizer.toggle_mode()
        elif event.key == pygame.K_p and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self._toggle_piano_view()
        elif event.key == pygame.K_F12:
            from PIL import Image
            bmp = "temp_shot.bmp"
            png = "pulse16_screenshot.png"
            pygame.image.save(self.screen, bmp)
            im = Image.open(bmp)
            im.save(png)
            if os.path.exists(bmp):
                os.remove(bmp)
            self.show_toast("Saved screenshot to pulse16_screenshot.png!")

        # Keypad / Alpha finger drumming
        char = event.unicode.upper()
        for track, bind in TRACK_KEYBINDS.items():
            if char == bind and track in self.pads:
                self.pads[track].trigger()

    def handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return
            elif event.type == pygame.KEYDOWN:
                self.handle_keyboard(event)

            # Header buttons
            self.btn_play.handle_event(event)
            self.btn_stop.handle_event(event)
            self.stepper_bpm.handle_event(event)
            self.stepper_swing.handle_event(event)
            self.slider_drive.handle_event(event)
            self.slider_filter.handle_event(event)
            self.slider_delay.handle_event(event)
            self.btn_random.handle_event(event)
            self.btn_clear.handle_event(event)
            self.btn_export.handle_event(event)
            self.btn_stems.handle_event(event)
            self.btn_song_mode.handle_event(event)
            self.btn_toggle_piano.handle_event(event)
            self.visualizer.handle_event(event)

            # Bank buttons
            for b_btn in self.bank_buttons:
                b_btn.handle_event(event)

            # Presets
            for p_btn in self.preset_buttons:
                p_btn.handle_event(event)

            # Category tabs
            for c_btn in self.cat_buttons:
                c_btn.handle_event(event)

            # Virtual Piano Keyboard
            if self.show_piano:
                self.piano_keyboard.handle_event(event)

            # Visible Track rows
            visible = self._get_visible_tracks()
            for track in visible:
                self.pads[track].handle_event(event)
                self.mute_btns[track].handle_event(event)
                self.solo_btns[track].handle_event(event)

                for step_idx, step_btn in enumerate(self.step_buttons[track]):
                    if step_btn.handle_event(event):
                        self.sequencer.cycle_step(track, step_idx)

    def update(self, dt: float) -> None:
        prev_step = self.sequencer.current_step
        self.sequencer.update()
        cur_step = self.sequencer.current_step
        
        # Beat flash on downbeat (steps 0, 4, 8, 12)
        if cur_step != prev_step:
            if cur_step % 4 == 0:
                self.visualizer.trigger_downbeat_pulse()
            # Step the arpeggiator in sync with sequencer
            if self.piano_keyboard.arp_enabled:
                self.piano_keyboard.step_arpeggiator()
            
        self.audio.decay_visualizer(dt)
        for pad in self.pads.values():
            pad.update(dt)
            
        if self.show_piano:
            self.piano_keyboard.update(dt)
            
        if self.toast_timer > 0:
            self.toast_timer = max(0.0, self.toast_timer - dt)

    def draw(self) -> None:
        self.screen.fill(BG_MAIN)
        
        # 1. Top Header Box
        top_rect = pygame.Rect(10, 10, self.width - 20, 90)
        pygame.draw.rect(self.screen, PANEL_BG, top_rect, border_radius=8)
        pygame.draw.rect(self.screen, PANEL_BORDER, top_rect, width=1, border_radius=8)
        
        # Logo & Subtitle
        t_surf = self.font_title.render("⚡ PULSE-16", True, (0, 245, 212))
        self.screen.blit(t_surf, (20, 20))
        sub_surf = self.font_tiny.render("PROCEDURAL STUDIO DAW", True, TEXT_MUTED)
        self.screen.blit(sub_surf, (22, 44))

        # Draw Top Controls
        self.btn_play.draw(self.screen)
        self.btn_stop.draw(self.screen)
        self.stepper_bpm.draw(self.screen)
        self.stepper_swing.draw(self.screen)
        self.slider_drive.draw(self.screen)
        self.slider_filter.draw(self.screen)
        self.slider_delay.draw(self.screen)
        self.btn_random.draw(self.screen)
        self.btn_clear.draw(self.screen)
        
        # Bank Selector & Song Mode
        bank_lbl = self.font_tiny.render("BANK:", True, TEXT_MUTED)
        self.screen.blit(bank_lbl, (170, 72))
        for b_btn in self.bank_buttons:
            b_btn.is_active = (b_btn.text == self.sequencer.active_bank)
            b_btn.draw(self.screen)
            
        self.btn_song_mode.draw(self.screen)
            
        # Preset Buttons
        pre_lbl = self.font_tiny.render("PRESETS:", True, TEXT_MUTED)
        self.screen.blit(pre_lbl, (474, 72))
        for p_btn in self.preset_buttons:
            p_btn.draw(self.screen)
            
        # Visualizer
        vis_buffer, peak = self.audio.get_visualizer_data()
        self.visualizer.draw(self.screen, vis_buffer, peak)

        # 2. Category Filter Bar
        cat_rect = pygame.Rect(10, 104, self.width - 20, 36)
        pygame.draw.rect(self.screen, (18, 22, 28), cat_rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, cat_rect, width=1, border_radius=6)
        
        for c_btn in self.cat_buttons:
            c_btn.is_active = (
                (self.active_category == "ORCHESTRA" and "ORCHESTRA" in c_btn.text) or
                (self.active_category == "DRUMS" and "DRUMS" in c_btn.text) or
                (self.active_category == "SYNTH" and "SYNTH" in c_btn.text) or
                (self.active_category == "ALL" and "ALL" in c_btn.text)
            )
            c_btn.draw(self.screen)
            
        self.btn_toggle_piano.draw(self.screen)
        self.btn_export.draw(self.screen)
        self.btn_stems.draw(self.screen)

        # 3. Step Number Header
        visible_tracks = self._get_visible_tracks()
        active_step = self.sequencer.current_step if self.sequencer.is_playing else -1
        
        if visible_tracks:
            first_t = visible_tracks[0]
            for s_idx in range(16):
                btn_rect = self.step_buttons[first_t][s_idx].rect
                is_downbeat = (s_idx % 4 == 0)
                is_cur = (s_idx == active_step)
                
                num_text = str(s_idx + 1)
                num_color = (0, 255, 220) if is_cur else (TEXT_PRIMARY if is_downbeat else TEXT_MUTED)
                num_surf = self.font_small.render(num_text, True, num_color)
                self.screen.blit(num_surf, num_surf.get_rect(center=(btn_rect.centerx, 154)))
                
                if is_downbeat:
                    b_num = (s_idx // 4) + 1
                    b_surf = self.font_tiny.render(f"B{b_num}", True, (255, 185, 30))
                    self.screen.blit(b_surf, b_surf.get_rect(center=(btn_rect.centerx, 142)))

        # 4. Sequencer Grid Rows
        grid_start_y = 168
        # If piano is visible, limit rows to max 11 or scale row_h
        row_h = 42 if self.show_piano else 45
        pad_w = 124
        mute_w = 25
        solo_w = 25

        # Render visible rows
        max_display = 11 if self.show_piano else len(visible_tracks)
        for r_i, track in enumerate(visible_tracks[:max_display]):
            ry = grid_start_y + r_i * row_h
            
            self.pads[track].rect.y = ry
            self.pads[track].draw(self.screen)
            
            self.mute_btns[track].rect.x = 18 + pad_w + 6
            self.mute_btns[track].rect.y = ry
            is_muted = self.audio.muted.get(track, False)
            self.mute_btns[track].bg_color = COLOR_MUTE if is_muted else (34, 40, 52)
            self.mute_btns[track].text_color = (255, 255, 255) if is_muted else TEXT_SECONDARY
            self.mute_btns[track].draw(self.screen)
            
            self.solo_btns[track].rect.x = 18 + pad_w + mute_w + 10
            self.solo_btns[track].rect.y = ry
            is_soloed = self.audio.soloed.get(track, False)
            self.solo_btns[track].bg_color = COLOR_SOLO if is_soloed else (34, 40, 52)
            self.solo_btns[track].text_color = (0, 0, 0) if is_soloed else TEXT_SECONDARY
            self.solo_btns[track].draw(self.screen)
            
            track_steps = self.sequencer.grid.get(track, [0] * 16)
            for s_idx, step_btn in enumerate(self.step_buttons[track]):
                step_btn.rect.y = ry
                state = track_steps[s_idx]
                is_playhead = (s_idx == active_step)
                step_btn.draw(self.screen, state, is_playhead)

        # 5. Interactive Virtual Piano Keyboard (If toggled on)
        if self.show_piano:
            self.piano_keyboard.draw(self.screen)

        # 6. Bottom Status Bar & Toast Notifications
        bot_rect = pygame.Rect(10, self.height - 38, self.width - 20, 28)
        pygame.draw.rect(self.screen, PANEL_BG, bot_rect, border_radius=6)
        pygame.draw.rect(self.screen, PANEL_BORDER, bot_rect, width=1, border_radius=6)
        
        if self.toast_timer > 0:
            toast_surf = self.font_main.render(self.toast_msg, True, (0, 245, 212))
            self.screen.blit(toast_surf, (20, self.height - 32))
        else:
            hint_str = "SPACE: Play/Pause  |  Z-M: Orchestra  |  1-0: Drums  |  Click Keys to Play Piano  |  Shift+P: Toggle Piano"
            hint_surf = self.font_small.render(hint_str, True, TEXT_MUTED)
            self.screen.blit(hint_surf, (20, self.height - 30))
            
        dsp_badge = "26 VOICES • VIRTUAL PIANO • ARP • STEMS EXPORTER • 16-BIT 44.1kHz"
        badge_surf = self.font_tiny.render(dsp_badge, True, (70, 200, 170))
        self.screen.blit(badge_surf, (self.width - 440, self.height - 30))

        pygame.display.flip()

    def run(self) -> None:
        last_time = time.perf_counter()
        while self.running:
            now = time.perf_counter()
            dt = min(0.1, now - last_time)
            last_time = now
            
            self.handle_events()
            self.update(dt)
            self.draw()
            self.clock.tick(60)
            
        pygame.quit()


def main():
    app = Pulse16App()
    app.run()


if __name__ == "__main__":
    main()
