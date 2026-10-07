"""
Click-and-drag hyperparameter sliders.

Each Slider binds a label to a getter/setter pair, so changes take
effect immediately. Wrap a list of them in a SliderPanel for layout.
"""
import pygame


class Slider:
    def __init__(self, label, vmin, vmax, getter, setter, fmt="{:.2f}"):
        self.label  = label
        self.vmin   = vmin
        self.vmax   = vmax
        self.getter = getter
        self.setter = setter
        self.fmt    = fmt
        self.rect   = None   # filled in on draw, used for hit-testing


class SliderPanel:
    def __init__(self, sliders, title="HYPERPARAMS"):
        self.sliders  = sliders
        self.title    = title
        self.visible  = False
        self.dragging = None

    def toggle(self):
        self.visible = not self.visible

    def handle_event(self, event):
        if not self.visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for s in self.sliders:
                if s.rect and s.rect.collidepoint(event.pos):
                    self.dragging = s
                    self._update_from_mouse(s, event.pos[0])
                    return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self.dragging = None
                return True
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_from_mouse(self.dragging, event.pos[0])
            return True
        return False

    def _update_from_mouse(self, s, mouse_x):
        x = max(s.rect.x, min(s.rect.x + s.rect.w, mouse_x))
        t = (x - s.rect.x) / max(s.rect.w, 1)
        s.setter(s.vmin + t * (s.vmax - s.vmin))

    def draw(self, surface, font_sm, rect):
        if not self.visible:
            return
        rx, ry, rw, rh = rect

        bg = pygame.Surface((rw, rh), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 210))
        surface.blit(bg, (rx, ry))
        pygame.draw.rect(surface, (110, 110, 110), (rx, ry, rw, rh), 1)

        hdr = font_sm.render(f"{self.title}  ·  [Y] hide", True, (255, 200, 80))
        surface.blit(hdr, (rx + 8, ry + 6))

        row_h    = 34
        bar_w    = rw - 24
        bar_h    = 8

        for i, s in enumerate(self.sliders):
            sy = ry + 28 + i * row_h
            sx = rx + 12

            v = s.getter()
            txt = font_sm.render(f"{s.label}: {s.fmt.format(v)}", True, (220, 220, 220))
            surface.blit(txt, (sx, sy))

            # Hit rect (slightly taller than the bar for forgiving clicks)
            bar_y  = sy + 16
            s.rect = pygame.Rect(sx, bar_y - 4, bar_w, bar_h + 8)

            # Bar background + fill + knob
            pygame.draw.rect(surface, (50, 50, 50), (sx, bar_y, bar_w, bar_h))
            t      = (v - s.vmin) / max(s.vmax - s.vmin, 1e-9)
            t      = max(0.0, min(1.0, t))
            fill_w = int(bar_w * t)
            pygame.draw.rect(surface, (100, 180, 255),
                             (sx, bar_y, fill_w, bar_h))
            pygame.draw.rect(surface, (200, 200, 200),
                             (sx, bar_y, bar_w, bar_h), 1)
            knob_x = sx + fill_w
            pygame.draw.circle(surface, (255, 255, 255),
                               (knob_x, bar_y + bar_h // 2), 6)
            pygame.draw.circle(surface, (60, 60, 60),
                               (knob_x, bar_y + bar_h // 2), 6, 1)
