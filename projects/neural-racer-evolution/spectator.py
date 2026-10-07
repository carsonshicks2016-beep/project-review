"""
Broadcast-style spectator camera.

When enabled, every ~5 seconds the camera locks onto a different
"interesting" car (the leader, a drifter, a speedster, an underdog,
a wildcard) and procedural commentary appears in a lower-third banner.

Each car is given a procedural nickname (e.g. "Crimson Phantom #42")
and the Commentator produces a fresh, dedup'd one-liner about it.
"""
import random
import pygame


CYCLE_SECONDS = 5.0     # how long to spend on each car
MIN_DWELL     = 2.0     # don't cut away faster than this even if uninteresting


# ----------------------------------------------------------------------
# Picker functions — each finds an "interesting" car of a given type
# ----------------------------------------------------------------------

def _pick_leader(cars):
    if not cars: return None
    return max(cars, key=lambda c: c.fitness)


def _pick_drifter(cars):
    drifters = [c for c in cars if getattr(c, 'is_drifting', False)]
    if not drifters: return None
    return max(drifters, key=lambda c: c.speed)


def _pick_speedster(cars):
    fast = [c for c in cars if getattr(c, 'speed', 0) > 380]
    if not fast: return None
    return max(fast, key=lambda c: c.speed)


def _pick_underdog(cars):
    """A mid-pack car that's still alive — not the leader, not the worst."""
    if len(cars) < 4: return None
    by_fit = sorted(cars, key=lambda c: c.fitness)
    mid_start = len(by_fit) // 3
    mid_end   = (len(by_fit) * 2) // 3
    mid = by_fit[mid_start:mid_end]
    if not mid: return None
    return random.choice(mid)


def _pick_rookie(cars):
    new = [c for c in cars if getattr(c, 'time_alive', 99) < 4.0]
    if not new: return None
    return random.choice(new)


def _pick_survivor(cars):
    """Late in gen, a car with low fitness but still alive — gritty."""
    if not cars: return None
    by_fit = sorted(cars, key=lambda c: c.fitness)
    bottom = by_fit[:max(3, len(cars) // 4)]
    return random.choice(bottom)


def _pick_wildcard(cars):
    if not cars: return None
    return random.choice(cars)


PICKERS = [
    ('LEADER',     'LEADER',    _pick_leader),
    ('DRIFTER',    'DRIFT',     _pick_drifter),
    ('SPEEDSTER',  'SPEEDSTER', _pick_speedster),
    ('UNDERDOG',   'UNDERDOG',  _pick_underdog),
    ('ROOKIE',     'ROOKIE',    _pick_rookie),
    ('SURVIVOR',   'SURVIVOR',  _pick_survivor),
    ('WILDCARD',   'WILDCARD',  _pick_wildcard),
]


# ----------------------------------------------------------------------
# Spectator
# ----------------------------------------------------------------------

class Spectator:
    def __init__(self, commentator):
        self.commentator    = commentator
        self.enabled        = False
        self.current_car    = None
        self.current_label  = ""
        self.current_text   = ""
        self.next_switch    = 0.0
        self.switched_at    = 0.0
        self._picker_order  = list(range(len(PICKERS)))
        self._picker_idx    = 0
        self._pulse         = 0.0

    def toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.next_switch = 0.0   # force immediate pick
        else:
            self.current_car = None
            self.current_text = ""

    def update(self, alive_cars, sim_time):
        if not self.enabled:
            return
        if not alive_cars:
            self.current_car  = None
            self.current_text = "...waiting for cars to spawn..."
            return

        # Drop the spectated car if it died — switch immediately
        if self.current_car is not None and not self.current_car.alive:
            self.next_switch = sim_time

        if sim_time >= self.next_switch:
            self._pick_next(alive_cars, sim_time)

    def _pick_next(self, alive_cars, sim_time):
        # Walk through pickers in a shuffled order each cycle
        if self._picker_idx == 0:
            random.shuffle(self._picker_order)

        tries = 0
        while tries < len(PICKERS):
            tag, cat, picker = PICKERS[self._picker_order[self._picker_idx]]
            self._picker_idx = (self._picker_idx + 1) % len(PICKERS)
            tries += 1
            car = picker(alive_cars)
            if car is None:
                continue
            if car is self.current_car and len(alive_cars) > 1:
                continue
            self.current_car   = car
            self.current_label = tag
            self.current_text  = self.commentator.comment(car, cat)
            self.switched_at   = sim_time
            self.next_switch   = sim_time + CYCLE_SECONDS
            return

        # Nothing interesting — fall back to wildcard
        car = _pick_wildcard(alive_cars)
        if car:
            self.current_car   = car
            self.current_label = "WILDCARD"
            self.current_text  = self.commentator.comment(car, 'WILDCARD')
            self.switched_at   = sim_time
            self.next_switch   = sim_time + CYCLE_SECONDS

    # ------------------------------------------------------------------
    # Drawing — broadcast-style lower-third banner
    # ------------------------------------------------------------------

    def draw(self, surface, font_sm, font_lg, screen_w, screen_h):
        if not self.enabled:
            return

        import math
        self._pulse = (self._pulse + 0.06) % (math.pi * 2)

        # Banner geometry — bottom centre, leaves room for fitness graph + minimap
        box_w = 700
        box_h = 84
        box_x = (screen_w - box_w) // 2
        box_y = screen_h - box_h - 26   # sits above the progress bar

        bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
        bg.fill((10, 10, 14, 215))
        surface.blit(bg, (box_x, box_y))
        pygame.draw.rect(surface, (220, 180, 60), (box_x, box_y, box_w, box_h), 2)

        # Pulsing red "LIVE" dot
        pulse_r = 5 + int(2 * (1 + math.sin(self._pulse)))
        pygame.draw.circle(surface, (255, 60, 60),
                           (box_x + 18, box_y + 18), pulse_r)
        live_txt = font_sm.render("LIVE", True, (255, 200, 200))
        surface.blit(live_txt, (box_x + 30, box_y + 12))

        # Header line: car nickname + tag
        if self.current_car is None:
            header = "SPECTATOR  ·  waiting..."
        else:
            name = self.commentator.name_for(self.current_car)
            header = f"NOW SHOWING:  {name}   ·   {self.current_label}"
        h_surf = font_lg.render(header, True, (255, 220, 120))
        surface.blit(h_surf, (box_x + 80, box_y + 10))

        # Commentary line — wrapped if needed
        text  = self.current_text or ""
        lines = _word_wrap(text, font_sm, box_w - 24)
        for i, ln in enumerate(lines[:2]):
            t_surf = font_sm.render(ln, True, (235, 235, 235))
            surface.blit(t_surf, (box_x + 14, box_y + 44 + i * 18))


def _word_wrap(text, font, max_width):
    """Naive word-wrap to fit text within max_width pixels."""
    words = text.split(" ")
    lines = []
    cur   = ""
    for w in words:
        cand = cur + (" " if cur else "") + w
        if font.size(cand)[0] <= max_width:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines
