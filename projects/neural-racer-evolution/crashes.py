"""
Crash highlight recorder + slow-motion replay.

Maintains a rolling 2-second history of every car's pose. When a car dies
at high speed, snapshots the surrounding context as a "highlight."
Press P to play back the most dramatic crash in slow-motion with a
camera locked to the victim.
"""
import math
import pygame


HISTORY_FRAMES = 120        # ~2 seconds at 60 Hz
MAX_HIGHLIGHTS = 6          # cap stored crashes
MIN_IMPACT_PXS = 80.0       # ignore deaths slower than this
CAR_W, CAR_H   = 12, 22


class CrashRecorder:
    def __init__(self):
        self.history    = []   # list of frames; each frame = list of car records
        self.highlights = []   # (impact, victim_id, frames_snapshot, gen, lap)

    def record(self, cars, generation):
        """Snapshot current frame; detect deaths since the previous record."""
        frame = [
            (id(c), c.x, c.y, c.angle, c.alive, c.vx, c.vy, tuple(c.color),
             c.laps)
            for c in cars
        ]
        self.history.append(frame)
        if len(self.history) > HISTORY_FRAMES:
            self.history.pop(0)

        if len(self.history) < 2:
            return

        prev = {row[0]: row for row in self.history[-2]}
        curr = {row[0]: row for row in self.history[-1]}
        for cid, cf in curr.items():
            pf = prev.get(cid)
            if pf is None:
                continue
            # Just died this frame
            if pf[4] and not cf[4]:
                impact = (pf[5] * pf[5] + pf[6] * pf[6]) ** 0.5
                if impact < MIN_IMPACT_PXS:
                    continue
                snapshot = [list(f) for f in self.history]   # deep copy
                self.highlights.append((impact, cid, snapshot, generation, cf[8]))

        # Cap highlights — keep the best ones
        if len(self.highlights) > MAX_HIGHLIGHTS:
            self.highlights.sort(key=lambda h: -h[0])
            self.highlights = self.highlights[:MAX_HIGHLIGHTS]

    def reset_generation(self):
        self.history.clear()
        # Keep highlights across gens so [P] always has something to play

    def best_highlight(self):
        if not self.highlights:
            return None
        return max(self.highlights, key=lambda h: h[0])


# ----------------------------------------------------------------------
# Replay player
# ----------------------------------------------------------------------

def _draw_car(surface, x, y, angle, color, offset):
    fx, fy = math.cos(angle), math.sin(angle)
    lx, ly = -fy, fx
    hw, hh = CAR_W / 2, CAR_H / 2
    corners = [
        (x + fx*hh + lx*hw, y + fy*hh + ly*hw),
        (x + fx*hh - lx*hw, y + fy*hh - ly*hw),
        (x - fx*hh - lx*hw, y - fy*hh - ly*hw),
        (x - fx*hh + lx*hw, y - fy*hh + ly*hw),
    ]
    pts = [(int(c[0] - offset[0]), int(c[1] - offset[1])) for c in corners]
    pygame.draw.polygon(surface, color, pts)


def play_replay(highlight, screen, track, font_sm, font_lg, screen_w, screen_h):
    """
    Block until replay completes or user aborts.
    Renders the captured frames in ~0.3x slow-motion with camera on the victim.
    """
    if highlight is None:
        return
    impact, victim_id, frames, gen, lap = highlight
    if not frames:
        return

    # Where was the victim each frame? (for camera lock)
    victim_pos = []
    last = (0.0, 0.0)
    for f in frames:
        for ent in f:
            if ent[0] == victim_id:
                last = (ent[1], ent[2])
                break
        victim_pos.append(last)

    SLOWMO_FPS = 18   # ~0.3x of 60 Hz
    clock = pygame.time.Clock()

    for i, frame in enumerate(frames):
        # Pump events; abort on ESC or P
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); import sys; sys.exit()
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_p):
                return

        tx, ty = victim_pos[i]
        offset = (int(tx - screen_w / 2), int(ty - screen_h / 2))

        # Background + track
        screen.fill((18, 18, 22))
        track.draw(screen, offset)

        # Cars
        for ent in frame:
            cid, x, y, ang, alive, vx, vy, col, _laps = ent
            if not alive and cid != victim_id:
                continue
            if cid == victim_id:
                draw_col = (255, 70, 70) if alive else (255, 210, 60)
            else:
                draw_col = tuple(int(c * 0.6) for c in col)
            _draw_car(screen, x, y, ang, draw_col, offset)

        # Highlight circle around victim
        vsx = int(tx - offset[0])
        vsy = int(ty - offset[1])
        ring_r = 22 + int(6 * math.sin(i * 0.4))   # pulse
        pygame.draw.circle(screen, (255, 80, 80), (vsx, vsy), ring_r, 2)

        # HUD overlay
        title = font_lg.render(
            f"CRASH REPLAY   ·   Gen {gen}   ·   {impact:.0f} px/s impact   ·   lap {lap}",
            True, (255, 120, 80))
        screen.blit(title, (screen_w // 2 - title.get_width() // 2, 18))

        progress = (i + 1) / len(frames)
        bar_w = 480
        bar_x = screen_w // 2 - bar_w // 2
        pygame.draw.rect(screen, (60, 60, 60),    (bar_x, 56, bar_w, 4))
        pygame.draw.rect(screen, (255, 100, 100), (bar_x, 56, int(bar_w * progress), 4))

        hint = font_sm.render("[ESC] or [P] to skip   ·   playing at 0.3x", True, (150, 150, 150))
        screen.blit(hint, (screen_w // 2 - hint.get_width() // 2, screen_h - 28))

        pygame.display.flip()
        clock.tick(SLOWMO_FPS)
