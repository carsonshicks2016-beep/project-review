import numpy as np
import pygame
import random


class TrackGrid:
    """Spatial grid for fast nearest-centerline lookups."""

    def __init__(self, centerline, half_width, cell=200):
        self.cell = cell
        self.half_width = half_width
        self.hw2 = half_width ** 2
        self.grid = {}
        self.centerline = centerline

        for i, (x, y) in enumerate(centerline):
            gx = int(x // cell)
            gy = int(y // cell)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    key = (gx + dx, gy + dy)
                    self.grid.setdefault(key, []).append(i)

    def is_on_track(self, x, y):
        key = (int(x // self.cell), int(y // self.cell))
        candidates = self.grid.get(key)
        if not candidates:
            return False
        for i in candidates:
            cx, cy = self.centerline[i]
            if (x - cx) ** 2 + (y - cy) ** 2 < self.hw2:
                return True
        return False


WORLD_CX = 5000   # world-space centre X
WORLD_CY = 4000   # world-space centre Y


class Track:
    WIDTH = 175   # road width in world-pixels

    def __init__(self, seed=None, difficulty=1.0):
        """
        difficulty: 0.0 (gentle oval) → 1.0 (full chaos with hairpins).
        Used by curriculum learning to start populations on easy tracks.
        """
        self.seed = seed if seed is not None else random.randint(0, 99999)
        self.difficulty = max(0.0, min(1.0, float(difficulty)))
        self._rng = random.Random(self.seed)
        self.centerline = []
        self.left_wall = []
        self.right_wall = []
        self.checkpoints = []
        self.obstacles = []
        self.width = self.WIDTH
        self._grid = None
        self.bbox = (0, 0, 1, 1)   # (xmin, ymin, xmax, ymax) filled after gen
        self._generate()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self):
        """
        Generate a non-self-intersecting loop.

        Key fix: use EVENLY-SPACED base angles with only a small bounded jitter
        (≤ 30 % of the sector width).  This guarantees the control polygon is
        star-shaped relative to the centre, so the Catmull-Rom spline cannot
        produce a figure-8 or knot.  We also verify with a fast crossing-check
        and retry up to 15 times if needed.
        """
        cx, cy = WORLD_CX, WORLD_CY

        # Large world — radii of 900-2200 px give a perimeter of ~6-14 km
        # (at 1 px ≈ 0.5 m that's 3-7 miles per lap).
        # Curriculum-tunable bounds.
        # At d=0:    4-5 turns,  ~2% jitter, near-equal radii    → gentle oval
        # At d=0.5:  ~9 turns, ~25% jitter, radii 950-1850         → real corners
        # At d=1.0:  12-18 turns, ~50% jitter, radii 350-2300      → chaotic, hairpins
        d        = self.difficulty
        n_min    = int(4 + d * 8)         # 4 → 12
        n_max    = int(5 + d * 13)        # 5 → 18
        jitter_k = 0.02 + d * 0.48        # 0.02 → 0.50  (much wilder at high d)
        r_lo     = 1400 - d * 1050        # 1400 → 350   (genuine hairpins at max)
        r_hi     = 1500 + d * 800         # 1500 → 2300

        for _attempt in range(30):
            n = self._rng.randint(n_min, n_max)
            sector = 2 * np.pi / n
            max_jitter = sector * jitter_k

            pts = []
            for i in range(n):
                base   = sector * i
                jitter = self._rng.uniform(-max_jitter, max_jitter)
                r      = self._rng.uniform(r_lo, r_hi)
                a      = base + jitter
                pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))

            candidate = self._catmull_rom(pts, 900)   # more samples for larger world

            if not self._has_crossing(candidate):
                self.centerline = candidate
                break
        else:
            # Fallback: large clean oval
            pts = [(cx + 1600 * np.cos(2*np.pi*i/10),
                    cy + 1100 * np.sin(2*np.pi*i/10)) for i in range(10)]
            self.centerline = self._catmull_rom(pts, 900)

        self._start_idx = self._find_straight_start()

        self._build_walls()
        self._build_checkpoints()
        self._build_obstacles()
        self._grid = TrackGrid(self.centerline, self.width / 2, cell=200)

        # Bounding box over the centerline (used by minimap)
        xs = [p[0] for p in self.centerline]
        ys = [p[1] for p in self.centerline]
        self.bbox = (min(xs), min(ys), max(xs), max(ys))

    def _has_crossing(self, cl):
        """
        Cheap crossing detector: sample every ~8 pts and check that no sampled
        point is unreasonably close to a far-away sampled point (would indicate
        the spline doubled back on itself / overlapped).

        Threshold tightens as difficulty drops — high-difficulty tracks are
        allowed to weave more tightly so we can get real hairpins through.
        """
        step  = 8
        pts   = cl[::step]
        n     = len(pts)
        min_sep = max(4, n // 5)
        # Loosen the safety margin at high difficulty so hairpins are allowed.
        margin = 1.10 - 0.20 * self.difficulty   # 1.10 → 0.90 of track width
        threshold = (self.width * margin) ** 2

        for i in range(n):
            xi, yi = pts[i]
            for j in range(i + min_sep, n - min_sep + 1):
                xj, yj = pts[j]
                if (xi - xj) ** 2 + (yi - yj) ** 2 < threshold:
                    return True
        return False

    def _find_straight_start(self):
        """
        Return the index along the centerline with the lowest curvature
        (most straight) so the start line isn't on a hairpin.
        """
        cl = self.centerline
        n  = len(cl)
        look = 12   # points ahead/behind to measure curvature

        best_idx = 0
        best_straight = -1.0

        for i in range(0, n, 4):
            prev = cl[(i - look) % n]
            cur  = cl[i]
            nxt  = cl[(i + look) % n]

            # Dot product of normalised forward vectors — 1.0 = perfectly straight
            d1x, d1y = cur[0]-prev[0], cur[1]-prev[1]
            d2x, d2y = nxt[0]-cur[0],  nxt[1]-cur[1]
            l1 = (d1x**2 + d1y**2) ** 0.5 + 1e-9
            l2 = (d2x**2 + d2y**2) ** 0.5 + 1e-9
            dot = (d1x/l1)*(d2x/l2) + (d1y/l1)*(d2y/l2)

            if dot > best_straight:
                best_straight = dot
                best_idx = i

        return best_idx

    def _catmull_rom(self, pts, total_samples):
        n = len(pts)
        result = []
        per = total_samples // n
        for i in range(n):
            p0 = pts[(i - 1) % n]
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            p3 = pts[(i + 2) % n]
            for j in range(per):
                t = j / per
                t2, t3 = t * t, t * t * t
                x = 0.5 * ((2 * p1[0]) + (-p0[0] + p2[0]) * t
                            + (2*p0[0] - 5*p1[0] + 4*p2[0] - p3[0]) * t2
                            + (-p0[0] + 3*p1[0] - 3*p2[0] + p3[0]) * t3)
                y = 0.5 * ((2 * p1[1]) + (-p0[1] + p2[1]) * t
                            + (2*p0[1] - 5*p1[1] + 4*p2[1] - p3[1]) * t2
                            + (-p0[1] + 3*p1[1] - 3*p2[1] + p3[1]) * t3)
                result.append((x, y))
        return result

    def _build_walls(self):
        n = len(self.centerline)
        hw = self.width / 2
        for i in range(n):
            px, py = self.centerline[i]
            nx_pt, ny_pt = self.centerline[(i + 1) % n]
            dx, dy = nx_pt - px, ny_pt - py
            length = (dx*dx + dy*dy) ** 0.5 + 1e-9
            nx, ny = -dy / length, dx / length
            self.left_wall.append((px + nx * hw, py + ny * hw))
            self.right_wall.append((px - nx * hw, py - ny * hw))

    def _build_checkpoints(self):
        n    = len(self.centerline)
        step = max(1, n // 18)
        hw   = self.width / 2 + 6

        # Build a full list of evenly-spaced indices starting at _start_idx
        # so checkpoint 0 is always the start/finish line.
        start = getattr(self, '_start_idx', 0)
        indices = [(start + i * step) % n for i in range(n // step)]

        for idx in indices:
            px, py   = self.centerline[idx]
            nx_pt, ny_pt = self.centerline[(idx + 1) % n]
            dx, dy   = nx_pt - px, ny_pt - py
            length   = (dx*dx + dy*dy) ** 0.5 + 1e-9
            nx, ny   = -dy / length, dx / length
            self.checkpoints.append((
                (px + nx * hw, py + ny * hw),
                (px - nx * hw, py - ny * hw),
                idx,
            ))

    def _build_obstacles(self):
        pass  # obstacles removed — car-to-car collisions are the hazard now

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_start_pos(self):
        """Return (x, y, heading) at the straightest point on the track."""
        idx  = self._start_idx
        n    = len(self.centerline)
        px, py = self.centerline[idx]
        # Heading = average direction over next several points (stable on straights)
        look = 10
        ax, ay = self.centerline[(idx + look) % n]
        angle = np.arctan2(ay - py, ax - px)
        return px, py, angle

    def is_on_track(self, x, y):
        return self._grid.is_on_track(x, y)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def draw(self, surface, offset=(0, 0)):
        ox, oy = offset
        n = len(self.centerline)
        if n < 3:
            return

        def s(x, y):
            return (int(x - ox), int(y - oy))

        def shift(pts):
            return [s(x, y) for x, y in pts]

        # --- Filled track surface ---
        lw = shift(self.left_wall)
        rw = shift(list(reversed(self.right_wall)))
        if len(lw) > 2:
            pygame.draw.polygon(surface, (45, 45, 45), lw + rw)

        # --- Walls ---
        if len(lw) > 1:
            pygame.draw.lines(surface, (210, 210, 210), True, lw, 2)
        rw2 = shift(self.right_wall)
        if len(rw2) > 1:
            pygame.draw.lines(surface, (210, 210, 210), True, rw2, 2)

        # --- Dashed centreline ---
        dash_on = True
        for i in range(0, n - 4, 6):
            dash_on = not dash_on
            if not dash_on:
                continue
            p1 = s(*self.centerline[i])
            p2 = s(*self.centerline[min(i + 5, n - 1)])
            pygame.draw.line(surface, (100, 90, 40), p1, p2, 1)

        # --- Direction chevrons every ~30 points ---
        for i in range(0, n, 30):
            mid  = self.centerline[i]
            fwd  = self.centerline[(i + 9) % n]
            dx   = fwd[0] - mid[0]
            dy   = fwd[1] - mid[1]
            ln   = (dx*dx + dy*dy) ** 0.5 + 1e-9
            dx, dy   = dx/ln, dy/ln     # forward unit vector
            px, py   = -dy, dx          # perpendicular (left)

            arm  = 7.0   # half-width of chevron
            reach = 8.0  # forward reach of tip

            tip   = s(mid[0] + dx*reach,         mid[1] + dy*reach)
            left  = s(mid[0] - dx*4 + px*arm,    mid[1] - dy*4 + py*arm)
            right = s(mid[0] - dx*4 - px*arm,    mid[1] - dy*4 - py*arm)

            col = (190, 170, 45)
            pygame.draw.line(surface, col, left,  tip, 2)
            pygame.draw.line(surface, col, right, tip, 2)

        # --- Start / finish line (chequered) ---
        if self.checkpoints:
            cp      = self.checkpoints[0]
            p_start = s(*cp[0])
            p_end   = s(*cp[1])
            # Draw thick white line
            pygame.draw.line(surface, (255, 255, 255), p_start, p_end, 4)
            # Draw a small "S" arrow showing the launch direction at start
            si = self._start_idx
            fwd_pt = self.centerline[(si + 14) % n]
            bk_pt  = self.centerline[si]
            ddx    = fwd_pt[0] - bk_pt[0]
            ddy    = fwd_pt[1] - bk_pt[1]
            ll     = (ddx**2 + ddy**2) ** 0.5 + 1e-9
            ddx, ddy = ddx/ll, ddy/ll
            mid_x  = (cp[0][0] + cp[1][0]) / 2
            mid_y  = (cp[0][1] + cp[1][1]) / 2
            arrow_tip  = s(mid_x + ddx*22, mid_y + ddy*22)
            arrow_base = s(mid_x - ddx*8,  mid_y - ddy*8)
            ppx, ppy   = -ddy, ddx
            arrow_l = s(mid_x + ddx*8 + ppx*8, mid_y + ddy*8 + ppy*8)
            arrow_r = s(mid_x + ddx*8 - ppx*8, mid_y + ddy*8 - ppy*8)
            pygame.draw.line(surface, (50, 255, 50), arrow_base, arrow_tip, 3)
            pygame.draw.line(surface, (50, 255, 50), arrow_l, arrow_tip, 3)
            pygame.draw.line(surface, (50, 255, 50), arrow_r, arrow_tip, 3)

        # --- Other checkpoints (faint blue) ---
        for cp in self.checkpoints[1:]:
            pygame.draw.line(surface, (60, 60, 170),
                             s(*cp[0]), s(*cp[1]), 1)


    # ------------------------------------------------------------------
    # Mini-map
    # ------------------------------------------------------------------

    def draw_minimap(self, surface, rect, cars, cam_offset, screen_w, screen_h):
        """
        Draw an overview mini-map inside `rect` (x, y, w, h).
        Shows the full track, all cars, and the current camera viewport.
        """
        mx, my, mw, mh = rect

        # Background panel
        bg = pygame.Surface((mw, mh), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 180))
        surface.blit(bg, (mx, my))

        xmin, ymin, xmax, ymax = self.bbox
        bw = max(xmax - xmin, 1)
        bh = max(ymax - ymin, 1)

        scale  = min(mw / bw, mh / bh) * 0.88
        pad_x  = (mw - bw * scale) / 2
        pad_y  = (mh - bh * scale) / 2

        def to_map(wx, wy):
            return (
                int(mx + pad_x + (wx - xmin) * scale),
                int(my + pad_y + (wy - ymin) * scale),
            )

        # Track fill (sampled walls)
        step = max(1, len(self.left_wall) // 300)
        lpts = [to_map(*p) for p in self.left_wall[::step]]
        rpts = [to_map(*p) for p in reversed(self.right_wall[::step])]
        if len(lpts) > 2:
            pygame.draw.polygon(surface, (70, 70, 70), lpts + rpts)
        if len(lpts) > 1:
            pygame.draw.lines(surface, (170, 170, 170), True, lpts, 1)
            pygame.draw.lines(surface, (170, 170, 170), True,
                              [to_map(*p) for p in self.right_wall[::step]], 1)

        # Start/finish marker
        if self.checkpoints:
            cp = self.checkpoints[0]
            pygame.draw.line(surface, (255, 255, 255),
                             to_map(*cp[0]), to_map(*cp[1]), 2)


        # Cars — faint grey for dead, colour-coded for alive
        for car in cars:
            pt = to_map(car.x, car.y)
            col = car.color if car.alive else (55, 55, 55)
            pygame.draw.circle(surface, col, pt, 2)

        # Best alive — white dot, slightly larger
        alive = [c for c in cars if c.alive]
        if alive:
            best = max(alive, key=lambda c: c.fitness)
            bpt  = to_map(best.x, best.y)
            pygame.draw.circle(surface, (255, 255, 255), bpt, 4)
            pygame.draw.circle(surface, (0, 0, 0), bpt, 4, 1)

        # Viewport rectangle
        vx1, vy1 = to_map(cam_offset[0], cam_offset[1])
        vx2, vy2 = to_map(cam_offset[0] + screen_w, cam_offset[1] + screen_h)
        rct = pygame.Rect(vx1, vy1, max(vx2 - vx1, 4), max(vy2 - vy1, 4))
        pygame.draw.rect(surface, (255, 220, 50), rct, 1)

        # Border
        pygame.draw.rect(surface, (120, 120, 120), (mx, my, mw, mh), 1)
