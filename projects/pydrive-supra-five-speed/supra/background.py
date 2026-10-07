import math
import numpy as np
import pygame

from . import props

class TrackEnvironmentBackground:
    def __init__(self, trk, seed: int = 7):
        self.rng = np.random.default_rng(seed)
        self.trk = trk
        self.stars = []
        self.light_surface = None

        # State: day, dusk, night
        self.time_of_day = "day"
        self.modes = {
            "day": {
                "terrain_color": (38, 52, 30),
                "ambient": 1.0,
                "shadow_offset": (0.18, 0.25),
                "shadow_color": (16, 22, 14),
                "stars": False,
                "streetlights": False,
            },
            "dusk": {
                "terrain_color": (30, 24, 18),
                "ambient": 0.65,
                "shadow_offset": (0.65, 0.85),
                "shadow_color": (12, 10, 8),
                "stars": True,
                "stars_brightness": 100,
                "streetlights": True,
                "light_alpha": 18,
            },
            "night": {
                "terrain_color": (10, 14, 8),
                "ambient": 0.25,
                "shadow_offset": (0.0, 0.0), # No global shadows
                "shadow_color": None,
                "stars": True,
                "stars_brightness": 220,
                "streetlights": True,
                "light_alpha": 42,
            }
        }

        # 1. Generate ambient sky stars
        for _ in range(40):
            rx = self.rng.uniform(0.0, 1.0)
            ry = self.rng.uniform(0.0, 0.6)
            size = self.rng.choice([1, 2], p=[0.9, 0.1])
            brightness = self.rng.integers(60, 150)
            self.stars.append((rx, ry, size, brightness))

        # 2. Extract track layout geometry
        c = trk.center
        normal = trk.normal
        tangent = trk.tangent
        half = trk.half
        M = len(c)
        self.long_track = M > 1800
        self.active_window = 18 if self.long_track else 30
        self.max_active_objects = 70 if self.long_track else 120

        self.track_center = c
        self.segment_objects = [[] for _ in range(M)]
        self.segment_terrain = [[] for _ in range(M)]

        # Pre-plan trackside buildings so we can carve a tree-clearing in front of
        # each one (otherwise they vanish into the dense forest band).
        building_segments = {}      # seg_idx -> (side, kind)
        tree_block = set()          # (seg_idx, side) where trees are suppressed
        building_stride = 150 if self.long_track else 26
        for i in range(M):
            if i % building_stride == building_stride // 2:
                side = int(self.rng.choice([1, -1]))
                kind = "garage" if self.rng.random() < 0.5 else "grandstand"
                building_segments[i] = (side, kind)
                for j in (i - 1, i, i + 1):
                    tree_block.add((j % M, side))

        # 3. Dynamic track-relative object placement
        light_stride = 48 if self.long_track else 8
        tree_stride = 5 if self.long_track else 2
        guard_stride = 10 if self.long_track else 5
        for i in range(M):
            # --- Streetlights ---
            if i % light_stride == 0:
                for sgn in [1, -1]:
                    offset = 2.0
                    pos_lamp = c[i] + normal[i] * (half + offset) * sgn
                    self.segment_objects[i].append({
                        "type": "streetlight",
                        "x": pos_lamp[0],
                        "y": pos_lamp[1],
                        "nx": normal[i][0],
                        "ny": normal[i][1],
                        "side": sgn,
                        "height": 8.0,
                    })

            # --- Forest Trees (Pine Trees only) ---
            # Trees dominate the frame's polygon count, so place a bit more
            # sparsely (1-2 per side) — still reads as dense forest because they
            # overlap heavily, but it roughly halves the per-frame tree faces.
            if i % tree_stride == 0:
                for sgn in [1, -1]:
                    if (i, sgn) in tree_block:      # leave a clearing for buildings
                        continue
                    num_trees = 1 if self.long_track else self.rng.integers(1, 3)
                    for _ in range(num_trees):
                        offset = self.rng.uniform(8.0, 38.0)
                        pos_tree = c[i] + normal[i] * (half + offset) * sgn

                        height = self.rng.uniform(10.0, 18.0)
                        width = self.rng.uniform(4.0, 7.5)

                        green = self.rng.choice([
                            (34, 52, 28),   # Dark Pine
                            (28, 46, 24),   # Spruce Green
                            (42, 60, 36),   # Olive Green
                            (48, 68, 42),   # Foliage Green
                        ])

                        self.segment_objects[i].append({
                            "type": "tree",
                            "x": pos_tree[0],
                            "y": pos_tree[1],
                            "height": height,
                            "width": width,
                            "color": green,
                            "tree_key": "tree_pine",
                        })

            # --- Trackside props (carart-style faux-3D furniture) ---
            tx, ty = float(tangent[i][0]), float(tangent[i][1])
            nx, ny = float(normal[i][0]), float(normal[i][1])

            # Armco guardrail along both shoulders, run along the tangent.
            if i % guard_stride == 0:
                for sgn in (1, -1):
                    edge = c[i] + normal[i] * (half + 1.3) * sgn
                    self.segment_objects[i].append({
                        "type": "guardrail", "x": float(edge[0]), "y": float(edge[1]),
                        "hx": tx, "hy": ty, "length": 7.0,
                    })

            # One piece of roadside furniture per segment (sparse), facing the road.
            roll = self.rng.random() * (3.0 if self.long_track else 1.0)
            sgn = int(self.rng.choice([1, -1]))
            face_x, face_y = -nx * sgn, -ny * sgn   # heading: prop -> track centre
            if roll < 0.05:
                for k in range(3):                  # a little run of cones
                    base = c[i] + normal[i] * (half + 1.0) * sgn + tangent[i] * (k - 1) * 1.0
                    self.segment_objects[i].append({
                        "type": "traffic_cone", "x": float(base[0]), "y": float(base[1]),
                        "hx": face_x, "hy": face_y, "color": (236, 110, 28),
                    })
            elif roll < 0.08:
                pos = c[i] + normal[i] * (half + 2.2) * sgn
                self.segment_objects[i].append({
                    "type": "tire_stack", "x": float(pos[0]), "y": float(pos[1]),
                    "hx": face_x, "hy": face_y, "count": int(self.rng.integers(3, 6)),
                })
            elif roll < 0.11:
                pos = c[i] + normal[i] * (half + 2.6) * sgn
                col = tuple(int(v) for v in self.rng.choice([(208, 44, 44), (40, 90, 200), (235, 150, 30)]))
                self.segment_objects[i].append({
                    "type": "vending_machine", "x": float(pos[0]), "y": float(pos[1]),
                    "hx": face_x, "hy": face_y, "color": col,
                })
            elif roll < 0.15:
                pos = c[i] + normal[i] * (half + 2.2) * sgn
                col = tuple(int(v) for v in self.rng.choice([(40, 96, 176), (225, 200, 40), (200, 40, 40)]))
                self.segment_objects[i].append({
                    "type": "road_sign", "x": float(pos[0]), "y": float(pos[1]),
                    "hx": face_x, "hy": face_y, "color": col,
                })
            elif roll < 0.17:
                pos = c[i] + normal[i] * (half + 6.5) * sgn
                col = tuple(int(v) for v in self.rng.choice([(220, 196, 64), (210, 70, 70), (70, 140, 200)]))
                self.segment_objects[i].append({
                    "type": "billboard", "x": float(pos[0]), "y": float(pos[1]),
                    "hx": face_x, "hy": face_y, "color": col, "accent": (32, 36, 48),
                })

            # Buildings: trackside, in a forest clearing (see pre-plan above).
            if i in building_segments:
                bsgn, kind = building_segments[i]
                bnx, bny = float(normal[i][0]), float(normal[i][1])
                if kind == "garage":
                    pos = c[i] + normal[i] * (half + 7.5) * bsgn
                    self.segment_objects[i].append({
                        "type": "garage", "x": float(pos[0]), "y": float(pos[1]),
                        "hx": -bnx * bsgn, "hy": -bny * bsgn,   # door faces the road
                        "width": 8.0, "depth": 6.5,
                        "color": tuple(int(v) for v in self.rng.choice(
                            [(176, 172, 162), (150, 156, 150), (186, 170, 150)])),
                    })
                else:
                    pos = c[i] + normal[i] * (half + 3.5) * bsgn
                    self.segment_objects[i].append({
                        "type": "grandstand", "x": float(pos[0]), "y": float(pos[1]),
                        "hx": bnx * bsgn, "hy": bny * bsgn,     # rakes up away from track
                        "width": 14.0, "seed": int(i),
                    })

        # 4. Generate terrain grass/dirt patches
        terrain_count = 35 if self.long_track else 70
        for _ in range(terrain_count):
            rand_idx = self.rng.integers(0, M)
            pos_base = c[rand_idx]
            offset_dist = self.rng.uniform(20.0, 200.0)
            offset_angle = self.rng.uniform(0.0, 2 * np.pi)
            px = pos_base[0] + np.cos(offset_angle) * offset_dist
            py = pos_base[1] + np.sin(offset_angle) * offset_dist

            radius = self.rng.uniform(20.0, 70.0)
            col = self.rng.choice([
                (20, 27, 16),
                (30, 26, 18),
                (18, 24, 15),
            ])
            self.segment_terrain[rand_idx].append({
                "x": px,
                "y": py,
                "radius": radius,
                "color": col,
            })

    def cycle_time_of_day(self):
        """Cycles time of day: Day -> Dusk -> Night -> Day."""
        if self.time_of_day == "day":
            self.time_of_day = "dusk"
        elif self.time_of_day == "dusk":
            self.time_of_day = "night"
        else:
            self.time_of_day = "day"

    def _get_light_surface(self, scale: float, alpha: int) -> pygame.Surface:
        """Creates a cached soft alpha-blended radial light cone for streetlights."""
        rad = int(scale * 15.0)
        if rad < 2:
            rad = 2
        w = rad * 2

        if self.light_surface is None or self.light_surface.get_width() != w:
            surf = pygame.Surface((w, w), pygame.SRCALPHA)
            for r in range(rad, 0, -2):
                curr_alpha = int(alpha * (1.0 - r / rad))
                pygame.draw.circle(surf, (255, 235, 170, curr_alpha), (rad, rad), r)
            self.light_surface = surf

        return self.light_surface

    def _get_active_segments(self, cam) -> list:
        """Track segments to populate this frame. A follow-cam only needs a window
        around the car; a zoomed-out whole-track view (the live training screens)
        sets `cam.view_all` to draw the entire course."""
        M = len(self.track_center)
        if getattr(cam, "view_all", False):
            if self.long_track:
                step = max(1, M // 450)
                return list(range(0, M, step))
            return list(range(M))

        cx, cy = cam.cx, cam.cy
        if hasattr(self.trk, "nearest"):
            close_idx = int(self.trk.nearest(cx, cy))
        else:
            diff = self.track_center - [cx, cy]
            dists_sq = diff[:, 0]**2 + diff[:, 1]**2
            close_idx = int(np.argmin(dists_sq))

        window = self.active_window
        return [(close_idx + offset) % M for offset in range(-window, window + 1)]

    def draw_terrain(self, screen: pygame.Surface, cam):
        """Phase 1: Draw the terrain floor, stars, and grass patches."""
        w, h = screen.get_size()
        scale = cam.scale
        time_sec = pygame.time.get_ticks() / 1000.0

        mode = self.modes[self.time_of_day]
        ambient = mode["ambient"]

        # 1. Fill ground background
        screen.fill(mode["terrain_color"])

        # 2. Draw static stars (if active)
        if mode["stars"]:
            star_brightness = mode["stars_brightness"]
            for rx, ry, size, brightness in self.stars:
                cycle = np.sin(time_sec * 2.0 + rx * 10) * 0.25 + 0.75
                b_val = int(star_brightness * (brightness / 150.0) * cycle)
                b_val = max(0, min(255, b_val))
                col = (b_val, b_val, int(b_val * 0.8))
                sx, sy = int(rx * w), int(ry * h)
                pygame.draw.circle(screen, col, (sx, sy), size)

        # 3. Project helper
        H_cam = 100.0
        cx, cy = cam.cx, cam.cy
        def project_pt(wx: float, wy: float, wz: float):
            dist = H_cam - wz
            f = H_cam / dist
            apx = cx + (wx - cx) * f
            apy = cy + (wy - cy) * f
            sx, sy = cam.to_screen(apx, apy)
            return (int(sx), int(sy))

        # 4. Get active segments
        visible_indices = self._get_active_segments(cam)

        # 5. Draw active terrain detail patches (grass/mud)
        for idx in visible_indices:
            for p in self.segment_terrain[idx]:
                sx, sy = project_pt(p["x"], p["y"], 0.0)
                r_px = int(p["radius"] * scale)
                if r_px > 1:
                    c_org = p["color"]
                    c_shaded = (int(c_org[0] * ambient), int(c_org[1] * ambient), int(c_org[2] * ambient))
                    pygame.draw.circle(screen, c_shaded, (sx, sy), r_px)

    def draw_objects(self, screen: pygame.Surface, cam, veh):
        """Phase 2: Draw streetlights' tarmac glow, projected 3D shadows, and 3D geometries."""
        scale = cam.scale

        mode = self.modes[self.time_of_day]
        ambient = mode["ambient"]
        sh_color = mode["shadow_color"]
        dx, dy = mode["shadow_offset"]

        H_cam = 100.0
        cx, cy = cam.cx, cam.cy

        def project_pt(wx: float, wy: float, wz: float):
            dist = H_cam - wz
            f = H_cam / dist
            apx = cx + (wx - cx) * f
            apy = cy + (wy - cy) * f
            sx, sy = cam.to_screen(apx, apy)
            return (int(sx), int(sy))

        # 1. Fetch active objects, frustum-culled to the viewport. The ±30-segment
        #    window runs well past the screen edges (especially on straights), so
        #    skipping off-screen objects here avoids projecting / sorting / filling
        #    faces nobody can see — the main framerate win for the trackside detail.
        W_scr, H_scr = screen.get_size()
        # Height projects points radially AWAY from screen centre, so an object
        # whose base is off-screen has its top even further off — it can never
        # intrude. A small safety margin is all we need (tighter = fewer faces).
        MARGIN = 60
        visible_indices = self._get_active_segments(cam)
        active_objects = []
        for idx in visible_indices:
            for obj in self.segment_objects[idx]:
                sx, sy = project_pt(obj["x"], obj["y"], 0.0)
                if -MARGIN <= sx <= W_scr + MARGIN and -MARGIN <= sy <= H_scr + MARGIN:
                    active_objects.append(obj)
        if len(active_objects) > self.max_active_objects:
            active_objects.sort(key=lambda obj: (obj["x"] - cx) ** 2 + (obj["y"] - cy) ** 2)
            active_objects = active_objects[:self.max_active_objects]

        # 2. Draw Streetlight Tarmac Glow Cones (Dusk and Night only)
        if mode["streetlights"]:
            alpha = mode["light_alpha"]
            glow_surf = self._get_light_surface(scale, alpha)
            rad_w = glow_surf.get_width() // 2

            for obj in active_objects:
                if obj["type"] == "streetlight":
                    sx, sy, nx, ny, side = obj["x"], obj["y"], obj["nx"], obj["ny"], obj["side"]
                    light_x = sx - nx * 2.0 * side
                    light_y = sy - ny * 2.0 * side
                    sl_x, sl_y = project_pt(light_x, light_y, 0.0)
                    screen.blit(glow_surf, (sl_x - rad_w, sl_y - rad_w))

        # 3. Draw Projected 3D Shadows (Day and Dusk only)
        if sh_color is not None:
            for obj in active_objects:
                obj_type = obj["type"]
                ox, oy = obj["x"], obj["y"]

                if obj_type == "tree":
                    th, tw = obj["height"], obj["width"]
                    sh_base = project_pt(ox, oy, 0.0)
                    sh_trunk_top = project_pt(ox - (0.25 * th) * dx, oy - (0.25 * th) * dy, 0.0)
                    pygame.draw.line(screen, sh_color, sh_base, sh_trunk_top, max(1, int(scale * 0.22)))
                    for z_b, z_a, w_f in [(0.25, 0.6, 1.0), (0.5, 0.8, 0.75), (0.75, 1.0, 0.5)]:
                        w_t = tw * w_f
                        sb0 = project_pt(ox - w_t/2 - z_b * th * dx, oy - w_t/2 - z_b * th * dy, 0.0)
                        sb1 = project_pt(ox + w_t/2 - z_b * th * dx, oy - w_t/2 - z_b * th * dy, 0.0)
                        sb2 = project_pt(ox + w_t/2 - z_b * th * dx, oy + w_t/2 - z_b * th * dy, 0.0)
                        sb3 = project_pt(ox - w_t/2 - z_b * th * dx, oy + w_t/2 - z_b * th * dy, 0.0)
                        sa = project_pt(ox - z_a * th * dx, oy - z_a * th * dy, 0.0)
                        pygame.draw.polygon(screen, sh_color, [sb0, sb1, sa])
                        pygame.draw.polygon(screen, sh_color, [sb1, sb2, sa])
                        pygame.draw.polygon(screen, sh_color, [sb2, sb3, sa])
                        pygame.draw.polygon(screen, sh_color, [sb3, sb0, sa])

                elif obj_type == "streetlight":
                    sh = obj["height"]
                    s_base = project_pt(ox, oy, 0.0)
                    sh_lamp = project_pt(ox - sh * dx, oy - sh * dy, 0.0)
                    pygame.draw.line(screen, sh_color, s_base, sh_lamp, max(1, int(scale * 0.08)))

                elif obj_type in props.SHADOW_TYPES:
                    # Simple skewed ground blob under big props.
                    h_est = {"vending_machine": 1.9, "billboard": 4.5, "garage": 4.4,
                             "grandstand": 3.0, "tire_stack": 1.5}.get(obj_type, 2.0)
                    bx, by = project_pt(ox - h_est * dx * 0.5, oy - h_est * dy * 0.5, 0.0)
                    rw = int(scale * (3.5 if obj_type in ("garage", "grandstand", "billboard") else 1.2))
                    rh = max(2, int(rw * 0.5))
                    if rw > 1:
                        blob = pygame.Surface((rw * 2, rh * 2), pygame.SRCALPHA)
                        pygame.draw.ellipse(blob, (*sh_color, 130), blob.get_rect())
                        screen.blit(blob, (bx - rw, by - rh))

        # 4. Project and sort active 3D geometries
        faces_to_render = []
        light_dir = (0.2, -0.35, 0.9) if self.time_of_day != "night" else (0.0, 0.0, 1.0)

        for obj in active_objects:
            obj_type = obj["type"]
            ox, oy = obj["x"], obj["y"]

            if obj_type in props.PROP_TYPES:
                faces_to_render.extend(
                    props.emit_faces(obj, project_pt, cx, cy, H_cam, light_dir, ambient))
                continue

            if obj_type == "tree":
                th, tw, t_col = obj["height"], obj["width"], obj["color"]

                # 3D Pine (3-tier pyramids)
                s_base = project_pt(ox, oy, 0.0)
                s_trunk_top = project_pt(ox, oy, 0.25 * th)
                dist_trunk = math.sqrt((ox - cx)**2 + (oy - cy)**2 + (0.125 * th - H_cam)**2)
                faces_to_render.append({
                    "dist": dist_trunk,
                    "poly_pts": [s_base, s_trunk_top],
                    "color": (80, 52, 30),
                    "f_name": "tree_trunk",
                    "width": max(1, int(scale * 0.25))
                })

                for z_b, z_a, w_f in [(0.25, 0.6, 1.0), (0.5, 0.8, 0.75), (0.75, 1.0, 0.5)]:
                    w_t = tw * w_f
                    b0 = project_pt(ox - w_t/2, oy - w_t/2, z_b * th)
                    b1 = project_pt(ox + w_t/2, oy - w_t/2, z_b * th)
                    b2 = project_pt(ox + w_t/2, oy + w_t/2, z_b * th)
                    b3 = project_pt(ox - w_t/2, oy + w_t/2, z_b * th)
                    s_apex = project_pt(ox, oy, z_a * th)

                    faces_spec = [
                        ("tree_north", [b0, b1, s_apex], (0.0, -1.0, 0.1), t_col),
                        ("tree_east", [b1, b2, s_apex], (1.0, 0.0, 0.1), t_col),
                        ("tree_south", [b2, b3, s_apex], (0.0, 1.0, 0.1), t_col),
                        ("tree_west", [b3, b0, s_apex], (-1.0, 0.0, 0.1), t_col),
                    ]
                    for f_name, poly_pts, normal, col in faces_spec:
                        fx_c, fy_c, fz_c = ox, oy, (z_b + z_a)/2 * th
                        dist = math.sqrt((fx_c - cx)**2 + (fy_c - cy)**2 + (fz_c - H_cam)**2)
                        dot = normal[0]*light_dir[0] + normal[1]*light_dir[1] + normal[2]*light_dir[2]
                        shade = (0.6 + 0.4 * max(0.0, dot)) * ambient
                        shaded_col = (int(col[0] * shade), int(col[1] * shade), int(col[2] * shade))
                        faces_to_render.append({
                            "dist": dist,
                            "poly_pts": poly_pts,
                            "color": shaded_col,
                            "f_name": f_name,
                            "outline_color": (int(shaded_col[0] * 0.75), int(shaded_col[1] * 0.75), int(shaded_col[2] * 0.75)),
                        })

            elif obj_type == "streetlight":
                nx, ny, side, sh = obj["nx"], obj["ny"], obj["side"], obj["height"]

                s_base = project_pt(ox, oy, 0.0)
                s_lamp = project_pt(ox, oy, sh)

                arm_wx = ox - nx * 1.5 * side
                arm_wy = oy - ny * 1.5 * side
                s_lamp_arm = project_pt(arm_wx, arm_wy, sh)

                dist = math.sqrt((ox - cx)**2 + (oy - cy)**2 + (sh/2 - H_cam)**2)
                p_col = (int(110 * ambient), int(112 * ambient), int(115 * ambient))

                faces_to_render.append({
                    "dist": dist,
                    "poly_pts": [s_base, s_lamp, s_lamp_arm],
                    "color": p_col,
                    "f_name": "streetlight",
                    "trim_color": (255, 250, 210) if mode["streetlights"] else (120, 120, 100),
                })

        # 5. Sort faces globally: furthest away first
        faces_to_render.sort(key=lambda x: x["dist"], reverse=True)

        # 6. Render Sorted Faces
        for f in faces_to_render:
            f_name = f["f_name"]

            if f_name == "streetlight":
                poly_pts = f["poly_pts"]
                col = f["color"]
                pygame.draw.line(screen, col, poly_pts[0], poly_pts[1], max(1, int(scale * 0.08)))
                pygame.draw.line(screen, col, poly_pts[1], poly_pts[2], max(1, int(scale * 0.06)))
                pygame.draw.circle(screen, f["trim_color"], poly_pts[2], max(2, int(scale * 0.12)))
            elif f_name == "tree_trunk":
                poly_pts = f["poly_pts"]
                col = f["color"]
                width = f["width"]
                pygame.draw.line(screen, col, poly_pts[0], poly_pts[1], width)
            else:
                poly_pts = f["poly_pts"]
                pygame.draw.polygon(screen, f["color"], poly_pts)
                # The dark edge outline is a SECOND polygon draw per face; only
                # worth it when the face is big enough to read it. Skipping it on
                # tiny/distant polygons roughly halves draw calls in dense forest.
                xs = [p[0] for p in poly_pts]
                ys = [p[1] for p in poly_pts]
                if (max(xs) - min(xs)) >= 7 or (max(ys) - min(ys)) >= 7:
                    pygame.draw.polygon(screen, f["outline_color"], poly_pts, max(1, int(scale * 0.02)))
