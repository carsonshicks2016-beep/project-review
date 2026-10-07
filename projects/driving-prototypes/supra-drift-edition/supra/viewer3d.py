"""
3D OpenGL rendering engine for Supra-AI using moderngl + pygame.

Renders track geometry, car models, particle effects (smoke/flames),
sky gradient and a HUD overlay.  All shaders target GLSL 330 core
for macOS Core Profile compatibility.

World convention: X,Y = simulation plane, Z = up/elevation.
"""

from __future__ import annotations
import math
import struct

import numpy as np

try:
    import moderngl
except ImportError:
    moderngl = None

try:
    import pygame
except ImportError:
    pygame = None

from .config import Config


# ──────────────────────────────────────────────────────────────────────
# Shader sources
# ──────────────────────────────────────────────────────────────────────

_TRACK_VERT = """
#version 330 core
uniform mat4 u_mvp;
in vec3 in_pos;
in vec3 in_color;
in vec3 in_normal;
out vec3 v_color;
out vec3 v_normal;
out vec3 v_fragpos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_color = in_color;
    v_normal = in_normal;
    v_fragpos = in_pos;
}
"""

_TRACK_FRAG = """
#version 330 core
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_ambient;
in vec3 v_color;
in vec3 v_normal;
in vec3 v_fragpos;
out vec4 fragColor;
void main() {
    vec3 n = normalize(v_normal);
    float diff = max(dot(n, normalize(u_sun_dir)), 0.0);
    vec3 lit = v_color * (u_ambient + u_sun_color * diff);
    fragColor = vec4(lit, 1.0);
}
"""

_CAR_VERT = """
#version 330 core
uniform mat4 u_mvp;
uniform mat4 u_model;
in vec3 in_pos;
in vec3 in_normal;
out vec3 v_normal;
out vec3 v_fragpos;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_normal = mat3(u_model) * in_normal;
    v_fragpos = (u_model * vec4(in_pos, 1.0)).xyz;
}
"""

_CAR_FRAG = """
#version 330 core
uniform vec3 u_base_color;
uniform vec3 u_sun_dir;
uniform vec3 u_sun_color;
uniform vec3 u_ambient;
uniform vec3 u_eye;
in vec3 v_normal;
in vec3 v_fragpos;
out vec4 fragColor;
void main() {
    vec3 n = normalize(v_normal);
    vec3 L = normalize(u_sun_dir);
    float diff = max(dot(n, L), 0.0);
    vec3 V = normalize(u_eye - v_fragpos);
    vec3 H = normalize(L + V);
    float spec = pow(max(dot(n, H), 0.0), 64.0);
    vec3 color = u_base_color * (u_ambient + u_sun_color * diff) + vec3(0.4) * spec;
    fragColor = vec4(color, 1.0);
}
"""

_PARTICLE_VERT = """
#version 330 core
uniform mat4 u_vp;
uniform vec3 u_cam_right;
uniform vec3 u_cam_up;
in vec3 in_center;
in vec2 in_offset;
in float in_size;
in float in_life;
in vec4 in_color;
out float v_life;
out vec4 v_color;
out vec2 v_uv;
void main() {
    vec3 pos = in_center
             + u_cam_right * in_offset.x * in_size
             + u_cam_up    * in_offset.y * in_size;
    gl_Position = u_vp * vec4(pos, 1.0);
    v_life = in_life;
    v_color = in_color;
    v_uv = in_offset;
}
"""

_PARTICLE_FRAG = """
#version 330 core
in float v_life;
in vec4 v_color;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    float d = length(v_uv);
    if (d > 1.0) discard;
    float falloff = 1.0 - d;
    float alpha = v_color.a * v_life * falloff;
    fragColor = vec4(v_color.rgb, alpha);
}
"""

_SKY_VERT = """
#version 330 core
in vec2 in_pos;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.999, 1.0);
    v_uv = in_pos * 0.5 + 0.5;
}
"""

_SKY_FRAG = """
#version 330 core
uniform vec3 u_top_color;
uniform vec3 u_bot_color;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec3 col = mix(u_bot_color, u_top_color, v_uv.y);
    fragColor = vec4(col, 1.0);
}
"""


# ──────────────────────────────────────────────────────────────────────
# Matrix helpers (pure numpy, column-major for GL)
# ──────────────────────────────────────────────────────────────────────

def _perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) * 0.5)
    nf = 1.0 / (near - far)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) * nf, 2 * far * near * nf],
        [0, 0, -1, 0],
    ], dtype=np.float32)


def _look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    f = target - eye
    f = f / (np.linalg.norm(f) + 1e-9)
    s = np.cross(f, up)
    s = s / (np.linalg.norm(s) + 1e-9)
    u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[0, 3] = -np.dot(s, eye)
    M[1, 3] = -np.dot(u, eye)
    M[2, 3] = np.dot(f, eye)
    return M


def _translate(tx, ty, tz):
    M = np.eye(4, dtype=np.float32)
    M[0, 3] = tx
    M[1, 3] = ty
    M[2, 3] = tz
    return M


def _rotate_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = c; M[0, 1] = -s
    M[1, 0] = s; M[1, 1] = c
    return M


def _rotate_x(angle):
    c, s = math.cos(angle), math.sin(angle)
    M = np.eye(4, dtype=np.float32)
    M[1, 1] = c; M[1, 2] = -s
    M[2, 1] = s; M[2, 2] = c
    return M


def _rotate_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = c; M[0, 2] = s
    M[2, 0] = -s; M[2, 2] = c
    return M


def _scale(sx, sy, sz):
    M = np.eye(4, dtype=np.float32)
    M[0, 0] = sx; M[1, 1] = sy; M[2, 2] = sz
    return M


# ──────────────────────────────────────────────────────────────────────
# Renderer
# ──────────────────────────────────────────────────────────────────────

class Renderer3D:
    """Full 3D rendering engine for the Supra-AI driving sim.

    Requires a pygame OPENGL display to already be set up; creates a
    moderngl context from the existing window.
    """

    def __init__(self, cfg: Config, width: int, height: int):
        if moderngl is None:
            raise ImportError(
                "moderngl is required for 3D rendering.  Install it with:\n"
                "  pip install moderngl"
            )
        self.cfg = cfg
        self.width = width
        self.height = height

        # Create context from existing pygame GL surface
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        # ---- compile shaders ----
        self.prog_track = self.ctx.program(
            vertex_shader=_TRACK_VERT, fragment_shader=_TRACK_FRAG)
        self.prog_car = self.ctx.program(
            vertex_shader=_CAR_VERT, fragment_shader=_CAR_FRAG)
        self.prog_particle = self.ctx.program(
            vertex_shader=_PARTICLE_VERT, fragment_shader=_PARTICLE_FRAG)
        self.prog_sky = self.ctx.program(
            vertex_shader=_SKY_VERT, fragment_shader=_SKY_FRAG)

        # ---- projection ----
        aspect = width / max(1, height)
        self.proj = _perspective(60.0, aspect, 0.5, 5000.0)
        self.view = np.eye(4, dtype=np.float32)

        # ---- sky fullscreen quad ----
        sky_verts = np.array([
            -1, -1,  1, -1,  -1, 1,
             1, -1,  1,  1,  -1, 1,
        ], dtype=np.float32)
        sky_vbo = self.ctx.buffer(sky_verts.tobytes())
        self.sky_vao = self.ctx.vertex_array(
            self.prog_sky, [(sky_vbo, '2f', 'in_pos')])

        # ---- sun / lighting ----
        self.sun_dir = np.array([0.4, 0.3, 1.0], dtype=np.float32)
        self.sun_dir /= np.linalg.norm(self.sun_dir)
        self.sun_color = np.array([1.0, 0.95, 0.85], dtype=np.float32)
        self.ambient = np.array([0.25, 0.27, 0.32], dtype=np.float32)

        # ---- camera state (spring-damper chase cam) ----
        self.cam_pos = np.array([0.0, 0.0, 10.0], dtype=np.float32)
        self.cam_target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._cam_yaw = 0.0
        self._current_track = None

        # ---- track mesh ----
        self.track_vao = None
        self.track_n_verts = 0
        self.ground_vao = None
        self.ground_n_verts = 0

        # ---- car mesh ----
        self._car_vao = None
        self._car_n_verts = 0
        self._car_detail_vao = None
        self._car_detail_n_verts = 0
        self.build_car_mesh()

        # ---- particle buffers ----
        self._max_particles = 1200
        self._particle_vbo = self.ctx.buffer(reserve=self._max_particles * 6 * 48)
        # billboard quad offsets
        self._quad_offsets = np.array([
            [-1, -1], [1, -1], [-1, 1],
            [1, -1], [1, 1], [-1, 1],
        ], dtype=np.float32)

        # ---- skid mark buffers ----
        self._skid_vao = None
        self._skid_n_verts = 0

    # ================================================================== #
    #  Track mesh
    # ================================================================== #

    def set_track(self, track):
        """Build triangle mesh from left/right/elevation arrays."""
        self._current_track = track
        N = track.N
        left = track.left    # (N, 2)
        right = track.right  # (N, 2)
        elev = getattr(track, 'elevation', None)
        if elev is None:
            elev = np.zeros(N, dtype=np.float64)

        # ---- main track surface ----
        # Triangle strip: for each segment i→i+1, two triangles from
        # left[i], right[i], left[i+1], right[i+1]
        positions = []
        colors = []
        normals = []

        # Asphalt base colors
        c_asphalt1 = np.array([0.22, 0.23, 0.25], dtype=np.float32)
        c_asphalt2 = np.array([0.20, 0.21, 0.23], dtype=np.float32)
        c_kerb_r   = np.array([0.82, 0.27, 0.25], dtype=np.float32)
        c_kerb_w   = np.array([0.92, 0.93, 0.96], dtype=np.float32)
        c_edge     = np.array([0.86, 0.88, 0.90], dtype=np.float32)
        c_grass    = np.array([0.13, 0.22, 0.12], dtype=np.float32)
        c_grass2   = np.array([0.11, 0.19, 0.10], dtype=np.float32)

        wrap = track.is_loop

        def _idx(i):
            if wrap:
                return i % N
            return min(i, N - 1)

        n_segs = N if wrap else N - 1

        for i in range(n_segs):
            j = _idx(i + 1)
            # Triangle 1: left[i], right[i], left[j]
            # Triangle 2: right[i], right[j], left[j]
            l_i = np.array([left[i, 0], left[i, 1], elev[i]], dtype=np.float32)
            r_i = np.array([right[i, 0], right[i, 1], elev[i]], dtype=np.float32)
            l_j = np.array([left[j, 0], left[j, 1], elev[j]], dtype=np.float32)
            r_j = np.array([right[j, 0], right[j, 1], elev[j]], dtype=np.float32)

            # Face normal (approximately up for flat track)
            e1 = r_i - l_i
            e2 = l_j - l_i
            n = np.cross(e1, e2)
            nl = np.linalg.norm(n)
            if nl > 1e-9:
                n /= nl
            else:
                n = np.array([0, 0, 1], dtype=np.float32)

            # Alternating stripe color
            col = c_asphalt1 if (i // 6) % 2 == 0 else c_asphalt2

            positions.extend([l_i, r_i, l_j, r_i, r_j, l_j])
            colors.extend([col] * 6)
            normals.extend([n] * 6)

            # ---- Kerbs on corners ----
            if hasattr(track, 'difficulty') and track.difficulty[i] > 0.28:
                kc = c_kerb_r if (i // 3) % 2 == 0 else c_kerb_w
                kerb_w = 1.1
                # Left kerb (outside left edge)
                norm_i = track.normal[i]
                norm_j = track.normal[j]
                kl_i = np.array([left[i, 0] - norm_i[0] * kerb_w,
                                 left[i, 1] - norm_i[1] * kerb_w, elev[i]], dtype=np.float32)
                kl_j = np.array([left[j, 0] - norm_j[0] * kerb_w,
                                 left[j, 1] - norm_j[1] * kerb_w, elev[j]], dtype=np.float32)
                positions.extend([l_i, kl_i, l_j, kl_i, kl_j, l_j])
                colors.extend([kc] * 6)
                normals.extend([n] * 6)
                # Right kerb
                kr_i = np.array([right[i, 0] + norm_i[0] * kerb_w,
                                 right[i, 1] + norm_i[1] * kerb_w, elev[i]], dtype=np.float32)
                kr_j = np.array([right[j, 0] + norm_j[0] * kerb_w,
                                 right[j, 1] + norm_j[1] * kerb_w, elev[j]], dtype=np.float32)
                positions.extend([r_i, r_j, kr_i, r_j, kr_j, kr_i])
                colors.extend([kc] * 6)
                normals.extend([n] * 6)

        # ---- Edge lines (thin strips along left/right edges) ----
        edge_w = 0.3
        for i in range(n_segs):
            j = _idx(i + 1)
            for side, arr in [('left', left), ('right', right)]:
                norm_i = track.normal[i]
                norm_j = track.normal[j]
                sign = 1.0 if side == 'left' else -1.0
                a = np.array([arr[i, 0], arr[i, 1], elev[i] + 0.01], dtype=np.float32)
                b = np.array([arr[j, 0], arr[j, 1], elev[j] + 0.01], dtype=np.float32)
                oa = np.array([arr[i, 0] - sign * norm_i[0] * edge_w,
                               arr[i, 1] - sign * norm_i[1] * edge_w,
                               elev[i] + 0.01], dtype=np.float32)
                ob = np.array([arr[j, 0] - sign * norm_j[0] * edge_w,
                               arr[j, 1] - sign * norm_j[1] * edge_w,
                               elev[j] + 0.01], dtype=np.float32)
                up = np.array([0, 0, 1], dtype=np.float32)
                positions.extend([a, oa, b, oa, ob, b])
                colors.extend([c_edge] * 6)
                normals.extend([up] * 6)

        # ---- Mountainous terrain (slopes from road edges) ----
        ground_pos = []
        ground_col = []
        ground_nrm = []

        # Terrain strip definitions: (distance_start, distance_end, drop, color)
        terrain_strips = [
            (0.0, 15.0, 8.0,   np.array([0.16, 0.24, 0.13], dtype=np.float32)),
            (15.0, 40.0, 25.0, np.array([0.12, 0.18, 0.10], dtype=np.float32)),
            (40.0, 100.0, 60.0, np.array([0.09, 0.14, 0.08], dtype=np.float32)),
        ]

        # Skip every 2nd segment for performance
        step = 2
        for i in range(0, n_segs, step):
            j = _idx(i + step)
            norm_i = track.normal[i]
            norm_j = track.normal[j]
            el_i = float(elev[i])
            el_j = float(elev[j])

            for d_start, d_end, drop, t_col in terrain_strips:
                # Alternate color slightly
                t_col2 = t_col * 0.88
                col_use = t_col if (i // 4) % 2 == 0 else t_col2

                # LEFT side (extend in -normal direction)
                for side_sign, side_arr in [(-1.0, left), (1.0, right)]:
                    inner_i = np.array([
                        side_arr[i, 0] + side_sign * norm_i[0] * d_start,
                        side_arr[i, 1] + side_sign * norm_i[1] * d_start,
                        el_i - d_start * 0.15
                    ], dtype=np.float32)
                    inner_j = np.array([
                        side_arr[_idx(i + step), 0] + side_sign * norm_j[0] * d_start,
                        side_arr[_idx(i + step), 1] + side_sign * norm_j[1] * d_start,
                        el_j - d_start * 0.15
                    ], dtype=np.float32)
                    outer_i = np.array([
                        side_arr[i, 0] + side_sign * norm_i[0] * d_end,
                        side_arr[i, 1] + side_sign * norm_i[1] * d_end,
                        el_i - drop
                    ], dtype=np.float32)
                    outer_j = np.array([
                        side_arr[_idx(i + step), 0] + side_sign * norm_j[0] * d_end,
                        side_arr[_idx(i + step), 1] + side_sign * norm_j[1] * d_end,
                        el_j - drop
                    ], dtype=np.float32)

                    # Face normal
                    te1 = outer_i - inner_i
                    te2 = inner_j - inner_i
                    tn = np.cross(te1, te2)
                    tnl = np.linalg.norm(tn)
                    if tnl > 1e-9:
                        tn = tn / tnl
                    else:
                        tn = np.array([0, 0, 1], dtype=np.float32)
                    # Flip normal to face outward if needed
                    if tn[2] < 0:
                        tn = -tn

                    ground_pos.extend([inner_i, outer_i, inner_j, outer_i, outer_j, inner_j])
                    ground_col.extend([col_use] * 6)
                    ground_nrm.extend([tn] * 6)

        # ---- Valley floor (large flat plane at lowest elevation) ----
        all_pts = np.concatenate([left, right], axis=0)
        min_x, min_y = all_pts.min(axis=0) - 500
        max_x, max_y = all_pts.max(axis=0) + 500
        ground_z = float(elev.min()) - 65.0
        c_valley = np.array([0.06, 0.10, 0.06], dtype=np.float32)
        c_valley2 = np.array([0.05, 0.09, 0.05], dtype=np.float32)
        up_vec = np.array([0, 0, 1], dtype=np.float32)

        tile_size = 80.0
        nx_tiles = min(int((max_x - min_x) / tile_size) + 1, 50)
        ny_tiles = min(int((max_y - min_y) / tile_size) + 1, 50)

        for ix in range(nx_tiles):
            for iy in range(ny_tiles):
                x0 = min_x + ix * tile_size
                y0 = min_y + iy * tile_size
                x1 = x0 + tile_size
                y1 = y0 + tile_size
                gc = c_valley if (ix + iy) % 2 == 0 else c_valley2
                p0 = np.array([x0, y0, ground_z], dtype=np.float32)
                p1 = np.array([x1, y0, ground_z], dtype=np.float32)
                p2 = np.array([x0, y1, ground_z], dtype=np.float32)
                p3 = np.array([x1, y1, ground_z], dtype=np.float32)
                ground_pos.extend([p0, p1, p2, p1, p3, p2])
                ground_col.extend([gc] * 6)
                ground_nrm.extend([up_vec] * 6)

        # Build ground VAO
        if ground_pos:
            g_pos_arr = np.array(ground_pos, dtype=np.float32).reshape(-1, 3)
            g_col_arr = np.array(ground_col, dtype=np.float32).reshape(-1, 3)
            g_nrm_arr = np.array(ground_nrm, dtype=np.float32).reshape(-1, 3)
            g_data = np.hstack([g_pos_arr, g_col_arr, g_nrm_arr]).astype(np.float32)
            g_vbo = self.ctx.buffer(g_data.tobytes())
            self.ground_vao = self.ctx.vertex_array(
                self.prog_track,
                [(g_vbo, '3f 3f 3f', 'in_pos', 'in_color', 'in_normal')])
            self.ground_n_verts = len(ground_pos)

        # Build track VAO
        if positions:
            pos_arr = np.array(positions, dtype=np.float32).reshape(-1, 3)
            col_arr = np.array(colors, dtype=np.float32).reshape(-1, 3)
            nrm_arr = np.array(normals, dtype=np.float32).reshape(-1, 3)
            data = np.hstack([pos_arr, col_arr, nrm_arr]).astype(np.float32)
            vbo = self.ctx.buffer(data.tobytes())
            self.track_vao = self.ctx.vertex_array(
                self.prog_track,
                [(vbo, '3f 3f 3f', 'in_pos', 'in_color', 'in_normal')])
            self.track_n_verts = len(positions)

    # ================================================================== #
    #  Car mesh — low-poly Supra extruded from 2D silhouette
    # ================================================================== #

    def build_car_mesh(self):
        """Build a low-poly Toyota Supra Mk4 mesh from a 2D side silhouette.

        Heights: ground clearance 0.15m, beltline 0.55m, roof peak 1.15m.
        The mesh is at the origin facing +X, in body coords.
        Also builds detail geometry (wheels, headlights, taillights, glass).
        """
        # Side-profile silhouette (x=forward, z=up), as a series of sections
        # Each section: (x, half_width_bottom, half_width_top, z_bottom, z_top)
        clearance = 0.15
        beltline = 0.55
        roof = 1.15

        # Body cross-sections from front to rear (x position, half-width)
        # Based on the 2D silhouette in app.py
        sections = [
            # x,     hw_bot, hw_top, z_bot,    z_top
            ( 2.30,  0.10,   0.10,   clearance + 0.10, clearance + 0.30),  # nose tip
            ( 2.10,  0.34,   0.28,   clearance,        beltline - 0.05),   # front
            ( 1.55,  0.78,   0.40,   clearance,        beltline + 0.10),   # hood
            ( 0.55,  0.86,   0.45,   clearance,        roof),              # A-pillar
            ( 0.05,  0.86,   0.54,   clearance,        roof),              # windshield top
            (-0.65,  0.86,   0.52,   clearance,        roof - 0.05),       # roof peak
            (-1.12,  0.84,   0.36,   clearance,        roof - 0.15),       # C-pillar
            (-1.55,  0.80,   0.30,   clearance,        beltline + 0.15),   # rear deck
            (-1.95,  0.60,   0.25,   clearance,        beltline + 0.08),   # tail
            (-2.12,  0.34,   0.15,   clearance + 0.05, beltline),          # tail tip
        ]

        positions = []
        norms = []

        def _add_tri(p0, p1, p2):
            e1 = np.array(p1) - np.array(p0)
            e2 = np.array(p2) - np.array(p0)
            n = np.cross(e1, e2)
            nl = np.linalg.norm(n)
            if nl > 1e-9:
                n = n / nl
            else:
                n = np.array([0, 0, 1], dtype=np.float32)
            for p in (p0, p1, p2):
                positions.append(np.array(p, dtype=np.float32))
                norms.append(n.astype(np.float32))

        def _add_quad(p0, p1, p2, p3):
            _add_tri(p0, p1, p2)
            _add_tri(p2, p3, p0)

        for k in range(len(sections) - 1):
            x0, hw0_b, hw0_t, z0_b, z0_t = sections[k]
            x1, hw1_b, hw1_t, z1_b, z1_t = sections[k + 1]

            # 8 corners of this segment
            # Bottom quad: (x, ±hw_b, z_b)
            # Top quad:    (x, ±hw_t, z_t)
            bl0 = (x0, -hw0_b, z0_b)  # bottom-left-front
            br0 = (x0,  hw0_b, z0_b)  # bottom-right-front
            tl0 = (x0, -hw0_t, z0_t)  # top-left-front
            tr0 = (x0,  hw0_t, z0_t)  # top-right-front

            bl1 = (x1, -hw1_b, z1_b)
            br1 = (x1,  hw1_b, z1_b)
            tl1 = (x1, -hw1_t, z1_t)
            tr1 = (x1,  hw1_t, z1_t)

            # Left side (y negative)
            _add_quad(bl0, bl1, tl1, tl0)
            # Right side (y positive)
            _add_quad(br1, br0, tr0, tr1)
            # Top
            _add_quad(tl0, tl1, tr1, tr0)
            # Bottom
            _add_quad(bl1, bl0, br0, br1)

        # Front face
        s = sections[0]
        x, hw_b, hw_t, z_b, z_t = s
        _add_quad((x, -hw_b, z_b), (x, hw_b, z_b),
                  (x, hw_t, z_t), (x, -hw_t, z_t))

        # Rear face
        s = sections[-1]
        x, hw_b, hw_t, z_b, z_t = s
        _add_quad((x, hw_b, z_b), (x, -hw_b, z_b),
                  (x, -hw_t, z_t), (x, hw_t, z_t))

        # ---- Rear wing ----
        wing_top = beltline + 0.40
        # Wing top
        _add_quad(
            (-1.95, -0.94, wing_top), (-1.95, 0.94, wing_top),
            (-2.34, 0.94, wing_top), (-2.34, -0.94, wing_top))
        # Wing bottom
        _add_quad(
            (-2.34, -0.94, beltline + 0.30), (-2.34, 0.94, beltline + 0.30),
            (-1.95, 0.94, beltline + 0.30), (-1.95, -0.94, beltline + 0.30))
        # Wing endplate left
        _add_quad(
            (-1.95, -0.94, beltline + 0.30), (-1.95, -0.94, wing_top),
            (-2.34, -0.94, wing_top), (-2.34, -0.94, beltline + 0.30))
        # Wing endplate right
        _add_quad(
            (-2.34, 0.94, beltline + 0.30), (-2.34, 0.94, wing_top),
            (-1.95, 0.94, wing_top), (-1.95, 0.94, beltline + 0.30))
        # Wing stays
        for sy in (0.18, -0.18):
            _add_quad(
                (-1.8, sy - 0.03, beltline), (-1.8, sy + 0.03, beltline),
                (-1.8, sy + 0.03, beltline + 0.30), (-1.8, sy - 0.03, beltline + 0.30))

        pos_arr = np.array(positions, dtype=np.float32).reshape(-1, 3)
        nrm_arr = np.array(norms, dtype=np.float32).reshape(-1, 3)
        data = np.hstack([pos_arr, nrm_arr]).astype(np.float32)

        vbo = self.ctx.buffer(data.tobytes())
        self._car_vao = self.ctx.vertex_array(
            self.prog_car, [(vbo, '3f 3f', 'in_pos', 'in_normal')])
        self._car_n_verts = len(positions)

        # ================================================================ #
        #  Detail geometry: wheels, headlights, taillights, glass
        #  Rendered with the track shader (vertex-colored)
        # ================================================================ #
        detail_pos = []
        detail_col = []
        detail_nrm = []

        def _d_tri(p0, p1, p2, color):
            e1 = np.array(p1) - np.array(p0)
            e2 = np.array(p2) - np.array(p0)
            n = np.cross(e1, e2)
            nl = np.linalg.norm(n)
            n = (n / nl).astype(np.float32) if nl > 1e-9 else np.array([0, 0, 1], dtype=np.float32)
            for p in (p0, p1, p2):
                detail_pos.append(np.array(p, dtype=np.float32))
                detail_col.append(np.array(color, dtype=np.float32))
                detail_nrm.append(n)

        def _d_quad(p0, p1, p2, p3, color):
            _d_tri(p0, p1, p2, color)
            _d_tri(p2, p3, p0, color)

        # ---- Wheels (4 octagonal cylinders) ----
        tire_color = [0.12, 0.12, 0.14]
        rim_color = [0.55, 0.55, 0.60]
        wheel_positions = [
            (1.45, -0.82, 0.33),   # Front-left
            (1.45,  0.82, 0.33),   # Front-right
            (-1.20, -0.82, 0.33),  # Rear-left
            (-1.20,  0.82, 0.33),  # Rear-right
        ]
        w_radius = 0.33
        w_width = 0.22
        n_seg = 10

        for (cx, cy, cz) in wheel_positions:
            for i in range(n_seg):
                a0 = 2 * math.pi * i / n_seg
                a1 = 2 * math.pi * (i + 1) / n_seg
                cos0, sin0 = math.cos(a0) * w_radius, math.sin(a0) * w_radius
                cos1, sin1 = math.cos(a1) * w_radius, math.sin(a1) * w_radius
                y_l = cy - w_width / 2
                y_r = cy + w_width / 2
                # Side quad (tire)
                p0 = [cx, y_l, cz + sin0]  # note: x stays at cx, z has the circle
                p1 = [cx, y_l, cz + sin1]
                p2 = [cx, y_r, cz + sin1]
                p3 = [cx, y_r, cz + sin0]
                # For a wheel oriented with axle along Y, circle is in X-Z plane
                p0 = [cx + cos0, y_l, cz + sin0]
                p1 = [cx + cos1, y_l, cz + sin1]
                p2 = [cx + cos1, y_r, cz + sin1]
                p3 = [cx + cos0, y_r, cz + sin0]
                _d_quad(p0, p1, p2, p3, tire_color)
                # End caps (rim)
                center_l = [cx, y_l, cz]
                center_r = [cx, y_r, cz]
                _d_tri([cx + cos0, y_l, cz + sin0], center_l, [cx + cos1, y_l, cz + sin1], rim_color)
                _d_tri(center_r, [cx + cos0, y_r, cz + sin0], [cx + cos1, y_r, cz + sin1], rim_color)

        # ---- Headlights (front, bright white/yellow) ----
        hl_color = [1.0, 0.95, 0.85]
        hw, hh = 0.15, 0.08
        for hy in [-0.55, 0.55]:
            hx, hz = 2.12, 0.38
            _d_quad(
                (hx, hy - hw, hz - hh), (hx, hy + hw, hz - hh),
                (hx, hy + hw, hz + hh), (hx, hy - hw, hz + hh),
                hl_color)

        # ---- Taillights (rear, bright red) ----
        tl_color = [0.9, 0.1, 0.05]
        tw, th = 0.18, 0.06
        for ty in [-0.55, 0.55]:
            tx, tz = -2.12, 0.38
            _d_quad(
                (tx, ty - tw, tz - th), (tx, ty + tw, tz - th),
                (tx, ty + tw, tz + th), (tx, ty - tw, tz + th),
                tl_color)

        # ---- Glass / windshield (top faces of roof sections in dark tint) ----
        glass_color = [0.12, 0.15, 0.20]
        # Sections 3-6 are the roof sections (A-pillar through C-pillar)
        for k in range(3, min(7, len(sections) - 1)):
            x0, _, hw0_t, _, z0_t = sections[k]
            x1, _, hw1_t, _, z1_t = sections[k + 1]
            # Top face as a glass quad (slightly raised)
            _d_quad(
                (x0, -hw0_t, z0_t + 0.005), (x1, -hw1_t, z1_t + 0.005),
                (x1,  hw1_t, z1_t + 0.005), (x0,  hw0_t, z0_t + 0.005),
                glass_color)

        # Build detail VAO using the track shader (which supports vertex colors)
        if detail_pos:
            d_pos_arr = np.array(detail_pos, dtype=np.float32).reshape(-1, 3)
            d_col_arr = np.array(detail_col, dtype=np.float32).reshape(-1, 3)
            d_nrm_arr = np.array(detail_nrm, dtype=np.float32).reshape(-1, 3)
            d_data = np.hstack([d_pos_arr, d_col_arr, d_nrm_arr]).astype(np.float32)
            d_vbo = self.ctx.buffer(d_data.tobytes())
            self._car_detail_vao = self.ctx.vertex_array(
                self.prog_track,
                [(d_vbo, '3f 3f 3f', 'in_pos', 'in_color', 'in_normal')])
            self._car_detail_n_verts = len(detail_pos)

    # ================================================================== #
    #  Elevation helper
    # ================================================================== #

    def _smooth_elev_at(self, track, car_idx, x, y):
        """Interpolate elevation using car's track index and position."""
        elev_arr = getattr(track, 'elevation', None)
        if elev_arr is None:
            return 0.0
        N = track.N
        idx = int(car_idx) % N
        if track.is_loop:
            idx_next = (idx + 1) % N
        else:
            idx_next = min(idx + 1, N - 1)
        # Fraction along the segment
        seg = track.center[idx_next] - track.center[idx]
        seg_len_sq = float(np.dot(seg, seg)) + 1e-9
        pos = np.array([x, y]) - track.center[idx]
        t = float(np.clip(np.dot(pos, seg) / seg_len_sq, 0.0, 1.0))
        return float(elev_arr[idx] * (1.0 - t) + elev_arr[idx_next] * t)

    # ================================================================== #
    #  Camera
    # ================================================================== #

    def update_camera(self, car, dt):
        """Chase cam behind + above the car with spring-damper follow."""
        if car is None:
            return
        v = car.vehicle
        elev = 0.0
        # Use smooth elevation interpolation
        if hasattr(car, '_idx') and self._current_track is not None:
            elev = self._smooth_elev_at(self._current_track, car._idx, v.x, v.y)

        # Target position: the car
        car_pos = np.array([v.x, v.y, elev + 0.5], dtype=np.float32)

        # Smoothly follow yaw
        target_yaw = v.yaw
        dyaw = target_yaw - self._cam_yaw
        # Normalize angle
        dyaw = math.atan2(math.sin(dyaw), math.cos(dyaw))
        self._cam_yaw += dyaw * min(1.0, 3.0 * dt)

        # Look-ahead: shift target forward based on speed
        speed = v.speed
        look_ahead = min(speed * 0.35, 25.0)
        ahead = np.array([
            math.cos(self._cam_yaw) * look_ahead,
            math.sin(self._cam_yaw) * look_ahead,
            0.0
        ], dtype=np.float32)

        # Camera offset behind and above car
        dist_back = 8.0 + speed * 0.08
        dist_up = 3.5 + speed * 0.02
        cam_offset = np.array([
            -math.cos(self._cam_yaw) * dist_back,
            -math.sin(self._cam_yaw) * dist_back,
            dist_up
        ], dtype=np.float32)

        desired_pos = car_pos + cam_offset
        desired_target = car_pos + ahead

        # Ensure camera never goes below the track surface
        if self._current_track is not None:
            min_z = elev + 2.0
            desired_pos[2] = max(desired_pos[2], min_z)

        # Spring-damper
        k_pos = min(1.0, 4.0 * dt)
        k_tgt = min(1.0, 8.0 * dt)
        self.cam_pos += (desired_pos - self.cam_pos) * k_pos
        self.cam_target += (desired_target - self.cam_target) * k_tgt

        # Build view matrix
        self.view = _look_at(self.cam_pos, self.cam_target,
                             np.array([0, 0, 1], dtype=np.float32))

    # ================================================================== #
    #  Render
    # ================================================================== #

    def render(self, sim, smoke, flames, skid=None, hud_info=None):
        """Main render call: sky → ground → track → cars → particles → skids → HUD."""
        self.ctx.clear(0.05, 0.06, 0.07, 1.0)

        vp = self.proj @ self.view

        # ---- Sky ----
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.prog_sky['u_top_color'].value = (0.10, 0.13, 0.22)
        self.prog_sky['u_bot_color'].value = (0.35, 0.42, 0.50)
        self.sky_vao.render()
        self.ctx.enable(moderngl.DEPTH_TEST)

        # ---- Ground ----
        if self.ground_vao is not None:
            self.prog_track['u_mvp'].write(vp.T.astype(np.float32).tobytes())
            self.prog_track['u_sun_dir'].value = tuple(self.sun_dir)
            self.prog_track['u_sun_color'].value = tuple(self.sun_color)
            self.prog_track['u_ambient'].value = tuple(self.ambient)
            self.ground_vao.render()

        # ---- Track ----
        if self.track_vao is not None:
            self.prog_track['u_mvp'].write(vp.T.astype(np.float32).tobytes())
            self.prog_track['u_sun_dir'].value = tuple(self.sun_dir)
            self.prog_track['u_sun_color'].value = tuple(self.sun_color)
            self.prog_track['u_ambient'].value = tuple(self.ambient)
            self.track_vao.render()

        # ---- Skid marks ----
        if skid:
            self._render_skids(skid, vp)

        # ---- Cars ----
        track = sim.track if sim is not None else None
        if sim is not None:
            for ci, car in enumerate(sim.cars):
                self._render_car(car, vp, track, focus=(ci == sim.focus))

        # ---- Particles (smoke + flames) ----
        self._render_particles(smoke, flames, vp, sim)

    def _render_skids(self, skid, vp):
        """Render skid marks as small dark quads on the track surface."""
        if not skid:
            return
        track = self._current_track
        skid_pos = []
        skid_col = []
        skid_nrm = []
        c_skid = np.array([0.05, 0.05, 0.08], dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)
        sq = 0.10  # half-size of skid quad

        for pt in skid:
            sx, sy = float(pt[0]), float(pt[1])
            # Approximate elevation
            sz = 0.0
            if track is not None:
                elev_arr = getattr(track, 'elevation', None)
                if elev_arr is not None:
                    dx = track.center[:, 0] - sx
                    dy = track.center[:, 1] - sy
                    dists = dx * dx + dy * dy
                    idx = int(np.argmin(dists))
                    sz = float(elev_arr[idx]) + 0.02
            p0 = np.array([sx - sq, sy - sq, sz], dtype=np.float32)
            p1 = np.array([sx + sq, sy - sq, sz], dtype=np.float32)
            p2 = np.array([sx + sq, sy + sq, sz], dtype=np.float32)
            p3 = np.array([sx - sq, sy + sq, sz], dtype=np.float32)
            skid_pos.extend([p0, p1, p2, p2, p3, p0])
            skid_col.extend([c_skid] * 6)
            skid_nrm.extend([up] * 6)

        if not skid_pos:
            return
        s_pos_arr = np.array(skid_pos, dtype=np.float32).reshape(-1, 3)
        s_col_arr = np.array(skid_col, dtype=np.float32).reshape(-1, 3)
        s_nrm_arr = np.array(skid_nrm, dtype=np.float32).reshape(-1, 3)
        s_data = np.hstack([s_pos_arr, s_col_arr, s_nrm_arr]).astype(np.float32)
        s_vbo = self.ctx.buffer(s_data.tobytes())
        s_vao = self.ctx.vertex_array(
            self.prog_track,
            [(s_vbo, '3f 3f 3f', 'in_pos', 'in_color', 'in_normal')])
        self.prog_track['u_mvp'].write(vp.T.astype(np.float32).tobytes())
        self.prog_track['u_sun_dir'].value = tuple(self.sun_dir)
        self.prog_track['u_sun_color'].value = tuple(self.sun_color)
        self.prog_track['u_ambient'].value = tuple(self.ambient)
        s_vao.render()
        s_vao.release()
        s_vbo.release()

    def _render_car(self, car, vp, track, focus=False):
        """Render a single car with Phong lighting + detail geometry."""
        v = car.vehicle

        # Smooth elevation at car position
        elev = 0.0
        if track is not None:
            elev = self._smooth_elev_at(track, car._idx, v.x, v.y)

        # Model matrix: translate to world pos, rotate by yaw, apply roll/pitch
        model = _translate(v.x, v.y, elev)
        model = model @ _rotate_z(v.yaw)
        # Roll around X (body leans in corners)
        if hasattr(v, 'roll'):
            model = model @ _rotate_x(math.radians(v.roll))
        # Pitch around Y (braking/accel)
        if hasattr(v, 'pitch'):
            model = model @ _rotate_y(math.radians(-v.pitch))

        mvp = vp @ model

        # Color
        if focus:
            base_color = (1.0, 0.84, 0.24)  # gold
        elif car.alive:
            c = car.color
            base_color = (c[0] / 255.0, c[1] / 255.0, c[2] / 255.0)
        else:
            c = car.color
            base_color = (c[0] / 765.0, c[1] / 765.0, c[2] / 765.0)

        # Render body
        self.prog_car['u_mvp'].write(mvp.T.astype(np.float32).tobytes())
        self.prog_car['u_model'].write(model.T.astype(np.float32).tobytes())
        self.prog_car['u_base_color'].value = base_color
        self.prog_car['u_sun_dir'].value = tuple(self.sun_dir)
        self.prog_car['u_sun_color'].value = tuple(self.sun_color)
        self.prog_car['u_ambient'].value = tuple(self.ambient)
        self.prog_car['u_eye'].value = tuple(self.cam_pos)
        self._car_vao.render()

        # Render detail geometry (wheels, headlights, taillights, glass)
        if self._car_detail_vao is not None:
            self.prog_track['u_mvp'].write(mvp.T.astype(np.float32).tobytes())
            self.prog_track['u_sun_dir'].value = tuple(self.sun_dir)
            self.prog_track['u_sun_color'].value = tuple(self.sun_color)
            self.prog_track['u_ambient'].value = tuple(self.ambient)
            self._car_detail_vao.render()

    def _render_particles(self, smoke, flames, vp, sim=None):
        """Render smoke and flame particles as camera-facing billboard quads."""
        if not smoke and not flames:
            return

        # Approximate base elevation from focus car (particles are near cars)
        base_elev = 0.0
        if self._current_track is not None and sim is not None:
            fc = sim.focus_car if sim else None
            if fc is not None:
                base_elev = self._smooth_elev_at(
                    self._current_track, fc._idx, fc.vehicle.x, fc.vehicle.y)

        # Camera right/up vectors from view matrix (rows, not columns)
        cam_right = np.array([self.view[0, 0], self.view[0, 1], self.view[0, 2]],
                             dtype=np.float32)
        cam_up = np.array([self.view[1, 0], self.view[1, 1], self.view[1, 2]],
                          dtype=np.float32)

        # Build vertex data for all particles
        # Per vertex: center(3) + offset(2) + size(1) + life(1) + color(4) = 11 floats
        verts = []
        count = 0

        for p in smoke:
            if p['life'] <= 0:
                continue
            age = 1.0 - p['life']
            size = p['size'] * (1.0 + age * 1.6)
            center = [p['x'], p['y'], base_elev + 0.3 + age * 2.0]
            r, g, b = 0.80, 0.81, 0.83
            a = 0.5 * p['life']
            for off in self._quad_offsets:
                verts.extend(center)
                verts.extend(off)
                verts.append(size)
                verts.append(p['life'])
                verts.extend([r, g, b, a])
            count += 6

        for p in flames:
            if p['life'] <= 0:
                continue
            size = p['size']
            center = [p['x'], p['y'], base_elev + 0.4]
            life = p['life']
            r = 1.0
            g = 0.4 + 0.4 * life
            b = 0.08 + 0.5 * life
            a = 0.85 * life
            for off in self._quad_offsets:
                verts.extend(center)
                verts.extend(off)
                verts.append(size)
                verts.append(life)
                verts.extend([r, g, b, a])
            count += 6

        if count == 0:
            return

        # Clamp to buffer capacity
        max_verts = self._max_particles * 6
        if count > max_verts:
            count = max_verts
            verts = verts[:count * 11]

        data = np.array(verts, dtype=np.float32)
        self._particle_vbo.orphan(size=data.nbytes)
        self._particle_vbo.write(data.tobytes())

        vao = self.ctx.vertex_array(
            self.prog_particle,
            [(self._particle_vbo, '3f 2f 1f 1f 4f',
              'in_center', 'in_offset', 'in_size', 'in_life', 'in_color')])

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.prog_particle['u_vp'].write(vp.T.astype(np.float32).tobytes())
        self.prog_particle['u_cam_right'].value = tuple(cam_right)
        self.prog_particle['u_cam_up'].value = tuple(cam_up)
        vao.render(vertices=count)
        self.ctx.enable(moderngl.DEPTH_TEST)
        vao.release()

    # ================================================================== #
    #  Cleanup
    # ================================================================== #

    def release(self):
        """Release all GL resources."""
        if self.ctx:
            self.ctx.release()
