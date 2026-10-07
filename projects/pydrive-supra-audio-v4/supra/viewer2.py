"""2D Viewer V2 — the projected 2.5D ("faux 3D") viewer, built from scratch.

V1 (supra/app.py) pioneered the look: 3D car meshes oblique-projected onto a
top-down world (screen_y -= z * lift), physics-driven body tilt, ground
shadows, day/night moods. V2 keeps that idea and makes it the LAW of the whole
world — one projector for everything:

  * the ROAD is elevation-projected: real DEM crests bulge, dips sink, and
    banking tilts the edges (v1 drew a flat road under 3D cars);
  * a HEADING-UP chase camera: the world rotates around the car, with a
    velocity lead so you see down the road (C toggles classic north-up);
  * the car's shadow stays glued to the ROAD while the body lifts with
    veh.z — jumps finally read as jumps;
  * sun-lit road shading (slopes facing the light brighten), kerbs on tight
    corners, a checkered start gate, trackside props with height;
  * a Fable-styled HUD: amber monospace telemetry, analog tach dial with
    shift lights + upshift flash, lap board, drift bar, AIR pill,
    elevation-tinted minimap.

Same call signature as supra.app.run — a drop-in. Multi-car modes (racers /
opponents) delegate to the classic v1 viewer, which stays untouched and is
always reachable with --classic (or SUPRA_CLASSIC_VIEWER=1).

Keys: arrows/WASD drive · Space handbrake · Shift clutch · T auto box ·
Q/E shift · C camera · V broadcast director · 1-5 direct shots ·
G day/dusk/night · B beams · M mute · R reset · Esc.
"""
from __future__ import annotations

import math
import os
from collections import deque
from types import SimpleNamespace

import numpy as np

from .config import CarSpec, SimSpec, get_car
from .physics import Controls, Vehicle
from . import track as track_mod
from .app import AutoBox
from .lightfx import draw_dual_headlight_beam


LIFT = 0.85          # screen px of height per world metre, at scale 1
SUN = (0.35, -0.42, 0.84)
MPS_TO_MPH = 2.2369362920544


def _display_refresh_target(default=120):
    """Best-effort presentation rate; env override supports unusual displays."""
    override = os.environ.get("SUPRA_RENDER_FPS")
    if override:
        try:
            return int(np.clip(int(override), 30, 240))
        except ValueError:
            pass
    try:                                    # macOS: real active display mode
        import Quartz
        mode = Quartz.CGDisplayCopyDisplayMode(Quartz.CGMainDisplayID())
        hz = float(Quartz.CGDisplayModeGetRefreshRate(mode))
        if hz >= 30.0:
            return int(np.clip(round(hz), 30, 240))
    except Exception:
        pass
    return int(default)


# --------------------------------------------------------------------------- #
# palettes (G cycles)
# --------------------------------------------------------------------------- #
MOODS = [
    dict(name="day",
         ground=(44, 58, 36), ground_dot=(52, 68, 42),
         road=(52, 54, 58), road_alt=(49, 51, 55), edge=(228, 230, 234),
         shoulder=(84, 78, 65), shoulder_alt=(69, 72, 58),
         aggregate=(92, 94, 96), shoulder_speck=(132, 121, 94),
         fog=(126, 139, 108), fog_near=52.0, fog_far=145.0,
         shadow_reach=0.82, sun_strength=1.10,
         line=(210, 205, 120), kerb_a=(196, 60, 52), kerb_b=(226, 222, 210),
         shadow=(16, 20, 13), tint=None, headlights=False,
         hud=(255, 205, 120), hud_dim=(150, 130, 90)),
    dict(name="dusk",
         ground=(40, 32, 30), ground_dot=(50, 38, 34),
         road=(46, 42, 48), road_alt=(43, 39, 45), edge=(216, 190, 170),
         shoulder=(70, 58, 51), shoulder_alt=(60, 55, 48),
         aggregate=(82, 76, 78), shoulder_speck=(118, 93, 72),
         fog=(104, 78, 70), fog_near=44.0, fog_far=120.0,
         shadow_reach=1.48, sun_strength=1.16,
         line=(235, 185, 100), kerb_a=(176, 54, 48), kerb_b=(206, 196, 180),
         shadow=(12, 10, 10), tint=(255, 130, 55, 26), headlights=True,
         hud=(255, 195, 110), hud_dim=(150, 120, 80)),
    dict(name="night",
         ground=(13, 16, 14), ground_dot=(18, 22, 19),
         road=(30, 32, 40), road_alt=(28, 30, 37), edge=(120, 140, 190),
         shoulder=(38, 39, 38), shoulder_alt=(32, 35, 33),
         aggregate=(60, 64, 76), shoulder_speck=(66, 67, 61),
         fog=(29, 36, 48), fog_near=48.0, fog_far=132.0,
         shadow_reach=0.24, sun_strength=0.72,
         line=(110, 130, 190), kerb_a=(120, 44, 44), kerb_b=(150, 148, 150),
         shadow=(6, 7, 8), tint=(25, 35, 80, 46), headlights=True,
         hud=(255, 200, 115), hud_dim=(130, 110, 80)),
]


def _shade(c, f):
    return (max(0, min(255, int(c[0] * f))),
            max(0, min(255, int(c[1] * f))),
            max(0, min(255, int(c[2] * f))))


def _mix_color(a, b, t):
    t = max(0.0, min(1.0, float(t)))
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def _fog_amount(view, x, y, mood):
    """Smooth world-distance haze used consistently by road and props."""
    d = math.hypot(float(x) - view.cx, float(y) - view.cy)
    near = float(mood.get("fog_near", 55.0))
    far = max(near + 1.0, float(mood.get("fog_far", 140.0)))
    t = max(0.0, min(1.0, (d - near) / (far - near)))
    return t * t * (3.0 - 2.0 * t)


def _fog_color(color, mood, amount, strength=1.0):
    return _mix_color(color, mood.get("fog", color), amount * strength)


# --------------------------------------------------------------------------- #
# The one projector everything shares
# --------------------------------------------------------------------------- #
class View:
    """Rotating, zooming, elevation-lifting camera.

    project(x, y, z): world -> screen. The world is rotated so the camera
    heading points UP, scaled, and lifted by (z - camera ground z) so local
    elevation deforms the picture without launching it off screen.
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        self.cx = self.cy = 0.0            # camera world position
        self.z0 = 0.0                      # road height under the camera
        self.rot = 0.0                     # world rotation (heading-up)
        self.scale = 15.0
        self.base_scale = 15.0
        self.mode = "chase"                # chase (heading-up) | north
        self.anchor_x = w * 0.5
        self.anchor_y = h * 0.66           # car sits low -> long view ahead
        self.shake_x = self.shake_y = 0.0
        self.audio_shake_x = self.audio_shake_y = 0.0
        self._shake_tx = self._shake_ty = 0.0
        self._cos, self._sin = 1.0, 0.0
        # director hooks (all optional; None/1.0 = classic behaviour)
        self.cam_override = None           # (x, y): park the camera here
        self.rot_override = None           # fixed world rotation
        self.rot_offset = 0.0              # additive twist on the chase rot
        self.anchor_override = None        # anchor_y fraction of screen height
        self.zoom_mult = 1.0               # >1 closer, <1 wider
        self.lead_mult = 1.0               # scales the velocity lead
        self.cut = False                   # snap next update (broadcast cut)

    def update(self, veh, road_z, dt):
        # lead the camera along the velocity so the view looks down the road
        sp = veh.speed
        if self.cam_override is not None:
            wx, wy = self.cam_override
        else:
            lead = min(18.0, sp * 0.35) * self.lead_mult
            wx = veh.x + math.cos(veh.yaw) * lead
            wy = veh.y + math.sin(veh.yaw) * lead
        k = min(1.0, dt * 3.5)
        self.cx += (wx - self.cx) * k
        self.cy += (wy - self.cy) * k
        self.z0 += (road_z - self.z0) * min(1.0, dt * 3.0)

        # screen rotation phi: chase = heading points UP, north = identity;
        # a sliver of the body roll leaks into the camera (banking feel)
        if self.rot_override is not None:
            target = self.rot_override
            ay = self.h * (self.anchor_override or 0.52)
        elif self.mode == "chase":
            target = -math.pi / 2.0 - veh.yaw + self.rot_offset \
                + float(np.clip(getattr(veh, "roll", 0.0), -0.10, 0.10)) * 0.45
            ay = self.h * (self.anchor_override or 0.66)
        else:
            target = 0.0
            ay = self.h * (self.anchor_override or 0.54)
        if self.cut:                       # hard broadcast cut: no swooping
            self.cx, self.cy = wx, wy
            self.rot = target
            self.anchor_y = ay
            self.shake_x = self.shake_y = 0.0
            self._shake_tx = self._shake_ty = 0.0
            self.scale = self.base_scale * (1.0 - 0.30 * min(1.0, sp / 75.0)) \
                * self.zoom_mult
            self.cut = False
        d = (target - self.rot + math.pi) % (2 * math.pi) - math.pi
        self.rot += d * min(1.0, dt * 2.6)
        self.anchor_y += (ay - self.anchor_y) * min(1.0, dt * 3.0)

        # speed zoom + slide shake
        f = min(1.0, sp / 75.0)
        target_s = self.base_scale * (1.0 - 0.30 * f) * self.zoom_mult
        slide = abs(math.degrees(veh.slip_angle)) if sp > 6 else 0.0
        if slide > 14:
            a = min(1.0, (slide - 14) / 40.0) * 2.2
            k_noise = min(1.0, dt * 8.0)
            self._shake_tx += (np.random.uniform(-a, a) - self._shake_tx) * k_noise
            self._shake_ty += (np.random.uniform(-a, a) - self._shake_ty) * k_noise
        else:
            self._shake_tx *= 0.82
            self._shake_ty *= 0.82
        k_shake = min(1.0, dt * 12.0)
        self.shake_x += (self._shake_tx - self.shake_x) * k_shake
        self.shake_y += (self._shake_ty - self.shake_y) * k_shake
        self.scale += (target_s - self.scale) * min(1.0, dt * 3.0)
        self._cos, self._sin = math.cos(self.rot), math.sin(self.rot)

    def impulse(self, mag):
        """One-frame camera kick (hard landings)."""
        self.shake_x += np.random.uniform(-mag, mag)
        self.shake_y += np.random.uniform(-mag, mag)

    def project(self, x, y, z=None):
        dx, dy = x - self.cx, y - self.cy
        rx = dx * self._cos - dy * self._sin
        ry = dx * self._sin + dy * self._cos
        sx = self.anchor_x + rx * self.scale + self.shake_x + self.audio_shake_x
        sy = self.anchor_y + ry * self.scale + self.shake_y + self.audio_shake_y
        if z is not None:
            sy -= (z - self.z0) * LIFT * self.scale
        return sx, sy

    def to_screen(self, x, y):               # v1-compat (flat) projector
        return self.project(x, y, None)


class RenderVehicle:
    """Viewer-only fixed-step interpolation wrapper.

    The simulator, controller, sensors, lap logic, and audio vehicle list keep
    using the real Vehicle. This proxy retains the last two 120 Hz physics
    poses and draws between them using the accumulator remainder. That removes
    visible 0/1/2-step cadence without changing simulation timing.
    """

    FIELDS = ("x", "y", "yaw", "z", "road_z", "pitch", "roll", "ax", "ay",
              "steer_angle")

    def __init__(self):
        self._veh = None
        self._ready = False

    def __getattr__(self, name):
        veh = object.__getattribute__(self, "_veh")
        if veh is None:
            raise AttributeError(name)
        return getattr(veh, name)

    def reset(self, veh):
        self._veh = veh
        self._ready = True
        snap = self._snapshot(veh)
        self._prev = dict(snap)
        self._curr = dict(snap)
        self._apply(snap)

    @classmethod
    def _snapshot(cls, veh):
        return {name: float(getattr(veh, name, 0.0)) for name in cls.FIELDS}

    def _apply(self, snap):
        for name, value in snap.items():
            setattr(self, name, value)

    def capture(self, veh):
        """Commit one completed physics step to the presentation history."""
        if self._veh is not veh or not self._ready:
            self.reset(veh)
            return
        snap = self._snapshot(veh)
        dx = snap["x"] - self._curr["x"]
        dy = snap["y"] - self._curr["y"]
        if dx * dx + dy * dy > 64.0:
            self.reset(veh)
            return
        self._prev = self._curr
        self._curr = snap

    def interpolate(self, veh, alpha):
        if self._veh is not veh or not self._ready:
            self.reset(veh)
            return
        a = float(np.clip(alpha, 0.0, 1.0))
        pose = {}
        for name in self.FIELDS:
            old, new = self._prev[name], self._curr[name]
            if name == "yaw":
                delta = (new - old + math.pi) % (2.0 * math.pi) - math.pi
                pose[name] = old + delta * a
            else:
                pose[name] = old + (new - old) * a
        self._apply(pose)


# --------------------------------------------------------------------------- #
# Track geometry, prepared once
# --------------------------------------------------------------------------- #
class RoadGeom:
    def __init__(self, trk, seed=7):
        n = len(trk.center)
        self.n = n
        self.trk = trk
        c = np.asarray(trk.center, float)
        nrm = np.asarray(trk.normal, float)
        half = np.asarray(getattr(trk, "half_width", None)
                          if getattr(trk, "half_width", None) is not None
                          else np.full(n, trk.half), float)
        z = np.asarray(getattr(trk, "z", None)
                       if getattr(trk, "z", None) is not None
                       else np.zeros(n), float)
        bank = np.asarray(getattr(trk, "bank", None)
                          if getattr(trk, "bank", None) is not None
                          else np.zeros(n), float)
        grade = np.asarray(getattr(trk, "grade", None)
                           if getattr(trk, "grade", None) is not None
                           else np.zeros(n), float)
        tang = np.asarray(trk.tangent, float)
        self.center, self.z = c, z
        self.normal, self.half = nrm, half
        self.bank = bank
        self.left = c + nrm * half[:, None]
        self.right = c - nrm * half[:, None]
        self.lz = z + np.sin(bank) * half          # banking tilts the edges
        self.rz = z - np.sin(bank) * half
        self.spacing = float(trk.length / n)
        signed_curv = np.asarray(trk.curvature, float)
        self.signed_curvature = signed_curv
        curv = np.abs(signed_curv)
        k = np.maximum(curv, np.maximum(np.roll(curv, 3), np.roll(curv, -3)))
        self.curvature = k
        self.kerb = k > 0.016

        # Render-only surface geometry. Shoulders sit just below the asphalt,
        # while the rubber lane leans gently toward corner apexes. None of this
        # is fed back into collision, grip, sensors, or vehicle state.
        surface_rng = np.random.default_rng((seed or 7) + 1907)
        shoulder_w = surface_rng.uniform(0.72, 1.18, n)
        shoulder_w = (shoulder_w + np.roll(shoulder_w, 1)
                      + np.roll(shoulder_w, -1)) / 3.0
        self.left_outer = self.left + nrm * shoulder_w[:, None]
        self.right_outer = self.right - nrm * shoulder_w[:, None]
        self.loz = self.lz - 0.035
        self.roz = self.rz - 0.035
        self.shoulder_tone = surface_rng.uniform(0.88, 1.10, n)

        curv_smooth = sum(np.roll(signed_curv, q) for q in range(-4, 5)) / 9.0
        apex_offset = np.clip(curv_smooth * 46.0, -0.24, 0.24) * half
        rubber_half = np.clip(0.95 + half * 0.035, 1.05, 1.38)
        self.rubber_left = c + nrm * (apex_offset + rubber_half)[:, None]
        self.rubber_right = c + nrm * (apex_offset - rubber_half)[:, None]
        bank_sin = np.sin(bank)
        self.rubber_lz = z + bank_sin * (apex_offset + rubber_half) + 0.012
        self.rubber_rz = z + bank_sin * (apex_offset - rubber_half) + 0.012
        self.rubber_tone = surface_rng.uniform(0.78, 0.88, n)

        # Sparse patched sections and tiny aggregate/gravel points are prepared
        # once so the texture is stable in world space instead of swimming under
        # the camera.
        self.patch_mask = surface_rng.random(n) < 0.055
        self.patch_a = surface_rng.uniform(0.08, 0.56, n)
        self.patch_b = np.minimum(0.94, self.patch_a + surface_rng.uniform(0.18, 0.38, n))
        self.patch_tone = surface_rng.choice((0.82, 0.88, 1.07), size=n,
                                             p=(0.38, 0.46, 0.16))
        self.surface_bits = [[] for _ in range(n)]
        bit_step = max(1, int(3.2 / max(self.spacing, 0.05)))
        for i in range(0, n, bit_step):
            for _ in range(2):
                lat = surface_rng.uniform(-0.78, 0.78) * half[i]
                pos = c[i] + nrm[i] * lat + tang[i] * surface_rng.uniform(-0.35, 0.35)
                pz = z[i] + bank_sin[i] * lat + 0.018
                self.surface_bits[i].append((float(pos[0]), float(pos[1]), float(pz),
                                             "road", surface_rng.uniform(0.72, 1.12)))
            side = 1.0 if surface_rng.random() < 0.5 else -1.0
            lat = side * (half[i] + surface_rng.uniform(0.18, shoulder_w[i] * 0.92))
            pos = c[i] + nrm[i] * lat
            pz = z[i] + bank_sin[i] * lat + 0.01
            self.surface_bits[i].append((float(pos[0]), float(pos[1]), float(pz),
                                         "shoulder", surface_rng.uniform(0.75, 1.18)))

        # sun-facing slope shading per sample (slope vector . light)
        slope = tang * grade[:, None] + nrm * np.sin(bank)[:, None]
        dot = slope[:, 0] * SUN[0] + slope[:, 1] * SUN[1]
        self.light = np.clip(1.0 - dot * 3.05, 0.66, 1.38)
        # altitude tint: higher ground reads slightly lighter
        zr = (z - z.min()) / max(1e-6, float(z.max() - z.min()))
        self.alt = 0.92 + 0.16 * zr

        # Trackside world: bucketed by segment so long circuits only examine
        # nearby scenery each frame. Trees grow in clusters; infrastructure is
        # placed in coherent runs and small marshal/spectator scenes.
        rng = np.random.default_rng(seed)
        self.props_by_segment = [[] for _ in range(n)]

        def add_prop(i, kind, pos, height=1.0, scale=1.0, **extra):
            o = dict(x=float(pos[0]), y=float(pos[1]), z=float(z[i]),
                     height=float(height), scale=float(scale), kind=kind,
                     seg=int(i), hx=float(tang[i, 0]), hy=float(tang[i, 1]))
            o.update(extra)
            self.props_by_segment[i].append(o)

        def add_parked_models(i, pos, scale=1.0, seed=1, side=1):
            """Place real faux-3D chassis as individually sorted world props."""
            for q, model in enumerate(("supra", "lr4", "rx7", "f150")):
                along = (q - 1.5) * 5.8 * scale
                lateral = (0.62 if q % 2 else -0.62) * scale
                car_pos = np.asarray(pos) + tang[i] * along + nrm[i] * lateral
                j = int(trk.nearest(float(car_pos[0]), float(car_pos[1])))
                yaw = math.atan2(float(tang[j, 1]), float(tang[j, 0]))
                if (q + int(seed)) % 3 == 0:
                    yaw += math.pi
                add_prop(j, "parked_model", car_pos, height=1.9, scale=1.0,
                         side=side, model=model, yaw=yaw,
                         seed=int(seed) * 17 + q * 31)

        # Forest clusters. Clear gaps between groups make the dense sections
        # feel intentional and let buildings/signs remain readable.
        tree_step = max(3, int(24.0 / self.spacing))
        tree_kinds = ("pine", "pine", "pine", "deciduous", "birch")
        for i in range(0, n, tree_step):
            if rng.random() < 0.22:
                continue
            primary_side = 1 if rng.random() < 0.5 else -1
            sides = (primary_side, -primary_side) if rng.random() < 0.38 else (primary_side,)
            for side in sides:
                count = int(rng.integers(2, 5))
                anchor_d = half[i] + rng.uniform(9.0, 19.0)
                anchor = c[i] + nrm[i] * side * anchor_d
                nearest = trk.nearest(anchor[0], anchor[1])
                if np.hypot(*(anchor - c[nearest])) < half[nearest] + 4.5:
                    continue
                for _ in range(count):
                    along = rng.uniform(-8.5, 8.5)
                    away = rng.uniform(-2.5, 8.0)
                    pos = anchor + tang[i] * along + nrm[i] * side * away
                    kind = str(rng.choice(tree_kinds))
                    h = rng.uniform(8.5, 15.5) if kind == "pine" else rng.uniform(7.0, 13.0)
                    add_prop(i, kind, pos, height=h, scale=rng.uniform(0.78, 1.18),
                             seed=int(rng.integers(0, 100000)), side=side)

        # Guardrail runs hug faster bends and appear intermittently on straights.
        guard_step = max(2, int(11.5 / self.spacing))
        for i in range(0, n, guard_step):
            bend = k[i] > 0.009
            if not bend and rng.random() > 0.42:
                continue
            sides = (1, -1) if bend and rng.random() < 0.72 else (
                1 if rng.random() < 0.5 else -1,)
            for side in sides:
                pos = c[i] + nrm[i] * side * (half[i] + 1.42)
                add_prop(i, "guardrail", pos, height=0.95,
                         length=guard_step * self.spacing * 1.08, side=side)

        # Catch-fence runs sit behind selected barriers, sparse enough to retain
        # the clean faux-3D silhouette in wide shots.
        fence_step = max(5, int(38.0 / self.spacing))
        for i in range(fence_step // 2, n, fence_step):
            if rng.random() < 0.48:
                continue
            side = 1 if rng.random() < 0.5 else -1
            pos = c[i] + nrm[i] * side * (half[i] + 2.75)
            add_prop(i, "fence", pos, height=rng.uniform(2.2, 3.1),
                     length=fence_step * self.spacing * 0.72, side=side)

        # Braking boards appear as 150/100/50 trios before substantial turns.
        onset = np.logical_and(k > 0.013, np.roll(k, 4) <= 0.013)
        last_board = -10 * n
        min_gap = max(12, int(145.0 / self.spacing))
        for apex in np.flatnonzero(onset):
            if apex - last_board < min_gap:
                continue
            last_board = int(apex)
            side = -1 if signed_curv[apex] >= 0.0 else 1
            for metres, label in ((150.0, "150"), (100.0, "100"), (50.0, "50")):
                bi = (int(apex) - int(metres / self.spacing)) % n
                pos = c[bi] + nrm[bi] * side * (half[bi] + 2.0)
                add_prop(bi, "brake_board", pos, height=2.1, scale=1.0,
                         label=label, side=side)

        # Scene anchors: a hut, light mast, billboard and spectator bank are
        # grouped together, with long empty intervals between them.
        scene_step = max(38, int(310.0 / self.spacing))
        for scene_no, i in enumerate(range(scene_step // 2, n, scene_step)):
            side = 1 if (scene_no + (seed or 7)) % 2 == 0 else -1
            base = c[i] + nrm[i] * side * (half[i] + 5.4)
            add_prop(i, "marshal_hut", base, height=2.7, scale=rng.uniform(0.9, 1.12),
                     side=side, accent=(214, 82, 42) if scene_no % 2 == 0 else (52, 112, 190))
            lamp = base + tang[i] * 4.2
            add_prop(i, "post", lamp, height=4.1, scale=1.0, side=side)
            if scene_no % 2 == 0:
                board = base - tang[i] * 7.0 + nrm[i] * side * 1.8
                add_prop(i, "billboard", board, height=4.4, scale=rng.uniform(0.9, 1.15),
                         side=side, accent=(234, 188, 70), seed=scene_no + 301,
                         messages=("FABLE FIVE", "AI ON TRACK", "LIVE TIMING"))
            else:
                crowd = base - tang[i] * 5.0 + nrm[i] * side * 2.2
                add_prop(i, "spectators", crowd, height=2.2, scale=1.0,
                         side=side, seed=scene_no + 101)
            if scene_no % 3 == 0:
                parked = base + tang[i] * 9.0 + nrm[i] * side * 4.5
                add_parked_models(i, parked, side=side, seed=scene_no + 501)

        # Sparse, distant turbines occupy high/open ground well beyond the
        # barriers. Their true world anchors give them slower parallax than
        # nearby trees without requiring a second camera layer.
        turbine_step = max(80, int(2400.0 / self.spacing))
        high_cut = float(np.percentile(z, 56))
        turbine_no = 0
        for i in range(turbine_step // 2, n, turbine_step):
            if z[i] < high_cut or k[i] > 0.014:
                continue
            side = 1 if (turbine_no + (seed or 7)) % 2 == 0 else -1
            distance = half[i] + rng.uniform(43.0, 61.0)
            pos = (c[i] + nrm[i] * side * distance +
                   tang[i] * rng.uniform(-16.0, 16.0))
            nearest = int(trk.nearest(float(pos[0]), float(pos[1])))
            if np.hypot(*(pos - c[nearest])) < half[nearest] + 28.0:
                continue
            add_prop(i, "wind_turbine", pos, height=rng.uniform(17.0, 23.0),
                     scale=rng.uniform(0.82, 1.08), side=side,
                     seed=9100 + turbine_no * 43)
            turbine_no += 1

        # Nordschleife identity layer, sourced from the real track asset's
        # named OSM landmark arcs. Generic/procedural tracks skip this block.
        profile = str(getattr(trk, "metadata", {}).get("profile", "")).lower()
        self.is_nordschleife = "nordschleife" in profile
        self.landmark_name = np.empty(n, dtype=object)
        self.landmark_name[:] = None
        self.concrete = np.zeros(n, dtype=bool)
        self.graffiti = {}

        if self.is_nordschleife:
            arc = np.asarray(trk.arc, float)
            landmarks = list(getattr(trk, "landmarks", ()))

            def arc_index(metres):
                return min(n - 1, max(0, int(np.searchsorted(arc, float(metres)))))

            starts = {}
            seen_names = set()
            for lm in landmarks:
                name = str(lm.get("name", "")).strip()
                ia, ib = arc_index(lm.get("start_arc", 0.0)), arc_index(lm.get("end_arc", 0.0))
                self.landmark_name[ia:max(ia + 1, ib)] = name
                starts.setdefault(name, ia)
                if not name or name in seen_names or name == "Nürburgring Nordschleife":
                    continue
                seen_names.add(name)
                side = 1 if len(seen_names) % 2 else -1
                pos = c[ia] + nrm[ia] * side * (half[ia] + 3.7)
                add_prop(ia, "section_sign", pos, height=2.65, scale=1.0,
                         side=side, label=name.upper())

                if "Karussell" in name:
                    self.concrete[ia:max(ia + 1, ib)] = True

            def at(name, kind, side=1, distance=7.0, along=0.0,
                   height=3.0, scale=1.0, **extra):
                i = starts.get(name)
                if i is None:
                    return None
                pos = c[i] + nrm[i] * side * (half[i] + distance) + tang[i] * along
                add_prop(i, kind, pos, height=height, scale=scale, side=side, **extra)
                return i

            def parked_at(name, side=1, distance=11.0, along=0.0,
                          scale=1.0, seed=1):
                i = starts.get(name)
                if i is None:
                    return
                pos = (c[i] + nrm[i] * side * (half[i] + distance) +
                       tang[i] * along)
                add_parked_models(i, pos, scale=scale, side=side, seed=seed)

            # T13/start area: pit buildings, timing bridge and event signage.
            t13 = starts.get("T13", 0)
            for q, col in ((-16.0, (166, 172, 174)), (-6.0, (188, 184, 172)),
                           (6.0, (156, 164, 170)), (18.0, (182, 176, 164))):
                i = (t13 + int(q / self.spacing)) % n
                pos = c[i] - nrm[i] * (half[i] + 10.0)
                add_prop(i, "pit_building", pos, height=4.1, scale=1.0,
                         side=-1, color=col, label="NÜRBURGRING")
            # Animated personnel occupy a distinct pit apron between the
            # buildings and barrier. Even their widest animation remains more
            # than 3.5 m outside the racing edge.
            pit_roles = ("marshal", "lollipop", "tyre", "mechanic", "pit_board",
                         "mechanic", "tyre")
            for q, role in zip((-18.0, -12.0, -6.0, 0.0, 7.0, 13.0, 19.0),
                               pit_roles):
                i = (t13 + int(q / self.spacing)) % n
                apron = c[i] - nrm[i] * (half[i] + 5.2)
                add_prop(i, "pit_personnel", apron, height=1.82,
                         scale=0.96 if role == "mechanic" else 1.0,
                         side=-1, role=role, seed=820 + len(self.props_by_segment[i]) + int(q))
            add_prop(t13, "gantry", c[t13], height=5.8, scale=1.0,
                     label="NÜRBURGRING", span=float(half[t13] * 2.0 + 3.0),
                     seed=613, messages=("NÜRBURGRING", "FABLE FIVE", "GREEN TRACK"))

            # Signature spectator/camping locations and camera infrastructure.
            at("Adenauer Forst", "spectators", side=1, distance=7.0,
               height=2.2, seed=701)
            at("Brünnchen", "spectators", side=-1, distance=5.0,
               height=2.2, scale=1.35, seed=702)
            at("Brünnchen", "camp", side=-1, distance=15.0,
               along=13.0, height=3.2, scale=1.25, seed=703)
            parked_at("Brünnchen", side=-1, distance=11.0,
                      along=-12.0, scale=1.25, seed=713)
            at("Brünnchen", "camera_tower", side=1, distance=5.5,
               along=-8.0, height=7.2)
            at("Pflanzgarten", "camp", side=1, distance=14.0,
               along=-10.0, height=3.0, seed=704)
            parked_at("Pflanzgarten", side=1, distance=10.5,
                      along=11.0, scale=1.25, seed=714)
            at("Pflanzgarten", "camera_tower", side=-1, distance=5.5,
               height=7.5)
            at("Sprunghügel", "jump_sign", side=1, distance=3.2,
               height=2.8, label="SPRUNG")
            at("Hohe Acht", "tower", side=1, distance=22.0,
               height=18.0, scale=1.15)
            at("Döttinger Höhe", "camp", side=-1, distance=15.0,
               along=18.0, height=3.1, scale=1.30, seed=705)
            parked_at("Döttinger Höhe", side=-1, distance=11.0,
                      along=-14.0, scale=1.3, seed=715)
            dottinger = at("Döttinger Höhe", "gantry", side=1, distance=-half[starts.get("Döttinger Höhe", 0)],
                           height=5.8, label="DÖTTINGER HÖHE",
                           span=float(half[starts.get("Döttinger Höhe", 0)] * 2.0 + 3.0),
                           seed=716, messages=("DÖTTINGER HÖHE", "LIVE SPEED", "FABLE FIVE"))

            # Stable hand-painted road marks around the best-known viewing zones.
            graffiti_cols = ((208, 68, 62), (228, 214, 88), (76, 160, 220),
                              (226, 226, 218), (94, 190, 108))
            for name in ("Adenauer Forst", "Karussell", "Brünnchen",
                         "Pflanzgarten", "Döttinger Höhe"):
                start = starts.get(name)
                if start is None:
                    continue
                for q in range(3):
                    gi = (start + 5 + q * max(2, int(7.0 / self.spacing))) % n
                    self.graffiti[gi] = (graffiti_cols[(q + len(name)) % len(graffiti_cols)],
                                         0.18 + 0.19 * q, 0.34 + 0.19 * q)

            # Low-elevation mist pockets reinforce Fuchsröhre/Breidscheid's
            # valley character without turning the entire circuit foggy.
            mist_cut = float(np.percentile(z, 23))
            mist_step = max(24, int(185.0 / self.spacing))
            for i in range(mist_step // 2, n, mist_step):
                if z[i] <= mist_cut and rng.random() < 0.72:
                    add_prop(i, "mist_bank", c[i], height=2.2,
                             scale=rng.uniform(1.2, 2.0), seed=int(rng.integers(1000, 9999)))

    def window(self, i0, radius_m):
        k = int(radius_m / self.spacing) + 2
        ka = min(self.n - 1, int(k * 1.7))
        kb = min(self.n - 1 - ka if ka < self.n - 1 else 0, k)
        return [(i0 + d) % self.n for d in range(-kb, ka)]


# --------------------------------------------------------------------------- #
# world rendering
# --------------------------------------------------------------------------- #
def draw_world(pygame, screen, view, geom, mood, i0, veh, skids, t_now,
               wet=False, session=None):
    W, H = view.w, view.h
    screen.fill(mood["ground"])
    wet_level = float(np.clip(float(wet), 0.0, 1.0))

    reach_x = (W / view.scale) * 0.62
    reach_y = (H / view.scale) * 0.62

    # tonal grass patches: big hashed world-space blotches in nearby greens.
    # Drawn under everything, they break the flat ground into meadow.
    cellp = 24.0
    px0 = int((view.cx - reach_x - cellp) / cellp)
    px1 = int((view.cx + reach_x + cellp) / cellp)
    py0 = int((view.cy - reach_y - cellp) / cellp)
    py1 = int((view.cy + reach_y + cellp) / cellp)
    for gx in range(px0, px1 + 1):
        for gy in range(py0, py1 + 1):
            hsh = ((gx * 2654435761) ^ (gy * 40503)) & 0x7fffffff
            if hsh % 5 > 2:
                continue
            f = 0.94 + ((hsh >> 8) % 40) / 330.0        # 0.94 .. 1.06
            wx = gx * cellp + (hsh % 13) - 6.0
            wy = gy * cellp + ((hsh >> 4) % 13) - 6.0
            sx, sy = view.project(wx, wy, view.z0)
            r = int((7.0 + (hsh >> 6) % 6) * view.scale)
            if -r < sx < W + r and -r < sy < H + r:
                pygame.draw.circle(screen, _shade(mood["ground"], f),
                                   (int(sx), int(sy)), r)

    # ground speckle: a hashed world-space grid of dots -> rotation/motion cue
    dot_c = mood["ground_dot"]
    cell = 9.0
    gx0 = int((view.cx - reach_x) / cell)
    gx1 = int((view.cx + reach_x) / cell)
    gy0 = int((view.cy - reach_y) / cell)
    gy1 = int((view.cy + reach_y) / cell)
    for gx in range(gx0, gx1 + 1):
        for gy in range(gy0, gy1 + 1):
            hsh = (gx * 73856093) ^ (gy * 19349663)
            if hsh % 3:
                continue
            px = gx * cell + (hsh % 7) - 3.0
            py = gy * cell + ((hsh >> 3) % 7) - 3.0
            sx, sy = view.project(px, py, view.z0)
            if -8 < sx < W + 8 and -8 < sy < H + 8:
                r = 1 + (hsh >> 5) % 2
                pygame.draw.circle(screen, dot_c, (int(sx), int(sy)), r)

    if session is not None:
        session.draw_ground(pygame, screen, view, mood)

    # road strips, far -> near by projected y so crests overlap correctly
    radius = float(np.hypot(W, H)) * 0.60 / max(view.scale, 0.05)
    idx = geom.window(i0, radius)
    strips = []
    P = {}

    def pt(i, which):
        key = (i, which)
        p = P.get(key)
        if p is None:
            if which == 0:
                w = geom.left[i]
                p = view.project(w[0], w[1], geom.lz[i])
            elif which == 1:
                w = geom.right[i]
                p = view.project(w[0], w[1], geom.rz[i])
            elif which == 2:
                w = geom.left_outer[i]
                p = view.project(w[0], w[1], geom.loz[i])
            elif which == 3:
                w = geom.right_outer[i]
                p = view.project(w[0], w[1], geom.roz[i])
            elif which == 4:
                w = geom.rubber_left[i]
                p = view.project(w[0], w[1], geom.rubber_lz[i])
            else:
                w = geom.rubber_right[i]
                p = view.project(w[0], w[1], geom.rubber_rz[i])
            P[key] = p
        return p

    n = geom.n
    for a in range(len(idx) - 1):
        i, j = idx[a], idx[a + 1]
        if (j - i) % n != 1:
            continue
        l0, r0, l1, r1 = pt(i, 0), pt(i, 1), pt(j, 0), pt(j, 1)
        xs = (l0[0], r0[0], l1[0], r1[0])
        ys = (l0[1], r0[1], l1[1], r1[1])
        if max(xs) < -30 or min(xs) > W + 30 or max(ys) < -30 or min(ys) > H + 30:
            continue
        strips.append((min(ys), i, j, l0, r0, r1, l1))
    strips.sort(key=lambda s: s[0])

    base = mood["road"]
    alt = mood["road_alt"]
    for _, i, j, l0, r0, r1, l1 in strips:
        f = geom.light[i] * geom.alt[i]
        fog_t = _fog_amount(view, geom.center[i, 0], geom.center[i, 1], mood)
        surface_base = ((108, 106, 98) if geom.concrete[i]
                        else (base if (i // 2) % 2 == 0 else alt))
        col = _shade(surface_base,
                     f * mood.get("sun_strength", 1.0))
        col = _fog_color(col, mood, fog_t, 0.82)
        if wet_level > 0.02:
            col = _mix_color(col, (23, 30, 39), 0.29 * wet_level)

        # Gravel/packed-earth shoulders extend the silhouette of the road and
        # keep the white edge line from floating directly against flat grass.
        lo0, ro0 = pt(i, 2), pt(i, 3)
        lo1, ro1 = pt(j, 2), pt(j, 3)
        sh_base = mood["shoulder"] if (i // 3) % 2 == 0 else mood["shoulder_alt"]
        sh_col = _shade(sh_base, geom.shoulder_tone[i] * geom.alt[i])
        sh_col = _fog_color(sh_col, mood, fog_t, 0.88)
        if wet_level > 0.02:
            sh_col = _mix_color(sh_col, (43, 48, 49), 0.22 * wet_level)
        pygame.draw.polygon(screen, sh_col, (lo0, l0, l1, lo1))
        pygame.draw.polygon(screen, _shade(sh_col, 0.94), (r0, ro0, ro1, r1))

        quad = (l0, r0, r1, l1)
        pygame.draw.polygon(screen, col, quad)

        if geom.concrete[i]:
            # Pale slab joints and an inner wear band distinguish the two
            # Karussell bowls from ordinary asphalt.
            joint = _fog_color(_shade(col, 0.70), mood, fog_t, 0.76)
            if (i // 2) % 2 == 0:
                pygame.draw.line(screen, joint, l0, r0, 1)
            inner0 = (l0[0] + (r0[0] - l0[0]) * 0.20,
                      l0[1] + (r0[1] - l0[1]) * 0.20)
            inner1 = (l1[0] + (r1[0] - l1[0]) * 0.20,
                      l1[1] + (r1[1] - l1[1]) * 0.20)
            pygame.draw.line(screen, _fog_color((78, 76, 72), mood, fog_t, 0.78),
                             inner0, inner1, max(1, int(view.scale * 0.08)))

        # The darker, slightly mottled rubber lane follows a corner-biased
        # centerline. It is deliberately subtle: visible as track history, not
        # a gamey navigation ribbon.
        rl0, rr0 = pt(i, 4), pt(i, 5)
        rl1, rr1 = pt(j, 4), pt(j, 5)
        session_rubber = float(session.rubber[i]) if session is not None else 0.0
        rubber_tone = max(0.58, float(geom.rubber_tone[i]) - session_rubber * 0.18)
        pygame.draw.polygon(screen, _shade(col, rubber_tone),
                            (rl0, rr0, rr1, rl1))

        if wet_level > 0.05:
            # Broken specular traces and shallow pooled repairs make the road
            # read as wet without changing grip or the physical surface.
            if (i // 4) % 3 == 0:
                glint = _fog_color(_mix_color(col, (132, 157, 174), 0.34 * wet_level),
                                   mood, fog_t, 0.72)
                q0 = (l0[0] + (r0[0] - l0[0]) * 0.18,
                      l0[1] + (r0[1] - l0[1]) * 0.18)
                q1 = (l1[0] + (r1[0] - l1[0]) * 0.18,
                      l1[1] + (r1[1] - l1[1]) * 0.18)
                pygame.draw.line(screen, glint, q0, q1, 1)
            if geom.patch_mask[i]:
                ta, tb = float(geom.patch_a[i]), float(geom.patch_b[i])
                wa0 = (l0[0] + (r0[0] - l0[0]) * ta, l0[1] + (r0[1] - l0[1]) * ta)
                wb0 = (l0[0] + (r0[0] - l0[0]) * tb, l0[1] + (r0[1] - l0[1]) * tb)
                wa1 = (l1[0] + (r1[0] - l1[0]) * ta, l1[1] + (r1[1] - l1[1]) * ta)
                wb1 = (l1[0] + (r1[0] - l1[0]) * tb, l1[1] + (r1[1] - l1[1]) * tb)
                pygame.draw.polygon(screen, _mix_color(col, (94, 122, 146), 0.28 * wet_level),
                                    (wa0, wb0, wb1, wa1))

        # Occasional partial-width resurfacing repairs break the perfect strips
        # without changing any physical road property.
        if geom.patch_mask[i] and i >= max(2, int(7.0 / geom.spacing)):
            ta, tb = float(geom.patch_a[i]), float(geom.patch_b[i])

            def across(a, b, t):
                return (a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t)

            patch = (across(l0, r0, ta), across(l0, r0, tb),
                     across(l1, r1, tb), across(l1, r1, ta))
            pygame.draw.polygon(screen, _shade(col, float(geom.patch_tone[i])), patch)
            pygame.draw.line(screen, _shade(col, 0.72), patch[0], patch[1], 1)
        if i in geom.graffiti:
            gc, ta, tb = geom.graffiti[i]
            ga = (l0[0] + (r0[0] - l0[0]) * ta,
                  l0[1] + (r0[1] - l0[1]) * ta)
            gb = (l0[0] + (r0[0] - l0[0]) * tb,
                  l0[1] + (r0[1] - l0[1]) * tb)
            gm = ((ga[0] + gb[0]) * 0.5,
                  (ga[1] + gb[1]) * 0.5 - view.scale * 0.16)
            faded = _fog_color(_mix_color(col, gc, 0.76), mood, fog_t, 0.82)
            pygame.draw.line(screen, faded, ga, gm, max(2, int(view.scale * 0.12)))
            pygame.draw.line(screen, faded, gm, gb, max(2, int(view.scale * 0.12)))
        # start gate: a checkered band across the first few metres
        if i < max(2, int(6.0 / geom.spacing)):
            for kk in range(6):
                t0, t1 = kk / 6.0, (kk + 1) / 6.0
                qa = (l0[0] + (r0[0] - l0[0]) * t0, l0[1] + (r0[1] - l0[1]) * t0)
                qb = (l0[0] + (r0[0] - l0[0]) * t1, l0[1] + (r0[1] - l0[1]) * t1)
                qc = (l1[0] + (r1[0] - l1[0]) * t1, l1[1] + (r1[1] - l1[1]) * t1)
                qd = (l1[0] + (r1[0] - l1[0]) * t0, l1[1] + (r1[1] - l1[1]) * t0)
                raw_cc = (228, 228, 224) if (kk + i) % 2 == 0 else (26, 26, 28)
                cc = _fog_color(raw_cc, mood, fog_t, 0.82)
                pygame.draw.polygon(screen, cc, (qa, qb, qc, qd))
        # kerbs on tight corners: red/white bites on both edges
        if geom.kerb[i]:
            ka = _fog_color(mood["kerb_a"] if i % 2 == 0 else mood["kerb_b"],
                            mood, fog_t, 0.84)
            def bite(e0, e1, o0, o1):
                m0 = (e0[0] + (o0[0] - e0[0]) * 0.12, e0[1] + (o0[1] - e0[1]) * 0.12)
                m1 = (e1[0] + (o1[0] - e1[0]) * 0.12, e1[1] + (o1[1] - e1[1]) * 0.12)
                pygame.draw.polygon(screen, ka, (e0, e1, m1, m0))
            bite(l0, l1, r0, r1)
            bite(r0, r1, l0, l1)
        # dashed centreline
        if (i // 3) % 2 == 0:
            c0 = ((l0[0] + r0[0]) / 2, (l0[1] + r0[1]) / 2)
            c1 = ((l1[0] + r1[0]) / 2, (l1[1] + r1[1]) / 2)
            pygame.draw.line(screen, _fog_color(mood["line"], mood, fog_t, 0.86), c0, c1,
                             max(1, int(view.scale * 0.06)))

    # Stable aggregate and shoulder gravel: tiny near-camera flecks add motion
    # and material scale while remaining nearly invisible in wide director shots.
    if view.scale > 7.0:
        for i in idx:
            for wx, wy, wz, kind, tone in geom.surface_bits[i]:
                sx, sy = view.project(wx, wy, wz)
                if -3 < sx < W + 3 and -3 < sy < H + 3:
                    bc = mood["aggregate"] if kind == "road" else mood["shoulder_speck"]
                    ft = _fog_amount(view, wx, wy, mood)
                    pygame.draw.circle(screen, _fog_color(_shade(bc, tone), mood, ft, 0.9),
                                       (int(sx), int(sy)), 1)

    # Continuous edge lines with per-segment fogging, so distant white paint
    # settles into the atmosphere instead of glowing through it.
    ew = max(1, int(view.scale * 0.09))
    for which in (0, 1):
        prev = None
        for i in idx:
            if prev is not None and (i - prev) % n == 1:
                mx = (geom.center[prev, 0] + geom.center[i, 0]) * 0.5
                my = (geom.center[prev, 1] + geom.center[i, 1]) * 0.5
                ft = _fog_amount(view, mx, my, mood)
                edge_col = _fog_color(mood["edge"], mood, ft, 0.88)
                pygame.draw.line(screen, edge_col, pt(prev, which), pt(i, which), ew)
            prev = i

    # skid marks ride ON the road surface (projected with its z)
    skids.draw(pygame, screen, view, mood)

    # props with height + soft shadows, painter-sorted with the car later
    props_out = []
    seen = set()
    for seg in idx:
        if seg in seen:
            continue
        seen.add(seg)
        for o in geom.props_by_segment[seg]:
            px, py, pz = o["x"], o["y"], o["z"]
            if abs(px - view.cx) > radius + 35 or abs(py - view.cy) > radius + 35:
                continue
            bx, by = view.project(px, py, pz)
            if -120 < bx < W + 120 and -180 < by < H + 100:
                props_out.append((by, o))
    return props_out


_PROP_FONTS = {}


def _prop_font(pygame, size, bold=False):
    key = (size, bold)
    if key not in _PROP_FONTS:
        _PROP_FONTS[key] = pygame.font.SysFont("menlo,consolas,monospace", size,
                                               bold=bold)
    return _PROP_FONTS[key]


def draw_prop(pygame, screen, view, mood, p, t_now=0.0):
    _, o = p
    px, py, pz = o["x"], o["y"], o["z"]
    h, kind, s = o["height"], o["kind"], o["scale"]
    bx, by = view.project(px, py, pz)
    tx, ty = view.project(px, py, pz + h)
    sc = view.scale
    night = 0.56 if mood["name"] == "night" else (0.82 if mood["name"] == "dusk" else 1.0)
    fog_t = _fog_amount(view, px, py, mood)

    def pshade(color, factor=1.0, fog_strength=0.92):
        return _fog_color(_shade(color, factor), mood, fog_t, fog_strength)

    # Every substantial object casts the same world-directional shadow. Dusk
    # stretches it; night nearly removes it, leaving local lamps/headlights to
    # define the scene.
    if kind not in ("fence", "post", "brake_board", "mist_bank", "gantry"):
        reach = float(mood.get("shadow_reach", 0.8))
        shadow_h = min(h, 12.0)
        swx = px - SUN[0] * shadow_h * reach / max(SUN[2], 0.1)
        swy = py - SUN[1] * shadow_h * reach / max(SUN[2], 0.1)
        shadow_tip = view.project(swx, swy, pz)
        shadow_w = max(2, int(sc * s * (0.30 if kind in ("pine", "deciduous", "birch") else 0.18)))
        soft_shadow = _mix_color(mood["ground"], mood["shadow"],
                                 0.54 if mood["name"] != "night" else 0.30)
        pygame.draw.line(screen, _fog_color(soft_shadow, mood, fog_t, 0.48),
                         (bx, by), shadow_tip, shadow_w)

    if kind in ("pine", "deciduous", "birch"):
        r = max(4, int((2.05 if kind == "pine" else 2.45) * s * sc))
        contact_shadow = _mix_color(mood["ground"], mood["shadow"], 0.66)
        pygame.draw.ellipse(screen, _fog_color(contact_shadow, mood, fog_t, 0.48),
                            (int(bx - r * 0.95), int(by - r * 0.27),
                             int(r * 2.1), max(3, int(r * 0.62))))
        trunk = (64, 48, 34) if kind != "birch" else (158, 151, 132)
        pygame.draw.line(screen, pshade(trunk, night), (bx, by), (tx, ty),
                         max(2, int(sc * (0.22 if kind == "pine" else 0.30) * s)))

        if kind == "pine":
            palette = ((22, 54, 30), (29, 68, 35), (38, 82, 42))
            for kf, wf, hh, col in ((0.35, 1.00, 0.34, palette[0]),
                                    (0.60, 0.78, 0.29, palette[1]),
                                    (0.82, 0.54, 0.23, palette[2])):
                cx = bx + (tx - bx) * kf
                cy = by + (ty - by) * kf
                top_y = cy - abs(by - ty) * hh
                pts = ((cx, top_y), (cx - r * wf, cy + r * 0.28),
                       (cx - r * 0.16, cy + r * 0.12),
                       (cx, cy + r * 0.34),
                       (cx + r * 0.16, cy + r * 0.12),
                       (cx + r * wf, cy + r * 0.28))
                pygame.draw.polygon(screen, pshade(col, night), pts)
                pygame.draw.line(screen, pshade(_shade(col, 1.32), night, 0.82),
                                 pts[0], pts[1], max(1, int(sc * 0.045)))
        else:
            seed = int(o.get("seed", 1))
            ccx = bx + (tx - bx) * 0.72
            ccy = by + (ty - by) * 0.72
            pts = []
            for q in range(12):
                ang = math.tau * q / 12.0
                wobble = 0.82 + 0.20 * math.sin(seed * 0.017 + q * 2.37)
                pts.append((ccx + math.cos(ang) * r * wobble,
                            ccy + math.sin(ang) * r * 0.78 * wobble))
            base_col = (48, 92, 43) if kind == "deciduous" else (68, 105, 56)
            pygame.draw.polygon(screen, pshade(base_col, 0.70 * night), pts)
            hi = (70, 122, 58) if kind == "deciduous" else (102, 132, 78)
            pygame.draw.circle(screen, pshade(hi, night),
                               (int(ccx - r * 0.18), int(ccy - r * 0.22)),
                               max(3, int(r * 0.62)))

    elif kind == "guardrail":
        hx, hy = o["hx"], o["hy"]
        L = o.get("length", 8.0) * 0.5
        ends = [(px - hx * L, py - hy * L), (px + hx * L, py + hy * L)]
        steel = pshade((178, 184, 190), night)
        dark = pshade((92, 98, 104), night)
        for zoff, width in ((0.58, max(2, int(sc * 0.11))),
                            (0.91, max(1, int(sc * 0.07)))):
            a = view.project(ends[0][0], ends[0][1], pz + zoff)
            b = view.project(ends[1][0], ends[1][1], pz + zoff)
            pygame.draw.line(screen, steel, a, b, width)
            pygame.draw.line(screen, dark, (a[0], a[1] + 1), (b[0], b[1] + 1), 1)
        for q in (-0.42, 0.0, 0.42):
            wx, wy = px + hx * L * q * 2.0, py + hy * L * q * 2.0
            a = view.project(wx, wy, pz)
            b = view.project(wx, wy, pz + 0.92)
            pygame.draw.line(screen, dark, a, b, max(1, int(sc * 0.07)))

    elif kind == "fence":
        hx, hy = o["hx"], o["hy"]
        L = o.get("length", 16.0) * 0.5
        steel = pshade((112, 124, 126), night)
        nodes = []
        for q in (-1.0, -0.5, 0.0, 0.5, 1.0):
            wx, wy = px + hx * L * q, py + hy * L * q
            bot = view.project(wx, wy, pz)
            top = view.project(wx, wy, pz + h)
            nodes.append((bot, top))
            pygame.draw.line(screen, steel, bot, top, max(1, int(sc * 0.055)))
        pygame.draw.lines(screen, steel, False, [n[1] for n in nodes], 1)
        pygame.draw.lines(screen, pshade(steel, 0.80), False,
                          [(n[0][0], n[0][1] + (n[1][1] - n[0][1]) * 0.45) for n in nodes], 1)
        for q in range(len(nodes) - 1):
            pygame.draw.line(screen, pshade(steel, 0.72), nodes[q][0], nodes[q + 1][1], 1)

    elif kind == "brake_board":
        pole_top = view.project(px, py, pz + h * 0.70)
        pygame.draw.line(screen, pshade((138, 142, 148), night), (bx, by), pole_top,
                         max(1, int(sc * 0.07)))
        bw, bh = max(16, int(sc * 1.18)), max(13, int(sc * 0.92))
        rect = pygame.Rect(int(pole_top[0] - bw / 2), int(pole_top[1] - bh), bw, bh)
        pygame.draw.rect(screen, pshade((235, 234, 222), night), rect, border_radius=2)
        pygame.draw.rect(screen, pshade((42, 44, 48), night), rect, max(1, bw // 14),
                         border_radius=2)
        font = _prop_font(pygame, max(9, int(sc * 0.56)), bold=True)
        label = font.render(o.get("label", "100"), True, pshade((28, 30, 32), night))
        screen.blit(label, label.get_rect(center=rect.center))

    elif kind == "marshal_hut":
        w = 2.2 * s * sc
        top = view.project(px, py, pz + h * 0.72)
        accent = pshade(o.get("accent", (210, 84, 44)), night)
        wall = pshade((174, 172, 158), night)
        body = ((bx - w, by), (bx + w, by), (top[0] + w, top[1]), (top[0] - w, top[1]))
        pygame.draw.polygon(screen, wall, body)
        roof_y = top[1] - abs(by - top[1]) * 0.28
        pygame.draw.polygon(screen, accent,
                            ((top[0] - w * 1.16, top[1]), (top[0] + w * 1.16, top[1]),
                             (top[0], roof_y)))
        win = pygame.Rect(int(top[0] - w * 0.55), int(top[1] + 5),
                          max(5, int(w * 1.1)), max(4, int(abs(by - top[1]) * 0.30)))
        pygame.draw.rect(screen, pshade((42, 58, 66), night), win)

    elif kind == "billboard":
        hx, hy = o["hx"], o["hy"]
        half_w = 3.0 * s
        corners = []
        for zoff in (2.2, h):
            corners.append(view.project(px - hx * half_w, py - hy * half_w, pz + zoff))
            corners.append(view.project(px + hx * half_w, py + hy * half_w, pz + zoff))
        for side in (-0.68, 0.68):
            wx, wy = px + hx * half_w * side, py + hy * half_w * side
            pygame.draw.line(screen, pshade((92, 94, 100), night),
                             view.project(wx, wy, pz), view.project(wx, wy, pz + 2.3),
                             max(1, int(sc * 0.08)))
        panel = (corners[0], corners[1], corners[3], corners[2])
        accent = pshade(o.get("accent", (230, 190, 68)), night)
        pygame.draw.polygon(screen, pshade((14, 20, 23), night), panel)
        pygame.draw.lines(screen, accent, True, panel,
                          max(1, int(sc * 0.08)))
        messages = tuple(o.get("messages", ("FABLE FIVE", "LIVE TIMING", "AI ON TRACK")))
        seed = int(o.get("seed", 0))
        msg_i = int((t_now + seed * 0.037) / 2.55) % max(1, len(messages))
        font = _prop_font(pygame, max(8, int(sc * 0.48)), bold=True)
        label = font.render(messages[msg_i], True, accent)
        pcx = sum(q[0] for q in panel) / 4.0
        pcy = sum(q[1] for q in panel) / 4.0
        screen.blit(label, label.get_rect(center=(int(pcx), int(pcy))))
        # Slow electronic scan and alternating status beacons communicate that
        # this is an active display, not a painted sponsor board.
        scan_t = ((t_now * 0.42 + seed * 0.011) % 1.0)
        left = (panel[0][0] + (panel[2][0] - panel[0][0]) * scan_t,
                panel[0][1] + (panel[2][1] - panel[0][1]) * scan_t)
        right = (panel[1][0] + (panel[3][0] - panel[1][0]) * scan_t,
                 panel[1][1] + (panel[3][1] - panel[1][1]) * scan_t)
        pygame.draw.line(screen, pshade((90, 126, 118), night), left, right, 1)
        beacon_on = int((t_now + seed * 0.13) * 2.0) % 2 == 0
        beacon_col = (255, 185, 62) if beacon_on else (82, 64, 30)
        for q in (panel[0], panel[1]):
            pygame.draw.circle(screen, pshade(beacon_col, night),
                               (int(q[0]), int(q[1])), max(1, int(sc * 0.10)))

    elif kind == "section_sign":
        pole_top = view.project(px, py, pz + h * 0.76)
        pygame.draw.line(screen, pshade((118, 126, 122), night), (bx, by), pole_top,
                         max(1, int(sc * 0.08)))
        label_text = o.get("label", "NORDSCHLEIFE")
        font = _prop_font(pygame, max(8, int(sc * 0.48)), bold=True)
        label = font.render(label_text, True, pshade((236, 238, 226), night))
        bw = max(label.get_width() + 14, int(sc * 3.2))
        bh = max(label.get_height() + 8, int(sc * 0.92))
        rect = pygame.Rect(int(pole_top[0] - bw / 2), int(pole_top[1] - bh), bw, bh)
        pygame.draw.rect(screen, pshade((30, 78, 48), night), rect, border_radius=3)
        pygame.draw.rect(screen, pshade((224, 224, 204), night), rect, 1, border_radius=3)
        screen.blit(label, label.get_rect(center=rect.center))

    elif kind == "gantry":
        nx, ny = -o["hy"], o["hx"]
        span = float(o.get("span", 14.0)) * 0.5
        steel = pshade((142, 150, 152), night)
        tops = []
        for side in (-1.0, 1.0):
            wx, wy = px + nx * span * side, py + ny * span * side
            bot = view.project(wx, wy, pz)
            top = view.project(wx, wy, pz + h)
            tops.append(top)
            pygame.draw.line(screen, steel, bot, top, max(2, int(sc * 0.10)))
        pygame.draw.line(screen, steel, tops[0], tops[1], max(3, int(sc * 0.16)))
        pygame.draw.line(screen, pshade((56, 62, 64), night),
                         (tops[0][0], tops[0][1] + 3), (tops[1][0], tops[1][1] + 3),
                         max(2, int(sc * 0.10)))
        seed = int(o.get("seed", 0))
        messages = tuple(o.get("messages", (o.get("label", "NÜRBURGRING"),
                                             "FABLE FIVE", "LIVE TIMING")))
        msg_i = int((t_now + seed * 0.021) / 3.0) % max(1, len(messages))
        font = _prop_font(pygame, max(8, int(sc * 0.45)), bold=True)
        label = font.render(messages[msg_i], True, pshade((245, 205, 112), night))
        mid = ((tops[0][0] + tops[1][0]) * 0.5,
               (tops[0][1] + tops[1][1]) * 0.5)
        screen.blit(label, label.get_rect(center=(int(mid[0]), int(mid[1] - 7))))
        pulse = 0.5 + 0.5 * math.sin(t_now * 3.2 + seed)
        lamp_col = pshade((255, int(110 + 110 * pulse), 46), night)
        for q in tops:
            pygame.draw.circle(screen, lamp_col, (int(q[0]), int(q[1] - 2)),
                               max(2, int(sc * 0.12)))

    elif kind == "pit_building":
        w = 4.6 * s * sc
        top = view.project(px, py, pz + h * 0.78)
        wall = pshade(o.get("color", (172, 174, 170)), night)
        body = ((bx - w, by), (bx + w, by), (top[0] + w, top[1]), (top[0] - w, top[1]))
        pygame.draw.polygon(screen, wall, body)
        pygame.draw.line(screen, pshade((62, 68, 72), night),
                         (top[0] - w * 1.05, top[1]), (top[0] + w * 1.05, top[1]),
                         max(2, int(sc * 0.18)))
        floors = 2
        for row in range(floors):
            wy = top[1] + (by - top[1]) * (0.28 + row * 0.32)
            for q in range(-3, 4):
                ww = max(4, int(w * 0.18))
                rect = pygame.Rect(int(top[0] + q * w * 0.25 - ww / 2), int(wy),
                                   ww, max(3, int(sc * 0.34)))
                pygame.draw.rect(screen, pshade((42, 62, 72), night), rect)

    elif kind == "pit_personnel":
        role = str(o.get("role", "mechanic"))
        seed = int(o.get("seed", 1))
        phase = t_now * (2.25 if role in ("tyre", "marshal") else 1.55) + seed * 0.37
        hx, hy = o["hx"], o["hy"]

        # Tyre carriers pace a short, deterministic loop along the apron. All
        # other roles stay at their assigned workstation.
        walk = math.sin(phase) * 0.78 if role == "tyre" else 0.0
        wx, wy = px + hx * walk, py + hy * walk
        foot = view.project(wx, wy, pz + 0.015)
        person_h = h * s * sc * LIFT
        crouch = 0.34 if role == "mechanic" else 0.0
        bob = (abs(math.sin(phase * 1.8)) * sc * 0.025
               if role == "tyre" else math.sin(phase) * sc * 0.012)
        hip = (foot[0], foot[1] - person_h * (0.43 - crouch * 0.14) - bob)
        shoulder = (foot[0], foot[1] - person_h * (0.76 - crouch * 0.30) - bob)
        head = (int(foot[0]), int(foot[1] - person_h * (0.91 - crouch * 0.34) - bob))

        suits = {
            "marshal": ((232, 116, 38), (248, 218, 86)),
            "lollipop": ((52, 92, 160), (222, 226, 218)),
            "tyre": ((42, 48, 56), (204, 56, 48)),
            "mechanic": ((28, 34, 42), (66, 126, 202)),
            "pit_board": ((52, 92, 160), (236, 204, 82)),
        }
        suit, accent = suits.get(role, suits["mechanic"])
        suit = pshade(suit, night)
        accent = pshade(accent, night)
        skin = pshade((190, 145, 108), night)
        limb_w = max(1, int(sc * 0.10 * s))

        # Contact patch and animated legs.
        pygame.draw.ellipse(screen, pshade((42, 42, 40), night),
                            (int(foot[0] - sc * 0.30), int(foot[1] - sc * 0.08),
                             max(3, int(sc * 0.60)), max(2, int(sc * 0.18))))
        stride = math.sin(phase * 2.0) * sc * 0.18 if role == "tyre" else sc * 0.11
        pygame.draw.line(screen, suit, hip,
                         (foot[0] - stride, foot[1]), limb_w)
        pygame.draw.line(screen, suit, hip,
                         (foot[0] + stride, foot[1]), limb_w)
        pygame.draw.line(screen, suit, hip, shoulder, max(2, int(sc * 0.18 * s)))
        pygame.draw.line(screen, accent,
                         (shoulder[0] - sc * 0.12, shoulder[1] + person_h * 0.08),
                         (shoulder[0] + sc * 0.12, shoulder[1] + person_h * 0.08),
                         max(1, int(sc * 0.055)))
        pygame.draw.circle(screen, skin, head, max(2, int(sc * 0.115 * s)))
        pygame.draw.line(screen, accent,
                         (head[0] - sc * 0.12, head[1] - sc * 0.07),
                         (head[0] + sc * 0.13, head[1] - sc * 0.07),
                         max(1, int(sc * 0.055)))

        if role == "marshal":
            # One hand signals the lane; the other waves a real hinged flag.
            hand = (shoulder[0] + sc * 0.32, shoulder[1] - person_h * 0.08)
            pygame.draw.line(screen, suit, shoulder, hand, limb_w)
            pole_ang = -1.28 + math.sin(phase * 1.7) * 0.38
            tip = (hand[0] + math.cos(pole_ang) * sc * 1.15,
                   hand[1] + math.sin(pole_ang) * sc * 1.15)
            pygame.draw.line(screen, pshade((170, 174, 168), night), hand, tip,
                             max(1, int(sc * 0.045)))
            flag_col = (72, 204, 98) if int(t_now / 3.5 + seed) % 2 else (244, 210, 62)
            flap = math.sin(phase * 3.1) * sc * 0.18
            pygame.draw.polygon(screen, pshade(flag_col, night),
                                (tip, (tip[0] + sc * 0.68, tip[1] + flap),
                                 (tip[0] + sc * 0.56, tip[1] + sc * 0.42)))
        elif role == "lollipop":
            hand = (shoulder[0] + sc * 0.34, shoulder[1] - person_h * 0.02)
            top = (hand[0], hand[1] - sc * (0.78 + 0.08 * math.sin(phase)))
            pygame.draw.line(screen, suit, shoulder, hand, limb_w)
            pygame.draw.line(screen, pshade((184, 188, 184), night), hand, top,
                             max(1, int(sc * 0.06)))
            rr = max(3, int(sc * 0.25))
            pygame.draw.circle(screen, pshade((28, 32, 34), night),
                               (int(top[0]), int(top[1])), rr)
            lamp = (76, 224, 104) if int((t_now + seed) * 2.0) % 2 else (214, 66, 48)
            pygame.draw.circle(screen, pshade(lamp, night),
                               (int(top[0]), int(top[1])), max(1, rr // 2))
        elif role == "tyre":
            arm_swing = math.sin(phase * 2.0) * sc * 0.16
            hand = (shoulder[0] + sc * 0.30, shoulder[1] + person_h * 0.22 + arm_swing)
            pygame.draw.line(screen, suit, shoulder, hand, limb_w)
            tyre = (int(hand[0] + sc * 0.16), int(hand[1] + sc * 0.10))
            rr = max(3, int(sc * 0.29))
            pygame.draw.circle(screen, pshade((20, 22, 24), night), tyre, rr,
                               max(2, int(sc * 0.12)))
            pygame.draw.circle(screen, pshade((126, 130, 132), night), tyre,
                               max(1, int(rr * 0.30)))
        elif role == "mechanic":
            # Crouched mechanic works a low wheel-gun with a cyclic wrist.
            hand = (shoulder[0] + sc * 0.40,
                    shoulder[1] + person_h * (0.20 + 0.04 * math.sin(phase * 2.4)))
            pygame.draw.line(screen, suit, shoulder, hand, limb_w)
            tool_end = (hand[0] + sc * 0.33, hand[1] + sc * 0.05)
            pygame.draw.line(screen, pshade((164, 170, 174), night), hand, tool_end,
                             max(2, int(sc * 0.09)))
            pygame.draw.circle(screen, accent, (int(tool_end[0]), int(tool_end[1])),
                               max(1, int(sc * 0.08)))
        else:  # pit-board operator
            lift = math.sin(phase) * sc * 0.08
            board_c = (shoulder[0] + sc * 0.34, shoulder[1] - sc * 0.43 + lift)
            pygame.draw.line(screen, suit, shoulder, board_c, limb_w)
            bw, bh = max(12, int(sc * 0.92)), max(9, int(sc * 0.62))
            rect = pygame.Rect(int(board_c[0] - bw / 2), int(board_c[1] - bh / 2), bw, bh)
            pygame.draw.rect(screen, pshade((16, 22, 26), night), rect, border_radius=2)
            pygame.draw.rect(screen, accent, rect, max(1, int(sc * 0.055)), border_radius=2)
            messages = ("BOX", "P3", "GO")
            msg = messages[int((t_now + seed * 0.17) / 2.2) % len(messages)]
            font = _prop_font(pygame, max(7, int(sc * 0.38)), bold=True)
            label = font.render(msg, True, accent)
            screen.blit(label, label.get_rect(center=rect.center))

    elif kind == "parked_model":
        # Use the project's full procedural car meshes. Every parked vehicle is
        # an individual world prop, so projection and painter order match trees.
        model = str(o.get("model", "supra"))
        parked_spec = get_car(model)
        parked = SimpleNamespace(
            x=px, y=py, z=pz, road_z=pz, yaw=float(o.get("yaw", 0.0)),
            ax=0.0, ay=0.0, pitch=0.0, roll=0.0, steer_angle=0.0,
        )
        draw_car_v2(pygame, screen, view, parked, parked_spec, mood,
                    fog_amount=fog_t, allow_lod=True)

        # Hazards remain, now attached to the actual rear corners of each mesh.
        seed = int(o.get("seed", 1))
        if ((t_now * 1.38 + seed * 0.173) % 1.0) < 0.43:
            yaw = parked.yaw
            cyw, syw = math.cos(yaw), math.sin(yaw)
            HL = (parked_spec.wheelbase + 1.7) / 2.0
            HW = (parked_spec.track_width + 0.36) / 2.0
            lamp_z = pz + (0.68 if model in ("lr4", "f150") else 0.42)
            for lamp_side in (-1.0, 1.0):
                lx = px - cyw * HL * 0.96 - syw * lamp_side * HW * 0.62
                ly = py - syw * HL * 0.96 + cyw * lamp_side * HW * 0.62
                lp = view.project(lx, ly, lamp_z)
                pygame.draw.circle(screen, pshade((255, 178, 62), night, 0.70),
                                   (int(lp[0]), int(lp[1])),
                                   max(1, int(sc * 0.075)))

    elif kind == "camp":
        rng = np.random.default_rng(int(o.get("seed", 1)))
        cols = ((196, 62, 54), (62, 112, 188), (226, 194, 82), (214, 216, 210))
        for q in range(6):
            along = rng.uniform(-7.0, 7.0)
            lateral = rng.uniform(-2.0, 2.0)
            wx = px + o["hx"] * along - o["hy"] * lateral
            wy = py + o["hy"] * along + o["hx"] * lateral
            base = view.project(wx, wy, pz)
            top = view.project(wx, wy, pz + rng.uniform(1.2, 2.0))
            size = sc * rng.uniform(0.75, 1.25)
            col = pshade(cols[q % len(cols)], night)
            if q % 3 == 0:                 # camper van
                rect = pygame.Rect(int(base[0] - size), int(top[1]),
                                   max(4, int(size * 2)), max(4, int(base[1] - top[1])))
                pygame.draw.rect(screen, col, rect, border_radius=2)
                pygame.draw.rect(screen, pshade((48, 66, 72), night),
                                 (rect.x + 3, rect.y + 3, max(2, rect.w // 3), max(2, rect.h // 3)))
            else:                           # small spectator tent
                pygame.draw.polygon(screen, col,
                                    ((base[0] - size, base[1]), (base[0] + size, base[1]),
                                     (top[0], top[1])))

    elif kind == "camera_tower":
        top = view.project(px, py, pz + h)
        steel = pshade((104, 112, 116), night)
        spread = max(5, int(sc * 0.42))
        pygame.draw.line(screen, steel, (bx - spread, by), (top[0] - spread, top[1]), 2)
        pygame.draw.line(screen, steel, (bx + spread, by), (top[0] + spread, top[1]), 2)
        for q in (0.25, 0.50, 0.75):
            yy = by + (top[1] - by) * q
            pygame.draw.line(screen, pshade((74, 80, 84), night),
                             (bx - spread, yy), (bx + spread, yy), 1)
        platform = pygame.Rect(int(top[0] - sc * 0.9), int(top[1] - sc * 0.24),
                               max(5, int(sc * 1.8)), max(3, int(sc * 0.48)))
        pygame.draw.rect(screen, pshade((46, 50, 54), night), platform)
        pygame.draw.circle(screen, pshade((32, 34, 36), night),
                           (int(top[0] + sc * 0.52), int(top[1] - sc * 0.34)),
                           max(2, int(sc * 0.18)))

    elif kind == "wind_turbine":
        seed = int(o.get("seed", 1))
        hx, hy = o["hx"], o["hy"]
        nx, ny = -hy, hx
        hub_z = pz + h
        hub = view.project(px, py, hub_z)
        base_half = 0.42 * s
        top_half = 0.13 * s
        tower = (view.project(px + nx * base_half, py + ny * base_half, pz),
                 view.project(px - nx * base_half, py - ny * base_half, pz),
                 view.project(px - nx * top_half, py - ny * top_half, hub_z),
                 view.project(px + nx * top_half, py + ny * top_half, hub_z))
        pygame.draw.polygon(screen, pshade((194, 199, 194), night), tower)
        pygame.draw.line(screen, pshade((116, 124, 124), night),
                         tower[0], tower[3], max(1, int(sc * 0.055)))

        rotor_r = h * 0.27 * s
        direction = -1.0 if seed % 2 else 1.0
        rotor_phase = seed * 0.071 + t_now * 0.58 * direction

        def rotor_point(angle, radius):
            lateral = math.cos(angle) * radius
            vertical = math.sin(angle) * radius
            return view.project(px + hx * lateral, py + hy * lateral,
                                hub_z + vertical)

        blade_col = pshade((224, 228, 220), night)
        edge_col = pshade((142, 150, 150), night)
        for blade in range(3):
            angle = rotor_phase + blade * math.tau / 3.0
            root = rotor_r * 0.16
            # Tapered swept blade in the true vertical world plane. The small
            # angular widths feather into the hub instead of forming hard rods.
            poly = (rotor_point(angle - 0.16, root),
                    rotor_point(angle - 0.045, rotor_r),
                    rotor_point(angle + 0.035, rotor_r * 0.97),
                    rotor_point(angle + 0.12, root))
            pygame.draw.polygon(screen, blade_col, poly)
            pygame.draw.line(screen, edge_col, poly[0], poly[1], 1)
        pygame.draw.circle(screen, pshade((176, 184, 182), night),
                           (int(hub[0]), int(hub[1])), max(2, int(sc * 0.16 * s)))
        if ((t_now + seed * 0.11) % 2.4) < 0.16:
            pygame.draw.circle(screen, pshade((242, 62, 48), night, 0.55),
                               (int(hub[0]), int(hub[1] - sc * 0.10)),
                               max(1, int(sc * 0.075)))

    elif kind == "tower":
        top = view.project(px, py, pz + h)
        steel = pshade((116, 120, 120), night)
        spread = max(8, int(sc * 0.8))
        pygame.draw.line(screen, steel, (bx - spread, by), top, max(2, int(sc * 0.10)))
        pygame.draw.line(screen, steel, (bx + spread, by), top, max(2, int(sc * 0.10)))
        for q in range(1, 6):
            yy = by + (top[1] - by) * q / 6.0
            ww = spread * (1.0 - q / 7.0)
            pygame.draw.line(screen, steel, (top[0] - ww, yy), (top[0] + ww, yy), 1)
        pygame.draw.circle(screen, pshade((180, 72, 58), night),
                           (int(top[0]), int(top[1] - sc * 0.28)), max(2, int(sc * 0.22)))

    elif kind == "jump_sign":
        pole_top = view.project(px, py, pz + h * 0.68)
        pygame.draw.line(screen, pshade((134, 138, 140), night), (bx, by), pole_top, 2)
        r = max(8, int(sc * 0.75))
        tri = ((pole_top[0], pole_top[1] - r),
               (pole_top[0] - r, pole_top[1] + r * 0.65),
               (pole_top[0] + r, pole_top[1] + r * 0.65))
        pygame.draw.polygon(screen, pshade((238, 206, 62), night), tri)
        pygame.draw.lines(screen, pshade((42, 42, 40), night), True, tri, 2)
        pygame.draw.line(screen, pshade((42, 42, 40), night),
                         (pole_top[0] - r * 0.45, pole_top[1] + r * 0.2),
                         (pole_top[0] + r * 0.45, pole_top[1] - r * 0.12), 2)

    elif kind == "mist_bank":
        center = view.project(px, py, pz + h * 0.48)
        rw = max(30, int(sc * 8.5 * s))
        rh = max(10, int(sc * 2.2 * s))
        surf = pygame.Surface((rw * 2, rh * 2), pygame.SRCALPHA)
        for q in range(5, 0, -1):
            alpha = int(5 + 4 * (6 - q))
            rect = pygame.Rect(rw - rw * q / 5, rh - rh * q / 5,
                               rw * 2 * q / 5, rh * 2 * q / 5)
            pygame.draw.ellipse(surf, (*mood.get("fog", (128, 138, 122)), alpha), rect)
        screen.blit(surf, (center[0] - rw, center[1] - rh))

    elif kind == "spectators":
        seed = int(o.get("seed", 1))
        rng = np.random.default_rng(seed)
        pygame.draw.ellipse(screen, pshade((64, 62, 56), night),
                            (int(bx - sc * 4.5), int(by - sc * 0.5),
                             int(sc * 9.0), max(3, int(sc * 1.2))))
        cols = ((210, 80, 70), (72, 128, 210), (232, 202, 76),
                (88, 170, 104), (220, 220, 218))
        camera_heads = []
        for q in range(16):
            ox = rng.uniform(-4.0, 4.0)
            oy = rng.uniform(-0.7, 0.7)
            base = view.project(px + o["hx"] * ox, py + o["hy"] * ox, pz + oy * 0.08)
            ph = rng.uniform(1.25, 1.85) * sc * LIFT
            col = pshade(cols[q % len(cols)], night)
            pygame.draw.line(screen, col, base, (base[0], base[1] - ph),
                             max(1, int(sc * 0.10)))
            pygame.draw.circle(screen, pshade((190, 151, 116), night),
                               (int(base[0]), int(base[1] - ph - max(2, sc * 0.10))),
                               max(1, int(sc * 0.10)))
            if q % 3 == 0:
                camera_heads.append((base[0] + sc * 0.16, base[1] - ph * 0.72))

        period = 1.65 + (seed % 9) * 0.19
        phase = (t_now + seed * 0.071) % period
        distance = math.hypot(px - view.cx, py - view.cy)
        if phase < 0.070 and distance < 155.0 and camera_heads:
            flash_no = int((t_now + seed) / period) % len(camera_heads)
            fx, fy = camera_heads[flash_no]
            radius = max(8, int(sc * (0.72 if mood["name"] == "night" else 0.48)))
            glow = pygame.Surface((radius * 2 + 2, radius * 2 + 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (218, 232, 255, 38),
                               (radius + 1, radius + 1), radius)
            pygame.draw.circle(glow, (255, 252, 225, 220),
                               (radius + 1, radius + 1), max(2, radius // 5))
            screen.blit(glow, (fx - radius - 1, fy - radius - 1),
                        special_flags=pygame.BLEND_RGBA_ADD)
            pygame.draw.line(screen, (255, 248, 220),
                             (fx - radius * 0.65, fy), (fx + radius * 0.65, fy), 1)
            pygame.draw.line(screen, (255, 248, 220),
                             (fx, fy - radius * 0.65), (fx, fy + radius * 0.65), 1)

    else:                                      # marshal post: pole + amber light
        if mood["headlights"]:                 # dusk/night: pooled light on the ground
            pr = max(6, int(sc * 1.9))
            pool = pygame.Surface((pr * 2, pr), pygame.SRCALPHA)
            pygame.draw.ellipse(pool, (255, 200, 110, 26), pool.get_rect())
            screen.blit(pool, (bx - pr, by - pr // 2))
        pygame.draw.line(screen, pshade((150, 150, 158), night), (bx, by), (tx, ty),
                         max(1, int(sc * 0.14)))
        pygame.draw.circle(screen, pshade((255, 190, 90), night, 0.55), (int(tx), int(ty)),
                           max(2, int(sc * 0.16)))
        if mood["headlights"]:
            pygame.draw.circle(screen, pshade((255, 240, 190), night, 0.45), (int(tx), int(ty)),
                               max(1, int(sc * 0.08)))


def draw_headlight_beams_v2(pygame, screen, view, veh, spec, mood):
    if not mood.get("headlights"):
        return

    HL = (spec.wheelbase + 1.7) / 2.0
    HW = (spec.track_width + 0.36) / 2.0
    yaw = veh.yaw
    cyw, syw = math.cos(yaw), math.sin(yaw)
    side_x, side_y = -syw, cyw
    height = max(0.0, float(getattr(veh, "z", 0.0) - getattr(veh, "road_z", 0.0)))
    road_z = float(getattr(veh, "road_z", 0.0))

    nose_x = veh.x + cyw * HL * 0.96
    nose_y = veh.y + syw * HL * 0.96
    lamp_sep = HW * 0.50
    left_src = view.project(nose_x + side_x * lamp_sep,
                            nose_y + side_y * lamp_sep,
                            road_z + height + 0.44)
    right_src = view.project(nose_x - side_x * lamp_sep,
                             nose_y - side_y * lamp_sep,
                             road_z + height + 0.44)

    night = mood.get("name") == "night"
    beam_len = 30.0 if night else 25.0
    beam_half_width = 5.3 if night else 4.4
    far_x = nose_x + cyw * beam_len
    far_y = nose_y + syw * beam_len
    far_center = view.project(far_x, far_y, road_z)
    far_edge = view.project(far_x + side_x * beam_half_width,
                            far_y + side_y * beam_half_width,
                            road_z)
    far_half_width_px = math.hypot(far_edge[0] - far_center[0],
                                   far_edge[1] - far_center[1])

    draw_dual_headlight_beam(
        pygame, screen, left_src, right_src, far_center, far_half_width_px,
        intensity=1.16 if night else 0.74,
        source_radius_px=max(2.0, view.scale * 0.055),
        strips=48,
        blur_scale=0.48,
    )


# --------------------------------------------------------------------------- #
# car: reuse the v1 meshes, project them through the V2 camera
# --------------------------------------------------------------------------- #
def draw_car_v2(pygame, screen, view, veh, spec, mood, brake=0.0, dirt=0.0,
                vibration=(0.0, 0.0), fog_amount=0.0, allow_lod=False):
    from .carart import CAR_COLORS, get_car_mesh, add_wheels_to_mesh, _shade as csh

    name = getattr(spec, "name", "supra")
    base = CAR_COLORS.get(name, (198, 32, 38))
    HL = (spec.wheelbase + 1.7) / 2.0
    HW = (spec.track_width + 0.36) / 2.0
    vertices, faces = get_car_mesh(name, HL, HW)
    lod_distance = math.hypot(float(veh.x) - view.cx, float(veh.y) - view.cy)
    if allow_lod and (lod_distance > 44.0 or fog_amount > 0.32):
        # Distant roadside cars retain their chassis footprint, roof and color
        # but skip wheels and face sorting once atmospheric fog dominates.
        yaw = float(veh.yaw)
        cyw, syw = math.cos(yaw), math.sin(yaw)
        road_z = float(getattr(veh, "road_z", 0.0))

        def lod_point(lx, ly, lz=0.0):
            wx = veh.x + lx * cyw - ly * syw
            wy = veh.y + lx * syw + ly * cyw
            return view.project(wx, wy, road_z + lz)

        footprint = []
        for idxp in (0, 8, 18, 19, 17, 7, 29, 31, 30, 20):
            footprint.append(lod_point(*vertices[idxp]))
        shadow = [(x + 2, y + 3) for x, y in footprint]
        pygame.draw.polygon(screen, _mix_color(mood["shadow"], mood["fog"],
                                               fog_amount * 0.45), shadow)
        body_col = _mix_color(base, mood["fog"], fog_amount * 0.76)
        pygame.draw.polygon(screen, body_col, footprint)
        pygame.draw.lines(screen, csh(body_col, 0.62), True, footprint, 1)
        roof_h = 1.12 if name in ("lr4", "f150") else 0.82
        roof = [lod_point(HL * x, HW * y, roof_h)
                for x, y in ((0.35, 0.48), (0.35, -0.48),
                             (-0.38, -0.48), (-0.38, 0.48))]
        glass = _mix_color((40, 50, 65), mood["fog"], fog_amount * 0.78)
        pygame.draw.polygon(screen, glass, roof)
        return
    vertices, faces = add_wheels_to_mesh(vertices, faces, spec, veh)

    height = max(0.0, float(getattr(veh, "z", 0.0) - getattr(veh, "road_z", 0.0)))
    road_z = float(getattr(veh, "road_z", 0.0))
    yaw = veh.yaw
    cyw, syw = math.cos(yaw), math.sin(yaw)

    gain = getattr(spec, "body_roll_gain", 1.0)
    tilt_p = -np.clip(veh.ax, -20, 20) * 0.003 * gain + float(getattr(veh, "pitch", 0.0)) * 0.55
    tilt_r = -np.clip(veh.ay, -20, 20) * 0.003 * gain + float(getattr(veh, "roll", 0.0)) * 0.55

    # shadow glued to the ROAD (the point of the whole projector): while the
    # body lifts with veh.z, this stays at road_z — airborne gap reads instantly
    sh_pts = []
    for idxp in (0, 8, 18, 19, 17, 7, 29, 31, 30, 20):
        lx, ly, _ = vertices[idxp]
        wx = veh.x + lx * cyw - ly * syw
        wy = veh.y + lx * syw + ly * cyw
        sh_pts.append(view.project(wx, wy, road_z))
    grow = 1.0 + min(0.5, height * 0.10)
    cxs = sum(p[0] for p in sh_pts) / len(sh_pts)
    cys = sum(p[1] for p in sh_pts) / len(sh_pts)
    sh_pts = [((p[0] - cxs) * grow + cxs + 2, (p[1] - cys) * grow + cys + 3) for p in sh_pts]
    alpha = max(60, 150 - int(height * 14))
    mn_x = min(p[0] for p in sh_pts); mx_x = max(p[0] for p in sh_pts)
    mn_y = min(p[1] for p in sh_pts); mx_y = max(p[1] for p in sh_pts)
    if mx_x > mn_x and mx_y > mn_y and mx_x - mn_x < 4000:
        sh = pygame.Surface((int(mx_x - mn_x) + 4, int(mx_y - mn_y) + 4), pygame.SRCALPHA)
        pygame.draw.polygon(sh, (*mood["shadow"], alpha),
                            [(p[0] - mn_x + 2, p[1] - mn_y + 2) for p in sh_pts])
        screen.blit(sh, (mn_x - 2, mn_y - 2))

    # project mesh: body frame -> world -> V2 projector (rotation-aware)
    scr = []
    depth = []
    for lx, ly, lz in vertices:
        tx = lx + tilt_p * lz
        ty = ly + tilt_r * lz
        wx = veh.x + tx * cyw - ty * syw
        wy = veh.y + tx * syw + ty * cyw
        sx, sy = view.project(wx, wy, road_z + height + lz)
        _, dy = view.project(wx, wy, None)
        scr.append((int(sx + vibration[0]), int(sy + vibration[1])))
        depth.append(dy)

    night = 0.62 if mood["name"] == "night" else 1.0
    order = []
    for f_name, f_idx, (nx_b, ny_b, nz_b), ctype in faces:
        nx_w = nx_b * cyw - ny_b * syw
        ny_w = nx_b * syw + ny_b * cyw
        dot = nx_w * SUN[0] + ny_w * SUN[1] + nz_b * SUN[2]
        f = (0.46 + 0.62 * max(0.0, dot)) * night * mood.get("sun_strength", 1.0)
        if ctype == "body":
            col = csh(base, f)
        elif ctype == "glass":
            col = csh((40, 50, 65), f)
        elif ctype == "wheel_tread":
            col = csh((32, 32, 36), f)
        elif ctype == "wheel_rim":
            col = csh((140, 130, 105), f)
        elif ctype == "wheel_inner":
            col = csh((20, 20, 22), f)
        elif isinstance(ctype, tuple):
            col = csh(ctype, f)
        else:
            col = csh(base, f)
        if fog_amount > 0.0:
            col = _mix_color(col, mood.get("fog", col),
                             min(1.0, float(fog_amount)) * 0.76)
        d = sum(depth[i] for i in f_idx) / len(f_idx)
        order.append((d, [scr[i] for i in f_idx], col))
    order.sort(key=lambda o: o[0])
    lw = max(1, int(view.scale * 0.025))
    for _, pts, col in order:
        pygame.draw.polygon(screen, col, pts)
        pygame.draw.polygon(screen, csh(col, 0.72), pts, lw)

    def P(lx, ly, lz):
        """Project a body-local point through the tilt + V2 camera."""
        tx = lx + tilt_p * lz
        ty = ly + tilt_r * lz
        wx = veh.x + tx * cyw - ty * syw
        wy = veh.y + tx * syw + ty * cyw
        sx, sy = view.project(wx, wy, road_z + height + lz)
        return (int(sx + vibration[0]), int(sy + vibration[1]))

    # --- 787B identity pass (drawn over the faces, so it never z-fights) ---
    if name == "mazda787b":
        sc = view.scale
        # covered headlights: perspex fairings low on the nose
        lit = mood["headlights"]
        hl_col = (255, 250, 215) if lit else (168, 182, 192)
        for sgn in (1, -1):
            cover = [P(0.97 * HL, sgn * 0.26 * HW, 0.105),
                     P(0.87 * HL, sgn * 0.78 * HW, 0.195),
                     P(0.76 * HL, sgn * 0.70 * HW, 0.245),
                     P(0.87 * HL, sgn * 0.24 * HW, 0.165)]
            pygame.draw.polygon(screen, hl_col, cover)
            pygame.draw.polygon(screen, (10, 12, 14), cover, max(1, int(sc * 0.02)))
        # nose roundel with the 55
        if 2 * HL * sc > 36:
            rc = P(0.60 * HL, 0.0, 0.245)
            rr = max(3, int(sc * 0.26))
            pygame.draw.circle(screen, (242, 242, 234), rc, rr)
            pygame.draw.circle(screen, (30, 32, 34), rc, rr, max(1, int(sc * 0.02)))
            if rr >= 7:
                ink = (28, 30, 32)

                def five(cx0):
                    w = max(3, int(rr * 0.40))
                    h = max(5, int(rr * 0.85))
                    t = max(1, int(rr * 0.16))
                    x1, x2 = cx0 - w // 2, cx0 + w // 2
                    y1, y2 = rc[1] - h // 2, rc[1] + h // 2
                    ym = rc[1]
                    pygame.draw.line(screen, ink, (x1, y1), (x2, y1), t)
                    pygame.draw.line(screen, ink, (x1, y1), (x1, ym), t)
                    pygame.draw.line(screen, ink, (x1, ym), (x2, ym), t)
                    pygame.draw.line(screen, ink, (x2, ym), (x2, y2), t)
                    pygame.draw.line(screen, ink, (x1, y2), (x2, y2), t)

                five(rc[0] - int(rr * 0.30))
                five(rc[0] + int(rr * 0.34))
        # tail light bar + centre rain light
        bar = [P(-1.035 * HL, 0.82 * HW, 0.30), P(-1.035 * HL, -0.82 * HW, 0.30),
               P(-1.035 * HL, -0.82 * HW, 0.21), P(-1.035 * HL, 0.82 * HW, 0.21)]
        glow = int(120 + 130 * min(1.0, brake))
        pygame.draw.polygon(screen, (glow, 22, 20), bar)
        pygame.draw.circle(screen, (255, 90, 60) if brake > 0.1 else (150, 40, 30),
                           P(-1.04 * HL, 0.0, 0.255), max(2, int(sc * 0.05)))

    # Session-persistent grime after off-track excursions. Fixed body-space
    # flecks avoid random shimmer and remain purely a paint-layer effect.
    dirt = float(np.clip(dirt, 0.0, 1.0))
    if dirt > 0.03:
        flecks = ((-0.72, 0.68, 0.22), (-0.52, -0.72, 0.18),
                  (-0.18, 0.76, 0.16), (0.05, -0.70, 0.14),
                  (-0.88, 0.18, 0.25), (0.42, 0.66, 0.12),
                  (0.28, -0.62, 0.13), (-0.36, 0.12, 0.28))
        count = max(1, int(len(flecks) * dirt))
        mud = (88, 68, 42) if mood["name"] != "night" else (48, 43, 38)
        for lx_f, ly_f, z_f in flecks[:count]:
            fp = P(lx_f * HL, ly_f * HW, z_f)
            pygame.draw.circle(screen, mud, fp,
                               max(1, int(view.scale * (0.035 + dirt * 0.045))))

    # brake glow
    if brake > 0.1:
        for sgn in (1, -1):
            wx = veh.x - cyw * HL * 0.98 - syw * sgn * HW * 0.55
            wy = veh.y - syw * HL * 0.98 + cyw * sgn * HW * 0.55
            sx, sy = view.project(wx, wy, road_z + height + 0.6)
            pygame.draw.circle(screen, (255, 60, 48),
                               (int(sx + vibration[0]), int(sy + vibration[1])),
                               max(2, int(view.scale * 0.10)))


# --------------------------------------------------------------------------- #
# FX live in supra/fx.py (V2 section) — projected surface marks, particles,
# atmosphere grade, and speed tension. Imported here so the viewer stays lean.
# --------------------------------------------------------------------------- #
from .fx import (Skids, Puffs, Cinematic, SpeedTension,   # noqa: E402
                 FlybyDopplerFX, draw_exhaust_flame,
                 draw_audio_reactive_exhaust)


# --------------------------------------------------------------------------- #
# Render-only replay recorder
# --------------------------------------------------------------------------- #
class ReplaySystem:
    """Records drawn poses, then replays them while the live sim keeps moving.

    Frames contain only presentation state. Playback never writes into Vehicle,
    track, controller, sensors, lap timing, or audio state.
    """

    def __init__(self, max_frames=420):
        self.buffer = deque(maxlen=max_frames)
        self.frames = []
        self.index = 0.0
        self.playing = False
        self.kind = None
        self.camera = "quarter"
        self.pending = None
        self.pending_t = 0.0
        self.cooldown = 0.0
        self.just_cut = False
        self._rot0 = 0.0

    @staticmethod
    def _snapshot(veh):
        vals = {}
        for name, default in (
                ("x", 0.0), ("y", 0.0), ("yaw", 0.0), ("z", 0.0),
                ("road_z", 0.0), ("pitch", 0.0), ("roll", 0.0),
                ("ax", 0.0), ("ay", 0.0), ("steer_angle", 0.0),
                ("speed", 0.0), ("slip_angle", 0.0), ("rpm", 0.0),
                ("gear", 1), ("landing_g", 0.0), ("airborne", False)):
            value = getattr(veh, name, default)
            vals[name] = bool(value) if name == "airborne" else (
                int(value) if name == "gear" else float(value))
        vals["spec"] = veh.spec
        return vals

    def capture(self, veh, brake, mood_i, wet):
        self.buffer.append(dict(pose=self._snapshot(veh), brake=float(brake),
                                mood_i=int(mood_i), wet=float(wet)))

    def request(self, kind, delay=0.48):
        if self.playing or self.pending is not None or self.cooldown > 0.0:
            return False
        if len(self.buffer) < 100:
            return False
        self.pending = str(kind)
        self.pending_t = float(delay)
        return True

    def manual(self):
        if self.playing:
            self.playing = False
            self.cooldown = 1.0
            self.just_cut = True
            return False
        if len(self.buffer) < 75:
            return False
        self._begin("MANUAL REPLAY")
        return True

    def _begin(self, kind):
        count = min(len(self.buffer), 185)
        self.frames = list(self.buffer)[-count:]
        self.index = 0.0
        self.kind = kind
        upper = kind.upper()
        self.camera = ("side" if "DRIFT" in upper else
                       "helicopter" if "OFF" in upper else "quarter")
        first = self.frames[0]["pose"]
        self._rot0 = -math.pi / 2.0 - first["yaw"] + (
            math.pi / 2.0 if self.camera == "side" else 0.0)
        self.playing = True
        self.pending = None
        self.just_cut = True

    def update(self, dt):
        self.just_cut = False
        self.cooldown = max(0.0, self.cooldown - dt)
        if self.playing:
            self.index += dt * 60.0 * 0.66       # restrained broadcast slow motion
            if self.index >= len(self.frames) - 1:
                self.playing = False
                self.cooldown = 7.5
                self.just_cut = True
            return
        if self.pending is not None:
            self.pending_t -= dt
            if self.pending_t <= 0.0:
                self._begin(self.pending)

    def current(self):
        if not self.playing or not self.frames:
            return None
        return self.frames[min(len(self.frames) - 1, int(self.index))]

    def pose(self):
        frame = self.current()
        return SimpleNamespace(**frame["pose"]) if frame else None

    def apply_view(self, view, dt):
        view.cam_override = None
        view.rot_override = None
        view.rot_offset = 0.0
        view.anchor_override = None
        view.zoom_mult = 1.0
        view.lead_mult = 1.0
        if self.camera == "side":
            view.rot_override = self._rot0
            view.zoom_mult = 1.18
            view.anchor_override = 0.52
        elif self.camera == "helicopter":
            self._rot0 += 0.045 * dt
            view.rot_override = self._rot0
            view.zoom_mult = 0.62
            view.anchor_override = 0.50
        else:
            view.zoom_mult = 1.42
            view.anchor_override = 0.59
            view.rot_offset = 0.48
            view.lead_mult = 0.34
        if self.just_cut:
            view.cut = True

    def label(self):
        return f"REPLAY · {self.kind}" if self.playing else None


# --------------------------------------------------------------------------- #
# Toggleable AI intent layer
# --------------------------------------------------------------------------- #
class AIVision:
    """One-switch augmented broadcast layer built from policy read-only data."""

    def __init__(self, pygame, W, H, enabled=False):
        self.W, self.H = W, H
        self.enabled = bool(enabled)
        self.path = []
        self.path3 = []
        self._path_token = None
        self.brake_zone = []
        self.apex = None
        self.confidence = 0.5
        self.uncertainty = 0.5
        self.intent = (0.0, 0.0, 0.0)
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 10)
        self.font_bold = pygame.font.SysFont("menlo,consolas,monospace", 12, bold=True)

    def toggle(self):
        self.enabled = not self.enabled
        return self.enabled

    def update(self, agent, veh, trk):
        if not self.enabled or agent is None:
            return
        # The agent already amortises this copied-vehicle rollout across frames.
        self.path = agent.step_prediction(veh, trk, budget=14, seconds=1.08)
        token = id(self.path)
        if token != self._path_token:
            self.path3 = [(x, y, self._path_z(trk, x, y)) for x, y in self.path]
            self._path_token = token
        self.confidence = float(np.clip(agent.confidence(), 0.0, 1.0))
        self.uncertainty = float(np.clip(agent.uncertainty(), 0.0, 1.0))
        act = np.asarray(getattr(agent, "last", {}).get("act", (0.0, 0.0)), float)
        steer = float(np.clip(act[0] if len(act) else 0.0, -1.0, 1.0))
        longitudinal = float(np.clip(act[1] if len(act) > 1 else 0.0, -1.0, 1.0))
        self.intent = (steer, max(0.0, longitudinal), max(0.0, -longitudinal))

        n = len(trk.center)
        spacing = max(0.05, float(trk.length / n))
        i0 = trk.nearest(veh.x, veh.y)
        lo = max(2, int(18.0 / spacing))
        hi = max(lo + 1, int(145.0 / spacing))
        vref = getattr(trk, "fable_vref", None)
        best = None
        for q in range(lo, hi):
            i = (i0 + q) % n
            curve = abs(float(trk.curvature[i]))
            speed_risk = (max(0.0, veh.speed - float(vref[i])) / 24.0
                          if vref is not None else 0.0)
            risk = curve * 58.0 + speed_risk
            if best is None or risk > best[0]:
                best = (risk, q, i)

        self.brake_zone = []
        self.apex = None
        if best is not None and best[0] > 0.38:
            _, q_apex, i_apex = best
            braking_dist = min(62.0, max(22.0, veh.speed * 0.68))
            q0 = max(2, q_apex - int(braking_dist / spacing))
            q1 = max(q0 + 1, q_apex - max(1, int(7.0 / spacing)))
            step = max(1, int(4.5 / spacing))
            self.brake_zone = [(i0 + q) % n for q in range(q0, q1 + 1, step)]

            signed = float(trk.curvature[i_apex])
            inside = 1.0 if signed >= 0.0 else -1.0
            half = (float(trk.half_width[i_apex])
                    if getattr(trk, "half_width", None) is not None else float(trk.half))
            pos = np.asarray(trk.center[i_apex], float) + np.asarray(trk.normal[i_apex], float) * inside * half * 0.54
            z = float(trk.z[i_apex]) if getattr(trk, "z", None) is not None else 0.0
            self.apex = (float(pos[0]), float(pos[1]), z)

    @staticmethod
    def _path_z(trk, x, y):
        i = trk.nearest(x, y)
        return float(trk.z[i]) + 0.07 if getattr(trk, "z", None) is not None else 0.07

    def draw_world(self, pygame, screen, view, trk, mood):
        if not self.enabled:
            return
        conf_col = _mix_color((74, 190, 255), (255, 211, 92), self.confidence)

        # Predicted policy rollout: a tapered dual-stroke ribbon, grounded to
        # the rendered road rather than floating in screen space.
        if len(self.path3) > 1:
            pts = [view.project(x, y, z) for x, y, z in self.path3]
            for i in range(len(pts) - 1):
                fade = 1.0 - i / max(1, len(pts) - 1)
                outer = _mix_color(mood["road"], (20, 35, 44), 0.70 * fade)
                inner = _mix_color(mood["road"], conf_col, 0.88 * fade)
                width = max(2, int(view.scale * (0.34 * fade + 0.09)))
                pygame.draw.line(screen, outer, pts[i], pts[i + 1], width + 3)
                pygame.draw.line(screen, inner, pts[i], pts[i + 1], width)

        # Upcoming brake zone: transverse amber/red bands leading into the apex.
        for q, i in enumerate(self.brake_zone):
            c = np.asarray(trk.center[i], float)
            nrm = np.asarray(trk.normal[i], float)
            half = (float(trk.half_width[i]) if getattr(trk, "half_width", None) is not None
                    else float(trk.half))
            z = float(trk.z[i]) + 0.085 if getattr(trk, "z", None) is not None else 0.085
            a = view.project(*(c + nrm * half * 0.53), z)
            b = view.project(*(c - nrm * half * 0.53), z)
            t = q / max(1, len(self.brake_zone) - 1)
            col = _mix_color((255, 188, 58), (255, 72, 58), t)
            pygame.draw.line(screen, col, a, b, max(1, int(view.scale * 0.075)))

        if self.apex is not None:
            sx, sy = view.project(*self.apex)
            r = max(7, int(view.scale * 0.48))
            col = (255, 216, 92)
            pygame.draw.polygon(screen, col,
                                ((sx, sy - r), (sx + r, sy), (sx, sy + r), (sx - r, sy)), 2)
            pygame.draw.circle(screen, (255, 245, 188), (int(sx), int(sy)), max(2, r // 4))
            label = self.font.render("APEX", True, col)
            screen.blit(label, (sx + r + 4, sy - label.get_height() // 2))

    def draw_hud(self, pygame, screen):
        if not self.enabled:
            return
        x, y, w, h = 16, 226, 190, 128
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (7, 12, 16, 205), panel.get_rect(), border_radius=11)
        pygame.draw.rect(panel, (76, 190, 226, 92), panel.get_rect(), 1, border_radius=11)
        panel.blit(self.font_bold.render("AI VISION", True, (118, 218, 244)), (12, 10))

        def bar(label, value, yy, color, bipolar=False):
            panel.blit(self.font.render(label, True, (116, 142, 150)), (12, yy))
            bx, by, bw, bh = 72, yy + 1, 102, 7
            pygame.draw.rect(panel, (30, 40, 44), (bx, by, bw, bh), border_radius=3)
            if bipolar:
                pygame.draw.line(panel, (88, 98, 102), (bx + bw // 2, by - 1),
                                 (bx + bw // 2, by + bh + 1), 1)
                fill = int((bw * 0.5) * abs(value))
                fx = bx + bw // 2 - fill if value < 0 else bx + bw // 2
            else:
                fill = int(bw * max(0.0, min(1.0, value)))
                fx = bx
            if fill > 0:
                pygame.draw.rect(panel, color, (fx, by, fill, bh), border_radius=3)

        bar("CONF", self.confidence, 35, (255, 205, 92))
        bar("UNCERT", self.uncertainty, 52, (255, 112, 92))
        bar("STEER", self.intent[0], 76, (92, 198, 240), bipolar=True)
        bar("THROT", self.intent[1], 93, (108, 224, 146))
        bar("BRAKE", self.intent[2], 110, (255, 102, 82))
        screen.blit(panel, (x, y))


# --------------------------------------------------------------------------- #
# Weather, persistent session patina, and optional high-quality post FX
# --------------------------------------------------------------------------- #
class WeatherSystem:
    MODES = (
        dict(name="clear", label="CLEAR", wet=0.0, rain=0, tint=(0, 0, 0, 0)),
        dict(name="overcast", label="OVERCAST", wet=0.0, rain=0, tint=(74, 82, 91, 28)),
        dict(name="rain", label="RAIN", wet=0.72, rain=92, tint=(48, 67, 84, 28)),
        dict(name="storm", label="STORM", wet=1.0, rain=180, tint=(24, 37, 58, 48)),
        dict(name="mist", label="FOREST MIST", wet=0.18, rain=0, tint=(118, 132, 121, 22)),
    )

    def __init__(self, W, H, rng):
        self.W, self.H, self.rng = W, H, rng
        self.index = 0
        self.flash = 0.0
        self.drops = [[rng.uniform(0, W), rng.uniform(0, H),
                       rng.uniform(620, 1120), rng.uniform(12, 28)] for _ in range(180)]
        self.mist = [[rng.uniform(-0.2, 1.0), rng.uniform(0.15, 0.78),
                      rng.uniform(0.12, 0.28), rng.uniform(0.018, 0.045)] for _ in range(8)]

    @property
    def mode(self):
        return self.MODES[self.index]

    def cycle(self):
        self.index = (self.index + 1) % len(self.MODES)
        return self.mode["label"]

    def update(self, dt):
        self.flash = max(0.0, self.flash - dt * 2.8)
        if self.mode["name"] == "storm" and self.flash <= 0.0:
            if self.rng.random() < dt * 0.07:
                self.flash = self.rng.uniform(0.62, 1.0)
        count = int(self.mode["rain"])
        for q in self.drops[:count]:
            q[0] -= q[2] * dt * 0.10
            q[1] += q[2] * dt
            if q[1] > self.H + 35 or q[0] < -35:
                q[0] = self.rng.uniform(0, self.W + 120)
                q[1] = self.rng.uniform(-self.H * 0.25, -10)
        for q in self.mist:
            q[0] += q[3] * dt
            if q[0] > 1.2:
                q[0] = -0.25

    def draw(self, pygame, screen):
        mode = self.mode
        tint = mode["tint"]
        if tint[3] > 0:
            veil = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            veil.fill(tint)
            screen.blit(veil, (0, 0))

        if mode["name"] in ("mist", "rain", "storm"):
            fog = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            strength = 18 if mode["name"] == "mist" else (9 if mode["name"] == "rain" else 13)
            for x, y, size, _ in self.mist:
                rect = pygame.Rect(int((x - size) * self.W), int((y - size * 0.35) * self.H),
                                   int(size * self.W * 2.0), int(size * self.H * 0.70))
                pygame.draw.ellipse(fog, (174, 188, 184, strength), rect)
            screen.blit(fog, (0, 0))

        count = int(mode["rain"])
        if count:
            rain = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            alpha = 78 if mode["name"] == "rain" else 112
            for x, y, speed, length in self.drops[:count]:
                slant = length * 0.20
                pygame.draw.line(rain, (188, 214, 230, alpha),
                                 (x, y), (x - slant, y + length), 1)
            screen.blit(rain, (0, 0))

        if self.flash > 0.0:
            lightning = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
            lightning.fill((208, 220, 255, int(115 * self.flash)))
            screen.blit(lightning, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


class SessionVisualState:
    """Session-long visual memory; never consulted by simulation logic."""

    def __init__(self, n):
        self.rubber = np.zeros(n, dtype=np.float32)
        self.wetness = 0.0
        self.car_dirt = 0.0
        self.ground_marks = []
        self._last_ground = None

    def update(self, veh, fr, i_near, dt, wet_target):
        self.wetness += (float(wet_target) - self.wetness) * min(1.0, dt * 0.34)
        slide = abs(math.degrees(getattr(veh, "slip_angle", 0.0)))
        if not fr["off_track"] and veh.speed > 7.0:
            lay = max(0.0, (slide - 6.0) / 34.0) * dt * 0.60
            if lay > 0.0:
                for d, falloff in ((0, 1.0), (-1, 0.72), (1, 0.72), (-2, 0.35), (2, 0.35)):
                    j = (i_near + d) % len(self.rubber)
                    self.rubber[j] = min(1.0, self.rubber[j] + lay * falloff)

        if fr["off_track"] and veh.speed > 4.0 and not veh.airborne:
            self.car_dirt = min(1.0, self.car_dirt + dt * (0.035 + veh.speed * 0.0022))
            point = (float(veh.x), float(veh.y), float(fr["z"]),
                     min(1.0, 0.25 + veh.speed / 55.0))
            if (self._last_ground is None or
                    (point[0] - self._last_ground[0]) ** 2 +
                    (point[1] - self._last_ground[1]) ** 2 > 0.22):
                self.ground_marks.append(point)
                self._last_ground = point
                if len(self.ground_marks) > 1600:
                    del self.ground_marks[:len(self.ground_marks) - 1600]
        else:
            self._last_ground = None
            self.car_dirt = max(0.0, self.car_dirt - dt * 0.0008)

    def draw_ground(self, pygame, screen, view, mood):
        for x, y, z, strength in self.ground_marks:
            if abs(x - view.cx) > 180 or abs(y - view.cy) > 180:
                continue
            sx, sy = view.project(x, y, z + 0.015)
            r = max(1, int(view.scale * (0.10 + strength * 0.08)))
            col = _mix_color(mood["ground"], (54, 42, 29), 0.48 * strength)
            pygame.draw.ellipse(screen, col,
                                (int(sx - r * 1.8), int(sy - r * 0.6),
                                 max(2, int(r * 3.6)), max(2, int(r * 1.2))))


class PostFX:
    """Optional higher-cost bloom and high-speed temporal persistence."""

    def __init__(self, pygame, W, H, enabled=False):
        self.W, self.H = W, H
        self.enabled = bool(enabled)
        self.previous = None
        self.bloom = None
        self.frame = 0

    def toggle(self):
        self.enabled = not self.enabled
        self.previous = None
        return self.enabled

    def apply(self, pygame, screen, speed, impact=0.0):
        raw = screen.copy()
        if not self.enabled:
            self.previous = None
            return
        self.frame += 1

        if self.previous is not None and speed > 48.0:
            ghost = self.previous.copy()
            ghost.set_alpha(int(min(42, 10 + (speed - 48.0) * 0.8)))
            screen.blit(ghost, (0, 1))

        if self.frame % 2 == 1 or self.bloom is None:
            sw, sh = max(8, self.W // 6), max(8, self.H // 6)
            small = pygame.transform.smoothscale(raw, (sw, sh))
            arr = pygame.surfarray.pixels3d(small)
            lum = arr.max(axis=2).astype(np.float32)
            gain = np.clip((lum - 132.0) / 92.0, 0.0, 1.0)[:, :, None]
            arr[:, :, :] = np.clip(arr.astype(np.float32) * gain * 0.72, 0, 255).astype(np.uint8)
            del arr
            self.bloom = pygame.transform.smoothscale(small, (self.W, self.H))
            self.bloom.set_alpha(88)
        screen.blit(self.bloom, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

        if impact > 0.15:
            split = max(1, int(2 + impact * 4))
            red = raw.copy(); blue = raw.copy()
            red.fill((255, 48, 48), special_flags=pygame.BLEND_RGB_MULT)
            blue.fill((48, 90, 255), special_flags=pygame.BLEND_RGB_MULT)
            red.set_alpha(int(22 * impact)); blue.set_alpha(int(22 * impact))
            screen.blit(red, (split, 0), special_flags=pygame.BLEND_RGB_ADD)
            screen.blit(blue, (-split, 0), special_flags=pygame.BLEND_RGB_ADD)
        self.previous = raw


class AudioReactiveVisuals:
    """Render-only engine/event telemetry shared by every audio-linked effect.

    This observes the same public vehicle values already used by the HUD and
    sound mixer. It never writes to Vehicle or samples/modifies the audio stream.
    """

    def __init__(self):
        self.time = 0.0
        self.engine_phase = 0.0
        self.rpm_norm = 0.0
        self.load = 0.0
        self.harmonic = 0.0
        self.resonance = 0.0
        self.shift = 0.0
        self.backfire = 0.0
        self.limiter_active = False
        self.limiter_gate = 0.0
        self.limiter_tick = 0.0
        self.limiter_just_cut = False
        self.shift_wave = 0.0
        self.waveform = deque(maxlen=38)
        self.closing_speed = 0.0
        self.radial_velocity = 0.0
        self.camera_speed = 0.0
        self.camera_distance = 0.0
        self.doppler_strength = 0.0
        self.flyby_pulse = 0.0
        self.flyby_just_passed = False
        self.events = deque(maxlen=48)
        self._prev_gear = None
        self._prev_rpm = 0.0
        self._prev_cam = None
        self._limiter_prev = False
        self._was_approaching = False

    def _event(self, kind, **data):
        self.events.append(dict(kind=kind, time=self.time, **data))

    def update(self, veh, throttle, brake, dt, camera_pos, fixed_camera=False):
        dt = max(1e-5, float(dt))
        self.time += dt
        rpm = max(0.0, float(getattr(veh, "rpm", 0.0)))
        cutoff = max(1.0, float(getattr(veh.spec, "cutoff_rpm", 8000.0)))
        redline = max(1.0, float(getattr(veh.spec, "redline_rpm", cutoff * 0.9)))
        self.rpm_norm = float(np.clip(rpm / cutoff, 0.0, 1.25))
        rotations = rpm / 60.0
        self.engine_phase = (self.engine_phase + math.tau * rotations * dt) % math.tau

        # A stable harmonic signal replaces random noise as the basis for later
        # exhaust, body and camera responses.
        raw_harmonic = (math.sin(self.engine_phase) * 0.58 +
                        math.sin(self.engine_phase * 2.0 + 0.35) * 0.27 +
                        math.sin(self.engine_phase * 4.0 + 1.1) * 0.15)
        target_load = float(np.clip(throttle, 0.0, 1.0)) * (0.28 + 0.72 * min(1.0, self.rpm_norm))
        self.load += (target_load - self.load) * min(1.0, dt * 10.0)
        self.harmonic = raw_harmonic * self.load

        # Narrow mechanical resonance bands; the smoothed envelope is exposed
        # for step 3 rather than immediately shaking the camera here.
        resonance_raw = 0.0
        for center, width, gain in ((0.43, 0.055, 0.34),
                                    (0.68, 0.045, 0.55),
                                    (0.87, 0.035, 0.82)):
            resonance_raw += math.exp(-0.5 * ((self.rpm_norm - center) / width) ** 2) * gain
        resonance_raw = min(1.0, resonance_raw) * self.load
        self.resonance += (resonance_raw - self.resonance) * min(1.0, dt * 8.0)

        self.shift = max(0.0, self.shift - dt * 4.4)
        self.backfire = max(0.0, self.backfire - dt * 7.2)
        self.shift_wave = max(0.0, self.shift_wave - dt * 1.55)
        self.limiter_tick = max(0.0, self.limiter_tick - dt * 12.0)
        self.flyby_pulse = max(0.0, self.flyby_pulse - dt * 2.8)
        self.limiter_just_cut = False
        self.flyby_just_passed = False
        self.waveform.append(self.rpm_norm)
        gear = int(getattr(veh, "gear", 1))
        if self._prev_gear is None:
            self._prev_gear = gear
        elif gear != self._prev_gear:
            direction = "up" if gear > self._prev_gear else "down"
            rpm_delta = rpm - self._prev_rpm
            self.shift = 1.0
            self.shift_wave = 1.0
            self._event("shift", direction=direction, old_gear=self._prev_gear,
                        new_gear=gear, rpm_delta=float(rpm_delta), speed=float(veh.speed))
            if direction == "down" and veh.speed > 5.0:
                self.backfire = 1.0
                self._event("backfire", gear=gear, rpm=rpm, speed=float(veh.speed))
            self._prev_gear = gear

        self.limiter_active = bool(rpm >= cutoff * 0.965 and throttle > 0.42)
        self.limiter_gate = (1.0 if self.limiter_active and
                             math.sin(self.time * math.tau * 10.5) > -0.05 else 0.0)
        gate_rising = self.limiter_gate > 0.5 and getattr(self, "_gate_prev", 0.0) <= 0.5
        if gate_rising:
            self.limiter_tick = 1.0
            self.limiter_just_cut = True
            self._event("limiter_cut", rpm=rpm, gear=gear)
        self._gate_prev = self.limiter_gate
        if self.limiter_active and not self._limiter_prev:
            self._event("limiter", rpm=rpm, gear=gear)
        self._limiter_prev = self.limiter_active

        # Camera-relative radial velocity is centralized here for the later
        # trackside flyby/doppler visual pass.
        cam_x, cam_y = float(camera_pos[0]), float(camera_pos[1])
        if self._prev_cam is None:
            cam_vx = cam_vy = 0.0
        else:
            cam_vx = (cam_x - self._prev_cam[0]) / dt
            cam_vy = (cam_y - self._prev_cam[1]) / dt
        self._prev_cam = (cam_x, cam_y)
        self.camera_speed = math.hypot(cam_vx, cam_vy)
        dx, dy = float(veh.x) - cam_x, float(veh.y) - cam_y
        dist = max(1e-5, math.hypot(dx, dy))
        self.camera_distance = dist
        ux, uy = dx / dist, dy / dist
        car_vx = float(veh.speed) * math.cos(float(veh.yaw))
        car_vy = float(veh.speed) * math.sin(float(veh.yaw))
        self.radial_velocity = (car_vx - cam_vx) * ux + (car_vy - cam_vy) * uy
        self.closing_speed = -self.radial_velocity
        fixed_camera = bool(fixed_camera)
        target_doppler = (min(1.0, abs(self.radial_velocity) / 72.0) *
                          min(1.0, float(veh.speed) / 48.0)
                          if fixed_camera and dist < 150.0 else 0.0)
        self.doppler_strength += (target_doppler - self.doppler_strength) * min(1.0, dt * 9.0)
        approaching = fixed_camera and self.closing_speed > 1.5
        if (fixed_camera and self._was_approaching and not approaching and
                self.closing_speed < -1.5 and dist < 42.0 and veh.speed > 24.0):
            self.flyby_pulse = 1.0
            self.flyby_just_passed = True
            self._event("flyby_pass", speed=float(veh.speed), distance=dist,
                        radial_velocity=self.radial_velocity)
        self._was_approaching = approaching
        self._prev_rpm = rpm

    def body_vibration(self):
        amp = self.resonance * (0.28 + self.load * 0.72)
        return (self.harmonic * amp * 1.15,
                math.sin(self.engine_phase * 1.5 + 0.8) * amp * 0.72)

    def camera_vibration(self):
        amp = self.resonance * 0.34 + self.limiter_gate * 0.16
        return (math.sin(self.engine_phase * 0.52) * amp,
                math.sin(self.engine_phase * 0.73 + 1.2) * amp * 0.72)

    def latest(self, kind):
        for event in reversed(self.events):
            if event["kind"] == kind:
                return event
        return None


# --------------------------------------------------------------------------- #
# Camera director — auto-cuts, plus manual shot locks on number keys
# --------------------------------------------------------------------------- #
class Director:
    """Race-broadcast shot caller. Owns the View's override hooks; cuts
    between chase, low rear-quarter, locked side pan, drone/helicopter, bumper,
    apex and fixed trackside cameras. Auto selection reads speed, slip, air and
    upcoming curvature; number keys keep the five original manual families."""

    SHOTS = ("chase", "quarter", "side", "drone", "trackside",
             "bumper", "helicopter", "apex")
    WEIGHTS = (0.19, 0.15, 0.11, 0.09, 0.14, 0.10, 0.10, 0.12)
    MANUAL_SIDE_DUR = 8.0

    def __init__(self, trk, geom, rng):
        self.trk, self.geom, self.rng = trk, geom, rng
        self.on = False
        self.mode = None                    # None | "auto" | "man"
        self.shot = "chase"
        self.t = 0.0
        self.dur = 8.0
        self._rot0 = 0.0
        self._point = None
        self._i_cam = 0

    def _clear(self, view):
        view.cam_override = None
        view.rot_override = None
        view.rot_offset = 0.0
        view.anchor_override = None
        view.zoom_mult = 1.0
        view.lead_mult = 1.0

    def toggle(self, view):
        if self.mode == "auto":
            self.disable(view)
        else:
            self.start_auto(view)
        return self.on

    def start_auto(self, view):
        self.on = True
        self.mode = "auto"
        self._clear(view)
        self.shot = "chase"
        self.t, self.dur = 0.0, 6.0
        view.mode = "chase"
        view.cut = True
        return self.on

    def disable(self, view):
        self.on = False
        self.mode = None
        self._clear(view)
        view.cut = True
        return self.on

    def select_manual(self, view, veh, i_car, shot):
        if shot not in self.SHOTS:
            return
        self.on = True
        self.mode = "man"
        view.mode = "chase"
        self._start_shot(view, veh, i_car, shot, manual=True)

    def label(self):
        if not self.on:
            return None
        prefix = "MAN" if self.mode == "man" else "AUTO"
        return f"{prefix} {self.shot}"

    def _heading_at(self, i):
        tang = self.trk.tangent[i]
        return math.atan2(tang[1], tang[0])

    def _start_shot(self, view, veh, i_car, shot, manual=False):
        self.shot = shot
        self.t = 0.0
        self.dur = self.MANUAL_SIDE_DUR if manual and shot == "side" else 8.0
        self._clear(view)
        view.cut = True                    # broadcast cut, not a swoop

        if shot == "side":
            # lock the rotation so the car crosses a steady frame
            self._rot0 = -math.pi / 2.0 - veh.yaw + math.copysign(
                math.pi / 2.0, self.rng.random() - 0.5)
        elif shot in ("drone", "helicopter"):
            self._rot0 = -math.pi / 2.0 - veh.yaw
        elif shot in ("trackside", "apex"):
            n = self.geom.n
            if shot == "apex":
                lo = max(1, int(35.0 / self.geom.spacing))
                hi = max(lo + 1, int(210.0 / self.geom.spacing))
                choices = [(i_car + q) % n for q in range(lo, hi)]
                self._i_cam = max(choices, key=lambda q: self.geom.curvature[q])
            else:
                ahead = int(110.0 / self.geom.spacing)
                self._i_cam = (i_car + ahead) % n
            c = self.geom.center[self._i_cam]
            nrm = self.trk.normal[self._i_cam]
            if shot == "apex":
                side = -1.0 if self.geom.signed_curvature[self._i_cam] >= 0.0 else 1.0
            else:
                side = 1.0 if self.rng.random() < 0.5 else -1.0
            half = float(self.geom.half[self._i_cam])
            off = half + (3.0 if shot == "apex" else 2.0)
            self._point = (float(c[0] + nrm[0] * side * off),
                           float(c[1] + nrm[1] * side * off))
            self._rot0 = -math.pi / 2.0 - self._heading_at(self._i_cam)
            self.dur = 9999.0 if manual else (10.0 if shot == "apex" else 9.0)

    def _pick(self, view, veh, i_car):
        prev = self.shot
        weights = dict(zip(self.SHOTS, self.WEIGHTS))
        slide = abs(math.degrees(getattr(veh, "slip_angle", 0.0)))
        n = self.geom.n
        lo = max(1, int(35.0 / self.geom.spacing))
        hi = max(lo + 1, int(180.0 / self.geom.spacing))
        upcoming = max(self.geom.curvature[(i_car + q) % n] for q in range(lo, hi))
        if getattr(veh, "airborne", False):
            weights.update(quarter=0.42, helicopter=0.28, side=0.12,
                           bumper=0.01, apex=0.03)
        elif slide > 18.0:
            weights.update(side=0.38, quarter=0.31, helicopter=0.14,
                           bumper=0.02, trackside=0.07)
        elif upcoming > 0.014:
            weights.update(apex=0.34, side=0.17, quarter=0.17,
                           trackside=0.16, bumper=0.04)
        elif veh.speed > 55.0:
            weights.update(bumper=0.27, trackside=0.23, helicopter=0.18,
                           chase=0.13, drone=0.06)
        while True:
            r = self.rng.random() * sum(weights.values())
            acc = 0.0
            for s in self.SHOTS:
                w = weights[s]
                acc += w
                if r <= acc:
                    shot = s
                    break
            if shot != prev:
                break
        self._start_shot(view, veh, i_car, shot)
        if shot not in ("trackside", "apex"):
            self.dur = self.rng.uniform(6.0, 11.0)

    def _trackside_done(self, veh, i_car):
        if self._point is None:
            return True
        n = self.geom.n
        rel = ((i_car - self._i_cam) % n) * self.geom.spacing
        if 20.0 < rel < 400.0:             # car has passed the camera
            return True
        dx = veh.x - self._point[0]
        dy = veh.y - self._point[1]
        return math.hypot(dx, dy) > 320.0  # never coming — bail

    def _apply(self, view, dt):
        s = self.shot
        if s == "chase":
            pass                           # classic chase defaults
        elif s == "quarter":
            # low rear-quarter: closer, car low in frame, twisted a touch
            view.zoom_mult = 1.35
            view.anchor_override = 0.58
            view.rot_offset = 0.42
            view.lead_mult = 0.45
        elif s == "side":
            view.rot_override = self._rot0
            view.zoom_mult = 1.12
            view.anchor_override = 0.52
        elif s == "drone":
            self._rot0 += 0.035 * dt       # slow orbital drift
            view.rot_override = self._rot0
            view.zoom_mult = 0.52
            view.anchor_override = 0.50
        elif s == "helicopter":
            self._rot0 += 0.020 * dt
            view.rot_override = self._rot0
            view.zoom_mult = 0.68
            view.anchor_override = 0.48
            view.lead_mult = 0.72
        elif s == "bumper":
            view.zoom_mult = 1.72
            view.anchor_override = 0.74
            view.lead_mult = 0.10
            view.rot_offset = 0.05
        elif s in ("trackside", "apex"):
            view.cam_override = self._point
            view.rot_override = self._rot0
            view.zoom_mult = 1.34 if s == "apex" else 1.18
            view.anchor_override = 0.50

    def update(self, view, veh, i_car, dt):
        if not self.on:
            return
        self.t += dt

        if self.mode == "auto":
            done = self.t >= self.dur
            protected = (getattr(veh, "airborne", False) or
                         abs(math.degrees(getattr(veh, "slip_angle", 0.0))) > 22.0)
            if protected and self.t < self.dur + 4.0:
                done = False               # never cut out of the middle of the moment
            if self.shot in ("trackside", "apex") and self._trackside_done(veh, i_car):
                done = True
            if done:
                self._pick(view, veh, i_car)
        elif self.mode == "man":
            if self.shot in ("trackside", "apex") and self._trackside_done(veh, i_car):
                self._start_shot(view, veh, i_car, "trackside", manual=True)
            elif self.shot == "side" and self.t >= self.dur:
                self._start_shot(view, veh, i_car, "side", manual=True)

        self._apply(view, dt)


# --------------------------------------------------------------------------- #
# HUD — Fable-styled: amber monospace, analog tach dial, lap board, minimap
# --------------------------------------------------------------------------- #
AMBER = (255, 205, 120)
AMBER_HI = (255, 226, 170)


def _fmt_lap(t):
    if t is None:
        return "--:--.--"
    m = int(t // 60)
    return f"{m}:{t - m * 60:05.2f}"


def _rot_points(points, angle, cx, cy, scale=1.0):
    ca, sa = math.cos(angle), math.sin(angle)
    out = []
    for x, y in points:
        out.append((cx + (x * ca - y * sa) * scale,
                    cy + (x * sa + y * ca) * scale))
    return out


class PerformanceMonitor:
    """Rolling delivered-frame telemetry and optional unobtrusive HUD."""

    def __init__(self, pygame, target_fps):
        self.target_fps = int(target_fps)
        self.samples = deque(maxlen=600)
        self.step_samples = deque(maxlen=600)
        self.visible = False
        self.font = pygame.font.SysFont("menlo,consolas,monospace", 11, bold=True)

    def update(self, dt, steps):
        if 0.0 < dt < 0.5:
            self.samples.append(float(dt))
            self.step_samples.append(int(steps))

    def metrics(self):
        if not self.samples:
            return 0.0, 0.0, 0.0, 0
        recent = list(self.samples)[-240:]
        avg = sum(recent) / len(recent)
        sorted_dt = sorted(recent)
        p99 = sorted_dt[min(len(sorted_dt) - 1,
                            max(0, int(len(sorted_dt) * 0.99)))]
        return 1.0 / avg, avg * 1000.0, 1.0 / max(p99, 1e-6), max(self.step_samples or (0,))

    def draw(self, pygame, screen, quality_label):
        if not self.visible:
            return
        fps, ms, low, max_steps = self.metrics()
        text = (f"{fps:5.1f} FPS  {ms:4.1f} ms  1% {low:4.0f}  "
                f"SIM×{max_steps}  FX {quality_label}")
        label = self.font.render(text, True, (216, 228, 220))
        rect = label.get_rect(midtop=(screen.get_width() // 2, 12)).inflate(16, 9)
        panel = pygame.Surface(rect.size, pygame.SRCALPHA)
        panel.fill((12, 16, 18, 190))
        screen.blit(panel, rect)
        screen.blit(label, label.get_rect(center=rect.center))


class PresentationGovernor:
    """Protect motion cadence by trimming decorative FX before geometry."""

    LABELS = ("RESCUE", "BALANCED", "FULL")

    def __init__(self, target_fps):
        self.target_dt = 1.0 / max(30.0, float(target_fps))
        self.quality = 2
        self.elapsed = 0.0
        self.bad_time = 0.0
        self.good_time = 0.0

    @property
    def label(self):
        return self.LABELS[self.quality]

    def update(self, dt):
        dt = min(0.25, max(0.0, float(dt)))
        self.elapsed += dt
        if self.elapsed < 2.0:              # ignore window/display warm-up
            return
        if dt > self.target_dt * 1.28:
            self.bad_time += dt
            self.good_time = 0.0
        else:
            self.bad_time = max(0.0, self.bad_time - dt * 0.45)
            if dt <= self.target_dt * 1.10:
                self.good_time += dt
            else:
                self.good_time = 0.0
        if self.bad_time > 0.70 and self.quality > 0:
            self.quality -= 1
            self.bad_time = self.good_time = 0.0
        elif self.good_time > 5.0 and self.quality < 2:
            self.quality += 1
            self.bad_time = self.good_time = 0.0


class Hud:
    def __init__(self, pygame, W, H, trk, geom):
        self.W, self.H = W, H
        self.f_big = pygame.font.SysFont("menlo,consolas,monospace", 44, bold=True)
        self.f_med = pygame.font.SysFont("menlo,consolas,monospace", 19, bold=True)
        self.f_sml = pygame.font.SysFont("menlo,consolas,monospace", 12)
        self.f_tin = pygame.font.SysFont("menlo,consolas,monospace", 10)
        # minimap: prerender the path once, elevation-tinted
        mw, mh = 210, 158
        self.map_surf = pygame.Surface((mw, mh), pygame.SRCALPHA)
        c = geom.center
        mn = c.min(axis=0); mx = c.max(axis=0)
        span = max((mx - mn).max(), 1e-6)
        pad = 10
        sc = min(mw - 2 * pad, mh - 2 * pad) / span
        z = geom.z
        zr = (z - z.min()) / max(1e-6, z.max() - z.min())
        self._map = lambda x, y: (pad + (x - mn[0]) * sc + (mw - 2 * pad - (mx[0] - mn[0]) * sc) / 2,
                                  pad + (y - mn[1]) * sc + (mh - 2 * pad - (mx[1] - mn[1]) * sc) / 2)
        step = max(1, geom.n // 900)
        for i in range(0, geom.n, step):
            j = (i + step) % geom.n
            zc = zr[i]
            col = (int(60 + 180 * zc), int(120 + 80 * zc), int(90 - 40 * zc))
            pygame.draw.line(self.map_surf, col, self._map(*c[i]), self._map(*c[j]), 2)
        self._prev_gear = 1
        self._gear_pop = 0.0
        self._flash_t = 0.0
        self._rpm_disp = 0.0
        self._att = dict(p=0.0, r=0.0, p_hi=0.0, p_lo=0.0, r_hi=0.0, r_lo=0.0,
                         g_trail=deque(maxlen=22))
        self.f_gear = pygame.font.SysFont("menlo,consolas,monospace", 46, bold=True)
        self.f_ban = pygame.font.SysFont("menlo,consolas,monospace", 30, bold=True)
        self.events = []            # broadcast moments (sector flash, best lap)

    # ------------------------------------------------------------------ #
    def event(self, text, color=(120, 230, 160), sub=None, dur=2.4, kind="flash"):
        self.events.append(dict(text=text, sub=sub, color=color,
                                t=0.0, dur=dur, kind=kind))

    def _panel(self, pygame, screen, x, y, w, h, hud):
        panel = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(panel, (8, 9, 7, 168), panel.get_rect(), border_radius=12)
        pygame.draw.rect(panel, (*hud, 46), panel.get_rect(), 1, border_radius=12)
        screen.blit(panel, (x, y))

    def _draw_side_pitch(self, pygame, screen, cx, cy, pitch, hud, dim):
        # Side profile: sparse 787B-style outline for the pitch readout.
        # Nose is drawn at -x (screen left); Vehicle.pitch + = climbing/nose
        # up = positive screen rotation lifts the left end.
        angle = float(np.clip(pitch, -0.55, 0.55))
        body = [(-34, 6), (-31, 0), (-19, -1), (-9, -4), (-2, -13),
                (8, -13), (15, -5), (28, -3), (34, 3), (34, 6)]
        canopy = [(-4, -12), (7, -12), (14, -5)]
        wing = [(24, -8), (34, -8)]
        wing_leg = [(30, -8), (28, -2)]
        pygame.draw.lines(screen, hud, False,
                          _rot_points(body, angle, cx, cy, 1.0), 2)
        pygame.draw.lines(screen, dim, False,
                          _rot_points(canopy, angle, cx, cy, 1.0), 1)
        pygame.draw.line(screen, hud,
                         *_rot_points(wing, angle, cx, cy, 1.0), 1)
        pygame.draw.line(screen, dim,
                         *_rot_points(wing_leg, angle, cx, cy, 1.0), 1)
        pygame.draw.line(screen, dim,
                         *_rot_points([(-16, 1), (20, 1)], angle, cx, cy, 1.0), 1)
        for wx in (-22, 22):
            wc = _rot_points([(wx, 9)], angle, cx, cy, 1.0)[0]
            pygame.draw.circle(screen, hud, (int(wc[0]), int(wc[1])), 5, 1)
            pygame.draw.circle(screen, dim, (int(wc[0]), int(wc[1])), 2, 1)
        pygame.draw.line(screen, (70, 66, 56), (cx - 36, cy + 17), (cx + 36, cy + 17), 1)

    def _draw_rear_roll(self, pygame, screen, cx, cy, roll, hud, dim):
        # Rear view: positive roll raises the left side, matching Vehicle.roll.
        angle = float(np.clip(roll, -0.55, 0.55))
        body = [(-31, 6), (-28, -3), (-18, -8), (18, -8), (28, -3), (31, 6)]
        pygame.draw.lines(screen, hud, False,
                          _rot_points(body, angle, cx, cy, 1.0), 2)
        pygame.draw.line(screen, hud,
                         *_rot_points([(-29, -13), (29, -13)], angle, cx, cy, 1.0), 1)
        for wx in (-18, 18):
            pygame.draw.line(screen, dim,
                             *_rot_points([(wx, -13), (wx, -7)], angle, cx, cy, 1.0), 1)
        pygame.draw.line(screen, dim,
                         *_rot_points([(-10, -6), (10, -6)], angle, cx, cy, 1.0), 1)
        pygame.draw.line(screen, dim,
                         *_rot_points([(-24, 1), (24, 1)], angle, cx, cy, 1.0), 1)
        for wx in (-23, 23):
            tire = [(wx - 5, 5), (wx + 5, 5), (wx + 5, 11), (wx - 5, 11)]
            pts = _rot_points(tire, angle, cx, cy, 1.0)
            pygame.draw.polygon(screen, (8, 9, 8), pts)
            pygame.draw.lines(screen, hud, True, pts, 1)
            pygame.draw.line(screen, dim, pts[0], pts[1], 1)
        pygame.draw.line(screen, (70, 66, 56), (cx - 36, cy + 17), (cx + 36, cy + 17), 1)

    def _draw_attitude(self, pygame, screen, veh, hud, dim, dt):
        # Attitude station: protractor-scaled pitch/roll gauges with a damped
        # needle, peak-hold ghosts, and a central G-ball with a fading trail.
        x, y, w, h = 16, 94, 252, 150
        self._panel(pygame, screen, x, y, w, h, hud)
        st = self._att
        p_deg = math.degrees(float(getattr(veh, "pitch", 0.0)))
        r_deg = math.degrees(float(getattr(veh, "roll", 0.0)))
        if dt > 0.0:
            k = min(1.0, dt * 12.0)
            st["p"] += (p_deg - st["p"]) * k
            st["r"] += (r_deg - st["r"]) * k
            decay = dt * 5.0
            st["p_hi"] = max(p_deg, st["p_hi"] - decay)
            st["p_lo"] = min(p_deg, st["p_lo"] + decay)
            st["r_hi"] = max(r_deg, st["r_hi"] - decay)
            st["r_lo"] = min(r_deg, st["r_lo"] + decay)

        screen.blit(self.f_tin.render("PITCH", True, dim), (x + 16, y + 10))
        gtag = self.f_tin.render("G", True, dim)
        screen.blit(gtag, (x + w // 2 - gtag.get_width() // 2, y + 10))
        rtag = self.f_tin.render("ROLL", True, dim)
        screen.blit(rtag, (x + w - 16 - rtag.get_width(), y + 10))

        LIM = 24.0                          # gauge range, degrees
        arc_k = math.radians(55.0) / LIM    # degrees → arc sweep
        top = math.radians(270.0)

        def gauge(cx, cy, d, lo, hi):
            def arc_pt(deg, r):
                a = top - deg * arc_k
                return (cx + math.cos(a) * r, cy + math.sin(a) * r)

            t = -24
            while t <= 24:
                major = t % 12 == 0
                col = ((255, 110, 96) if abs(t) >= 16
                       else (dim if major else (86, 80, 66)))
                pygame.draw.line(screen, col, arc_pt(t, 46),
                                 arc_pt(t, 46 - (7 if major else 4)),
                                 2 if major else 1)
                t += 4
            pygame.draw.polygon(screen, dim, ((cx, cy - 49),
                                              (cx - 3, cy - 54), (cx + 3, cy - 54)))
            sev = ((255, 110, 96) if abs(d) >= 16
                   else (AMBER_HI if abs(d) >= 8 else hud))
            dc = max(-LIM, min(LIM, d))
            if abs(dc) > 0.4:
                steps = max(2, int(abs(dc)))
                pygame.draw.lines(screen, sev, False,
                                  [arc_pt(dc * i / steps, 34)
                                   for i in range(steps + 1)], 4)
            nx, ny = arc_pt(dc, 34)
            pygame.draw.circle(screen, sev, (int(nx), int(ny)), 3)
            for pk in (lo, hi):             # peak-hold ghost ticks
                if abs(pk) > 2.0:
                    pc = max(-LIM, min(LIM, pk))
                    pygame.draw.line(screen, (255, 170, 110),
                                     arc_pt(pc, 45), arc_pt(pc, 38), 1)
            return sev

        pcx, rcx, gy = x + 60, x + 192, y + 84
        p_col = gauge(pcx, gy, st["p"], st["p_lo"], st["p_hi"])
        r_col = gauge(rcx, gy, st["r"], st["r_lo"], st["r_hi"])
        self._draw_side_pitch(pygame, screen, pcx, gy,
                              math.radians(st["p"]), hud, dim)
        self._draw_rear_roll(pygame, screen, rcx, gy,
                             math.radians(st["r"]), hud, dim)
        p_lab = self.f_med.render(f"{p_deg:+5.1f}°", True, p_col)
        screen.blit(p_lab, (pcx - p_lab.get_width() // 2, y + h - 40))
        r_lab = self.f_med.render(f"{r_deg:+5.1f}°", True, r_col)
        screen.blit(r_lab, (rcx - r_lab.get_width() // 2, y + h - 40))

        # G-ball: acceleration vector inside 1g/2g rings, braking plots up.
        gcx, gcy, ppg = x + w // 2, y + 80, 12
        ax_g = float(getattr(veh, "ax", 0.0)) / 9.81
        ay_g = float(getattr(veh, "ay", 0.0)) / 9.81
        st["g_trail"].append((ay_g, ax_g))
        for rg in (ppg, 2 * ppg):
            pygame.draw.circle(screen, (60, 57, 48), (gcx, gcy), rg, 1)
        pygame.draw.line(screen, (60, 57, 48), (gcx - 26, gcy), (gcx + 26, gcy), 1)
        pygame.draw.line(screen, (60, 57, 48), (gcx, gcy - 26), (gcx, gcy + 26), 1)

        def g_pt(lat, lon):
            dx = max(-25.0, min(25.0, -lat * ppg))
            dy = max(-25.0, min(25.0, lon * ppg))
            return (int(gcx + dx), int(gcy + dy))

        n = len(st["g_trail"])
        for i, (lat, lon) in enumerate(st["g_trail"]):
            if i == n - 1:
                continue
            f = i / max(1, n - 1)
            pygame.draw.circle(screen, _mix_color((44, 42, 36), hud, f * 0.85),
                               g_pt(lat, lon), 2 if f > 0.5 else 1)
        pygame.draw.circle(screen, AMBER_HI, g_pt(ay_g, ax_g), 3)
        g_tot = math.hypot(ax_g, ay_g)
        g_col = AMBER_HI if g_tot > 1.2 else hud
        g_lab = self.f_sml.render(f"{g_tot:0.1f}G", True, g_col)
        screen.blit(g_lab, (gcx - g_lab.get_width() // 2, y + h - 36))

    def _draw_drive_cluster(self, pygame, screen, veh, tele, hud, dim, dt):
        # Bottom-left: analog tach dial + shift lights + gear box + speed.
        H = self.H
        px, py, pw, ph = 16, H - 200, 332, 184
        self._panel(pygame, screen, px, py, pw, ph, hud)

        spec = veh.spec
        cutoff = max(1.0, float(getattr(spec, "cutoff_rpm", 8000.0)))
        redline = min(cutoff, float(getattr(spec, "redline_rpm", cutoff * 0.9)))
        rpm = max(0.0, float(veh.rpm))
        if dt > 0.0:
            self._rpm_disp += (rpm - self._rpm_disp) * min(1.0, dt * 16.0)

        top_gear = len(getattr(spec, "gear_ratios", ())) or 6
        cur_gear = max(1, veh.gear)
        at_red = rpm >= redline * 0.995
        want_shift = rpm >= redline * 0.965 and cur_gear < top_gear
        limiter_gate = tele.get("limiter_gate")
        strobe_on = (bool(limiter_gate) if (at_red and limiter_gate is not None)
                     else int(self._flash_t * (9.0 if at_red else 7.0)) % 2 == 0)
        flash = want_shift and strobe_on

        # ---- shift lights across the top ---- #
        led_gap = 24
        lx0 = px + pw // 2 - (5 * led_gap) // 2
        ly = py + 16
        urge = rpm / redline
        for i, (th, col) in enumerate((
                (0.84, (255, 196, 120)), (0.88, (255, 196, 120)),
                (0.92, AMBER_HI), (0.95, AMBER_HI),
                (0.98, (255, 92, 66)), (1.0, (255, 92, 66)))):
            cx_i = lx0 + i * led_gap
            if at_red:
                col, lit = ((255, 240, 225) if strobe_on else (255, 70, 55)), True
            else:
                lit = urge >= th
            if lit:
                pygame.draw.circle(screen, col, (cx_i, ly), 5)
            else:
                pygame.draw.circle(screen, (40, 38, 33), (cx_i, ly), 5)
                pygame.draw.circle(screen, (72, 68, 57), (cx_i, ly), 5, 1)

        # ---- analog tach: 270° sweep, 0 rpm at 7-o'clock ---- #
        cx, cy, R = px + 94, py + 106, 62
        a0, sweep = math.radians(135.0), math.radians(270.0)

        def dial_pt(f, r):
            a = a0 + sweep * min(1.0, max(0.0, f))
            return (cx + math.cos(a) * r, cy + math.sin(a) * r)

        pygame.draw.circle(screen, (11, 12, 10), (cx, cy), R)
        pygame.draw.circle(screen, (74, 70, 58), (cx, cy), R, 1)

        f_red = redline / cutoff
        band_col = (255, 92, 66) if (at_red and strobe_on) else (128, 44, 36)
        steps = max(4, int((1.0 - f_red) * 68))
        band = [dial_pt(f_red + (1.0 - f_red) * i / steps, R - 7)
                for i in range(steps + 1)]
        pygame.draw.lines(screen, band_col, False, band, 6)

        v = 0.0
        while v <= cutoff + 1.0:
            f = v / cutoff
            in_red = v >= redline - 1.0
            if int(v) % 1000 == 0:
                col = (255, 110, 90) if in_red else hud
                pygame.draw.line(screen, col, dial_pt(f, R - 3), dial_pt(f, R - 13), 2)
                num = self.f_tin.render(f"{int(v) // 1000}", True,
                                        col if in_red else dim)
                nx, ny = dial_pt(f, R - 22)
                screen.blit(num, (nx - num.get_width() / 2, ny - num.get_height() / 2))
            else:
                pygame.draw.line(screen, (86, 80, 66),
                                 dial_pt(f, R - 3), dial_pt(f, R - 9), 1)
            v += 500.0

        lab = self.f_tin.render("x1000", True, dim)
        screen.blit(lab, (cx - lab.get_width() // 2, cy - 28))
        needle_col = (255, 110, 90) if at_red else (255, 236, 200)
        f_now = self._rpm_disp / cutoff
        pygame.draw.line(screen, needle_col, dial_pt(f_now, -12),
                         dial_pt(f_now, R - 14), 3)
        pygame.draw.circle(screen, (26, 25, 21), (cx, cy), 7)
        pygame.draw.circle(screen, hud, (cx, cy), 7, 1)
        digits = self.f_tin.render(f"{rpm:4.0f} RPM", True, hud)
        screen.blit(digits, (cx - digits.get_width() // 2, cy + 36))

        # ---- gear box: flashes when the engine wants an upshift ---- #
        gx, gy, gw, gh = px + 192, py + 34, 122, 72
        box = pygame.Surface((gw, gh), pygame.SRCALPHA)
        if flash:
            fill = (255, 70, 55, 105) if at_red else (255, 190, 110, 60)
            pygame.draw.rect(box, fill, box.get_rect(), border_radius=10)
        border = (255, 92, 66, 220) if flash else (*hud, 70)
        pygame.draw.rect(box, border, box.get_rect(), 1, border_radius=10)
        screen.blit(box, (gx, gy))
        screen.blit(self.f_tin.render("GEAR", True, dim), (gx + 9, gy + 7))
        if want_shift:
            tag_col = (255, 240, 225) if strobe_on else (255, 92, 66)
            tag = self.f_tin.render("SHIFT", True, tag_col)
            screen.blit(tag, (gx + gw - tag.get_width() - 9, gy + 7))

        if cur_gear != self._prev_gear:
            self._gear_pop = 0.28
            self._prev_gear = cur_gear
        self._gear_pop = max(0.0, self._gear_pop - dt)
        pop = self._gear_pop / 0.28
        digit = self.f_gear.render(str(cur_gear), True,
                                   AMBER_HI if pop > 0.4 else hud)
        if pop > 0.0:
            dw, dh = digit.get_size()
            s = 1.0 + 0.30 * pop
            digit = pygame.transform.smoothscale(digit, (int(dw * s), int(dh * s)))
        screen.blit(digit, (gx + (gw - digit.get_width()) // 2,
                            gy + (gh - digit.get_height()) // 2 + 5))

        mph = veh.speed * MPS_TO_MPH
        spd = self.f_big.render(f"{mph:3.0f}", True, AMBER_HI)
        screen.blit(spd, (gx + 2, py + 112))
        screen.blit(self.f_sml.render("MPH", True, dim),
                    (gx + 10 + spd.get_width(), py + 134))

        # A shift briefly turns the recent normalized RPM history into a tiny
        # oscilloscope. It uses coordinator samples, so the visual drop matches
        # the exact gear event that drives the exhaust pressure pulse.
        wave = tele.get("engine_wave") or ()
        wave_alpha = float(tele.get("shift_wave", 0.0))
        if wave_alpha > 0.02 and len(wave) > 1:
            wx0, wy0, ww, wh = gx, py + ph - 24, gw, 16
            wave_col = _mix_color((50, 47, 40), AMBER_HI, min(1.0, wave_alpha))
            pts = []
            for j, sample in enumerate(wave):
                wpx = wx0 + j * ww / max(1, len(wave) - 1)
                wpy = wy0 + wh - float(np.clip(sample, 0.0, 1.1)) / 1.1 * wh
                pts.append((int(wpx), int(wpy)))
            pygame.draw.lines(screen, wave_col, False, pts, 1)

    def draw(self, pygame, screen, veh, mood, lap, tele):
        W, H = self.W, self.H
        hud, dim = mood["hud"], mood["hud_dim"]
        dt = tele.get("dt", 0.0)
        self._flash_t += dt

        # ---------------- top-centre: broadcast chip + lap progress -------- #
        cw = 560
        cx0 = (W - cw) // 2
        self._panel(pygame, screen, cx0, 10, cw, 30, hud)
        screen.blit(self.f_sml.render("FABLE FIVE", True, AMBER_HI), (cx0 + 14, 18))
        screen.blit(self.f_tin.render(tele.get("course", "NORDSCHLEIFE"), True, dim),
                    (cx0 + 106, 20))
        # LIVE dot (pulses) + current camera
        pulse = 0.5 + 0.5 * math.sin(self._flash_t * 4.0)
        shot = tele.get("shot")
        replay_label = tele.get("replay")
        live_dot_x = cx0 + cw - (222 if shot else 130)
        pygame.draw.circle(screen, (255, int(60 + 40 * pulse), 50),
                           (live_dot_x, 25), 4)
        screen.blit(self.f_tin.render("REPLAY" if replay_label else "LIVE", True,
                                      (255, 120, 100)),
                    (live_dot_x + 10, 20))
        if shot and not replay_label:
            screen.blit(self.f_tin.render(f"CAM·{shot.upper()}", True, dim),
                        (cx0 + cw - 170, 20))
        elif replay_label:
            screen.blit(self.f_tin.render("0.66×  SLOW MOTION", True, AMBER_HI),
                        (cx0 + cw - 170, 20))

        prog = tele.get("progress")
        if prog is not None:
            bx, by, bw = cx0 + 8, 44, cw - 16
            pygame.draw.rect(screen, (42, 40, 36), (bx, by, bw, 5), border_radius=2)
            clean = tele.get("clean", True)
            fill = (120, 230, 160) if clean else (255, 110, 96)
            pygame.draw.rect(screen, fill, (bx, by, int(bw * min(1.0, prog)), 5),
                             border_radius=2)
            for tick in (1.0 / 3.0, 2.0 / 3.0):
                tx = bx + int(bw * tick)
                pygame.draw.line(screen, (200, 200, 200), (tx, by - 2), (tx, by + 7), 1)
            mx = bx + int(bw * min(1.0, prog))
            pygame.draw.polygon(screen, AMBER_HI,
                                ((mx, by - 4), (mx + 4, by + 2), (mx, by + 8), (mx - 4, by + 2)))
            sec = tele.get("sector", 1)
            tag = self.f_tin.render(f"S{sec}", True, hud)
            screen.blit(tag, (bx + bw + 2 - tag.get_width(), by + 9))
            pill_txt = "CLEAN" if clean else "INVALID"
            pill_col = (120, 230, 160) if clean else (255, 110, 96)
            screen.blit(self.f_tin.render(pill_txt, True, pill_col), (bx, by + 9))

        if replay_label:
            rw = 330
            rx = (W - rw) // 2
            panel = pygame.Surface((rw, 38), pygame.SRCALPHA)
            pygame.draw.rect(panel, (10, 10, 12, 205), panel.get_rect(), border_radius=9)
            pygame.draw.rect(panel, (255, 92, 66, 150), panel.get_rect(), 1, border_radius=9)
            screen.blit(panel, (rx, 76))
            label = self.f_med.render(replay_label, True, (255, 188, 132))
            screen.blit(label, (rx + (rw - label.get_width()) // 2, 85))

        # ---------------- bottom-left: tach dial / gear / speed ------------ #
        self._draw_drive_cluster(pygame, screen, veh, tele, hud, dim, dt)

        # ---------------- bottom-right: lap board -------------------------- #
        self._panel(pygame, screen, W - 268, H - 148, 252, 132, hud)
        delta = tele.get("delta")
        if delta is None:
            d_txt, d_col = "--.--", dim
        else:
            d_txt = f"{delta:+5.2f}"
            d_col = (120, 230, 160) if delta <= 0.0 else (255, 150, 110)
        rows = [("LAP", f"{lap['count']}", hud),
                ("TIME", _fmt_lap(lap['t']), AMBER_HI if not lap['invalid'] else (255, 110, 96)),
                ("LAST", _fmt_lap(lap['last']), dim),
                ("BEST CLEAN", _fmt_lap(lap['best']), (120, 230, 160) if lap['best'] else dim),
                ("Δ BEST", d_txt, d_col)]
        yy = H - 138
        for name, val, col in rows:
            screen.blit(self.f_tin.render(name, True, dim), (W - 252, yy + 3))
            screen.blit(self.f_med.render(val, True, col), (W - 156, yy))
            yy += 25

        # ---------------- top-right: minimap ------------------------------- #
        screen.blit(self.map_surf, (W - 226, 14))
        mx, my = self._map(veh.x, veh.y)
        pygame.draw.circle(screen, AMBER_HI, (W - 226 + int(mx), 14 + int(my)), 4)
        pygame.draw.circle(screen, (20, 20, 20), (W - 226 + int(mx), 14 + int(my)), 4, 1)

        # ---------------- top-left: pace / drift / air --------------------- #
        status = []
        if tele.get("weather") and tele.get("weather") != "CLEAR":
            status.append(str(tele["weather"]))
        if float(tele.get("wet", 0.0)) > 0.12:
            status.append(f"WET {int(float(tele['wet']) * 100):02d}%")
        if tele.get("hq_post"):
            status.append("HQ FX")
        title_txt = tele["title"] + (("  ·  " + "  ·  ".join(status)) if status else "")
        title_col = (132, 194, 220) if status else dim
        screen.blit(self.f_tin.render(title_txt, True, title_col), (16, 12))
        slide = abs(math.degrees(veh.slip_angle)) if veh.speed > 5 else 0.0
        if slide > 6:
            f = min(1.0, slide / 60.0)
            pygame.draw.rect(screen, (50, 48, 44), (16, 30, 150, 7), border_radius=3)
            col = hud if slide < 40 else (255, 90, 70)
            pygame.draw.rect(screen, col, (16, 30, int(150 * f), 7), border_radius=3)
            screen.blit(self.f_tin.render(f"DRIFT {slide:2.0f}°", True, col), (172, 27))
        pace = tele.get("pace")
        if pace is not None and veh.speed > 8:
            d_mph = pace * MPS_TO_MPH
            col = (120, 230, 160) if d_mph >= -2.0 else \
                  ((255, 205, 120) if d_mph >= -9.0 else (255, 110, 96))
            screen.blit(self.f_tin.render(f"PACE {d_mph:+5.0f} MPH vs LIMIT", True, col),
                        (16, 74))
        self._draw_attitude(pygame, screen, veh, hud, dim, dt)
        if veh.airborne:
            pill = self.f_med.render("AIR", True, (10, 10, 10))
            pygame.draw.rect(screen, AMBER_HI, (16, 46, pill.get_width() + 16, 24),
                             border_radius=12)
            screen.blit(pill, (24, 48))
        elif getattr(veh, "landing_g", 0.0) > 1.5 and tele.get("landing_flash", 0) > 0:
            g = self.f_sml.render(f"LANDING {veh.landing_g:0.1f} G", True, (255, 160, 90))
            screen.blit(g, (16, 48))
        if tele.get("hint"):
            hint = self.f_tin.render(tele["hint"], True, (110, 112, 104))
            screen.blit(hint, ((W - hint.get_width()) // 2, H - 20))

        # ---------------- broadcast moments -------------------------------- #
        keep = []
        yy = int(H * 0.24)
        for ev in self.events:
            ev["t"] += dt
            if ev["t"] >= ev["dur"]:
                continue
            keep.append(ev)
            fade = min(1.0, 4.0 * (1.0 - ev["t"] / ev["dur"]))
            a = int(230 * fade)
            if ev["kind"] == "finish":
                # checkered strip sweeping the screen
                strip = pygame.Surface((W, 26), pygame.SRCALPHA)
                sq = 26
                for kk in range(0, W // sq + 1):
                    c = (232, 232, 228, a) if kk % 2 == 0 else (20, 20, 22, a)
                    pygame.draw.rect(strip, c, (kk * sq, 0, sq, 26))
                screen.blit(strip, (0, yy - 34))
            txt = self.f_ban.render(ev["text"], True, ev["color"])
            bgw = txt.get_width() + 48
            panel = pygame.Surface((bgw, 46), pygame.SRCALPHA)
            pygame.draw.rect(panel, (8, 9, 7, int(150 * fade)), panel.get_rect(),
                             border_radius=10)
            pygame.draw.rect(panel, (*ev["color"], int(90 * fade)), panel.get_rect(),
                             1, border_radius=10)
            screen.blit(panel, ((W - bgw) // 2, yy))
            txt.set_alpha(a)
            screen.blit(txt, ((W - txt.get_width()) // 2, yy + 7))
            if ev.get("sub"):
                sub = self.f_sml.render(ev["sub"], True, (210, 208, 200))
                sub.set_alpha(a)
                screen.blit(sub, ((W - sub.get_width()) // 2, yy + 48))
            yy += 66
        self.events = keep


# --------------------------------------------------------------------------- #
# main loop
# --------------------------------------------------------------------------- #
def run(car: str = "supra", track_name: str = "random", seed: int | None = 7,
        audio_on: bool = True, controller=None, track=None, title=None,
        max_frames=None, agent=None, start_speed: float = 0.0, opponents: list = None,
        racers: list = None, sensor_spec=None, shift_controller=None,
        wet_visual: bool = False, ai_vision: bool = False,
        weather_mode: int = 0, hq_post: bool = False):
    """V2 projected 2.5D viewer — same contract as supra.app.run.

    `shift_controller(veh, throttle, dt) -> (clutch, up, down)` optionally
    replaces the AutoBox while an AI controller drives (Fable's gear-aware
    policies shift for themselves through their RaceBox)."""
    # multi-car scenes still run on the classic viewer (untouched v1; it has
    # no shift_controller — gear-aware policies degrade to its AutoBox there)
    if racers or opponents or os.environ.get("SUPRA_CLASSIC_VIEWER"):
        from .app import run as classic_run
        return classic_run(car=car, track_name=track_name, seed=seed,
                           audio_on=audio_on, controller=controller, track=track,
                           title=title, max_frames=max_frames, agent=agent,
                           start_speed=start_speed, opponents=opponents,
                           racers=racers, sensor_spec=sensor_spec)

    try:
        import pygame
    except ImportError:
        raise SystemExit("pygame is required. pip3 install pygame")

    from .sensors import SensorSuite
    from .sound import SpatialAudioMixer

    pygame.init()
    W, H = 1480, 820
    render_fps = _display_refresh_target(120)
    display_flags = pygame.SCALED | pygame.RESIZABLE
    screen = pygame.display.set_mode((W, H), display_flags, vsync=1)
    pygame.display.set_caption(title or f"Supra Drift V2 — {car} [{track_name}]")
    clock = pygame.time.Clock()
    perf = PerformanceMonitor(pygame, render_fps)
    governor = PresentationGovernor(render_fps)
    fullscreen = False

    sim = SimSpec()
    spec = get_car(car)
    veh = Vehicle(spec, sim)
    sensors = SensorSuite(sensor_spec)

    tracks = {
        "oval": lambda: track_mod.oval(),
        "random": lambda: track_mod.random_circuit(seed=seed),
        "touge": lambda: track_mod.touge(seed=seed),
    }
    trk = track if track is not None else tracks.get(track_name, tracks["random"])()
    geom = RoadGeom(trk, seed=seed or 7)
    view = View(W, H)
    render_veh = RenderVehicle()
    hud = Hud(pygame, W, H, trk, geom)
    ai_layer = AIVision(pygame, W, H, enabled=bool(ai_vision and agent is not None))
    skids = Skids()
    puffs = Puffs()
    rng = np.random.default_rng(seed or 7)
    cine = Cinematic(pygame, W, H, rng)
    tension = SpeedTension(pygame, W, H, rng)
    flyby_fx = FlybyDopplerFX(W, H)
    director = Director(trk, geom, rng)
    replay = ReplaySystem()
    weather = WeatherSystem(W, H, rng)
    weather.index = int(weather_mode) % len(weather.MODES)
    session_visual = SessionVisualState(geom.n)
    post_fx = PostFX(pygame, W, H, enabled=hq_post)
    mood_tints = {}
    audio_visuals = AudioReactiveVisuals()
    director_shot_keys = {
        pygame.K_1: "chase", pygame.K_KP1: "chase",
        pygame.K_2: "quarter", pygame.K_KP2: "quarter",
        pygame.K_3: "side", pygame.K_KP3: "side",
        pygame.K_4: "drone", pygame.K_KP4: "drone",
        pygame.K_5: "trackside", pygame.K_KP5: "trackside",
    }
    vref_arr = getattr(trk, "fable_vref", None)   # for the HUD pace read-out

    sx, sy, syaw = trk.start_pose()
    veh.reset(sx, sy, syaw, speed=start_speed)
    fr = trk.frame(veh.x, veh.y)
    veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])
    render_veh.reset(veh)
    view.cx, view.cy, view.z0 = veh.x, veh.y, fr["z"]
    view.rot = -math.pi / 2.0 - syaw

    audio = SpatialAudioMixer()
    if audio_on:
        audio.start([veh])

    autobox = AutoBox(spec)
    auto = True
    mood_i = 0
    show_beams = False
    obs = sensors.observe(veh, trk)
    ctrl_period = max(1, round(1.0 / (30 * sim.dt)))
    ctrl_ctr = 0
    cur_act = (0.0, 0.0, 0.0, 0.0)
    t_in = b_in = s_in = 0.0
    handbrake = 0.0

    lap = dict(count=0, t=0.0, last=None, best=None, invalid=False)
    prev_frac = fr["progress"]
    stall_t = 0.0
    landing_flash = 0.0
    drive_time = 0.0
    prev_air = False
    prev_off = False
    prev_sec = 0
    drift_moment_t = 0.0
    prev_landmark = geom.landmark_name[trk.nearest(veh.x, veh.y)]
    post_impact = 0.0
    trace_p, trace_t = [], []       # this lap's progress -> time trace
    ref_p = ref_t = None            # best clean lap's trace (for Δ BEST)
    delta_live = None

    # broadcast director: on by default when an AI is driving (watch mode)
    if controller is not None:
        director.toggle(view)

    hint = ("WASD/arrows drive · SPACE handbrake · C camera · V director · "
            "1-5 shots · P replay · I AI · H weather · O HQ FX · G time · Y wet · B beams · T auto · Q/E shift · R reset · "
            "F3 performance · F11 fullscreen · M mute · ESC quit")

    MAX_CATCHUP = 0.20
    accum = 0.0
    frame_no = 0
    running = True
    while running:
        frame_dt = clock.tick(render_fps) / 1000.0
        accum += min(frame_dt, MAX_CATCHUP)
        frame_no += 1
        if max_frames and frame_no > max_frames:
            running = False

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    running = False
                elif ev.key == pygame.K_r:
                    veh.reset(sx, sy, syaw, speed=start_speed)
                    fr_reset = trk.frame(veh.x, veh.y)
                    veh.set_road(fr_reset["grade"], fr_reset["bank"],
                                 fr_reset["heading"], fr_reset["z"],
                                 fr_reset["vcurv"])
                    render_veh.reset(veh)
                    lap.update(count=0, t=0.0, invalid=False)
                    prev_frac = trk.frame(veh.x, veh.y)["progress"]
                    trace_p, trace_t = [], []
                    prev_sec = 0
                    view.cut = True
                    audio.reset([veh])
                elif ev.key in director_shot_keys:
                    i_manual = trk.nearest(veh.x, veh.y)
                    director.select_manual(view, veh, i_manual,
                                           director_shot_keys[ev.key])
                elif ev.key == pygame.K_c:
                    if director.on:
                        director.disable(view)
                    view.mode = "north" if view.mode == "chase" else "chase"
                elif ev.key == pygame.K_v:
                    director.toggle(view)
                elif ev.key == pygame.K_p:
                    replay.manual()
                    view.cut = True
                elif ev.key == pygame.K_i:
                    if agent is not None:
                        enabled = ai_layer.toggle()
                        hud.event(f"AI VISION {'ON' if enabled else 'OFF'}",
                                  color=(102, 210, 240) if enabled else (150, 150, 150),
                                  dur=1.5)
                    else:
                        hud.event("AI VISION UNAVAILABLE", color=(255, 145, 100), dur=1.5)
                elif ev.key == pygame.K_h:
                    label = weather.cycle()
                    hud.event(f"WEATHER · {label}", color=(142, 202, 230), dur=1.7)
                elif ev.key == pygame.K_o:
                    enabled = post_fx.toggle()
                    hud.event(f"HQ POST FX {'ON' if enabled else 'OFF'}",
                              color=(194, 156, 244) if enabled else (150, 150, 150), dur=1.5)
                elif ev.key == pygame.K_F3:
                    perf.visible = not perf.visible
                elif ev.key == pygame.K_F11:
                    try:
                        result = pygame.display.toggle_fullscreen()
                        if result == 0:
                            fullscreen = not fullscreen
                            hud.event(f"FULLSCREEN {'ON' if fullscreen else 'OFF'}",
                                      color=(126, 214, 170), dur=1.2)
                    except pygame.error:
                        hud.event("FULLSCREEN UNAVAILABLE", color=(255, 145, 100), dur=1.5)
                elif ev.key == pygame.K_g:
                    mood_i = (mood_i + 1) % len(MOODS)
                elif ev.key == pygame.K_y:
                    wet_visual = not wet_visual
                elif ev.key == pygame.K_b:
                    show_beams = not show_beams
                elif ev.key == pygame.K_t:
                    auto = not auto
                elif ev.key == pygame.K_m:
                    audio.toggle_mute()
                elif ev.key == pygame.K_q and not auto:
                    veh.shift_down()
                elif ev.key == pygame.K_e and not auto:
                    veh.shift_up()

        keys = pygame.key.get_pressed()

        # --- physics: consume real elapsed time at 120 Hz ---
        steps = 0
        while accum >= sim.dt and steps < int(MAX_CATCHUP / sim.dt) + 2:
            fr = trk.frame(veh.x, veh.y)
            veh.surface_grip = spec.offtrack_grip if fr["off_track"] else 1.0
            veh.set_road(fr["grade"], fr["bank"], fr["heading"], fr["z"], fr["vcurv"])

            if controller is not None:
                if ctrl_ctr % ctrl_period == 0:
                    obs = sensors.observe(veh, trk)
                    cur_act = controller(veh, obs)
                ctrl_ctr += 1
                s_in, t_raw, b_raw, handbrake = (list(cur_act) + [0, 0, 0, 0])[:4]
                t_in, b_in = t_raw, b_raw
            else:
                up = keys[pygame.K_UP] or keys[pygame.K_w]
                dn = keys[pygame.K_DOWN] or keys[pygame.K_s]
                lf = keys[pygame.K_LEFT] or keys[pygame.K_a]
                rt = keys[pygame.K_RIGHT] or keys[pygame.K_d]
                t_in = min(1.0, t_in + 3.5 * sim.dt) if up else max(0.0, t_in - 6 * sim.dt)
                b_in = min(1.0, b_in + 5.0 * sim.dt) if dn else max(0.0, b_in - 8 * sim.dt)
                steer_target = (1.0 if lf else 0.0) - (1.0 if rt else 0.0)
                s_in += (steer_target - s_in) * min(1.0, 7.0 * sim.dt)
                handbrake = 1.0 if keys[pygame.K_SPACE] else 0.0

            if shift_controller is not None and controller is not None:
                clutch, up_s, dn_s = shift_controller(veh, max(t_in, 0.0), sim.dt)
            elif auto:
                clutch, up_s, dn_s = autobox.update(veh, max(t_in, 0.0), sim.dt)
            else:
                clutch, up_s, dn_s = 0.0, False, False
            if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
                clutch = 1.0
            veh.step(Controls(steer=float(np.clip(s_in, -1, 1)),
                              throttle=float(np.clip(t_in, 0, 1)),
                              brake=float(np.clip(b_in, 0, 1)),
                              clutch=clutch, handbrake=handbrake,
                              shift_up=up_s, shift_down=dn_s))
            render_veh.capture(veh)
            accum -= sim.dt
            steps += 1
            drive_time += sim.dt
            lap["t"] += sim.dt

            # lap accounting (clean-lap rules: off-track invalidates)
            frac = fr["progress"]
            d = frac - prev_frac
            if d < -0.5:
                lap["count"] += 1
                lap["last"] = lap["t"]
                lap_time = lap["t"]
                real_lap = lap_time > 30.0
                new_best = (not lap["invalid"] and real_lap
                            and (lap["best"] is None or lap_time < lap["best"]))
                if not lap["invalid"] and real_lap:
                    lap["best"] = lap_time if lap["best"] is None else min(lap["best"], lap_time)
                # broadcast moments
                if real_lap:
                    hud.event(f"LAP {lap['count']}  ·  {_fmt_lap(lap_time)}",
                              color=AMBER_HI, dur=2.6, kind="finish")
                    if new_best:
                        hud.event("NEW BEST LAP", color=(255, 214, 90),
                                  sub=_fmt_lap(lap_time), dur=3.6)
                if new_best and len(trace_p) > 8:
                    ref_p = np.asarray(trace_p)
                    ref_t = np.asarray(trace_t)
                trace_p, trace_t = [], []
                prev_sec = 0
                lap["t"] = 0.0
                lap["invalid"] = False
            prev_frac = frac
            if fr["off_track"]:
                lap["invalid"] = True

            # sector splits (3 equal track thirds) + best-lap delta trace
            sec = min(2, int(frac * 3.0))
            if sec == prev_sec + 1:
                if not lap["invalid"] and lap["t"] > 5.0:
                    hud.event(f"SECTOR {sec}  ·  {_fmt_lap(lap['t'])}",
                              color=(120, 230, 160), dur=1.8)
                prev_sec = sec
            if not trace_p or frac > trace_p[-1] + 0.0015:
                trace_p.append(frac)
                trace_t.append(lap["t"])

            # AI watch: auto-reset when stuck
            if controller is not None:
                stall_t = stall_t + sim.dt if (drive_time > 3 and veh.speed < 0.6) else 0.0
                if stall_t > 3.5 or abs(fr["lateral"]) > fr.get("half_width", trk.half) + 18:
                    veh.reset(sx, sy, syaw, speed=start_speed)
                    fr_reset = trk.frame(veh.x, veh.y)
                    veh.set_road(fr_reset["grade"], fr_reset["bank"],
                                 fr_reset["heading"], fr_reset["z"],
                                 fr_reset["vcurv"])
                    render_veh.reset(veh)
                    prev_frac = trk.frame(veh.x, veh.y)["progress"]
                    lap.update(t=0.0, invalid=False)
                    stall_t = 0.0
                    trace_p, trace_t = [], []
                    prev_sec = 0
                    view.cut = True        # broadcast cut, don't swoop back
                    audio.reset([veh])

        perf.update(frame_dt, steps)
        governor.update(frame_dt)

        fr = trk.frame(veh.x, veh.y)
        reactive_camera = ((veh.x, veh.y) if replay.playing else (view.cx, view.cy))
        fixed_flyby_camera = (director.on and director.shot in ("trackside", "apex")
                              and not replay.playing)
        audio_visuals.update(veh, t_in, b_in, frame_dt, reactive_camera,
                             fixed_camera=fixed_flyby_camera)
        if not veh.airborne and getattr(veh, "landing_g", 0) > 1.5:
            landing_flash = max(landing_flash, 0.9)
        landing_flash = max(0.0, landing_flash - frame_dt)

        # landing: dust kicked up + a camera jolt scaled by the hit
        if prev_air and not veh.airborne:
            g = float(getattr(veh, "landing_g", 0.0))
            if g > 1.2:
                puffs.burst(veh.x, veh.y, fr["z"], rng,
                            n=int(min(18, 5 + g * 3)),
                            kind="spray" if wet_visual and not fr["off_track"] else "dust")
                view.impulse(min(9.0, g * 2.0))
                post_impact = max(post_impact, min(1.0, g / 4.5))
                if controller is not None and director.mode == "auto" and g > 2.15:
                    replay.request("HARD LANDING")
        prev_air = veh.airborne

        # kerb strike: sparks + audio rattle flag + a nibble of camera shake
        i_near = trk.nearest(veh.x, veh.y)
        landmark_now = geom.landmark_name[i_near]
        if landmark_now != prev_landmark:
            if landmark_now and drive_time > 1.0:
                hud.event(str(landmark_now).upper(), color=(126, 214, 170),
                          sub="NORDSCHLEIFE SECTION", dur=2.2)
            prev_landmark = landmark_now
        on_kerb = (geom.kerb[i_near] and veh.speed > 10.0 and not veh.airborne
                   and abs(fr["lateral"]) > fr.get("half_width", trk.half) * 0.55)
        veh._snd_kerb = 1.0 if on_kerb else 0.0
        if on_kerb and veh.speed > 14.0 and rng.random() < min(0.9, veh.speed / 60.0):
            b = getattr(veh.spec, "b", 1.3)
            ht = getattr(veh.spec, "half_track", 0.8)
            sgn = 1.0 if fr["lateral"] > 0 else -1.0
            cyw_s, syw_s = math.cos(veh.yaw), math.sin(veh.yaw)
            wx = veh.x - cyw_s * b - syw_s * sgn * ht
            wy = veh.y - syw_s * b + cyw_s * sgn * ht
            puffs.spark(wx + rng.uniform(-0.3, 0.3), wy + rng.uniform(-0.3, 0.3),
                        fr["z"], rng, vx=veh.speed * cyw_s, vy=veh.speed * syw_s)
            view.impulse(0.9)
            post_impact = max(post_impact, 0.24)

        # Each real limiter gate edge gets one crisp kick; continuous resonance
        # comes from the deterministic audio vibration offset below.
        if audio_visuals.limiter_just_cut and veh.speed > 10.0:
            view.impulse(0.20)
        if audio_visuals.flyby_just_passed:
            view.impulse(1.15)
        if fr["off_track"] and not prev_off and veh.speed > 22.0:
            view.impulse(5.0)
            post_impact = max(post_impact, min(0.85, veh.speed / 75.0))
            if controller is not None and director.mode == "auto" and veh.speed > 34.0:
                replay.request("OFF-TRACK MOMENT", delay=0.62)
        prev_off = fr["off_track"]

        slide_now = abs(math.degrees(veh.slip_angle)) if veh.speed > 8.0 else 0.0
        if slide_now > 23.0 and not veh.airborne:
            drift_moment_t += frame_dt
        elif drift_moment_t > 1.15 and slide_now < 15.0:
            if controller is not None and director.mode == "auto":
                replay.request("DRIFT REPLAY", delay=0.34)
            drift_moment_t = 0.0
        elif slide_now < 9.0:
            drift_moment_t = max(0.0, drift_moment_t - frame_dt * 2.0)

        weather.update(frame_dt)
        wet_target = 1.0 if wet_visual else float(weather.mode["wet"])
        session_visual.update(veh, fr, i_near, frame_dt, wet_target)
        post_impact = max(0.0, post_impact - frame_dt * 1.8)

        skids.update(veh, fr, frame_dt, brake=b_in)
        puffs.update(veh, fr, frame_dt, rng, wet=session_visual.wetness > 0.14)
        render_veh.interpolate(veh, accum / sim.dt)
        replay.capture(render_veh, b_in, mood_i, session_visual.wetness)
        replay.update(frame_dt)
        director.update(view, veh, i_near, frame_dt)
        render_flyby_camera = (director.on and director.shot in ("trackside", "apex")
                               and not replay.playing)
        replay_frame = replay.current()
        display_veh = replay.pose() if replay_frame is not None else render_veh
        if replay.playing:
            replay.apply_view(view, frame_dt)
        elif replay.just_cut:
            view.cut = True
        view.update(display_veh, display_veh.road_z, frame_dt)
        if replay.playing:
            view.audio_shake_x = view.audio_shake_y = 0.0
        else:
            view.audio_shake_x, view.audio_shake_y = audio_visuals.camera_vibration()
        if ai_layer.enabled and not replay.playing:
            ai_layer.update(agent, veh, trk)

        # live delta vs the best clean lap (same-progress comparison)
        delta_live = None
        if ref_p is not None and lap["t"] > 2.0 and fr["progress"] > 0.005:
            delta_live = lap["t"] - float(np.interp(fr["progress"], ref_p, ref_t))

        car_vx = veh.speed * math.cos(veh.yaw)
        car_vy = veh.speed * math.sin(veh.yaw)
        listener_vel = (car_vx, car_vy)
        # The director's trackside shot is the stand-still flyby camera: park
        # the listener so the spatial mixer can produce a real doppler sweep.
        if director.on and director.shot in ("trackside", "apex") and not replay.playing:
            listener_vel = (0.0, 0.0)
        # Viewer-only telemetry enriches sound without entering physics or the
        # Fable observation/reward fingerprint.
        veh._audio_brake = b_in
        veh._audio_clutch = float(getattr(veh, "clutch", 1.0))
        veh._audio_landing_force = float(getattr(veh, "landing_g", 0.0)) if not veh.airborne else 0.0
        veh._audio_surface = "wet" if wet_visual else ("grass" if fr["off_track"] else ("kerb" if getattr(veh, "_snd_kerb", 0) else "dry"))
        perspective = 1 if replay.playing or view.mode == "chase" else 0
        camera_cut = bool(view.cut)
        if replay.playing:
            audio.update((veh.x, veh.y), listener_vel, veh.yaw, [veh], [t_in],
                         brakes=[b_in], perspective=perspective, camera_cut=camera_cut)
        else:
            listener_yaw = -math.pi / 2.0 - view.rot
            audio.update((view.cx, view.cy), listener_vel, listener_yaw, [veh], [t_in],
                         brakes=[b_in], perspective=perspective, camera_cut=camera_cut)

        # --- render ---
        render_mood_i = replay_frame["mood_i"] if replay_frame is not None else mood_i
        render_wet = (replay_frame["wet"] if replay_frame is not None
                      else session_visual.wetness)
        render_brake = replay_frame["brake"] if replay_frame is not None else b_in
        render_mood = MOODS[render_mood_i]
        i0 = trk.nearest(display_veh.x, display_veh.y)
        props = draw_world(pygame, screen, view, geom, render_mood, i0, display_veh,
                           skids, drive_time, wet=render_wet, session=session_visual)
        cine.draw_clouds(pygame, screen, view, drive_time)
        draw_headlight_beams_v2(pygame, screen, view, display_veh, spec, render_mood)
        # painter: props behind the car first, car, then props in front
        car_sy = view.project(display_veh.x, display_veh.y, display_veh.road_z)[1]
        props.sort(key=lambda p: p[0])
        for p in props:
            if p[0] <= car_sy:
                draw_prop(pygame, screen, view, render_mood, p, drive_time)
        if not replay.playing:
            puffs.draw(pygame, screen, view)
            ai_layer.draw_world(pygame, screen, view, trk, render_mood)
        if show_beams and obs is not None and not replay.playing:
            for bd, (bx, by) in zip(obs.beams, obs.beam_points):
                p0 = view.project(display_veh.x, display_veh.y,
                                  display_veh.road_z + 0.4)
                p1 = view.project(bx, by, fr["z"] + 0.4)
                pygame.draw.line(screen, (90, 200, 220), p0, p1, 1)
        if not replay.playing:
            flyby_fx.draw(pygame, screen, view, display_veh, audio_visuals,
                          active=render_flyby_camera)
        body_vibration = ((0.0, 0.0) if replay.playing else
                          audio_visuals.body_vibration())
        draw_car_v2(pygame, screen, view, display_veh, spec, render_mood,
                    brake=render_brake, dirt=session_visual.car_dirt,
                    vibration=body_vibration)
        if not replay.playing:
            draw_exhaust_flame(pygame, screen, view, display_veh, spec,
                               audio_visuals.backfire * 0.14, rng)
            draw_audio_reactive_exhaust(pygame, screen, view, display_veh, spec,
                                        audio_visuals)
        for p in props:
            if p[0] > car_sy:
                draw_prop(pygame, screen, view, render_mood, p, drive_time)

        tension.draw(pygame, screen, view, display_veh, frame_dt,
                     quality=governor.quality)
        tint = render_mood["tint"]
        if tint:
            tint_key = tuple(tint)
            if tint_key not in mood_tints:
                mood_tints[tint_key] = pygame.Surface((W, H), pygame.SRCALPHA)
                mood_tints[tint_key].fill(tint)
            screen.blit(mood_tints[tint_key], (0, 0))
        cine.draw_grade(pygame, screen, render_mood, frame_no,
                        quality=governor.quality)
        weather.draw(pygame, screen)
        post_fx.apply(pygame, screen, display_veh.speed, impact=post_impact)
        pace = None
        if vref_arr is not None:
            pace = float(veh.speed - vref_arr[i0])
        hud.draw(pygame, screen, display_veh, render_mood, lap,
                 {"title": (title or f"SUPRA DRIFT V2 — {car.upper()}"),
                  "course": (str(prev_landmark).upper() if geom.is_nordschleife and prev_landmark
                             else track_name.upper() if track is None else "NORDSCHLEIFE"),
                  "hint": hint if controller is None else
                  "WATCHING — V director · P replay · I AI · H weather · O HQ FX · F3 performance · F11 fullscreen · R reset · ESC quit",
                  "landing_flash": landing_flash, "dt": frame_dt, "pace": pace,
                  "delta": delta_live, "progress": fr["progress"],
                  "limiter_gate": audio_visuals.limiter_gate,
                  "shift_wave": audio_visuals.shift_wave,
                  "engine_wave": list(audio_visuals.waveform),
                  "sector": prev_sec + 1, "clean": not lap["invalid"],
                  "wet": render_wet, "weather": weather.mode["label"],
                  "hq_post": post_fx.enabled, "replay": replay.label(),
                  "shot": replay.label() or director.label()})
        if not replay.playing:
            ai_layer.draw_hud(pygame, screen)
        perf.draw(pygame, screen, governor.label)

        pygame.display.flip()
        if max_frames and frame_no in (max_frames, max_frames // 2):
            out = os.environ.get("V2_SHOT")
            if out:
                pygame.image.save(screen, out.replace(".png", f"_{frame_no}.png"))

    try:
        audio.stop()
    except Exception:
        pass
    pygame.quit()
