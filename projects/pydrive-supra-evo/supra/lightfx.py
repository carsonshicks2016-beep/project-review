"""Shared lightweight lighting effects for the 2D car renderers."""
from __future__ import annotations

import math


def _finite_point(pt):
    if pt is None or len(pt) != 2:
        return None
    x, y = float(pt[0]), float(pt[1])
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _smoothstep(t):
    t = _clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _scaled_point(pt, scale):
    return pt[0] * scale, pt[1] * scale


def _draw_beam_band(pygame, surf, src, axis, normal, length, near_hw, far_hw,
                    color, max_alpha, strips, width_scale, end_t, falloff):
    if max_alpha <= 0 or length <= 1.0 or strips <= 0:
        return
    r, g, b = color
    for i in range(strips):
        t0 = end_t * i / strips
        t1 = end_t * (i + 1) / strips
        tm = (t0 + t1) * 0.5
        ease0 = _smoothstep(t0)
        ease1 = _smoothstep(t1)
        w0 = (near_hw + (far_hw - near_hw) * ease0) * width_scale
        w1 = (near_hw + (far_hw - near_hw) * ease1) * width_scale
        fade_in = _clamp(tm / 0.08, 0.0, 1.0)
        fade_out = (1.0 - tm / max(end_t, 0.001)) ** falloff
        alpha = int(max_alpha * fade_in * fade_out)
        if alpha <= 0:
            continue
        c0 = (src[0] + axis[0] * length * t0, src[1] + axis[1] * length * t0)
        c1 = (src[0] + axis[0] * length * t1, src[1] + axis[1] * length * t1)
        pts = [
            (int(c0[0] + normal[0] * w0), int(c0[1] + normal[1] * w0)),
            (int(c1[0] + normal[0] * w1), int(c1[1] + normal[1] * w1)),
            (int(c1[0] - normal[0] * w1), int(c1[1] - normal[1] * w1)),
            (int(c0[0] - normal[0] * w0), int(c0[1] - normal[1] * w0)),
        ]
        pygame.draw.polygon(surf, (r, g, b, alpha), pts)


def _draw_single_headlight_beam(pygame, beam, src, far, near_hw, far_hw,
                                intensity, strips):
    dx = far[0] - src[0]
    dy = far[1] - src[1]
    length = math.hypot(dx, dy)
    if length <= 1.0:
        return

    axis = (dx / length, dy / length)
    normal = (-axis[1], axis[0])
    for color, alpha, layer_strips, width_scale, end_t, falloff in (
        ((110, 126, 188), int(6 * intensity), strips, 1.42, 1.0, 1.9),
        ((132, 150, 205), int(11 * intensity), strips, 1.02, 1.0, 2.2),
        ((255, 226, 170), int(25 * intensity), strips, 0.58, 0.88, 2.65),
        ((255, 246, 205), int(20 * intensity), max(12, strips // 2), 0.24, 0.46, 3.1),
    ):
        layer = pygame.Surface(beam.get_size(), pygame.SRCALPHA)
        _draw_beam_band(pygame, layer, src, axis, normal, length, near_hw, far_hw,
                        color, alpha, layer_strips, width_scale, end_t, falloff)
        beam.blit(layer, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)


def draw_dual_headlight_beam(pygame, target, left_src, right_src, far_center,
                             far_half_width_px, *, intensity=1.0,
                             source_radius_px=5.0, strips=48,
                             blur_scale=0.5):
    """Draw two cinematic headlight beams in screen space."""
    lpt = _finite_point(left_src)
    rpt = _finite_point(right_src)
    fpt = _finite_point(far_center)
    if lpt is None or rpt is None or fpt is None:
        return

    width, height = target.get_size()
    if width <= 0 or height <= 0:
        return

    intensity = _clamp(float(intensity), 0.0, 1.5)
    if intensity <= 0.0:
        return

    blur_scale = _clamp(float(blur_scale), 0.25, 1.0)
    sw = max(1, int(width * blur_scale))
    sh = max(1, int(height * blur_scale))
    scale = blur_scale

    left = _scaled_point(lpt, scale)
    right = _scaled_point(rpt, scale)
    far = _scaled_point(fpt, scale)
    src_center = ((left[0] + right[0]) * 0.5, (left[1] + right[1]) * 0.5)

    dx = far[0] - src_center[0]
    dy = far[1] - src_center[1]
    base_length = math.hypot(dx, dy)
    if base_length < 10.0 * scale:
        return

    base_axis = (dx / base_length, dy / base_length)
    base_normal = (-base_axis[1], base_axis[0])
    lamp_gap = math.hypot(left[0] - right[0], left[1] - right[1])
    near_hw = max(1.8 * scale, source_radius_px * 0.52 * scale, lamp_gap * 0.10)
    requested_far_hw = abs(float(far_half_width_px)) * scale
    far_hw = _clamp(requested_far_hw * 0.38, near_hw * 2.1,
                    max(near_hw * 2.2, base_length * 0.18))

    beam = pygame.Surface((sw, sh), pygame.SRCALPHA)

    for lamp in (left, right):
        lateral = ((lamp[0] - src_center[0]) * base_normal[0] +
                   (lamp[1] - src_center[1]) * base_normal[1])
        far_lamp = (far[0] + base_normal[0] * lateral * 0.72,
                    far[1] + base_normal[1] * lateral * 0.72)
        _draw_single_headlight_beam(pygame, beam, lamp, far_lamp, near_hw, far_hw,
                                    intensity, strips)

    glow_r = max(1, int(source_radius_px * scale))
    glow = pygame.Surface((sw, sh), pygame.SRCALPHA)
    for lamp in (left, right):
        pygame.draw.circle(glow, (255, 235, 185, int(40 * intensity)),
                           (int(lamp[0]), int(lamp[1])), glow_r * 2)
        pygame.draw.circle(glow, (255, 250, 220, int(58 * intensity)),
                           (int(lamp[0]), int(lamp[1])), glow_r)
    beam.blit(glow, (0, 0), special_flags=pygame.BLEND_RGBA_ADD)

    if scale < 0.999:
        beam = pygame.transform.smoothscale(beam, (width, height))
    target.blit(beam, (0, 0))
