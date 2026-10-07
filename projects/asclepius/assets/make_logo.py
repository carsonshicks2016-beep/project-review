"""Render the Asclepius mark.

The rod of Asclepius drawn the way a mid-century-futurist console would draw it:
thin amber vector strokes on a warm near-black panel, hexagonal frame, gauge
ticks, corner brackets, and a soft bloom over everything.

    uv run --with pillow python assets/make_logo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

S = 1024          # logical canvas; all geometry below is in these units
SS = 4            # supersample factor
N = S * SS
C = S / 2

OUT = Path(__file__).resolve().parent

# Console palette: warm black panel, amber phosphor, one hot accent.
BG_DEEP = (10, 8, 7)
BG_WARM = (48, 24, 9)
ROD_FILL = (16, 11, 7)
AMBER = (255, 158, 44)
AMBER_HI = (255, 220, 162)
AMBER_DIM = (150, 80, 21)
HOT = (255, 78, 26)


# --------------------------------------------------------------------------- #
# drawing helpers (logical units in, device units out)
# --------------------------------------------------------------------------- #

def px(p):
    return (p[0] * SS, p[1] * SS)


def line(d, pts, w, col, a=255):
    d.line([px(p) for p in pts], fill=col + (a,),
           width=max(1, round(w * SS)), joint="curve")


def dot(d, x, y, r, col, a=255):
    d.ellipse([(x - r) * SS, (y - r) * SS, (x + r) * SS, (y + r) * SS],
              fill=col + (a,))


def arc(d, r, a0, a1, w, col, a=255, cx=C, cy=C):
    box = [(cx - r) * SS, (cy - r) * SS, (cx + r) * SS, (cy + r) * SS]
    d.arc(box, a0, a1, fill=col + (a,), width=max(1, round(w * SS)))


def poly(d, pts, col, a=255):
    d.polygon([px(p) for p in pts], fill=col + (a,))


def tapered(d, path, col, a=255):
    """A stroke of varying width, laid down as overlapping discs."""
    for x, y, w in path:
        dot(d, x, y, w / 2, col, a)


def edges(path):
    """The two edge curves of a tapered tube, for drawing it hollow."""
    n = len(path)
    left, right = [], []
    for i, (x, y, w) in enumerate(path):
        a_, b_ = path[max(0, i - 1)], path[min(n - 1, i + 1)]
        tx, ty = norm(b_[0] - a_[0], b_[1] - a_[1])
        nx, ny = -ty, tx
        left.append((x + nx * w / 2, y + ny * w / 2))
        right.append((x - nx * w / 2, y - ny * w / 2))
    return left, right


def runs(coil):
    """Split the coil into contiguous behind-the-rod / in-front-of-it runs."""
    out, cur, sign = [], [], coil[0][3] >= 0
    for x, y, w, z in coil:
        s = z >= 0
        if s != sign and cur:
            out.append((sign, cur))
            cur, sign = [cur[-1]], s
        cur.append((x, y, w))
    if cur:
        out.append((sign, cur))
    return out


def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def norm(vx, vy):
    m = math.hypot(vx, vy) or 1.0
    return vx / m, vy / m


def hexagon(r, rot=90.0):
    return [(C + r * math.cos(math.radians(rot + 60 * i)),
             C - r * math.sin(math.radians(rot + 60 * i))) for i in range(6)]


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

HEX_R = 436
HEX_R2 = 404
ROD_X = C
ROD_TOP, ROD_BOT = 150, 902
ROD_W = 46

COIL_A = 158              # helix radius
# 3.25 turns puts both the tail and the head on the near side of the rod, so
# the coil starts and ends solid rather than mid-way through a hidden run.
COIL_TURNS = 3.25
COIL_BOT, COIL_TOP = 806, 374


def helix():
    """Sampled coil around the rod: (x, y, width, depth) per step.

    Depth is the z of a 3D helix, so segments with z < 0 pass behind the rod.
    """
    total = COIL_TURNS * 2 * math.pi
    # End the coil front-and-right, so the neck lifts away cleanly.
    phase = 0.80 - total
    out = []
    steps = 1400
    for i in range(steps + 1):
        u = i / steps
        t = u * total
        ang = t + phase
        x = ROD_X + COIL_A * math.sin(ang)
        y = COIL_BOT + (COIL_TOP - COIL_BOT) * u
        w = (10 + 24 * smoothstep(0.0, 0.13, u)) * (1 - 0.18 * smoothstep(0.72, 1.0, u))
        out.append((x, y, w, math.cos(ang)))
    return out


def bezier(p0, p1, p2, steps):
    for i in range(steps + 1):
        t = i / steps
        m = 1 - t
        yield (m * m * p0[0] + 2 * m * t * p1[0] + t * t * p2[0],
               m * m * p0[1] + 2 * m * t * p1[1] + t * t * p2[1])


def neck_and_head(coil):
    """Neck curve lifting off the last coil, plus the head polygon and eye."""
    x0, y0, w0, _ = coil[-1]
    xp, yp, _, _ = coil[-30]
    tx, ty = norm(x0 - xp, y0 - yp)

    p0 = (x0, y0)
    p1 = (x0 + tx * 88, y0 + ty * 88)
    p2 = (x0 + 26, y0 - 124)

    pts = list(bezier(p0, p1, p2, 160))
    neck = [(x, y, w0 * (1 - 0.22 * (i / len(pts))))
            for i, (x, y) in enumerate(pts)]

    # Head: an ellipse in the neck's frame, pinched toward the snout.
    hx, hy = pts[-1]
    dx, dy = norm(hx - pts[-14][0], hy - pts[-14][1])
    ux, uy = -dy, dx
    half, wide = 44, 23

    upper, lower = [], []
    for i in range(41):
        s = -1 + 2 * i / 40
        r = wide * math.sqrt(max(0.0, 1 - s * s)) * (1 - 0.52 * max(s, 0.0))
        cx_ = hx + dx * (half * s) + dx * half * 0.55
        cy_ = hy + dy * (half * s) + dy * half * 0.55
        upper.append((cx_ + ux * r, cy_ + uy * r))
        lower.append((cx_ - ux * r, cy_ - uy * r))
    head = upper + lower[::-1]

    eye = (hx + dx * half * 0.75 + ux * wide * 0.30,
           hy + dy * half * 0.75 + uy * wide * 0.30)
    return neck, head, eye


# --------------------------------------------------------------------------- #
# the mark
# --------------------------------------------------------------------------- #

def render(d, emissive: bool):
    """Draw the mark. `emissive` skips opaque dark fills so the bloom pass
    only ever blurs light, never holes."""

    # corner brackets
    for sx, sy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        ox, oy = C - sx * (C - 74), C - sy * (C - 74)
        line(d, [(ox + sx * 78, oy), (ox, oy), (ox, oy + sy * 78)], 7, AMBER_DIM, 200)

    # double hexagon frame
    h1, h2 = hexagon(HEX_R), hexagon(HEX_R2)
    line(d, h1 + [h1[0]], 8, AMBER, 205)
    line(d, h2 + [h2[0]], 3, AMBER_DIM, 165)
    for v in h1:
        dot(d, v[0], v[1], 10, AMBER_HI, 235)

    # gauge ticks stepping inward off each hexagon edge
    for i in range(6):
        a, b = h2[i], h2[(i + 1) % 6]
        ex, ey = norm(b[0] - a[0], b[1] - a[1])
        nx, ny = norm(C - (a[0] + b[0]) / 2, C - (a[1] + b[1]) / 2)
        for k in range(1, 10):
            t = k / 10
            mx, my = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
            ln, col, al = (26, AMBER, 190) if k == 5 else (13, AMBER_DIM, 150)
            line(d, [(mx, my), (mx + nx * ln, my + ny * ln)], 4, col, al)
        del ex, ey

    coil = helix()
    segments = runs(coil)

    # The far side of the coil, hollow — the way a schematic shows what the
    # rod is hiding.
    for front, seg in segments:
        if not front:
            for e in edges(seg):
                line(d, e, 5, AMBER_DIM, 210)

    # the rod itself, drawn as a wireframe tube
    half = ROD_W / 2
    if not emissive:
        d.rectangle([(ROD_X - half) * SS, ROD_TOP * SS,
                     (ROD_X + half) * SS, ROD_BOT * SS], fill=ROD_FILL + (255,))
    for s in (-1, 1):
        line(d, [(ROD_X + s * half, ROD_TOP), (ROD_X + s * half, ROD_BOT)], 7, AMBER, 240)
    line(d, [(ROD_X - half + 12, ROD_TOP + 9), (ROD_X - half + 12, ROD_BOT - 9)],
         4, AMBER_DIM, 155)
    for y in (ROD_TOP, ROD_BOT):
        line(d, [(ROD_X - half - 15, y), (ROD_X + half + 15, y)], 7, AMBER_HI, 240)

    # the near side, solid
    for front, seg in segments:
        if front:
            tapered(d, seg, AMBER, 255)

    neck, head, eye = neck_and_head(coil)
    tapered(d, neck, AMBER, 255)
    poly(d, head, AMBER, 255)
    if not emissive:
        dot(d, eye[0], eye[1], 8, ROD_FILL, 255)
    dot(d, eye[0], eye[1], 5, HOT, 255)


def background():
    g = 512
    yy, xx = np.mgrid[0:g, 0:g]
    r = np.sqrt(((xx - g / 2) / (g / 2)) ** 2 + ((yy - g / 2) / (g / 2)) ** 2)
    t = (np.clip(1 - r / 0.98, 0, 1) ** 2.3)[..., None]
    rgb = np.array(BG_DEEP) + (np.array(BG_WARM) - np.array(BG_DEEP)) * t
    img = Image.fromarray(rgb.astype("uint8"), "RGB")
    return img.resize((N, N), Image.BICUBIC)


def bloom(base, emis, radius, gain):
    g = emis.filter(ImageFilter.GaussianBlur(radius * SS))
    a = g.getchannel("A").point(lambda v: int(v * gain))
    lit = Image.composite(Image.merge("RGB", g.split()[:3]),
                          Image.new("RGB", g.size, (0, 0, 0)), a)
    return ImageChops.add(base, lit)


def main():
    art = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    render(ImageDraw.Draw(art), emissive=False)

    emis = Image.new("RGBA", (N, N), (0, 0, 0, 0))
    render(ImageDraw.Draw(emis), emissive=True)

    img = background()
    img = bloom(img, emis, 7, 0.42)
    img = bloom(img, emis, 26, 0.34)
    img.paste(art.convert("RGB"), (0, 0), art.getchannel("A"))

    master = img.resize((1024, 1024), Image.LANCZOS)
    master.save(OUT / "asclepius-logo-1024.png")
    master.resize((512, 512), Image.LANCZOS).save(OUT.parent / "logo.png")
    print("wrote assets/asclepius-logo-1024.png and logo.png")


if __name__ == "__main__":
    main()
