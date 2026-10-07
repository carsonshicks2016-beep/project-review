#!/usr/bin/env python3
"""Render a seed (or any genome JSON) to a PNG so you can eyeball the creature.

    python3 scripts/view_creature.py                      # both seeds -> renders/
    python3 scripts/view_creature.py agent_zero quadruped
    python3 scripts/view_creature.py path/to/genome.json --out renders/foo.png

Offscreen render via mujoco.Renderer; PNG written with the stdlib only (no
Pillow/imageio needed). For an interactive view use mujoco.viewer.launch(model).
"""
import argparse
import os
import struct
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco

from personal_cambrian.seeds import SEEDS
from personal_cambrian.encoding import Genome
from personal_cambrian.morphogenesis import develop
from personal_cambrian.morphogenesis.to_mujoco import compile_morphology

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_png(path: str, rgb: np.ndarray) -> None:
    """Write an HxWx3 uint8 array to a PNG using only the standard library."""
    h, w, _ = rgb.shape
    scanlines = np.hstack([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)])
    raw = scanlines.astype(np.uint8).tobytes()

    def chunk(typ: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + typ + data
                + struct.pack(">I", zlib.crc32(typ + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(chunk(b"IHDR", ihdr))
        f.write(chunk(b"IDAT", zlib.compress(raw, 9)))
        f.write(chunk(b"IEND", b""))


def load_genome(spec: str) -> Genome:
    if spec in SEEDS:
        return SEEDS[spec]()
    with open(spec) as f:
        return Genome.from_json(f.read())


def render(genome: Genome, out: str, width=640, height=480) -> np.ndarray:
    model, _ = compile_morphology(develop(genome), add_floor=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # lift the creature so its lowest point sits just above the floor
    cre = [i for i in range(model.ngeom) if model.geom(i).name != "floor"]
    zmin = min(float(data.geom_xpos[i, 2] - model.geom_rbound[i]) for i in cre)
    data.qpos[2] += 0.03 - zmin
    mujoco.mj_forward(model, data)

    # frame a camera on the creature
    pts = data.geom_xpos[cre]
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat = pts.mean(axis=0)
    cam.distance = 2.2 * (float(np.linalg.norm(pts.max(0) - pts.min(0))) + 0.4)
    cam.azimuth, cam.elevation = 140, -20

    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    write_png(out, img)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("targets", nargs="*", default=list(SEEDS),
                    help="seed name(s) or genome JSON path(s)")
    ap.add_argument("--out", default=None, help="output PNG (single target only)")
    args = ap.parse_args()

    targets = args.targets or list(SEEDS)
    for t in targets:
        label = t if t in SEEDS else os.path.splitext(os.path.basename(t))[0]
        out = args.out or os.path.join(ROOT, "renders", f"{label}.png")
        img = render(load_genome(t), out)
        print(f"{label:12s} -> {out}  ({img.shape[1]}x{img.shape[0]}, "
              f"var={img.var():.0f})")


if __name__ == "__main__":
    main()
