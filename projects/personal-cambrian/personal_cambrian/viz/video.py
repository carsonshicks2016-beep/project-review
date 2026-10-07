"""Stitch a rendered PNG frame sequence into a replay video (ROADMAP Stage 10.2).

Blender on some builds (incl. the one here) ships WITHOUT ffmpeg, so the animation is
rendered as a PNG sequence and assembled into a video on the Python side: an MP4 if an
imageio ffmpeg backend is present, otherwise a GIF (always works via Pillow). Either
is a valid locomotion replay.
"""
from __future__ import annotations

import glob
import os


def _frames(frames_dir: str) -> list:
    return sorted(glob.glob(os.path.join(frames_dir, "*.png")))


def stitch(frames_dir: str, out_stem: str, *, fps: int = 30, cleanup: bool = True) -> str:
    """Assemble the PNG frames in `frames_dir` into `out_stem.(mp4|gif)`. Returns the
    path written. Prefers MP4 (imageio-ffmpeg), falls back to GIF (Pillow)."""
    paths = _frames(frames_dir)
    if not paths:
        raise FileNotFoundError(f"no PNG frames in {frames_dir}")

    out = None
    try:                                     # MP4 if an imageio ffmpeg backend exists
        import imageio.v2 as imageio
        import numpy as np
        from PIL import Image
        out = out_stem + ".mp4"
        with imageio.get_writer(out, fps=fps, macro_block_size=None) as w:
            for p in paths:
                w.append_data(np.asarray(Image.open(p).convert("RGB")))
    except Exception:                        # noqa: BLE001 -- no ffmpeg backend -> GIF
        from PIL import Image
        out = out_stem + ".gif"
        imgs = [Image.open(p).convert("RGB") for p in paths]
        imgs[0].save(out, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / max(fps, 1)), loop=0)

    if cleanup:
        for p in paths:
            os.remove(p)
        if not os.listdir(frames_dir):
            os.rmdir(frames_dir)
    return out
