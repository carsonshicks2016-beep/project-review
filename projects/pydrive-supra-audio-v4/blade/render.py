"""Offscreen rendering helpers for BLADE fights — a tracking camera + mp4 writer."""
from __future__ import annotations

import numpy as np
import mujoco


def tracking_camera(distance=6.2, elevation=-18.0, azimuth=90.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    cam.lookat[:] = [0, 0, 0.9]
    return cam


def aim_camera(cam, env, smooth=0.15):
    a = env.data.xpos[env.fi["a_"]["torso"]]
    b = env.data.xpos[env.fi["b_"]["torso"]]
    mid = (a + b) / 2.0
    target = np.array([mid[0], mid[1], 0.95])
    cam.lookat[:] = (1 - smooth) * np.array(cam.lookat) + smooth * target
    # slowly orbit so the duel reads in 3D
    cam.azimuth = (cam.azimuth + 0.15) % 360


def save_mp4(frames, path, fps=40):
    import imageio
    imageio.mimsave(path, frames, fps=fps, quality=8, macro_block_size=8)
