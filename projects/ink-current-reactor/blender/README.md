# Frutiger Aero Blender Loop

This folder contains a self-building Blender scene for a glossy early-2000s Frutiger Aero loop rendered in Cycles.

There is also a saved Blender scene at [frutiger_aero_loop.blend](/Users/REVIEW_USER/Documents/New project/blender/frutiger_aero_loop.blend).

## Run It

1. Open Blender.
2. Open the Scripting workspace.
3. Load [frutiger_aero_loop.py](/Users/REVIEW_USER/Documents/New project/blender/frutiger_aero_loop.py).
4. Press `Run Script`.

The script builds:

- a glass pond bowl
- shimmering water
- mossy ground forms
- translucent leaves
- a floating orb and aqua ribbon
- orbiting droplets
- lighting and camera

## Render Settings

- Engine: `Cycles`
- Length: frames `1-240`
- FPS: `24`
- Output: `1920x1080`

If the `.blend` file has already been saved, the render path is set next to it.

- When Blender exposes `FFMPEG`, the script targets `frutiger_aero_loop.mp4`.
- On builds without `FFMPEG` in the Python render API, it falls back to a PNG sequence.

## Tweak Points

- Change palette and density near the top of the material builders in [frutiger_aero_loop.py](/Users/REVIEW_USER/Documents/New project/blender/frutiger_aero_loop.py).
- Increase `scene.cycles.samples` in `configure_scene()` for cleaner glass.
- If you want a faster preview, drop samples to `64` and disable depth of field.

## Note

This scene was validated in Blender `5.0.1` in background mode and saved out as a `.blend` file from that run.
