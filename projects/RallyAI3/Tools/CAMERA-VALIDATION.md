# Viewer Camera Validation

`SpectatorDirector` owns per-frame field of view. The half-second presentation
refresh must never reset it; that previously created periodic zoom pulses in
all watch modes. Managed viewers disable desktop VSync so their existing 30 FPS
cap is respected. This does not change training timestep or headless rendering.

For an explicitly requested local diagnostic, set `RALLY_CAMERA_TRACE` to an
absolute JSONL path when launching a viewer. The trace is disabled by default
and stops at 1,200 frames. It records render/physics positions, frame intervals,
camera mode, FOV, and frame pacing settings. It is not an evaluation trajectory.

```sh
.venv/bin/python Tools/camera_trace.py /absolute/path/to/trace.jsonl
```

The summary reports per-mode FOV discontinuities and measured frame intervals.
Cuts between different modes or effective cinematic shots, and their first
half-second of settling, are excluded. Deliberate cinematic shot changes must
be reviewed separately from unexpected zoom resets.
