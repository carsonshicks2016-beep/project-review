# Agent E — Weather system

**Files:** `src/weather/`  ·  **Contract in:** `WorldSpec`, `AudioFrame`
·  **Depends on:** Agent D scene conventions

## Goal
Implement the `WeatherState` machine and its visuals.

## Scope
- States: `clear, cloudy, drizzle, rain, storm, snow, fog, ash`.
- GPU-instanced precipitation; volumetric-ish fog; lightning flashes triggered
  on `bass` transients in `storm`.
- Intensity ramps from live `AudioFrame.rms`/`energy`; smooth transitions
  between states (no popping).

## Acceptance
- Each state renders distinctly and performs (capped particle counts, 60 fps).
- Lightning fires on bass onsets; intensity tracks energy; offline audio →
  steady baseline weather.
