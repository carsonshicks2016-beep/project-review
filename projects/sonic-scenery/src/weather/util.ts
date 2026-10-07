/**
 * Small shared helpers for the weather system: color parsing, easing, and a
 * critically-damped smoothing helper used for pop-free intensity ramps.
 */
import * as THREE from "three";
import type { Palette } from "../contracts";

/** Clamp a value to [min, max]. */
export function clamp(v: number, min = 0, max = 1): number {
  return v < min ? min : v > max ? max : v;
}

/** Linear interpolate. */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Linear remap from [inMin,inMax] to [outMin,outMax], clamped to the out range. */
export function remap(
  v: number,
  inMin: number,
  inMax: number,
  outMin: number,
  outMax: number
): number {
  if (inMax === inMin) return outMin;
  const t = clamp((v - inMin) / (inMax - inMin));
  return outMin + (outMax - outMin) * t;
}

/**
 * Frame-rate independent exponential smoothing. `rate` is roughly "how fast"
 * (higher = snappier). Returns the new smoothed value moving `current` toward
 * `target`. Stable for any dt.
 */
export function damp(current: number, target: number, rate: number, dt: number): number {
  return lerp(current, target, 1 - Math.exp(-rate * dt));
}

/** Parse a palette hex string into a THREE.Color (falls back to white). */
export function paletteColor(hex: string | undefined, fallback = "#ffffff"): THREE.Color {
  try {
    return new THREE.Color(hex ?? fallback);
  } catch {
    return new THREE.Color(fallback);
  }
}

/**
 * A reasonable "ambient" tint for a palette: the background color, nudged
 * toward the primary so fog/precipitation feel native to the world rather than
 * a flat gray.
 */
export function ambientTint(palette: Palette): THREE.Color {
  const bg = paletteColor(palette?.bg, "#1a1f2b");
  const primary = paletteColor(palette?.primary, "#8090a0");
  return bg.clone().lerp(primary, 0.35);
}
