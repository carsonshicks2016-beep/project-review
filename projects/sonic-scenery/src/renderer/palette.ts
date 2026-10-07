/**
 * Palette helpers — turn the WorldSpec hex palette into Three.js colors and
 * derive sky / fog / light tints from `timeOfDay`.
 */
import * as THREE from "three";
import type { Palette } from "../contracts";

export interface ResolvedPalette {
  primary: THREE.Color;
  secondary: THREE.Color;
  accent: THREE.Color;
  bg: THREE.Color;
  /** Sky color blended from bg + accent by time of day. */
  sky: THREE.Color;
  /** Fog color (slightly lighter than bg). */
  fog: THREE.Color;
  /** Directional "sun/moon" light color. */
  sun: THREE.Color;
  /** Ambient/hemisphere fill color. */
  ambient: THREE.Color;
}

function safeColor(hex: string, fallback = "#222222"): THREE.Color {
  try {
    return new THREE.Color(hex);
  } catch {
    return new THREE.Color(fallback);
  }
}

/**
 * `timeOfDay` is 0..1 (0 = midnight, 0.5 = noon). We map it to a daylight
 * factor with a smooth dawn/dusk curve so colors warm at the edges of day.
 */
export function daylight(timeOfDay: number): number {
  // Triangle peaking at noon, clamped — 0 at midnight, 1 at noon.
  const t = THREE.MathUtils.clamp(timeOfDay, 0, 1);
  return THREE.MathUtils.clamp(1 - Math.abs(t - 0.5) * 2, 0, 1);
}

export function resolvePalette(p: Palette, timeOfDay: number): ResolvedPalette {
  const primary = safeColor(p.primary, "#3f6b3a");
  const secondary = safeColor(p.secondary, "#8a6b3f");
  const accent = safeColor(p.accent, "#ffd27f");
  const bg = safeColor(p.bg, "#1a2230");

  const day = daylight(timeOfDay);

  // Sky: night = bg, day = bg lifted toward accent + a blue lift.
  const sky = bg.clone().lerp(accent, day * 0.35);
  sky.lerp(new THREE.Color(0x6fa8ff), day * 0.25);

  const fog = bg.clone().lerp(sky, 0.45);

  // Sun warms at dawn/dusk (low day), cools to white at noon; moon is cool.
  const warm = new THREE.Color(0xffb066);
  const noon = new THREE.Color(0xffffff);
  const moon = new THREE.Color(0x6678a8);
  const sun =
    day < 0.15
      ? moon.clone()
      : warm.clone().lerp(noon, THREE.MathUtils.smoothstep(day, 0.1, 0.9));

  const ambient = bg.clone().lerp(sky, 0.6).lerp(primary, 0.15);

  return { primary, secondary, accent, bg, sky, fog, sun, ambient };
}

/** [inMin, inMax, outMin, outMax] linear remap with clamping, per ModifierCurves. */
export function remap(
  value: number,
  curve: readonly [number, number, number, number],
): number {
  const [inMin, inMax, outMin, outMax] = curve;
  const span = inMax - inMin;
  const tRaw = span === 0 ? 0 : (value - inMin) / span;
  const t = THREE.MathUtils.clamp(tRaw, 0, 1);
  return outMin + (outMax - outMin) * t;
}
