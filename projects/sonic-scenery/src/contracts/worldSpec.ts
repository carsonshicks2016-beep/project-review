/**
 * WorldSpec — produced by the generation core from a TrackContext (+ optional
 * live AudioFrame prior). Fully describes a renderable world. Deterministic
 * given the same trackId.
 *
 * Owner: Agent C (src/generation/). Consumers: Agent D/E/F.
 */
import type { Palette } from "./trackContext";

export type Biome =
  | "lofi-rooftop"
  | "classical-peaks"
  | "edm-grid"
  | "metal-volcanic"
  | "folk-forest"
  | "jazz-noir"
  | "ambient-dream"
  | "pop-coast";

export type WeatherState =
  | "clear"
  | "cloudy"
  | "drizzle"
  | "rain"
  | "storm"
  | "snow"
  | "fog"
  | "ash";

export interface TerrainParams {
  octaves: number;
  amplitude: number;
  frequency: number;
  seed: number;
}

export interface ScatterItem {
  kind: string; // e.g. "pine", "rock", "neon-pillar"
  x: number;
  z: number;
  scale: number;
  rotation: number;
}

export interface CreatureSpec {
  species: string; // resolved from the biome creature pool
  count: number; // f(energy, seed)
  baseSpeed: number; // f(tempo/energy)
  behavior: "boids" | "wander" | "drift";
}

/**
 * How live AudioFrame values remap onto scene parameters. Each curve is a
 * simple [inMin, inMax, outMin, outMax] remap applied per frame by the renderer.
 */
export interface ModifierCurves {
  bassToLightning: [number, number, number, number];
  energyToParticles: [number, number, number, number];
  centroidToBrightness: [number, number, number, number];
}

export interface WorldSpec {
  biome: Biome;
  seed: number;
  terrain: TerrainParams;
  scatter: ScatterItem[];
  /** 0..1 (0 = midnight, 0.5 = noon). */
  timeOfDay: number;
  weatherState: WeatherState;
  creaturePool: CreatureSpec[];
  palette: Palette;
  modifierCurves: ModifierCurves;
}
