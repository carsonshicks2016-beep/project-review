/**
 * A hand-written, contract-valid WorldSpec so the renderer can run standalone
 * without the generation module (Agent C). Two variants are provided so the
 * crossfade/dissolve path can be exercised by swapping between them.
 */
import type { WorldSpec, ScatterItem } from "../contracts";

/** Cheap deterministic scatter so the demo doesn't depend on the gen core. */
function demoScatter(
  kinds: string[],
  count: number,
  seed: number,
  spread = 90,
): ScatterItem[] {
  // Small LCG — keeps the demo self-contained and deterministic.
  let s = (seed >>> 0) || 1;
  const rng = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0xffffffff;
  };
  const items: ScatterItem[] = [];
  for (let i = 0; i < count; i++) {
    items.push({
      kind: kinds[Math.floor(rng() * kinds.length)]!,
      x: (rng() * 2 - 1) * spread,
      z: (rng() * 2 - 1) * spread,
      scale: 0.6 + rng() * 1.8,
      rotation: rng() * Math.PI * 2,
    });
  }
  return items;
}

export const DEMO_WORLD: WorldSpec = {
  biome: "folk-forest",
  seed: 0x5eed1,
  terrain: { octaves: 5, amplitude: 9, frequency: 0.012, seed: 0x5eed1 },
  scatter: demoScatter(["pine", "rock", "shrub"], 220, 0x5eed1),
  timeOfDay: 0.32, // warm late afternoon
  weatherState: "clear",
  creaturePool: [
    { species: "firefly", count: 140, baseSpeed: 1.4, behavior: "drift" },
    { species: "sparrow", count: 36, baseSpeed: 3.0, behavior: "boids" },
    { species: "deer", count: 6, baseSpeed: 1.0, behavior: "wander" },
  ],
  palette: {
    primary: "#3f6b3a",
    secondary: "#8a6b3f",
    accent: "#ffd27f",
    bg: "#1a2230",
  },
  modifierCurves: {
    bassToLightning: [0.55, 1.0, 0.0, 1.0],
    energyToParticles: [0.0, 1.0, 0.2, 1.6],
    centroidToBrightness: [0.0, 1.0, 0.7, 1.8],
  },
};

/** Second world for demonstrating crossfades (an EDM-grid night). */
export const DEMO_WORLD_ALT: WorldSpec = {
  biome: "edm-grid",
  seed: 0xbeef2,
  terrain: { octaves: 4, amplitude: 5, frequency: 0.02, seed: 0xbeef2 },
  scatter: demoScatter(["neon-pillar", "rock"], 160, 0xbeef2),
  timeOfDay: 0.92, // night
  weatherState: "clear",
  creaturePool: [
    { species: "drone", count: 80, baseSpeed: 4.0, behavior: "boids" },
  ],
  palette: {
    primary: "#ff2db5",
    secondary: "#2d7bff",
    accent: "#00ffd5",
    bg: "#05030f",
  },
  modifierCurves: {
    bassToLightning: [0.45, 1.0, 0.0, 1.0],
    energyToParticles: [0.0, 1.0, 0.4, 2.2],
    centroidToBrightness: [0.0, 1.0, 0.9, 2.4],
  },
};
