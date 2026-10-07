/**
 * Biome registry — the authoritative genre → biome map and per-biome generation
 * priors (terrain shape, scatter kinds, creature pool, weather options,
 * time-of-day window). Follows PLAN.md §4.
 *
 * Pure data + pure lookup helpers. No randomness, no I/O.
 */
import type { Biome, WeatherState } from "../contracts";

export interface BiomeProfile {
  biome: Biome;
  /** Terrain heightfield priors (jittered per-seed in generationCore). */
  terrain: {
    octaves: number;
    /** Base vertical amplitude. */
    amplitude: number;
    /** Base noise frequency. */
    frequency: number;
  };
  /** Object kinds the Poisson-disk scatter can place. */
  scatterKinds: string[];
  /** Mean number of scattered objects (jittered per-seed). */
  scatterDensity: number;
  /** Minimum spacing for Poisson-disk sampling (world units). */
  scatterRadius: number;
  /** Candidate creature species for this biome (PRNG selects a subset). */
  creatureSpecies: string[];
  /** Default movement style for this biome's creatures. */
  creatureBehavior: "boids" | "wander" | "drift";
  /**
   * Weather options ordered by ascending energy. A low-energy track picks an
   * early entry; a high-energy track picks a later one.
   */
  weatherByEnergy: WeatherState[];
  /** Time-of-day window [min, max] in 0..1 (0 = midnight, 0.5 = noon). */
  timeOfDay: [number, number];
}

/**
 * Genre → biome map (PLAN.md §4). Keys are lowercase substrings matched against
 * Spotify artist genres, which are often compound (e.g. "melodic death metal").
 * Order matters: earlier, more-specific keys win.
 */
export const GENRE_TO_BIOME: ReadonlyArray<readonly [string, Biome]> = [
  // lo-fi / chillhop
  ["lo-fi", "lofi-rooftop"],
  ["lofi", "lofi-rooftop"],
  ["chillhop", "lofi-rooftop"],
  ["chillwave", "lofi-rooftop"],
  ["chill", "lofi-rooftop"],
  // classical
  ["classical", "classical-peaks"],
  ["orchestra", "classical-peaks"],
  ["baroque", "classical-peaks"],
  ["opera", "classical-peaks"],
  ["piano", "classical-peaks"],
  // edm / electronic
  ["edm", "edm-grid"],
  ["electronic", "edm-grid"],
  ["house", "edm-grid"],
  ["techno", "edm-grid"],
  ["trance", "edm-grid"],
  ["dubstep", "edm-grid"],
  ["synthwave", "edm-grid"],
  ["electro", "edm-grid"],
  // metal
  ["metal", "metal-volcanic"],
  ["hardcore", "metal-volcanic"],
  ["thrash", "metal-volcanic"],
  ["death", "metal-volcanic"],
  ["doom", "metal-volcanic"],
  // folk / acoustic
  ["folk", "folk-forest"],
  ["acoustic", "folk-forest"],
  ["bluegrass", "folk-forest"],
  ["singer-songwriter", "folk-forest"],
  ["americana", "folk-forest"],
  // jazz
  ["jazz", "jazz-noir"],
  ["bebop", "jazz-noir"],
  ["swing", "jazz-noir"],
  ["blues", "jazz-noir"],
  ["soul", "jazz-noir"],
  // ambient
  ["ambient", "ambient-dream"],
  ["drone", "ambient-dream"],
  ["new age", "ambient-dream"],
  ["soundscape", "ambient-dream"],
  // pop
  ["pop", "pop-coast"],
  ["dance pop", "pop-coast"],
  ["indie pop", "pop-coast"],
  ["k-pop", "pop-coast"],
];

export const ALL_BIOMES: readonly Biome[] = [
  "lofi-rooftop",
  "classical-peaks",
  "edm-grid",
  "metal-volcanic",
  "folk-forest",
  "jazz-noir",
  "ambient-dream",
  "pop-coast",
];

export const BIOME_PROFILES: Readonly<Record<Biome, BiomeProfile>> = {
  "lofi-rooftop": {
    biome: "lofi-rooftop",
    terrain: { octaves: 3, amplitude: 4, frequency: 0.05 },
    scatterKinds: ["ac-unit", "potted-plant", "antenna", "neon-sign", "water-tower"],
    scatterDensity: 60,
    scatterRadius: 3.5,
    creatureSpecies: ["cat", "pigeon", "sparrow", "moth"],
    creatureBehavior: "wander",
    weatherByEnergy: ["clear", "cloudy", "drizzle", "rain"],
    timeOfDay: [0.78, 0.9], // dusk
  },
  "classical-peaks": {
    biome: "classical-peaks",
    terrain: { octaves: 6, amplitude: 48, frequency: 0.012 },
    scatterKinds: ["pine", "boulder", "snow-cap", "ridge-grass"],
    scatterDensity: 90,
    scatterRadius: 4,
    creatureSpecies: ["crane", "deer", "eagle", "ibex"],
    creatureBehavior: "boids",
    weatherByEnergy: ["clear", "fog", "cloudy", "snow"],
    timeOfDay: [0.2, 0.32], // golden dawn
  },
  "edm-grid": {
    biome: "edm-grid",
    terrain: { octaves: 2, amplitude: 6, frequency: 0.08 },
    scatterKinds: ["neon-pillar", "grid-tile", "laser-spire", "hologram-cube"],
    scatterDensity: 120,
    scatterRadius: 2.5,
    creatureSpecies: ["drone", "glow-orb", "synth-bird"],
    creatureBehavior: "boids",
    weatherByEnergy: ["clear", "cloudy", "rain", "storm"],
    timeOfDay: [0.85, 0.98], // synthwave night
  },
  "metal-volcanic": {
    biome: "metal-volcanic",
    terrain: { octaves: 5, amplitude: 40, frequency: 0.02 },
    scatterKinds: ["obsidian-shard", "lava-rock", "dead-tree", "ash-mound"],
    scatterDensity: 80,
    scatterRadius: 3.5,
    creatureSpecies: ["raven", "bat", "ember-moth"],
    creatureBehavior: "boids",
    weatherByEnergy: ["cloudy", "ash", "storm", "storm"],
    timeOfDay: [0.04, 0.14], // pre-dawn dark
  },
  "folk-forest": {
    biome: "folk-forest",
    terrain: { octaves: 4, amplitude: 12, frequency: 0.03 },
    scatterKinds: ["oak", "fern", "wildflower", "log", "mushroom"],
    scatterDensity: 140,
    scatterRadius: 2.8,
    creatureSpecies: ["deer", "firefly", "rabbit", "songbird"],
    creatureBehavior: "wander",
    weatherByEnergy: ["clear", "cloudy", "drizzle", "rain"],
    timeOfDay: [0.4, 0.55], // bright meadow midday
  },
  "jazz-noir": {
    biome: "jazz-noir",
    terrain: { octaves: 2, amplitude: 3, frequency: 0.06 },
    scatterKinds: ["lamppost", "puddle", "trash-can", "fire-escape", "neon-sign"],
    scatterDensity: 70,
    scatterRadius: 3,
    creatureSpecies: ["alley-cat", "rat", "moth", "pigeon"],
    creatureBehavior: "wander",
    weatherByEnergy: ["fog", "drizzle", "rain", "storm"],
    timeOfDay: [0.92, 0.99], // deep night
  },
  "ambient-dream": {
    biome: "ambient-dream",
    terrain: { octaves: 3, amplitude: 18, frequency: 0.015 },
    scatterKinds: ["floating-crystal", "soft-monolith", "glow-reed", "cloud-shelf"],
    scatterDensity: 50,
    scatterRadius: 5,
    creatureSpecies: ["jellyfish-drifter", "luminescent-ray", "spore"],
    creatureBehavior: "drift",
    weatherByEnergy: ["clear", "fog", "cloudy", "snow"],
    timeOfDay: [0.6, 0.75], // aurora evening
  },
  "pop-coast": {
    biome: "pop-coast",
    terrain: { octaves: 3, amplitude: 8, frequency: 0.04 },
    scatterKinds: ["palm", "beach-umbrella", "shell", "dune-grass", "rock"],
    scatterDensity: 100,
    scatterRadius: 3,
    creatureSpecies: ["butterfly", "seagull", "crab", "songbird"],
    creatureBehavior: "boids",
    weatherByEnergy: ["clear", "clear", "cloudy", "drizzle"],
    timeOfDay: [0.42, 0.58], // pastel midday
  },
};

/**
 * Resolve a biome from Spotify artist genres. Returns null when nothing matches
 * (caller then uses the audio-prior mood fallback). Never throws.
 */
export function biomeFromGenres(genres: readonly string[] | undefined): Biome | null {
  if (!genres || genres.length === 0) return null;
  const normalized = genres.map((g) => g.toLowerCase().trim()).filter((g) => g.length > 0);
  for (const [needle, biome] of GENRE_TO_BIOME) {
    for (const g of normalized) {
      if (g.includes(needle)) return biome;
    }
  }
  return null;
}
