/**
 * Agent C — Generation core (pure, deterministic, testable).
 * Brief: docs/tasks/agent-C-generation.md
 *
 *  - hash(trackId) -> seed; seeded PRNG drives terrain/scatter/creatures/sky.
 *  - genres -> biome, with a live-audio mood fallback when genres are sparse.
 *  - album-art palette tints the world.
 *
 * Same trackId MUST always produce the same WorldSpec.
 *
 * PURITY: no I/O, no Date.now(), no Math.random(). All variation flows from the
 * seeded PRNG (seedrandom) keyed on seedFromTrackId(ctx.trackId). The optional
 * AudioFrame prior only influences the mood fallback + a couple of intensity
 * choices; it is read, never mutated.
 */
import seedrandom from "seedrandom";
import type {
  AudioFrame,
  Biome,
  CreatureSpec,
  ScatterItem,
  TerrainParams,
  TrackContext,
  WeatherState,
  WorldSpec,
} from "../contracts";
import { seedFromTrackId } from "./hash";
import { BIOME_PROFILES, biomeFromGenres, type BiomeProfile } from "./biomes";
import { poissonDisk } from "./scatter";

export { seedFromTrackId } from "./hash";

/** World area half-extent (world units). Terrain + scatter live in [-HALF, HALF]. */
const WORLD_HALF = 60;

/** Linear interpolation. */
function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** Clamp to [lo, hi]. */
function clamp(v: number, lo: number, hi: number): number {
  return v < lo ? lo : v > hi ? hi : v;
}

/**
 * Mood fallback: pick a biome from the live audio prior when genres are
 * empty/ambiguous. Uses energy (rms) on one axis and brightness (centroid) on
 * the other. Deterministic given the same prior; with no prior at all we fall
 * back through the seed so a world is still produced. Never throws.
 */
export function biomeFromAudioPrior(prior: AudioFrame | undefined, seed: number): Biome {
  // No prior: derive a stable pseudo-mood from the seed itself so unknown-genre
  // tracks with no audio still spread across biomes deterministically.
  const energy = prior ? clamp(prior.rms, 0, 1) : ((seed % 1000) / 1000) * 0.6 + 0.2;
  const brightness = prior
    ? clamp(prior.centroid, 0, 1)
    : (((seed >>> 10) % 1000) / 1000) * 0.6 + 0.2;

  // 2x2 mood quadrants → biome, with energy extremes overriding.
  const low = 0.35;
  const high = 0.66;

  if (energy >= high) {
    // Loud: bright -> edm, dark -> metal.
    return brightness >= 0.5 ? "edm-grid" : "metal-volcanic";
  }
  if (energy <= low) {
    // Quiet: bright -> ambient dream, dark/intimate -> jazz noir.
    return brightness >= 0.5 ? "ambient-dream" : "jazz-noir";
  }
  // Mid energy.
  if (brightness >= high) return "pop-coast"; // bright + lively
  if (brightness <= low) return "lofi-rooftop"; // warm + mellow
  return brightness >= 0.5 ? "folk-forest" : "classical-peaks";
}

/** Resolve the biome: genres first (PLAN §4), then audio-prior mood fallback. */
function resolveBiome(ctx: TrackContext, prior: AudioFrame | undefined, seed: number): Biome {
  return biomeFromGenres(ctx.genres) ?? biomeFromAudioPrior(prior, seed);
}

/**
 * Layered-simplex terrain params. The heightfield itself is evaluated by the
 * renderer (Agent D) using simplex-noise seeded on TerrainParams.seed; here we
 * only choose deterministic params jittered around the biome priors.
 */
function buildTerrain(profile: BiomeProfile, rng: seedrandom.PRNG, seed: number): TerrainParams {
  const { octaves, amplitude, frequency } = profile.terrain;
  return {
    octaves: octaves + Math.floor(rng() * 2), // +0..1 octaves
    amplitude: amplitude * lerp(0.85, 1.2, rng()),
    frequency: frequency * lerp(0.85, 1.25, rng()),
    seed,
  };
}

/** Poisson-disk scatter, each point assigned a biome-appropriate kind. */
function buildScatter(
  profile: BiomeProfile,
  rng: seedrandom.PRNG,
  energy: number,
): ScatterItem[] {
  // Energy nudges density up (PLAN: energy -> element density).
  const densityScale = lerp(0.8, 1.25, clamp(energy, 0, 1));
  const maxPoints = Math.round(profile.scatterDensity * densityScale);

  const points = poissonDisk(rng, {
    half: WORLD_HALF,
    radius: profile.scatterRadius,
    maxPoints,
  });

  const kinds = profile.scatterKinds;
  return points.map((p) => {
    const kind = kinds[Math.floor(rng() * kinds.length)] ?? kinds[0]!;
    return {
      kind,
      x: p.x,
      z: p.z,
      scale: lerp(0.6, 1.6, rng()),
      rotation: rng() * Math.PI * 2,
    };
  });
}

/** Select a creature subset for the biome and size/speed it from energy. */
function buildCreaturePool(
  profile: BiomeProfile,
  rng: seedrandom.PRNG,
  energy: number,
): CreatureSpec[] {
  const species = profile.creatureSpecies;
  // Pick 2..min(3,N) distinct species deterministically.
  const wanted = Math.min(species.length, 2 + Math.floor(rng() * 2));
  const pool = [...species];
  const chosen: string[] = [];
  for (let i = 0; i < wanted && pool.length > 0; i++) {
    const idx = Math.floor(rng() * pool.length);
    chosen.push(pool.splice(idx, 1)[0]!);
  }

  const e = clamp(energy, 0, 1);
  return chosen.map((sp) => ({
    species: sp,
    // count = f(energy, seed); speed = f(energy).
    count: Math.round(lerp(6, 40, e) * lerp(0.7, 1.3, rng())),
    baseSpeed: Number(lerp(0.4, 2.2, e).toFixed(4)),
    behavior: profile.creatureBehavior,
  }));
}

/** Time of day within the biome window, jittered by the seed. */
function buildTimeOfDay(profile: BiomeProfile, rng: seedrandom.PRNG): number {
  const [lo, hi] = profile.timeOfDay;
  return Number(lerp(lo, hi, rng()).toFixed(4));
}

/** Pick weather from the biome's energy-ordered options. */
function buildWeather(profile: BiomeProfile, energy: number): WeatherState {
  const opts = profile.weatherByEnergy;
  const i = clamp(Math.floor(clamp(energy, 0, 0.999) * opts.length), 0, opts.length - 1);
  return opts[i]!;
}

/**
 * Effective energy used for density/weather/creatures. Prefers the live audio
 * prior (rms); falls back to a stable per-seed pseudo-energy when absent so the
 * output stays deterministic without a prior.
 */
function effectiveEnergy(prior: AudioFrame | undefined, seed: number): number {
  if (prior) return clamp(prior.rms, 0, 1);
  return ((seed >>> 5) % 1000) / 1000;
}

/**
 * Deterministically turn a track into a fully-described world.
 * Same ctx.trackId ⇒ deep-equal WorldSpec on every call.
 */
export function generateWorld(ctx: TrackContext, audioPrior?: AudioFrame): WorldSpec {
  const seed = seedFromTrackId(ctx.trackId);

  // Single PRNG stream keyed on the seed. The *order* of draws below is fixed,
  // so the whole spec is reproducible. Keyed string => stable across runs.
  const rng = seedrandom(String(seed));

  const biome = resolveBiome(ctx, audioPrior, seed);
  const profile = BIOME_PROFILES[biome];
  const energy = effectiveEnergy(audioPrior, seed);

  const terrain = buildTerrain(profile, rng, seed);
  const scatter = buildScatter(profile, rng, energy);
  const creaturePool = buildCreaturePool(profile, rng, energy);
  const timeOfDay = buildTimeOfDay(profile, rng);
  const weatherState = buildWeather(profile, energy);

  return {
    biome,
    seed,
    terrain,
    scatter,
    timeOfDay,
    weatherState,
    // Album-art palette tints the world (PLAN §3). Copied so the spec owns it.
    palette: { ...ctx.palette },
    creaturePool,
    // Param -> modifier map (PLAN §4): how live AudioFrame values remap per frame.
    modifierCurves: {
      // bass transient -> lightning intensity.
      bassToLightning: [0.55, 1, 0, 1],
      // energy -> particle / element density.
      energyToParticles: [0, 1, 0.2, 1],
      // spectral centroid -> scene brightness.
      centroidToBrightness: [0, 1, 0.35, 1.25],
    },
  };
}
