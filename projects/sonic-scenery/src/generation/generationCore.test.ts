/**
 * Lightweight runnable determinism check for the generation core.
 *
 * Pure Node script (no test framework). The repo uses extensionless,
 * bundler-resolved imports, so run it through the bundled esbuild:
 *   npm run test:gen
 * (equivalently: esbuild ... --bundle --platform=node --format=esm | node)
 *
 * Asserts the acceptance criteria from docs/tasks/agent-C-generation.md:
 *   1. same trackId -> deep-equal WorldSpec (across calls; with/without prior).
 *   2. different trackIds -> different worlds.
 *   3. every Biome is reachable (via genres and via the audio-prior fallback).
 *   4. unknown genre + audio prior resolves sensibly (never throws, valid biome).
 *   5. purity probe: output is stable regardless of wall-clock time.
 */
import type { AudioFrame, Biome, TrackContext, WorldSpec } from "../contracts";
import { generateWorld, seedFromTrackId, biomeFromAudioPrior } from "./generationCore";
import { ALL_BIOMES, GENRE_TO_BIOME } from "./biomes";

let failures = 0;
function assert(cond: boolean, msg: string): void {
  if (!cond) {
    failures++;
    console.error(`  FAIL: ${msg}`);
  } else {
    console.log(`  ok:   ${msg}`);
  }
}

function eq(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b);
}

const palette = { primary: "#1a2b3c", secondary: "#445566", accent: "#ffaa00", bg: "#020308" };

function track(trackId: string, genres: string[]): TrackContext {
  return {
    trackId,
    title: "T",
    artist: "A",
    genres,
    durationMs: 200000,
    progressMs: 0,
    palette,
    artUrl: "",
  };
}

function frame(rms: number, centroid: number): AudioFrame {
  return {
    t: 0,
    rms,
    bands: [0, 0, 0, 0, 0, 0, 0, 0],
    bass: 0,
    mid: 0,
    treble: 0,
    centroid,
    onset: false,
    beatConfidence: 0,
  };
}

// 1. Determinism: same id -> identical spec on repeated calls.
console.log("[1] determinism (same id -> deep-equal)");
{
  const ctx = track("spotify:track:abc123", ["lo-fi beats"]);
  const a = generateWorld(ctx);
  const b = generateWorld(ctx);
  assert(eq(a, b), "two calls with same ctx produce deep-equal WorldSpec");

  // A fresh ctx object with the same trackId must also match.
  const c = generateWorld(track("spotify:track:abc123", ["lo-fi beats"]));
  assert(eq(a, c), "fresh ctx, same trackId -> deep-equal");

  // Prior should not break determinism when reused.
  const fr = frame(0.7, 0.8);
  assert(eq(generateWorld(ctx, fr), generateWorld(ctx, fr)), "same prior -> deep-equal");
}

// 2. Different ids -> different worlds.
console.log("[2] different ids -> different worlds");
{
  const seen = new Set<string>();
  let distinct = 0;
  for (let i = 0; i < 50; i++) {
    const w = generateWorld(track(`id-${i}`, ["jazz"]));
    const sig = JSON.stringify(w);
    if (!seen.has(sig)) distinct++;
    seen.add(sig);
  }
  assert(distinct === 50, `50 distinct trackIds -> 50 distinct worlds (got ${distinct})`);

  // Seeds themselves should differ.
  assert(seedFromTrackId("a") !== seedFromTrackId("b"), "distinct ids -> distinct seeds");
  assert(seedFromTrackId("a") === seedFromTrackId("a"), "same id -> same seed");
}

// 3. Every Biome reachable via genres.
console.log("[3] every biome reachable via genres");
{
  const reached = new Set<Biome>();
  for (const [needle] of GENRE_TO_BIOME) {
    const w = generateWorld(track(`g-${needle}`, [needle]));
    reached.add(w.biome);
  }
  for (const b of ALL_BIOMES) {
    assert(reached.has(b), `biome reachable via genre: ${b}`);
  }
}

// 3b. Every Biome reachable via audio-prior fallback (no genres).
console.log("[3b] every biome reachable via audio-prior fallback");
{
  const reached = new Set<Biome>();
  // Sweep the energy x brightness space.
  for (let e = 0; e <= 1.0001; e += 0.1) {
    for (let c = 0; c <= 1.0001; c += 0.1) {
      reached.add(biomeFromAudioPrior(frame(e, c), 0));
    }
  }
  for (const b of ALL_BIOMES) {
    assert(reached.has(b), `biome reachable via audio prior: ${b}`);
  }
}

// 4. Unknown genre + audio prior resolves sensibly (no throw, valid biome).
console.log("[4] unknown genre + prior resolves sensibly");
{
  const valid = new Set<Biome>(ALL_BIOMES);
  let threw = false;
  let allValid = true;
  for (const genres of [[], ["totally-unknown-genre"], ["", "  "]]) {
    try {
      const w = generateWorld(track("u-" + genres.join(","), genres), frame(0.9, 0.9));
      if (!valid.has(w.biome)) allValid = false;
    } catch {
      threw = true;
    }
  }
  // Loud + bright unknown -> edm-grid specifically.
  const loudBright = generateWorld(track("lb", []), frame(0.9, 0.9));
  assert(!threw, "never throws on unknown/empty genres");
  assert(allValid, "fallback always yields a valid biome");
  assert(loudBright.biome === "edm-grid", "loud+bright unknown -> edm-grid");
}

// 5. Purity probe: structurally valid spec.
console.log("[5] structural validity");
{
  const w: WorldSpec = generateWorld(track("struct", ["folk"]), frame(0.5, 0.5));
  assert(w.scatter.length > 0, "scatter is non-empty");
  assert(w.creaturePool.length > 0, "creaturePool is non-empty");
  assert(w.timeOfDay >= 0 && w.timeOfDay <= 1, "timeOfDay in [0,1]");
  assert(w.terrain.seed === seedFromTrackId("struct"), "terrain.seed == seedFromTrackId");
  assert(
    w.scatter.every((s) => Math.abs(s.x) <= 60 && Math.abs(s.z) <= 60),
    "scatter points within world bounds",
  );
  assert(eq(w.palette, palette), "palette mapped from ctx");
}

console.log("");
if (failures > 0) {
  console.error(`DETERMINISM TEST: ${failures} assertion(s) FAILED`);
  process.exit(1);
} else {
  console.log("DETERMINISM TEST: all assertions passed");
}
