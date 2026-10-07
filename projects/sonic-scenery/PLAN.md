# Sonic Scenery — Plan (authoritative)

A desktop app that generates a **3D immersive, procedurally-generated world** —
scenery, ambient creatures, and weather — driven by whatever is playing on
Spotify. Each song deterministically produces its own unique world; genre sets
the biome, the live audio animates it.

**Direction (locked):** 3D immersive world · native macOS real-time audio capture.

---

## 1. The constraint that shapes everything

Spotify **deprecated the Audio Features / Audio Analysis APIs in Nov 2024** for
newly-created apps (also Recommendations, Related Artists, reliable
`preview_url`). Those endpoints used to give `tempo`, `energy`, `valence`,
`danceability`, `key`, `mode` per track. A new app gets `403`.

So musical character comes from two sources:

| Source | Gives us | Notes |
|---|---|---|
| Spotify Web API (works) | track/artist/album, **artist genres**, album art → palette, playback progress | genres are *artist*-level and often sparse |
| Native audio capture + FFT (our machine) | live energy, spectral bands, onsets/beats, brightness | genre-agnostic, reactive |

Hybrid: **metadata themes + seeds** the world; **live audio animates** it.

---

## 2. Architecture

```
Spotify Web API ──(track change · genres · album art)──┐
                                                        ├─► Generation core ──► Three.js scene
Swift helper (ScreenCaptureKit + vDSP FFT) ──(WS)───────┘   (seed + theme)      (reactive modifiers)
```

| Layer | Tech | Role | Agent |
|---|---|---|---|
| Renderer | Electron + TypeScript + Three.js | 3D world, instanced creatures/particles, bloom | D |
| Audio | Swift + ScreenCaptureKit + Accelerate/vDSP | capture system audio → FFT → AudioFrame over WS | B |
| Metadata | Spotify Web API (PKCE OAuth) | poll currently-playing, genres, album-art palette | A |
| Generation | TS core, seeded PRNG | trackId → deterministic world; genre/params → modifiers | C |
| Weather | TS + GPU particles | precipitation, lightning, fog | E |
| Creatures | TS + boids/instancing | ambient agents per biome | F |

**macOS audio:** On macOS 13+ (this machine is far newer) ScreenCaptureKit
captures system audio with **no BlackHole/driver install** — just a one-time
Screen Recording permission grant. BlackHole is the fallback for pre-13.

---

## 3. Contracts (shared interfaces — build against these in parallel)

See `src/contracts/`. Summary:

- **`AudioFrame`** (helper → renderer/gen, ~60 Hz over WS):
  `{ t, rms, bands[8], bass, mid, treble, centroid, onset, beatConfidence }`
- **`TrackContext`** (spotify → gen, on track change):
  `{ trackId, title, artist, genres[], durationMs, progressMs, palette, artUrl }`
- **`WorldSpec`** (gen → renderer/weather/creatures):
  `{ biome, seed, terrain, scatter[], timeOfDay, weatherState, creaturePool[], palette, modifierCurves }`

---

## 4. Generation pipeline (per track)

1. **Genre → biome.** Artist genres pick a biome archetype. Fallback when
   genres are missing/vague: derive mood from live audio energy + brightness.
2. **`seed = hash(trackId)`** → seeded PRNG → terrain heightfield (layered
   simplex noise), Poisson-disk scatter, creature species, sky jitter.
   Same song → same world, always.
3. **Album-art palette** tints materials, sky, fog, lights (distinct worlds
   within one genre).
4. **Live audio modifiers** (continuous): bass → ground pulse/lightning;
   energy → particle density + weather intensity; centroid → brightness;
   onsets → discrete events.

### Genre → biome map (starter)

| Genre | Biome | Weather / creatures |
|---|---|---|
| lo-fi / chillhop | `lofi-rooftop` (dusk, neon) | soft rain, drifting cats/birds |
| classical | `classical-peaks` (misty, golden) | gentle wind, cranes/deer |
| edm / electronic | `edm-grid` (neon, synthwave sun) | strobes/lightning, glowing drones |
| metal | `metal-volcanic` (obsidian, ash) | storms, ravens/bats |
| folk / acoustic | `folk-forest` (meadow) | breeze, fireflies, deer |
| jazz | `jazz-noir` (rainy city) | drizzle, smoke, alley cats |
| ambient | `ambient-dream` (auroras) | weightless jellyfish-like drifters |
| pop | `pop-coast` (pastel) | light clouds, butterflies |

### Param → modifier map

- tempo → wind speed, creature movement, particle velocity
- energy/loudness → element density, weather intensity (drizzle→storm), saturation
- valence/mood → warm vs cold palette, day vs night, lush vs barren
- live amplitude/onsets → pulsing, lightning, creature bursts

---

## 5. Systems

- **Weather:** state machine `{clear,cloudy,drizzle,rain,storm,snow,fog,ash}`,
  biome-selected, intensity from live energy, GPU-instanced precipitation,
  lightning on bass transients.
- **Creatures:** Reynolds boids (birds/fish/fireflies) + wander agents (ground),
  GPU-instanced; count = f(energy, seed), speed = f(tempo/energy).
- **Transitions:** crossfade/dissolve between songs ~2–4s — never hard-cut.
- **Perf:** GPU instancing, LOD, capped particle counts, target 60 fps.

---

## 6. Honest risks

- Genres are **artist-level** and often sparse → live-audio mood fallback must
  ship in M1, not later.
- ScreenCaptureKit is great for **relative energy + onsets**; **exact BPM is
  unreliable** — supplement with a genre tempo prior, don't promise beat-lock.
- One-time Screen Recording permission prompt — document in setup.

---

## 7. Milestones & /batch decomposition

Agents A–D can run **in parallel** against the contracts.

| Milestone | Task | Agent |
|---|---|---|
| M0 | OAuth + currently-playing poller + genre/palette service | A |
| M1 | Swift capture helper → AudioFrame over WS | B |
| M2 | Generation core: seed, biome registry, param mapping (pure, testable) | C |
| M3 | Three.js renderer + 1 biome + post-processing | D |
| M4 | Weather system | E |
| M5 | Creature/boids system | F |
| M6 | 6–10 biomes, transitions, polish, presets | integration |

Per-agent briefs: `docs/tasks/agent-*.md`.

---

## 8. First build steps (after `go ahead`)

1. `npm install` (Electron + Three.js + TS toolchain).
2. Create a Spotify app + set redirect URI for PKCE.
3. `/batch` the A–D agents against their task briefs.
