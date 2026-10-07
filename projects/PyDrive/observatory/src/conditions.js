// Visual condition table for the Observatory stage.
//
// Pure data + pure functions (no three.js import) so the contract is unit
// testable under node.  Colors are hex ints; the renderer owns how they are
// applied.  Weather is presentation only — the single simulation-facing value
// is `audio`, the coarse ambience word the Python audio renderer accepts.
//
// Fog color should sit near `horizon` so distant terrain dissolves into the
// sky band instead of cutting against a mismatched slab.

// Fog densities sit near legacy Observatory (~0.0012–0.0016 clear day). The
// remake's earlier 0.0005x values left mid-distance asphalt crushing to black
// while terrain still hazed — denser FogExp2 dissolves that ribbon.
const CONDITIONS = {
  "clear-day": {
    label: "Clear day", light: "day", wetness: 0.0, rain: 0,
    zenith: 0x3a7eb8, horizon: 0xc5d8e0, sun: 0xfff3dc, sunIntensity: 2.916,
    sunAltitude: 0.82, ambient: 0x9fb8c4, ambientIntensity: 0.28,
    fogColor: 0xc5d8e0, fogDensity: 0.0014, stars: 0,
    cloudAmount: 0.3, cloudTint: 0xf2f6f8,
  },
  "overcast-day": {
    label: "Overcast day", light: "day", wetness: 0.15, rain: 0,
    zenith: 0x5b6b74, horizon: 0xaab4b8, sun: 0xdfe6e6, sunIntensity: 1.458,
    sunAltitude: 0.75, ambient: 0x97a4aa, ambientIntensity: 0.38,
    fogColor: 0xaab4b8, fogDensity: 0.00155, stars: 0,
    cloudAmount: 0.92, cloudTint: 0xb9c2c6,
  },
  "wet-day": {
    label: "Wet day", light: "day", wetness: 0.62, rain: 0,
    zenith: 0x4c5f6b, horizon: 0x9cadb4, sun: 0xd8e2e4, sunIntensity: 1.242,
    sunAltitude: 0.7, ambient: 0x8b9aa2, ambientIntensity: 0.36,
    fogColor: 0x9cadb4, fogDensity: 0.0017, stars: 0,
    cloudAmount: 0.88, cloudTint: 0xa9b6bc,
  },
  "rain-day": {
    label: "Rain day", light: "day", wetness: 0.9, rain: 1.0,
    zenith: 0x3d4c56, horizon: 0x84939b, sun: 0xc4cfd2, sunIntensity: 0.972,
    sunAltitude: 0.66, ambient: 0x7c8b93, ambientIntensity: 0.4,
    fogColor: 0x84939b, fogDensity: 0.0020, stars: 0,
    cloudAmount: 0.97, cloudTint: 0x8e9ba2,
  },
  "clear-dusk": {
    label: "Clear dusk", light: "dusk", wetness: 0.0, rain: 0,
    zenith: 0x1c2a4e, horizon: 0xd98a4e, sun: 0xffbc7a, sunIntensity: 1.998,
    sunAltitude: 0.12, ambient: 0x6a5d70, ambientIntensity: 0.28,
    fogColor: 0xb87a52, fogDensity: 0.0015, stars: 0.25,
    cloudAmount: 0.34, cloudTint: 0xf0c49a,
  },
  "wet-dusk": {
    label: "Wet dusk", light: "dusk", wetness: 0.62, rain: 0,
    zenith: 0x1a2440, horizon: 0xb97a52, sun: 0xf2a878, sunIntensity: 1.512,
    sunAltitude: 0.1, ambient: 0x5c5566, ambientIntensity: 0.3,
    fogColor: 0x8a6450, fogDensity: 0.0018, stars: 0.2,
    cloudAmount: 0.8, cloudTint: 0xc79a7c,
  },
  "rain-dusk": {
    label: "Rain dusk", light: "dusk", wetness: 0.9, rain: 1.0,
    zenith: 0x161e33, horizon: 0x8a6350, sun: 0xc9906e, sunIntensity: 0.918,
    sunAltitude: 0.09, ambient: 0x4c4a58, ambientIntensity: 0.32,
    fogColor: 0x6a5248, fogDensity: 0.0021, stars: 0,
    cloudAmount: 0.95, cloudTint: 0x9c7663,
  },
  "clear-night": {
    label: "Clear night", light: "night", wetness: 0.0, rain: 0,
    zenith: 0x050a18, horizon: 0x14243a, sun: 0xbcd2ff, sunIntensity: 0.486,
    sunAltitude: 0.55, ambient: 0x2a3550, ambientIntensity: 0.22,
    fogColor: 0x101b2a, fogDensity: 0.0014, stars: 1.0,
    cloudAmount: 0.16, cloudTint: 0x2a3854,
  },
  "rain-night": {
    label: "Rain night", light: "night", wetness: 0.9, rain: 1.0,
    zenith: 0x04070f, horizon: 0x0e1722, sun: 0x93a8cc, sunIntensity: 0.346,
    sunAltitude: 0.5, ambient: 0x222c40, ambientIntensity: 0.24,
    fogColor: 0x0c141d, fogDensity: 0.0019, stars: 0.15,
    cloudAmount: 0.95, cloudTint: 0x1b2432,
  },
};

export const CONDITION_IDS = Object.freeze(Object.keys(CONDITIONS));

export function condition(id) {
  const entry = CONDITIONS[String(id ?? "").toLowerCase()] ?? CONDITIONS["clear-day"];
  return { id: entry === CONDITIONS[id] ? id : conditionId(id), ...entry };
}

function conditionId(id) {
  return CONDITIONS[String(id ?? "").toLowerCase()] ? String(id).toLowerCase() : "clear-day";
}

/** Coarse ambience word for supra.audio_v4 — the only weather the sim hears. */
export function ambienceWord(id) {
  const light = condition(id).light;
  if (light === "night") return "night";
  if (light === "dusk") return "dusk";
  return "dry";
}
