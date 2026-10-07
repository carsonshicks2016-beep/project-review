/**
 * Presentation-only weather palettes. Never mutates sim / physics.
 * clear · dusk · night · rain — fog density + sky/asphalt tint (+ rain streaks).
 */

export const WEATHER_IDS = ["clear", "dusk", "night", "rain"];

const PALETTES = {
  clear: {
    id: "clear",
    label: "CLEAR",
    fog: 0x6e7c74,
    density: 0.011,
    hemiSky: 0xb8c4b8,
    hemiGround: 0x3a4038,
    hemiIntensity: 0.55,
    asphaltTint: [1.0, 1.0, 1.0],
    rain: false,
  },
  dusk: {
    id: "dusk",
    label: "DUSK",
    fog: 0x8a6a58,
    density: 0.014,
    hemiSky: 0xd4a078,
    hemiGround: 0x3a2820,
    hemiIntensity: 0.42,
    asphaltTint: [1.05, 0.92, 0.82],
    rain: false,
  },
  night: {
    id: "night",
    label: "NIGHT",
    fog: 0x1a2228,
    density: 0.018,
    hemiSky: 0x3a4a58,
    hemiGround: 0x101418,
    hemiIntensity: 0.28,
    asphaltTint: [0.55, 0.6, 0.72],
    rain: false,
  },
  rain: {
    id: "rain",
    label: "RAIN",
    fog: 0x5a6870,
    density: 0.02,
    hemiSky: 0x8a9aa4,
    hemiGround: 0x2a3034,
    hemiIntensity: 0.4,
    asphaltTint: [0.78, 0.82, 0.88],
    rain: true,
  },
};

export function resolveWeather(raw) {
  const key = String(raw || "clear").toLowerCase().trim();
  if (PALETTES[key]) return PALETTES[key];
  // Aliases from Observatory-ish query strings.
  if (key.includes("night")) return PALETTES.night;
  if (key.includes("dusk") || key.includes("sunset")) return PALETTES.dusk;
  if (key.includes("rain") || key.includes("wet")) return PALETTES.rain;
  return PALETTES.clear;
}

export function nextWeather(currentId) {
  const i = WEATHER_IDS.indexOf(currentId);
  const next = WEATHER_IDS[(i + 1) % WEATHER_IDS.length];
  return PALETTES[next];
}

export function weatherPalette(id) {
  return resolveWeather(id);
}
