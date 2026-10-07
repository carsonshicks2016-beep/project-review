export const VISUAL_CONDITIONS = Object.freeze({
  "clear-day": { id: "clear-day", label: "Clear daylight", lighting: "day", precipitation: "clear", surface: "dry", wetness: 0 },
  "overcast-day": { id: "overcast-day", label: "Overcast daylight", lighting: "day", precipitation: "overcast", surface: "dry", wetness: 0.08 },
  "wet-day": { id: "wet-day", label: "Wet daylight", lighting: "day", precipitation: "clear", surface: "wet", wetness: 0.56 },
  "rain-day": { id: "rain-day", label: "Rain daylight", lighting: "day", precipitation: "rain", surface: "wet", wetness: 0.86 },
  "clear-dusk": { id: "clear-dusk", label: "Clear dusk", lighting: "dusk", precipitation: "clear", surface: "dry", wetness: 0 },
  "wet-dusk": { id: "wet-dusk", label: "Wet dusk", lighting: "dusk", precipitation: "clear", surface: "wet", wetness: 0.62 },
  "rain-dusk": { id: "rain-dusk", label: "Rain dusk", lighting: "dusk", precipitation: "rain", surface: "wet", wetness: 0.9 },
  "clear-night": { id: "clear-night", label: "Clear night", lighting: "night", precipitation: "clear", surface: "dry", wetness: 0 },
  "rain-night": { id: "rain-night", label: "Rain night", lighting: "night", precipitation: "rain", surface: "wet", wetness: 0.9 },
});

const LEGACY = Object.freeze({ dry: "clear-day", dusk: "clear-dusk", night: "clear-night" });

export function resolveVisualCondition(value) {
  const id = VISUAL_CONDITIONS[value] ? value : LEGACY[value] || "clear-day";
  return VISUAL_CONDITIONS[id];
}
