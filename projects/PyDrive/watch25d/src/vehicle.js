import { createPorsche919Proxy, PORSCHE_919_META } from "./vehicle-919.js";
import { createMazda787bProxy, MAZDA_787B_META } from "./vehicle-787b.js";

/** Resolve PS1 proxy kind from edition query and/or session checkpoint car. */
export function resolveVehicleKind(...hints) {
  for (const hint of hints) {
    const key = String(hint || "").toLowerCase();
    if (!key) continue;
    if (key.includes("787") || key.includes("mazda")) return "787b";
    if (key.includes("919") || key.includes("porsche")) return "919";
  }
  return "919";
}

export function createVehicleProxy(kind) {
  if (kind === "787b") {
    return { kind: "787b", ...createMazda787bProxy(), meta: MAZDA_787B_META };
  }
  return { kind: "919", ...createPorsche919Proxy(), meta: PORSCHE_919_META };
}

export { PORSCHE_919_META, MAZDA_787B_META };
