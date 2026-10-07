import type { SonicSceneryApi } from "../preload/preload";

declare global {
  interface Window {
    /** Exposed by the preload bridge when running inside Electron. */
    sonicScenery?: SonicSceneryApi;
  }
}

export {};
