/**
 * Preload bridge (M6 integration).
 *
 * Runs in the isolated preload context (contextIsolation: true) and exposes a
 * tiny, typed API on `window.sonicScenery` so the renderer can receive
 * generated worlds + status from the main process without Node access.
 */
import { contextBridge, ipcRenderer } from "electron";
import type { WorldSpec } from "../contracts";

export interface SonicStatus {
  mode: "spotify" | "demo";
  message?: string;
}

const api = {
  /** Subscribe to WorldSpecs generated from the current Spotify track. Returns an unsubscribe fn. */
  onWorld(cb: (spec: WorldSpec) => void): () => void {
    const listener = (_e: unknown, spec: WorldSpec) => cb(spec);
    ipcRenderer.on("sonic:world", listener);
    return () => ipcRenderer.removeListener("sonic:world", listener);
  },
  /** Subscribe to status/mode updates (spotify vs demo, errors). */
  onStatus(cb: (status: SonicStatus) => void): () => void {
    const listener = (_e: unknown, s: SonicStatus) => cb(s);
    ipcRenderer.on("sonic:status", listener);
    return () => ipcRenderer.removeListener("sonic:status", listener);
  },
};

export type SonicSceneryApi = typeof api;

contextBridge.exposeInMainWorld("sonicScenery", api);
