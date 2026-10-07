/**
 * Fixed visual regression viewpoints.
 *
 * Open `?benchmark=<id>` to load the named replay, seek to the exact recorded
 * timestamp, select the fixed camera and pause after deterministic effect
 * reconstruction.
 */

import type { CameraMode } from "./camera";

export interface VisualBenchmark {
  id: string;
  replay: "demo" | "proving_ground";
  time: number;
  camera: CameraMode;
  purpose: string;
}

export const VISUAL_BENCHMARKS: readonly VisualBenchmark[] = [
  {
    id: "corner-roll",
    replay: "proving_ground",
    time: 8.7,
    camera: "chase",
    purpose: "normal gravel cornering, lateral load transfer and suspension roll",
  },
  {
    id: "landing-pitch",
    replay: "proving_ground",
    time: 22.0333,
    camera: "chase",
    purpose:
      "peak recorded landing load, 23,615 N total 0.17 s after the 21.8417 touchdown",
  },
  {
    id: "airborne-droop",
    replay: "proving_ground",
    time: 21.2333,
    camera: "sideline",
    purpose: "maximum recorded airborne height, 3.243 m, shadow separation and full droop",
  },
  {
    id: "gravel-spray",
    replay: "demo",
    time: 48.4667,
    camera: "sideline",
    purpose:
      "highest recorded gravel cornering load, 0.110 rad wheel slip at 31.1 m/s",
  },
  {
    id: "launch-wheelspin",
    replay: "demo",
    time: 2.8,
    camera: "chase",
    purpose:
      "highest recorded gravel slip ratio, 0.336 under full throttle at 16.4 m/s",
  },
  {
    id: "tarmac-skid",
    replay: "proving_ground",
    time: 22.1333,
    camera: "sideline",
    purpose:
      "skid trail accumulated through the 21.87-21.97 lock-up, peak slip ratio 0.833",
  },
] as const;

export function visualBenchmark(
  search: string,
): VisualBenchmark | undefined {
  const id = new URLSearchParams(search).get("benchmark");
  if (!id) return undefined;
  return VISUAL_BENCHMARKS.find((benchmark) => benchmark.id === id);
}
