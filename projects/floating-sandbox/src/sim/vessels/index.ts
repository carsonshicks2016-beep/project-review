import type { Ship, VesselPreset } from '../types';
import {
  VESSEL_PRESETS,
  buildBarge,
  buildCraneBarge,
  buildFreighter,
  buildIceberg,
  buildLiner,
  buildRaft,
  buildTug,
  spawnVessel,
} from './presets';
import {
  importBlueprintFromFile,
  importBlueprintFromImageData,
  materialFromColor,
} from './blueprint';

export {
  VESSEL_PRESETS,
  spawnVessel,
  buildLiner,
  buildFreighter,
  buildTug,
  buildBarge,
  buildRaft,
  buildCraneBarge,
  buildIceberg,
  importBlueprintFromFile,
  importBlueprintFromImageData,
  materialFromColor,
};

/** Full VesselPreset registry with build callbacks for UI / spawn menus. */
export const VESSEL_REGISTRY: VesselPreset[] = [
  { id: 'liner', name: 'Ocean Liner', build: buildLiner },
  { id: 'freighter', name: 'Freighter', build: buildFreighter },
  { id: 'tug', name: 'Tugboat', build: buildTug },
  { id: 'barge', name: 'Barge', build: buildBarge },
  { id: 'raft', name: 'Raft', build: buildRaft },
  { id: 'craneBarge', name: 'Crane Barge', build: buildCraneBarge },
  { id: 'iceberg', name: 'Iceberg', build: buildIceberg },
];

/** Spawn from registry by id (same as spawnVessel, kept for callers that use the registry). */
export function buildVesselById(
  id: string,
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1,
): Ship | null {
  return spawnVessel(id, cx, cy, densityScale, strengthScale);
}
