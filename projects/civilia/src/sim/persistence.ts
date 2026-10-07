import type { World } from './types'

/**
 * Persistence v0: the World is plain serializable data, so a snapshot is a
 * versioned JSON envelope. Loading a snapshot and continuing must produce
 * exactly the history an uninterrupted run would have produced — randomness
 * derives from seed + day + phase labels, never from in-memory generator
 * state, so nothing is lost across a save/load boundary.
 *
 * SQLite (event-log tables, periodic snapshots, timeline branching) is
 * deferred until a Node server process exists (stage 29); the browser-only
 * app uses localStorage and file download as its storage backends.
 */

// v2: relationships gained grievances/lastHarmDay (stage 13).
// v3: agent memories — world.memories, agent.memoryIds (stage 14).
// v4: graded beliefs — world.beliefs, agent.beliefIds (stage 15).
export const SAVE_VERSION = 4

interface SaveEnvelope {
  saveVersion: number
  savedAt: string
  world: World
}

export function serializeWorld(world: World): string {
  const envelope: SaveEnvelope = {
    saveVersion: SAVE_VERSION,
    savedAt: new Date().toISOString(),
    world,
  }
  return JSON.stringify(envelope)
}

export function deserializeWorld(json: string): World {
  let envelope: SaveEnvelope
  try {
    envelope = JSON.parse(json)
  } catch {
    throw new Error('Save file is not valid JSON')
  }
  if (typeof envelope !== 'object' || envelope === null) {
    throw new Error('Save file is not a Civilia snapshot')
  }
  if (envelope.saveVersion !== SAVE_VERSION) {
    throw new Error(
      `Save version ${envelope.saveVersion} is not supported (expected ${SAVE_VERSION})`,
    )
  }
  const world = envelope.world
  if (
    !world ||
    typeof world.seed !== 'string' ||
    typeof world.day !== 'number' ||
    !world.agents ||
    !world.places ||
    !Array.isArray(world.events)
  ) {
    throw new Error('Save file is missing core world state')
  }
  return world
}
