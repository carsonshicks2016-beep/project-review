/**
 * Agent F — Shared per-species agent buffers.
 *
 * One `Flock` holds the simulation state for a single CreatureSpec. Positions
 * and velocities live in flat Float32Arrays (x,y,z interleaved) so they can be
 * written straight into an InstancedMesh matrix loop without allocating a
 * Vector3 per agent per frame.
 */
import * as THREE from "three";

export interface Flock {
  /** Number of live agents. */
  count: number;
  /** Interleaved xyz positions, length === count * 3. */
  positions: Float32Array;
  /** Interleaved xyz velocities, length === count * 3. */
  velocities: Float32Array;
  /** Per-agent phase used for wing-flap / pulse animation, length === count. */
  phases: Float32Array;
  /** Half-extents of the containment box, centered on `center`. */
  bounds: THREE.Vector3;
  /** World-space center of the containment box. */
  center: THREE.Vector3;
  /** Cruising speed (units/sec) from CreatureSpec.baseSpeed. */
  baseSpeed: number;
}

/**
 * Allocate and seed a flock with deterministic-ish random spread.
 *
 * @param count     number of agents
 * @param baseSpeed cruising speed
 * @param center    box center in world space
 * @param bounds    box half-extents
 * @param rand      0..1 RNG (defaults to Math.random)
 */
export function createFlock(
  count: number,
  baseSpeed: number,
  center: THREE.Vector3,
  bounds: THREE.Vector3,
  rand: () => number = Math.random,
): Flock {
  const positions = new Float32Array(count * 3);
  const velocities = new Float32Array(count * 3);
  const phases = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    positions[i3] = center.x + (rand() * 2 - 1) * bounds.x;
    positions[i3 + 1] = center.y + (rand() * 2 - 1) * bounds.y;
    positions[i3 + 2] = center.z + (rand() * 2 - 1) * bounds.z;

    // Random initial heading at cruising speed.
    const theta = rand() * Math.PI * 2;
    const phi = Math.acos(rand() * 2 - 1);
    const sp = baseSpeed * (0.5 + rand() * 0.5);
    velocities[i3] = Math.sin(phi) * Math.cos(theta) * sp;
    velocities[i3 + 1] = Math.cos(phi) * sp * 0.4; // less vertical drift
    velocities[i3 + 2] = Math.sin(phi) * Math.sin(theta) * sp;

    phases[i] = rand() * Math.PI * 2;
  }

  return {
    count,
    positions,
    velocities,
    phases,
    bounds: bounds.clone(),
    center: center.clone(),
    baseSpeed,
  };
}
