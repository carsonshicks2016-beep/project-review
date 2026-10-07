/**
 * Agent F — Wander behavior for ground agents (deer, cats, etc.).
 *
 * Each agent steers toward a slowly drifting heading projected onto a wander
 * circle in front of it, stays clamped to the ground plane (y === center.y),
 * and turns back when it nears the box edge. Onset agitation makes them bolt.
 */
import * as THREE from "three";
import type { Flock } from "./flock";

export interface WanderParams {
  /** How sharply heading can change per second (radians). */
  turnRate: number;
  /** Max steering acceleration per second. */
  maxForce: number;
}

export const DEFAULT_WANDER_PARAMS: WanderParams = {
  turnRate: 1.2,
  maxForce: 6,
};

const _pos = new THREE.Vector3();
const _vel = new THREE.Vector3();
const _steer = new THREE.Vector3();

/**
 * Advance ground wanderers one step.
 *
 * @param flock     buffers + bounds + baseSpeed
 * @param dt        seconds since last frame
 * @param params    turn rate + force limits
 * @param agitation 0..1 from live audio; onsets make them sprint + scatter
 */
export function stepWander(
  flock: Flock,
  dt: number,
  params: WanderParams,
  agitation: number,
): void {
  const { positions, velocities, phases, count } = flock;
  const half = flock.bounds;
  const groundY = flock.center.y;
  const speed = flock.baseSpeed * (1 + agitation * 2.0);
  const maxSpeed = Math.max(0.001, speed);

  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    _pos.set(positions[i3]!, positions[i3 + 1]!, positions[i3 + 2]!);
    _vel.set(velocities[i3]!, 0, velocities[i3 + 2]!);

    // Drift the wander phase; agitation speeds up direction changes.
    const wobble = params.turnRate * (1 + agitation * 3);
    phases[i] = phases[i]! + (Math.random() - 0.5) * wobble * dt;
    const heading = phases[i]!;

    // Desired velocity along the wander heading.
    _steer.set(Math.cos(heading), 0, Math.sin(heading)).multiplyScalar(maxSpeed);
    _steer.sub(_vel);
    const len = _steer.length();
    if (len > params.maxForce) _steer.multiplyScalar(params.maxForce / len);

    // Turn back inside the box (XZ only — ground agents).
    const margin = 3;
    if (_pos.x > half.x - margin) _steer.x -= params.maxForce;
    else if (_pos.x < -half.x + margin) _steer.x += params.maxForce;
    if (_pos.z > half.z - margin) _steer.z -= params.maxForce;
    else if (_pos.z < -half.z + margin) _steer.z += params.maxForce;

    _vel.addScaledVector(_steer, dt);
    _vel.y = 0;
    const vlen = _vel.length();
    if (vlen > maxSpeed) _vel.multiplyScalar(maxSpeed / vlen);

    _pos.addScaledVector(_vel, dt);
    _pos.y = groundY; // pinned to the ground

    // Hard clamp inside the box on XZ.
    if (_pos.x > half.x) {
      _pos.x = half.x;
      _vel.x = -Math.abs(_vel.x);
    } else if (_pos.x < -half.x) {
      _pos.x = -half.x;
      _vel.x = Math.abs(_vel.x);
    }
    if (_pos.z > half.z) {
      _pos.z = half.z;
      _vel.z = -Math.abs(_vel.z);
    } else if (_pos.z < -half.z) {
      _pos.z = -half.z;
      _vel.z = Math.abs(_vel.z);
    }

    positions[i3] = _pos.x;
    positions[i3 + 1] = _pos.y;
    positions[i3 + 2] = _pos.z;
    velocities[i3] = _vel.x;
    velocities[i3 + 1] = 0;
    velocities[i3 + 2] = _vel.z;
  }
}
