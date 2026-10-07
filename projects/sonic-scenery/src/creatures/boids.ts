/**
 * Agent F — Reynolds boids flocking.
 *
 * Classic three-rule steering (separation / alignment / cohesion) plus a soft
 * boundary force that keeps the flock inside an axis-aligned box. Operates on
 * flat Float32Array position/velocity buffers so the same data can feed an
 * InstancedMesh without per-agent object allocation.
 */
import * as THREE from "three";
import type { Flock } from "./flock";

/** Tunable weights for the three Reynolds rules + steering limits. */
export interface BoidParams {
  separation: number;
  alignment: number;
  cohesion: number;
  /** Squared neighbour radius used for all three rules. */
  neighborRadiusSq: number;
  /** Squared distance under which separation kicks in hard. */
  separationRadiusSq: number;
  /** Max steering acceleration applied per second. */
  maxForce: number;
}

export const DEFAULT_BOID_PARAMS: BoidParams = {
  separation: 1.6,
  alignment: 1.0,
  cohesion: 0.9,
  neighborRadiusSq: 36, // 6 units
  separationRadiusSq: 6.25, // 2.5 units
  maxForce: 8,
};

// Scratch vectors reused every step to avoid per-frame allocation.
const _sep = new THREE.Vector3();
const _ali = new THREE.Vector3();
const _coh = new THREE.Vector3();
const _diff = new THREE.Vector3();
const _steer = new THREE.Vector3();
const _pos = new THREE.Vector3();
const _vel = new THREE.Vector3();
const _other = new THREE.Vector3();

/**
 * Advance a flock one step under boid rules.
 *
 * @param flock     position/velocity buffers + bounds + speed limits
 * @param dt        seconds since last frame
 * @param params    rule weights
 * @param agitation 0..1 extra urgency from live audio (raises speed + jitter)
 */
export function stepBoids(
  flock: Flock,
  dt: number,
  params: BoidParams,
  agitation: number,
): void {
  const { positions, velocities, count } = flock;
  const half = flock.bounds; // THREE.Vector3 half-extents
  const speed = flock.baseSpeed * (1 + agitation * 1.5);
  const maxSpeed = Math.max(0.001, speed);
  const minSpeed = maxSpeed * 0.35;

  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    _pos.set(positions[i3]!, positions[i3 + 1]!, positions[i3 + 2]!);
    _vel.set(velocities[i3]!, velocities[i3 + 1]!, velocities[i3 + 2]!);

    _sep.set(0, 0, 0);
    _ali.set(0, 0, 0);
    _coh.set(0, 0, 0);
    let neighbors = 0;
    let sepCount = 0;

    for (let j = 0; j < count; j++) {
      if (j === i) continue;
      const j3 = j * 3;
      _other.set(positions[j3]!, positions[j3 + 1]!, positions[j3 + 2]!);
      _diff.subVectors(_pos, _other);
      const d2 = _diff.lengthSq();
      if (d2 > params.neighborRadiusSq || d2 === 0) continue;

      // Alignment: average neighbour heading.
      _ali.x += velocities[j3]!;
      _ali.y += velocities[j3 + 1]!;
      _ali.z += velocities[j3 + 2]!;
      // Cohesion: average neighbour position.
      _coh.add(_other);
      neighbors++;

      // Separation: push away, weighted by inverse distance.
      if (d2 < params.separationRadiusSq) {
        _diff.multiplyScalar(1 / Math.max(d2, 1e-4));
        _sep.add(_diff);
        sepCount++;
      }
    }

    _steer.set(0, 0, 0);

    if (neighbors > 0) {
      _ali.multiplyScalar(1 / neighbors);
      limitToForce(_ali, params.maxForce);
      _steer.addScaledVector(_ali, params.alignment);

      _coh.multiplyScalar(1 / neighbors).sub(_pos);
      limitToForce(_coh, params.maxForce);
      _steer.addScaledVector(_coh, params.cohesion);
    }
    if (sepCount > 0) {
      _sep.multiplyScalar(1 / sepCount);
      limitToForce(_sep, params.maxForce);
      _steer.addScaledVector(_sep, params.separation);
    }

    // Soft containment: steer back toward the box before clipping.
    boundarySteer(_pos, half, params.maxForce, _steer);

    // Onset agitation adds a little random jitter so the flock scatters.
    if (agitation > 0.001) {
      _steer.x += (Math.random() - 0.5) * agitation * params.maxForce * 2;
      _steer.y += (Math.random() - 0.5) * agitation * params.maxForce * 2;
      _steer.z += (Math.random() - 0.5) * agitation * params.maxForce * 2;
    }

    _vel.addScaledVector(_steer, dt);
    clampSpeed(_vel, minSpeed, maxSpeed);

    _pos.addScaledVector(_vel, dt);
    hardClampToBounds(_pos, _vel, half);

    positions[i3] = _pos.x;
    positions[i3 + 1] = _pos.y;
    positions[i3 + 2] = _pos.z;
    velocities[i3] = _vel.x;
    velocities[i3 + 1] = _vel.y;
    velocities[i3 + 2] = _vel.z;
  }
}

function limitToForce(v: THREE.Vector3, maxForce: number): void {
  const len = v.length();
  if (len > maxForce) v.multiplyScalar(maxForce / len);
}

function clampSpeed(v: THREE.Vector3, min: number, max: number): void {
  const len = v.length();
  if (len > max) v.multiplyScalar(max / len);
  else if (len < min && len > 1e-5) v.multiplyScalar(min / len);
}

/** Adds an inward force when an agent nears/exceeds a wall of the box. */
function boundarySteer(
  pos: THREE.Vector3,
  half: THREE.Vector3,
  maxForce: number,
  out: THREE.Vector3,
): void {
  const margin = 2;
  if (pos.x > half.x - margin) out.x -= maxForce;
  else if (pos.x < -half.x + margin) out.x += maxForce;
  if (pos.y > half.y - margin) out.y -= maxForce;
  else if (pos.y < -half.y + margin) out.y += maxForce;
  if (pos.z > half.z - margin) out.z -= maxForce;
  else if (pos.z < -half.z + margin) out.z += maxForce;
}

/** Last-resort clamp: never let an agent leave the box; bounce velocity. */
function hardClampToBounds(
  pos: THREE.Vector3,
  vel: THREE.Vector3,
  half: THREE.Vector3,
): void {
  if (pos.x > half.x) {
    pos.x = half.x;
    vel.x = -Math.abs(vel.x);
  } else if (pos.x < -half.x) {
    pos.x = -half.x;
    vel.x = Math.abs(vel.x);
  }
  if (pos.y > half.y) {
    pos.y = half.y;
    vel.y = -Math.abs(vel.y);
  } else if (pos.y < -half.y) {
    pos.y = -half.y;
    vel.y = Math.abs(vel.y);
  }
  if (pos.z > half.z) {
    pos.z = half.z;
    vel.z = -Math.abs(vel.z);
  } else if (pos.z < -half.z) {
    pos.z = -half.z;
    vel.z = Math.abs(vel.z);
  }
}
