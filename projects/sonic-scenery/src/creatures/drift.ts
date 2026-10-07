/**
 * Agent F — Drift behavior for weightless creatures (jellyfish, dream motes).
 *
 * Slow, buoyant, near-frictionless motion: agents bob on sine waves around a
 * gentle baseline velocity, are pulled softly back toward the box center, and
 * on an onset get an upward/outward "pulse" that reads as the whole field
 * inhaling. No flocking — each agent is independent and calm.
 */
import * as THREE from "three";
import type { Flock } from "./flock";

export interface DriftParams {
  /** Vertical bob amplitude (units/sec contribution). */
  bob: number;
  /** Bob frequency (radians/sec). */
  bobFreq: number;
  /** Velocity damping per second (0..1, higher = more drag). */
  damping: number;
}

export const DEFAULT_DRIFT_PARAMS: DriftParams = {
  bob: 0.6,
  bobFreq: 0.8,
  damping: 0.6,
};

const _pos = new THREE.Vector3();
const _vel = new THREE.Vector3();
const _toCenter = new THREE.Vector3();

/**
 * Advance drifters one step.
 *
 * @param flock     buffers + bounds + baseSpeed
 * @param dt        seconds since last frame
 * @param params    bob + damping
 * @param agitation 0..1 from live audio; onsets pulse the field upward/outward
 */
export function stepDrift(
  flock: Flock,
  dt: number,
  params: DriftParams,
  agitation: number,
): void {
  const { positions, velocities, phases, count } = flock;
  const half = flock.bounds;
  const center = flock.center;
  const speed = flock.baseSpeed * (1 + agitation * 0.8);
  // Exponential damping that is stable regardless of dt.
  const damp = Math.exp(-params.damping * dt);

  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    _pos.set(positions[i3]!, positions[i3 + 1]!, positions[i3 + 2]!);
    _vel.set(velocities[i3]!, velocities[i3 + 1]!, velocities[i3 + 2]!);

    phases[i] = phases[i]! + params.bobFreq * dt;
    const ph = phases[i]!;

    // Gentle restoring pull toward center keeps the field cohesive.
    _toCenter.subVectors(center, _pos).multiplyScalar(0.05);

    _vel.multiplyScalar(damp);
    _vel.addScaledVector(_toCenter, dt);
    // Buoyant bob on Y, lazy sway on XZ.
    _vel.y += Math.sin(ph) * params.bob * dt;
    _vel.x += Math.cos(ph * 0.7) * params.bob * 0.3 * dt;
    _vel.z += Math.sin(ph * 0.5) * params.bob * 0.3 * dt;

    // Onset: push outward from center + a buoyant kick (the "inhale").
    if (agitation > 0.001) {
      _toCenter.subVectors(_pos, center).normalize();
      _vel.addScaledVector(_toCenter, agitation * speed * 1.5 * dt * 4);
      _vel.y += agitation * speed * dt * 4;
    }

    const vlen = _vel.length();
    const maxSpeed = Math.max(0.001, speed);
    if (vlen > maxSpeed) _vel.multiplyScalar(maxSpeed / vlen);

    _pos.addScaledVector(_vel, dt);

    // Soft bounce off the box so drifters never escape.
    if (_pos.x > half.x || _pos.x < -half.x) {
      _pos.x = THREE.MathUtils.clamp(_pos.x, -half.x, half.x);
      _vel.x *= -0.5;
    }
    if (_pos.y > half.y || _pos.y < -half.y) {
      _pos.y = THREE.MathUtils.clamp(_pos.y, -half.y, half.y);
      _vel.y *= -0.5;
    }
    if (_pos.z > half.z || _pos.z < -half.z) {
      _pos.z = THREE.MathUtils.clamp(_pos.z, -half.z, half.z);
      _vel.z *= -0.5;
    }

    positions[i3] = _pos.x;
    positions[i3 + 1] = _pos.y;
    positions[i3 + 2] = _pos.z;
    velocities[i3] = _vel.x;
    velocities[i3 + 1] = _vel.y;
    velocities[i3 + 2] = _vel.z;
  }
}
