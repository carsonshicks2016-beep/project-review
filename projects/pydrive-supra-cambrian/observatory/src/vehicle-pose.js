import * as THREE from "three";

const vehicleEuler = new THREE.Euler(0, 0, 0, "YZX");

/**
 * Convert the simulator's x-forward/y-left/z-up pose into Observatory world
 * space (x-forward/y-up/z-right). The order is deliberately yaw, pitch, roll:
 * yaw follows the road in plan view, positive pitch raises the nose, and
 * positive roll raises the car's left side.
 *
 * This is render-only. The authoritative simulator remains the sole owner of
 * pose telemetry, including the airborne pitch and roll state.
 */
export function vehiclePoseQuaternion(pose = {}, target = new THREE.Quaternion()) {
  const yaw = Number.isFinite(Number(pose.yaw)) ? Number(pose.yaw) : 0;
  const pitch = Number.isFinite(Number(pose.pitch)) ? Number(pose.pitch) : 0;
  const roll = Number.isFinite(Number(pose.roll)) ? Number(pose.roll) : 0;
  vehicleEuler.set(roll, yaw, pitch, "YZX");
  return target.setFromEuler(vehicleEuler).normalize();
}

function lerpAngle(a, b, t) {
  const delta = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + delta * t;
}

/**
 * Conservative render-only filtering for the kinematic road attitude.
 *
 * Yaw already follows the buffered racing line smoothly and stays exact so
 * steering response is not made woolly. Pitch and roll contain small
 * sample-to-sample road-plane changes that read as body hopping in 3D, so they
 * receive a short frame-rate-independent low-pass. Nothing from this object is
 * returned to telemetry, policy inference, sensors, or the simulator.
 */
export class RenderAttitudeFilter {
  constructor(responseRate = 11) {
    this.responseRate = responseRate;
    this.pitch = 0;
    this.roll = 0;
    this.initialized = false;
  }

  reset(pose = {}) {
    this.pitch = Number(pose.pitch) || 0;
    this.roll = Number(pose.roll) || 0;
    this.initialized = true;
  }

  update(pose = {}, deltaSeconds = 1 / 60, snap = false) {
    const pitch = Number(pose.pitch) || 0;
    const roll = Number(pose.roll) || 0;
    if (!this.initialized || snap) {
      this.reset({ pitch, roll });
    } else {
      const dt = THREE.MathUtils.clamp(Number(deltaSeconds) || 0, 0, 0.1);
      const alpha = 1 - Math.exp(-dt * this.responseRate);
      this.pitch = lerpAngle(this.pitch, pitch, alpha);
      this.roll = lerpAngle(this.roll, roll, alpha);
    }
    return {
      ...pose,
      yaw: Number(pose.yaw) || 0,
      pitch: this.pitch,
      roll: this.roll,
    };
  }
}

/**
 * Predictive render-height filter for the 30 Hz road-height samples.
 *
 * A plain low-pass visibly trails the road on a sustained climb. This filter
 * estimates the current vertical trend, leads the target by roughly one filter
 * time constant, then eases toward it. The final clamp keeps the visual body
 * within a few centimetres of the authoritative surface even through sharp
 * grade changes.
 */
export class RenderHeightFilter {
  constructor({ responseRate = 14, velocityResponse = 10, lookAheadSeconds = 0.055, maxError = 0.09 } = {}) {
    this.responseRate = responseRate;
    this.velocityResponse = velocityResponse;
    this.lookAheadSeconds = lookAheadSeconds;
    this.maxError = maxError;
    this.height = 0;
    this.velocity = 0;
    this.previousTarget = 0;
    this.initialized = false;
  }

  reset(targetHeight = 0) {
    const target = Number.isFinite(Number(targetHeight)) ? Number(targetHeight) : 0;
    this.height = target;
    this.velocity = 0;
    this.previousTarget = target;
    this.initialized = true;
    return this.height;
  }

  update(targetHeight = 0, deltaSeconds = 1 / 60, snap = false) {
    const target = Number.isFinite(Number(targetHeight)) ? Number(targetHeight) : 0;
    if (!this.initialized || snap) return this.reset(target);
    const dt = THREE.MathUtils.clamp(Number(deltaSeconds) || 0, 1 / 240, 0.1);
    const measuredVelocity = (target - this.previousTarget) / dt;
    const velocityAlpha = 1 - Math.exp(-dt * this.velocityResponse);
    this.velocity += (measuredVelocity - this.velocity) * velocityAlpha;
    const predictedTarget = target + this.velocity * this.lookAheadSeconds;
    const positionAlpha = 1 - Math.exp(-dt * this.responseRate);
    this.height += (predictedTarget - this.height) * positionAlpha;
    this.height = THREE.MathUtils.clamp(
      this.height,
      target - this.maxError,
      target + this.maxError,
    );
    this.previousTarget = target;
    return this.height;
  }
}
