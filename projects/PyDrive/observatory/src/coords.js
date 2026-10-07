import * as THREE from "three";

// Python/sim: X forward-plane, Y lateral-plane, Z up.
// Three.js:   X right-plane,  Y up,          Z toward camera (right-handed).
// Mapping: three(x,y,z) = (sim.x, sim.z, -sim.y)

export function simToThree(x, y, z, target = new THREE.Vector3()) {
  return target.set(x, z, -y);
}

export function threeToSim(x, y, z) {
  return { x, y: -z, z: y };
}

/**
 * Attitude for a Z-up authored vehicle after a fixed root rotateX(-π/2)
 * that aligns vehicle +Z with Three +Y.
 *
 * Sim body axes (supra/physics.py): +X nose, +Y left, +Z up, right-handed.
 *   yaw   ψ about +Z, positive = turn left
 *   pitch θ = atan(grade_body), positive = NOSE UP  → about sim −Y
 *   roll  φ = atan(bank_body),  positive = LEFT UP  → about sim +X
 * so the body rotation in sim space is Rz(ψ)·Ry(−θ)·Rx(φ).
 *
 * simToThree is the proper rotation Rx(−π/2), which carries
 *   sim +X → three +X,  sim +Y → three −Z,  sim +Z → three +Y.
 * Conjugating each factor through it (axes move, angles don't) gives
 *   yaw → about three +Y, pitch → about three +Z, roll → about three +X,
 * i.e. R = Ry(ψ)·Rz(θ)·Rx(φ) — exactly Euler order "YZX".
 *
 * Every sign here is load-bearing: cameras.js derives forward as
 * (cos ψ, 0, −sin ψ), so a flipped yaw crabs the body away from the shot.
 */
export function vehicleOrientation(yaw, pitch, roll, target = new THREE.Quaternion()) {
  const euler = new THREE.Euler(roll, yaw, pitch, "YZX");
  return target.setFromEuler(euler);
}

/** First-order attitude filter — softens policy-tick jitter without inventing pose. */
export class AttitudeFilter {
  constructor(tau = 0.045) {
    this.tau = tau;
    this.q = new THREE.Quaternion();
    this.ready = false;
  }

  reset(q) {
    this.q.copy(q);
    this.ready = true;
  }

  step(target, dt) {
    if (!this.ready) {
      this.reset(target);
      return this.q;
    }
    const alpha = 1 - Math.exp(-Math.max(0, dt) / Math.max(1e-4, this.tau));
    this.q.slerp(target, alpha);
    return this.q;
  }
}

export class HeightFilter {
  constructor(tau = 0.035) {
    this.tau = tau;
    this.value = 0;
    this.ready = false;
  }

  reset(value) {
    this.value = value;
    this.ready = true;
  }

  step(target, dt) {
    if (!this.ready) {
      this.reset(target);
      return this.value;
    }
    const alpha = 1 - Math.exp(-Math.max(0, dt) / Math.max(1e-4, this.tau));
    this.value += (target - this.value) * alpha;
    return this.value;
  }
}
