import * as THREE from "three";

/**
 * Presentation cameras only — never written back to the session / sim.
 * Chase: low GT1 / Rally Championship style. Broadcast: higher TV follow.
 */

export const CAMERA_MODES = Object.freeze({
  CHASE: "chase",
  BROADCAST: "broadcast",
});

const PRESETS = {
  [CAMERA_MODES.CHASE]: {
    fov: 58,
    back: 11.5,
    height: 3.2,
    lookUp: 0.85,
    lateral: 0.35,
    lerp: 0.16,
    lookLerp: 0.22,
  },
  [CAMERA_MODES.BROADCAST]: {
    fov: 48,
    back: 22,
    height: 9.5,
    lookUp: 0.4,
    lateral: 4.5,
    lerp: 0.08,
    lookLerp: 0.12,
  },
};

export class WatchCamera {
  constructor(camera) {
    this.camera = camera;
    this.mode = CAMERA_MODES.CHASE;
    this.cut = 0;
    this._desired = new THREE.Vector3();
    this._look = new THREE.Vector3();
    this._lookSmooth = new THREE.Vector3();
    this._right = new THREE.Vector3();
    this._fwd = new THREE.Vector3();
    this._ready = false;
    this.applyFov();
  }

  setMode(mode) {
    if (!PRESETS[mode] || mode === this.mode) return;
    this.mode = mode;
    this.cut += 1;
    this._ready = false;
    this.applyFov();
  }

  /** FOA1 listener metadata — presentation labels only; no client pitch shift. */
  audioState() {
    return {
      camera: this.mode,
      perspective: "external",
      cut: this.cut,
    };
  }

  cycle() {
    const next = this.mode === CAMERA_MODES.CHASE
      ? CAMERA_MODES.BROADCAST
      : CAMERA_MODES.CHASE;
    this.setMode(next);
    return this.mode;
  }

  applyFov() {
    const preset = PRESETS[this.mode];
    this.camera.fov = preset.fov;
    this.camera.updateProjectionMatrix();
  }

  /**
   * @param {THREE.Vector3} carPos three-space position (already using roadZ)
   * @param {number} yaw sim yaw (radians)
   */
  update(carPos, yaw) {
    const p = PRESETS[this.mode];
    // Match coords.js / Observatory: forward ≈ (cos ψ, 0, −sin ψ) in Three.
    this._fwd.set(Math.cos(yaw), 0, -Math.sin(yaw));
    this._right.set(Math.sin(yaw), 0, Math.cos(yaw));

    this._desired.copy(carPos)
      .addScaledVector(this._fwd, -p.back)
      .addScaledVector(this._right, p.lateral);
    this._desired.y = carPos.y + p.height;

    this._look.set(carPos.x, carPos.y + p.lookUp, carPos.z);
    this._look.addScaledVector(this._fwd, 6);

    if (!this._ready) {
      this.camera.position.copy(this._desired);
      this._lookSmooth.copy(this._look);
      this._ready = true;
    } else {
      this.camera.position.lerp(this._desired, p.lerp);
      this._lookSmooth.lerp(this._look, p.lookLerp);
    }
    this.camera.lookAt(this._lookSmooth);
  }

  reset() {
    this._ready = false;
  }
}
