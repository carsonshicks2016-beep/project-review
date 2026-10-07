// Chase/cinematic/hood rigs. Numbers carried over from legacy; Phase 7 tunes
// framing against the inspiration pics (car low-center, road dominating).
import * as THREE from "../vendor/three.module.min.js";

const MODES = ["chase", "cinematic", "hood"];

export function createCameraRig(camera) {
  let mode = "chase";
  const fwd = new THREE.Vector3();
  const side = new THREE.Vector3();
  const desired = new THREE.Vector3();
  const look = new THREE.Vector3();

  return {
    get mode() {
      return mode;
    },
    cycle() {
      mode = MODES[(MODES.indexOf(mode) + 1) % MODES.length];
      return mode;
    },
    update({ carPos, yaw, speed = 0, time = 0, snap = false }) {
      fwd.set(Math.cos(yaw), 0, -Math.sin(yaw));
      side.set(Math.sin(yaw), 0, Math.cos(yaw));

      if (mode === "hood") {
        desired.copy(carPos).addScaledVector(fwd, 1.66);
        desired.y += 1.05;
        look.copy(carPos).addScaledVector(fwd, 25);
        look.y += 0.82;
      } else if (mode === "cinematic") {
        desired.copy(carPos).addScaledVector(fwd, -7.2).addScaledVector(side, 4.2);
        desired.y += 2.35;
        look.copy(carPos).addScaledVector(fwd, 7.2);
        look.y += 0.82;
      } else {
        // pic framing: car low in frame, road dominating, horizon upper third
        desired.copy(carPos).addScaledVector(fwd, -9.0 - Math.min(speed * 0.08, 3.6));
        desired.y += 2.72;
        look.copy(carPos).addScaledVector(fwd, 15.5);
        look.y += 1.15;
      }
      if (snap) camera.position.copy(desired);
      else camera.position.lerp(desired, 0.095);
      camera.lookAt(look);
    },
  };
}
