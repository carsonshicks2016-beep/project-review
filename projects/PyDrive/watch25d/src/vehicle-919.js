import * as THREE from "three";
import { makePs1Flat } from "./ps1-material.js";

/**
 * Low-poly Porsche 919 Evo proxy for Watch 2.5D.
 *
 * Authored in Three Y-up local space matching vehicleOrientation:
 *   +X nose, +Y up, +Z right  (sim left +Y → three −Z).
 * Dimensions follow assets/vehicles/porsche_919evo/asset_manifest.json
 * (length 5.078, width 1.9, wheelbase 2.957, wheel radius 0.355).
 *
 * Flat MeshBasicMaterial only — no PBR / cinema materials.
 */

const LENGTH = 5.078;
const WIDTH = 1.9;
const WHEEL_R = 0.355;
const WHEEL_W = 0.34;
const AXLE_F = 1.4785;
const AXLE_R = -1.4785;
const TRACK_F = 0.768;
const TRACK_R = 0.771;

const COL = {
  white: 0xf2f4f0,
  black: 0x14181c,
  carbon: 0x1c2228,
  red: 0xc4282c,
  canopy: 0x1a2834,
  glass: 0x2a3a48,
  rubber: 0x0c0e10,
  amber: 0xd08028,
  lamp: 0xd8e0e8,
  tail: 0xa01820,
  silver: 0x8a9498,
};

const WHEEL_ORDER = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"];

function paint(hex) {
  return makePs1Flat(hex);
}

function box(sx, sy, sz, x, y, z, color, parent) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(sx, sy, sz), paint(color));
  mesh.position.set(x, y, z);
  mesh.userData.ps1 = true;
  parent.add(mesh);
  return mesh;
}

/** Low-segment tyre + hub; cylinder default is +Y, rotated so axle = +Z. */
function makeWheel() {
  const group = new THREE.Group();
  const tyre = new THREE.Mesh(
    new THREE.CylinderGeometry(WHEEL_R, WHEEL_R, WHEEL_W, 10, 1),
    paint(COL.rubber),
  );
  tyre.rotation.x = Math.PI / 2;
  group.add(tyre);
  const hub = new THREE.Mesh(
    new THREE.CylinderGeometry(WHEEL_R * 0.42, WHEEL_R * 0.42, WHEEL_W * 0.55, 8, 1),
    paint(COL.silver),
  );
  hub.rotation.x = Math.PI / 2;
  group.add(hub);
  return group;
}

/**
 * Build the 919 proxy group.
 * @returns {{
 *   group: THREE.Group,
 *   wheels: Record<string, THREE.Object3D>,
 *   applyWheels: (frame: {wheels?: Array<{rotation?: number, steer?: number}>}) => void,
 *   dimensions: {length: number, width: number, height: number},
 * }}
 */
export function createPorsche919Proxy() {
  const group = new THREE.Group();
  group.name = "porsche_919evo_ps1";

  // --- Lower body / undertray ---
  box(4.55, 0.10, 1.22, -0.02, 0.08, 0, COL.carbon, group);

  // Main white shell — long LMP tub with slightly taller midsection.
  box(4.40, 0.38, 1.72, -0.05, 0.32, 0, COL.white, group);
  // Sidepods (wider mid-body).
  box(2.40, 0.34, 1.88, -0.35, 0.36, 0, COL.white, group);

  // Nose taper blocks (readable at chase distance).
  box(0.95, 0.28, 1.55, 1.95, 0.28, 0, COL.white, group);
  box(0.55, 0.20, 1.28, 2.35, 0.22, 0, COL.white, group);
  // Nose blackout / tunnels.
  box(0.70, 0.14, 1.50, 2.15, 0.14, 0, COL.black, group);
  // Front splitter.
  box(0.28, 0.04, 1.78, 2.42, 0.05, 0, COL.carbon, group);
  // Front flaps (left / right in Three: −Z left, +Z right).
  box(0.42, 0.04, 0.32, 2.18, 0.18, -0.55, COL.carbon, group);
  box(0.42, 0.04, 0.32, 2.18, 0.18, 0.55, COL.carbon, group);

  // Red nose number plate + thin red accent stripe.
  box(0.18, 0.16, 0.42, 2.08, 0.42, 0, COL.red, group);
  box(0.55, 0.05, 1.70, 1.55, 0.48, 0, COL.red, group);

  // Cockpit canopy bubble (dark glass look).
  box(1.35, 0.38, 0.95, 0.05, 0.72, 0, COL.canopy, group);
  box(1.05, 0.22, 0.78, 0.02, 0.96, 0, COL.glass, group);
  // Black roof shell behind canopy.
  box(0.85, 0.18, 0.85, -0.55, 0.88, 0, COL.black, group);
  // Stacked airbox.
  box(0.42, 0.20, 0.36, -0.38, 1.02, 0, COL.white, group);
  box(0.08, 0.10, 0.22, -0.16, 1.02, 0, COL.black, group);

  // Long rear deck / engine cover.
  box(1.55, 0.28, 1.55, -1.55, 0.42, 0, COL.white, group);
  box(0.95, 0.22, 1.40, -2.05, 0.36, 0, COL.black, group);

  // Shark fin (signature Evo silhouette).
  box(1.55, 0.42, 0.03, -1.25, 0.78, 0, COL.white, group);

  // Massive rear wing + endplates + pylons.
  box(0.22, 0.08, WIDTH, -2.42, 0.88, 0, COL.carbon, group);
  box(0.24, 0.42, 0.04, -2.40, 0.74, -0.94, COL.carbon, group);
  box(0.24, 0.42, 0.04, -2.40, 0.74, 0.94, COL.carbon, group);
  box(0.08, 0.42, 0.05, -2.20, 0.58, -0.50, COL.carbon, group);
  box(0.08, 0.42, 0.05, -2.20, 0.58, 0.50, COL.carbon, group);
  // Wing "PORSCHE" bar suggestion — flat red stripe on wing face.
  box(0.04, 0.05, 1.40, -2.30, 0.90, 0, COL.red, group);

  // Rear light bar + exhaust hint.
  box(0.05, 0.06, 0.55, -2.28, 0.34, 0, COL.tail, group);
  box(0.18, 0.10, 0.12, -2.30, 0.20, 0, COL.silver, group);

  // Mirrors.
  box(0.10, 0.06, 0.08, 0.42, 0.72, -0.82, COL.white, group);
  box(0.10, 0.06, 0.08, 0.42, 0.72, 0.82, COL.white, group);

  // Headlights.
  box(0.08, 0.08, 0.10, 2.20, 0.26, -0.72, COL.lamp, group);
  box(0.08, 0.08, 0.10, 2.20, 0.26, 0.72, COL.lamp, group);

  // Side markers.
  box(0.12, 0.04, 0.03, 1.05, 0.40, -0.94, COL.amber, group);
  box(0.12, 0.04, 0.03, 1.05, 0.40, 0.94, COL.amber, group);

  // Wheel arches (open bay hint — dark voids over tyre tops).
  for (const [ax, tz] of [
    [AXLE_F, -TRACK_F], [AXLE_F, TRACK_F],
    [AXLE_R, -TRACK_R], [AXLE_R, TRACK_R],
  ]) {
    box(0.72, 0.12, 0.38, ax, 0.58, tz * 0.92, COL.black, group);
  }

  // Soft contact blob — flat disc under the tub.
  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(1.55, 12),
    new THREE.MeshBasicMaterial({
      color: 0x000000,
      transparent: true,
      opacity: 0.35,
      depthWrite: false,
      fog: true,
    }),
  );
  shadow.rotation.x = -Math.PI / 2;
  shadow.position.y = 0.02;
  shadow.scale.set(LENGTH * 0.22, WIDTH * 0.55, 1);
  group.add(shadow);

  // --- Wheels ---
  const wheels = {};
  const placements = {
    wheel_fl: [AXLE_F, WHEEL_R, -TRACK_F],
    wheel_fr: [AXLE_F, WHEEL_R, TRACK_F],
    wheel_rl: [AXLE_R, WHEEL_R, -TRACK_R],
    wheel_rr: [AXLE_R, WHEEL_R, TRACK_R],
  };
  for (const [name, [x, y, z]] of Object.entries(placements)) {
    const wheel = makeWheel();
    wheel.name = name;
    wheel.position.set(x, y, z);
    group.add(wheel);
    wheels[name] = wheel;
  }

  function applyWheels(frame) {
    const list = frame?.wheels;
    if (!Array.isArray(list) || !list.length) return;
    for (let i = 0; i < WHEEL_ORDER.length; i += 1) {
      const node = wheels[WHEEL_ORDER[i]];
      const data = list[i];
      if (!node || !data) continue;
      // Steer about +Y (up), spin about +Z (axle). Steer outside spin.
      node.rotation.set(0, data.steer || 0, data.rotation || 0, "YZX");
    }
  }

  return {
    group,
    wheels,
    applyWheels,
    dimensions: { length: LENGTH, width: WIDTH, height: 1.05 },
  };
}

export const PORSCHE_919_META = {
  vehicleId: "porsche_919evo",
  displayName: "Porsche 919 Hybrid Evo",
  style: "watch25d-ps1-proxy-v1",
  length_m: LENGTH,
  width_m: WIDTH,
  wheelbase_m: 2.957,
  wheel_radius_m: WHEEL_R,
};
