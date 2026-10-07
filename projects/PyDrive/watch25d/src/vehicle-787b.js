import * as THREE from "three";
import { makePs1Flat } from "./ps1-material.js";

/**
 * Low-poly Mazda 787B #55 Renown proxy for Watch 2.5D.
 *
 * Authored in Three Y-up local space matching vehicleOrientation:
 *   +X nose, +Y up, +Z right  (sim left +Y → three −Z).
 * Dimensions follow assets/vehicles/mazda787b/asset_manifest.json
 * (length 4.782, width 1.994, wheelbase 2.662).
 *
 * Distinct from the 919: green/orange Renown blocks, rotary coupe canopy,
 * no shark fin, shorter wheelbase, different wing stance.
 */

const LENGTH = 4.782;
const WIDTH = 1.994;
const WHEEL_RF = 0.32;
const WHEEL_RR = 0.355;
const WHEEL_W = 0.34;
const AXLE_F = 1.335;
const AXLE_R = -1.327;
const TRACK_F = 0.767;
const TRACK_R = 0.752;

const COL = {
  green: 0x1a6b3c,
  greenDark: 0x0e3f24,
  orange: 0xe85a1a,
  orangeDeep: 0xc44812,
  white: 0xf0f2ec,
  black: 0x14181c,
  carbon: 0x1c2228,
  canopy: 0x1a2834,
  glass: 0x243848,
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

function makeWheel(radius) {
  const group = new THREE.Group();
  const tyre = new THREE.Mesh(
    new THREE.CylinderGeometry(radius, radius, WHEEL_W, 10, 1),
    paint(COL.rubber),
  );
  tyre.rotation.x = Math.PI / 2;
  group.add(tyre);
  const hub = new THREE.Mesh(
    new THREE.CylinderGeometry(radius * 0.42, radius * 0.42, WHEEL_W * 0.55, 8, 1),
    paint(COL.silver),
  );
  hub.rotation.x = Math.PI / 2;
  group.add(hub);
  return group;
}

/**
 * Build the 787B proxy group.
 * @returns {{
 *   group: THREE.Group,
 *   wheels: Record<string, THREE.Object3D>,
 *   applyWheels: (frame: {wheels?: Array<{rotation?: number, steer?: number}>}) => void,
 *   dimensions: {length: number, width: number, height: number},
 * }}
 */
export function createMazda787bProxy() {
  const group = new THREE.Group();
  group.name = "mazda787b_ps1";

  // Undertray / floor.
  box(4.25, 0.09, 1.35, -0.02, 0.07, 0, COL.carbon, group);

  // Main green tub — Group C coupe, shorter than the 919 LMP.
  box(4.05, 0.36, 1.78, -0.08, 0.30, 0, COL.green, group);
  // Wider mid sidepods.
  box(2.15, 0.32, 1.96, -0.25, 0.34, 0, COL.green, group);
  // Dark lower rocker.
  box(3.80, 0.12, 1.85, -0.10, 0.14, 0, COL.greenDark, group);

  // Nose — blunter Group C prow + orange tip.
  box(0.85, 0.26, 1.58, 1.78, 0.26, 0, COL.green, group);
  box(0.48, 0.18, 1.30, 2.18, 0.20, 0, COL.green, group);
  box(0.22, 0.14, 1.10, 2.38, 0.18, 0, COL.orange, group);
  // Front splitter.
  box(0.26, 0.04, 1.85, 2.28, 0.05, 0, COL.carbon, group);
  // Nose tunnels.
  box(0.55, 0.10, 1.42, 2.00, 0.12, 0, COL.black, group);

  // Renown orange spine + side flash (signature #55 read).
  box(3.60, 0.08, 0.28, -0.15, 0.50, 0, COL.orange, group);
  box(2.20, 0.06, 0.10, -0.20, 0.46, -0.88, COL.orange, group);
  box(2.20, 0.06, 0.10, -0.20, 0.46, 0.88, COL.orange, group);
  // White number-plate patch on nose.
  box(0.16, 0.14, 0.36, 1.95, 0.40, 0, COL.white, group);
  box(0.10, 0.10, 0.22, 1.96, 0.40, 0, COL.orangeDeep, group);

  // Rotary coupe canopy — lower bubble, further forward than 919 airbox stack.
  box(1.45, 0.34, 0.92, 0.22, 0.66, 0, COL.canopy, group);
  box(1.15, 0.18, 0.74, 0.18, 0.88, 0, COL.glass, group);
  // Roof scoop / intake (rotary snorkel hint).
  box(0.55, 0.16, 0.42, -0.35, 0.92, 0, COL.greenDark, group);
  box(0.28, 0.10, 0.28, -0.55, 1.02, 0, COL.black, group);

  // Rear deck — shorter engine cover, twin-exhaust suggestion.
  box(1.35, 0.26, 1.62, -1.45, 0.38, 0, COL.green, group);
  box(0.85, 0.20, 1.45, -1.95, 0.34, 0, COL.greenDark, group);
  box(0.16, 0.08, 0.10, -2.15, 0.18, -0.22, COL.silver, group);
  box(0.16, 0.08, 0.10, -2.15, 0.18, 0.22, COL.silver, group);

  // Compact rear wing (no shark fin — coupe silhouette).
  box(0.20, 0.07, WIDTH * 0.95, -2.28, 0.78, 0, COL.carbon, group);
  box(0.22, 0.36, 0.04, -2.26, 0.66, -0.92, COL.carbon, group);
  box(0.22, 0.36, 0.04, -2.26, 0.66, 0.92, COL.carbon, group);
  box(0.07, 0.34, 0.05, -2.08, 0.52, -0.48, COL.carbon, group);
  box(0.07, 0.34, 0.05, -2.08, 0.52, 0.48, COL.carbon, group);
  // Orange wing stripe.
  box(0.04, 0.04, 1.35, -2.16, 0.80, 0, COL.orange, group);

  // Tail lights.
  box(0.05, 0.06, 0.22, -2.18, 0.32, -0.38, COL.tail, group);
  box(0.05, 0.06, 0.22, -2.18, 0.32, 0.38, COL.tail, group);

  // Mirrors.
  box(0.10, 0.06, 0.08, 0.55, 0.68, -0.86, COL.green, group);
  box(0.10, 0.06, 0.08, 0.55, 0.68, 0.86, COL.green, group);

  // Headlights.
  box(0.08, 0.08, 0.10, 2.10, 0.24, -0.70, COL.lamp, group);
  box(0.08, 0.08, 0.10, 2.10, 0.24, 0.70, COL.lamp, group);

  // Side markers.
  box(0.12, 0.04, 0.03, 0.95, 0.38, -0.98, COL.amber, group);
  box(0.12, 0.04, 0.03, 0.95, 0.38, 0.98, COL.amber, group);

  // Wheel arches.
  for (const [ax, tz, r] of [
    [AXLE_F, -TRACK_F, WHEEL_RF], [AXLE_F, TRACK_F, WHEEL_RF],
    [AXLE_R, -TRACK_R, WHEEL_RR], [AXLE_R, TRACK_R, WHEEL_RR],
  ]) {
    box(0.68, 0.11, 0.36, ax, r + 0.22, tz * 0.92, COL.black, group);
  }

  const shadow = new THREE.Mesh(
    new THREE.CircleGeometry(1.45, 12),
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

  const wheels = {};
  const placements = {
    wheel_fl: [AXLE_F, WHEEL_RF, -TRACK_F, WHEEL_RF],
    wheel_fr: [AXLE_F, WHEEL_RF, TRACK_F, WHEEL_RF],
    wheel_rl: [AXLE_R, WHEEL_RR, -TRACK_R, WHEEL_RR],
    wheel_rr: [AXLE_R, WHEEL_RR, TRACK_R, WHEEL_RR],
  };
  for (const [name, [x, y, z, radius]] of Object.entries(placements)) {
    const wheel = makeWheel(radius);
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

export const MAZDA_787B_META = {
  vehicleId: "mazda787b",
  displayName: "Mazda 787B #55 Renown",
  style: "watch25d-ps1-proxy-v1",
  length_m: LENGTH,
  width_m: WIDTH,
  wheelbase_m: 2.662,
  wheel_radius_m: WHEEL_RR,
};
