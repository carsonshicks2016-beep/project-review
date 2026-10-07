// Track spine: sim->three conversion, arc-length profile, road sampling, and
// the corridor "shape" — smoothed curvature, curvature classes, and the
// cliff/valley side assignment that terrain, furniture, and vegetation all
// key off. The client NEVER invents elevation — the y in these points is
// whatever the server streamed (VIEWER3D_V2_PLAN.md §2.2).
import * as THREE from "../vendor/three.module.min.js";
import { hash01 } from "./noise.js";

// Sim: X forward, Y left, Z up. Three: Y up, Z mirrored from sim Y.
export function simPointToThree(p) {
  return new THREE.Vector3(p[0], p[2], -p[1]);
}

export function createCorridor(trackMsg) {
  const center = trackMsg.center.map(simPointToThree);
  const left = trackMsg.left.map(simPointToThree);
  const right = trackMsg.right.map(simPointToThree);

  const distances = [0];
  for (let i = 1; i < center.length; i++) {
    distances.push(distances[i - 1] + center[i].distanceTo(center[i - 1]));
  }
  let widthSum = 0;
  for (let i = 0; i < center.length; i++) {
    widthSum += center[i].distanceTo(left[i]) + center[i].distanceTo(right[i]);
  }

  const corridor = {
    center,
    left,
    right,
    distances,
    length: distances[distances.length - 1] ?? 0,
    halfWidth: widthSum / Math.max(1, center.length * 2),
    meta: trackMsg.meta || {},
    seed: Number(trackMsg.meta?.seed ?? 7) || 7,
  };
  computeShape(corridor);
  return corridor;
}

function angleDelta(a, b) {
  let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

function boxSmooth(arr, radius) {
  const n = arr.length;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let sum = 0;
    let count = 0;
    for (let k = Math.max(0, i - radius); k <= Math.min(n - 1, i + radius); k++) {
      sum += arr[k];
      count++;
    }
    out[i] = sum / Math.max(1, count);
  }
  return out;
}

// Adds: corridor.curv (signed rad/m, smoothed), corridor.curvClass
// (0 straight / 1 bend / 2 hairpin), corridor.cliff in [-1, 1]
// (+1 = mountain fully on the LEFT, -1 = on the RIGHT, ~0 = saddle).
// The cliff signal mixes seeded low-frequency waves (integer cycles, so it is
// loop-periodic) with curvature — turns tend to hug the mountain on their
// inside, like a road cut into a slope.
function computeShape(corridor) {
  const n = corridor.center.length;
  const seed = corridor.seed;

  const yaw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const a = corridor.center[Math.max(0, i - 1)];
    const b = corridor.center[Math.min(n - 1, i + 1)];
    yaw[i] = Math.atan2(-(b.z - a.z), b.x - a.x);
  }

  const curvRaw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const i0 = Math.max(0, i - 2);
    const i1 = Math.min(n - 1, i + 2);
    const arc = Math.max(1e-3, corridor.distances[i1] - corridor.distances[i0]);
    curvRaw[i] = angleDelta(yaw[i0], yaw[i1]) / arc;
  }
  const curv = boxSmooth(curvRaw, 5);

  const curvClass = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    const c = Math.abs(curv[i]);
    curvClass[i] = c > 0.030 ? 2 : c > 0.012 ? 1 : 0;
  }

  const L = Math.max(1, corridor.length);
  const phase = hash01(seed * 13.7) * Math.PI * 2;
  const raw = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const u = (corridor.distances[i] / L) * Math.PI * 2;
    raw[i] = Math.tanh(
      1.6 * (
        0.9 * Math.sin(u * 2 + phase) +
        0.55 * Math.sin(u * 5 + phase * 2.17) +
        22.0 * curv[i]
      ),
    );
  }

  corridor.curv = curv;
  corridor.curvClass = curvClass;
  corridor.cliff = boxSmooth(raw, 9);
}

// +1 if (x,z) lies on the LEFT side of the road at this corridor index.
export function sideSignAt(corridor, index, x, z) {
  const c = corridor.center[index];
  const l = corridor.left[index];
  return ((x - c.x) * (l.x - c.x) + (z - c.z) * (l.z - c.z)) >= 0 ? 1 : -1;
}

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

export function sampleRoadAt(corridor, x, z) {
  if (!corridor || corridor.center.length < 2) return null;
  const pts = corridor.center;
  let bestDist = Infinity;
  let bestY = pts[0].y;
  let bestIx = 0;
  let bestT = 0;

  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    const vx = b.x - a.x;
    const vz = b.z - a.z;
    const len2 = vx * vx + vz * vz;
    if (len2 <= 1e-6) continue;
    const t = clamp01(((x - a.x) * vx + (z - a.z) * vz) / len2);
    const sx = a.x + vx * t;
    const sz = a.z + vz * t;
    const dx = x - sx;
    const dz = z - sz;
    const d2 = dx * dx + dz * dz;
    if (d2 < bestDist) {
      bestDist = d2;
      bestY = a.y + (b.y - a.y) * t;
      bestIx = i;
      bestT = t;
    }
  }

  return { y: bestY, distance: Math.sqrt(bestDist), index: bestIx, t: bestT };
}

export function roadHeightAt(corridor, x, z, fallbackY = 0) {
  const sample = sampleRoadAt(corridor, x, z);
  return sample ? sample.y : fallbackY;
}

export function roadPitchAt(corridor, x, z, yaw) {
  const halfProbe = 1.55;
  const fx = Math.cos(yaw) * halfProbe;
  const fz = -Math.sin(yaw) * halfProbe;
  const frontY = roadHeightAt(corridor, x + fx, z + fz);
  const rearY = roadHeightAt(corridor, x - fx, z - fz);
  return Math.atan2(frontY - rearY, halfProbe * 2);
}
