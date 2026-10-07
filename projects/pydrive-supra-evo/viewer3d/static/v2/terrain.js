// Carved terrain: a heightfield that conforms to the road (bench under the
// corridor), rises into a mountain mass on the cliff side, and drops into a
// valley on the other — plus crisp near-road cliff walls and valley lips that
// follow the corridor and pinch away where the side assignment swaps.
// Look: flat-shaded facets + day-authored vertex colors; night comes from
// lighting and the mood-bound terrainTint multiplier.
import * as THREE from "../vendor/three.module.min.js";
import { sampleRoadAt, sideSignAt } from "./corridor.js";
import { smoothstep, hash01, fbm2 } from "./noise.js";

const TAG = "world";

// Day-authored facet palette (hex). Variations picked per face by hash.
const FACETS = {
  grass: [0x5d7355, 0x55694d, 0x647a5b],
  grassDark: [0x45593f, 0x3e5039],
  valleyDeep: [0x36493c, 0x2f4136],
  dirt: [0x6b5f49, 0x615640],
  rock: [0x7d7f7c, 0x585b58, 0x6e716d],
  rockLight: [0x9a9b94, 0x8f918c],
  pale: [0x9aa09b, 0x8b9290],
};

function pick(list, h) {
  return list[Math.min(list.length - 1, Math.floor(h * list.length))];
}

// Accumulates loose triangles with per-face color -> one flat-shaded mesh.
export class FacetBuffer {
  constructor() {
    this.pos = [];
    this.col = [];
    this._c = new THREE.Color();
  }

  tri(a, b, c, hex, bright = 1) {
    this.pos.push(a.x, a.y, a.z, b.x, b.y, b.z, c.x, c.y, c.z);
    const col = this._c.set(hex).multiplyScalar(bright);
    for (let k = 0; k < 3; k++) this.col.push(col.r, col.g, col.b);
  }

  quad(a, b, c, d, hex, bright = 1) {
    this.tri(a, b, c, hex, bright);
    this.tri(b, d, c, hex, bright);
  }

  build(material) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(this.pos, 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(this.col, 3));
    geo.computeVertexNormals(); // non-indexed => true flat facets
    return new THREE.Mesh(geo, material);
  }
}

// The height model, queryable by other systems (vegetation, furniture) so
// everything sits exactly on the ground the field mesh renders.
export function terrainHeightAt(corridor, px, pz, s = null) {
  const seed = corridor.seed;
  const hw = corridor.halfWidth;
  const dShoulder = hw + 3.0;
  s = s || sampleRoadAt(corridor, px, pz);
  if (!s) return { y: 0, dist: 9999, roadY: 0 };

  const side = sideSignAt(corridor, s.index, px, pz);
  const cliff = corridor.cliff[s.index];
  const cliffness = side > 0 ? Math.max(0, cliff) : Math.max(0, -cliff);
  const valleyness = side > 0 ? Math.max(0, -cliff) : Math.max(0, cliff);
  const neutral = 1 - Math.abs(cliff);
  const d = s.distance;

  const beyond = Math.max(0, d - dShoulder);
  const rise = Math.min(95, Math.pow(beyond, 0.92) * 0.55);
  const drop = Math.min(60, Math.pow(beyond, 0.95) * 0.42);
  let relief = cliffness * rise - valleyness * drop;
  // back the near-road cliff band up with real mass so it doesn't read
  // as a floating fin
  relief += cliffness * smoothstep(dShoulder, dShoulder + 16, d) * 9.0;

  const n1 = fbm2(px * 0.012 + seed * 0.37, pz * 0.012 - seed * 0.61, 4);
  const n2 = fbm2(px * 0.05 + 31.7, pz * 0.05 + 11.3, 3);
  const amp = 2.0 + 10.0 * smoothstep(dShoulder, 150, d);
  relief += n1 * amp + n2 * amp * 0.3 + neutral * n1 * 5.0;

  // containment rim so the world reads as a mountain bowl, not a plate
  relief += smoothstep(180, 460, d) * 18 * (0.65 + 0.35 * n1);

  const t = smoothstep(dShoulder, dShoulder + 24, d);
  let y = s.y - 0.10 + t * relief;
  if (d < hw + 1.2) y = s.y - 0.14; // strictly under the road ribbon
  return { y, dist: d, roadY: s.y };
}

export function buildTerrain(corridor, mood) {
  const group = new THREE.Group();
  group.name = "terrain-v2";

  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    color: 0xffffff,
    side: THREE.DoubleSide,
  });
  mood.bindColor(mat, "color", "terrainTint", TAG);

  const field = buildField(corridor, mat);
  field.receiveShadow = true;
  group.add(field);

  const bands = buildBands(corridor, mat);
  bands.castShadow = true;
  bands.receiveShadow = true;
  group.add(bands);

  return group;
}

/* ── The heightfield ──────────────────────────────────────────── */

function buildField(corridor, mat) {
  const seed = corridor.seed;
  const hw = corridor.halfWidth;
  const dShoulder = hw + 3.0;

  const box = new THREE.Box3().setFromPoints(corridor.center);
  const c = box.getCenter(new THREE.Vector3());
  const margin = 420;
  const span = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) + margin * 2;
  const seg = 150;
  const N = seg + 1;

  const ys = new Float32Array(N * N);
  const dist = new Float32Array(N * N);
  const roadY = new Float32Array(N * N);

  for (let gz = 0; gz < N; gz++) {
    for (let gx = 0; gx < N; gx++) {
      const px = c.x + (gx / seg - 0.5) * span;
      const pz = c.z + (gz / seg - 0.5) * span;
      const k = gz * N + gx;
      const r = terrainHeightAt(corridor, px, pz);
      ys[k] = r.y;
      dist[k] = r.dist;
      roadY[k] = r.roadY;
    }
  }

  const buf = new FacetBuffer();
  const v = [new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3(), new THREE.Vector3()];
  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const nrm = new THREE.Vector3();

  const at = (gx, gz, out) => {
    out.set(
      c.x + (gx / seg - 0.5) * span,
      ys[gz * N + gx],
      c.z + (gz / seg - 0.5) * span,
    );
    return out;
  };

  let face = 0;
  for (let gz = 0; gz < seg; gz++) {
    for (let gx = 0; gx < seg; gx++) {
      const p00 = at(gx, gz, v[0]);
      const p01 = at(gx, gz + 1, v[1]);
      const p10 = at(gx + 1, gz, v[2]);
      const p11 = at(gx + 1, gz + 1, v[3]);
      // alternate the diagonal for a nicer facet pattern
      const flip = (gx + gz) % 2 === 1;
      const tris = flip
        ? [[p00, p01, p11], [p00, p11, p10]]
        : [[p00, p01, p10], [p01, p11, p10]];

      for (const [a, b, cc] of tris) {
        face++;
        const avgY = (a.y + b.y + cc.y) / 3;
        // metrics from the nearest stored vertex (cheap, good enough per-face)
        const k = gz * N + gx;
        const avgDist = dist[k];
        const hAbove = avgY - roadY[k];
        nrm.copy(e1.copy(b).sub(a)).cross(e2.copy(cc).sub(a)).normalize();
        const slope = 1 - Math.abs(nrm.y);

        const h = hash01(face * 7.13 + seed);
        let hex;
        if (avgDist < hw + 3.4) hex = pick(FACETS.dirt, h);
        else if (slope > 0.50) hex = pick(FACETS.rock, h);
        else if (hAbove > 40) hex = pick(FACETS.pale, h);
        else if (hAbove < -12) hex = pick(FACETS.valleyDeep, h);
        else if (slope > 0.30) hex = pick(FACETS.grassDark, h);
        else hex = pick(FACETS.grass, h);

        buf.tri(a, b, cc, hex, 0.93 + hash01(face * 3.7) * 0.14);
      }
    }
  }

  return buf.build(mat);
}

/* ── Near-road cliff walls + valley lips ──────────────────────── */

function cliffRails(corridor, i, sideSign) {
  const seed = corridor.seed;
  const cliff = corridor.cliff[i];
  const amt = sideSign > 0 ? Math.max(0, cliff) : Math.max(0, -cliff);
  const hf = smoothstep(0.10, 0.50, amt);
  const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
  const ctr = corridor.center[i];
  const out = new THREE.Vector3(edge.x - ctr.x, 0, edge.z - ctr.z).normalize();
  const d = corridor.distances[i];

  const n1 = fbm2(d * 0.045 + seed * 3.1, seed * 7.7, 3);
  const n2 = fbm2(d * 0.020 - seed * 1.7, seed * 4.3, 3);
  const j = hash01(i * 3.71 + seed);

  const toe = edge.clone().addScaledVector(out, 1.9 + j * 0.5);
  toe.y = edge.y - 0.08;
  const mid = edge.clone().addScaledVector(out, 4.6 + j * 1.4 + (n1 + 1) * 0.9);
  mid.y = edge.y + 2.6 + (n1 + 1) * 1.4;
  const top = edge.clone().addScaledVector(out, 10.5 + j * 2.5 + (n2 + 1) * 1.8);
  top.y = edge.y + 8.0 + (n2 + 1) * 3.0;

  // pinch to nothing where this side is not a cliff
  mid.lerp(toe, 1 - hf);
  top.lerp(toe, 1 - hf);
  return { toe, mid, top, hf };
}

function valleyRails(corridor, i, sideSign) {
  const seed = corridor.seed;
  const cliff = corridor.cliff[i];
  const amt = sideSign > 0 ? Math.max(0, -cliff) : Math.max(0, cliff);
  const vf = smoothstep(0.10, 0.50, amt);
  const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
  const ctr = corridor.center[i];
  const out = new THREE.Vector3(edge.x - ctr.x, 0, edge.z - ctr.z).normalize();
  const d = corridor.distances[i];

  const n1 = fbm2(d * 0.038 + seed * 5.3, seed * 2.9, 3);
  const j = hash01(i * 5.17 + seed);

  const lip = edge.clone().addScaledVector(out, 1.7 + j * 0.4);
  lip.y = edge.y - 0.10;
  const bench = edge.clone().addScaledVector(out, 6.5 + j * 2.0);
  bench.y = edge.y - (1.7 + (n1 + 1) * 0.5);
  const slope = edge.clone().addScaledVector(out, 16.0 + j * 5.0);
  slope.y = edge.y - (5.5 + (n1 + 1) * 1.5);

  bench.lerp(lip, 1 - vf);
  slope.lerp(lip, 1 - vf);
  return { lip, bench, slope, vf };
}

function buildBands(corridor, mat) {
  const seed = corridor.seed;
  const buf = new FacetBuffer();
  const n = corridor.center.length;

  for (const sideSign of [1, -1]) {
    let prevC = cliffRails(corridor, 0, sideSign);
    let prevV = valleyRails(corridor, 0, sideSign);
    for (let i = 1; i < n; i++) {
      const curC = cliffRails(corridor, i, sideSign);
      const curV = valleyRails(corridor, i, sideSign);
      const h = hash01(i * 9.31 + seed + sideSign * 41);

      if (prevC.hf > 0.02 || curC.hf > 0.02) {
        buf.quad(prevC.toe, curC.toe, prevC.mid, curC.mid,
          pick(FACETS.rock, h), 0.92 + hash01(i * 2.3) * 0.16);
        buf.quad(prevC.mid, curC.mid, prevC.top, curC.top,
          h > 0.78 ? pick(FACETS.rockLight, h) : pick(FACETS.rock, hash01(i * 6.7 + 2)),
          0.90 + hash01(i * 4.9) * 0.18);
      }
      if (prevV.vf > 0.02 || curV.vf > 0.02) {
        buf.quad(prevV.lip, curV.lip, prevV.bench, curV.bench,
          pick(FACETS.grassDark, h), 0.93 + hash01(i * 3.1) * 0.12);
        buf.quad(prevV.bench, curV.bench, prevV.slope, curV.slope,
          h > 0.6 ? pick(FACETS.valleyDeep, h) : pick(FACETS.grassDark, hash01(i * 8.3 + 1)),
          0.90 + hash01(i * 5.7) * 0.14);
      }
      prevC = curC;
      prevV = curV;
    }
  }

  return buf.build(mat);
}
