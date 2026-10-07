// Mountain backdrop, two layers:
//  1) Mid ring — real low-poly peak clusters (two-tier jagged cones, merged
//     into ONE flat-shaded mesh). Snow by facet rule: flat-enough facets above
//     a jittered snowline go white, steep facets stay rock — the day pic's
//     peak look. Scene fog supplies day haze and night fade.
//  2) Far ring — two unlit silhouette ridge strips, fog-EXEMPT, with authored
//     aerial-perspective tints (mood-bound). Grayscale vertex colors carry the
//     snow band so one material color tints body+snow together.
import * as THREE from "../vendor/three.module.min.js";
import { FacetBuffer } from "./terrain.js";
import { sampleRoadAt } from "./corridor.js";
import { hash01, smoothstep } from "./noise.js";

const TAG = "world";

// Day-authored facet colors for mid peaks (night via mountainTint + lighting).
// NOTE: this three build reads hex as LINEAR (no sRGB conversion), so darks
// must be authored much darker than they'd look in a color picker. Snow stays
// bright, rock goes deep — fog at range eats contrast.
const SNOW = [0xf2f7fa, 0xe7eff5];
const SNOW_SHADE = [0xaabccc, 0x9cb0c2];
const ROCK = [0x2a2e35, 0x20242a, 0x383d44, 0x24282e];

function pick(list, h) {
  return list[Math.min(list.length - 1, Math.floor(h * list.length))];
}

function trackBounds(corridor) {
  const box = new THREE.Box3().setFromPoints(corridor.center);
  return {
    center: box.getCenter(new THREE.Vector3()),
    ring: Math.max(box.max.x - box.min.x, box.max.z - box.min.z) * 0.5,
  };
}

export function buildMountains(corridor, mood) {
  const group = new THREE.Group();
  group.name = "mountains-v2";
  const { center, ring } = trackBounds(corridor);
  const seed = corridor.seed;

  group.add(buildMidPeaks(corridor, center, ring, seed, mood));
  group.add(ridgeStrip(center, ring + 950, 260, 120, seed + 31, mood, "ridgeMid"));
  group.add(ridgeStrip(center, ring + 1500, 420, 185, seed + 77, mood, "ridgeFar"));
  return group;
}

/* ── Mid ring: jagged snow-capped peak clusters ───────────────── */

function addPeak(buf, px, pz, baseY, R, H, seed) {
  const K = 7 + Math.floor(hash01(seed * 3.1) * 3);
  const snowY = baseY + H * (0.40 + hash01(seed * 5.7) * 0.12);

  const apex = new THREE.Vector3(
    px + (hash01(seed * 7.3) - 0.5) * R * 0.22,
    baseY + H,
    pz + (hash01(seed * 9.9) - 0.5) * R * 0.22,
  );
  const upper = [];
  const base = [];
  for (let k = 0; k < K; k++) {
    const a = (k / K) * Math.PI * 2;
    const ju = 0.75 + hash01(seed * 11.3 + k * 1.7) * 0.5;
    const jb = 0.80 + hash01(seed * 13.9 + k * 2.3) * 0.4;
    const yu = baseY + H * (0.46 + (hash01(seed * 17.1 + k * 3.1) - 0.5) * 0.16);
    upper.push(new THREE.Vector3(
      px + Math.cos(a) * R * 0.42 * ju, yu, pz + Math.sin(a) * R * 0.42 * ju,
    ));
    // base driven deep (−320) so it always sits below the terrain rim that
    // occludes it — a shallow base poked above low/valley horizons and read
    // as a floating flat-bottomed cone
    base.push(new THREE.Vector3(
      px + Math.cos(a) * R * jb, baseY - 320, pz + Math.sin(a) * R * jb,
    ));
  }

  const e1 = new THREE.Vector3();
  const e2 = new THREE.Vector3();
  const nrm = new THREE.Vector3();
  const colorFor = (a, b, c, h) => {
    nrm.copy(e1.copy(b).sub(a)).cross(e2.copy(c).sub(a)).normalize();
    const slope = 1 - Math.abs(nrm.y);
    const cy = (a.y + b.y + c.y) / 3;
    if (cy > snowY && slope < 0.72) {
      return pick(slope > 0.55 ? SNOW_SHADE : SNOW, h);
    }
    return pick(ROCK, h);
  };

  for (let k = 0; k < K; k++) {
    const k2 = (k + 1) % K;
    const h1 = hash01(seed * 19.3 + k * 5.1);
    const h2 = hash01(seed * 23.7 + k * 7.7);
    const capCol = colorFor(apex, upper[k], upper[k2], h1);
    buf.tri(apex, upper[k], upper[k2], capCol, 0.94 + h1 * 0.12);
    const flankColA = colorFor(upper[k], base[k], base[k2], h2);
    buf.tri(upper[k], base[k], base[k2], flankColA, 0.92 + h2 * 0.14);
    const flankColB = colorFor(upper[k], base[k2], upper[k2], h2);
    buf.tri(upper[k], base[k2], upper[k2], flankColB, 0.92 + h2 * 0.14);
  }
}

function buildMidPeaks(corridor, center, ring, seed, mood) {
  const buf = new FacetBuffer();
  const slots = 16;
  // square-ish tracks reach diagonally past the bbox ring — never drop a
  // peak onto (or hard against) the road itself
  const safe = (px, pz, R) => {
    const s = sampleRoadAt(corridor, px, pz);
    return !s || s.distance > R + 60;
  };
  for (let s = 0; s < slots; s++) {
    const hs = hash01(seed * 41.7 + s * 13.1);
    if (hs < 0.22) continue; // gaps so the ring isn't a perfect crown
    const a = ((s + (hs - 0.5) * 0.9) / slots) * Math.PI * 2;
    const radius = ring + 430 + hash01(seed * 47.3 + s * 17.9) * 320;
    const px = center.x + Math.cos(a) * radius;
    const pz = center.z + Math.sin(a) * radius;
    const R = 70 + hash01(seed * 53.1 + s * 19.3) * 75;
    const H = 110 + hash01(seed * 59.9 + s * 23.7) * 150;

    if (safe(px, pz, R)) addPeak(buf, px, pz, 0, R, H, seed + s * 101);
    // sub-summits make multi-peak massifs
    const subs = hash01(seed * 61.3 + s * 29.1) > 0.45 ? 2 : 1;
    for (let q = 0; q < subs; q++) {
      const sa = a + (hash01(seed * 67.7 + s * 31.7 + q) - 0.5) * 0.30;
      const sr = radius + (hash01(seed * 71.3 + s * 37.1 + q) - 0.5) * 160;
      const sx = center.x + Math.cos(sa) * sr;
      const sz = center.z + Math.sin(sa) * sr;
      const subR = R * (0.5 + hash01(seed * 73.9 + q * 3.3) * 0.25);
      if (!safe(sx, sz, subR)) continue;
      addPeak(
        buf,
        sx,
        sz,
        0,
        subR,
        H * (0.45 + hash01(seed * 79.1 + q * 5.9) * 0.30),
        seed + s * 101 + q * 977 + 13,
      );
    }
  }

  const mat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    color: 0xffffff,
  });
  mood.bindColor(mat, "color", "mountainTint", TAG);
  const mesh = buf.build(mat);
  mesh.receiveShadow = false;
  mesh.castShadow = false;
  return mesh;
}

/* ── Far ring: unlit silhouette ridge strips ──────────────────── */

function ridgeStrip(center, radius, height, snowY, seed, mood, paletteKey) {
  const segs = 220;
  const phase1 = hash01(seed * 1.3) * Math.PI * 2;
  const phase2 = hash01(seed * 2.7) * Math.PI * 2;
  const phase3 = hash01(seed * 4.1) * Math.PI * 2;

  const pos = [];
  const col = [];
  const index = [];

  for (let s = 0; s < segs; s++) {
    const th = (s / segs) * Math.PI * 2;
    // integer frequencies => loop-periodic, no seam
    const n =
      0.50 * Math.sin(th * 3 + phase1) +
      0.30 * Math.sin(th * 7 + phase2) +
      0.20 * Math.sin(th * 13 + phase3);
    const serration = Math.abs(Math.sin(th * 23 + phase2)) * 0.18;
    const jag = Math.pow(0.5 + 0.5 * n, 1.35) + serration;
    const r = radius * (1 + 0.06 * Math.sin(th * 5 + phase3));
    const x = center.x + Math.cos(th) * r;
    const z = center.z + Math.sin(th) * r;
    const topY = height * (0.42 + 0.58 * jag);

    pos.push(x, -320, z, x, topY, z);   // base deep below the terrain rim
    const snow = smoothstep(snowY - 25, snowY + 35, topY);
    const bodyG = 0.62 + hash01(seed * 91.1 + s * 3.7) * 0.10;
    const topG = bodyG + (1.0 - bodyG) * snow;
    col.push(bodyG * 0.82, bodyG * 0.82, bodyG * 0.82, topG, topG, topG);

    const a = s * 2;
    const b = ((s + 1) % segs) * 2;
    index.push(a, b, a + 1, a + 1, b, b + 1);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("color", new THREE.Float32BufferAttribute(col, 3));
  geo.setIndex(index);

  const mat = new THREE.MeshBasicMaterial({
    vertexColors: true,
    color: 0xffffff,
    fog: false, // aerial perspective is authored, not accidental
    side: THREE.DoubleSide,
  });
  mood.bindColor(mat, "color", paletteKey, TAG);
  return new THREE.Mesh(geo, mat);
}
