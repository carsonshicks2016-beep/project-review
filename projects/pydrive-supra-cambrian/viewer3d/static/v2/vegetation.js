// Vegetation: multi-tier pines (trunk + 3 stacked cones merged into ONE
// geometry -> one InstancedMesh = one draw call for the whole forest) and
// jittered boulders. Placement keys off the corridor side signal: dense bands
// descending the valley side into the mist, clusters perched above the cliff
// walls, sparse scatter on neutral stretches, nothing near the road, nothing
// above the treeline.
import * as THREE from "../vendor/three.module.min.js";
import { hash01 } from "./noise.js";
import { terrainHeightAt } from "./terrain.js";

const TAG = "world";

function mergedPineGeometry() {
  const parts = [];
  const add = (geo, y, hex) => {
    geo.translate(0, y, 0);
    parts.push({ geo: geo.toNonIndexed(), hex });
  };
  // authored dark (hex reads linear in this three build)
  add(new THREE.CylinderGeometry(0.09, 0.17, 1.1, 5), 0.55, 0x241a10);
  add(new THREE.ConeGeometry(1.05, 1.35, 7), 1.55, 0x0e2414);
  add(new THREE.ConeGeometry(0.80, 1.20, 7), 2.42, 0x112a18);
  add(new THREE.ConeGeometry(0.55, 1.05, 6), 3.20, 0x14301b);

  let total = 0;
  for (const p of parts) total += p.geo.getAttribute("position").count;
  const pos = new Float32Array(total * 3);
  const nrm = new Float32Array(total * 3);
  const col = new Float32Array(total * 3);
  const c = new THREE.Color();
  let o = 0;
  for (const p of parts) {
    const pp = p.geo.getAttribute("position");
    const pn = p.geo.getAttribute("normal");
    pos.set(pp.array, o * 3);
    nrm.set(pn.array, o * 3);
    c.set(p.hex);
    for (let i = 0; i < pp.count; i++) {
      col[(o + i) * 3] = c.r;
      col[(o + i) * 3 + 1] = c.g;
      col[(o + i) * 3 + 2] = c.b;
    }
    o += pp.count;
    p.geo.dispose();
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(nrm, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
  return geo;
}

function boulderGeometry(seed) {
  const geo = new THREE.BoxGeometry(1.5, 1.15, 1.35, 2, 2, 2).toNonIndexed();
  const p = geo.getAttribute("position");
  for (let i = 0; i < p.count; i++) {
    p.setXYZ(
      i,
      p.getX(i) + (hash01(seed + i * 3.13) - 0.5) * 0.34,
      p.getY(i) + (hash01(seed + i * 5.29) - 0.5) * 0.26,
      p.getZ(i) + (hash01(seed + i * 7.71) - 0.5) * 0.34,
    );
  }
  geo.computeVertexNormals();
  return geo;
}

export function buildVegetation(corridor, mood) {
  const group = new THREE.Group();
  group.name = "vegetation-v2";
  const n = corridor.center.length;
  const seed = corridor.seed;
  const dummy = new THREE.Object3D();
  const tint = new THREE.Color();
  const out = new THREE.Vector3();

  /* ── Pines ── */
  const MAXP = 420;
  const pineMat = new THREE.MeshLambertMaterial({
    vertexColors: true,
    color: 0xffffff,
    flatShading: true,
  });
  mood.bindColor(pineMat, "color", "pineTint", TAG);
  const pines = new THREE.InstancedMesh(mergedPineGeometry(), pineMat, MAXP);
  pines.castShadow = true;
  let pCount = 0;

  for (let i = 2; i < n - 2 && pCount < MAXP; i += 3) {
    for (const sideSign of [1, -1]) {
      if (pCount >= MAXP) break;
      const cliff = corridor.cliff[i];
      const amtV = sideSign > 0 ? Math.max(0, -cliff) : Math.max(0, cliff);
      const amtC = sideSign > 0 ? Math.max(0, cliff) : Math.max(0, -cliff);
      const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
      const ctr = corridor.center[i];
      out.set(edge.x - ctr.x, 0, edge.z - ctr.z).normalize();
      const h0 = hash01(i * 13.7 + seed + sideSign * 29);

      const spots = [];
      if (amtV > 0.4) {
        // descending bands toward the mist
        const rows = 1 + Math.floor(h0 * 3);
        for (let r = 0; r < rows; r++) {
          if (hash01(i * 3.3 + r * 7.1 + seed) < 0.25) continue;
          spots.push(7 + r * 9 + hash01(i * 5.9 + r * 3.7 + seed) * 6);
        }
      } else if (amtC > 0.4) {
        // perched above the rock wall
        if (h0 > 0.45) spots.push(13 + h0 * 16);
      } else if (h0 > 0.62) {
        spots.push(8 + h0 * 24);
      }

      for (const off of spots) {
        if (pCount >= MAXP) break;
        const px = edge.x + out.x * off + (hash01(i * 9.1 + off * 1.7) - 0.5) * 4;
        const pz = edge.z + out.z * off + (hash01(i * 11.3 + off * 2.3) - 0.5) * 4;
        const t = terrainHeightAt(corridor, px, pz);
        if (t.dist < corridor.halfWidth + 4.5) continue; // road exclusion
        if (t.y - t.roadY > 26) continue;                // treeline
        const s = 1.15 + hash01(i * 17.9 + off * 3.1 + seed) * 1.1;
        dummy.position.set(px, t.y - 0.25, pz);
        dummy.rotation.set(0, hash01(i * 19.3 + off) * Math.PI * 2, 0);
        dummy.scale.set(s, s * (0.9 + hash01(i * 23.1 + off) * 0.3), s);
        dummy.updateMatrix();
        pines.setMatrixAt(pCount, dummy.matrix);
        const cj = 0.78 + hash01(i * 29.7 + off * 5.3) * 0.5;
        tint.setRGB(cj * (0.88 + hash01(i * 31.1 + off) * 0.26), cj, cj * 0.92);
        pines.setColorAt(pCount, tint);
        pCount++;
      }
    }
  }
  pines.count = pCount;
  pines.instanceMatrix.needsUpdate = true;
  if (pines.instanceColor) pines.instanceColor.needsUpdate = true;
  group.add(pines);

  /* ── Boulders ── */
  const MAXB = 90;
  const boulderMat = new THREE.MeshLambertMaterial({
    color: 0x76776f,
    flatShading: true,
  });
  mood.bindColor(boulderMat, "color", "boulder", TAG);
  const boulders = new THREE.InstancedMesh(boulderGeometry(seed * 1.7), boulderMat, MAXB);
  boulders.castShadow = true;
  boulders.receiveShadow = true;
  let bCount = 0;

  for (let i = 4; i < n - 4 && bCount < MAXB; i += 5) {
    for (const sideSign of [1, -1]) {
      if (bCount >= MAXB) break;
      const cliff = corridor.cliff[i];
      const amtV = sideSign > 0 ? Math.max(0, -cliff) : Math.max(0, cliff);
      const amtC = sideSign > 0 ? Math.max(0, cliff) : Math.max(0, -cliff);
      const h = hash01(i * 41.3 + seed + sideSign * 13);
      let off = null;
      if (amtC > 0.25 && amtC < 0.8 && h > 0.55) off = 2.6 + h * 3.2; // cliff toe
      else if (amtV > 0.4 && h > 0.78) off = 6 + h * 7;               // valley bench
      if (off === null) continue;

      const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
      const ctr = corridor.center[i];
      out.set(edge.x - ctr.x, 0, edge.z - ctr.z).normalize();
      const px = edge.x + out.x * off;
      const pz = edge.z + out.z * off;
      const t = terrainHeightAt(corridor, px, pz);
      // tight tracks: don't let a boulder placed off THIS segment land on a
      // neighboring pass of the road
      if (t.dist < corridor.halfWidth + 1.8) continue;
      const s = 0.5 + hash01(i * 47.7 + seed) * 1.1;
      dummy.position.set(px, t.y + 0.32 * s, pz);
      dummy.rotation.set(0, hash01(i * 53.9) * Math.PI * 2, 0);
      dummy.scale.set(s, s * (0.8 + h * 0.5), s * (0.85 + h * 0.4));
      dummy.updateMatrix();
      boulders.setMatrixAt(bCount++, dummy.matrix);
    }
  }
  boulders.count = bCount;
  boulders.instanceMatrix.needsUpdate = true;
  group.add(boulders);

  return group;
}
