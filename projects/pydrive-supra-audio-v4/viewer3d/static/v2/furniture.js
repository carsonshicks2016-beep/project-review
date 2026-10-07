// Road furniture: guardrails, curved-arm sodium lamps (instanced, with a
// small pool of real PointLights that follows the car), candy delineators,
// hairpin chevrons, masonry retaining walls with snow caps, and one tunnel
// portal at the track's best cliff pinch. Repeated items are instanced or
// merged; the lamp glow halos are a single THREE.Points draw call.
import * as THREE from "../vendor/three.module.min.js";
import { hash01, smoothstep } from "./noise.js";
import { terrainHeightAt } from "./terrain.js";

// Bury a wall's base below the local terrain so it never floats over a drop
// (the terrain mesh is built from this same sampler, so the base is guaranteed
// to be under the visible ground — no see-through gap).
function groundedBase(corridor, point, roadFallback, skirt = 1.2) {
  const t = terrainHeightAt(corridor, point.x, point.z);
  return Math.min(roadFallback, t.y - skirt);
}

const TAG = "world";

/* ── Generated textures ───────────────────────────────────────── */

function masonryTexture() {
  const s = 256;
  const canvas = document.createElement("canvas");
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#56544c";
  ctx.fillRect(0, 0, s, s);
  const rowH = 32;
  for (let row = 0; row < s / rowH; row++) {
    const offset = row % 2 === 0 ? 0 : 36;
    for (let x = -36; x < s + 36; x += 72) {
      const v = 120 + ((row * 7 + x) % 38);
      ctx.fillStyle = `rgb(${v},${v - 4},${v - 12})`;
      ctx.fillRect(x + offset + 2, row * rowH + 2, 68, rowH - 4);
      ctx.fillStyle = "rgba(255,255,255,0.07)";
      ctx.fillRect(x + offset + 2, row * rowH + 2, 68, 4);
    }
  }
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function chevronTexture() {
  const s = 128;
  const canvas = document.createElement("canvas");
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#d8b32e";
  ctx.fillRect(0, 0, s, s);
  ctx.fillStyle = "#15171a";
  for (let x = -28; x < s + 28; x += 44) {
    ctx.beginPath();
    ctx.moveTo(x, 16);
    ctx.lineTo(x + 22, s / 2);
    ctx.lineTo(x, s - 16);
    ctx.lineTo(x + 12, s - 16);
    ctx.lineTo(x + 34, s / 2);
    ctx.lineTo(x + 12, 16);
    ctx.closePath();
    ctx.fill();
  }
  ctx.fillStyle = "#15171a";
  ctx.fillRect(0, 0, s, 6);
  ctx.fillRect(0, s - 6, s, 6);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function delineatorTexture() {
  const s = 64;
  const canvas = document.createElement("canvas");
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#e8eaec";
  ctx.fillRect(0, 0, s, s);
  ctx.fillStyle = "#c33232";
  ctx.fillRect(0, 6, s, 16);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function glowTexture() {
  const s = 128;
  const canvas = document.createElement("canvas");
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 2, 64, 64, 62);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.25, "rgba(255,235,200,0.55)");
  g.addColorStop(1, "rgba(255,220,160,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, s, s);
  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

/* ── Small geometry helpers ───────────────────────────────────── */

function ribbonUV(aPts, bPts, material, vScale) {
  const verts = [];
  const uvs = [];
  const indices = [];
  let dist = 0;
  for (let i = 0; i < aPts.length; i++) {
    if (i > 0) dist += aPts[i].distanceTo(aPts[i - 1]);
    verts.push(aPts[i].x, aPts[i].y, aPts[i].z, bPts[i].x, bPts[i].y, bPts[i].z);
    uvs.push(0, dist * vScale, 1, dist * vScale);
  }
  for (let i = 0; i < aPts.length - 1; i++) {
    const a = i * 2;
    indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, material);
}

function outwardAt(corridor, i, sideSign) {
  const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
  const ctr = corridor.center[i];
  return new THREE.Vector3(edge.x - ctr.x, 0, edge.z - ctr.z).normalize();
}

function roadYawAt(corridor, i) {
  const n = corridor.center.length;
  const a = corridor.center[Math.max(0, i - 1)];
  const b = corridor.center[Math.min(n - 1, i + 1)];
  return Math.atan2(-(b.z - a.z), b.x - a.x);
}

function cliffAmtAt(corridor, i, sideSign) {
  const c = corridor.cliff[i];
  return sideSign > 0 ? Math.max(0, c) : Math.max(0, -c);
}

/* ── Tunnel span selection ────────────────────────────────────── */

function findTunnelSpan(corridor) {
  const n = corridor.center.length;
  let best = null;
  let runStart = -1;
  for (let i = 0; i <= n; i++) {
    const strong = i < n && Math.abs(corridor.cliff[i]) > 0.6;
    if (strong && runStart < 0) runStart = i;
    if (!strong && runStart >= 0) {
      const len = i - runStart;
      if (!best || len > best.len) best = { start: runStart, len };
      runStart = -1;
    }
  }
  if (!best || best.len < 12) return null;
  const span = Math.min(best.len, 22);
  const mid = best.start + Math.floor(best.len / 2);
  const i0 = Math.max(1, mid - Math.floor(span / 2));
  const i1 = Math.min(n - 2, i0 + span);
  return { i0, i1, side: Math.sign(corridor.cliff[mid]) || 1 };
}

/* ── Main builder ─────────────────────────────────────────────── */

export function buildFurniture(corridor, mood) {
  const group = new THREE.Group();
  group.name = "furniture-v2";

  const tunnel = findTunnelSpan(corridor);
  const inTunnel = (i) => tunnel && i >= tunnel.i0 - 2 && i <= tunnel.i1 + 2;

  const masonry = masonryTexture();
  const wallMat = new THREE.MeshLambertMaterial({
    map: masonry,
    color: 0xffffff,
    side: THREE.DoubleSide,
  });
  mood.bindColor(wallMat, "color", "wall", TAG);

  const heads = [];
  const glowPositions = [];

  buildGuardrails(corridor, mood, group, inTunnel);
  buildRetainingWalls(corridor, mood, group, inTunnel, wallMat);
  buildLamps(corridor, mood, group, inTunnel, heads, glowPositions);
  buildDelineators(corridor, mood, group, inTunnel);
  buildChevrons(corridor, mood, group, inTunnel);
  if (tunnel) buildTunnel(corridor, mood, group, tunnel, wallMat, glowPositions);
  buildGlowPoints(mood, group, glowPositions);
  buildLampPools(mood, group, heads);
  const update = buildLightPool(mood, group, heads);

  return { group, update };
}

/* ── Guardrails ───────────────────────────────────────────────── */

function buildGuardrails(corridor, mood, group, inTunnel) {
  const n = corridor.center.length;
  const railMat = new THREE.MeshLambertMaterial({
    color: 0x8a9298,
    side: THREE.DoubleSide,
  });
  mood.bindColor(railMat, "color", "guardrail", TAG);

  const postGeo = new THREE.BoxGeometry(0.12, 0.78, 0.14);
  const posts = new THREE.InstancedMesh(postGeo, railMat, 700);
  posts.castShadow = true;
  const dummy = new THREE.Object3D();
  let postCount = 0;

  for (const sideSign of [1, -1]) {
    const include = (i) => {
      if (inTunnel(i)) return false;
      // cliff walls replace rails on strong cliff stretches
      return cliffAmtAt(corridor, i, sideSign) < 0.35;
    };

    let run = [];
    const flush = () => {
      if (run.length > 2) {
        const a1 = [];
        const b1 = [];
        const a2 = [];
        const b2 = [];
        for (const i of run) {
          const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
          const out = outwardAt(corridor, i, sideSign);
          const jitter = (hash01(i * 3.91 + corridor.seed) - 0.5) * 0.04;
          const base = edge.clone().addScaledVector(out, 1.15 + jitter);
          const mk = (y0, y1, arrA, arrB) => {
            const lo = base.clone();
            lo.y = edge.y + y0;
            const hi = base.clone();
            hi.y = edge.y + y1;
            arrA.push(lo);
            arrB.push(hi);
          };
          mk(0.46, 0.62, a1, b1);
          mk(0.66, 0.82, a2, b2);
        }
        group.add(ribbonUV(a1, b1, railMat, 0.2));
        group.add(ribbonUV(a2, b2, railMat, 0.2));
        for (let k = 0; k < run.length; k += 3) {
          if (postCount >= 700) break;
          const i = run[k];
          const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
          const out = outwardAt(corridor, i, sideSign);
          dummy.position.copy(edge).addScaledVector(out, 1.17);
          dummy.position.y = edge.y + 0.33;
          dummy.rotation.set(0, roadYawAt(corridor, i), 0);
          dummy.updateMatrix();
          posts.setMatrixAt(postCount++, dummy.matrix);
        }
      }
      run = [];
    };

    for (let i = 0; i < n; i++) {
      if (include(i)) run.push(i);
      else flush();
    }
    flush();
  }

  posts.count = postCount;
  posts.instanceMatrix.needsUpdate = true;
  group.add(posts);
}

/* ── Retaining walls (cliff side) ─────────────────────────────── */

function buildRetainingWalls(corridor, mood, group, inTunnel, wallMat) {
  const n = corridor.center.length;
  const snowMat = new THREE.MeshLambertMaterial({
    color: 0xf2f7fa,
    side: THREE.DoubleSide,
  });
  mood.bindColor(snowMat, "color", "wallSnow", TAG);

  for (const sideSign of [1, -1]) {
    let run = [];
    const flush = () => {
      if (run.length > 3) {
        const faceA = [];
        const faceB = [];
        const capA = [];
        const capB = [];
        const snowA = [];
        const snowB = [];
        for (const i of run) {
          const f = smoothstep(0.45, 0.62, cliffAmtAt(corridor, i, sideSign));
          const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
          const out = outwardAt(corridor, i, sideSign);
          const h = (1.85 + hash01(i * 7.7 + corridor.seed) * 0.25) * f;
          const inner = edge.clone().addScaledVector(out, 1.55);
          const lo = inner.clone();
          lo.y = groundedBase(corridor, inner, edge.y - 0.15);
          const hi = inner.clone();
          hi.y = edge.y + Math.max(0.02, h);
          faceA.push(lo);
          faceB.push(hi);
          const capOut = hi.clone().addScaledVector(out, 0.45);
          capA.push(hi);
          capB.push(capOut);
          const sA = hi.clone();
          sA.y += 0.045;
          const sB = capOut.clone();
          sB.y += 0.045;
          snowA.push(sA);
          snowB.push(sB);
        }
        group.add(ribbonUV(faceA, faceB, wallMat, 0.45));
        group.add(ribbonUV(capA, capB, wallMat, 0.45));
        group.add(ribbonUV(snowA, snowB, snowMat, 0.45));
      }
      run = [];
    };

    for (let i = 0; i < n; i++) {
      const f = smoothstep(0.45, 0.62, cliffAmtAt(corridor, i, sideSign));
      if (f > 0.02 && !inTunnel(i)) run.push(i);
      else flush();
    }
    flush();
  }
}

/* ── Street lamps ─────────────────────────────────────────────── */

function buildLamps(corridor, mood, group, inTunnel, heads, glowPositions) {
  const n = corridor.center.length;
  const seed = corridor.seed;
  const MAX = 84;

  const poleMat = new THREE.MeshLambertMaterial({ color: 0x3a4046 });
  mood.bindColor(poleMat, "color", "lampPole", TAG);
  const headMat = new THREE.MeshBasicMaterial({ color: 0xffc46a });
  mood.bindColor(headMat, "color", "lampHead", TAG);

  const poleGeo = new THREE.BoxGeometry(0.14, 4.7, 0.14);
  const arm1Geo = new THREE.BoxGeometry(0.11, 0.11, 1.35);
  const arm2Geo = new THREE.BoxGeometry(0.11, 0.11, 1.05);
  const headGeo = new THREE.BoxGeometry(0.30, 0.14, 0.78);

  const poles = new THREE.InstancedMesh(poleGeo, poleMat, MAX);
  const arms1 = new THREE.InstancedMesh(arm1Geo, poleMat, MAX);
  const arms2 = new THREE.InstancedMesh(arm2Geo, poleMat, MAX);
  const headBoxes = new THREE.InstancedMesh(headGeo, headMat, MAX);
  poles.castShadow = true;

  const dummy = new THREE.Object3D();
  let count = 0;
  let parity = 0;
  let i = 0;

  for (let d = 26; d < corridor.length - 12 && count < MAX;
       d += 40 + hash01(d * 0.13 + seed) * 16) {
    while (i < n - 1 && corridor.distances[i] < d) i++;
    if (inTunnel(i)) continue;

    const cliff = corridor.cliff[i];
    let sideSign;
    if (Math.abs(cliff) > 0.18) sideSign = cliff > 0 ? 1 : -1;
    else sideSign = (parity++ % 2) === 0 ? 1 : -1;

    const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
    const out = outwardAt(corridor, i, sideSign);
    const inward = out.clone().multiplyScalar(-1);
    const base = edge.clone().addScaledVector(out, 1.6);
    base.y = edge.y - 0.05;
    const theta = Math.atan2(inward.x, inward.z);

    dummy.rotation.set(0, theta, 0);
    dummy.position.copy(base);
    dummy.position.y += 2.35;
    dummy.updateMatrix();
    poles.setMatrixAt(count, dummy.matrix);

    dummy.rotation.set(-0.5, theta, 0, "YXZ");
    dummy.position.copy(base).addScaledVector(inward, 0.55);
    dummy.position.y = base.y + 4.62 + 0.28;
    dummy.updateMatrix();
    arms1.setMatrixAt(count, dummy.matrix);

    dummy.rotation.set(0, theta, 0);
    dummy.position.copy(base).addScaledVector(inward, 1.62);
    dummy.position.y = base.y + 5.18;
    dummy.updateMatrix();
    arms2.setMatrixAt(count, dummy.matrix);

    const headPos = base.clone().addScaledVector(inward, 2.05);
    headPos.y = base.y + 5.10;
    dummy.rotation.set(0, theta, 0);
    dummy.position.copy(headPos);
    dummy.updateMatrix();
    headBoxes.setMatrixAt(count, dummy.matrix);

    const poolPos = base.clone().addScaledVector(inward, 2.6);
    poolPos.y = edge.y + 0.05;
    heads.push({ head: headPos, poolPos, yaw: roadYawAt(corridor, i) });
    glowPositions.push(headPos.x, headPos.y - 0.10, headPos.z);
    count++;
  }

  for (const m of [poles, arms1, arms2, headBoxes]) {
    m.count = count;
    m.instanceMatrix.needsUpdate = true;
    group.add(m);
  }
}

/* ── Lamp glow halos: one Points draw call ────────────────────── */

function buildGlowPoints(mood, group, glowPositions) {
  if (!glowPositions.length) return;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(glowPositions, 3));
  const mat = new THREE.PointsMaterial({
    map: glowTexture(),
    color: 0xffc674,
    size: 5.2,
    sizeAttenuation: true,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  mood.bindNumber(mat, "opacity", "lampGlowOpacity", TAG);
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  points.renderOrder = 8;
  group.add(points);
}

/* ── Warm pools on the tarmac under each lamp ─────────────────── */

function buildLampPools(mood, group, heads) {
  if (!heads.length) return;
  const pos = [];
  const uv = [];
  const index = [];
  let q = 0;
  for (const { poolPos, yaw } of heads) {
    const fx = Math.cos(yaw);
    const fz = -Math.sin(yaw);
    const sx = Math.sin(yaw);
    const sz = Math.cos(yaw);
    const hw = 3.6; // along road
    const hd = 2.3; // across
    const corners = [
      [poolPos.x - fx * hw - sx * hd, poolPos.z - fz * hw - sz * hd],
      [poolPos.x + fx * hw - sx * hd, poolPos.z + fz * hw - sz * hd],
      [poolPos.x - fx * hw + sx * hd, poolPos.z - fz * hw + sz * hd],
      [poolPos.x + fx * hw + sx * hd, poolPos.z + fz * hw + sz * hd],
    ];
    for (const [cx, cz] of corners) pos.push(cx, poolPos.y, cz);
    uv.push(0, 0, 1, 0, 0, 1, 1, 1);
    const b = q * 4;
    index.push(b, b + 1, b + 2, b + 1, b + 3, b + 2);
    q++;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  geo.setIndex(index);
  const mat = new THREE.MeshBasicMaterial({
    map: glowTexture(),
    color: 0xffb257,
    transparent: true,
    opacity: 0.5,
    depthWrite: false,
    side: THREE.DoubleSide,
    blending: THREE.AdditiveBlending,
  });
  mood.bindNumber(mat, "opacity", "lampPoolOpacity", TAG);
  const mesh = new THREE.Mesh(geo, mat);
  mesh.renderOrder = 7;
  group.add(mesh);
}

/* ── Pooled real lights that follow the car ───────────────────── */

function buildLightPool(mood, group, heads) {
  const POOL = 6;
  const lights = [];
  for (let k = 0; k < POOL; k++) {
    const l = new THREE.PointLight(0xffb257, 0, 40, 1.8);
    group.add(l);
    lights.push(l);
  }
  const holder = { value: 14 };
  mood.bindNumber(holder, "value", "lampIntensity", TAG);

  return function update(carPos) {
    if (!heads.length) {
      for (const l of lights) l.intensity = 0;
      return;
    }
    const order = heads
      .map((h, idx) => ({ idx, d: h.head.distanceToSquared(carPos) }))
      .sort((a, b) => a.d - b.d);
    for (let k = 0; k < lights.length; k++) {
      const src = order[Math.min(k, order.length - 1)];
      lights[k].position.copy(heads[src.idx].head);
      lights[k].position.y -= 0.15;
      lights[k].intensity = holder.value;
    }
  };
}

/* ── Candy-stripe delineator posts ────────────────────────────── */

function buildDelineators(corridor, mood, group, inTunnel) {
  const n = corridor.center.length;
  const MAX = 280;
  const mat = new THREE.MeshBasicMaterial({ map: delineatorTexture() });
  mood.bindColor(mat, "color", "delineator", TAG);
  const geo = new THREE.BoxGeometry(0.09, 0.78, 0.09);
  const mesh = new THREE.InstancedMesh(geo, mat, MAX);
  const dummy = new THREE.Object3D();
  let count = 0;

  for (let i = 2; i < n - 2 && count < MAX; i += 7) {
    for (const sideSign of [1, -1]) {
      if (count >= MAX) break;
      if (inTunnel(i)) continue;
      const cliffAmt = cliffAmtAt(corridor, i, sideSign);
      if (cliffAmt > 0.45) continue; // wall stretches: no posts
      const hasRail = cliffAmt < 0.35;
      const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
      const out = outwardAt(corridor, i, sideSign);
      dummy.position.copy(edge).addScaledVector(out, hasRail ? 1.55 : 1.15);
      dummy.position.y = edge.y + 0.34;
      dummy.rotation.set(0, roadYawAt(corridor, i), 0);
      dummy.updateMatrix();
      mesh.setMatrixAt(count++, dummy.matrix);
    }
  }
  mesh.count = count;
  mesh.instanceMatrix.needsUpdate = true;
  group.add(mesh);
}

/* ── Chevron boards on hairpins ───────────────────────────────── */

function buildChevrons(corridor, mood, group, inTunnel) {
  const n = corridor.center.length;
  const MAX = 70;
  const panelMat = new THREE.MeshBasicMaterial({ map: chevronTexture() });
  mood.bindColor(panelMat, "color", "chevron", TAG);
  const postMat = new THREE.MeshLambertMaterial({ color: 0x3a4046 });
  mood.bindColor(postMat, "color", "lampPole", TAG);

  const panels = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.08, 0.62, 0.92), panelMat, MAX);
  const posts = new THREE.InstancedMesh(
    new THREE.BoxGeometry(0.09, 0.95, 0.09), postMat, MAX);
  const dummy = new THREE.Object3D();
  let count = 0;

  let sinceLast = 99;
  for (let i = 2; i < n - 2 && count < MAX; i++) {
    sinceLast++;
    if (corridor.curvClass[i] !== 2 || inTunnel(i) || sinceLast < 5) continue;
    sinceLast = 0;
    const sideSign = corridor.curv[i] > 0 ? -1 : 1; // outside of the bend
    if (cliffAmtAt(corridor, i, sideSign) > 0.45) continue;
    const edge = sideSign > 0 ? corridor.left[i] : corridor.right[i];
    const out = outwardAt(corridor, i, sideSign);
    const yaw = roadYawAt(corridor, i);

    dummy.position.copy(edge).addScaledVector(out, 2.05);
    dummy.position.y = edge.y + 1.05;
    dummy.rotation.set(0, yaw, 0);
    dummy.updateMatrix();
    panels.setMatrixAt(count, dummy.matrix);

    dummy.position.y = edge.y + 0.42;
    dummy.updateMatrix();
    posts.setMatrixAt(count, dummy.matrix);
    count++;
  }
  panels.count = posts.count = count;
  panels.instanceMatrix.needsUpdate = true;
  posts.instanceMatrix.needsUpdate = true;
  panels.castShadow = true;
  group.add(panels, posts);
}

/* ── Tunnel ───────────────────────────────────────────────────── */

function buildTunnel(corridor, mood, group, tunnel, wallMat, glowPositions) {
  const { i0, i1, side } = tunnel;
  const innerMat = new THREE.MeshBasicMaterial({
    color: 0x060708,
    side: THREE.DoubleSide,
  });

  const lw = [];
  const lwT = [];
  const rw = [];
  const rwT = [];
  const ceilA = [];
  const ceilB = [];
  const outLs = [];
  const outRs = [];
  for (let i = i0; i <= i1; i++) {
    const outL = outwardAt(corridor, i, 1);
    const outR = outwardAt(corridor, i, -1);
    outLs.push(outL);
    outRs.push(outR);
    const l = corridor.left[i].clone().addScaledVector(outL, 0.9);
    const r = corridor.right[i].clone().addScaledVector(outR, 0.9);
    const lLo = l.clone();
    lLo.y = groundedBase(corridor, l, corridor.left[i].y - 0.2);
    const lHi = l.clone();
    lHi.y = corridor.left[i].y + 4.6;
    const rLo = r.clone();
    rLo.y = groundedBase(corridor, r, corridor.right[i].y - 0.2);
    const rHi = r.clone();
    rHi.y = corridor.right[i].y + 4.6;
    lw.push(lLo);
    lwT.push(lHi);
    rw.push(rLo);
    rwT.push(rHi);
    ceilA.push(lHi);
    ceilB.push(rHi);
  }
  group.add(ribbonUV(lw, lwT, wallMat, 0.45));
  group.add(ribbonUV(rw, rwT, wallMat, 0.45));
  group.add(ribbonUV(ceilA, ceilB, wallMat, 0.45));

  // dark liner so the bore reads black from outside. Offset INWARD (toward
  // the bore) by 0.3 m as well as in Y — a Y-only inset left it coplanar with
  // the masonry wall and z-fought into flickering squares.
  const inset = (pts, tops, outs) => {
    const a = [];
    const b = [];
    for (let k = 0; k < pts.length; k++) {
      const lo = pts[k].clone().addScaledVector(outs[k], -0.3);
      const hi = tops[k].clone().addScaledVector(outs[k], -0.3);
      lo.y += 0.05;
      hi.y -= 0.12;
      a.push(lo);
      b.push(hi);
    }
    return [a, b];
  };
  const [la, lb] = inset(lw, lwT, outLs);
  group.add(ribbonUV(la, lb, innerMat, 1));
  const [ra, rb] = inset(rw, rwT, outRs);
  group.add(ribbonUV(ra, rb, innerMat, 1));

  // facades at both mouths
  for (const i of [i0, i1]) {
    const ctr = corridor.center[i].clone();
    const lat = outwardAt(corridor, i, 1); // points left
    const hwIn = corridor.halfWidth + 1.3;
    const outCliff = 5.5;
    const outValley = 2.6;
    const leftOut = hwIn + (side > 0 ? outCliff : outValley);
    const rightOut = hwIn + (side > 0 ? outValley : outCliff);
    const y0 = ctr.y - 0.2;

    const quad = (x0, x1, yA, yB) => {
      const p = (x, y) => new THREE.Vector3(
        ctr.x + lat.x * x, y, ctr.z + lat.z * x);
      return [
        [p(x0, yA), p(x1, yA)],
        [p(x0, yB), p(x1, yB)],
      ];
    };
    const addQuad = (x0, x1, yA, yB) => {
      const [[a0, a1], [b0, b1]] = quad(x0, x1, yA, yB);
      group.add(ribbonUV([a0, b0], [a1, b1], wallMat, 0.12));
    };
    addQuad(hwIn, leftOut, y0, y0 + 6.8);        // left pillar+band
    addQuad(-rightOut, -hwIn, y0, y0 + 6.8);     // right pillar+band
    addQuad(-hwIn, hwIn, y0 + 4.6, y0 + 6.8);    // lintel over the bore
  }

  // faint amber service lights inside
  const span = i1 - i0;
  for (const t of [0.25, 0.5, 0.75]) {
    const i = i0 + Math.round(span * t);
    const c = corridor.center[i];
    glowPositions.push(c.x, c.y + 4.0, c.z);
  }
}
