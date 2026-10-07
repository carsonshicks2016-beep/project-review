import * as THREE from "three";
import { simToThree } from "./coords.js";
import {
  makeCurbStripeTexture,
  makePs1Flat,
  makePs1TexturedMaterial,
  makePs1VertexMaterial,
} from "./ps1-material.js";

const CURB_W = 0.55;
const GRASS_W = 7.5;
const ARMCO_EVERY_M = 5.5;
const POST_H = 0.72;
const POST_SIZE = 0.12;
const RAIL_H = 0.38;
// Phase 5 densify — still instanced / PS1-flat.
const TREE_EVERY_M = 11;
const TREE_OFFSET_M = 4.0;
const BILLBOARD_EVERY_M = 95;
const MARSHAL_EVERY_M = 140;

/**
 * Build a low-poly Nordschleife ribbon world from the published track JSON.
 * Presentation only — does not affect sim pose / road_z.
 */
export async function loadTrackWorld(url = "./assets/track/nordschleife_ribbon.json") {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`track asset ${res.status}: ${url}`);
  const data = await res.json();
  if (data.schema !== "watch25d-track-v1") {
    throw new Error(`unexpected track schema: ${data.schema}`);
  }
  return buildTrackWorld(data);
}

export function buildTrackWorld(data) {
  const group = new THREE.Group();
  group.name = "track-world";

  const center = data.center_xyz;
  const halfW = data.half_width_m;
  const n = center.length;
  if (n < 3) throw new Error("track ribbon too short");

  // Drop the duplicated closing sample when present (builder appends index 0).
  let count = n;
  const first = center[0];
  const last = center[n - 1];
  if (
    Math.abs(first[0] - last[0]) < 1e-3
    && Math.abs(first[1] - last[1]) < 1e-3
    && Math.abs(first[2] - last[2]) < 1e-3
  ) {
    count = n - 1;
  }

  const leftEdge = new Array(count);
  const rightEdge = new Array(count);
  const leftCurbOut = new Array(count);
  const rightCurbOut = new Array(count);
  const leftGrass = new Array(count);
  const rightGrass = new Array(count);
  const headings = data.heading_rad;

  const _l = new THREE.Vector3();
  const _r = new THREE.Vector3();

  for (let i = 0; i < count; i += 1) {
    const [sx, sy, sz] = center[i];
    const hw = halfW[i] ?? data.width_m * 0.5;
    const yaw = headings[i];
    // Sim left normal = (-sin, cos) from heading atan2(ty, tx).
    const nx = -Math.sin(yaw);
    const ny = Math.cos(yaw);

    simToThree(sx + nx * hw, sy + ny * hw, sz, _l);
    simToThree(sx - nx * hw, sy - ny * hw, sz, _r);
    leftEdge[i] = _l.clone();
    rightEdge[i] = _r.clone();

    simToThree(sx + nx * (hw + CURB_W), sy + ny * (hw + CURB_W), sz, _l);
    simToThree(sx - nx * (hw + CURB_W), sy - ny * (hw + CURB_W), sz, _r);
    leftCurbOut[i] = _l.clone();
    rightCurbOut[i] = _r.clone();

    simToThree(sx + nx * (hw + CURB_W + GRASS_W), sy + ny * (hw + CURB_W + GRASS_W), sz - 0.15, _l);
    simToThree(sx - nx * (hw + CURB_W + GRASS_W), sy - ny * (hw + CURB_W + GRASS_W), sz - 0.15, _r);
    leftGrass[i] = _l.clone();
    rightGrass[i] = _r.clone();
  }

  const asphaltMat = makePs1VertexMaterial({ snap: true });
  const curbTex = makeCurbStripeTexture();
  curbTex.repeat.set(count * 0.35, 1);
  const curbStripeMat = makePs1TexturedMaterial(curbTex, { snap: true, wobble: 1 });
  const grassMat = makePs1VertexMaterial({ snap: true });
  const postMat = makePs1Flat(0x8a9096);
  const railMat = makePs1Flat(0x6e747a);

  const asphalt = ribbonMesh(leftEdge, rightEdge, {
    closed: true,
    colorFn: (i, side) => asphaltShade(i, side),
  }, asphaltMat);
  asphalt.name = "asphalt";
  group.add(asphalt);

  const leftCurb = ribbonMesh(leftEdge, leftCurbOut, {
    closed: true,
    uAlong: true,
  }, curbStripeMat);
  leftCurb.name = "curb-left";
  group.add(leftCurb);

  const rightCurb = ribbonMesh(rightCurbOut, rightEdge, {
    closed: true,
    uAlong: true,
  }, curbStripeMat);
  rightCurb.name = "curb-right";
  group.add(rightCurb);

  const leftSkirt = ribbonMesh(leftCurbOut, leftGrass, {
    closed: true,
    colorFn: (i, side) => grassShade(i, side),
  }, grassMat);
  leftSkirt.name = "grass-left";
  group.add(leftSkirt);

  const rightSkirt = ribbonMesh(rightGrass, rightCurbOut, {
    closed: true,
    colorFn: (i, side) => grassShade(i, side),
  }, grassMat);
  rightSkirt.name = "grass-right";
  group.add(rightSkirt);

  const armco = buildArmco(leftCurbOut, rightCurbOut, {
    everyM: ARMCO_EVERY_M,
    postMat,
    railMat,
  });
  group.add(armco);

  const trunkMat = makePs1Flat(0x4a3828);
  const canopyMat = makePs1Flat(0x2f5a32);
  const canopyAltMat = makePs1Flat(0x3a6a38);
  const trees = buildRoadsideTrees(leftGrass, rightGrass, {
    everyM: TREE_EVERY_M,
    offsetM: TREE_OFFSET_M,
    trunkMat,
    canopyMat,
    canopyAltMat,
  });
  group.add(trees);

  const boardMatA = makePs1Flat(0xc44a2a);
  const boardMatB = makePs1Flat(0xe8e0c8);
  const boardMatC = makePs1Flat(0x2a4a8a);
  const boardPostMat = makePs1Flat(0x5a5048);
  const billboards = buildBillboards(leftGrass, rightGrass, {
    everyM: BILLBOARD_EVERY_M,
    mats: [boardMatA, boardMatB, boardMatC],
    postMat: boardPostMat,
  });
  group.add(billboards);

  const marshalPoleMat = makePs1Flat(0x9a9ea4);
  const marshalFlagMat = makePs1Flat(0xe8a020);
  const marshals = buildMarshalPosts(leftCurbOut, rightCurbOut, {
    everyM: MARSHAL_EVERY_M,
    poleMat: marshalPoleMat,
    flagMat: marshalFlagMat,
  });
  group.add(marshals);

  const materials = [
    asphaltMat, curbStripeMat, grassMat, postMat, railMat,
    trunkMat, canopyMat, canopyAltMat,
    boardMatA, boardMatB, boardMatC, boardPostMat,
    marshalPoleMat, marshalFlagMat,
  ];

  return {
    group,
    materials,
    meta: {
      id: data.id,
      sampleCount: count,
      widthM: data.width_m,
      lengthM: data.length_m,
    },
  };
}

function asphaltShade(i, side) {
  // Mild lengthwise dither — flat greys, no pace heat.
  const band = ((i >> 1) & 1) === 0 ? 0.0 : 0.03;
  const edge = side === 0 || side === 1 ? 0.02 : 0;
  const g = 0.22 + band + edge;
  return new THREE.Color(g, g + 0.01, g - 0.01);
}

function grassShade(i, side) {
  const dither = ((i + side) & 3) === 0 ? 0.04 : 0;
  return new THREE.Color(0.18 + dither, 0.32 + dither * 0.5, 0.14);
}

/**
 * Build a strip between two parallel polylines (same length).
 * Vertices: for each sample i → left[i], right[i].
 */
function ribbonMesh(left, right, { closed = true, colorFn = null, uAlong = false } = {}, material) {
  const n = left.length;
  const seg = closed ? n : n - 1;
  const positions = new Float32Array(n * 2 * 3);
  const colors = colorFn ? new Float32Array(n * 2 * 3) : null;
  const uvs = uAlong ? new Float32Array(n * 2 * 2) : null;
  const indices = [];

  for (let i = 0; i < n; i += 1) {
    const L = left[i];
    const R = right[i];
    const base = i * 2;
    positions[base * 3 + 0] = L.x;
    positions[base * 3 + 1] = L.y;
    positions[base * 3 + 2] = L.z;
    positions[(base + 1) * 3 + 0] = R.x;
    positions[(base + 1) * 3 + 1] = R.y;
    positions[(base + 1) * 3 + 2] = R.z;

    if (colors) {
      const cL = colorFn(i, 0);
      const cR = colorFn(i, 1);
      colors[base * 3 + 0] = cL.r;
      colors[base * 3 + 1] = cL.g;
      colors[base * 3 + 2] = cL.b;
      colors[(base + 1) * 3 + 0] = cR.r;
      colors[(base + 1) * 3 + 1] = cR.g;
      colors[(base + 1) * 3 + 2] = cR.b;
    }
    if (uvs) {
      const u = i / Math.max(1, n - 1);
      uvs[base * 2 + 0] = u;
      uvs[base * 2 + 1] = 0;
      uvs[(base + 1) * 2 + 0] = u;
      uvs[(base + 1) * 2 + 1] = 1;
    }
  }

  for (let i = 0; i < seg; i += 1) {
    const a = i * 2;
    const b = a + 1;
    const j = ((i + 1) % n) * 2;
    const c = j;
    const d = j + 1;
    indices.push(a, b, d, a, d, c);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  if (colors) geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  if (uvs) geo.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();

  const mesh = new THREE.Mesh(geo, material);
  mesh.frustumCulled = true;
  return mesh;
}

function buildArmco(leftOut, rightOut, { everyM, postMat, railMat }) {
  const root = new THREE.Group();
  root.name = "armco";

  const postGeo = new THREE.BoxGeometry(POST_SIZE, POST_H, POST_SIZE);
  const matrices = [];
  const railPositions = [];
  const railIndices = [];
  let railVert = 0;

  const sides = [
    { edge: leftOut, sign: 1 },
    { edge: rightOut, sign: -1 },
  ];

  for (const { edge } of sides) {
    let traveled = 0;
    let nextAt = 0;
    const posts = [];

    for (let i = 0; i < edge.length; i += 1) {
      if (i > 0) traveled += edge[i].distanceTo(edge[i - 1]);
      if (traveled + 1e-6 < nextAt && i !== 0) continue;
      nextAt += everyM;
      posts.push({ i, p: edge[i] });

      const m = new THREE.Matrix4();
      const q = new THREE.Quaternion();
      const prev = edge[(i - 1 + edge.length) % edge.length];
      const next = edge[(i + 1) % edge.length];
      const dir = new THREE.Vector3().subVectors(next, prev).normalize();
      const yaw = Math.atan2(dir.x, dir.z);
      q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);
      m.compose(
        new THREE.Vector3(edge[i].x, edge[i].y + POST_H * 0.5, edge[i].z),
        q,
        new THREE.Vector3(1, 1, 1),
      );
      matrices.push(m);
    }

    for (let p = 0; p < posts.length; p += 1) {
      const a = posts[p].p;
      const b = posts[(p + 1) % posts.length].p;
      const y0 = RAIL_H * 0.35;
      const y1 = RAIL_H;
      const base = railVert;
      railPositions.push(
        a.x, a.y + y0, a.z,
        a.x, a.y + y1, a.z,
        b.x, b.y + y1, b.z,
        b.x, b.y + y0, b.z,
      );
      railIndices.push(base, base + 1, base + 2, base, base + 2, base + 3);
      railVert += 4;
    }
  }

  if (matrices.length) {
    const posts = new THREE.InstancedMesh(postGeo, postMat, matrices.length);
    matrices.forEach((m, i) => posts.setMatrixAt(i, m));
    posts.instanceMatrix.needsUpdate = true;
    posts.name = "armco-posts";
    root.add(posts);
  }

  if (railPositions.length) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(railPositions, 3));
    geo.setIndex(railIndices);
    geo.computeVertexNormals();
    const rail = new THREE.Mesh(geo, railMat);
    rail.name = "armco-rail";
    root.add(rail);
  }

  return root;
}

/**
 * Cheap PS1 roadside trees: box trunk + cone canopy, instanced.
 */
function buildRoadsideTrees(leftGrass, rightGrass, {
  everyM,
  offsetM,
  trunkMat,
  canopyMat,
  canopyAltMat,
}) {
  const root = new THREE.Group();
  root.name = "roadside-trees";

  const trunkGeo = new THREE.BoxGeometry(0.28, 1.1, 0.28);
  const canopyGeo = new THREE.ConeGeometry(1.15, 2.4, 5);
  const trunkMatrices = [];
  const canopyA = [];
  const canopyB = [];

  const sides = [
    { edge: leftGrass, outward: 1 },
    { edge: rightGrass, outward: -1 },
  ];

  for (const { edge, outward } of sides) {
    let traveled = 0;
    let nextAt = everyM * 0.35;
    let treeIdx = 0;

    for (let i = 0; i < edge.length; i += 1) {
      if (i > 0) traveled += edge[i].distanceTo(edge[i - 1]);
      if (traveled + 1e-6 < nextAt && i !== 0) continue;
      nextAt += everyM + ((treeIdx % 3) - 1) * 2.2;
      treeIdx += 1;

      // Occasional double-row densify further out.
      const rows = (treeIdx % 5) === 0 ? 2 : 1;
      for (let row = 0; row < rows; row += 1) {
        const prev = edge[(i - 1 + edge.length) % edge.length];
        const next = edge[(i + 1) % edge.length];
        const along = new THREE.Vector3().subVectors(next, prev);
        along.y = 0;
        if (along.lengthSq() < 1e-6) continue;
        along.normalize();
        const out = new THREE.Vector3(along.z * outward, 0, -along.x * outward);
        const p = edge[i];
        const jitter = ((treeIdx * 17 + row * 9) % 7) * 0.14;
        const depth = offsetM + jitter + row * 3.4;
        const x = p.x + out.x * depth;
        const z = p.z + out.z * depth;
        const y = p.y;
        const scale = 0.8 + ((treeIdx * 13 + row) % 5) * 0.09;
        const yaw = ((treeIdx * 41 + row * 17) % 360) * (Math.PI / 180);

        const q = new THREE.Quaternion().setFromAxisAngle(
          new THREE.Vector3(0, 1, 0),
          yaw,
        );
        const trunkM = new THREE.Matrix4();
        trunkM.compose(
          new THREE.Vector3(x, y + 0.55 * scale, z),
          q,
          new THREE.Vector3(scale, scale, scale),
        );
        trunkMatrices.push(trunkM);

        const canopyM = new THREE.Matrix4();
        canopyM.compose(
          new THREE.Vector3(x, y + (1.1 + 1.15) * scale, z),
          q,
          new THREE.Vector3(scale, scale, scale),
        );
        if (((treeIdx + row) & 1) === 0) canopyA.push(canopyM);
        else canopyB.push(canopyM);
      }
    }
  }

  if (trunkMatrices.length) {
    const trunks = new THREE.InstancedMesh(trunkGeo, trunkMat, trunkMatrices.length);
    trunkMatrices.forEach((m, i) => trunks.setMatrixAt(i, m));
    trunks.instanceMatrix.needsUpdate = true;
    trunks.name = "tree-trunks";
    trunks.frustumCulled = true;
    root.add(trunks);
  }
  if (canopyA.length) {
    const c = new THREE.InstancedMesh(canopyGeo, canopyMat, canopyA.length);
    canopyA.forEach((m, i) => c.setMatrixAt(i, m));
    c.instanceMatrix.needsUpdate = true;
    c.name = "tree-canopy-a";
    c.frustumCulled = true;
    root.add(c);
  }
  if (canopyB.length) {
    const c = new THREE.InstancedMesh(canopyGeo, canopyAltMat, canopyB.length);
    canopyB.forEach((m, i) => c.setMatrixAt(i, m));
    c.instanceMatrix.needsUpdate = true;
    c.name = "tree-canopy-b";
    c.frustumCulled = true;
    root.add(c);
  }

  return root;
}

/**
 * Occasional flat billboard ads — PS1 quads, instanced, face track.
 */
function buildBillboards(leftGrass, rightGrass, { everyM, mats, postMat }) {
  const root = new THREE.Group();
  root.name = "billboards";

  const boardGeo = new THREE.PlaneGeometry(3.2, 1.6);
  const postGeo = new THREE.BoxGeometry(0.12, 2.2, 0.12);
  const buckets = mats.map(() => []);
  const postMatrices = [];

  const sides = [
    { edge: leftGrass, outward: 1 },
    { edge: rightGrass, outward: -1 },
  ];

  let idx = 0;
  for (const { edge, outward } of sides) {
    let traveled = 0;
    let nextAt = everyM * (0.4 + (outward > 0 ? 0 : 0.35));

    for (let i = 0; i < edge.length; i += 1) {
      if (i > 0) traveled += edge[i].distanceTo(edge[i - 1]);
      if (traveled + 1e-6 < nextAt && i !== 0) continue;
      nextAt += everyM + ((idx % 4) - 1.5) * 12;
      idx += 1;

      const prev = edge[(i - 1 + edge.length) % edge.length];
      const next = edge[(i + 1) % edge.length];
      const along = new THREE.Vector3().subVectors(next, prev);
      along.y = 0;
      if (along.lengthSq() < 1e-6) continue;
      along.normalize();
      const out = new THREE.Vector3(along.z * outward, 0, -along.x * outward);
      const p = edge[i];
      const depth = 6.5 + ((idx * 7) % 5) * 0.4;
      const x = p.x + out.x * depth;
      const z = p.z + out.z * depth;
      const y = p.y;
      // Face toward track (inward).
      const yaw = Math.atan2(-out.x, -out.z);
      const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);

      const boardM = new THREE.Matrix4();
      boardM.compose(
        new THREE.Vector3(x, y + 1.85, z),
        q,
        new THREE.Vector3(1, 1, 1),
      );
      buckets[idx % mats.length].push(boardM);

      const postM = new THREE.Matrix4();
      postM.compose(
        new THREE.Vector3(x, y + 1.1, z),
        q,
        new THREE.Vector3(1, 1, 1),
      );
      postMatrices.push(postM);
    }
  }

  if (postMatrices.length) {
    const posts = new THREE.InstancedMesh(postGeo, postMat, postMatrices.length);
    postMatrices.forEach((m, i) => posts.setMatrixAt(i, m));
    posts.instanceMatrix.needsUpdate = true;
    posts.name = "billboard-posts";
    root.add(posts);
  }

  buckets.forEach((list, bi) => {
    if (!list.length) return;
    const mesh = new THREE.InstancedMesh(boardGeo, mats[bi], list.length);
    list.forEach((m, i) => mesh.setMatrixAt(i, m));
    mesh.instanceMatrix.needsUpdate = true;
    mesh.name = `billboard-face-${bi}`;
    root.add(mesh);
  });

  return root;
}

/**
 * Occasional marshal posts with a flat orange flag — PS1 boxes.
 */
function buildMarshalPosts(leftOut, rightOut, { everyM, poleMat, flagMat }) {
  const root = new THREE.Group();
  root.name = "marshals";

  const poleGeo = new THREE.BoxGeometry(0.08, 2.4, 0.08);
  const flagGeo = new THREE.PlaneGeometry(0.7, 0.45);
  const poleMatrices = [];
  const flagMatrices = [];

  const sides = [
    { edge: leftOut, outward: 1 },
    { edge: rightOut, outward: -1 },
  ];

  let idx = 0;
  for (const { edge, outward } of sides) {
    let traveled = 0;
    let nextAt = everyM * (0.55 + (outward > 0 ? 0 : 0.2));

    for (let i = 0; i < edge.length; i += 1) {
      if (i > 0) traveled += edge[i].distanceTo(edge[i - 1]);
      if (traveled + 1e-6 < nextAt && i !== 0) continue;
      nextAt += everyM + ((idx % 3) - 1) * 18;
      idx += 1;

      const prev = edge[(i - 1 + edge.length) % edge.length];
      const next = edge[(i + 1) % edge.length];
      const along = new THREE.Vector3().subVectors(next, prev);
      along.y = 0;
      if (along.lengthSq() < 1e-6) continue;
      along.normalize();
      const out = new THREE.Vector3(along.z * outward, 0, -along.x * outward);
      const p = edge[i];
      const x = p.x + out.x * 1.2;
      const z = p.z + out.z * 1.2;
      const y = p.y;
      const yaw = Math.atan2(along.x, along.z);
      const q = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), yaw);

      const poleM = new THREE.Matrix4();
      poleM.compose(
        new THREE.Vector3(x, y + 1.2, z),
        q,
        new THREE.Vector3(1, 1, 1),
      );
      poleMatrices.push(poleM);

      const flagM = new THREE.Matrix4();
      flagM.compose(
        new THREE.Vector3(x + along.x * 0.35, y + 2.15, z + along.z * 0.35),
        q,
        new THREE.Vector3(1, 1, 1),
      );
      flagMatrices.push(flagM);
    }
  }

  if (poleMatrices.length) {
    const poles = new THREE.InstancedMesh(poleGeo, poleMat, poleMatrices.length);
    poleMatrices.forEach((m, i) => poles.setMatrixAt(i, m));
    poles.instanceMatrix.needsUpdate = true;
    poles.name = "marshal-poles";
    root.add(poles);
  }
  if (flagMatrices.length) {
    const flags = new THREE.InstancedMesh(flagGeo, flagMat, flagMatrices.length);
    flagMatrices.forEach((m, i) => flags.setMatrixAt(i, m));
    flags.instanceMatrix.needsUpdate = true;
    flags.name = "marshal-flags";
    root.add(flags);
  }

  return root;
}
