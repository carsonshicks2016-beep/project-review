// Road surface + painted lines + shoulders. The asphalt is a generated
// texture (speckle + lane wear baked in); lines are geometry ribbons so they
// stay crisp at any resolution. Mood drives tints only.
import * as THREE from "../vendor/three.module.min.js";

const TAG = "world";

function asphaltTexture() {
  const s = 256;
  const canvas = document.createElement("canvas");
  canvas.width = s;
  canvas.height = s;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#393b3f";
  ctx.fillRect(0, 0, s, s);

  // aggregate speckle
  for (let i = 0; i < 1100; i++) {
    const v = 44 + ((i * 37) % 46);
    ctx.fillStyle = `rgba(${v},${v + 2},${v + 4},0.55)`;
    ctx.fillRect((i * 31) % s, (i * 17) % s, 1 + (i % 2), 1 + ((i >> 1) % 2));
  }
  // darker patches and tar streaks
  ctx.fillStyle = "rgba(18,19,22,0.35)";
  for (let i = 0; i < 22; i++) {
    ctx.fillRect((i * 53) % s, (i * 97) % s, 14 + (i % 4) * 12, 2 + (i % 3));
  }
  // lane wear: two darker vertical bands where tyres run
  for (const u of [0.30, 0.70]) {
    const g = ctx.createLinearGradient((u - 0.10) * s, 0, (u + 0.10) * s, 0);
    g.addColorStop(0, "rgba(0,0,0,0)");
    g.addColorStop(0.5, "rgba(8,9,11,0.42)");
    g.addColorStop(1, "rgba(0,0,0,0)");
    ctx.fillStyle = g;
    ctx.fillRect((u - 0.10) * s, 0, 0.2 * s, s);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

export function ribbon(aPts, bPts, material, lift = 0) {
  const verts = [];
  const indices = [];
  for (let i = 0; i < aPts.length; i++) {
    const a = aPts[i];
    const b = bPts[i];
    verts.push(a.x, a.y + lift, a.z, b.x, b.y + lift, b.z);
  }
  for (let i = 0; i < aPts.length - 1; i++) {
    const a = i * 2;
    indices.push(a, a + 2, a + 1, a + 1, a + 2, a + 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  return new THREE.Mesh(geo, material);
}

function roadStrip(corridor, material) {
  const verts = [];
  const uvs = [];
  const indices = [];
  for (let i = 0; i < corridor.left.length; i++) {
    const l = corridor.left[i];
    const r = corridor.right[i];
    const v = corridor.distances[i] * 0.085; // ~12m per texture tile
    verts.push(l.x, l.y, l.z, r.x, r.y, r.z);
    uvs.push(0, v, 1, v);
  }
  for (let i = 0; i < corridor.left.length - 1; i++) {
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

function lateralRails(corridor, lateral, halfWidth) {
  const a = [];
  const b = [];
  const dir = new THREE.Vector3();
  for (let i = 0; i < corridor.center.length; i++) {
    dir.copy(corridor.left[i]).sub(corridor.center[i]).setY(0).normalize();
    const base = corridor.center[i].clone().addScaledVector(dir, lateral);
    a.push(base.clone().addScaledVector(dir, halfWidth));
    b.push(base.clone().addScaledVector(dir, -halfWidth));
  }
  return [a, b];
}

function edgeRails(corridor, edge, insetNear, insetFar) {
  const a = [];
  const b = [];
  const dir = new THREE.Vector3();
  for (let i = 0; i < edge.length; i++) {
    dir.copy(corridor.center[i]).sub(edge[i]).setY(0).normalize();
    a.push(edge[i].clone().addScaledVector(dir, insetNear));
    b.push(edge[i].clone().addScaledVector(dir, insetFar));
  }
  return [a, b];
}

function shoulderRails(corridor, edge) {
  const a = [];
  const b = [];
  const dir = new THREE.Vector3();
  for (let i = 0; i < edge.length; i++) {
    dir.copy(edge[i]).sub(corridor.center[i]).setY(0).normalize();
    const inner = edge[i].clone();
    inner.y += 0.005;
    const outer = edge[i].clone().addScaledVector(dir, 1.5);
    outer.y -= 0.06;
    a.push(inner);
    b.push(outer);
  }
  return [a, b];
}

export function buildRoad(corridor, mood) {
  const group = new THREE.Group();
  group.name = "road-v2";

  const roadMat = new THREE.MeshLambertMaterial({
    map: asphaltTexture(),
    color: 0xffffff,
    side: THREE.DoubleSide,
  });
  mood.bindColor(roadMat, "color", "road", TAG);
  const road = roadStrip(corridor, roadMat);
  road.receiveShadow = true;
  group.add(road);

  const shoulderMat = new THREE.MeshLambertMaterial({
    color: 0x4a4337,
    side: THREE.DoubleSide,
  });
  mood.bindColor(shoulderMat, "color", "shoulder", TAG);
  for (const edge of [corridor.left, corridor.right]) {
    const [sa, sb] = shoulderRails(corridor, edge);
    const shoulder = ribbon(sa, sb, shoulderMat);
    shoulder.receiveShadow = true;
    group.add(shoulder);
  }

  const laneMat = new THREE.MeshBasicMaterial({
    color: 0xd9b13b,
    side: THREE.DoubleSide,
  });
  mood.bindColor(laneMat, "color", "laneLine", TAG);
  const edgeMat = new THREE.MeshBasicMaterial({
    color: 0xa9ada7,
    side: THREE.DoubleSide,
  });
  mood.bindColor(edgeMat, "color", "edgeLine", TAG);

  const [clA, clB] = lateralRails(corridor, 0, 0.09);
  group.add(ribbon(clA, clB, laneMat, 0.03));
  const [leA, leB] = edgeRails(corridor, corridor.left, 0.30, 0.45);
  group.add(ribbon(leA, leB, edgeMat, 0.03));
  const [reA, reB] = edgeRails(corridor, corridor.right, 0.30, 0.45);
  group.add(ribbon(reA, reB, edgeMat, 0.03));

  return group;
}
