import * as THREE from "./vendor/three.module.min.js";

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

function smoothstep(edge0, edge1, x) {
  const t = clamp01((x - edge0) / Math.max(1e-6, edge1 - edge0));
  return t * t * (3 - 2 * t);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function hash01(seed) {
  const x = Math.sin(seed * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

function trackDistances(points) {
  const distances = [0];
  for (let i = 1; i < points.length; i++) {
    distances.push(distances[i - 1] + points[i].distanceTo(points[i - 1]));
  }
  return distances;
}

function distanceIndex(distances, d) {
  if (d <= 0) return { index: 0, t: 0 };
  const last = distances.length - 1;
  if (d >= distances[last]) return { index: Math.max(0, last - 1), t: 1 };
  let lo = 0;
  let hi = last;
  while (hi - lo > 1) {
    const mid = Math.floor((lo + hi) / 2);
    if (distances[mid] < d) lo = mid;
    else hi = mid;
  }
  const span = Math.max(1e-6, distances[hi] - distances[lo]);
  return { index: lo, t: (d - distances[lo]) / span };
}

function trackSampleAtDistance(profile, d) {
  const { center, left, right, distances } = profile;
  const { index, t } = distanceIndex(distances, d);
  const next = Math.min(center.length - 1, index + 1);
  const c = center[index].clone().lerp(center[next], t);
  const l = left[index].clone().lerp(left[next], t);
  const r = right[index].clone().lerp(right[next], t);
  const tangent = center[next].clone().sub(center[index]).setY(0).normalize();
  return {
    center: c,
    left: l,
    right: r,
    yaw: Math.atan2(-tangent.z, tangent.x),
    index,
    t,
  };
}

function createTrackProfile(center, left, right) {
  let widthSum = 0;
  for (let i = 0; i < center.length; i++) {
    widthSum += center[i].distanceTo(left[i]) + center[i].distanceTo(right[i]);
  }
  const distances = trackDistances(center);
  return {
    center,
    left,
    right,
    distances,
    length: distances[distances.length - 1] ?? 0,
    halfWidth: widthSum / Math.max(1, center.length * 2),
  };
}

export function sampleRoadAt(profile, x, z) {
  if (!profile || profile.center.length < 2) return null;
  const pts = profile.center;
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
      bestY = lerp(a.y, b.y, t);
      bestIx = i;
      bestT = t;
    }
  }

  return {
    y: bestY,
    distance: Math.sqrt(bestDist),
    index: bestIx,
    t: bestT,
  };
}

export function roadHeightAt(profile, x, z, fallbackY = 0) {
  const sample = sampleRoadAt(profile, x, z);
  return sample ? sample.y : fallbackY;
}

export function roadPitchAt(profile, x, z, yaw) {
  const forward = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
  const halfProbe = 1.55;
  const frontY = roadHeightAt(profile, x + forward.x * halfProbe, z + forward.z * halfProbe);
  const rearY = roadHeightAt(profile, x - forward.x * halfProbe, z - forward.z * halfProbe);
  return Math.atan2(frontY - rearY, halfProbe * 2);
}

function terrainHeightAt(profile, px, pz) {
  if (!profile) {
    return -2.3 + Math.sin(px * 0.012) * 1.2 + Math.cos(pz * 0.010) * 0.9;
  }
  const sample = sampleRoadAt(profile, px, pz);
  const roadY = sample ? sample.y : 0;
  const dist = sample ? sample.distance : 9999;
  const shoulderEdge = profile.halfWidth + 3.0;
  const blendEnd = profile.halfWidth + 95.0;
  const nearRoad = roadY + 0.012;
  const mountainNoise = (
    Math.sin(px * 0.011 + pz * 0.004) * 1.15 +
    Math.cos(pz * 0.009 - px * 0.003) * 0.82 +
    Math.sin((px + pz) * 0.005) * 0.55
  );
  const rise = Math.min(dist * 0.030, 8.5);
  const regional = roadY - 1.35 + mountainNoise + rise;
  const t = smoothstep(shoulderEdge, blendEnd, dist);
  return lerp(nearRoad, regional, t);
}

function trackYawAt(points, i) {
  const a = points[Math.max(0, i - 1)];
  const b = points[Math.min(points.length - 1, i + 1)];
  const tangent = b.clone().sub(a).setY(0).normalize();
  return Math.atan2(-tangent.z, tangent.x);
}

function angleDelta(a, b) {
  let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
  if (d < -Math.PI) d += Math.PI * 2;
  return d;
}

function curvatureAt(profile, i, span = 5) {
  const a = trackYawAt(profile.center, Math.max(1, i - span));
  const b = trackYawAt(profile.center, Math.min(profile.center.length - 2, i + span));
  return angleDelta(a, b);
}

function roadCamberAt(profile, i, lateral) {
  const curve = Math.max(-0.42, Math.min(0.42, curvatureAt(profile, i)));
  return Math.max(-0.18, Math.min(0.18, -curve * lateral * 0.38));
}

function visualRoadPoint(profile, p, i, lateral, lift = 0) {
  return p.clone().add(new THREE.Vector3(0, lift + roadCamberAt(profile, i, lateral), 0));
}

function outwardAt(center, edge, i) {
  return edge[i].clone().sub(center[i]).setY(0).normalize();
}

function setInstancedTransform(mesh, index, pos, yaw, scale = null) {
  const dummy = setInstancedTransform._dummy || (setInstancedTransform._dummy = new THREE.Object3D());
  dummy.position.copy(pos);
  dummy.rotation.set(0, yaw, 0);
  if (scale) dummy.scale.copy(scale);
  else dummy.scale.set(1, 1, 1);
  dummy.updateMatrix();
  mesh.setMatrixAt(index, dummy.matrix);
}

function jitterGeometry(geo, amount, seed = 1) {
  const pos = geo.getAttribute("position");
  for (let i = 0; i < pos.count; i++) {
    const x = pos.getX(i) + (hash01(seed + i * 3.13) - 0.5) * amount;
    const y = pos.getY(i) + (hash01(seed + i * 5.29) - 0.5) * amount;
    const z = pos.getZ(i) + (hash01(seed + i * 7.71) - 0.5) * amount;
    pos.setXYZ(i, x, y, z);
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();
  return geo;
}

export function buildMountainPassTrack({ roadGroup, roadsideGroup, center, left, right, mats, surfaceLift }) {
  const profile = createTrackProfile(center, left, right);
  buildRoad(roadGroup, profile, mats, surfaceLift);
  buildShoulders(roadGroup, profile, mats);
  buildWetRoadDetails(roadGroup, profile, mats, surfaceLift);
  buildRoadEdgeShadows(roadGroup, profile, mats, surfaceLift);
  addLine(roadGroup, left.map((p, i) => visualRoadPoint(profile, p, i, 1, surfaceLift)), mats.edgeLine, 0.13);
  addLine(roadGroup, right.map((p, i) => visualRoadPoint(profile, p, i, -1, surfaceLift)), mats.edgeLine, 0.13);
  addDashedCenter(roadGroup, profile, mats, surfaceLift);
  buildTerrain(roadsideGroup, profile, mats);
  const fogPlanes = buildRoadside(roadsideGroup, profile, mats, surfaceLift);
  return { profile, fogPlanes };
}

function buildRoad(group, profile, mats, surfaceLift) {
  const verts = [];
  const uvs = [];
  const indices = [];
  for (let i = 0; i < profile.left.length; i++) {
    const v = profile.distances[i] * 0.052;
    const leftWarp = (hash01(i * 3.71) - 0.5) * 0.009;
    const rightWarp = (hash01(i * 4.97) - 0.5) * 0.009;
    const vWarp = (hash01(i * 6.43) - 0.5) * 0.012;
    const left = visualRoadPoint(profile, profile.left[i], i, 1, surfaceLift);
    const right = visualRoadPoint(profile, profile.right[i], i, -1, surfaceLift);
    verts.push(left.x, left.y, left.z);
    verts.push(right.x, right.y, right.z);
    uvs.push(leftWarp, v + vWarp, 1 + rightWarp, v - vWarp);
  }
  for (let i = 0; i < profile.left.length - 1; i++) {
    const a = i * 2, b = a + 1, c = a + 2, d = a + 3;
    indices.push(a, c, b, b, c, d);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const road = new THREE.Mesh(geo, mats.road);
  road.renderOrder = 3;
  road.receiveShadow = true;
  group.add(road);
}

function buildShoulders(group, profile, mats) {
  const leftOuter = [];
  const rightOuter = [];
  for (let i = 0; i < profile.center.length; i++) {
    const ldir = profile.left[i].clone().sub(profile.center[i]).setY(0).normalize();
    const rdir = profile.right[i].clone().sub(profile.center[i]).setY(0).normalize();
    leftOuter.push(visualRoadPoint(profile, profile.left[i].clone().add(ldir.multiplyScalar(2.4)), i, 1.35, -0.04));
    rightOuter.push(visualRoadPoint(profile, profile.right[i].clone().add(rdir.multiplyScalar(2.4)), i, -1.35, -0.04));
  }
  group.add(makeRibbon(leftOuter, profile.left.map((p, i) => visualRoadPoint(profile, p, i, 1, -0.02)), mats.shoulder, 2));
  group.add(makeRibbon(profile.right.map((p, i) => visualRoadPoint(profile, p, i, -1, -0.02)), rightOuter, mats.shoulder, 2));
}

function makeRibbon(aPts, bPts, material, order = 1) {
  const verts = [];
  const uvs = [];
  const indices = [];
  const mids = aPts.map((p, i) => p.clone().add(bPts[i]).multiplyScalar(0.5));
  const distances = trackDistances(mids);
  for (let i = 0; i < aPts.length; i++) {
    const a = aPts[i].clone().add(new THREE.Vector3(0, 0.12, 0));
    const b = bPts[i].clone().add(new THREE.Vector3(0, 0.12, 0));
    const v = distances[i] * 0.050;
    const warp = (hash01(i * 9.17) - 0.5) * 0.010;
    verts.push(a.x, a.y, a.z, b.x, b.y, b.z);
    uvs.push(warp, v + warp, 1 - warp, v - warp);
  }
  for (let i = 0; i < aPts.length - 1; i++) {
    const a = i * 2, b = a + 1, c = a + 2, d = a + 3;
    indices.push(a, c, b, b, c, d);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const mesh = new THREE.Mesh(geo, material);
  mesh.receiveShadow = true;
  mesh.renderOrder = order;
  return mesh;
}

function addLine(group, points, material, lift) {
  const geo = new THREE.BufferGeometry().setFromPoints(points.map(p => p.clone().add(new THREE.Vector3(0, lift, 0))));
  const line = new THREE.Line(geo, material);
  line.renderOrder = 5;
  group.add(line);
}

function addDashedCenter(group, profile, mats, surfaceLift) {
  const dashPts = [];
  for (let i = 0; i < profile.center.length - 1; i += 5) {
    const j = Math.min(i + 3, profile.center.length - 1);
    const a = visualRoadPoint(profile, profile.center[i], i, 0, surfaceLift + 0.16);
    const b = visualRoadPoint(profile, profile.center[j], j, 0, surfaceLift + 0.16);
    dashPts.push(a, b);
  }
  const dashes = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(dashPts), mats.centerLine);
  dashes.renderOrder = 6;
  group.add(dashes);
}

function buildWetRoadDetails(group, profile, mats, surfaceLift) {
  const patchCount = Math.min(76, Math.max(18, profile.center.length));
  const sheenCount = Math.min(110, Math.max(24, Math.floor(profile.center.length * 1.2)));
  const shoulderCount = Math.min(54, Math.max(16, Math.floor(profile.center.length * 0.65)));
  const patchGeo = new THREE.BoxGeometry(2.0, 0.018, 0.64);
  const sheenGeo = new THREE.BoxGeometry(3.8, 0.020, 0.050);
  const shoulderGeo = new THREE.BoxGeometry(1.35, 0.020, 0.075);
  const patches = new THREE.InstancedMesh(patchGeo, mats.roadPatch, patchCount);
  const sheens = new THREE.InstancedMesh(sheenGeo, mats.roadSheen, sheenCount);
  const shoulderCracks = new THREE.InstancedMesh(shoulderGeo, mats.roadPatch, shoulderCount);
  let pN = 0, sN = 0, cN = 0;
  for (let i = 2; i < profile.center.length - 2 && (pN < patchCount || sN < sheenCount || cN < shoulderCount); i += 2) {
    const side = profile.left[i].clone().sub(profile.center[i]).setY(0).normalize();
    const laneHalf = Math.min(profile.center[i].distanceTo(profile.left[i]), profile.center[i].distanceTo(profile.right[i])) * 0.72;
    const yaw = trackYawAt(profile.center, i);
    if (pN < patchCount) {
      const lateral = (hash01(i * 5.17) * 2 - 1) * laneHalf;
      const pos = profile.center[i].clone().add(side.clone().multiplyScalar(lateral));
      pos.y = roadHeightAt(profile, pos.x, pos.z, profile.center[i].y) + surfaceLift + 0.044;
      const scale = new THREE.Vector3(0.46 + hash01(i * 11.3) * 1.05, 1, 0.42 + hash01(i * 7.9) * 1.05);
      setInstancedTransform(patches, pN, pos, yaw + (hash01(i * 2.1) - 0.5) * 0.14, scale);
      pN++;
    }
    if (sN < sheenCount) {
      const lateral = (hash01(i * 13.47) * 2 - 1) * laneHalf * 0.72;
      const pos = profile.center[i].clone().add(side.clone().multiplyScalar(lateral));
      pos.y = roadHeightAt(profile, pos.x, pos.z, profile.center[i].y) + surfaceLift + 0.052;
      const scale = new THREE.Vector3(0.60 + hash01(i * 3.7) * 1.65, 1, 0.50 + hash01(i * 5.5) * 1.35);
      setInstancedTransform(sheens, sN, pos, yaw + (hash01(i * 17.0) - 0.5) * 0.08, scale);
      sN++;
    }
    if (cN < shoulderCount && i % 4 === 0) {
      const edge = hash01(i * 4.4) > 0.5 ? profile.left[i] : profile.right[i];
      const outward = edge.clone().sub(profile.center[i]).setY(0).normalize();
      const pos = edge.clone().add(outward.multiplyScalar(0.86));
      pos.y = roadHeightAt(profile, pos.x, pos.z, edge.y) + surfaceLift + 0.055;
      const scale = new THREE.Vector3(0.55 + hash01(i * 8.8), 1, 0.9 + hash01(i * 9.9));
      setInstancedTransform(shoulderCracks, cN, pos, yaw + Math.PI / 2 + (hash01(i * 10.1) - 0.5) * 0.35, scale);
      cN++;
    }
  }
  patches.count = pN;
  sheens.count = sN;
  shoulderCracks.count = cN;
  patches.renderOrder = sheens.renderOrder = shoulderCracks.renderOrder = 4;
  group.add(patches, sheens, shoulderCracks);
}

function buildRoadEdgeShadows(group, profile, mats, surfaceLift) {
  if (!mats.roadEdgeShadow) return;
  const rightEdge = [];
  const rightInner = [];
  const leftEdge = [];
  const leftInner = [];
  for (let i = 0; i < profile.center.length; i++) {
    rightEdge.push(visualRoadPoint(profile, profile.right[i], i, -1, surfaceLift + 0.010));
    rightInner.push(visualRoadPoint(profile, profile.right[i].clone().lerp(profile.center[i], 0.36), i, -0.55, surfaceLift + 0.010));
    leftEdge.push(visualRoadPoint(profile, profile.left[i], i, 1, surfaceLift + 0.010));
    leftInner.push(visualRoadPoint(profile, profile.left[i].clone().lerp(profile.center[i], 0.24), i, 0.65, surfaceLift + 0.010));
  }
  const cliffShadow = makeRibbon(rightEdge, rightInner, mats.roadEdgeShadow, 4);
  const valleyShadow = makeRibbon(leftEdge, leftInner, mats.roadEdgeShadow, 4);
  cliffShadow.renderOrder = valleyShadow.renderOrder = 4;
  group.add(cliffShadow, valleyShadow);
}

function buildTerrain(group, profile, mats) {
  const points = profile.center.concat(profile.left, profile.right);
  const box = new THREE.Box3().setFromPoints(points);
  const terrainCenter = box.getCenter(new THREE.Vector3());
  const size = Math.max(box.max.x - box.min.x, box.max.z - box.min.z) + 480;
  const seg = 92;
  const verts = [];
  const uvs = [];
  const indices = [];
  for (let z = 0; z <= seg; z++) {
    for (let x = 0; x <= seg; x++) {
      const px = terrainCenter.x + (x / seg - 0.5) * size;
      const pz = terrainCenter.z + (z / seg - 0.5) * size;
      const sample = sampleRoadAt(profile, px, pz);
      const dist = sample ? sample.distance : 9999;
      const jitterGate = smoothstep(profile.halfWidth + 8, profile.halfWidth + 90, dist);
      const py = terrainHeightAt(profile, px, pz) + (hash01(px * 0.117 + pz * 0.073) - 0.5) * 0.42 * jitterGate;
      const uvJitter = (hash01(x * 17.1 + z * 23.7) - 0.5) * 0.018;
      verts.push(px, py, pz);
      uvs.push(x / 5.8 + uvJitter, z / 5.8 - uvJitter);
    }
  }
  for (let z = 0; z < seg; z++) {
    for (let x = 0; x < seg; x++) {
      const a = z * (seg + 1) + x;
      indices.push(a, a + seg + 1, a + 1, a + 1, a + seg + 1, a + seg + 2);
    }
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
  geo.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geo.setIndex(indices);
  geo.computeVertexNormals();
  const terrain = new THREE.Mesh(geo, mats.terrain);
  terrain.receiveShadow = true;
  group.add(terrain);
}

function buildHeroPassCorridor(group, profile, mats, surfaceLift) {
  buildValleyDropoff(group, profile, mats, surfaceLift);
  buildValleyDepthAccents(group, profile, mats, surfaceLift);
  buildValleyMistRibbons(group, profile, mats, surfaceLift);
  buildCliffFootWall(group, profile, mats, surfaceLift);
  buildCliffWall(group, profile, mats, surfaceLift);
  buildCliffButtresses(group, profile, mats, surfaceLift);
  buildCliffStrata(group, profile, mats, surfaceLift);
  buildCliffCrevices(group, profile, mats, surfaceLift);
  buildCliffLampWashes(group, profile, mats, surfaceLift);
  buildValleyPineBands(group, profile, mats, surfaceLift);
}

function buildValleyDepthAccents(group, profile, mats, surfaceLift) {
  if (!mats.valleyDepthAccent) return;
  const rimNear = [];
  const rimFar = [];
  const dropNear = [];
  const dropFar = [];

  for (let i = 0; i < profile.center.length; i++) {
    const d = profile.distances[i] || 0;
    const edge = profile.left[i];
    const outward = edge.clone().sub(profile.center[i]).setY(0).normalize();
    const roadY = roadHeightAt(profile, edge.x, edge.z, edge.y) + surfaceLift;
    const wave = Math.sin(d * 0.026) * 0.45 + Math.cos(d * 0.018) * 0.35;
    const rough = hash01(i * 4.23 + d * 0.016);

    const a = edge.clone().add(outward.clone().multiplyScalar(4.6 + rough * 1.2));
    a.y = roadY - 0.74 + wave * 0.20;
    const b = edge.clone().add(outward.clone().multiplyScalar(12.0 + rough * 3.5));
    b.y = roadY - 2.2 - rough * 0.95 + wave * 0.30;
    rimNear.push(a);
    rimFar.push(b);

    const c = edge.clone().add(outward.clone().multiplyScalar(18.0 + rough * 5.5));
    c.y = roadY - 4.2 - rough * 1.8 + wave * 0.40;
    const e = edge.clone().add(outward.clone().multiplyScalar(38.0 + rough * 11.0));
    e.y = roadY - 7.6 - rough * 2.7 + wave * 0.55;
    dropNear.push(c);
    dropFar.push(e);
  }

  const rim = makeRibbon(rimNear, rimFar, mats.valleyDepthAccent, 0);
  const drop = makeRibbon(dropNear, dropFar, mats.valleyDepthAccent, 0);
  rim.renderOrder = drop.renderOrder = 0;
  group.add(rim, drop);
}

function buildValleyMistRibbons(group, profile, mats, surfaceLift) {
  if (!mats.valleyMistRibbon) return;
  const near = [];
  const far = [];
  const lowerNear = [];
  const lowerFar = [];

  for (let i = 0; i < profile.center.length; i++) {
    const d = profile.distances[i] || 0;
    const edge = profile.left[i];
    const outward = edge.clone().sub(profile.center[i]).setY(0).normalize();
    const roadY = roadHeightAt(profile, edge.x, edge.z, edge.y) + surfaceLift;
    const wave = Math.sin(d * 0.035) * 0.34 + Math.cos(d * 0.017) * 0.42;
    const drift = hash01(i * 3.91 + d * 0.021) * 2.8;

    const a = edge.clone().add(outward.clone().multiplyScalar(13.0 + drift));
    a.y = roadY - 2.2 + wave;
    const b = edge.clone().add(outward.clone().multiplyScalar(35.0 + drift * 2.2));
    b.y = roadY - 3.4 + wave * 0.65;
    near.push(a);
    far.push(b);

    const c = edge.clone().add(outward.clone().multiplyScalar(28.0 + drift));
    c.y = roadY - 6.0 + wave * 0.45;
    const e = edge.clone().add(outward.clone().multiplyScalar(62.0 + drift * 2.5));
    e.y = roadY - 7.4 + wave * 0.32;
    lowerNear.push(c);
    lowerFar.push(e);
  }

  const highMist = makeRibbon(near, far, mats.valleyMistRibbon, -1);
  const lowMist = makeRibbon(lowerNear, lowerFar, mats.valleyMistRibbon, -1);
  highMist.renderOrder = lowMist.renderOrder = -1;
  group.add(highMist, lowMist);
}

function buildCliffFootWall(group, profile, mats, surfaceLift) {
  const toe = [];
  const lip = [];
  const cap = [];

  for (let i = 0; i < profile.center.length; i++) {
    const d = profile.distances[i] || 0;
    const edge = profile.right[i];
    const outward = edge.clone().sub(profile.center[i]).setY(0).normalize();
    const roadY = roadHeightAt(profile, edge.x, edge.z, edge.y) + surfaceLift;
    const rough = hash01(i * 8.41 + d * 0.037);

    const toeP = edge.clone().add(outward.clone().multiplyScalar(1.65 + rough * 0.30));
    toeP.y = roadY - 0.05;

    const lipP = edge.clone().add(outward.clone().multiplyScalar(2.65 + rough * 0.55));
    lipP.y = roadY + 1.02 + rough * 0.36;

    const capP = edge.clone().add(outward.clone().multiplyScalar(3.45 + rough * 0.70));
    capP.y = roadY + 1.28 + rough * 0.42;

    toe.push(toeP);
    lip.push(lipP);
    cap.push(capP);
  }

  const face = makeRibbon(toe, lip, mats.cliffFace, 1);
  const top = makeRibbon(lip, cap, mats.cliffTop, 1);
  for (const mesh of [face, top]) {
    mesh.castShadow = true;
    mesh.receiveShadow = true;
  }
  group.add(face, top);
}

function buildCliffWall(group, profile, mats, surfaceLift) {
  const foot = [];
  const mid = [];
  const upper = [];
  const cap = [];

  for (let i = 0; i < profile.center.length; i++) {
    const d = profile.distances[i] || 0;
    const edge = profile.right[i];
    const outward = edge.clone().sub(profile.center[i]).setY(0).normalize();
    const roadY = roadHeightAt(profile, edge.x, edge.z, edge.y) + surfaceLift;
    const roughA = hash01(i * 4.91 + d * 0.031);
    const roughB = hash01(i * 7.33 + d * 0.017);
    const wave = Math.sin(d * 0.045) * 0.9 + Math.cos(d * 0.021) * 0.7;

    const footP = edge.clone().add(outward.clone().multiplyScalar(2.9 + roughA * 0.85));
    footP.y = roadY - 0.34;

    const midP = edge.clone().add(outward.clone().multiplyScalar(7.0 + roughB * 2.5));
    midP.y = roadY + 2.1 + roughA * 2.3 + wave * 0.42;

    const upperP = edge.clone().add(outward.clone().multiplyScalar(13.0 + roughA * 4.4));
    upperP.y = roadY + 7.6 + roughB * 6.4 + wave;

    const capP = edge.clone().add(outward.clone().multiplyScalar(22.0 + roughB * 7.5));
    capP.y = upperP.y + 0.9 + roughA * 2.2;

    foot.push(footP);
    mid.push(midP);
    upper.push(upperP);
    cap.push(capP);
  }

  const lower = makeRibbon(foot, mid, mats.cliffFace, 0);
  const high = makeRibbon(mid, upper, mats.cliffDark, 0);
  const top = makeRibbon(upper, cap, mats.cliffTop, 0);
  for (const mesh of [lower, high, top]) {
    mesh.castShadow = true;
    mesh.receiveShadow = true;
  }
  group.add(lower, high, top);
}

function buildCliffButtresses(group, profile, mats, surfaceLift) {
  const count = Math.min(150, Math.max(22, Math.floor(profile.length / 5.8)));
  const ledgeCount = Math.min(120, Math.max(18, Math.floor(profile.length / 7.0)));
  const chunkGeo = jitterGeometry(new THREE.BoxGeometry(1, 1, 1), 0.18, 1240);
  const ledgeGeo = jitterGeometry(new THREE.BoxGeometry(1, 1, 1), 0.10, 1241);
  const chunks = new THREE.InstancedMesh(chunkGeo, mats.cliffLedge || mats.propRock || mats.rock, count);
  const ledges = new THREE.InstancedMesh(ledgeGeo, mats.cliffLedge || mats.cliffFace, ledgeCount);
  chunks.castShadow = chunks.receiveShadow = true;
  ledges.castShadow = ledges.receiveShadow = true;

  let n = 0;
  let lN = 0;
  for (let d = 8; d < profile.length - 8 && (n < count || lN < ledgeCount); d += 5.8) {
    const sample = trackSampleAtDistance(profile, d);
    const outward = sample.right.clone().sub(sample.center).setY(0).normalize();
    const roadY = roadHeightAt(profile, sample.right.x, sample.right.z, sample.right.y) + surfaceLift;
    const rA = hash01(d * 0.91);
    const rB = hash01(d * 1.77);
    const rC = hash01(d * 2.63);
    const yaw = sample.yaw + (rB - 0.5) * 0.32;

    if (n < count) {
      const height = 1.2 + rA * 3.4;
      const depth = 1.4 + rB * 3.0;
      const length = 2.8 + rC * 4.8;
      const p = sample.right.clone().add(outward.clone().multiplyScalar(2.35 + depth * 0.56 + rA * 1.3));
      p.y = roadY + height * 0.50 - 0.18 + rB * 0.72;
      setInstancedTransform(chunks, n, p, yaw, new THREE.Vector3(length, height, depth));
      n++;
    }

    if (lN < ledgeCount && rC > 0.22) {
      const shelfHeight = 0.12 + rA * 0.10;
      const shelfLength = 3.2 + rB * 4.2;
      const shelfDepth = 1.7 + rC * 2.2;
      const p = sample.right.clone().add(outward.clone().multiplyScalar(5.2 + rA * 5.0));
      p.y = roadY + 1.45 + rB * 4.5;
      setInstancedTransform(ledges, lN, p, sample.yaw + (rA - 0.5) * 0.48, new THREE.Vector3(shelfLength, shelfHeight, shelfDepth));
      lN++;
    }
  }

  chunks.count = n;
  ledges.count = lN;
  group.add(chunks, ledges);
}

function buildValleyDropoff(group, profile, mats, surfaceLift) {
  const lip = [];
  const bench = [];
  const lower = [];
  const far = [];

  for (let i = 0; i < profile.center.length; i++) {
    const d = profile.distances[i] || 0;
    const edge = profile.left[i];
    const outward = edge.clone().sub(profile.center[i]).setY(0).normalize();
    const roadY = roadHeightAt(profile, edge.x, edge.z, edge.y) + surfaceLift;
    const roughA = hash01(i * 5.37 + d * 0.023);
    const roughB = hash01(i * 9.19 + d * 0.019);
    const valleyWave = Math.sin(d * 0.032) * 0.7 + Math.cos(d * 0.014) * 0.9;

    const lipP = edge.clone().add(outward.clone().multiplyScalar(2.7 + roughA * 0.8));
    lipP.y = roadY - 0.18;

    const benchP = edge.clone().add(outward.clone().multiplyScalar(8.5 + roughB * 3.5));
    benchP.y = roadY - 1.45 - roughA * 1.4 + valleyWave * 0.25;

    const lowerP = edge.clone().add(outward.clone().multiplyScalar(23.0 + roughA * 11.0));
    lowerP.y = roadY - 5.3 - roughB * 4.5 + valleyWave * 0.5;

    const farP = edge.clone().add(outward.clone().multiplyScalar(48.0 + roughB * 20.0));
    farP.y = roadY - 8.4 - roughA * 5.4 + valleyWave * 0.7;

    lip.push(lipP);
    bench.push(benchP);
    lower.push(lowerP);
    far.push(farP);
  }

  const shoulder = makeRibbon(lip, bench, mats.valleyShoulder, 0);
  const slope = makeRibbon(bench, lower, mats.valleySlope, 0);
  const darkness = makeRibbon(lower, far, mats.valleyDark, 0);
  for (const mesh of [shoulder, slope, darkness]) {
    mesh.receiveShadow = true;
    mesh.castShadow = false;
  }
  group.add(shoulder, slope, darkness);
}

function buildCliffStrata(group, profile, mats, surfaceLift) {
  const count = Math.min(260, Math.max(32, Math.floor(profile.length / 4) * 2));
  const stripeGeo = new THREE.BoxGeometry(4.8, 0.045, 0.08);
  const stripes = new THREE.InstancedMesh(stripeGeo, mats.cliffStrata, count);
  stripes.renderOrder = 4;
  let n = 0;
  for (let d = 9; d < profile.length - 8 && n < count; d += 6.5) {
    const sample = trackSampleAtDistance(profile, d);
    const outward = sample.right.clone().sub(sample.center).setY(0).normalize();
    const roadY = roadHeightAt(profile, sample.right.x, sample.right.z, sample.right.y) + surfaceLift;
    const stripeRows = hash01(d * 0.97) > 0.48 ? 2 : 1;
    for (let row = 0; row < stripeRows && n < count; row++) {
      const lateral = 5.4 + hash01(d * 1.71 + row) * 8.0;
      const pos = sample.right.clone().add(outward.clone().multiplyScalar(lateral));
      pos.y = roadY + 1.25 + row * 2.3 + hash01(d * 2.31 + row) * 7.2;
      const yaw = sample.yaw + (hash01(d * 3.19 + row) - 0.5) * 0.22;
      const scale = new THREE.Vector3(0.45 + hash01(d * 4.13 + row) * 1.25, 1, 1);
      setInstancedTransform(stripes, n, pos, yaw, scale);
      n++;
    }
  }
  stripes.count = n;
  group.add(stripes);
}

function buildCliffCrevices(group, profile, mats, surfaceLift) {
  if (!mats.cliffCrevice) return;
  const count = Math.min(210, Math.max(24, Math.floor(profile.length / 5)));
  const crackGeo = new THREE.BoxGeometry(0.055, 1.0, 0.08);
  const cracks = new THREE.InstancedMesh(crackGeo, mats.cliffCrevice, count);
  cracks.renderOrder = 4;
  let n = 0;
  for (let d = 10; d < profile.length - 10 && n < count; d += 4.9) {
    if (hash01(d * 0.45) < 0.22) continue;
    const sample = trackSampleAtDistance(profile, d);
    const outward = sample.right.clone().sub(sample.center).setY(0).normalize();
    const roadY = roadHeightAt(profile, sample.right.x, sample.right.z, sample.right.y) + surfaceLift;
    const lateral = 3.0 + hash01(d * 1.33) * 13.0;
    const p = sample.right.clone().add(outward.multiplyScalar(lateral));
    p.y = roadY + 1.0 + hash01(d * 2.17) * 9.0;
    const scale = new THREE.Vector3(1, 0.7 + hash01(d * 3.03) * 2.8, 1);
    setInstancedTransform(cracks, n, p, sample.yaw + (hash01(d * 4.1) - 0.5) * 0.35, scale);
    n++;
  }
  cracks.count = n;
  group.add(cracks);
}

function buildCliffLampWashes(group, profile, mats, surfaceLift) {
  const count = Math.min(72, Math.max(8, Math.floor(profile.length / 42)));
  const washGeo = new THREE.PlaneGeometry(1, 1, 1, 1);
  const washes = new THREE.InstancedMesh(washGeo, mats.lampWallWash, count);
  washes.renderOrder = 5;
  let n = 0;
  for (let d = 32; d < Math.max(40, profile.length - 18) && n < count; d += 44) {
    const sample = trackSampleAtDistance(profile, d);
    const outward = sample.right.clone().sub(sample.center).setY(0).normalize();
    const roadY = roadHeightAt(profile, sample.right.x, sample.right.z, sample.right.y) + surfaceLift;
    const pos = sample.right.clone().add(outward.multiplyScalar(5.0));
    pos.y = roadY + 2.35 + hash01(d * 1.3) * 1.1;
    const scale = new THREE.Vector3(4.6 + hash01(d * 2.2) * 2.2, 3.1 + hash01(d * 3.1) * 1.6, 1);
    setInstancedTransform(washes, n, pos, sample.yaw, scale);
    n++;
  }
  washes.count = n;
  group.add(washes);
}

function buildValleyPineBands(group, profile, mats, surfaceLift) {
  const count = Math.min(170, Math.max(36, Math.floor(profile.length / 4)));
  const trunkGeo = new THREE.CylinderGeometry(0.11, 0.17, 1.28, 5);
  const crownGeo = new THREE.ConeGeometry(0.92, 2.72, 6);
  const trunks = new THREE.InstancedMesh(trunkGeo, mats.propPineTrunk || mats.pineTrunk, count);
  const crowns = new THREE.InstancedMesh(crownGeo, mats.propPineNeedles || mats.pineNeedles, count);
  trunks.castShadow = crowns.castShadow = true;
  let n = 0;
  for (let d = 18; d < profile.length - 12 && n < count; d += 8) {
    const sample = trackSampleAtDistance(profile, d);
    const outward = sample.left.clone().sub(sample.center).setY(0).normalize();
    for (let row = 0; row < 2 && n < count; row++) {
      if (hash01(d * 0.41 + row * 8.0) < 0.18) continue;
      const offset = 12 + row * 15 + hash01(d * 1.7 + row) * 9;
      const p = sample.left.clone().add(outward.clone().multiplyScalar(offset));
      const roadY = roadHeightAt(profile, p.x, p.z, sample.left.y) + surfaceLift;
      p.y = roadY - 1.5 - row * 2.7 + (hash01(d * 2.8 + row) - 0.5) * 0.9;
      const h = 0.68 + hash01(d * 4.1 + row) * 1.1;
      const yaw = sample.yaw + hash01(d * 6.2 + row) * Math.PI;
      setInstancedTransform(trunks, n, p.clone().add(new THREE.Vector3(0, 0.62 * h, 0)), yaw, new THREE.Vector3(h, h, h));
      setInstancedTransform(crowns, n, p.clone().add(new THREE.Vector3(0, 2.10 * h, 0)), yaw, new THREE.Vector3(h, h, h));
      n++;
    }
  }
  trunks.count = crowns.count = n;
  group.add(trunks, crowns);
}

function buildRoadside(group, profile, mats, surfaceLift) {
  buildHeroPassCorridor(group, profile, mats, surfaceLift);
  buildDistantRidges(group, profile, mats);
  buildMountainPeaks(group, profile, mats);
  buildGuardrails(group, profile, mats);
  buildReflectorPosts(group, profile, mats, surfaceLift);
  buildSparseStreetLamps(group, profile, mats, surfaceLift);
  buildRockWalls(group, profile, mats);
  buildPines(group, profile, mats);
  buildCurveChevrons(group, profile, mats, surfaceLift);
  buildReflectiveSigns(group, profile, mats);
  const fogPlanes = buildValleyFog(group, profile, mats);
  return fogPlanes;
}

function buildDistantRidges(group, profile, mats) {
  const points = profile.center.concat(profile.left, profile.right);
  const box = new THREE.Box3().setFromPoints(points);
  const center = box.getCenter(new THREE.Vector3());
  const spanX = box.max.x - box.min.x;
  const spanZ = box.max.z - box.min.z;
  const ring = Math.max(spanX, spanZ) * 0.68 + 120;
  const count = Math.min(58, Math.max(16, Math.floor(profile.length / 18)));
  const ridgeGeo = jitterGeometry(new THREE.BoxGeometry(9.5, 5.2, 2.6), 0.95, 620);
  const ridges = new THREE.InstancedMesh(ridgeGeo, mats.ridge, count);
  ridges.castShadow = ridges.receiveShadow = true;
  for (let i = 0; i < count; i++) {
    const a = (i / count) * Math.PI * 2;
    const wobble = (hash01(i * 4.77) - 0.5) * 0.62;
    const radius = ring + hash01(i * 7.31) * 90;
    const px = center.x + Math.cos(a + wobble) * radius;
    const pz = center.z + Math.sin(a + wobble) * radius;
    const p = new THREE.Vector3(px, terrainHeightAt(profile, px, pz) + 2.2 + hash01(i * 8.2) * 4.8, pz);
    const scale = new THREE.Vector3(
      1.7 + hash01(i * 2.1) * 3.2,
      1.0 + hash01(i * 3.3) * 2.6,
      1.4 + hash01(i * 5.5) * 2.5,
    );
    setInstancedTransform(ridges, i, p, -a + Math.PI / 2 + (hash01(i * 9.1) - 0.5) * 0.35, scale);
  }
  group.add(ridges);
}

function buildGuardrails(group, profile, mats) {
  const segmentCount = Math.min(240, Math.floor(profile.center.length / 3) * 2);
  const railGeo = new THREE.BoxGeometry(3.75, 0.16, 0.17);
  const lowRailGeo = new THREE.BoxGeometry(3.45, 0.10, 0.12);
  const postGeo = new THREE.BoxGeometry(0.15, 0.92, 0.15);
  const glintGeo = new THREE.BoxGeometry(0.035, 0.12, 0.20);
  const shadowGeo = new THREE.BoxGeometry(3.65, 0.014, 0.58);
  const upperRails = new THREE.InstancedMesh(railGeo, mats.propGuardrail || mats.guardrail, segmentCount);
  const lowerRails = new THREE.InstancedMesh(lowRailGeo, mats.propGuardrail || mats.guardrail, segmentCount);
  const posts = new THREE.InstancedMesh(postGeo, mats.propPole || mats.pole, segmentCount);
  const glints = new THREE.InstancedMesh(glintGeo, mats.reflectorCool, segmentCount);
  const shadows = new THREE.InstancedMesh(shadowGeo, mats.railShadow || mats.roadEdgeShadow, segmentCount);
  upperRails.castShadow = lowerRails.castShadow = posts.castShadow = true;
  shadows.renderOrder = 4;
  let n = 0;
  let gN = 0;
  let sN = 0;
  for (let i = 2; i < profile.center.length - 2 && n < segmentCount; i += 3) {
    for (const side of ["left", "right"]) {
      const edge = side === "left" ? profile.left : profile.right;
      const outward = outwardAt(profile.center, edge, i);
      const yaw = trackYawAt(profile.center, i);
      const isValley = side === "left";
      const lateral = isValley ? 1.34 : 1.02;
      const railScale = new THREE.Vector3(isValley ? 1.04 : 0.88, 1, 1);
      const railPos = edge[i].clone().add(outward.clone().multiplyScalar(lateral));
      const railY = roadHeightAt(profile, railPos.x, railPos.z, edge[i].y);
      railPos.y = railY + (isValley ? 0.88 : 0.74);
      setInstancedTransform(upperRails, n, railPos, yaw, railScale);
      const lowPos = railPos.clone();
      lowPos.y = railY + (isValley ? 0.55 : 0.46);
      setInstancedTransform(lowerRails, n, lowPos, yaw, railScale);
      const postPos = railPos.clone().add(outward.clone().multiplyScalar(0.04));
      postPos.y = railY + (isValley ? 0.40 : 0.33);
      setInstancedTransform(posts, n, postPos, yaw);
      if (sN < segmentCount) {
        const shadow = railPos.clone().add(outward.clone().multiplyScalar(isValley ? -0.42 : -0.32));
        shadow.y = railY + 0.24;
        const shadowScale = new THREE.Vector3(isValley ? 1.10 : 0.82, 1, isValley ? 1.10 : 0.84);
        setInstancedTransform(shadows, sN, shadow, yaw, shadowScale);
        sN++;
      }
      if ((i + (isValley ? 0 : 1)) % 6 === 0 && gN < segmentCount) {
        const glint = railPos.clone().add(outward.clone().multiplyScalar(-0.10));
        glint.y += 0.02;
        setInstancedTransform(glints, gN, glint, yaw + Math.PI / 2);
        gN++;
      }
      n++;
      if (n >= segmentCount) break;
    }
  }
  upperRails.count = lowerRails.count = posts.count = n;
  glints.count = gN;
  shadows.count = sN;
  group.add(upperRails, lowerRails, posts, glints, shadows);
}

function buildReflectorPosts(group, profile, mats, surfaceLift) {
  const count = Math.min(220, Math.max(24, Math.floor(profile.length / 5.5) * 2));
  const postGeo = new THREE.BoxGeometry(0.09, 0.54, 0.09);
  const reflectorGeo = new THREE.BoxGeometry(0.04, 0.12, 0.18);
  const posts = new THREE.InstancedMesh(postGeo, mats.propPole || mats.pole, count);
  const warm = new THREE.InstancedMesh(reflectorGeo, mats.reflectorWarm, count);
  const cool = new THREE.InstancedMesh(reflectorGeo, mats.reflectorCool, count);
  let n = 0;
  let warmN = 0;
  let coolN = 0;
  for (let d = 8; d < profile.length - 6 && n < count; d += 11) {
    const sample = trackSampleAtDistance(profile, d);
    const sidePair = d % 33 < 11 ? [sample.left] : [sample.left, sample.right];
    for (const edge of sidePair) {
      if (n >= count) break;
      const outward = edge.clone().sub(sample.center).setY(0).normalize();
      const base = edge.clone().add(outward.clone().multiplyScalar(1.95));
      base.y = roadHeightAt(profile, base.x, base.z, edge.y) + surfaceLift + 0.27;
      const yaw = sample.yaw + Math.PI / 2;
      setInstancedTransform(posts, n, base, yaw);
      const head = base.clone().add(new THREE.Vector3(0, 0.17, 0));
      if (hash01(d * 4.2 + n) > 0.45) {
        setInstancedTransform(warm, warmN, head, yaw);
        warmN++;
      } else {
        setInstancedTransform(cool, coolN, head, yaw);
        coolN++;
      }
      n++;
    }
  }
  posts.count = n;
  warm.count = warmN;
  cool.count = coolN;
  group.add(posts, warm, cool);
}

function buildSparseStreetLamps(group, profile, mats, surfaceLift) {
  const placements = [];
  let station = 0;
  for (let d = 32; d < Math.max(40, profile.length - 18) && placements.length < 88; d += 44) {
    placements.push({ distance: d, side: station % 4 === 2 ? "left" : "right" });
    if (station % 5 === 4) placements.push({ distance: d + 10, side: "left" });
    station++;
  }
  if (!placements.length && profile.center.length > 2) placements.push({ distance: profile.length * 0.35, side: "left" });

  const count = Math.min(78, placements.length);
  const poleGeo = new THREE.BoxGeometry(0.15, 4.35, 0.15);
  const armGeo = new THREE.BoxGeometry(0.13, 0.13, 3.15);
  const headGeo = new THREE.BoxGeometry(0.94, 0.18, 0.38);
  const glowGeo = new THREE.BoxGeometry(0.58, 0.28, 0.08);
  const poolGeo = new THREE.CircleGeometry(3.7, 18);
  poolGeo.rotateX(-Math.PI / 2);
  const streakGeo = new THREE.BoxGeometry(6.2, 0.018, 0.82);
  const poles = new THREE.InstancedMesh(poleGeo, mats.propPole || mats.pole, count);
  const arms = new THREE.InstancedMesh(armGeo, mats.propPole || mats.pole, count);
  const heads = new THREE.InstancedMesh(headGeo, mats.streetLamp, count);
  const glows = new THREE.InstancedMesh(glowGeo, mats.lampGlow, count);
  const pools = new THREE.InstancedMesh(poolGeo, mats.lightPool, count);
  const streaks = new THREE.InstancedMesh(streakGeo, mats.lampRoadStreak, count);
  poles.castShadow = arms.castShadow = heads.castShadow = true;
  pools.renderOrder = 7;
  streaks.renderOrder = 8;
  glows.renderOrder = 8;
  let n = 0;
  let lit = 0;
  for (const placement of placements) {
    if (n >= count) break;
    const sample = trackSampleAtDistance(profile, placement.distance);
    const rig = makeStreetLampRig(profile, sample, placement.side, surfaceLift);
    setInstancedTransform(poles, n, rig.pole, rig.yaw);
    setInstancedTransform(arms, n, rig.arm, rig.yaw);
    setInstancedTransform(heads, n, rig.head, rig.yaw);
    setInstancedTransform(glows, n, rig.glow, rig.yaw);
    setInstancedTransform(pools, n, rig.pool, rig.yaw, rig.poolScale);
    setInstancedTransform(streaks, n, rig.streak, rig.yaw, rig.streakScale);
    if (lit < 34) {
      const light = new THREE.PointLight(0xffb35f, 2.05, 52, 1.42);
      light.position.copy(rig.head);
      light.castShadow = false;
      group.add(light);
      lit++;
    }
    n++;
  }
  poles.count = arms.count = heads.count = glows.count = pools.count = streaks.count = n;
  group.add(poles, arms, heads, pools, streaks, glows);
}

function makeStreetLampRig(profile, sample, side, surfaceLift) {
  const edge = side === "left" ? sample.left : sample.right;
  const outward = edge.clone().sub(sample.center).setY(0).normalize();
  const base = edge.clone().add(outward.clone().multiplyScalar(3.2));
  base.y = terrainHeightAt(profile, base.x, base.z);
  const pool = edge.clone().add(outward.clone().multiplyScalar(-0.55));
  pool.y = roadHeightAt(profile, pool.x, pool.z, edge.y) + surfaceLift + 0.040;
  const head = pool.clone();
  head.y = Math.max(base.y + 4.10, pool.y + 3.95);
  const pole = base.clone().add(new THREE.Vector3(0, 2.175, 0));
  const arm = base.clone().lerp(pool, 0.5);
  arm.y = head.y - 0.10;
  const glow = head.clone().add(new THREE.Vector3(0, -0.08, 0));
  const streak = pool.clone();
  streak.y += 0.028;
  return {
    pole,
    arm,
    head,
    glow,
    streak,
    pool,
    poolScale: new THREE.Vector3(0.90, 1, 0.52),
    streakScale: new THREE.Vector3(0.72, 1, 0.48),
    yaw: sample.yaw,
  };
}

function buildRockWalls(group, profile, mats) {
  const count = Math.min(150, Math.floor(profile.length / 5.5));
  const rockGeo = jitterGeometry(new THREE.BoxGeometry(3.8, 2.7, 1.3), 0.36, 200);
  const rocks = new THREE.InstancedMesh(rockGeo, mats.propRock || mats.rock, count);
  rocks.castShadow = rocks.receiveShadow = true;
  let n = 0;
  for (let d = 14; d < profile.length - 10 && n < count; d += 5.8) {
    const sample = trackSampleAtDistance(profile, d);
    const edge = hash01(d * 0.21) > 0.18 ? sample.right : sample.left;
    const outward = edge.clone().sub(sample.center).setY(0).normalize();
    const cliffBias = edge === sample.right ? 1 : 0;
    const p = edge.clone().add(outward.clone().multiplyScalar((cliffBias ? 3.5 : 6.2) + hash01(d * 0.7) * (cliffBias ? 6.0 : 4.0)));
    p.y = terrainHeightAt(profile, p.x, p.z) + 0.80 + hash01(d * 1.7) * (cliffBias ? 2.2 : 1.0);
    const scale = new THREE.Vector3(
      0.78 + hash01(d * 2.1) * (cliffBias ? 1.45 : 0.95),
      0.76 + hash01(d * 3.2) * (cliffBias ? 1.90 : 1.10),
      0.78 + hash01(d * 4.3) * 0.90
    );
    setInstancedTransform(rocks, n, p, sample.yaw + (hash01(d * 5.2) - 0.5) * 0.55, scale);
    n++;
  }
  rocks.count = n;
  group.add(rocks);
}

function buildPines(group, profile, mats) {
  const count = Math.min(180, Math.floor(profile.length / 4));
  const trunkGeo = new THREE.CylinderGeometry(0.10, 0.15, 1.15, 5);
  const crownGeo = new THREE.ConeGeometry(0.82, 2.35, 6);
  const trunks = new THREE.InstancedMesh(trunkGeo, mats.propPineTrunk || mats.pineTrunk, count);
  const crowns = new THREE.InstancedMesh(crownGeo, mats.propPineNeedles || mats.pineNeedles, count);
  trunks.castShadow = crowns.castShadow = true;
  let n = 0;
  for (let d = 12; d < profile.length - 8 && n < count; d += 5.5) {
    if (hash01(d * 0.33) < 0.28) continue;
    const sample = trackSampleAtDistance(profile, d);
    const side = hash01(d * 1.9) > 0.5 ? sample.left : sample.right;
    const outward = side.clone().sub(sample.center).setY(0).normalize();
    const offset = 8 + hash01(d * 2.7) * 30;
    const p = side.clone().add(outward.multiplyScalar(offset));
    p.y = terrainHeightAt(profile, p.x, p.z);
    const h = 0.85 + hash01(d * 4.1) * 0.85;
    setInstancedTransform(trunks, n, p.clone().add(new THREE.Vector3(0, 0.55 * h, 0)), sample.yaw, new THREE.Vector3(h, h, h));
    setInstancedTransform(crowns, n, p.clone().add(new THREE.Vector3(0, 1.95 * h, 0)), sample.yaw + hash01(d * 7.2), new THREE.Vector3(h, h, h));
    n++;
  }
  trunks.count = crowns.count = n;
  group.add(trunks, crowns);
}

function buildCurveChevrons(group, profile, mats, surfaceLift) {
  const count = Math.min(72, Math.max(8, Math.floor(profile.length / 18)));
  const panelGeo = new THREE.BoxGeometry(0.09, 0.62, 1.36);
  const postGeo = new THREE.BoxGeometry(0.10, 1.10, 0.10);
  const panels = new THREE.InstancedMesh(panelGeo, mats.chevronSign || mats.sign, count);
  const posts = new THREE.InstancedMesh(postGeo, mats.propPole || mats.pole, count);
  let n = 0;
  for (let d = 26; d < profile.length - 26 && n < count; d += 18) {
    const before = trackSampleAtDistance(profile, d - 9);
    const after = trackSampleAtDistance(profile, d + 9);
    const turn = angleDelta(before.yaw, after.yaw);
    if (Math.abs(turn) < 0.145) continue;

    const sample = trackSampleAtDistance(profile, d);
    const edge = turn > 0 ? sample.right : sample.left;
    const outward = edge.clone().sub(sample.center).setY(0).normalize();
    const base = edge.clone().add(outward.clone().multiplyScalar(3.8));
    base.y = roadHeightAt(profile, base.x, base.z, edge.y) + surfaceLift;
    const yaw = sample.yaw + Math.PI / 2;
    const post = base.clone().add(new THREE.Vector3(0, 0.58, 0));
    const panel = base.clone().add(new THREE.Vector3(0, 1.32, 0));
    setInstancedTransform(posts, n, post, yaw);
    setInstancedTransform(panels, n, panel, yaw);
    n++;
  }
  panels.count = posts.count = n;
  panels.castShadow = posts.castShadow = true;
  group.add(posts, panels);
}

function buildReflectiveSigns(group, profile, mats) {
  const count = Math.min(22, Math.floor(profile.length / 55));
  const panelGeo = new THREE.BoxGeometry(0.10, 0.92, 1.72);
  const postGeo = new THREE.BoxGeometry(0.11, 1.55, 0.11);
  const panels = new THREE.InstancedMesh(panelGeo, mats.propSign || mats.sign, count);
  const posts = new THREE.InstancedMesh(postGeo, mats.propPole || mats.pole, count);
  let n = 0;
  for (let d = 28; d < profile.length - 16 && n < count; d += 58) {
    const sample = trackSampleAtDistance(profile, d);
    const edge = n % 2 === 0 ? sample.right : sample.left;
    const outward = edge.clone().sub(sample.center).setY(0).normalize();
    const yaw = sample.yaw + Math.PI / 2;
    const base = edge.clone().add(outward.multiplyScalar(4.8));
    base.y = terrainHeightAt(profile, base.x, base.z);
    setInstancedTransform(posts, n, base.clone().add(new THREE.Vector3(0, 0.78, 0)), yaw);
    setInstancedTransform(panels, n, base.clone().add(new THREE.Vector3(0, 1.88, 0)), yaw);
    n++;
  }
  panels.count = posts.count = n;
  group.add(posts, panels);
}

/* ── Dramatic mountain peaks with snow caps ─────────────────── */

function snowCapScale(peakW, peakD, peakH, snowH, seed) {
  const capBaseRatio = Math.max(0.08, Math.min(0.38, snowH / Math.max(1e-6, peakH)));
  const tuck = 0.68 + hash01(seed) * 0.14;
  return new THREE.Vector3(
    peakW * capBaseRatio * tuck,
    snowH,
    peakD * capBaseRatio * tuck
  );
}

function buildMountainPeaks(group, profile, mats) {
  const points = profile.center.concat(profile.left, profile.right);
  const box = new THREE.Box3().setFromPoints(points);
  const center = box.getCenter(new THREE.Vector3());
  const spanX = box.max.x - box.min.x;
  const spanZ = box.max.z - box.min.z;
  const baseRing = Math.max(spanX, spanZ) * 0.68 + 120;

  // --- Ring 1: Mid-range peaks (closer, medium height) ---
  const midCount = Math.min(32, Math.max(10, Math.floor(profile.length / 24)));
  const midPeakGeo = jitterGeometry(new THREE.ConeGeometry(1.0, 1.0, 6), 0.12, 800);
  const midSnowGeo = jitterGeometry(new THREE.ConeGeometry(1.0, 1.0, 6), 0.08, 801);
  const midPeaks = new THREE.InstancedMesh(midPeakGeo, mats.mountainMid, midCount);
  const midSnow  = new THREE.InstancedMesh(midSnowGeo, mats.snowCap, midCount);
  midPeaks.castShadow = midPeaks.receiveShadow = true;
  midSnow.receiveShadow = true;

  for (let i = 0; i < midCount; i++) {
    const a = (i / midCount) * Math.PI * 2;
    const wobble = (hash01(i * 6.31) - 0.5) * 0.52;
    const radius = baseRing * 1.04 + hash01(i * 3.17) * 80;
    const px = center.x + Math.cos(a + wobble) * radius;
    const pz = center.z + Math.sin(a + wobble) * radius;
    const baseY = terrainHeightAt(profile, px, pz);
    const peakH = 14 + hash01(i * 5.3) * 22;
    const peakW = 9.0 + hash01(i * 2.9) * 12.5;
    const peakD = 7.0 + hash01(i * 4.1) * 9.5;
    const p = new THREE.Vector3(px, baseY + peakH * 0.5 - 1.5, pz);
    const scale = new THREE.Vector3(peakW, peakH, peakD);
    const yaw = -a + Math.PI / 2 + (hash01(i * 8.7) - 0.5) * 0.6;
    setInstancedTransform(midPeaks, i, p, yaw, scale);

    // Snow cap: smaller cone at the top
    const snowH = peakH * (0.15 + hash01(i * 7.1) * 0.09);
    const snowP = new THREE.Vector3(px, baseY + peakH - snowH * 0.5 - 1.5, pz);
    setInstancedTransform(midSnow, i, snowP, yaw, snowCapScale(peakW, peakD, peakH, snowH, i * 6.2));
  }
  group.add(midPeaks, midSnow);

  // --- Ring 2: Far background peaks (taller, further away) ---
  const farCount = Math.min(24, Math.max(8, Math.floor(profile.length / 34)));
  const farPeakGeo = jitterGeometry(new THREE.ConeGeometry(1.0, 1.0, 5), 0.15, 900);
  const farSnowGeo = jitterGeometry(new THREE.ConeGeometry(1.0, 1.0, 5), 0.10, 901);
  const farPeaks = new THREE.InstancedMesh(farPeakGeo, mats.mountainDark, farCount);
  const farSnow  = new THREE.InstancedMesh(farSnowGeo, mats.snowCap, farCount);
  farPeaks.receiveShadow = true;
  farSnow.receiveShadow = true;

  for (let i = 0; i < farCount; i++) {
    const a = (i / farCount) * Math.PI * 2;
    const wobble = (hash01(i * 11.31) - 0.5) * 0.42;
    const radius = baseRing * 1.6 + hash01(i * 9.17) * 120;
    const px = center.x + Math.cos(a + wobble) * radius;
    const pz = center.z + Math.sin(a + wobble) * radius;
    const baseY = terrainHeightAt(profile, px, pz) - 5;
    const peakH = 32 + hash01(i * 12.3) * 50;
    const peakW = 24 + hash01(i * 8.9) * 34;
    const peakD = 18 + hash01(i * 7.1) * 27;
    const p = new THREE.Vector3(px, baseY + peakH * 0.5, pz);
    const scale = new THREE.Vector3(peakW, peakH, peakD);
    const yaw = -a + (hash01(i * 13.1) - 0.5) * 0.8;
    setInstancedTransform(farPeaks, i, p, yaw, scale);

    // Snow cap
    const snowH = peakH * (0.14 + hash01(i * 14.1) * 0.08);
    const snowP = new THREE.Vector3(px, baseY + peakH - snowH * 0.5, pz);
    setInstancedTransform(farSnow, i, snowP, yaw, snowCapScale(peakW, peakD, peakH, snowH, i * 10.2));
  }
  group.add(farPeaks, farSnow);

  // --- Ring 3: Massive distant silhouette peaks ---
  const silCount = Math.min(12, Math.max(5, Math.floor(profile.length / 58)));
  const silGeo = jitterGeometry(new THREE.ConeGeometry(1.0, 1.0, 4), 0.18, 1000);
  const silSnowGeo = jitterGeometry(new THREE.ConeGeometry(1.0, 1.0, 4), 0.12, 1001);
  const silPeaks = new THREE.InstancedMesh(silGeo, mats.mountainDark, silCount);
  const silSnow  = new THREE.InstancedMesh(silSnowGeo, mats.snowCap, silCount);

  for (let i = 0; i < silCount; i++) {
    const a = (i / silCount) * Math.PI * 2;
    const wobble = (hash01(i * 15.7) - 0.5) * 0.55;
    const radius = baseRing * 2.5 + hash01(i * 17.3) * 160;
    const px = center.x + Math.cos(a + wobble) * radius;
    const pz = center.z + Math.sin(a + wobble) * radius;
    const baseY = -12;
    const peakH = 55 + hash01(i * 20.1) * 85;
    const peakW = 44 + hash01(i * 18.2) * 70;
    const peakD = 34 + hash01(i * 16.4) * 55;
    const p = new THREE.Vector3(px, baseY + peakH * 0.5, pz);
    const scale = new THREE.Vector3(peakW, peakH, peakD);
    setInstancedTransform(silPeaks, i, p, hash01(i * 21.0) * Math.PI, scale);

    const snowH = peakH * (0.11 + hash01(i * 22.1) * 0.07);
    const snowP = new THREE.Vector3(px, baseY + peakH - snowH * 0.5, pz);
    setInstancedTransform(silSnow, i, snowP, hash01(i * 21.0) * Math.PI, snowCapScale(peakW, peakD, peakH, snowH, i * 19.2));
  }
  group.add(silPeaks, silSnow);
}

/* ── Layered valley fog planes ─────────────────────────────── */

function buildValleyFog(group, profile, mats) {
  const points = profile.center.concat(profile.left, profile.right);
  const box = new THREE.Box3().setFromPoints(points);
  const center = box.getCenter(new THREE.Vector3());
  const span = Math.max(box.max.x - box.min.x, box.max.z - box.min.z);
  const fogPlanes = [];

  // Layer config: [height offset from road, size multiplier, opacity night, opacity day, color night]
  const layers = [
    { yOff: -3.5,  sizeMul: 2.8,  nightOp: 0.20, dayOp: 0.18, nightCol: 0x2a3845 },
    { yOff: -6.0,  sizeMul: 3.5,  nightOp: 0.16, dayOp: 0.14, nightCol: 0x1e2d3a },
    { yOff: -1.0,  sizeMul: 2.2,  nightOp: 0.10, dayOp: 0.12, nightCol: 0x384858 },
    { yOff: -10.0, sizeMul: 4.0,  nightOp: 0.12, dayOp: 0.10, nightCol: 0x1a2535 },
  ];

  const avgRoadY = profile.center.reduce((s, p) => s + p.y, 0) / Math.max(1, profile.center.length);

  for (let li = 0; li < layers.length; li++) {
    const cfg = layers[li];
    const fogSize = span + cfg.sizeMul * 100;
    const geo = new THREE.PlaneGeometry(fogSize, fogSize, 1, 1);
    geo.rotateX(-Math.PI / 2);
    const mat = new THREE.MeshBasicMaterial({
      color: cfg.nightCol,
      transparent: true,
      opacity: cfg.nightOp,
      depthWrite: false,
      side: THREE.DoubleSide,
      blending: THREE.NormalBlending,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(center.x, avgRoadY + cfg.yOff, center.z);
    mesh.renderOrder = -1;
    mesh.userData.baseY = mesh.position.y;
    mesh.userData.nightColor = cfg.nightCol;
    mesh.userData.nightOpacity = cfg.nightOp;
    mesh.userData.dayOpacity = cfg.dayOp;
    mesh.userData.bobSpeed = 0.15 + li * 0.08;
    mesh.userData.bobAmount = 0.35 + li * 0.15;
    group.add(mesh);
    fogPlanes.push(mesh);
  }

  return fogPlanes;
}
