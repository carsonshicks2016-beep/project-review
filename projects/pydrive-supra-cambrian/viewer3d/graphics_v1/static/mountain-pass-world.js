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
  addLine(roadGroup, left, mats.edgeLine, 0.30);
  addLine(roadGroup, right, mats.edgeLine, 0.30);
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
    verts.push(profile.left[i].x, profile.left[i].y + surfaceLift, profile.left[i].z);
    verts.push(profile.right[i].x, profile.right[i].y + surfaceLift, profile.right[i].z);
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
    leftOuter.push(profile.left[i].clone().add(ldir.multiplyScalar(2.4)).add(new THREE.Vector3(0, -0.04, 0)));
    rightOuter.push(profile.right[i].clone().add(rdir.multiplyScalar(2.4)).add(new THREE.Vector3(0, -0.04, 0)));
  }
  group.add(makeRibbon(leftOuter, profile.left, mats.shoulder, 2));
  group.add(makeRibbon(profile.right, rightOuter, mats.shoulder, 2));
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
  const dashPts2 = [];
  for (let i = 0; i < profile.center.length - 1; i += 7) {
    const j = Math.min(i + 3, profile.center.length - 1);
    const sideA = profile.left[i].clone().sub(profile.center[i]).setY(0).normalize();
    const sideB = profile.left[j].clone().sub(profile.center[j]).setY(0).normalize();
    const a = profile.center[i].clone().add(sideA.clone().multiplyScalar(0.22)).add(new THREE.Vector3(0, surfaceLift + 0.16, 0));
    const b = profile.center[j].clone().add(sideB.clone().multiplyScalar(0.22)).add(new THREE.Vector3(0, surfaceLift + 0.16, 0));
    const c = profile.center[i].clone().add(sideA.clone().multiplyScalar(-0.22)).add(new THREE.Vector3(0, surfaceLift + 0.16, 0));
    const d = profile.center[j].clone().add(sideB.clone().multiplyScalar(-0.22)).add(new THREE.Vector3(0, surfaceLift + 0.16, 0));
    dashPts.push(a, b);
    dashPts2.push(c, d);
  }
  const dashes = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(dashPts), mats.centerLine);
  dashes.renderOrder = 6;
  group.add(dashes);
  const dashes2 = new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(dashPts2), mats.centerLine);
  dashes2.renderOrder = 6;
  group.add(dashes2);
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

function buildRoadside(group, profile, mats, surfaceLift) {
  buildDistantRidges(group, profile, mats);
  buildMountainPeaks(group, profile, mats);
  buildGuardrails(group, profile, mats);
  buildReflectorPosts(group, profile, mats, surfaceLift);
  buildSparseStreetLamps(group, profile, mats, surfaceLift);
  buildRockWalls(group, profile, mats);
  buildPines(group, profile, mats);
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
  const count = Math.min(90, Math.max(24, Math.floor(profile.length / 12)));
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
  const railGeo = new THREE.BoxGeometry(4.3, 0.18, 0.15);
  const postGeo = new THREE.BoxGeometry(0.15, 0.96, 0.15);
  const count = Math.min(190, Math.floor(profile.center.length / 4) * 2);
  const rails = new THREE.InstancedMesh(railGeo, mats.guardrail, count);
  const posts = new THREE.InstancedMesh(postGeo, mats.pole, count);
  rails.castShadow = posts.castShadow = true;
  let n = 0;
  for (let i = 2; i < profile.center.length - 2 && n < count; i += 4) {
    for (const edge of [profile.left, profile.right]) {
      const outward = outwardAt(profile.center, edge, i);
      const yaw = trackYawAt(profile.center, i);
      const railPos = edge[i].clone().add(outward.multiplyScalar(1.35));
      railPos.y = roadHeightAt(profile, railPos.x, railPos.z, edge[i].y) + 0.83;
      setInstancedTransform(rails, n, railPos, yaw);
      const postPos = railPos.clone();
      postPos.y -= 0.34;
      setInstancedTransform(posts, n, postPos, yaw);
      n++;
      if (n >= count) break;
    }
  }
  rails.count = posts.count = n;
  group.add(rails, posts);
}

function buildReflectorPosts(group, profile, mats, surfaceLift) {
  const count = Math.min(220, Math.max(24, Math.floor(profile.length / 5.5) * 2));
  const postGeo = new THREE.BoxGeometry(0.09, 0.54, 0.09);
  const reflectorGeo = new THREE.BoxGeometry(0.04, 0.12, 0.18);
  const posts = new THREE.InstancedMesh(postGeo, mats.pole, count);
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
  for (let d = 32; d < Math.max(40, profile.length - 18) && placements.length < 72; d += 54) {
    placements.push({ distance: d, side: station % 2 === 0 ? "left" : "right" });
    if (station % 5 === 4) placements.push({ distance: d + 10, side: station % 2 === 0 ? "right" : "left" });
    station++;
  }
  if (!placements.length && profile.center.length > 2) placements.push({ distance: profile.length * 0.35, side: "left" });

  const count = Math.min(78, placements.length);
  const poleGeo = new THREE.BoxGeometry(0.15, 4.35, 0.15);
  const armGeo = new THREE.BoxGeometry(0.13, 0.13, 3.15);
  const headGeo = new THREE.BoxGeometry(0.94, 0.18, 0.38);
  const glowGeo = new THREE.BoxGeometry(0.58, 0.28, 0.08);
  const poolGeo = new THREE.CircleGeometry(5.8, 14);
  poolGeo.rotateX(-Math.PI / 2);
  const poles = new THREE.InstancedMesh(poleGeo, mats.pole, count);
  const arms = new THREE.InstancedMesh(armGeo, mats.pole, count);
  const heads = new THREE.InstancedMesh(headGeo, mats.streetLamp, count);
  const glows = new THREE.InstancedMesh(glowGeo, mats.lampGlow, count);
  const pools = new THREE.InstancedMesh(poolGeo, mats.lightPool, count);
  poles.castShadow = arms.castShadow = heads.castShadow = true;
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
    if (lit < 34) {
      const light = new THREE.PointLight(0xffb35f, 2.05, 52, 1.42);
      light.position.copy(rig.head);
      light.castShadow = false;
      group.add(light);
      lit++;
    }
    n++;
  }
  poles.count = arms.count = heads.count = glows.count = pools.count = n;
  group.add(poles, arms, heads, glows, pools);
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
  return {
    pole,
    arm,
    head,
    glow,
    pool,
    poolScale: new THREE.Vector3(1.12, 1, 0.70),
    yaw: sample.yaw,
  };
}

function buildRockWalls(group, profile, mats) {
  const count = Math.min(96, Math.floor(profile.length / 8));
  const rockGeo = jitterGeometry(new THREE.BoxGeometry(3.8, 2.7, 1.3), 0.36, 200);
  const rocks = new THREE.InstancedMesh(rockGeo, mats.rock, count);
  rocks.castShadow = rocks.receiveShadow = true;
  let n = 0;
  for (let d = 18; d < profile.length - 10 && n < count; d += 9) {
    const sample = trackSampleAtDistance(profile, d);
    const edge = Math.floor(d / 55) % 2 === 0 ? sample.left : sample.right;
    const outward = edge.clone().sub(sample.center).setY(0).normalize();
    const p = edge.clone().add(outward.clone().multiplyScalar(4.2 + hash01(d * 0.7) * 5.5));
    p.y = terrainHeightAt(profile, p.x, p.z) + 1.05 + hash01(d * 1.7) * 0.95;
    const scale = new THREE.Vector3(0.72 + hash01(d * 2.1) * 1.10, 0.70 + hash01(d * 3.2) * 1.35, 0.75 + hash01(d * 4.3) * 0.80);
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
  const trunks = new THREE.InstancedMesh(trunkGeo, mats.pineTrunk, count);
  const crowns = new THREE.InstancedMesh(crownGeo, mats.pineNeedles, count);
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

function buildReflectiveSigns(group, profile, mats) {
  const count = Math.min(22, Math.floor(profile.length / 55));
  const panelGeo = new THREE.BoxGeometry(0.10, 0.92, 1.72);
  const postGeo = new THREE.BoxGeometry(0.11, 1.55, 0.11);
  const panels = new THREE.InstancedMesh(panelGeo, mats.sign, count);
  const posts = new THREE.InstancedMesh(postGeo, mats.pole, count);
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

function buildMountainPeaks(group, profile, mats) {
  const points = profile.center.concat(profile.left, profile.right);
  const box = new THREE.Box3().setFromPoints(points);
  const center = box.getCenter(new THREE.Vector3());
  const spanX = box.max.x - box.min.x;
  const spanZ = box.max.z - box.min.z;
  const baseRing = Math.max(spanX, spanZ) * 0.68 + 120;

  // --- Ring 1: Mid-range peaks (closer, medium height) ---
  const midCount = Math.min(48, Math.max(16, Math.floor(profile.length / 15)));
  const midPeakGeo = jitterGeometry(new THREE.ConeGeometry(1.0, 1.0, 6), 0.12, 800);
  const midSnowGeo = jitterGeometry(new THREE.ConeGeometry(1.0, 1.0, 6), 0.08, 801);
  const midPeaks = new THREE.InstancedMesh(midPeakGeo, mats.mountainMid, midCount);
  const midSnow  = new THREE.InstancedMesh(midSnowGeo, mats.snowCap, midCount);
  midPeaks.castShadow = midPeaks.receiveShadow = true;
  midSnow.receiveShadow = true;

  for (let i = 0; i < midCount; i++) {
    const a = (i / midCount) * Math.PI * 2;
    const wobble = (hash01(i * 6.31) - 0.5) * 0.52;
    const radius = baseRing * 0.85 + hash01(i * 3.17) * 65;
    const px = center.x + Math.cos(a + wobble) * radius;
    const pz = center.z + Math.sin(a + wobble) * radius;
    const baseY = terrainHeightAt(profile, px, pz);
    const peakH = 14 + hash01(i * 5.3) * 22;
    const peakW = 5.5 + hash01(i * 2.9) * 8.5;
    const peakD = 4.2 + hash01(i * 4.1) * 6.0;
    const p = new THREE.Vector3(px, baseY + peakH * 0.5 - 1.5, pz);
    const scale = new THREE.Vector3(peakW, peakH, peakD);
    const yaw = -a + Math.PI / 2 + (hash01(i * 8.7) - 0.5) * 0.6;
    setInstancedTransform(midPeaks, i, p, yaw, scale);

    // Snow cap: smaller cone at the top
    const snowH = peakH * (0.25 + hash01(i * 7.1) * 0.20);
    const snowW = peakW * (0.55 + hash01(i * 6.2) * 0.20);
    const snowD = peakD * (0.55 + hash01(i * 5.8) * 0.20);
    const snowP = new THREE.Vector3(px, baseY + peakH - snowH * 0.5 - 1.5, pz);
    setInstancedTransform(midSnow, i, snowP, yaw, new THREE.Vector3(snowW, snowH, snowD));
  }
  group.add(midPeaks, midSnow);

  // --- Ring 2: Far background peaks (taller, further away) ---
  const farCount = Math.min(36, Math.max(12, Math.floor(profile.length / 22)));
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
    const peakW = 12 + hash01(i * 8.9) * 18;
    const peakD = 10 + hash01(i * 7.1) * 14;
    const p = new THREE.Vector3(px, baseY + peakH * 0.5, pz);
    const scale = new THREE.Vector3(peakW, peakH, peakD);
    const yaw = -a + (hash01(i * 13.1) - 0.5) * 0.8;
    setInstancedTransform(farPeaks, i, p, yaw, scale);

    // Snow cap
    const snowH = peakH * (0.30 + hash01(i * 14.1) * 0.18);
    const snowW = peakW * (0.50 + hash01(i * 10.2) * 0.18);
    const snowD = peakD * (0.50 + hash01(i * 9.8) * 0.18);
    const snowP = new THREE.Vector3(px, baseY + peakH - snowH * 0.5, pz);
    setInstancedTransform(farSnow, i, snowP, yaw, new THREE.Vector3(snowW, snowH, snowD));
  }
  group.add(farPeaks, farSnow);

  // --- Ring 3: Massive distant silhouette peaks ---
  const silCount = Math.min(20, Math.max(8, Math.floor(profile.length / 35)));
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
    const peakW = 22 + hash01(i * 18.2) * 35;
    const peakD = 18 + hash01(i * 16.4) * 28;
    const p = new THREE.Vector3(px, baseY + peakH * 0.5, pz);
    const scale = new THREE.Vector3(peakW, peakH, peakD);
    setInstancedTransform(silPeaks, i, p, hash01(i * 21.0) * Math.PI, scale);

    const snowH = peakH * (0.22 + hash01(i * 22.1) * 0.15);
    const snowW = peakW * (0.45 + hash01(i * 19.2) * 0.15);
    const snowD = peakD * (0.45 + hash01(i * 18.8) * 0.15);
    const snowP = new THREE.Vector3(px, baseY + peakH - snowH * 0.5, pz);
    setInstancedTransform(silSnow, i, snowP, hash01(i * 21.0) * Math.PI, new THREE.Vector3(snowW, snowH, snowD));
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
