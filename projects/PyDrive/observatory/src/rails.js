import * as THREE from "three";

/**
 * Authored guardrail ribbons in world_visual.glb are nearly coplanar with the
 * road (~0.5 m tall). Erect presentation-only posts + beams so rails read
 * within chase distance. Never feeds physics.
 *
 * Sampling walks a medial / centerline path (voxel collapse + NN chain), not
 * every-Nth vertex of the whole ribbon — cross-section verts otherwise plant
 * posts as hairline spikes across the verge.
 */
export class RailPosts {
  constructor() {
    this.root = new THREE.Group();
    this.root.name = "rail_posts";
    this._meshes = [];
  }

  buildFromVisual(visualRoot) {
    this.dispose();
    if (!visualRoot) return 0;

    const targets = [];
    visualRoot.traverse((obj) => {
      if (!obj.isMesh) return;
      const name = obj.name || "";
      if (/guardrail/i.test(name)) targets.push({ mesh: obj, kind: "rail" });
      else if (/safety_fence/i.test(name)) targets.push({ mesh: obj, kind: "fence" });
    });

    let placed = 0;
    for (const { mesh, kind } of targets) {
      placed += this._erect(mesh, kind);
    }
    return placed;
  }

  _erect(source, kind) {
    source.updateWorldMatrix(true, false);
    const pos = source.geometry?.attributes?.position;
    if (!pos || pos.count < 8) return 0;

    const worldPoints = [];
    const v = new THREE.Vector3();
    // Dense enough to survive voxel collapse; still O(n) over authored verts.
    const stride = Math.max(1, Math.floor(pos.count / 4000));
    for (let i = 0; i < pos.count; i += stride) {
      v.fromBufferAttribute(pos, i).applyMatrix4(source.matrixWorld);
      worldPoints.push(v.clone());
    }
    if (worldPoints.length < 2) return 0;

    const centerline = extractMedialPath(worldPoints);
    if (centerline.length < 2) return 0;

    // Greedy spacing along the medial path — keep posts ~2.8 m apart.
    const spaced = [centerline[0]];
    const minGap = kind === "fence" ? 3.6 : 2.8;
    for (let i = 1; i < centerline.length; i += 1) {
      if (spaced[spaced.length - 1].distanceTo(centerline[i]) >= minGap) {
        spaced.push(centerline[i]);
      }
    }
    if (spaced.length < 2) return 0;

    const postH = kind === "fence" ? 1.65 : 0.92;
    // ~0.18–0.25 m posts so they read at chase distance (legacy ~0.13).
    const postW = kind === "fence" ? 0.14 : 0.22;
    const postD = kind === "fence" ? 0.12 : 0.18;
    const postGeo = new THREE.BoxGeometry(postW, postH, postD);
    postGeo.translate(0, postH * 0.5, 0);
    // Unit-X beam; each instance is scaled to the actual post-to-post span.
    const beamThick = kind === "fence" ? 0.05 : 0.1;
    const beamDepth = kind === "fence" ? 0.06 : 0.08;
    const beamGeo = new THREE.BoxGeometry(1, beamThick, beamDepth);
    beamGeo.translate(0, kind === "fence" ? postH * 0.82 : postH * 0.78, 0);

    const metal = new THREE.MeshStandardMaterial({
      color: kind === "fence" ? 0x7a8680 : 0xd0d8d4,
      metalness: kind === "fence" ? 0.55 : 0.82,
      roughness: kind === "fence" ? 0.45 : 0.26,
    });

    const posts = new THREE.InstancedMesh(postGeo, metal, spaced.length);
    const beams = new THREE.InstancedMesh(beamGeo, metal.clone(), Math.max(1, spaced.length - 1));
    posts.castShadow = true;
    beams.castShadow = true;
    posts.frustumCulled = false;
    beams.frustumCulled = false;

    const dummy = new THREE.Object3D();
    let beamCount = 0;
    for (let i = 0; i < spaced.length; i += 1) {
      const p = spaced[i];
      dummy.position.copy(p);
      dummy.scale.set(1, 1, 1);
      if (i + 1 < spaced.length) {
        const q = spaced[i + 1];
        dummy.rotation.y = Math.atan2(-(q.z - p.z), q.x - p.x);
      } else if (i > 0) {
        const prev = spaced[i - 1];
        dummy.rotation.y = Math.atan2(-(p.z - prev.z), p.x - prev.x);
      } else {
        dummy.rotation.set(0, 0, 0);
      }
      dummy.updateMatrix();
      posts.setMatrixAt(i, dummy.matrix);

      if (i + 1 < spaced.length) {
        const q = spaced[i + 1];
        const span = p.distanceTo(q);
        // Skip pathological jumps (ribbon fold / bad chain) — no kilometre beams.
        if (span > 0.4 && span < minGap * 2.4) {
          dummy.position.lerpVectors(p, q, 0.5);
          dummy.position.y = (p.y + q.y) * 0.5;
          dummy.rotation.y = Math.atan2(-(q.z - p.z), q.x - p.x);
          dummy.scale.set(span * 0.92, 1, 1);
          dummy.updateMatrix();
          beams.setMatrixAt(beamCount, dummy.matrix);
          beamCount += 1;
        }
      }
    }
    posts.instanceMatrix.needsUpdate = true;
    beams.instanceMatrix.needsUpdate = true;
    beams.count = Math.max(1, beamCount);

    this.root.add(posts);
    this.root.add(beams);
    this._meshes.push(posts, beams);
    return spaced.length;
  }

  setCondition({ night = false } = {}) {
    for (const mesh of this._meshes) {
      if (!mesh.material?.color) continue;
      mesh.material.color.setHex(night ? 0x5a6460 : 0xd0d8d4);
      mesh.material.metalness = night ? 0.55 : 0.82;
    }
  }

  dispose() {
    while (this.root.children.length) {
      const child = this.root.children.pop();
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    }
    this._meshes = [];
  }
}

/**
 * Collapse ribbon cross-section verts into a walkable centerline.
 * 1) Voxelize (~0.55 m) so left/right/top ribbon edges share a cell.
 * 2) Nearest-neighbour chain from an extreme voxel.
 */
function extractMedialPath(worldPoints) {
  const voxel = 0.55;
  const cells = new Map();
  for (const p of worldPoints) {
    const ix = Math.round(p.x / voxel);
    const iy = Math.round(p.y / voxel);
    const iz = Math.round(p.z / voxel);
    const key = `${ix}|${iy}|${iz}`;
    const existing = cells.get(key);
    if (!existing) {
      cells.set(key, { x: p.x, y: p.y, z: p.z, n: 1 });
    } else {
      existing.x += p.x;
      existing.y += p.y;
      existing.z += p.z;
      existing.n += 1;
    }
  }
  const pts = [];
  for (const c of cells.values()) {
    pts.push(new THREE.Vector3(c.x / c.n, c.y / c.n, c.z / c.n));
  }
  if (pts.length < 2) return pts;

  // Start at the point farthest from the cloud centroid (an end-ish extreme).
  const centroid = new THREE.Vector3();
  for (const p of pts) centroid.add(p);
  centroid.multiplyScalar(1 / pts.length);
  let startIdx = 0;
  let best = -1;
  for (let i = 0; i < pts.length; i += 1) {
    const d = pts[i].distanceToSquared(centroid);
    if (d > best) {
      best = d;
      startIdx = i;
    }
  }

  const used = new Uint8Array(pts.length);
  const path = [];
  let current = startIdx;
  const searchR = voxel * 4.5;
  const searchR2 = searchR * searchR;

  while (current >= 0) {
    used[current] = 1;
    path.push(pts[current]);
    const cur = pts[current];
    let next = -1;
    let nextDist = Infinity;
    // Prefer the nearest unused neighbour inside the search radius so the
    // walk follows the ribbon instead of leaping to a parallel verge.
    for (let i = 0; i < pts.length; i += 1) {
      if (used[i]) continue;
      const d2 = cur.distanceToSquared(pts[i]);
      if (d2 > searchR2 || d2 >= nextDist) continue;
      nextDist = d2;
      next = i;
    }
    if (next < 0) {
      // Gap — jump to nearest unused within a wider radius, else stop.
      let jump = -1;
      let jumpD = (voxel * 12) ** 2;
      for (let i = 0; i < pts.length; i += 1) {
        if (used[i]) continue;
        const d2 = cur.distanceToSquared(pts[i]);
        if (d2 < jumpD) {
          jumpD = d2;
          jump = i;
        }
      }
      current = jump;
    } else {
      current = next;
    }
    // Safety: one pass per voxel.
    if (path.length >= pts.length) break;
  }

  return path;
}
