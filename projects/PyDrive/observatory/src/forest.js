import * as THREE from "three";
import { simToThree } from "./coords.js";
import { stylize } from "./stylize.js";

const ASSET_ROOT = "./assets/observatory";

/**
 * Deterministic Eifel forest from world_data.json placements.
 * Presentation only — never feeds physics / collision.
 */
function makeTreeGeometry(species = 0) {
  const trunkH = species === 0 ? 2.6 : species === 1 ? 3.1 : 2.2;
  const canopyH = species === 0 ? 4.2 : species === 1 ? 5.0 : 3.4;
  // Wider canopies + thicker trunks so far-tier trees read as mass, not hairline spikes.
  const canopyR = species === 0 ? 1.85 : species === 1 ? 1.55 : 2.1;
  const trunkRTop = species === 0 ? 0.28 : species === 1 ? 0.24 : 0.32;
  const trunkRBot = species === 0 ? 0.42 : species === 1 ? 0.36 : 0.48;

  const trunk = new THREE.CylinderGeometry(trunkRTop, trunkRBot, trunkH, 6);
  trunk.translate(0, trunkH * 0.5, 0);
  const canopy = new THREE.ConeGeometry(canopyR, canopyH, 7);
  canopy.translate(0, trunkH + canopyH * 0.35, 0);

  // Merge into one non-indexed buffer for a single InstancedMesh material...
  // Two-material trees need two InstancedMeshes; keep one canopy-dominant mesh
  // with a trunk color baked via vertex colors for cheap readability.
  const trunkPos = trunk.attributes.position;
  const canopyPos = canopy.attributes.position;
  const count = trunkPos.count + canopyPos.count;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const trunkColor = new THREE.Color(0x4a3828);
  const canopyColor = species === 1
    ? new THREE.Color(0x2f5a38)
    : species === 2
      ? new THREE.Color(0x3d6a34)
      : new THREE.Color(0x355a32);

  for (let i = 0; i < trunkPos.count; i += 1) {
    positions[i * 3] = trunkPos.getX(i);
    positions[i * 3 + 1] = trunkPos.getY(i);
    positions[i * 3 + 2] = trunkPos.getZ(i);
    colors[i * 3] = trunkColor.r;
    colors[i * 3 + 1] = trunkColor.g;
    colors[i * 3 + 2] = trunkColor.b;
  }
  const base = trunkPos.count;
  for (let i = 0; i < canopyPos.count; i += 1) {
    const o = (base + i) * 3;
    positions[o] = canopyPos.getX(i);
    positions[o + 1] = canopyPos.getY(i);
    positions[o + 2] = canopyPos.getZ(i);
    colors[o] = canopyColor.r;
    colors[o + 1] = canopyColor.g;
    colors[o + 2] = canopyColor.b;
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  trunk.dispose();
  canopy.dispose();
  return geo;
}

function tierBudget(tier, quality = "high") {
  if (tier === "hero") return quality === "low" ? 120 : 279;
  if (tier === "mid") return quality === "low" ? 280 : quality === "medium" ? 700 : 1460;
  return quality === "low" ? 180 : quality === "medium" ? 450 : 941;
}

export class ForestField {
  constructor() {
    this.root = new THREE.Group();
    this.root.name = "forest_field";
    this._records = [];
    this._meshes = [];
    this._dummy = new THREE.Object3D();
    this._tmp = new THREE.Vector3();
    this._quality = "high";
    this.visible = true;
  }

  async load() {
    const data = await fetch(`${ASSET_ROOT}/world/world_data.json`).then((r) => r.json());
    this._records = Array.isArray(data.forest) ? data.forest : [];
    this._rebuild();
    return this._records.length;
  }

  setQuality(level) {
    if (level === this._quality) return;
    this._quality = level;
    this._rebuild();
  }

  setCondition({ night = false, wetness = 0 } = {}) {
    for (const mesh of this._meshes) {
      if (!mesh.material?.color) continue;
      // Vertex colors carry species; tint the material multiply-ish via color.
      mesh.material.color.setHex(night ? 0x889988 : wetness > 0.5 ? 0xc8d4c0 : 0xffffff);
      mesh.material.opacity = night ? 0.92 : 1;
      mesh.material.transparent = night;
    }
  }

  _rebuild() {
    while (this.root.children.length) {
      const child = this.root.children.pop();
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    }
    this._meshes = [];
    if (!this._records.length) return;

    const byKey = new Map();
    for (const rec of this._records) {
      const key = `${rec.tier}|${rec.species}`;
      if (!byKey.has(key)) byKey.set(key, []);
      byKey.get(key).push(rec);
    }

    for (const [key, list] of byKey) {
      const [tier, speciesStr] = key.split("|");
      const species = Number(speciesStr) || 0;
      const budget = tierBudget(tier, this._quality);
      const slice = list.slice(0, Math.min(list.length, budget));
      if (!slice.length) continue;

      const geo = makeTreeGeometry(species);
      const mat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        roughness: 0.92,
        metalness: 0.02,
        flatShading: true,
      });
      // Fewer bands than the car: canopies read better as flat graphic shapes.
      stylize(mat, { bands: 2, softness: 0.14, rimStrength: 0.0 });
      const mesh = new THREE.InstancedMesh(geo, mat, slice.length);
      mesh.castShadow = tier === "hero";
      mesh.receiveShadow = true;
      mesh.frustumCulled = true;
      mesh.userData.tier = tier;
      mesh.userData.records = slice;

      for (let i = 0; i < slice.length; i += 1) {
        const rec = slice[i];
        const [x, y, z] = rec.position;
        simToThree(x, y, z, this._tmp);
        this._dummy.position.copy(this._tmp);
        this._dummy.rotation.set(0, -Number(rec.rotation) || 0, 0);
        const s = Number(rec.scale) || 1;
        // Oversized for chase readability — authored placements sit ~20–60 m
        // off the racing line and vanish at stock 1× inside a 48° FOV.
        // Far/horizon tiers get extra bulk so they read as canopy mass, not pins.
        const tierScale = tier === "hero" ? 2.6 : tier === "mid" ? 2.15 : 1.85;
        this._dummy.scale.setScalar(s * tierScale);
        this._dummy.updateMatrix();
        mesh.setMatrixAt(i, this._dummy.matrix);
      }
      mesh.instanceMatrix.needsUpdate = true;
      mesh.computeBoundingSphere();
      mesh.frustumCulled = tier === "horizon";
      this.root.add(mesh);
      this._meshes.push(mesh);
    }
  }

  /**
   * Terrain/forest density is quality-gated at build time; keep instances on
   * so trees remain readable within ~80 m of the car in chase framing.
   */
  updateLod(_subject, _camera) {
    for (const mesh of this._meshes) mesh.visible = this.visible;
  }
}
