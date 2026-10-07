import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { mergeGeometries } from "three/addons/utils/BufferGeometryUtils.js";

import { RenderAttitudeFilter, RenderHeightFilter, vehiclePoseQuaternion } from "./vehicle-pose.js";
import { sampleTrackSurface } from "./track-surface.js";

const ASSET_ROOT = `${import.meta.env.BASE_URL}assets/observatory`;
const TRACK_SAMPLE_COUNT = 6944;

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} while loading ${url}`);
  return response.json();
}

function loadGltf(loader, url) {
  return new Promise((resolve, reject) => loader.load(url, resolve, undefined, reject));
}

export function simToWorld(point, target = new THREE.Vector3()) {
  return target.set(Number(point?.[0]) || 0, Number(point?.[2]) || 0, -(Number(point?.[1]) || 0));
}

const shadowPlaneBasis = new THREE.Quaternion().setFromAxisAngle(
  new THREE.Vector3(1, 0, 0),
  -Math.PI / 2,
);

function vehicleKey(identity) {
  const key = String(identity?.car || identity || "").trim();
  if (!key || key === "UNRESOLVED" || !/^[a-z0-9_-]+$/.test(key)) {
    throw new Error("Cannot load a vehicle without a safe server-resolved car identity");
  }
  return key;
}

function seededNoise(x, y = 0) {
  const value = Math.sin(x * 127.1 + y * 311.7) * 43758.5453123;
  return value - Math.floor(value);
}

function surfaceTexture(renderer, type) {
  const size = type === "road" ? 1024 : 512;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = 512;
  const context = canvas.getContext("2d", { willReadFrequently: false });
  const image = context.createImageData(size, 512);
  const pixels = image.data;
  const road = type === "road";
  for (let y = 0; y < 512; y += 1) {
    const across = Math.abs(y / 511 - 0.5) * 2;
    for (let x = 0; x < size; x += 1) {
      const index = (y * size + x) * 4;
      const grain = seededNoise(x, y);
      const coarse = seededNoise(Math.floor(x / 7), Math.floor(y / 7));
      if (road) {
        const rubber = Math.exp(-Math.pow(across / 0.29, 2));
        const aggregate = 38 + grain * 16 + coarse * 6 - rubber * 9;
        pixels[index] = aggregate;
        pixels[index + 1] = aggregate + 2;
        pixels[index + 2] = aggregate + 3;
      } else {
        const gravel = 74 + grain * 36 + coarse * 15;
        pixels[index] = gravel * 0.88;
        pixels[index + 1] = gravel * 0.9;
        pixels[index + 2] = gravel * 0.82;
      }
      pixels[index + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  if (road) {
    context.globalAlpha = 0.34;
    for (let x = 30; x < size; x += 137) {
      context.fillStyle = x % 274 ? "#25282a" : "#4b4c49";
      context.fillRect(x, 20 + (x % 71), 2, 398 - (x % 89));
    }
    context.globalAlpha = 0.72;
    context.fillStyle = "#d7d5c7";
    context.fillRect(0, 9, size, 5);
    context.fillRect(0, 498, size, 5);
    context.globalAlpha = 0.12;
    context.fillStyle = "#07090a";
    context.fillRect(0, 194, size, 124);
    context.globalAlpha = 1;
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.name = `observatory_procedural_${type}_surface`;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  return texture;
}

function radialShadowTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(128, 64, 6, 128, 64, 120);
  gradient.addColorStop(0, "rgba(0,0,0,.62)");
  gradient.addColorStop(0.45, "rgba(0,0,0,.31)");
  gradient.addColorStop(1, "rgba(0,0,0,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 256, 128);
  return new THREE.CanvasTexture(canvas);
}

function trackUvFallback(geometry, centerline) {
  if (geometry.getAttribute("uv") || geometry.getAttribute("position").count !== centerline.length * 2) return;
  const uv = new Float32Array(centerline.length * 4);
  let arc = 0;
  for (let index = 0; index < centerline.length; index += 1) {
    if (index > 0) {
      const previous = centerline[index - 1];
      const point = centerline[index];
      arc += Math.hypot(point[0] - previous[0], point[1] - previous[1], point[2] - previous[2]);
    }
    uv[index * 4] = arc / 18;
    uv[index * 4 + 1] = 0;
    uv[index * 4 + 2] = arc / 18;
    uv[index * 4 + 3] = 1;
  }
  geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
}

function terrainTexture(renderer) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 512;
  const context = canvas.getContext("2d");
  const image = context.createImageData(512, 512);
  for (let y = 0; y < 512; y += 1) {
    for (let x = 0; x < 512; x += 1) {
      const index = (y * 512 + x) * 4;
      const fine = seededNoise(x, y);
      const broad = seededNoise(Math.floor(x / 11), Math.floor(y / 11));
      const value = 176 + fine * 36 + broad * 18;
      image.data[index] = value * 0.92;
      image.data[index + 1] = value;
      image.data[index + 2] = value * 0.87;
      image.data[index + 3] = 255;
    }
  }
  context.putImageData(image, 0, 0);
  const texture = new THREE.CanvasTexture(canvas);
  texture.name = "observatory_procedural_terrain_detail";
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  return texture;
}

function terrainMaterial(map) {
  return new THREE.MeshStandardMaterial({
    name: "observatory_dgm_terrain_material",
    map,
    vertexColors: true,
    roughness: 1,
    metalness: 0,
    flatShading: false,
  });
}

function colorTerrain(mesh, fine = false, map = null) {
  const position = mesh.geometry.getAttribute("position");
  const color = new Float32Array(position.count * 3);
  const uv = new Float32Array(position.count * 2);
  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const y = position.getY(index);
    const z = position.getZ(index);
    const variation = seededNoise(Math.floor(x * 0.07), Math.floor(y * 0.07));
    const altitude = THREE.MathUtils.clamp((z - 430) / 210, 0, 1);
    const r = 0.105 + variation * 0.055 + altitude * 0.035;
    const g = 0.205 + variation * 0.075 - altitude * 0.025;
    const b = 0.115 + variation * 0.035 + altitude * 0.025;
    color[index * 3] = fine ? r * 1.08 : r;
    color[index * 3 + 1] = fine ? g * 1.06 : g;
    color[index * 3 + 2] = b;
    uv[index * 2] = x / 12;
    uv[index * 2 + 1] = y / 12;
  }
  mesh.geometry.setAttribute("color", new THREE.BufferAttribute(color, 3));
  mesh.geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  mesh.material = terrainMaterial(map);
}

function mergedCrown(species) {
  if (species === 0) {
    const lower = new THREE.ConeGeometry(1.55, 3.7, 9).translate(0, 1.5, 0);
    const middle = new THREE.ConeGeometry(1.18, 3.1, 9).translate(0, 3.0, 0);
    const top = new THREE.ConeGeometry(0.78, 2.4, 9).translate(0, 4.2, 0);
    return mergeGeometries([lower, middle, top], false);
  }
  if (species === 1) {
    const a = new THREE.DodecahedronGeometry(1.35, 1).translate(-0.42, 2.4, 0.12);
    const b = new THREE.DodecahedronGeometry(1.5, 1).translate(0.45, 2.75, -0.12);
    const c = new THREE.DodecahedronGeometry(1.16, 1).translate(0, 3.85, 0.2);
    return mergeGeometries([a, b, c], false);
  }
  const lower = new THREE.ConeGeometry(1.32, 3.15, 10).translate(0, 1.5, 0);
  const upper = new THREE.ConeGeometry(0.88, 2.8, 10).translate(0, 3.25, 0);
  return mergeGeometries([lower, upper], false);
}

function signTexture(label) {
  const canvas = document.createElement("canvas");
  canvas.width = 1024;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  context.fillStyle = "#111719";
  context.fillRect(0, 0, 1024, 256);
  context.strokeStyle = "#f1efe5";
  context.lineWidth = 16;
  context.strokeRect(12, 12, 1000, 232);
  context.fillStyle = "#f5f2e8";
  context.font = "700 74px Arial, sans-serif";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(String(label).toUpperCase().slice(0, 24), 512, 128);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

export class ObservatoryWorld {
  constructor(scene, renderer) {
    this.scene = scene;
    this.renderer = renderer;
    this.loader = new GLTFLoader();
    this.trackPoints = [];
    this.trackWidth = [];
    this.vehicleRoot = new THREE.Group();
    this.vehicleRoot.name = "live_vehicle_root";
    this.scene.add(this.vehicleRoot);
    this.renderAttitude = new RenderAttitudeFilter();
    this.renderHeight = new RenderHeightFilter();
    this.previousRenderInput = new THREE.Vector3();
    this.renderPosition = new THREE.Vector3();
    this.renderPoseInitialized = false;
    this.previousAirborne = false;
    this.vehicleAsset = null;
    this.vehicleKey = "";
    this.pendingVehicleKey = "";
    this.terrainCoarseGroup = new THREE.Group();
    this.terrainCoarseGroup.name = "visual_dgm_terrain_lod1";
    this.terrainFineGroup = new THREE.Group();
    this.terrainFineGroup.name = "visual_dgm_terrain_lod0";
    this.forestMeshes = [];
    this.forestDensity = 1;
    this.terrainTiles = new Map();
    this.terrainFineDistance = 900;
    this.terrainCoarseDistance = 4000;
    this.weather = "dry";
    this.engineeringVisible = false;
    this.brainGroup = new THREE.Group();
    this.brainGroup.name = "brain_xray_truth_overlay";
    this.scene.add(this.brainGroup);
    this.predictionLine = this.makeLine(0x71f6ff, 1, 0.95);
    this.paceLine = this.makeLine(0xffb24b, 1, 0.72, true);
    this.brainGroup.add(this.predictionLine, this.paceLine);
    this.rayLines = [];
    this.vehicleVisualLength = 0;
    this.footprintHasData = false;
    this.footprintLine = new THREE.LineLoop(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0xffbb4d, transparent: true, opacity: 0.94, depthTest: false }),
    );
    this.footprintLine.name = "authoritative_collision_footprint";
    this.footprintLine.renderOrder = 20;
    this.scene.add(this.footprintLine);
    this.contactShadow = new THREE.Mesh(
      new THREE.PlaneGeometry(5.4, 2.35),
      new THREE.MeshBasicMaterial({ map: radialShadowTexture(), transparent: true, depthWrite: false, opacity: 0.72 }),
    );
    this.contactShadow.rotation.x = -Math.PI / 2;
    this.contactShadow.renderOrder = 1;
    this.scene.add(this.contactShadow);
    this.brainWasVisible = false;
    this.headlights = [];
    this.vehicleFill = null;
    this.rearLights = [];
    this.frontLightMaterials = [];
    this.terrainDetailTexture = terrainTexture(renderer);
    this.trackMaterial = this.makeTrackMaterial();
    this.shoulderMaterial = this.makeShoulderMaterial();
  }

  makeTrackMaterial() {
    const map = surfaceTexture(this.renderer, "road");
    const bump = map.clone();
    bump.colorSpace = THREE.NoColorSpace;
    bump.needsUpdate = true;
    const material = new THREE.MeshPhysicalMaterial({
      name: "observatory_truth_track_visual_skin",
      map,
      bumpMap: bump,
      bumpScale: 0.035,
      color: 0xffffff,
      roughness: 0.91,
      metalness: 0.02,
      clearcoat: 0.08,
      clearcoatRoughness: 0.88,
      side: THREE.DoubleSide,
      // The authoritative road is coplanar with the visual DGM verge at the
      // shoulder seam. Pull it toward the camera in the depth buffer so the
      // terrain skin can never z-fight up through the surface the car drives on.
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
    material.userData.truthBoundary = "material-only visual skin over exact runtime track vertices";
    return material;
  }

  makeShoulderMaterial() {
    return new THREE.MeshStandardMaterial({
      name: "observatory_truth_shoulder_visual_skin",
      map: surfaceTexture(this.renderer, "shoulder"),
      color: 0x9a9a8c,
      roughness: 1,
      metalness: 0,
      side: THREE.DoubleSide,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
  }

  async load(onProgress = () => {}, initialVehicle = null) {
    onProgress(0.06, "Reading versioned world manifest");
    const manifestUrl = `${ASSET_ROOT}/world/world.manifest.json`;
    const manifest = await fetchJson(manifestUrl);
    if (manifest.schema !== "supra-observatory-world-v1" || manifest.track?.sample_count !== TRACK_SAMPLE_COUNT) {
      throw new Error("World asset contract is invalid or does not contain all 6,944 track samples");
    }
    onProgress(0.18, "Loading exact road and visual-world layers");
    const [truth, visual, data] = await Promise.all([
      loadGltf(this.loader, `${ASSET_ROOT}/world/${manifest.files.truth.path}`),
      loadGltf(this.loader, `${ASSET_ROOT}/world/${manifest.files.visual.path}`),
      fetchJson(`${ASSET_ROOT}/world/${manifest.files.data.path}`),
    ]);
    this.trackPoints = data.track.centerline;
    this.trackWidth = data.track.width_m;
    truth.scene.name = "nordschleife_truth_layer";
    visual.scene.name = "nordschleife_visual_layer";
    truth.scene.rotation.x = -Math.PI / 2;
    visual.scene.rotation.x = -Math.PI / 2;
    truth.scene.traverse((object) => {
      if (!object.isMesh) return;
      object.receiveShadow = true;
      object.castShadow = false;
      object.frustumCulled = false;
      // Draw the authoritative surface after the terrain so it wins the depth
      // tie at the seam (paired with the road/terrain polygon offsets).
      object.renderOrder = 1;
      if (object.name === "truth_road") {
        trackUvFallback(object.geometry, this.trackPoints);
        object.material = this.trackMaterial;
      } else {
        trackUvFallback(object.geometry, this.trackPoints);
        object.material = this.shoulderMaterial;
      }
    });
    visual.scene.traverse((object) => {
      if (!object.isMesh) return;
      object.frustumCulled = false;
      object.receiveShadow = true;
      object.castShadow = object.name.includes("landmark") || object.name.includes("guardrail");
      if (object.name.includes("visual_dgm_verge")) colorTerrain(object, true, this.terrainDetailTexture);
      else if (object.name.includes("guardrail")) object.material = new THREE.MeshStandardMaterial({ color: 0xb6b8b3, metalness: 0.78, roughness: 0.31 });
      else if (object.name.includes("fence")) object.visible = false;
      else if (object.name.includes("curb")) {
        // Curbs ride the road edge just above the surface — bias them forward
        // like the road so the verge cannot z-fight through them either.
        const accent = object.name.includes("accent");
        object.material = new THREE.MeshStandardMaterial({
          color: accent ? 0xd34a2c : 0xe4dfce,
          roughness: accent ? 0.74 : 0.78,
          polygonOffset: true,
          polygonOffsetFactor: -2,
          polygonOffsetUnits: -2,
        });
        object.renderOrder = 1;
      }
    });
    this.scene.add(visual.scene, truth.scene);
    this.addTerrainUnderlay(data);
    onProgress(0.48, "Loading offline DGM terrain tiles");
    await this.loadTerrainLevel(manifest, 1, this.terrainCoarseGroup);
    this.scene.add(this.terrainCoarseGroup);
    onProgress(0.61, "Planting multi-species deterministic forest");
    this.addForest(data.forest || []);
    this.addTracksideFurniture(data);
    onProgress(0.76, "Preparing the selected endurance prototype");
    if (initialVehicle) await this.ensureVehicle(initialVehicle);
    this.loadTerrainLevel(manifest, 0, this.terrainFineGroup).then(() => {
      this.scene.add(this.terrainFineGroup);
    }).catch((error) => console.warn("Fine DGM terrain LOD unavailable; retaining coarse tiles", error));
    onProgress(1, `World ${manifest.track.hash.slice(0, 12)} verified`);
    return { manifest, data };
  }

  addForest(entries) {
    if (!entries.length) return;
    const speciesEntries = [[], [], []];
    entries.forEach((entry, index) => speciesEntries[index % 3].push(entry));
    const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x3c2b20, roughness: 1 });
    const crownMaterials = [0x173924, 0x284d27, 0x1d432f].map((color) => new THREE.MeshStandardMaterial({ color, roughness: 0.96, vertexColors: false }));
    const matrix = new THREE.Matrix4();
    const quaternion = new THREE.Quaternion();
    const position = new THREE.Vector3();
    const scale = new THREE.Vector3();
    speciesEntries.forEach((records, species) => {
      const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.15, 0.25, 2.8, 7), trunkMaterial, records.length);
      const crown = new THREE.InstancedMesh(mergedCrown(species), crownMaterials[species], records.length);
      records.forEach((tree, index) => {
        simToWorld(tree.position, position);
        const size = Number(tree.scale) || 1;
        quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Number(tree.rotation) || 0);
        scale.set(size, size, size);
        position.y += 1.4 * size;
        matrix.compose(position, quaternion, scale);
        trunk.setMatrixAt(index, matrix);
        position.y += 0.7 * size;
        matrix.compose(position, quaternion, scale);
        crown.setMatrixAt(index, matrix);
      });
      trunk.name = `visual_forest_trunks_${species}`;
      crown.name = `visual_forest_crowns_${species}`;
      trunk.receiveShadow = true;
      crown.receiveShadow = true;
      crown.castShadow = species !== 1;
      this.forestMeshes.push({ mesh: trunk, capacity: records.length }, { mesh: crown, capacity: records.length });
      this.scene.add(trunk, crown);
    });
    const underbrushRecords = entries.filter((_, index) => index % 2 === 0);
    const underbrush = new THREE.InstancedMesh(
      new THREE.ConeGeometry(0.35, 0.9, 5),
      new THREE.MeshStandardMaterial({ color: 0x315a2b, roughness: 1 }),
      underbrushRecords.length,
    );
    underbrushRecords.forEach((tree, index) => {
      simToWorld(tree.position, position);
      position.x += (seededNoise(index, 3) - 0.5) * 3.5;
      position.z += (seededNoise(index, 9) - 0.5) * 3.5;
      position.y += 0.4;
      const size = 0.55 + seededNoise(index, 11) * 0.8;
      scale.set(size, size, size);
      quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), seededNoise(index, 21) * Math.PI * 2);
      matrix.compose(position, quaternion, scale);
      underbrush.setMatrixAt(index, matrix);
    });
    underbrush.name = "visual_forest_underbrush";
    this.forestMeshes.push({ mesh: underbrush, capacity: underbrushRecords.length });
    this.scene.add(underbrush);
    this.setForestDensity(this.forestDensity);
  }

  addTerrainUnderlay(data) {
    const centerline = data.track?.centerline || [];
    if (centerline.length < 3) return;
    const stride = 8;
    const samples = [];
    for (let index = 0; index < centerline.length; index += stride) samples.push(index);
    const positions = new Float32Array(samples.length * 6);
    const colors = new Float32Array(samples.length * 6);
    samples.forEach((sampleIndex, ring) => {
      const point = centerline[sampleIndex];
      const next = centerline[(sampleIndex + stride) % centerline.length];
      const tx = next[0] - point[0];
      const ty = next[1] - point[1];
      const length = Math.max(0.001, Math.hypot(tx, ty));
      const nx = -ty / length;
      const ny = tx / length;
      for (let side = 0; side < 2; side += 1) {
        const sign = side === 0 ? -1 : 1;
        const offset = ring * 6 + side * 3;
        positions[offset] = point[0] + nx * 45 * sign;
        positions[offset + 1] = point[2] - 4;
        positions[offset + 2] = -(point[1] + ny * 45 * sign);
        const variation = seededNoise(ring, side) * 0.035;
        colors[offset] = 0.11 + variation;
        colors[offset + 1] = 0.20 + variation;
        colors[offset + 2] = 0.105 + variation * 0.55;
      }
    });
    const indices = [];
    for (let ring = 0; ring < samples.length; ring += 1) {
      const next = (ring + 1) % samples.length;
      indices.push(ring * 2, next * 2, ring * 2 + 1, ring * 2 + 1, next * 2, next * 2 + 1);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    const underlay = new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 1, side: THREE.DoubleSide }),
    );
    underlay.name = "visual_terrain_gap_underlay";
    underlay.receiveShadow = true;
    underlay.frustumCulled = false;
    underlay.userData.collisionAuthority = false;
    this.scene.add(underlay);
  }

  addTracksideFurniture(data) {
    if (!this.trackPoints.length) return;
    const records = [];
    for (let index = 0; index < this.trackPoints.length; index += 84) {
      const point = this.trackPoints[index];
      const next = this.trackPoints[(index + 3) % this.trackPoints.length];
      const tx = next[0] - point[0];
      const ty = next[1] - point[1];
      const length = Math.max(0.001, Math.hypot(tx, ty));
      const nx = -ty / length;
      const ny = tx / length;
      const offset = (Number(this.trackWidth[index]) || 10) * 0.5 + 3.15;
      for (const sign of [-1, 1]) records.push({
        point: [point[0] + nx * offset * sign, point[1] + ny * offset * sign, point[2]],
        yaw: Math.atan2(ty, tx),
      });
    }
    const posts = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.11, 0.82, 0.11),
      new THREE.MeshStandardMaterial({ color: 0xe3e0d4, roughness: 0.75 }),
      records.length,
    );
    const matrix = new THREE.Matrix4();
    const position = new THREE.Vector3();
    const quaternion = new THREE.Quaternion();
    const scale = new THREE.Vector3(1, 1, 1);
    records.forEach((record, index) => {
      simToWorld(record.point, position);
      position.y += 0.41;
      quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), record.yaw);
      matrix.compose(position, quaternion, scale);
      posts.setMatrixAt(index, matrix);
    });
    posts.name = "visual_trackside_reflector_posts";
    posts.receiveShadow = true;
    posts.frustumCulled = false;
    this.scene.add(posts);
    this.addSafetyFencing();

    (data.landmarks || []).forEach((landmark) => {
      const group = new THREE.Group();
      const board = new THREE.Mesh(
        new THREE.PlaneGeometry(3.6, 0.9),
        new THREE.MeshBasicMaterial({ map: signTexture(landmark.name), side: THREE.DoubleSide, toneMapped: false }),
      );
      board.position.y = 2.4;
      const poleMaterial = new THREE.MeshStandardMaterial({ color: 0x777b78, metalness: 0.72, roughness: 0.4 });
      for (const x of [-1.45, 1.45]) {
        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 2.2, 8), poleMaterial);
        pole.position.set(x, 1.1, 0);
        group.add(pole);
      }
      group.add(board);
      simToWorld(landmark.position, group.position);
      const index = Number(landmark.sample_index) || 0;
      const point = this.trackPoints[index];
      const next = this.trackPoints[(index + 4) % this.trackPoints.length];
      group.rotation.y = Math.atan2(-(next[1] - point[1]), next[0] - point[0]) + Math.PI / 2;
      group.name = `visual_section_board_${index}`;
      this.scene.add(group);
    });
  }

  addSafetyFencing() {
    const stride = 12;
    const wirePositions = [];
    const postRecords = [];
    for (let index = 0; index < this.trackPoints.length; index += stride) {
      const nextIndex = (index + stride) % this.trackPoints.length;
      const point = this.trackPoints[index];
      const next = this.trackPoints[nextIndex];
      const tx = next[0] - point[0];
      const ty = next[1] - point[1];
      const length = Math.max(0.001, Math.hypot(tx, ty));
      const nx = -ty / length;
      const ny = tx / length;
      for (const sign of [-1, 1]) {
        const width = (Number(this.trackWidth[index]) || 10) * 0.5 + 8.5;
        const nextWidth = (Number(this.trackWidth[nextIndex]) || 10) * 0.5 + 8.5;
        const a = simToWorld([point[0] + nx * width * sign, point[1] + ny * width * sign, point[2]]);
        const b = simToWorld([next[0] + nx * nextWidth * sign, next[1] + ny * nextWidth * sign, next[2]]);
        for (const height of [0.48, 1.05, 1.58]) {
          wirePositions.push(a.x, a.y + height, a.z, b.x, b.y + height, b.z);
        }
        if (index % (stride * 3) === 0) postRecords.push(a);
      }
    }
    const wireGeometry = new THREE.BufferGeometry();
    wireGeometry.setAttribute("position", new THREE.Float32BufferAttribute(wirePositions, 3));
    const wires = new THREE.LineSegments(
      wireGeometry,
      new THREE.LineBasicMaterial({ color: 0x73817d, transparent: true, opacity: 0.38, depthWrite: false }),
    );
    wires.name = "visual_safety_fence_wires";
    wires.frustumCulled = false;
    this.scene.add(wires);
    const posts = new THREE.InstancedMesh(
      new THREE.BoxGeometry(0.055, 1.65, 0.055),
      new THREE.MeshStandardMaterial({ color: 0x68716e, metalness: 0.62, roughness: 0.44 }),
      postRecords.length,
    );
    const matrix = new THREE.Matrix4();
    postRecords.forEach((position, index) => {
      matrix.makeTranslation(position.x, position.y + 0.825, position.z);
      posts.setMatrixAt(index, matrix);
    });
    posts.name = "visual_safety_fence_posts";
    posts.frustumCulled = false;
    this.scene.add(posts);
  }

  async loadTerrainLevel(manifest, level, group) {
    const contract = manifest.terrain?.levels?.find((entry) => Number(entry.level) === level);
    if (!contract || !Array.isArray(contract.files) || contract.files.length === 0) {
      throw new Error(`DGM terrain LOD ${level} is missing from the world manifest`);
    }
    if (level === 0) this.terrainFineDistance = Number(contract.display_distance_m) || 900;
    if (level === 1) this.terrainCoarseDistance = Number(contract.display_distance_m) || 4000;
    const tiles = await Promise.all(contract.files.map((record) => loadGltf(this.loader, `${ASSET_ROOT}/world/${record.path}`)));
    tiles.forEach((tile, index) => {
      const record = contract.files[index];
      tile.scene.rotation.x = -Math.PI / 2;
      tile.scene.traverse((object) => {
        if (!object.isMesh) return;
        object.receiveShadow = true;
        object.castShadow = false;
        object.frustumCulled = false;
        colorTerrain(object, level === 0, this.terrainDetailTexture);
      });
      const key = record.tile.join("_");
      const bounds = record.sim_bounds_xy_m || [0, 0, 0, 0];
      const entry = this.terrainTiles.get(key) || {
        center: new THREE.Vector2((bounds[0] + bounds[2]) * 0.5, (bounds[1] + bounds[3]) * 0.5),
        coarse: null,
        fine: null,
      };
      entry[level === 0 ? "fine" : "coarse"] = tile.scene;
      this.terrainTiles.set(key, entry);
      group.add(tile.scene);
    });
  }

  updateTerrainLod(cameraPosition) {
    const simX = cameraPosition.x;
    const simY = -cameraPosition.z;
    this.terrainTiles.forEach((tile) => {
      const distance = Math.hypot(simX - tile.center.x, simY - tile.center.y);
      const useFine = Boolean(tile.fine) && distance <= this.terrainFineDistance;
      if (tile.fine) tile.fine.visible = useFine;
      if (tile.coarse) tile.coarse.visible = !useFine;
    });
  }

  setForestDensity(ratio) {
    this.forestDensity = THREE.MathUtils.clamp(ratio, 0, 1);
    this.forestMeshes.forEach(({ mesh, capacity }) => { mesh.count = Math.floor(capacity * this.forestDensity); });
  }

  async ensureVehicle(identity) {
    const key = vehicleKey(identity);
    if (key === this.vehicleKey || key === this.pendingVehicleKey) return;
    this.pendingVehicleKey = key;
    const manifest = await fetchJson(`${ASSET_ROOT}/vehicles/${key}/asset_manifest.json`);
    if (manifest.vehicle_id !== key) throw new Error(`Vehicle package identity mismatch: requested ${key}, received ${manifest.vehicle_id || "unknown"}`);
    const gltf = await loadGltf(this.loader, `${ASSET_ROOT}/vehicles/${key}/${manifest.entry_glb}`);
    if (this.pendingVehicleKey !== key) return;
    this.vehicleRoot.clear();
    const basis = new THREE.Group();
    basis.name = `${key}_z_up_basis`;
    basis.rotation.x = -Math.PI / 2;
    this.enhanceVehicle(gltf.scene, key, manifest);
    basis.add(gltf.scene);
    this.vehicleRoot.add(basis);
    this.addVehicleLights();
    this.vehicleAsset = gltf.scene;
    this.vehicleVisualLength = Number(manifest.dimensions_m?.length) || 0;
    this.vehicleKey = key;
    this.pendingVehicleKey = "";
    this.setWeather(this.weather);
  }

  enhanceVehicle(asset, key, manifest) {
    this.rearLights = [];
    this.frontLightMaterials = [];
    asset.traverse((object) => {
      if (!object.isMesh) return;
      const source = object.material;
      const material = new THREE.MeshPhysicalMaterial({
        color: source?.color?.clone() || new THREE.Color(0x909090),
        map: source?.map || null,
        normalMap: source?.normalMap || null,
        roughnessMap: source?.roughnessMap || null,
        metalnessMap: source?.metalnessMap || null,
        emissiveMap: source?.emissiveMap || null,
        metalness: Number.isFinite(source?.metalness) ? source.metalness : 0,
        roughness: Number.isFinite(source?.roughness) ? source.roughness : 0.55,
        transparent: Boolean(source?.transparent),
        opacity: Number.isFinite(source?.opacity) ? source.opacity : 1,
        alphaTest: Number(source?.alphaTest) || 0,
        side: source?.side ?? THREE.FrontSide,
        vertexColors: Boolean(source?.vertexColors),
      });
      material.name = source?.name || `${object.name}_material`;
      const label = `${object.name} ${source?.name || ""}`.toLowerCase();
      if (/body|shell|white|orange|green|livery|panel|fender|deck|spine/.test(label)) {
        material.clearcoat = 0.86;
        material.clearcoatRoughness = 0.2;
        material.roughness = Math.min(material.roughness, 0.34);
      }
      if (/carbon|undertray|splitter|wing|flap|fin|diffuser/.test(label)) {
        material.metalness = 0.34;
        material.roughness = 0.44;
        material.clearcoat = 0.32;
      }
      if (/glass|canopy/.test(label)) {
        material.transparent = true;
        material.opacity = 0.72;
        material.roughness = 0.08;
        material.metalness = 0.06;
        material.clearcoat = 1;
      }
      if (/headlight|headlamp/.test(label)) {
        material.emissive.setHex(0xe8f4ff);
        material.emissiveIntensity = 0.4;
        this.frontLightMaterials.push(material);
      }
      if (/rear_light|rear_lamp|tail/.test(label)) {
        material.emissive.setHex(0xff170f);
        material.emissiveIntensity = 1.2;
        this.rearLights.push(material);
      }
      object.material = material;
      object.castShadow = true;
      object.receiveShadow = true;
    });
    if (manifest?.runtime_detail_kit !== false) {
      const detail = new THREE.Group();
      detail.name = "observatory_vehicle_detail_kit";
      const carbon = new THREE.MeshPhysicalMaterial({ color: 0x111518, metalness: 0.42, roughness: 0.4, clearcoat: 0.4 });
      const mirror = new THREE.MeshPhysicalMaterial({ color: key === "mazda787b" ? 0xd84c16 : 0xe9e8e2, metalness: 0.05, roughness: 0.3, clearcoat: 0.9 });
      for (const side of [-1, 1]) {
        const housing = new THREE.Mesh(new THREE.SphereGeometry(0.09, 18, 10), mirror);
        housing.scale.set(1.55, 0.62, 0.72);
        housing.position.set(0.38, side * (key === "mazda787b" ? 0.86 : 0.82), 0.69);
        detail.add(housing);
        const stalk = new THREE.Mesh(new THREE.CylinderGeometry(0.012, 0.018, 0.22, 8), carbon);
        stalk.rotation.x = Math.PI / 2;
        stalk.position.set(0.38, side * 0.75, 0.63);
        detail.add(stalk);
      }
      const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.008, 0.011, 0.28, 8), carbon);
      antenna.rotation.x = Math.PI / 2;
      antenna.position.set(-0.3, 0, 0.89);
      detail.add(antenna);
      for (const side of [-1, 1]) {
        for (let index = 0; index < 3; index += 1) {
          const strake = new THREE.Mesh(new THREE.BoxGeometry(0.52, 0.025, 0.085), carbon);
          strake.position.set(-1.94, side * (0.24 + index * 0.18), 0.12);
          detail.add(strake);
        }
      }
      asset.add(detail);
      ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"].forEach((name) => {
        const wheel = asset.getObjectByName(name);
        if (!wheel) return;
        const hardware = new THREE.Group();
        hardware.name = `${name}_visual_hardware`;
        const rim = new THREE.MeshPhysicalMaterial({ color: 0x9fa3a3, metalness: 0.9, roughness: 0.2 });
        const torus = new THREE.Mesh(new THREE.TorusGeometry(0.205, 0.018, 10, 36), rim);
        torus.rotation.x = Math.PI / 2;
        hardware.add(torus);
        for (let index = 0; index < 10; index += 1) {
          const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.31, 0.018, 0.024), rim);
          spoke.rotation.y = index * Math.PI / 5;
          hardware.add(spoke);
        }
        const hub = new THREE.Mesh(new THREE.CylinderGeometry(0.054, 0.054, 0.035, 20), carbon);
        hardware.add(hub);
        wheel.add(hardware);
      });
    }
  }

  addVehicleLights() {
    this.headlights = [];
    for (const side of [-1, 1]) {
      const light = new THREE.SpotLight(0xd7eeff, 0, 72, Math.PI / 6.8, 0.56, 1.35);
      light.position.set(1.65, 0.42, side * 0.48);
      light.castShadow = false;
      const target = new THREE.Object3D();
      target.position.set(42, -0.65, side * 1.2);
      light.target = target;
      this.vehicleRoot.add(light, target);
      this.headlights.push(light);
    }
    this.vehicleFill = new THREE.PointLight(0x93afd0, 0, 8, 1.8);
    this.vehicleFill.position.set(-0.4, 2.2, 0);
    this.vehicleRoot.add(this.vehicleFill);
  }

  applyFrame(frame, showBrain = false, telemetryChanged = true, deltaSeconds = 1 / 60) {
    const rawPosition = simToWorld(frame.vehicle.position);
    const airborne = Boolean(frame.vehicle.airborne);
    const trackSurface = airborne
      ? null
      : sampleTrackSurface(this.trackPoints, frame.vehicle.position, frame.telemetry.progress);
    const visualRoadHeight = trackSurface?.height ?? rawPosition.y;
    const attitudeTarget = trackSurface
      ? { ...frame.vehicle, pitch: trackSurface.pitch }
      : frame.vehicle;
    const snapPose = !this.renderPoseInitialized
      || this.previousRenderInput.distanceToSquared(rawPosition) > 625
      || Boolean(frame.replay.scrubbing)
      || airborne !== this.previousAirborne;
    const renderPose = this.renderAttitude.update(attitudeTarget, deltaSeconds, snapPose);
    const position = this.renderPosition.copy(rawPosition);
    position.y = this.renderHeight.update(visualRoadHeight, deltaSeconds, snapPose || airborne);
    this.previousRenderInput.copy(rawPosition);
    this.previousAirborne = airborne;
    this.renderPoseInitialized = true;
    this.vehicleRoot.position.copy(position);
    vehiclePoseQuaternion(renderPose, this.vehicleRoot.quaternion);
    this.vehicleRoot.visible = true;
    this.contactShadow.position.set(rawPosition.x, visualRoadHeight + 0.035, rawPosition.z);
    this.contactShadow.quaternion.copy(this.vehicleRoot.quaternion).multiply(shadowPlaneBasis);
    frame.vehicle.visualLength = this.vehicleVisualLength;
    this.updateVehicleNodes(frame);
    if (telemetryChanged) this.updateFootprint(frame);
    this.footprintLine.visible = this.engineeringVisible && this.footprintHasData;
    this.brainGroup.visible = showBrain;
    if (showBrain && (telemetryChanged || !this.brainWasVisible)) this.updateBrain(frame, position);
    this.brainWasVisible = showBrain;
    return {
      position,
      yaw: frame.vehicle.yaw,
      progress: frame.telemetry.progress,
      speedMps: frame.telemetry.speedMps,
      steer: frame.telemetry.steer,
      car: frame.identity.car,
    };
  }

  updateVehicleNodes(frame) {
    if (!this.vehicleAsset) return;
    ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"].forEach((name, index) => {
      const rotation = frame.vehicle.wheelRotation[index] || 0;
      const steer = (name.endsWith("fl") || name.endsWith("fr")) ? (frame.vehicle.wheelSteer[index] || frame.vehicle.steer * 0.42) : 0;
      this.vehicleAsset.traverse((node) => {
        if (node.name !== name && !node.name.startsWith(`${name}__`)) return;
        node.rotation.y = rotation;
        node.rotation.z = steer;
      });
    });
    const lowDrag = Number(frame.telemetry.activeAeroLowDrag) || 0;
    ["aero_front_flap_left", "aero_front_flap_right", "aero_rear_wing"].forEach((name, index) => {
      const node = this.vehicleAsset.getObjectByName(name);
      if (node) node.rotation.y = (index === 2 ? -1 : 1) * lowDrag * 0.13;
    });
    const brakeGlow = 0.9 + Number(frame.telemetry.brake) * 5.2;
    this.rearLights.forEach((material) => { material.emissiveIntensity = brakeGlow; });
  }

  updateFootprint(frame) {
    const footprint = frame.vehicle.footprint;
    if (!Array.isArray(footprint) || footprint.length < 3) {
      this.footprintHasData = false;
      return;
    }
    const z = frame.vehicle.position[2] + 0.055;
    const points = footprint.map((point) => simToWorld([point[0], point[1], z]));
    let positions = this.footprintLine.geometry.getAttribute("position");
    if (!positions || positions.count !== points.length) {
      this.footprintLine.geometry.dispose();
      positions = new THREE.BufferAttribute(new Float32Array(points.length * 3), 3).setUsage(THREE.DynamicDrawUsage);
      this.footprintLine.geometry = new THREE.BufferGeometry();
      this.footprintLine.geometry.setAttribute("position", positions);
    }
    points.forEach((point, index) => positions.setXYZ(index, point.x, point.y, point.z));
    positions.needsUpdate = true;
    this.footprintLine.geometry.computeBoundingSphere();
    this.footprintHasData = true;
  }

  updateBrain(frame, origin) {
    this.setLinePoints(this.predictionLine, frame.brain.predictedPath, origin);
    this.setLinePoints(this.paceLine, frame.brain.pacePoints, origin);
    const rays = frame.brain.rays.slice(0, 48);
    while (this.rayLines.length < rays.length) {
      const index = this.rayLines.length;
      const color = new THREE.Color().setHSL(0.53 + (index / Math.max(1, rays.length)) * 0.17, 0.92, 0.66);
      const line = this.makeLine(color, 1, 0.34 + (index % 4) * 0.08);
      this.rayLines.push(line);
      this.brainGroup.add(line);
    }
    this.rayLines.forEach((line, index) => {
      line.visible = index < rays.length;
      if (!line.visible) return;
      const ray = rays[index];
      const start = ray.start || frame.vehicle.position;
      const end = ray.end || ray.hit || ray;
      this.setLinePoints(line, [start, end], origin);
    });
  }

  makeLine(color, width = 1, opacity = 0.9, dashed = false) {
    const material = dashed
      ? new THREE.LineDashedMaterial({ color, transparent: true, opacity, dashSize: 1.6, gapSize: 0.7, depthTest: false })
      : new THREE.LineBasicMaterial({ color, transparent: true, opacity, linewidth: width, depthTest: false, blending: THREE.AdditiveBlending });
    const line = new THREE.Line(new THREE.BufferGeometry(), material);
    line.renderOrder = 30;
    return line;
  }

  setLinePoints(line, points, fallbackOrigin) {
    if (!Array.isArray(points) || points.length < 2) {
      line.visible = false;
      return;
    }
    const vectors = points.map((point) => {
      if (Array.isArray(point)) return simToWorld(point);
      if (point && Array.isArray(point.position)) return simToWorld(point.position);
      return fallbackOrigin.clone();
    });
    line.geometry.dispose();
    line.geometry = new THREE.BufferGeometry().setFromPoints(vectors);
    if (line.material.isLineDashedMaterial) line.computeLineDistances();
    line.visible = true;
  }

  setEngineeringVisible(visible) {
    this.engineeringVisible = Boolean(visible);
    this.footprintLine.visible = this.engineeringVisible && this.footprintHasData;
  }

  setWeather(mode) {
    this.weather = ["dry", "dusk", "night"].includes(mode) ? mode : "dry";
    const night = this.weather === "night";
    const dusk = this.weather === "dusk";
    this.trackMaterial.color.setHex(night ? 0xaab4bc : dusk ? 0xd7c9bd : 0xffffff);
    this.shoulderMaterial.color.setHex(night ? 0x7d8884 : dusk ? 0x8f817c : 0x9a9a8c);
    this.headlights.forEach((light) => { light.intensity = night ? 3200 : dusk ? 520 : 0; });
    if (this.vehicleFill) this.vehicleFill.intensity = night ? 75 : dusk ? 18 : 0;
    this.frontLightMaterials.forEach((material) => { material.emissiveIntensity = night ? 4 : dusk ? 1.8 : 0.4; });
  }
}
