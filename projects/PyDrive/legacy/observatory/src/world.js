import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { resolveVisualCondition } from "./visual-conditions.js";
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
        const rubber = Math.exp(-Math.pow(across / 0.26, 2));
        const groove = Math.exp(-Math.pow((across - 0.62) / 0.08, 2))
          + Math.exp(-Math.pow((across - 0.78) / 0.06, 2));
        const aggregate = 34 + grain * 18 + coarse * 8 - rubber * 12 - groove * 7;
        pixels[index] = aggregate;
        pixels[index + 1] = aggregate + 1;
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
    // tire rubber streaks
    context.globalAlpha = 0.28;
    for (let x = 24; x < size; x += 97) {
      context.fillStyle = x % 194 ? "#1c1f21" : "#3a3b38";
      context.fillRect(x, 28 + (x % 53), 2, 456 - (x % 71));
    }
    // solid white edge lines (v≈0 and v≈1 map to track edges)
    context.globalAlpha = 0.92;
    context.fillStyle = "#ece8dc";
    context.fillRect(0, 6, size, 7);
    context.fillRect(0, 499, size, 7);
    // soft yellow/white dashed centerline
    context.globalAlpha = 0.78;
    for (let x = 0; x < size; x += 56) {
      const dash = 28;
      context.fillStyle = x % 112 < 56 ? "#f0e6b8" : "#ddd8c4";
      context.fillRect(x, 250, dash, 5);
      context.fillRect(x, 257, dash, 5);
    }
    // secondary lane seams
    context.globalAlpha = 0.22;
    context.fillStyle = "#cfcabe";
    context.fillRect(0, 118, size, 2);
    context.fillRect(0, 392, size, 2);
    // dark racing line wear band
    context.globalAlpha = 0.16;
    context.fillStyle = "#07090a";
    context.fillRect(0, 168, size, 176);
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

function curbStripeTexture(renderer) {
  const canvas = document.createElement("canvas");
  canvas.width = 512;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  const stripe = 28;
  for (let x = 0; x < 512; x += stripe) {
    const red = Math.floor(x / stripe) % 2 === 0;
    context.fillStyle = red ? "#c93a28" : "#efe8d8";
    context.fillRect(x, 0, stripe, 64);
  }
  // subtle bevel shading
  context.globalAlpha = 0.18;
  context.fillStyle = "#000";
  context.fillRect(0, 0, 512, 10);
  context.fillStyle = "#fff";
  context.fillRect(0, 54, 512, 10);
  context.globalAlpha = 1;
  const texture = new THREE.CanvasTexture(canvas);
  texture.name = "observatory_procedural_curb_stripes";
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.ClampToEdgeWrapping;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = Math.min(8, renderer.capabilities.getMaxAnisotropy());
  texture.repeat.set(1, 1);
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

function carbonWeaveTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 128;
  canvas.height = 128;
  const context = canvas.getContext("2d");
  context.fillStyle = "#808080";
  context.fillRect(0, 0, 128, 128);
  context.globalAlpha = 0.3;
  context.strokeStyle = "#c0c0c0";
  context.lineWidth = 2;
  for (let offset = -128; offset < 256; offset += 8) {
    context.beginPath(); context.moveTo(offset, 0); context.lineTo(offset - 128, 128); context.stroke();
    context.beginPath(); context.moveTo(offset, 0); context.lineTo(offset + 128, 128); context.stroke();
  }
  const texture = new THREE.CanvasTexture(canvas);
  texture.name = "observatory_procedural_carbon_weave";
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(5, 5);
  texture.colorSpace = THREE.NoColorSpace;
  return texture;
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
      const moss = seededNoise(Math.floor(x / 3), Math.floor(y / 5));
      const value = 148 + fine * 42 + broad * 22;
      image.data[index] = value * 0.78 + moss * 8;
      image.data[index + 1] = value * 0.98 + moss * 14;
      image.data[index + 2] = value * 0.72 + moss * 6;
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

function terrainMaterial(map, level = 1) {
  return new THREE.MeshStandardMaterial({
    name: `observatory_dgm_terrain_lod${level}_material`,
    map,
    vertexColors: true,
    roughness: 0.94,
    metalness: 0,
    flatShading: false,
    side: THREE.DoubleSide,
  });
}

function colorTerrain(mesh, fine = false, map = null, level = fine ? 0 : 1) {
  const position = mesh.geometry.getAttribute("position");
  const color = new Float32Array(position.count * 3);
  const uv = new Float32Array(position.count * 2);
  for (let index = 0; index < position.count; index += 1) {
    const x = position.getX(index);
    const y = position.getY(index);
    const z = position.getZ(index);
    const variation = seededNoise(Math.floor(x * 0.07), Math.floor(y * 0.07));
    const moss = seededNoise(Math.floor(x * 0.19), Math.floor(y * 0.17));
    const altitude = THREE.MathUtils.clamp((z - 430) / 210, 0, 1);
    // Cooler Eifel greens with moss mottling and pale high ground
    const r = 0.08 + variation * 0.04 + altitude * 0.05 + moss * 0.02;
    const g = 0.18 + variation * 0.08 - altitude * 0.02 + moss * 0.05;
    const b = 0.1 + variation * 0.03 + altitude * 0.04 + moss * 0.015;
    color[index * 3] = fine ? r * 1.1 : r;
    color[index * 3 + 1] = fine ? g * 1.08 : g;
    color[index * 3 + 2] = fine ? b * 1.05 : b;
    uv[index * 2] = x / 12;
    uv[index * 2 + 1] = y / 12;
  }
  mesh.geometry.setAttribute("color", new THREE.BufferAttribute(color, 3));
  mesh.geometry.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  mesh.material = terrainMaterial(map, level);
}

function mergedCrown(species) {
  if (species === 0) {
    // layered spruce — denser mass reading
    const layers = [
      new THREE.ConeGeometry(1.75, 3.4, 12).translate(0, 1.35, 0),
      new THREE.ConeGeometry(1.42, 3.0, 12).translate(0, 2.55, 0),
      new THREE.ConeGeometry(1.08, 2.6, 12).translate(0, 3.65, 0),
      new THREE.ConeGeometry(0.72, 2.1, 10).translate(0, 4.55, 0),
    ];
    return mergeGeometries(layers, false);
  }
  if (species === 1) {
    const a = new THREE.SphereGeometry(1.45, 10, 7).translate(-0.48, 2.35, 0.14);
    const b = new THREE.SphereGeometry(1.62, 10, 7).translate(0.52, 2.7, -0.18);
    const c = new THREE.SphereGeometry(1.28, 10, 7).translate(0.05, 3.7, 0.22);
    const d = new THREE.SphereGeometry(0.95, 8, 6).translate(-0.2, 4.35, -0.1);
    return mergeGeometries([a, b, c, d], false);
  }
  const lower = new THREE.ConeGeometry(1.4, 3.0, 10).translate(0, 1.45, 0);
  const mid = new THREE.ConeGeometry(1.05, 2.7, 10).translate(0, 2.95, 0);
  const upper = new THREE.ConeGeometry(0.72, 2.3, 10).translate(0, 4.05, 0);
  const tip = new THREE.SphereGeometry(0.55, 8, 6).translate(0.12, 4.85, -0.06);
  return mergeGeometries([lower, mid, upper, tip], false);
}

function canopyMassGeometry() {
  // flattened ellipsoid used as distant forest canopy mass
  const geo = new THREE.SphereGeometry(1, 8, 5);
  geo.scale(1.8, 0.55, 1.35);
  geo.translate(0, 0.45, 0);
  return geo;
}

function shrubGeometry(kind = 0) {
  if (kind === 0) {
    return mergeGeometries([
      new THREE.SphereGeometry(0.55, 7, 5).translate(0, 0.4, 0),
      new THREE.SphereGeometry(0.42, 6, 4).translate(0.28, 0.55, 0.1),
      new THREE.SphereGeometry(0.38, 6, 4).translate(-0.22, 0.48, -0.12),
    ], false);
  }
  return mergeGeometries([
    new THREE.ConeGeometry(0.48, 1.1, 6).translate(0, 0.5, 0),
    new THREE.SphereGeometry(0.32, 6, 4).translate(0.18, 0.72, 0.08),
  ], false);
}

/** Visual-only ribbon strip along the runtime centerline (markings / kerb cues). */
function buildTrackRibbon(centerline, widths, {
  lateralSign = 0,
  lateralFrac = 0,
  lateralMeters = null,
  halfWidth = 0.08,
  lift = 0.035,
  stride = 3,
} = {}) {
  const samples = [];
  for (let index = 0; index < centerline.length; index += stride) samples.push(index);
  if (samples[samples.length - 1] !== centerline.length - 1) samples.push(centerline.length - 1);
  const positions = [];
  const uvs = [];
  const indices = [];
  let arc = 0;
  for (let i = 0; i < samples.length; i += 1) {
    const index = samples[i];
    const point = centerline[index];
    const next = centerline[(index + Math.max(1, stride)) % centerline.length];
    if (i > 0) {
      const prev = centerline[samples[i - 1]];
      arc += Math.hypot(point[0] - prev[0], point[1] - prev[1], point[2] - prev[2]);
    }
    const dx = next[0] - point[0];
    const dy = next[1] - point[1];
    const length = Math.max(0.001, Math.hypot(dx, dy));
    const nx = -dy / length;
    const ny = dx / length;
    const half = Number(widths?.[index]) || 10;
    const offset = lateralMeters != null
      ? lateralMeters * lateralSign
      : half * lateralFrac * lateralSign;
    const left = halfWidth;
    const ax = point[0] + nx * (offset - left);
    const ay = point[1] + ny * (offset - left);
    const bx = point[0] + nx * (offset + left);
    const by = point[1] + ny * (offset + left);
    const z = point[2] + lift;
    // sim xy-up-z → world x,y,z with y-up
    positions.push(ax, z, -ay, bx, z, -by);
    uvs.push(arc / 8, 0, arc / 8, 1);
    if (i > 0) {
      const base = (i - 1) * 2;
      indices.push(base, base + 1, base + 2, base + 1, base + 3, base + 2);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uvs, 2));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  return geometry;
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
    this.visualGroundRaycaster = new THREE.Raycaster();
    this.visualGroundRaycaster.ray.direction.set(0, -1, 0);
    this.weather = "clear-day";
    this.visualCondition = { lighting: "day", precipitation: "clear", surface: "dry" };
    this.wettableTerrainMaterials = [];
    this.wetSurfaceGroup = new THREE.Group();
    this.wetSurfaceGroup.name = "visual_condition_surface_details";
    this.wetPuddleMaterial = new THREE.MeshStandardMaterial({
      color: 0x142428,
      metalness: 0,
      roughness: 0.38,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    this.rainRippleMaterial = new THREE.MeshBasicMaterial({
      color: 0xc4dce1,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.rainRipples = [];
    this.scene.add(this.wetSurfaceGroup);
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
    this.vehicleMaterials = [];
    this.tyreMaterials = [];
    this.brakeDiscMaterials = [];
    this.carbonWeave = carbonWeaveTexture();
    this.vehicleGroundContact = { ground_anchor_offset_m: 0, contact_tolerance_m: 0.02 };
    this.debugVisuals = false;
    this.debugGroup = new THREE.Group();
    this.debugGroup.name = "observatory_visual_grounding_debug";
    this.debugGroup.visible = false;
    this.scene.add(this.debugGroup);
    this.terrainDetailTexture = terrainTexture(renderer);
    this.curbTexture = curbStripeTexture(renderer);
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
      bumpScale: 0.055,
      color: 0xe8eaec,
      roughness: 0.88,
      metalness: 0.04,
      clearcoat: 0.12,
      clearcoatRoughness: 0.82,
      side: THREE.DoubleSide,
    });
    material.userData.truthBoundary = "material-only visual skin over exact runtime track vertices";
    return material;
  }

  makeShoulderMaterial() {
    return new THREE.MeshStandardMaterial({
      name: "observatory_truth_shoulder_visual_skin",
      map: surfaceTexture(this.renderer, "shoulder"),
      color: 0x8e9084,
      roughness: 0.96,
      metalness: 0,
      side: THREE.DoubleSide,
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
    this.addDebugOverlays(data);
    truth.scene.name = "nordschleife_truth_layer";
    visual.scene.name = "nordschleife_visual_layer";
    truth.scene.rotation.x = -Math.PI / 2;
    visual.scene.rotation.x = -Math.PI / 2;
    truth.scene.traverse((object) => {
      if (!object.isMesh) return;
      object.receiveShadow = true;
      object.castShadow = false;
      object.frustumCulled = false;
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
      if (object.name.includes("visual_dgm_verge") || object.name.includes("seam_skirt") || object.name.includes("containment_shell")) {
        colorTerrain(object, false, this.terrainDetailTexture, 1);
        this.registerWettableTerrainMaterial(object.material);
      }
      else if (object.name.includes("guardrail")) object.material = new THREE.MeshStandardMaterial({ color: 0xb6b8b3, metalness: 0.78, roughness: 0.31 });
      else if (object.name.includes("drainage")) object.material = new THREE.MeshStandardMaterial({ color: 0x3c423d, metalness: 0.08, roughness: 0.94, side: THREE.DoubleSide });
      else if (object.name.includes("fence")) object.material = new THREE.MeshStandardMaterial({ color: 0x73817d, metalness: 0.54, roughness: 0.46, transparent: true, opacity: 0.38, depthWrite: false, side: THREE.DoubleSide });
      else if (object.name.includes("curb")) {
        const accent = object.name.includes("accent");
        object.material = new THREE.MeshStandardMaterial({
          name: accent ? "observatory_curb_accent_stripes" : "observatory_curb_stripes",
          map: this.curbTexture,
          color: accent ? 0xffffff : 0xf2eee4,
          roughness: accent ? 0.7 : 0.76,
          metalness: 0.02,
          side: THREE.DoubleSide,
        });
        // Stretch stripe repeat along the curb length via existing UVs when present.
        if (object.geometry.getAttribute("uv")) {
          object.material.map = this.curbTexture.clone();
          object.material.map.needsUpdate = true;
          object.material.map.wrapS = THREE.RepeatWrapping;
          object.material.map.repeat.set(accent ? 18 : 12, 1);
        }
      }
    });
    this.scene.add(visual.scene, truth.scene);
    onProgress(0.48, "Loading offline DGM terrain tiles");
    await this.loadTerrainLevel(manifest, 1, this.terrainCoarseGroup);
    this.scene.add(this.terrainCoarseGroup);
    onProgress(0.61, "Planting multi-species deterministic forest");
    this.addForest(data.forest || []);
    this.addTracksideFurniture(data);
    this.addTrackMarkings(data);
    this.addWetSurfaceDetails(data);
    onProgress(0.76, "Preparing the selected endurance prototype");
    if (initialVehicle) await this.ensureVehicle(initialVehicle);
    this.loadTerrainLevel(manifest, 0, this.terrainFineGroup).then(() => {
      this.scene.add(this.terrainFineGroup);
      this.setVisualCondition(this.weather);
    }).catch((error) => console.warn("Fine DGM terrain LOD unavailable; retaining coarse tiles", error));
    onProgress(1, `World ${manifest.track.hash.slice(0, 12)} verified`);
    this.setVisualCondition(this.weather);
    return { manifest, data };
  }

  registerWettableTerrainMaterial(material) {
    if (!material || this.wettableTerrainMaterials.includes(material)) return;
    material.userData.weatherDryColor = material.color?.clone?.() || new THREE.Color(0xffffff);
    material.userData.weatherDryRoughness = Number(material.roughness) || 1;
    this.wettableTerrainMaterials.push(material);
  }

  addWetSurfaceDetails(data) {
    const points = data.track?.centerline || [];
    const widths = data.track?.width_m || [];
    if (points.length < 30) return;
    const window = Math.max(160, Math.round(points.length / 19));
    for (let start = 0; start < points.length; start += window) {
      let index = start;
      const end = Math.min(points.length, start + window);
      for (let candidate = start + 1; candidate < end; candidate += 1) {
        if (points[candidate][2] < points[index][2]) index = candidate;
      }
      const point = points[index];
      const next = points[(index + 5) % points.length];
      const dx = next[0] - point[0];
      const dy = next[1] - point[1];
      const length = Math.max(0.001, Math.hypot(dx, dy));
      const side = (Math.floor(start / window) % 2 ? -1 : 1);
      const laneOffset = Math.min(1.25, (Number(widths[index]) || 9) * 0.14) * side;
      const x = point[0] - dy / length * laneOffset;
      const y = point[1] + dx / length * laneOffset;
      const base = simToWorld([x, y, point[2] + 0.026]);
      const puddle = new THREE.Mesh(new THREE.CircleGeometry(1, 28), this.wetPuddleMaterial);
      puddle.name = `visual_lowspot_puddle_${index}`;
      puddle.position.copy(base);
      puddle.rotation.set(-Math.PI / 2, Math.atan2(-dy, dx), 0);
      const size = 0.75 + (index % 7) * 0.09;
      puddle.scale.set(1.75 + size, 0.9 + size * 0.35, 1);
      this.wetSurfaceGroup.add(puddle);
      for (let ringIndex = 0; ringIndex < 2; ringIndex += 1) {
        const ripple = new THREE.Mesh(new THREE.RingGeometry(0.72, 0.79, 20), this.rainRippleMaterial);
        ripple.name = `visual_rain_ripple_${index}_${ringIndex}`;
        ripple.position.copy(base);
        ripple.position.y += 0.012 + ringIndex * 0.003;
        ripple.rotation.set(-Math.PI / 2, Math.atan2(-dy, dx), 0);
        ripple.userData.phase = (index * 0.013 + ringIndex * 0.47) % 1;
        this.rainRipples.push(ripple);
        this.wetSurfaceGroup.add(ripple);
      }
    }
  }

  addForest(entries) {
    if (!entries.length) return;
    const speciesEntries = [[], [], []];
    const horizonEntries = [[], [], []];
    const midPool = [];
    entries.forEach((entry, index) => {
      const species = THREE.MathUtils.clamp(Number(entry.species ?? index % 3), 0, 2);
      if (entry.tier === "horizon") horizonEntries[species].push(entry);
      else {
        speciesEntries[species].push(entry);
        midPool.push({ entry, species, index });
      }
    });

    // Deterministic densification: satellite saplings between banked trees.
    midPool.forEach(({ entry, species, index }) => {
      const satellites = entry.tier === "hero" ? 2 : (index % 3 === 0 ? 2 : 1);
      for (let s = 0; s < satellites; s += 1) {
        const angle = seededNoise(index, 17 + s) * Math.PI * 2;
        const radius = 3.2 + seededNoise(index, 29 + s) * 7.5;
        const nx = Math.cos(angle) * radius;
        const ny = Math.sin(angle) * radius;
        speciesEntries[species].push({
          position: [
            entry.position[0] + nx,
            entry.position[1] + ny,
            entry.position[2] + (seededNoise(index, 41 + s) - 0.5) * 0.6,
          ],
          scale: (Number(entry.scale) || 1) * (0.45 + seededNoise(index, 53 + s) * 0.38),
          rotation: seededNoise(index, 61 + s) * Math.PI * 2,
          species,
          tier: "fill",
        });
      }
    });

    const trunkMaterial = new THREE.MeshStandardMaterial({ color: 0x2e221a, roughness: 0.98 });
    const crownMaterials = [0x143520, 0x1f3f22, 0x17382a, 0x254828].map((color) =>
      new THREE.MeshStandardMaterial({ color, roughness: 0.92, flatShading: true }));
    const matrix = new THREE.Matrix4();
    const quaternion = new THREE.Quaternion();
    const position = new THREE.Vector3();
    const scale = new THREE.Vector3();
    speciesEntries.forEach((records, species) => {
      if (!records.length) return;
      const trunk = new THREE.InstancedMesh(new THREE.CylinderGeometry(0.15, 0.25, 2.8, 7), trunkMaterial, records.length);
      const crown = new THREE.InstancedMesh(mergedCrown(species), crownMaterials[species], records.length);
      records.forEach((tree, index) => {
        simToWorld(tree.position, position);
        const groundY = position.y;
        const size = Number(tree.scale) || 1;
        quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Number(tree.rotation) || 0);
        scale.set(size, size, size);
        position.y = groundY + 1.4 * size;
        matrix.compose(position, quaternion, scale);
        trunk.setMatrixAt(index, matrix);
        // Crown geometries are authored from ground level already; applying
        // the trunk lift a second time made deciduous trees float on stilts.
        position.y = groundY;
        matrix.compose(position, quaternion, scale);
        crown.setMatrixAt(index, matrix);
      });
      trunk.name = `visual_forest_trunks_${species}`;
      crown.name = `visual_forest_crowns_${species}`;
      trunk.receiveShadow = true;
      crown.receiveShadow = true;
      crown.castShadow = records.some((tree) => tree.tier === "hero") && species !== 1;
      this.forestMeshes.push({ mesh: trunk, capacity: records.length }, { mesh: crown, capacity: records.length });
      this.scene.add(trunk, crown);
    });

    // Soft canopy mass — distant forest volume without per-tree cost.
    const canopyRecords = midPool.filter((_, index) => index % 2 === 0);
    if (canopyRecords.length) {
      const canopy = new THREE.InstancedMesh(
        canopyMassGeometry(),
        new THREE.MeshStandardMaterial({
          color: 0x163424,
          roughness: 0.98,
          flatShading: true,
          transparent: true,
          opacity: 0.88,
          depthWrite: true,
        }),
        canopyRecords.length,
      );
      canopyRecords.forEach(({ entry, index }, i) => {
        simToWorld(entry.position, position);
        const size = 4.5 + seededNoise(index, 71) * 5.5 + (entry.tier === "hero" ? 2.2 : 0);
        scale.set(size, size * (0.55 + seededNoise(index, 73) * 0.25), size * 0.9);
        quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), seededNoise(index, 79) * Math.PI * 2);
        position.y += 2.4 + seededNoise(index, 83) * 1.8;
        matrix.compose(position, quaternion, scale);
        canopy.setMatrixAt(i, matrix);
      });
      canopy.name = "visual_forest_canopy_mass";
      canopy.receiveShadow = false;
      canopy.castShadow = false;
      canopy.frustumCulled = true;
      this.forestMeshes.push({ mesh: canopy, capacity: canopyRecords.length });
      this.scene.add(canopy);
    }

    horizonEntries.forEach((records, species) => {
      if (!records.length) return;
      // Duplicate each horizon tree with a nearer silhouette twin for denser ridges.
      const expanded = [];
      records.forEach((tree, index) => {
        expanded.push(tree);
        if (index % 2 === 0) {
          expanded.push({
            ...tree,
            position: [
              tree.position[0] + (seededNoise(index, 91) - 0.5) * 14,
              tree.position[1] + (seededNoise(index, 97) - 0.5) * 14,
              tree.position[2],
            ],
            scale: (Number(tree.scale) || 0.7) * (0.75 + seededNoise(index, 101) * 0.35),
            rotation: seededNoise(index, 103) * Math.PI * 2,
          });
        }
      });
      const horizon = new THREE.InstancedMesh(
        new THREE.ConeGeometry(1.25 + species * 0.14, 4.2 + species * 0.35, 8),
        new THREE.MeshStandardMaterial({ color: [0x122e1c, 0x1a3820, 0x153528][species], roughness: 0.96, flatShading: true }),
        expanded.length,
      );
      expanded.forEach((tree, index) => {
        simToWorld(tree.position, position);
        const size = Number(tree.scale) || 0.7;
        quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Number(tree.rotation) || 0);
        scale.set(size, size * (0.9 + seededNoise(index, species) * 0.28), size);
        position.y += 2.0 * size;
        matrix.compose(position, quaternion, scale);
        horizon.setMatrixAt(index, matrix);
      });
      horizon.name = `visual_horizon_forest_${species}`;
      horizon.receiveShadow = false;
      horizon.castShadow = false;
      this.forestMeshes.push({ mesh: horizon, capacity: expanded.length });
      this.scene.add(horizon);
    });

    const brushSource = entries.filter((entry) => entry.tier !== "horizon");
    const underbrushRecords = [];
    brushSource.forEach((tree, index) => {
      underbrushRecords.push(tree);
      if (index % 2 === 0) underbrushRecords.push(tree);
    });
    const underbrush = new THREE.InstancedMesh(
      shrubGeometry(0),
      new THREE.MeshStandardMaterial({ color: 0x244828, roughness: 0.95, flatShading: true }),
      underbrushRecords.length,
    );
    underbrushRecords.forEach((tree, index) => {
      simToWorld(tree.position, position);
      position.x += (seededNoise(index, 3) - 0.5) * 5.5;
      position.z += (seededNoise(index, 9) - 0.5) * 5.5;
      position.y += 0.05;
      const size = 0.7 + seededNoise(index, 11) * 1.15;
      scale.set(size, size * (0.7 + seededNoise(index, 13) * 0.5), size);
      quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), seededNoise(index, 21) * Math.PI * 2);
      matrix.compose(position, quaternion, scale);
      underbrush.setMatrixAt(index, matrix);
    });
    underbrush.name = "visual_forest_underbrush";
    this.forestMeshes.push({ mesh: underbrush, capacity: underbrushRecords.length });
    this.scene.add(underbrush);

    // Second shrub layer — lower ferns/scrub for ground mass.
    const scrubRecords = brushSource.filter((_, index) => index % 2 === 1);
    if (scrubRecords.length) {
      const scrub = new THREE.InstancedMesh(
        shrubGeometry(1),
        new THREE.MeshStandardMaterial({ color: 0x1c3a24, roughness: 0.97, flatShading: true }),
        scrubRecords.length,
      );
      scrubRecords.forEach((tree, index) => {
        simToWorld(tree.position, position);
        position.x += (seededNoise(index, 33) - 0.5) * 8;
        position.z += (seededNoise(index, 39) - 0.5) * 8;
        const size = 0.85 + seededNoise(index, 43) * 1.4;
        scale.set(size * 1.2, size * 0.55, size * 1.1);
        quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), seededNoise(index, 47) * Math.PI * 2);
        matrix.compose(position, quaternion, scale);
        scrub.setMatrixAt(index, matrix);
      });
      scrub.name = "visual_forest_scrub";
      this.forestMeshes.push({ mesh: scrub, capacity: scrubRecords.length });
      this.scene.add(scrub);
    }
    this.setForestDensity(this.forestDensity);
  }

  addTrackMarkings(data) {
    const points = data.track?.centerline || [];
    const widths = data.track?.width_m || [];
    if (points.length < 40) return;
    const group = new THREE.Group();
    group.name = "visual_track_markings";

    const edgeMaterial = new THREE.MeshStandardMaterial({
      color: 0xf2eee0,
      roughness: 0.62,
      metalness: 0.04,
      emissive: 0x222018,
      emissiveIntensity: 0.08,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const centerMaterial = new THREE.MeshStandardMaterial({
      color: 0xe8d9a0,
      roughness: 0.58,
      metalness: 0.05,
      emissive: 0x2a2410,
      emissiveIntensity: 0.06,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });

    for (const sign of [-1, 1]) {
      const edge = new THREE.Mesh(
        buildTrackRibbon(points, widths, {
          lateralSign: sign,
          lateralFrac: 0.96,
          halfWidth: 0.07,
          lift: 0.028,
          stride: 3,
        }),
        edgeMaterial,
      );
      edge.name = `visual_edge_line_${sign < 0 ? "left" : "right"}`;
      edge.receiveShadow = true;
      edge.frustumCulled = false;
      group.add(edge);
    }

    // One merged dashed centerline (not hundreds of meshes — that was enough
    // geometry pressure to risk a black WebGL viewport on laptop GPUs).
    const dashLen = 5;
    const gapLen = 9;
    const dashGeometries = [];
    for (let start = 0; start < points.length; start += dashLen + gapLen) {
      const local = [];
      const localW = [];
      for (let i = 0; i < dashLen; i += 1) {
        const index = (start + i) % points.length;
        local.push(points[index]);
        localW.push(widths[index] ?? 10);
      }
      if (local.length < 2) continue;
      dashGeometries.push(buildTrackRibbon(local, localW, {
        lateralSign: 0,
        lateralFrac: 0,
        halfWidth: 0.055,
        lift: 0.03,
        stride: 1,
      }));
    }
    if (dashGeometries.length) {
      const merged = mergeGeometries(dashGeometries, false);
      dashGeometries.forEach((geometry) => geometry.dispose());
      if (merged) {
        const dash = new THREE.Mesh(merged, centerMaterial);
        dash.name = "visual_center_dashes";
        dash.frustumCulled = false;
        group.add(dash);
      }
    }

    const kerbCueMaterial = new THREE.MeshStandardMaterial({
      map: this.curbTexture.clone(),
      color: 0xffffff,
      roughness: 0.72,
      metalness: 0.02,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    kerbCueMaterial.map.needsUpdate = true;
    kerbCueMaterial.map.wrapS = THREE.RepeatWrapping;
    kerbCueMaterial.map.repeat.set(40, 1);
    for (const sign of [-1, 1]) {
      const kerb = new THREE.Mesh(
        buildTrackRibbon(points, widths, {
          lateralSign: sign,
          lateralFrac: 1.02,
          halfWidth: 0.16,
          lift: 0.04,
          stride: 4,
        }),
        kerbCueMaterial,
      );
      kerb.name = `visual_kerb_cue_${sign < 0 ? "left" : "right"}`;
      kerb.frustumCulled = false;
      group.add(kerb);
    }

    this.scene.add(group);
  }

  addTracksideFurniture(data) {
    const records = data.trackside?.reflectors || [];
    if (!records.length) return;
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
      simToWorld(record.position, position);
      position.y += 0.41;
      quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), record.yaw);
      matrix.compose(position, quaternion, scale);
      posts.setMatrixAt(index, matrix);
    });
    posts.name = "visual_trackside_reflector_posts";
    posts.receiveShadow = true;
    posts.frustumCulled = false;
    this.scene.add(posts);

    const placeInstances = (recordsForAsset, geometry, material, name, height = 0, scaleValue = 1) => {
      if (!recordsForAsset.length) return;
      const mesh = new THREE.InstancedMesh(geometry, material, recordsForAsset.length);
      recordsForAsset.forEach((record, index) => {
        simToWorld(record.position, position);
        position.y += height;
        quaternion.setFromAxisAngle(new THREE.Vector3(0, 1, 0), Number(record.yaw) || 0);
        scale.set(scaleValue, scaleValue, scaleValue);
        matrix.compose(position, quaternion, scale);
        mesh.setMatrixAt(index, matrix);
      });
      mesh.name = name;
      mesh.receiveShadow = true;
      mesh.castShadow = false;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
    };
    placeInstances(
      data.trackside?.guardrail_posts || [],
      new THREE.BoxGeometry(0.13, 0.92, 0.12),
      new THREE.MeshStandardMaterial({ color: 0x69706d, metalness: 0.74, roughness: 0.35 }),
      "visual_guardrail_support_posts", 0.46,
    );
    placeInstances(
      data.trackside?.drainage_markers || [],
      new THREE.BoxGeometry(1.05, 0.035, 0.22),
      new THREE.MeshStandardMaterial({ color: 0x2f3431, roughness: 0.97 }),
      "visual_drainage_grates", 0.055,
    );
    placeInstances(
      data.trackside?.marshal_stations || [],
      new THREE.BoxGeometry(1.2, 0.84, 0.075),
      new THREE.MeshStandardMaterial({ color: 0xe3a83b, roughness: 0.62, emissive: 0x332008, emissiveIntensity: 0.18 }),
      "visual_marshal_station_boards", 1.72,
    );

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
        colorTerrain(object, level === 0, this.terrainDetailTexture, level);
        this.registerWettableTerrainMaterial(object.material);
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
    const hysteresis = 80;
    this.terrainTiles.forEach((tile) => {
      const distance = Math.hypot(simX - tile.center.x, simY - tile.center.y);
      // A hysteresis band avoids repeated LOD swaps at the threshold without
      // ever drawing two terrain surfaces at the same depth.
      if (tile.fineActive == null) tile.fineActive = Boolean(tile.fine) && distance <= this.terrainFineDistance;
      if (distance < this.terrainFineDistance - hysteresis) tile.fineActive = Boolean(tile.fine);
      else if (distance > this.terrainFineDistance + hysteresis) tile.fineActive = false;
      if (tile.fine) tile.fine.visible = tile.fineActive;
      if (tile.coarse) tile.coarse.visible = !tile.fineActive;
    });
  }

  visualGroundHeightAt(x, z, fallback = 0) {
    // Editorial cameras are presentation-only, but they need the same visual
    // DGM reference as scenery. Restrict the ray to terrain groups so tree,
    // fence, and guardrail meshes cannot be mistaken for ground.
    this.visualGroundRaycaster.set(new THREE.Vector3(x, 10000, z), new THREE.Vector3(0, -1, 0));
    const hits = this.visualGroundRaycaster.intersectObjects([
      this.terrainFineGroup,
      this.terrainCoarseGroup,
    ], true);
    return hits[0]?.point?.y ?? fallback;
  }

  setVisualCondition(condition = "clear-day") {
    this.visualCondition = resolveVisualCondition(condition);
    this.weather = this.visualCondition.id;
    const wetness = this.visualCondition.wetness;
    const dampGrass = new THREE.Color(0x1e3526);
    this.wettableTerrainMaterials.forEach((material) => {
      const dryColor = material.userData.weatherDryColor;
      if (dryColor && material.color) material.color.copy(dryColor).lerp(dampGrass, wetness * 0.48);
      if (Number.isFinite(material.userData.weatherDryRoughness)) {
        material.roughness = THREE.MathUtils.lerp(material.userData.weatherDryRoughness, Math.max(0.55, material.userData.weatherDryRoughness * 0.7), wetness);
      }
    });
    this.wetSurfaceGroup.visible = wetness > 0.28;
    this.wetPuddleMaterial.opacity = wetness > 0.28 ? THREE.MathUtils.lerp(0.14, 0.34, wetness) : 0;
    this.rainRippleMaterial.opacity = this.visualCondition.precipitation === "rain" ? 0.28 : 0;
    this.rainRipples.forEach((ripple) => { ripple.visible = this.visualCondition.precipitation === "rain"; });
    this.trackMaterial.roughness = THREE.MathUtils.lerp(0.88, 0.38, wetness);
    this.trackMaterial.clearcoat = THREE.MathUtils.lerp(0.12, 0.42, wetness);
    this.trackMaterial.clearcoatRoughness = THREE.MathUtils.lerp(0.82, 0.22, wetness);
    this.shoulderMaterial.roughness = THREE.MathUtils.lerp(0.96, 0.62, wetness);
    const lighting = this.visualCondition.lighting;
    this.trackMaterial.color.setHex(lighting === "night" ? 0x9aa4ac : lighting === "dusk" ? 0xd4c4b6 : 0xe8eaec);
    this.shoulderMaterial.color.setHex(lighting === "night" ? 0x6f7874 : lighting === "dusk" ? 0x847874 : 0x8e9084);
    this.vehicleMaterials.forEach(({ material, kind }) => {
      if (kind === "paint" || kind === "glass") material.roughness = Math.max(0.055, material.userData.dryRoughness * (1 - wetness * 0.38));
      if (kind === "carbon") material.roughness = material.userData.dryRoughness * (1 - wetness * 0.24);
      if (kind === "tyre") material.roughness = THREE.MathUtils.lerp(material.userData.dryRoughness, 0.54, wetness);
    });
    this.tyreMaterials.forEach((material) => {
      if (Number.isFinite(material.userData.dryRoughness)) material.roughness = THREE.MathUtils.lerp(material.userData.dryRoughness, 0.54, wetness);
    });
    this.contactShadow.material.opacity = THREE.MathUtils.lerp(0.72, 0.52, wetness);
    const night = lighting === "night";
    const dusk = lighting === "dusk";
    this.headlights.forEach((light) => { light.intensity = night ? 7.2 : dusk ? 3.6 : this.visualCondition.precipitation === "rain" ? 1.4 : 0; });
    if (this.vehicleFill) this.vehicleFill.intensity = night ? 95 : dusk ? 24 : 0;
    this.frontLightMaterials.forEach((material) => { material.emissiveIntensity = night ? 2.4 : dusk ? 1.35 : 0.28; });
    this.rearLights.forEach((material) => { material.emissiveIntensity = night ? 2.2 : 1.25; });
  }

  updateSurfaceDetails(elapsed) {
    if (this.visualCondition.precipitation !== "rain" || !this.rainRipples.length) return;
    this.rainRipples.forEach((ripple) => {
      const phase = (ripple.userData.phase + elapsed * 0.9) % 1;
      const scale = 0.35 + phase * 1.75;
      ripple.scale.setScalar(scale);
      ripple.visible = true;
    });
  }

  setWeather(mode) {
    this.setVisualCondition(mode);
  }

  setDebugVisuals(enabled) {
    this.debugVisuals = Boolean(enabled);
    this.debugGroup.visible = this.debugVisuals;
  }

  addDebugOverlays(data) {
    const centerline = data.track?.centerline || [];
    if (centerline.length < 3) return;
    const road = centerline.map((point) => simToWorld(point));
    const envelope = centerline.map((point, index) => {
      const next = centerline[(index + 3) % centerline.length];
      const dx = next[0] - point[0];
      const dy = next[1] - point[1];
      const length = Math.max(0.001, Math.hypot(dx, dy));
      const side = index % 2 ? 1 : -1;
      const distance = (Number(data.track.width_m?.[index]) || 10) * 0.5 + 18;
      return simToWorld([point[0] - dy / length * distance * side, point[1] + dx / length * distance * side, point[2] + 0.18]);
    });
    const line = (points, color, name) => {
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const mesh = new THREE.Line(geometry, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.78, depthTest: false }));
      mesh.name = name;
      mesh.renderOrder = 40;
      this.debugGroup.add(mesh);
    };
    line(road, 0xffd34d, "debug_truth_road_centerline");
    line(envelope, 0x55e6ff, "debug_terrain_cutout_envelope");
    const positions = (data.trackside?.reflectors || []).map((record) => simToWorld(record.position));
    if (positions.length) {
      const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints(positions), new THREE.PointsMaterial({ color: 0x6eff8d, size: 0.45, sizeAttenuation: true, depthTest: false }));
      dots.name = "debug_grounded_trackside_props";
      dots.renderOrder = 41;
      this.debugGroup.add(dots);
    }
  }

  updateGroundingDebug(position, roadHeight, airborne) {
    if (!this.debugArrow) {
      this.debugArrow = new THREE.ArrowHelper(new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1.2, 0xe2ff5e, 0.2, 0.12);
      this.debugArrow.name = "visual_ground_contact_anchor";
      this.debugGroup.add(this.debugArrow);
    }
    this.debugArrow.position.set(position.x, roadHeight, position.z);
    this.debugArrow.setColor(new THREE.Color(airborne ? 0xff665d : 0xe2ff5e));
    this.debugArrow.visible = true;
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
    const contact = manifest.visual_ground_contact;
    const wheelNodes = contact?.wheel_nodes || {};
    if (!contact || !Number.isFinite(Number(contact.ground_anchor_offset_m))
      || !Number.isFinite(Number(contact.contact_tolerance_m)) || Number(contact.contact_tolerance_m) > 0.02
      || ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"].some((name) => !wheelNodes[name])) {
      throw new Error(`Vehicle '${key}' is missing its visual ground-contact contract`);
    }
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
    this.vehicleGroundContact = manifest.visual_ground_contact;
    this.vehicleVisualLength = Number(manifest.dimensions_m?.length) || 0;
    this.vehicleKey = key;
    this.pendingVehicleKey = "";
    this.setVisualCondition(this.weather);
  }

  enhanceVehicle(asset, key, manifest) {
    this.rearLights = [];
    this.frontLightMaterials = [];
    this.vehicleMaterials = [];
    this.tyreMaterials = [];
    this.brakeDiscMaterials = [];
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
        material.normalMap = this.carbonWeave;
        material.normalScale.setScalar(0.11);
      }
      if (/glass|canopy/.test(label)) {
        material.transparent = true;
        material.opacity = 0.76;
        material.roughness = 0.08;
        material.metalness = 0.06;
        material.clearcoat = 1;
        material.ior = 1.45;
        material.reflectivity = 0.92;
        material.iridescence = 0.14;
      }
      if (/tyre|tire/.test(label)) {
        material.color.setHex(0x0d0f10);
        material.roughness = 0.82;
        material.metalness = 0;
      }
      if (/brake_disc|brakedisc|brake.*rotor|carbon_brake/.test(label)) {
        material.color.setHex(0x3f4444);
        material.roughness = 0.56;
        material.metalness = 0.2;
        material.emissive.setHex(0x4b0600);
        material.emissiveIntensity = 0;
      }
      const kind = /tyre|tire/.test(label) ? "tyre"
        : /glass|canopy/.test(label) ? "glass"
        : /carbon|undertray|splitter|wing|flap|fin|diffuser/.test(label) ? "carbon"
          : /body|shell|white|orange|green|livery|panel|fender|deck|spine/.test(label) ? "paint" : "other";
      material.userData.dryRoughness = material.roughness;
      this.vehicleMaterials.push({ material, kind });
      if (kind === "tyre") this.tyreMaterials.push(material);
      if (/brake_disc|brakedisc|brake.*rotor|carbon_brake/.test(label)) this.brakeDiscMaterials.push(material);
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
    // Only packages that explicitly opt in get a supplemental runtime kit.
    // The rebuilt 919 has complete native wheel assemblies, so adding a second
    // set here would create duplicate tyres, brake hardware, and clipping.
    if (manifest?.runtime_detail_kit === true) {
      const detail = new THREE.Group();
      detail.name = "observatory_vehicle_detail_kit";
      const carbon = new THREE.MeshPhysicalMaterial({ color: 0x111518, metalness: 0.42, roughness: 0.4, clearcoat: 0.4, normalMap: this.carbonWeave, normalScale: new THREE.Vector2(0.11, 0.11) });
      const mirror = new THREE.MeshPhysicalMaterial({ color: key === "mazda787b" ? 0xd84c16 : 0xe9e8e2, metalness: 0.05, roughness: 0.3, clearcoat: 0.9 });
      const tyre = new THREE.MeshStandardMaterial({ color: 0x0a0c0d, roughness: 0.84, metalness: 0 });
      tyre.userData.dryRoughness = tyre.roughness;
      this.tyreMaterials.push(tyre);
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
        const sideSign = name.endsWith("fl") || name.endsWith("rl") ? 1 : -1;
        const sidewall = new THREE.Mesh(new THREE.TorusGeometry(0.285, 0.074, 12, 36), tyre);
        sidewall.rotation.x = Math.PI / 2;
        hardware.add(sidewall);
        const brake = new THREE.MeshStandardMaterial({ color: 0x3f4444, metalness: 0.74, roughness: 0.34, emissive: 0x4b0600, emissiveIntensity: 0 });
        this.brakeDiscMaterials.push(brake);
        const disc = new THREE.Mesh(new THREE.CylinderGeometry(0.17, 0.17, 0.024, 24), brake);
        disc.position.y = sideSign * 0.045;
        hardware.add(disc);
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
    const contactOffset = airborne ? 0 : Number(this.vehicleGroundContact?.ground_anchor_offset_m) || 0;
    position.y = this.renderHeight.update(visualRoadHeight + contactOffset, deltaSeconds, snapPose || airborne);
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
    if (this.debugVisuals) this.updateGroundingDebug(position, visualRoadHeight, airborne);
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
    const speed = Math.max(0, Number(frame.telemetry.speedMps) || 0);
    const braking = THREE.MathUtils.smoothstep(Number(frame.telemetry.brake) || 0, 0.08, 0.82);
    const heat = braking * THREE.MathUtils.smoothstep(speed, 12, 46);
    const nightBoost = this.visualCondition.lighting === "night" ? 1.35 : this.visualCondition.lighting === "dusk" ? 1.08 : 0.8;
    this.brakeDiscMaterials.forEach((material) => {
      material.emissive.setRGB(0.95, 0.06 + heat * 0.18, 0.005);
      material.emissiveIntensity = heat * 2.4 * nightBoost;
    });
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

}
