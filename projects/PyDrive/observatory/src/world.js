import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { AttitudeFilter, HeightFilter, simToThree, vehicleOrientation } from "./coords.js";
import { condition } from "./conditions.js";
import { ForestField } from "./forest.js";
import { RailPosts } from "./rails.js";
import { setStylizeTree, stylize, stylizeTree } from "./stylize.js";
import {
  createTrackMaterial,
  setTrackLighting,
  setTrackPalette,
  setTrackWetness,
} from "./track-material.js";

// Band counts per category. The car is the hero and carries the rim; terrain is
// large soft landform that only needs enough steps to read as sculpted.
export const STYLE_VEHICLE = { bands: 4, softness: 0.07, rimStrength: 0.42, rimStart: 0.48 };
export const STYLE_SCENERY = { bands: 4, softness: 0.10, rimStrength: 0.12, rimStart: 0.62 };
// Nordschleife elevation runs ~330-620 m; keying colour to that range and to
// steepness is what turns one flat green slab back into readable landform.
export const STYLE_TERRAIN = {
  bands: 3,
  softness: 0.16,
  rimStrength: 0.0,
  terrain: {
    low: 0x4e6b3a,
    high: 0x76854e,
    slope: 0x36452f,
    heightLow: 330,
    heightHigh: 620,
    slopeStart: 0.22,
    steps: 4,
  },
};

const ASSET_ROOT = "./assets/observatory";
const loader = new GLTFLoader();

function loadGltf(url) {
  return new Promise((resolve, reject) => {
    loader.load(url, resolve, undefined, reject);
  });
}

function findNode(root, name) {
  let found = null;
  root.traverse((obj) => {
    if (!found && obj.name === name) found = obj;
  });
  return found;
}

function wetnessForMaterial(material, wetness, night) {
  if (!material) return;
  const mats = Array.isArray(material) ? material : [material];
  for (const mat of mats) {
    if (mat.roughness !== undefined) {
      mat.roughness = THREE.MathUtils.lerp(0.92, 0.22, wetness);
    }
    if (mat.metalness !== undefined) {
      mat.metalness = THREE.MathUtils.lerp(0.02, 0.08, wetness);
    }
    if (mat.color && /asphalt|truth_road|road/i.test(mat.name || "")) {
      mat.color.setHex(night ? 0x1a2228 : wetness > 0.4 ? 0x242e34 : 0x4a5258);
    }
  }
}

/**
 * Soft ambient-occlusion blob under the car — grounds the vehicle even where
 * the sun shadow falls at a grazing angle.  Built as a unit plane so
 * `sizeContactShadow` can stretch it to the real footprint: a radial gradient
 * scaled non-uniformly reads as a correctly elongated ellipse.
 */
function createContactShadow() {
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(0,0,0,0.62)");
  g.addColorStop(0.45, "rgba(0,0,0,0.28)");
  g.addColorStop(0.75, "rgba(0,0,0,0.07)");
  g.addColorStop(1, "rgba(0,0,0,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;

  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(1, 1),
    new THREE.MeshBasicMaterial({
      map: texture,
      transparent: true,
      opacity: 0.6,
      depthWrite: false,
      depthTest: true,
      fog: true,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    }),
  );
  mesh.renderOrder = 2;
  mesh.name = "contact_shadow";
  mesh.receiveShadow = false;
  mesh.castShadow = false;
  // Local +X runs along the car, local +Y across it — see applyFrame, which
  // lays the plane flat with a fixed rotateX(-π/2).
  mesh.scale.set(5.6, 2.7, 1);
  return mesh;
}

/** Stretch the blob to the vehicle's own footprint plus a little penumbra. */
function sizeContactShadow(mesh, dimensions = {}) {
  if (!mesh) return;
  const length = Number(dimensions.length);
  const width = Number(dimensions.width);
  if (!Number.isFinite(length) || !Number.isFinite(width)) return;
  mesh.scale.set(length * 1.22, width * 1.55, 1);
}

function prepareShadowMaterial(material) {
  const mats = Array.isArray(material) ? material : [material];
  for (const mat of mats) {
    if (!mat) continue;
    mat.shadowSide = THREE.FrontSide;
    if (mat.roughness !== undefined && mat.roughness < 0.15) {
      mat.roughness = Math.max(mat.roughness, 0.28);
    }
  }
}

/** Make rails / curbs / landmarks / verges actually readable beside the car. */
function styleVisualMesh(obj) {
  if (!obj.isMesh || !obj.material) return;
  const name = obj.name || "";
  const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
  for (const mat of mats) {
    if (!mat?.color) continue;
    if (/guardrail/i.test(name)) {
      mat.color.setHex(0xc2cac6);
      if (mat.metalness !== undefined) mat.metalness = 0.78;
      if (mat.roughness !== undefined) mat.roughness = 0.28;
      obj.castShadow = true;
    } else if (/fence/i.test(name)) {
      mat.color.setHex(0x6a7470);
      if (mat.metalness !== undefined) mat.metalness = 0.45;
      if (mat.roughness !== undefined) mat.roughness = 0.5;
    } else if (/curb.*accent/i.test(name)) {
      mat.color.setHex(0xe06028);
      if (mat.roughness !== undefined) mat.roughness = 0.72;
      if (mat.metalness !== undefined) mat.metalness = 0.05;
    } else if (/curb.*light/i.test(name)) {
      mat.color.setHex(0xece8dc);
      if (mat.roughness !== undefined) mat.roughness = 0.78;
    } else if (/drainage/i.test(name)) {
      mat.color.setHex(0x5a6064);
      if (mat.roughness !== undefined) mat.roughness = 0.55;
      if (mat.metalness !== undefined) mat.metalness = 0.35;
    } else if (/landmark|gantry/i.test(name)) {
      mat.color.setHex(0xe4e8dc);
      if (mat.emissive) {
        mat.emissive.setHex(0x2a3320);
        mat.emissiveIntensity = 0.08;
      }
      if (mat.roughness !== undefined) mat.roughness = 0.65;
      obj.castShadow = true;
    } else if (/verge|grass|forest_floor/i.test(name)) {
      mat.color.setHex(0x4a6b3e);
      if (mat.roughness !== undefined) mat.roughness = 0.96;
      if (mat.metalness !== undefined) mat.metalness = 0;
    } else if (/seam_skirt/i.test(name)) {
      mat.color.setHex(0x3f5a38);
      if (mat.roughness !== undefined) mat.roughness = 0.98;
    } else if (/containment/i.test(name)) {
      // Gap filler sitting ~14 m *below* the DGM surface. It was translucent,
      // so instead of only showing through holes it blended over every tile in
      // its 10 km footprint and washed the whole world teal. Opaque means the
      // terrain simply wins the depth test wherever terrain exists.
      mat.color.setHex(0x3f5233);
      if (mat.roughness !== undefined) mat.roughness = 1;
      mat.transparent = false;
      mat.opacity = 1;
      obj.castShadow = false;
      obj.receiveShadow = true;
      obj.userData.containment = true;
      // Gap-filler geometry: inking its silhouette draws a hairline across the
      // whole sky, because the shell extends well past the visible hills.
      obj.userData.noOutline = true;
    }
    prepareShadowMaterial(mat);
  }
}

/**
 * Per-material treatment for the vehicle, matched on the material names the
 * asset generators author (`919_canopy_glass`, `michelin_slick_rubber`,
 * `carbon_brake_disc`, ...). Returns the stylize overrides for that surface, so
 * glass does not get the same banding as bodywork.
 */
function vehicleStyleFor(materialName) {
  const name = String(materialName || "").toLowerCase();
  if (/glass|canopy|polycarbonate|lens|lamp|light/.test(name)) {
    // Glazing: almost no banding, strong rim. Bands on a dark transparent
    // surface just look like dirt.
    return { bands: 2, softness: 0.3, rimStrength: 0.9, rimStart: 0.25 };
  }
  if (/rubber|slick/.test(name)) {
    return { bands: 2, softness: 0.22, rimStrength: 0.18, rimStart: 0.55 };
  }
  if (/carbon|brake|caliper/.test(name)) {
    return { bands: 3, softness: 0.12, rimStrength: 0.3, rimStart: 0.5 };
  }
  if (/magnesium|metal|rim|gold/.test(name)) {
    return { bands: 5, softness: 0.06, rimStrength: 0.55, rimStart: 0.42 };
  }
  return STYLE_VEHICLE;
}

function softenTerrainMaterial(material) {
  const mats = Array.isArray(material) ? material : [material];
  for (const mat of mats) {
    if (!mat) continue;
    if (mat.color) mat.color.setHex(0x3a5536);
    if (mat.roughness !== undefined) mat.roughness = 0.98;
    if (mat.metalness !== undefined) mat.metalness = 0;
    if (mat.flatShading !== undefined) mat.flatShading = false;
    prepareShadowMaterial(mat);
  }
}

/**
 * World stage: truth road, visual scenery, DGM terrain LODs, and the car.
 * Geometry is presentation-only — road height comes from frame.roadZ.
 */
export class WorldStage {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    this.root.name = "world_root";
    scene.add(this.root);
    // Authored world GLBs are Z-up (sim axes). Rotate into Three Y-up so
    // vertex (x,y,z)_sim lands at (x,z,-y)_three — same as simToThree().
    this.ground = new THREE.Group();
    this.ground.name = "sim_ground";
    this.ground.rotation.x = -Math.PI / 2;
    this.root.add(this.ground);
    this.truth = null;
    this.visual = null;
    this.terrainNear = new THREE.Group();
    this.terrainFar = new THREE.Group();
    this.ground.add(this.terrainNear);
    this.ground.add(this.terrainFar);
    this._terrainByKey = new Map();
    this.forest = new ForestField();
    this.root.add(this.forest.root);
    this.rails = new RailPosts();
    this.root.add(this.rails.root);
    this.car = null;
    this.carRoot = new THREE.Group();
    this.carRoot.name = "vehicle_root";
    this.root.add(this.carRoot);
    this.contactShadow = createContactShadow();
    this.carRoot.add(this.contactShadow);
    this._flatShadowQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);
    this._yawQ = new THREE.Quaternion();
    this._upAxis = new THREE.Vector3(0, 1, 0);
    this._shadowOffset = new THREE.Vector3();
    this._invQ = new THREE.Quaternion();
    this.wheels = {};
    this.brakeMaterials = [];
    this._brakeHeat = 0;
    this.attitude = new AttitudeFilter(0.04);
    this.height = new HeightFilter(0.03);
    this._q = new THREE.Quaternion();
    this._p = new THREE.Vector3();
    this._groundOffset = 0.012;
    this.conditionId = "clear-day";
    this.beamGroup = new THREE.Group();
    this.beamGroup.visible = false;
    this.root.add(this.beamGroup);
    this.footprint = null;
    this.manifest = null;
    this.vehicleId = null;
    this.trackMaterial = null;
  }

  async boot(vehicleId = "mazda787b") {
    const worldManifest = await fetch(`${ASSET_ROOT}/world/world.manifest.json`).then((r) => r.json());
    this.manifest = worldManifest;
    const [truthGltf, visualGltf] = await Promise.all([
      loadGltf(`${ASSET_ROOT}/world/world_truth.glb`),
      loadGltf(`${ASSET_ROOT}/world/world_visual.glb`),
    ]);
    this.truth = truthGltf.scene;
    this.visual = visualGltf.scene;
    this.trackMaterial = createTrackMaterial();
    this.truth.traverse((obj) => {
      if (!obj.isMesh) return;
      const name = obj.name || "";
      if (name === "truth_road" || name.includes("truth_road")) {
        // Exact runtime vertices — replace stock PBR with shader skin only.
        obj.material = this.trackMaterial;
        obj.castShadow = false;
        obj.receiveShadow = true;
        return;
      }
      obj.castShadow = false;
      obj.receiveShadow = true;
      if (obj.material) {
        obj.material = Array.isArray(obj.material)
          ? obj.material.map((m) => m.clone())
          : obj.material.clone();
        prepareShadowMaterial(obj.material);
      }
    });
    this.visual.traverse((obj) => {
      if (!obj.isMesh) return;
      obj.castShadow = true;
      obj.receiveShadow = true;
      if (obj.material) {
        obj.material = Array.isArray(obj.material)
          ? obj.material.map((m) => m.clone())
          : obj.material.clone();
      }
      styleVisualMesh(obj);
    });
    // Verges, seam skirts and the containment shell are ground, not props —
    // give them the same landform palette as the DGM tiles or they read as a
    // different material stitched along the track edge.
    this.visual.traverse((obj) => {
      if (!obj.isMesh || !obj.material) return;
      const ground = /verge|seam_skirt|containment|grass|forest_floor/i.test(obj.name || "");
      const list = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of list) stylize(mat, ground ? STYLE_TERRAIN : STYLE_SCENERY);
    });
    this.ground.add(this.truth);
    this.ground.add(this.visual);
    this.rails.buildFromVisual(this.visual);
    stylizeTree(this.rails.root, STYLE_SCENERY);
    await Promise.all([
      this._loadTerrain(worldManifest),
      this.forest.load().catch((err) => {
        console.warn("forest load skipped", err);
      }),
    ]);
    await this.loadVehicle(vehicleId);
    return worldManifest;
  }

  async _loadTerrain(manifest) {
    const levels = manifest?.terrain?.levels || [];
    const jobs = [];
    for (const level of levels) {
      const lod = level.display_distance_m <= 900 ? 0 : 1;
      const group = lod === 0 ? this.terrainNear : this.terrainFar;
      for (const file of level.files || []) {
        const bounds = file.sim_bounds_xy_m || [];
        const [minX, minY, maxX, maxY] = bounds.length >= 4
          ? bounds
          : [0, 0, 0, 0];
        const cx = (minX + maxX) * 0.5;
        const cy = (minY + maxY) * 0.5;
        const halfDiag = Math.hypot(maxX - minX, maxY - minY) * 0.5 || 400;
        const tileKey = Array.isArray(file.tile) ? file.tile.join("_") : file.node || file.path;
        jobs.push(
          loadGltf(`${ASSET_ROOT}/world/${file.path}`).then((gltf) => {
            gltf.scene.traverse((obj) => {
              if (obj.isMesh) {
                obj.receiveShadow = true;
                obj.castShadow = false;
                if (obj.material) {
                  obj.material = Array.isArray(obj.material)
                    ? obj.material.map((m) => m.clone())
                    : obj.material.clone();
                  softenTerrainMaterial(obj.material);
                }
              }
            });
            stylizeTree(gltf.scene, STYLE_TERRAIN);
            group.add(gltf.scene);
            const entry = this._terrainByKey.get(tileKey) || { cx, cy, halfDiag };
            entry.cx = cx;
            entry.cy = cy;
            entry.halfDiag = halfDiag;
            if (lod === 0) entry.near = gltf.scene;
            else entry.far = gltf.scene;
            this._terrainByKey.set(tileKey, entry);
          }).catch(() => {
            /* tile miss is non-fatal; containment shell still covers gaps */
          }),
        );
      }
    }
    await Promise.all(jobs);
  }

  async loadVehicle(vehicleId) {
    // Keep the contact shadow; drop everything else.
    const keep = this.contactShadow;
    while (this.carRoot.children.length) {
      this.carRoot.remove(this.carRoot.children[0]);
    }
    this.wheels = {};
    const manifest = await fetch(`${ASSET_ROOT}/vehicles/${vehicleId}/asset_manifest.json`)
      .then((r) => r.json());
    this.vehicleId = vehicleId;
    this._groundOffset = Number(manifest.visual_ground_contact?.ground_anchor_offset_m ?? 0.012);
    const gltf = await loadGltf(`${ASSET_ROOT}/vehicles/${vehicleId}/${manifest.entry_glb}`);
    this.car = gltf.scene;
    // Authoring is Z-up; Three is Y-up.
    this.car.rotation.x = -Math.PI / 2;
    this.car.traverse((obj) => {
      if (obj.isMesh) {
        obj.castShadow = true;
        obj.receiveShadow = true;
        if (obj.material) prepareShadowMaterial(obj.material);
      }
    });
    this.brakeMaterials = [];
    this.car.traverse((obj) => {
      if (!obj.isMesh || !obj.material) return;
      const list = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of list) {
        stylize(mat, vehicleStyleFor(mat.name));
        if (/brake_disc/i.test(mat.name || "") && mat.emissive) {
          mat.emissive.setHex(0xff5a1e);
          mat.emissiveIntensity = 0;
          this.brakeMaterials.push(mat);
        }
      }
    });
    this.carRoot.add(this.car);
    if (keep) this.carRoot.add(keep);
    else {
      this.contactShadow = createContactShadow();
      this.carRoot.add(this.contactShadow);
    }
    sizeContactShadow(this.contactShadow, manifest.dimensions_m);
    for (const name of manifest.named_nodes?.wheels || []) {
      const node = findNode(this.car, name);
      if (node) this.wheels[name] = node;
    }
    return manifest;
  }

  setCondition(id) {
    const c = condition(id);
    this.conditionId = c.id;
    const night = c.light === "night";
    const dusk = c.light === "dusk";

    setTrackWetness(this.trackMaterial, c.wetness);
    setTrackPalette(this.trackMaterial, {
      dry: night ? 0x1a2228 : dusk ? 0x3a3c42 : 0x4a5258,
      wet: night ? 0x0c1218 : dusk ? 0x1a2228 : 0x1e2830,
      edge: night ? 0xb8b4a8 : 0xece8dc,
      night,
    });

    this.truth?.traverse((obj) => {
      if (!obj.isMesh || obj.material === this.trackMaterial) return;
      // Shoulders: stock materials with wetness response.
      wetnessForMaterial(obj.material, c.wetness, night);
      if (obj.material?.color && /shoulder/i.test(obj.name || obj.material.name || "")) {
        obj.material.color.setHex(
          night ? 0x6f7874 : dusk ? 0x847874 : c.wetness > 0.5 ? 0x6e7468 : 0x8e9084,
        );
        if (obj.material.roughness !== undefined) {
          obj.material.roughness = THREE.MathUtils.lerp(0.96, 0.55, c.wetness);
        }
      }
    });
    this.visual?.traverse((obj) => {
      if (!obj.isMesh || !obj.material?.color) return;
      const name = obj.name || "";
      if (obj.userData.containment) return;
      if (/verge|grass|forest_floor|seam_skirt/i.test(name)) {
        obj.material.color.setHex(night ? 0x1a2a1c : c.wetness > 0.5 ? 0x2f4a30 : 0x3d5b34);
      } else if (/guardrail/i.test(name)) {
        obj.material.color.setHex(night ? 0x6a7270 : 0xc2cac6);
      } else if (/curb.*accent/i.test(name)) {
        obj.material.color.setHex(night ? 0x8a4020 : 0xe06028);
      } else if (/curb.*light/i.test(name)) {
        obj.material.color.setHex(night ? 0x8a8878 : 0xece8dc);
      }
    });
    this.forest?.setCondition({ night, wetness: c.wetness });
    this.rails?.setCondition({ night });
    if (this.contactShadow?.material) {
      this.contactShadow.material.opacity = night ? 0.45 : c.wetness > 0.5 ? 0.7 : 0.95;
    }
    this.terrainNear.visible = true;
    this.terrainFar.visible = c.light !== "night";
  }

  setForestQuality(level) {
    this.forest?.setQuality(level);
    this._quality = level;
    // Fewer light bands on the low tier: the extra steps cost shader work and
    // are the least visible part of the look at reduced resolution.
    const bands = level === "low" ? 2 : level === "medium" ? 3 : 4;
    setStylizeTree(this.car, { bands });
  }

  /**
   * Per-tile DGM LOD: dense near mesh within ~850 m of the car, coarse hinterland
   * beyond. Forest stays on so trees read within chase distance.
   */
  updateSceneryLod(frame, camera) {
    if (!frame?.pose) return;
    const sx = frame.pose.x;
    const sy = frame.pose.y;
    const night = condition(this.conditionId).light === "night";
    // The dense near tiles are the biggest terrain cost; pull them in hard on
    // the low tier and let the coarse level cover more ground instead.
    const nearRange = this._quality === "low" ? 260 : this._quality === "medium" ? 550 : 850;
    // Keep coarse far tiles on while near is active so the LOD handoff overlaps
    // instead of hard-hiding hinterland (which left a sharp horizon seam).
    const farKeep = this._quality === "low" ? 3800 : 4200;
    for (const entry of this._terrainByKey.values()) {
      const d = Math.hypot(sx - entry.cx, sy - entry.cy) - entry.halfDiag;
      const useNear = Boolean(entry.near) && d < nearRange;
      if (entry.near) entry.near.visible = useNear;
      if (entry.far) entry.far.visible = !night && d < farKeep;
    }
    this.forest?.updateLod(this.carRoot.position, camera);
  }

  /** Keep asphalt lit by the same sun the sky uses. */
  syncTrackLighting(sky) {
    if (!this.trackMaterial || !sky?.sun) return;
    const dir = sky._tmp
      ? sky._tmp.copy(sky.sun.position).sub(sky.sun.target.position).normalize()
      : sky.sun.position.clone().sub(sky.sun.target.position).normalize();
    const ambIntensity = THREE.MathUtils.clamp((sky.ambient?.intensity ?? 0.4) * 0.85, 0.28, 0.62);
    const amb = (sky.ambient?.color ?? new THREE.Color(0x6a7880)).clone().multiplyScalar(ambIntensity);
    setTrackLighting(this.trackMaterial, {
      direction: dir,
      color: sky.sun.color,
      intensity: Math.max(0.85, sky.sun.intensity * 0.62),
      ambient: amb,
    });
  }

  setBrainVisible(visible) {
    this.beamGroup.visible = Boolean(visible);
  }

  applyFrame(frame, dt) {
    if (!frame) return;
    vehicleOrientation(frame.pose.yaw, frame.pose.pitch, frame.pose.roll, this._q);
    this.attitude.step(this._q, dt);
    const targetY = this.height.step(frame.roadZ + this._groundOffset, dt);
    simToThree(frame.pose.x, frame.pose.y, frame.pose.z, this._p);
    this._p.y = targetY;
    this.carRoot.position.copy(this._p);
    this.carRoot.quaternion.copy(this.attitude.q);

    // Blob stays flat to world-up (pitch must not lift an edge off the asphalt)
    // but keeps heading, or the elongated ellipse slides out from under the car
    // on every corner.  Child = parent⁻¹ · yaw · flat.
    if (this.contactShadow) {
      this._invQ.copy(this.carRoot.quaternion).invert();
      this._yawQ.setFromAxisAngle(this._upAxis, frame.pose.yaw);
      this.contactShadow.quaternion
        .copy(this._invQ)
        .multiply(this._yawQ)
        .multiply(this._flatShadowQ);
      // Lift along world up, not the body's tilted up.
      this.contactShadow.position.set(0, 0.03, 0).applyQuaternion(this._invQ);
    }

    // Brake discs glow under braking. `brake` is the policy's own control
    // channel, so this is telemetry made visible, not invented drama. Heat
    // bleeds off rather than snapping, which is both truer and less flickery.
    if (this.brakeMaterials?.length) {
      const target = Math.min(1, (frame.brake || 0) * (0.35 + Math.min(1, frame.speedMps / 60) * 0.65));
      this._brakeHeat += (target - this._brakeHeat) * Math.min(1, dt * (target > this._brakeHeat ? 6 : 1.6));
      for (const mat of this.brakeMaterials) mat.emissiveIntensity = this._brakeHeat * 1.6;
    }

    const order = ["wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"];
    frame.wheels.forEach((wheel, index) => {
      const node = this.wheels[order[index]];
      if (!node) return;
      // Wheel-local axes are the authored Z-up set: spin about +Y (the axle),
      // steer about +Z (up).  "ZYX" puts steer outside spin so the wheel spins
      // about its *steered* axle; the default "XYZ" spins about the straight-
      // ahead axle and visibly wobbles at lock.
      node.rotation.set(0, wheel.rotation, wheel.steer, "ZYX");
    });

    if (this.beamGroup.visible) this._drawBeams(frame);
  }

  _drawBeams(frame) {
    while (this.beamGroup.children.length) {
      const child = this.beamGroup.children.pop();
      child.geometry?.dispose?.();
      child.material?.dispose?.();
    }
    const sensors = frame.sensors;
    if (!sensors?.beam_points?.length) return;
    const origin = sensors.beam_origin || [frame.pose.x, frame.pose.y, frame.pose.z];
    const o = simToThree(origin[0], origin[1], origin[2] ?? frame.pose.z);
    const positions = [];
    for (const point of sensors.beam_points) {
      if (!Array.isArray(point) || point.length < 2) continue;
      const tip = simToThree(point[0], point[1], point[2] ?? frame.roadZ);
      positions.push(o.x, o.y, o.z, tip.x, tip.y, tip.z);
    }
    if (!positions.length) return;
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
    const line = new THREE.LineSegments(
      geo,
      new THREE.LineBasicMaterial({ color: 0x5ee1ff, transparent: true, opacity: 0.55 }),
    );
    this.beamGroup.add(line);

    if (Array.isArray(frame.footprint) && frame.footprint.length >= 3) {
      const ring = [];
      for (const xy of frame.footprint) {
        const p = simToThree(xy[0], xy[1], frame.roadZ + 0.05);
        ring.push(p.x, p.y, p.z);
      }
      const first = frame.footprint[0];
      const close = simToThree(first[0], first[1], frame.roadZ + 0.05);
      ring.push(close.x, close.y, close.z);
      const fgeo = new THREE.BufferGeometry();
      fgeo.setAttribute("position", new THREE.Float32BufferAttribute(ring, 3));
      this.beamGroup.add(new THREE.Line(
        fgeo,
        new THREE.LineBasicMaterial({ color: 0xff6a4d, transparent: true, opacity: 0.8 }),
      ));
    }
  }

  snapFilters(frame) {
    if (!frame) return;
    vehicleOrientation(frame.pose.yaw, frame.pose.pitch, frame.pose.roll, this._q);
    this.attitude.reset(this._q);
    this.height.reset(frame.roadZ + this._groundOffset);
  }
}
