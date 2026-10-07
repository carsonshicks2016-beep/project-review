import * as THREE from "three";
import { condition } from "./conditions.js";

function clamp01(v) {
  return Math.min(1, Math.max(0, Number(v) || 0));
}

function smoothstep(edge0, edge1, x) {
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/** Soft valley mist — legacy-style attenuated points, not a hard plane edge. */
function mistSpriteTexture() {
  const size = 64;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0, "rgba(230,240,242,1)");
  g.addColorStop(0.45, "rgba(200,220,222,0.55)");
  g.addColorStop(1, "rgba(180,200,205,0)");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

export class GroundMist {
  constructor(scene, count = 110) {
    this.count = count;
    this.positions = new Float32Array(count * 3);
    this.seeds = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      this.seeds[i * 3] = ((i * 0.754877666) % 1 - 0.5) * 140;
      this.seeds[i * 3 + 1] = (i * 0.569840291) % 1;
      this.seeds[i * 3 + 2] = ((i * 0.438579) % 1 - 0.5) * 140;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    this.material = new THREE.PointsMaterial({
      color: 0xc8d4d5,
      size: 22,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.14,
      depthWrite: false,
      fog: true,
      map: mistSpriteTexture(),
      alphaTest: 0.01,
    });
    this.mesh = new THREE.Points(geometry, this.material);
    this.mesh.name = "visual_localized_ground_mist";
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = -10;
    scene.add(this.mesh);
    this._baseOpacity = 0.14;
    this._elapsed = 0;
  }

  set(id) {
    const c = condition(id);
    const moist = 0.1 + c.wetness * 0.55 + (c.light === "dusk" ? 0.14 : 0) + (c.rain > 0 ? 0.18 : 0);
    this._baseOpacity = Math.min(0.36, Math.max(0.08, 0.08 + moist * 0.3));
    this.material.opacity = this._baseOpacity;
    this.material.size = c.rain > 0 ? 28 : 20;
    this.material.color.setHex(
      c.light === "night" ? 0x8a9aaa : c.light === "dusk" ? 0xb89278 : 0xc8d4d5,
    );
    this.mesh.visible = this._baseOpacity > 0.05;
  }

  update(subject, dt = 0.016) {
    if (!this.mesh.visible) return;
    this._elapsed += dt;
    const t = this._elapsed;
    for (let i = 0; i < this.count; i += 1) {
      const o = i * 3;
      this.positions[o] = subject.x + this.seeds[o] + Math.sin(t * 0.11 + i) * 1.4;
      this.positions[o + 1] = subject.y - 1.2 + this.seeds[o + 1] * 2.8;
      this.positions[o + 2] = subject.z + this.seeds[o + 2] + Math.cos(t * 0.09 + i * 0.7) * 1.2;
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
}

/** Presentation-only rain — camera-local streaks, never a physics input. */
export class RainField {
  constructor(scene, count = 1100) {
    this.count = count;
    this.positions = new Float32Array(count * 6);
    this.seeds = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      this.seeds[i * 3] = ((i * 0.754877666) % 1 - 0.5) * 92;
      this.seeds[i * 3 + 1] = (i * 0.569840291) % 1;
      this.seeds[i * 3 + 2] = ((i * 0.438579) % 1 - 0.5) * 92;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.material = new THREE.LineBasicMaterial({
      color: 0xb8d5e3,
      transparent: true,
      opacity: 0.28,
      depthWrite: false,
      fog: true,
    });
    this.mesh = new THREE.LineSegments(geometry, this.material);
    this.mesh.name = "visual_camera_local_rain";
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    scene.add(this.mesh);
    this.intensity = 0;
    this._light = "day";
    this._elapsed = 0;
  }

  set(id) {
    const c = condition(id);
    this.intensity = c.rain;
    this._light = c.light;
    this.material.opacity = 0.18 + c.rain * 0.22;
    this.material.color.setHex(c.light === "night" ? 0x8aa0b0 : 0xb8d5e3);
    this.mesh.visible = c.rain > 0.05;
  }

  update(cameraPosition, dt) {
    if (!this.mesh.visible) return;
    this._elapsed += dt;
    const rate = this._light === "night" ? 16 : 22;
    const length = this._light === "night" ? 1.45 : 1.15;
    const elapsed = this._elapsed;
    for (let i = 0; i < this.count; i += 1) {
      const seed = i * 3;
      const phase = (this.seeds[seed + 1] + elapsed * rate / 44) % 1;
      const x = cameraPosition.x + this.seeds[seed] + Math.sin(elapsed * 1.7 + i) * 0.7;
      const y = cameraPosition.y + 2 + phase * 38;
      const z = cameraPosition.z + this.seeds[seed + 2] + elapsed * 4.1;
      const o = i * 6;
      this.positions[o] = x;
      this.positions[o + 1] = y;
      this.positions[o + 2] = z;
      this.positions[o + 3] = x - 0.15;
      this.positions[o + 4] = y - length;
      this.positions[o + 5] = z - 0.06;
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
}

/** Visual-only wet-road spray plume from already-rendered pose/speed. */
export function tyreSprayStrength(wetness, speedMps = 0) {
  const w = clamp01(wetness);
  return clamp01(w * smoothstep(14, 42, Number(speedMps) || 0));
}

/**
 * How hard the tyres are working, from the sim's own per-wheel slip.
 * Presentation only — this reads telemetry, it never feeds it.
 */
export function tyreSlipStrength(wheels, speedMps = 0) {
  if (!Array.isArray(wheels) || !wheels.length) return 0;
  let worst = 0;
  for (const wheel of wheels) {
    if (!wheel || wheel.contact === false) continue;
    const ratio = Math.abs(Number(wheel.slipRatio) || 0);
    const angle = Math.abs(Number(wheel.slipAngle) || 0);
    worst = Math.max(worst, ratio, angle * 1.6);
  }
  // Below ~8 m/s a spinning wheel is a standing burnout, not a slide; above it
  // the plume should build with speed.
  return clamp01(smoothstep(0.12, 0.55, worst) * smoothstep(3, 12, Number(speedMps) || 0));
}

export class TyreSpray {
  constructor(scene, count = 110) {
    this.count = count;
    this.seeds = new Float32Array(count * 3);
    this.positions = new Float32Array(count * 3);
    for (let i = 0; i < count; i += 1) {
      this.seeds[i * 3] = ((i * 0.618033989) % 1 - 0.5);
      this.seeds[i * 3 + 1] = (i * 0.381966011) % 1;
      this.seeds[i * 3 + 2] = ((i * 0.754877666) % 1 - 0.5);
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(this.positions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    this.material = new THREE.PointsMaterial({
      color: 0xaebfc0,
      size: 0.06,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      fog: true,
    });
    this.mesh = new THREE.Points(geometry, this.material);
    this.mesh.name = "visual_vehicle_tyre_spray";
    this.mesh.frustumCulled = false;
    this.mesh.visible = false;
    scene.add(this.mesh);
    this.wetness = 0;
    this.rain = 0;
    this._elapsed = 0;
  }

  set(id) {
    const c = condition(id);
    this.wetness = c.wetness;
    this.rain = c.rain;
    this.mesh.visible = c.wetness > 0.28;
  }

  update(vehicle, dt) {
    if (!vehicle) {
      this.mesh.visible = false;
      return;
    }
    // Two sources feed the same plume: standing water when wet, and tyre smoke
    // whenever the sim reports the tyres slipping — so a dry-weather slide
    // still reads as a slide.
    const slip = tyreSlipStrength(vehicle.wheels, vehicle.speedMps);
    const wet = this.wetness > 0.28 ? tyreSprayStrength(this.wetness, vehicle.speedMps) : 0;
    if (wet <= 0 && slip <= 0.02) {
      this.mesh.visible = false;
      return;
    }
    this.mesh.visible = true;
    this._elapsed += dt;
    const strength = Math.max(wet, slip);
    this.material.color.setHex(wet >= slip ? 0xaebfc0 : 0xd8d4cc);
    this.mesh.visible = strength > 0.015;
    if (!this.mesh.visible) return;
    this.material.opacity = 0.04 + strength * (this.rain > 0 ? 0.14 : 0.08);
    this.material.size = 0.035 + strength * 0.08;
    const yaw = Number(vehicle.yaw) || 0;
    const forwardX = Math.cos(yaw);
    const forwardZ = -Math.sin(yaw);
    const sideX = -forwardZ;
    const sideZ = forwardX;
    const originX = vehicle.position.x - forwardX * 1.55;
    const originY = vehicle.position.y;
    const originZ = vehicle.position.z - forwardZ * 1.55;
    const elapsed = this._elapsed;
    for (let i = 0; i < this.count; i += 1) {
      const seed = i * 3;
      const age = (this.seeds[seed + 1] + elapsed * (0.9 + strength * 1.8)) % 1;
      const spread = 0.28 + age * (0.8 + strength * 2.0);
      const lateral = this.seeds[seed] * spread;
      const backward = 0.35 + age * (1.4 + strength * 3.9);
      const rise = 0.08 + age * (0.28 + strength * 0.72) + Math.abs(this.seeds[seed + 2]) * 0.12;
      this.positions[seed] = originX - forwardX * backward + sideX * lateral;
      this.positions[seed + 1] = originY + rise;
      this.positions[seed + 2] = originZ - forwardZ * backward + sideZ * lateral;
    }
    this.mesh.geometry.attributes.position.needsUpdate = true;
  }
}

export class QualityGovernor {
  constructor(renderer, appEl) {
    this.renderer = renderer;
    this.appEl = appEl;
    this.level = "high";
    this.onQuality = null;
    this._samples = [];
    this._last = performance.now();
  }

  get pixelRatio() {
    if (this.level === "low") return 0.75;
    if (this.level === "medium") return 1.0;
    return Math.min(window.devicePixelRatio || 1, 1.75);
  }

  apply() {
    this.renderer.setPixelRatio(this.pixelRatio);
    this.appEl.dataset.quality = this.level;
    // Shadows stay on for medium+ — Phase 2 contact/CSM depends on it.
    this.renderer.shadowMap.enabled = this.level !== "low";
    this.onQuality?.(this.level);
  }

  sample() {
    const now = performance.now();
    const dt = now - this._last;
    this._last = now;
    if (dt <= 0 || dt > 200) return;
    this._samples.push(dt);
    if (this._samples.length < 90) return;
    const sorted = [...this._samples].sort((a, b) => a - b);
    const p95 = sorted[Math.floor(sorted.length * 0.95)];
    this._samples.length = 0;
    let next = this.level;
    if (p95 > 28) next = this.level === "high" ? "medium" : "low";
    else if (p95 < 14 && this.level !== "high") next = this.level === "low" ? "medium" : "high";
    if (next !== this.level) {
      this.level = next;
      this.apply();
    }
  }
}
