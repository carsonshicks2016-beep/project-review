import * as THREE from "three";
import { Sky } from "three/addons/objects/Sky.js";
import { resolveVisualCondition } from "./visual-conditions.js";

/**
 * Eifel Broadcast atmosphere — cool, moist European forest race cinema.
 * Presets push depth (fog + haze), softer sun discs, and denser night skies
 * without changing any physics / track-truth contract.
 */
const PRESETS = Object.freeze({
  day: {
    elevation: 42,
    azimuth: 214,
    turbidity: 5.2,
    rayleigh: 1.55,
    mieCoefficient: 0.0042,
    mieDirectionalG: 0.74,
    fog: 0x9aafb4,
    fogDensity: 0.00155,
    sunColor: 0xfff0d4,
    sunIntensity: 2.45,
    hemiSky: 0xd0e4ea,
    hemiGround: 0x2a3828,
    hemiIntensity: 1.48,
    exposure: 0.92,
    bloom: 0.04,
    hazeOpacity: 0.34,
    hazeColor: 0xc5d5d8,
  },
  dusk: {
    elevation: 6.2,
    azimuth: 242,
    turbidity: 10.5,
    rayleigh: 3.1,
    mieCoefficient: 0.016,
    mieDirectionalG: 0.9,
    fog: 0x6a5560,
    fogDensity: 0.00215,
    sunColor: 0xff8a55,
    sunIntensity: 2.05,
    hemiSky: 0x9a7f8f,
    hemiGround: 0x1a1618,
    hemiIntensity: 1.12,
    exposure: 0.86,
    bloom: 0.28,
    hazeOpacity: 0.48,
    hazeColor: 0xb88878,
  },
  night: {
    elevation: -18,
    azimuth: 210,
    turbidity: 2.4,
    rayleigh: 0.18,
    mieCoefficient: 0.0008,
    mieDirectionalG: 0.68,
    fog: 0x0e1824,
    fogDensity: 0.00165,
    sunColor: 0x8eacc8,
    sunIntensity: 0.42,
    hemiSky: 0x3d5674,
    hemiGround: 0x121a18,
    hemiIntensity: 1.05,
    exposure: 1.05,
    bloom: 0.32,
    hazeOpacity: 0.22,
    hazeColor: 0x1a2838,
  },
});

function radialTexture(inner = "rgba(255,255,242,1)", mid = "rgba(255,210,150,.78)", outer = "rgba(255,140,50,0)") {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(128, 128, 2, 128, 128, 126);
  gradient.addColorStop(0, inner);
  gradient.addColorStop(0.14, mid);
  gradient.addColorStop(0.48, "rgba(255,160,90,.18)");
  gradient.addColorStop(1, outer);
  context.fillStyle = gradient;
  context.fillRect(0, 0, 256, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function mistGeometry(count = 110) {
  const positions = new Float32Array(count * 3);
  const seeds = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    seeds[index * 3] = ((index * 0.754877666) % 1 - 0.5) * 140;
    seeds[index * 3 + 1] = (index * 0.569840291) % 1;
    seeds[index * 3 + 2] = ((index * 0.438579) % 1 - 0.5) * 140;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
  return { geometry, positions, seeds };
}

function starsGeometry(count = 2800) {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const a = index * 12.9898;
    const b = index * 78.233;
    const u = Math.abs(Math.sin(a) * 43758.5453) % 1;
    const v = Math.abs(Math.sin(b) * 24634.6345) % 1;
    const theta = u * Math.PI * 2;
    const phi = Math.acos(THREE.MathUtils.clamp(v * 1.65 - 0.65, -1, 1));
    const radius = 2200 + (index % 23) * 9;
    positions[index * 3] = Math.sin(phi) * Math.cos(theta) * radius;
    positions[index * 3 + 1] = Math.max(70, Math.cos(phi) * radius);
    positions[index * 3 + 2] = Math.sin(phi) * Math.sin(theta) * radius;
    const warmth = 0.72 + (index % 9) * 0.028;
    const cool = index % 5 === 0;
    colors[index * 3] = cool ? 0.72 : 0.82 + warmth * 0.18;
    colors[index * 3 + 1] = cool ? 0.84 : 0.86 + warmth * 0.12;
    colors[index * 3 + 2] = cool ? 1 : 0.94 + warmth * 0.06;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

function makeHorizonHaze() {
  const geometry = new THREE.PlaneGeometry(5200, 900, 1, 1);
  const material = new THREE.MeshBasicMaterial({
    color: 0xc5d5d8,
    transparent: true,
    opacity: 0.34,
    depthWrite: false,
    fog: false,
    side: THREE.DoubleSide,
    blending: THREE.NormalBlending,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.name = "observatory_horizon_haze";
  mesh.frustumCulled = false;
  mesh.renderOrder = -2;
  return mesh;
}

export class AtmosphereRig {
  constructor(scene) {
    this.scene = scene;
    this.sky = new Sky();
    this.sky.name = "observatory_procedural_sky";
    this.sky.scale.setScalar(450000);
    this.sky.frustumCulled = false;
    this.scene.add(this.sky);

    this.sunDirection = new THREE.Vector3();
    this.sunDisc = new THREE.Sprite(new THREE.SpriteMaterial({
      map: radialTexture(),
      color: 0xffd5a0,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      opacity: 0.78,
    }));
    this.sunDisc.name = "observatory_sun_disc";
    this.sunDisc.scale.set(240, 240, 1);
    this.scene.add(this.sunDisc);

    this.sunGlow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: radialTexture("rgba(255,230,200,0.9)", "rgba(255,180,110,.35)", "rgba(255,120,40,0)"),
      color: 0xffc89a,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: THREE.AdditiveBlending,
      opacity: 0.35,
    }));
    this.sunGlow.name = "observatory_sun_glow";
    this.sunGlow.scale.set(520, 520, 1);
    this.scene.add(this.sunGlow);

    this.horizonHaze = makeHorizonHaze();
    this.scene.add(this.horizonHaze);

    this.stars = new THREE.Points(
      starsGeometry(),
      new THREE.PointsMaterial({
        size: 2.4,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0,
        vertexColors: true,
        depthWrite: false,
        fog: false,
        blending: THREE.AdditiveBlending,
      }),
    );
    this.stars.name = "observatory_night_stars";
    this.stars.frustumCulled = false;
    this.scene.add(this.stars);

    const mist = mistGeometry();
    this.mistPositions = mist.positions;
    this.mistSeeds = mist.seeds;
    this.mist = new THREE.Points(mist.geometry, new THREE.PointsMaterial({
      color: 0xc8d4d5,
      size: 22,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      fog: true,
      map: radialTexture("rgba(230,240,242,1)", "rgba(200,220,222,.55)", "rgba(180,200,205,0)"),
      alphaTest: 0.01,
    }));
    this.mist.name = "visual_localized_ground_mist";
    this.mist.frustumCulled = false;
    this.mist.visible = false;
    this.scene.add(this.mist);

    this.mode = "day";
    this.condition = resolveVisualCondition("clear-day");
    this._hazeTarget = new THREE.Vector3();
    this.setVisualCondition("clear-day");
  }

  setVisualCondition(condition) {
    this.condition = resolveVisualCondition(condition);
    this.mode = this.condition.lighting;
    const preset = PRESETS[this.mode];
    const phi = THREE.MathUtils.degToRad(90 - preset.elevation);
    const theta = THREE.MathUtils.degToRad(preset.azimuth);
    this.sunDirection.setFromSphericalCoords(1, phi, theta).normalize();
    const uniforms = this.sky.material.uniforms;
    uniforms.turbidity.value = preset.turbidity;
    uniforms.rayleigh.value = preset.rayleigh;
    uniforms.mieCoefficient.value = preset.mieCoefficient;
    uniforms.mieDirectionalG.value = preset.mieDirectionalG;
    uniforms.sunPosition.value.copy(this.sunDirection);

    const rain = this.condition.precipitation === "rain";
    const overcast = this.condition.precipitation === "overcast";

    let fogHex = preset.fog;
    let fogDensity = preset.fogDensity;
    let hazeOpacity = preset.hazeOpacity;
    let hazeHex = preset.hazeColor;
    let turbidity = preset.turbidity;
    let rayleigh = preset.rayleigh;
    let sunOpacity = this.mode === "dusk" ? 0.95 : 0.72;
    let sunIntensityScale = 1;

    if (rain) {
      fogHex = this.mode === "night" ? 0x0c141c : this.mode === "dusk" ? 0x5a5058 : 0x7a8a90;
      fogDensity *= 1.85;
      hazeOpacity = Math.min(0.62, hazeOpacity + 0.18);
      hazeHex = this.mode === "night" ? 0x15202c : 0x9aabaf;
      turbidity += 4;
      rayleigh *= 0.72;
      sunOpacity = 0.14;
      sunIntensityScale = 0.55;
    } else if (overcast) {
      fogHex = this.mode === "night" ? 0x121c28 : 0x8a9aa0;
      fogDensity *= 1.32;
      hazeOpacity = Math.min(0.55, hazeOpacity + 0.12);
      turbidity += 6;
      rayleigh *= 0.85;
      sunOpacity = 0.28;
      sunIntensityScale = 0.7;
    }

    uniforms.turbidity.value = turbidity;
    uniforms.rayleigh.value = rayleigh;

    this.scene.fog.color.setHex(fogHex);
    this.scene.fog.density = fogDensity;
    this.horizonHaze.material.color.setHex(hazeHex);
    this.horizonHaze.material.opacity = hazeOpacity;

    this.stars.material.opacity = this.mode === "night"
      ? (rain ? 0.14 : 0.88)
      : this.mode === "dusk" ? 0.1 : 0;

    this.sunDisc.visible = this.mode !== "night";
    this.sunGlow.visible = this.mode !== "night";
    this.sunDisc.material.color.setHex(preset.sunColor);
    this.sunGlow.material.color.setHex(preset.sunColor);
    this.sunDisc.material.opacity = sunOpacity;
    this.sunGlow.material.opacity = sunOpacity * (this.mode === "dusk" ? 0.55 : 0.38);

    const mistStrength = rain
      ? 0.52
      : this.condition.wetness > 0.5 && this.mode !== "day"
        ? 0.3
        : overcast && this.mode === "dusk"
          ? 0.18
          : 0;
    this.mist.visible = mistStrength > 0;
    this.mist.material.opacity = mistStrength;
    this.mist.material.size = rain ? 28 : 20;
    this.mist.material.color.setHex(this.mode === "night" ? 0x8a9aaa : 0xc8d4d5);

    return {
      ...preset,
      ...this.condition,
      sunIntensity: preset.sunIntensity * sunIntensityScale,
      fog: fogHex,
      fogDensity,
    };
  }

  setPreset(mode) {
    return this.setVisualCondition(mode);
  }

  update(cameraPosition, elapsed = 0) {
    this.stars.position.copy(cameraPosition);
    this.sunDisc.position.copy(cameraPosition).addScaledVector(this.sunDirection, 1600);
    this.sunGlow.position.copy(cameraPosition).addScaledVector(this.sunDirection, 1550);

    // Soft horizon band sits ahead of the camera, low in the view.
    const forward = this._hazeTarget;
    forward.set(-this.sunDirection.x, 0, -this.sunDirection.z);
    if (forward.lengthSq() < 1e-6) forward.set(0, 0, -1);
    forward.normalize();
    this.horizonHaze.position.copy(cameraPosition);
    this.horizonHaze.position.y = cameraPosition.y + (this.mode === "night" ? 40 : 28);
    this.horizonHaze.position.addScaledVector(forward, 900);
    this.horizonHaze.lookAt(cameraPosition.x, this.horizonHaze.position.y, cameraPosition.z);

    if (!this.mist.visible) return;
    for (let index = 0; index < this.mistSeeds.length / 3; index += 1) {
      const offset = index * 3;
      this.mistPositions[offset] = cameraPosition.x + this.mistSeeds[offset]
        + Math.sin(index * 1.7 + elapsed * 0.1) * 4;
      this.mistPositions[offset + 1] = cameraPosition.y - 3.2 + this.mistSeeds[offset + 1] * 2.8;
      this.mistPositions[offset + 2] = cameraPosition.z + this.mistSeeds[offset + 2]
        + Math.cos(index * 1.1 + elapsed * 0.08) * 2;
    }
    this.mist.geometry.attributes.position.needsUpdate = true;
  }
}

export function atmospherePreset(mode) {
  return PRESETS[resolveVisualCondition(mode).lighting] || PRESETS.day;
}
