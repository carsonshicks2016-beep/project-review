import * as THREE from "three";
import { Sky } from "three/addons/objects/Sky.js";

const PRESETS = Object.freeze({
  dry: {
    elevation: 38,
    azimuth: 218,
    turbidity: 3.8,
    rayleigh: 1.35,
    mieCoefficient: 0.0025,
    mieDirectionalG: 0.78,
    fog: 0x8fa6ac,
    fogDensity: 0.0013,
    sunColor: 0xfff2d1,
    sunIntensity: 2.8,
    hemiSky: 0xcbe3ec,
    hemiGround: 0x243325,
    hemiIntensity: 1.32,
    exposure: 0.88,
    bloom: 0.0,
  },
  dusk: {
    elevation: 7.5,
    azimuth: 238,
    turbidity: 8.5,
    rayleigh: 2.8,
    mieCoefficient: 0.012,
    mieDirectionalG: 0.88,
    fog: 0x7c6670,
    fogDensity: 0.0019,
    sunColor: 0xff9b68,
    sunIntensity: 2.15,
    hemiSky: 0x8f8197,
    hemiGround: 0x211d22,
    hemiIntensity: 1.05,
    exposure: 0.84,
    bloom: 0.18,
  },
  night: {
    elevation: -16,
    azimuth: 214,
    turbidity: 3,
    rayleigh: 0.25,
    mieCoefficient: 0.001,
    mieDirectionalG: 0.72,
    fog: 0x0b1724,
    fogDensity: 0.00165,
    sunColor: 0x9ab8db,
    sunIntensity: 0.32,
    hemiSky: 0x304b70,
    hemiGround: 0x111716,
    hemiIntensity: 0.72,
    exposure: 0.82,
    bloom: 0.12,
  },
});

function radialTexture() {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const context = canvas.getContext("2d");
  const gradient = context.createRadialGradient(128, 128, 2, 128, 128, 126);
  gradient.addColorStop(0, "rgba(255,255,242,1)");
  gradient.addColorStop(0.12, "rgba(255,222,165,.88)");
  gradient.addColorStop(0.42, "rgba(255,168,94,.22)");
  gradient.addColorStop(1, "rgba(255,120,40,0)");
  context.fillStyle = gradient;
  context.fillRect(0, 0, 256, 256);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

function starsGeometry(count = 1800) {
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  for (let index = 0; index < count; index += 1) {
    const a = index * 12.9898;
    const b = index * 78.233;
    const u = Math.abs(Math.sin(a) * 43758.5453) % 1;
    const v = Math.abs(Math.sin(b) * 24634.6345) % 1;
    const theta = u * Math.PI * 2;
    const phi = Math.acos(THREE.MathUtils.clamp(v * 1.65 - 0.65, -1, 1));
    const radius = 2200 + (index % 19) * 7;
    positions[index * 3] = Math.sin(phi) * Math.cos(theta) * radius;
    positions[index * 3 + 1] = Math.max(90, Math.cos(phi) * radius);
    positions[index * 3 + 2] = Math.sin(phi) * Math.sin(theta) * radius;
    const warmth = 0.78 + (index % 7) * 0.03;
    colors[index * 3] = 0.72 + warmth * 0.25;
    colors[index * 3 + 1] = 0.80 + warmth * 0.16;
    colors[index * 3 + 2] = 1;
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  return geometry;
}

export class AtmosphereRig {
  constructor(scene) {
    this.scene = scene;
    this.sky = new Sky();
    this.sky.name = "observatory_procedural_sky";
    this.sky.scale.setScalar(350000);
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
      opacity: 0.82,
    }));
    this.sunDisc.name = "observatory_sun_disc";
    this.sunDisc.scale.set(190, 190, 1);
    this.scene.add(this.sunDisc);

    this.stars = new THREE.Points(
      starsGeometry(),
      new THREE.PointsMaterial({
        size: 2.15,
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
    this.mode = "dry";
    this.setPreset("dry");
  }

  setPreset(mode) {
    this.mode = PRESETS[mode] ? mode : "dry";
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
    this.stars.material.opacity = this.mode === "night" ? 0.78 : this.mode === "dusk" ? 0.08 : 0;
    this.sunDisc.visible = this.mode !== "night";
    this.sunDisc.material.color.setHex(preset.sunColor);
    this.sunDisc.material.opacity = this.mode === "dusk" ? 0.95 : 0.72;
    return preset;
  }

  update(cameraPosition) {
    this.stars.position.copy(cameraPosition);
    this.sunDisc.position.copy(cameraPosition).addScaledVector(this.sunDirection, 1450);
  }
}

export function atmospherePreset(mode) {
  return PRESETS[mode] || PRESETS.dry;
}
