/**
 * WorldView — everything renderable for a single WorldSpec, parented under one
 * Group so it can be crossfaded in/out as a unit on track change.
 *
 *  - Terrain: a heightfield PlaneGeometry displaced by layered simplex noise
 *    (octaves/amplitude/frequency/seed from TerrainParams).
 *  - Scatter: GPU InstancedMesh per "kind" (one draw call each).
 *  - Particles: an instanced/point cloud whose count + brightness react to
 *    the live AudioFrame via WorldSpec.modifierCurves.
 *  - Weather (Agent E) and Creatures (Agent F): mounted behind their exported
 *    interfaces; guarded with try/catch so the scene still renders if their
 *    factories throw (they're built on other branches).
 */
import * as THREE from "three";
import { SimplexNoise } from "three/examples/jsm/math/SimplexNoise.js";
import type { WorldSpec, AudioFrame, ScatterItem } from "../contracts";
import {
  createWeatherSystem,
  type WeatherSystem,
} from "../weather/weatherSystem";
import {
  createCreatureSystem,
  type CreatureSystem,
} from "../creatures/creatureSystem";
import { resolvePalette, remap, daylight, type ResolvedPalette } from "./palette";

const TERRAIN_SIZE = 220;
const TERRAIN_SEGMENTS = 160;
const MAX_PARTICLES = 1500;

/** A seeded PRNG (mulberry32) so scatter/particle jitter is deterministic. */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class WorldView {
  readonly group = new THREE.Group();
  readonly spec: WorldSpec;
  readonly palette: ResolvedPalette;

  private noise: SimplexNoise;
  private disposables: { dispose: () => void }[] = [];
  private materials: THREE.Material[] = [];

  private terrain!: THREE.Mesh;
  private particles!: THREE.Points;
  private particleMat!: THREE.PointsMaterial;
  private particleBase: Float32Array;
  private particleSeeds: Float32Array;

  private weather: WeatherSystem | null = null;
  private creatures: CreatureSystem | null = null;

  /** 0 = fully faded out, 1 = fully present. Drives crossfades. */
  private opacity = 1;

  constructor(spec: WorldSpec) {
    this.spec = spec;
    this.palette = resolvePalette(spec.palette, spec.timeOfDay);
    this.noise = new SimplexNoise(mulberry32(spec.terrain.seed >>> 0));
    this.particleBase = new Float32Array(MAX_PARTICLES * 3);
    this.particleSeeds = new Float32Array(MAX_PARTICLES);

    this.buildTerrain();
    this.buildScatter();
    this.buildParticles();
    this.mountSubsystems();
  }

  /** Layered simplex height at world (x, z). */
  private heightAt(x: number, z: number): number {
    const { octaves, amplitude, frequency } = this.spec.terrain;
    let h = 0;
    let amp = 1;
    let freq = frequency;
    let norm = 0;
    for (let o = 0; o < Math.max(1, octaves); o++) {
      h += this.noise.noise(x * freq, z * freq) * amp;
      norm += amp;
      amp *= 0.5;
      freq *= 2;
    }
    return (h / (norm || 1)) * amplitude;
  }

  private buildTerrain(): void {
    const geo = new THREE.PlaneGeometry(
      TERRAIN_SIZE,
      TERRAIN_SIZE,
      TERRAIN_SEGMENTS,
      TERRAIN_SEGMENTS,
    );
    geo.rotateX(-Math.PI / 2); // lie flat on XZ
    const pos = geo.attributes.position as THREE.BufferAttribute;
    const colors = new Float32Array(pos.count * 3);
    const low = this.palette.secondary.clone().multiplyScalar(0.7);
    const high = this.palette.primary.clone();
    const c = new THREE.Color();
    for (let i = 0; i < pos.count; i++) {
      const x = pos.getX(i);
      const z = pos.getZ(i);
      const y = this.heightAt(x, z);
      pos.setY(i, y);
      const t = THREE.MathUtils.clamp(
        y / (this.spec.terrain.amplitude || 1) * 0.5 + 0.5,
        0,
        1,
      );
      c.copy(low).lerp(high, t);
      colors[i * 3] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }
    geo.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    geo.computeVertexNormals();

    const mat = new THREE.MeshStandardMaterial({
      vertexColors: true,
      roughness: 0.95,
      metalness: 0.0,
      flatShading: false,
      transparent: true,
      opacity: 1,
    });
    this.materials.push(mat);
    const mesh = new THREE.Mesh(geo, mat);
    mesh.receiveShadow = true;
    this.terrain = mesh;
    this.group.add(mesh);
    this.disposables.push(geo);
  }

  private buildScatter(): void {
    // Group scatter items by kind → one InstancedMesh (one draw call) each.
    const byKind = new Map<string, ScatterItem[]>();
    for (const item of this.spec.scatter) {
      const arr = byKind.get(item.kind) ?? [];
      arr.push(item);
      byKind.set(item.kind, arr);
    }

    const dummy = new THREE.Object3D();
    for (const [kind, items] of byKind) {
      const { geometry, material } = this.scatterPrototype(kind);
      this.materials.push(material);
      this.disposables.push(geometry);
      const inst = new THREE.InstancedMesh(geometry, material, items.length);
      inst.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      items.forEach((item, i) => {
        const y = this.heightAt(item.x, item.z);
        dummy.position.set(item.x, y, item.z);
        dummy.rotation.set(0, item.rotation, 0);
        dummy.scale.setScalar(item.scale);
        dummy.updateMatrix();
        inst.setMatrixAt(i, dummy.matrix);
      });
      inst.instanceMatrix.needsUpdate = true;
      inst.castShadow = true;
      this.group.add(inst);
    }
  }

  /** Cheap stand-in geometry/material per scatter kind. */
  private scatterPrototype(kind: string): {
    geometry: THREE.BufferGeometry;
    material: THREE.Material;
  } {
    const opts: THREE.MeshStandardMaterialParameters = {
      transparent: true,
      opacity: 1,
      roughness: 0.85,
    };
    switch (kind) {
      case "pine":
        return {
          geometry: new THREE.ConeGeometry(1.1, 4.5, 7),
          material: new THREE.MeshStandardMaterial({
            ...opts,
            color: this.palette.primary.clone().multiplyScalar(0.8),
          }),
        };
      case "neon-pillar": {
        const m = new THREE.MeshStandardMaterial({
          ...opts,
          color: this.palette.accent,
          emissive: this.palette.accent,
          emissiveIntensity: 1.4,
        });
        return { geometry: new THREE.BoxGeometry(0.6, 8, 0.6), material: m };
      }
      case "shrub":
        return {
          geometry: new THREE.IcosahedronGeometry(1.0, 0),
          material: new THREE.MeshStandardMaterial({
            ...opts,
            color: this.palette.primary.clone().lerp(this.palette.secondary, 0.4),
          }),
        };
      case "rock":
      default:
        return {
          geometry: new THREE.DodecahedronGeometry(0.9, 0),
          material: new THREE.MeshStandardMaterial({
            ...opts,
            color: this.palette.secondary.clone().multiplyScalar(0.9),
            roughness: 1,
          }),
        };
    }
  }

  private buildParticles(): void {
    const rng = mulberry32((this.spec.seed ^ 0x9e3779b9) >>> 0);
    const positions = new Float32Array(MAX_PARTICLES * 3);
    for (let i = 0; i < MAX_PARTICLES; i++) {
      const x = (rng() * 2 - 1) * TERRAIN_SIZE * 0.5;
      const z = (rng() * 2 - 1) * TERRAIN_SIZE * 0.5;
      const y = this.heightAt(x, z) + 2 + rng() * 30;
      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
      this.particleBase[i * 3] = x;
      this.particleBase[i * 3 + 1] = y;
      this.particleBase[i * 3 + 2] = z;
      this.particleSeeds[i] = rng() * Math.PI * 2;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geo.setDrawRange(0, Math.floor(MAX_PARTICLES * 0.4));
    this.disposables.push(geo);

    const mat = new THREE.PointsMaterial({
      color: this.palette.accent,
      size: 0.6,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
      sizeAttenuation: true,
    });
    this.materials.push(mat);
    this.particleMat = mat;
    this.particles = new THREE.Points(geo, mat);
    this.group.add(this.particles);
  }

  private mountSubsystems(): void {
    // Both systems are built on other branches; their factories may throw.
    // Guard so the scene still renders.
    try {
      const w = createWeatherSystem();
      w.init(this.group as unknown as THREE.Scene, this.spec);
      this.weather = w;
    } catch {
      this.weather = null;
    }
    try {
      const c = createCreatureSystem();
      c.init(this.group as unknown as THREE.Scene, this.spec);
      this.creatures = c;
    } catch {
      this.creatures = null;
    }
  }

  /** Apply WorldSpec.modifierCurves from the live (or synthetic) AudioFrame. */
  update(dt: number, elapsed: number, audio: AudioFrame): void {
    const curves = this.spec.modifierCurves;

    // centroid → brightness: scale particle + emissive presence.
    const brightness = remap(audio.centroid, curves.centroidToBrightness);
    this.particleMat.opacity = THREE.MathUtils.clamp(
      0.35 * brightness * this.opacity,
      0,
      1,
    );

    // energy (rms) → particle count (draw range) + drift speed.
    const density = remap(audio.rms, curves.energyToParticles);
    const count = THREE.MathUtils.clamp(
      Math.floor(MAX_PARTICLES * THREE.MathUtils.clamp(density / 2.2, 0.05, 1)),
      0,
      MAX_PARTICLES,
    );
    this.particles.geometry.setDrawRange(0, count);

    // Animate particles (gentle float). bass nudges them upward = "pulse".
    const pos = this.particles.geometry.attributes.position as THREE.BufferAttribute;
    const pulse = audio.bass * 4;
    for (let i = 0; i < count; i++) {
      const ph = this.particleSeeds[i]!;
      const bx = this.particleBase[i * 3]!;
      const by = this.particleBase[i * 3 + 1]!;
      const bz = this.particleBase[i * 3 + 2]!;
      const drift = elapsed * (0.4 + audio.mid * 0.8);
      pos.setX(i, bx + Math.sin(drift + ph) * 1.5);
      pos.setY(i, by + Math.sin(drift * 0.7 + ph) * 1.0 + pulse);
      pos.setZ(i, bz + Math.cos(drift * 0.6 + ph) * 1.5);
    }
    pos.needsUpdate = true;

    // bass → lightning hook: a quick emissive lift on terrain when a strong
    // bass transient lands (the weather system owns real lightning bolts).
    const lightning = remap(audio.bass, curves.bassToLightning);
    const tm = this.terrain.material as THREE.MeshStandardMaterial;
    const flash = audio.onset ? lightning * 0.4 : 0;
    tm.emissive.copy(this.palette.accent);
    tm.emissiveIntensity = THREE.MathUtils.lerp(
      tm.emissiveIntensity,
      flash,
      0.3,
    );

    if (this.weather) {
      try {
        this.weather.update(dt, audio);
      } catch {
        this.weather = null;
      }
    }
    if (this.creatures) {
      try {
        this.creatures.update(dt, audio);
      } catch {
        this.creatures = null;
      }
    }
  }

  /** Crossfade control: 0..1 presence. Applied to every owned material. */
  setOpacity(o: number): void {
    this.opacity = THREE.MathUtils.clamp(o, 0, 1);
    for (const m of this.materials) {
      (m as THREE.Material & { opacity: number }).opacity = this.opacity;
    }
    this.group.visible = this.opacity > 0.001;
  }

  /** A target camera framing hint derived from the biome (gentle variation). */
  get cameraHeight(): number {
    return 14 + this.spec.terrain.amplitude * 0.6;
  }

  dispose(): void {
    if (this.weather) {
      try {
        this.weather.dispose();
      } catch {
        /* ignore */
      }
    }
    if (this.creatures) {
      try {
        this.creatures.dispose();
      } catch {
        /* ignore */
      }
    }
    for (const d of this.disposables) d.dispose();
    for (const m of this.materials) m.dispose();
    this.group.removeFromParent();
  }

  /** Sun elevation factor for the directional light, from time of day. */
  get day(): number {
    return daylight(this.spec.timeOfDay);
  }
}
