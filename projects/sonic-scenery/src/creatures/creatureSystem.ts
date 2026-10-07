/**
 * Agent F — Creature system.  Brief: docs/tasks/agent-F-creatures.md
 *
 *  - Reynolds boids (birds/fish/fireflies) + wander agents (ground) + drift.
 *  - GPU-instanced; count = f(energy, seed), speed = f(tempo/energy).
 *  - Onset reactions (bursts/scatter) from live AudioFrame.
 *
 * One InstancedMesh per CreatureSpec. Simulation runs on flat Float32Array
 * buffers (see flock.ts) and is written into instance matrices each frame —
 * no per-agent object allocation in the hot loop. When `audio` is undefined
 * (helper offline) the system falls back to calm ambient motion.
 */
import * as THREE from "three";
import type { WorldSpec, AudioFrame, CreatureSpec } from "../contracts";
import { createFlock, type Flock } from "./flock";
import { stepBoids, DEFAULT_BOID_PARAMS } from "./boids";
import { stepWander, DEFAULT_WANDER_PARAMS } from "./wander";
import { stepDrift, DEFAULT_DRIFT_PARAMS } from "./drift";

export interface CreatureSystem {
  init(scene: THREE.Scene, world: WorldSpec): void;
  update(dt: number, audio?: AudioFrame): void;
  dispose(): void;
}

/** Performance ceiling per species so a dense WorldSpec can't blow past 60fps. */
const MAX_INSTANCES_PER_SPECIES = 2000;

interface SpeciesGroup {
  spec: CreatureSpec;
  flock: Flock;
  mesh: THREE.InstancedMesh;
  /** Reused per frame to compose instance matrices. */
  baseScale: number;
}

const _mat = new THREE.Matrix4();
const _quat = new THREE.Quaternion();
const _scale = new THREE.Vector3();
const _pos = new THREE.Vector3();
const _vel = new THREE.Vector3();
const _up = new THREE.Vector3(0, 1, 0);
const _fwd = new THREE.Vector3();

export function createCreatureSystem(): CreatureSystem {
  let groups: SpeciesGroup[] = [];
  let sceneRef: THREE.Scene | null = null;

  /** Smoothed agitation so onset spikes decay instead of snapping back. */
  let agitation = 0;

  return {
    init(scene: THREE.Scene, world: WorldSpec): void {
      sceneRef = scene;
      groups = [];

      // Containment box scales with terrain amplitude; flocks fly above ground.
      const span = Math.max(60, world.terrain.amplitude * 6 + 80);
      const skyBounds = new THREE.Vector3(span * 0.5, 24, span * 0.5);
      const groundBounds = new THREE.Vector3(span * 0.5, 1, span * 0.5);
      const skyCenter = new THREE.Vector3(0, 28, 0);
      const groundCenter = new THREE.Vector3(0, 1.2, 0);

      const accent = new THREE.Color(safeColor(world.palette.accent, "#ffffff"));
      const primary = new THREE.Color(safeColor(world.palette.primary, "#cccccc"));

      for (const spec of world.creaturePool) {
        const count = Math.max(0, Math.min(spec.count | 0, MAX_INSTANCES_PER_SPECIES));
        if (count === 0) continue;

        const isGround = spec.behavior === "wander";
        const bounds = isGround ? groundBounds : skyBounds;
        const center = isGround ? groundCenter : skyCenter;

        const flock = createFlock(count, spec.baseSpeed, center, bounds);
        const { geometry, baseScale, color, emissive } = makeSpeciesAppearance(
          spec,
          accent,
          primary,
        );

        const material = new THREE.MeshStandardMaterial({
          color,
          emissive,
          emissiveIntensity: emissive.getHex() === 0 ? 0 : 1.2,
          roughness: 0.6,
          metalness: 0.0,
        });

        const mesh = new THREE.InstancedMesh(geometry, material, count);
        mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        mesh.frustumCulled = false; // agents roam the whole box
        mesh.name = `creatures:${spec.species}`;
        mesh.count = count;

        const group: SpeciesGroup = { spec, flock, mesh, baseScale };
        writeMatrices(group); // seed initial transforms
        scene.add(mesh);
        groups.push(group);
      }
    },

    update(dt: number, audio?: AudioFrame): void {
      // Clamp dt so a stutter / tab-restore can't fling agents out of bounds.
      const step = Math.min(Math.max(dt, 0), 0.05);

      // Drive agitation from live audio; decay smoothly toward calm.
      if (audio) {
        const energy = clamp01(audio.rms * 0.6 + audio.bass * 0.4);
        const target = audio.onset ? 1 : energy * 0.4;
        // Onsets jump agitation up instantly; otherwise ease toward target.
        agitation = Math.max(target, agitation);
      }
      // Exponential decay (~0.4s to settle) regardless of dt.
      agitation *= Math.exp(-3.0 * step);
      if (agitation < 1e-3) agitation = 0;

      for (const group of groups) {
        switch (group.spec.behavior) {
          case "boids":
            stepBoids(group.flock, step, DEFAULT_BOID_PARAMS, agitation);
            break;
          case "wander":
            stepWander(group.flock, step, DEFAULT_WANDER_PARAMS, agitation);
            break;
          case "drift":
            stepDrift(group.flock, step, DEFAULT_DRIFT_PARAMS, agitation);
            break;
        }
        writeMatrices(group);
      }
    },

    dispose(): void {
      for (const group of groups) {
        if (sceneRef) sceneRef.remove(group.mesh);
        group.mesh.geometry.dispose();
        const mat = group.mesh.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else mat.dispose();
        group.mesh.dispose();
      }
      groups = [];
      sceneRef = null;
      agitation = 0;
    },
  };
}

/** Compose and upload instance matrices from the flock's position/velocity. */
function writeMatrices(group: SpeciesGroup): void {
  const { flock, mesh, baseScale } = group;
  const { positions, velocities, phases, count } = flock;
  const flap = group.spec.behavior === "boids" ? 1 : 0;

  for (let i = 0; i < count; i++) {
    const i3 = i * 3;
    _pos.set(positions[i3]!, positions[i3 + 1]!, positions[i3 + 2]!);
    _vel.set(velocities[i3]!, velocities[i3 + 1]!, velocities[i3 + 2]!);

    // Orient along velocity (fall back to +Z when nearly still).
    if (_vel.lengthSq() > 1e-6) _fwd.copy(_vel).normalize();
    else _fwd.set(0, 0, 1);
    _quat.setFromUnitVectors(_up, _fwd);

    // Subtle wing-flap / pulse: squash on the local up axis using phase.
    const pulse = flap ? 1 + Math.sin(phases[i]! * 6) * 0.25 : 1;
    _scale.set(baseScale, baseScale * pulse, baseScale);

    _mat.compose(_pos, _quat, _scale);
    mesh.setMatrixAt(i, _mat);
  }
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.count !== count) mesh.count = count;
}

interface SpeciesAppearance {
  geometry: THREE.BufferGeometry;
  baseScale: number;
  color: THREE.Color;
  emissive: THREE.Color;
}

/**
 * Pick a cheap low-poly geometry, size, and color per species. Geometry is
 * chosen by keyword so new species names degrade gracefully to a sensible
 * default for their behavior.
 */
function makeSpeciesAppearance(
  spec: CreatureSpec,
  accent: THREE.Color,
  primary: THREE.Color,
): SpeciesAppearance {
  const s = spec.species.toLowerCase();
  const black = new THREE.Color(0x000000);

  // Glowing fliers: fireflies, drones, dream motes.
  if (/firefly|fireflies|drone|mote|spark|glow/.test(s)) {
    return {
      geometry: new THREE.SphereGeometry(0.25, 6, 5),
      baseScale: 1,
      color: accent.clone(),
      emissive: accent.clone(),
    };
  }
  // Jellyfish / weightless drifters.
  if (/jelly|drifter|ghost|wisp/.test(s) || spec.behavior === "drift") {
    return {
      geometry: new THREE.IcosahedronGeometry(0.9, 1),
      baseScale: 1,
      color: accent.clone().lerp(primary, 0.5),
      emissive: accent.clone().multiplyScalar(0.6),
    };
  }
  // Butterflies — small, bright, flappy.
  if (/butterfly|butterflies/.test(s)) {
    return {
      geometry: new THREE.ConeGeometry(0.45, 0.15, 4),
      baseScale: 1,
      color: accent.clone(),
      emissive: black,
    };
  }
  // Fish — slim cones, mid palette.
  if (/fish|koi/.test(s)) {
    return {
      geometry: new THREE.ConeGeometry(0.35, 1.3, 6),
      baseScale: 1,
      color: primary.clone(),
      emissive: black,
    };
  }
  // Ground mammals — chunkier boxes.
  if (spec.behavior === "wander") {
    return {
      geometry: new THREE.BoxGeometry(0.8, 0.9, 1.6),
      baseScale: 1,
      color: primary.clone(),
      emissive: black,
    };
  }
  // Default flier (birds/bats/cranes/ravens): a small dart cone.
  return {
    geometry: new THREE.ConeGeometry(0.4, 1.2, 5),
    baseScale: 1,
    color: primary.clone(),
    emissive: black,
  };
}

function clamp01(v: number): number {
  return v < 0 ? 0 : v > 1 ? 1 : v;
}

/** Guard against malformed palette hex strings from upstream. */
function safeColor(hex: string, fallback: string): string {
  return /^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(hex) ? hex : fallback;
}
