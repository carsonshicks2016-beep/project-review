/**
 * Deterministic per-wheel surface spray.
 *
 * Replay wheel slip, contact, load, speed and surface are the only inputs.
 * Braking/throttle merely choose how strongly those recorded tyre states are
 * presented. Pools clear and deterministically prewarm after every seek.
 */

import {
  Color,
  DodecahedronGeometry,
  DynamicDrawUsage,
  Group,
  IcosahedronGeometry,
  InstancedMesh,
  Matrix4,
  MeshBasicMaterial,
  Quaternion,
  Vector3,
} from "three";

import type { LoadedReplay, SurfaceKind } from "../replay";
import type {
  WheelVisualController,
  WheelVisualState,
} from "./surface";

const PUFF_CAPACITY = 180;
const DEBRIS_CAPACITY = 120;
const HIDDEN_SCALE = new Vector3(0, 0, 0);
const UP = new Vector3(0, 1, 0);

const SURFACE_COLOUR: Record<SurfaceKind, number> = {
  gravel: 0xc2ad87,
  mud: 0x745437,
  snow: 0xe3edf1,
  tarmac: 0x595d62,
};

interface Controls {
  brake: number;
  throttle: number;
}

interface Particle {
  age: number;
  life: number;
  size: number;
  pos: Vector3;
  vel: Vector3;
  spin: number;
  colour: Color;
}

function makeParticles(count: number): Particle[] {
  return Array.from({ length: count }, () => ({
    age: -1,
    life: 1,
    size: 1,
    pos: new Vector3(),
    vel: new Vector3(),
    spin: 0,
    colour: new Color(),
  }));
}

function clamp(value: number, low: number, high: number): number {
  return Math.min(Math.max(value, low), high);
}

export class SurfaceEffects {
  readonly root = new Group();
  readonly puffs: InstancedMesh;
  readonly debris: InstancedMesh;

  private readonly puffParticles = makeParticles(PUFF_CAPACITY);
  private readonly debrisParticles = makeParticles(DEBRIS_CAPACITY);
  private readonly puffCarry = [0, 0, 0, 0];
  private readonly debrisCarry = [0, 0, 0, 0];
  private readonly matrix = new Matrix4();
  private readonly rotation = new Quaternion();
  private readonly scale = new Vector3();
  private readonly tint = new Color();
  private puffCursor = 0;
  private debrisCursor = 0;
  private randomState = 0x5555c0de;

  constructor() {
    this.root.name = "per_wheel_surface_effects";
    this.puffs = new InstancedMesh(
      new IcosahedronGeometry(0.5, 0),
      new MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        // Near-wheel spray needs to read against a matching road albedo.
        // Per-instance alpha is not available on this material, so young
        // particles are also brightened in writePool to carry the weight.
        opacity: 0.82,
        depthWrite: false,
        fog: true,
      }),
      PUFF_CAPACITY,
    );
    this.puffs.name = "surface_puffs";
    this.puffs.renderOrder = 2;
    this.puffs.frustumCulled = false;
    this.puffs.instanceMatrix.setUsage(DynamicDrawUsage);

    this.debris = new InstancedMesh(
      new DodecahedronGeometry(0.5, 0),
      new MeshBasicMaterial({
        color: 0xffffff,
        depthWrite: true,
        fog: true,
      }),
      DEBRIS_CAPACITY,
    );
    this.debris.name = "surface_fragments";
    this.debris.frustumCulled = false;
    this.debris.instanceMatrix.setUsage(DynamicDrawUsage);
    this.root.add(this.puffs, this.debris);
    this.clear();
  }

  private random(): number {
    this.randomState = (this.randomState * 1664525 + 1013904223) >>> 0;
    return this.randomState / 4294967296;
  }

  clear(): void {
    this.randomState = 0x5555c0de;
    this.puffCursor = 0;
    this.debrisCursor = 0;
    this.puffCarry.fill(0);
    this.debrisCarry.fill(0);
    for (const particle of [
      ...this.puffParticles,
      ...this.debrisParticles,
    ]) {
      particle.age = -1;
    }
    this.writeInstances();
  }

  prewarm(
    replay: LoadedReplay,
    time: number,
    controller: WheelVisualController,
  ): void {
    this.clear();
    const start = Math.max(0, time - 1.6);
    for (const frame of replay.frames) {
      if (frame.t < start) continue;
      if (frame.t > time + 1e-6) break;
      this.update(replay.dt, controller.fromFrame(frame), {
        brake: frame.in?.b ?? 0,
        throttle: frame.in?.t ?? 0,
      });
    }
  }

  update(
    dt: number,
    wheels: readonly WheelVisualState[],
    controls: Controls,
  ): void {
    if (!(dt > 0)) {
      this.writeInstances();
      return;
    }

    this.advance(this.puffParticles, dt, false);
    this.advance(this.debrisParticles, dt, true);

    for (const wheel of wheels) {
      const loose =
        wheel.surface === "gravel" ||
        wheel.surface === "mud" ||
        wheel.surface === "snow";
      if (!wheel.contact || !loose || wheel.speed < 3) {
        this.puffCarry[wheel.index] = Math.min(
          this.puffCarry[wheel.index]!,
          0.5,
        );
        this.debrisCarry[wheel.index] = Math.min(
          this.debrisCarry[wheel.index]!,
          0.5,
        );
        continue;
      }

      const longEnergy = Math.abs(wheel.slipRatio) * wheel.speed;
      const lateralEnergy = Math.abs(wheel.slipAngle) * wheel.speed;
      const axleInput =
        wheel.index < 2 ? controls.brake : controls.throttle * 0.65;
      const loadFactor = clamp(wheel.load / 3600, 0.25, 1.65);
      const puffRate =
        (2.5 +
          wheel.speed * 0.18 +
          longEnergy * 1.15 +
          lateralEnergy * 1.7 +
          axleInput * 2.4) *
        loadFactor;
      this.puffCarry[wheel.index] =
        this.puffCarry[wheel.index]! + puffRate * dt;
      let emitted = 0;
      while (this.puffCarry[wheel.index]! >= 1 && emitted < 5) {
        this.puffCarry[wheel.index]!--;
        this.spawnPuff(wheel, longEnergy, lateralEnergy);
        emitted++;
      }

      const fragmentRate =
        wheel.surface === "snow"
          ? 0.5 + longEnergy * 0.16 + lateralEnergy * 0.22
          : 0.8 + longEnergy * 0.28 + lateralEnergy * 0.42;
      this.debrisCarry[wheel.index] =
        this.debrisCarry[wheel.index]! + fragmentRate * dt;
      if (this.debrisCarry[wheel.index]! >= 1) {
        this.debrisCarry[wheel.index]!--;
        this.spawnDebris(wheel, lateralEnergy);
      }
    }

    this.writeInstances();
  }

  private advance(
    particles: Particle[],
    dt: number,
    gravity: boolean,
  ): void {
    for (const particle of particles) {
      if (particle.age < 0) continue;
      particle.age += dt;
      if (particle.age >= particle.life) {
        particle.age = -1;
        continue;
      }
      particle.pos.addScaledVector(particle.vel, dt);
      // Puffs keep vertical speed longer than lateral so they rise 1.5–3 m
      // over their life without changing how many are emitted.
      if (gravity) {
        particle.vel.multiplyScalar(Math.exp(-0.45 * dt));
        particle.vel.y += -5.4 * dt;
      } else {
        particle.vel.x *= Math.exp(-0.85 * dt);
        particle.vel.z *= Math.exp(-0.85 * dt);
        particle.vel.y *= Math.exp(-0.28 * dt);
        particle.vel.y += 0.55 * dt;
      }
      particle.spin += dt * (gravity ? 8 : 0.7);
    }
  }

  private spawnPuff(
    wheel: WheelVisualState,
    longEnergy: number,
    lateralEnergy: number,
  ): void {
    const particle = this.puffParticles[this.puffCursor]!;
    this.puffCursor = (this.puffCursor + 1) % PUFF_CAPACITY;
    const lateralSign =
      Math.abs(wheel.slipAngle) > 0.006
        ? -Math.sign(wheel.slipAngle)
        : (wheel.index % 2 === 0 ? -1 : 1) * 0.18;
    particle.pos
      .copy(wheel.contactPoint)
      .addScaledVector(wheel.normal, 0.10 + this.random() * 0.07);
    particle.vel
      .copy(wheel.forward)
      .multiplyScalar(-(0.75 + Math.min(wheel.speed * 0.08 + longEnergy * 0.12, 4.0)))
      .addScaledVector(
        wheel.right,
        lateralSign * Math.min(0.55 + lateralEnergy * 0.72, 5.3),
      )
      // 1.5–3 m of vertical rise over a 1.2–2.1 s life.
      .addScaledVector(wheel.normal, 1.85 + this.random() * 1.55);
    particle.age = 0;
    particle.life = 1.2 + this.random() * 0.9;
    particle.size =
      0.42 +
      Math.min(wheel.speed * 0.012, 0.38) +
      this.random() * 0.22;
    particle.spin = this.random() * Math.PI;
    // Brighten above road albedo so tan spray reads on tan gravel; a minority
    // of puffs stay as a darker core for volume rather than a flat haze.
    particle.colour
      .setHex(SURFACE_COLOUR[wheel.surface])
      .offsetHSL((this.random() - 0.5) * 0.03, 0.04, 0);
    if (this.random() < 0.28) {
      particle.colour.multiplyScalar(0.42 + this.random() * 0.14);
    } else {
      particle.colour.offsetHSL(0, 0, 0.14 + this.random() * 0.16);
      particle.colour.multiplyScalar(1.12 + this.random() * 0.18);
    }
  }

  private spawnDebris(
    wheel: WheelVisualState,
    lateralEnergy: number,
  ): void {
    const particle = this.debrisParticles[this.debrisCursor]!;
    this.debrisCursor = (this.debrisCursor + 1) % DEBRIS_CAPACITY;
    const side =
      -Math.sign(wheel.slipAngle || (wheel.index % 2 === 0 ? -1 : 1));
    particle.pos
      .copy(wheel.contactPoint)
      .addScaledVector(wheel.normal, 0.06);
    particle.vel
      .copy(wheel.forward)
      .multiplyScalar(-(0.5 + this.random() * 2.1))
      .addScaledVector(
        wheel.right,
        side * (0.25 + Math.min(lateralEnergy * 0.48, 3.2)),
      )
      .addScaledVector(wheel.normal, 0.75 + this.random() * 1.65);
    particle.age = 0;
    particle.life = 0.42 + this.random() * 0.46;
    particle.size =
      wheel.surface === "snow"
        ? 0.055 + this.random() * 0.075
        : 0.035 + this.random() * 0.07;
    particle.spin = this.random() * Math.PI;
    particle.colour
      .setHex(SURFACE_COLOUR[wheel.surface])
      .multiplyScalar(0.72 + this.random() * 0.30);
  }

  private writeInstances(): void {
    this.writePool(this.puffs, this.puffParticles, false);
    this.writePool(this.debris, this.debrisParticles, true);
  }

  private writePool(
    mesh: InstancedMesh,
    particles: Particle[],
    gravity: boolean,
  ): void {
    for (let i = 0; i < particles.length; i++) {
      const particle = particles[i]!;
      if (particle.age < 0) {
        this.scale.copy(HIDDEN_SCALE);
        this.rotation.identity();
        mesh.setMatrixAt(
          i,
          this.matrix.compose(
            particle.pos.set(0, -1000, 0),
            this.rotation,
            this.scale,
          ),
        );
        mesh.setColorAt(i, particle.colour.setHex(0xffffff));
        continue;
      }
      const f = particle.age / particle.life;
      const fade = f < 0.72 ? 1 : Math.max(0, (1 - f) / 0.28);
      const size = gravity
        ? particle.size * fade
        : particle.size * (0.85 + f * 3.4) * fade;
      this.scale.set(
        size,
        size * (gravity ? 0.72 : 0.9),
        size * (gravity ? 1.15 : 1.05),
      );
      this.rotation.setFromAxisAngle(UP, particle.spin);
      mesh.setMatrixAt(
        i,
        this.matrix.compose(particle.pos, this.rotation, this.scale),
      );
      // Young (near-wheel) puffs read hotter; age cools them toward the albedo.
      if (!gravity) {
        const near = 1 + (1 - f) * 0.35;
        this.tint.copy(particle.colour).multiplyScalar(near * fade);
        mesh.setColorAt(i, this.tint);
      } else {
        mesh.setColorAt(i, particle.colour);
      }
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
}
