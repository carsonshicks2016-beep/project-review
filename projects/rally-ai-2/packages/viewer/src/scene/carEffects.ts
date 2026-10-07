/**
 * Replay-led vehicle feedback.
 *
 * Direct channels:
 * - brake-light intensity: recorded brake input;
 * - exhaust pulse rate: recorded RPM and throttle;
 * - damage visibility: explicit replay impact/crash/contact events only.
 *
 * Documented viewer derivations:
 * - anti-lag: a recorded throttle drop while RPM is above 3,800;
 * - grime: accumulated recorded distance, weighted by recorded surface.
 *
 * None of these values feed back into playback or vehicle state.
 */

import {
  Color,
  DynamicDrawUsage,
  Group,
  IcosahedronGeometry,
  InstancedMesh,
  Matrix4,
  MeshBasicMaterial,
  Quaternion,
  TetrahedronGeometry,
  Vector3,
} from "three";

import { orientation, xyz } from "../coords";
import type { PlaybackSample } from "../playback";
import type {
  Frame,
  LoadedReplay,
  ReplayEvent,
  SurfaceKind,
} from "../replay";
import type { CarRig } from "./car";

const SMOKE_CAPACITY = 72;
const FLASH_CAPACITY = 24;
const EXHAUST_LOCAL = new Vector3(-2.30, 0.28, 0.51);
const HIDDEN_SCALE = new Vector3(0, 0, 0);
const UP = new Vector3(0, 1, 0);

const GRIME_COLOUR: Record<SurfaceKind, number> = {
  gravel: 0x806648,
  mud: 0x4e3524,
  snow: 0xb8c6cc,
  tarmac: 0x4a4947,
};

interface CarInput {
  t: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  v: number;
  rpm: number;
  throttle: number;
  brake: number;
  surface: SurfaceKind;
}

interface Particle {
  age: number;
  life: number;
  size: number;
  pos: Vector3;
  vel: Vector3;
  colour: Color;
  spin: number;
}

function particles(count: number): Particle[] {
  return Array.from({ length: count }, () => ({
    age: -1,
    life: 1,
    size: 1,
    pos: new Vector3(),
    vel: new Vector3(),
    colour: new Color(),
    spin: 0,
  }));
}

function inputFromFrame(frame: Frame): CarInput {
  return {
    t: frame.t,
    x: frame.x,
    y: frame.y,
    z: frame.z,
    yaw: frame.yaw,
    pitch: frame.pitch,
    roll: frame.roll,
    v: frame.v,
    rpm: frame.rpm ?? 0,
    throttle: frame.in?.t ?? 0,
    brake: frame.in?.b ?? 0,
    surface: frame.surf ?? "gravel",
  };
}

function inputFromSample(sample: PlaybackSample): CarInput {
  return {
    t: sample.t,
    x: sample.x,
    y: sample.y,
    z: sample.z,
    yaw: sample.yaw,
    pitch: sample.pitch,
    roll: sample.roll,
    v: sample.v,
    rpm: sample.rpm,
    throttle: sample.throttle,
    brake: sample.brake,
    surface: sample.surface ?? "gravel",
  };
}

function isDamageEvent(event: ReplayEvent): boolean {
  return /crash|impact|collision|contact|damage|puncture/i.test(event.kind);
}

export class CarStateEffects {
  readonly root = new Group();
  readonly smoke: InstancedMesh;
  readonly flashes: InstancedMesh;

  private readonly smokeParticles = particles(SMOKE_CAPACITY);
  private readonly flashParticles = particles(FLASH_CAPACITY);
  private readonly antiLagTimes: number[] = [];
  private readonly grimeAtFrame: number[] = [];
  private readonly grimeSurfaceAtFrame: SurfaceKind[] = [];
  private readonly matrix = new Matrix4();
  private readonly rotation = new Quaternion();
  private readonly scale = new Vector3();
  private readonly exhaust = new Vector3();
  private readonly carRotation = new Quaternion();
  private readonly randomVelocity = new Vector3();
  private smokeCursor = 0;
  private flashCursor = 0;
  private emitCarry = 0;
  private randomState = 0x51ba7e;
  private lastTime = 0;

  constructor(
    private readonly replay: LoadedReplay,
    private readonly rig: CarRig,
  ) {
    this.root.name = "car_state_effects";
    this.smoke = new InstancedMesh(
      new IcosahedronGeometry(0.5, 0),
      new MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.46,
        depthWrite: false,
        fog: true,
      }),
      SMOKE_CAPACITY,
    );
    this.smoke.name = "exhaust_and_antilag_smoke";
    this.smoke.renderOrder = 2;
    this.smoke.frustumCulled = false;
    this.smoke.instanceMatrix.setUsage(DynamicDrawUsage);

    this.flashes = new InstancedMesh(
      new TetrahedronGeometry(0.5, 0),
      new MeshBasicMaterial({
        color: 0xffffff,
        transparent: true,
        opacity: 0.92,
        depthWrite: false,
        fog: true,
      }),
      FLASH_CAPACITY,
    );
    this.flashes.name = "antilag_flashes";
    this.flashes.renderOrder = 3;
    this.flashes.frustumCulled = false;
    this.flashes.instanceMatrix.setUsage(DynamicDrawUsage);
    this.root.add(this.smoke, this.flashes);

    this.precomputeDerivedChannels();
    this.clearParticles();
    this.applyState(0, inputFromFrame(replay.frames[0]!));
  }

  private precomputeDerivedChannels(): void {
    let grimeDistance = 0;
    let lastLoose: SurfaceKind = "gravel";
    let lastAntiLag = -10;
    for (let i = 0; i < this.replay.frames.length; i++) {
      const frame = this.replay.frames[i]!;
      const surface = frame.surf ?? "gravel";
      const weight =
        surface === "mud"
          ? 1.35
          : surface === "gravel"
            ? 0.82
            : surface === "snow"
              ? 0.48
              : 0.035;
      grimeDistance += Math.abs(frame.v) * this.replay.dt * weight;
      if (surface !== "tarmac") lastLoose = surface;
      this.grimeAtFrame.push(Math.min(1, grimeDistance / 520));
      this.grimeSurfaceAtFrame.push(lastLoose);

      if (i === 0) continue;
      const previous = this.replay.frames[i - 1]!;
      const throttleDrop =
        (previous.in?.t ?? 0) - (frame.in?.t ?? 0);
      const rpm = Math.max(previous.rpm ?? 0, frame.rpm ?? 0);
      if (
        throttleDrop > 0.28 &&
        rpm > 3800 &&
        Math.abs(frame.v) > 8 &&
        frame.t - lastAntiLag > 0.18
      ) {
        this.antiLagTimes.push(frame.t);
        lastAntiLag = frame.t;
      }
    }
  }

  private random(): number {
    this.randomState = (this.randomState * 1664525 + 1013904223) >>> 0;
    return this.randomState / 4294967296;
  }

  rebuild(time: number): void {
    this.clearParticles();
    const start = Math.max(0, time - 1.35);
    this.lastTime = start;
    for (const frame of this.replay.frames) {
      if (frame.t < start) continue;
      if (frame.t > time + 1e-6) break;
      this.stepParticles(this.replay.dt, inputFromFrame(frame));
      this.lastTime = frame.t;
    }
    const index = Math.min(
      Math.max(Math.round(time / this.replay.dt), 0),
      this.replay.frames.length - 1,
    );
    this.applyState(time, inputFromFrame(this.replay.frames[index]!));
    this.lastTime = time;
    this.writeInstances();
  }

  update(dt: number, sample: PlaybackSample): void {
    const input = inputFromSample(sample);
    this.applyState(sample.t, input);
    if (dt > 0) this.stepParticles(dt, input);
    this.lastTime = sample.t;
    this.writeInstances();
  }

  private applyState(time: number, input: CarInput): void {
    for (const light of this.rig.brakeLights) {
      light.emissiveIntensity = 0.06 + input.brake * 0.72;
      light.color.setHex(input.brake > 0.04 ? 0xff3237 : 0xd52b2f);
    }

    const index = Math.min(
      Math.max(Math.floor(time / this.replay.dt), 0),
      this.grimeAtFrame.length - 1,
    );
    const grime = this.grimeAtFrame[index] ?? 0;
    const grimeSurface = this.grimeSurfaceAtFrame[index] ?? "gravel";
    for (const material of this.rig.grimeMaterials) {
      material.opacity = grime * 0.76;
      material.color.setHex(GRIME_COLOUR[grimeSurface]);
    }

    const damage = (this.replay.events ?? [])
      .filter((event) => event.t <= time && isDamageEvent(event))
      .reduce((sum, event) => sum + Math.max(1, event.severity ?? 1), 0);
    this.rig.damageParts.forEach((part, partIndex) => {
      part.visible = damage > partIndex * 2.5;
    });
  }

  private stepParticles(dt: number, input: CarInput): void {
    this.advance(this.smokeParticles, dt, false);
    this.advance(this.flashParticles, dt, true);
    this.exhaustAt(input);

    // Exhaust pulse frequency is a direct visual mapping from recorded engine
    // state. At idle it only breathes; high-RPM throttle produces a clear train.
    if (input.rpm > 500) {
      const rate =
        1.1 +
        Math.min(input.rpm / 1600, 4.8) +
        input.throttle * 4.2;
      this.emitCarry += rate * dt;
      let emitted = 0;
      while (this.emitCarry >= 1 && emitted < 3) {
        this.emitCarry -= 1;
        this.spawnSmoke(input, false);
        emitted++;
      }
    }

    for (const eventTime of this.antiLagTimes) {
      if (eventTime > this.lastTime + 1e-6 && eventTime <= input.t + 1e-6) {
        this.spawnFlash(input);
        this.spawnSmoke(input, true);
        this.spawnSmoke(input, true);
      }
    }
  }

  private exhaustAt(input: CarInput): Vector3 {
    xyz(input.x, input.y, input.z, this.exhaust);
    orientation(
      input.yaw,
      input.pitch,
      input.roll,
      this.carRotation,
    );
    return this.exhaust.add(
      this.randomVelocity
        .copy(EXHAUST_LOCAL)
        .applyQuaternion(this.carRotation),
    );
  }

  private spawnSmoke(input: CarInput, antiLag: boolean): void {
    const particle = this.smokeParticles[this.smokeCursor]!;
    this.smokeCursor = (this.smokeCursor + 1) % SMOKE_CAPACITY;
    this.exhaustAt(input);
    particle.pos.copy(this.exhaust);
    const rear = new Vector3(-1, 0, 0).applyQuaternion(this.carRotation);
    particle.vel
      .copy(rear)
      .multiplyScalar(0.45 + Math.abs(input.v) * 0.035 + this.random() * 0.55)
      .addScaledVector(UP, 0.18 + this.random() * 0.28);
    particle.age = 0;
    particle.life = antiLag
      ? 0.42 + this.random() * 0.25
      : 0.56 + this.random() * 0.38;
    particle.size = antiLag
      ? 0.12 + this.random() * 0.08
      : 0.075 + this.random() * 0.055;
    particle.spin = this.random() * Math.PI;
    particle.colour.setHex(antiLag ? 0x5b5b58 : 0x8a8d89);
  }

  private spawnFlash(input: CarInput): void {
    const particle = this.flashParticles[this.flashCursor]!;
    this.flashCursor = (this.flashCursor + 1) % FLASH_CAPACITY;
    this.exhaustAt(input);
    particle.pos.copy(this.exhaust);
    particle.vel.set(0, 0, 0);
    particle.age = 0;
    particle.life = 0.075 + this.random() * 0.055;
    particle.size = 0.18 + this.random() * 0.14;
    particle.spin = this.random() * Math.PI;
    particle.colour.setHex(this.random() > 0.45 ? 0xffa01f : 0xffdc55);
  }

  private advance(
    pool: Particle[],
    dt: number,
    flash: boolean,
  ): void {
    for (const particle of pool) {
      if (particle.age < 0) continue;
      particle.age += dt;
      if (particle.age >= particle.life) {
        particle.age = -1;
        continue;
      }
      particle.pos.addScaledVector(particle.vel, dt);
      if (!flash) {
        particle.vel.multiplyScalar(Math.exp(-1.2 * dt));
        particle.vel.y += 0.12 * dt;
      }
      particle.spin += dt * (flash ? 18 : 0.9);
    }
  }

  private clearParticles(): void {
    this.randomState = 0x51ba7e;
    this.emitCarry = 0;
    this.smokeCursor = 0;
    this.flashCursor = 0;
    for (const particle of [
      ...this.smokeParticles,
      ...this.flashParticles,
    ]) {
      particle.age = -1;
    }
    this.writeInstances();
  }

  private writeInstances(): void {
    this.writePool(this.smoke, this.smokeParticles, false);
    this.writePool(this.flashes, this.flashParticles, true);
  }

  private writePool(
    mesh: InstancedMesh,
    pool: Particle[],
    flash: boolean,
  ): void {
    for (let i = 0; i < pool.length; i++) {
      const particle = pool[i]!;
      if (particle.age < 0) {
        mesh.setMatrixAt(
          i,
          this.matrix.compose(
            particle.pos.set(0, -1000, 0),
            this.rotation.identity(),
            HIDDEN_SCALE,
          ),
        );
        mesh.setColorAt(i, particle.colour.setHex(0xffffff));
        continue;
      }
      const f = particle.age / particle.life;
      const fade = Math.max(0, 1 - f);
      const size = flash
        ? particle.size * fade
        : particle.size * (0.72 + f * 1.65) * fade;
      this.scale.set(size, size * (flash ? 0.72 : 0.88), size);
      this.rotation.setFromAxisAngle(UP, particle.spin);
      mesh.setMatrixAt(
        i,
        this.matrix.compose(particle.pos, this.rotation, this.scale),
      );
      mesh.setColorAt(i, particle.colour);
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }
}
