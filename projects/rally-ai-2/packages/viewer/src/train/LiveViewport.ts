/**
 * G3 live viewport — the same WATCH scene stack, fed by F3 frame packets
 * instead of a finished replay file.
 */

import {
  Color,
  DirectionalLight,
  Fog,
  Group,
  HemisphereLight,
  Object3D,
  Scene,
  Vector3,
  WebGLRenderer,
  type Mesh,
} from "three";

import { ChaseCamera } from "../camera";
import { orientation, xyz } from "../coords";
import { Hud } from "../hud";
import type { PlaybackSample } from "../playback";
import { applyVertexJitter, Ps1Pass } from "../render/ps1";
import type { Frame, LoadedReplay, Stage } from "../replay";
import {
  buildCar,
  driveWheels,
  resetSuspension,
  updateSuspension,
  type CarRig,
} from "../scene/car";
import { SurfaceEffects } from "../scene/dust";
import { GroundingShadows } from "../scene/shadows";
import { buildSky, HORIZON_COLOUR } from "../scene/sky";
import { buildStage } from "../scene/stage";
import { StageSurface, WheelVisualController } from "../scene/surface";
import { buildBanners, buildRoadside } from "../scene/trees";
import { SensorOverlay, type SenseDebug } from "./sensors";

const SKY = new Color(HORIZON_COLOUR);
const FOG_NEAR = 42;
const FOG_FAR = 210;
const WHEEL_R = 0.32;

/** Map a live frame onto the same sample shape WATCH interpolates. */
export function frameToSample(frame: Frame): PlaybackSample {
  return {
    t: frame.t,
    x: frame.x,
    y: frame.y,
    z: frame.z,
    yaw: frame.yaw,
    pitch: frame.pitch,
    roll: frame.roll,
    v: frame.v,
    s: frame.s,
    vx: frame.vx ?? 0,
    vy: frame.vy ?? 0,
    vz: frame.vz ?? 0,
    steer: frame.in?.s ?? 0,
    throttle: frame.in?.t ?? 0,
    brake: frame.in?.b ?? 0,
    handbrake: frame.in?.h ?? 0,
    rpm: frame.rpm ?? 0,
    gear: frame.gear ?? 0,
    boost: frame.boost ?? 0,
    slipAngle: frame.sa ?? 0,
    airborne: Boolean(frame.air),
    height: frame.hgt ?? 0,
    surface: frame.surf,
    note: frame.note ?? "",
    wheels: frame.w ?? [],
  };
}

function stubReplay(stage: Stage): LoadedReplay {
  return {
    schema_version: 1,
    dt: 1 / 30,
    source: "agent",
    meta: {
      termination: "aborted",
      time_s: 0,
      car: "live",
      physics_version: "live",
    },
    frames: [
      {
        t: 0,
        x: stage.start?.s === 0 ? (stage.centerline[0]?.x ?? 0) : 0,
        y: stage.centerline[0]?.y ?? 0,
        z: stage.centerline[0]?.z ?? 0,
        yaw: stage.start?.heading ?? 0,
        pitch: 0,
        roll: 0,
        v: 0,
        s: 0,
      },
    ],
    stage,
  };
}

export class LiveViewport {
  private readonly wrap: HTMLElement;
  private readonly canvas: HTMLCanvasElement;
  private readonly hudHost: HTMLElement;
  private readonly emptyEl: HTMLElement;

  private renderer: WebGLRenderer | null = null;
  private scene: Scene | null = null;
  private chase: ChaseCamera | null = null;
  private ps1: Ps1Pass | null = null;
  private car: CarRig | null = null;
  private road: StageSurface | null = null;
  private wheels: WheelVisualController | null = null;
  private shadows: GroundingShadows | null = null;
  private surfaceEffects: SurfaceEffects | null = null;
  private sensors: SensorOverlay | null = null;
  private hud: Hud | null = null;
  private stageRoot: Group | null = null;

  private lastFrame: Frame | null = null;
  private lastSample: PlaybackSample | null = null;
  private wheelAngle = 0;
  private raf = 0;
  private last = performance.now();
  private running = false;

  private readonly pos = new Vector3();
  private readonly fwd = new Vector3();

  constructor(parent: HTMLElement) {
    this.wrap = document.createElement("div");
    this.wrap.className = "train-viewport-wrap";
    this.wrap.innerHTML = `
      <canvas class="train-view"></canvas>
      <div class="train-hud-host"></div>
      <div class="train-viewport-empty">
        Watch live — reference pilot until a checkpoint exists
      </div>
    `;
    parent.appendChild(this.wrap);
    this.canvas = this.wrap.querySelector("canvas")!;
    this.hudHost = this.wrap.querySelector(".train-hud-host")!;
    this.emptyEl = this.wrap.querySelector(".train-viewport-empty")!;
  }

  start(): void {
    if (this.running) return;
    this.running = true;
    this.renderer = new WebGLRenderer({ canvas: this.canvas, antialias: true });
    this.renderer.setClearColor(SKY);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

    this.scene = new Scene();
    this.scene.fog = new Fog(SKY, FOG_NEAR, FOG_FAR);
    this.scene.add(new HemisphereLight(0xcfe7ef, 0x50613d, 2.05));
    const sun = new DirectionalLight(0xfff2cf, 2.65);
    sun.position.set(-38, 74, 46);
    this.scene.add(sun);

    this.stageRoot = new Group();
    this.stageRoot.name = "liveStage";
    this.scene.add(this.stageRoot);

    this.car = buildCar();
    this.scene.add(this.car.root);
    this.shadows = new GroundingShadows();
    this.scene.add(this.shadows.root);
    this.surfaceEffects = new SurfaceEffects();
    this.scene.add(this.surfaceEffects.root);

    this.sensors = new SensorOverlay();
    this.scene.add(this.sensors.group);

    this.chase = new ChaseCamera(1);
    this.chase.camera.add(buildSky());
    this.scene.add(this.chase.camera);

    this.ps1 = new Ps1Pass(this.renderer);
    this.resize();
    this.last = performance.now();
    const loop = (now: number): void => {
      if (!this.running) return;
      const dt = Math.min((now - this.last) / 1000, 0.1);
      this.last = now;
      this.tick(dt);
      this.raf = requestAnimationFrame(loop);
    };
    this.raf = requestAnimationFrame(loop);
  }

  setStage(stage: Stage): void {
    if (!this.scene || !this.stageRoot || !this.chase) return;
    this.setEmpty(false);

    while (this.stageRoot.children.length) {
      const child = this.stageRoot.children[0]!;
      this.stageRoot.remove(child);
      disposeObject(child);
    }

    this.stageRoot.add(buildStage(stage));
    this.stageRoot.add(buildRoadside(stage));
    this.stageRoot.add(buildBanners(stage));

    this.road = new StageSurface(stage);
    this.wheels = new WheelVisualController(this.road);

    this.hud?.reset();
    this.hudHost.replaceChildren();
    this.hud = new Hud(this.hudHost, stubReplay(stage));

    this.sensors?.clear();
    this.chase.reset();
    if (this.car) resetSuspension(this.car);
    this.wheelAngle = 0;
    this.lastFrame = null;
    this.lastSample = null;

    // Soft vertex jitter on environment only — same rule as WATCH.
    const belongsTo = (o: Object3D, name: string): boolean => {
      for (let p: Object3D | null = o; p; p = p.parent) {
        if (p.name === name) return true;
      }
      return false;
    };
    this.scene.traverse((o) => {
      if (
        belongsTo(o, "sky") ||
        belongsTo(o, "car") ||
        belongsTo(o, "banners") ||
        belongsTo(o, "sensorDebug") ||
        belongsTo(o, "vehicle_grounding_shadows") ||
        belongsTo(o, "per_wheel_surface_effects")
      ) {
        return;
      }
      const m = (o as Mesh).material;
      if (!m) return;
      for (const mat of Array.isArray(m) ? m : [m]) {
        applyVertexJitter(mat, false);
      }
    });
  }

  applyFrame(frame: Frame, sense: SenseDebug | null): void {
    this.setEmpty(false);
    this.lastFrame = frame;
    this.lastSample = frameToSample(frame);
    // Sense is applied immediately so beams never lag one frame behind the car.
    if (this.sensors) {
      this.sensors.apply({ x: frame.x, y: frame.y, z: frame.z }, sense);
    }
  }

  clearSense(): void {
    this.sensors?.clear();
  }

  setEmpty(empty: boolean): void {
    this.emptyEl.hidden = !empty;
  }

  resize(): void {
    if (!this.renderer || !this.chase || !this.ps1) return;
    const rect = this.wrap.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    this.renderer.setSize(w, h, false);
    this.chase.resize(w / h);
    this.ps1.resize();
  }

  destroy(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
    this.sensors?.dispose();
    this.sensors = null;
    this.ps1 = null;
    this.renderer?.dispose();
    this.renderer = null;
    this.scene = null;
    this.chase = null;
    this.car = null;
    this.road = null;
    this.wheels = null;
    this.shadows = null;
    this.surfaceEffects = null;
    this.hud = null;
    this.stageRoot = null;
    this.wrap.remove();
  }

  private tick(dt: number): void {
    if (!this.renderer || !this.scene || !this.chase || !this.ps1 || !this.car) {
      return;
    }

    const sample = this.lastSample;
    const frame = this.lastFrame;
    if (sample && frame) {
      xyz(sample.x, sample.y, sample.z, this.pos);
      this.car.root.position.copy(this.pos);
      this.car.root.quaternion.copy(
        orientation(sample.yaw, sample.pitch, sample.roll),
      );
      updateSuspension(this.car, sample.wheels, sample.airborne, dt);
      this.wheelAngle += (sample.v / WHEEL_R) * dt;
      driveWheels(this.car, sample.steer, this.wheelAngle);

      if (this.wheels && this.road && this.shadows && this.surfaceEffects) {
        const wheelState = this.wheels.update(sample, this.car);
        this.shadows.update(
          this.pos,
          sample.s,
          sample.height,
          this.road,
          wheelState,
        );
        this.surfaceEffects.update(dt, wheelState, {
          brake: sample.brake,
          throttle: sample.throttle,
        });
      }

      xyz(Math.cos(sample.yaw), Math.sin(sample.yaw), 0, this.fwd);
      this.chase.update(this.pos, this.fwd, sample.v, dt, {
        time: sample.t,
        surface: sample.surface,
        slipAngle: sample.slipAngle,
        landingKick: 0,
      });
      this.hud?.update(frame, sample.t, sample.v);
    }

    this.ps1.render(this.scene, this.chase.camera);
  }
}

function disposeObject(root: Object3D): void {
  root.traverse((o) => {
    const mesh = o as Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const m = mesh.material;
    if (!m) return;
    for (const mat of Array.isArray(m) ? m : [m]) {
      mat.dispose();
    }
  });
}
