/**
 * Shared road and wheel presentation state.
 *
 * Everything that appears at a tyre contact patch consumes this module:
 * shadows, loose-surface spray and persistent tracks. It is deliberately a
 * read-only projection of replay data. Nothing here integrates vehicle motion
 * or writes back to playback, physics, sensing or training.
 */

import { Matrix4, Quaternion, Vector3 } from "three";

import { orientation, rightDir, xyz } from "../coords";
import type { PlaybackSample } from "../playback";
import type {
  CenterlinePoint,
  Frame,
  Stage,
  SurfaceKind,
  WheelFrame,
} from "../replay";
import { surfaceAt } from "../replay";
import {
  CAR_DIMENSIONS,
  type CarRig,
  wheelVisualCompression,
} from "./car";

const UP = new Vector3(0, 1, 0);
const ROAD_LIFT = 0.036;

export interface RoadFrame {
  point: Vector3;
  forward: Vector3;
  right: Vector3;
  normal: Vector3;
  width: number;
  camber: number;
  surface: SurfaceKind;
}

function findSegment(points: CenterlinePoint[], s: number): number {
  let lo = 0;
  let hi = points.length - 1;
  while (lo + 1 < hi) {
    const mid = (lo + hi) >> 1;
    if (points[mid]!.s <= s) lo = mid;
    else hi = mid;
  }
  return Math.min(lo, points.length - 2);
}

/** Interpolates the stage's actual road plane at an arc length. */
export class StageSurface {
  private readonly point = new Vector3();
  private readonly forward = new Vector3();
  private readonly right = new Vector3();
  private readonly normal = new Vector3();

  constructor(readonly stage: Stage) {}

  sample(s: number): RoadFrame {
    const pts = this.stage.centerline;
    const clamped = Math.min(Math.max(s, pts[0]!.s), pts[pts.length - 1]!.s);
    const i = findSegment(pts, clamped);
    const a = pts[i]!;
    const b = pts[i + 1]!;
    const span = b.s - a.s;
    const f = span > 1e-9 ? (clamped - a.s) / span : 0;
    const x = a.x + (b.x - a.x) * f;
    const y = a.y + (b.y - a.y) * f;
    const z = a.z + (b.z - a.z) * f;
    const width = a.width + (b.width - a.width) * f;
    const camber = a.camber + (b.camber - a.camber) * f;
    const heading = Math.atan2(b.y - a.y, b.x - a.x);

    xyz(x, y, z, this.point);
    xyz(b.x - a.x, b.y - a.y, b.z - a.z, this.forward).normalize();
    rightDir(heading, this.right)
      .multiplyScalar(Math.cos(camber))
      .addScaledVector(UP, Math.sin(camber))
      .normalize();
    this.normal.copy(this.right).cross(this.forward).normalize();
    // Re-orthogonalise the right vector after grade and camber are combined.
    this.right.copy(this.forward).cross(this.normal).normalize();

    return {
      point: this.point,
      forward: this.forward,
      right: this.right,
      normal: this.normal,
      width,
      camber,
      surface: surfaceAt(this.stage, clamped)?.type ?? "gravel",
    };
  }

  /** Project a world point onto the sampled road plane. */
  project(point: Vector3, s: number, out = new Vector3()): Vector3 {
    const road = this.sample(s);
    const distance = out.copy(point).sub(road.point).dot(road.normal);
    return out.copy(point).addScaledVector(road.normal, -distance);
  }
}

export interface WheelVisualState {
  /** FL, FR, RL, RR. */
  index: number;
  position: Vector3;
  contactPoint: Vector3;
  forward: Vector3;
  right: Vector3;
  normal: Vector3;
  contact: boolean;
  compression: number;
  load: number;
  slipRatio: number;
  slipAngle: number;
  speed: number;
  surface: SurfaceKind;
}

interface WheelStateSource {
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  s: number;
  v: number;
  airborne: boolean;
  height: number;
  surface: SurfaceKind | undefined;
  wheels: WheelFrame[];
}

const WHEEL_LOCAL: readonly [number, number, number][] = [
  [+CAR_DIMENSIONS.wheelbase / 2, CAR_DIMENSIONS.wheelRadius, -CAR_DIMENSIONS.track / 2],
  [+CAR_DIMENSIONS.wheelbase / 2, CAR_DIMENSIONS.wheelRadius, +CAR_DIMENSIONS.track / 2],
  [-CAR_DIMENSIONS.wheelbase / 2, CAR_DIMENSIONS.wheelRadius, -CAR_DIMENSIONS.track / 2],
  [-CAR_DIMENSIONS.wheelbase / 2, CAR_DIMENSIONS.wheelRadius, +CAR_DIMENSIONS.track / 2],
];

function frameSource(frame: Frame): WheelStateSource {
  return {
    x: frame.x,
    y: frame.y,
    z: frame.z,
    yaw: frame.yaw,
    pitch: frame.pitch,
    roll: frame.roll,
    s: frame.s,
    v: frame.v,
    airborne: Boolean(frame.air),
    height: frame.hgt ?? 0,
    surface: frame.surf,
    wheels: frame.w ?? [],
  };
}

/**
 * Maintains four stable state objects, avoiding per-frame allocations in every
 * downstream effect. `fromFrame` also lets seek reconstruction replay exactly
 * the recorded 30 Hz samples.
 */
export class WheelVisualController {
  readonly states: WheelVisualState[] = Array.from({ length: 4 }, (_, index) => ({
    index,
    position: new Vector3(),
    contactPoint: new Vector3(),
    forward: new Vector3(),
    right: new Vector3(),
    normal: new Vector3(),
    contact: true,
    compression: 0,
    load: 0,
    slipRatio: 0,
    slipAngle: 0,
    speed: 0,
    surface: "gravel",
  }));

  private readonly rootPosition = new Vector3();
  private readonly rootRotation = new Quaternion();
  private readonly local = new Vector3();

  constructor(private readonly road: StageSurface) {}

  update(sample: PlaybackSample, rig: CarRig): readonly WheelVisualState[] {
    rig.root.updateMatrixWorld(true);
    for (let i = 0; i < 4; i++) {
      rig.wheels[i]!.getWorldPosition(this.states[i]!.position);
    }
    this.populate(sample, rig.compression);
    return this.states;
  }

  fromFrame(frame: Frame): readonly WheelVisualState[] {
    const source = frameSource(frame);
    xyz(source.x, source.y, source.z, this.rootPosition);
    orientation(
      source.yaw,
      source.pitch,
      source.roll,
      this.rootRotation,
    );
    for (let i = 0; i < 4; i++) {
      const [x, y, z] = WHEEL_LOCAL[i]!;
      this.states[i]!.position
        .copy(this.local.set(x, y, z))
        .applyQuaternion(this.rootRotation)
        .add(this.rootPosition);
    }
    const compression = Array.from({ length: 4 }, (_, i) =>
      wheelVisualCompression(source.wheels[i], i, source.airborne),
    );
    this.populate(source, compression);
    return this.states;
  }

  private populate(
    source: WheelStateSource,
    compression: readonly number[],
  ): void {
    const road = this.road.sample(source.s);
    for (let i = 0; i < 4; i++) {
      const wheel = source.wheels[i];
      const state = this.states[i]!;
      this.road.project(state.position, source.s, state.contactPoint)
        .addScaledVector(road.normal, ROAD_LIFT);
      state.forward.copy(road.forward);
      state.right.copy(road.right);
      state.normal.copy(road.normal);
      state.contact =
        !source.airborne &&
        source.height < 0.12 &&
        wheel?.con !== false &&
        (wheel?.fz === undefined || wheel.fz > 35);
      state.compression = compression[i] ?? 0;
      state.load = wheel?.fz ?? 0;
      state.slipRatio = wheel?.sr ?? 0;
      state.slipAngle = wheel?.sa ?? 0;
      state.speed = Math.abs(source.v);
      state.surface = source.surface ?? road.surface;
    }
  }
}

/** Quaternion whose local x/y/z axes are road forward/normal/right. */
export function roadQuaternion(
  road: RoadFrame,
  out = new Quaternion(),
): Quaternion {
  const matrix = new Matrix4().makeBasis(
    road.forward,
    road.normal,
    road.right,
  );
  return out.setFromRotationMatrix(matrix);
}
