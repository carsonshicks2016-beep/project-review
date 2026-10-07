/**
 * Bounded, replay-safe tyre marks.
 *
 * The mesh is presentation only and is rebuilt from recorded frames after a
 * seek. A hard segment cap prevents a long replay from growing GPU memory.
 */

import {
  BufferGeometry,
  Color,
  DoubleSide,
  Float32BufferAttribute,
  Mesh,
  MeshBasicMaterial,
  Vector3,
} from "three";

import type { LoadedReplay, SurfaceKind } from "../replay";
import type {
  WheelVisualController,
  WheelVisualState,
} from "./surface";

const MAX_SEGMENTS = 1400;
const TYRE_MARK_WIDTH = 0.24;
const MIN_STEP = 0.11;
const MAX_STEP = 2.2;

const TRACK_COLOUR: Record<SurfaceKind, number> = {
  gravel: 0xa99d85,
  tarmac: 0x090a0c,
  mud: 0x34261a,
  snow: 0xa9b8c4,
};

interface TrackEnd {
  point: Vector3;
  right: Vector3;
  surface: SurfaceKind;
}

interface TrackSegment {
  a: Vector3;
  b: Vector3;
  rightA: Vector3;
  rightB: Vector3;
  surface: SurfaceKind;
  strength: number;
}

function shouldMark(wheel: WheelVisualState): number {
  if (!wheel.contact || wheel.speed < 2) return 0;
  const longitudinal = Math.abs(wheel.slipRatio);
  const lateral = Math.abs(wheel.slipAngle);
  switch (wheel.surface) {
    case "tarmac":
      return Math.max(longitudinal - 0.10, lateral - 0.14) * 3.2;
    case "mud":
      return 0.62 + Math.min(0.38, longitudinal + lateral);
    case "snow":
      return 0.48 + Math.min(0.42, longitudinal + lateral);
    case "gravel":
    default:
      return 0.30 + Math.min(0.50, longitudinal * 1.8 + lateral * 0.8);
  }
}

export class TyreTracks {
  readonly root: Mesh;

  private readonly geometry = new BufferGeometry();
  private readonly segments: TrackSegment[] = [];
  private readonly previous: Array<TrackEnd | undefined> = [
    undefined,
    undefined,
    undefined,
    undefined,
  ];
  private readonly delta = new Vector3();
  private readonly colour = new Color();

  constructor() {
    this.root = new Mesh(
      this.geometry,
      new MeshBasicMaterial({
        color: 0xffffff,
        vertexColors: true,
        transparent: true,
        opacity: 0.78,
        depthWrite: false,
        fog: true,
        side: DoubleSide,
      }),
    );
    this.root.name = "persistent_tyre_tracks";
    this.root.renderOrder = 1;
    this.root.frustumCulled = false;
    this.refreshGeometry();
  }

  clear(): void {
    this.segments.length = 0;
    this.previous.fill(undefined);
    this.refreshGeometry();
  }

  update(wheels: readonly WheelVisualState[]): void {
    if (this.append(wheels)) this.refreshGeometry();
  }

  /**
   * Reconstruct marks from authoritative 30 Hz frames. Old history naturally
   * falls out of the ring cap, so a seek has the same bounded result as playing
   * forward to that timestamp.
   */
  rebuild(
    replay: LoadedReplay,
    time: number,
    controller: WheelVisualController,
  ): void {
    this.segments.length = 0;
    this.previous.fill(undefined);
    for (const frame of replay.frames) {
      if (frame.t > time + 1e-6) break;
      this.append(controller.fromFrame(frame));
    }
    this.refreshGeometry();
  }

  private append(wheels: readonly WheelVisualState[]): boolean {
    let changed = false;
    for (const wheel of wheels) {
      const strength = shouldMark(wheel);
      const previous = this.previous[wheel.index];
      if (strength <= 0) {
        this.previous[wheel.index] = undefined;
        continue;
      }

      if (!previous || previous.surface !== wheel.surface) {
        this.previous[wheel.index] = {
          point: wheel.contactPoint.clone(),
          right: wheel.right.clone(),
          surface: wheel.surface,
        };
        continue;
      }

      const distance = this.delta
        .copy(wheel.contactPoint)
        .sub(previous.point)
        .length();
      if (distance > MAX_STEP) {
        previous.point.copy(wheel.contactPoint);
        previous.right.copy(wheel.right);
        previous.surface = wheel.surface;
        continue;
      }
      if (distance < MIN_STEP) continue;

      this.segments.push({
        a: previous.point.clone(),
        b: wheel.contactPoint.clone(),
        rightA: previous.right.clone(),
        rightB: wheel.right.clone(),
        surface: wheel.surface,
        strength: Math.min(1, strength),
      });
      if (this.segments.length > MAX_SEGMENTS) {
        this.segments.splice(0, this.segments.length - MAX_SEGMENTS);
      }
      previous.point.copy(wheel.contactPoint);
      previous.right.copy(wheel.right);
      changed = true;
    }
    return changed;
  }

  private refreshGeometry(): void {
    const positions: number[] = [];
    const colours: number[] = [];
    const half = TYRE_MARK_WIDTH / 2;
    const count = Math.max(1, this.segments.length);
    for (let i = 0; i < this.segments.length; i++) {
      const segment = this.segments[i]!;
      const ageFade = 0.42 + 0.58 * ((i + 1) / count);
      this.colour
        .setHex(TRACK_COLOUR[segment.surface])
        .multiplyScalar(
          ageFade * (0.62 + segment.strength * 0.38),
        );
      const aLeft = segment.a.clone().addScaledVector(segment.rightA, -half);
      const aRight = segment.a.clone().addScaledVector(segment.rightA, half);
      const bLeft = segment.b.clone().addScaledVector(segment.rightB, -half);
      const bRight = segment.b.clone().addScaledVector(segment.rightB, half);
      for (const point of [
        aLeft,
        bLeft,
        aRight,
        aRight,
        bLeft,
        bRight,
      ]) {
        positions.push(point.x, point.y, point.z);
        colours.push(this.colour.r, this.colour.g, this.colour.b);
      }
    }
    this.geometry.setAttribute(
      "position",
      new Float32BufferAttribute(positions, 3),
    );
    this.geometry.setAttribute(
      "color",
      new Float32BufferAttribute(colours, 3),
    );
    this.geometry.computeBoundingSphere();
  }
}
