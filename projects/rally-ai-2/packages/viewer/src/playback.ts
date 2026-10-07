/**
 * Playback clock and frame interpolation.
 *
 * Replays are recorded at the control rate (30 Hz) and watched at whatever the
 * display runs at, so frames are interpolated rather than snapped. Without it
 * the car visibly steps at speed — at 40 m/s a 30 Hz frame is 1.3 m of road.
 *
 * Angles are interpolated the short way round. Yaw crosses ±pi on any stage
 * with a hairpin, and a naive lerp spins the car through a full revolution in
 * one frame. It looks exactly like a physics explosion and is not one.
 */

import type {
  Frame,
  Replay,
  SurfaceKind,
  WheelFrame,
} from "./replay";

/** Shortest-path angular interpolation, radians. */
export function lerpAngle(a: number, b: number, f: number): number {
  let d = b - a;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return a + d * f;
}

const lerp = (a: number, b: number, f: number): number => a + (b - a) * f;

function lerpOptional(
  a: number | undefined,
  b: number | undefined,
  f: number,
): number | undefined {
  if (a === undefined) return b;
  if (b === undefined) return a;
  return lerp(a, b, f);
}

function sampleWheel(
  a: WheelFrame | undefined,
  b: WheelFrame | undefined,
  f: number,
): WheelFrame {
  const out: WheelFrame = {};
  for (const key of ["sa", "sr", "fz", "grip", "comp", "om"] as const) {
    const value = lerpOptional(a?.[key], b?.[key], f);
    if (value !== undefined) out[key] = value;
  }
  const contact = f < 0.5 ? a?.con : b?.con;
  if (contact !== undefined) out.con = contact;
  return out;
}

export interface PlaybackSample {
  t: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  v: number;
  s: number;
  vx: number;
  vy: number;
  vz: number;
  steer: number;
  throttle: number;
  brake: number;
  handbrake: number;
  rpm: number;
  gear: number;
  /** Turbo boost 0..1, interpolated. 0 where the replay predates the field. */
  boost: number;
  slipAngle: number;
  airborne: boolean;
  height: number;
  surface: SurfaceKind | undefined;
  /** Pace note text of the frame at-or-before now — discrete, like gear. */
  note: string;
  wheels: WheelFrame[];
}

export class Playback {
  /** Stage time in seconds. The single source of truth for where we are. */
  time = 0;
  playing = true;
  rate = 1;

  private readonly frames: Frame[];
  private readonly dt: number;
  readonly duration: number;

  constructor(replay: Replay) {
    this.frames = replay.frames;
    this.dt = replay.dt;
    // From the frame stamps, not from meta.time_s: if a replay is ever
    // truncated on write, trusting the header would scrub past the end.
    this.duration = this.frames[this.frames.length - 1]!.t;
  }

  /** Advance by wall-clock seconds. Stops at the end rather than wrapping. */
  advance(wallDt: number): void {
    if (!this.playing) return;
    this.time += wallDt * this.rate;
    if (this.time >= this.duration) {
      this.time = this.duration;
      this.playing = false;
    } else if (this.time < 0) {
      this.time = 0;
    }
  }

  seek(t: number): void {
    this.time = Math.min(Math.max(t, 0), this.duration);
  }

  togglePlay(): void {
    // Restarting from the end is the common case after watching a run through.
    if (!this.playing && this.time >= this.duration) this.time = 0;
    this.playing = !this.playing;
  }

  /** Index of the frame at or before the current time. */
  get index(): number {
    const i = Math.floor(this.time / this.dt);
    return Math.min(Math.max(i, 0), this.frames.length - 1);
  }

  /** The frame at or before now — for discrete values like gear and surface. */
  get frame(): Frame {
    return this.frames[this.index]!;
  }

  /**
   * Interpolated pose and continuous channels at the current time.
   *
   * Only the values that are visibly stepped get interpolated. Gear, surface
   * and the like are read from {@link frame} instead, because interpolating a
   * gear number produces 3.4 and there is no such gear.
   */
  sample(): PlaybackSample {
    const i = this.index;
    const a = this.frames[i]!;
    const b = this.frames[Math.min(i + 1, this.frames.length - 1)]!;
    const span = b.t - a.t;
    const f = span > 1e-9 ? Math.min(Math.max((this.time - a.t) / span, 0), 1) : 0;
    const wheels = Array.from({ length: 4 }, (_, wheel) =>
      sampleWheel(a.w?.[wheel], b.w?.[wheel], f),
    );

    return {
      t: this.time,
      x: lerp(a.x, b.x, f),
      y: lerp(a.y, b.y, f),
      z: lerp(a.z, b.z, f),
      yaw: lerpAngle(a.yaw, b.yaw, f),
      pitch: lerpAngle(a.pitch, b.pitch, f),
      roll: lerpAngle(a.roll, b.roll, f),
      v: lerp(a.v, b.v, f),
      s: lerp(a.s, b.s, f),
      vx: lerp(a.vx ?? 0, b.vx ?? 0, f),
      vy: lerp(a.vy ?? 0, b.vy ?? 0, f),
      vz: lerp(a.vz ?? 0, b.vz ?? 0, f),
      steer: lerp(a.in?.s ?? 0, b.in?.s ?? 0, f),
      throttle: lerp(a.in?.t ?? 0, b.in?.t ?? 0, f),
      brake: lerp(a.in?.b ?? 0, b.in?.b ?? 0, f),
      handbrake: lerp(a.in?.h ?? 0, b.in?.h ?? 0, f),
      rpm: lerp(a.rpm ?? 0, b.rpm ?? 0, f),
      gear: a.gear ?? 0,
      boost: lerp(a.boost ?? 0, b.boost ?? 0, f),
      slipAngle: lerp(a.sa ?? 0, b.sa ?? 0, f),
      airborne: f < 0.5 ? Boolean(a.air) : Boolean(b.air),
      height: lerp(a.hgt ?? 0, b.hgt ?? 0, f),
      surface: (f < 0.5 ? a.surf : b.surf) ?? a.surf ?? b.surf,
      note: a.note ?? "",
      wheels,
    };
  }
}
