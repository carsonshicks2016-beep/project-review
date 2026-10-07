/**
 * The replay contract, as TypeScript.
 *
 * These types are a hand-maintained mirror of
 * `packages/shared/schemas/replay.schema.json` and `stage.schema.json`. The
 * schema is authoritative — where these disagree, the schema wins and this file
 * is the bug.
 *
 * Fields are optional here exactly where the schema leaves them optional, so
 * the compiler forces a decision about missing data rather than letting
 * `undefined` reach the renderer as a silent NaN.
 */

/** One corridor sample. `width` is FULL width; half-width is `width / 2`. */
export interface CenterlinePoint {
  s: number;
  x: number;
  y: number;
  z: number;
  width: number;
  camber: number;
}

export interface SurfaceSegment {
  s_start: number;
  s_end: number;
  type: SurfaceKind;
  mu: number;
}

export type SurfaceKind = "gravel" | "tarmac" | "snow" | "mud";

export interface Obstacle {
  kind: "tree" | "rock" | "bank" | "post" | "hay_bale";
  x: number;
  y: number;
  z: number;
  radius: number;
  height: number;
  s: number;
  lateral: number;
}

export interface PaceNote {
  s: number;
  dir: "left" | "right" | "straight" | "crest" | "jump" | "dip" | "caution";
  severity: number;
  modifiers: string[];
  text: string;
}

export interface Stage {
  schema_version: number;
  id: string;
  name: string;
  length_m: number;
  centerline: CenterlinePoint[];
  surfaces: SurfaceSegment[];
  obstacles: Obstacle[];
  pace_notes: PaceNote[];
  start: { s: number; heading: number };
  finish: { s: number };
  generator?: { version: number; tier: number; seed?: number; authored?: boolean };
  content_hash?: string;
  meta?: Record<string, unknown>;
}

/** Per-wheel state, order FL, FR, RL, RR. */
export interface WheelFrame {
  sa?: number;
  sr?: number;
  fz?: number;
  grip?: number;
  comp?: number;
  om?: number;
  con?: boolean;
}

export interface Frame {
  t: number;
  x: number;
  y: number;
  z: number;
  yaw: number;
  pitch: number;
  roll: number;
  v: number;
  s: number;

  vx?: number;
  vy?: number;
  vz?: number;
  yr?: number;
  /** Chassis slip angle. The opposite-lock tell. */
  sa?: number;

  gear?: number;
  rpm?: number;
  boost?: number;

  in?: { s: number; t: number; b: number; h: number };
  w?: WheelFrame[];

  air?: boolean;
  hgt?: number;
  surf?: SurfaceKind;
  mu?: number;

  prog?: number;
  lat?: number;
  note?: string;
  fx?: Record<string, unknown>;
}

export type Termination =
  | "finish"
  | "crash"
  | "off_course"
  | "stuck"
  | "spun"
  | "timeout"
  | "aborted";

export interface ReplayMeta {
  termination: Termination;
  time_s: number;
  car: string;
  physics_version: string;
  seed?: number;
  checkpoint?: string;
  policy_hash?: string;
  clean?: boolean;
  obstacle_contacts?: number;
  off_course_s?: number;
}

export interface ReplayEvent {
  t: number;
  kind: string;
  severity?: number;
}

export interface Replay {
  schema_version: number;
  dt: number;
  source: "agent" | "human" | "eval" | "replay_test";
  meta: ReplayMeta;
  frames: Frame[];
  events?: ReplayEvent[];
  stage?: Stage;
  stage_ref?: { id: string; content_hash: string };
  content_hash?: string;
}

/** A replay that is guaranteed to carry its stage inline. */
export interface LoadedReplay extends Replay {
  stage: Stage;
}

/**
 * Fetch and structurally check a replay.
 *
 * This is deliberately not a JSON Schema validator — the sim already validated
 * on write and stamped a content hash. What it checks is the handful of things
 * whose absence would otherwise surface as a blank screen or a NaN transform
 * twenty frames in, which is far harder to diagnose than a thrown error here.
 */
export async function loadReplay(url: string): Promise<LoadedReplay> {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`replay ${url}: HTTP ${res.status} ${res.statusText}`);
  }
  const doc = (await res.json()) as Replay;
  return checkReplay(doc, url);
}

/** The structural checks, split out so tests can run them without fetching. */
export function checkReplay(doc: Replay, label = "replay"): LoadedReplay {
  if (doc.schema_version !== 1) {
    throw new Error(`${label}: schema_version ${doc.schema_version}, expected 1`);
  }
  if (!doc.stage) {
    // `stage_ref` is legal in the contract but means fetching and hash-checking
    // a stage this viewer has no store for. Fail loudly rather than half-render.
    throw new Error(
      `${label}: no inline stage. stage_ref is valid in the schema but ` +
        `unsupported here — export the replay with the stage embedded.`,
    );
  }
  if (!Array.isArray(doc.frames) || doc.frames.length === 0) {
    throw new Error(`${label}: no frames`);
  }
  if (!(doc.dt > 0)) {
    throw new Error(`${label}: dt is ${doc.dt}; physics is fixed-step, not a hint`);
  }
  const cl = doc.stage.centerline;
  if (!Array.isArray(cl) || cl.length < 2) {
    throw new Error(`${label}: centerline has ${cl?.length ?? 0} points, need >= 2`);
  }
  return doc as LoadedReplay;
}

/** Surface type at an arc length, for tyre noise and road colour. */
export function surfaceAt(stage: Stage, s: number): SurfaceSegment | undefined {
  // Segments are contiguous and cover 0..length_m, so a linear scan over the
  // handful of them is cheaper than the bookkeeping to binary-search it.
  for (const seg of stage.surfaces) {
    if (s >= seg.s_start && s <= seg.s_end) return seg;
  }
  return stage.surfaces[stage.surfaces.length - 1];
}
