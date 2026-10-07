/**
 * TRAIN G2 — sensor debug overlay.
 *
 * Draws the agent's streamed Observation debug (beams + look-ahead) into the
 * Three.js scene. Endpoints come from `sense.beam_points` / `lookahead_points`
 * in **sim z-up** coordinates and are converted once via coords.ts.
 * Never recompute rays from car pose in the viewer.
 */

import {
  BufferAttribute,
  BufferGeometry,
  Color,
  Group,
  LineBasicMaterial,
  LineSegments,
  Mesh,
  MeshBasicMaterial,
  SphereGeometry,
  type Material,
} from "three";

import { xyz } from "../coords";

/** Corridor edge (survivable). Matches Track.HIT_EDGE. */
export const KIND_EDGE = 0;
/** Obstacle / tree (HIT_OBSTACLE). Matches Track.HIT_OBSTACLE. */
export const KIND_OBSTACLE = 1;

const COLOR_EDGE = new Color(0xf5e6a8);
const COLOR_OBSTACLE = new Color(0xe11d48);
const COLOR_LOOKAHEAD = 0xc8ff3d;
const COLOR_MISS = new Color(0x6a7058);

const N_BEAMS = 9;
const N_LOOK = 6;
/** Lift lines slightly so they sit above the road mesh (three y). */
const LIFT = 0.35;

export interface SensePaceNote {
  s?: number;
  dir?: string;
  severity?: number;
  text?: string;
}

/** Observation debug sidecar from F3 live envelope / optional frame.sense. */
export interface SenseDebug {
  beam_distances?: number[];
  beam_points: Array<number[]>;
  beam_kinds: number[];
  lookahead_s?: number[];
  lookahead_points: Array<number[]>;
  pace_note?: SensePaceNote | null;
}

export function isSenseDebug(v: unknown): v is SenseDebug {
  if (!v || typeof v !== "object") return false;
  const s = v as SenseDebug;
  return (
    Array.isArray(s.beam_points) &&
    Array.isArray(s.beam_kinds) &&
    Array.isArray(s.lookahead_points)
  );
}

/**
 * Parse sense from a live frame packet. Prefer the envelope `sense` field;
 * fall back to `frame.sense` if a concatenated replay embedded it.
 */
export function senseFromPacket(packet: {
  sense?: unknown;
  frame?: { sense?: unknown };
}): SenseDebug | null {
  if (isSenseDebug(packet.sense)) return packet.sense;
  if (packet.frame && isSenseDebug(packet.frame.sense)) return packet.frame.sense;
  return null;
}

export class SensorOverlay {
  readonly group = new Group();

  private readonly beamPos: Float32Array;
  private readonly beamCol: Float32Array;
  private readonly beamGeo: BufferGeometry;
  private readonly lookMeshes: Mesh[] = [];
  private readonly _o = xyz(0, 0, 0);
  private readonly _p = xyz(0, 0, 0);

  constructor() {
    this.group.name = "sensorDebug";
    this.group.renderOrder = 10;

    this.beamPos = new Float32Array(N_BEAMS * 2 * 3);
    this.beamCol = new Float32Array(N_BEAMS * 2 * 3);
    this.beamGeo = new BufferGeometry();
    this.beamGeo.setAttribute("position", new BufferAttribute(this.beamPos, 3));
    this.beamGeo.setAttribute("color", new BufferAttribute(this.beamCol, 3));
    const beamMat = new LineBasicMaterial({
      vertexColors: true,
      depthTest: true,
      transparent: true,
      opacity: 0.92,
    });
    const beams = new LineSegments(this.beamGeo, beamMat);
    beams.frustumCulled = false;
    this.group.add(beams);

    const lookGeo = new SphereGeometry(0.55, 6, 4);
    const lookMat = new MeshBasicMaterial({
      color: COLOR_LOOKAHEAD,
      depthTest: true,
      transparent: true,
      opacity: 0.95,
    });
    for (let i = 0; i < N_LOOK; i++) {
      const m = new Mesh(lookGeo, lookMat);
      m.visible = false;
      m.frustumCulled = false;
      this.lookMeshes.push(m);
      this.group.add(m);
    }

    this.group.visible = false;
  }

  /**
   * Apply streamed sense.
   * `origin` is the car pose from the same frame in **sim** coordinates
   * (line start only). Hit points and kinds are taken verbatim from `sense`.
   */
  apply(
    origin: { x: number; y: number; z: number },
    sense: SenseDebug | null | undefined,
  ): void {
    if (!sense || !sense.beam_points?.length) {
      this.group.visible = false;
      return;
    }
    this.group.visible = true;

    xyz(origin.x, origin.y, origin.z, this._o);
    const ox = this._o.x;
    const oy = this._o.y + LIFT;
    const oz = this._o.z;
    const n = Math.min(
      N_BEAMS,
      sense.beam_points.length,
      sense.beam_kinds?.length ?? 0,
    );

    for (let i = 0; i < N_BEAMS; i++) {
      const i0 = i * 6;
      this.beamPos[i0] = ox;
      this.beamPos[i0 + 1] = oy;
      this.beamPos[i0 + 2] = oz;

      if (i < n) {
        const p = sense.beam_points[i]!;
        const kind = sense.beam_kinds[i] ?? KIND_EDGE;
        xyz(p[0] ?? origin.x, p[1] ?? origin.y, p[2] ?? origin.z, this._p);
        this.beamPos[i0 + 3] = this._p.x;
        this.beamPos[i0 + 4] = this._p.y + LIFT;
        this.beamPos[i0 + 5] = this._p.z;
        const c = kind === KIND_OBSTACLE ? COLOR_OBSTACLE : COLOR_EDGE;
        this.beamCol[i0] = c.r;
        this.beamCol[i0 + 1] = c.g;
        this.beamCol[i0 + 2] = c.b;
        this.beamCol[i0 + 3] = c.r;
        this.beamCol[i0 + 4] = c.g;
        this.beamCol[i0 + 5] = c.b;
      } else {
        this.beamPos[i0 + 3] = ox;
        this.beamPos[i0 + 4] = oy;
        this.beamPos[i0 + 5] = oz;
        this.beamCol[i0] = COLOR_MISS.r;
        this.beamCol[i0 + 1] = COLOR_MISS.g;
        this.beamCol[i0 + 2] = COLOR_MISS.b;
        this.beamCol[i0 + 3] = COLOR_MISS.r;
        this.beamCol[i0 + 4] = COLOR_MISS.g;
        this.beamCol[i0 + 5] = COLOR_MISS.b;
      }
    }
    this.beamGeo.attributes.position!.needsUpdate = true;
    this.beamGeo.attributes.color!.needsUpdate = true;
    this.beamGeo.computeBoundingSphere();

    const looks = sense.lookahead_points ?? [];
    for (let i = 0; i < N_LOOK; i++) {
      const m = this.lookMeshes[i]!;
      const p = looks[i];
      if (!p) {
        m.visible = false;
        continue;
      }
      m.visible = true;
      const s = 0.75 + i * 0.12;
      m.scale.setScalar(s);
      xyz(p[0] ?? 0, p[1] ?? 0, p[2] ?? 0, this._p);
      m.position.set(this._p.x, this._p.y + LIFT + 0.2, this._p.z);
    }
  }

  clear(): void {
    this.group.visible = false;
    for (const m of this.lookMeshes) m.visible = false;
  }

  dispose(): void {
    this.group.removeFromParent();
    this.beamGeo.dispose();
    const beam = this.group.children[0] as LineSegments | undefined;
    if (beam?.material) (beam.material as Material).dispose();
    const look0 = this.lookMeshes[0];
    if (look0) {
      look0.geometry.dispose();
      (look0.material as Material).dispose();
    }
  }
}
