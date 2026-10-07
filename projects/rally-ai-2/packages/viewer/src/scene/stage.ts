/**
 * The road, built from the stage document.
 *
 * The corridor is a ribbon swept along the centerline: at each sample the
 * cross-section is the full `width` rotated by `camber` about the tangent. Both
 * the physics and this renderer read the same geometry, so the road the car
 * drives and the road you see are the same surface by construction — not by two
 * implementations agreeing.
 *
 * The road is split into **one mesh per contiguous surface run**, because each
 * surface has its own texture. Runs share their boundary vertices, so a gravel
 * to tarmac change is a hard seam across the road with no gap and no overlap —
 * which is exactly what it looks like in reality.
 *
 * ## Camber, and the sign that was wrong
 *
 * Positive camber raises the **right** edge, which is what banks a road into a
 * left-hander. That is the contract's definition and `geometry.py` matches it,
 * computing surface height as `z + lat * sin(camber)`.
 *
 * The sim shipped with this backwards at the physics bridge — the vendored
 * model's `bank` is positive when the LEFT side is up — so every corner the
 * generator banked into the turn was played off-camber. If a replay ever looks
 * like the car is leaning out of banked corners, that bug is back, and it is
 * not a renderer bug.
 */

import {
  BufferAttribute,
  BufferGeometry,
  Color,
  DoubleSide,
  Group,
  Material,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  Vector3,
} from "three";

import { rightDir, xyz } from "../coords";
import type { CenterlinePoint, Stage, SurfaceKind } from "../replay";
import { surfaceAt } from "../replay";
import { textures } from "./textures";

/** Verge and far-terrain tints, multiplied over the shared grass texture. */
const VERGE_TINT: Record<SurfaceKind, number> = {
  gravel: 0xa8bf90,
  tarmac: 0xa8bf90,
  snow: 0xe4ecf2,
  mud: 0x93a878,
};

const TERRAIN_TINT: Record<SurfaceKind, number> = {
  gravel: 0x6d8560,
  tarmac: 0x6d8560,
  snow: 0xbcc8d2,
  mud: 0x5f7050,
};

const VERGE_M = 6.0;
const VERGE_DROP_M = 0.35;
const DITCH_WIDTH_M = 0.82;
const DITCH_DROP_M = 0.14;

/**
 * Terrain skirt: how far the ground extends past the verge, and how far it
 * falls over that distance.
 *
 * The sim has no terrain — only a corridor — so there is nothing here to be
 * faithful to. What there IS to get right is that the ground must reach past
 * the fog, because a world that visibly ends in mid-air reads as a broken
 * renderer. It follows the centerline elevation rather than sitting at a fixed
 * height, so it stays put under a stage that climbs 40 m.
 */
const SKIRT_M = 110.0;
const SKIRT_DROP_M = 5.0;

/**
 * Vertical terrain fall at a distance beyond the road edge.
 *
 * Shared with roadside scenery so cosmetic trees, banners and course stakes
 * sit on the same piecewise-linear verge/skirt surface the ground mesh draws.
 */
export function terrainFall(depthFromRoadEdge: number): number {
  const d = Math.max(0, depthFromRoadEdge);
  if (d <= VERGE_M) return (d / VERGE_M) * VERGE_DROP_M;
  return (
    VERGE_DROP_M +
    Math.min((d - VERGE_M) / SKIRT_M, 1) * SKIRT_DROP_M
  );
}

/** Metres of road per texture repeat, along and across. */
const ROAD_REPEAT_M = 7.0;
const GRASS_REPEAT_M = 9.0;

/**
 * How far the road is extended past the start and finish lines.
 *
 * The corridor is defined over 0..length_m exactly, so without this the world
 * stops at the start line — and the start line is the first thing anyone sees
 * on every replay.
 */
const APRON_M = 45.0;

/**
 * Tangent heading at a centerline sample, in radians about +z.
 *
 * Central difference where possible. The endpoints one-side it rather than
 * wrapping: the stage is point-to-point, and wrapping would splice the finish
 * into the start line — the same class of mistake as a lap-wrapped look-ahead
 * on a stage that does not lap.
 */
function headingAt(pts: CenterlinePoint[], i: number): number {
  const a = pts[Math.max(0, i - 1)]!;
  const b = pts[Math.min(pts.length - 1, i + 1)]!;
  return Math.atan2(b.y - a.y, b.x - a.x);
}

/** The centerline with an apron spliced onto each end. */
function withApron(pts: CenterlinePoint[]): CenterlinePoint[] {
  const extend = (from: CenterlinePoint, heading: number, dist: number): CenterlinePoint => ({
    s: from.s + dist,
    x: from.x + Math.cos(heading) * dist,
    y: from.y + Math.sin(heading) * dist,
    z: from.z,
    width: from.width,
    camber: from.camber,
  });
  const first = pts[0]!;
  const last = pts[pts.length - 1]!;
  return [
    extend(first, headingAt(pts, 0), -APRON_M),
    ...pts,
    extend(last, headingAt(pts, pts.length - 1), APRON_M),
  ];
}

interface Ribbon {
  positions: number[];
  uvs: number[];
  colors: number[];
  indices: number[];
  rows: number;
}

const emptyRibbon = (): Ribbon => ({
  positions: [],
  uvs: [],
  colors: [],
  indices: [],
  rows: 0,
});

function pushVertex(r: Ribbon, v: Vector3, u: number, vv: number, c: Color): void {
  r.positions.push(v.x, v.y, v.z);
  r.uvs.push(u, vv);
  r.colors.push(c.r, c.g, c.b);
}

/** Two triangles joining vertex pairs (a0,a1) and (b0,b1). */
function pushQuad(r: Ribbon, a0: number, a1: number, b0: number, b1: number): void {
  r.indices.push(a0, b0, a1, a1, b0, b1);
}

function toMesh(r: Ribbon, name: string, material: Material): Mesh | null {
  if (r.indices.length === 0) return null;
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(new Float32Array(r.positions), 3));
  g.setAttribute("uv", new BufferAttribute(new Float32Array(r.uvs), 2));
  g.setAttribute("color", new BufferAttribute(new Float32Array(r.colors), 3));
  g.setIndex(r.indices);
  g.computeVertexNormals();
  g.computeBoundingSphere();
  const m = new Mesh(g, material);
  m.name = name;
  return m;
}

/**
 * Cross-section helper: a point `d` metres right of the centerline, at `fall`
 * metres below the road plane. Positive camber lifts the right edge, so +d
 * gains height.
 */
function crossSection(
  centre: Vector3,
  right: Vector3,
  camber: number,
  d: number,
  fall: number,
  out: Vector3,
): Vector3 {
  const cc = Math.cos(camber);
  const sc = Math.sin(camber);
  return out
    .copy(centre)
    .addScaledVector(right, d * cc)
    .setY(centre.y + d * sc - fall);
}

/** Build the road, its verges, terrain, and the start/finish markers. */
export function buildStage(stage: Stage): Group {
  const tex = textures();
  const pts = withApron(stage.centerline);
  const group = new Group();
  group.name = "stage";

  const centre = new Vector3();
  const right = new Vector3();
  const edge = new Vector3();
  // The road texture already carries the surface's colour, so the road's own
  // vertex colours are neutral and the texture is what you see.
  const white = new Color(0xffffff);
  const vergeColour = new Color();
  const terrainColour = new Color();

  // --- terrain and verge: one mesh, since they share the grass texture ---
  const ground = emptyRibbon();
  const roadEdge = emptyRibbon();

  // --- road: a ribbon per contiguous surface run ---
  let runKind: SurfaceKind | null = null;
  let run = emptyRibbon();
  const roadMeshes: Mesh[] = [];

  const closeRun = (): void => {
    if (runKind === null) return;
    const mesh = toMesh(run, `road_${runKind}`, new MeshLambertMaterial({
      map: tex.road[runKind],
      vertexColors: true,
      side: DoubleSide,
      flatShading: true,
    }));
    if (mesh) roadMeshes.push(mesh);
  };

  for (let i = 0; i < pts.length; i++) {
    const p = pts[i]!;
    const h = headingAt(pts, i);
    const half = p.width / 2;

    xyz(p.x, p.y, p.z, centre);
    rightDir(h, right);

    // Aprons inherit the nearest legal stage surface. `surfaceAt` deliberately
    // falls back to the final segment for uncovered values, so passing -45 m
    // directly would paint the start apron with the finish surface.
    const surfaceS = Math.min(Math.max(p.s, 0), stage.length_m);
    const kind = surfaceAt(stage, surfaceS)?.type ?? "gravel";
    const vLong = p.s / ROAD_REPEAT_M;

    // Close the previous run AT this row, then repeat the same row in the new
    // run. Without the shared transition row, each material change leaves one
    // centerline-sample-length of grass visible across the road.
    if (kind !== runKind) {
      if (runKind !== null && run.rows > 0) {
        pushVertex(run, crossSection(centre, right, p.camber, -half, 0, edge), 0, vLong, white);
        pushVertex(run, crossSection(centre, right, p.camber, +half, 0, edge), 1, vLong, white);
        run.rows++;
        const a = (run.rows - 2) * 2;
        const b = (run.rows - 1) * 2;
        pushQuad(run, a, a + 1, b, b + 1);
      }
      closeRun();
      run = emptyRibbon();
      runKind = kind;
    }

    pushVertex(run, crossSection(centre, right, p.camber, -half, 0, edge), 0, vLong, white);
    pushVertex(run, crossSection(centre, right, p.camber, +half, 0, edge), 1, vLong, white);
    run.rows++;
    if (run.rows > 1) {
      const a = (run.rows - 2) * 2;
      const b = (run.rows - 1) * 2;
      pushQuad(run, a, a + 1, b, b + 1);
    }

    // --- ground: outer-left, verge-left, road-left, road-right, verge-right,
    //     outer-right ---
    vergeColour.setHex(VERGE_TINT[kind]);
    terrainColour.setHex(TERRAIN_TINT[kind]);
    const vG = p.s / GRASS_REPEAT_M;
    const uOuter = (SKIRT_M + VERGE_M) / GRASS_REPEAT_M;
    const uVerge = VERGE_M / GRASS_REPEAT_M;

    pushVertex(ground, crossSection(centre, right, p.camber, -half - VERGE_M - SKIRT_M, VERGE_DROP_M + SKIRT_DROP_M, edge), -uOuter, vG, terrainColour);
    pushVertex(ground, crossSection(centre, right, p.camber, -half - VERGE_M, VERGE_DROP_M, edge), -uVerge, vG, vergeColour);
    pushVertex(ground, crossSection(centre, right, p.camber, -half, 0, edge), 0, vG, vergeColour);
    pushVertex(ground, crossSection(centre, right, p.camber, +half, 0, edge), 0, vG, vergeColour);
    pushVertex(ground, crossSection(centre, right, p.camber, +half + VERGE_M, VERGE_DROP_M, edge), uVerge, vG, vergeColour);
    pushVertex(ground, crossSection(centre, right, p.camber, +half + VERGE_M + SKIRT_M, VERGE_DROP_M + SKIRT_DROP_M, edge), uOuter, vG, terrainColour);
    ground.rows++;

    if (ground.rows > 1) {
      const a = (ground.rows - 2) * 6;
      const b = (ground.rows - 1) * 6;
      pushQuad(ground, a, a + 1, b, b + 1); // left skirt
      pushQuad(ground, a + 1, a + 2, b + 1, b + 2); // left shoulder
      pushQuad(ground, a + 3, a + 4, b + 3, b + 4); // right shoulder
      pushQuad(ground, a + 4, a + 5, b + 4, b + 5); // right skirt
    }

    // Narrow, shallow roadside drainage cuts. These sit on the same sampled
    // cross-section as the road and replace the old perfectly clean road/grass
    // seam with a coarse, textured edge that still follows camber exactly.
    pushVertex(
      roadEdge,
      crossSection(
        centre,
        right,
        p.camber,
        -half - DITCH_WIDTH_M,
        DITCH_DROP_M,
        edge,
      ),
      0,
      vLong,
      white,
    );
    pushVertex(
      roadEdge,
      crossSection(centre, right, p.camber, -half, 0, edge),
      1,
      vLong,
      white,
    );
    pushVertex(
      roadEdge,
      crossSection(centre, right, p.camber, +half, 0, edge),
      0,
      vLong,
      white,
    );
    pushVertex(
      roadEdge,
      crossSection(
        centre,
        right,
        p.camber,
        +half + DITCH_WIDTH_M,
        DITCH_DROP_M,
        edge,
      ),
      1,
      vLong,
      white,
    );
    roadEdge.rows++;
    if (roadEdge.rows > 1) {
      const a = (roadEdge.rows - 2) * 4;
      const b = (roadEdge.rows - 1) * 4;
      pushQuad(roadEdge, a, a + 1, b, b + 1);
      pushQuad(roadEdge, a + 2, a + 3, b + 2, b + 3);
    }
  }
  closeRun();

  // Ground first: it shares the road's edge vertices exactly, and drawing the
  // road after lets it win the depth fight along that seam.
  const groundMesh = toMesh(ground, "ground", new MeshLambertMaterial({
    map: tex.grass,
    vertexColors: true,
    side: DoubleSide,
    flatShading: true,
  }));
  if (groundMesh) group.add(groundMesh);
  const roadEdgeMesh = toMesh(
    roadEdge,
    "textured_road_edge_ditches",
    new MeshLambertMaterial({
      map: tex.ditch,
      vertexColors: true,
      side: DoubleSide,
      flatShading: true,
    }),
  );
  if (roadEdgeMesh) group.add(roadEdgeMesh);
  for (const m of roadMeshes) group.add(m);

  group.add(buildLine(stage, stage.start.s, 0xffffff));
  group.add(buildLine(stage, stage.finish.s, 0xd83a3a));
  return group;
}

/**
 * A banner across the corridor at an arc length — start and finish.
 *
 * Placed by interpolating the centerline rather than by index, so it lands at
 * the declared `s` even though samples are unevenly spaced.
 */
function buildLine(stage: Stage, s: number, colour: number): Mesh {
  const pts = stage.centerline;
  let i = pts.findIndex((p) => p.s >= s);
  if (i < 0) i = pts.length - 1;
  const i0 = Math.max(0, i - 1);
  const a = pts[i0]!;
  const b = pts[Math.min(pts.length - 1, i0 + 1)]!;
  const span = b.s - a.s;
  const f = span > 1e-9 ? (s - a.s) / span : 0;

  const p = {
    x: a.x + (b.x - a.x) * f,
    y: a.y + (b.y - a.y) * f,
    z: a.z + (b.z - a.z) * f,
    width: a.width + (b.width - a.width) * f,
    camber: a.camber + (b.camber - a.camber) * f,
  };
  const h = Math.atan2(b.y - a.y, b.x - a.x);
  const half = p.width / 2;
  const centre = xyz(p.x, p.y, p.z);
  const right = rightDir(h);
  const fwd = xyz(Math.cos(h), Math.sin(h), 0);
  const DEPTH = 1.4;
  const LIFT = 0.02; // clear of the road, below the car

  const r = emptyRibbon();
  const c = new Color(colour);
  const v = new Vector3();
  let k = 0;
  for (const along of [-DEPTH / 2, DEPTH / 2]) {
    for (const lat of [-half, half]) {
      crossSection(centre, right, p.camber, lat, -LIFT, v).addScaledVector(fwd, along);
      pushVertex(r, v, k % 2, Math.floor(k / 2), c);
      k++;
    }
  }
  pushQuad(r, 0, 1, 2, 3);
  return toMesh(r, `line_${s.toFixed(0)}`, new MeshBasicMaterial({
    color: 0xffffff,
    vertexColors: true,
    side: DoubleSide,
  }))!;
}
