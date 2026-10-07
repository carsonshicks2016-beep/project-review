/**
 * A low-poly late-1990s works rally saloon.
 *
 * Dimensions still come from `evo_rally`'s CarSpec: wheelbase 2.51 m, track
 * 1.47 m, body 4.35 x 1.77 m, wheel radius 0.32 m and CG height 0.49 m. The
 * roof envelope is kept near the period saloon's 1.42 m rather than exaggerated
 * for the chase camera. The skin remains reference-led: deep works blue,
 * yellow 555-era graphics, gold wheels and a three-box Impreza silhouette.
 *
 * The project is personal and the user explicitly approved real historic
 * branding for this viewer skin. The shapes stay broad and graphic so the car
 * reads immediately at speed without depending on a pixelated framebuffer.
 *
 * Model frame is +x forward, +y up, +z right, matching `coords.ts`.
 */

import {
  BoxGeometry,
  BufferGeometry,
  CanvasTexture,
  CylinderGeometry,
  DoubleSide,
  Float32BufferAttribute,
  Group,
  LinearFilter,
  Mesh,
  MeshBasicMaterial,
  MeshLambertMaterial,
  PlaneGeometry,
  SRGBColorSpace,
  type Object3D,
  type Texture,
} from "three";

import type { WheelFrame } from "../replay";

const BODY_LENGTH = 4.35;
const BODY_WIDTH = 1.77;
const WHEELBASE = 2.51;
const TRACK = 1.47;
const WHEEL_R = 0.32;
const WHEEL_W = 0.27;
const CG_HEIGHT = 0.49;
const MASS = 1250;
const FRONT_WEIGHT = 0.58;
const MAX_BUMP = 0.085;
const MAX_DROOP = 0.095;
const LOAD_TO_TRAVEL = 0.055;

export const CAR_DIMENSIONS = {
  bodyLength: BODY_LENGTH,
  bodyWidth: BODY_WIDTH,
  wheelbase: WHEELBASE,
  track: TRACK,
  wheelRadius: WHEEL_R,
} as const;

/** Static per-corner loads from the authoritative `evo_rally` CarSpec. */
const STATIC_FZ = [
  (MASS * 9.81 * FRONT_WEIGHT) / 2,
  (MASS * 9.81 * FRONT_WEIGHT) / 2,
  (MASS * 9.81 * (1 - FRONT_WEIGHT)) / 2,
  (MASS * 9.81 * (1 - FRONT_WEIGHT)) / 2,
] as const;

const BLUE = 0x123d9a;
const BLUE_LIT = 0x1d55c5;
const BLUE_DARK = 0x071a4b;
const TYRE = 0x15171b;
const RIM = 0xb78c31;
const LAMP = 0xf3e8be;
const LAMP_RED = 0xd52b2f;
const PLATE = 0xd9ad22;
const TRIM = 0x161b24;
const MUD = 0x5f4a35;

function material(colour: number): MeshLambertMaterial {
  return new MeshLambertMaterial({
    color: colour,
    side: DoubleSide,
    flatShading: true,
  });
}

function box(
  w: number,
  h: number,
  d: number,
  colour: number,
  x = 0,
  y = 0,
  z = 0,
): Mesh {
  const m = new Mesh(new BoxGeometry(w, h, d), material(colour));
  m.position.set(x, y, z);
  return m;
}

/**
 * A faceted eight-vertex prism. Front/rear inset slopes the top in profile and
 * the narrower top width gives the body its shouldered, box-flared stance.
 */
function taperedPrism(
  length: number,
  height: number,
  bottomWidth: number,
  topWidth: number,
  frontInset: number,
  rearInset: number,
  colour: number,
  x: number,
  yBottom: number,
): Mesh {
  const xb0 = -length / 2;
  const xb1 = length / 2;
  const xt0 = xb0 + rearInset;
  const xt1 = xb1 - frontInset;
  const zb = bottomWidth / 2;
  const zt = topWidth / 2;
  const p = [
    xb0, 0, -zb, xb1, 0, -zb, xb1, 0, zb, xb0, 0, zb,
    xt0, height, -zt, xt1, height, -zt, xt1, height, zt, xt0, height, zt,
  ];
  const g = new BufferGeometry();
  g.setAttribute("position", new Float32BufferAttribute(p, 3));
  g.setIndex([
    0, 1, 2, 0, 2, 3,
    4, 6, 5, 4, 7, 6,
    0, 4, 5, 0, 5, 1,
    1, 5, 6, 1, 6, 2,
    2, 6, 7, 2, 7, 3,
    3, 7, 4, 3, 4, 0,
  ]);
  g.computeVertexNormals();
  const m = new Mesh(g, material(colour));
  m.position.set(x, yBottom, 0);
  return m;
}

interface BodyStation {
  /** Longitudinal position, rear to front. */
  x: number;
  bottomY: number;
  lowerHalf: number;
  shoulderY: number;
  shoulderHalf: number;
  deckY: number;
  deckHalf: number;
}

/**
 * One continuous, deliberately low-resolution body shell.
 *
 * The previous car was assembled from overlapping rectangular boxes. It had
 * the right proportions, but its silhouette read as voxel art. These eight
 * cross-sections create a single chamfered shell instead: the plan tapers at
 * both ends, bulges over each axle, and changes deck height in large, readable
 * polygons. There are still very few faces — the difference is that the faces
 * now describe a car rather than a stack of cubes.
 */
function bodyShell(sideMap: Texture): Mesh {
  const stations: readonly BodyStation[] = [
    { x: -2.175, bottomY: 0.29, lowerHalf: 0.60, shoulderY: 0.55, shoulderHalf: 0.72, deckY: 0.68, deckHalf: 0.61 },
    { x: -1.93, bottomY: 0.23, lowerHalf: 0.76, shoulderY: 0.61, shoulderHalf: 0.84, deckY: 0.80, deckHalf: 0.74 },
    { x: -1.30, bottomY: 0.20, lowerHalf: 0.82, shoulderY: 0.64, shoulderHalf: 0.885, deckY: 0.86, deckHalf: 0.76 },
    { x: -0.66, bottomY: 0.20, lowerHalf: 0.79, shoulderY: 0.65, shoulderHalf: 0.86, deckY: 0.84, deckHalf: 0.74 },
    { x: 0.57, bottomY: 0.20, lowerHalf: 0.80, shoulderY: 0.65, shoulderHalf: 0.87, deckY: 0.86, deckHalf: 0.75 },
    { x: 1.30, bottomY: 0.20, lowerHalf: 0.82, shoulderY: 0.63, shoulderHalf: 0.885, deckY: 0.86, deckHalf: 0.75 },
    { x: 1.92, bottomY: 0.23, lowerHalf: 0.75, shoulderY: 0.59, shoulderHalf: 0.82, deckY: 0.79, deckHalf: 0.68 },
    { x: 2.175, bottomY: 0.30, lowerHalf: 0.57, shoulderY: 0.54, shoulderHalf: 0.69, deckY: 0.66, deckHalf: 0.55 },
  ];

  const ring = (s: BodyStation): [number, number, number][] => [
    [s.x, s.bottomY, -s.lowerHalf],
    [s.x, s.shoulderY, -s.shoulderHalf],
    [s.x, s.deckY, -s.deckHalf],
    [s.x, s.deckY, +s.deckHalf],
    [s.x, s.shoulderY, +s.shoulderHalf],
    [s.x, s.bottomY, +s.lowerHalf],
  ];
  const rings = stations.map(ring);
  const positions: number[] = [];
  const uvs: number[] = [];
  const geometry = new BufferGeometry();
  const vByRing = [0, 0.42, 1, 1, 0.42, 0] as const;
  const uAt = (x: number): number => (x + BODY_LENGTH / 2) / BODY_LENGTH;

  const pushTriangle = (
    a: readonly [number, number, number],
    b: readonly [number, number, number],
    c: readonly [number, number, number],
    au: number,
    av: number,
    bu: number,
    bv: number,
    cu: number,
    cv: number,
  ): void => {
    positions.push(...a, ...b, ...c);
    uvs.push(au, av, bu, bv, cu, cv);
  };

  const addQuad = (
    a: readonly [number, number, number],
    b: readonly [number, number, number],
    c: readonly [number, number, number],
    d: readonly [number, number, number],
    aUv: readonly [number, number],
    bUv: readonly [number, number],
    cUv: readonly [number, number],
    dUv: readonly [number, number],
    materialIndex: number,
  ): void => {
    const start = positions.length / 3;
    pushTriangle(a, b, c, ...aUv, ...bUv, ...cUv);
    pushTriangle(a, c, d, ...aUv, ...cUv, ...dUv);
    geometry.addGroup(start, 6, materialIndex);
  };

  for (let i = 0; i < rings.length - 1; i++) {
    const a = rings[i]!;
    const b = rings[i + 1]!;
    const u0 = uAt(stations[i]!.x);
    const u1 = uAt(stations[i + 1]!.x);
    for (let edge = 0; edge < 6; edge++) {
      const next = (edge + 1) % 6;
      const materialIndex = edge === 2 ? 1 : edge === 5 ? 2 : 0;
      addQuad(
        a[edge]!,
        b[edge]!,
        b[next]!,
        a[next]!,
        [u0, vByRing[edge]!],
        [u1, vByRing[edge]!],
        [u1, vByRing[next]!],
        [u0, vByRing[next]!],
        materialIndex,
      );
    }
  }

  // Low-poly end caps. The fan shape gives the nose and tail diagonal planes
  // around the lamps instead of one enormous rectangular end face.
  for (const stationIndex of [0, rings.length - 1]) {
    const r = rings[stationIndex]!;
    const s = stations[stationIndex]!;
    const centre: [number, number, number] = [
      s.x,
      (s.bottomY + s.deckY) * 0.5,
      0,
    ];
    for (let edge = 0; edge < 6; edge++) {
      const next = (edge + 1) % 6;
      const start = positions.length / 3;
      if (stationIndex === 0) {
        pushTriangle(centre, r[next]!, r[edge]!, 0.5, 0.5, 1, 0, 0, 0);
      } else {
        pushTriangle(centre, r[edge]!, r[next]!, 0.5, 0.5, 0, 0, 1, 0);
      }
      geometry.addGroup(start, 3, edge === 5 ? 2 : 1);
    }
  }

  geometry.setAttribute(
    "position",
    new Float32BufferAttribute(positions, 3),
  );
  geometry.setAttribute("uv", new Float32BufferAttribute(uvs, 2));
  geometry.computeVertexNormals();
  geometry.computeBoundingSphere();

  const skin = new MeshLambertMaterial({
    map: sideMap,
    side: DoubleSide,
    flatShading: true,
  });
  const mesh = new Mesh(geometry, [skin, material(BLUE_LIT), material(BLUE_DARK)]);
  mesh.name = "faceted_body_shell";
  return mesh;
}

/** One dark side-window polygon on the tapered cabin. */
function sideWindow(side: -1 | 1): Mesh {
  const z = side * 0.741;
  const g = new BufferGeometry();
  g.setAttribute(
    "position",
    new Float32BufferAttribute(
      [
        -1.03, 0.91, z,
        -0.80, 1.36, z,
        0.35, 1.36, z,
        0.70, 0.91, z,
      ],
      3,
    ),
  );
  g.setAttribute("uv", new Float32BufferAttribute([0, 0, 0, 1, 1, 1, 1, 0], 2));
  g.setIndex(side > 0 ? [0, 1, 2, 0, 2, 3] : [0, 2, 1, 0, 3, 2]);
  return new Mesh(g, glassMaterial());
}

/** Front or rear glass laid exactly over the cabin's sloped prism face. */
function endWindow(end: "front" | "rear"): Mesh {
  const front = end === "front";
  const xb = front ? 0.76 : -1.13;
  const xt = front ? 0.36 : -0.80;
  const g = new BufferGeometry();
  g.setAttribute(
    "position",
    new Float32BufferAttribute(
      [
        xb, 0.91, -0.64,
        xb, 0.91, 0.64,
        xt, 1.37, 0.52,
        xt, 1.37, -0.52,
      ],
      3,
    ),
  );
  g.setAttribute("uv", new Float32BufferAttribute([0, 0, 1, 0, 1, 1, 0, 1], 2));
  g.setIndex(front ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3]);
  return new Mesh(g, glassMaterial());
}

/** Flush four-vertex tail lamp, shaped to the shell instead of protruding. */
function rearLamp(
  side: -1 | 1,
  brakeMaterial: MeshLambertMaterial,
): Mesh {
  const outer = side * 0.61;
  const inner = side * 0.31;
  const x = -2.181;
  const g = new BufferGeometry();
  g.setAttribute(
    "position",
    new Float32BufferAttribute(
      [
        x, 0.70, outer,
        x, 0.69, inner,
        x, 0.53, inner,
        x, 0.54, outer,
      ],
      3,
    ),
  );
  g.setIndex(side > 0 ? [0, 2, 1, 0, 3, 2] : [0, 1, 2, 0, 2, 3]);
  g.computeVertexNormals();
  return new Mesh(g, brakeMaterial);
}

function canvasTexture(
  width: number,
  height: number,
  draw: (g: CanvasRenderingContext2D, w: number, h: number) => void,
): Texture {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const g = c.getContext("2d")!;
  g.clearRect(0, 0, width, height);
  draw(g, width, height);
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  t.magFilter = LinearFilter;
  t.minFilter = LinearFilter;
  t.generateMipmaps = true;
  return t;
}

let glassCache: Texture | undefined;

/**
 * Tiny authored window map: dark glass with one hard diagonal reflection.
 * It reads as a period texture polygon rather than a glossy modern shader.
 */
function glassTexture(): Texture {
  if (glassCache) return glassCache;
  glassCache = canvasTexture(64, 32, (g, w, h) => {
    g.fillStyle = "#132238";
    g.fillRect(0, 0, w, h);
    g.fillStyle = "#314963";
    g.beginPath();
    g.moveTo(0, 0);
    g.lineTo(w * 0.72, 0);
    g.lineTo(w * 0.34, h);
    g.lineTo(0, h);
    g.closePath();
    g.fill();
    g.fillStyle = "rgba(135,170,188,0.34)";
    g.beginPath();
    g.moveTo(w * 0.61, 0);
    g.lineTo(w * 0.78, 0);
    g.lineTo(w * 0.43, h);
    g.lineTo(w * 0.34, h);
    g.closePath();
    g.fill();
  });
  return glassCache;
}

function glassMaterial(): MeshLambertMaterial {
  return new MeshLambertMaterial({
    map: glassTexture(),
    side: DoubleSide,
    flatShading: true,
  });
}

let liveryCache:
  | { side: Texture; rear: Texture; roof: Texture; wing: Texture }
  | undefined;

function livery(): { side: Texture; rear: Texture; roof: Texture; wing: Texture } {
  if (liveryCache) return liveryCache;

  const side = canvasTexture(256, 128, (g, w, h) => {
    // The texture is intentionally small and graphic, like an authored period
    // body map, but it is filtered at display resolution rather than enlarged
    // into screen pixels.
    g.fillStyle = "#123d9a";
    g.fillRect(0, 0, w, h);

    // Large painted tonal facets make the material participate in the body
    // planes without trying to fake modern reflections.
    g.fillStyle = "#1d55c5";
    g.beginPath();
    g.moveTo(0, 0);
    g.lineTo(w, 0);
    g.lineTo(w * 0.88, h * 0.30);
    g.lineTo(w * 0.26, h * 0.43);
    g.closePath();
    g.fill();
    g.fillStyle = "#0d2f7a";
    g.beginPath();
    g.moveTo(0, h * 0.72);
    g.lineTo(w, h * 0.62);
    g.lineTo(w, h);
    g.lineTo(0, h);
    g.closePath();
    g.fill();

    g.lineCap = "round";
    g.strokeStyle = "#f2d21a";
    g.lineWidth = 13;
    g.beginPath();
    g.moveTo(8, 82);
    g.bezierCurveTo(43, 34, 88, 30, 124, 70);
    g.stroke();
    g.lineWidth = 6;
    g.beginPath();
    g.moveTo(17, 97);
    g.bezierCurveTo(54, 55, 94, 54, 126, 86);
    g.stroke();

    g.fillStyle = "#ffffff";
    g.strokeStyle = "#071a4b";
    g.lineWidth = 4;
    g.font = "900 55px Arial Black, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.strokeText("5", 150, 63);
    g.fillText("5", 150, 63);

    g.fillStyle = "#f2d21a";
    g.strokeStyle = "#071a4b";
    g.lineWidth = 3;
    g.font = "900 30px Arial Black, sans-serif";
    g.strokeText("555", 210, 64);
    g.fillText("555", 210, 64);
    g.fillStyle = "#ffffff";
    g.font = "900 9px Arial Black, sans-serif";
    g.fillText("SUBARU", 207, 88);

    // Door and quarter seams are texture detail, not extra geometry.
    g.strokeStyle = "rgba(4,18,55,0.72)";
    g.lineWidth = 2;
    for (const x of [77, 132, 181]) {
      g.beginPath();
      g.moveTo(x, 21);
      g.lineTo(x + 3, 112);
      g.stroke();
    }
    g.beginPath();
    g.moveTo(0, 105);
    g.lineTo(w, 101);
    g.stroke();

    // Painted wheel-well shadow visually cuts arches into the continuous shell.
    g.fillStyle = "#111824";
    for (const x of [54, 201]) {
      g.beginPath();
      g.ellipse(x, h + 1, 27, 31, 0, Math.PI, Math.PI * 2);
      g.fill();
    }

    // Dirt is a separate runtime layer. Keeping it out of the base texture
    // lets a clean start actually look clean and makes replay seeks repeatable.
  });

  const rear = canvasTexture(128, 64, (g, w, h) => {
    g.strokeStyle = "#f2d21a";
    g.lineWidth = 4;
    g.beginPath();
    g.ellipse(w / 2, h / 2, 43, 22, 0, 0, Math.PI * 2);
    g.stroke();
    g.fillStyle = "#f2d21a";
    const stars = [
      [44, 31, 7],
      [61, 22, 5],
      [68, 38, 4],
      [80, 25, 4],
      [88, 36, 3],
      [55, 43, 3],
    ];
    for (const [x, y, r] of stars) {
      g.beginPath();
      g.ellipse(x!, y!, r!, Math.max(2, r! * 0.55), -0.35, 0, Math.PI * 2);
      g.fill();
    }
  });

  const roof = canvasTexture(128, 64, (g, w, h) => {
    g.fillStyle = "#f2d21a";
    g.font = "900 42px Arial Black, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText("555", w / 2, h / 2);
  });

  const wing = canvasTexture(256, 48, (g, w, h) => {
    g.fillStyle = "#ffffff";
    g.font = "900 29px Arial Black, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText("SUBARU", w / 2, h / 2);
  });

  liveryCache = { side, rear, roof, wing };
  return liveryCache;
}

function decal(map: Texture, width: number, height: number): Mesh {
  return new Mesh(
    new PlaneGeometry(width, height),
    new MeshBasicMaterial({
      map,
      transparent: true,
      alphaTest: 0.12,
      depthWrite: false,
      side: DoubleSide,
    }),
  );
}

let grimeMaskCache: Texture | undefined;

/** Broad transparent shapes, tinted at runtime to the recorded surface. */
function grimeMask(): Texture {
  if (grimeMaskCache) return grimeMaskCache;
  grimeMaskCache = canvasTexture(128, 32, (g, _w, h) => {
    g.fillStyle = "rgba(255,255,255,0.92)";
    const patches = [
      [0, 21, 23, 6],
      [17, 16, 49, 12],
      [43, 23, 75, 5],
      [69, 13, 98, 14],
      [92, 20, 128, 8],
    ] as const;
    for (const [x0, y0, x1, rise] of patches) {
      g.beginPath();
      g.moveTo(x0, h);
      g.lineTo(x0, y0);
      g.lineTo(x1, h - rise);
      g.lineTo(x1, h);
      g.closePath();
      g.fill();
    }
    g.fillStyle = "rgba(255,255,255,0.48)";
    for (const [x, y, r] of [
      [16, 14, 3],
      [55, 12, 2],
      [82, 9, 3],
      [111, 15, 2],
    ] as const) {
      g.beginPath();
      g.arc(x, y, r, 0, Math.PI * 2);
      g.fill();
    }
  });
  return grimeMaskCache;
}

function grimePanel(
  width: number,
  height: number,
): { mesh: Mesh; material: MeshBasicMaterial } {
  const grimeMaterial = new MeshBasicMaterial({
    map: grimeMask(),
    color: 0x806648,
    transparent: true,
    opacity: 0,
    depthWrite: false,
    side: DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: -2,
  });
  return {
    mesh: new Mesh(new PlaneGeometry(width, height), grimeMaterial),
    material: grimeMaterial,
  };
}

/**
 * One wheel: steering group -> spin group -> faceted tyre and gold rim.
 */
function wheel(): { steer: Group; spin: Group } {
  const steer = new Group();
  const spin = new Group();

  const tyre = new Mesh(
    new CylinderGeometry(WHEEL_R, WHEEL_R, WHEEL_W, 10),
    material(TYRE),
  );
  tyre.rotation.x = Math.PI / 2;
  const rim = new Mesh(
    new CylinderGeometry(WHEEL_R * 0.62, WHEEL_R * 0.62, WHEEL_W * 1.07, 8),
    material(RIM),
  );
  rim.rotation.x = Math.PI / 2;
  const hub = new Mesh(
    new CylinderGeometry(WHEEL_R * 0.19, WHEEL_R * 0.19, WHEEL_W * 1.12, 8),
    material(TRIM),
  );
  hub.rotation.x = Math.PI / 2;

  spin.add(tyre, rim, hub);
  steer.add(spin);
  steer.name = "wheel";
  return { steer, spin };
}

export interface CarRig {
  root: Group;
  /** Chassis motion about the real CG; wheels remain on the recorded road plane. */
  body: Group;
  /** Order FL, FR, RL, RR — the wheel order used everywhere in the contracts. */
  wheels: Group[];
  spin: Group[];
  compression: number[];
  /** Recorded brake input controls these emissive materials directly. */
  brakeLights: MeshLambertMaterial[];
  /** Viewer-derived accumulated dirt, reset deterministically on seek. */
  grimeMaterials: MeshBasicMaterial[];
  /** Hidden unless a matching replay impact/crash event has occurred. */
  damageParts: Object3D[];
}

export function buildCar(): CarRig {
  const root = new Group();
  root.name = "car";
  const body = new Group();
  body.name = "chassis_motion";
  body.position.y = CG_HEIGHT;
  const bodyParts = new Group();
  bodyParts.name = "chassis_geometry";
  bodyParts.position.y = -CG_HEIGHT;
  body.add(bodyParts);
  root.add(body);

  const hw = BODY_WIDTH / 2;
  const art = livery();
  const brakeMaterial = new MeshLambertMaterial({
    color: LAMP_RED,
    emissive: 0xff1018,
    emissiveIntensity: 0.12,
    flatShading: true,
    side: DoubleSide,
  });
  const grimeMaterials: MeshBasicMaterial[] = [];
  const damageParts: Object3D[] = [];

  // One tapered, textured shell replaces the old stack of rectangular body
  // sections and box flares.
  bodyParts.add(bodyShell(art.side));
  bodyParts.add(box(0.54, 0.075, 0.59, BLUE_DARK, 1.20, 0.90, 0)); // bonnet scoop

  // Cabin and glazing.
  bodyParts.add(
    taperedPrism(1.95, 0.55, 1.48, 1.18, 0.38, 0.30, BLUE, -0.18, 0.85),
  );
  bodyParts.add(sideWindow(-1), sideWindow(1));
  for (const side of [-1, 1] as const) {
    bodyParts.add(
      box(0.068, 0.40, 0.032, BLUE_DARK, -0.24, 1.14, side * 0.753),
    );
  }
  bodyParts.add(endWindow("front"), endWindow("rear"));
  bodyParts.add(box(1.24, 0.045, 1.21, BLUE_LIT, -0.24, 1.415, 0));

  // Mirrors, mud flaps and bumper grime.
  bodyParts.add(box(0.20, 0.10, 0.14, BLUE_LIT, 0.39, 1.08, hw + 0.025));
  bodyParts.add(box(0.20, 0.10, 0.14, BLUE_LIT, 0.39, 1.08, -hw - 0.025));
  for (const z of [-0.76, 0.76]) {
    bodyParts.add(box(0.08, 0.36, 0.29, TRIM, -1.60, 0.31, z));
  }
  bodyParts.add(box(0.13, 0.11, 1.28, MUD, -2.11, 0.31, 0));

  // Period high wing, kept below the roof/antenna envelope.
  bodyParts.add(box(0.10, 0.34, 0.07, BLUE, -1.78, 1.10, +0.69));
  bodyParts.add(box(0.10, 0.34, 0.07, BLUE, -1.78, 1.10, -0.69));
  bodyParts.add(box(0.38, 0.065, 1.72, BLUE, -1.86, 1.31, 0));
  bodyParts.add(box(0.29, 0.045, 1.62, BLUE_LIT, -1.94, 1.38, 0));

  // Lamps, plate, bumper cut and exhaust.
  for (const z of [-0.54, 0.54]) {
    bodyParts.add(box(0.045, 0.15, 0.39, LAMP, +2.16, 0.61, z));
  }
  bodyParts.add(
    rearLamp(-1, brakeMaterial),
    rearLamp(1, brakeMaterial),
  );
  bodyParts.add(box(0.045, 0.15, 0.52, PLATE, -2.18, 0.42, 0));
  bodyParts.add(box(0.05, 0.10, 0.58, TRIM, -2.17, 0.27, 0));
  const exhaust = new Mesh(
    new CylinderGeometry(0.045, 0.058, 0.22, 8),
    material(TRIM),
  );
  exhaust.rotation.z = Math.PI / 2;
  exhaust.position.set(-2.17, 0.28, 0.51);
  bodyParts.add(exhaust);

  // Progressive dirt is a transparent, coarse polygon mask over the sill and
  // tail. Runtime opacity comes from replay distance on loose surfaces.
  for (const side of [-1, 1] as const) {
    const grime = grimePanel(4.05, 0.54);
    grime.mesh.position.set(-0.02, 0.47, side * 0.891);
    if (side < 0) grime.mesh.rotation.y = Math.PI;
    grime.mesh.name = `progressive_grime_side_${side}`;
    bodyParts.add(grime.mesh);
    grimeMaterials.push(grime.material);
  }
  const rearGrime = grimePanel(1.38, 0.45);
  rearGrime.mesh.position.set(-2.187, 0.48, 0);
  rearGrime.mesh.rotation.y = -Math.PI / 2;
  rearGrime.mesh.name = "progressive_grime_rear";
  bodyParts.add(rearGrime.mesh);
  grimeMaterials.push(rearGrime.material);

  // Optional event-driven damage. These low-poly loose panels stay hidden for
  // clean replays and never alter the authoritative collision silhouette.
  const damagedBumperLeft = box(
    0.24,
    0.11,
    0.62,
    BLUE_DARK,
    2.16,
    0.34,
    -0.46,
  );
  damagedBumperLeft.name = "damage_front_left";
  damagedBumperLeft.visible = false;
  damagedBumperLeft.rotation.z = 0.11;
  bodyParts.add(damagedBumperLeft);
  damageParts.push(damagedBumperLeft);

  const damagedBumperRight = box(
    0.24,
    0.11,
    0.62,
    BLUE_DARK,
    2.16,
    0.31,
    0.46,
  );
  damagedBumperRight.name = "damage_front_right";
  damagedBumperRight.visible = false;
  damagedBumperRight.rotation.z = -0.12;
  bodyParts.add(damagedBumperRight);
  damageParts.push(damagedBumperRight);

  // Rear, roof and wing marks remain separate cards; the side livery is now
  // genuinely wrapped into the body-shell material above.
  const rearMark = decal(art.rear, 0.88, 0.43);
  rearMark.position.set(-2.183, 0.69, 0);
  rearMark.rotation.y = -Math.PI / 2;
  bodyParts.add(rearMark);

  const roofMark = decal(art.roof, 0.86, 0.44);
  roofMark.position.set(-0.24, 1.44, 0);
  roofMark.rotation.x = -Math.PI / 2;
  bodyParts.add(roofMark);

  const wingMark = decal(art.wing, 1.42, 0.25);
  wingMark.position.set(-2.057, 1.315, 0);
  wingMark.rotation.y = -Math.PI / 2;
  bodyParts.add(wingMark);

  // Period roof antenna.
  const antenna = new Mesh(
    new CylinderGeometry(0.010, 0.016, 0.38, 5),
    material(TRIM),
  );
  antenna.position.set(-0.66, 1.60, 0);
  antenna.rotation.z = -0.32;
  bodyParts.add(antenna);

  const wheels: Group[] = [];
  const spin: Group[] = [];
  const placement: [number, number][] = [
    [+WHEELBASE / 2, -TRACK / 2],
    [+WHEELBASE / 2, +TRACK / 2],
    [-WHEELBASE / 2, -TRACK / 2],
    [-WHEELBASE / 2, +TRACK / 2],
  ];
  for (const [x, z] of placement) {
    const w = wheel();
    w.steer.position.set(x, WHEEL_R, z);
    root.add(w.steer);
    wheels.push(w.steer);
    spin.push(w.spin);
  }

  return {
    root,
    body,
    wheels,
    spin,
    compression: [0, 0, 0, 0],
    brakeLights: [brakeMaterial],
    grimeMaterials,
    damageParts,
  };
}

const STEER_VISUAL_MAX = 0.52;

export function driveWheels(rig: CarRig, steer: number, rollRad: number): void {
  const angle = steer * STEER_VISUAL_MAX;
  rig.wheels[0]!.rotation.y = angle;
  rig.wheels[1]!.rotation.y = angle;
  for (const s of rig.spin) s.rotation.z = -rollRad;
}

function clamp(value: number, lo: number, hi: number): number {
  return Math.min(Math.max(value, lo), hi);
}

/**
 * Reconstruct visual spring travel from recorded wheel telemetry.
 *
 * Newer replay producers may provide `comp` directly. Existing replays carry
 * the authoritative per-corner vertical load instead, so the fallback maps
 * load relative to that corner's static load into a conservative gravel-rally
 * bump/droop range. This remains presentation-only: it never feeds simulation,
 * sensing, reward or replay state.
 */
export function wheelVisualCompression(
  wheel: WheelFrame | undefined,
  index: number,
  airborne: boolean,
): number {
  if (airborne || wheel?.con === false) return -MAX_DROOP;
  if (Number.isFinite(wheel?.comp)) {
    return clamp(wheel!.comp!, -MAX_DROOP, MAX_BUMP);
  }
  const fz = wheel?.fz;
  if (!Number.isFinite(fz)) return 0;
  if (fz! < 50) return -MAX_DROOP;
  return clamp(
    (fz! / STATIC_FZ[index]! - 1) * LOAD_TO_TRAVEL,
    -MAX_DROOP,
    MAX_BUMP,
  );
}

/**
 * Keep the wheel centres on the recorded road plane and move the sprung body
 * around the CarSpec CG. The four spring values produce heave, pitch and roll:
 *
 * - front compression lowers the nose under braking;
 * - outside compression rolls the body into lateral load transfer;
 * - zero load extends the suspension in flight.
 */
export function updateSuspension(
  rig: CarRig,
  wheelFrames: WheelFrame[],
  airborne: boolean,
  dt: number,
): void {
  const follow = 1 - Math.exp(-14 * Math.min(Math.max(dt, 0), 0.1));
  for (let i = 0; i < 4; i++) {
    const target = wheelVisualCompression(wheelFrames[i], i, airborne);
    const current = rig.compression[i]!;
    rig.compression[i] = current + (target - current) * follow;
  }

  const [fl, fr, rl, rr] = rig.compression as [
    number,
    number,
    number,
    number,
  ];
  const front = (fl + fr) * 0.5;
  const rear = (rl + rr) * 0.5;
  const left = (fl + rl) * 0.5;
  const right = (fr + rr) * 0.5;
  const heave = (fl + fr + rl + rr) * 0.25;

  rig.body.position.y = CG_HEIGHT - heave;
  // Local +z rotation lifts the nose; local +x rotation lowers the right side.
  rig.body.rotation.z = Math.atan2(rear - front, WHEELBASE);
  rig.body.rotation.x = Math.atan2(right - left, TRACK);
}

export function resetSuspension(rig: CarRig): void {
  rig.compression.fill(0);
  rig.body.position.y = CG_HEIGHT;
  rig.body.rotation.set(0, 0, 0);
}

export function carMeshes(rig: CarRig): Object3D[] {
  const out: Object3D[] = [];
  rig.root.traverse((o) => out.push(o));
  return out;
}
