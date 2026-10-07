/**
 * Sky and distant terrain.
 *
 * A flat clear colour behind fog is what a renderer does before anyone has
 * looked at it. The era's stages had a graded sky, blocky clouds, and — most
 * importantly — **layered hills sitting on the fog line**, which is what gives a
 * stage the sense of being somewhere rather than on a table.
 *
 * All of it is fake and none of it is in the stage document. That is fine and
 * deliberate: the sim has no terrain beyond the corridor, so there is nothing
 * here to be faithful to. What matters is that it is *behind* everything and
 * never occludes road, car or obstacle.
 */

import {
  BackSide,
  CanvasTexture,
  Color,
  DoubleSide,
  Group,
  LinearFilter,
  Mesh,
  MeshBasicMaterial,
  PlaneGeometry,
  SphereGeometry,
  SRGBColorSpace,
  type Texture,
} from "three";

/** Colours of the sky gradient, horizon first. */
const HORIZON = "#c3d4d4";
const ZENITH = "#6da6cc";

/** Hill bands, near to far: colour and how high they sit. */
const HILLS: [string, number][] = [
  ["#4d6850", 0.055],
  ["#587060", 0.085],
  ["#758b91", 0.12],
];

function prng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/**
 * The sky dome texture: vertical gradient plus broad angular cumulus.
 *
 * A handful of clean polygon lobes echo the environment's geometry without
 * turning the sky into a grid of tiny squares.
 */
function skyCanvas(): HTMLCanvasElement {
  const W = 512;
  const H = 256;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d")!;

  const grad = g.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, ZENITH);
  grad.addColorStop(0.62, "#9fc3d7");
  grad.addColorStop(1, HORIZON);
  g.fillStyle = grad;
  g.fillRect(0, 0, W, H);

  const r = prng(0xc10d);
  // Kept in the upper two thirds: clouds sitting on the horizon would fight the
  // hills and the fog band for the same pixels.
  for (let i = 0; i < 10; i++) {
    const cx = r() * W;
    const cy = H * (0.08 + r() * 0.45);
    const scale = 0.4 + r() * 0.75;
    for (let j = 0; j < 6; j++) {
      const dx = (r() - 0.5) * 74 * scale;
      const dy = (r() - 0.5) * 16 * scale;
      const w = (18 + r() * 30) * scale;
      const h = (8 + r() * 14) * scale;
      const bright = r();
      g.fillStyle =
        bright > 0.75 ? "#f4f6f1" : bright > 0.35 ? "#e5ecee" : "#c4d2d8";
      g.beginPath();
      g.moveTo(cx + dx - w * 0.5, cy + dy + h * 0.28);
      g.lineTo(cx + dx - w * 0.28, cy + dy - h * 0.45);
      g.lineTo(cx + dx + w * 0.12, cy + dy - h * 0.55);
      g.lineTo(cx + dx + w * 0.52, cy + dy - h * 0.02);
      g.lineTo(cx + dx + w * 0.34, cy + dy + h * 0.48);
      g.closePath();
      g.fill();
    }
  }
  return c;
}

/**
 * One band of hills, as a wrapping silhouette strip with alpha above it.
 *
 * Built from a coarse sequence of peaks. The profile is intentionally angular:
 * a stylised low-poly ridgeline, not a smooth procedural sine wave.
 */
function hillCanvas(colour: string, roughness: number, seed: number): HTMLCanvasElement {
  const W = 1024;
  const H = 128;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d")!;
  g.clearRect(0, 0, W, H);
  g.fillStyle = colour;

  const r = prng(seed);
  const step = 32;
  g.beginPath();
  g.moveTo(0, H);
  for (let x = 0; x <= W; x += step) {
    const broad = Math.sin((x / W) * Math.PI * 4 + seed * 0.001) * 0.12;
    const h = 0.36 + broad + (r() - 0.5) * 0.24 * roughness;
    g.lineTo(x, Math.max(4, H * (1 - h)));
  }
  g.lineTo(W, H);
  g.closePath();
  g.fill();
  return c;
}

function tex(c: HTMLCanvasElement): Texture {
  const t = new CanvasTexture(c);
  t.colorSpace = SRGBColorSpace;
  // The sky is a smooth gradient across a huge area, so linear sampling avoids
  // visible rings while the clouds and ridgelines keep their angular edges.
  t.magFilter = LinearFilter;
  t.minFilter = LinearFilter;
  t.generateMipmaps = false;
  return t;
}

/**
 * Build the sky dome and hill bands.
 *
 * Returned as a `Group` the caller parents to the camera, so it travels with
 * the viewer and can never be reached. Everything in it has depth writing off
 * and a low render order — it is a backdrop, not geometry.
 */
export function buildSky(): Group {
  const group = new Group();
  group.name = "sky";

  const dome = new Mesh(
    new SphereGeometry(900, 24, 16),
    new MeshBasicMaterial({
      map: tex(skyCanvas()),
      side: BackSide,
      depthWrite: false,
      fog: false,
    }),
  );
  dome.renderOrder = -10;
  group.add(dome);

  HILLS.forEach(([colour, height], i) => {
    const radius = 780 - i * 90;
    const c = hillCanvas(colour, 1 + i * 0.6, 0x51e + i * 977);
    const t = tex(c);
    // A cylinder would need caps and a seam; a wide plane per band, curved by
    // being placed on a ring, costs more geometry than it earns. One belt of
    // quads around the camera is enough because the bands never rotate.
    const belt = new Mesh(
      new PlaneGeometry(radius * 2 * Math.PI, radius * height * 2, 64, 1),
      new MeshBasicMaterial({
        map: t,
        transparent: true,
        depthWrite: false,
        side: DoubleSide,
        fog: false,
        color: new Color(0xffffff),
      }),
    );
    // Wrap the belt into a ring by bending its vertices.
    const pos = belt.geometry.attributes.position!;
    for (let v = 0; v < pos.count; v++) {
      const x = pos.getX(v);
      const angle = x / radius;
      pos.setX(v, Math.sin(angle) * radius);
      pos.setZ(v, -Math.cos(angle) * radius);
    }
    pos.needsUpdate = true;
    belt.geometry.computeBoundingSphere();
    belt.position.y = radius * height * 0.32;
    belt.renderOrder = -9 + i;
    group.add(belt);
  });

  return group;
}

/** The horizon colour, so fog and clear colour cannot drift from the sky. */
export const HORIZON_COLOUR = HORIZON;
