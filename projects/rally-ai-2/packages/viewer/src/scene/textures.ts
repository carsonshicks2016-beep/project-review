/**
 * Procedural textures, drawn to canvases at load.
 *
 * Nothing is fetched. The whole viewer stays self-contained — a replay plus this
 * file is enough to render a stage — and there is no asset pipeline to keep in
 * sync with the sim. It also means surfaces are generated from the same names
 * the contract uses, so a new surface kind cannot render as "missing texture".
 *
 * Everything is authored as broad, hard-edged colour patches rather than tiny
 * flecks. The low-poly geometry should remain the star: these maps add material
 * variation without turning the road and grass into a pixel-noise overlay.
 */

import {
  CanvasTexture,
  LinearFilter,
  LinearMipmapLinearFilter,
  RepeatWrapping,
  SRGBColorSpace,
  type Texture,
} from "three";

import type { SurfaceKind } from "../replay";

/**
 * Deterministic noise.
 *
 * A seeded PRNG rather than `Math.random`, so a texture looks the same every
 * load. Nothing depends on it, but a road that resurfaces itself on refresh
 * makes "did that change?" impossible to answer while iterating.
 */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

function canvas(size: number): { c: HTMLCanvasElement; g: CanvasRenderingContext2D } {
  const c = document.createElement("canvas");
  c.width = c.height = size;
  const g = c.getContext("2d")!;
  return { c, g };
}

function finish(c: HTMLCanvasElement, repeat: number): Texture {
  const t = new CanvasTexture(c);
  t.wrapS = t.wrapT = RepeatWrapping;
  // Surface maps are deliberately modest, but their texels are not the style.
  // Filtering keeps the broad painted facets calm as they recede into fog.
  t.magFilter = LinearFilter;
  t.minFilter = LinearMipmapLinearFilter;
  t.generateMipmaps = true;
  t.colorSpace = SRGBColorSpace;
  t.repeat.set(repeat, repeat);
  return t;
}

/** Broad overlapping polygon patches: material variation without pixel noise. */
function paintedFacets(
  size: number,
  base: string,
  patches: string[],
  seed: number,
): HTMLCanvasElement {
  const { c, g } = canvas(size);
  g.fillStyle = base;
  g.fillRect(0, 0, size, size);
  const r = rng(seed);
  for (const colour of patches) {
    g.fillStyle = colour;
    for (let i = 0; i < 18; i++) {
      const x = r() * size;
      const y = r() * size;
      const radius = size * (0.055 + r() * 0.13);
      const sides = 3 + Math.floor(r() * 3);
      g.beginPath();
      for (let p = 0; p < sides; p++) {
        const a = (p / sides) * Math.PI * 2 + r() * 0.5;
        const rr = radius * (0.6 + r() * 0.55);
        const px = x + Math.cos(a) * rr;
        const py = y + Math.sin(a) * rr;
        if (p === 0) g.moveTo(px, py);
        else g.lineTo(px, py);
      }
      g.closePath();
      g.fill();
    }
  }
  return c;
}

/**
 * Road surfaces.
 *
 * Each carries a faint pair of darker wheel tracks down its length. That single
 * detail does more than the aggregate noise does: it tells you where the racing
 * line is, and it makes the road read as *travelled* rather than as a ribbon.
 */
function roadCanvas(kind: SurfaceKind): HTMLCanvasElement {
  const S = 128;
  const spec: Record<SurfaceKind, { base: string; patches: string[]; track: string }> = {
    gravel: {
      base: "#a39a89",
      patches: ["#918879", "#b5ad9e", "#7f786d", "#c4bcad"],
      track: "rgba(78,74,68,0.22)",
    },
    tarmac: {
      base: "#4a4a50",
      patches: ["#414147", "#55555b", "#36363b"],
      track: "rgba(30,30,34,0.28)",
    },
    snow: {
      base: "#dee5ec",
      patches: ["#ccd5de", "#eef3f8", "#bdc8d3"],
      track: "rgba(150,164,180,0.34)",
    },
    mud: {
      base: "#63503a",
      patches: ["#54432f", "#756045", "#473824"],
      track: "rgba(40,31,20,0.34)",
    },
  };
  const s = spec[kind];
  const c = paintedFacets(S, s.base, s.patches, 0x9e3779 + kind.length * 31);
  const g = c.getContext("2d")!;

  // Wheel tracks. u runs across the corridor, so these land at fixed fractions
  // of the road width whatever the road is doing.
  g.fillStyle = s.track;
  g.fillRect(Math.round(S * 0.28), 0, Math.round(S * 0.09), S);
  g.fillRect(Math.round(S * 0.63), 0, Math.round(S * 0.09), S);
  // A few broad atlas seams are intentional. They suggest small authored
  // texture pages and stay legible without exposing individual screen pixels.
  g.fillStyle =
    kind === "snow"
      ? "rgba(126,145,162,0.15)"
      : "rgba(26,28,27,0.13)";
  g.fillRect(0, Math.round(S * 0.32), S, 3);
  g.fillRect(0, Math.round(S * 0.78), S, 2);
  return c;
}

/** Grass verge: broad low-poly colour masses, clearly distinct from the road. */
function grassCanvas(): HTMLCanvasElement {
  const S = 128;
  const c = paintedFacets(
    S,
    "#4f7136",
    ["#3c5d2c", "#648542", "#36532b", "#738c4c"],
    0x5eed,
  );
  const g = c.getContext("2d")!;
  // Coarse authored seams and flattened grass bands read like a small texture
  // atlas without turning the verge into a noisy pixel field.
  g.fillStyle = "rgba(39,72,33,0.18)";
  g.fillRect(0, 38, S, 4);
  g.fillRect(0, 91, S, 3);
  g.fillStyle = "rgba(124,151,79,0.16)";
  g.beginPath();
  g.moveTo(0, 67);
  g.lineTo(S, 54);
  g.lineTo(S, 69);
  g.lineTo(0, 79);
  g.closePath();
  g.fill();
  return c;
}

function ditchCanvas(): HTMLCanvasElement {
  const S = 96;
  const c = paintedFacets(
    S,
    "#665d45",
    ["#514b38", "#7a704f", "#46533a", "#8b8061"],
    0xd17c4,
  );
  const g = c.getContext("2d")!;
  g.fillStyle = "rgba(37,45,31,0.34)";
  g.fillRect(0, 0, 18, S);
  g.fillRect(76, 0, 20, S);
  g.strokeStyle = "rgba(142,126,89,0.32)";
  g.lineWidth = 4;
  g.beginPath();
  g.moveTo(7, 0);
  g.lineTo(31, S);
  g.moveTo(67, 0);
  g.lineTo(89, S);
  g.stroke();
  return c;
}

function rockCanvas(): HTMLCanvasElement {
  const S = 64;
  const c = paintedFacets(
    S,
    "#77776d",
    ["#5e625d", "#8e8b7d", "#4f5553", "#a29d8a"],
    0x70c5,
  );
  const g = c.getContext("2d")!;
  g.strokeStyle = "rgba(45,48,47,0.38)";
  g.lineWidth = 3;
  g.beginPath();
  g.moveTo(0, 18);
  g.lineTo(64, 35);
  g.moveTo(14, 0);
  g.lineTo(42, 64);
  g.stroke();
  return c;
}

function treeAtlas(
  kind: "bark" | "foliage",
): HTMLCanvasElement {
  const W = 96;
  const H = 64;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d")!;
  const bases =
    kind === "bark"
      ? ["#4d3825", "#57412c", "#46372b"]
      : ["#294f32", "#38623a", "#486f3f"];
  const accents =
    kind === "bark"
      ? ["#725239", "#37291f", "#846146"]
      : ["#3e6a40", "#24462e", "#67834b"];
  for (let variant = 0; variant < 3; variant++) {
    const x = variant * 32;
    g.fillStyle = bases[variant]!;
    g.fillRect(x, 0, 32, H);
    g.fillStyle = accents[variant]!;
    if (kind === "bark") {
      for (let stripe = 0; stripe < 5; stripe++) {
        const sx = x + 3 + stripe * 7 + (variant + stripe) % 3;
        g.beginPath();
        g.moveTo(sx, 0);
        g.lineTo(sx + ((stripe % 2) * 4 - 2), H);
        g.lineTo(sx + 3, H);
        g.lineTo(sx + 2, 0);
        g.closePath();
        g.fill();
      }
    } else {
      const r = rng(0xf011a9 + variant * 83);
      for (let patch = 0; patch < 15; patch++) {
        const px = x + r() * 32;
        const py = r() * H;
        const radius = 3 + r() * 7;
        g.beginPath();
        g.moveTo(px, py - radius);
        g.lineTo(px + radius, py);
        g.lineTo(px, py + radius * 0.65);
        g.lineTo(px - radius * 0.8, py);
        g.closePath();
        g.fill();
      }
    }
    // The atlas boundary is deliberately visible as a coarse UV seam.
    g.fillStyle = "rgba(18,24,19,0.26)";
    g.fillRect(x + 30, 0, 2, H);
  }
  return c;
}

function farTreeCanvas(variant: number): HTMLCanvasElement {
  const c = document.createElement("canvas");
  c.width = 64;
  c.height = 128;
  const g = c.getContext("2d")!;
  const foliage = ["#294a31", "#31543a", "#3d6040"][variant]!;
  const light = ["#3c6340", "#477048", "#54794b"][variant]!;
  g.fillStyle = "#493522";
  g.fillRect(29, 70, 7, 58);
  g.fillStyle = foliage;
  for (const [y, width] of [
    [8, 16],
    [28, 24],
    [50, 31],
    [74, 28],
  ] as const) {
    g.beginPath();
    g.moveTo(32, y);
    g.lineTo(32 + width, y + 45);
    g.lineTo(32 - width, y + 45);
    g.closePath();
    g.fill();
  }
  g.fillStyle = light;
  g.beginPath();
  g.moveTo(32, 17);
  g.lineTo(49, 58);
  g.lineTo(32, 50);
  g.closePath();
  g.fill();
  return c;
}

function atlasSlices(source: Texture): [Texture, Texture, Texture] {
  return [0, 1, 2].map((variant) => {
    const texture = source.clone();
    texture.repeat.set(1 / 3, 1);
    texture.offset.set(variant / 3, 0);
    texture.needsUpdate = true;
    return texture;
  }) as [Texture, Texture, Texture];
}

/**
 * Roadside hoarding.
 *
 * Sponsor boards lining the verge are one of the most recognisable things about
 * the era's rally stages — they mark the corner, they give the eye something to
 * measure speed against, and they are why those games read as *events* rather
 * than as empty countryside.
 *
 * This personal viewer skin intentionally uses the event/manufacturer names
 * authorised by the user. Large colour blocks and wordmarks read clearly at
 * speed; tiny logo sticker sheets do not.
 */
function bannerCanvas(): HTMLCanvasElement {
  const W = 256;
  const H = 64;
  const c = document.createElement("canvas");
  c.width = W;
  c.height = H;
  const g = c.getContext("2d")!;

  const panels: [string, string, string][] = [
    ["#123d9a", "#f2d21a", "SUBARU"],
    ["#f2d21a", "#123d9a", "555"],
    ["#ffffff", "#1768b0", "MICHELIN"],
    ["#d8242f", "#ffffff", "MOTUL"],
    ["#f4d62b", "#d3222a", "PIRELLI"],
  ];
  const pw = W / panels.length;
  panels.forEach(([bg, fg, text], i) => {
    g.fillStyle = bg;
    g.fillRect(i * pw, 0, pw, H);
    g.fillStyle = fg;
    g.font = "bold 15px monospace";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(text, i * pw + pw / 2, H / 2);
  });
  // Dark lip top and bottom so the board reads as a board, not a light leak.
  g.fillStyle = "rgba(0,0,0,0.45)";
  g.fillRect(0, 0, W, 3);
  g.fillRect(0, H - 4, W, 4);
  return c;
}

/** Soft round blob, used as the car's contact shadow. */
function shadowCanvas(): HTMLCanvasElement {
  const S = 64;
  const { c, g } = canvas(S);
  const grad = g.createRadialGradient(S / 2, S / 2, 0, S / 2, S / 2, S / 2);
  grad.addColorStop(0, "rgba(0,0,0,0.55)");
  grad.addColorStop(0.55, "rgba(0,0,0,0.30)");
  grad.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, S, S);
  return c;
}

/**
 * Every texture, built once.
 *
 * Lazy and cached: building them costs a few milliseconds, and a stage with
 * three surface segments should not build three copies of the same gravel.
 */
let cache: TextureSet | null = null;

export interface TextureSet {
  road: Record<SurfaceKind, Texture>;
  grass: Texture;
  ditch: Texture;
  rock: Texture;
  banner: Texture;
  shadow: Texture;
  bark: [Texture, Texture, Texture];
  foliage: [Texture, Texture, Texture];
  farTree: [Texture, Texture, Texture];
}

export function textures(): TextureSet {
  if (cache) return cache;

  const road = {} as Record<SurfaceKind, Texture>;
  for (const kind of ["gravel", "tarmac", "snow", "mud"] as SurfaceKind[]) {
    road[kind] = finish(roadCanvas(kind), 1);
  }

  const sprite = (c: HTMLCanvasElement): Texture => {
    const t = finish(c, 1);
    t.repeat.set(1, 1);
    return t;
  };

  const barkAtlas = finish(treeAtlas("bark"), 1);
  const foliageAtlas = finish(treeAtlas("foliage"), 1);
  cache = {
    road,
    grass: finish(grassCanvas(), 1),
    ditch: finish(ditchCanvas(), 1),
    rock: finish(rockCanvas(), 1),
    banner: sprite(bannerCanvas()),
    shadow: sprite(shadowCanvas()),
    bark: atlasSlices(barkAtlas),
    foliage: atlasSlices(foliageAtlas),
    farTree: [
      sprite(farTreeCanvas(0)),
      sprite(farTreeCanvas(1)),
      sprite(farTreeCanvas(2)),
    ],
  };
  return cache;
}
