import { MATERIALS, MATERIAL_LIST, indexOfMaterial } from '../materials';
import {
  BEAM_BRACE,
  BEAM_SKIN,
  FLAG_SKIN,
  type MaterialId,
  type Ship,
} from '../types';
import {
  addBeamFromMaterials,
  addNode,
  createBeams,
  createNodes,
} from '../ship/node-beam';

/** Parse a CSS hex color (#rgb / #rrggbb) into 0–255 RGB. */
function hexToRgb(hex: string): [number, number, number] {
  let h = hex.replace('#', '');
  if (h.length === 3) {
    h = h[0]! + h[0]! + h[1]! + h[1]! + h[2]! + h[2]!;
  }
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

const MATERIAL_RGB: { id: MaterialId; r: number; g: number; b: number }[] =
  MATERIAL_LIST.map((m) => {
    const [r, g, b] = hexToRgb(m.color);
    return { id: m.id, r, g, b };
  });

function rgbDist2(
  r: number,
  g: number,
  b: number,
  mr: number,
  mg: number,
  mb: number,
): number {
  const dr = r - mr;
  const dg = g - mg;
  const db = b - mb;
  return dr * dr + dg * dg + db * db;
}

/** Map pixel RGB to a material via nearest palette color, with hue-bucket fallback. */
export function materialFromColor(r: number, g: number, b: number): MaterialId {
  let best: MaterialId = 'steel';
  let bestD = Infinity;
  for (const m of MATERIAL_RGB) {
    const d = rgbDist2(r, g, b, m.r, m.g, m.b);
    if (d < bestD) {
      bestD = d;
      best = m.id;
    }
  }

  // If the nearest palette hit is still far, fall back to simple hue buckets
  if (bestD > 70 * 70) {
    return hueBucketMaterial(r, g, b);
  }
  return best;
}

function hueBucketMaterial(r: number, g: number, b: number): MaterialId {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const chroma = max - min;
  const lightness = (max + min) / 2;
  const sat = chroma < 1 ? 0 : chroma / (255 - Math.abs(2 * lightness - 255) || 1);

  // Black / near-black → engine
  if (max < 50) return 'engine';
  // Dark desaturated → keel / steel
  if (lightness < 80 && sat < 0.25) return 'keel';
  if (lightness < 110 && sat < 0.2) return 'steel';

  // Brown → wood / deck
  if (r > g && g >= b && r > 80 && g > 40 && b < 120 && r - b > 30) {
    return lightness > 120 ? 'deck' : 'wood';
  }

  // Cyan / ice
  if (b > r && g > r && b > 140 && sat > 0.1) return 'ice';

  // Blue-ish → glass
  if (b > r + 15 && b > g && sat > 0.15) return 'glass';

  // Light gray → superstructure
  if (sat < 0.18 && lightness > 160) return 'superstructure';

  // Mid gray steel / funnel
  if (sat < 0.2 && lightness < 120) return 'funnel';

  return 'steel';
}

function nodeMass(mat: MaterialId, spacing: number, densityScale: number): number {
  return MATERIALS[mat].density * spacing * spacing * densityScale;
}

/**
 * Build a ship from raw ImageData.
 * Each opaque pixel (alpha > 128) becomes a structural node; 4-connected
 * neighbors get beams; missing neighbors mark skin; long shapes get braces.
 */
export function importBlueprintFromImageData(
  img: ImageData,
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1,
): Ship {
  const { width, height, data } = img;
  const spacing = 5;

  // Count opaque pixels for capacity
  let opaque = 0;
  for (let i = 0; i < width * height; i++) {
    if (data[i * 4 + 3]! > 128) opaque++;
  }
  if (opaque === 0) {
    return {
      nodes: createNodes(1),
      beams: createBeams(1),
      name: 'Blueprint',
    };
  }

  const nodes = createNodes(opaque + 8);
  const beams = createBeams(opaque * 4 + 128);
  /** Pixel index → node index, or -1 */
  const grid = new Int32Array(width * height).fill(-1);

  const x0 = cx - ((width - 1) * spacing) * 0.5;
  // Bottom of the image sits near cy (image row 0 is top)
  const y0 = cy;
  const yTop = y0 + (height - 1) * spacing;

  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const p = (py * width + px) * 4;
      const a = data[p + 3]!;
      if (a <= 128) continue;

      const r = data[p]!;
      const g = data[p + 1]!;
      const b = data[p + 2]!;
      const mat = materialFromColor(r, g, b);

      // Image y grows downward; world y grows upward
      const worldX = x0 + px * spacing;
      const worldY = yTop - py * spacing;

      // Skin if any 4-neighbor is missing / out of bounds / transparent
      const skin = !isOpaque(data, width, height, px - 1, py)
        || !isOpaque(data, width, height, px + 1, py)
        || !isOpaque(data, width, height, px, py - 1)
        || !isOpaque(data, width, height, px, py + 1);

      const ni = addNode(
        nodes,
        worldX,
        worldY,
        indexOfMaterial(mat),
        nodeMass(mat, spacing, densityScale),
        skin ? FLAG_SKIN : 0,
        1,
      );
      grid[py * width + px] = ni;
    }
  }

  // 4-connected beams
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width; px++) {
      const i = grid[py * width + px]!;
      if (i < 0) continue;

      const right = px + 1 < width ? grid[py * width + (px + 1)]! : -1;
      const down = py + 1 < height ? grid[(py + 1) * width + px]! : -1;

      if (right >= 0) {
        const skinEdge =
          (nodes.flags[i]! & FLAG_SKIN) !== 0 &&
          (nodes.flags[right]! & FLAG_SKIN) !== 0 &&
          (!isOpaque(data, width, height, px, py - 1) ||
            !isOpaque(data, width, height, px, py + 1) ||
            !isOpaque(data, width, height, px + 1, py - 1) ||
            !isOpaque(data, width, height, px + 1, py + 1));
        addBeamFromMaterials(
          beams,
          nodes,
          i,
          right,
          densityScale,
          strengthScale,
          skinEdge ? BEAM_SKIN : 0,
        );
      }
      if (down >= 0) {
        const skinEdge =
          (nodes.flags[i]! & FLAG_SKIN) !== 0 &&
          (nodes.flags[down]! & FLAG_SKIN) !== 0 &&
          (!isOpaque(data, width, height, px - 1, py) ||
            !isOpaque(data, width, height, px + 1, py) ||
            !isOpaque(data, width, height, px - 1, py + 1) ||
            !isOpaque(data, width, height, px + 1, py + 1));
        addBeamFromMaterials(
          beams,
          nodes,
          i,
          down,
          densityScale,
          strengthScale,
          skinEdge ? BEAM_SKIN : 0,
        );
      }
    }
  }

  // Longitudinal braces when the silhouette is much wider than tall
  if (width > height * 1.6) {
    addBlueprintBraces(nodes, beams, grid, width, height, densityScale, strengthScale);
  }

  return { nodes, beams, name: 'Blueprint' };
}

function isOpaque(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  px: number,
  py: number,
): boolean {
  if (px < 0 || py < 0 || px >= width || py >= height) return false;
  return data[(py * width + px) * 4 + 3]! > 128;
}

/** Add long-axis brace beams every few columns on occupied rows. */
function addBlueprintBraces(
  nodes: Ship['nodes'],
  beams: Ship['beams'],
  grid: Int32Array,
  width: number,
  height: number,
  densityScale: number,
  strengthScale: number,
): void {
  const step = 3;
  for (let py = 0; py < height; py++) {
    for (let px = 0; px < width - step; px++) {
      const a = grid[py * width + px]!;
      const b = grid[py * width + (px + step)]!;
      if (a < 0 || b < 0) continue;
      // Only brace if the span is mostly solid
      let solid = true;
      for (let k = 1; k < step; k++) {
        if (grid[py * width + (px + k)]! < 0) {
          solid = false;
          break;
        }
      }
      if (!solid) continue;
      addBeamFromMaterials(
        beams,
        nodes,
        a,
        b,
        densityScale,
        strengthScale * 1.1,
        BEAM_BRACE,
      );
    }
  }
}

/** Load a PNG/image File via canvas and convert to a Ship lattice. */
export async function importBlueprintFromFile(
  file: File,
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1,
): Promise<Ship> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    bitmap.close();
    throw new Error('Could not get 2D canvas context for blueprint import');
  }
  ctx.drawImage(bitmap, 0, 0);
  const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
  bitmap.close();
  return importBlueprintFromImageData(img, cx, cy, densityScale, strengthScale);
}
