import { indexOfMaterial, MATERIALS } from '../materials';
import type { BeamState, MaterialId, NodeState, Ship } from '../types';
import { BEAM_BRACE, BEAM_SKIN, FLAG_SKIN } from '../types';
import { addBeamFromMaterials, addNode, createBeams, createNodes } from './node-beam';

function gridIndex(col: number, row: number, cols: number): number {
  return row * cols + col;
}

function isPerimeter(col: number, row: number, cols: number, rows: number): boolean {
  return col === 0 || row === 0 || col === cols - 1 || row === rows - 1;
}

/**
 * Connect a row-major node grid with 4-neighborhood (and optional diagonals).
 * When `skin` is true, perimeter edges get BEAM_SKIN and nodes get FLAG_SKIN.
 */
export function connectGrid(
  nodes: NodeState,
  beams: BeamState,
  cols: number,
  rows: number,
  densScale = 1,
  strengthScale = 1,
  diagonals = false,
  skin = false,
): void {
  if (cols < 1 || rows < 1) return;
  const expected = cols * rows;
  if (nodes.count < expected) {
    throw new Error(`connectGrid: need ${expected} nodes, have ${nodes.count}`);
  }

  if (skin) {
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (isPerimeter(c, r, cols, rows)) {
          const i = gridIndex(c, r, cols);
          nodes.flags[i]! |= FLAG_SKIN;
        }
      }
    }
  }

  const link = (a: number, b: number, asSkin: boolean) => {
    const flags = asSkin ? BEAM_SKIN : 0;
    addBeamFromMaterials(beams, nodes, a, b, densScale, strengthScale, flags);
  };

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = gridIndex(c, r, cols);
      if (c + 1 < cols) {
        const j = gridIndex(c + 1, r, cols);
        const edgeSkin = skin && (r === 0 || r === rows - 1);
        link(i, j, edgeSkin);
      }
      if (r + 1 < rows) {
        const j = gridIndex(c, r + 1, cols);
        const edgeSkin = skin && (c === 0 || c === cols - 1);
        link(i, j, edgeSkin);
      }
      if (diagonals) {
        if (c + 1 < cols && r + 1 < rows) {
          link(i, gridIndex(c + 1, r + 1, cols), false);
        }
        if (c + 1 < cols && r - 1 >= 0) {
          link(i, gridIndex(c + 1, r - 1, cols), false);
        }
      }
    }
  }
}

/**
 * Add long longitudinal tie beams along each row every `everyN` columns.
 * Marked BEAM_BRACE for higher effective stiffness in the solver.
 */
export function addLongitudinalBraces(
  nodes: NodeState,
  beams: BeamState,
  cols: number,
  rows: number,
  everyN: number,
  strengthScale = 1,
): void {
  const step = Math.max(2, everyN | 0);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c + step < cols; c += step) {
      const a = gridIndex(c, r, cols);
      const b = gridIndex(c + step, r, cols);
      addBeamFromMaterials(beams, nodes, a, b, 1, strengthScale, BEAM_BRACE);
    }
  }
}

/**
 * Build a rectangular lattice ship from world bounds with uniform spacing.
 * Mass = material.density * spacing² * densityScale.
 */
export function buildRectLattice(opts: {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  spacing: number;
  material: MaterialId;
  densityScale?: number;
  strengthScale?: number;
  skin: boolean;
  braces?: boolean;
  compartmentId?: number;
}): Ship {
  const {
    spacing,
    material,
    skin,
    braces = false,
    compartmentId = 0,
  } = opts;
  const densityScale = opts.densityScale ?? 1;
  const strengthScale = opts.strengthScale ?? 1;

  const x0 = Math.min(opts.x0, opts.x1);
  const x1 = Math.max(opts.x0, opts.x1);
  const y0 = Math.min(opts.y0, opts.y1);
  const y1 = Math.max(opts.y0, opts.y1);
  const sp = Math.max(spacing, 1e-4);

  const cols = Math.max(2, Math.floor((x1 - x0) / sp) + 1);
  const rows = Math.max(2, Math.floor((y1 - y0) / sp) + 1);
  const nodeCap = cols * rows;
  // 4-neigh + optional diags + braces headroom
  const beamCap = cols * rows * 6 + cols * rows;

  const nodes = createNodes(nodeCap);
  const beams = createBeams(beamCap);
  const mat = MATERIALS[material];
  const matIndex = indexOfMaterial(material);
  const mass = mat.density * sp * sp * densityScale;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = cols === 1 ? x0 : x0 + (c / (cols - 1)) * (x1 - x0);
      const y = rows === 1 ? y0 : y0 + (r / (rows - 1)) * (y1 - y0);
      const flags = skin && isPerimeter(c, r, cols, rows) ? FLAG_SKIN : 0;
      addNode(nodes, x, y, matIndex, mass, flags, compartmentId);
    }
  }

  connectGrid(nodes, beams, cols, rows, densityScale, strengthScale, false, skin);

  if (braces) {
    const everyN = Math.max(2, Math.round(cols / 4) || 2);
    addLongitudinalBraces(nodes, beams, cols, rows, everyN, strengthScale * 1.25);
  }

  return {
    nodes,
    beams,
    name: `rect-${material}`,
  };
}
