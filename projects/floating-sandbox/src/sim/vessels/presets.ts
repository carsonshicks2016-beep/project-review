import { MATERIALS, indexOfMaterial } from '../materials';
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
import {
  addLongitudinalBraces,
  connectGrid,
} from '../ship/mesh-builder';

/** Mass for a structural node from material density and cell size. */
function nodeMass(mat: MaterialId, spacing: number, densityScale: number): number {
  return MATERIALS[mat].density * spacing * spacing * densityScale;
}

function idx(col: number, row: number, cols: number): number {
  return row * cols + col;
}

/**
 * Build a rectangular hull lattice with keel, deck, skin, bulkheads, and
 * length-wise compartments. Keel (deepest, largest y) sits at `cy`.
 * Returns the ship plus grid dimensions (for attaching superstructure).
 */
function buildHull(opts: {
  name: string;
  cx: number;
  cy: number;
  cols: number;
  rows: number;
  spacing: number;
  densityScale: number;
  strengthScale: number;
  compartments?: number;
  engineAft?: boolean;
  bracesEvery?: number;
}): Ship & { cols: number; rows: number; spacing: number } {
  const {
    name,
    cx,
    cy,
    cols,
    rows,
    spacing,
    densityScale,
    strengthScale,
    compartments = 1,
    engineAft = false,
    bracesEvery = 3,
  } = opts;

  const width = (cols - 1) * spacing;
  const x0 = cx - width * 0.5;
  // y-down: keel at cy (deepest); row index increases toward the deck (−y)
  const yKeel = cy;

  const nCap = cols * rows + 256;
  const bCap = cols * rows * 6 + 512;
  const nodes = createNodes(nCap);
  const beams = createBeams(bCap);

  const engineColStart = engineAft ? Math.floor(cols * 0.72) : cols;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x = x0 + c * spacing;
      const y = yKeel - r * spacing;
      const onBoundary = c === 0 || c === cols - 1 || r === 0 || r === rows - 1;

      let mat: MaterialId = 'steel';
      if (r === 0) {
        mat = 'keel';
      } else if (r === rows - 1) {
        mat = 'deck';
      } else if (engineAft && c >= engineColStart && r <= Math.floor(rows * 0.45)) {
        mat = 'engine';
      } else if (onBoundary) {
        mat = 'steel';
      } else {
        // Vertical bulkhead where compartment id changes along length
        const compHere = Math.floor((c / cols) * compartments);
        const compPrev = Math.floor(((c - 1) / cols) * compartments);
        const nearBulkhead = compartments > 1 && c > 0 && compHere !== compPrev;
        mat = nearBulkhead ? 'bulkhead' : 'compartment';
      }

      const flags = onBoundary ? FLAG_SKIN : 0;
      const compartmentId =
        compartments <= 1
          ? 1
          : Math.min(compartments, Math.floor((c / cols) * compartments) + 1);

      addNode(
        nodes,
        x,
        y,
        indexOfMaterial(mat),
        nodeMass(mat, spacing, densityScale),
        flags,
        compartmentId,
      );
    }
  }

  // 4-connected + diagonals; perimeter edges marked as watertight skin
  connectGrid(nodes, beams, cols, rows, densityScale, strengthScale, true, true);

  if (bracesEvery > 0) {
    addLongitudinalBraces(nodes, beams, cols, rows, bracesEvery, strengthScale);
  }

  return { nodes, beams, name, cols, rows, spacing };
}

/** Attach a smaller lattice above the hull (superstructure / funnel / crane). */
function attachLattice(
  ship: Ship,
  opts: {
    x0: number;
    y0: number;
    cols: number;
    rows: number;
    spacing: number;
    material: MaterialId;
    densityScale: number;
    strengthScale: number;
    skin?: boolean;
    compartmentId?: number;
    braces?: boolean;
  },
): void {
  const {
    x0,
    y0,
    cols,
    rows,
    spacing,
    material,
    densityScale,
    strengthScale,
    skin = false,
    compartmentId = 0,
    braces = false,
  } = opts;

  const offset = ship.nodes.count;
  const matIdx = indexOfMaterial(material);
  const mass = nodeMass(material, spacing, densityScale);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const onBoundary = c === 0 || c === cols - 1 || r === 0 || r === rows - 1;
      const flags = skin && onBoundary ? FLAG_SKIN : 0;
      addNode(
        ship.nodes,
        x0 + c * spacing,
        // y-down: r=0 at deck attachment, higher r toward sky (−y)
        y0 - r * spacing,
        matIdx,
        mass,
        flags,
        compartmentId,
      );
    }
  }

  // Connect new grid in isolation (indices offset..offset+cols*rows-1)
  connectAttachedGrid(
    ship,
    offset,
    cols,
    rows,
    densityScale,
    strengthScale,
    skin,
  );

  if (braces) {
    // Longitudinal braces within the attached block via direct long ties
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols - 2; c += 2) {
        const a = offset + idx(c, r, cols);
        const b = offset + idx(c + 2, r, cols);
        addBeamFromMaterials(
          ship.beams,
          ship.nodes,
          a,
          b,
          densityScale,
          strengthScale,
          BEAM_BRACE,
        );
      }
    }
  }

  // Pin-attach bottom row of attachment to nearest existing hull nodes below
  stitchToHull(ship, offset, cols, spacing);
}

/** 4-neigh (+ optional diagonals) for a node block starting at `offset`. */
function connectAttachedGrid(
  ship: Ship,
  offset: number,
  cols: number,
  rows: number,
  densityScale: number,
  strengthScale: number,
  skin: boolean,
): void {
  const link = (a: number, b: number, skinEdge: boolean) => {
    addBeamFromMaterials(
      ship.beams,
      ship.nodes,
      a,
      b,
      densityScale,
      strengthScale,
      skinEdge ? BEAM_SKIN : 0,
    );
  };

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = offset + idx(c, r, cols);
      if (c + 1 < cols) {
        const j = offset + idx(c + 1, r, cols);
        const edge = skin && (r === 0 || r === rows - 1);
        link(i, j, edge);
      }
      if (r + 1 < rows) {
        const j = offset + idx(c, r + 1, cols);
        const edge = skin && (c === 0 || c === cols - 1);
        link(i, j, edge);
      }
      // Shear diagonals
      if (c + 1 < cols && r + 1 < rows) {
        link(i, offset + idx(c + 1, r + 1, cols), false);
      }
    }
  }
}

/** Weld bottom row of an attached lattice to nearby existing nodes underneath. */
function stitchToHull(
  ship: Ship,
  offset: number,
  cols: number,
  spacing: number,
): void {
  const radius = spacing * 1.35;
  for (let c = 0; c < cols; c++) {
    const ai = offset + c; // bottom row of attachment (r=0)
    const ax = ship.nodes.x[ai]!;
    const ay = ship.nodes.y[ai]!;
    let best = -1;
    let bestD = radius * radius;
    for (let i = 0; i < offset; i++) {
      const dx = ship.nodes.x[i]! - ax;
      const dy = ship.nodes.y[i]! - ay;
      const d2 = dx * dx + dy * dy;
      if (d2 < bestD) {
        bestD = d2;
        best = i;
      }
    }
    if (best >= 0) {
      addBeamFromMaterials(ship.beams, ship.nodes, ai, best, 1, 1.2, 0);
    }
  }
}

// ---------------------------------------------------------------------------
// Presets
// ---------------------------------------------------------------------------

export function buildLiner(
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1.25,
): Ship {
  const spacing = 5;
  const hull = buildHull({
    name: 'Ocean Liner',
    cx,
    cy,
    cols: 36,
    rows: 8,
    spacing,
    densityScale,
    strengthScale,
    compartments: 6,
    engineAft: true,
    bracesEvery: 3,
  });

  const width = (hull.cols - 1) * spacing;
  const x0 = cx - width * 0.5;
  const deckY = cy - (hull.rows - 1) * spacing;

  // Midships superstructure (grows toward sky / −y from deck)
  attachLattice(hull, {
    x0: x0 + spacing * 10,
    y0: deckY,
    cols: 12,
    rows: 4,
    spacing,
    material: 'superstructure',
    densityScale,
    strengthScale,
    skin: true,
    compartmentId: 0,
  });

  // Funnel
  attachLattice(hull, {
    x0: x0 + spacing * 14,
    y0: deckY - 4 * spacing,
    cols: 3,
    rows: 5,
    spacing,
    material: 'funnel',
    densityScale,
    strengthScale,
    skin: false,
  });

  return { nodes: hull.nodes, beams: hull.beams, name: hull.name };
}

export function buildFreighter(
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1.25,
): Ship {
  const spacing = 5;
  const hull = buildHull({
    name: 'Freighter',
    cx,
    cy,
    cols: 32,
    rows: 7,
    spacing,
    densityScale,
    strengthScale,
    compartments: 5,
    engineAft: true,
    bracesEvery: 3,
  });

  const width = (hull.cols - 1) * spacing;
  const x0 = cx - width * 0.5;
  const deckY = cy - (hull.rows - 1) * spacing;

  // Small aft bridge house
  attachLattice(hull, {
    x0: x0 + spacing * 24,
    y0: deckY,
    cols: 5,
    rows: 3,
    spacing,
    material: 'superstructure',
    densityScale,
    strengthScale,
    skin: true,
  });

  return { nodes: hull.nodes, beams: hull.beams, name: hull.name };
}

export function buildTug(
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1.25,
): Ship {
  const spacing = 4.5;
  const hull = buildHull({
    name: 'Tugboat',
    cx,
    cy,
    cols: 16,
    rows: 6,
    spacing,
    densityScale,
    strengthScale,
    compartments: 3,
    engineAft: true,
    bracesEvery: 2,
  });

  const width = (hull.cols - 1) * spacing;
  const x0 = cx - width * 0.5;
  const deckY = cy - (hull.rows - 1) * spacing;

  attachLattice(hull, {
    x0: x0 + spacing * 5,
    y0: deckY,
    cols: 5,
    rows: 3,
    spacing,
    material: 'superstructure',
    densityScale,
    strengthScale,
    skin: true,
  });

  // Short funnel
  attachLattice(hull, {
    x0: x0 + spacing * 7,
    y0: deckY - 3 * spacing,
    cols: 2,
    rows: 3,
    spacing,
    material: 'funnel',
    densityScale,
    strengthScale,
  });

  return { nodes: hull.nodes, beams: hull.beams, name: hull.name };
}

export function buildBarge(
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1.25,
): Ship {
  const spacing = 5;
  const hull = buildHull({
    name: 'Barge',
    cx,
    cy,
    cols: 28,
    rows: 5,
    spacing,
    densityScale,
    strengthScale,
    compartments: 4,
    engineAft: false,
    bracesEvery: 4,
  });
  return { nodes: hull.nodes, beams: hull.beams, name: hull.name };
}

export function buildRaft(
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1,
): Ship {
  const spacing = 6;
  const cols = 12;
  const rows = 2;
  const width = (cols - 1) * spacing;
  const x0 = cx - width * 0.5;
  // y-down: lower slat deeper (larger y)
  const yDeck = cy - spacing;
  const yKeel = cy;

  const nodes = createNodes(cols * rows + 16);
  const beams = createBeams(cols * rows * 4 + 32);
  const mat = 'wood' as const;
  const mass = nodeMass(mat, spacing, densityScale);

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      // Sparse: skip every other interior cell on the lower (keel) row for a slatted look
      if (r === 0 && c > 0 && c < cols - 1 && c % 2 === 1) continue;
      addNode(
        nodes,
        x0 + c * spacing,
        r === 0 ? yKeel : yDeck,
        indexOfMaterial(mat),
        mass,
        FLAG_SKIN,
        1,
      );
    }
  }

  // Connect by proximity (sparse grid is not a full rectangle)
  for (let i = 0; i < nodes.count; i++) {
    for (let j = i + 1; j < nodes.count; j++) {
      const dx = nodes.x[j]! - nodes.x[i]!;
      const dy = nodes.y[j]! - nodes.y[i]!;
      const d = Math.hypot(dx, dy);
      if (d > 0.5 && d < spacing * 1.55) {
        addBeamFromMaterials(
          beams,
          nodes,
          i,
          j,
          densityScale,
          strengthScale,
          BEAM_SKIN,
        );
      }
    }
  }

  // A few long ties along the deck
  for (let i = 0; i < nodes.count; i++) {
    for (let j = i + 1; j < nodes.count; j++) {
      const dx = nodes.x[j]! - nodes.x[i]!;
      const dy = nodes.y[j]! - nodes.y[i]!;
      if (Math.abs(dy) < spacing * 0.25 && Math.abs(dx - spacing * 2) < spacing * 0.3) {
        addBeamFromMaterials(
          beams,
          nodes,
          i,
          j,
          densityScale,
          strengthScale,
          BEAM_BRACE,
        );
      }
    }
  }

  return { nodes, beams, name: 'Raft' };
}

export function buildCraneBarge(
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1.25,
): Ship {
  const spacing = 5;
  const hull = buildHull({
    name: 'Crane Barge',
    cx,
    cy,
    cols: 24,
    rows: 5,
    spacing,
    densityScale,
    strengthScale,
    compartments: 3,
    engineAft: false,
    bracesEvery: 3,
  });

  const width = (hull.cols - 1) * spacing;
  const x0 = cx - width * 0.5;
  const deckY = cy - (hull.rows - 1) * spacing;

  // Tall crane mast midship (steel)
  attachLattice(hull, {
    x0: x0 + spacing * 10,
    y0: deckY,
    cols: 2,
    rows: 10,
    spacing,
    material: 'steel',
    densityScale,
    strengthScale,
    braces: true,
  });

  // Boom angled forward as a horizontal lattice at the top
  attachLattice(hull, {
    x0: x0 + spacing * 4,
    y0: deckY - 9 * spacing,
    cols: 10,
    rows: 2,
    spacing,
    material: 'funnel',
    densityScale,
    strengthScale,
    braces: true,
  });

  return { nodes: hull.nodes, beams: hull.beams, name: hull.name };
}

export function buildIceberg(
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1,
): Ship {
  const spacing = 5;
  // Irregular iceberg silhouette sampled on a grid
  const cols = 18;
  const rows = 14;
  const width = (cols - 1) * spacing;
  const height = (rows - 1) * spacing;
  const x0 = cx - width * 0.5;
  // Sit most of the mass below cy (iceberg hangs deep)
  const y0 = cy - height * 0.55;

  const nodes = createNodes(cols * rows + 8);
  const beams = createBeams(cols * rows * 5 + 64);
  const grid: Int32Array = new Int32Array(cols * rows).fill(-1);

  const inside = (c: number, r: number): boolean => {
    const u = (c / (cols - 1)) * 2 - 1;
    const v = (r / (rows - 1)) * 2 - 1;
    // Soft blob: wider at waterline (mid rows), pointed tip above, deep keel below
    const waterline = 0.15;
    const halfW =
      v > waterline
        ? 0.55 * (1 - (v - waterline) / (1 - waterline))
        : 0.75 * (1 - Math.abs(v - waterline) * 0.35);
    return Math.abs(u) < halfW + 0.08 * Math.sin(r * 1.7 + c * 0.4);
  };

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (!inside(c, r)) continue;
      const neighborMissing =
        !inside(c - 1, r) ||
        !inside(c + 1, r) ||
        !inside(c, r - 1) ||
        !inside(c, r + 1);
      const flags = neighborMissing ? FLAG_SKIN : 0;
      const i = addNode(
        nodes,
        x0 + c * spacing,
        y0 + r * spacing,
        indexOfMaterial('ice'),
        nodeMass('ice', spacing, densityScale),
        flags,
        1,
      );
      grid[idx(c, r, cols)] = i;
    }
  }

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const i = grid[idx(c, r, cols)]!;
      if (i < 0) continue;
      const right = c + 1 < cols ? grid[idx(c + 1, r, cols)]! : -1;
      const up = r + 1 < rows ? grid[idx(c, r + 1, cols)]! : -1;
      const diag = c + 1 < cols && r + 1 < rows ? grid[idx(c + 1, r + 1, cols)]! : -1;

      const skinH =
        (nodes.flags[i]! & FLAG_SKIN) !== 0 &&
        right >= 0 &&
        (nodes.flags[right]! & FLAG_SKIN) !== 0;
      const skinV =
        (nodes.flags[i]! & FLAG_SKIN) !== 0 &&
        up >= 0 &&
        (nodes.flags[up]! & FLAG_SKIN) !== 0;

      if (right >= 0) {
        addBeamFromMaterials(
          beams,
          nodes,
          i,
          right,
          densityScale,
          strengthScale,
          skinH ? BEAM_SKIN : 0,
        );
      }
      if (up >= 0) {
        addBeamFromMaterials(
          beams,
          nodes,
          i,
          up,
          densityScale,
          strengthScale,
          skinV ? BEAM_SKIN : 0,
        );
      }
      if (diag >= 0) {
        addBeamFromMaterials(beams, nodes, i, diag, densityScale, strengthScale, 0);
      }
    }
  }

  return { nodes, beams, name: 'Iceberg' };
}

/** Registry metadata for UI pickers. */
export const VESSEL_PRESETS: { id: string; name: string }[] = [
  { id: 'liner', name: 'Ocean Liner' },
  { id: 'freighter', name: 'Freighter' },
  { id: 'tug', name: 'Tugboat' },
  { id: 'barge', name: 'Barge' },
  { id: 'raft', name: 'Raft' },
  { id: 'craneBarge', name: 'Crane Barge' },
  { id: 'iceberg', name: 'Iceberg' },
];

const BUILDERS: Record<
  string,
  (cx: number, cy: number, densityScale: number, strengthScale: number) => Ship
> = {
  liner: buildLiner,
  freighter: buildFreighter,
  tug: buildTug,
  barge: buildBarge,
  raft: buildRaft,
  craneBarge: buildCraneBarge,
  iceberg: buildIceberg,
};

export function spawnVessel(
  id: string,
  cx: number,
  cy: number,
  densityScale = 1,
  strengthScale = 1,
): Ship | null {
  const build = BUILDERS[id];
  if (!build) return null;
  return build(cx, cy, densityScale, strengthScale);
}
