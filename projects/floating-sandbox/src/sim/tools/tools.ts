import type { Ship, ToolId, WaterState } from '../types';
import { FLAG_PINNED } from '../types';
import type { ShipLatticeSolver } from '../ship/lattice.ts';
import { addParticle, removeParticle } from '../water/water-state.ts';
import { detonateBomb } from './bombs.ts';
import type { ToolState } from './tool-state.ts';

export type { ToolState } from './tool-state.ts';
export { createToolState } from './tool-state.ts';
export { detonateBomb } from './bombs.ts';

const DRAG_PULL = 0.45;
const BOMB_FUSE_MS = 350;
const WATER_SPRINKLE = 12;

function dist2(ax: number, ay: number, bx: number, by: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  return dx * dx + dy * dy;
}

/** Nearest node index within maxDist, or -1. */
export function findNearestNode(
  ship: Ship,
  x: number,
  y: number,
  maxDist: number,
): number {
  const nodes = ship.nodes;
  let best = -1;
  let bestD = maxDist * maxDist;
  for (let i = 0; i < nodes.count; i++) {
    const d2 = dist2(nodes.x[i]!, nodes.y[i]!, x, y);
    if (d2 <= bestD) {
      bestD = d2;
      best = i;
    }
  }
  return best;
}

function refreshNodeInvMass(ship: Ship, i: number): void {
  const nodes = ship.nodes;
  if ((nodes.flags[i]! & FLAG_PINNED) !== 0) {
    nodes.invMass[i] = 0;
    return;
  }
  const m = nodes.mass[i]! + nodes.floodMass[i]!;
  nodes.invMass[i] = m > 0 ? 1 / m : 1;
}

function sprinkleWater(water: WaterState, x: number, y: number, radius: number): void {
  const n = Math.max(1, Math.min(WATER_SPRINKLE, Math.floor(radius / Math.max(water.spacing, 0.5)) + 2));
  for (let k = 0; k < n; k++) {
    const a = Math.random() * Math.PI * 2;
    const rr = Math.random() * radius;
    addParticle(water, x + Math.cos(a) * rr, y + Math.sin(a) * rr);
  }
}

function scoopWater(water: WaterState, x: number, y: number, radius: number): void {
  const r2 = radius * radius;
  for (let i = water.count - 1; i >= 0; i--) {
    if (dist2(water.x[i]!, water.y[i]!, x, y) <= r2) {
      removeParticle(water, i);
    }
  }
}

function reduceFloodInRadius(ship: Ship, x: number, y: number, radius: number): void {
  const nodes = ship.nodes;
  const r2 = radius * radius;
  for (let i = 0; i < nodes.count; i++) {
    if (dist2(nodes.x[i]!, nodes.y[i]!, x, y) > r2) continue;
    nodes.floodMass[i] = Math.max(0, nodes.floodMass[i]! * 0.82);
    if ((nodes.flags[i]! & FLAG_PINNED) === 0) {
      refreshNodeInvMass(ship, i);
    }
  }
}

/** Per-tool pointer handlers — kept as a clean ToolId map. */
export interface ToolPointerFns {
  down: (
    ctrl: ToolController,
    worldX: number,
    worldY: number,
    water: WaterState,
    ship: Ship | null,
    lattice: ShipLatticeSolver,
  ) => void;
  move: (
    ctrl: ToolController,
    worldX: number,
    worldY: number,
    water: WaterState,
    ship: Ship | null,
    lattice: ShipLatticeSolver,
  ) => void;
  up: (
    ctrl: ToolController,
    worldX: number,
    worldY: number,
    water: WaterState,
    ship: Ship | null,
    lattice: ShipLatticeSolver,
  ) => void;
}

const toolHandlers: Record<ToolId, ToolPointerFns> = {
  drag: {
    down(ctrl, worldX, worldY, _water, ship) {
      if (!ship) return;
      const i = findNearestNode(ship, worldX, worldY, ctrl.tools.brushRadius);
      if (i < 0) return;
      const nodes = ship.nodes;
      ctrl.dragNode = i;
      ctrl.dragWasPinned = (nodes.flags[i]! & FLAG_PINNED) !== 0;
      nodes.flags[i]! |= FLAG_PINNED;
      nodes.invMass[i] = 0;
      nodes.vx[i] = 0;
      nodes.vy[i] = 0;
      nodes.x[i] = worldX;
      nodes.y[i] = worldY;
      nodes.px[i] = worldX;
      nodes.py[i] = worldY;
    },
    move(ctrl, worldX, worldY, _water, ship) {
      if (!ship || ctrl.dragNode < 0) return;
      const i = ctrl.dragNode;
      const nodes = ship.nodes;
      if (i >= nodes.count) {
        ctrl.dragNode = -1;
        return;
      }
      nodes.x[i]! += (worldX - nodes.x[i]!) * DRAG_PULL;
      nodes.y[i]! += (worldY - nodes.y[i]!) * DRAG_PULL;
      nodes.px[i] = nodes.x[i]!;
      nodes.py[i] = nodes.y[i]!;
      nodes.vx[i] = 0;
      nodes.vy[i] = 0;
      nodes.flags[i]! |= FLAG_PINNED;
      nodes.invMass[i] = 0;
    },
    up(ctrl, _worldX, _worldY, _water, ship) {
      if (!ship || ctrl.dragNode < 0) {
        ctrl.dragNode = -1;
        return;
      }
      const i = ctrl.dragNode;
      const nodes = ship.nodes;
      if (i < nodes.count && !ctrl.dragWasPinned) {
        nodes.flags[i]! &= ~FLAG_PINNED;
        refreshNodeInvMass(ship, i);
      }
      ctrl.dragNode = -1;
      ctrl.dragWasPinned = false;
    },
  },

  smash: {
    down(ctrl, worldX, worldY, _water, ship, lattice) {
      if (!ship) return;
      const damage = 0.55 * ctrl.tools.strengthScale;
      lattice.smash(ship, worldX, worldY, ctrl.tools.brushRadius, damage);
    },
    move(ctrl, worldX, worldY, _water, ship, lattice) {
      if (!ship) return;
      const damage = 0.35 * ctrl.tools.strengthScale;
      lattice.smash(ship, worldX, worldY, ctrl.tools.brushRadius * 0.85, damage);
    },
    up() {},
  },

  cut: {
    down(ctrl, worldX, worldY) {
      ctrl.cutX = worldX;
      ctrl.cutY = worldY;
      ctrl.cutting = true;
    },
    move(ctrl, worldX, worldY, _water, ship, lattice) {
      if (!ship || !ctrl.cutting) return;
      const width = Math.max(2, ctrl.tools.brushRadius * 0.35);
      lattice.cut(ship, ctrl.cutX, ctrl.cutY, worldX, worldY, width);
      ctrl.cutX = worldX;
      ctrl.cutY = worldY;
    },
    up(ctrl, worldX, worldY, _water, ship, lattice) {
      if (ship && ctrl.cutting) {
        const width = Math.max(2, ctrl.tools.brushRadius * 0.35);
        lattice.cut(ship, ctrl.cutX, ctrl.cutY, worldX, worldY, width);
      }
      ctrl.cutting = false;
    },
  },

  bomb: {
    down(ctrl, worldX, worldY, water, ship, lattice) {
      if (!ship) return;
      const power = 55 * ctrl.tools.bombPower * ctrl.tools.strengthScale;
      const radius = ctrl.tools.brushRadius * (1.8 + ctrl.tools.bombPower);
      // Short fuse — delayed detonation; strain tear happens on blast.
      const sx = worldX;
      const sy = worldY;
      window.setTimeout(() => {
        detonateBomb(ship, water, sx, sy, power, radius, lattice);
      }, BOMB_FUSE_MS);
    },
    move() {},
    up() {},
  },

  pin: {
    down(ctrl, worldX, worldY, _water, ship, lattice) {
      if (!ship) return;
      lattice.pinToggle(ship, worldX, worldY, ctrl.tools.brushRadius);
    },
    move() {},
    up() {},
  },

  repair: {
    down(ctrl, worldX, worldY, _water, ship, lattice) {
      if (!ship) return;
      lattice.repair(ship, worldX, worldY, ctrl.tools.brushRadius);
      reduceFloodInRadius(ship, worldX, worldY, ctrl.tools.brushRadius);
    },
    move(ctrl, worldX, worldY, _water, ship, lattice) {
      if (!ship) return;
      lattice.repair(ship, worldX, worldY, ctrl.tools.brushRadius);
      reduceFloodInRadius(ship, worldX, worldY, ctrl.tools.brushRadius);
    },
    up() {},
  },

  addWater: {
    down(ctrl, worldX, worldY, water) {
      sprinkleWater(water, worldX, worldY, ctrl.tools.brushRadius);
    },
    move(ctrl, worldX, worldY, water) {
      sprinkleWater(water, worldX, worldY, ctrl.tools.brushRadius);
    },
    up() {},
  },

  removeWater: {
    down(ctrl, worldX, worldY, water) {
      scoopWater(water, worldX, worldY, ctrl.tools.brushRadius);
    },
    move(ctrl, worldX, worldY, water) {
      scoopWater(water, worldX, worldY, ctrl.tools.brushRadius);
    },
    up() {},
  },
};

export class ToolController {
  readonly tools: ToolState;

  /** Internal drag / cut gesture state (handlers read/write). */
  dragNode = -1;
  dragWasPinned = false;
  cutting = false;
  cutX = 0;
  cutY = 0;
  private pressing = false;

  constructor(tools: ToolState) {
    this.tools = tools;
  }

  setTool(id: ToolId): void {
    // Drop any in-progress drag/cut when switching tools.
    this.dragNode = -1;
    this.dragWasPinned = false;
    this.cutting = false;
    this.pressing = false;
    this.tools.active = id;
  }

  onPointerDown(
    worldX: number,
    worldY: number,
    water: WaterState,
    ship: Ship | null,
    lattice: ShipLatticeSolver,
  ): void {
    this.pressing = true;
    toolHandlers[this.tools.active].down(this, worldX, worldY, water, ship, lattice);
  }

  onPointerMove(
    worldX: number,
    worldY: number,
    water: WaterState,
    ship: Ship | null,
    lattice: ShipLatticeSolver,
    buttons: number,
  ): void {
    if (!this.pressing && (buttons & 1) === 0) return;
    if ((buttons & 1) !== 0) this.pressing = true;
    toolHandlers[this.tools.active].move(this, worldX, worldY, water, ship, lattice);
  }

  onPointerUp(
    worldX: number,
    worldY: number,
    water: WaterState,
    ship: Ship | null,
    lattice: ShipLatticeSolver,
  ): void {
    toolHandlers[this.tools.active].up(this, worldX, worldY, water, ship, lattice);
    this.pressing = false;
  }
}

export { toolHandlers };
