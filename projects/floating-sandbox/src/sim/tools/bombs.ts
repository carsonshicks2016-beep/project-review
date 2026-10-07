import type { Ship, WaterState } from '../types';
import type { ShipLatticeSolver } from '../ship/lattice.ts';

const EPS = 1e-8;

/**
 * Radial shock: impulse ship nodes outward (tear by strain, no delete-circle)
 * and push nearby water particles away from the blast.
 */
export function detonateBomb(
  ship: Ship,
  water: WaterState,
  x: number,
  y: number,
  power: number,
  radius: number,
  lattice: ShipLatticeSolver,
): void {
  const r = Math.max(radius, EPS);
  const r2 = r * r;
  const waterScale = 0.55;

  // Ship: radial impulses via lattice.impulseAt, localized at each node so
  // the directional API produces an outward blast field.
  const nodes = ship.nodes;
  for (let i = 0; i < nodes.count; i++) {
    if (nodes.invMass[i]! === 0) continue;
    const dx = nodes.x[i]! - x;
    const dy = nodes.y[i]! - y;
    const d2 = dx * dx + dy * dy;
    if (d2 > r2 || d2 < EPS) continue;
    const d = Math.sqrt(d2);
    const w = 1 - d / r;
    const mag = power * w * w;
    lattice.impulseAt(
      ship,
      nodes.x[i]!,
      nodes.y[i]!,
      (dx / d) * mag,
      (dy / d) * mag,
      Math.max(r * 0.08, 1),
    );
  }

  // Water: push particles outward (velocity only — no delete).
  for (let i = 0; i < water.count; i++) {
    if (water.invMass[i]! === 0) continue;
    const dx = water.x[i]! - x;
    const dy = water.y[i]! - y;
    const d2 = dx * dx + dy * dy;
    if (d2 > r2 || d2 < EPS) continue;
    const d = Math.sqrt(d2);
    const w = 1 - d / r;
    const mag = power * waterScale * w * w;
    water.vx[i]! += (dx / d) * mag;
    water.vy[i]! += (dy / d) * mag;
  }
}
