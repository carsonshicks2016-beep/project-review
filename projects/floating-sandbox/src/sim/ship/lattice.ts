import type { BeamState, NodeState, Ship } from '../types';
import {
  BEAM_BRACE,
  BEAM_SKIN,
  FLAG_BROKEN,
  FLAG_PINNED,
  FLAG_SKIN,
} from '../types';
import { refreshInvMass } from './node-beam';

const EPS = 1e-8;
/** Yield starts at this fraction of the break limit (ductile materials). */
const YIELD_FRACTION = 0.55;
/** Plastic rest blend rate per constraint iteration at ductility=1. */
const PLASTIC_RATE = 0.12;
/** Mild velocity damping per step. */
const DAMPING = 0.998;
/** Brace beams get this compliance multiplier (stiffer). */
const BRACE_COMPLIANCE_SCALE = 0.25;
const DEFAULT_ITERATIONS = 6;

function dist2(ax: number, ay: number, bx: number, by: number): number {
  const dx = ax - bx;
  const dy = ay - by;
  return dx * dx + dy * dy;
}

/** True if segment (x0,y0)-(x1,y1) comes within `halfW` of segment (ax,ay)-(bx,by). */
function segmentsNear(
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
  halfW: number,
): boolean {
  const hw2 = halfW * halfW;
  // Sample endpoints + closest-point style check via parameterized distances.
  if (dist2(x0, y0, ax, ay) <= hw2) return true;
  if (dist2(x0, y0, bx, by) <= hw2) return true;
  if (dist2(x1, y1, ax, ay) <= hw2) return true;
  if (dist2(x1, y1, bx, by) <= hw2) return true;

  const abx = bx - ax;
  const aby = by - ay;
  const cdx = x1 - x0;
  const cdy = y1 - y0;
  const den = abx * cdy - aby * cdx;
  if (Math.abs(den) > EPS) {
    const t = ((x0 - ax) * cdy - (y0 - ay) * cdx) / den;
    const u = ((x0 - ax) * aby - (y0 - ay) * abx) / den;
    if (t >= 0 && t <= 1 && u >= 0 && u <= 1) return true;
  }

  // Distance from cut endpoints to beam segment and vice versa.
  return (
    pointSegDist2(x0, y0, ax, ay, bx, by) <= hw2 ||
    pointSegDist2(x1, y1, ax, ay, bx, by) <= hw2 ||
    pointSegDist2(ax, ay, x0, y0, x1, y1) <= hw2 ||
    pointSegDist2(bx, by, x0, y0, x1, y1) <= hw2
  );
}

function pointSegDist2(
  px: number,
  py: number,
  ax: number,
  ay: number,
  bx: number,
  by: number,
): number {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const ab2 = abx * abx + aby * aby;
  let t = ab2 > EPS ? (apx * abx + apy * aby) / ab2 : 0;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  const cx = ax + abx * t - px;
  const cy = ay + aby * t - py;
  return cx * cx + cy * cy;
}

function breakBeam(ship: Ship, bi: number): void {
  const beams = ship.beams;
  if (beams.alive[bi] === 0) return;
  beams.alive[bi] = 0;
  beams.strain[bi] = 0;

  if ((beams.flags[bi]! & BEAM_SKIN) === 0) return;

  const a = beams.a[bi]!;
  const b = beams.b[bi]!;
  clearSkinIfOrphaned(ship, a);
  clearSkinIfOrphaned(ship, b);
}

function clearSkinIfOrphaned(ship: Ship, ni: number): void {
  const nodes = ship.nodes;
  const beams = ship.beams;
  if ((nodes.flags[ni]! & FLAG_SKIN) === 0) return;

  for (let i = 0; i < beams.count; i++) {
    if (beams.alive[i] === 0) continue;
    if ((beams.flags[i]! & BEAM_SKIN) === 0) continue;
    if (beams.a[i] === ni || beams.b[i] === ni) return;
  }

  nodes.flags[ni]! &= ~FLAG_SKIN;
  nodes.flags[ni]! |= FLAG_BROKEN;
}

function restoreSkinFlags(ship: Ship, bi: number): void {
  const beams = ship.beams;
  if ((beams.flags[bi]! & BEAM_SKIN) === 0) return;
  const nodes = ship.nodes;
  const a = beams.a[bi]!;
  const b = beams.b[bi]!;
  nodes.flags[a]! |= FLAG_SKIN;
  nodes.flags[a]! &= ~FLAG_BROKEN;
  nodes.flags[b]! |= FLAG_SKIN;
  nodes.flags[b]! &= ~FLAG_BROKEN;
}

function solveBeams(nodes: NodeState, beams: BeamState, ship: Ship, dt: number): void {
  const alphaScale = dt > 0 ? 1 / (dt * dt) : 0;

  for (let i = 0; i < beams.count; i++) {
    if (beams.alive[i] === 0) continue;

    const ia = beams.a[i]!;
    const ib = beams.b[i]!;
    const wa = nodes.invMass[ia]!;
    const wb = nodes.invMass[ib]!;
    const wSum = wa + wb;
    if (wSum < EPS) continue;

    let rest = beams.plasticRest[i]!;
    if (rest < EPS) rest = beams.rest[i]!;
    if (rest < EPS) continue;

    const dx = nodes.px[ib]! - nodes.px[ia]!;
    const dy = nodes.py[ib]! - nodes.py[ia]!;
    const len = Math.hypot(dx, dy);
    if (len < EPS) continue;

    const strain = (len - rest) / rest;
    beams.strain[i] = strain;

    // Ductile plastic rest update (keep the dent).
    const duct = beams.ductility[i]!;
    if (duct > 0) {
      const lim = strain >= 0 ? beams.tensionLimit[i]! : beams.compressionLimit[i]!;
      if (Math.abs(strain) > YIELD_FRACTION * lim) {
        const blend = duct * PLASTIC_RATE;
        rest = rest + (len - rest) * blend;
        if (rest < EPS) rest = EPS;
        beams.plasticRest[i] = rest;
      }
    }

    // Recompute strain against (possibly updated) plastic rest for break / solve.
    const strain2 = (len - rest) / rest;
    beams.strain[i] = strain2;

    if (strain2 > beams.tensionLimit[i]! || -strain2 > beams.compressionLimit[i]!) {
      breakBeam(ship, i);
      continue;
    }

    let compliance = beams.compliance[i]!;
    if ((beams.flags[i]! & BEAM_BRACE) !== 0) {
      compliance *= BRACE_COMPLIANCE_SCALE;
    }
    const alpha = compliance * alphaScale;
    const C = len - rest;
    const dl = -C / (wSum + alpha);
    const nx = dx / len;
    const ny = dy / len;
    const corrAx = -wa * dl * nx;
    const corrAy = -wa * dl * ny;
    const corrBx = wb * dl * nx;
    const corrBy = wb * dl * ny;

    nodes.px[ia]! += corrAx;
    nodes.py[ia]! += corrAy;
    nodes.px[ib]! += corrBx;
    nodes.py[ib]! += corrBy;
  }
}

export class ShipLatticeSolver {
  step(ship: Ship, dt: number, gravity: number, iterations = DEFAULT_ITERATIONS): void {
    if (dt <= 0 || ship.nodes.count === 0) return;

    const nodes = ship.nodes;
    const n = nodes.count;

    // Integrate: gravity + mild damping → predicted positions.
    for (let i = 0; i < n; i++) {
      if ((nodes.flags[i]! & FLAG_PINNED) !== 0 || nodes.invMass[i]! === 0) {
        nodes.px[i] = nodes.x[i]!;
        nodes.py[i] = nodes.y[i]!;
        nodes.vx[i] = 0;
        nodes.vy[i] = 0;
        continue;
      }
      nodes.vy[i]! += gravity * dt;
      nodes.vx[i]! *= DAMPING;
      nodes.vy[i]! *= DAMPING;
      nodes.px[i] = nodes.x[i]! + nodes.vx[i]! * dt;
      nodes.py[i] = nodes.y[i]! + nodes.vy[i]! * dt;
    }

    const iters = iterations > 0 ? iterations | 0 : DEFAULT_ITERATIONS;
    for (let k = 0; k < iters; k++) {
      solveBeams(nodes, ship.beams, ship, dt);
      // Keep pins stuck.
      for (let i = 0; i < n; i++) {
        if ((nodes.flags[i]! & FLAG_PINNED) !== 0 || nodes.invMass[i]! === 0) {
          nodes.px[i] = nodes.x[i]!;
          nodes.py[i] = nodes.y[i]!;
        }
      }
    }

    const invDt = 1 / dt;
    for (let i = 0; i < n; i++) {
      if ((nodes.flags[i]! & FLAG_PINNED) !== 0 || nodes.invMass[i]! === 0) {
        nodes.vx[i] = 0;
        nodes.vy[i] = 0;
        continue;
      }
      nodes.vx[i] = (nodes.px[i]! - nodes.x[i]!) * invDt;
      nodes.vy[i] = (nodes.py[i]! - nodes.y[i]!) * invDt;
      nodes.x[i] = nodes.px[i]!;
      nodes.y[i] = nodes.py[i]!;
    }
  }

  /** Apply impulse at world point (soft falloff inside radius). */
  impulseAt(ship: Ship, x: number, y: number, ix: number, iy: number, radius: number): void {
    const nodes = ship.nodes;
    const r = Math.max(radius, EPS);
    const r2 = r * r;
    for (let i = 0; i < nodes.count; i++) {
      if (nodes.invMass[i]! === 0) continue;
      const d2 = dist2(nodes.x[i]!, nodes.y[i]!, x, y);
      if (d2 > r2) continue;
      const d = Math.sqrt(d2);
      const w = 1 - d / r;
      const s = w * w * nodes.invMass[i]!;
      nodes.vx[i]! += ix * s;
      nodes.vy[i]! += iy * s;
    }
  }

  /** Radial smash: impulse outward + damage beams by proximity. */
  smash(ship: Ship, x: number, y: number, radius: number, damage: number): void {
    const nodes = ship.nodes;
    const beams = ship.beams;
    const r = Math.max(radius, EPS);
    const r2 = r * r;
    const impulseMag = damage * 40;

    for (let i = 0; i < nodes.count; i++) {
      if (nodes.invMass[i]! === 0) continue;
      const dx = nodes.x[i]! - x;
      const dy = nodes.y[i]! - y;
      const d2 = dx * dx + dy * dy;
      if (d2 > r2 || d2 < EPS) continue;
      const d = Math.sqrt(d2);
      const w = 1 - d / r;
      const s = (w * w * impulseMag) * nodes.invMass[i]!;
      nodes.vx[i]! += (dx / d) * s;
      nodes.vy[i]! += (dy / d) * s;
    }

    // Tear beams by reducing effective limits / snapping when damage is high.
    for (let i = 0; i < beams.count; i++) {
      if (beams.alive[i] === 0) continue;
      const ia = beams.a[i]!;
      const ib = beams.b[i]!;
      const mx = (nodes.x[ia]! + nodes.x[ib]!) * 0.5;
      const my = (nodes.y[ia]! + nodes.y[ib]!) * 0.5;
      const d2 = dist2(mx, my, x, y);
      if (d2 > r2) continue;
      const d = Math.sqrt(d2);
      const w = 1 - d / r;
      const hit = damage * w * w;
      beams.tensionLimit[i]! *= Math.max(0, 1 - hit * 0.85);
      beams.compressionLimit[i]! *= Math.max(0, 1 - hit * 0.85);
      if (hit > 0.55 || beams.tensionLimit[i]! < 0.02 || beams.compressionLimit[i]! < 0.02) {
        breakBeam(ship, i);
      }
    }
  }

  /** Cut beams intersecting the segment thickened by `width`. */
  cut(ship: Ship, x0: number, y0: number, x1: number, y1: number, width: number): void {
    const nodes = ship.nodes;
    const beams = ship.beams;
    const halfW = Math.max(width * 0.5, 0.5);

    for (let i = 0; i < beams.count; i++) {
      if (beams.alive[i] === 0) continue;
      const ia = beams.a[i]!;
      const ib = beams.b[i]!;
      if (
        segmentsNear(
          x0,
          y0,
          x1,
          y1,
          nodes.x[ia]!,
          nodes.y[ia]!,
          nodes.x[ib]!,
          nodes.y[ib]!,
          halfW,
        )
      ) {
        breakBeam(ship, i);
      }
    }
  }

  /** Toggle pin on nodes inside radius. */
  pinToggle(ship: Ship, x: number, y: number, radius: number): void {
    const nodes = ship.nodes;
    const r2 = Math.max(radius, EPS) ** 2;
    let anyPinned = false;
    let anyUnpinned = false;

    // If any unpinned node is hit, pin all hit nodes; else unpin all hit.
    for (let i = 0; i < nodes.count; i++) {
      if (dist2(nodes.x[i]!, nodes.y[i]!, x, y) > r2) continue;
      if ((nodes.flags[i]! & FLAG_PINNED) !== 0) anyPinned = true;
      else anyUnpinned = true;
    }

    const shouldPin = anyUnpinned || !anyPinned;
    for (let i = 0; i < nodes.count; i++) {
      if (dist2(nodes.x[i]!, nodes.y[i]!, x, y) > r2) continue;
      if (shouldPin) {
        nodes.flags[i]! |= FLAG_PINNED;
        nodes.invMass[i] = 0;
        nodes.vx[i] = 0;
        nodes.vy[i] = 0;
        nodes.px[i] = nodes.x[i]!;
        nodes.py[i] = nodes.y[i]!;
      } else {
        nodes.flags[i]! &= ~FLAG_PINNED;
        refreshInvMass(nodes, i);
      }
    }
  }

  /** Restore broken beams whose midpoint lies in radius. */
  repair(ship: Ship, x: number, y: number, radius: number): void {
    const nodes = ship.nodes;
    const beams = ship.beams;
    const r2 = Math.max(radius, EPS) ** 2;

    for (let i = 0; i < beams.count; i++) {
      const ia = beams.a[i]!;
      const ib = beams.b[i]!;
      const mx = (nodes.x[ia]! + nodes.x[ib]!) * 0.5;
      const my = (nodes.y[ia]! + nodes.y[ib]!) * 0.5;
      if (dist2(mx, my, x, y) > r2) continue;

      beams.alive[i] = 1;
      beams.plasticRest[i] = beams.rest[i]!;
      beams.strain[i] = 0;
      // Nudge endpoints toward rest length gently via positions unchanged;
      // restoring skin / clearing broken markers.
      restoreSkinFlags(ship, i);
      nodes.flags[ia]! &= ~FLAG_BROKEN;
      nodes.flags[ib]! &= ~FLAG_BROKEN;
    }
  }
}
