import {
  BEAM_SKIN,
  FLAG_PINNED,
  type Ship,
  type WaterState,
} from '../types';

/** Particle–hull contact radius relative to water spacing */
const PARTICLE_RADIUS_FACTOR = 0.4;
/** Node treated as circle for particle contacts */
const NODE_RADIUS_FACTOR = 0.3;
/** Max position correction per contact (× spacing) */
const MAX_CORR = 0.25;
/** Max impulse magnitude per contact — keep low so particle spam doesn't shred beams */
const MAX_IMPULSE = 0.45;
/** Soft restitution — nearly inelastic */
const RESTITUTION = 0.02;
/** Fraction of contact reaction applied to the ship (particles still fully corrected) */
const SHIP_FEEDBACK = 0.06;
/** Extra node-circle contacts are expensive and violent; skin segments suffice */
const ENABLE_NODE_COLLIDE = false;
const SLOP = 1e-5;

/**
 * Uniform-grid acceleration structure over alive BEAM_SKIN segments.
 * Particles collide with skin edges and with nodes as small circles;
 * equal/opposite impulses go to beam endpoints (barycentric weights).
 *
 * Coordinate convention: y increases downward (canvas-style). Up = −y.
 */
export class HullCollider {
  /** Skin beam indices */
  private beamIds: number[] = [];
  /** Flat entry store: each cell lists entry indices into entryBeam / entryNext */
  private entryBeam: Int32Array = new Int32Array(0);
  private entryNext: Int32Array = new Int32Array(0);
  private entryCount = 0;
  private heads: Int32Array = new Int32Array(0);
  private cellSize = 1;
  private originX = 0;
  private originY = 0;
  private cols = 0;
  private rows = 0;
  private dirty = true;

  invalidate(): void {
    this.dirty = true;
  }

  /** Rebuild skin segment list + grid from current ship topology/pose. */
  rebuild(ship: Ship, spacing: number): void {
    const { beams, nodes } = ship;
    this.beamIds.length = 0;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (let i = 0; i < beams.count; i++) {
      if (!beams.alive[i] || (beams.flags[i]! & BEAM_SKIN) === 0) continue;
      const ai = beams.a[i]!;
      const bi = beams.b[i]!;
      if (ai >= nodes.count || bi >= nodes.count) continue;
      this.beamIds.push(i);
      const ax = nodes.x[ai]!;
      const ay = nodes.y[ai]!;
      const bx = nodes.x[bi]!;
      const by = nodes.y[bi]!;
      if (ax < minX) minX = ax;
      if (bx < minX) minX = bx;
      if (ay < minY) minY = ay;
      if (by < minY) minY = by;
      if (ax > maxX) maxX = ax;
      if (bx > maxX) maxX = bx;
      if (ay > maxY) maxY = ay;
      if (by > maxY) maxY = by;
    }

    const nSeg = this.beamIds.length;
    if (nSeg === 0 || !Number.isFinite(minX)) {
      this.cols = 0;
      this.rows = 0;
      this.entryCount = 0;
      this.dirty = false;
      return;
    }

    const pad = Math.max(spacing, 0.5);
    this.cellSize = Math.max(spacing * 2.5, 1e-3);
    this.originX = minX - pad;
    this.originY = minY - pad;
    const width = maxX - minX + pad * 2;
    const height = maxY - minY + pad * 2;
    this.cols = Math.max(1, Math.ceil(width / this.cellSize) + 1);
    this.rows = Math.max(1, Math.ceil(height / this.cellSize) + 1);
    const nCells = this.cols * this.rows;

    if (this.heads.length < nCells) this.heads = new Int32Array(Math.max(nCells, this.heads.length * 2 || nCells));
    for (let c = 0; c < nCells; c++) this.heads[c] = -1;

    this.entryCount = 0;

    for (let s = 0; s < nSeg; s++) {
      const beamId = this.beamIds[s]!;
      const ai = beams.a[beamId]!;
      const bj = beams.b[beamId]!;
      const ax = nodes.x[ai]!;
      const ay = nodes.y[ai]!;
      const bx = nodes.x[bj]!;
      const by = nodes.y[bj]!;
      const x0 = Math.min(ax, bx);
      const y0 = Math.min(ay, by);
      const x1 = Math.max(ax, bx);
      const y1 = Math.max(ay, by);
      const c0 = this.clampCol(((x0 - this.originX) / this.cellSize) | 0);
      const c1 = this.clampCol(((x1 - this.originX) / this.cellSize) | 0);
      const r0 = this.clampRow(((y0 - this.originY) / this.cellSize) | 0);
      const r1 = this.clampRow(((y1 - this.originY) / this.cellSize) | 0);
      for (let r = r0; r <= r1; r++) {
        for (let c = c0; c <= c1; c++) {
          if (this.entryCount >= this.entryBeam.length) {
            const cap = Math.max(this.entryCount * 2, 256);
            const nb = new Int32Array(cap);
            const nn = new Int32Array(cap);
            nb.set(this.entryBeam);
            nn.set(this.entryNext);
            this.entryBeam = nb;
            this.entryNext = nn;
          }
          const cell = r * this.cols + c;
          const e = this.entryCount++;
          this.entryBeam[e] = beamId;
          this.entryNext[e] = this.heads[cell]!;
          this.heads[cell] = e;
        }
      }
    }

    this.dirty = false;
  }

  private clampCol(c: number): number {
    if (c < 0) return 0;
    if (c >= this.cols) return this.cols - 1;
    return c;
  }

  private clampRow(r: number): number {
    if (r < 0) return 0;
    if (r >= this.rows) return this.rows - 1;
    return r;
  }

  /**
   * Resolve particle ↔ hull contacts. Mutates water + ship positions/velocities.
   * Call after both systems have integrated for this substep.
   */
  collide(water: WaterState, ship: Ship, dt: number): void {
    if (water.count === 0 || ship.nodes.count === 0) return;

    const spacing = Math.max(water.spacing, 1e-4);
    if (this.dirty) this.rebuild(ship, spacing);

    const pR = spacing * PARTICLE_RADIUS_FACTOR;
    const nR = spacing * NODE_RADIUS_FACTOR;
    const contactR = pR + Math.max(spacing * 0.15, 0.05);
    const maxCorr = spacing * MAX_CORR;
    const invDt = dt > 1e-8 ? 1 / dt : 0;

    if (this.beamIds.length > 0 && this.cols > 0) {
      this.collideSegments(water, ship, contactR, maxCorr, invDt);
    }

    if (ENABLE_NODE_COLLIDE) {
      this.collideNodes(water, ship, pR + nR, maxCorr, invDt);
    }
  }

  private collideSegments(
    water: WaterState,
    ship: Ship,
    contactR: number,
    maxCorr: number,
    invDt: number,
  ): void {
    const { beams, nodes } = ship;
    const r2 = contactR * contactR;
    const queryPad = contactR + this.cellSize;

    for (let i = 0; i < water.count; i++) {
      if (water.invMass[i] === 0) continue;
      const px = water.x[i]!;
      const py = water.y[i]!;

      const c0 = this.clampCol(((px - queryPad - this.originX) / this.cellSize) | 0);
      const c1 = this.clampCol(((px + queryPad - this.originX) / this.cellSize) | 0);
      const r0 = this.clampRow(((py - queryPad - this.originY) / this.cellSize) | 0);
      const r1 = this.clampRow(((py + queryPad - this.originY) / this.cellSize) | 0);

      let bestD2 = r2;
      let bestSeg = -1;
      let bestT = 0;
      let bestQx = 0;
      let bestQy = 0;

      // Track visited beams this particle (small local set via last-seen stamp)
      for (let row = r0; row <= r1; row++) {
        for (let col = c0; col <= c1; col++) {
          let e = this.heads[row * this.cols + col]!;
          while (e >= 0) {
            const beamId = this.entryBeam[e]!;
            e = this.entryNext[e]!;
            if (!beams.alive[beamId] || (beams.flags[beamId]! & BEAM_SKIN) === 0) continue;

            const ai = beams.a[beamId]!;
            const bi = beams.b[beamId]!;
            const ax = nodes.x[ai]!;
            const ay = nodes.y[ai]!;
            const bx = nodes.x[bi]!;
            const by = nodes.y[bi]!;
            const abx = bx - ax;
            const aby = by - ay;
            const apx = px - ax;
            const apy = py - ay;
            const abLen2 = abx * abx + aby * aby;
            let t = abLen2 > 1e-12 ? (apx * abx + apy * aby) / abLen2 : 0;
            if (t < 0) t = 0;
            else if (t > 1) t = 1;
            const qx = ax + abx * t;
            const qy = ay + aby * t;
            const dx = px - qx;
            const dy = py - qy;
            const d2 = dx * dx + dy * dy;
            if (d2 < bestD2) {
              bestD2 = d2;
              bestSeg = beamId;
              bestT = t;
              bestQx = qx;
              bestQy = qy;
            }
          }
        }
      }

      if (bestSeg < 0) continue;

      const dist = Math.sqrt(Math.max(bestD2, 0));
      let nx: number;
      let ny: number;
      if (dist > 1e-8) {
        nx = (px - bestQx) / dist;
        ny = (py - bestQy) / dist;
      } else {
        const ai = beams.a[bestSeg]!;
        const bi = beams.b[bestSeg]!;
        const abx = nodes.x[bi]! - nodes.x[ai]!;
        const aby = nodes.y[bi]! - nodes.y[ai]!;
        const len = Math.hypot(abx, aby) || 1;
        nx = -aby / len;
        ny = abx / len;
      }

      const penetration = contactR - dist;
      if (penetration <= 0) continue;
      const corr = Math.min(penetration, maxCorr);

      water.x[i]! += nx * corr;
      water.y[i]! += ny * corr;
      water.px[i]! = water.x[i]!;
      water.py[i]! = water.y[i]!;

      const ai = beams.a[bestSeg]!;
      const bi = beams.b[bestSeg]!;
      const wA = 1 - bestT;
      const wB = bestT;

      const svx = nodes.vx[ai]! * wA + nodes.vx[bi]! * wB;
      const svy = nodes.vy[ai]! * wA + nodes.vy[bi]! * wB;
      const rvx = water.vx[i]! - svx;
      const rvy = water.vy[i]! - svy;
      const vn = rvx * nx + rvy * ny;

      if (vn > SLOP) continue;

      const pInv = water.invMass[i]!;
      const aInv = (nodes.flags[ai]! & FLAG_PINNED) !== 0 ? 0 : nodes.invMass[ai]!;
      const bInv = (nodes.flags[bi]! & FLAG_PINNED) !== 0 ? 0 : nodes.invMass[bi]!;
      const shipInv = aInv * wA * wA + bInv * wB * wB;
      const invSum = pInv + shipInv;
      if (invSum < 1e-12) continue;

      let jn = (-(1 + RESTITUTION) * vn) / invSum;
      jn += (corr * invDt * 0.35) / invSum;
      const jAbs = Math.abs(jn);
      if (jAbs > MAX_IMPULSE) jn *= MAX_IMPULSE / jAbs;

      water.vx[i]! += jn * nx * pInv;
      water.vy[i]! += jn * ny * pInv;

      this.applyEndpointImpulse(
        nodes,
        ai,
        bi,
        wA,
        wB,
        -jn * nx * SHIP_FEEDBACK,
        -jn * ny * SHIP_FEEDBACK,
      );

      // Soft positional reaction on hull
      const shipShare = pInv / invSum;
      const sc = corr * shipShare * 0.4 * SHIP_FEEDBACK;
      if (aInv > 0) {
        nodes.x[ai]! -= nx * sc * wA;
        nodes.y[ai]! -= ny * sc * wA;
        nodes.px[ai]! = nodes.x[ai]!;
        nodes.py[ai]! = nodes.y[ai]!;
      }
      if (bInv > 0) {
        nodes.x[bi]! -= nx * sc * wB;
        nodes.y[bi]! -= ny * sc * wB;
        nodes.px[bi]! = nodes.x[bi]!;
        nodes.py[bi]! = nodes.y[bi]!;
      }
    }
  }

  private collideNodes(
    water: WaterState,
    ship: Ship,
    contactR: number,
    maxCorr: number,
    invDt: number,
  ): void {
    const { nodes } = ship;
    const r2 = contactR * contactR;
    const spacing = Math.max(water.spacing, 1e-4);
    const cell = Math.max(contactR * 2, spacing * 2);
    const useHash = water.count > 4000 && nodes.count > 20;

    if (!useHash) {
      for (let i = 0; i < water.count; i++) {
        if (water.invMass[i] === 0) continue;
        for (let n = 0; n < nodes.count; n++) {
          this.resolveCircle(water, ship, i, n, contactR, r2, maxCorr, invDt);
        }
      }
      return;
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (let i = 0; i < water.count; i++) {
      const x = water.x[i]!;
      const y = water.y[i]!;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
    }
    if (!Number.isFinite(minX)) return;

    const cols = Math.max(1, Math.ceil((maxX - minX) / cell) + 1);
    const rows = Math.max(1, Math.ceil((maxY - minY) / cell) + 1);
    const nCells = cols * rows;
    const heads = new Int32Array(nCells).fill(-1);
    const next = new Int32Array(water.count);

    for (let i = 0; i < water.count; i++) {
      const cx = Math.min(cols - 1, Math.max(0, ((water.x[i]! - minX) / cell) | 0));
      const cy = Math.min(rows - 1, Math.max(0, ((water.y[i]! - minY) / cell) | 0));
      const c = cy * cols + cx;
      next[i] = heads[c]!;
      heads[c] = i;
    }

    const reach = Math.ceil(contactR / cell) + 1;
    for (let n = 0; n < nodes.count; n++) {
      const nx = nodes.x[n]!;
      const ny = nodes.y[n]!;
      const cx = Math.min(cols - 1, Math.max(0, ((nx - minX) / cell) | 0));
      const cy = Math.min(rows - 1, Math.max(0, ((ny - minY) / cell) | 0));
      for (let dy = -reach; dy <= reach; dy++) {
        const ry = cy + dy;
        if (ry < 0 || ry >= rows) continue;
        for (let dx = -reach; dx <= reach; dx++) {
          const rx = cx + dx;
          if (rx < 0 || rx >= cols) continue;
          let i = heads[ry * cols + rx]!;
          while (i >= 0) {
            const ni = next[i]!;
            if (water.invMass[i] !== 0) {
              this.resolveCircle(water, ship, i, n, contactR, r2, maxCorr, invDt);
            }
            i = ni;
          }
        }
      }
    }
  }

  private resolveCircle(
    water: WaterState,
    ship: Ship,
    i: number,
    n: number,
    contactR: number,
    r2: number,
    maxCorr: number,
    invDt: number,
  ): void {
    const nodes = ship.nodes;
    const dx = water.x[i]! - nodes.x[n]!;
    const dy = water.y[i]! - nodes.y[n]!;
    const d2 = dx * dx + dy * dy;
    if (d2 >= r2) return;

    let nx: number;
    let ny: number;
    let dist: number;
    if (d2 < 1e-14) {
      nx = 0;
      ny = -1;
      dist = 0;
    } else {
      dist = Math.sqrt(d2);
      nx = dx / dist;
      ny = dy / dist;
    }

    const penetration = contactR - dist;
    if (penetration <= 0) return;
    const corr = Math.min(penetration, maxCorr);

    water.x[i]! += nx * corr;
    water.y[i]! += ny * corr;
    water.px[i]! = water.x[i]!;
    water.py[i]! = water.y[i]!;

    const pinned = (nodes.flags[n]! & FLAG_PINNED) !== 0;
    const nInv = pinned ? 0 : nodes.invMass[n]!;
    const pInv = water.invMass[i]!;
    const invSum = pInv + nInv;
    if (invSum < 1e-12) return;

    const rvx = water.vx[i]! - nodes.vx[n]!;
    const rvy = water.vy[i]! - nodes.vy[n]!;
    const vn = rvx * nx + rvy * ny;
    if (vn > SLOP) return;

    let jn = (-(1 + RESTITUTION) * vn) / invSum;
    jn += (corr * invDt * 0.3) / invSum;
    const jAbs = Math.abs(jn);
    if (jAbs > MAX_IMPULSE) jn *= MAX_IMPULSE / jAbs;

    water.vx[i]! += jn * nx * pInv;
    water.vy[i]! += jn * ny * pInv;
    if (nInv > 0) {
      nodes.vx[n]! -= jn * nx * nInv;
      nodes.vy[n]! -= jn * ny * nInv;
      const shipCorr = corr * (pInv / invSum);
      nodes.x[n]! -= nx * shipCorr * 0.5;
      nodes.y[n]! -= ny * shipCorr * 0.5;
      nodes.px[n]! = nodes.x[n]!;
      nodes.py[n]! = nodes.y[n]!;
    }
  }

  private applyEndpointImpulse(
    nodes: Ship['nodes'],
    ai: number,
    bi: number,
    wA: number,
    wB: number,
    ix: number,
    iy: number,
  ): void {
    if ((nodes.flags[ai]! & FLAG_PINNED) === 0 && nodes.invMass[ai]! > 0) {
      const s = nodes.invMass[ai]! * wA;
      nodes.vx[ai]! += ix * s;
      nodes.vy[ai]! += iy * s;
    }
    if ((nodes.flags[bi]! & FLAG_PINNED) === 0 && nodes.invMass[bi]! > 0) {
      const s = nodes.invMass[bi]! * wB;
      nodes.vx[bi]! += ix * s;
      nodes.vy[bi]! += iy * s;
    }
  }
}
