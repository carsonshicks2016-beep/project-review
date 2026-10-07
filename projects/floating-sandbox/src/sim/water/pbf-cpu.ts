/**
 * Position-Based Fluids (Macklin & Müller) — 2D CPU solver.
 * Poly6 density + Spiky gradient; artificial pressure for mild cohesion; XSPH viscosity.
 */

import type { TankBounds, WaterState } from '../types';
import { SpatialHash } from './spatial-hash';

export interface PbfCpuParams {
  gravity: number;
  viscosity: number;
  surfaceTension: number;
  /** Density-constraint iterations (default 3). */
  iterations: number;
  dt: number;
  /** Optional mild horizontal wind acceleration (default 0). */
  wind?: number;
}

/** Cap neighbours per particle (2D packing with h≈2.2·s is ~15–20). */
const MAX_NEIGHBORS = 32;

/** Poly6 W from r². */
function poly6(r2: number, h2: number, coeff: number): number {
  if (r2 >= h2) return 0;
  const d = h2 - r2;
  return coeff * d * d * d;
}

export class PbfCpuSolver {
  private bounds: TankBounds;
  private hash: SpatialHash;

  private lambda = new Float32Array(0);
  private dpx = new Float32Array(0);
  private dpy = new Float32Array(0);

  /** CSR neighbour lists: offsets[i] .. offsets[i]+counts[i]. */
  private neighOffsets = new Int32Array(0);
  private neighCounts = new Uint8Array(0);
  private neighJ = new Int32Array(0);
  private neighDx = new Float32Array(0);
  private neighDy = new Float32Array(0);
  private neighD2 = new Float32Array(0);

  constructor(bounds: TankBounds) {
    this.bounds = { ...bounds };
    this.hash = new SpatialHash(16384);
  }

  setBounds(b: TankBounds): void {
    this.bounds = { ...b };
  }

  private ensureScratch(n: number): void {
    if (n > this.lambda.length) {
      let cap = this.lambda.length || 1024;
      while (cap < n) cap <<= 1;
      this.lambda = new Float32Array(cap);
      this.dpx = new Float32Array(cap);
      this.dpy = new Float32Array(cap);
      this.neighOffsets = new Int32Array(cap);
      this.neighCounts = new Uint8Array(cap);
    }

    const needSlots = n * MAX_NEIGHBORS;
    if (needSlots > this.neighJ.length) {
      let slots = this.neighJ.length || MAX_NEIGHBORS * 1024;
      while (slots < needSlots) slots <<= 1;
      this.neighJ = new Int32Array(slots);
      this.neighDx = new Float32Array(slots);
      this.neighDy = new Float32Array(slots);
      this.neighD2 = new Float32Array(slots);
    }
  }

  private rebuildNeighborLists(
    px: Float32Array,
    py: Float32Array,
    n: number,
    h: number,
  ): void {
    this.hash.rebuild(px, py, n, h);
    const offsets = this.neighOffsets;
    const counts = this.neighCounts;
    let cursor = 0;
    for (let i = 0; i < n; i++) {
      offsets[i] = cursor;
      const c = this.hash.gatherNeighbors(
        i,
        px,
        py,
        h,
        this.neighJ,
        this.neighDx,
        this.neighDy,
        this.neighD2,
        cursor,
        MAX_NEIGHBORS,
      );
      counts[i] = c;
      cursor += c;
    }
  }

  step(water: WaterState, params: PbfCpuParams): void {
    const n = water.count;
    if (n === 0) return;

    const dt = params.dt;
    if (dt <= 0) return;

    const iters = params.iterations > 0 ? params.iterations | 0 : 3;
    const gAccel = params.gravity;
    const wind = params.wind ?? 0;
    const visc = Math.max(0, Math.min(1, params.viscosity));
    // Artificial pressure strength (Macklin tensile instability term)
    const kCorr = Math.max(0, params.surfaceTension) * 0.004;
    const dqRel = 0.3;

    const spacing = water.spacing;
    const h = spacing * 2;
    const h2 = h * h;
    const rho0 = water.restDensity;
    // Soft constraint ε in |∇C|²-space (Σ|∇C|² is typically O(10–100) here)
    const epsGrad = 1;
    const minDist = spacing * 0.1;
    const minDist2 = minDist * minDist;
    const omega = 0.5; // under-relaxation

    const h4 = h2 * h2;
    const poly6Coeff = 4 / (Math.PI * h4 * h4);
    const spikyBase = -30 / (Math.PI * h2 * h2 * h);
    const wDq = poly6(dqRel * dqRel * h2, h2, poly6Coeff);
    const invRho0 = 1 / rho0;
    const rho0Sq = rho0 * rho0;
    const poly6Self = poly6Coeff * h2 * h2 * h2; // W(0)

    const { x, y, px, py, vx, vy, invMass } = water;
    const { x0, y0, x1, y1 } = this.bounds;

    this.ensureScratch(n);
    this.hash.ensureCapacity(n);
    const lambda = this.lambda;
    const dpx = this.dpx;
    const dpy = this.dpy;
    const offsets = this.neighOffsets;
    const counts = this.neighCounts;
    const nJ = this.neighJ;
    const nDx = this.neighDx;
    const nDy = this.neighDy;
    const nD2 = this.neighD2;

    // 1. Gravity (+ wind) and predict positions
    for (let i = 0; i < n; i++) {
      if (invMass[i] === 0) {
        px[i] = x[i];
        py[i] = y[i];
        continue;
      }
      vx[i] += wind * dt;
      vy[i] += gAccel * dt;
      px[i] = x[i] + vx[i] * dt;
      py[i] = y[i] + vy[i] * dt;
    }

    // 2–3. Neighbour hash + density-constraint iterations
    for (let iter = 0; iter < iters; iter++) {
      this.rebuildNeighborLists(px, py, n, h);

      // Densities + λ
      for (let i = 0; i < n; i++) {
        if (invMass[i] === 0) {
          lambda[i] = 0;
          continue;
        }

        let rho = poly6Self;
        let sumGradSq = 0;
        let gradIx = 0;
        let gradIy = 0;

        const base = offsets[i];
        const c = counts[i];
        for (let k = 0; k < c; k++) {
          const slot = base + k;
          const d2raw = nD2[slot];
          const dx = nDx[slot]; // xⱼ − xᵢ
          const dy = nDy[slot];

          const diff = h2 - d2raw;
          rho += poly6Coeff * diff * diff * diff;

          // Spiky ∇W(pᵢ − pⱼ); clamp near-zero pairs so they still repel
          let rx = -dx; // pᵢ − pⱼ
          let ry = -dy;
          let r2 = d2raw;
          if (r2 < minDist2) {
            if (r2 < 1e-20) {
              // deterministic push axis from index
              rx = minDist;
              ry = 0;
            } else {
              const scale = minDist / Math.sqrt(r2);
              rx *= scale;
              ry *= scale;
            }
            r2 = minDist2;
          }
          const r = Math.sqrt(r2);
          const t = h - r;
          const s = (spikyBase * t * t) / r;
          const gx = s * rx;
          const gy = s * ry;

          // Stash ∇W for the delta pass
          nDx[slot] = gx;
          nDy[slot] = gy;

          gradIx += gx;
          gradIy += gy;
          sumGradSq += gx * gx + gy * gy;
        }

        sumGradSq += gradIx * gradIx + gradIy * gradIy;
        const C = rho * invRho0 - 1;
        // λ = −C / (Σ|∇C|² + ε),  ∇C = ∇W/ρ₀
        lambda[i] = -C / (sumGradSq / rho0Sq + epsGrad);
      }

      // Position deltas
      dpx.fill(0, 0, n);
      dpy.fill(0, 0, n);

      for (let i = 0; i < n; i++) {
        if (invMass[i] === 0) continue;
        const li = lambda[i];
        const base = offsets[i];
        const c = counts[i];

        for (let k = 0; k < c; k++) {
          const slot = base + k;
          const j = nJ[slot];
          const d2 = nD2[slot];
          const gx = nDx[slot];
          const gy = nDy[slot];

          let scorr = 0;
          if (kCorr > 0 && wDq > 0 && d2 < h2) {
            const diff = h2 - d2;
            const w = poly6Coeff * diff * diff * diff;
            const ratio = w / wDq;
            const r2w = ratio * ratio;
            scorr = -kCorr * r2w * r2w; // ^4
          }

          const coeff = (li + lambda[j] + scorr) * invRho0;
          dpx[i] += coeff * gx;
          dpy[i] += coeff * gy;
        }
      }

      // Apply under-relaxed corrections + soft wall projection
      for (let i = 0; i < n; i++) {
        if (invMass[i] === 0) continue;
        px[i] += dpx[i] * omega;
        py[i] += dpy[i] * omega;
      }
      this.collideBounds(px, py, vx, vy, invMass, n, x0, y0, x1, y1, false);
    }

    // 4. Velocities from position delta
    const invDt = 1 / dt;
    for (let i = 0; i < n; i++) {
      if (invMass[i] === 0) {
        vx[i] = 0;
        vy[i] = 0;
        continue;
      }
      vx[i] = (px[i] - x[i]) * invDt;
      vy[i] = (py[i] - y[i]) * invDt;
    }

    // 5. XSPH viscosity
    if (visc > 0) {
      if (iters !== 1) this.rebuildNeighborLists(px, py, n, h);

      dpx.fill(0, 0, n);
      dpy.fill(0, 0, n);

      for (let i = 0; i < n; i++) {
        if (invMass[i] === 0) continue;
        const base = offsets[i];
        const c = counts[i];
        let ax = 0;
        let ay = 0;
        const vxi = vx[i];
        const vyi = vy[i];
        for (let k = 0; k < c; k++) {
          const slot = base + k;
          const j = nJ[slot];
          const d2 = nD2[slot];
          if (d2 >= h2) continue;
          const diff = h2 - d2;
          const w = poly6Coeff * diff * diff * diff;
          ax += (vx[j] - vxi) * w;
          ay += (vy[j] - vyi) * w;
        }
        dpx[i] = ax;
        dpy[i] = ay;
      }

      const scale = visc * invRho0;
      for (let i = 0; i < n; i++) {
        if (invMass[i] === 0) continue;
        vx[i] += dpx[i] * scale;
        vy[i] += dpy[i] * scale;
      }
    }

    // 6. Walls (restitution ≈ 0) + commit
    this.collideBounds(px, py, vx, vy, invMass, n, x0, y0, x1, y1, true);
    for (let i = 0; i < n; i++) {
      x[i] = px[i];
      y[i] = py[i];
    }
  }

  private collideBounds(
    px: Float32Array,
    py: Float32Array,
    vx: Float32Array,
    vy: Float32Array,
    invMass: Float32Array,
    n: number,
    x0: number,
    y0: number,
    x1: number,
    y1: number,
    killVelocity: boolean,
  ): void {
    for (let i = 0; i < n; i++) {
      if (invMass[i] === 0) continue;
      if (px[i] < x0) {
        px[i] = x0;
        if (killVelocity && vx[i] < 0) vx[i] = 0;
      } else if (px[i] > x1) {
        px[i] = x1;
        if (killVelocity && vx[i] > 0) vx[i] = 0;
      }
      if (py[i] < y0) {
        py[i] = y0;
        if (killVelocity && vy[i] < 0) vy[i] = 0;
      } else if (py[i] > y1) {
        py[i] = y1;
        if (killVelocity && vy[i] > 0) vy[i] = 0;
      }
    }
  }
}
