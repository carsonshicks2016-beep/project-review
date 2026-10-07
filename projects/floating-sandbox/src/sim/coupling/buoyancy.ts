import {
  FLAG_PINNED,
  type Ship,
  type TankBounds,
  type WaterState,
} from '../types';

/** Water density relative to material densities (~1) */
const WATER_DENSITY = 1.0;
/**
 * Buoyancy gain: upward accel ≈ −VOLUME_SCALE · subFrac · g · (baseMass / totalMass).
 * Mass-proportional so the lattice rises together. Tuned for ~½–⅔ draft.
 * Must allow peak accel > g when fully submerged (see MAX_ACCEL_G).
 */
const VOLUME_SCALE = 2.15;
/** Linear drag as acceleration damping in water */
const DRAG = 4.0;
/**
 * Clamp |buoyancy+drag accel| to this multiple of |g|.
 * Must be > 1 so submerged nodes can cancel gravity and restore upward.
 */
const MAX_ACCEL_G = 2.2;
/** Gravity magnitude used when caller doesn't pass one (matches world default) */
const DEFAULT_G = 900;
/** Overlap radius for particle proximity submerged test */
const OVERLAP_FACTOR = 1.05;
/** Max bins for free-surface sampling */
const MAX_BINS = 256;

/**
 * Hybrid buoyancy:
 * - Column free-surface from water particles (with rest-surface fallback)
 * - Mass-proportional lift so the lattice accelerates together
 *
 * y-down: surface = min particle y in column; submerged when node.y > surface.
 * floodMass reduces upward gain (baseMass/totalMass) so breached ships sink.
 */
export class BuoyancySolver {
  private binCount = 0;
  private binMinX = 0;
  private binWidth = 1;
  /** Free-surface y per bin (y-down: top of water). +Infinity = no water */
  private surfaceY: Float64Array = new Float64Array(0);
  private binFill: Uint8Array = new Uint8Array(0);
  private gravity = DEFAULT_G;
  /** Shallowest free-surface seen (survives particle-ocean collapse) */
  private oceanSurface = Number.NaN;

  setGravity(g: number): void {
    this.gravity = Math.abs(g) || DEFAULT_G;
  }

  /** Call when the ocean is refilled so rest-surface tracking resets. */
  resetOcean(): void {
    this.oceanSurface = Number.NaN;
  }

  /**
   * Sample approximate free-surface height from water particles in x-bins.
   * Ignores deep stray particles in hull voids so the ship doesn't "see"
   * the seabed as the waterline.
   */
  sampleSurface(water: WaterState, bounds: TankBounds): void {
    const width = Math.max(bounds.x1 - bounds.x0, 1e-3);
    const spacing = Math.max(water.spacing, 1e-3);
    const desired = Math.ceil(width / Math.max(spacing * 2, 1));
    this.binCount = Math.max(8, Math.min(MAX_BINS, desired));
    this.binMinX = bounds.x0;
    this.binWidth = width / this.binCount;

    if (this.surfaceY.length < this.binCount) {
      this.surfaceY = new Float64Array(this.binCount);
      this.binFill = new Uint8Array(this.binCount);
    }

    let globalSurf = Infinity;
    for (let i = 0; i < water.count; i++) {
      const y = water.y[i]!;
      if (y < globalSurf) globalSurf = y;
    }

    // Remember a healthy ocean top; if the particle fluid collapses to the
    // seabed, keep using the rest surface so ships can still float.
    const floorY = bounds.y1;
    if (
      Number.isFinite(globalSurf) &&
      globalSurf < floorY - spacing * 6 &&
      (!Number.isFinite(this.oceanSurface) || globalSurf < this.oceanSurface)
    ) {
      this.oceanSurface = globalSurf;
    }
    let usableSurf = globalSurf;
    if (
      Number.isFinite(this.oceanSurface) &&
      (!Number.isFinite(globalSurf) || globalSurf > this.oceanSurface + spacing * 5)
    ) {
      usableSurf = this.oceanSurface;
    }

    for (let b = 0; b < this.binCount; b++) {
      this.surfaceY[b] = Infinity;
      this.binFill[b] = 0;
    }

    // Only particles near the usable free surface count (skip void-floor leftovers)
    const band = spacing * 10;
    const deepCut = Number.isFinite(usableSurf) ? usableSurf + band : Infinity;

    for (let i = 0; i < water.count; i++) {
      const x = water.x[i]!;
      const y = water.y[i]!;
      if (y > deepCut) continue;
      let b = ((x - this.binMinX) / this.binWidth) | 0;
      if (b < 0) b = 0;
      else if (b >= this.binCount) b = this.binCount - 1;
      if (y < this.surfaceY[b]!) {
        this.surfaceY[b] = y;
        this.binFill[b] = 1;
      }
    }

    // Fill empty bins from neighbors so buoyancy doesn't drop out mid-hull
    this.fillGaps();

    // Any remaining empties → usable ocean surface
    if (Number.isFinite(usableSurf)) {
      for (let b = 0; b < this.binCount; b++) {
        if (!Number.isFinite(this.surfaceY[b]!)) this.surfaceY[b] = usableSurf;
      }
    }
  }

  private fillGaps(): void {
    // Forward
    let last = Infinity;
    for (let b = 0; b < this.binCount; b++) {
      if (this.binFill[b]) last = this.surfaceY[b]!;
      else if (Number.isFinite(last)) this.surfaceY[b] = last;
    }
    // Backward
    last = Infinity;
    for (let b = this.binCount - 1; b >= 0; b--) {
      if (this.binFill[b]) {
        last = this.surfaceY[b]!;
      } else if (Number.isFinite(last)) {
        const cur = this.surfaceY[b]!;
        if (!Number.isFinite(cur)) this.surfaceY[b] = last;
        else this.surfaceY[b] = Math.min(cur, last);
      }
    }
  }

  /** Free-surface y at world x. Returns +Infinity if unknown / empty. */
  surfaceAt(x: number): number {
    if (this.binCount <= 0) return Infinity;
    let b = ((x - this.binMinX) / this.binWidth) | 0;
    if (b < 0) b = 0;
    else if (b >= this.binCount) b = this.binCount - 1;
    return this.surfaceY[b]!;
  }

  /**
   * Apply Archimedes-ish lift + linear drag to submerged ship nodes.
   */
  apply(ship: Ship, water: WaterState, dt: number): void {
    if (ship.nodes.count === 0 || dt <= 0) return;

    const nodes = ship.nodes;
    const spacing = Math.max(water.spacing, 1e-3);
    const nodeSize = spacing * 1.75;
    const g = this.gravity;
    const maxAccel = MAX_ACCEL_G * g;
    const overlapR = spacing * OVERLAP_FACTOR;
    const overlapR2 = overlapR * overlapR;

    // Overlap boost is noisy on dense lattices; free-surface column is enough.
    const markOverlap = false;
    const overlapped = markOverlap ? this.findOverlappedNodes(water, ship, overlapR2) : null;

    for (let n = 0; n < nodes.count; n++) {
      if ((nodes.flags[n]! & FLAG_PINNED) !== 0) continue;
      if (nodes.invMass[n]! <= 0) continue;

      const nx = nodes.x[n]!;
      const ny = nodes.y[n]!;

      const surface = this.surfaceAt(nx);
      let subFrac = 0;

      if (Number.isFinite(surface)) {
        // Submerged depth below free surface (y-down)
        const depth = ny - surface;
        if (depth > 0) {
          subFrac = Math.min(1, depth / nodeSize);
        }
      }

      // Particle overlap can mark local submersion even without column sample
      if (overlapped && overlapped[n]) {
        subFrac = Math.max(subFrac, 0.35);
      }

      if (subFrac <= 1e-4) continue;

      const baseMass = Math.max(nodes.mass[n]!, 1e-6);
      const totalMass = Math.max(baseMass + nodes.floodMass[n]!, 1e-6);
      // Mass-proportional lift → shared accel; flood ballast reduces upward gain
      const buoyGain = WATER_DENSITY * VOLUME_SCALE * (baseMass / totalMass);
      let ax = -DRAG * subFrac * nodes.vx[n]!;
      let ay = -buoyGain * subFrac * g - DRAG * subFrac * nodes.vy[n]!;

      const aMag = Math.hypot(ax, ay);
      if (aMag > maxAccel) {
        const s = maxAccel / aMag;
        ax *= s;
        ay *= s;
      }

      nodes.vx[n]! += ax * dt;
      nodes.vy[n]! += ay * dt;
    }
  }

  /**
   * Mark nodes that overlap at least one water particle (coarse hash).
   */
  private findOverlappedNodes(
    water: WaterState,
    ship: Ship,
    r2: number,
  ): Uint8Array {
    const nodes = ship.nodes;
    const out = new Uint8Array(nodes.count);
    const r = Math.sqrt(r2);
    const cell = Math.max(r * 2, water.spacing * 2);

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
    if (!Number.isFinite(minX)) return out;

    const cols = Math.max(1, Math.ceil((maxX - minX) / cell) + 1);
    const rows = Math.max(1, Math.ceil((maxY - minY) / cell) + 1);
    const heads = new Int32Array(cols * rows).fill(-1);
    const next = new Int32Array(water.count);
    for (let i = 0; i < water.count; i++) {
      const cx = Math.min(cols - 1, Math.max(0, ((water.x[i]! - minX) / cell) | 0));
      const cy = Math.min(rows - 1, Math.max(0, ((water.y[i]! - minY) / cell) | 0));
      const c = cy * cols + cx;
      next[i] = heads[c]!;
      heads[c] = i;
    }

    const reach = 1;
    for (let n = 0; n < nodes.count; n++) {
      const nx = nodes.x[n]!;
      const ny = nodes.y[n]!;
      const cx = Math.min(cols - 1, Math.max(0, ((nx - minX) / cell) | 0));
      const cy = Math.min(rows - 1, Math.max(0, ((ny - minY) / cell) | 0));
      let hit = false;
      for (let dy = -reach; dy <= reach && !hit; dy++) {
        const ry = cy + dy;
        if (ry < 0 || ry >= rows) continue;
        for (let dx = -reach; dx <= reach && !hit; dx++) {
          const rx = cx + dx;
          if (rx < 0 || rx >= cols) continue;
          let i = heads[ry * cols + rx]!;
          while (i >= 0) {
            const ddx = water.x[i]! - nx;
            const ddy = water.y[i]! - ny;
            if (ddx * ddx + ddy * ddy <= r2) {
              hit = true;
              break;
            }
            i = next[i]!;
          }
        }
      }
      if (hit) out[n] = 1;
    }
    return out;
  }
}
