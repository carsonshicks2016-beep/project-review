import {
  BEAM_SKIN,
  FLAG_FLOODED,
  FLAG_PINNED,
  type Ship,
} from '../types';

/** Flood mass fraction of base mass that sets FLAG_FLOODED */
const FLOOD_FLAG_FRAC = 0.1;
/** Cap flood mass as multiple of structural mass (full compartment) — enough to sink when awash */
const FLOOD_CAP_FRAC = 2.2;
/** Base fill rate (mass / s) per submerged breach, scaled by spacing proxy */
const BASE_FILL_RATE = 24;
/** Extra rate per additional broken skin beam touching the compartment */
const PER_BREACH_RATE = 16;
/** Minimum submerged depth (world units) for a breach to admit water */
const MIN_BREACH_DEPTH = 0.35;
/** Drain rate when breaches are above water / repaired (slow) */
const DRAIN_RATE = 1.5;

export interface FloodingResult {
  floodedNodes: number;
  breachCount: number;
}

/**
 * Compartment flooding from broken hull skin.
 * Intact BEAM_SKIN = watertight. Broken skin → breaches on endpoint compartments.
 * floodMass increases effective mass and invMass; FLAG_FLOODED when significant.
 */
export class FloodingSystem {
  /** Scratch: breach count per compartment id */
  private breachByComp: Int32Array = new Int32Array(64);
  /** Scratch: deepest submerged breach depth per compartment */
  private depthByComp: Float64Array = new Float64Array(64);
  private maxCompId = 0;

  /**
   * @param surfaceAt free-surface y sampler (y-down: submerged when y > surface)
   * @returns flooded node count
   */
  step(ship: Ship, surfaceAt: (x: number) => number, dt: number, spacing = 8): FloodingResult {
    const { nodes, beams } = ship;
    if (nodes.count === 0 || dt <= 0) {
      return { floodedNodes: 0, breachCount: 0 };
    }

    // Discover max compartment id
    let maxC = 0;
    for (let i = 0; i < nodes.count; i++) {
      const c = nodes.compartment[i]!;
      if (c > maxC) maxC = c;
    }
    this.maxCompId = maxC;
    const need = maxC + 1;
    if (this.breachByComp.length < need) {
      this.breachByComp = new Int32Array(need);
      this.depthByComp = new Float64Array(need);
    } else {
      this.breachByComp.fill(0, 0, need);
      this.depthByComp.fill(0, 0, need);
    }

    let breachCount = 0;

    // Broken skin beams → breaches at endpoints' compartments
    for (let i = 0; i < beams.count; i++) {
      if (beams.alive[i]) continue;
      if ((beams.flags[i]! & BEAM_SKIN) === 0) continue;

      const ai = beams.a[i]!;
      const bi = beams.b[i]!;
      if (ai >= nodes.count || bi >= nodes.count) continue;

      const mx = (nodes.x[ai]! + nodes.x[bi]!) * 0.5;
      const my = (nodes.y[ai]! + nodes.y[bi]!) * 0.5;
      const surface = surfaceAt(mx);
      let depth = 0;
      if (Number.isFinite(surface)) {
        depth = my - surface; // y-down
      }
      if (depth < MIN_BREACH_DEPTH) continue;

      breachCount++;
      this.markBreach(nodes.compartment[ai]!, depth);
      this.markBreach(nodes.compartment[bi]!, depth);
    }

    const fillScale = Math.max(spacing * spacing * 0.02, 0.5);
    let floodedNodes = 0;

    for (let n = 0; n < nodes.count; n++) {
      const comp = nodes.compartment[n]!;
      const baseMass = Math.max(nodes.mass[n]!, 1e-6);
      const cap = baseMass * FLOOD_CAP_FRAC;
      let flood = nodes.floodMass[n]!;

      if (comp > 0) {
        const breaches = this.breachByComp[comp]!;
        const depth = this.depthByComp[comp]!;
        if (breaches > 0 && depth > 0) {
          // Small hole = slow; many broken skin beams = fast
          const rate =
            (BASE_FILL_RATE + PER_BREACH_RATE * (breaches - 1)) *
            fillScale *
            Math.min(2.5, 0.35 + depth / Math.max(spacing, 1));
          flood = Math.min(cap, flood + rate * dt);
        } else if (flood > 0 && breaches === 0) {
          // Intact again or above water — slow drain / pump-out
          flood = Math.max(0, flood - DRAIN_RATE * fillScale * dt);
        }
      } else {
        // Exterior / debris: no compartment flooding, but clear residual
        // (exterior water interaction is via particles + buoyancy)
      }

      nodes.floodMass[n] = flood;

      // Effective mass + invMass
      const pinned = (nodes.flags[n]! & FLAG_PINNED) !== 0;
      if (pinned) {
        nodes.invMass[n] = 0;
      } else {
        const m = baseMass + flood;
        nodes.invMass[n] = m > 1e-8 ? 1 / m : 0;
      }

      // FLAG_FLOODED
      let flags = nodes.flags[n]!;
      if (flood >= baseMass * FLOOD_FLAG_FRAC) {
        flags |= FLAG_FLOODED;
        floodedNodes++;
      } else {
        flags &= ~FLAG_FLOODED;
      }
      nodes.flags[n] = flags;
    }

    return { floodedNodes, breachCount };
  }

  private markBreach(comp: number, depth: number): void {
    if (comp <= 0) return;
    this.breachByComp[comp]!++;
    if (depth > this.depthByComp[comp]!) {
      this.depthByComp[comp] = depth;
    }
  }
}
