import type { Ship, TankBounds, WaterState } from '../types';
import { BuoyancySolver } from './buoyancy';
import { FloodingSystem } from './flooding';
import { HullCollider } from './hull-collision';

/**
 * Fluid ↔ ship coupling orchestrator.
 * Call `step` each substep after water + ship integrate, before render.
 *
 * Order:
 * 1. Sample free surface
 * 2. Hull collision (particles ↔ skin / nodes)
 * 3. Buoyancy + drag on submerged nodes
 * 4. Compartment flooding from broken skin
 */
export class CouplingSystem {
  private bounds: TankBounds;
  private hull = new HullCollider();
  private buoyancy = new BuoyancySolver();
  private flooding = new FloodingSystem();
  private floodedCount = 0;
  private gravity = 900;

  constructor(bounds: TankBounds) {
    this.bounds = { ...bounds };
  }

  setBounds(bounds: TankBounds): void {
    this.bounds = { ...bounds };
  }

  /** Optional: match sim gravity magnitude (y-down, positive). */
  setGravity(g: number): void {
    this.gravity = Math.abs(g) || 900;
    this.buoyancy.setGravity(this.gravity);
  }

  /** Rebuild skin segment acceleration structure (topology or ship swap). */
  invalidateShip(): void {
    this.hull.invalidate();
  }

  /** Call when the ocean is refilled. */
  resetOcean(): void {
    this.buoyancy.resetOcean();
  }

  getFloodedCount(): number {
    return this.floodedCount;
  }

  /**
   * Call each substep after water + ship integrate, before render.
   */
  step(water: WaterState, ship: Ship, dt: number): void {
    if (dt <= 0) return;

    this.hull.invalidate();

    this.buoyancy.setGravity(this.gravity);
    this.buoyancy.sampleSurface(water, this.bounds);

    // Full particle–hull contacts are expensive; always run for small oceans,
    // sample for large particle counts (buoyancy still carries float).
    if (water.count <= 12000) {
      this.hull.collide(water, ship, dt);
    } else if ((this._coupleFrame++ & 1) === 0) {
      this.hull.collide(water, ship, dt);
    }

    this.buoyancy.apply(ship, water, dt);

    const spacing = Math.max(water.spacing, 1e-3);
    const result = this.flooding.step(
      ship,
      (x) => this.buoyancy.surfaceAt(x),
      dt,
      spacing,
    );
    this.floodedCount = result.floodedNodes;
  }

  private _coupleFrame = 0;
}

export { BuoyancySolver } from './buoyancy';
export { FloodingSystem } from './flooding';
export { HullCollider } from './hull-collision';
