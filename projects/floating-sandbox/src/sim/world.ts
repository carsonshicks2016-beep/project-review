import type { Ship, SimConfig, SimStats, TankBounds, WaterState } from './types';
import { createWaterState, fillRect, clearWater } from './water/water-state';
import { PbfCpuSolver, type PbfCpuParams } from './water/pbf-cpu';
import { CouplingSystem } from './coupling/coupling';
import { ShipLatticeSolver } from './ship/lattice';

export const DEFAULT_CONFIG: SimConfig = {
  dt: 1 / 60,
  substeps: 1,
  gravity: 900,
  waterViscosity: 0.1,
  surfaceTension: 0.06,
  wind: 0,
  waveAmp: 0,
  particleTarget: 8000,
  /** Prefer CPU for correct coupling; enable GPU for 50k–100k+ demos. */
  useGpu: false,
  stressOverlay: false,
  paused: false,
};

export interface WorldOptions {
  bounds?: TankBounds;
  particleCapacity?: number;
  spacing?: number;
}

/**
 * Fixed-timestep simulation world: water → ship → coupling each substep.
 * GPU water backend is attached later by App when available.
 */
export class World {
  bounds: TankBounds;
  water: WaterState;
  ship: Ship | null = null;
  config: SimConfig;
  readonly lattice = new ShipLatticeSolver();
  readonly coupling: CouplingSystem;
  readonly cpuWater: PbfCpuSolver;

  /** Optional GPU stepper injected by App (must download into water before return). */
  gpuStep: ((params: PbfCpuParams & { wind?: number }, water: WaterState) => Promise<void>) | null =
    null;
  gpuUpload: ((w: WaterState) => void) | null = null;
  backend: 'cpu' | 'webgpu' = 'cpu';

  private accumulator = 0;
  private stats: SimStats = {
    waterCount: 0,
    nodeCount: 0,
    beamCount: 0,
    brokenBeams: 0,
    floodedNodes: 0,
    fps: 0,
    backend: 'cpu',
  };
  private frames = 0;
  private fpsTimer = 0;

  constructor(opts: WorldOptions = {}) {
    this.bounds = opts.bounds ?? { x0: 0, y0: 0, x1: 1200, y1: 700 };
    const spacing = opts.spacing ?? 8;
    const cap = opts.particleCapacity ?? 140000;
    this.water = createWaterState(cap, spacing);
    this.config = { ...DEFAULT_CONFIG };
    this.cpuWater = new PbfCpuSolver(this.bounds);
    this.coupling = new CouplingSystem(this.bounds);
    this.fillOcean(0.42);
  }

  fillOcean(depthFrac: number): void {
    clearWater(this.water);
    const { x0, y0, x1, y1 } = this.bounds;
    const h = (y1 - y0) * depthFrac;
    const target = Math.min(this.config.particleTarget, this.water.capacity);
    // Temporarily limit fill by capacity; fillRect stops when full
    const oldCap = this.water.capacity;
    // Fill from seabed up
    fillRect(this.water, x0 + 4, y1 - h, x1 - 4, y1 - 4, this.water.spacing);
    // If over target, trim by reducing count (keep bottom particles)
    if (this.water.count > target) {
      this.water.count = target;
    }
    void oldCap;
    this.coupling.resetOcean();
    this.gpuUpload?.(this.water);
  }

  setShip(ship: Ship | null): void {
    this.ship = ship;
    this.coupling.invalidateShip();
  }

  reset(keepShip = false): void {
    this.fillOcean(0.45);
    if (!keepShip) this.setShip(null);
    this.accumulator = 0;
  }

  getStats(): SimStats {
    return this.stats;
  }

  /** Advance by real frame dt with fixed inner step */
  async update(frameDt: number): Promise<void> {
    this.frames++;
    this.fpsTimer += frameDt;
    if (this.fpsTimer >= 0.5) {
      this.stats.fps = this.frames / this.fpsTimer;
      this.frames = 0;
      this.fpsTimer = 0;
    }

    if (this.config.paused) {
      this.refreshStats();
      return;
    }

    const maxFrame = 0.05;
    let dt = Math.min(frameDt, maxFrame);
    this.accumulator += dt;
    const step = this.config.dt;
    let guard = 0;
    while (this.accumulator >= step && guard++ < 5) {
      await this.fixedStep(step);
      this.accumulator -= step;
    }
    this.refreshStats();
  }

  /** Single paused step */
  async stepOnce(): Promise<void> {
    await this.fixedStep(this.config.dt);
    this.refreshStats();
  }

  private async fixedStep(dt: number): Promise<void> {
    const sub = Math.max(1, this.config.substeps);
    const h = dt / sub;
    const waterParams: PbfCpuParams & { wind?: number } = {
      gravity: this.config.gravity,
      viscosity: this.config.waterViscosity,
      surfaceTension: this.config.surfaceTension,
      iterations: this.backend === 'webgpu' ? 3 : 1,
      dt: h,
      wind: this.config.wind,
    };

    for (let s = 0; s < sub; s++) {
      if (this.backend === 'webgpu' && this.gpuStep) {
        await this.gpuStep(waterParams, this.water);
      } else {
        this.cpuWater.setBounds(this.bounds);
        this.cpuWater.step(this.water, waterParams);
      }

      if (this.ship) {
        this.lattice.step(this.ship, h, this.config.gravity, 8);
        this.coupling.setGravity(this.config.gravity);
        this.coupling.step(this.water, this.ship, h);
      }
    }
  }

  private refreshStats(): void {
    let broken = 0;
    let nodes = 0;
    let beams = 0;
    if (this.ship) {
      nodes = this.ship.nodes.count;
      beams = this.ship.beams.count;
      for (let i = 0; i < beams; i++) {
        if (!this.ship.beams.alive[i]) broken++;
      }
    }
    this.stats = {
      waterCount: this.water.count,
      nodeCount: nodes,
      beamCount: beams,
      brokenBeams: broken,
      floodedNodes: this.coupling.getFloodedCount(),
      fps: this.stats.fps,
      backend: this.backend,
    };
  }
}
