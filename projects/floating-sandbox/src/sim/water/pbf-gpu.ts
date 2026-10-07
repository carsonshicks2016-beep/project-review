/**
 * WebGPU Position-Based Fluids solver — same physics intent as PbfCpuSolver.
 * Particle SoA packed into a 32-byte struct buffer; linked-list spatial hash on GPU.
 */

import type { TankBounds, WaterState } from '../types';
import {
  nextPow2,
  PARTICLE_STRIDE_BYTES,
  tryInitWebGPU,
  workgroupCount,
} from '../../gpu/webgpu-device';
import { PBF_SHADER_CODE, PBF_WORKGROUP_SIZE } from './shaders/pbfShaders';

export type { WaterBackend } from './backend';

export interface PbfGpuStepParams {
  gravity: number;
  viscosity: number;
  surfaceTension: number;
  iterations: number;
  dt: number;
  wind?: number;
}

const FLOATS_PER_PARTICLE = 8;
/** SimParams is 5 × 16-byte rows (no flags — soft/kill are separate entry points). */
const UNIFORM_BYTES = 80;

const ENTRY_POINTS = [
  'integrate',
  'clearHash',
  'insertHash',
  'computeLambda',
  'computeDelta',
  'applyDelta',
  'collideBoundsSoft',
  'collideBoundsKill',
  'updateVel',
  'viscosityAccum',
  'viscosityApply',
  'commitPositions',
] as const;

type EntryPoint = (typeof ENTRY_POINTS)[number];

export class PbfGpuSolver {
  private device: GPUDevice;
  private capacity: number;
  private bounds: TankBounds;
  private spacing: number;
  private restDensity: number;
  private count = 0;
  private cellCount: number;
  private destroyed = false;

  private particleBuf!: GPUBuffer;
  private lambdaBuf!: GPUBuffer;
  private deltaXBuf!: GPUBuffer;
  private deltaYBuf!: GPUBuffer;
  private cellHeadBuf!: GPUBuffer;
  private cellNextBuf!: GPUBuffer;
  private uniformBuf!: GPUBuffer;
  private stagingBuf!: GPUBuffer;

  private pipelines = new Map<EntryPoint, GPUComputePipeline>();
  private bindGroup!: GPUBindGroup;

  private uniformScratch = new ArrayBuffer(UNIFORM_BYTES);
  private uniformView = new DataView(this.uniformScratch);
  /** Interleaved particle scratch for upload / download. */
  private packed: Float32Array;
  /** Latest completed GPU→CPU snapshot (for downloadSync). */
  private cachePacked: Float32Array;
  private cacheValid = false;
  private readbackInFlight: Promise<void> | null = null;
  private readbackEvery = 0;
  private stepIndex = 0;

  private constructor(
    device: GPUDevice,
    capacity: number,
    bounds: TankBounds,
    spacing: number,
  ) {
    this.device = device;
    this.capacity = Math.max(1, capacity | 0);
    this.bounds = { ...bounds };
    this.spacing = spacing;
    this.restDensity = restDensityFromSpacing(spacing);
    this.cellCount = nextPow2(Math.max(this.capacity, 65_536));
    this.packed = new Float32Array(this.capacity * FLOATS_PER_PARTICLE);
    this.cachePacked = new Float32Array(this.capacity * FLOATS_PER_PARTICLE);
  }

  static async create(
    device: GPUDevice,
    capacity: number,
    bounds: TankBounds,
    spacing: number,
  ): Promise<PbfGpuSolver> {
    const solver = new PbfGpuSolver(device, capacity, bounds, spacing);
    await solver.initGpu();
    return solver;
  }

  private async initGpu(): Promise<void> {
    const d = this.device;
    const cap = this.capacity;
    const cells = this.cellCount;

    this.particleBuf = d.createBuffer({
      label: 'pbf-particles',
      size: cap * PARTICLE_STRIDE_BYTES,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.lambdaBuf = d.createBuffer({
      label: 'pbf-lambda',
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.deltaXBuf = d.createBuffer({
      label: 'pbf-delta-x',
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.deltaYBuf = d.createBuffer({
      label: 'pbf-delta-y',
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.cellHeadBuf = d.createBuffer({
      label: 'pbf-cell-head',
      size: cells * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.cellNextBuf = d.createBuffer({
      label: 'pbf-cell-next',
      size: cap * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.uniformBuf = d.createBuffer({
      label: 'pbf-uniforms',
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.stagingBuf = d.createBuffer({
      label: 'pbf-staging',
      size: cap * PARTICLE_STRIDE_BYTES,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    const module = d.createShaderModule({
      label: 'pbf-shaders',
      code: PBF_SHADER_CODE,
    });

    const info = await module.getCompilationInfo();
    for (const msg of info.messages) {
      if (msg.type === 'error') {
        console.error('[PBF WGSL]', msg.message, `L${msg.lineNum}:${msg.linePos}`);
      }
    }

    // Explicit layout so every entry point shares one bind group (layout:'auto' would diverge).
    const bgl = d.createBindGroupLayout({
      label: 'pbf-bgl',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    const pipeLayout = d.createPipelineLayout({
      label: 'pbf-pll',
      bindGroupLayouts: [bgl],
    });

    for (const entryPoint of ENTRY_POINTS) {
      const pipeline = d.createComputePipeline({
        label: `pbf-${entryPoint}`,
        layout: pipeLayout,
        compute: { module, entryPoint },
      });
      this.pipelines.set(entryPoint, pipeline);
    }

    this.bindGroup = d.createBindGroup({
      label: 'pbf-bg',
      layout: bgl,
      entries: [
        { binding: 0, resource: { buffer: this.uniformBuf } },
        { binding: 1, resource: { buffer: this.particleBuf } },
        { binding: 2, resource: { buffer: this.lambdaBuf } },
        { binding: 3, resource: { buffer: this.deltaXBuf } },
        { binding: 4, resource: { buffer: this.deltaYBuf } },
        { binding: 5, resource: { buffer: this.cellHeadBuf } },
        { binding: 6, resource: { buffer: this.cellNextBuf } },
      ],
    });
  }

  setBounds(b: TankBounds): void {
    this.bounds = { ...b };
  }

  /** Auto GPU→CPU snapshot every N steps (0 = off). Keeps downloadSync warm. */
  setReadbackInterval(n: number): void {
    this.readbackEvery = Math.max(0, n | 0);
  }

  getCount(): number {
    return this.count;
  }

  /** Upload CPU water buffers to GPU (call when count changes or after CPU edits). */
  upload(water: WaterState): void {
    if (this.destroyed) return;
    const n = Math.min(water.count, this.capacity);
    this.count = n;
    this.spacing = water.spacing;
    this.restDensity = water.restDensity > 0 ? water.restDensity : restDensityFromSpacing(water.spacing);

    const src = this.packed;
    for (let i = 0; i < n; i++) {
      const o = i * FLOATS_PER_PARTICLE;
      src[o] = water.x[i]!;
      src[o + 1] = water.y[i]!;
      src[o + 2] = water.px[i]!;
      src[o + 3] = water.py[i]!;
      src[o + 4] = water.vx[i]!;
      src[o + 5] = water.vy[i]!;
      src[o + 6] = water.invMass[i]!;
      src[o + 7] = 0;
    }

    if (n > 0) {
      this.device.queue.writeBuffer(
        this.particleBuf,
        0,
        src.buffer as ArrayBuffer,
        src.byteOffset,
        n * PARTICLE_STRIDE_BYTES,
      );
    }
    this.cachePacked.set(src.subarray(0, n * FLOATS_PER_PARTICLE));
    this.cacheValid = true;
  }

  /**
   * One PBF step on GPU.
   * Subpasses: integrate → buildHash → (λ → Δp → apply → softBounds)×iters
   *          → updateVel → viscosity → killBounds → commit.
   */
  step(params: PbfGpuStepParams): void {
    if (this.destroyed || this.count === 0 || params.dt <= 0) return;

    const visc = Math.max(0, Math.min(1, params.viscosity));
    this.writeUniforms(params, visc);

    const enc = this.device.createCommandEncoder({ label: 'pbf-step' });
    const n = this.count;
    const cells = this.cellCount;
    const iters = params.iterations > 0 ? params.iterations | 0 : 3;

    // 1. Gravity + wind → predict
    this.dispatch(enc, 'integrate', n);
    // 2. Spatial hash once (matches CPU — not rebuilt each constraint iter)
    this.dispatch(enc, 'clearHash', cells);
    this.dispatch(enc, 'insertHash', n);

    // 3. Density-constraint iterations
    for (let i = 0; i < iters; i++) {
      this.dispatch(enc, 'computeLambda', n);
      this.dispatch(enc, 'computeDelta', n);
      this.dispatch(enc, 'applyDelta', n);
      this.dispatch(enc, 'collideBoundsSoft', n);
    }

    // 4. Velocities from position change
    this.dispatch(enc, 'updateVel', n);

    // 5. XSPH viscosity
    if (visc > 0) {
      this.dispatch(enc, 'clearHash', cells);
      this.dispatch(enc, 'insertHash', n);
      this.dispatch(enc, 'viscosityAccum', n);
      this.dispatch(enc, 'viscosityApply', n);
    }

    // 6. Final walls (kill normal v) + commit positions
    this.dispatch(enc, 'collideBoundsKill', n);
    this.dispatch(enc, 'commitPositions', n);

    this.device.queue.submit([enc.finish()]);

    this.stepIndex++;
    if (this.readbackEvery > 0 && this.stepIndex % this.readbackEvery === 0) {
      void this.kickReadback();
    }
  }

  /** Read positions/velocities back into WaterState (for coupling / render fallback). */
  async download(water: WaterState): Promise<void> {
    if (this.destroyed) return;
    await this.readbackToCache();
    this.applyCacheToWater(water);
  }

  /**
   * Fast sync download for CPU coupling.
   * Applies the latest completed GPU snapshot (upload or finished readback).
   * Browsers cannot truly stall on mapAsync; prefer `await download()` for same-frame data,
   * or `setReadbackInterval(1)` so the cache stays warm.
   */
  downloadSync(water: WaterState): void {
    if (this.destroyed) return;
    if (this.cacheValid) {
      this.applyCacheToWater(water);
    }
    void this.kickReadback();
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;
    this.particleBuf.destroy();
    this.lambdaBuf.destroy();
    this.deltaXBuf.destroy();
    this.deltaYBuf.destroy();
    this.cellHeadBuf.destroy();
    this.cellNextBuf.destroy();
    this.uniformBuf.destroy();
    this.stagingBuf.destroy();
    this.pipelines.clear();
  }

  // ── Internals ────────────────────────────────────────────────────────────

  private dispatch(encoder: GPUCommandEncoder, entry: EntryPoint, threads: number): void {
    const pipeline = this.pipelines.get(entry)!;
    const pass = encoder.beginComputePass({ label: entry });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(workgroupCount(threads, PBF_WORKGROUP_SIZE));
    pass.end();
  }

  private writeUniforms(params: PbfGpuStepParams, visc: number): void {
    const h = this.spacing * 2;
    const h2 = h * h;
    const rho0 = this.restDensity;
    const poly6Coeff = 4 / (Math.PI * h ** 8);
    const spikyBase = -30 / (Math.PI * h ** 5);
    const dqRel = 0.25;
    const dq2 = (dqRel * h) ** 2;
    let wDq = 0;
    if (dq2 < h2) {
      const d = h2 - dq2;
      wDq = poly6Coeff * d * d * d;
    }
    const kCorr = Math.max(0, params.surfaceTension) * 0.05;
    const invCell = h > 0 ? 1 / h : 1;
    const cellMask = this.cellCount - 1;

    const v = this.uniformView;
    let o = 0;
    const f32 = (x: number) => {
      v.setFloat32(o, x, true);
      o += 4;
    };
    const u32 = (x: number) => {
      v.setUint32(o, x >>> 0, true);
      o += 4;
    };

    f32(params.gravity);
    f32(visc);
    f32(params.dt);
    f32(params.wind ?? 0);

    f32(h);
    f32(h2);
    f32(rho0);
    f32(rho0 * 1e-6);

    f32(poly6Coeff);
    f32(spikyBase);
    f32(wDq);
    f32(kCorr);

    f32(4); // nCorr
    f32(invCell);
    u32(this.count);
    u32(cellMask);

    f32(this.bounds.x0);
    f32(this.bounds.y0);
    f32(this.bounds.x1);
    f32(this.bounds.y1);

    this.device.queue.writeBuffer(this.uniformBuf, 0, this.uniformScratch);
  }

  private applyCacheToWater(water: WaterState): void {
    const n = Math.min(this.count, water.capacity);
    water.count = n;
    const src = this.cachePacked;
    for (let i = 0; i < n; i++) {
      const o = i * FLOATS_PER_PARTICLE;
      water.x[i] = src[o]!;
      water.y[i] = src[o + 1]!;
      water.px[i] = src[o + 2]!;
      water.py[i] = src[o + 3]!;
      water.vx[i] = src[o + 4]!;
      water.vy[i] = src[o + 5]!;
      water.invMass[i] = src[o + 6]!;
    }
  }

  private kickReadback(): Promise<void> {
    if (this.readbackInFlight) return this.readbackInFlight;
    this.readbackInFlight = this.readbackToCache().finally(() => {
      this.readbackInFlight = null;
    });
    return this.readbackInFlight;
  }

  private async readbackToCache(): Promise<void> {
    if (this.destroyed || this.count === 0) {
      this.cacheValid = this.count === 0;
      return;
    }
    const n = this.count;
    const bytes = n * PARTICLE_STRIDE_BYTES;

    if (this.stagingBuf.mapState === 'mapped') {
      this.stagingBuf.unmap();
    }

    const enc = this.device.createCommandEncoder({ label: 'pbf-readback' });
    enc.copyBufferToBuffer(this.particleBuf, 0, this.stagingBuf, 0, bytes);
    this.device.queue.submit([enc.finish()]);

    await this.stagingBuf.mapAsync(GPUMapMode.READ, 0, bytes);
    const mapped = new Float32Array(this.stagingBuf.getMappedRange(0, bytes));
    this.cachePacked.set(mapped);
    this.stagingBuf.unmap();
    this.cacheValid = true;
  }
}

/** Local copy of water-state helper so GPU module does not depend on CPU agent timing. */
function restDensityFromSpacing(spacing: number): number {
  const h = spacing * 2;
  const h2 = h * h;
  const coeff = 4 / (Math.PI * h ** 8);
  let rho = 0;
  for (let iy = -2; iy <= 2; iy++) {
    for (let ix = -2; ix <= 2; ix++) {
      const r2 = (ix * spacing) ** 2 + (iy * spacing) ** 2;
      if (r2 >= h2) continue;
      const d = h2 - r2;
      rho += coeff * d * d * d;
    }
  }
  return rho > 0 ? rho : 1 / (spacing * spacing);
}

export { tryInitWebGPU };
