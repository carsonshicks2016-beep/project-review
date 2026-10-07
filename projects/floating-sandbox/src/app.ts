import type { SimConfig, ToolId, Vec2 } from './sim/types';
import { World, DEFAULT_CONFIG } from './sim/world';
import { spawnVessel, importBlueprintFromFile, VESSEL_PRESETS } from './sim/vessels';
import { createToolState, ToolController } from './sim/tools/tools';
import { Camera, attachCameraControls } from './render/camera';
import { SimRenderer } from './render/renderer';
import { mountToolbar, mountPanel, mountHud } from './ui';
import { pickWaterBackend } from './sim/water/backend';
import { PbfGpuSolver } from './sim/water/pbf-gpu';

/**
 * Application shell: wires world, tools, UI, renderer, and optional WebGPU water.
 */
export class App {
  readonly world: World;
  readonly camera: Camera;
  readonly renderer: SimRenderer;
  readonly tools = createToolState();
  readonly toolController: ToolController;

  private densityScale = 1;
  private strengthScale = 1;
  private cursorWorld: Vec2 | null = null;
  private gpu: PbfGpuSolver | null = null;
  private raf = 0;
  private lastT = 0;
  private running = false;
  private hud: { update: (s: import('./sim/types').SimStats) => void };
  private panelRefresh: () => void;
  private toolbarRefresh: () => void;

  constructor(
    private canvas: HTMLCanvasElement,
    toolbarEl: HTMLElement,
    panelEl: HTMLElement,
    hudEl: HTMLElement,
  ) {
    this.world = new World({ particleCapacity: 140000, spacing: 8 });
    this.camera = new Camera();
    this.renderer = new SimRenderer(canvas);
    this.toolController = new ToolController(this.tools);

    this.toolbarRefresh = mountToolbar(toolbarEl, {
      onTool: (id) => {
        this.toolController.setTool(id);
        this.toolbarRefresh();
      },
      getTool: () => this.tools.active,
    }).refresh;

    this.panelRefresh = mountPanel(panelEl, {
      onSpawn: (id) => this.spawn(id),
      onDensity: (v) => {
        this.densityScale = v;
      },
      onStrength: (v) => {
        this.strengthScale = v;
      },
      onParticleTarget: (n) => {
        this.world.config.particleTarget = n;
        this.world.fillOcean(0.45);
      },
      onToggleGpu: (on) => {
        this.world.config.useGpu = on;
        void this.setupBackend();
      },
      onToggleStress: (on) => {
        this.world.config.stressOverlay = on;
      },
      onPause: () => {
        this.world.config.paused = !this.world.config.paused;
        this.panelRefresh();
      },
      onStep: () => void this.world.stepOnce(),
      onReset: () => this.reset(),
      onBlueprint: (file) => void this.loadBlueprint(file),
      onWind: (v) => {
        this.world.config.wind = v;
      },
      getConfig: () => ({
        ...this.world.config,
        densityScale: this.densityScale,
        strengthScale: this.strengthScale,
      }),
    }).refresh;

    this.hud = mountHud(hudEl);

    // Pan with Alt or middle mouse; wheel zooms. Left-drag reserved for tools.
    attachCameraControls(canvas, this.camera, () => true);

    this.bindPointer();
    this.camera.fitBounds(this.world.bounds, canvas, 40);
    this.renderer.resize();

    // Default vessel
    this.spawn('liner');
  }

  async init(): Promise<void> {
    await this.setupBackend();
  }

  private async setupBackend(): Promise<void> {
    const prefer = this.world.config.useGpu;
    const { backend, device } = await pickWaterBackend(prefer);
    if (backend === 'webgpu' && device) {
      try {
        this.gpu?.destroy();
        this.gpu = await PbfGpuSolver.create(
          device,
          this.world.water.capacity,
          this.world.bounds,
          this.world.water.spacing,
        );
        this.gpu.upload(this.world.water);
        this.world.gpuStep = async (params, water) => {
          this.gpu!.setBounds(this.world.bounds);
          this.gpu!.step(params);
          await this.gpu!.download(water);
        };
        this.world.gpuUpload = (w) => this.gpu!.upload(w);
        this.world.backend = 'webgpu';
      } catch (err) {
        console.warn('WebGPU water init failed, using CPU', err);
        this.fallbackCpu();
      }
    } else {
      this.fallbackCpu();
    }
    this.panelRefresh();
  }

  private fallbackCpu(): void {
    this.gpu?.destroy();
    this.gpu = null;
    this.world.gpuStep = null;
    this.world.gpuUpload = null;
    this.world.backend = 'cpu';
    if (this.world.config.particleTarget > 20000) {
      this.world.config.particleTarget = 12000;
      this.world.fillOcean(0.45);
    }
  }

  spawn(vesselId: string): void {
    const b = this.world.bounds;
    const cx = (b.x0 + b.x1) * 0.5;
    const cy = b.y1 - (b.y1 - b.y0) * 0.45 - 30;
    const ship = spawnVessel(vesselId, cx, cy, this.densityScale, this.strengthScale);
    if (ship) this.world.setShip(ship);
  }

  async loadBlueprint(file: File): Promise<void> {
    const b = this.world.bounds;
    const cx = (b.x0 + b.x1) * 0.5;
    const cy = b.y1 - (b.y1 - b.y0) * 0.45 - 30;
    const ship = await importBlueprintFromFile(file, cx, cy, this.densityScale, this.strengthScale);
    this.world.setShip(ship);
  }

  reset(): void {
    this.world.reset(false);
    this.spawn('liner');
  }

  private bindPointer(): void {
    const canvas = this.canvas;
    const getWorld = (e: PointerEvent) =>
      this.camera.screenToWorld(e.clientX, e.clientY, canvas);

    canvas.addEventListener('pointerdown', (e) => {
      if (e.button === 1 || e.altKey) return; // camera
      canvas.setPointerCapture(e.pointerId);
      const w = getWorld(e);
      this.cursorWorld = w;
      this.toolController.onPointerDown(w.x, w.y, this.world.water, this.world.ship, this.world.lattice);
    });
    canvas.addEventListener('pointermove', (e) => {
      const w = getWorld(e);
      this.cursorWorld = w;
      this.toolController.onPointerMove(
        w.x,
        w.y,
        this.world.water,
        this.world.ship,
        this.world.lattice,
        e.buttons,
      );
    });
    canvas.addEventListener('pointerup', (e) => {
      const w = getWorld(e);
      this.cursorWorld = w;
      this.toolController.onPointerUp(w.x, w.y, this.world.water, this.world.ship, this.world.lattice);
    });
  }

  start(): void {
    this.running = true;
    this.lastT = performance.now();
    let busy = false;

    const tick = (t: number) => {
      if (!this.running) return;
      // Always schedule next frame first so a slow/hung step cannot freeze the loop forever.
      this.raf = requestAnimationFrame(tick);
      if (busy) return;
      busy = true;

      const dt = Math.min(0.05, (t - this.lastT) / 1000);
      this.lastT = t;

      void this.world
        .update(dt)
        .then(() => {
          this.renderer.resize();
          this.renderer.render({
            camera: this.camera,
            bounds: this.world.bounds,
            water: this.world.water,
            ship: this.world.ship,
            stressOverlay: this.world.config.stressOverlay,
            toolRadius: this.tools.brushRadius,
            cursorWorld: this.cursorWorld,
          });
          this.hud.update(this.world.getStats());
        })
        .catch((err) => {
          console.error('sim step failed', err);
        })
        .finally(() => {
          busy = false;
        });
    };

    this.raf = requestAnimationFrame(tick);
  }

  stop(): void {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }
}

// Re-export for debugging
export { DEFAULT_CONFIG, VESSEL_PRESETS };
export type { SimConfig, ToolId };
