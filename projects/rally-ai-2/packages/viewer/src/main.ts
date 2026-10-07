/**
 * WATCH — replay playback.
 *
 * The whole surface: load a replay, build the stage it embeds, drive the car
 * along the recorded frames, and let someone watch it. There is no simulation
 * here and there must never be. **Python owns the truth**; this renders state
 * it is handed and never steps the world.
 *
 * The replay is self-contained and hash-verified on write, so nothing in this
 * package needs the sim running, a server, or a trained policy to work.
 */

import {
  Color,
  DirectionalLight,
  Fog,
  HemisphereLight,
  Object3D,
  Scene,
  Vector3,
  WebGLRenderer,
  type Mesh,
} from "three";

import { RallyAudio } from "./audio/RallyAudio";
import {
  visualBenchmark,
  type VisualBenchmark,
} from "./benchmarks";
import { ChaseCamera, type CameraMode } from "./camera";
import { CameraDirector } from "./camera/director";
import { orientation, xyz } from "./coords";
import { Hud } from "./hud";
import { Playback } from "./playback";
import { applyVertexJitter, Ps1Pass } from "./render/ps1";
import { loadReplay, type LoadedReplay } from "./replay";
import {
  buildCar,
  driveWheels,
  resetSuspension,
  updateSuspension,
  type CarRig,
} from "./scene/car";
import { CarStateEffects } from "./scene/carEffects";
import { SurfaceEffects } from "./scene/dust";
import { GroundingShadows } from "./scene/shadows";
import { buildSky, HORIZON_COLOUR } from "./scene/sky";
import { buildStage } from "./scene/stage";
import {
  StageSurface,
  WheelVisualController,
} from "./scene/surface";
import { TyreTracks } from "./scene/tracks";
import { buildBanners, buildRoadside } from "./scene/trees";

/**
 * The colour distance fades into.
 *
 * Taken from the sky's own horizon stop so the fog and the sky cannot drift
 * apart — a fog colour that does not match the horizon puts a visible band
 * across the bottom of the sky, and it is the kind of thing that looks
 * "slightly wrong" for a long time before anyone works out why.
 */
const SKY = new Color(HORIZON_COLOUR);
/**
 * Draw distance, and where it starts to bite.
 *
 * Aggressive on purpose. Hardware of the era could not draw far, so the games
 * fogged hard and the fog became part of the look rather than an apology for
 * it. The compact 42/210 m band also keeps the denser mixed forest from reading
 * as saturated cardboard at the horizon.
 */
const FOG_NEAR = 42;
const FOG_FAR = 210;

/** Rolling radius, metres — matches `WHEEL_R` in the car mesh. */
const WHEEL_R = 0.32;

/** Rolling window for the dev-only frame-time readout. */
const FRAME_BUDGET_SAMPLES = 120;

/** Dev-only rolling frame-time stats, attached to `window.watch`. */
class FrameBudget {
  private readonly samples = new Float64Array(FRAME_BUDGET_SAMPLES);
  private readonly sorted = new Float64Array(FRAME_BUDGET_SAMPLES);
  private cursor = 0;
  private filled = 0;

  push(ms: number): void {
    this.samples[this.cursor] = ms;
    this.cursor = (this.cursor + 1) % FRAME_BUDGET_SAMPLES;
    this.filled = Math.min(this.filled + 1, FRAME_BUDGET_SAMPLES);
  }

  /** Clear the window so a console measurement starts clean. */
  reset(): void {
    this.cursor = 0;
    this.filled = 0;
  }

  get count(): number {
    return this.filled;
  }

  mean(): number {
    if (this.filled === 0) return 0;
    let sum = 0;
    for (let i = 0; i < this.filled; i++) sum += this.samples[i]!;
    return sum / this.filled;
  }

  p95(): number {
    if (this.filled === 0) return 0;
    for (let i = 0; i < this.filled; i++) this.sorted[i] = this.samples[i]!;
    const view = this.sorted.subarray(0, this.filled);
    view.sort((a, b) => a - b);
    const idx = Math.min(
      this.filled - 1,
      Math.ceil(this.filled * 0.95) - 1,
    );
    return view[idx]!;
  }

  /** `watch.frameBudget.print()` from the console. */
  print(): string {
    const line =
      `frame budget (last ${this.filled}): ` +
      `mean ${this.mean().toFixed(2)} ms, ` +
      `p95 ${this.p95().toFixed(2)} ms`;
    console.log(line);
    return line;
  }
}

class Watch {
  private readonly renderer: WebGLRenderer;
  private readonly scene = new Scene();
  private readonly chase: ChaseCamera;
  private readonly ps1: Ps1Pass;
  private readonly car: CarRig;
  private readonly road: StageSurface;
  private readonly wheels: WheelVisualController;
  private readonly shadows: GroundingShadows;
  private readonly surfaceEffects: SurfaceEffects;
  private readonly tracks: TyreTracks;
  private readonly carEffects: CarStateEffects;
  private readonly playback: Playback;
  private readonly hud: Hud;
  private readonly replay: LoadedReplay;
  private readonly audio: RallyAudio;
  private readonly director: CameraDirector;

  private last = performance.now();
  private wheelAngle = 0;
  private ended = false;
  private controlsTimer: number | undefined;
  private readonly frameBudget = new FrameBudget();
  /** Replay-time auto edit. Off under a fixed benchmark's pinned camera. */
  private directing: boolean;
  private lastShot = -2;

  private readonly pos = new Vector3();
  private readonly fwd = new Vector3();

  constructor(
    canvas: HTMLCanvasElement,
    replay: LoadedReplay,
    benchmark?: VisualBenchmark,
  ) {
    this.replay = replay;
    this.renderer = new WebGLRenderer({ canvas, antialias: true });
    this.renderer.setClearColor(SKY);
    // The model carries the period look now, not an enlarged framebuffer.
    // Capping Retina at 1.5 keeps the image crisp without wasting fill rate.
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));

    this.scene.fog = new Fog(SKY, FOG_NEAR, FOG_FAR);
    // Broad sky fill plus one decisive "sun" makes every deliberately simple
    // plane readable. Flat-shaded Lambert materials turn that light into the
    // clean value bands that define the new high-resolution low-poly look.
    this.scene.add(new HemisphereLight(0xcfe7ef, 0x50613d, 2.05));
    const sun = new DirectionalLight(0xfff2cf, 2.65);
    sun.position.set(-38, 74, 46);
    this.scene.add(sun);
    this.scene.add(buildStage(replay.stage));
    this.scene.add(buildRoadside(replay.stage));
    this.scene.add(buildBanners(replay.stage));

    this.road = new StageSurface(replay.stage);
    this.wheels = new WheelVisualController(this.road);
    this.tracks = new TyreTracks();
    this.scene.add(this.tracks.root);

    this.car = buildCar();
    this.scene.add(this.car.root);
    this.shadows = new GroundingShadows();
    this.scene.add(this.shadows.root);
    this.surfaceEffects = new SurfaceEffects();
    this.scene.add(this.surfaceEffects.root);
    this.carEffects = new CarStateEffects(replay, this.car);
    this.scene.add(this.carEffects.root);

    this.chase = new ChaseCamera(1);
    this.director = new CameraDirector(replay);
    // The director is the default watching experience; a benchmark pins its
    // own camera and must stay pixel-identical, so it never directs.
    this.directing = !benchmark;
    if (benchmark) this.chase.mode = benchmark.camera;
    // The sky travels with the camera so it can never be reached or driven past.
    this.chase.camera.add(buildSky());
    this.scene.add(this.chase.camera);

    // A trace of geometry wobble remains in the environment, but the hero car
    // and graphic event dressing stay stable and clean. The nostalgia should
    // come from facets and silhouette, never from making the subject illegible.
    this.ps1 = new Ps1Pass(this.renderer);
    const belongsTo = (o: Object3D, name: string): boolean => {
      for (let p: Object3D | null = o; p; p = p.parent) {
        if (p.name === name) return true;
      }
      return false;
    };
    this.scene.traverse((o) => {
      if (
        belongsTo(o, "sky") ||
        belongsTo(o, "car") ||
        belongsTo(o, "banners") ||
        belongsTo(o, "persistent_tyre_tracks") ||
        belongsTo(o, "vehicle_grounding_shadows") ||
        belongsTo(o, "per_wheel_surface_effects") ||
        belongsTo(o, "car_state_effects")
      ) return;
      const m = (o as Mesh).material;
      if (!m) return;
      for (const mat of Array.isArray(m) ? m : [m]) {
        applyVertexJitter(mat, false);
      }
    });
    this.playback = new Playback(replay);
    if (benchmark) {
      this.playback.seek(benchmark.time);
      this.playback.playing = false;
    }
    this.hud = new Hud(document.getElementById("game")!, replay);

    this.audio = new RallyAudio();
    this.audio.setStage(replay.stage);
    // Browsers gate AudioContext behind a user gesture; any click or key
    // (space to play, a scrub, the canvas) unlocks it.
    this.audio.bindGestures(document);

    this.resize();
    window.addEventListener("resize", () => this.resize());
    window.addEventListener("pointermove", () => this.revealControls());
    this.bindKeys();
    this.bindScrub();
    this.resetPresentation();
  }

  private resize(): void {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const w = Math.max(1, Math.round(rect.width));
    const h = Math.max(1, Math.round(rect.height));
    this.renderer.setSize(w, h, false);
    this.chase.resize(w / h);
    this.ps1.resize();
  }

  private bindKeys(): void {
    window.addEventListener("keydown", (e) => {
      switch (e.key) {
        case " ":
          e.preventDefault();
          this.togglePlayback();
          break;
        case "ArrowLeft":
          this.playback.seek(this.playback.time - (e.shiftKey ? 10 : 2));
          this.resetPresentation();
          break;
        case "ArrowRight":
          this.playback.seek(this.playback.time + (e.shiftKey ? 10 : 2));
          this.resetPresentation();
          break;
        case "[":
          this.playback.rate = Math.max(0.1, this.playback.rate / 2);
          break;
        case "]":
          this.playback.rate = Math.min(8, this.playback.rate * 2);
          break;
        case "c":
          this.cycleCamera();
          break;
        case "m":
          this.audio.toggleMute();
          break;
        case "p":
          // A/B against a clean render. Any "does this effect actually look
          // better" question is unanswerable without being able to turn it off.
          this.ps1.enabled = !this.ps1.enabled;
          break;
        case "r":
          this.playback.seek(0);
          this.playback.playing = true;
          this.resetPresentation();
          break;
      }
      this.syncControls();
      this.revealControls();
    });
  }

  private togglePlayback(): void {
    const restarting =
      !this.playback.playing && this.playback.time >= this.playback.duration;
    this.playback.togglePlay();
    if (restarting) this.resetPresentation();
  }

  private resetPresentation(): void {
    this.ended = false;
    this.chase.reset();
    this.lastShot = -2;
    this.audio.reset();
    resetSuspension(this.car);
    const sample = this.playback.sample();
    this.wheelAngle = sample.s / WHEEL_R;
    this.tracks.rebuild(this.replay, this.playback.time, this.wheels);
    this.surfaceEffects.prewarm(
      this.replay,
      this.playback.time,
      this.wheels,
    );
    this.carEffects.rebuild(this.playback.time);
    this.hud.reset();
    // Apply the selected replay pose immediately. Fixed benchmark routes are
    // valid before the browser grants the first animation frame, and a seek
    // never flashes the previous car/shadow/camera state for one frame.
    this.step(1 / 60);
  }

  /** auto (director) → chase → bonnet → sideline → auto. */
  private cycleCamera(): void {
    if (this.directing) {
      this.directing = false;
      this.chase.mode = "chase";
    } else {
      const order: CameraMode[] = ["chase", "bonnet", "sideline"];
      const i = order.indexOf(this.chase.mode);
      if (i === order.length - 1) {
        this.directing = true;
        this.lastShot = -2;
      } else {
        this.chase.mode = order[i + 1]!;
      }
    }
    this.chase.reset();
  }

  private bindScrub(): void {
    const scrub = document.getElementById("scrub") as HTMLInputElement | null;
    if (!scrub) return;
    scrub.max = String(this.playback.duration);
    scrub.addEventListener("input", () => {
      this.playback.seek(Number(scrub.value));
      this.resetPresentation();
      this.revealControls();
    });
    const play = document.getElementById("play");
    play?.addEventListener("click", () => {
      this.togglePlayback();
      this.syncControls();
      this.revealControls();
    });
  }

  private revealControls(): void {
    const transport = document.querySelector<HTMLElement>(".transport");
    if (!transport) return;
    transport.classList.add("visible");
    if (this.controlsTimer !== undefined) window.clearTimeout(this.controlsTimer);
    if (this.playback.playing) {
      this.controlsTimer = window.setTimeout(() => {
        transport.classList.remove("visible");
        this.controlsTimer = undefined;
      }, 1800);
    }
  }

  private syncControls(): void {
    const rate = document.getElementById("rate");
    if (rate) rate.textContent = `${this.playback.rate}x`;
    const cam = document.getElementById("cam");
    if (cam) {
      cam.textContent = this.directing ? `auto·${this.chase.mode}` : this.chase.mode;
    }
    const mute = document.getElementById("mute");
    if (mute) mute.textContent = this.audio.muted ? "muted" : "sound";
    const play = document.getElementById("play");
    if (play) play.textContent = this.playback.playing ? "❚❚" : "▶";
  }

  start(): void {
    this.syncControls();
    this.revealControls();
    // Dev-only handle. Debugging a 3D scene from the outside means being able to
    // ask it where things actually are — guessing at geometry from a screenshot
    // is how you spend an hour on the wrong hypothesis.
    if (import.meta.env.DEV) {
      (window as unknown as { watch: unknown }).watch = {
        scene: this.scene,
        camera: this.chase.camera,
        car: this.car,
        chase: this.chase,
        road: this.road,
        wheels: this.wheels,
        shadows: this.shadows,
        surfaceEffects: this.surfaceEffects,
        tracks: this.tracks,
        carEffects: this.carEffects,
        playback: this.playback,
        frameBudget: this.frameBudget,
        audio: this.audio,
        director: this.director,
      };
    }
    const loop = (now: number): void => {
      // Clamped: a backgrounded tab returns a dt of many seconds, which would
      // teleport the car and snap the camera on the frame you come back.
      const dt = Math.min((now - this.last) / 1000, 0.1);
      this.last = now;

      const frameStart = performance.now();
      this.playback.advance(dt);
      this.step(dt);
      this.ps1.render(this.scene, this.chase.camera);
      if (import.meta.env.DEV) {
        this.frameBudget.push(performance.now() - frameStart);
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  private step(dt: number): void {
    const s = this.playback.sample();
    const replayDt = this.playback.playing ? dt * this.playback.rate : 0;

    xyz(s.x, s.y, s.z, this.pos);
    this.car.root.position.copy(this.pos);
    this.car.root.quaternion.copy(orientation(s.yaw, s.pitch, s.roll));
    updateSuspension(
      this.car,
      s.wheels,
      Boolean(this.playback.frame.air),
      dt,
    );

    // Wheels roll at road speed. Cosmetic, so it uses speed rather than the
    // per-wheel angular velocity — a locked wheel under braking still turns
    // here, which is wrong and not worth a wheel-state lookup to fix yet.
    this.wheelAngle += (s.v / WHEEL_R) * replayDt;
    driveWheels(this.car, s.steer, this.wheelAngle);
    const wheelState = this.wheels.update(s, this.car);
    this.shadows.update(this.pos, s.s, s.height, this.road, wheelState);
    this.tracks.update(wheelState);
    this.surfaceEffects.update(replayDt, wheelState, {
      brake: s.brake,
      throttle: s.throttle,
    });
    this.carEffects.update(replayDt, s);

    // Replay-time auto edit. A shot change re-seeds the damped camera so a
    // cut is a cut, not a swing of the boom across the stage. Vibration and
    // landing kick live inside ChaseCamera.apply() and run in every mode.
    if (this.directing) {
      const shot = this.director.shotIndexAt(this.playback.time);
      if (shot !== this.lastShot) {
        this.lastShot = shot;
        const mode = this.director.modeAt(this.playback.time);
        if (mode !== this.chase.mode) this.chase.mode = mode;
        this.chase.reset();
        this.syncControls();
      }
    }

    xyz(Math.cos(s.yaw), Math.sin(s.yaw), 0, this.fwd);
    this.chase.update(this.pos, this.fwd, s.v, dt, {
      time: this.playback.time,
      surface: s.surface,
      slipAngle: s.slipAngle,
      landingKick: this.landingKick(this.playback.time),
    });

    this.hud.update(this.playback.frame, this.playback.time, s.v);

    // Audio reads the same interpolated sample as the renderer, so RPM glides
    // between 30 Hz frames instead of zippering. Paused playback ducks to
    // silence rather than droning the engine at a frozen RPM.
    this.audio.setPaused(!this.playback.playing);
    if (this.playback.playing) this.audio.update(s, dt);

    const scrub = document.getElementById("scrub") as HTMLInputElement | null;
    if (scrub && document.activeElement !== scrub) {
      scrub.value = String(this.playback.time);
    }

    if (!this.ended && this.playback.time >= this.playback.duration) {
      this.ended = true;
      this.hud.finish(this.replay);
      this.syncControls();
      this.revealControls();
    }
  }

  private landingKick(time: number): number {
    let kick = 0;
    for (const event of this.replay.events ?? []) {
      if (event.kind !== "landing") continue;
      const age = time - event.t;
      if (age < 0 || age > 1.2) continue;
      const severity = Math.min(1.35, Math.max(0.45, (event.severity ?? 3) / 5));
      kick +=
        Math.sin(age * 15.5) *
        Math.exp(-age * 4.2) *
        0.19 *
        severity;
    }
    return kick;
  }
}

/**
 * Which replay to load.
 * Prefer `?r=name` (stem under public/replays/). Dashboard deep-links also
 * accept `?replay=<url-or-path>` for a full fetchable path.
 */
function replayUrl(benchmark?: VisualBenchmark): string {
  if (benchmark) return `replays/${benchmark.replay}.json`;
  const params = new URLSearchParams(location.search);
  const full = params.get("replay");
  if (full) return full;
  const p = params.get("r");
  return `replays/${p ?? "demo"}.json`;
}

/** Viewer surface: WATCH (default) or the program dashboard shell. */
function viewerMode(): string | null {
  return new URLSearchParams(location.search).get("mode");
}

async function bootWatch(): Promise<void> {
  const canvas = document.getElementById("view") as HTMLCanvasElement;
  const status = document.getElementById("status")!;
  const benchmark = visualBenchmark(location.search);
  const url = replayUrl(benchmark);
  try {
    const replay = await loadReplay(url);
    const watch = new Watch(canvas, replay, benchmark);
    status.remove();
    watch.start();
  } catch (err) {
    // Shown rather than logged. A blank canvas with an error buried in the
    // console is the single most time-wasting failure mode a viewer has.
    status.className = "status error";
    status.textContent = `Could not load ${url} — ${(err as Error).message}`;
  }
}

/** Hide the replay transport and loading line before a non-WATCH surface. */
function clearWatchChrome(): void {
  const transport = document.querySelector<HTMLElement>(".transport");
  if (transport) transport.hidden = true;
  document.getElementById("status")?.remove();
}

async function boot(): Promise<void> {
  const mode = viewerMode();
  const host = document.getElementById("game");
  if (!host) throw new Error("#game missing");

  // Both alternate surfaces are dynamically imported so WATCH — the default,
  // and the only one that works with no server running — never pays for them.
  if (mode === "dashboard") {
    clearWatchChrome();
    const { bootDashboard } = await import("./dashboard/DashboardApp");
    await bootDashboard(host);
    return;
  }

  if (mode === "train") {
    clearWatchChrome();
    // `train.css` is scoped to this attribute and TrainMode does not set it;
    // without it the studio mounts unstyled. The dashboard sets its own.
    document.body.dataset.mode = "train";
    const { bootTrain } = await import("./train/TrainMode");
    await bootTrain(host);
    return;
  }

  await bootWatch();
}

void boot();
