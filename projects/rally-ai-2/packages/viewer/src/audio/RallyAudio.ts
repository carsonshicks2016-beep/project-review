/**
 * Rally WATCH audio facade: unlock, mute, engine / tyres / wind / ambient / co-driver.
 *
 * Cost: main-thread update() is timed each frame; voices only automate AudioParams
 * (no ScriptProcessor / AudioWorklet process allocations).
 */

import type { PlaybackSample } from "../playback";
import type { Stage } from "../replay";
import { AmbientAudio } from "./AmbientAudio";
import { AudioBus } from "./AudioBus";
import { CoDriverAudio, paceDirFromNote } from "./CoDriverAudio";
import { EngineAudio } from "./EngineAudio";
import { TyreAudio } from "./TyreAudio";
import { WindAudio } from "./WindAudio";

/** Nominal frame budget at 60 Hz (ms). */
const FRAME_BUDGET_MS = 1000 / 60;

export type AudioCostSample = {
  /** Last update() wall time in ms (main thread). */
  lastMs: number;
  /** EMA of update cost in ms. */
  emaMs: number;
  /** lastMs / 16.67 as a fraction of a 60 Hz frame. */
  frameShare: number;
  samples: number;
};

export class RallyAudio {
  readonly bus = new AudioBus();
  readonly engine = new EngineAudio(this.bus);
  readonly tyres = new TyreAudio(this.bus);
  readonly wind = new WindAudio(this.bus);
  readonly ambient = new AmbientAudio(this.bus);
  readonly codriver = new CoDriverAudio(this.bus);

  private gestureRoot: HTMLElement | Document = document;
  private cost: AudioCostSample = { lastMs: 0, emaMs: 0, frameShare: 0, samples: 0 };

  get muted(): boolean {
    return this.bus.isMuted;
  }

  get costSample(): AudioCostSample {
    return this.cost;
  }

  bindGestures(root: HTMLElement | Document = document): void {
    this.gestureRoot = root;
    this.bus.bindUnlockGestures(root);
  }

  /** Call from scrub / any explicit user action. */
  async unlock(): Promise<void> {
    const ok = await this.bus.unlock();
    if (ok) this.startLayers();
  }

  setMuted(muted: boolean): void {
    this.bus.setMuted(muted);
  }

  toggleMute(): boolean {
    const next = !this.bus.isMuted;
    this.setMuted(next);
    return next;
  }

  /** Duck to silence while the transport is paused; mute state is untouched. */
  setPaused(paused: boolean): void {
    this.bus.setDucked(paused);
  }

  setStage(stage: Stage | null | undefined): void {
    this.ambient.setFromStage(stage);
  }

  /** Clear co-driver memory after seek/restart. */
  reset(): void {
    this.codriver.reset();
  }

  update(sample: PlaybackSample, dt: number): void {
    const t0 = performance.now();
    if (!this.bus.available) {
      this.recordCost(performance.now() - t0);
      return;
    }
    if (this.bus.ready) this.startLayers();

    this.engine.update(
      {
        rpm: sample.rpm,
        gear: sample.gear,
        throttle: sample.throttle,
        brake: sample.brake,
        speed: sample.v,
        boost: sample.boost,
        airborne: sample.airborne,
      },
      dt,
    );
    this.tyres.update(
      {
        surface: sample.surface,
        speed: sample.v,
        slip: sample.slipAngle,
        wheels: sample.wheels,
        airborne: sample.airborne,
      },
      dt,
    );
    this.wind.update({ speed: sample.v, airborne: sample.airborne }, dt);
    this.ambient.update({ surface: sample.surface, speed: sample.v }, dt);
    this.codriver.update(
      {
        pace_dir: paceDirFromNote(sample.note),
        pace_note: sample.note,
      },
      dt,
    );

    this.recordCost(performance.now() - t0);
  }

  dispose(): void {
    this.bus.unbindUnlockGestures(this.gestureRoot);
    this.engine.dispose();
    this.tyres.dispose();
    this.wind.dispose();
    this.ambient.dispose();
    this.codriver.dispose();
    this.bus.dispose();
  }

  private startLayers(): void {
    this.engine.start();
    this.tyres.start();
    this.wind.start();
    this.ambient.start();
    this.codriver.start();
  }

  private recordCost(ms: number): void {
    const lastMs = ms;
    const samples = this.cost.samples + 1;
    const emaMs = samples === 1 ? lastMs : this.cost.emaMs * 0.9 + lastMs * 0.1;
    this.cost = {
      lastMs,
      emaMs,
      frameShare: emaMs / FRAME_BUDGET_MS,
      samples,
    };
    (globalThis as unknown as { __rallyAudioCost?: AudioCostSample }).__rallyAudioCost =
      this.cost;
  }
}

export { AudioBus } from "./AudioBus";
export { EngineAudio } from "./EngineAudio";
export { AmbientAudio, surfaceToBiome, stageToBiome } from "./AmbientAudio";
export { TyreAudio } from "./TyreAudio";
export { WindAudio } from "./WindAudio";
export { CoDriverAudio, paceDirFromNote } from "./CoDriverAudio";
export type { AmbientBiome } from "./AmbientAudio";
export type { AudioCostSample as RallyAudioCost };
