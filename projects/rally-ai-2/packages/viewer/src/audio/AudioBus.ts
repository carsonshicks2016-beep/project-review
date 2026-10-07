/**
 * Shared Web Audio bus for RallyAI WATCH.
 * Handles autoplay unlock, master mute, and per-layer gain splits.
 *
 * Graph is static after ensure() — voices only automate AudioParams from the
 * main thread. No ScriptProcessor / AudioWorklet process() allocations.
 */

const MUTE_KEY = "rallyai.viewer.audioMuted";

export class AudioBus {
  ctx: AudioContext | null = null;
  master: GainNode | null = null;
  engineBus: GainNode | null = null;
  tyreBus: GainNode | null = null;
  windBus: GainNode | null = null;
  ambientBus: GainNode | null = null;
  voiceBus: GainNode | null = null;

  private unlocked = false;
  private unavailable = false;
  private unlockBound = false;
  private unlockHandler: (() => void) | null = null;
  private muted = false;
  private ducked = false;

  constructor() {
    try {
      this.muted = localStorage.getItem(MUTE_KEY) === "1";
    } catch {
      this.muted = false;
    }
  }

  /** True once AudioContext exists and is running (or was resumed). */
  get ready(): boolean {
    return !!this.ctx && this.ctx.state === "running" && this.unlocked;
  }

  get available(): boolean {
    return !this.unavailable && typeof AudioContext !== "undefined";
  }

  get isMuted(): boolean {
    return this.muted;
  }

  /**
   * Lazily create the graph. Returns false if Web Audio is unavailable.
   */
  ensure(): boolean {
    if (this.unavailable) return false;
    if (this.ctx) return true;
    try {
      const AC =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!AC) {
        this.unavailable = true;
        return false;
      }
      this.ctx = new AC();
      this.master = this.ctx.createGain();
      this.master.gain.value = this.muted ? 0 : 0.85;
      this.master.connect(this.ctx.destination);

      this.engineBus = this.ctx.createGain();
      this.engineBus.gain.value = 0.52;
      this.engineBus.connect(this.master);

      this.tyreBus = this.ctx.createGain();
      this.tyreBus.gain.value = 0.38;
      this.tyreBus.connect(this.master);

      this.windBus = this.ctx.createGain();
      this.windBus.gain.value = 0.28;
      this.windBus.connect(this.master);

      this.ambientBus = this.ctx.createGain();
      this.ambientBus.gain.value = 0.2;
      this.ambientBus.connect(this.master);

      this.voiceBus = this.ctx.createGain();
      this.voiceBus.gain.value = 0.62;
      this.voiceBus.connect(this.master);
      return true;
    } catch {
      this.unavailable = true;
      this.ctx = null;
      return false;
    }
  }

  setMuted(muted: boolean): void {
    this.muted = muted;
    try {
      localStorage.setItem(MUTE_KEY, muted ? "1" : "0");
    } catch {
      /* ignore */
    }
    this.applyMasterGain(0.04);
  }

  /**
   * Transport duck: silence everything while playback is paused without
   * touching the persisted mute preference. Frozen frames would otherwise
   * drone the engine at a fixed RPM forever.
   */
  setDucked(ducked: boolean): void {
    if (this.ducked === ducked) return;
    this.ducked = ducked;
    this.applyMasterGain(ducked ? 0.06 : 0.12);
  }

  private applyMasterGain(ramp: number): void {
    if (!this.master || !this.ctx) return;
    const now = this.ctx.currentTime;
    this.master.gain.cancelScheduledValues(now);
    this.master.gain.setTargetAtTime(
      this.muted || this.ducked ? 0 : 0.85,
      now,
      ramp,
    );
  }

  /**
   * Resume context after a user gesture. Safe to call repeatedly.
   */
  async unlock(): Promise<boolean> {
    if (!this.ensure() || !this.ctx) return false;
    try {
      if (this.ctx.state === "suspended") {
        await this.ctx.resume();
      }
      this.unlocked = this.ctx.state === "running";
      if (this.unlocked && this.master) {
        this.master.gain.value = this.muted || this.ducked ? 0 : 0.85;
      }
      return this.unlocked;
    } catch {
      this.unavailable = true;
      return false;
    }
  }

  /**
   * Bind one-shot unlock on pointer/key gestures.
   */
  bindUnlockGestures(root: HTMLElement | Document = document): void {
    if (this.unlockBound || this.unavailable) return;
    this.unlockBound = true;
    const go = () => {
      void this.unlock();
    };
    this.unlockHandler = go;
    root.addEventListener("pointerdown", go, { capture: true });
    root.addEventListener("keydown", go, { capture: true });
  }

  unbindUnlockGestures(root: HTMLElement | Document = document): void {
    if (!this.unlockBound || !this.unlockHandler) return;
    root.removeEventListener("pointerdown", this.unlockHandler, { capture: true });
    root.removeEventListener("keydown", this.unlockHandler, { capture: true });
    this.unlockBound = false;
    this.unlockHandler = null;
  }

  dispose(): void {
    this.unbindUnlockGestures();
    try {
      void this.ctx?.close();
    } catch {
      /* ignore */
    }
    this.ctx = null;
    this.master = null;
    this.engineBus = null;
    this.tyreBus = null;
    this.windBus = null;
    this.ambientBus = null;
    this.voiceBus = null;
    this.unlocked = false;
  }
}
