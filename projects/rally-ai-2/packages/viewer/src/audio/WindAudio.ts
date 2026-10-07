/**
 * Speed-scaled wind: filtered noise, separate from biome ambience.
 */

import type { AudioBus } from "./AudioBus";
import { clamp, makePinkishNoise, smooth } from "./noise";

export type WindAudioState = {
  speed?: number;
  airborne?: boolean;
};

export class WindAudio {
  private bus: AudioBus;
  private started = false;
  private muted = false;
  private src: AudioBufferSourceNode | null = null;
  private filter: BiquadFilterNode | null = null;
  private gain: GainNode | null = null;
  private cur = 0;

  constructor(bus: AudioBus) {
    this.bus = bus;
  }

  start(): void {
    if (this.started || !this.bus.ensure()) return;
    const ctx = this.bus.ctx!;
    const out = this.bus.windBus!;
    this.started = true;

    this.src = ctx.createBufferSource();
    this.src.buffer = makePinkishNoise(ctx, 2.2);
    this.src.loop = true;
    this.filter = ctx.createBiquadFilter();
    this.filter.type = "bandpass";
    this.filter.frequency.value = 1100;
    this.filter.Q.value = 0.45;
    this.gain = ctx.createGain();
    this.gain.gain.value = 0;
    this.src.connect(this.filter);
    this.filter.connect(this.gain);
    this.gain.connect(out);
    this.src.start();
  }

  update(state: WindAudioState, dt: number): void {
    if (!this.bus.available) return;
    if (!this.started) this.start();
    if (!this.started || !this.bus.ready) return;
    if (this.muted) return;

    const speed = Math.max(0, state.speed ?? 0);
    const speedN = clamp(speed / 48, 0, 1);
    const air = state.airborne ? 1.25 : 1;
    const target = speedN * speedN * 0.42 * air;
    this.cur = smooth(this.cur, target, dt, 4);

    const ctx = this.bus.ctx!;
    const now = ctx.currentTime;
    if (this.gain) this.gain.gain.setTargetAtTime(this.cur, now, 0.08);
    if (this.filter) {
      this.filter.frequency.setTargetAtTime(700 + speedN * 1600, now, 0.12);
    }
  }

  setMuted(m: boolean): void {
    this.muted = m;
    if (!m || !this.bus.ctx) return;
    this.gain?.gain.setTargetAtTime(0, this.bus.ctx.currentTime, 0.05);
  }

  dispose(): void {
    try {
      this.src?.stop();
    } catch {
      /* ignore */
    }
    this.src = null;
    this.started = false;
  }
}
