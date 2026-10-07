/**
 * Surface-dependent tyre noise gated by per-wheel slip.
 * Gravel/snow/mud peppering scales with speed on loose surfaces.
 * All synthesis via looped buffers + AudioParam automation (no render-thread alloc).
 */

import type { SurfaceKind, WheelFrame } from "../replay";
import type { AudioBus } from "./AudioBus";
import { clamp, makePinkishNoise, makeWhiteNoise, smooth } from "./noise";

export type TyreAudioState = {
  surface?: SurfaceKind | undefined;
  speed?: number;
  slip?: number;
  wheels?: WheelFrame[];
  airborne?: boolean;
};

export class TyreAudio {
  private bus: AudioBus;
  private started = false;
  private muted = false;

  private scrubSrc: AudioBufferSourceNode | null = null;
  private scrubFilter: BiquadFilterNode | null = null;
  private scrubGain: GainNode | null = null;

  private pepperSrc: AudioBufferSourceNode | null = null;
  private pepperFilter: BiquadFilterNode | null = null;
  private pepperGain: GainNode | null = null;
  private pepperGate: GainNode | null = null;

  private curScrub = 0;
  private curPepper = 0;
  private pepperPhase = 0;

  constructor(bus: AudioBus) {
    this.bus = bus;
  }

  start(): void {
    if (this.started || !this.bus.ensure()) return;
    const ctx = this.bus.ctx!;
    const out = this.bus.tyreBus!;
    this.started = true;

    const pink = makePinkishNoise(ctx, 2.0);
    const white = makeWhiteNoise(ctx, 1.5);

    this.scrubSrc = ctx.createBufferSource();
    this.scrubSrc.buffer = pink;
    this.scrubSrc.loop = true;
    this.scrubFilter = ctx.createBiquadFilter();
    this.scrubFilter.type = "bandpass";
    this.scrubFilter.frequency.value = 900;
    this.scrubFilter.Q.value = 0.85;
    this.scrubGain = ctx.createGain();
    this.scrubGain.gain.value = 0;
    this.scrubSrc.connect(this.scrubFilter);
    this.scrubFilter.connect(this.scrubGain);
    this.scrubGain.connect(out);
    this.scrubSrc.start();

    this.pepperSrc = ctx.createBufferSource();
    this.pepperSrc.buffer = white;
    this.pepperSrc.loop = true;
    this.pepperFilter = ctx.createBiquadFilter();
    this.pepperFilter.type = "highpass";
    this.pepperFilter.frequency.value = 2400;
    this.pepperFilter.Q.value = 0.7;
    this.pepperGate = ctx.createGain();
    this.pepperGate.gain.value = 0;
    this.pepperGain = ctx.createGain();
    this.pepperGain.gain.value = 0;
    this.pepperSrc.connect(this.pepperFilter);
    this.pepperFilter.connect(this.pepperGate);
    this.pepperGate.connect(this.pepperGain);
    this.pepperGain.connect(out);
    this.pepperSrc.start();
  }

  update(state: TyreAudioState, dt: number): void {
    if (!this.bus.available) return;
    if (!this.started) this.start();
    if (!this.started || !this.bus.ready) return;
    if (this.muted) return;

    const ctx = this.bus.ctx!;
    const now = ctx.currentTime;
    const surface = state.surface ?? "gravel";
    const speed = Math.max(0, state.speed ?? 0);
    const speedN = clamp(speed / 38, 0, 1);
    const airborne = !!state.airborne;

    const slip = meanWheelSlip(state);
    const gripUtil = meanGripUtil(state);

    let center = 900;
    let q = 0.85;
    let scrubScale = 0.22;
    let loose = 0;
    if (surface === "tarmac") {
      center = 1400;
      q = 1.1;
      scrubScale = 0.28;
      loose = 0;
    } else if (surface === "snow") {
      center = 700;
      q = 0.65;
      scrubScale = 0.2;
      loose = 0.85;
    } else if (surface === "mud") {
      center = 620;
      q = 0.7;
      scrubScale = 0.26;
      loose = 1.05;
    } else {
      center = 850;
      q = 0.75;
      scrubScale = 0.24;
      loose = 1;
    }

    const slipGate = clamp((slip - 0.04) / 0.55, 0, 1);
    const gripGate = clamp((gripUtil - 0.35) / 0.65, 0, 1);
    const targetScrub = airborne
      ? 0
      : (0.15 * speedN + slipGate * 0.85 + gripGate * 0.35) * scrubScale;

    this.curScrub = smooth(this.curScrub, targetScrub, dt, 10);
    if (this.scrubGain) this.scrubGain.gain.setTargetAtTime(this.curScrub, now, 0.05);
    if (this.scrubFilter) {
      this.scrubFilter.frequency.setTargetAtTime(center + slipGate * 400, now, 0.08);
      this.scrubFilter.Q.setTargetAtTime(q, now, 0.1);
    }

    const pepperAmt = airborne ? 0 : loose * speedN * (0.35 + slipGate * 0.65);
    this.curPepper = smooth(this.curPepper, pepperAmt * 0.16, dt, 6);
    if (this.pepperGain) this.pepperGain.gain.setTargetAtTime(this.curPepper, now, 0.06);
    if (this.pepperFilter) {
      this.pepperFilter.frequency.setTargetAtTime(surface === "snow" ? 1800 : 2600, now, 0.1);
    }

    this.pepperPhase += dt * (6 + speedN * 22 + slipGate * 10);
    const pulse = this.pepperPhase % 1;
    const open = pulse < 0.08 + slipGate * 0.06 ? 1 : 0.08;
    if (this.pepperGate) this.pepperGate.gain.setTargetAtTime(open * (pepperAmt > 0.02 ? 1 : 0), now, 0.01);
  }

  setMuted(m: boolean): void {
    this.muted = m;
    if (!m || !this.bus.ctx) return;
    const now = this.bus.ctx.currentTime;
    this.scrubGain?.gain.setTargetAtTime(0, now, 0.05);
    this.pepperGain?.gain.setTargetAtTime(0, now, 0.05);
  }

  dispose(): void {
    try {
      this.scrubSrc?.stop();
      this.pepperSrc?.stop();
    } catch {
      /* ignore */
    }
    this.scrubSrc = null;
    this.pepperSrc = null;
    this.started = false;
  }
}

function meanWheelSlip(state: TyreAudioState): number {
  const wheels = state.wheels;
  if (wheels && wheels.length) {
    let sum = 0;
    for (const w of wheels) {
      const sa = Math.abs(w.sa ?? 0);
      const sr = Math.abs(w.sr ?? 0);
      sum += 0.55 * sa + 0.45 * sr;
    }
    return sum / wheels.length;
  }
  return Math.abs(state.slip ?? 0);
}

function meanGripUtil(state: TyreAudioState): number {
  const wheels = state.wheels;
  if (!wheels || !wheels.length) return 0;
  let n = 0;
  let sum = 0;
  for (const w of wheels) {
    if (w.grip == null || !Number.isFinite(w.grip)) continue;
    sum += clamp(w.grip, 0, 1.5);
    n++;
  }
  return n ? sum / n : 0;
}
