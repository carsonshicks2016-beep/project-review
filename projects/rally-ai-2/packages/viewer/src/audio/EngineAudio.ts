/**
 * Procedural engine audio.
 * Oscillator stack driven by RPM; load/boost shape timbre; off-throttle overrun;
 * gear-change punctuation transient.
 *
 * RPM must be interpolated by the caller (Playback.sample) — stepping raw 30 Hz
 * frames here would zipper audibly.
 */

import type { AudioBus } from "./AudioBus";
import { clamp, makeWhiteNoise, smooth } from "./noise";

export type EngineAudioState = {
  rpm?: number;
  gear?: number;
  throttle?: number;
  brake?: number;
  speed?: number;
  boost?: number;
  airborne?: boolean;
};

export class EngineAudio {
  private bus: AudioBus;
  private started = false;

  private fundamental: OscillatorNode | null = null;
  private harmonic: OscillatorNode | null = null;
  private growl: OscillatorNode | null = null;
  private fundGain: GainNode | null = null;
  private harmGain: GainNode | null = null;
  private growlGain: GainNode | null = null;
  private growlFilter: BiquadFilterNode | null = null;
  private engineDuck: GainNode | null = null;

  private noiseSrc: AudioBufferSourceNode | null = null;
  private loadFilter: BiquadFilterNode | null = null;
  private loadGain: GainNode | null = null;

  private overrunOsc: OscillatorNode | null = null;
  private overrunGain: GainNode | null = null;

  private lastGear = 1;
  private shiftT = 0;
  private prevThrottle = 0;
  private overrunT = 0;
  private boostEst = 0;
  private muted = false;

  private curFund = 55;
  private curVol = 0;
  private curBright = 400;

  constructor(bus: AudioBus) {
    this.bus = bus;
  }

  /** Build graph once context is available. No-op if Web Audio missing. */
  start(): void {
    if (this.started || !this.bus.ensure()) return;
    const ctx = this.bus.ctx!;
    const out = this.bus.engineBus!;
    this.started = true;

    this.engineDuck = ctx.createGain();
    this.engineDuck.gain.value = 1;
    this.engineDuck.connect(out);

    this.fundGain = ctx.createGain();
    this.fundGain.gain.value = 0;
    this.fundGain.connect(this.engineDuck);
    this.fundamental = ctx.createOscillator();
    this.fundamental.type = "sawtooth";
    this.fundamental.frequency.value = 55;
    this.fundamental.connect(this.fundGain);
    this.fundamental.start();

    this.harmGain = ctx.createGain();
    this.harmGain.gain.value = 0;
    this.harmGain.connect(this.engineDuck);
    this.harmonic = ctx.createOscillator();
    this.harmonic.type = "square";
    this.harmonic.frequency.value = 110;
    this.harmonic.connect(this.harmGain);
    this.harmonic.start();

    this.growlFilter = ctx.createBiquadFilter();
    this.growlFilter.type = "lowpass";
    this.growlFilter.frequency.value = 400;
    this.growlFilter.Q.value = 1.2;
    this.growlGain = ctx.createGain();
    this.growlGain.gain.value = 0;
    this.growlFilter.connect(this.growlGain);
    this.growlGain.connect(this.engineDuck);
    this.growl = ctx.createOscillator();
    this.growl.type = "triangle";
    this.growl.frequency.value = 42;
    this.growl.connect(this.growlFilter);
    this.growl.start();

    const noiseBuf = makeWhiteNoise(ctx, 1.0);
    this.noiseSrc = ctx.createBufferSource();
    this.noiseSrc.buffer = noiseBuf;
    this.noiseSrc.loop = true;
    this.loadFilter = ctx.createBiquadFilter();
    this.loadFilter.type = "bandpass";
    this.loadFilter.frequency.value = 1800;
    this.loadFilter.Q.value = 0.7;
    this.loadGain = ctx.createGain();
    this.loadGain.gain.value = 0;
    this.noiseSrc.connect(this.loadFilter);
    this.loadFilter.connect(this.loadGain);
    this.loadGain.connect(out);
    this.noiseSrc.start();

    this.overrunGain = ctx.createGain();
    this.overrunGain.gain.value = 0;
    this.overrunGain.connect(this.engineDuck);
    this.overrunOsc = ctx.createOscillator();
    this.overrunOsc.type = "sawtooth";
    this.overrunOsc.frequency.value = 220;
    this.overrunOsc.connect(this.overrunGain);
    this.overrunOsc.start();
  }

  update(state: EngineAudioState, dt: number): void {
    if (!this.bus.available) return;
    if (!this.started) this.start();
    if (!this.started || !this.bus.ready) return;
    if (this.muted) return;

    const ctx = this.bus.ctx!;
    const speed = Math.max(0, state.speed ?? 0);
    const gear = Math.max(1, Math.round(state.gear ?? 1));
    const throttle = clamp(state.throttle ?? 0, 0, 1);
    const brake = clamp(state.brake ?? 0, 0, 1);
    const airborne = !!state.airborne;

    let rpm = state.rpm ?? 0;
    if (!Number.isFinite(rpm) || rpm < 200) {
      rpm = 900 + speed * (28 + gear * 9) + throttle * 1400 - brake * 400;
    }
    rpm = clamp(rpm, 700, 8200);
    const rpmN = clamp((rpm - 900) / (7800 - 900), 0, 1);
    const speedN = clamp(speed / 42, 0, 1);

    const boostTarget =
      state.boost != null && Number.isFinite(state.boost)
        ? clamp(state.boost, 0, 1)
        : throttle * throttle;
    this.boostEst = smooth(this.boostEst, boostTarget, dt, throttle > this.boostEst ? 3.2 : 1.6);
    const boost = this.boostEst;

    if (gear !== this.lastGear) {
      this.shiftT = 0.07;
      this.playShiftBlip(ctx);
      this.lastGear = gear;
    }
    if (this.shiftT > 0) this.shiftT = Math.max(0, this.shiftT - dt);

    const dThrottle = throttle - this.prevThrottle;
    if (dThrottle < -0.35 && rpmN > 0.45 && !airborne) {
      this.overrunT = 0.18 + rpmN * 0.12;
    }
    if (this.overrunT > 0) this.overrunT = Math.max(0, this.overrunT - dt);
    this.prevThrottle = throttle;

    const overrunLift = this.overrunT > 0 ? 18 + rpmN * 28 : 0;
    const targetFund =
      48 + rpmN * 130 + throttle * 14 - brake * 10 + boost * 8 + overrunLift - (airborne ? 6 : 0);
    this.curFund = smooth(this.curFund, targetFund, dt, 10);
    const fund = this.curFund;

    const idle = speedN < 0.02 && throttle < 0.05 ? 0.04 : 0;
    const targetVol =
      idle +
      0.06 +
      speedN * 0.14 +
      rpmN * 0.18 +
      throttle * 0.34 +
      boost * 0.1 -
      brake * 0.08 -
      (airborne ? 0.06 : 0);
    this.curVol = smooth(this.curVol, clamp(targetVol, 0, 0.78), dt, 8);

    const targetBright = 280 + rpmN * 900 + throttle * 700 + boost * 900;
    this.curBright = smooth(this.curBright, targetBright, dt, 6);

    const duck = this.shiftT > 0 ? 0.15 : 1;
    const now = ctx.currentTime;
    const t = 0.04;

    if (this.fundamental) this.fundamental.frequency.setTargetAtTime(fund, now, t);
    if (this.harmonic) this.harmonic.frequency.setTargetAtTime(fund * (2.02 + boost * 0.08), now, t);
    if (this.growl) this.growl.frequency.setTargetAtTime(fund * 0.5, now, t);
    if (this.growlFilter) this.growlFilter.frequency.setTargetAtTime(this.curBright, now, t);

    const harmMix = 0.1 + boost * 0.14;
    if (this.fundGain) this.fundGain.gain.setTargetAtTime(this.curVol * 0.45 * duck, now, t);
    if (this.harmGain) this.harmGain.gain.setTargetAtTime(this.curVol * harmMix * duck, now, t);
    if (this.growlGain) this.growlGain.gain.setTargetAtTime(this.curVol * 0.28 * duck, now, t);
    if (this.engineDuck) this.engineDuck.gain.setTargetAtTime(duck, now, 0.02);

    const load =
      throttle * (0.03 + boost * 0.05) + brake * (0.04 + speedN * 0.1) + (this.overrunT > 0 ? 0.04 : 0);
    if (this.loadGain) this.loadGain.gain.setTargetAtTime(load, now, 0.05);
    if (this.loadFilter) {
      this.loadFilter.frequency.setTargetAtTime(1200 + throttle * 900 + boost * 600 + brake * 400, now, 0.06);
    }

    const ov = this.overrunT > 0 ? 0.08 + rpmN * 0.1 : 0;
    if (this.overrunGain) this.overrunGain.gain.setTargetAtTime(ov, now, 0.02);
    if (this.overrunOsc) {
      this.overrunOsc.frequency.setTargetAtTime(fund * 2.4 + rpmN * 80, now, 0.03);
    }
  }

  setMuted(m: boolean): void {
    this.muted = m;
    if (!m || !this.bus.ctx) return;
    const now = this.bus.ctx.currentTime;
    this.fundGain?.gain.setTargetAtTime(0, now, 0.05);
    this.harmGain?.gain.setTargetAtTime(0, now, 0.05);
    this.growlGain?.gain.setTargetAtTime(0, now, 0.05);
    this.loadGain?.gain.setTargetAtTime(0, now, 0.05);
    this.overrunGain?.gain.setTargetAtTime(0, now, 0.05);
  }

  dispose(): void {
    try {
      this.fundamental?.stop();
      this.harmonic?.stop();
      this.growl?.stop();
      this.noiseSrc?.stop();
      this.overrunOsc?.stop();
    } catch {
      /* ignore */
    }
    this.fundamental = null;
    this.harmonic = null;
    this.growl = null;
    this.noiseSrc = null;
    this.overrunOsc = null;
    this.started = false;
  }

  private playShiftBlip(ctx: AudioContext): void {
    const out = this.bus.engineBus;
    if (!out) return;
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = 180 + Math.random() * 40;
    g.gain.value = 0.08;
    osc.connect(g);
    g.connect(out);
    const t0 = ctx.currentTime;
    g.gain.setValueAtTime(0.09, t0);
    g.gain.exponentialRampToValueAtTime(0.001, t0 + 0.05);
    osc.frequency.exponentialRampToValueAtTime(90, t0 + 0.05);
    osc.start(t0);
    osc.stop(t0 + 0.06);
  }
}
