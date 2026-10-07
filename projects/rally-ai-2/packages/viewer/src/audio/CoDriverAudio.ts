/**
 * Procedural co-driver pace calls — timed from the same pace note text the
 * HUD reads (`frame.note`). Synthesised beeps/formants only (no samples).
 */

import type { PaceNote } from "../replay";
import type { AudioBus } from "./AudioBus";

export type PaceDir = PaceNote["dir"] | "none";

export type CoDriverAudioState = {
  pace_dir?: PaceDir;
  pace_note?: string;
};

export class CoDriverAudio {
  private bus: AudioBus;
  private started = false;
  private muted = false;
  private lastKey = "";
  private coolT = 0;

  constructor(bus: AudioBus) {
    this.bus = bus;
  }

  start(): void {
    if (this.started || !this.bus.ensure()) return;
    this.started = true;
  }

  /** Clear call memory after seek/restart so notes can fire again. */
  reset(): void {
    this.lastKey = "";
    this.coolT = 0;
  }

  update(state: CoDriverAudioState, dt: number): void {
    if (!this.bus.available) return;
    if (!this.started) this.start();
    if (!this.started || !this.bus.ready) return;
    if (this.muted) return;

    if (this.coolT > 0) this.coolT = Math.max(0, this.coolT - dt);

    const dir = state.pace_dir ?? paceDirFromNote(state.pace_note ?? "");
    const note = (state.pace_note ?? "").trim();
    if (dir === "none" && !note) {
      this.lastKey = "";
      return;
    }

    const key = `${dir}|${note}`;
    if (key === this.lastKey || this.coolT > 0) return;
    this.lastKey = key;
    this.coolT = 0.85;

    const severity = severityFromNote(note);
    this.speak(dir, severity, note.length > 0);
  }

  setMuted(m: boolean): void {
    this.muted = m;
  }

  dispose(): void {
    this.started = false;
    this.lastKey = "";
  }

  private speak(dir: PaceDir, severity: number, hasLabel: boolean): void {
    const ctx = this.bus.ctx;
    const out = this.bus.voiceBus;
    if (!ctx || !out) return;
    const t0 = ctx.currentTime;

    if (dir === "left") {
      this.tone(ctx, out, t0, 520, 380, 0.09, 0.11);
      this.tone(ctx, out, t0 + 0.1, 440, 300, 0.08, 0.1);
      if (severity > 0.55) this.tone(ctx, out, t0 + 0.2, 360, 260, 0.07, 0.09);
    } else if (dir === "right") {
      this.tone(ctx, out, t0, 360, 480, 0.09, 0.11);
      this.tone(ctx, out, t0 + 0.1, 420, 560, 0.08, 0.1);
      if (severity > 0.55) this.tone(ctx, out, t0 + 0.2, 500, 640, 0.07, 0.09);
    } else if (dir === "caution") {
      this.buzz(ctx, out, t0, 180, 0.16, 0.12);
      this.tone(ctx, out, t0 + 0.18, 700, 500, 0.06, 0.08);
    } else if (dir === "crest" || dir === "jump") {
      this.tone(ctx, out, t0, 300, 720, 0.14, 0.1);
      this.tone(ctx, out, t0 + 0.12, 480, 900, 0.1, 0.08);
    } else if (dir === "straight" || dir === "dip") {
      this.tone(ctx, out, t0, 400, 400, 0.08, 0.07);
    } else if (hasLabel) {
      this.tone(ctx, out, t0, 450, 420, 0.09, 0.08);
    }
  }

  private tone(
    ctx: AudioContext,
    out: AudioNode,
    t0: number,
    f0: number,
    f1: number,
    dur: number,
    level: number,
  ): void {
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    const filt = ctx.createBiquadFilter();
    osc.type = "triangle";
    filt.type = "bandpass";
    filt.frequency.value = (f0 + f1) * 0.5;
    filt.Q.value = 2.2;
    osc.connect(filt);
    filt.connect(g);
    g.connect(out);
    osc.frequency.setValueAtTime(f0, t0);
    osc.frequency.exponentialRampToValueAtTime(Math.max(40, f1), t0 + dur);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(level, t0 + 0.015);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }

  private buzz(
    ctx: AudioContext,
    out: AudioNode,
    t0: number,
    f0: number,
    dur: number,
    level: number,
  ): void {
    const osc = ctx.createOscillator();
    const g = ctx.createGain();
    osc.type = "square";
    osc.frequency.value = f0;
    osc.connect(g);
    g.connect(out);
    g.gain.setValueAtTime(0.0001, t0);
    g.gain.exponentialRampToValueAtTime(level, t0 + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
    osc.start(t0);
    osc.stop(t0 + dur + 0.02);
  }
}

/** Derive pace direction from the recorded note text (same cues as the HUD). */
export function paceDirFromNote(note: string): PaceDir {
  const lower = note.toLowerCase();
  if (!lower) return "none";
  if (lower.includes("jump")) return "jump";
  if (lower.includes("crest")) return "crest";
  if (lower.includes("caution") || lower.includes("care") || lower.includes("danger")) {
    return "caution";
  }
  if (lower.includes("dip")) return "dip";
  if (lower.includes("left")) return "left";
  if (lower.includes("right")) return "right";
  if (lower.includes("straight")) return "straight";
  return "none";
}

function severityFromNote(note: string): number {
  const n = note.toLowerCase();
  if (!n) return 0.4;
  if (/\b(1|hairpin|tight)\b/.test(n)) return 0.95;
  if (/\b(2|acute)\b/.test(n)) return 0.8;
  if (/\b(3)\b/.test(n)) return 0.65;
  if (/\b(4|5|6|opens|long)\b/.test(n)) return 0.45;
  if (/caution|care|danger/.test(n)) return 0.9;
  return 0.5;
}
