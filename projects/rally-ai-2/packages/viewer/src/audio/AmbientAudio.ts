/**
 * Biome-dependent ambient beds: procedural noise, not music.
 * Crossfades on surface/biome change; sits under engine on ambientBus.
 */

import type { Stage, SurfaceKind } from "../replay";
import type { AudioBus } from "./AudioBus";
import { clamp, makePinkishNoise } from "./noise";

export type AmbientBiome = "snow" | "forest" | "tarmac";

export type AmbientAudioState = {
  surface?: SurfaceKind | undefined;
  speed?: number;
};

/** Map frame surface → ambient bed. Gravel/mud use forest hush. */
export function surfaceToBiome(surface?: string | null): AmbientBiome {
  if (surface === "snow") return "snow";
  if (surface === "tarmac") return "tarmac";
  return "forest";
}

/** Stage-level default when frames lack surface yet. */
export function stageToBiome(stage: Stage | null | undefined): AmbientBiome {
  if (!stage) return "forest";
  const id = (stage.id || stage.name || "").toLowerCase();
  if (id.includes("snow")) return "snow";
  if (id.includes("tarmac") || id.includes("asphalt")) return "tarmac";
  const surfaces = stage.surfaces ?? [];
  if (surfaces.length) {
    const total =
      surfaces.reduce((a, s) => a + Math.max(0, s.s_end - s.s_start), 0) || 1;
    const snow = surfaces
      .filter((s) => s.type === "snow")
      .reduce((a, s) => a + Math.max(0, s.s_end - s.s_start), 0);
    const tar = surfaces
      .filter((s) => s.type === "tarmac")
      .reduce((a, s) => a + Math.max(0, s.s_end - s.s_start), 0);
    if (snow / total > 0.45) return "snow";
    if (tar / total > 0.55) return "tarmac";
  }
  return "forest";
}

type Bed = {
  biome: AmbientBiome;
  src: AudioBufferSourceNode;
  filter: BiquadFilterNode;
  filter2: BiquadFilterNode | null;
  gain: GainNode;
};

export class AmbientAudio {
  private bus: AudioBus;
  private started = false;
  private beds = new Map<AmbientBiome, Bed>();
  private current: AmbientBiome = "forest";
  private target: AmbientBiome = "forest";
  private fade = 1;
  private muted = false;
  private noiseBuf: AudioBuffer | null = null;

  constructor(bus: AudioBus) {
    this.bus = bus;
  }

  start(): void {
    if (this.started || !this.bus.ensure()) return;
    const ctx = this.bus.ctx!;
    const out = this.bus.ambientBus!;
    this.noiseBuf = makePinkishNoise(ctx, 2.5);
    this.started = true;

    this.beds.set("snow", this.makeBed(ctx, out, "snow"));
    this.beds.set("forest", this.makeBed(ctx, out, "forest"));
    this.beds.set("tarmac", this.makeBed(ctx, out, "tarmac"));

    for (const bed of this.beds.values()) {
      bed.gain.gain.value = bed.biome === this.current ? 1 : 0;
    }
    for (const bed of this.beds.values()) {
      bed.src.start();
    }
  }

  setBiome(biome: AmbientBiome): void {
    if (biome === this.target) return;
    this.target = biome;
    this.fade = 0;
  }

  setFromStage(stage: Stage | null | undefined): void {
    this.setBiome(stageToBiome(stage));
    this.current = this.target;
    this.fade = 1;
    this.applyGainsImmediate();
  }

  update(state: AmbientAudioState, dt: number): void {
    if (!this.bus.available) return;
    if (!this.started) this.start();
    if (!this.started || !this.bus.ready) return;
    if (this.muted) return;

    if (state.surface) {
      this.setBiome(surfaceToBiome(state.surface));
    }

    const fadeSec = 1.1;
    if (this.fade < 1) {
      this.fade = Math.min(1, this.fade + dt / fadeSec);
      if (this.fade >= 1) this.current = this.target;
    }

    const speedN = clamp((state.speed ?? 0) / 50, 0, 1);
    const bedLevel = 0.5 + speedN * 0.18;
    const ctx = this.bus.ctx!;
    const now = ctx.currentTime;
    const from = this.current;
    const to = this.target;
    const a = this.fade;
    const fromG = (1 - a) * bedLevel;
    const toG = a * bedLevel;

    for (const bed of this.beds.values()) {
      let g = 0;
      if (from === to) {
        g = bed.biome === to ? bedLevel : 0;
      } else if (bed.biome === to) g = toG;
      else if (bed.biome === from) g = fromG;
      bed.gain.gain.setTargetAtTime(g, now, 0.08);
    }
  }

  setMuted(m: boolean): void {
    this.muted = m;
    if (!m || !this.bus.ctx) return;
    const now = this.bus.ctx.currentTime;
    for (const bed of this.beds.values()) {
      bed.gain.gain.setTargetAtTime(0, now, 0.05);
    }
  }

  dispose(): void {
    for (const bed of this.beds.values()) {
      try {
        bed.src.stop();
      } catch {
        /* ignore */
      }
    }
    this.beds.clear();
    this.started = false;
    this.noiseBuf = null;
  }

  private applyGainsImmediate(): void {
    if (!this.started) return;
    for (const bed of this.beds.values()) {
      bed.gain.gain.value = bed.biome === this.current ? 1 : 0;
    }
  }

  private makeBed(ctx: AudioContext, out: AudioNode, biome: AmbientBiome): Bed {
    const src = ctx.createBufferSource();
    src.buffer = this.noiseBuf!;
    src.loop = true;

    const filter = ctx.createBiquadFilter();
    const gain = ctx.createGain();
    gain.gain.value = 0;
    let filter2: BiquadFilterNode | null = null;

    if (biome === "snow") {
      filter.type = "bandpass";
      filter.frequency.value = 900;
      filter.Q.value = 0.55;
      filter2 = ctx.createBiquadFilter();
      filter2.type = "highpass";
      filter2.frequency.value = 280;
      src.connect(filter);
      filter.connect(filter2);
      filter2.connect(gain);
      const trim = ctx.createGain();
      trim.gain.value = 0.7;
      gain.connect(trim);
      trim.connect(out);
      return { biome, src, filter, filter2, gain };
    }

    if (biome === "tarmac") {
      filter.type = "lowpass";
      filter.frequency.value = 420;
      filter.Q.value = 0.6;
      src.connect(filter);
      filter.connect(gain);
      const trim = ctx.createGain();
      trim.gain.value = 0.35;
      gain.connect(trim);
      trim.connect(out);
      return { biome, src, filter, filter2: null, gain };
    }

    filter.type = "bandpass";
    filter.frequency.value = 380;
    filter.Q.value = 0.8;
    filter2 = ctx.createBiquadFilter();
    filter2.type = "bandpass";
    filter2.frequency.value = 1600;
    filter2.Q.value = 0.5;
    const g2 = ctx.createGain();
    g2.gain.value = 0.35;
    src.connect(filter);
    filter.connect(gain);
    src.connect(filter2);
    filter2.connect(g2);
    g2.connect(gain);
    const trim = ctx.createGain();
    trim.gain.value = 0.55;
    gain.connect(trim);
    trim.connect(out);
    return { biome, src, filter, filter2, gain };
  }
}
