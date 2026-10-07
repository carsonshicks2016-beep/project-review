/**
 * Lightning: a directional/point flash that fires on bass transients during a
 * storm. We don't simulate bolt geometry (too costly per frame for the payoff);
 * instead a high-intensity light bursts and decays, which the renderer's bloom
 * pass turns into a convincing sky flash. Exposes a 0..1 `flash` value the fog
 * controller uses to whiten the atmosphere in sync.
 *
 * Triggering: `audio.onset && audio.bass >= bassThreshold`, gated by a
 * refractory period so a sustained bass line doesn't strobe every frame.
 */
import * as THREE from "three";
import type { AudioFrame, Palette } from "../contracts";
import { clamp, damp, paletteColor } from "./util";

export interface LightningProfile {
  /** Minimum normalized bass to count as a strike trigger. */
  bassThreshold: number;
  /** Minimum seconds between strikes. */
  refractory: number;
  /** Peak light intensity on a strike. */
  peakIntensity: number;
  /** How fast the flash decays (higher = snappier). */
  decayRate: number;
  /** Chance per eligible onset that a strike actually fires (adds variety). */
  strikeChance: number;
}

export const LIGHTNING_DEFAULT: LightningProfile = {
  bassThreshold: 0.55,
  refractory: 0.45,
  peakIntensity: 6,
  decayRate: 6,
  strikeChance: 0.85,
};

export class Lightning {
  private light: THREE.DirectionalLight | null = null;
  private scene: THREE.Scene | null = null;
  private profile: LightningProfile = LIGHTNING_DEFAULT;

  /** 0..1 current flash brightness (also exposed to the fog). */
  private level = 0;
  private cooldown = 0;
  /** When offline, drive an occasional ambient flash so storms still feel alive. */
  private autoTimer = 0;
  private autoInterval = 4;

  init(scene: THREE.Scene, palette: Palette, profile: LightningProfile = LIGHTNING_DEFAULT): void {
    this.scene = scene;
    this.profile = profile;
    const color = paletteColor(palette?.accent, "#dfe9ff").lerp(new THREE.Color(0xffffff), 0.5);
    this.light = new THREE.DirectionalLight(color, 0);
    this.light.position.set(20, 80, 10);
    scene.add(this.light);
  }

  retint(palette: Palette): void {
    if (!this.light) return;
    this.light.color.copy(paletteColor(palette?.accent, "#dfe9ff").lerp(new THREE.Color(0xffffff), 0.5));
  }

  /** Force a strike (used by the auto/offline path and on demand). */
  strike(power = 1): void {
    this.level = clamp(power);
    this.cooldown = this.profile.refractory;
  }

  /**
   * @param active   whether the current weather is a storm.
   * @param audio    live frame, may be undefined (offline → auto flashes).
   * @param energy   smoothed 0..1 energy used to scale auto-flash cadence.
   * @returns current flash level 0..1 (for fog/atmosphere coupling).
   */
  update(dt: number, active: boolean, audio: AudioFrame | undefined, energy: number): number {
    if (!this.light) return 0;
    this.cooldown = Math.max(0, this.cooldown - dt);

    if (active) {
      if (audio) {
        if (
          audio.onset &&
          audio.bass >= this.profile.bassThreshold &&
          this.cooldown === 0 &&
          Math.random() < this.profile.strikeChance
        ) {
          // Stronger bass → brighter strike.
          this.strike(0.7 + 0.3 * clamp(audio.bass));
        }
      } else {
        // Offline: periodic auto strikes, faster when "energy" is high.
        this.autoInterval = 5.5 - 3.5 * clamp(energy); // 2..5.5s
        this.autoTimer += dt;
        if (this.autoTimer >= this.autoInterval && this.cooldown === 0) {
          this.autoTimer = 0;
          this.strike(0.7 + 0.3 * Math.random());
        }
      }
    }

    // Decay the flash toward 0 every frame.
    this.level = damp(this.level, 0, this.profile.decayRate, dt);
    if (this.level < 0.002) this.level = 0;
    this.light.intensity = this.level * this.profile.peakIntensity;
    return this.level;
  }

  get flash(): number {
    return this.level;
  }

  dispose(): void {
    if (this.light && this.scene) this.scene.remove(this.light);
    this.light?.dispose?.();
    this.light = null;
    this.scene = null;
  }
}
