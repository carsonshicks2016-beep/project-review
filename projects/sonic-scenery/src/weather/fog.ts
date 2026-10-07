/**
 * Volumetric-ish fog. We can't afford true volumetrics in the per-frame budget,
 * so we lean on THREE.FogExp2 (exponential distance fog) whose color is pulled
 * from the world palette, and whose density is smoothed toward a per-state
 * target. The "volumetric" feel comes from coupling fog color to the scene
 * background and letting density breathe gently with audio energy.
 */
import * as THREE from "three";
import type { Palette } from "../contracts";
import { ambientTint, damp, lerp } from "./util";

export interface FogProfile {
  /** Base exponential density when intensity === 0. */
  baseDensity: number;
  /** Additional density at intensity === 1. */
  intensityDensity: number;
  /** 0..1 — how far the fog color is pushed toward the palette accent. */
  accentMix: number;
}

export class FogController {
  private fog: THREE.FogExp2 | null = null;
  private scene: THREE.Scene | null = null;
  private readonly color = new THREE.Color();
  private readonly targetColor = new THREE.Color();
  private density = 0;
  private targetDensity = 0;
  private profile: FogProfile = {
    baseDensity: 0,
    intensityDensity: 0,
    accentMix: 0,
  };
  /** Whether the previous background was ours to restore. */
  private prevBackground: THREE.Scene["background"] = null;
  private ownsBackground = false;

  init(scene: THREE.Scene, palette: Palette, profile: FogProfile): void {
    this.scene = scene;
    this.profile = profile;
    this.prevBackground = scene.background;

    const tint = ambientTint(palette);
    const accent = new THREE.Color(palette?.accent ?? "#ffffff");
    this.targetColor.copy(tint).lerp(accent, profile.accentMix);
    this.color.copy(this.targetColor);
    this.density = 0; // ramp up from 0 so a freshly-mounted world doesn't pop.
    this.targetDensity = profile.baseDensity;

    this.fog = new THREE.FogExp2(this.color.getHex(), this.density);
    scene.fog = this.fog;

    // Only tint the background when the renderer hasn't set one (e.g. a skybox).
    if (scene.background === null) {
      scene.background = this.color.clone();
      this.ownsBackground = true;
    }
  }

  /** Reconfigure colors/profile (e.g. on a palette or state change). */
  retarget(palette: Palette, profile: FogProfile): void {
    this.profile = profile;
    const tint = ambientTint(palette);
    const accent = new THREE.Color(palette?.accent ?? "#ffffff");
    this.targetColor.copy(tint).lerp(accent, profile.accentMix);
  }

  /**
   * @param intensity 0..1 weather intensity.
   * @param flash 0..1 lightning flash amount (briefly brightens fog).
   */
  update(dt: number, intensity: number, flash = 0): void {
    if (!this.fog) return;
    this.targetDensity = this.profile.baseDensity + this.profile.intensityDensity * intensity;
    // Smooth density and color so state/intensity changes never pop.
    this.density = damp(this.density, this.targetDensity, 2.5, dt);
    this.fog.density = this.density;

    this.color.lerp(this.targetColor, 1 - Math.exp(-3 * dt));
    const lit = flash > 0 ? this.color.clone().lerp(new THREE.Color(0xffffff), 0.6 * flash) : this.color;
    this.fog.color.copy(lit);

    if (this.ownsBackground && this.scene && this.scene.background instanceof THREE.Color) {
      // Background tracks fog but stays slightly darker for depth.
      this.scene.background.copy(this.color).multiplyScalar(lerp(0.75, 1, flash));
    }
  }

  dispose(): void {
    if (this.scene) {
      this.scene.fog = null;
      if (this.ownsBackground) {
        this.scene.background = this.prevBackground;
      }
    }
    this.fog = null;
    this.scene = null;
  }
}
