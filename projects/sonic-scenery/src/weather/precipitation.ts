/**
 * GPU-resident precipitation using THREE.Points. A single BufferGeometry holds
 * a capped pool of particles; we recycle them in a spawn volume that follows
 * the camera/world origin so the player is always inside the weather.
 *
 * One Points cloud handles rain, snow, or ash by swapping kinematics + material:
 *  - rain: fast, near-vertical, thin bright streaks
 *  - snow: slow, lateral drift + sine sway, soft round flakes
 *  - ash:  slow, buoyant (rises/hangs), warm embers with flicker
 *
 * Visible particle count is driven by `intensity` (0..1) without re-allocating:
 * we simply draw the first N of the pool via `geometry.setDrawRange`, so going
 * from drizzle to storm never reallocates buffers (no GC hitch / pop).
 */
import * as THREE from "three";
import type { Palette } from "../contracts";
import { clamp, lerp, paletteColor } from "./util";

export type PrecipKind = "rain" | "snow" | "ash";

export interface PrecipProfile {
  kind: PrecipKind;
  /** Hard cap on pool size (allocated once). Tuned for 60fps. */
  maxCount: number;
  /** Fall speed range (world units / s). Negative vy = upward (ash). */
  speedMin: number;
  speedMax: number;
  /** Particle point size in world units. */
  size: number;
  /** Lateral drift magnitude (snow/ash). */
  drift: number;
  /** Base opacity at full intensity. */
  opacity: number;
  /** 0..1 how strongly to tint toward palette vs the kind's default color. */
  paletteMix: number;
}

/** Default per-kind profiles. Counts capped for a 60fps budget. */
export const PRECIP_DEFAULTS: Record<PrecipKind, PrecipProfile> = {
  rain: {
    kind: "rain",
    maxCount: 12000,
    speedMin: 55,
    speedMax: 80,
    size: 0.18,
    drift: 1.5,
    opacity: 0.55,
    paletteMix: 0.35,
  },
  snow: {
    kind: "snow",
    maxCount: 8000,
    speedMin: 3,
    speedMax: 7,
    size: 0.5,
    drift: 4,
    opacity: 0.9,
    paletteMix: 0.15,
  },
  ash: {
    kind: "ash",
    maxCount: 6000,
    speedMin: -4, // rises
    speedMax: 2, // some settle
    size: 0.35,
    drift: 3,
    opacity: 0.7,
    paletteMix: 0.6,
  },
};

const KIND_BASE_COLOR: Record<PrecipKind, number> = {
  rain: 0xaac4e0,
  snow: 0xffffff,
  ash: 0xff7b3a,
};

/** Half-extent of the cubic spawn volume centered on the follow target. */
const VOLUME = 90;
const VOLUME_HEIGHT = 70;

export class Precipitation {
  private points: THREE.Points | null = null;
  private geometry: THREE.BufferGeometry | null = null;
  private material: THREE.PointsMaterial | null = null;
  private scene: THREE.Scene | null = null;

  private profile: PrecipProfile = PRECIP_DEFAULTS.rain;
  private positions!: Float32Array;
  private velocities!: Float32Array; // vy per particle (xz drift derived from phase)
  private phase!: Float32Array; // per-particle sway phase
  private count = 0;

  /** Smoothed values so changes ramp instead of pop. */
  private intensity = 0;
  private opacity = 0;

  /** Center the weather follows (camera position, set by caller each frame). */
  private readonly center = new THREE.Vector3();

  init(scene: THREE.Scene, palette: Palette, profile: PrecipProfile): void {
    this.scene = scene;
    this.profile = profile;
    const n = profile.maxCount;

    this.positions = new Float32Array(n * 3);
    this.velocities = new Float32Array(n);
    this.phase = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      this.resetParticle(i, true);
    }

    this.geometry = new THREE.BufferGeometry();
    this.geometry.setAttribute("position", new THREE.BufferAttribute(this.positions, 3));
    this.geometry.setDrawRange(0, 0); // start empty; intensity ramps it up.

    const base = new THREE.Color(KIND_BASE_COLOR[profile.kind]);
    const tint = paletteColor(palette?.primary, "#ffffff");
    base.lerp(tint, profile.paletteMix);

    this.material = new THREE.PointsMaterial({
      color: base,
      size: profile.size,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: profile.kind === "ash" ? THREE.AdditiveBlending : THREE.NormalBlending,
    });

    this.points = new THREE.Points(this.geometry, this.material);
    this.points.frustumCulled = false; // volume always surrounds the camera
    this.points.renderOrder = 10;
    scene.add(this.points);
  }

  /** Re-tint on palette change without reallocating. */
  retint(palette: Palette): void {
    if (!this.material) return;
    const base = new THREE.Color(KIND_BASE_COLOR[this.profile.kind]);
    const tint = paletteColor(palette?.primary, "#ffffff");
    base.lerp(tint, this.profile.paletteMix);
    this.material.color.copy(base);
  }

  /** Set the world position the spawn volume should follow (typically camera). */
  setCenter(x: number, y: number, z: number): void {
    this.center.set(x, y, z);
  }

  private resetParticle(i: number, randomY: boolean): void {
    const i3 = i * 3;
    this.positions[i3] = this.center.x + (Math.random() - 0.5) * 2 * VOLUME;
    this.positions[i3 + 1] = randomY
      ? this.center.y + (Math.random() - 0.5) * 2 * VOLUME_HEIGHT
      : this.center.y + VOLUME_HEIGHT;
    this.positions[i3 + 2] = this.center.z + (Math.random() - 0.5) * 2 * VOLUME;
    this.velocities[i] = lerp(this.profile.speedMin, this.profile.speedMax, Math.random());
    this.phase[i] = Math.random() * Math.PI * 2;
  }

  /**
   * @param intensity 0..1 — fraction of the pool to render and base opacity.
   * @param wind extra horizontal push (e.g. storm gusts), world units/s.
   */
  update(dt: number, intensity: number, wind = 0, elapsed = 0): void {
    if (!this.points || !this.geometry) return;
    const p = this.profile;

    // Smoothed visible count + opacity → no popping on state/intensity changes.
    this.intensity = lerp(this.intensity, clamp(intensity), 1 - Math.exp(-3 * dt));
    this.opacity = lerp(this.opacity, clamp(intensity) * p.opacity, 1 - Math.exp(-4 * dt));
    if (this.material) this.material.opacity = this.opacity;

    const visible = Math.floor(this.intensity * p.maxCount);
    this.count = visible;
    this.geometry.setDrawRange(0, visible);
    if (visible === 0) return;

    const pos = this.positions;
    const ascending = p.speedMin < 0;
    const bottom = this.center.y - VOLUME_HEIGHT;
    const top = this.center.y + VOLUME_HEIGHT;

    for (let i = 0; i < visible; i++) {
      const i3 = i * 3;
      const ph = this.phase[i]!;
      const sway = Math.sin(elapsed * 0.6 + ph) * p.drift;

      // Read current position into locals (keeps the indexed access checked).
      const x = pos[i3]! + sway * dt + wind * dt;
      // Vertical motion (rain/snow fall = -vy; ash with negative speed rises).
      const y = pos[i3 + 1]! - this.velocities[i]! * dt;
      const z = pos[i3 + 2]! + Math.cos(elapsed * 0.5 + ph) * p.drift * 0.5 * dt;
      pos[i3] = x;
      pos[i3 + 1] = y;
      pos[i3 + 2] = z;

      // Recycle when out of the volume, or when too far laterally.
      const dx = x - this.center.x;
      const dz = z - this.center.z;
      const outVert = ascending ? y > top : y < bottom;
      const outLat = Math.abs(dx) > VOLUME || Math.abs(dz) > VOLUME;
      if (outVert || outLat) {
        this.resetParticle(i, false);
        if (ascending) {
          // ash respawns near the ground and rises
          pos[i3 + 1] = this.center.y - VOLUME_HEIGHT * 0.6;
        }
      }
    }

    (this.geometry.getAttribute("position") as THREE.BufferAttribute).needsUpdate = true;
  }

  get visibleCount(): number {
    return this.count;
  }

  dispose(): void {
    if (this.points && this.scene) this.scene.remove(this.points);
    this.geometry?.dispose();
    this.material?.dispose();
    this.points = null;
    this.geometry = null;
    this.material = null;
    this.scene = null;
  }
}
