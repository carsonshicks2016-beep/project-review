/**
 * Agent E — Weather system.  Brief: docs/tasks/agent-E-weather.md
 *
 *  - Drive a WeatherState machine selected by `world.weatherState`.
 *  - GPU-resident precipitation (rain/snow/ash via THREE.Points); fog from the
 *    world palette; lightning flashes on bass transients during `storm`.
 *  - Intensity ramps from live AudioFrame energy (rms/bands); when audio is
 *    undefined (helper offline) a steady per-state baseline runs instead.
 *  - All transitions are smoothed (critically-damped) so nothing pops.
 *
 * The renderer (Agent D) owns the scene/camera/loop and calls init → update*
 * → dispose. We assume standard Three.js conventions: a metric-ish world with
 * the camera near the origin. `setCameraPosition` lets the renderer keep the
 * weather volume centered on the viewer; if it's never called we sit at origin.
 */
import * as THREE from "three";
import type { AudioFrame, WeatherState, WorldSpec } from "../contracts";
import { FogController } from "./fog";
import { Lightning } from "./lightning";
import { Precipitation } from "./precipitation";
import { WEATHER_STATES } from "./states";
import { clamp, damp } from "./util";

export interface WeatherSystem {
  /** Build particle systems / fog for the given world. */
  init(scene: THREE.Scene, world: WorldSpec): void;
  /** Per-frame update; `audio` may be undefined if the helper is offline. */
  update(dt: number, audio?: AudioFrame): void;
  dispose(): void;
}

/** Optional extension surface the renderer can use; safe to ignore. */
export interface WeatherSystemEx extends WeatherSystem {
  /** Keep the precipitation volume centered on the viewer. */
  setCameraPosition(x: number, y: number, z: number): void;
  /** Override the active weather state (e.g. for transitions/presets). */
  setState(state: WeatherState): void;
}

class WeatherSystemImpl implements WeatherSystemEx {
  private scene: THREE.Scene | null = null;
  private world: WorldSpec | null = null;

  private readonly fog = new FogController();
  private readonly lightning = new Lightning();
  /** One reusable precipitation cloud; reconfigured per state. */
  private precip: Precipitation | null = null;
  private precipKind: WeatherState | null = null;

  private state: WeatherState = "clear";

  /** Smoothed 0..1 weather intensity (drives density/particle counts). */
  private intensity = 0;
  /** Smoothed 0..1 audio energy estimate. */
  private energy = 0;
  /** Seconds since init, for deterministic sway/auto-lightning timing. */
  private elapsed = 0;

  private readonly camera = new THREE.Vector3(0, 8, 0);
  private disposed = false;

  init(scene: THREE.Scene, world: WorldSpec): void {
    this.scene = scene;
    this.world = world;
    this.state = world.weatherState ?? "clear";
    this.disposed = false;

    const cfg = WEATHER_STATES[this.state];
    this.fog.init(scene, world.palette, cfg.fog);
    this.lightning.init(scene, world.palette);

    this.buildPrecip(cfg);

    // Start at the baseline so the first frame doesn't ramp from empty too hard,
    // but below target so there's still a gentle fade-in.
    this.energy = cfg.baselineIntensity * 0.5;
    this.intensity = 0;
  }

  /** Allocate / reconfigure the precipitation cloud for a state's config. */
  private buildPrecip(cfg: typeof WEATHER_STATES[WeatherState]): void {
    if (!this.scene || !this.world) return;
    // Tear down the old cloud only if the kind changed (avoids needless realloc).
    if (cfg.precip) {
      const sameKind =
        this.precip !== null && this.precipKind === this.state;
      if (!sameKind) {
        this.precip?.dispose();
        this.precip = new Precipitation();
        this.precip.init(this.scene, this.world.palette, cfg.precip.profile);
        this.precipKind = this.state;
      }
    } else {
      this.precip?.dispose();
      this.precip = null;
      this.precipKind = null;
    }
  }

  setCameraPosition(x: number, y: number, z: number): void {
    this.camera.set(x, y, z);
  }

  setState(state: WeatherState): void {
    if (this.disposed || state === this.state || !this.scene || !this.world) return;
    this.state = state;
    const cfg = WEATHER_STATES[state];
    // Retarget fog colors/profile and lightning tint; precipitation rebuilds
    // only when the particle KIND changes (rain↔snow↔ash↔none).
    this.fog.retarget(this.world.palette, cfg.fog);
    this.buildPrecip(cfg);
  }

  /**
   * Estimate a 0..1 "energy" from a frame. rms is the primary driver; we add a
   * little weighting from the band aggregates so percussive/full-spectrum
   * material reads as more intense.
   */
  private frameEnergy(audio: AudioFrame): number {
    const bandAvg = (audio.bass + audio.mid + audio.treble) / 3;
    return clamp(0.65 * audio.rms + 0.35 * bandAvg);
  }

  update(dt: number, audio?: AudioFrame): void {
    if (this.disposed || !this.scene) return;
    // Guard against pathological dt (tab restored, debugger pause, etc.).
    const step = clamp(dt, 0, 0.1);
    this.elapsed += step;

    const cfg = WEATHER_STATES[this.state];

    // --- Energy: live audio if present, else a steady baseline. ---
    const targetEnergy = audio ? this.frameEnergy(audio) : cfg.baselineIntensity;
    this.energy = damp(this.energy, targetEnergy, 3, step);

    // --- Intensity: map energy into the state's [baseline, max] band. ---
    const targetIntensity = clamp(
      cfg.baselineIntensity + (cfg.maxIntensity - cfg.baselineIntensity) * this.energy,
      0,
      cfg.maxIntensity
    );
    // Slower ramp than energy so precipitation density breathes, not flickers.
    this.intensity = damp(this.intensity, targetIntensity, 1.8, step);

    // --- Lightning (storm only); returns a 0..1 flash for atmosphere coupling.
    const flash = this.lightning.update(step, cfg.lightning, audio, this.energy);

    // --- Fog reacts to intensity and is briefly whitened by lightning. ---
    this.fog.update(step, this.intensity, flash);

    // --- Precipitation. ---
    if (this.precip) {
      this.precip.setCenter(this.camera.x, this.camera.y, this.camera.z);
      // Gusts scale with energy in stormy states.
      const wind = cfg.wind * (0.5 + this.energy);
      this.precip.update(step, this.intensity, wind, this.elapsed);
    }
  }

  dispose(): void {
    this.disposed = true;
    this.precip?.dispose();
    this.precip = null;
    this.lightning.dispose();
    this.fog.dispose();
    this.scene = null;
    this.world = null;
  }
}

export function createWeatherSystem(): WeatherSystem {
  return new WeatherSystemImpl();
}
