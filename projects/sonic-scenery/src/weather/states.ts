/**
 * Per-WeatherState configuration: which sub-systems are active and with what
 * parameters. Keeping this declarative makes the state machine in
 * weatherSystem.ts a thin selector and keeps every state visually distinct.
 */
import type { WeatherState } from "../contracts";
import type { FogProfile } from "./fog";
import type { PrecipKind, PrecipProfile } from "./precipitation";
import { PRECIP_DEFAULTS } from "./precipitation";

export interface WeatherStateConfig {
  /** Precipitation kind + profile, or null for none. */
  precip: { kind: PrecipKind; profile: PrecipProfile } | null;
  fog: FogProfile;
  /** Whether lightning is armed in this state. */
  lightning: boolean;
  /**
   * Baseline intensity used when audio is offline, and the floor applied even
   * when audio is present so a state never looks empty. 0..1.
   */
  baselineIntensity: number;
  /** Ceiling for precipitation intensity (storm reaches 1, drizzle stays low). */
  maxIntensity: number;
  /** Horizontal wind push (world units/s) applied to precipitation. */
  wind: number;
}

function precip(kind: PrecipKind, overrides: Partial<PrecipProfile> = {}) {
  return { kind, profile: { ...PRECIP_DEFAULTS[kind], ...overrides } };
}

export const WEATHER_STATES: Record<WeatherState, WeatherStateConfig> = {
  clear: {
    precip: null,
    fog: { baseDensity: 0.0015, intensityDensity: 0.001, accentMix: 0.05 },
    lightning: false,
    baselineIntensity: 0.2,
    maxIntensity: 0.4,
    wind: 0,
  },
  cloudy: {
    precip: null,
    fog: { baseDensity: 0.004, intensityDensity: 0.004, accentMix: 0.1 },
    lightning: false,
    baselineIntensity: 0.4,
    maxIntensity: 0.7,
    wind: 1,
  },
  drizzle: {
    precip: precip("rain", { maxCount: 5000, speedMin: 35, speedMax: 50, size: 0.12, opacity: 0.4 }),
    fog: { baseDensity: 0.006, intensityDensity: 0.006, accentMix: 0.12 },
    lightning: false,
    baselineIntensity: 0.3,
    maxIntensity: 0.55,
    wind: 1.5,
  },
  rain: {
    precip: precip("rain"),
    fog: { baseDensity: 0.008, intensityDensity: 0.008, accentMix: 0.15 },
    lightning: false,
    baselineIntensity: 0.5,
    maxIntensity: 0.85,
    wind: 3,
  },
  storm: {
    precip: precip("rain", { maxCount: 14000, speedMin: 70, speedMax: 100, size: 0.2, opacity: 0.65 }),
    fog: { baseDensity: 0.012, intensityDensity: 0.012, accentMix: 0.2 },
    lightning: true,
    baselineIntensity: 0.7,
    maxIntensity: 1,
    wind: 8,
  },
  snow: {
    precip: precip("snow"),
    fog: { baseDensity: 0.009, intensityDensity: 0.007, accentMix: 0.1 },
    lightning: false,
    baselineIntensity: 0.45,
    maxIntensity: 0.9,
    wind: 2,
  },
  fog: {
    precip: null,
    fog: { baseDensity: 0.03, intensityDensity: 0.02, accentMix: 0.18 },
    lightning: false,
    baselineIntensity: 0.6,
    maxIntensity: 1,
    wind: 0.5,
  },
  ash: {
    precip: precip("ash"),
    fog: { baseDensity: 0.018, intensityDensity: 0.012, accentMix: 0.4 },
    lightning: false,
    baselineIntensity: 0.5,
    maxIntensity: 0.95,
    wind: 2,
  },
};
