export const C = 343;
export const SAMPLE_RATE = 24000;
export const GAMMA = 1.4;
export const R_AIR = 287.05;
export const T0_STD = 288.15;
export const LAPSE_RATE_STD = 0.0065;
export const D_HEAD = 0.18;
export const H_OBS = 1.7;
export const GAMMA_GROUND = 0.90;

export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));
export const smoothstep = x => { x = clamp(x, 0, 1); return x * x * (3 - 2 * x); };
export function regime(m) { return m < 0.8 ? 'SUBSONIC' : m < 1.2 ? 'TRANSONIC' : m < 5 ? 'SUPERSONIC' : 'HYPERSONIC'; }
export function coneAngle(m) { return m > 1 ? Math.asin(1 / m) : null; }

// Slant range with lateral observer offset y
export function slantRange(h, y = 0) {
  return Math.hypot(h, y);
}

// Boom arrival time with lateral observer offset y: t_boom = (r_slant / C) * sqrt(1 - 1/M²)
export function boomTime(m, h, y = 0) {
  const r = Math.hypot(h, y);
  return m > 1 ? (r / C) * Math.sqrt(1 - 1 / (m * m)) : null;
}
export const boomTimeLateral = boomTime;

// Sonic boom carpet half-width on the ground: carpetWidth = 2 * h * sqrt(M² - 1)
export function carpetWidth(m, h) {
  return m > 1 ? 2 * h * Math.sqrt(m * m - 1) : 0;
}

// Whitham ray-tube distance attenuation: peak(y) = peak_0 * (h / r_slant)**0.75
export function lateralAttenuation(h, y = 0) {
  const r = Math.hypot(h, y);
  return r === 0 ? 1 : Math.pow(h / r, 0.75);
}
export const whithamAttenuation = lateralAttenuation;

export function lateralOverpressure(peak0, h, y = 0) {
  return peak0 * lateralAttenuation(h, y);
}
export const whithamOverpressure = lateralOverpressure;

// Transonic condensation / Prandtl-Glauert vapor cone intensity
export function vaporConeIntensity(m) {
  if (m < 0.85 || m > 1.20) return 0;
  const d = Math.abs(m - 1.0);
  if (d <= 0.04) return 1.0;
  const decay = Math.exp(-Math.pow((d - 0.04) / 0.05, 2));
  return clamp(decay, 0, 1);
}

export function timeWindow(m, h, y = 0) {
  const r = Math.hypot(h, y);
  const b = r / C;
  return { start: -Math.max(3, 2.5 * b / Math.max(0.6, m)), end: Math.max(7, 4 * b + 2) };
}

// Stable quadratic solution of (1-M²)τ² - 2tτ + t² - (r_slant/c)² = 0.
// Squaring also introduces advanced roots: retain only causal solutions.
export function retardedTimes(t, m, h, y = 0) {
  const r = Math.hypot(h, y);
  const a = 1 - m * m, b = r / C;
  if (Math.abs(a) < 1e-10) { return t > 0 ? [(t * t - b * b) / (2 * t)] : []; }
  const disc = m * m * t * t + a * b * b;
  if (disc < 0 || (m > 1 && t <= 0)) return [];
  const d = Math.sqrt(disc), q = t + (t < 0 ? -d : d), roots = q === 0 ? [(t - d) / a, (t + d) / a] : [q / a, (t * t - b * b) / q];
  return roots.filter((tau, i) => Number.isFinite(tau) && tau <= t + 1e-9 && Math.abs(tau + Math.hypot(m * tau, b) - t) < 1e-7 && (!i || Math.abs(tau - roots[0]) > 1e-8)).sort((x, y) => x - y);
}

export function doppler(tau, m, h, y = 0) {
  const r = Math.hypot(m * tau, Math.hypot(h, y) / C);
  return 1 / Math.abs(1 + m * m * tau / r);
}

// Aircraft presets with specific physical and acoustic signatures
export const AIRCRAFT_PRESETS = {
  custom: {
    id: 'custom',
    name: 'Standard (Custom N-wave)',
    peak: 60,
    duration: 0.18,
    rise: 0.002,
    length: 20
  },
  f16: {
    id: 'f16',
    name: 'F-16 Falcon',
    peak: 75,
    duration: 0.095,
    rise: 0.0018,
    length: 15
  },
  concorde: {
    id: 'concorde',
    name: 'Concorde SST',
    peak: 105,
    duration: 0.24,
    rise: 0.002,
    length: 62
  },
  sr71: {
    id: 'sr71',
    name: 'SR-71 Blackbird',
    peak: 90,
    duration: 0.16,
    rise: 0.0015,
    length: 33
  },
  x59: {
    id: 'x59',
    name: 'NASA X-59 QueSST',
    peak: 30,
    duration: 0.22,
    rise: 0.015,
    length: 30
  }
};
AIRCRAFT_PRESETS.standard = AIRCRAFT_PRESETS.custom;

export function nWave(t, arrival, peak = 60, duration = 0.18, rise = 0.002) {
  if (arrival === null) return 0;
  const u = t - arrival;
  if (u < 0 || u > duration) return 0;
  const a = 2 * rise / duration, z = 12 / (6 + 9 * a + Math.sqrt((6 + 9 * a) ** 2 - 192 * a));
  const peakFactor = (1 - a * z) * smoothstep(z);
  return peak / peakFactor * (1 - 2 * u / duration) * smoothstep(u / rise) * smoothstep((duration - u) / rise);
}

// Low-boom shaped wave or Whitham N-wave ground signature
export function shapedWave(t, arrival, aircraftType = 'custom', peak, duration, rise) {
  const preset = AIRCRAFT_PRESETS[aircraftType] || AIRCRAFT_PRESETS.custom;
  const p = peak !== undefined ? peak : preset.peak;
  const d = duration !== undefined ? duration : preset.duration;
  const r = rise !== undefined ? rise : preset.rise;

  if (arrival === null) return 0;
  const u = t - arrival;
  if (u < 0 || u > d) return 0;

  if (aircraftType === 'x59') {
    const half = d * 0.5;
    const rEff = Math.min(r, half * 0.35);
    const u1 = rEff;
    const u2 = 1.5 * rEff;
    const u3 = 2.5 * rEff;

    const calcLobe = (x) => {
      if (x <= 0) return 0;
      if (x < u1) {
        return 0.4 * smoothstep(x / u1);
      } else if (x <= u2) {
        return 0.4;
      } else if (x < u3) {
        return 0.4 + 0.6 * smoothstep((x - u2) / (u3 - u2));
      } else if (x <= half) {
        const theta = (x - u3) / (half - u3);
        return Math.cos(theta * Math.PI * 0.5);
      }
      return 0;
    };

    if (u <= half) {
      return p * calcLobe(u);
    } else {
      return -p * calcLobe(d - u);
    }
  }

  return nWave(t, arrival, p, d, r);
}

// Atmospheric Temperature Lapse & Refraction:
// Effective sound speed c(z) = sqrt(gamma * R * (T0 - L*z))
export function effectiveSoundSpeed(z, lapseRate = LAPSE_RATE_STD, T0 = T0_STD) {
  const T = Math.max(1, T0 - lapseRate * z);
  return Math.sqrt(GAMMA * R_AIR * T);
}
export const soundSpeed = effectiveSoundSpeed;

// Ray curvature angle theta(z) = acos(c(z)/c0 * cos(theta0))
export function acousticRefractionRay(z, lapseRate = LAPSE_RATE_STD, theta0 = 0, T0 = T0_STD) {
  const c0 = effectiveSoundSpeed(0, lapseRate, T0);
  const cz = effectiveSoundSpeed(z, lapseRate, T0);
  const cosTheta = clamp((cz / c0) * Math.cos(theta0), -1, 1);
  const theta = Math.acos(cosTheta);
  return {
    z,
    theta,
    c: cz,
    c0,
    lapseRate,
    theta0,
    valueOf() { return this.theta; },
    toString() { return String(this.theta); }
  };
}

// Ground Reflection & Multipath Phase Interference:
export function groundReflection(tau, m, h, h_obs = H_OBS, Gamma = GAMMA_GROUND) {
  const x = m * C * tau;
  const r_dir = Math.hypot(x, h - h_obs);
  const r_refl = Math.hypot(x, h + h_obs);
  const deltaR = r_refl - r_dir;
  const deltaT = deltaR / C;
  return { r_dir, r_refl, deltaR, deltaT, Gamma };
}

// Atmospheric Air Absorption (ISO 9613-1 approximation):
export function airAbsorptionCutoff(distance, sampleRate = SAMPLE_RATE) {
  const fmax = sampleRate * 0.45;
  const fmin = 350;
  return clamp(fmax / (1 + distance / 1200), fmin, fmax);
}

export function airAbsorptionAlpha(distance, sampleRate = SAMPLE_RATE) {
  const fc = airAbsorptionCutoff(distance, sampleRate);
  const alpha = 1 - Math.exp(-2 * Math.PI * fc / sampleRate);
  return clamp(alpha, 1e-4, 1.0);
}

export class AirAbsorptionFilter {
  constructor(sampleRate = SAMPLE_RATE) {
    this.sampleRate = sampleRate;
    this.state = 0;
  }
  process(sample, distance) {
    const alpha = airAbsorptionAlpha(distance, this.sampleRate);
    this.state += alpha * (sample - this.state);
    return this.state;
  }
  reset() {
    this.state = 0;
  }
}

// Binaural Stereo Sound Rendering: ITD & ILD
export function binauralITD(sinTheta, d_head = D_HEAD) {
  return (d_head / C) * clamp(sinTheta, -1, 1);
}

export function binauralILD(sinTheta, distance = 500) {
  const s = clamp(sinTheta, -1, 1);
  const distFactor = clamp(1000 / (1000 + distance), 0.5, 1.0);
  const shadowDepth = 1.2 * distFactor;
  const ildL = s > 0 ? 1 / (1 + shadowDepth * s) : 1.0;
  const ildR = s < 0 ? 1 / (1 + shadowDepth * (-s)) : 1.0;
  return { ildL, ildR };
}

export function binauralParams(tau, m, h, d_head = D_HEAD) {
  const x = m * C * tau;
  const r = Math.hypot(x, h);
  const sinTheta = r > 0 ? clamp(x / r, -1, 1) : 0;
  const theta = Math.asin(sinTheta);
  const itd = binauralITD(sinTheta, d_head);
  const { ildL, ildR } = binauralILD(sinTheta, r);
  return { theta, sinTheta, itd, ildL, ildR, d_head };
}

function sincKernel(cutoff, length) {
  const out = [];
  let sum = 0;
  for (let k = 0; k < length; k++) {
    const x = k - (length - 1) / 2;
    const y = (x === 0 ? 2 * cutoff : Math.sin(2 * Math.PI * cutoff * x) / (Math.PI * x)) * (0.5 - 0.5 * Math.cos(2 * Math.PI * k / (length - 1)));
    out.push(y);
    sum += y;
  }
  return out.map(x => x / sum);
}

let sourceLevels;
function getSource() {
  if (sourceLevels) return sourceLevels;
  const n = 1 << 18, raw = new Float32Array(n), base = new Float32Array(n);
  let seed = 271828;
  for (let i = 0; i < n; i++) {
    seed ^= seed << 13;
    seed ^= seed >>> 17;
    seed ^= seed << 5;
    raw[i] = (seed >>> 0) / 2147483648 - 1;
  }
  const kernel = sincKernel(2200 / SAMPLE_RATE, 31);
  const tones = [105, 210, 480].map(f => Math.round(f * n / SAMPLE_RATE));
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < kernel.length; k++) s += raw[(i + k - 15 + n) % n] * kernel[k];
    base[i] = s * 1.8 + 0.10 * Math.sin(2 * Math.PI * tones[0] * i / n) + 0.06 * Math.sin(2 * Math.PI * tones[1] * i / n) + 0.035 * Math.sin(2 * Math.PI * tones[2] * i / n);
  }
  sourceLevels = [base];
  const down = sincKernel(0.22, 31);
  while (sourceLevels.at(-1).length > 32) {
    const prev = sourceLevels.at(-1), next = new Float32Array(prev.length / 2);
    for (let i = 0; i < next.length; i++) {
      let s = 0;
      for (let k = 0; k < down.length; k++) s += prev[(2 * i + k - 15 + prev.length) % prev.length] * down[k];
      next[i] = s;
    }
    sourceLevels.push(next);
  }
  return sourceLevels;
}

function periodicSample(array, index) {
  const len = array.length;
  index = ((index % len) + len) % len;
  const i = Math.floor(index), f = index - i;
  return array[i] * (1 - f) + array[(i + 1) % len] * f;
}

function jetSample(tau, rate, levels) {
  // A low-pass mip pyramid prevents a rapidly compressed source aliasing.
  const lod = clamp(Math.log2(Math.max(1, rate)), 0, levels.length - 1);
  const lo = Math.floor(lod), hi = Math.min(lo + 1, levels.length - 1), f = lod - lo;
  return periodicSample(levels[lo], tau * SAMPLE_RATE / 2 ** lo) * (1 - f) + periodicSample(levels[hi], tau * SAMPLE_RATE / 2 ** hi) * f;
}

function sampleTone(tau, rate, sampleRate) {
  const freq = 200 * rate;
  const aa = 1 - smoothstep((freq - sampleRate * 0.35) / (sampleRate * 0.1));
  return 0.35 * Math.sin(2 * Math.PI * 200 * tau) * aa;
}

export function renderAudio(config, sampleRate = SAMPLE_RATE) {
  const {
    mach: m,
    altitude: h,
    lateralOffset = 0,
    aircraft = 'custom',
    aircraftType,
    mode = 'jet',
    pressure,
    duration,
    rise,
    h_obs = H_OBS,
    Gamma = GAMMA_GROUND,
    d_head = D_HEAD
  } = config;

  const y = Number(lateralOffset || config.y || 0);
  const selectedAircraft = aircraftType || aircraft || 'custom';
  const preset = AIRCRAFT_PRESETS[selectedAircraft] || AIRCRAFT_PRESETS.custom;
  const basePeak = pressure !== undefined ? pressure : preset.peak;
  const baseDuration = duration !== undefined ? duration : preset.duration;
  const baseRise = rise !== undefined ? rise : preset.rise;

  const r_slant = slantRange(h, y);
  const effectivePressure = lateralOverpressure(basePeak, h, y);

  const { start, end } = timeWindow(m, h, y);
  const arrival = boomTime(m, h, y);
  const len = Math.ceil((end - start) * sampleRate);
  const samplesL = new Float32Array(len);
  const samplesR = new Float32Array(len);
  const samples = new Float32Array(len);
  const levels = mode === 'jet' ? getSource() : null;

  let filterStateL = 0;
  let filterStateR = 0;
  let peak = 0;
  let energy = 0;

  for (let i = 0; i < len; i++) {
    const t = start + i / sampleRate;

    // Ahead of supersonic shock wave arrival: strictly silent
    if (arrival !== null && t < arrival) {
      samples[i] = 0;
      samplesL[i] = 0;
      samplesR[i] = 0;
      continue;
    }

    let pL_jet = 0;
    let pR_jet = 0;
    let minDistance = r_slant;

    if (mode !== 'boom') {
      const taus = retardedTimes(t, m, h, y);
      for (const tau of taus) {
        const x = m * C * tau;
        const r_dir = Math.hypot(x, y, h - h_obs);
        const r_refl = Math.hypot(x, y, h + h_obs);
        const r_center = Math.hypot(x, r_slant);
        minDistance = Math.min(minDistance, r_dir);

        const rate = doppler(tau, m, h, y);
        const jac = Math.sqrt(1 / (rate * rate) + 0.08 * 0.08);
        const gate = arrival === null ? 1 : smoothstep((t - arrival) / 0.06);

        // Binaural angles and delays
        const sinTheta = r_center > 0 ? clamp(x / r_center, -1, 1) : 0;
        const itd = binauralITD(sinTheta, d_head);
        const { ildL, ildR } = binauralILD(sinTheta, r_center);

        // Ground reflection multipath delay
        const deltaTau = (r_refl - r_dir) / C;

        // Effective retarded times for L and R ears
        const tauL = tau - 0.5 * itd;
        const tauR = tau + 0.5 * itd;

        let srcL_dir = 0, srcR_dir = 0, srcL_refl = 0, srcR_refl = 0;
        if (mode === 'tone') {
          srcL_dir = sampleTone(tauL, rate, sampleRate);
          srcR_dir = sampleTone(tauR, rate, sampleRate);
          srcL_refl = sampleTone(tauL - deltaTau, rate, sampleRate);
          srcR_refl = sampleTone(tauR - deltaTau, rate, sampleRate);
        } else {
          srcL_dir = jetSample(tauL, rate, levels);
          srcR_dir = jetSample(tauR, rate, levels);
          srcL_refl = jetSample(tauL - deltaTau, rate, levels);
          srcR_refl = jetSample(tauR - deltaTau, rate, levels);
        }

        const amp_dir = 8000 / r_dir;
        const amp_refl = Gamma * (8000 / r_refl);

        pL_jet += ((srcL_dir * amp_dir + srcL_refl * amp_refl) * ildL / jac) * gate;
        pR_jet += ((srcR_dir * amp_dir + srcR_refl * amp_refl) * ildR / jac) * gate;
      }
    }

    // Atmospheric low-pass filter (ISO 9613-1 HF attenuation)
    const baseAlpha = airAbsorptionAlpha(minDistance, sampleRate);
    filterStateL += baseAlpha * (pL_jet - filterStateL);
    filterStateR += baseAlpha * (pR_jet - filterStateR);

    let pL = filterStateL;
    let pR = filterStateR;

    // Sonic boom: shaped wave or N-wave
    if (arrival !== null && effectivePressure > 0) {
      const mEff = Math.max(1.001, m);
      const sinThetaBoom = clamp(1 / mEff, -1, 1);
      const itdBoom = binauralITD(sinThetaBoom, d_head);
      const { ildL: ildLBoom, ildR: ildRBoom } = binauralILD(sinThetaBoom, r_slant);

      const arrivalL = arrival + Math.max(0, itdBoom);
      const arrivalR = arrival + Math.max(0, -itdBoom);

      pL += shapedWave(t, arrivalL, selectedAircraft, effectivePressure, baseDuration, baseRise) * ildLBoom;
      pR += shapedWave(t, arrivalR, selectedAircraft, effectivePressure, baseDuration, baseRise) * ildRBoom;
    }

    const pMono = (pL + pR) * 0.5;
    const fade = smoothstep((t - start) / 0.035) * smoothstep((end - t) / 0.08);
    const sL = 0.95 * Math.tanh(pL / 200) * fade;
    const sR = 0.95 * Math.tanh(pR / 200) * fade;
    const sMono = 0.95 * Math.tanh(pMono / 200) * fade;

    samplesL[i] = sL;
    samplesR[i] = sR;
    samples[i] = sMono;

    const samplePeak = Math.max(Math.abs(sL), Math.abs(sR), Math.abs(sMono));
    peak = Math.max(peak, samplePeak);
    energy += (sL * sL + sR * sR) * 0.5;
  }

  const rms = Math.sqrt(energy / len);
  return {
    samples,
    samplesL,
    samplesR,
    isStereo: true,
    start,
    end,
    arrival,
    sampleRate,
    peak,
    rms,
    r_slant,
    effectivePressure,
    carpetWidth: carpetWidth(m, h)
  };
}
