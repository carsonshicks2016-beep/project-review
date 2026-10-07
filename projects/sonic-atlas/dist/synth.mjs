import {
  C,
  SAMPLE_RATE,
  D_HEAD,
  clamp,
  smoothstep,
  slantRange,
  boomTime,
  timeWindow,
  retardedTimes,
  doppler,
  binauralITD,
  binauralILD,
  airAbsorptionAlpha
} from './physics.mjs';

/**
 * Direct Form II Transposed Biquad Filter.
 * Supports lowpass, highpass, and bandpass topologies.
 */
export class Biquad {
  constructor(type, fc, fs = SAMPLE_RATE, Q = 0.7071) {
    this.fs = fs;
    this.s1 = 0;
    this.s2 = 0;
    this.setParams(type, fc, Q);
  }

  setParams(type, fc, Q = 0.7071) {
    const w0 = 2 * Math.PI * clamp(fc, 1, this.fs * 0.49) / this.fs;
    const cosw0 = Math.cos(w0);
    const sinw0 = Math.sin(w0);
    const alpha = sinw0 / (2 * Math.max(0.01, Q));
    let b0, b1, b2, a0, a1, a2;

    if (type === 'lowpass') {
      b0 = (1 - cosw0) / 2;
      b1 = 1 - cosw0;
      b2 = (1 - cosw0) / 2;
      a0 = 1 + alpha;
      a1 = -2 * cosw0;
      a2 = 1 - alpha;
    } else if (type === 'highpass') {
      b0 = (1 + cosw0) / 2;
      b1 = -(1 + cosw0);
      b2 = (1 + cosw0) / 2;
      a0 = 1 + alpha;
      a1 = -2 * cosw0;
      a2 = 1 - alpha;
    } else if (type === 'bandpass') {
      b0 = sinw0 / 2;
      b1 = 0;
      b2 = -sinw0 / 2;
      a0 = 1 + alpha;
      a1 = -2 * cosw0;
      a2 = 1 - alpha;
    } else {
      b0 = 1; b1 = 0; b2 = 0; a0 = 1; a1 = 0; a2 = 0;
    }

    this.b0 = b0 / a0;
    this.b1 = b1 / a0;
    this.b2 = b2 / a0;
    this.a1 = a1 / a0;
    this.a2 = a2 / a0;
  }

  process(x) {
    const y = this.b0 * x + this.s1;
    this.s1 = this.b1 * x - this.a1 * y + this.s2;
    this.s2 = this.b2 * x - this.a2 * y;
    return y;
  }

  reset() {
    this.s1 = 0;
    this.s2 = 0;
  }
}

/**
 * 1. Psychoacoustic Sub-Bass Synthesizer (Virtual Pitch / Missing Fundamental)
 *
 * Sonic booms and deep aerodynamic events concentrate acoustic energy below 40 Hz,
 * which standard mobile and laptop speakers physically roll off.
 * This processor:
 *  - Detects low-frequency shock events (< 45 Hz) using a 4th-order Butterworth lowpass filter.
 *  - Tracks shock envelope with fast attack and controlled decay.
 *  - Generates 2nd (e.g. 60-80 Hz) and 3rd (e.g. 90-120 Hz) harmonic overtones via Chebyshev
 *    polynomial waveshaping and tuned resonant overtone filter banks.
 *  - Bandpass-shapes harmonics to the speaker sweet-spot (55-200 Hz) with soft saturation (tanh).
 *  - Blends synthesized harmonics back into the signal so listeners feel chest-thump impact.
 *
 * @param {Float32Array|number[]} samples - Input audio samples
 * @param {number} [sampleRate=SAMPLE_RATE] - Audio sample rate in Hz
 * @param {object} [options={}] - Customization options
 * @returns {Float32Array} Enhanced audio buffer with attached harmonics & envelope metadata
 */
export function generateSubBassHarmonics(samples, sampleRate = SAMPLE_RATE, options = {}) {
  const src = samples instanceof Float32Array ? samples : Float32Array.from(samples || []);
  const len = src.length;
  const out = new Float32Array(len);
  const harmonics = new Float32Array(len);
  const envelope = new Float32Array(len);

  if (len === 0) {
    out.harmonics = harmonics;
    out.envelope = envelope;
    out.shockDetected = false;
    return out;
  }

  const {
    subCutoff = 45,
    harmonicsGain = 0.65,
    drive = 1.8,
    blend = 1.0,
    shockThreshold = 0.08
  } = options;

  // 4th-order Butterworth low-pass filter (cascaded biquad stages at subCutoff)
  const lp1 = new Biquad('lowpass', subCutoff, sampleRate, 0.5412);
  const lp2 = new Biquad('lowpass', subCutoff, sampleRate, 1.3065);

  // Tuned resonant overtone filters (virtual pitch bank: 60 Hz, 90 Hz, 120 Hz)
  const bp60 = new Biquad('bandpass', 60, sampleRate, 5.0);
  const bp90 = new Biquad('bandpass', 90, sampleRate, 5.0);
  const bp120 = new Biquad('bandpass', 120, sampleRate, 5.0);

  // Bandpass filter to constrain harmonics to speaker sweet-spot (55 Hz to 200 Hz)
  const hpAudible = new Biquad('highpass', 52, sampleRate, 0.7071);
  const lpAudible = new Biquad('lowpass', 210, sampleRate, 0.7071);

  // Dual-time-constant envelope follower
  const alphaAtt = 1 - Math.exp(-1 / (sampleRate * 0.003)); // 3 ms attack
  const alphaRel = 1 - Math.exp(-1 / (sampleRate * 0.065)); // 65 ms release

  let env = 0;
  let lastSub = 0;
  let shockDetected = false;

  for (let i = 0; i < len; i++) {
    const x = src[i];

    // Isolate sub-bass shock signal below 45 Hz
    const sub = lp2.process(lp1.process(x));

    // Envelope follower
    const absSub = Math.abs(sub);
    if (absSub > env) {
      env += alphaAtt * (absSub - env);
    } else {
      env += alphaRel * (absSub - env);
    }
    envelope[i] = env;

    if (env > shockThreshold) {
      shockDetected = true;
    }

    // Normalized sub-bass component for wave-shaping
    const norm = env > 1e-4 ? clamp(sub / env, -1.0, 1.0) : 0;

    // Chebyshev polynomial frequency multipliers:
    // T2(u) = 2*u^2 - 1 (doubles frequency -> 2nd harmonic, e.g. 30 Hz -> 60 Hz)
    // T3(u) = 4*u^3 - 3*u (triples frequency -> 3rd harmonic, e.g. 30 Hz -> 90 Hz)
    const h2 = (2 * norm * norm - 1);
    const h3 = (4 * norm * norm * norm - 3 * norm);
    const chebyshevHarmonics = (0.60 * h2 + 0.40 * h3) * env;

    // Resonant virtual-pitch overtone excitation (shock transient + sub-bass)
    const shockDiff = (sub - lastSub) * (sampleRate / 1000);
    lastSub = sub;
    const excitation = sub + 0.08 * shockDiff;

    const r60 = bp60.process(excitation);
    const r90 = bp90.process(excitation);
    const r120 = bp120.process(excitation);
    const resonantOvertones = 0.50 * r60 + 0.35 * r90 + 0.15 * r120;

    // Combined raw harmonic content
    const rawHarmonics = chebyshevHarmonics + 0.75 * resonantOvertones;

    // Bandpass to audible speaker sweet-spot (55 - 200 Hz) to reject DC / sub-40Hz
    const sweetHarmonics = lpAudible.process(hpAudible.process(rawHarmonics));

    // Soft saturation (tanh waveshaping) preventing digital clipping and adding warmth
    const satHarmonics = Math.tanh(sweetHarmonics * drive) * harmonicsGain;
    harmonics[i] = satHarmonics;

    // Blend synthesized harmonics back with dry input, softly saturated and bounded
    const mixed = x + blend * satHarmonics;
    out[i] = 0.96 * Math.tanh(mixed);
  }

  out.harmonics = harmonics;
  out.envelope = envelope;
  out.shockDetected = shockDetected;
  return out;
}

/**
 * 2. Mobile Haptic Shock Feedback (Web Vibration API)
 *
 * Generates a synchronized dual-vibration pulse for the bow shock and tail shock.
 * Falls back safely when the Web Vibration API is unsupported (desktop browsers, node).
 *
 * @param {number} [delayMs=120] - Time interval in ms between bow shock and tail shock
 * @param {number} [intensity=1.0] - Vibration intensity scale factor (0.0 to 1.0)
 * @returns {boolean} True if vibration was triggered successfully, false otherwise
 */
export function triggerHapticBoom(delayMs = 120, intensity = 1.0) {
  const d = Math.max(0, Math.round(Number(delayMs ?? 120)));
  const inten = clamp(Number(intensity ?? 1.0), 0, 1);

  if (inten <= 0) {
    triggerHapticBoom.lastPattern = [0, 0, 0];
    return false;
  }

  // Bow shock (sharp lead pulse) and tail shock (recompression pulse)
  const d1 = Math.max(1, Math.round(35 * inten));
  const d2 = Math.max(1, Math.round(25 * inten));
  const pattern = [d1, d, d2];
  triggerHapticBoom.lastPattern = pattern;

  if (typeof navigator !== 'undefined' && typeof navigator.vibrate === 'function') {
    try {
      return Boolean(navigator.vibrate(pattern));
    } catch {
      return false;
    }
  }
  return false;
}

/**
 * Computes the dual-pulse vibration pattern [bowDurationMs, delayMs, tailDurationMs].
 */
export function getHapticBoomPattern(delayMs = 120, intensity = 1.0) {
  const d = Math.max(0, Math.round(Number(delayMs ?? 120)));
  const inten = clamp(Number(intensity ?? 1.0), 0, 1);
  if (inten <= 0) return [0, 0, 0];
  const d1 = Math.max(1, Math.round(35 * inten));
  const d2 = Math.max(1, Math.round(25 * inten));
  return [d1, d, d2];
}

/**
 * 3. Environmental Impulse Response Synthesizer (Reverberation)
 *
 * Synthesizes stereo impulse response buffers for Web Audio ConvolverNode:
 *  - 'desert': Dry open salt flats / desert (minimal reflection, pure direct sound).
 *  - 'canyon': Mountain valley / canyon (discrete delayed flutter echoes at 180 ms, 320 ms, 490 ms with HF absorption).
 *  - 'urban': Urban airfield (dense early reflections off ground and buildings, 40-120 ms).
 *
 * @param {'desert'|'canyon'|'urban'} [environmentType='desert'] - Environment acoustic profile
 * @param {number} [sampleRate=SAMPLE_RATE] - Audio sample rate in Hz
 * @param {number} [duration=2.0] - Impulse response duration in seconds
 * @returns {{ channelL: Float32Array, channelR: Float32Array, environmentType: string, sampleRate: number, duration: number }}
 */
export function createImpulseResponse(environmentType = 'desert', sampleRate = SAMPLE_RATE, duration = 2.0) {
  const N = Math.max(256, Math.round(sampleRate * duration));
  const channelL = new Float32Array(N);
  const channelR = new Float32Array(N);

  const env = String(environmentType || 'desert').toLowerCase();

  // Helper to add a Gaussian-smoothed reflection pulse with stereo panning and HF absorption
  const addReflection = (tSec, ampL, ampR, spreadSec = 0.0006) => {
    const centerIdx = Math.round(tSec * sampleRate);
    const radius = Math.max(1, Math.round(spreadSec * 3 * sampleRate));
    for (let di = -radius; di <= radius; di++) {
      const idx = centerIdx + di;
      if (idx >= 0 && idx < N) {
        const dt = di / sampleRate;
        const weight = Math.exp(-0.5 * (dt / spreadSec) * (dt / spreadSec));
        channelL[idx] += ampL * weight;
        channelR[idx] += ampR * weight;
      }
    }
  };

  if (env === 'desert') {
    // Dry open desert / salt flats:
    // Pure direct sound at t = 0 plus hard ground reflection at ~2.2 ms with Gamma ~ 0.22
    channelL[0] = 1.0;
    channelR[0] = 1.0;

    const groundIdx = Math.max(1, Math.round(0.0022 * sampleRate));
    if (groundIdx < N) {
      channelL[groundIdx] += 0.22;
      channelR[groundIdx] += 0.22;
    }
  } else if (env === 'canyon') {
    // Mountain valley / canyon:
    // Direct sound + discrete delayed flutter echoes at 180 ms, 320 ms, 490 ms with HF absorption
    channelL[0] = 1.0;
    channelR[0] = 1.0;

    // Discrete flutter echo 1: 180 ms (panned left, moderate HF absorption)
    addReflection(0.180, 0.58, 0.36, 0.0007);

    // Discrete flutter echo 2: 320 ms (panned right, stronger HF absorption)
    addReflection(0.320, 0.30, 0.52, 0.0011);

    // Discrete flutter echo 3: 490 ms (diffuse center, deep HF absorption)
    addReflection(0.490, 0.35, 0.33, 0.0016);

    // Secondary canyon wall flutter reflections (multi-bounce canyon valley)
    const secondaryEchoes = [
      { t: 0.670, l: 0.22, r: 0.15, spread: 0.0022 },
      { t: 0.810, l: 0.14, r: 0.20, spread: 0.0028 },
      { t: 0.980, l: 0.16, r: 0.15, spread: 0.0035 },
      { t: 1.160, l: 0.10, r: 0.12, spread: 0.0042 }
    ];
    for (const sec of secondaryEchoes) {
      if (sec.t < duration) {
        addReflection(sec.t, sec.l, sec.r, sec.spread);
      }
    }

    // Canyon diffuse reverberant tail (filtered stereo decorrelated noise)
    let seedL = 13371;
    let seedR = 73319;
    let stateL = 0;
    let stateR = 0;
    const alphaAbs = 1 - Math.exp(-2 * Math.PI * 1100 / sampleRate); // 1100 Hz HF absorption

    for (let i = 0; i < N; i++) {
      const t = i / sampleRate;
      if (t < 0.05) continue;

      // PRNG for L and R channels
      seedL = (seedL * 1664525 + 1013904223) >>> 0;
      seedR = (seedR * 1664525 + 1013904223) >>> 0;
      const noiseL = (seedL / 2147483648) - 1;
      const noiseR = (seedR / 2147483648) - 1;

      // Lowpass absorption filter
      stateL += alphaAbs * (noiseL - stateL);
      stateR += alphaAbs * (noiseR - stateR);

      // Exponential decay envelope (T60 ~ 1.8s)
      const envTail = 0.14 * Math.exp(-2.2 * t) * smoothstep((t - 0.05) / 0.10);
      channelL[i] += stateL * envTail;
      channelR[i] += stateR * envTail;
    }
  } else {
    // 'urban': Urban airfield
    // Dense early reflections off ground and buildings between 40 ms and 120 ms
    channelL[0] = 1.0;
    channelR[0] = 1.0;

    // Cluster of 16 dense early reflections between 40 ms and 120 ms
    const urbanDelaysMs = [
      42, 47, 52, 57, 63, 68, 74, 80, 85, 91, 96, 102, 107, 112, 117, 122
    ];

    for (let k = 0; k < urbanDelaysMs.length; k++) {
      const tSec = urbanDelaysMs[k] / 1000;
      if (tSec >= duration) break;
      const decayFactor = 0.40 * Math.pow(0.040 / tSec, 0.75);
      // Alternating stereo panning across urban facade
      const pan = Math.sin(k * 1.7);
      const ampL = decayFactor * (0.5 + 0.45 * pan);
      const ampR = decayFactor * (0.5 - 0.45 * pan);
      const spread = 0.0004 + 0.00008 * k;
      addReflection(tSec, ampL, ampR, spread);
    }

    // Dense diffuse reverberation tail after 100 ms (airfield buildings and tarmac scattering)
    let seedL = 987654;
    let seedR = 456789;
    let stateL = 0;
    let stateR = 0;
    const alphaAbs = 1 - Math.exp(-2 * Math.PI * 1800 / sampleRate); // 1800 Hz building reflection absorption

    for (let i = 0; i < N; i++) {
      const t = i / sampleRate;
      if (t < 0.040) continue;

      seedL = (seedL * 1664525 + 1013904223) >>> 0;
      seedR = (seedR * 1664525 + 1013904223) >>> 0;
      const noiseL = (seedL / 2147483648) - 1;
      const noiseR = (seedR / 2147483648) - 1;

      stateL += alphaAbs * (noiseL - stateL);
      stateR += alphaAbs * (noiseR - stateR);

      const envTail = 0.22 * Math.exp(-3.2 * t) * smoothstep((t - 0.040) / 0.05);
      channelL[i] += stateL * envTail;
      channelR[i] += stateR * envTail;
    }
  }

  // Peak normalization across stereo channels
  let peak = 0;
  for (let i = 0; i < N; i++) {
    peak = Math.max(peak, Math.abs(channelL[i]), Math.abs(channelR[i]));
  }
  if (peak > 0) {
    const invPeak = 1.0 / peak;
    for (let i = 0; i < N; i++) {
      channelL[i] *= invPeak;
      channelR[i] *= invPeak;
    }
  }

  return {
    channelL,
    channelR,
    environmentType: env,
    sampleRate,
    duration
  };
}

/**
 * Helper to interpolate audio buffer at fractional sample indices.
 */
function interpolateSample(buf, idx, loop = true) {
  const len = buf.length;
  if (len === 0) return 0;
  if (!loop) {
    if (idx < 0 || idx >= len - 1) return 0;
    const i = Math.floor(idx);
    const f = idx - i;
    return buf[i] * (1 - f) + buf[i + 1] * f;
  }
  const wrapped = ((idx % len) + len) % len;
  const i = Math.floor(wrapped);
  const f = wrapped - i;
  const next = (i + 1) % len;
  return buf[i] * (1 - f) + buf[next] * f;
}

/**
 * 4. Custom Audio "Flyby-izer"
 *
 * Takes any arbitrary audio signal (e.g. siren, speech, music loop) and resamples
 * it through the exact retarded-time Doppler solver and distance attenuation.
 * Produces a physical flyby pass with true pitch warping, amplitude envelope,
 * binaural panning, and causality ahead of supersonic shocks.
 *
 * @param {Float32Array|number[]|object} sourceBuffer - Input arbitrary audio signal
 * @param {number} [sampleRate=SAMPLE_RATE] - Audio sample rate in Hz
 * @param {number} [mach=0.8] - Aircraft Mach number
 * @param {number} [altitude=500] - Flight altitude h in meters
 * @param {number} [lateralOffset=0] - Lateral observer offset y in meters
 * @param {object} [options={}] - Customization options
 * @returns {Float32Array} Mono pass buffer with attached .samplesL, .samplesR, and physical metadata
 */
export function flybyAudio(sourceBuffer, sampleRate = SAMPLE_RATE, mach = 0.8, altitude = 500, lateralOffset = 0, options = {}) {
  const m = Number(mach ?? 0.8);
  const h = Number(altitude ?? 500);
  const y = Number(lateralOffset ?? 0);
  const sr = Number(sampleRate || SAMPLE_RATE);

  // Extract source samples
  let src = null;
  if (sourceBuffer instanceof Float32Array) {
    src = sourceBuffer;
  } else if (Array.isArray(sourceBuffer)) {
    src = Float32Array.from(sourceBuffer);
  } else if (sourceBuffer && sourceBuffer.samples instanceof Float32Array) {
    src = sourceBuffer.samples;
  } else if (sourceBuffer && sourceBuffer.channelL instanceof Float32Array) {
    src = sourceBuffer.channelL;
  } else {
    src = new Float32Array(0);
  }

  const {
    loop = true,
    timeWindow: customTimeWindow,
    absorption = true,
    refDistance = 500,
    d_head = D_HEAD
  } = options;

  const win = customTimeWindow || timeWindow(m, h, y);
  const start = win.start;
  const end = win.end;
  const arrival = boomTime(m, h, y);
  const r_slant = slantRange(h, y);

  const len = Math.max(1, Math.ceil((end - start) * sr));
  const samplesL = new Float32Array(len);
  const samplesR = new Float32Array(len);
  const samples = new Float32Array(len);

  if (src.length === 0) {
    samples.samples = samples;
    samples.samplesL = samplesL;
    samples.samplesR = samplesR;
    samples.sampleRate = sr;
    samples.start = start;
    samples.end = end;
    samples.arrival = arrival;
    samples.peak = 0;
    samples.rms = 0;
    return samples;
  }

  let filterL = 0;
  let filterR = 0;
  let peak = 0;
  let energy = 0;

  for (let i = 0; i < len; i++) {
    const t = start + i / sr;

    // Ahead of supersonic shock wave arrival: strictly silent (causality)
    if (arrival !== null && t < arrival) {
      samples[i] = 0;
      samplesL[i] = 0;
      samplesR[i] = 0;
      continue;
    }

    let pL = 0;
    let pR = 0;
    let minDistance = r_slant;

    const taus = retardedTimes(t, m, h, y);
    for (const tau of taus) {
      const x = m * C * tau;
      const r = Math.hypot(x, r_slant);
      minDistance = Math.min(minDistance, r);

      const rate = doppler(tau, m, h, y);
      const jac = Math.sqrt(1 / (rate * rate) + 0.08 * 0.08);
      const gate = arrival === null ? 1 : smoothstep((t - arrival) / 0.04);

      // Binaural ITD and ILD
      const sinTheta = r > 0 ? clamp(x / r, -1, 1) : 0;
      const itd = binauralITD(sinTheta, d_head);
      const { ildL, ildR } = binauralILD(sinTheta, r);

      // Physical 1/r spherical spreading attenuation
      const distAmp = refDistance / Math.max(1, r);
      const weight = (distAmp / jac) * gate;

      // Retarded times for left and right ears
      const tauL = tau - 0.5 * itd;
      const tauR = tau + 0.5 * itd;

      const sL = interpolateSample(src, tauL * sr, loop);
      const sR = interpolateSample(src, tauR * sr, loop);

      pL += sL * ildL * weight;
      pR += sR * ildR * weight;
    }

    // Atmospheric air absorption
    if (absorption) {
      const alpha = airAbsorptionAlpha(minDistance, sr);
      filterL += alpha * (pL - filterL);
      filterR += alpha * (pR - filterR);
      pL = filterL;
      pR = filterR;
    }

    // Smooth boundary fade and soft saturation
    const fade = smoothstep((t - start) / 0.035) * smoothstep((end - t) / 0.08);
    const sL = 0.95 * Math.tanh(pL) * fade;
    const sR = 0.95 * Math.tanh(pR) * fade;
    const sMono = (sL + sR) * 0.5;

    samplesL[i] = sL;
    samplesR[i] = sR;
    samples[i] = sMono;

    const sPeak = Math.max(Math.abs(sL), Math.abs(sR), Math.abs(sMono));
    peak = Math.max(peak, sPeak);
    energy += (sL * sL + sR * sR) * 0.5;
  }

  const rms = Math.sqrt(energy / len);

  samples.samples = samples;
  samples.samplesL = samplesL;
  samples.samplesR = samplesR;
  samples.sampleRate = sr;
  samples.start = start;
  samples.end = end;
  samples.arrival = arrival;
  samples.peak = peak;
  samples.rms = rms;
  samples.mach = m;
  samples.altitude = h;
  samples.lateralOffset = y;

  return samples;
}
