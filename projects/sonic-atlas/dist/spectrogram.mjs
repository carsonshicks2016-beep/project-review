/**
 * Mach Lab — Real-time FFT Spectrogram Waterfall Renderer
 * High-performance Short-Time Fourier Transform (STFT) in pure JavaScript.
 * Visualizes time-frequency acoustics, Doppler S-curves, sonic boom broadband impulses,
 * and atmospheric high-frequency absorption roll-off across 0 Hz to 12 kHz.
 */

// Twiddle factor and bit-reversal caches for Radix-2 FFT
const BIT_REV_CACHE = new Map();
const TWIDDLE_CACHE = new Map();

/**
 * Get or compute bit-reversal index table for size N (power of 2)
 */
export function getBitReversal(n) {
  let table = BIT_REV_CACHE.get(n);
  if (table) return table;

  table = new Uint32Array(n);
  let bits = 0;
  while ((1 << bits) < n) bits++;

  for (let i = 0; i < n; i++) {
    let rev = 0;
    for (let b = 0; b < bits; b++) {
      if ((i >> b) & 1) {
        rev |= 1 << (bits - 1 - b);
      }
    }
    table[i] = rev;
  }
  BIT_REV_CACHE.set(n, table);
  return table;
}

/**
 * Get or compute trigonometric tables for size N (power of 2)
 */
export function getTwiddleTable(n) {
  let table = TWIDDLE_CACHE.get(n);
  if (table) return table;

  const half = n >> 1;
  const cos = new Float64Array(half);
  const sin = new Float64Array(half);

  for (let i = 0; i < half; i++) {
    const angle = (-2.0 * Math.PI * i) / n;
    cos[i] = Math.cos(angle);
    sin[i] = Math.sin(angle);
  }

  table = { cos, sin };
  TWIDDLE_CACHE.set(n, table);
  return table;
}

/**
 * In-place Cooley-Tukey Radix-2 Decimation-in-Time Forward Fast Fourier Transform
 * @param {Float32Array|Float64Array} re Real part array (length must be power of 2)
 * @param {Float32Array|Float64Array} im Imaginary part array (length must be power of 2)
 */
export function fft(re, im) {
  const n = re.length;
  if ((n & (n - 1)) !== 0) {
    throw new Error(`FFT size must be a power of 2, received ${n}`);
  }
  if (im.length !== n) {
    throw new Error('Real and imaginary arrays must have identical length');
  }

  // 1. Bit-reversal permutation
  const rev = getBitReversal(n);
  for (let i = 0; i < n; i++) {
    const j = rev[i];
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
  }

  // 2. Butterfly combinations
  const twiddles = getTwiddleTable(n);
  for (let len = 2; len <= n; len <<= 1) {
    const half = len >> 1;
    const step = n / len;
    for (let i = 0; i < n; i += len) {
      for (let j = 0; j < half; j++) {
        const k = j * step;
        const cosVal = twiddles.cos[k];
        const sinVal = twiddles.sin[k]; // e^{-i 2pi k / N} = cosVal + i * sinVal

        const tr = re[i + half + j] * cosVal - im[i + half + j] * sinVal;
        const ti = re[i + half + j] * sinVal + im[i + half + j] * cosVal;

        const ur = re[i + j];
        const ui = im[i + j];

        re[i + j] = ur + tr;
        im[i + j] = ui + ti;
        re[i + half + j] = ur - tr;
        im[i + half + j] = ui - ti;
      }
    }
  }
}

/**
 * Hann (Hanning) window of length N
 * w[n] = 0.5 * (1 - cos(2*pi*n / (N - 1)))
 */
export function hannWindow(n) {
  const w = new Float32Array(n);
  const denom = n > 1 ? n - 1 : 1;
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / denom));
  }
  return w;
}

/**
 * Blackman window of length N
 * w[n] = 0.42 - 0.5*cos(2*pi*n / (N - 1)) + 0.08*cos(4*pi*n / (N - 1))
 */
export function blackmanWindow(n) {
  const w = new Float32Array(n);
  const denom = n > 1 ? n - 1 : 1;
  for (let i = 0; i < n; i++) {
    w[i] = 0.42 - 0.5 * Math.cos((2 * Math.PI * i) / denom) + 0.08 * Math.cos((4 * Math.PI * i) / denom);
  }
  return w;
}

/**
 * Window sum of squares (energy gain factor S2)
 */
export function windowEnergy(windowArray) {
  let s2 = 0;
  for (let i = 0; i < windowArray.length; i++) {
    s2 += windowArray[i] * windowArray[i];
  }
  return s2;
}

/**
 * Color palettes for scientific spectrogram waterfall rendering
 */
export const SPECTROGRAM_PALETTES = {
  inferno: [
    { stop: 0.00, r: 0,   g: 0,   b: 4   }, // Dark void
    { stop: 0.20, r: 40,  g: 11,  b: 84  }, // Deep violet
    { stop: 0.40, r: 101, g: 21,  b: 110 }, // Magenta
    { stop: 0.60, r: 187, g: 55,  b: 84  }, // Fiery red
    { stop: 0.80, r: 249, g: 142, b: 9   }, // Amber orange
    { stop: 0.95, r: 254, g: 231, b: 36  }, // Bright yellow
    { stop: 1.00, r: 255, g: 255, b: 255 }  // Pure white impulse
  ],
  cyberCyan: [
    { stop: 0.00, r: 4,   g: 10,  b: 18  }, // Deep obsidian navy
    { stop: 0.20, r: 8,   g: 32,  b: 54  }, // Dark ocean
    { stop: 0.45, r: 14,  g: 90,  b: 125 }, // Deep cyan
    { stop: 0.70, r: 34,  g: 180, b: 210 }, // Electric cyan
    { stop: 0.88, r: 103, g: 232, b: 249 }, // Neon aqua
    { stop: 0.96, r: 210, g: 250, b: 255 }, // High-energy shimmer
    { stop: 1.00, r: 255, g: 255, b: 255 }  // Shock flash
  ],
  amber: [
    { stop: 0.00, r: 12,  g: 6,   b: 2   }, // Deep umber void
    { stop: 0.25, r: 60,  g: 25,  b: 8   }, // Rich sepia
    { stop: 0.50, r: 150, g: 70,  b: 15  }, // Burnt amber
    { stop: 0.75, r: 230, g: 140, b: 20  }, // Golden orange
    { stop: 0.90, r: 250, g: 205, b: 60  }, // Bright gold
    { stop: 1.00, r: 255, g: 255, b: 240 }  // Solar white
  ],
  viridis: [
    { stop: 0.00, r: 68,  g: 1,   b: 84  }, // Deep indigo
    { stop: 0.25, r: 59,  g: 82,  b: 139 }, // Royal blue
    { stop: 0.50, r: 33,  g: 145, b: 140 }, // Teal
    { stop: 0.75, r: 94,  g: 201, b: 98  }, // Emerald green
    { stop: 1.00, r: 253, g: 231, b: 37  }  // Bright lemon
  ]
};

/**
 * Interpolate RGB color from palette given normalized value [0, 1]
 */
export function colormap(value, paletteName = 'inferno') {
  const stops = SPECTROGRAM_PALETTES[paletteName] || SPECTROGRAM_PALETTES.inferno;
  const v = Math.max(0, Math.min(1, value));

  if (v <= stops[0].stop) {
    const s = stops[0];
    return [s.r, s.g, s.b];
  }
  if (v >= stops[stops.length - 1].stop) {
    const s = stops[stops.length - 1];
    return [s.r, s.g, s.b];
  }

  for (let i = 0; i < stops.length - 1; i++) {
    const a = stops[i];
    const b = stops[i + 1];
    if (v >= a.stop && v <= b.stop) {
      const t = (v - a.stop) / (b.stop - a.stop);
      const r = Math.round(a.r + (b.r - a.r) * t);
      const g = Math.round(a.g + (b.g - a.g) * t);
      const bColor = Math.round(a.b + (b.b - a.b) * t);
      return [r, g, bColor];
    }
  }

  const last = stops[stops.length - 1];
  return [last.r, last.g, last.b];
}

/**
 * Fast Short-Time Fourier Transform (STFT) calculation
 */
export function computeSTFT(samples, sampleRate = 24000, options = {}) {
  const windowSize = options.windowSize || 512;
  const hopSize = options.hopSize || 128;
  const windowType = options.windowType || 'hann';
  const minDb = options.minDb ?? -80;
  const maxDb = options.maxDb ?? 0;

  if ((windowSize & (windowSize - 1)) !== 0) {
    throw new Error('windowSize must be a power of 2');
  }

  const rawSamples = samples instanceof Float32Array ? samples : new Float32Array(samples);
  const totalSamples = rawSamples.length;

  if (totalSamples < windowSize) {
    // Pad with zeros if sample length is smaller than window
    const padded = new Float32Array(windowSize);
    padded.set(rawSamples);
    return computeSTFT(padded, sampleRate, options);
  }

  const win = windowType === 'blackman' ? blackmanWindow(windowSize) : hannWindow(windowSize);
  const s2 = windowEnergy(win);
  const numBins = (windowSize >> 1) + 1; // 0 Hz (DC) to Nyquist
  const numFrames = Math.max(1, Math.floor((totalSamples - windowSize) / hopSize) + 1);

  // Allocate 2D grid: frames x bins
  const grid = new Array(numFrames);
  const powerGrid = new Array(numFrames);
  const dbGrid = new Array(numFrames);

  const re = new Float32Array(windowSize);
  const im = new Float32Array(windowSize);

  let globalMinPower = Infinity;
  let globalMaxPower = 0;
  let globalMinDb = Infinity;
  let globalMaxDb = -Infinity;

  // Single-sided PSD normalization factor
  // To conserve energy: sum_{k=0}^{N/2} P[k] = sum_{n=0}^{N-1} (x[n]*w[n])^2 / S2
  const psdNorm = 1.0 / (windowSize * s2);

  for (let m = 0; m < numFrames; m++) {
    const offset = m * hopSize;

    // Apply window to frame
    for (let n = 0; n < windowSize; n++) {
      re[n] = rawSamples[offset + n] * win[n];
      im[n] = 0.0;
    }

    // Compute FFT
    fft(re, im);

    const framePower = new Float32Array(numBins);
    const frameDb = new Float32Array(numBins);
    const frameDensity = new Float32Array(numBins);

    // Compute power spectral density across 0 Hz to Nyquist
    for (let k = 0; k < numBins; k++) {
      const magSq = re[k] * re[k] + im[k] * im[k];
      // Single-sided spectrum: double intermediate bin power
      const scale = (k === 0 || k === numBins - 1) ? 1.0 : 2.0;
      const power = Math.max(0.0, magSq * psdNorm * scale);

      framePower[k] = power;
      if (power < globalMinPower) globalMinPower = power;
      if (power > globalMaxPower) globalMaxPower = power;

      // Power in decibels with regularization floor
      const db = 10.0 * Math.log10(power + 1e-12);
      frameDb[k] = db;
      if (db < globalMinDb) globalMinDb = db;
      if (db > globalMaxDb) globalMaxDb = db;
    }

    // Map to normalized density [0, 1]
    const dbRange = maxDb - minDb;
    for (let k = 0; k < numBins; k++) {
      const clamped = Math.max(minDb, Math.min(maxDb, frameDb[k]));
      frameDensity[k] = (clamped - minDb) / dbRange;
    }

    grid[m] = frameDensity;
    powerGrid[m] = framePower;
    dbGrid[m] = frameDb;
  }

  // Precalculate timestamp array (center of each frame in seconds)
  const times = new Float32Array(numFrames);
  for (let m = 0; m < numFrames; m++) {
    times[m] = (m * hopSize + windowSize * 0.5) / sampleRate;
  }

  // Precalculate frequency bin centers (Hz)
  const frequencies = new Float32Array(numBins);
  const freqStep = sampleRate / windowSize;
  for (let k = 0; k < numBins; k++) {
    frequencies[k] = k * freqStep;
  }

  return {
    numFrames,
    numBins,
    sampleRate,
    windowSize,
    hopSize,
    windowType,
    times,
    frequencies,
    grid,            // 2D density grid [frame][bin] in [0, 1]
    powerGrid,       // 2D raw PSD [frame][bin] >= 0
    dbGrid,          // 2D dB values [frame][bin]
    minPower: globalMinPower === Infinity ? 0 : globalMinPower,
    maxPower: globalMaxPower,
    minDb: globalMinDb,
    maxDb: globalMaxDb,
    duration: totalSamples / sampleRate,
    freqStep,
    nyquist: sampleRate * 0.5,
    getPower(frame, bin) {
      if (frame >= 0 && frame < numFrames && bin >= 0 && bin < numBins) {
        return powerGrid[frame][bin];
      }
      return 0;
    },
    getDensity(frame, bin) {
      if (frame >= 0 && frame < numFrames && bin >= 0 && bin < numBins) {
        return grid[frame][bin];
      }
      return 0;
    },
    getDb(frame, bin) {
      if (frame >= 0 && frame < numFrames && bin >= 0 && bin < numBins) {
        return dbGrid[frame][bin];
      }
      return minDb;
    }
  };
}

/**
 * Theoretical Doppler frequency trajectory for comparison and overlay
 * f(t) = f0 * doppler(tau(t), m, h)
 */
export function theoreticalDopplerCurve(m, h, tList, f0 = 200, c = 343) {
  const curve = [];
  const b = h / c;

  for (const t of tList) {
    if (m > 1 && t <= 0) {
      curve.push({ t, f: null, ratio: null });
      continue;
    }

    const a = 1 - m * m;
    let tau = null;

    if (Math.abs(a) < 1e-9) {
      if (t > 0) tau = (t * t - b * b) / (2 * t);
    } else {
      const disc = m * m * t * t + a * b * b;
      if (disc >= 0) {
        const d = Math.sqrt(disc);
        const roots = [(t - d) / a, (t + d) / a];
        // Select causal retarded time
        const valid = roots.filter(r => r <= t + 1e-6 && Math.abs(r + Math.hypot(m * r, b) - t) < 1e-4);
        if (valid.length > 0) {
          tau = valid[valid.length - 1]; // later emission branch
        }
      }
    }

    if (tau !== null) {
      const r = Math.hypot(m * tau, b);
      const ratio = 1 / Math.abs(1 + (m * m * tau) / r);
      curve.push({ t, f: f0 * ratio, ratio });
    } else {
      curve.push({ t, f: null, ratio: null });
    }
  }

  return curve;
}

/**
 * SpectrogramRenderer Class
 * Orchestrates STFT computation and real-time canvas waterfall rendering.
 */
export class SpectrogramRenderer {
  /**
   * @param {Object} options Configuration parameters
   * @param {number} [options.windowSize=512] FFT window size (power of 2)
   * @param {number} [options.hopSize=128] STFT hop size
   * @param {string} [options.windowType='hann'] 'hann' or 'blackman'
   * @param {string} [options.palette='inferno'] 'inferno' | 'cyberCyan' | 'amber' | 'viridis'
   * @param {number} [options.minDb=-80] Minimum dB threshold for waterfall display
   * @param {number} [options.maxDb=0] Maximum dB threshold
   * @param {boolean} [options.showTicks=true] Render frequency tick marks
   * @param {boolean} [options.showGrid=true] Render horizontal frequency grid lines
   * @param {boolean} [options.showPlayhead=true] Render synchronized playhead marker
   */
  constructor(options = {}) {
    this.windowSize = options.windowSize || 512;
    this.hopSize = options.hopSize || 128;
    this.windowType = options.windowType || 'hann';
    this.palette = options.palette || 'inferno';
    this.minDb = options.minDb ?? -80;
    this.maxDb = options.maxDb ?? 0;
    this.showTicks = options.showTicks ?? true;
    this.showGrid = options.showGrid ?? true;
    this.showPlayhead = options.showPlayhead ?? true;
    this.cachedCanvas = null;
    this.lastSpectrogramData = null;
  }

  /**
   * Generates a 2D time-frequency density grid from raw audio samples.
   * @param {Float32Array|Array<number>|Object} audioSamples Raw audio samples or audio result object
   * @param {number} [sampleRate=24000] Sample rate in Hz
   * @returns {Object} Spectrogram data object containing density grid, frequencies, and timestamps
   */
  generateSpectrogramData(audioSamples, sampleRate = 24000) {
    let samples = audioSamples;
    let sr = sampleRate;

    // Handle audio result object from physics.mjs / audio-worker
    if (audioSamples && typeof audioSamples === 'object' && !Array.isArray(audioSamples) && !(audioSamples instanceof Float32Array)) {
      if (audioSamples.samples) {
        samples = audioSamples.samples;
      } else if (audioSamples.samplesL && audioSamples.samplesR) {
        // Average stereo channels into mono
        const len = audioSamples.samplesL.length;
        const mono = new Float32Array(len);
        for (let i = 0; i < len; i++) {
          mono[i] = 0.5 * (audioSamples.samplesL[i] + audioSamples.samplesR[i]);
        }
        samples = mono;
      }
      if (audioSamples.sampleRate) {
        sr = audioSamples.sampleRate;
      }
    }

    const data = computeSTFT(samples, sr, {
      windowSize: this.windowSize,
      hopSize: this.hopSize,
      windowType: this.windowType,
      minDb: this.minDb,
      maxDb: this.maxDb
    });

    this.lastSpectrogramData = data;
    return data;
  }

  /**
   * Draws the real-time spectrogram waterfall to a canvas context.
   * Includes frequency tick marks (100 Hz, 500 Hz, 1 kHz, 5 kHz, 10 kHz),
   * time axis, and playback playhead marker.
   *
   * @param {HTMLCanvasElement|Object} canvas HTML Canvas or mock canvas
   * @param {Object} spectrogramData 2D density grid returned by generateSpectrogramData
   * @param {number} currentTime Current playback simulation time (seconds)
   * @param {Object} timeWindow Simulation time range { start, end }
   * @param {Object} [options={}] Additional drawing options
   */
  drawSpectrogram(canvas, spectrogramData, currentTime = 0, timeWindow = null, options = {}) {
    if (!canvas) return;
    const ctx = canvas.getContext ? canvas.getContext('2d') : canvas;
    if (!ctx) return;

    const data = spectrogramData || this.lastSpectrogramData;
    if (!data || !data.grid || data.grid.length === 0) return;

    const width = canvas.width || 600;
    const height = canvas.height || 260;

    const paletteName = options.palette || this.palette;
    const tStart = timeWindow ? timeWindow.start : (options.startTime ?? 0);
    const tEnd = timeWindow ? timeWindow.end : (options.endTime ?? data.duration);
    const duration = Math.max(1e-4, tEnd - tStart);

    // Margins for scientific instrumentation display
    const marginLeft = this.showTicks ? (options.marginLeft ?? 54) : 0;
    const marginRight = options.marginRight ?? 14;
    const marginTop = options.marginTop ?? 16;
    const marginBottom = options.marginBottom ?? 24;

    const plotWidth = Math.max(10, width - marginLeft - marginRight);
    const plotHeight = Math.max(10, height - marginTop - marginBottom);

    // Clear background
    if (ctx.fillStyle !== undefined) {
      ctx.fillStyle = options.backgroundColor || '#080e14';
      ctx.fillRect(0, 0, width, height);
    }

    // Render Waterfall Grid into an offscreen image buffer or columns
    const numFrames = data.numFrames;
    const numBins = data.numBins;
    const maxFreq = data.nyquist; // 12000 Hz

    // Render columns matching plot width
    for (let px = 0; px < plotWidth; px++) {
      const u = px / plotWidth;
      const targetTime = tStart + u * duration;

      // Find closest STFT frame
      // Map time to frame index
      const frameFrac = (targetTime - (data.times[0] ?? 0)) / (data.duration || 1);
      const frameIdx = Math.max(0, Math.min(numFrames - 1, Math.floor(frameFrac * numFrames)));

      const frameDensity = data.grid[frameIdx];
      if (!frameDensity) continue;

      for (let py = 0; py < plotHeight; py++) {
        // Vertical axis: frequency 0 at bottom, maxFreq (12 kHz) at top
        const v = 1.0 - (py / plotHeight);
        const freqTarget = v * maxFreq;

        const binIdx = Math.max(0, Math.min(numBins - 1, Math.round((freqTarget / maxFreq) * (numBins - 1))));
        const density = frameDensity[binIdx];

        if (density > 0.01) {
          const [r, g, b] = colormap(density, paletteName);
          ctx.fillStyle = `rgb(${r},${g},${b})`;
          ctx.fillRect(marginLeft + px, marginTop + py, 1, 1);
        }
      }
    }

    // Draw Frequency Ticks & Grid Lines: 100 Hz, 500 Hz, 1 kHz, 5 kHz, 10 kHz
    const requiredTicks = [
      { freq: 100,   label: '100 Hz' },
      { freq: 500,   label: '500 Hz' },
      { freq: 1000,  label: '1 kHz'  },
      { freq: 5000,  label: '5 kHz'  },
      { freq: 10000, label: '10 kHz' },
      { freq: 12000, label: '12 kHz' }
    ];

    if (this.showGrid || this.showTicks) {
      for (const tick of requiredTicks) {
        if (tick.freq > maxFreq) continue;
        const yNorm = 1.0 - (tick.freq / maxFreq);
        const y = marginTop + yNorm * plotHeight;

        // Subtle horizontal grid line
        if (this.showGrid && ctx.beginPath && ctx.stroke) {
          ctx.beginPath();
          ctx.strokeStyle = tick.freq === 1000 ? 'rgba(255,255,255,0.22)' : 'rgba(255,255,255,0.10)';
          ctx.lineWidth = 1;
          if (ctx.setLineDash) ctx.setLineDash(tick.freq === 1000 ? [] : [2, 4]);
          ctx.moveTo(marginLeft, y);
          ctx.lineTo(marginLeft + plotWidth, y);
          ctx.stroke();
          if (ctx.setLineDash) ctx.setLineDash([]);
        }

        // Axis Tick & Label
        if (this.showTicks && ctx.fillText) {
          ctx.font = '10px "DM Sans", -apple-system, BlinkMacSystemFont, monospace';
          ctx.fillStyle = tick.freq === 1000 || tick.freq === 10000 ? '#a5b7c0' : '#657e8c';
          ctx.textAlign = 'right';
          ctx.textBaseline = 'middle';
          ctx.fillText(tick.label, marginLeft - 6, y);

          // Small tick mark
          if (ctx.beginPath && ctx.stroke) {
            ctx.beginPath();
            ctx.strokeStyle = '#51656c';
            ctx.moveTo(marginLeft - 4, y);
            ctx.lineTo(marginLeft, y);
            ctx.stroke();
          }
        }
      }
    }

    // Time Axis Ticks at Bottom
    if (this.showTicks && ctx.fillText) {
      const timeSteps = 5;
      for (let i = 0; i <= timeSteps; i++) {
        const frac = i / timeSteps;
        const tVal = tStart + frac * duration;
        const x = marginLeft + frac * plotWidth;

        ctx.font = '9px "DM Sans", -apple-system, BlinkMacSystemFont, monospace';
        ctx.fillStyle = '#657e8c';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const label = (tVal >= 0 ? '+' : '') + tVal.toFixed(1) + ' s';
        ctx.fillText(label, x, marginTop + plotHeight + 6);
      }
    }

    // Playhead Marker synchronized with current flyby playback
    if (this.showPlayhead && currentTime !== undefined && currentTime !== null) {
      const playheadNorm = (currentTime - tStart) / duration;
      if (playheadNorm >= 0 && playheadNorm <= 1) {
        const x = marginLeft + playheadNorm * plotWidth;

        if (ctx.beginPath && ctx.stroke) {
          // Subtle glow
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
          ctx.lineWidth = 3;
          ctx.moveTo(x, marginTop);
          ctx.lineTo(x, marginTop + plotHeight);
          ctx.stroke();

          // Crisp center playhead line
          ctx.beginPath();
          ctx.strokeStyle = '#ffffff';
          ctx.lineWidth = 1.5;
          ctx.moveTo(x, marginTop);
          ctx.lineTo(x, marginTop + plotHeight);
          ctx.stroke();

          // Playhead cap cursor
          if (ctx.fill) {
            ctx.fillStyle = '#ffffff';
            ctx.beginPath();
            ctx.moveTo(x - 4, marginTop);
            ctx.lineTo(x + 4, marginTop);
            ctx.lineTo(x, marginTop + 6);
            ctx.closePath();
            ctx.fill();
          }
        }
      }
    }

    // Top Instrumentation Caption & Legend
    if (ctx.fillText) {
      ctx.font = '9px "DM Sans", -apple-system, BlinkMacSystemFont, monospace';
      ctx.fillStyle = '#819aa6';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText('STFT SPECTROGRAM · 0 Hz – 12 kHz WATERFALL', marginLeft, marginTop - 4);

      ctx.textAlign = 'right';
      ctx.fillText(`WINDOW ${data.windowSize} · HOP ${data.hopSize} · NYQUIST 12 kHz`, marginLeft + plotWidth, marginTop - 4);
    }
  }
}
