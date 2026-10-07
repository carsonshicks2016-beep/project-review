import test from 'node:test';
import assert from 'node:assert/strict';
import {
  fft,
  hannWindow,
  blackmanWindow,
  windowEnergy,
  colormap,
  computeSTFT,
  theoreticalDopplerCurve,
  SpectrogramRenderer,
  SPECTROGRAM_PALETTES
} from '../dist/spectrogram.mjs';
import {
  clamp,
  getShockGeometry,
  computeDensityGradient,
  rayDeflection,
  renderSchlieren,
  SCHLIEREN_PALETTES
} from '../dist/schlieren.mjs';
import { renderAudio } from '../dist/physics.mjs';

const close = (a, b, tol = 1e-5) => {
  assert.ok(
    Math.abs(a - b) < tol,
    `Expected ${a} to be close to ${b} (diff: ${Math.abs(a - b)}, tol: ${tol})`
  );
};

// ============================================================================
// 1. FFT WINDOWING AND ENERGY CONSERVATION TESTS
// ============================================================================

test('Hann and Blackman windows satisfy symmetry, boundary, and energy properties', () => {
  for (const n of [128, 256, 512, 1024]) {
    const hann = hannWindow(n);
    const blackman = blackmanWindow(n);

    assert.equal(hann.length, n);
    assert.equal(blackman.length, n);

    // End points
    close(hann[0], 0.0, 1e-6);
    close(hann[n - 1], 0.0, 1e-6);

    // Peak near center
    const mid = Math.floor(n / 2);
    assert.ok(hann[mid] > 0.99, 'Hann peak near 1');
    assert.ok(blackman[mid] > 0.99, 'Blackman peak near 1');

    // Symmetry
    for (let i = 0; i < n; i++) {
      close(hann[i], hann[n - 1 - i], 1e-6);
      close(blackman[i], blackman[n - 1 - i], 1e-6);
      assert.ok(hann[i] >= 0 && hann[i] <= 1.0001, 'Hann in [0, 1]');
      assert.ok(blackman[i] >= -1e-6 && blackman[i] <= 1.0001, 'Blackman in [0, 1]');
    }

    // Energy calculation
    const s2Hann = windowEnergy(hann);
    const s2Blackman = windowEnergy(blackman);
    assert.ok(s2Hann > 0 && s2Hann < n, 'Hann energy is strictly positive and < N');
    assert.ok(s2Blackman > 0 && s2Blackman < n, 'Blackman energy is strictly positive and < N');
    // For Hann window, theoretical sum of w^2 is ~ 0.375 * N
    close(s2Hann / n, 0.375, 0.02);
  }
});

test('FFT accurately computes spectrum for DC and pure sinusoidal tones', () => {
  const n = 256;

  // 1. DC Signal: x[n] = 3.5
  const reDc = new Float32Array(n).fill(3.5);
  const imDc = new Float32Array(n);
  fft(reDc, imDc);

  close(reDc[0], 3.5 * n, 1e-4);
  close(imDc[0], 0, 1e-4);
  for (let k = 1; k < n; k++) {
    close(reDc[k], 0, 1e-4);
    close(imDc[k], 0, 1e-4);
  }

  // 2. Pure Sine Wave at bin k0 = 8
  const k0 = 8;
  const reSine = new Float32Array(n);
  const imSine = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    reSine[i] = Math.cos((2 * Math.PI * k0 * i) / n);
  }
  fft(reSine, imSine);

  // Peak at bin k0 and (N - k0)
  close(reSine[k0], n / 2, 1e-4);
  close(reSine[n - k0], n / 2, 1e-4);
  for (let k = 0; k < n; k++) {
    if (k !== k0 && k !== n - k0) {
      close(reSine[k], 0, 1e-4);
      close(imSine[k], 0, 1e-4);
    }
  }
});

test('FFT satisfies Parsevals energy conservation theorem for arbitrary signals', () => {
  for (const n of [128, 256, 512]) {
    // Generate pseudo-random test signal
    const x = new Float32Array(n);
    let timeEnergy = 0;
    for (let i = 0; i < n; i++) {
      x[i] = Math.sin(0.12 * i) + 0.5 * Math.cos(0.45 * i) - 0.25 * Math.sin(0.88 * i);
      timeEnergy += x[i] * x[i];
    }

    const re = new Float32Array(x);
    const im = new Float32Array(n);
    fft(re, im);

    // Sum of |X[k]|^2 / N in frequency domain
    let freqEnergy = 0;
    for (let k = 0; k < n; k++) {
      freqEnergy += (re[k] * re[k] + im[k] * im[k]) / n;
    }

    close(timeEnergy, freqEnergy, 1e-4);
  }
});

test('Windowed FFT strictly conserves windowed signal energy', () => {
  const n = 512;
  const win = hannWindow(n);

  const x = new Float32Array(n);
  const xw = new Float32Array(n);
  let timeEnergy = 0;

  for (let i = 0; i < n; i++) {
    x[i] = Math.sin(0.3 * i) + Math.cos(0.07 * i * i);
    xw[i] = x[i] * win[i];
    timeEnergy += xw[i] * xw[i];
  }

  const re = new Float32Array(xw);
  const im = new Float32Array(n);
  fft(re, im);

  let freqEnergy = 0;
  for (let k = 0; k < n; k++) {
    freqEnergy += (re[k] * re[k] + im[k] * im[k]) / n;
  }

  close(timeEnergy, freqEnergy, 1e-4);
});

// ============================================================================
// 2. SPECTROGRAM DATA GENERATION, DIMENSIONS, AND POWER TESTS
// ============================================================================

test('Spectrogram STFT output matches exact dimensions and non-negative power bounds', () => {
  const sampleRate = 24000;
  const numSamples = 24000; // 1.0 second of audio
  const audio = new Float32Array(numSamples);

  // Synthesize mixture: 440 Hz + 2000 Hz + noise
  for (let i = 0; i < numSamples; i++) {
    const t = i / sampleRate;
    audio[i] = 0.5 * Math.sin(2 * Math.PI * 440 * t) + 0.3 * Math.sin(2 * Math.PI * 2000 * t);
  }

  const windowSize = 512;
  const hopSize = 128;
  const stft = computeSTFT(audio, sampleRate, { windowSize, hopSize });

  const expectedBins = (windowSize >> 1) + 1; // 257 bins
  const expectedFrames = Math.floor((numSamples - windowSize) / hopSize) + 1; // 184 frames

  assert.equal(stft.numBins, expectedBins);
  assert.equal(stft.numFrames, expectedFrames);
  assert.equal(stft.grid.length, expectedFrames);
  assert.equal(stft.powerGrid.length, expectedFrames);
  assert.equal(stft.frequencies.length, expectedBins);
  assert.equal(stft.times.length, expectedFrames);

  // Frequency range: 0 Hz to 12 kHz (Nyquist)
  close(stft.frequencies[0], 0, 1e-5);
  close(stft.frequencies[expectedBins - 1], 12000, 1e-5);
  assert.equal(stft.nyquist, 12000);

  // Verify non-negative power values
  assert.ok(stft.minPower >= 0, `minPower (${stft.minPower}) must be >= 0`);
  for (let m = 0; m < expectedFrames; m++) {
    assert.equal(stft.grid[m].length, expectedBins);
    assert.equal(stft.powerGrid[m].length, expectedBins);

    for (let k = 0; k < expectedBins; k++) {
      const p = stft.powerGrid[m][k];
      assert.ok(p >= 0, `Power at [${m}][${k}] must be non-negative, got ${p}`);
      assert.ok(Number.isFinite(p), `Power at [${m}][${k}] must be finite`);

      const density = stft.grid[m][k];
      assert.ok(density >= 0 && density <= 1.0, `Density must be in [0, 1], got ${density}`);
    }
  }

  // Frequency peak verification: 440 Hz bin
  const bin440 = Math.round(440 / stft.freqStep);
  const midFrame = Math.floor(expectedFrames / 2);
  const power440 = stft.powerGrid[midFrame][bin440];
  const powerDc = stft.powerGrid[midFrame][0];
  assert.ok(power440 > powerDc * 10, '440 Hz bin should have significantly higher power than DC');
});

test('Spectrogram broadband impulse flash exhibits spectral energy across all frequencies', () => {
  const sampleRate = 24000;
  const audio = new Float32Array(4096);
  // Sharp impulse spike at sample 2048 (simulating sonic boom shock)
  audio[2048] = 50.0;

  const stft = computeSTFT(audio, sampleRate, { windowSize: 256, hopSize: 64 });
  const spikeFrame = Math.floor((2048 - 128) / 64);

  // In the impulse frame, power should be distributed broadly across low, mid, and high frequencies
  const powers = stft.powerGrid[spikeFrame];
  const lowPower = powers[5]; // low freq ~468 Hz
  const midPower = powers[40]; // mid freq ~3.75 kHz
  const highPower = powers[100]; // high freq ~9.37 kHz

  assert.ok(lowPower > 0.01, 'Broadband impulse has low frequency energy');
  assert.ok(midPower > 0.01, 'Broadband impulse has mid frequency energy');
  assert.ok(highPower > 0.01, 'Broadband impulse has high frequency energy');
});

test('SpectrogramRenderer class integrates with simulated Mach Lab audio', () => {
  // Generate realistic Mach 2 flyby audio with sonic boom using physics.mjs
  const audioResult = renderAudio({
    mach: 2.0,
    altitude: 500,
    lateralOffset: 0,
    aircraft: 'custom',
    mode: 'jet'
  });

  assert.ok(audioResult.samples.length > 0);

  const renderer = new SpectrogramRenderer({
    windowSize: 512,
    hopSize: 128,
    palette: 'inferno'
  });

  const specData = renderer.generateSpectrogramData(audioResult);

  assert.ok(specData.numFrames > 50);
  assert.equal(specData.numBins, 257);
  assert.ok(specData.maxPower > specData.minPower);
  assert.ok(specData.minPower >= 0);

  // Mock Canvas Context to test drawSpectrogram execution
  const drawCalls = [];
  const mockCtx = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    font: '',
    textAlign: '',
    textBaseline: '',
    fillRect(x, y, w, h) { drawCalls.push({ type: 'fillRect', x, y, w, h }); },
    beginPath() { drawCalls.push({ type: 'beginPath' }); },
    moveTo(x, y) { drawCalls.push({ type: 'moveTo', x, y }); },
    lineTo(x, y) { drawCalls.push({ type: 'lineTo', x, y }); },
    stroke() { drawCalls.push({ type: 'stroke' }); },
    fill() { drawCalls.push({ type: 'fill' }); },
    closePath() { drawCalls.push({ type: 'closePath' }); },
    fillText(text, x, y) { drawCalls.push({ type: 'fillText', text, x, y }); },
    setLineDash(dash) { drawCalls.push({ type: 'setLineDash', dash }); }
  };

  const mockCanvas = {
    width: 640,
    height: 320,
    getContext: () => mockCtx
  };

  // Draw at currentTime = 1.0 s during flyby
  renderer.drawSpectrogram(mockCanvas, specData, 1.0, { start: audioResult.start, end: audioResult.end });

  // Ensure canvas operations were dispatched
  assert.ok(drawCalls.length > 100, 'Spectrogram rendering produced canvas draw commands');
  const fillTexts = drawCalls.filter(c => c.type === 'fillText').map(c => c.text);

  // Verify presence of required frequency tick marks: 100 Hz, 500 Hz, 1 kHz, 5 kHz, 10 kHz
  assert.ok(fillTexts.includes('100 Hz'), 'Contains 100 Hz tick');
  assert.ok(fillTexts.includes('500 Hz'), 'Contains 500 Hz tick');
  assert.ok(fillTexts.includes('1 kHz'), 'Contains 1 kHz tick');
  assert.ok(fillTexts.includes('5 kHz'), 'Contains 5 kHz tick');
  assert.ok(fillTexts.includes('10 kHz'), 'Contains 10 kHz tick');
});

test('Colormap interpolation covers the full [0, 1] range and palettes', () => {
  for (const pal of ['inferno', 'cyberCyan', 'amber', 'viridis']) {
    const c0 = colormap(0.0, pal);
    const cMid = colormap(0.5, pal);
    const c1 = colormap(1.0, pal);

    assert.equal(c0.length, 3);
    assert.equal(cMid.length, 3);
    assert.equal(c1.length, 3);

    for (const c of [c0, cMid, c1]) {
      for (const val of c) {
        assert.ok(val >= 0 && val <= 255, 'RGB values between 0 and 255');
      }
    }
  }
});

// ============================================================================
// 3. SCHLIEREN GRADIENT MATH AND BOUNDARY CLAMPING TESTS
// ============================================================================

test('Shock geometry accurately calculates Mach angle and oblique shock angles', () => {
  // Mach 2.0
  const g2 = getShockGeometry({ mach: 2.0, aircraft: 'custom' });
  assert.ok(g2.isSupersonic);
  close(g2.mu, Math.asin(1 / 2.0), 1e-6); // 30 degrees = 0.5236 rad
  close(g2.muDeg, 30.0, 1e-4);
  assert.ok(g2.beta > g2.mu, 'Oblique shock angle beta is steeper than Mach angle mu');
  assert.ok(g2.betaTail > g2.mu, 'Tail shock angle is steeper than mu');

  // Mach 1.0 (Sonic limit)
  const g1 = getShockGeometry({ mach: 1.0, aircraft: 'custom' });
  assert.ok(!g1.isSupersonic);

  // Subsonic M = 0.7
  const gSub = getShockGeometry({ mach: 0.7, aircraft: 'custom' });
  assert.ok(!gSub.isSupersonic);
  assert.equal(gSub.mu, null);

  // NASA X-59 QueSST comparison: ultra-slender needle nose reduces beta
  const gX59 = getShockGeometry({ mach: 2.0, aircraft: 'x59' });
  assert.ok(gX59.beta < g2.beta, 'X-59 needle nose produces shallower initial shock angle than standard jet');
});

test('Density gradient computation produces sharp peaks along shocks and expansion zones', () => {
  const state = {
    mach: 2.0,
    px: 300,
    py: 150,
    length: 160,
    aircraft: 'custom',
    shimmer: false
  };

  const noseX = 300 + 0.5 * 160; // 380
  const yCenter = 150;
  const geom = getShockGeometry(state);
  const tanBeta = Math.tan(geom.beta);

  // 1. Exactly on Bow Shock Line: 40 pixels downstream of nose
  const xTest = noseX - 40;
  const yShockUpper = yCenter - 40 * tanBeta;
  const gradOnShock = computeDensityGradient(xTest, yShockUpper, state);

  // 2. Far ahead of the bow shock (upstream): x = noseX + 60
  const gradUpstream = computeDensityGradient(noseX + 60, yCenter, state);

  assert.ok(gradOnShock.magnitude > 0.5, 'Gradient magnitude on shock should be pronounced');
  assert.ok(gradOnShock.magnitude > gradUpstream.magnitude * 5, 'Shock gradient is much higher than undisturbed freestream');
  close(gradUpstream.magnitude, 0, 0.05);

  // 3. Tail recompression shock: dx = 0.95*L behind nose
  const xTail = noseX - 0.95 * 160;
  const yTailShock = yCenter - 20 * Math.tan(geom.betaTail);
  const gradTail = computeDensityGradient(xTail - 20, yTailShock, state);
  assert.ok(gradTail.magnitude > 0.3, 'Tail recompression shock has significant gradient magnitude');
});

test('Density gradient math strictly handles boundary clamping and extreme inputs', () => {
  const state = { mach: 2.0, px: 300, py: 150, length: 160 };

  // Test extreme coordinates (far beyond canvas bounds)
  const coords = [
    [-1e6, -1e6],
    [1e6, 1e6],
    [1e5, 150],
    [-1e5, 150],
    [300, 1e5]
  ];

  for (const [x, y] of coords) {
    const res = computeDensityGradient(x, y, state);
    assert.ok(Number.isFinite(res.magnitude), `Magnitude at (${x}, ${y}) must be finite`);
    assert.ok(res.magnitude >= 0 && res.magnitude <= 10.0, `Magnitude at (${x}, ${y}) must be clamped in [0, 10]`);
    assert.ok(Number.isFinite(res.rho), 'Density must be finite');
    assert.ok(Number.isFinite(res.deflectionX) && Math.abs(res.deflectionX) <= 15, 'DeflectionX must be clamped');
    assert.ok(Number.isFinite(res.deflectionY) && Math.abs(res.deflectionY) <= 15, 'DeflectionY must be clamped');
  }

  // Ray deflection helper
  const ray = rayDeflection(200, 100, state);
  assert.ok(Number.isFinite(ray.dx));
  assert.ok(Number.isFinite(ray.dy));
  assert.ok(Number.isFinite(ray.magnitude));
});

test('Schlieren renderer operates seamlessly with mock context across regimes and palettes', () => {
  const drawCalls = [];
  const mockCtx = {
    fillStyle: '',
    strokeStyle: '',
    lineWidth: 1,
    globalAlpha: 1,
    font: '',
    textAlign: '',
    textBaseline: '',
    fillRect(x, y, w, h) { drawCalls.push({ type: 'fillRect', x, y, w, h }); },
    strokeRect(x, y, w, h) { drawCalls.push({ type: 'strokeRect', x, y, w, h }); },
    beginPath() { drawCalls.push({ type: 'beginPath' }); },
    moveTo(x, y) { drawCalls.push({ type: 'moveTo', x, y }); },
    lineTo(x, y) { drawCalls.push({ type: 'lineTo', x, y }); },
    stroke() { drawCalls.push({ type: 'stroke' }); },
    fill() { drawCalls.push({ type: 'fill' }); },
    closePath() { drawCalls.push({ type: 'closePath' }); },
    save() { drawCalls.push({ type: 'save' }); },
    restore() { drawCalls.push({ type: 'restore' }); },
    translate(x, y) { drawCalls.push({ type: 'translate', x, y }); },
    quadraticCurveTo(cx, cy, x, y) { drawCalls.push({ type: 'quadraticCurveTo', cx, cy, x, y }); },
    fillText(text, x, y) { drawCalls.push({ type: 'fillText', text, x, y }); },
    createLinearGradient() {
      return { addColorStop() {} };
    },
    createRadialGradient() {
      return { addColorStop() {} };
    }
  };

  // Test 1: Supersonic M = 2.0 with background refraction ON (Amber palette)
  renderSchlieren(mockCtx, 800, 400, {
    mach: 2.0,
    aircraft: 'custom',
    palette: 'amber',
    backgroundDistortion: true,
    time: 0.5
  });
  assert.ok(drawCalls.length > 50);

  // Test 2: Subsonic M = 0.8 with background refraction OFF (Cyber-Cyan palette)
  drawCalls.length = 0;
  renderSchlieren(mockCtx, 800, 400, {
    mach: 0.8,
    aircraft: 'f16',
    palette: 'cyber-cyan',
    backgroundDistortion: false,
    time: 0
  });
  assert.ok(drawCalls.length > 30);

  // Test 3: NASA X-59 with Monochrome palette
  drawCalls.length = 0;
  renderSchlieren(mockCtx, 800, 400, {
    mach: 1.4,
    aircraft: 'x59',
    palette: 'monochrome',
    backgroundDistortion: true,
    time: 1.2
  });
  assert.ok(drawCalls.length > 50);
});
