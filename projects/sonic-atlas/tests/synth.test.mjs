import test from 'node:test';
import assert from 'node:assert/strict';
import {
  C,
  SAMPLE_RATE,
  boomTime,
  slantRange,
  nWave
} from '../dist/physics.mjs';
import {
  Biquad,
  generateSubBassHarmonics,
  triggerHapticBoom,
  getHapticBoomPattern,
  createImpulseResponse,
  flybyAudio
} from '../dist/synth.mjs';

const close = (a, b, tol = 1e-4) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b} (diff: ${Math.abs(a - b)})`);

// Discrete Fourier Transform energy at specific frequency
function goertzelEnergy(samples, freq, sampleRate) {
  const len = samples.length;
  const k = Math.round(len * freq / sampleRate);
  const w = (2 * Math.PI * k) / len;
  const coeff = 2 * Math.cos(w);
  let s_prev = 0;
  let s_prev2 = 0;

  for (let i = 0; i < len; i++) {
    const s = samples[i] + coeff * s_prev - s_prev2;
    s_prev2 = s_prev;
    s_prev = s;
  }
  return s_prev * s_prev + s_prev2 * s_prev2 - coeff * s_prev * s_prev2;
}

// Count zero crossings in a window
function countZeroCrossings(samples, startIdx, endIdx) {
  let count = 0;
  for (let i = startIdx; i < endIdx - 1; i++) {
    if ((samples[i] >= 0 && samples[i + 1] < 0) || (samples[i] < 0 && samples[i + 1] >= 0)) {
      count++;
    }
  }
  return count;
}

/* ========================================================================= */
/* 1. Psychoacoustic Sub-Bass Synthesizer Tests                              */
/* ========================================================================= */

test('generateSubBassHarmonics preserves buffer length and guarantees boundedness', () => {
  const sr = 24000;
  const len = 4800; // 0.2 s

  // Test 1: Silence
  const silence = new Float32Array(len);
  const outSilence = generateSubBassHarmonics(silence, sr);
  assert.equal(outSilence.length, len);
  assert.equal(outSilence.harmonics.length, len);
  assert.equal(outSilence.envelope.length, len);
  assert.equal(outSilence.shockDetected, false);
  for (let i = 0; i < len; i++) {
    assert.equal(outSilence[i], 0);
  }

  // Test 2: Huge amplitude input (stress test)
  const loud = new Float32Array(len);
  for (let i = 0; i < len; i++) loud[i] = 20.0 * Math.sin(2 * Math.PI * 25 * i / sr);
  const outLoud = generateSubBassHarmonics(loud, sr);
  for (let i = 0; i < len; i++) {
    assert.ok(Number.isFinite(outLoud[i]), `Sample ${i} is not finite`);
    assert.ok(Math.abs(outLoud[i]) <= 1.0, `Sample ${i} exceeds 1.0: ${outLoud[i]}`);
  }

  // Test 3: Sonic boom N-wave
  const boomSamples = new Float32Array(len);
  for (let i = 0; i < len; i++) {
    boomSamples[i] = nWave(i / sr, 0.05, 1.0, 0.10, 0.002);
  }
  const outBoom = generateSubBassHarmonics(boomSamples, sr);
  assert.equal(outBoom.shockDetected, true);
  for (let i = 0; i < len; i++) {
    assert.ok(Number.isFinite(outBoom[i]));
    assert.ok(Math.abs(outBoom[i]) <= 1.0);
  }
  // Verify envelope rises after shock arrival (t = 0.05s -> i = 1200)
  const envPre = outBoom.envelope[Math.round(0.02 * sr)];
  const envPost = outBoom.envelope[Math.round(0.08 * sr)];
  assert.ok(envPost > envPre * 10, 'Envelope should rise sharply after shock event');
});

test('generateSubBassHarmonics synthesizes virtual pitch 2nd and 3rd harmonics from sub-40Hz content', () => {
  const sr = 24000;
  const duration = 0.5; // 0.5s
  const len = Math.round(duration * sr);
  const input = new Float32Array(len);

  // Pure 30 Hz sub-bass sine wave (inaudible on laptop speakers)
  const f0 = 30;
  for (let i = 0; i < len; i++) {
    input[i] = 0.8 * Math.sin(2 * Math.PI * f0 * i / sr);
  }

  // In input, 60 Hz and 90 Hz are essentially non-existent
  const in60 = goertzelEnergy(input, 60, sr);
  const in90 = goertzelEnergy(input, 90, sr);

  const enhanced = generateSubBassHarmonics(input, sr, { harmonicsGain: 0.8, drive: 2.0 });

  // In harmonics buffer, 60 Hz (2nd harmonic) and 90 Hz (3rd harmonic) are synthesized!
  const harm60 = goertzelEnergy(enhanced.harmonics, 60, sr);
  const harm90 = goertzelEnergy(enhanced.harmonics, 90, sr);

  assert.ok(harm60 > 1000 * (in60 + 1e-9), `2nd harmonic (60 Hz) should be synthesized: harm=${harm60}, in=${in60}`);
  assert.ok(harm90 > 1000 * (in90 + 1e-9), `3rd harmonic (90 Hz) should be synthesized: harm=${harm90}, in=${in90}`);

  // In enhanced mix, 60 Hz energy is prominently present
  const out60 = goertzelEnergy(enhanced, 60, sr);
  assert.ok(out60 > 500 * (in60 + 1e-9), `Enhanced mix must contain 60 Hz virtual pitch overtone`);
});

/* ========================================================================= */
/* 2. Mobile Haptic Shock Feedback Tests                                     */
/* ========================================================================= */

test('triggerHapticBoom safely falls back on unsupported environments (desktop)', () => {
  // In node.js, navigator is undefined by default
  const origNav = globalThis.navigator;
  delete globalThis.navigator;

  try {
    const result = triggerHapticBoom(120, 1.0);
    assert.equal(result, false, 'Should return false when navigator.vibrate is unsupported');
  } finally {
    if (origNav !== undefined) globalThis.navigator = origNav;
  }
});

test('getHapticBoomPattern formats dual-pulse pattern for bow and tail shocks', () => {
  const pattern100 = getHapticBoomPattern(120, 1.0);
  assert.deepEqual(pattern100, [35, 120, 25]);

  const patternHalf = getHapticBoomPattern(80, 0.5);
  assert.deepEqual(patternHalf, [18, 80, 13]);

  const patternZero = getHapticBoomPattern(120, 0.0);
  assert.deepEqual(patternZero, [0, 0, 0]);
});

test('triggerHapticBoom correctly invokes navigator.vibrate when supported', () => {
  let calledPattern = null;
  const mockNavigator = {
    vibrate(pattern) {
      calledPattern = pattern;
      return true;
    }
  };

  const origNav = globalThis.navigator;
  globalThis.navigator = mockNavigator;

  try {
    const res = triggerHapticBoom(160, 0.8);
    assert.equal(res, true);
    assert.deepEqual(calledPattern, [28, 160, 20]);
  } finally {
    if (origNav !== undefined) {
      globalThis.navigator = origNav;
    } else {
      delete globalThis.navigator;
    }
  }
});

/* ========================================================================= */
/* 3. Environmental Impulse Response Tests                                   */
/* ========================================================================= */

test('createImpulseResponse generates normalized stereo buffers for all environments', () => {
  const sr = 24000;
  const duration = 1.5;
  const expectedLen = Math.round(sr * duration);

  for (const env of ['desert', 'canyon', 'urban']) {
    const ir = createImpulseResponse(env, sr, duration);
    assert.ok(ir.channelL instanceof Float32Array, `${env} channelL is Float32Array`);
    assert.ok(ir.channelR instanceof Float32Array, `${env} channelR is Float32Array`);
    assert.equal(ir.channelL.length, expectedLen, `${env} length matches duration`);
    assert.equal(ir.channelR.length, expectedLen, `${env} length matches duration`);

    let peak = 0;
    for (let i = 0; i < expectedLen; i++) {
      peak = Math.max(peak, Math.abs(ir.channelL[i]), Math.abs(ir.channelR[i]));
    }
    close(peak, 1.0, 1e-4);
  }
});

test('createImpulseResponse exhibits physical energy decay over time', () => {
  const sr = 24000;
  const duration = 2.0;
  const N = Math.round(sr * duration);
  const mid = Math.floor(N / 2);

  for (const env of ['desert', 'canyon', 'urban']) {
    const ir = createImpulseResponse(env, sr, duration);
    let energyFirstHalf = 0;
    let energySecondHalf = 0;

    for (let i = 0; i < mid; i++) {
      energyFirstHalf += ir.channelL[i] * ir.channelL[i] + ir.channelR[i] * ir.channelR[i];
    }
    for (let i = mid; i < N; i++) {
      energySecondHalf += ir.channelL[i] * ir.channelL[i] + ir.channelR[i] * ir.channelR[i];
    }

    assert.ok(
      energyFirstHalf > energySecondHalf * 3,
      `${env} first half energy (${energyFirstHalf}) should be much greater than second half (${energySecondHalf})`
    );

    if (env === 'desert') {
      // Desert has virtually zero late energy
      assert.ok(energySecondHalf < 1e-5, `Desert must have near-zero late reverberant energy: ${energySecondHalf}`);
    }
  }
});

test('createImpulseResponse provides true stereo decorrelation for canyon and urban', () => {
  const sr = 24000;
  const duration = 2.0;
  const N = Math.round(sr * duration);

  for (const env of ['canyon', 'urban']) {
    const ir = createImpulseResponse(env, sr, duration);
    let dot = 0, sumL2 = 0, sumR2 = 0;
    for (let i = 0; i < N; i++) {
      dot += ir.channelL[i] * ir.channelR[i];
      sumL2 += ir.channelL[i] * ir.channelL[i];
      sumR2 += ir.channelR[i] * ir.channelR[i];
    }
    const correlation = dot / (Math.sqrt(sumL2 * sumR2) + 1e-9);

    // Stereo decorrelation: correlation should be strictly less than 0.95
    assert.ok(
      correlation < 0.95,
      `${env} stereo cross-correlation (${correlation}) should be decorrelated (< 0.95)`
    );
  }

  // Desert is primarily direct sound, so correlation is high (~1.0)
  const irDesert = createImpulseResponse('desert', sr, duration);
  let dot = 0, sumL2 = 0, sumR2 = 0;
  for (let i = 0; i < N; i++) {
    dot += irDesert.channelL[i] * irDesert.channelR[i];
    sumL2 += irDesert.channelL[i] * irDesert.channelL[i];
    sumR2 += irDesert.channelR[i] * irDesert.channelR[i];
  }
  const corrDesert = dot / (Math.sqrt(sumL2 * sumR2) + 1e-9);
  assert.ok(corrDesert > 0.99, 'Desert direct sound should be centered in stereo');
});

test('canyon impulse response contains discrete flutter echoes at 180 ms, 320 ms, and 490 ms', () => {
  const sr = 24000;
  const duration = 1.5;
  const ir = createImpulseResponse('canyon', sr, duration);

  const echoTimes = [0.180, 0.320, 0.490];
  for (const t of echoTimes) {
    const center = Math.round(t * sr);
    // Find peak within 4 ms window around expected arrival
    const win = Math.round(0.004 * sr);
    let peakAround = 0;
    for (let di = -win; di <= win; di++) {
      const idx = center + di;
      peakAround = Math.max(peakAround, Math.abs(ir.channelL[idx]), Math.abs(ir.channelR[idx]));
    }

    // Compare with background level slightly before echo (e.g. 15 ms before)
    const bgIdx = center - Math.round(0.015 * sr);
    let bgLevel = 0;
    for (let di = -win; di <= win; di++) {
      bgLevel = Math.max(bgLevel, Math.abs(ir.channelL[bgIdx + di]), Math.abs(ir.channelR[bgIdx + di]));
    }

    assert.ok(
      peakAround > bgLevel * 1.5,
      `Discrete echo at ${t * 1000} ms should have distinct peak (${peakAround}) above background (${bgLevel})`
    );
  }
});

/* ========================================================================= */
/* 4. Custom Audio "Flyby-izer" Tests                                        */
/* ========================================================================= */

test('flybyAudio matches calculated physical output length and structure', () => {
  const sr = 24000;
  const mach = 0.8;
  const alt = 400;
  const lateral = 50;

  // Arbitrary test source: 2000 samples of siren chirp
  const src = new Float32Array(2000);
  for (let i = 0; i < src.length; i++) {
    src[i] = Math.sin(2 * Math.PI * (300 + 0.1 * i) * i / sr);
  }

  const flyby = flybyAudio(src, sr, mach, alt, lateral);
  const expectedLen = Math.ceil((flyby.end - flyby.start) * sr);

  assert.equal(flyby.length, expectedLen, 'Output buffer length matches (end - start) * sr');
  assert.equal(flyby.samplesL.length, expectedLen, 'samplesL length matches');
  assert.equal(flyby.samplesR.length, expectedLen, 'samplesR length matches');
  assert.ok(flyby.peak > 0, 'Audio flyby has audible peak');
  assert.ok(flyby.rms > 0, 'Audio flyby has non-zero RMS energy');
  assert.equal(flyby.mach, mach);
  assert.equal(flyby.altitude, alt);
  assert.equal(flyby.lateralOffset, lateral);
});

test('flybyAudio maintains strict supersonic causality ahead of shock wave arrival', () => {
  const sr = 24000;
  const mach = 1.6;
  const alt = 500;
  const lateral = 100;

  // High-energy source signal
  const src = new Float32Array(1000);
  src.fill(0.8);

  const flyby = flybyAudio(src, sr, mach, alt, lateral);
  const arrival = boomTime(mach, alt, lateral);

  assert.ok(arrival !== null && arrival > 0, 'Supersonic arrival time exists');

  // Verify that every sample before arrival is strictly 0
  let nonZeroCountAhead = 0;
  for (let i = 0; i < flyby.length; i++) {
    const t = flyby.start + i / sr;
    if (t < arrival - 0.001) {
      if (flyby[i] !== 0 || flyby.samplesL[i] !== 0 || flyby.samplesR[i] !== 0) {
        nonZeroCountAhead++;
      }
    }
  }
  assert.equal(nonZeroCountAhead, 0, 'Causality violation: non-zero samples found ahead of supersonic shock wave');

  // Verify that after arrival, audio becomes active
  const arrivalIdx = Math.round((arrival - flyby.start) * sr);
  let postArrivalEnergy = 0;
  for (let i = arrivalIdx; i < Math.min(flyby.length, arrivalIdx + 500); i++) {
    postArrivalEnergy += flyby[i] * flyby[i];
  }
  assert.ok(postArrivalEnergy > 0, 'Audio must be active behind the Mach shock wave');
});

test('flybyAudio produces authentic physical Doppler frequency pitch shift', () => {
  const sr = 24000;
  const mach = 0.6;
  const alt = 250;
  const lateral = 0;
  const f0 = 400; // 400 Hz pure source tone

  // 1-second continuous pure tone loop
  const srcLen = sr;
  const src = new Float32Array(srcLen);
  for (let i = 0; i < srcLen; i++) {
    src[i] = Math.sin(2 * Math.PI * f0 * i / sr);
  }

  const flyby = flybyAudio(src, sr, mach, alt, lateral, {
    timeWindow: { start: -2.0, end: 2.5 }
  });

  // Approaching region: t in [-1.5, -0.7] (aircraft flying towards observer)
  const appStartIdx = Math.round((-1.5 - flyby.start) * sr);
  const appEndIdx = Math.round((-0.7 - flyby.start) * sr);
  const appCrossings = countZeroCrossings(flyby, appStartIdx, appEndIdx);
  const appDuration = (appEndIdx - appStartIdx) / sr;
  const freqApp = (appCrossings / 2) / appDuration;

  // Receding region: t in [0.7, 1.5] (aircraft flying away from observer)
  const recStartIdx = Math.round((0.7 - flyby.start) * sr);
  const recEndIdx = Math.round((1.5 - flyby.start) * sr);
  const recCrossings = countZeroCrossings(flyby, recStartIdx, recEndIdx);
  const recDuration = (recEndIdx - recStartIdx) / sr;
  const freqRec = (recCrossings / 2) / recDuration;

  // Theoretical limits:
  // Approaching Doppler: f_app ~ f0 / (1 - M) = 400 / 0.4 = 1000 Hz
  // Receding Doppler:    f_rec ~ f0 / (1 + M) = 400 / 1.6 = 250 Hz
  // In our slant geometry, freqApp > 600 Hz and freqRec < 350 Hz
  assert.ok(
    freqApp > 550,
    `Approaching pitch (${freqApp} Hz) should be significantly higher than source (${f0} Hz)`
  );
  assert.ok(
    freqRec < 350,
    `Receding pitch (${freqRec} Hz) should be significantly lower than source (${f0} Hz)`
  );
  assert.ok(
    freqApp > 1.8 * freqRec,
    `Doppler frequency ratio (${freqApp / freqRec}) should demonstrate major pitch drop`
  );
});

test('flybyAudio correctly applies distance attenuation with lateral offset', () => {
  const sr = 24000;
  const mach = 0.8;
  const alt = 400;

  const src = new Float32Array(sr);
  for (let i = 0; i < src.length; i++) {
    src[i] = Math.sin(2 * Math.PI * 300 * i / sr);
  }

  const flybyDirect = flybyAudio(src, sr, mach, alt, 0, {
    timeWindow: { start: -1, end: 1 }
  });
  const flybyFar = flybyAudio(src, sr, mach, alt, 1200, {
    timeWindow: { start: -1, end: 1 }
  });

  assert.ok(
    flybyDirect.rms > flybyFar.rms * 1.5,
    `Distant lateral flyby RMS (${flybyFar.rms}) should be lower than overhead flyby (${flybyDirect.rms})`
  );
});
