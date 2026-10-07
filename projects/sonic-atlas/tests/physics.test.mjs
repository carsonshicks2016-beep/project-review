import test from 'node:test';
import assert from 'node:assert/strict';
import {
  C,
  SAMPLE_RATE,
  GAMMA,
  R_AIR,
  T0_STD,
  LAPSE_RATE_STD,
  D_HEAD,
  H_OBS,
  GAMMA_GROUND,
  retardedTimes,
  doppler,
  boomTime,
  coneAngle,
  nWave,
  renderAudio,
  effectiveSoundSpeed,
  soundSpeed,
  acousticRefractionRay,
  groundReflection,
  airAbsorptionCutoff,
  airAbsorptionAlpha,
  AirAbsorptionFilter,
  binauralITD,
  binauralILD,
  binauralParams,
  AIRCRAFT_PRESETS,
  shapedWave,
  slantRange,
  boomTimeLateral,
  carpetWidth,
  lateralAttenuation,
  lateralOverpressure,
  whithamAttenuation,
  whithamOverpressure,
  vaporConeIntensity
} from '../dist/physics.mjs';

const close = (a, b, tol = 1e-7) => assert.ok(Math.abs(a - b) < tol, `${a} vs ${b}`);

test('every retained emission satisfies unsquared arrival equation across all regimes', () => {
  for (const m of [.3, .7, .99, 1, 1.001, 1.2, 2, 5, 8])
    for (const h of [100, 500, 2000])
      for (let t = -12; t < 27; t += .071) {
        const roots = retardedTimes(t, m, h);
        if (m < 1) assert.equal(roots.length, 1);
        for (const tau of roots) {
          assert.ok(tau <= t);
          close(tau + Math.hypot(m * C * tau, h) / C, t, 1e-6);
        }
      }
});

test('supersonic observer remains silent until analytical Mach cone arrival', () => {
  for (const m of [1.01, 1.2, 2, 5, 8]) {
    const h = 500, t = boomTime(m, h);
    assert.equal(retardedTimes(t - .0001, m, h).length, 0);
    assert.equal(retardedTimes(t + .0001, m, h).length, 2);
    close(m * C * t * Math.tan(coneAngle(m)), h);
  }
});

test('Doppler ratio agrees with numerical retarded-time derivative', () => {
  for (const m of [.3, .7, .99, 1, 2, 5, 8])
    for (const t of [2, 4, 9]) {
      const roots = retardedTimes(t, m, 500),
        prev = retardedTimes(t - 1e-6, m, 500),
        next = retardedTimes(t + 1e-6, m, 500);
      for (let i = 0; i < roots.length; i++)
        close(Math.abs((next[i] - prev[i]) / 2e-6), doppler(roots[i], m, 500), 1e-4);
    }
});

test('overhead emitted sound arrives at h/c regardless of speed', () => {
  for (const m of [.3, .7, 1, 2, 8])
    assert.ok(retardedTimes(500 / C, m, 500).some(x => Math.abs(x) < 1e-8));
});

test('sonic limit has a causal positive-time branch only', () => {
  assert.deepEqual(retardedTimes(-1, 1, 500), []);
  assert.deepEqual(retardedTimes(0, 1, 500), []);
  assert.equal(retardedTimes(.01, 1, 500).length, 1);
  assert.equal(boomTime(1, 500), null);
});

test('N wave has finite edges, zero net area, and exact silence outside its support', () => {
  const start = 1.2, d = .18;
  assert.equal(nWave(start - .001, start), 0);
  assert.equal(nWave(start + d + .001, start), 0);
  close(nWave(start + d / 2, start), 0);
  assert.ok(nWave(start + .003, start) > 50);
  assert.ok(nWave(start + d - .003, start) < -50);
  let area = 0;
  for (let i = 0; i < 10001; i++)
    area += nWave(start + i * d / 10000, start) * d / 10000;
  close(area, 0, 1e-6);
});

test('generated audio stays finite and silent ahead of the shock', () => {
  for (const m of [.3, 1, 1.01, 2, 8]) {
    const a = renderAudio({ mach: m, altitude: 500, mode: 'tone' });
    assert.ok(a.peak < 1);
    assert.ok(a.rms > 0);
    for (let i = 0; i < a.samples.length; i++) {
      assert.ok(Number.isFinite(a.samples[i]));
      if (a.arrival !== null && a.start + i / a.sampleRate < a.arrival)
        assert.equal(a.samples[i], 0);
    }
  }
});

test('boom-only subsonic pass is silent; seeded jet sound is deterministic', () => {
  assert.equal(renderAudio({ mach: .7, altitude: 500, mode: 'boom' }).peak, 0);
  const a = renderAudio({ mach: 2, altitude: 500, mode: 'jet' }),
    b = renderAudio({ mach: 2, altitude: 500, mode: 'jet' });
  assert.deepEqual(a.samples, b.samples);
  assert.deepEqual(a.samplesL, b.samplesL);
  assert.deepEqual(a.samplesR, b.samplesR);
  assert.ok(a.rms > 0);
});

test('binaural ITD adheres strictly to physical head diameter bounds (ITD <= 0.6 ms)', () => {
  // Test ITD formula: ITD = (d_head / C) * sin(theta)
  for (let sinTheta = -1; sinTheta <= 1.0001; sinTheta += 0.1) {
    const s = Math.max(-1, Math.min(1, sinTheta));
    const itd = binauralITD(s, D_HEAD);
    assert.ok(Number.isFinite(itd));
    assert.ok(Math.abs(itd) <= 0.0006, `ITD ${itd * 1000} ms exceeds 0.6 ms limit`);
  }
  // Max ITD with standard head diameter (0.18 m) is 0.18 / 343 = ~0.5248 ms
  const maxITD = binauralITD(1.0, 0.18);
  close(maxITD, 0.18 / C, 1e-7);
  assert.ok(maxITD <= 0.0006);
  close(binauralITD(0), 0, 1e-9);
  close(binauralITD(-1.0, 0.18), -0.18 / C, 1e-7);
});

test('binaural stereo channel separation and ILD head shadow rendering', () => {
  // When aircraft is to the right (sinTheta > 0), right ear is louder (ildR = 1, ildL < 1)
  const rightCase = binauralILD(0.8, 500);
  assert.equal(rightCase.ildR, 1.0);
  assert.ok(rightCase.ildL < 1.0);
  assert.ok(rightCase.ildL > 0.3);

  // When aircraft is to the left (sinTheta < 0), left ear is louder (ildL = 1, ildR < 1)
  const leftCase = binauralILD(-0.8, 500);
  assert.equal(leftCase.ildL, 1.0);
  assert.ok(leftCase.ildR < 1.0);
  assert.ok(leftCase.ildR > 0.3);

  // Overhead (sinTheta = 0), both channels are identical
  const centerCase = binauralILD(0, 500);
  assert.equal(centerCase.ildL, 1.0);
  assert.equal(centerCase.ildR, 1.0);

  // Audio rendering verification for stereo output
  const audio = renderAudio({ mach: 0.7, altitude: 500, mode: 'tone' });
  assert.equal(audio.isStereo, true);
  assert.ok(audio.samplesL instanceof Float32Array);
  assert.ok(audio.samplesR instanceof Float32Array);
  assert.equal(audio.samplesL.length, audio.samples.length);
  assert.equal(audio.samplesR.length, audio.samples.length);

  // Measure channel energy during approach (left) vs departure (right)
  const half = Math.floor(audio.samples.length / 2);
  let leftEnergyApproach = 0, rightEnergyApproach = 0;
  for (let i = 0; i < Math.floor(half * 0.4); i++) {
    leftEnergyApproach += audio.samplesL[i] ** 2;
    rightEnergyApproach += audio.samplesR[i] ** 2;
  }
  // During approach from the left, left channel energy dominates right channel
  assert.ok(leftEnergyApproach > rightEnergyApproach, 'Left ear should receive more energy as plane approaches from left');

  let leftEnergyDepart = 0, rightEnergyDepart = 0;
  for (let i = audio.samples.length - Math.floor(half * 0.4); i < audio.samples.length; i++) {
    leftEnergyDepart += audio.samplesL[i] ** 2;
    rightEnergyDepart += audio.samplesR[i] ** 2;
  }
  // During departure to the right, right channel energy dominates left channel
  assert.ok(rightEnergyDepart > leftEnergyDepart, 'Right ear should receive more energy as plane departs to right');
});

test('ground reflection multipath interference geometry and delay calculations', () => {
  const h = 500, h_obs = 1.7, Gamma = 0.90;

  // Overhead position (tau = 0)
  const overhead = groundReflection(0, 0.8, h, h_obs, Gamma);
  close(overhead.r_dir, h - h_obs, 1e-6);
  close(overhead.r_refl, h + h_obs, 1e-6);
  close(overhead.deltaR, 2 * h_obs, 1e-6);
  close(overhead.deltaT, 2 * h_obs / C, 1e-6);
  assert.equal(overhead.Gamma, Gamma);

  // Reflected path must be strictly longer than direct path for observer above ground
  for (let tau = -10; tau <= 10; tau += 1.5) {
    const refl = groundReflection(tau, 1.2, h, h_obs, Gamma);
    assert.ok(refl.r_refl > refl.r_dir, 'Reflected ray distance must exceed direct ray distance');
    assert.ok(refl.deltaR > 0);
    assert.ok(refl.deltaT > 0);
  }

  // As aircraft approaches grazing angle at large distance, path difference approaches 0
  const grazing = groundReflection(50, 2.0, h, h_obs, Gamma);
  assert.ok(grazing.deltaR < 0.1, 'Grazing path difference should shrink toward zero at long range');
  assert.ok(grazing.deltaT < 0.0003);

  // Comb-filter notch frequencies at overhead
  const f0 = 0.5 / overhead.deltaT;
  assert.ok(f0 > 45 && f0 < 55, `First notch frequency expected ~50.4 Hz, got ${f0}`);
});

test('atmospheric absorption filter stability and distance-dependent roll-off', () => {
  // Cutoff decreases monotonically with distance (ISO 9613-1 HF attenuation)
  assert.ok(airAbsorptionCutoff(100) > airAbsorptionCutoff(1000));
  assert.ok(airAbsorptionCutoff(1000) > airAbsorptionCutoff(10000));
  assert.ok(airAbsorptionCutoff(50000) >= 350);

  // Filter pole stability across extreme distances
  for (const d of [0, 10, 100, 500, 2000, 10000, 50000, 1e6]) {
    const alpha = airAbsorptionAlpha(d, SAMPLE_RATE);
    assert.ok(alpha > 0 && alpha <= 1.0, `Alpha ${alpha} must be in (0, 1]`);
    const pole = 1 - alpha;
    assert.ok(pole >= 0 && pole < 1.0, `Pole ${pole} must lie strictly inside unit circle`);
  }

  // Filter step response stability
  const filter = new AirAbsorptionFilter(SAMPLE_RATE);
  let val = 0;
  for (let i = 0; i < 500; i++) {
    val = filter.process(1.0, 2000);
    assert.ok(Number.isFinite(val));
    assert.ok(val >= 0 && val <= 1.0, `Step response ${val} must not overshoot 1.0`);
  }
  assert.ok(val > 0.99, 'Filter should asymptotically reach steady-state input');

  // Filter impulse response stability (no oscillation, monotonic decay)
  filter.reset();
  val = filter.process(1.0, 1000);
  let prevVal = val;
  for (let i = 0; i < 200; i++) {
    val = filter.process(0.0, 1000);
    assert.ok(Number.isFinite(val));
    assert.ok(val >= 0 && val <= prevVal, 'Impulse response must decay monotonically to zero without ringing');
    prevVal = val;
  }
  assert.ok(val < 1e-4, 'Filter should decay to silence');

  // Extreme stress test: rapidly jumping distances with white noise input
  filter.reset();
  let seed = 12345;
  for (let i = 0; i < 5000; i++) {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    const noise = (seed / 0x7fffffff) * 2 - 1;
    const distance = 50 + (i % 100) * 500;
    const out = filter.process(noise, distance);
    assert.ok(Number.isFinite(out));
    assert.ok(Math.abs(out) <= 1.0, `Filter output ${out} bounded by input magnitude`);
  }

  // Verify distance-dependent attenuation: 5 kHz tone at 100m vs 5000m
  const filterNear = new AirAbsorptionFilter(SAMPLE_RATE);
  const filterFar = new AirAbsorptionFilter(SAMPLE_RATE);
  let peakNear = 0, peakFar = 0;
  for (let i = 0; i < 1000; i++) {
    const tone = Math.sin(2 * Math.PI * 5000 * i / SAMPLE_RATE);
    const outNear = filterNear.process(tone, 100);
    const outFar = filterFar.process(tone, 5000);
    if (i > 200) {
      peakNear = Math.max(peakNear, Math.abs(outNear));
      peakFar = Math.max(peakFar, Math.abs(outFar));
    }
  }
  assert.ok(peakNear > peakFar * 2, `5 kHz tone at 100m (${peakNear}) must be significantly louder than at 5000m (${peakFar})`);
});

test('atmospheric temperature lapse and acoustic refraction sound speed variation', () => {
  // Analytical speed of sound at sea level T0 = 288.15 K
  const c0_expected = Math.sqrt(GAMMA * R_AIR * T0_STD);
  close(effectiveSoundSpeed(0), c0_expected, 1e-4);
  close(soundSpeed(0), c0_expected, 1e-4);

  // Normal positive lapse rate: temperature and sound speed decrease with altitude
  const c_ground = effectiveSoundSpeed(0, LAPSE_RATE_STD);
  const c_1km = effectiveSoundSpeed(1000, LAPSE_RATE_STD);
  const c_3km = effectiveSoundSpeed(3000, LAPSE_RATE_STD);
  assert.ok(c_3km < c_1km && c_1km < c_ground, 'Sound speed must decrease with altitude under positive lapse rate');

  // Exact check at 2000m: T = 288.15 - 0.0065 * 2000 = 275.15 K
  const c_2km_expected = Math.sqrt(1.4 * 287.05 * 275.15);
  close(effectiveSoundSpeed(2000, 0.0065), c_2km_expected, 1e-4);

  // Temperature inversion (negative lapse rate): sound speed increases with altitude
  const c_inv_0 = effectiveSoundSpeed(0, -0.005);
  const c_inv_2km = effectiveSoundSpeed(2000, -0.005);
  assert.ok(c_inv_2km > c_inv_0, 'Sound speed must increase with altitude during temperature inversion');

  // Isothermal atmosphere (zero lapse rate): sound speed constant
  close(effectiveSoundSpeed(3000, 0), effectiveSoundSpeed(0, 0), 1e-6);

  // Ray refraction curvature angle theta(z) = acos(c(z)/c0 * cos(theta0))
  // For isothermal atmosphere, ray angle is invariant
  const rayIso = acousticRefractionRay(2000, 0, Math.PI / 6);
  close(rayIso.theta, Math.PI / 6, 1e-6);

  // Standard lapse rate: c(z) < c0 => cos(theta(z)) < cos(theta0) => ray curves upward (shadow zone)
  const rayNormal = acousticRefractionRay(2000, LAPSE_RATE_STD, 0);
  assert.ok(rayNormal.theta > 0, 'Ray launched horizontally must curve upward under normal lapse rate');
  close(rayNormal.theta, Math.acos(rayNormal.c / rayNormal.c0), 1e-6);

  // Temperature inversion: c(z) > c0 => ray curves downward (ducting / channeling)
  const rayInv = acousticRefractionRay(2000, -0.005, Math.PI / 4);
  assert.ok(rayInv.theta < Math.PI / 4, 'Ray launched at 45 deg must curve downward under inversion');

  // Object coercion check (valueOf returns theta)
  assert.equal(typeof rayNormal.theta, 'number');
  assert.equal(typeof rayNormal.c, 'number');
  close(Number(rayNormal), rayNormal.theta, 1e-8);
});

test('aircraft preset data consistency and physical parameters', () => {
  const expectedPresets = ['x59', 'concorde', 'sr71', 'f16', 'custom'];
  for (const id of expectedPresets) {
    const p = AIRCRAFT_PRESETS[id];
    assert.ok(p, `Preset ${id} must exist`);
    assert.equal(p.id, id);
    assert.ok(typeof p.name === 'string' && p.name.length > 0);
    assert.ok(typeof p.peak === 'number' && p.peak > 0);
    assert.ok(typeof p.duration === 'number' && p.duration > 0);
    assert.ok(typeof p.rise === 'number' && p.rise > 0);
    assert.ok(typeof p.length === 'number' && p.length > 0);
  }

  // NASA X-59 QueSST: Low-Boom shaped signature (soft thump, peak ~30 Pa, duration ~220 ms, soft rise time ~15 ms)
  const x59 = AIRCRAFT_PRESETS.x59;
  close(x59.peak, 30, 1e-4);
  close(x59.duration, 0.22, 1e-4);
  close(x59.rise, 0.015, 1e-4);

  // Concorde SST: Heavy supersonic transport: length ~62m, peak ~105 Pa, duration ~240 ms, rise ~2 ms
  const concorde = AIRCRAFT_PRESETS.concorde;
  close(concorde.peak, 105, 1e-4);
  close(concorde.duration, 0.24, 1e-4);
  close(concorde.rise, 0.002, 1e-4);
  close(concorde.length, 62, 1e-4);

  // SR-71 Blackbird: High-speed recon: peak ~90 Pa, duration ~160 ms, rise ~1.5 ms
  const sr71 = AIRCRAFT_PRESETS.sr71;
  close(sr71.peak, 90, 1e-4);
  close(sr71.duration, 0.16, 1e-4);
  close(sr71.rise, 0.0015, 1e-4);

  // F-16 Falcon: Compact fighter: length ~15m, peak ~75 Pa, duration ~95 ms, rise ~1.8 ms
  const f16 = AIRCRAFT_PRESETS.f16;
  close(f16.peak, 75, 1e-4);
  close(f16.duration, 0.095, 1e-4);
  close(f16.rise, 0.0018, 1e-4);
  close(f16.length, 15, 1e-4);

  // Custom / Standard N-wave
  const custom = AIRCRAFT_PRESETS.custom;
  assert.equal(AIRCRAFT_PRESETS.standard, custom);
  assert.ok(custom.peak > 0);
});

test('X-59 shaped wave zero-net-area property, multi-stage ramp, and bounded amplitude', () => {
  const arrival = 1.5;
  const d = 0.22;
  const p = 30;
  const r = 0.015;

  // Silence strictly outside support [arrival, arrival + duration]
  assert.equal(shapedWave(arrival - 0.01, arrival, 'x59', p, d, r), 0);
  assert.equal(shapedWave(arrival + d + 0.01, arrival, 'x59', p, d, r), 0);
  assert.equal(shapedWave(0, null, 'x59', p, d, r), 0);

  // Zero-crossing exactly at midpoint
  close(shapedWave(arrival + d * 0.5, arrival, 'x59', p, d, r), 0, 1e-9);

  // Bounded amplitude: max <= 30 Pa, min >= -30 Pa, reaches peak ~30 Pa
  let maxP = -Infinity, minP = Infinity;
  let area = 0;
  const N = 20000;
  for (let i = 0; i <= N; i++) {
    const t = arrival + i * d / N;
    const val = shapedWave(t, arrival, 'x59', p, d, r);
    assert.ok(Number.isFinite(val));
    maxP = Math.max(maxP, val);
    minP = Math.min(minP, val);
    area += val * (d / N);
  }

  // Exact dipole/quadrupole zero-net-area conservation
  close(area, 0, 1e-6);

  // Reaches full peak and bounded within [-p, p]
  close(maxP, p, 1e-4);
  close(minP, -p, 1e-4);

  // Verify multi-stage ramp / staircase profile that prevents shock coalescence:
  // Initial stage rises to first plateau / shelf around 0.4 * peak (~12 Pa)
  const pShelf = shapedWave(arrival + r * 1.2, arrival, 'x59', p, d, r);
  close(pShelf, 0.4 * p, 0.5);

  // Soft rise time: at 2 ms, pressure is gentle (~2.3 Pa), unlike sharp N-wave which jumps immediately
  const pEarlyX59 = shapedWave(arrival + 0.002, arrival, 'x59', p, d, r);
  const pEarlyNWave = nWave(arrival + 0.002, arrival, p, d, 0.002);
  assert.ok(pEarlyX59 < pEarlyNWave * 0.5, 'X-59 soft rise must produce lower initial overpressure than sharp N-wave');

  // Verify that other aircraft types default to physical N-wave
  const f16Wave = shapedWave(arrival + 0.005, arrival, 'f16');
  const f16N = nWave(arrival + 0.005, arrival, AIRCRAFT_PRESETS.f16.peak, AIRCRAFT_PRESETS.f16.duration, AIRCRAFT_PRESETS.f16.rise);
  close(f16Wave, f16N, 1e-6);
});

test('lateral observer offset arrival time, slant range, and Whitham attenuation r^-0.75', () => {
  const m = 2.0;
  const h = 1000;

  // Slant range r_slant = Math.hypot(h, y)
  close(slantRange(h, 0), h, 1e-7);
  close(slantRange(h, 1000), Math.hypot(1000, 1000), 1e-7);
  close(slantRange(h, 2400), 2600, 1e-7);

  // Boom arrival time with lateral offset: t_boom(m, h, y) = (r_slant / C) * Math.sqrt(1 - 1/(m*m))
  const tCenter = boomTime(m, h, 0);
  const tLat1 = boomTime(m, h, 1000);
  const tLat2 = boomTime(m, h, 2000);
  assert.ok(tLat1 > tCenter, 'Lateral boom arrival must be delayed relative to centerline');
  assert.ok(tLat2 > tLat1, 'Boom arrival time must increase monotonically with lateral distance');

  const expectedArrival = (slantRange(h, 1000) / C) * Math.sqrt(1 - 1 / (m * m));
  close(tLat1, expectedArrival, 1e-6);
  close(boomTimeLateral(m, h, 1000), tLat1, 1e-6);

  // Subsonic has no boom arrival regardless of lateral offset
  assert.equal(boomTime(0.8, h, 1000), null);

  // Overpressure attenuation with lateral distance: peak(y) = peak_0 * (h / r_slant)**0.75
  const p0 = 100;
  const pAtZero = lateralOverpressure(p0, h, 0);
  close(pAtZero, p0, 1e-6);

  const pAt1000 = lateralOverpressure(p0, h, 1000);
  const r1000 = slantRange(h, 1000);
  const expectedP1000 = p0 * Math.pow(h / r1000, 0.75);
  close(pAt1000, expectedP1000, 1e-6);

  const pAt3000 = lateralOverpressure(p0, h, 3000);
  const r3000 = slantRange(h, 3000);
  const expectedP3000 = p0 * Math.pow(h / r3000, 0.75);
  close(pAt3000, expectedP3000, 1e-6);

  // Whitham power law exponent verification: log(p1/p2) / log(r2/r1) == 0.75
  const ratioP = pAt1000 / pAt3000;
  const ratioR = r3000 / r1000;
  close(Math.log(ratioP) / Math.log(ratioR), 0.75, 1e-6);

  // Verify aliases whithamAttenuation and whithamOverpressure
  close(whithamAttenuation(h, 1500), lateralAttenuation(h, 1500), 1e-7);
  close(whithamOverpressure(p0, h, 1500), lateralOverpressure(p0, h, 1500), 1e-7);

  // Verify renderAudio integration with lateral offset
  const audioCenter = renderAudio({ mach: 2, altitude: 1000, lateralOffset: 0, mode: 'boom', pressure: 100 });
  const audioOffset = renderAudio({ mach: 2, altitude: 1000, lateralOffset: 1500, mode: 'boom', pressure: 100 });
  assert.ok(audioOffset.arrival > audioCenter.arrival, 'Offset boom must arrive after centerline boom in rendered audio');
  assert.ok(audioOffset.peak < audioCenter.peak, 'Offset boom peak must be attenuated relative to centerline');
});

test('sonic boom carpet half-width geometry', () => {
  // carpetWidth(m, h) = 2 * h * Math.sqrt(m*m - 1)
  assert.equal(carpetWidth(0.5, 1000), 0);
  assert.equal(carpetWidth(1.0, 1000), 0);

  // At Mach sqrt(2) ~ 1.41421356: sqrt(M^2 - 1) = 1.0 => carpet width = 2 * h
  const mSqrt2 = Math.SQRT2;
  close(carpetWidth(mSqrt2, 1000), 2000, 1e-4);

  // At Mach 2, altitude 500m: 2 * 500 * sqrt(3) = 1000 * 1.73205...
  close(carpetWidth(2, 500), 1000 * Math.sqrt(3), 1e-4);

  // Carpet width scales linearly with altitude h
  const cw500 = carpetWidth(2, 500);
  const cw1000 = carpetWidth(2, 1000);
  close(cw1000, 2 * cw500, 1e-4);

  // Carpet width expands with Mach number
  assert.ok(carpetWidth(3, 1000) > carpetWidth(2, 1000));
  assert.ok(carpetWidth(5, 1000) > carpetWidth(3, 1000));
});

test('transonic Prandtl-Glauert vapor cone intensity profile', () => {
  // Below transonic regime (M < 0.85): zero condensation
  assert.equal(vaporConeIntensity(0.5), 0);
  assert.equal(vaporConeIntensity(0.7), 0);
  assert.equal(vaporConeIntensity(0.8), 0);

  // Above transonic regime (M > 1.20): zero condensation
  assert.equal(vaporConeIntensity(1.25), 0);
  assert.equal(vaporConeIntensity(2.0), 0);
  assert.equal(vaporConeIntensity(5.0), 0);

  // Peaks around Mach 0.96 - 1.04 where local flow reaches M = 1 over airframe
  for (let m = 0.96; m <= 1.0401; m += 0.02) {
    const intensity = vaporConeIntensity(m);
    assert.ok(intensity >= 0.99, `Intensity at Mach ${m.toFixed(2)} should be at peak, got ${intensity}`);
  }

  // Smooth rise approaching Mach 1 from subsonic
  assert.ok(vaporConeIntensity(0.94) > vaporConeIntensity(0.90));
  assert.ok(vaporConeIntensity(0.90) > vaporConeIntensity(0.86));

  // Smooth decay above Mach 1 into supersonic
  assert.ok(vaporConeIntensity(1.06) > vaporConeIntensity(1.10));
  assert.ok(vaporConeIntensity(1.10) > vaporConeIntensity(1.15));

  // Intensity is strictly bounded in [0, 1]
  for (let m = 0.5; m <= 2.5; m += 0.01) {
    const val = vaporConeIntensity(m);
    assert.ok(val >= 0 && val <= 1, `Intensity ${val} at M=${m} must be in [0, 1]`);
  }
});

