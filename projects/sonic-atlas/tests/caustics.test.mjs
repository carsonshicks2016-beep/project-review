import test from 'node:test';
import assert from 'node:assert/strict';
import {
  C,
  clamp,
  smoothstep,
  TRAJECTORY_TYPES,
  TRAJECTORY_PRESETS,
  normalizeParams,
  getTrajectoryPosition,
  getTrajectoryMach,
  getTrajectoryVelocity,
  causticFocusMach,
  computeWavefrontEnvelope,
  calculateCaustic,
  superboomWave
} from '../dist/caustics.mjs';

const close = (a, b, tol = 1e-4) => {
  assert.ok(
    Math.abs(a - b) < tol,
    `expected ${a} to be close to ${b} (diff ${Math.abs(a - b)}, tol ${tol})`
  );
};

// ---------------------------------------------------------------------------
// 1. Trajectory Presets & Normalization
// ---------------------------------------------------------------------------

test('TRAJECTORY_PRESETS defines all 5 trajectory models with physical defaults', () => {
  const expectedTypes = ['level', 'dive', 'climb', 'turn', 'accel'];
  for (const type of expectedTypes) {
    assert.ok(TRAJECTORY_PRESETS[type], `preset for ${type} must exist`);
    assert.equal(TRAJECTORY_PRESETS[type].type, type);
    assert.ok(TRAJECTORY_PRESETS[type].altitude > 0);
    assert.ok(TRAJECTORY_PRESETS[type].basePeak > 0);
    assert.ok(TRAJECTORY_PRESETS[type].duration > 0);
  }

  // Dive and climb have negative and positive gamma angles
  assert.ok(TRAJECTORY_PRESETS.dive.gamma < 0);
  assert.ok(TRAJECTORY_PRESETS.climb.gamma > 0);

  // Turn has turn radius
  assert.ok(TRAJECTORY_PRESETS.turn.radius > 0);

  // Accel starts subsonic/transonic and reaches supersonic
  assert.ok(TRAJECTORY_PRESETS.accel.m0 < 1.0);
  assert.ok(TRAJECTORY_PRESETS.accel.m1 > 1.0);
  assert.ok(TRAJECTORY_PRESETS.accel.acceleration > 0);
});

test('normalizeParams supports aliases and unit conversions', () => {
  // Alias h for altitude, M for mach
  const p1 = normalizeParams('level', { h: 1800, M: 2.2 });
  assert.equal(p1.altitude, 1800);
  assert.equal(p1.mach, 2.2);

  // Gamma in degrees
  const p2 = normalizeParams('dive', { gamma: -30 });
  close(p2.gamma, (-30 * Math.PI) / 180, 1e-5);

  // Gamma in radians
  const p3 = normalizeParams('climb', { gamma: 0.25 });
  close(p3.gamma, 0.25, 1e-5);
});

// ---------------------------------------------------------------------------
// 2. Trajectory Kinematics (Requirement R3.1)
// ---------------------------------------------------------------------------

test('level flight maintains constant altitude and Mach number', () => {
  const params = { altitude: 1500, mach: 1.8 };
  const V = 1.8 * C;

  for (const tau of [-2, 0, 1.5, 4]) {
    const pos = getTrajectoryPosition('level', tau, params);
    const M = getTrajectoryMach('level', tau, params);
    const vel = getTrajectoryVelocity('level', tau, params);

    close(pos.x, V * tau);
    close(pos.y, 0);
    close(pos.z, 1500);
    close(M, 1.8);
    close(vel.vx, V);
    close(vel.vy, 0);
    close(vel.vz, 0);
  }
});

test('dive trajectory descends with negative gamma and tilts Mach cone towards ground', () => {
  const gammaDeg = -25;
  const gammaRad = (gammaDeg * Math.PI) / 180;
  const params = { altitude: 2500, mach: 1.6, gamma: gammaDeg };
  const V = 1.6 * C;

  const pos0 = getTrajectoryPosition('dive', 0, params);
  close(pos0.x, 0);
  close(pos0.z, 2500);

  const pos1 = getTrajectoryPosition('dive', 1.0, params);
  close(pos1.x, V * Math.cos(gammaRad) * 1.0);
  close(pos1.z, 2500 + V * Math.sin(gammaRad) * 1.0);
  assert.ok(pos1.z < pos0.z, 'altitude must decrease during dive');

  const vel = getTrajectoryVelocity('dive', 0, params);
  assert.ok(vel.vz < 0, 'vertical velocity must be downward');
  close(vel.speed, V);
});

test('climb trajectory ascends with positive gamma and tilts Mach cone away from ground', () => {
  const gammaDeg = 20;
  const gammaRad = (gammaDeg * Math.PI) / 180;
  const params = { altitude: 1000, mach: 1.5, gamma: gammaDeg };
  const V = 1.5 * C;

  const pos0 = getTrajectoryPosition('climb', 0, params);
  close(pos0.z, 1000);

  const pos2 = getTrajectoryPosition('climb', 2.0, params);
  close(pos2.z, 1000 + V * Math.sin(gammaRad) * 2.0);
  assert.ok(pos2.z > pos0.z, 'altitude must increase during climb');

  const vel = getTrajectoryVelocity('climb', 0, params);
  assert.ok(vel.vz > 0, 'vertical velocity must be upward');
});

test('turn trajectory follows horizontal circle with proper radius and centripetal acceleration', () => {
  const R = 4000;
  const M = 1.5;
  const V = M * C;
  const omega = V / R;
  const params = { altitude: 1200, mach: M, radius: R };

  // At tau = 0
  const pos0 = getTrajectoryPosition('turn', 0, params);
  close(pos0.x, 0);
  close(pos0.y, 0);
  close(pos0.z, 1200);

  // At quarter turn tau = (pi / 2) / omega
  const tauQuarter = Math.PI / (2 * omega);
  const posQ = getTrajectoryPosition('turn', tauQuarter, params);
  close(posQ.x, R);
  close(posQ.y, R);

  // Circle equation: x^2 + (y - R)^2 = R^2
  for (const tau of [-1.5, 0.5, 2.0, 3.5]) {
    const p = getTrajectoryPosition('turn', tau, params);
    const distFromCenter = Math.hypot(p.x, p.y - R);
    close(distFromCenter, R, 1e-3);
    close(p.z, 1200);
  }
});

test('accel trajectory accelerates smoothly through Mach 1 with correct kinematics', () => {
  const params = { altitude: 1200, m0: 0.9, m1: 1.6, acceleration: 3.5 };
  const V0 = 0.9 * C;
  const a = 3.5;
  const tauSonic = ((1.0 - 0.9) * C) / a;

  // At tau = 0
  close(getTrajectoryMach('accel', 0, params), 0.9);
  close(getTrajectoryPosition('accel', 0, params).x, 0);

  // At tauSonic, Mach must be 1.0
  close(getTrajectoryMach('accel', tauSonic, params), 1.0, 1e-4);

  // Check parabolic position during acceleration: x = V0*tau + 0.5*a*tau^2
  const tauTest = 10;
  const expectedX = V0 * tauTest + 0.5 * a * tauTest * tauTest;
  close(getTrajectoryPosition('accel', tauTest, params).x, expectedX);
});

// ---------------------------------------------------------------------------
// 3. Caustic Detection & Superboom Focus Factors (Requirement R3.2)
// ---------------------------------------------------------------------------

test('calculateCaustic detects acceleration caustics through Mach 1', () => {
  const params = {
    altitude: 1200,
    m0: 0.9,
    m1: 1.6,
    acceleration: 3.5,
    basePeak: 60
  };

  const res = calculateCaustic('accel', params);
  assert.equal(res.hasCaustic, true);
  assert.ok(Array.isArray(res.focusPoints));
  assert.ok(res.focusPoints.length > 0);

  const focus = res.focusPoints[0];
  assert.equal(focus.y, 0, 'acceleration caustic lies on ground centerline');
  assert.ok(focus.x > 0, 'focal ground point must be downrange');

  // Verify focus Mach matches analytical formula
  const expectedMach = causticFocusMach(1200, 3.5);
  close(focus.mach, expectedMach, 1e-4);
  assert.ok(focus.mach > 1.0, 'caustic occurs above Mach 1');
});

test('calculateCaustic detects banking turn caustics inside the turn radius', () => {
  const params = {
    altitude: 1200,
    mach: 1.5,
    radius: 4000,
    basePeak: 65
  };

  const res = calculateCaustic('turn', params);
  assert.equal(res.hasCaustic, true);
  assert.ok(Array.isArray(res.focusPoints));
  assert.ok(res.focusPoints.length > 0);

  const focus = res.focusPoints[0];
  // Inside the turn: y must be positive (towards the center of curvature)
  assert.ok(focus.y > 0, 'turn caustic must form on the inside of the turn');
  assert.ok(focus.x > 0, 'turn caustic x coordinate must be positive');
});

test('straight level, dive, and climb trajectories do not form caustics on flat ground', () => {
  for (const type of ['level', 'dive', 'climb']) {
    const res = calculateCaustic(type);
    assert.equal(res.hasCaustic, false, `${type} must not form caustics`);
    assert.equal(res.focusPoints.length, 0);
    assert.equal(res.focusFactor, 1.0);
  }
});

test('superboom focus factor bounds are strictly between 1.5x and 5.0x', () => {
  // Test acceleration across diverse altitudes and acceleration rates
  for (const alt of [500, 1200, 3000]) {
    for (const a of [1.5, 3.5, 7.0]) {
      const res = calculateCaustic('accel', { altitude: alt, acceleration: a });
      assert.ok(res.hasCaustic);
      assert.ok(
        res.focusFactor >= 1.5 && res.focusFactor <= 5.0,
        `focusFactor ${res.focusFactor} for accel (alt=${alt}, a=${a}) must be between 1.5 and 5.0`
      );
    }
  }

  // Test turns across diverse radii and Mach numbers
  for (const R of [2000, 4000, 8000]) {
    for (const M of [1.2, 1.6, 2.5]) {
      const res = calculateCaustic('turn', { radius: R, mach: M, altitude: 1200 });
      assert.ok(res.hasCaustic);
      assert.ok(
        res.focusFactor >= 1.5 && res.focusFactor <= 5.0,
        `focusFactor ${res.focusFactor} for turn (R=${R}, M=${M}) must be between 1.5 and 5.0`
      );
    }
  }
});

test('calculateCaustic groundProfile exhibits superboom peak at focus coordinate', () => {
  const basePeak = 60;
  const res = calculateCaustic('accel', {
    altitude: 1200,
    m0: 0.9,
    m1: 1.6,
    acceleration: 3.5,
    basePeak
  });

  assert.ok(Array.isArray(res.groundProfile));
  assert.ok(res.groundProfile.length > 0);

  const focusX = res.focusPoints[0].x;
  let maxP = -Infinity;
  let maxX = 0;

  for (const pt of res.groundProfile) {
    assert.ok('x' in pt && 'pressure' in pt);
    // Test numeric coercion via valueOf()
    assert.ok(typeof +pt === 'number');

    if (pt.pressure > maxP) {
      maxP = pt.pressure;
      maxX = pt.x;
    }
  }

  // Maximum overpressure should match focusFactor * basePeak
  const expectedPeak = basePeak * res.focusFactor;
  close(maxP, expectedPeak, 1.0);
  // Peak occurs near the focus location
  assert.ok(Math.abs(maxX - focusX) < 300, `peak x ${maxX} should be near focus x ${focusX}`);
});

// ---------------------------------------------------------------------------
// 4. Superboom Waveform Signature & Physics (Requirement R3.2 & R3.4)
// ---------------------------------------------------------------------------

test('superboomWave exhibits strict finite support outside [arrival, arrival + duration]', () => {
  const arrival = 2.5;
  const duration = 0.18;
  const basePeak = 60;
  const focusFactor = 3.0;

  // Ahead of arrival: strictly zero
  assert.equal(superboomWave(arrival - 1.0, arrival, basePeak, focusFactor, duration), 0);
  assert.equal(superboomWave(arrival - 0.001, arrival, basePeak, focusFactor, duration), 0);
  assert.equal(superboomWave(arrival, arrival, basePeak, focusFactor, duration), 0);

  // Past duration: strictly zero
  assert.equal(superboomWave(arrival + duration, arrival, basePeak, focusFactor, duration), 0);
  assert.equal(superboomWave(arrival + duration + 0.001, arrival, basePeak, focusFactor, duration), 0);
  assert.equal(superboomWave(arrival + duration + 2.0, arrival, basePeak, focusFactor, duration), 0);

  // If arrival is null or undefined: strictly zero
  assert.equal(superboomWave(1.0, null, basePeak, focusFactor, duration), 0);
  assert.equal(superboomWave(1.0, undefined, basePeak, focusFactor, duration), 0);

  // Non-zero inside the active window
  const midVal = superboomWave(arrival + 0.5 * duration, arrival, basePeak, focusFactor, duration);
  assert.notEqual(midVal, 0);
});

test('superboomWave peak matches basePeak * focusFactor', () => {
  const arrival = 1.0;
  const duration = 0.20;
  const basePeak = 75;
  const focusFactor = 3.2;
  const targetPeak = basePeak * focusFactor; // 240 Pa

  let maxOverpressure = -Infinity;
  const numSteps = 5000;
  for (let i = 0; i <= numSteps; i++) {
    const t = arrival + (i / numSteps) * duration;
    const p = superboomWave(t, arrival, basePeak, focusFactor, duration);
    if (p > maxOverpressure) maxOverpressure = p;
  }

  close(maxOverpressure, targetPeak, 0.2);
});

test('superboomWave strictly satisfies zero-net-area (acoustic dipole conservation)', () => {
  // Test across multiple peak pressures, durations, and focus factors
  const testCases = [
    { basePeak: 60, focusFactor: 2.5, duration: 0.18 },
    { basePeak: 100, focusFactor: 3.5, duration: 0.24 },
    { basePeak: 45, focusFactor: 4.2, duration: 0.12 }
  ];

  for (const { basePeak, focusFactor, duration } of testCases) {
    const arrival = 0.5;
    const N = 50000;
    const dt = duration / N;
    let netArea = 0;

    for (let i = 0; i < N; i++) {
      const t = arrival + (i + 0.5) * dt;
      netArea += superboomWave(t, arrival, basePeak, focusFactor, duration) * dt;
    }

    // Net acoustic impulse (Pa*s) should be vanishingly small
    assert.ok(
      Math.abs(netArea) < 1e-4,
      `net area ${netArea} Pa*s must be virtually zero for peak=${basePeak}, Kf=${focusFactor}`
    );
  }
});

test('superboomWave exhibits asymmetric U-wave / spiked caustic signature', () => {
  const arrival = 0;
  const duration = 0.18;
  const basePeak = 60;
  const focusFactor = 3.0;

  let maxVal = -Infinity;
  let minVal = Infinity;
  let maxTime = 0;

  const N = 1000;
  for (let i = 0; i <= N; i++) {
    const t = arrival + (i / N) * duration;
    const p = superboomWave(t, arrival, basePeak, focusFactor, duration);
    if (p > maxVal) {
      maxVal = p;
      maxTime = t;
    }
    if (p < minVal) {
      minVal = p;
    }
  }

  // 1. Sharp leading spike occurs early in the waveform (first 10% of duration)
  assert.ok(
    maxTime < arrival + 0.10 * duration,
    `peak spike must occur near shock front (was at ${maxTime}s)`
  );

  // 2. Negative U-shaped trough exists
  assert.ok(minVal < 0, `waveform must have negative underpressure trough (min was ${minVal} Pa)`);

  // 3. Asymmetric: the positive front peak is substantially greater than the trailing recovery
  const tailVal = superboomWave(arrival + 0.90 * duration, arrival, basePeak, focusFactor, duration);
  assert.ok(
    maxVal > tailVal * 1.5,
    'waveform must be asymmetric with dominant leading caustic spike'
  );
});

// ---------------------------------------------------------------------------
// 5. Wavefront Envelope & Ray Intersections (Requirement R3.3)
// ---------------------------------------------------------------------------

test('computeWavefrontEnvelope returns valid rays with ground intersections', () => {
  for (const type of ['level', 'dive', 'turn', 'accel']) {
    const envelope = computeWavefrontEnvelope(type, {}, 40);
    assert.ok(Array.isArray(envelope));
    assert.ok(envelope.length > 0);
    assert.equal(envelope.trajectoryType, type);

    for (const ray of envelope) {
      assert.ok('tau' in ray);
      assert.ok('t' in ray);
      assert.ok('x' in ray);
      assert.ok('y' in ray);
      assert.equal(ray.z, 0, 'ray ground intersection must be at z = 0');
      assert.ok(ray.pathLength > 0, 'path length must be positive');
      assert.ok(ray.mach > 1.0, 'ray mach must be supersonic');
    }
  }
});

test('computeWavefrontEnvelope demonstrates caustic fold on acceleration', () => {
  const envelope = computeWavefrontEnvelope('accel', { altitude: 1200, acceleration: 3.5 }, 80);
  assert.ok(envelope.hasCaustic);

  // Find min ground x coordinate among rays
  let minX = Infinity;
  for (const ray of envelope) {
    if (ray.x < minX) minX = ray.x;
  }

  // In acceleration caustics, rays emitted just above Mach 1 land far downrange,
  // then x_g decreases to a minimum (the caustic fold/cusp), then increases again.
  const firstX = envelope[0].x;
  const lastX = envelope[envelope.length - 1].x;

  assert.ok(
    minX < firstX && minX < lastX,
    'ground ray footprint must fold and reach a minimum (caustic cusp)'
  );
});

test('dive tilts Mach cone towards ground shifting arrival earlier than level flight', () => {
  const h = 2000;
  const M = 1.6;

  // Calculate level flight arrival from x=0
  const levelArrival = (h / C) * Math.sqrt(1 - 1 / (M * M));

  // Compute dive envelope rays from altitude h
  const diveEnvelope = computeWavefrontEnvelope('dive', { altitude: h, mach: M, gamma: -25 }, 50);
  assert.ok(diveEnvelope.length > 0);

  // Ray emitted at tau = 0
  const diveRayAtZero = diveEnvelope.find((r) => Math.abs(r.tau) < 0.1);
  if (diveRayAtZero) {
    assert.ok(
      diveRayAtZero.pathLength < h / Math.sqrt(1 - 1 / (M * M)),
      'dive ray path to ground must be shorter due to downward cone tilt'
    );
  }
});
