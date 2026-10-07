/**
 * Flight Dynamics & Shock Caustics ("Superbooms")
 *
 * Implements:
 * - 3D trajectory kinematics: level, dive, climb, banking turn, and transonic acceleration.
 * - Shock wavefront ray envelopes and ground ray intersections.
 * - Fold/cusp caustic singularities on acceleration and turns.
 * - Physical focus factors K_focus (2.0x to 5.0x) regularized by nonlinear transonic scaling.
 * - Focused superboom waveforms with asymmetric U-wave / spiked caustic signatures,
 *   strictly maintaining finite support and zero net acoustic area (zero-net-area dipole).
 */

export const C = 343; // Standard speed of sound in m/s

export const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

export const smoothstep = (x) => {
  x = clamp(x, 0, 1);
  return x * x * (3 - 2 * x);
};

export const TRAJECTORY_TYPES = ['level', 'dive', 'climb', 'turn', 'accel'];

export const TRAJECTORY_PRESETS = {
  level: {
    id: 'level',
    type: 'level',
    name: 'Straight Level Flight',
    mach: 1.5,
    altitude: 1200,
    basePeak: 60,
    duration: 0.18
  },
  dive: {
    id: 'dive',
    type: 'dive',
    name: 'Supersonic Dive (-20°)',
    mach: 1.6,
    altitude: 2000,
    gamma: -20, // degrees
    basePeak: 75,
    duration: 0.16
  },
  climb: {
    id: 'climb',
    type: 'climb',
    name: 'Supersonic Climb (+15°)',
    mach: 1.5,
    altitude: 1000,
    gamma: 15, // degrees
    basePeak: 55,
    duration: 0.20
  },
  turn: {
    id: 'turn',
    type: 'turn',
    name: 'Horizontal Banking Turn',
    mach: 1.5,
    altitude: 1200,
    radius: 4000, // meters
    basePeak: 65,
    duration: 0.18
  },
  accel: {
    id: 'accel',
    type: 'accel',
    name: 'Transonic Acceleration (M0.9 -> M1.6)',
    m0: 0.9,
    m1: 1.6,
    altitude: 1200,
    acceleration: 3.5, // m/s^2
    basePeak: 60,
    duration: 0.18
  }
};

/**
 * Normalize and validate trajectory parameters with sensible defaults.
 */
export function normalizeParams(type, params = {}) {
  const preset = TRAJECTORY_PRESETS[type] || TRAJECTORY_PRESETS.level;
  const altitude = Number(params.altitude ?? params.h ?? preset.altitude ?? 1200);
  const mach = Number(params.mach ?? params.M ?? preset.mach ?? 1.5);
  const basePeak = Number(params.basePeak ?? params.pressure ?? preset.basePeak ?? 60);
  const duration = Number(params.duration ?? preset.duration ?? 0.18);

  // Gamma for dive/climb (supports degrees or radians)
  let gamma = params.gamma ?? preset.gamma ?? 0;
  if (params.gammaDeg !== undefined) {
    gamma = (Number(params.gammaDeg) * Math.PI) / 180;
  } else if (Math.abs(gamma) > 1.5) {
    // Value given in degrees (e.g. -20 or 15)
    gamma = (Number(gamma) * Math.PI) / 180;
  } else {
    gamma = Number(gamma);
  }

  // Turn parameters
  const radius = Number(params.radius ?? params.R ?? preset.radius ?? 4000);
  const V_nom = mach * C;
  const turnRate = Number(params.turnRate ?? params.omega ?? (V_nom / Math.max(1, radius)));

  // Acceleration parameters
  const m0 = Number(params.m0 ?? params.M0 ?? preset.m0 ?? 0.9);
  const m1 = Number(params.m1 ?? params.M1 ?? preset.m1 ?? 1.6);
  const acceleration = Number(params.acceleration ?? params.a ?? preset.acceleration ?? 3.5);

  return {
    altitude,
    h: altitude,
    mach,
    M: mach,
    basePeak,
    duration,
    gamma,
    radius,
    R: radius,
    turnRate,
    omega: turnRate,
    m0,
    M0: m0,
    m1,
    M1: m1,
    acceleration,
    a: acceleration
  };
}

/**
 * Get 3D aircraft position { x, y, z } in meters at emission time tau (seconds).
 */
export function getTrajectoryPosition(type, tau, params = {}) {
  const p = normalizeParams(type, params);
  tau = Number(tau);

  switch (type) {
    case 'dive': {
      const V = p.mach * C;
      const x = V * Math.cos(p.gamma) * tau;
      const y = 0;
      const z = Math.max(0, p.altitude + V * Math.sin(p.gamma) * tau);
      return { x, y, z };
    }
    case 'climb': {
      const V = p.mach * C;
      const x = V * Math.cos(p.gamma) * tau;
      const y = 0;
      const z = p.altitude + V * Math.sin(p.gamma) * tau;
      return { x, y, z };
    }
    case 'turn': {
      const omega = p.turnRate;
      const R = p.radius;
      // At tau = 0, aircraft is at (0, 0, h) heading +x, turning towards +y
      const x = R * Math.sin(omega * tau);
      const y = R * (1 - Math.cos(omega * tau));
      const z = p.altitude;
      return { x, y, z };
    }
    case 'accel': {
      const V0 = p.m0 * C;
      const V1 = p.m1 * C;
      const a = p.acceleration;
      const tau1 = a > 0 ? (V1 - V0) / a : 0;
      let x = 0;
      if (tau <= 0) {
        x = V0 * tau;
      } else if (tau <= tau1) {
        x = V0 * tau + 0.5 * a * tau * tau;
      } else {
        const x1 = V0 * tau1 + 0.5 * a * tau1 * tau1;
        x = x1 + V1 * (tau - tau1);
      }
      return { x, y: 0, z: p.altitude };
    }
    case 'level':
    default: {
      const V = p.mach * C;
      const x = V * tau;
      const y = 0;
      const z = p.altitude;
      return { x, y, z };
    }
  }
}

/**
 * Get instantaneous Mach number at emission time tau (seconds).
 */
export function getTrajectoryMach(type, tau, params = {}) {
  const p = normalizeParams(type, params);
  tau = Number(tau);

  if (type === 'accel') {
    const V0 = p.m0 * C;
    const V1 = p.m1 * C;
    const a = p.acceleration;
    const tau1 = a > 0 ? (V1 - V0) / a : 0;
    if (tau <= 0) return p.m0;
    if (tau >= tau1) return p.m1;
    return (V0 + a * tau) / C;
  }

  return p.mach;
}

/**
 * Get instantaneous 3D velocity vector { vx, vy, vz, speed } in m/s at emission time tau.
 */
export function getTrajectoryVelocity(type, tau, params = {}) {
  const p = normalizeParams(type, params);
  tau = Number(tau);

  switch (type) {
    case 'dive': {
      const V = p.mach * C;
      return {
        vx: V * Math.cos(p.gamma),
        vy: 0,
        vz: V * Math.sin(p.gamma),
        speed: V
      };
    }
    case 'climb': {
      const V = p.mach * C;
      return {
        vx: V * Math.cos(p.gamma),
        vy: 0,
        vz: V * Math.sin(p.gamma),
        speed: V
      };
    }
    case 'turn': {
      const V = p.mach * C;
      const omega = p.turnRate;
      return {
        vx: V * Math.cos(omega * tau),
        vy: V * Math.sin(omega * tau),
        vz: 0,
        speed: V
      };
    }
    case 'accel': {
      const M = getTrajectoryMach('accel', tau, p);
      const V = M * C;
      return { vx: V, vy: 0, vz: 0, speed: V };
    }
    case 'level':
    default: {
      const V = p.mach * C;
      return { vx: V, vy: 0, vz: 0, speed: V };
    }
  }
}

/**
 * Analytical acceleration caustic Mach number where ray density dX_g/dtau = 0.
 * Derived from (M^2 - 1)^(3/2) = (h * a) / C^2.
 */
export function causticFocusMach(altitude, acceleration) {
  const C2 = C * C;
  const param = Math.max(0, (altitude * acceleration) / C2);
  return Math.sqrt(1 + Math.pow(param, 2 / 3));
}

/**
 * Computes ray paths and ground intersections for the acoustic wavefront envelope.
 * Returns an array of ray objects with ground impact coordinates and arrival times.
 */
export function computeWavefrontEnvelope(trajectoryType, params = {}, numRays = 100) {
  const type = trajectoryType || 'level';
  const p = normalizeParams(type, params);
  const count = Math.max(10, Math.min(2000, Number(numRays) || 100));
  const rays = [];

  switch (type) {
    case 'accel': {
      const a = p.acceleration;
      const tauSonic = a > 0 ? Math.max(0, ((1.001 - p.m0) * C) / a) : 0;
      const tauEnd = a > 0 ? Math.max(tauSonic + 0.1, ((p.m1 - p.m0) * C) / a + 2) : 5;
      const dTau = (tauEnd - tauSonic) / (count - 1);

      for (let i = 0; i < count; i++) {
        const tau = tauSonic + i * dTau;
        const M = getTrajectoryMach('accel', tau, p);
        if (M <= 1.0001) continue;

        const pos = getTrajectoryPosition('accel', tau, p);
        const mu = Math.asin(1 / M); // Mach angle
        const rayAngle = Math.PI / 2 - mu; // angle from velocity vector
        const dx = Math.cos(rayAngle);
        const dy = 0;
        const dz = -Math.sin(rayAngle); // downward

        const pathLength = pos.z / Math.sin(rayAngle);
        const xg = pos.x + pos.z / Math.tan(rayAngle);
        const yg = 0;
        const arrival = tau + pathLength / C;

        rays.push({
          tau,
          t: arrival,
          arrival,
          x: xg,
          y: yg,
          z: 0,
          sourcePos: pos,
          groundPos: { x: xg, y: yg, z: 0 },
          direction: { dx, dy, dz },
          mach: M,
          pathLength
        });
      }
      break;
    }
    case 'turn': {
      const omega = p.turnRate;
      const M = p.mach;
      const R = p.radius;
      const h = p.altitude;
      const mu = M > 1 ? Math.asin(1 / M) : Math.PI / 2;
      const rayAngle = Math.PI / 2 - mu;

      // Inside-turn focus ray elevation/azimuth
      const tanPhi = (R / Math.max(1, h)) * ((M * M - 1) / (M * M));
      const phiFocus = Math.atan(tanPhi);

      const spanAngle = Math.PI * 0.75; // 135 deg arc
      const tauMax = spanAngle / Math.max(1e-4, omega);
      const tauMin = -tauMax;
      const dTau = (tauMax - tauMin) / (count - 1);

      for (let i = 0; i < count; i++) {
        const tau = tauMin + i * dTau;
        const pos = getTrajectoryPosition('turn', tau, p);
        const psi = omega * tau;

        // Tangent and inward normal unit vectors
        const tx = Math.cos(psi), ty = Math.sin(psi);
        const inX = -Math.sin(psi), inY = Math.cos(psi);

        // Ray direction with azimuth phiFocus towards inside of turn
        const cosRay = Math.cos(rayAngle);
        const sinRay = Math.sin(rayAngle);
        const cosPhi = Math.cos(phiFocus);
        const sinPhi = Math.sin(phiFocus);

        const dx = cosRay * tx + sinRay * sinPhi * inX;
        const dy = cosRay * ty + sinRay * sinPhi * inY;
        const dz = -sinRay * cosPhi;

        if (dz >= -1e-4) continue;

        const pathLength = -pos.z / dz;
        const xg = pos.x + pathLength * dx;
        const yg = pos.y + pathLength * dy;
        const arrival = tau + pathLength / C;

        rays.push({
          tau,
          t: arrival,
          arrival,
          x: xg,
          y: yg,
          z: 0,
          sourcePos: pos,
          groundPos: { x: xg, y: yg, z: 0 },
          direction: { dx, dy, dz },
          mach: M,
          pathLength
        });
      }
      break;
    }
    case 'dive': {
      const M = p.mach;
      const tauSpan = 6;
      const dTau = tauSpan / (count - 1);
      const tauStart = -tauSpan * 0.5;

      for (let i = 0; i < count; i++) {
        const tau = tauStart + i * dTau;
        const pos = getTrajectoryPosition('dive', tau, p);
        if (pos.z <= 0) continue;

        const mu = M > 1 ? Math.asin(1 / M) : Math.PI / 2;
        const rayAngleFromV = Math.PI / 2 - mu;
        const thetaRay = p.gamma - rayAngleFromV; // tilted steeper by gamma < 0

        const dx = Math.cos(thetaRay);
        const dy = 0;
        const dz = Math.sin(thetaRay); // negative (downward)

        if (dz >= 0) continue;

        const pathLength = -pos.z / dz;
        const xg = pos.x + pathLength * dx;
        const yg = 0;
        const arrival = tau + pathLength / C;

        rays.push({
          tau,
          t: arrival,
          arrival,
          x: xg,
          y: yg,
          z: 0,
          sourcePos: pos,
          groundPos: { x: xg, y: yg, z: 0 },
          direction: { dx, dy, dz },
          mach: M,
          pathLength
        });
      }
      break;
    }
    case 'climb': {
      const M = p.mach;
      const tauSpan = 6;
      const dTau = tauSpan / (count - 1);
      const tauStart = -tauSpan * 0.5;

      for (let i = 0; i < count; i++) {
        const tau = tauStart + i * dTau;
        const pos = getTrajectoryPosition('climb', tau, p);

        const mu = M > 1 ? Math.asin(1 / M) : Math.PI / 2;
        const rayAngleFromV = Math.PI / 2 - mu;
        const thetaRay = p.gamma - rayAngleFromV; // tilted away from ground by gamma > 0

        const dx = Math.cos(thetaRay);
        const dy = 0;
        const dz = Math.sin(thetaRay);

        // Rays only hit ground if pointing downwards
        if (dz >= -1e-4) continue;

        const pathLength = -pos.z / dz;
        const xg = pos.x + pathLength * dx;
        const yg = 0;
        const arrival = tau + pathLength / C;

        rays.push({
          tau,
          t: arrival,
          arrival,
          x: xg,
          y: yg,
          z: 0,
          sourcePos: pos,
          groundPos: { x: xg, y: yg, z: 0 },
          direction: { dx, dy, dz },
          mach: M,
          pathLength
        });
      }
      break;
    }
    case 'level':
    default: {
      const M = p.mach;
      const tauSpan = 8;
      const dTau = tauSpan / (count - 1);
      const tauStart = -tauSpan * 0.5;

      for (let i = 0; i < count; i++) {
        const tau = tauStart + i * dTau;
        const pos = getTrajectoryPosition('level', tau, p);

        if (M <= 1) continue;
        const mu = Math.asin(1 / M);
        const rayAngle = Math.PI / 2 - mu;
        const dx = Math.cos(rayAngle);
        const dy = 0;
        const dz = -Math.sin(rayAngle);

        const pathLength = pos.z / Math.sin(rayAngle);
        const xg = pos.x + pos.z / Math.tan(rayAngle);
        const yg = 0;
        const arrival = tau + pathLength / C;

        rays.push({
          tau,
          t: arrival,
          arrival,
          x: xg,
          y: yg,
          z: 0,
          sourcePos: pos,
          groundPos: { x: xg, y: yg, z: 0 },
          direction: { dx, dy, dz },
          mach: M,
          pathLength
        });
      }
      break;
    }
  }

  // Attach metadata to the returned array for maximum ergonomics
  const causticInfo = calculateCaustic(type, p);
  rays.trajectoryType = type;
  rays.hasCaustic = causticInfo.hasCaustic;
  rays.focusPoints = causticInfo.focusPoints;
  rays.focusFactor = causticInfo.focusFactor;
  rays.rays = rays;

  return rays;
}

/**
 * Calculate caustic fold/cusp singularities and superboom magnification factors.
 * Returns:
 * - hasCaustic: boolean
 * - focusPoints: array of ground locations { x, y } where caustic hits ground
 * - focusFactor: overpressure magnification factor (e.g. 2.2x to 4.5x)
 * - groundProfile: array of { x, pressure, p, valueOf() } along ground coordinate x
 */
export function calculateCaustic(typeOrParams, maybeParams) {
  let type = 'level';
  let params = {};
  if (typeof typeOrParams === 'object' && typeOrParams !== null) {
    params = typeOrParams;
    type = params.type || params.trajectoryType || 'level';
  } else {
    type = typeOrParams || 'level';
    params = maybeParams || {};
  }

  const p = normalizeParams(type, params);
  const basePeak = p.basePeak;
  const h = p.altitude;
  const C2 = C * C;

  let hasCaustic = false;
  let focusPoints = [];
  let focusFactor = 1.0;
  const groundProfile = [];

  if (type === 'accel') {
    const a = p.acceleration;
    const m0 = p.m0;
    const m1 = p.m1;

    // Linear acceleration through Mach 1 causes caustic cusp when dX_g/dtau = 0
    if (a > 0 && m1 > 1.0) {
      const M_focus = causticFocusMach(h, a);

      // Check if caustic Mach falls within acceleration window
      if (M_focus >= m0 && M_focus <= m1 + 0.1) {
        hasCaustic = true;

        const tauFocus = Math.max(0, ((M_focus - m0) * C) / a);
        const x_ac = m0 * C * tauFocus + 0.5 * a * tauFocus * tauFocus;
        const cotMu = Math.sqrt(Math.max(1e-4, M_focus * M_focus - 1));
        const x_focus = x_ac + h / cotMu;
        const y_focus = 0;
        const t_focus = tauFocus + (h * M_focus) / (C * cotMu);

        focusPoints.push({
          x: x_focus,
          y: y_focus,
          t: t_focus,
          mach: M_focus,
          altitude: h
        });

        // Nonlinear transonic focus factor scaling (Guiraud similitude: 2.2x - 4.5x)
        const dimless = (a * h) / C2;
        focusFactor = clamp(2.0 + 1.5 * Math.pow(dimless, 1 / 6), 2.0, 5.0);

        // Ground profile along x around the focus cusp
        const xSpan = 12000;
        const nSteps = 101;
        const xMin = x_focus - xSpan * 0.5;
        const dx = xSpan / (nSteps - 1);

        for (let i = 0; i < nSteps; i++) {
          const x = xMin + i * dx;
          let pressure = basePeak;

          if (x <= x_focus) {
            // Illuminated zone behind caustic: smooth decay back to nominal boom
            const dist = (x_focus - x) / 1800;
            const boost = (focusFactor - 1.0) * Math.exp(-Math.pow(dist, 1.25));
            pressure = basePeak * (1.0 + boost);
          } else {
            // Shadow diffraction zone ahead of caustic: exponential decay
            const dist = (x - x_focus) / 850;
            pressure = basePeak * focusFactor * Math.exp(-dist);
          }

          groundProfile.push({
            x,
            pressure,
            p: pressure,
            valueOf() {
              return this.pressure;
            },
            toString() {
              return String(this.pressure);
            }
          });
        }
      }
    }
  } else if (type === 'turn') {
    const R = p.radius;
    const M = p.mach;

    if (M > 1 && R > 0) {
      hasCaustic = true;
      const V = M * C;
      const aN = (V * V) / R;

      // Inward banking turn ray envelope focal coordinates
      const tanPhi = (R / Math.max(1, h)) * ((M * M - 1) / (M * M));
      const cotMu = Math.sqrt(M * M - 1);
      const y_focus = R * ((M * M - 1) / (M * M));
      const x_focus = (h * Math.sqrt(1 + tanPhi * tanPhi)) / cotMu;

      focusPoints.push({
        x: x_focus,
        y: y_focus,
        radius: R,
        mach: M,
        altitude: h,
        centripetalAccel: aN
      });

      // Turning flight caustic focus factor scaling (2.2x - 4.5x)
      const dimless = (aN * h) / C2;
      focusFactor = clamp(2.0 + 1.2 * Math.pow(dimless, 1 / 6), 2.0, 4.5);

      // Ground profile along x coordinate near the focus line
      const xSpan = 10000;
      const nSteps = 101;
      const xMin = x_focus - xSpan * 0.5;
      const dx = xSpan / (nSteps - 1);

      for (let i = 0; i < nSteps; i++) {
        const x = xMin + i * dx;
        const dist = Math.abs(x - x_focus) / 1500;
        const boost = (focusFactor - 1.0) * Math.exp(-dist * dist);
        const pressure = basePeak * (1.0 + boost);

        groundProfile.push({
          x,
          pressure,
          p: pressure,
          valueOf() {
            return this.pressure;
          },
          toString() {
            return String(this.pressure);
          }
        });
      }
    }
  } else {
    // Non-focusing trajectories: level, dive, climb
    hasCaustic = false;
    focusFactor = 1.0;
    focusPoints = [];

    // Steeper ground impact in dive vs shallower in climb
    let impactFactor = 1.0;
    if (type === 'dive') {
      const mu = p.mach > 1 ? Math.asin(1 / p.mach) : Math.PI / 2;
      const normalAngle = Math.PI / 2 - mu;
      const diveRayAngle = Math.abs(p.gamma) + normalAngle;
      impactFactor = clamp(Math.sin(diveRayAngle) / Math.max(1e-4, Math.sin(normalAngle)), 1.0, 1.35);
    } else if (type === 'climb') {
      const mu = p.mach > 1 ? Math.asin(1 / p.mach) : Math.PI / 2;
      const normalAngle = Math.PI / 2 - mu;
      const climbRayAngle = Math.max(0.01, normalAngle - Math.abs(p.gamma));
      impactFactor = clamp(Math.sin(climbRayAngle) / Math.max(1e-4, Math.sin(normalAngle)), 0.6, 1.0);
    }

    const xSpan = 10000;
    const nSteps = 80;
    const dx = xSpan / (nSteps - 1);
    for (let i = 0; i < nSteps; i++) {
      const x = -xSpan * 0.5 + i * dx;
      const pressure = basePeak * impactFactor;
      groundProfile.push({
        x,
        pressure,
        p: pressure,
        valueOf() {
          return this.pressure;
        },
        toString() {
          return String(this.pressure);
        }
      });
    }
  }

  return {
    hasCaustic,
    focusPoints,
    focusFactor,
    groundProfile
  };
}

/**
 * Precomputed coefficients for the analytical asymmetric U-wave / caustic spiked waveform.
 * Exact analytical integral over [0, 1] is mathematically ZERO, ensuring strict zero-net-area.
 */
const B1 = 50;
const B3 = 25;
const I1 = 2 / ((B1 + 1) * (B1 + 2) * (B1 + 3));
const I2 = 1 / 140; // xi^3 * (1 - xi)^3
const I3 = 2 / ((B3 + 1) * (B3 + 2) * (B3 + 3));

const XI1_STAR = 2 / (B1 + 2);
const PEAK1_RAW = Math.pow(XI1_STAR, 2) * Math.pow(1 - XI1_STAR, B1);

const XI3_STAR = B3 / (B3 + 2);
const PEAK3_RAW = Math.pow(XI3_STAR, B3) * Math.pow(1 - XI3_STAR, 2);

const C1 = 1.0 / PEAK1_RAW;
const C3 = 0.40 / PEAK3_RAW;
const C2 = (C1 * I1 + C3 * I3) / I2;

function rawUWave(xi) {
  if (xi <= 0 || xi >= 1) return 0;
  const t1 = C1 * xi * xi * Math.pow(1 - xi, B1);
  const t2 = C2 * Math.pow(xi, 3) * Math.pow(1 - xi, 3);
  const t3 = C3 * Math.pow(xi, B3) * (1 - xi) * (1 - xi);
  return t1 - t2 + t3;
}

// Compute maximum of raw waveform for exact peak normalization
let NORM_PEAK = 0;
for (let i = 1; i <= 20000; i++) {
  const v = rawUWave(i / 20000);
  if (v > NORM_PEAK) NORM_PEAK = v;
}

/**
 * Generates a focused superboom waveform with asymmetric U-wave and spiked caustic signature.
 *
 * Guarantees:
 * - Finite support: strictly zero outside [arrival, arrival + duration]
 * - Zero net area: integral_{arrival}^{arrival+duration} p(t) dt = 0
 * - Peak overpressure: exactly reaches basePeak * focusFactor at the leading caustic shock
 * - Asymmetric U-wave profile with sharp leading spike, negative trough, and recompression
 */
export function superboomWave(t, arrival, basePeak = 60, focusFactor = 2.5, duration = 0.18, rise = 0.002) {
  if (arrival === null || arrival === undefined) return 0;

  // Handle optional object argument
  if (typeof basePeak === 'object' && basePeak !== null) {
    const opts = basePeak;
    basePeak = opts.basePeak ?? opts.pressure ?? 60;
    focusFactor = opts.focusFactor ?? 2.5;
    duration = opts.duration ?? 0.18;
    rise = opts.rise ?? 0.002;
  }

  t = Number(t);
  arrival = Number(arrival);
  basePeak = Number(basePeak);
  focusFactor = Number(focusFactor);
  duration = Number(duration);

  const u = t - arrival;
  // Finite support check: strictly zero ahead of shock arrival or after duration
  if (u <= 0 || u >= duration) return 0;

  const xi = u / duration;
  if (xi <= 0 || xi >= 1) return 0;

  const normalizedProfile = rawUWave(xi) / NORM_PEAK;
  return basePeak * focusFactor * normalizedProfile;
}
