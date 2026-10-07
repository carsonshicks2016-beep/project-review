/**
 * WGSL compute shaders for 2D Position-Based Fluids (Macklin & Müller).
 * Kernels / constants match the CPU solver in pbf-cpu.ts.
 *
 * Passes:
 *   integrate     — gravity + wind, predict positions
 *   clearHash     — reset spatial-hash cell heads
 *   insertHash    — atomic linked-list insert of predicted positions
 *   computeLambda — density + constraint λ (neighbour gather)
 *   computeDelta  — pressure / artificial-pressure position deltas
 *   applyDelta    — add Δp to predicted positions
 *   collideBoundsSoft / collideBoundsKill — tank walls (soft vs restitution≈0)
 *   updateVel     — v = (p − x)/dt
 *   viscosityAccum / viscosityApply — XSPH
 */

export const PBF_WORKGROUP_SIZE = 64;

export const PBF_SHADER_CODE = /* wgsl */ `
// ── Shared uniforms (std140-ish packing; keep 16-byte groups) ───────────────
struct SimParams {
  gravity: f32,
  viscosity: f32,
  dt: f32,
  wind: f32,

  h: f32,
  h2: f32,
  restDensity: f32,
  eps: f32,

  poly6Coeff: f32,
  spikyBase: f32,
  wDq: f32,
  kCorr: f32,

  nCorr: f32,
  invCell: f32,
  particleCount: u32,
  cellMask: u32,

  bounds: vec4<f32>, // x0, y0, x1, y1
};

// Packed particle (32 bytes) — matches PARTICLE_STRIDE_BYTES on CPU.
struct Particle {
  x: f32,
  y: f32,
  px: f32,
  py: f32,
  vx: f32,
  vy: f32,
  invMass: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> params: SimParams;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> lambdaBuf: array<f32>;
@group(0) @binding(3) var<storage, read_write> deltaX: array<f32>;
@group(0) @binding(4) var<storage, read_write> deltaY: array<f32>;
@group(0) @binding(5) var<storage, read_write> cellHead: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> cellNext: array<u32>;

const EMPTY: u32 = 0xffffffffu;
const EPS_R: f32 = 1e-8;

fn poly6(r2: f32) -> f32 {
  if (r2 >= params.h2) { return 0.0; }
  let d = params.h2 - r2;
  return params.poly6Coeff * d * d * d;
}

// Spiky gradient coeff c such that ∇_i W ≈ c * (dx, dy) with dx = xj − xi (c ≤ 0).
fn spikyGradCoeff(r: f32) -> f32 {
  if (r <= EPS_R || r >= params.h) { return 0.0; }
  let t = params.h - r;
  return (params.spikyBase * t * t) / r;
}

fn cellHash(cx: i32, cy: i32) -> u32 {
  // Match JS SpatialHash: ((cx * 73856093) ^ (cy * 19349663)) & mask
  let a = cx * 73856093;
  let b = cy * 19349663;
  return u32(a ^ b) & params.cellMask;
}

fn cellOf(px: f32, py: f32) -> vec2<i32> {
  return vec2<i32>(i32(floor(px * params.invCell)), i32(floor(py * params.invCell)));
}

// ── Pass: integrate — apply gravity/wind, write predicted (px, py) ──────────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  var p = particles[i];
  if (p.invMass == 0.0) {
    p.px = p.x;
    p.py = p.y;
    particles[i] = p;
    return;
  }

  let dt = params.dt;
  p.vx += params.wind * dt;
  p.vy += params.gravity * dt;
  p.px = p.x + p.vx * dt;
  p.py = p.y + p.vy * dt;
  particles[i] = p;
}

// ── Pass: clearHash — set every cell head to EMPTY ──────────────────────────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn clearHash(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  // cellMask is pow2−1; cell count = mask+1
  if (c > params.cellMask) { return; }
  atomicStore(&cellHead[c], EMPTY);
}

// ── Pass: insertHash — atomic linked-list insert by predicted position ──────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn insertHash(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  let p = particles[i];
  let cell = cellOf(p.px, p.py);
  let h = cellHash(cell.x, cell.y);
  let old = atomicExchange(&cellHead[h], i);
  cellNext[i] = old;
}

// ── Pass: computeLambda — density constraint λ_i ────────────────────────────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn computeLambda(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  let pi = particles[i];
  if (pi.invMass == 0.0) {
    lambdaBuf[i] = 0.0;
    return;
  }

  let rho0 = params.restDensity;
  var rho = poly6(0.0); // self
  var sumGradSq = 0.0;
  var gradIx = 0.0;
  var gradIy = 0.0;

  let base = cellOf(pi.px, pi.py);
  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      let h = cellHash(base.x + ox, base.y + oy);
      var j = atomicLoad(&cellHead[h]);
      loop {
        if (j == EMPTY) { break; }
        if (j != i) {
          let pj = particles[j];
          let dx = pj.px - pi.px;
          let dy = pj.py - pi.py;
          let d2 = dx * dx + dy * dy;
          if (d2 < params.h2) {
            rho += poly6(d2);
            let r = sqrt(d2);
            let c = spikyGradCoeff(r);
            // Match CPU: gx = -c * dx (c = spikyGradCoeff ≤ 0)
            let gx = -c * dx;
            let gy = -c * dy;
            gradIx += gx;
            gradIy += gy;
            sumGradSq += gx * gx + gy * gy;
          }
        }
        j = cellNext[j];
      }
    }
  }

  sumGradSq += gradIx * gradIx + gradIy * gradIy;
  let C = rho / rho0 - 1.0;
  lambdaBuf[i] = -C / (sumGradSq / (rho0 * rho0) + params.eps);
}

// ── Pass: computeDelta — Σ (λi+λj+s_corr) ∇W / ρ0 ───────────────────────────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn computeDelta(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  let pi = particles[i];
  if (pi.invMass == 0.0) {
    deltaX[i] = 0.0;
    deltaY[i] = 0.0;
    return;
  }

  let li = lambdaBuf[i];
  let rho0 = params.restDensity;
  var dpx = 0.0;
  var dpy = 0.0;

  let base = cellOf(pi.px, pi.py);
  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      let h = cellHash(base.x + ox, base.y + oy);
      var j = atomicLoad(&cellHead[h]);
      loop {
        if (j == EMPTY) { break; }
        if (j != i) {
          let pj = particles[j];
          let dx = pj.px - pi.px;
          let dy = pj.py - pi.py;
          let d2 = dx * dx + dy * dy;
          if (d2 < params.h2) {
            let r = sqrt(d2);
            let c = spikyGradCoeff(r);

            var scorr = 0.0;
            if (params.kCorr > 0.0 && params.wDq > 0.0) {
              let ratio = poly6(d2) / params.wDq;
              scorr = -params.kCorr * pow(ratio, params.nCorr);
            }

            // CPU: s = ((li + lambda[j] + scorr) * -c) / rho0
            let s = ((li + lambdaBuf[j] + scorr) * (-c)) / rho0;
            dpx += s * dx;
            dpy += s * dy;
          }
        }
        j = cellNext[j];
      }
    }
  }

  deltaX[i] = dpx;
  deltaY[i] = dpy;
}

// ── Pass: applyDelta — predicted += Δp ──────────────────────────────────────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn applyDelta(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  var p = particles[i];
  if (p.invMass == 0.0) { return; }
  p.px += deltaX[i];
  p.py += deltaY[i];
  particles[i] = p;
}

fn collideBoundsImpl(i: u32, kill: bool) {
  var p = particles[i];
  if (p.invMass == 0.0) { return; }

  let x0 = params.bounds.x;
  let y0 = params.bounds.y;
  let x1 = params.bounds.z;
  let y1 = params.bounds.w;

  if (p.px < x0) {
    p.px = x0;
    if (kill && p.vx < 0.0) { p.vx = 0.0; }
  } else if (p.px > x1) {
    p.px = x1;
    if (kill && p.vx > 0.0) { p.vx = 0.0; }
  }
  if (p.py < y0) {
    p.py = y0;
    if (kill && p.vy < 0.0) { p.vy = 0.0; }
  } else if (p.py > y1) {
    p.py = y1;
    if (kill && p.vy > 0.0) { p.vy = 0.0; }
  }
  particles[i] = p;
}

// ── Pass: collideBoundsSoft — clamp predicted pos each constraint iter ──────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn collideBoundsSoft(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }
  collideBoundsImpl(i, false);
}

// ── Pass: collideBoundsKill — final walls, restitution ≈ 0 (kill normal v) ─
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn collideBoundsKill(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }
  collideBoundsImpl(i, true);
}

// ── Pass: updateVel — v = (p − x)/dt (positions stay predicted until commit)
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn updateVel(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  var p = particles[i];
  if (p.invMass == 0.0) {
    p.vx = 0.0;
    p.vy = 0.0;
    particles[i] = p;
    return;
  }
  let invDt = 1.0 / params.dt;
  p.vx = (p.px - p.x) * invDt;
  p.vy = (p.py - p.y) * invDt;
  particles[i] = p;
}

// ── Pass: viscosityAccum — XSPH relative-velocity weighted by poly6 → delta ─
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn viscosityAccum(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  let pi = particles[i];
  if (pi.invMass == 0.0) {
    deltaX[i] = 0.0;
    deltaY[i] = 0.0;
    return;
  }

  var ax = 0.0;
  var ay = 0.0;
  let base = cellOf(pi.px, pi.py);

  for (var oy = -1; oy <= 1; oy++) {
    for (var ox = -1; ox <= 1; ox++) {
      let h = cellHash(base.x + ox, base.y + oy);
      var j = atomicLoad(&cellHead[h]);
      loop {
        if (j == EMPTY) { break; }
        if (j != i) {
          let pj = particles[j];
          let dx = pj.px - pi.px;
          let dy = pj.py - pi.py;
          let d2 = dx * dx + dy * dy;
          if (d2 < params.h2) {
            let w = poly6(d2);
            ax += (pj.vx - pi.vx) * w;
            ay += (pj.vy - pi.vy) * w;
          }
        }
        j = cellNext[j];
      }
    }
  }

  deltaX[i] = ax;
  deltaY[i] = ay;
}

// ── Pass: viscosityApply — v += (visc/ρ0) * accum ───────────────────────────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn viscosityApply(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }

  var p = particles[i];
  if (p.invMass == 0.0) { return; }
  let scale = params.viscosity / params.restDensity;
  p.vx += deltaX[i] * scale;
  p.vy += deltaY[i] * scale;
  particles[i] = p;
}

// ── Pass: commitPositions — x = px, y = py ──────────────────────────────────
@compute @workgroup_size(${PBF_WORKGROUP_SIZE})
fn commitPositions(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.particleCount) { return; }
  var p = particles[i];
  p.x = p.px;
  p.y = p.py;
  particles[i] = p;
}
`;
