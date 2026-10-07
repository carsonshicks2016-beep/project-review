/** SoA water buffer alloc / resize / fill helpers. */

import type { WaterState } from '../types';

/** Rest density for unit-mass particles on a square lattice (SPH poly6, h = 2·spacing). */
export function restDensityFromSpacing(spacing: number): number {
  const h = spacing * 2;
  const h2 = h * h;
  const coeff = 4 / (Math.PI * h ** 8);
  let rho = 0;
  const extent = Math.ceil(h / spacing);
  for (let iy = -extent; iy <= extent; iy++) {
    for (let ix = -extent; ix <= extent; ix++) {
      const r2 = (ix * spacing) ** 2 + (iy * spacing) ** 2;
      if (r2 >= h2) continue;
      const d = h2 - r2;
      rho += coeff * d * d * d;
    }
  }
  return rho > 0 ? rho : 1 / (spacing * spacing);
}

function allocBuffers(capacity: number) {
  return {
    x: new Float32Array(capacity),
    y: new Float32Array(capacity),
    px: new Float32Array(capacity),
    py: new Float32Array(capacity),
    vx: new Float32Array(capacity),
    vy: new Float32Array(capacity),
    invMass: new Float32Array(capacity),
  };
}

export function createWaterState(capacity: number, spacing: number): WaterState {
  const cap = Math.max(0, capacity | 0);
  const buf = allocBuffers(cap);
  // Default invMass = 1 for unused slots; live particles set on add
  buf.invMass.fill(1);
  return {
    count: 0,
    capacity: cap,
    ...buf,
    spacing,
    restDensity: restDensityFromSpacing(spacing),
  };
}

/** Grow (or shrink) capacity; preserves particles up to min(oldCount, newCapacity). */
export function resizeWaterCapacity(state: WaterState, capacity: number): WaterState {
  const cap = Math.max(0, capacity | 0);
  if (cap === state.capacity) return state;

  const keep = Math.min(state.count, cap);
  const buf = allocBuffers(cap);
  buf.invMass.fill(1);

  if (keep > 0) {
    buf.x.set(state.x.subarray(0, keep));
    buf.y.set(state.y.subarray(0, keep));
    buf.px.set(state.px.subarray(0, keep));
    buf.py.set(state.py.subarray(0, keep));
    buf.vx.set(state.vx.subarray(0, keep));
    buf.vy.set(state.vy.subarray(0, keep));
    buf.invMass.set(state.invMass.subarray(0, keep));
  }

  state.count = keep;
  state.capacity = cap;
  state.x = buf.x;
  state.y = buf.y;
  state.px = buf.px;
  state.py = buf.py;
  state.vx = buf.vx;
  state.vy = buf.vy;
  state.invMass = buf.invMass;
  return state;
}

/** Append a particle. Returns index, or -1 if full. */
export function addParticle(
  state: WaterState,
  x: number,
  y: number,
  vx = 0,
  vy = 0,
): number {
  if (state.count >= state.capacity) return -1;
  const i = state.count++;
  state.x[i] = x;
  state.y[i] = y;
  state.px[i] = x;
  state.py[i] = y;
  state.vx[i] = vx;
  state.vy[i] = vy;
  state.invMass[i] = 1;
  return i;
}

/** Swap-remove particle i with the last live particle. */
export function removeParticle(state: WaterState, i: number): void {
  const n = state.count;
  if (i < 0 || i >= n) return;
  const last = n - 1;
  if (i !== last) {
    state.x[i] = state.x[last];
    state.y[i] = state.y[last];
    state.px[i] = state.px[last];
    state.py[i] = state.py[last];
    state.vx[i] = state.vx[last];
    state.vy[i] = state.vy[last];
    state.invMass[i] = state.invMass[last];
  }
  state.count = last;
}

/**
 * Fill axis-aligned rectangle [x0,x1]×[y0,y1] with a square grid.
 * Stops early if capacity is exhausted. Writes SoA buffers directly (no per-cell calls).
 */
export function fillRect(
  state: WaterState,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  spacing?: number,
): void {
  const s = spacing ?? state.spacing;
  if (s <= 0) return;

  const minX = Math.min(x0, x1);
  const maxX = Math.max(x0, x1);
  const minY = Math.min(y0, y1);
  const maxY = Math.max(y0, y1);

  // Inset by half-spacing so particles sit inside the region
  const startX = minX + s * 0.5;
  const startY = minY + s * 0.5;
  const endX = maxX - s * 0.5 + 1e-9;
  const endY = maxY - s * 0.5 + 1e-9;
  if (startX > endX || startY > endY) return;

  const { x, y, px, py, vx, vy, invMass } = state;
  const cap = state.capacity;
  let i = state.count;

  for (let yy = startY; yy <= endY; yy += s) {
    for (let xx = startX; xx <= endX; xx += s) {
      if (i >= cap) {
        state.count = i;
        return;
      }
      x[i] = xx;
      y[i] = yy;
      px[i] = xx;
      py[i] = yy;
      vx[i] = 0;
      vy[i] = 0;
      invMass[i] = 1;
      i++;
    }
  }
  state.count = i;
}

export function clearWater(state: WaterState): void {
  state.count = 0;
}
