// Tiny deterministic noise/math kit shared by the world builders.
// Everything is seeded — the same track must always dress the same way.

export function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

export function smoothstep(edge0, edge1, x) {
  const t = clamp01((x - edge0) / Math.max(1e-6, edge1 - edge0));
  return t * t * (3 - 2 * t);
}

export function lerp(a, b, t) {
  return a + (b - a) * t;
}

export function hash01(n) {
  const x = Math.sin(n * 127.1 + 311.7) * 43758.5453123;
  return x - Math.floor(x);
}

export function hash2(x, z) {
  const s = Math.sin(x * 127.1 + z * 311.7) * 43758.5453123;
  return s - Math.floor(s);
}

export function valueNoise2(x, z) {
  const xi = Math.floor(x);
  const zi = Math.floor(z);
  const xf = x - xi;
  const zf = z - zi;
  const u = xf * xf * (3 - 2 * xf);
  const v = zf * zf * (3 - 2 * zf);
  const a = hash2(xi, zi);
  const b = hash2(xi + 1, zi);
  const c = hash2(xi, zi + 1);
  const d = hash2(xi + 1, zi + 1);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}

// Fractal noise in [-1, 1].
export function fbm2(x, z, octaves = 4) {
  let sum = 0;
  let amp = 0.5;
  let max = 0;
  let fx = x;
  let fz = z;
  for (let o = 0; o < octaves; o++) {
    sum += valueNoise2(fx, fz) * amp;
    max += amp;
    amp *= 0.5;
    fx = fx * 2.07 + 13.7;
    fz = fz * 2.07 - 7.3;
  }
  return (sum / max) * 2 - 1;
}
