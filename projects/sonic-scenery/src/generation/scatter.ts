/**
 * Deterministic Poisson-disk sampling (Bridson's algorithm) over a square area
 * centered on the origin. Driven entirely by an injected PRNG so the same seed
 * yields the same point set. Pure — no randomness, no I/O.
 */

export interface PoissonOptions {
  /** Half-extent of the square area; points fall in [-half, half] on x and z. */
  half: number;
  /** Minimum distance between any two points. */
  radius: number;
  /** Hard cap on emitted points (also bounds the loop). */
  maxPoints: number;
  /** Candidate attempts per active sample before it is retired. */
  k?: number;
}

export interface Point2D {
  x: number;
  z: number;
}

/**
 * Bridson Poisson-disk sampling. `rng()` must return a deterministic value in
 * [0, 1). Returns points in deterministic order for a given rng sequence.
 */
export function poissonDisk(rng: () => number, opts: PoissonOptions): Point2D[] {
  const { half, radius, maxPoints } = opts;
  const k = opts.k ?? 30;
  const size = half * 2;
  const cellSize = radius / Math.SQRT2;
  const gridW = Math.max(1, Math.ceil(size / cellSize));
  const gridH = gridW;
  const grid: Int32Array = new Int32Array(gridW * gridH).fill(-1);

  const samples: Point2D[] = [];
  const active: number[] = [];

  const gridIndex = (x: number, z: number): number => {
    const gx = Math.min(gridW - 1, Math.max(0, Math.floor((x + half) / cellSize)));
    const gz = Math.min(gridH - 1, Math.max(0, Math.floor((z + half) / cellSize)));
    return gz * gridW + gx;
  };

  const addSample = (p: Point2D): void => {
    const idx = samples.length;
    samples.push(p);
    grid[gridIndex(p.x, p.z)] = idx;
    active.push(idx);
  };

  // Seed the first sample deterministically from the rng.
  addSample({ x: (rng() - 0.5) * size, z: (rng() - 0.5) * size });

  while (active.length > 0 && samples.length < maxPoints) {
    const activePick = Math.floor(rng() * active.length);
    const parentIdx = active[activePick]!;
    const parent = samples[parentIdx]!;
    let placed = false;

    for (let attempt = 0; attempt < k; attempt++) {
      const angle = rng() * Math.PI * 2;
      const dist = radius * (1 + rng()); // [radius, 2*radius)
      const nx = parent.x + Math.cos(angle) * dist;
      const nz = parent.z + Math.sin(angle) * dist;

      if (nx < -half || nx > half || nz < -half || nz > half) continue;

      // Check neighbouring grid cells for a too-close existing sample.
      const gx = Math.min(gridW - 1, Math.max(0, Math.floor((nx + half) / cellSize)));
      const gz = Math.min(gridH - 1, Math.max(0, Math.floor((nz + half) / cellSize)));
      let ok = true;
      for (let dz = -2; dz <= 2 && ok; dz++) {
        for (let dx = -2; dx <= 2 && ok; dx++) {
          const cx = gx + dx;
          const cz = gz + dz;
          if (cx < 0 || cx >= gridW || cz < 0 || cz >= gridH) continue;
          const other = grid[cz * gridW + cx]!;
          if (other === -1) continue;
          const o = samples[other]!;
          const ddx = o.x - nx;
          const ddz = o.z - nz;
          if (ddx * ddx + ddz * ddz < radius * radius) ok = false;
        }
      }

      if (ok) {
        addSample({ x: nx, z: nz });
        placed = true;
        if (samples.length >= maxPoints) break;
        break;
      }
    }

    if (!placed) {
      // Retire this active sample (swap-remove keeps it deterministic).
      active[activePick] = active[active.length - 1]!;
      active.pop();
    }
  }

  return samples;
}
