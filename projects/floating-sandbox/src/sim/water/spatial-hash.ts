/**
 * Uniform-grid spatial hash for 2D PBF neighbour queries.
 * Linked-list cells — no per-frame heap allocations in the hot path.
 */

export type NeighborCallback = (
  j: number,
  dx: number,
  dy: number,
  distSq: number,
) => void;

const DEFAULT_CELLS = 4096; // power of two for fast mask

export class SpatialHash {
  private head: Int32Array;
  private next: Int32Array;
  private readonly nCells: number;
  private readonly cellMask: number;
  private cellSize = 1;
  private invCell = 1;

  constructor(maxParticles = 16384, nCells = DEFAULT_CELLS) {
    // Round cell count up to power of two
    let cells = 1;
    while (cells < nCells) cells <<= 1;
    this.nCells = cells;
    this.cellMask = cells - 1;
    this.head = new Int32Array(cells);
    this.head.fill(-1);
    this.next = new Int32Array(Math.max(1, maxParticles));
  }

  /** Ensure next[] can hold `count` particle links. */
  ensureCapacity(count: number): void {
    if (count <= this.next.length) return;
    let n = this.next.length;
    while (n < count) n <<= 1;
    this.next = new Int32Array(n);
  }

  private hash(cx: number, cy: number): number {
    // 32-bit spatial hash → cell index
    return ((cx * 73856093) ^ (cy * 19349663)) & this.cellMask;
  }

  /**
   * Rebuild the grid from particle positions.
   * `cellSize` should be ≥ neighbour query radius (typically the SPH support h).
   */
  rebuild(
    x: Float32Array,
    y: Float32Array,
    count: number,
    cellSize: number,
  ): void {
    this.ensureCapacity(count);
    this.cellSize = cellSize > 0 ? cellSize : 1;
    this.invCell = 1 / this.cellSize;

    const head = this.head;
    head.fill(-1);
    const next = this.next;
    const inv = this.invCell;

    for (let i = 0; i < count; i++) {
      const cx = Math.floor(x[i] * inv);
      const cy = Math.floor(y[i] * inv);
      const h = this.hash(cx, cy);
      next[i] = head[h];
      head[h] = i;
    }
  }

  /**
   * Visit all particles within `radius` of particle i (excluding i itself).
   * Requires cellSize ≥ radius so a 3×3 neighbourhood is sufficient.
   * Optional `maxNeighbors` early-outs once enough neighbours are found.
   */
  forNeighbors(
    i: number,
    x: Float32Array,
    y: Float32Array,
    _count: number,
    radius: number,
    callback: NeighborCallback,
    maxNeighbors = 0,
  ): void {
    const xi = x[i];
    const yi = y[i];
    const r2 = radius * radius;
    const inv = this.invCell;
    const cx = Math.floor(xi * inv);
    const cy = Math.floor(yi * inv);
    const head = this.head;
    const next = this.next;
    const limit = maxNeighbors > 0 ? maxNeighbors : 0x7fffffff;
    let found = 0;

    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        let j = head[this.hash(cx + ox, cy + oy)];
        while (j !== -1) {
          if (j !== i) {
            const dx = x[j] - xi;
            const dy = y[j] - yi;
            const d2 = dx * dx + dy * dy;
            if (d2 < r2) {
              callback(j, dx, dy, d2);
              if (++found >= limit) return;
            }
          }
          j = next[j];
        }
      }
    }
  }

  /**
   * Gather neighbours of i into SoA scratch (no callback).
   * Returns how many were written (≤ maxOut). Requires cellSize ≥ radius.
   */
  gatherNeighbors(
    i: number,
    x: Float32Array,
    y: Float32Array,
    radius: number,
    outJ: Int32Array,
    outDx: Float32Array,
    outDy: Float32Array,
    outD2: Float32Array,
    outBase: number,
    maxOut: number,
  ): number {
    if (maxOut <= 0) return 0;

    const xi = x[i];
    const yi = y[i];
    const r2 = radius * radius;
    const inv = this.invCell;
    const cx = Math.floor(xi * inv);
    const cy = Math.floor(yi * inv);
    const head = this.head;
    const next = this.next;
    let found = 0;

    for (let oy = -1; oy <= 1; oy++) {
      for (let ox = -1; ox <= 1; ox++) {
        let j = head[this.hash(cx + ox, cy + oy)];
        while (j !== -1) {
          if (j !== i) {
            const dx = x[j] - xi;
            const dy = y[j] - yi;
            const d2 = dx * dx + dy * dy;
            if (d2 < r2) {
              const slot = outBase + found;
              outJ[slot] = j;
              outDx[slot] = dx;
              outDy[slot] = dy;
              outD2[slot] = d2;
              if (++found >= maxOut) return found;
            }
          }
          j = next[j];
        }
      }
    }
    return found;
  }
}
