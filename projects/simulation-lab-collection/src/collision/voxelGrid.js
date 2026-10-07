// Sparse occupancy grid built from the LiDAR scan, so the house becomes solid geometry the
// drone can hit and sense rather than a backdrop it flies through.
//
// Stored as a bitset: at 1.5m cells the house is roughly 342 x 112 x 449 = 17M cells,
// which is 2.1MB packed one bit per cell. Small enough to hand a copy to every rollout
// worker, and a lookup is a shift and a mask.

export const DEFAULT_CELL_SIZE = 1.5;

export class VoxelGrid {
  constructor({ origin, dims, cellSize, bits }) {
    this.origin = origin;   // world position of cell (0,0,0)
    this.dims = dims;       // [nx, ny, nz]
    this.cellSize = cellSize;
    this.bits = bits;       // Uint8Array, one bit per cell
    this.inv = 1 / cellSize;
    this.strideY = dims[0];
    this.strideZ = dims[0] * dims[1];
  }

  // `positions` is the raw geometry attribute in local space. The scan is rendered with a
  // uniform scale and a y offset that sits it on the ground; the same transform has to be
  // applied here or collision would not line up with what you can see.
  static fromPoints(positions, { scale = 1, offsetY = 0, cellSize = DEFAULT_CELL_SIZE, minPoints = 2 } = {}) {
    const n = (positions.length / 3) | 0;

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = positions[i * 3] * scale;
      const y = positions[i * 3 + 1] * scale + offsetY;
      const z = positions[i * 3 + 2] * scale;
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    if (!Number.isFinite(minX)) throw new Error('VoxelGrid.fromPoints: no finite points');

    // A cell of padding so boundary points cannot fall outside the grid.
    const origin = [minX - cellSize, minY - cellSize, minZ - cellSize];
    const dims = [
      Math.ceil((maxX - minX) / cellSize) + 3,
      Math.ceil((maxY - minY) / cellSize) + 3,
      Math.ceil((maxZ - minZ) / cellSize) + 3
    ];
    const total = dims[0] * dims[1] * dims[2];

    // Count first, then threshold: one stray point should not become a wall. The scan's
    // own denoiser works in its local units, so this is a second, world-space pass.
    const counts = new Uint8Array(total);
    const inv = 1 / cellSize;
    const strideY = dims[0];
    const strideZ = dims[0] * dims[1];

    for (let i = 0; i < n; i++) {
      const cx = ((positions[i * 3] * scale - origin[0]) * inv) | 0;
      const cy = ((positions[i * 3 + 1] * scale + offsetY - origin[1]) * inv) | 0;
      const cz = ((positions[i * 3 + 2] * scale - origin[2]) * inv) | 0;
      if (cx < 0 || cy < 0 || cz < 0 || cx >= dims[0] || cy >= dims[1] || cz >= dims[2]) continue;
      const idx = cx + cy * strideY + cz * strideZ;
      if (counts[idx] < 255) counts[idx]++;
    }

    const bits = new Uint8Array((total + 7) >> 3);
    let solid = 0;
    for (let i = 0; i < total; i++) {
      if (counts[i] >= minPoints) {
        bits[i >> 3] |= 1 << (i & 7);
        solid++;
      }
    }

    const grid = new VoxelGrid({ origin, dims, cellSize, bits });
    grid.solidCount = solid;
    grid.totalCells = total;
    return grid;
  }

  cellIndex(cx, cy, cz) {
    return cx + cy * this.strideY + cz * this.strideZ;
  }

  isSolidCell(cx, cy, cz) {
    const d = this.dims;
    if (cx < 0 || cy < 0 || cz < 0 || cx >= d[0] || cy >= d[1] || cz >= d[2]) return false;
    const i = cx + cy * this.strideY + cz * this.strideZ;
    return (this.bits[i >> 3] & (1 << (i & 7))) !== 0;
  }

  isSolid(x, y, z) {
    return this.isSolidCell(
      ((x - this.origin[0]) * this.inv) | 0,
      ((y - this.origin[1]) * this.inv) | 0,
      ((z - this.origin[2]) * this.inv) | 0
    );
  }

  // Amanatides & Woo voxel traversal. Returns the distance to the first solid cell, or
  // maxDist if the ray leaves the grid or reaches the limit without hitting anything.
  raycast(ox, oy, oz, dx, dy, dz, maxDist) {
    const d = this.dims;

    // The grid only spans the scanned geometry, so a ray can easily start outside it.
    // Clip to the grid AABB first (slab method) and start traversal at the entry point;
    // without this any ray originating outside immediately fails the bounds test.
    const cs0 = this.cellSize;
    let t0 = 0, t1 = maxDist;
    const o = [ox, oy, oz], dir = [dx, dy, dz];
    for (let a = 0; a < 3; a++) {
      const lo = this.origin[a];
      const hi = this.origin[a] + d[a] * cs0;
      if (Math.abs(dir[a]) < 1e-12) {
        if (o[a] < lo || o[a] > hi) return maxDist;
      } else {
        let ta = (lo - o[a]) / dir[a];
        let tb = (hi - o[a]) / dir[a];
        if (ta > tb) { const tmp = ta; ta = tb; tb = tmp; }
        if (ta > t0) t0 = ta;
        if (tb < t1) t1 = tb;
        if (t0 > t1) return maxDist;
      }
    }
    // Nudge just inside so the entry cell is computed on the correct side of the boundary.
    const tStart = t0 > 0 ? t0 + 1e-4 : 0;
    if (tStart >= maxDist) return maxDist;
    ox += dx * tStart; oy += dy * tStart; oz += dz * tStart;

    let cx = ((ox - this.origin[0]) * this.inv) | 0;
    let cy = ((oy - this.origin[1]) * this.inv) | 0;
    let cz = ((oz - this.origin[2]) * this.inv) | 0;

    // Starting inside geometry: report zero rather than tunnelling out of it.
    if (this.isSolidCell(cx, cy, cz)) return tStart;

    const stepX = dx > 0 ? 1 : dx < 0 ? -1 : 0;
    const stepY = dy > 0 ? 1 : dy < 0 ? -1 : 0;
    const stepZ = dz > 0 ? 1 : dz < 0 ? -1 : 0;

    const cs = this.cellSize;
    // Distance along the ray to the next cell boundary on each axis.
    const nextBoundary = (c, o, origin, step) =>
      origin + (c + (step > 0 ? 1 : 0)) * cs - o;

    let tMaxX = stepX === 0 ? Infinity : nextBoundary(cx, ox, this.origin[0], stepX) / dx;
    let tMaxY = stepY === 0 ? Infinity : nextBoundary(cy, oy, this.origin[1], stepY) / dy;
    let tMaxZ = stepZ === 0 ? Infinity : nextBoundary(cz, oz, this.origin[2], stepZ) / dz;

    const tDeltaX = stepX === 0 ? Infinity : Math.abs(cs / dx);
    const tDeltaY = stepY === 0 ? Infinity : Math.abs(cs / dy);
    const tDeltaZ = stepZ === 0 ? Infinity : Math.abs(cs / dz);

    let t = 0;
    // Bounded so a ray parallel to a wall inside a large empty grid still terminates.
    const maxSteps = Math.ceil(maxDist / cs) * 3 + 3;
    for (let s = 0; s < maxSteps; s++) {
      if (tMaxX < tMaxY) {
        if (tMaxX < tMaxZ) { cx += stepX; t = tMaxX; tMaxX += tDeltaX; }
        else { cz += stepZ; t = tMaxZ; tMaxZ += tDeltaZ; }
      } else {
        if (tMaxY < tMaxZ) { cy += stepY; t = tMaxY; tMaxY += tDeltaY; }
        else { cz += stepZ; t = tMaxZ; tMaxZ += tDeltaZ; }
      }
      if (tStart + t > maxDist) return maxDist;
      if (cx < 0 || cy < 0 || cz < 0 || cx >= d[0] || cy >= d[1] || cz >= d[2]) return maxDist;
      if (this.isSolidCell(cx, cy, cz)) return tStart + t;
    }
    return maxDist;
  }

  // Transferable form for handing the grid to workers.
  serialize() {
    return { origin: this.origin, dims: this.dims, cellSize: this.cellSize, bits: this.bits.buffer };
  }

  static deserialize(o) {
    return new VoxelGrid({
      origin: o.origin, dims: o.dims, cellSize: o.cellSize, bits: new Uint8Array(o.bits)
    });
  }
}
