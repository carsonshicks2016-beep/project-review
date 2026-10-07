/**
 * ApexFlock - 3D Spatial Partitioning Grid
 * Enables O(1) average-time neighbor lookups and raycast queries
 * for hundreds of agents at 60 FPS.
 */

export class SpatialGrid {
  constructor(bounds = { minX: -250, maxX: 250, minY: -100, maxY: 150, minZ: -250, maxZ: 250 }, cellSize = 35) {
    this.bounds = bounds;
    this.cellSize = cellSize;
    this.invCellSize = 1.0 / cellSize;

    this.cols = Math.ceil((bounds.maxX - bounds.minX) / cellSize);
    this.rows = Math.ceil((bounds.maxY - bounds.minY) / cellSize);
    this.layers = Math.ceil((bounds.maxZ - bounds.minZ) / cellSize);
    this.numCells = this.cols * this.rows * this.layers;

    // Array of arrays for cell occupants
    this.cells = new Array(this.numCells);
    for (let i = 0; i < this.numCells; i++) {
      this.cells[i] = [];
    }

    // Fast reusable buffer for query results
    this._queryBuffer = [];
  }

  clear() {
    for (let i = 0; i < this.numCells; i++) {
      this.cells[i].length = 0;
    }
  }

  _hash(cx, cy, cz) {
    if (cx < 0 || cx >= this.cols || cy < 0 || cy >= this.rows || cz < 0 || cz >= this.layers) {
      return -1;
    }
    return cx + cy * this.cols + cz * this.cols * this.rows;
  }

  insert(agent, x, y, z) {
    const cx = Math.floor((x - this.bounds.minX) * this.invCellSize);
    const cy = Math.floor((y - this.bounds.minY) * this.invCellSize);
    const cz = Math.floor((z - this.bounds.minZ) * this.invCellSize);
    const idx = this._hash(cx, cy, cz);
    if (idx >= 0 && idx < this.numCells) {
      this.cells[idx].push(agent);
    }
  }

  /**
   * Queries all agents within a given spherical radius of (x, y, z).
   * @param {number} x 
   * @param {number} y 
   * @param {number} z 
   * @param {number} radius 
   * @param {Array} outResults - Optional destination array to avoid allocation
   * @returns {Array} agents within radius
   */
  queryRadius(x, y, z, radius, outResults = null) {
    const results = outResults || this._queryBuffer;
    results.length = 0;

    const r2 = radius * radius;
    const minCx = Math.floor((x - radius - this.bounds.minX) * this.invCellSize);
    const maxCx = Math.floor((x + radius - this.bounds.minX) * this.invCellSize);
    const minCy = Math.floor((y - radius - this.bounds.minY) * this.invCellSize);
    const maxCy = Math.floor((y + radius - this.bounds.minY) * this.invCellSize);
    const minCz = Math.floor((z - radius - this.bounds.minZ) * this.invCellSize);
    const maxCz = Math.floor((z + radius - this.bounds.minZ) * this.invCellSize);

    const c0 = Math.max(0, minCx);
    const c1 = Math.min(this.cols - 1, maxCx);
    const r0 = Math.max(0, minCy);
    const r1 = Math.min(this.rows - 1, maxCy);
    const l0 = Math.max(0, minCz);
    const l1 = Math.min(this.layers - 1, maxCz);

    for (let cz = l0; cz <= l1; cz++) {
      const zOffset = cz * this.cols * this.rows;
      for (let cy = r0; cy <= r1; cy++) {
        const yzOffset = cy * this.cols + zOffset;
        for (let cx = c0; cx <= c1; cx++) {
          const cell = this.cells[cx + yzOffset];
          for (let i = 0; i < cell.length; i++) {
            const a = cell[i];
            const dx = a.position.x - x;
            const dy = a.position.y - y;
            const dz = a.position.z - z;
            const d2 = dx * dx + dy * dy + dz * dz;
            if (d2 <= r2) {
              results.push(a);
            }
          }
        }
      }
    }

    return results;
  }
}
