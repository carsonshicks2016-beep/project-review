// Soft-body node/beam solver, BeamNG-style.
//
// Every object is a cloud of point masses ("nodes") wired together by
// spring-dampers ("beams"). Beams yield plastically past a strain threshold and
// snap past a higher one, so bodies crumple and tear instead of bouncing like
// rigid boxes. Runs at a high fixed substep rate; the renderer samples whatever
// state it finds.

export const GRAVITY = -9.81;
export const SUBSTEP_HZ = 1500;
const SUB_DT = 1 / SUBSTEP_HZ;
const MAX_SUBSTEPS = 40;

let nextGroupId = 1;

export function makeNode(x, y, z, mass, opts = {}) {
  return {
    x, y, z,
    vx: 0, vy: 0, vz: 0,
    fx: 0, fy: 0, fz: 0,
    m: mass,
    invM: mass > 0 ? 1 / mass : 0,
    r: opts.r ?? 0.16,
    mu: opts.mu ?? 0.8,
    restitution: opts.restitution ?? 0.08,
    drag: opts.drag ?? 0.04,
    isWheel: opts.isWheel ?? false,
    skipGround: opts.skipGround ?? false,
    group: null,
  };
}

// ratio params are fractions of critical damping, so stiffness and damping stay
// coupled when either is tuned.
export function makeBeam(a, b, opts = {}) {
  const rest = dist(a, b);
  const k = opts.k ?? 1.5e6;
  const mr = (a.m * b.m) / (a.m + b.m || 1);
  return {
    a, b,
    rest,
    restOrig: rest,
    k,
    d: opts.d ?? (opts.dampRatio ?? 0.1) * 2 * Math.sqrt(k * mr),
    deform: opts.deform ?? 0.03,  // strain where plastic yielding starts
    // Per-substep rate at which rest length chases the deformed length. This
    // is what separates a crumple zone from a spring: yield fast enough and the
    // beam force plateaus, so impact energy is absorbed instead of returned.
    plastic: opts.plastic ?? 0.1,
    brk: opts.brk ?? 0.55,        // strain where the beam snaps
    minRest: rest * 0.25,
    broken: false,
    damage: 0,
  };
}

function dist(a, b) {
  const dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

export class World {
  constructor() {
    this.groups = [];   // { id, nodes, beams, bounds, active }
    this.boxes = [];    // static AABBs: { min:{x,y,z}, max:{x,y,z}, mu }
    this.preSolve = []; // fn(dt) — tire forces, engine, etc.
    this.onBeamBreak = null;
    this.onImpact = null; // (node, speed, kind)
    this._boxCandidates = new Map(); // groupId -> box[]
    this._groupPairs = [];
    this._acc = 0;
  }

  addGroup(nodes, beams) {
    const g = { id: nextGroupId++, nodes, beams, bounds: newBounds(), active: true };
    for (const n of nodes) n.group = g;
    this.groups.push(g);
    return g;
  }

  addBox(min, max, mu = 0.9) {
    const box = { min, max, mu };
    this.boxes.push(box);
    return box;
  }

  // Called once per rendered frame, not per substep — bodies cannot cross a
  // building in 1/60s, so the candidate sets stay valid for the whole frame.
  refreshBroadphase() {
    for (const g of this.groups) {
      if (!g.active) continue;
      updateBounds(g);
      const pad = 1.5;
      const cands = [];
      for (const box of this.boxes) {
        if (g.bounds.maxX + pad < box.min.x || g.bounds.minX - pad > box.max.x) continue;
        if (g.bounds.maxY + pad < box.min.y || g.bounds.minY - pad > box.max.y) continue;
        if (g.bounds.maxZ + pad < box.min.z || g.bounds.minZ - pad > box.max.z) continue;
        cands.push(box);
      }
      this._boxCandidates.set(g.id, cands);
    }

    this._groupPairs.length = 0;
    for (let i = 0; i < this.groups.length; i++) {
      if (!this.groups[i].active) continue;
      for (let j = i + 1; j < this.groups.length; j++) {
        if (!this.groups[j].active) continue;
        const a = this.groups[i].bounds, b = this.groups[j].bounds;
        const pad = 1.0;
        if (a.maxX + pad < b.minX || a.minX - pad > b.maxX) continue;
        if (a.maxY + pad < b.minY || a.minY - pad > b.maxY) continue;
        if (a.maxZ + pad < b.minZ || a.minZ - pad > b.maxZ) continue;
        this._groupPairs.push([this.groups[i], this.groups[j]]);
      }
    }
  }

  // Accumulator keeps physics deterministic regardless of framerate; the
  // MAX_SUBSTEPS cap means a stalled tab drops time instead of spiralling.
  step(frameDt) {
    this._acc += Math.min(frameDt, 0.1);
    let n = 0;
    while (this._acc >= SUB_DT && n < MAX_SUBSTEPS) {
      this.substep(SUB_DT);
      this._acc -= SUB_DT;
      n++;
    }
    if (n === MAX_SUBSTEPS) this._acc = 0;
  }

  substep(dt) {
    for (const g of this.groups) {
      if (!g.active) continue;
      for (const nd of g.nodes) {
        nd.fx = 0;
        nd.fy = GRAVITY * nd.m;
        nd.fz = 0;
        const sp = Math.sqrt(nd.vx * nd.vx + nd.vy * nd.vy + nd.vz * nd.vz);
        if (sp > 0.01) {
          const q = nd.drag * sp;
          nd.fx -= nd.vx * q;
          nd.fy -= nd.vy * q;
          nd.fz -= nd.vz * q;
        }
      }
    }

    for (const fn of this.preSolve) fn(dt);

    for (const g of this.groups) {
      if (!g.active) continue;
      this.solveBeams(g, dt);
    }

    for (const g of this.groups) {
      if (!g.active) continue;
      for (const nd of g.nodes) {
        nd.vx += nd.fx * nd.invM * dt;
        nd.vy += nd.fy * nd.invM * dt;
        nd.vz += nd.fz * nd.invM * dt;
        nd.x += nd.vx * dt;
        nd.y += nd.vy * dt;
        nd.z += nd.vz * dt;
      }
      this.collideStatic(g);
    }

    for (const [ga, gb] of this._groupPairs) this.collideGroups(ga, gb);
  }

  solveBeams(g, dt) {
    const beams = g.beams;
    for (let i = 0; i < beams.length; i++) {
      const bm = beams[i];
      if (bm.broken) continue;
      const a = bm.a, b = bm.b;
      let dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
      const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (len < 1e-6) continue;
      const inv = 1 / len;
      dx *= inv; dy *= inv; dz *= inv;

      const ext = len - bm.rest;
      const rv = (b.vx - a.vx) * dx + (b.vy - a.vy) * dy + (b.vz - a.vz) * dz;
      const f = bm.k * ext + bm.d * rv;

      const fx = f * dx, fy = f * dy, fz = f * dz;
      a.fx += fx; a.fy += fy; a.fz += fz;
      b.fx -= fx; b.fy -= fy; b.fz -= fz;

      const strain = ext / bm.rest;
      const abs = strain < 0 ? -strain : strain;
      if (abs > bm.deform) {
        if (abs > bm.brk) {
          bm.broken = true;
          if (this.onBeamBreak) this.onBeamBreak(bm, g);
          continue;
        }
        // Yield toward the current length so the dent stays after unloading.
        const excess = ext - Math.sign(ext) * bm.deform * bm.rest;
        bm.rest += excess * bm.plastic;
        if (bm.rest < bm.minRest) bm.rest = bm.minRest;
        bm.damage += Math.abs(excess) * bm.plastic;
      }
    }
  }

  collideStatic(g) {
    const boxes = this._boxCandidates.get(g.id) || [];
    for (const nd of g.nodes) {
      if (!nd.skipGround && nd.y - nd.r < 0) {
        this.resolveContact(nd, 0, 1, 0, nd.r - nd.y, 0.9, 'ground');
      }
      for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        const cx = clamp(nd.x, box.min.x, box.max.x);
        const cy = clamp(nd.y, box.min.y, box.max.y);
        const cz = clamp(nd.z, box.min.z, box.max.z);
        let dx = nd.x - cx, dy = nd.y - cy, dz = nd.z - cz;
        let d2 = dx * dx + dy * dy + dz * dz;

        if (d2 > 1e-12) {
          if (d2 > nd.r * nd.r) continue;
          const d = Math.sqrt(d2);
          this.resolveContact(nd, dx / d, dy / d, dz / d, nd.r - d, box.mu, 'box');
        } else {
          // Centre is inside the box — escape along the shallowest axis.
          const px = Math.min(nd.x - box.min.x, box.max.x - nd.x);
          const py = Math.min(nd.y - box.min.y, box.max.y - nd.y);
          const pz = Math.min(nd.z - box.min.z, box.max.z - nd.z);
          if (px <= py && px <= pz) {
            const s = nd.x - (box.min.x + box.max.x) * 0.5 < 0 ? -1 : 1;
            this.resolveContact(nd, s, 0, 0, px + nd.r, box.mu, 'box');
          } else if (py <= pz) {
            const s = nd.y - (box.min.y + box.max.y) * 0.5 < 0 ? -1 : 1;
            this.resolveContact(nd, 0, s, 0, py + nd.r, box.mu, 'box');
          } else {
            const s = nd.z - (box.min.z + box.max.z) * 0.5 < 0 ? -1 : 1;
            this.resolveContact(nd, 0, 0, s, pz + nd.r, box.mu, 'box');
          }
        }
      }
    }
  }

  resolveContact(nd, nx, ny, nz, pen, mu, kind) {
    if (pen > 0) {
      nd.x += nx * pen;
      nd.y += ny * pen;
      nd.z += nz * pen;
    }
    const vn = nd.vx * nx + nd.vy * ny + nd.vz * nz;
    if (vn >= 0) return;

    if (this.onImpact && -vn > 3) this.onImpact(nd, -vn, kind);

    const j = -(1 + nd.restitution) * vn;
    nd.vx += j * nx; nd.vy += j * ny; nd.vz += j * nz;

    // Tangential drag, capped by Coulomb friction against the normal impulse.
    let tx = nd.vx - (nd.vx * nx + nd.vy * ny + nd.vz * nz) * nx;
    let ty = nd.vy - (nd.vx * nx + nd.vy * ny + nd.vz * nz) * ny;
    let tz = nd.vz - (nd.vx * nx + nd.vy * ny + nd.vz * nz) * nz;
    const tmag = Math.sqrt(tx * tx + ty * ty + tz * tz);
    if (tmag > 1e-5) {
      const maxF = mu * nd.mu * j;
      const scale = Math.min(1, maxF / tmag);
      nd.vx -= tx * scale;
      nd.vy -= ty * scale;
      nd.vz -= tz * scale;
    }
  }

  // Node-vs-node between separate bodies. Node radii are sized so the spheres
  // roughly tile the hull, which is what makes car-on-car hits deform both.
  collideGroups(ga, gb) {
    for (const a of ga.nodes) {
      for (const b of gb.nodes) {
        const rr = a.r + b.r;
        let dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 > rr * rr || d2 < 1e-9) continue;
        const d = Math.sqrt(d2);
        const inv = 1 / d;
        dx *= inv; dy *= inv; dz *= inv;
        const pen = rr - d;

        const totalInv = a.invM + b.invM;
        if (totalInv <= 0) continue;
        const corr = pen * 0.6;
        a.x -= dx * corr * (a.invM / totalInv);
        a.y -= dy * corr * (a.invM / totalInv);
        a.z -= dz * corr * (a.invM / totalInv);
        b.x += dx * corr * (b.invM / totalInv);
        b.y += dy * corr * (b.invM / totalInv);
        b.z += dz * corr * (b.invM / totalInv);

        const rvx = b.vx - a.vx, rvy = b.vy - a.vy, rvz = b.vz - a.vz;
        const vn = rvx * dx + rvy * dy + rvz * dz;
        if (vn >= 0) continue;
        if (this.onImpact && -vn > 4) this.onImpact(a, -vn, 'car');
        // Fully inelastic: any restitution here goes into shoving cars apart
        // instead of into the crumple zones, which is what makes rams read as
        // a bump rather than a crash.
        const jn = -vn / totalInv;
        a.vx -= dx * jn * a.invM;
        a.vy -= dy * jn * a.invM;
        a.vz -= dz * jn * a.invM;
        b.vx += dx * jn * b.invM;
        b.vy += dy * jn * b.invM;
        b.vz += dz * jn * b.invM;
      }
    }
  }
}

function newBounds() {
  return { minX: 0, minY: 0, minZ: 0, maxX: 0, maxY: 0, maxZ: 0, cx: 0, cy: 0, cz: 0 };
}

function updateBounds(g) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (const n of g.nodes) {
    if (n.x < minX) minX = n.x;
    if (n.y < minY) minY = n.y;
    if (n.z < minZ) minZ = n.z;
    if (n.x > maxX) maxX = n.x;
    if (n.y > maxY) maxY = n.y;
    if (n.z > maxZ) maxZ = n.z;
  }
  const b = g.bounds;
  b.minX = minX; b.minY = minY; b.minZ = minZ;
  b.maxX = maxX; b.maxY = maxY; b.maxZ = maxZ;
  b.cx = (minX + maxX) * 0.5;
  b.cy = (minY + maxY) * 0.5;
  b.cz = (minZ + maxZ) * 0.5;
}

function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
export { clamp };
