// Car built as a soft-body lattice: 20 chassis nodes + 4 wheel nodes.
//
// The body has no rigid transform anywhere — its forward/right/up axes are
// re-derived every substep from where the nodes actually are, so a car with a
// caved-in front end genuinely drives like one.

import { makeNode, makeBeam, clamp } from './physics.js';

// Local-space layout, +z forward, +x right, y up from ground.
const CHASSIS = [
  [-0.85, 0.30,  1.95], [ 0.85, 0.30,  1.95],   // 0,1  floor front
  [-0.85, 0.30,  0.75], [ 0.85, 0.30,  0.75],   // 2,3  floor mid-front
  [-0.85, 0.30, -0.75], [ 0.85, 0.30, -0.75],   // 4,5  floor mid-rear
  [-0.85, 0.30, -1.95], [ 0.85, 0.30, -1.95],   // 6,7  floor rear
  [-0.88, 0.95,  1.95], [ 0.88, 0.95,  1.95],   // 8,9  beltline front
  [-0.88, 0.95,  0.75], [ 0.88, 0.95,  0.75],   // 10,11
  [-0.88, 0.95, -0.75], [ 0.88, 0.95, -0.75],   // 12,13
  [-0.88, 0.95, -1.95], [ 0.88, 0.95, -1.95],   // 14,15
  [-0.72, 1.55,  0.55], [ 0.72, 1.55,  0.55],   // 16,17 roof front
  [-0.72, 1.55, -1.15], [ 0.72, 1.55, -1.15],   // 18,19 roof rear
];

// The extreme front and rear planes. Every beam touching one of these is a
// crumple-zone beam; everything else forms the rigid safety cell. Keeping the
// two groups disjoint is what lets the nose fold without the car going floppy:
// driving loads enter through the wheels into the cell, not through the tips.
const FRONT_TIP = new Set([0, 1, 8, 9]);
const REAR_TIP = new Set([6, 7, 14, 15]);

const WHEELS = [
  { pos: [-0.95, 0.36,  1.35], steer: true,  drive: 0.35 },
  { pos: [ 0.95, 0.36,  1.35], steer: true,  drive: 0.35 },
  { pos: [-0.95, 0.36, -1.35], steer: false, drive: 0.65 },
  { pos: [ 0.95, 0.36, -1.35], steer: false, drive: 0.65 },
];

// Surface quads over the chassis nodes. Vertices ARE nodes, so the skin
// crumples with the structure for free.
export const SKIN = [
  [0, 2, 3, 1], [2, 4, 5, 3], [4, 6, 7, 5],       // underside
  [0, 8, 10, 2], [2, 10, 12, 4], [4, 12, 14, 6],  // left flank
  [1, 3, 11, 9], [3, 5, 13, 11], [5, 7, 15, 13],  // right flank
  [0, 1, 9, 8],                                    // front
  [6, 14, 15, 7],                                  // rear
  [8, 9, 11, 10],                                  // hood
  [10, 11, 17, 16],                                // windshield
  [16, 17, 19, 18],                                // roof
  [18, 19, 15, 14],                                // rear glass / trunk
  [10, 16, 18, 12], [11, 13, 19, 17],              // cabin sides
];

// Quads rendered as glass rather than painted bodywork.
export const GLASS_QUADS = new Set([12, 14, 15, 16]);
export const BODY_SKIN = SKIN.filter((_, i) => !GLASS_QUADS.has(i));
export const GLASS_SKIN = SKIN.filter((_, i) => GLASS_QUADS.has(i));

export const WHEEL_RADIUS = 0.36;
const NODE_MASS = 52;
const WHEEL_MASS = 32;

const TOP_SPEED = 46;      // m/s, drive force fades toward this
const DRIVE_FORCE = 15500; // N total at full throttle
const BRAKE_FORCE = 22000;
const TIRE_K = 110000;
const TIRE_D = 4200;
const GRIP_K = 14000;
const TIRE_MU = 1.45;
const MAX_STEER = 0.58;

export class Vehicle {
  constructor(world, opts = {}) {
    this.world = world;
    this.spawn = { x: opts.x ?? 0, z: opts.z ?? 0, heading: opts.heading ?? 0 };
    this.color = opts.color ?? 0xcc3322;
    this.isPlayer = opts.isPlayer ?? false;
    this.name = opts.name ?? 'car';

    this.controls = { throttle: 0, steer: 0, brake: 0, handbrake: false };
    this.steerAngle = 0;
    this.wheelSpin = [0, 0, 0, 0];
    this.wheelLoad = [0, 0, 0, 0];
    this.wheelSlip = [0, 0, 0, 0];
    this.speed = 0;
    this.rpm = 0;
    this.disabled = false;

    this.fwd = { x: 0, y: 0, z: 1 };
    this.right = { x: 1, y: 0, z: 0 };
    this.up = { x: 0, y: 1, z: 0 };
    this.pos = { x: 0, y: 0, z: 0 };
    this.vel = { x: 0, y: 0, z: 0 };

    this.build();
    world.preSolve.push((dt) => this.update(dt));
  }

  build() {
    const { x, z, heading } = this.spawn;
    const ch = Math.cos(heading), sh = Math.sin(heading);
    const toWorld = (lx, ly, lz) => ({
      x: x + lx * ch + lz * sh,
      y: ly,
      z: z - lx * sh + lz * ch,
    });

    this.nodes = [];
    this.chassisNodes = [];
    for (const [lx, ly, lz] of CHASSIS) {
      const p = toWorld(lx, ly, lz);
      // Low restitution: crash energy should go into bending metal, not bounce.
      const n = makeNode(p.x, p.y, p.z, NODE_MASS,
        { r: 0.26, mu: 0.55, drag: 0.05, restitution: 0.02 });
      this.nodes.push(n);
      this.chassisNodes.push(n);
    }

    this.wheelNodes = [];
    for (const w of WHEELS) {
      const p = toWorld(...w.pos);
      // skipGround: the tire model owns ground contact for wheels.
      const n = makeNode(p.x, p.y, p.z, WHEEL_MASS, {
        r: WHEEL_RADIUS, mu: 1.0, drag: 0.02, isWheel: true, skipGround: true,
      });
      this.nodes.push(n);
      this.wheelNodes.push(n);
    }

    // Connectivity is deliberately sparse. An all-pairs truss is effectively
    // incompressible: a frontal hit spreads strain over every beam at once, so
    // none reaches yield and the car rebounds elastically instead of denting.
    // Local beams crumple; a handful of explicit long rails carry global shape.
    this.beams = [];
    const N = this.chassisNodes;
    const made = new Set();
    const link = (i, j, opts) => {
      const key = i < j ? `${i},${j}` : `${j},${i}`;
      if (made.has(key)) return;
      made.add(key);
      this.beams.push(makeBeam(N[i], N[j], opts));
    };

    for (let i = 0; i < N.length; i++) {
      for (let j = i + 1; j < N.length; j++) {
        const d = len3(N[i], N[j]);
        const crumple = FRONT_TIP.has(i) || FRONT_TIP.has(j)
          || REAR_TIP.has(i) || REAR_TIP.has(j);
        // Tips attach with a few short beams only. Long tip-to-cell diagonals
        // multiply the crush force (yield scales with rest length), which is
        // what kept impacts invisible.
        if (d > (crumple ? 1.9 : 2.65)) continue;

        if (crumple) {
          // Yields at ~1% strain so the force plateaus almost immediately and
          // the nose concertinas instead of stopping the car dead. This is the
          // entire reason impacts are visible rather than merely numeric.
          link(i, j, { k: 6e5, dampRatio: 0.22, deform: 0.01, brk: 0.7, plastic: 0.16 });
        } else {
          // Safety cell: near-complete graph over the 12 cabin/floor nodes and
          // stiff enough that it effectively never deforms. Carries all the
          // driving loads so the crumple zones never have to.
          link(i, j, { k: 1.3e6, dampRatio: 0.25, deform: 0.06, brk: 0.6, plastic: 0.08 });
        }
      }
    }

    for (const wn of this.wheelNodes) {
      const sorted = N.map((n, i) => ({ n, d: len3(wn, n), i }))
        .sort((a, b) => a.d - b.d).slice(0, 6);
      for (const s of sorted) {
        this.beams.push(makeBeam(wn, s.n, {
          k: 2.6e5, dampRatio: 0.32, deform: 0.28, brk: 1.25, plastic: 0.003,
        }));
      }
    }

    this.totalBeams = this.beams.length;
    this.group = this.world.addGroup(this.nodes, this.beams);
    this.group.vehicle = this;
    this.updateFrame();
  }

  reset(x, z, heading) {
    if (x !== undefined) this.spawn = { x, z, heading: heading ?? 0 };
    const { x: sx, z: sz, heading: sh } = this.spawn;
    const c = Math.cos(sh), s = Math.sin(sh);
    const place = (node, lx, ly, lz) => {
      node.x = sx + lx * c + lz * s;
      node.y = ly;
      node.z = sz - lx * s + lz * c;
      node.vx = node.vy = node.vz = 0;
    };
    CHASSIS.forEach((p, i) => place(this.chassisNodes[i], ...p));
    WHEELS.forEach((w, i) => place(this.wheelNodes[i], ...w.pos));
    for (const b of this.beams) {
      b.rest = b.restOrig;
      b.broken = false;
      b.damage = 0;
    }
    this.disabled = false;
    this.steerAngle = 0;
    this.updateFrame();
  }

  // Measured straight off permanent beam-length change, so it always matches
  // the visible dents and resets cleanly with the car.
  get damage() {
    let broken = 0, deformed = 0;
    for (const b of this.beams) {
      if (b.broken) { broken++; continue; }
      deformed += Math.abs(b.rest - b.restOrig) / b.restOrig;
    }
    // Calibrated so a ~90mph wall hit totals the car and a 30mph nudge is
    // cosmetic; snapped beams count for far more than bent ones.
    const n = this.totalBeams;
    return clamp((broken / n) * 3 + (deformed / n) * 16, 0, 1);
  }

  // Body axes straight off the node cloud — no stored orientation to go stale.
  updateFrame() {
    const N = this.chassisNodes;
    const frontX = (N[0].x + N[1].x) * 0.5, frontY = (N[0].y + N[1].y) * 0.5, frontZ = (N[0].z + N[1].z) * 0.5;
    const rearX = (N[6].x + N[7].x) * 0.5, rearY = (N[6].y + N[7].y) * 0.5, rearZ = (N[6].z + N[7].z) * 0.5;
    let f = norm3(frontX - rearX, frontY - rearY, frontZ - rearZ);
    if (!f) f = { x: 0, y: 0, z: 1 };

    const leftX = (N[0].x + N[2].x) * 0.5, leftY = (N[0].y + N[2].y) * 0.5, leftZ = (N[0].z + N[2].z) * 0.5;
    const rightX = (N[1].x + N[3].x) * 0.5, rightY = (N[1].y + N[3].y) * 0.5, rightZ = (N[1].z + N[3].z) * 0.5;
    let r = norm3(rightX - leftX, rightY - leftY, rightZ - leftZ);
    if (!r) r = { x: 1, y: 0, z: 0 };

    let u = norm3(
      f.y * r.z - f.z * r.y,
      f.z * r.x - f.x * r.z,
      f.x * r.y - f.y * r.x
    ) || { x: 0, y: 1, z: 0 };
    // Re-orthogonalise so a twisted chassis still yields a sane basis.
    r = norm3(
      u.y * f.z - u.z * f.y,
      u.z * f.x - u.x * f.z,
      u.x * f.y - u.y * f.x
    ) || r;

    this.fwd = f; this.right = r; this.up = u;

    let px = 0, py = 0, pz = 0, vx = 0, vy = 0, vz = 0;
    for (const n of this.nodes) {
      px += n.x; py += n.y; pz += n.z;
      vx += n.vx; vy += n.vy; vz += n.vz;
    }
    const inv = 1 / this.nodes.length;
    this.pos = { x: px * inv, y: py * inv, z: pz * inv };
    this.vel = { x: vx * inv, y: vy * inv, z: vz * inv };
    this.speed = Math.hypot(this.vel.x, this.vel.z);
  }

  update(dt) {
    if (this.group && !this.group.active) return;
    this.updateFrame();
    const c = this.controls;

    if (this.damage > 0.72) this.disabled = true;

    // Steering rate-limited and tightened down at speed.
    const speedFactor = 1 / (1 + this.speed * 0.045);
    const target = clamp(c.steer, -1, 1) * MAX_STEER * speedFactor;
    const rate = 5.5 * dt;
    this.steerAngle += clamp(target - this.steerAngle, -rate, rate);

    const throttle = this.disabled ? 0 : clamp(c.throttle, -1, 1);
    const fade = clamp(1 - this.speed / TOP_SPEED, 0, 1);

    for (let i = 0; i < 4; i++) {
      const w = this.wheelNodes[i];
      const cfg = WHEELS[i];

      const pen = WHEEL_RADIUS - w.y;
      if (pen <= 0) {
        this.wheelLoad[i] = 0;
        this.wheelSlip[i] = 0;
        continue;
      }

      let Fn = TIRE_K * pen - TIRE_D * w.vy;
      if (Fn < 0) Fn = 0;
      w.fy += Fn;
      this.wheelLoad[i] = Fn;

      // Contact-patch axes, flattened to the ground plane.
      const ang = cfg.steer ? this.steerAngle : 0;
      const ca = Math.cos(ang), sa = Math.sin(ang);
      let dx = this.fwd.x * ca + this.right.x * sa;
      let dz = this.fwd.z * ca + this.right.z * sa;
      const dl = Math.hypot(dx, dz) || 1;
      dx /= dl; dz /= dl;
      const rx = -dz, rz = dx;

      const vf = w.vx * dx + w.vz * dz;
      const vr = w.vx * rx + w.vz * rz;

      let Flong = 0;
      if (throttle > 0) Flong = throttle * DRIVE_FORCE * cfg.drive * fade;
      else if (throttle < 0) {
        // Same stick does braking while rolling forward, reverse once stopped.
        // Reverse needs its own speed fade or it winds up to forward top speed.
        Flong = vf > 0.6
          ? -BRAKE_FORCE * cfg.drive
          : throttle * DRIVE_FORCE * 0.45 * cfg.drive * clamp(1 + vf / 13, 0, 1);
      }
      if (c.brake > 0) Flong -= Math.sign(vf) * BRAKE_FORCE * cfg.drive * c.brake;
      Flong -= vf * 55; // rolling resistance

      let grip = GRIP_K;
      let mu = TIRE_MU;
      if (c.handbrake && !cfg.steer) {
        Flong = -vf * 2600;   // rear wheels dragged toward locked
        grip *= 0.32;
        mu *= 0.55;
      }
      let Flat = -vr * grip;

      // Friction circle: longitudinal and lateral demand share one budget.
      const maxF = mu * Fn;
      const mag = Math.hypot(Flong, Flat);
      if (mag > maxF && mag > 1e-6) {
        const s = maxF / mag;
        Flong *= s; Flat *= s;
        this.wheelSlip[i] = clamp((mag / maxF - 1) * 0.6, 0, 1);
      } else {
        this.wheelSlip[i] = 0;
      }

      w.fx += Flong * dx + Flat * rx;
      w.fz += Flong * dz + Flat * rz;

      this.wheelSpin[i] += (vf / WHEEL_RADIUS) * dt;
    }

    const fwdSpeed = this.vel.x * this.fwd.x + this.vel.z * this.fwd.z;
    this.rpm = clamp(Math.abs(fwdSpeed) / TOP_SPEED, 0, 1);
  }
}

function len3(a, b) {
  return Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z);
}

function norm3(x, y, z) {
  const l = Math.hypot(x, y, z);
  if (l < 1e-9) return null;
  return { x: x / l, y: y / l, z: z / l };
}
