/**
 * physics3d.js — 3-D Verlet Physics Engine
 * Siege Mech Evolution
 *
 * Node3D          – point-mass with Verlet integration
 * Constraint3D    – bone / muscle spring between two nodes
 * Terrain         – procedural height-field (flat, hills, craters)
 * MechBody        – skeletal mech assembled from nodes + constraints
 * Projectile      – ballistic shell with trail
 * Target          – destructible tower / drone
 */

/* ================================================================== */
/*  Node3D                                                            */
/* ================================================================== */

class Node3D {
  /**
   * @param {number} x
   * @param {number} y
   * @param {number} z
   * @param {number} [mass=1]
   * @param {boolean} [pinned=false]
   */
  constructor(x, y, z, mass = 1.0, pinned = false) {
    this.x = x;
    this.y = y;
    this.z = z;

    /** Previous-frame positions (Verlet) */
    this.ox = x;
    this.oy = y;
    this.oz = z;

    this.mass = mass;
    this.pinned = pinned;
    this.radius = 3;
  }

  /**
   * Verlet integration step.
   * @param {number} gravity  Downward acceleration (positive = stronger pull)
   * @param {number} drag     Velocity damping 0-1 (e.g. 0.99)
   * @param {{ x: number, z: number }} wind  Wind force vector
   */
  update(gravity, drag, wind) {
    if (this.pinned) return;

    const vx = (this.x - this.ox) * drag;
    const vy = (this.y - this.oy) * drag;
    const vz = (this.z - this.oz) * drag;

    this.ox = this.x;
    this.oy = this.y;
    this.oz = this.z;

    const invMass = 1 / this.mass;

    this.x += vx + wind.x * invMass;
    this.y += vy - gravity;
    this.z += vz + wind.z * invMass;
  }
}

/* ================================================================== */
/*  Constraint3D                                                      */
/* ================================================================== */

class Constraint3D {
  /**
   * @param {Node3D} nodeA
   * @param {Node3D} nodeB
   * @param {number} restLength
   * @param {number} [stiffness=1.0]
   * @param {string} [type='bone']  'bone' | 'muscle'
   */
  constructor(nodeA, nodeB, restLength, stiffness = 1.0, type = 'bone') {
    this.a = nodeA;
    this.b = nodeB;
    this.restLength = restLength;
    this.stiffness = stiffness;
    this.type = type;

    /** Muscle oscillation parameters (set externally for muscles) */
    this.amplitude = 0;
    this.frequency = 0;
    this.phase = 0;

    /** Rendering colour — overridden per frame for muscles */
    this.color = type === 'bone' ? '#88AACC' : '#44CC66';
  }

  /**
   * Resolve the distance constraint (with optional periodic actuation).
   * @param {number} time  Simulation clock in seconds
   */
  resolve(time) {
    let dx = this.b.x - this.a.x;
    let dy = this.b.y - this.a.y;
    let dz = this.b.z - this.a.z;
    let dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 0.0001;

    let target;
    if (this.type === 'muscle') {
      target = this.restLength * (1 + this.amplitude * Math.sin(this.frequency * time + this.phase));
    } else {
      target = this.restLength;
    }

    if (this.type === 'min-bone' && dist >= target) {
      return;
    }

    const diff = (dist - target) / dist * this.stiffness;
    const totalMass = this.a.mass + this.b.mass;
    const wA = this.a.pinned ? 0 : this.b.mass / totalMass;
    const wB = this.b.pinned ? 0 : this.a.mass / totalMass;

    const cx = dx * diff;
    const cy = dy * diff;
    const cz = dz * diff;

    if (!this.a.pinned) {
      this.a.x += cx * wA;
      this.a.y += cy * wA;
      this.a.z += cz * wA;
    }
    if (!this.b.pinned) {
      this.b.x -= cx * wB;
      this.b.y -= cy * wB;
      this.b.z -= cz * wB;
    }
  }
}

/* ================================================================== */
/*  Terrain                                                           */
/* ================================================================== */

class Terrain {
  /**
   * @param {string} [type='flat']  'flat' | 'hills' | 'craters'
   * @param {number} [width=800]
   * @param {number} [depth=400]
   */
  constructor(type = 'flat', width = 800, depth = 400) {
    this.type = type;
    this.width = width;
    this.depth = depth;

    /** Crater positions for the 'craters' variant */
    this.craters = [
      { cx: 100, cz: 60, r: 70, d: 25 },
      { cx: -150, cz: 100, r: 55, d: 20 },
      { cx: 250, cz: -50, r: 80, d: 30 },
      { cx: -50, cz: -100, r: 60, d: 22 },
      { cx: 350, cz: 80, r: 65, d: 18 }
    ];
  }

  /**
   * Return interpolated height at a world (x, z) position.
   * @param {number} x
   * @param {number} z
   * @returns {number}
   */
  getHeight(x, z) {
    switch (this.type) {
      case 'hills':
        return Math.sin(x * 0.015) * 12 +
               Math.sin(z * 0.02) * 8 +
               Math.sin(x * 0.007 + z * 0.009) * 6;

      case 'craters': {
        let h = 0;
        for (let i = 0; i < this.craters.length; i++) {
          const c = this.craters[i];
          const dx = x - c.cx;
          const dz = z - c.cz;
          const dist = Math.sqrt(dx * dx + dz * dz);
          if (dist < c.r) {
            const t = dist / c.r;
            h = Math.min(h, -c.d * (1 - t * t));
          }
        }
        return h;
      }

      default: // 'flat'
        return 0;
    }
  }

  /**
   * Generate an array of line-segment descriptors for rendering
   * a wireframe mesh of the terrain.
   *
   * Each entry: { x1, y1, z1, x2, y2, z2 }
   * @returns {object[]}
   */
  getGridLines() {
    const lines = [];
    const step = 40;
    const hw = this.width * 0.5;
    const hd = this.depth * 0.5;

    for (let gx = -hw; gx <= hw; gx += step) {
      for (let gz = -hd; gz <= hd; gz += step) {
        const h = this.getHeight(gx, gz);

        // line toward +X
        if (gx + step <= hw) {
          const h2 = this.getHeight(gx + step, gz);
          lines.push({ x1: gx, y1: h, z1: gz, x2: gx + step, y2: h2, z2: gz });
        }
        // line toward +Z
        if (gz + step <= hd) {
          const h2 = this.getHeight(gx, gz + step);
          lines.push({ x1: gx, y1: h, z1: gz, x2: gx, y2: h2, z2: gz + step });
        }
      }
    }
    return lines;
  }
}

/* ================================================================== */
/*  MechBody                                                          */
/* ================================================================== */

/**
 * Internal helper: Euclidean distance between two 3-D points.
 */
function _dist3(ax, ay, az, bx, by, bz) {
  const dx = bx - ax, dy = by - ay, dz = bz - az;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

class MechBody {
  constructor(template, startX, startZ, config = null, terrainHeight = 0) {
    this.template = template;
    this.nodes = [];
    this.constraints = [];
    this.turretNode = null;
    this.turretAnglePitch = -0.2;
    this.turretAngleYaw = 0;
    this.isDead = false;
    this.startX = startX;
    this.startZ = startZ;
    this.distanceMoved = 0;
    this.hits = 0;
    this.misses = 0;
    this.shotsRemaining = 10;
    this.fireCooldown = 0;
    this.hasFired = false;

    this.footIndices = [];
    this.fallIndices = [];
    this.isFallen = false;
    this.survivalTime = 0;

    this._buildSkeleton(template, startX, startZ, config, terrainHeight);

    // Reset velocities (previous positions = current positions) to start at rest
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      n.ox = n.x;
      n.oy = n.y;
      n.oz = n.z;
    }
  }

  /* -------- skeleton builders -------- */

  /**
   * Dispatch to the correct template builder.
   */
  _buildSkeleton(template, sx, sz, cfg, toy = 0) {
    switch (template) {
      case 'biped':    this._buildBiped(sx, sz, cfg, toy); break;
      case 'tripod':   this._buildTripod(sx, sz, cfg, toy); break;
      case 'quadruped': this._buildQuadruped(sx, sz, cfg, toy); break;
      default:         this._buildBiped(sx, sz, cfg, toy); break;
    }
  }

  _addNode(x, y, z, mass) {
    const n = new Node3D(x, y, z, mass);
    this.nodes.push(n);
    return n;
  }

  _addBone(a, b, stiff, cfg, boneIdx) {
    let rest = _dist3(a.x, a.y, a.z, b.x, b.y, b.z);
    let s = stiff;
    if (cfg && cfg.limbLengths && cfg.limbLengths[boneIdx] !== undefined) {
      rest *= cfg.limbLengths[boneIdx];
    }
    const c = new Constraint3D(a, b, rest, s, 'bone');
    this.constraints.push(c);
    return c;
  }

  _addBoneDirect(a, b, stiff, type = 'bone') {
    let rest = _dist3(a.x, a.y, a.z, b.x, b.y, b.z);
    const c = new Constraint3D(a, b, rest, stiff, type);
    this.constraints.push(c);
    return c;
  }

  _addMuscle(a, b, amp, freq, phase, cfg, muscleIdx) {
    let rest = _dist3(a.x, a.y, a.z, b.x, b.y, b.z);
    if (cfg && cfg.limbLengths) {
      const off = this.constraints.filter(c => c.type === 'bone').length + muscleIdx;
      if (cfg.limbLengths[off] !== undefined) rest *= cfg.limbLengths[off];
    }
    const c = new Constraint3D(a, b, rest, 0.9, 'muscle');
    c.amplitude = cfg && cfg.muscleAmps ? cfg.muscleAmps[muscleIdx] : amp;
    c.frequency = cfg && cfg.muscleFreqs ? cfg.muscleFreqs[muscleIdx] : freq;
    c.phase     = cfg && cfg.musclePhases ? cfg.musclePhases[muscleIdx] : phase;
    this.constraints.push(c);
    return c;
  }

  _addMuscleDirect(a, b, amp, freq, phase, cfg, muscleIdx) {
    let rest = _dist3(a.x, a.y, a.z, b.x, b.y, b.z);
    const c = new Constraint3D(a, b, rest, 0.9, 'muscle');
    c.amplitude = cfg && cfg.muscleAmps ? cfg.muscleAmps[muscleIdx] : amp;
    c.frequency = cfg && cfg.muscleFreqs ? cfg.muscleFreqs[muscleIdx] : freq;
    c.phase     = cfg && cfg.musclePhases ? cfg.musclePhases[muscleIdx] : phase;
    this.constraints.push(c);
    return c;
  }

  _addMinBoneDirect(a, b, fraction, stiff = 0.8) {
    let rest = _dist3(a.x, a.y, a.z, b.x, b.y, b.z) * fraction;
    const c = new Constraint3D(a, b, rest, stiff, 'min-bone');
    this.constraints.push(c);
    return c;
  }

  /** Apply node mass multipliers from config */
  _applyMasses(cfg) {
    if (cfg && cfg.nodeMasses) {
      for (let i = 0; i < this.nodes.length && i < cfg.nodeMasses.length; i++) {
        this.nodes[i].mass = cfg.nodeMasses[i];
      }
    }
  }

  /* ---------- BIPED ---------- */
  _buildBiped(sx, sz, cfg, toy = 0) {
    // 17 Nodes
    const hip       = this._addNode(sx, 32 + toy, sz, 2.0);       // 0 (pelvis)
    const waist     = this._addNode(sx, 42 + toy, sz, 1.6);       // 1 (spine mid)
    const torso     = this._addNode(sx, 52 + toy, sz, 1.5);       // 2 (chest)
    const neck      = this._addNode(sx, 60 + toy, sz, 1.0);       // 3
    const head      = this._addNode(sx, 68 + toy, sz, 1.1);       // 4 turret

    const lShoulder = this._addNode(sx, 52 + toy, sz - 8, 1.0);   // 5
    const lElbow    = this._addNode(sx - 6, 42 + toy, sz - 11, 0.9); // 6
    const lHand     = this._addNode(sx - 10, 32 + toy, sz - 13, 0.9); // 7

    const rShoulder = this._addNode(sx, 52 + toy, sz + 8, 1.0);   // 8
    const rElbow    = this._addNode(sx - 6, 42 + toy, sz + 11, 0.9); // 9
    const rHand     = this._addNode(sx - 10, 32 + toy, sz + 13, 0.9); // 10

    const lKnee     = this._addNode(sx, 18 + toy, sz - 20, 1.1);  // 11
    const lAnkle    = this._addNode(sx - 6, 3 + toy, sz - 20, 1.1);  // 12
    const lFoot     = this._addNode(sx + 8, 3 + toy, sz - 20, 1.2);  // 13

    const rKnee     = this._addNode(sx, 18 + toy, sz + 20, 1.1);  // 14
    const rAnkle    = this._addNode(sx - 6, 3 + toy, sz + 20, 1.1);  // 15
    const rFoot     = this._addNode(sx + 8, 3 + toy, sz + 20, 1.2);  // 16

    this.turretNode = head;
    this.footIndices = [12, 13, 15, 16]; // ankles + feet
    this.fallIndices = [1, 2, 3, 4]; // waist, torso, neck, head

    // 16 Core Bones
    this._addBone(hip, waist, 1.0, cfg, 0);
    this._addBone(waist, torso, 1.0, cfg, 1);
    this._addBone(torso, neck, 1.0, cfg, 2);
    this._addBone(neck, head, 1.0, cfg, 3);

    this._addBone(torso, lShoulder, 1.0, cfg, 4);
    this._addBone(torso, rShoulder, 1.0, cfg, 5);

    this._addBone(lShoulder, lElbow, 0.9, cfg, 6);
    this._addBone(lElbow, lHand, 0.9, cfg, 7);

    this._addBone(rShoulder, rElbow, 0.9, cfg, 8);
    this._addBone(rElbow, rHand, 0.9, cfg, 9);

    this._addBone(hip, lKnee, 0.95, cfg, 10);
    this._addBone(lKnee, lAnkle, 0.95, cfg, 11);
    this._addBone(lAnkle, lFoot, 0.95, cfg, 12);

    this._addBone(hip, rKnee, 0.95, cfg, 13);
    this._addBone(rKnee, rAnkle, 0.95, cfg, 14);
    this._addBone(rAnkle, rFoot, 0.95, cfg, 15);

    this._applyMasses(cfg);

    // Phase 1: Settle core bones
    for (let iter = 0; iter < 100; iter++) {
      for (let i = 0; i < this.constraints.length; i++) {
        if (this.constraints[i].type === 'bone') {
          this.constraints[i].resolve(0);
        }
      }
      for (let i = 0; i < this.nodes.length; i++) {
        const n = this.nodes[i];
        const isFoot = this.footIndices.includes(i);
        if (isFoot) {
          n.y = toy + n.radius;
        } else if (n.y < toy + n.radius) {
          n.y = toy + n.radius;
        }
      }
    }

    // Phase 2: Add braces using settled node positions (4 braces + stability truss)
    this._addBoneDirect(lShoulder, rShoulder, 0.8, 'bone'); // shoulder brace
    this._addBoneDirect(lKnee, rKnee, 0.7, 'bone'); // knee lateral stabilization
    this._addBoneDirect(lAnkle, rAnkle, 0.7, 'bone'); // ankle lateral stabilization
    this._addBoneDirect(hip, torso, 0.8, 'bone'); // spine brace

    // New stability cross-bracing and tendons to keep upright
    this._addBoneDirect(lKnee, rAnkle, 0.3, 'bone');
    this._addBoneDirect(rKnee, lAnkle, 0.3, 'bone');
    this._addBoneDirect(hip, lAnkle, 0.3, 'bone');
    this._addBoneDirect(hip, rAnkle, 0.3, 'bone');
    this._addBoneDirect(lKnee, lFoot, 0.3, 'bone');
    this._addBoneDirect(rKnee, rFoot, 0.3, 'bone');
    this._addBoneDirect(lFoot, rFoot, 0.3, 'bone'); // foot yaw stabilizer
    this._addBoneDirect(torso, lKnee, 0.3, 'bone');
    this._addBoneDirect(torso, rKnee, 0.3, 'bone');
    this._addBoneDirect(torso, lAnkle, 0.3, 'bone');
    this._addBoneDirect(torso, rAnkle, 0.3, 'bone');
    this._addBoneDirect(torso, head, 0.3, 'bone'); // neck stabilizer

    // Add 11 Muscles
    this._addMuscleDirect(waist, lKnee, 0.25, 4, 0, cfg, 0); // left hip flexor
    this._addMuscleDirect(hip, lAnkle, 0.2, 4, Math.PI * 0.5, cfg, 1); // left knee actuator
    this._addMuscleDirect(lKnee, lFoot, 0.15, 4, Math.PI, cfg, 2); // left ankle actuator

    this._addMuscleDirect(waist, rKnee, 0.25, 4, Math.PI, cfg, 3); // right hip flexor
    this._addMuscleDirect(hip, rAnkle, 0.2, 4, Math.PI * 1.5, cfg, 4); // right knee actuator
    this._addMuscleDirect(rKnee, rFoot, 0.15, 4, 0, cfg, 5); // right ankle actuator

    this._addMuscleDirect(torso, lElbow, 0.15, 4, Math.PI * 0.5, cfg, 6); // left arm swing
    this._addMuscleDirect(lShoulder, lHand, 0.1, 4, Math.PI, cfg, 7); // left elbow flexor

    this._addMuscleDirect(torso, rElbow, 0.15, 4, Math.PI * 1.5, cfg, 8); // right arm swing
    this._addMuscleDirect(rShoulder, rHand, 0.1, 4, 0, cfg, 9); // right elbow flexor

    this._addMuscleDirect(hip, neck, 0.15, 4, Math.PI * 0.25, cfg, 10); // torso stabilizer
  }

  /* ---------- TRIPOD ---------- */
  _buildTripod(sx, sz, cfg, toy = 0) {
    const hub    = this._addNode(sx, 45 + toy, sz, 2.5);      // 0
    const turret = this._addNode(sx, 65 + toy, sz, 1);        // 1

    const k0 = this._addNode(sx, 25 + toy, sz + 18, 1);       // 2
    const f0 = this._addNode(sx, 2 + toy, sz + 24, 1.2);      // 3
    const k1 = this._addNode(sx - 16, 25 + toy, sz - 10, 1);  // 4
    const f1 = this._addNode(sx - 22, 2 + toy, sz - 14, 1.2); // 5
    const k2 = this._addNode(sx + 16, 25 + toy, sz - 10, 1);  // 6
    const f2 = this._addNode(sx + 22, 2 + toy, sz - 14, 1.2); // 7

    this.turretNode = turret;
    this.footIndices = [3, 5, 7];
    this.fallIndices = [1];

    // 7 Core Bones
    this._addBone(hub, turret, 1, cfg, 0);
    this._addBone(hub, k0, 0.9, cfg, 1);
    this._addBone(k0, f0, 0.9, cfg, 2);
    this._addBone(hub, k1, 0.9, cfg, 3);
    this._addBone(k1, f1, 0.9, cfg, 4);
    this._addBone(hub, k2, 0.9, cfg, 5);
    this._addBone(k2, f2, 0.9, cfg, 6);

    this._applyMasses(cfg);

    // Phase 1: Settle core bones
    for (let iter = 0; iter < 100; iter++) {
      for (let i = 0; i < this.constraints.length; i++) {
        if (this.constraints[i].type === 'bone') {
          this.constraints[i].resolve(0);
        }
      }
      for (let i = 0; i < this.nodes.length; i++) {
        const n = this.nodes[i];
        const isFoot = this.footIndices.includes(i);
        if (isFoot) {
          n.y = toy + n.radius;
        } else if (n.y < toy + n.radius) {
          n.y = toy + n.radius;
        }
      }
    }

    // Phase 2: Braces (3 braces)
    this._addBoneDirect(turret, k0, 0.3, 'bone');
    this._addBoneDirect(turret, k1, 0.3, 'bone');
    this._addBoneDirect(turret, k2, 0.3, 'bone');

    // Muscles (3 muscles)
    this._addMuscleDirect(hub, f0, 0.3, 3.5, 0, cfg, 0);
    this._addMuscleDirect(hub, f1, 0.3, 3.5, Math.PI * 0.67, cfg, 1);
    this._addMuscleDirect(hub, f2, 0.3, 3.5, Math.PI * 1.33, cfg, 2);
  }

  /* ---------- QUADRUPED ---------- */
  _buildQuadruped(sx, sz, cfg, toy = 0) {
    const fHip   = this._addNode(sx, 40 + toy, sz + 15, 2);       // 0
    const rHip   = this._addNode(sx, 40 + toy, sz - 15, 2);       // 1
    const torso  = this._addNode(sx, 50 + toy, sz, 1.5);          // 2
    const turret = this._addNode(sx, 62 + toy, sz + 10, 1);       // 3

    // Front-left
    const flK = this._addNode(sx - 14, 22 + toy, sz + 18, 1);     // 4
    const flF = this._addNode(sx - 16, 2 + toy, sz + 20, 1.2);    // 5
    // Front-right
    const frK = this._addNode(sx + 14, 22 + toy, sz + 18, 1);     // 6
    const frF = this._addNode(sx + 16, 2 + toy, sz + 20, 1.2);    // 7
    // Rear-left
    const rlK = this._addNode(sx - 14, 22 + toy, sz - 18, 1);     // 8
    const rlF = this._addNode(sx - 16, 2 + toy, sz - 20, 1.2);    // 9
    // Rear-right
    const rrK = this._addNode(sx + 14, 22 + toy, sz - 18, 1);     // 10
    const rrF = this._addNode(sx + 16, 2 + toy, sz - 20, 1.2);    // 11

    this.turretNode = turret;
    this.footIndices = [5, 7, 9, 11];
    this.fallIndices = [2, 3];

    // 12 Core Bones
    this._addBone(fHip, rHip, 1, cfg, 0);
    this._addBone(fHip, torso, 1, cfg, 1);
    this._addBone(rHip, torso, 1, cfg, 2);
    this._addBone(torso, turret, 1, cfg, 3);

    this._addBone(fHip, flK, 0.9, cfg, 4);
    this._addBone(flK, flF, 0.9, cfg, 5);
    this._addBone(fHip, frK, 0.9, cfg, 6);
    this._addBone(frK, frF, 0.9, cfg, 7);

    this._addBone(rHip, rlK, 0.9, cfg, 8);
    this._addBone(rlK, rlF, 0.9, cfg, 9);
    this._addBone(rHip, rrK, 0.9, cfg, 10);
    this._addBone(rrK, rrF, 0.9, cfg, 11);

    this._applyMasses(cfg);

    // Phase 1: Settle core bones
    for (let iter = 0; iter < 100; iter++) {
      for (let i = 0; i < this.constraints.length; i++) {
        if (this.constraints[i].type === 'bone') {
          this.constraints[i].resolve(0);
        }
      }
      for (let i = 0; i < this.nodes.length; i++) {
        const n = this.nodes[i];
        const isFoot = this.footIndices.includes(i);
        if (isFoot) {
          n.y = toy + n.radius;
        } else if (n.y < toy + n.radius) {
          n.y = toy + n.radius;
        }
      }
    }

    // Phase 2: Braces (4 braces + knee stability truss)
    this._addBoneDirect(torso, flK, 0.3, 'bone');
    this._addBoneDirect(torso, frK, 0.3, 'bone');
    this._addBoneDirect(torso, rlK, 0.3, 'bone');
    this._addBoneDirect(torso, rrK, 0.3, 'bone');

    // Add passive knee cross-braces to form a stable box
    this._addBoneDirect(flK, frK, 0.3, 'bone');
    this._addBoneDirect(rlK, rrK, 0.3, 'bone');
    this._addBoneDirect(flK, rlK, 0.3, 'bone');
    this._addBoneDirect(frK, rrK, 0.3, 'bone');
    this._addBoneDirect(flK, rrK, 0.3, 'bone');
    this._addBoneDirect(frK, rlK, 0.3, 'bone');

    // Muscles (8 muscles)
    this._addMuscleDirect(torso, flK, 0.25, 4, 0, cfg, 0);
    this._addMuscleDirect(fHip, flF, 0.2, 4, Math.PI * 0.5, cfg, 1);
    this._addMuscleDirect(torso, frK, 0.25, 4, Math.PI, cfg, 2);
    this._addMuscleDirect(fHip, frF, 0.2, 4, Math.PI * 1.5, cfg, 3);
    this._addMuscleDirect(torso, rlK, 0.25, 4, Math.PI * 0.25, cfg, 4);
    this._addMuscleDirect(rHip, rlF, 0.2, 4, Math.PI * 0.75, cfg, 5);
    this._addMuscleDirect(torso, rrK, 0.25, 4, Math.PI * 1.25, cfg, 6);
    this._addMuscleDirect(rHip, rrF, 0.2, 4, Math.PI * 1.75, cfg, 7);

    this.footIndices = [5, 7, 9, 11];
    this.fallIndices = [2, 3];
    this._applyMasses(cfg);
  }

  /* ================================================================ */
  /*  Physics tick                                                    */
  /* ================================================================ */

  /**
   * Advance physics one time-step.
   * @param {number} time     Simulation clock (s)
   * @param {number} gravity  Gravity strength
   * @param {number} drag     Velocity damping
   * @param {{ x: number, z: number }} wind
   * @param {Terrain} terrain
   */
  update(time, gravity, drag, wind, terrain) {
    if (this.isDead || this.isFallen) return;

    this.survivalTime += 1 / 60; // tracks survival time (s) before falling

    // Integrate nodes
    for (let i = 0; i < this.nodes.length; i++) {
      this.nodes[i].update(gravity, drag, wind);
    }

    // Solve constraints and terrain collision together (25 iterations for stability)
    for (let iter = 0; iter < 25; iter++) {
      for (let i = 0; i < this.constraints.length; i++) {
        this.constraints[i].resolve(time);
      }
      // Intermediate ground projection
      for (let i = 0; i < this.nodes.length; i++) {
        const n = this.nodes[i];
        const ground = terrain.getHeight(n.x, n.z);
        if (n.y < ground + n.radius) {
          n.y = ground + n.radius;
        }
      }
    }

    // Final terrain collision (applying friction and fall checks)
    for (let i = 0; i < this.nodes.length; i++) {
      const n = this.nodes[i];
      const ground = terrain.getHeight(n.x, n.z);
      if (n.y < ground + n.radius) {
        n.y = ground + n.radius;
        
        // Grip/friction depends on whether it's a foot
        const isFoot = this.footIndices && this.footIndices.includes(i);
        const frictionCoeff = isFoot ? 0.85 : 0.15;
        n.ox = n.ox + (n.x - n.ox) * frictionCoeff;
        n.oz = n.oz + (n.z - n.oz) * frictionCoeff;
        
        // Kill downward velocity
        if (n.oy > n.y) n.oy = n.y;

        // If a critical fall node touches the ground (after 2s grace period), mark as fallen
        if (time > 2.0 && this.fallIndices && this.fallIndices.includes(i)) {
          this.isFallen = true;
        }
      }
    }

    // Track locomotion (Euclidean distance in XZ plane)
    const com = this.getCenterOfMass();
    this.distanceMoved = Math.hypot(com.x - this.startX, com.z - this.startZ);

    // Cooldown
    if (this.fireCooldown > 0) this.fireCooldown--;
  }

  /**
   * Average position of all nodes.
   * @returns {{ x: number, y: number, z: number }}
   */
  getCenterOfMass() {
    let sx = 0, sy = 0, sz = 0;
    for (let i = 0; i < this.nodes.length; i++) {
      sx += this.nodes[i].x;
      sy += this.nodes[i].y;
      sz += this.nodes[i].z;
    }
    const n = this.nodes.length;
    return { x: sx / n, y: sy / n, z: sz / n };
  }

  /**
   * Attempt to fire a projectile from the turret.
   * @param {Target[]} targets
   * @param {Terrain} terrain
   * @param {{ x: number, z: number }} wind
   * @returns {Projectile|null}
   */
  fire(targets, terrain, wind) {
    if (this.fireCooldown > 0 || this.shotsRemaining <= 0) return null;

    const t = this.turretNode;
    const speed = 5;

    // Direction from turret pitch/yaw
    const cosP = Math.cos(this.turretAnglePitch);
    const sinP = Math.sin(this.turretAnglePitch);
    const cosY = Math.cos(this.turretAngleYaw);
    const sinY = Math.sin(this.turretAngleYaw);

    const vx = cosP * sinY * speed;
    const vy = sinP * speed;
    const vz = cosP * cosY * speed;

    this.shotsRemaining--;
    this.fireCooldown = 30; // half-second at 60fps
    this.hasFired = true;

    return new Projectile(t.x, t.y, t.z, vx, vy, vz);
  }
}

/* ================================================================== */
/*  Projectile                                                        */
/* ================================================================== */

class Projectile {
  /**
   * @param {number} x  Start X
   * @param {number} y  Start Y
   * @param {number} z  Start Z
   * @param {number} vx Velocity X
   * @param {number} vy Velocity Y
   * @param {number} vz Velocity Z
   */
  constructor(x, y, z, vx, vy, vz) {
    this.x = x;
    this.y = y;
    this.z = z;
    this.vx = vx;
    this.vy = vy;
    this.vz = vz;

    /** Trail of past positions for arc rendering */
    this.trail = [{ x, y, z }];

    this.alive = true;
    this.age = 0;
  }

  /**
   * Advance projectile one step.
   * @param {number} gravity
   * @param {{ x: number, z: number }} wind
   * @param {Terrain} terrain
   * @param {Target[]} targets
   * @returns {{ hit: Target|null, impactPos: {x:number,y:number,z:number} }}
   */
  update(gravity, wind, terrain, targets) {
    if (!this.alive) return { hit: null, impactPos: null };

    this.vy -= gravity * 0.3;
    this.vx += wind.x * 0.002;
    this.vz += wind.z * 0.002;

    this.x += this.vx;
    this.y += this.vy;
    this.z += this.vz;
    this.age++;

    // Store trail (cap at 80 points)
    this.trail.push({ x: this.x, y: this.y, z: this.z });
    if (this.trail.length > 80) this.trail.shift();

    const impactPos = { x: this.x, y: this.y, z: this.z };

    // Terrain collision
    const ground = terrain.getHeight(this.x, this.z);
    if (this.y <= ground) {
      this.alive = false;
      impactPos.y = ground;
      return { hit: null, impactPos };
    }

    // Target collision
    for (let i = 0; i < targets.length; i++) {
      const tgt = targets[i];
      if (tgt.isDestroyed) continue;
      const dx = this.x - tgt.x;
      const dy = this.y - tgt.y;
      const dz = this.z - tgt.z;
      const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
      if (dist < tgt.radius) {
        this.alive = false;
        tgt.isDestroyed = true;
        return { hit: tgt, impactPos };
      }
    }

    // Timeout
    if (this.age > 300) {
      this.alive = false;
      return { hit: null, impactPos };
    }

    return { hit: null, impactPos: null };
  }
}

/* ================================================================== */
/*  Target                                                            */
/* ================================================================== */

class Target {
  /**
   * @param {number} x
   * @param {number} y
   * @param {number} z
   * @param {string} [type='static']  'static' | 'drone'
   */
  constructor(x, y, z, type = 'static') {
    this.x = x;
    this.y = y;
    this.z = z;
    this.baseX = x;
    this.baseY = y;
    this.baseZ = z;
    this.type = type;
    this.radius = type === 'drone' ? 10 : 15;
    this.isDestroyed = false;
    this.color = type === 'drone' ? '#FFD700' : '#FF6B35';

    /** Drone movement parameters */
    this.pathRadius = 40 + Math.random() * 40;
    this.pathSpeed = 0.5 + Math.random() * 0.5;
    this.pathPhase = Math.random() * Math.PI * 2;
    this.pathAxis = Math.random() > 0.5 ? 'xz' : 'xy';
  }

  /**
   * Update target (drones move, static targets stay put).
   * @param {number} time  Simulation clock
   */
  update(time) {
    if (this.isDestroyed) return;
    if (this.type !== 'drone') return;

    const angle = this.pathSpeed * time + this.pathPhase;
    if (this.pathAxis === 'xz') {
      this.x = this.baseX + Math.cos(angle) * this.pathRadius;
      this.z = this.baseZ + Math.sin(angle) * this.pathRadius;
    } else {
      this.x = this.baseX + Math.cos(angle) * this.pathRadius;
      this.y = this.baseY + Math.sin(angle) * this.pathRadius * 0.5;
    }
  }
}
