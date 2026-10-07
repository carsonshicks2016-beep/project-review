/**
 * ApexFlock - Apex Predator Agent
 * Co-Evolutionary Pack Hunter: Deep Policy Brain + Dive Kinematics + Cooperative Signaling
 */

import { NeuralNetwork } from './neuralNet.js';

let nextPredatorId = 1;

export class Predator {
  constructor(x = 0, y = 0, z = 0, brain = null, genome = null) {
    this.id = nextPredatorId++;
    this.type = 'predator';
    this.lineageId = genome?.lineageId || this.id;
    this.generation = genome?.generation || 1;

    // 3D Kinematics
    this.position = { x, y, z };
    const initialSpeed = 16 + Math.random() * 6;
    const theta = Math.random() * Math.PI * 2;
    const phi = (Math.random() - 0.5) * Math.PI * 0.4;
    this.velocity = {
      x: Math.cos(theta) * Math.cos(phi) * initialSpeed,
      y: Math.sin(phi) * initialSpeed,
      z: Math.sin(theta) * Math.cos(phi) * initialSpeed
    };
    this.acceleration = { x: 0, y: 0, z: 0 };

    // Orientation vectors
    this.forward = { x: 0, y: 0, z: 1 };
    this.up = { x: 0, y: 1, z: 0 };
    this.right = { x: 1, y: 0, z: 0 };
    this._updateOrientation();

    // Morphological Genome (Physical traits subject to evolution)
    // Organic predatory palette: Obsidian/Crimson (350), Burnished Bronze (28), Deep Plum (310), Smoked Amber (38)
    const predHues = [350, 28, 310, 38];
    const baseHue = predHues[Math.floor(Math.random() * predHues.length)];
    this.genome = genome || {
      lineageId: this.id,
      generation: 1,
      size: 1.6 + (Math.random() - 0.5) * 0.3,             // Visual & collision scale
      diveSpeed: 1.1 + (Math.random() - 0.5) * 0.3,         // Max sprint top speed
      jawReach: 5.5 + (Math.random() - 0.5) * 1.5,          // Strike catch radius
      pouncePower: 26.0 + (Math.random() - 0.5) * 6.0,      // Sprint acceleration torque
      packCooperation: 0.5 + (Math.random() - 0.5) * 0.4,   // Pack formation affinity
      visionRange: 110 + (Math.random() - 0.5) * 30,        // Sight radius
      staminaMax: 130 + (Math.random() - 0.5) * 30,
      staminaRecovery: 3.5 + (Math.random() - 0.5) * 1.0,
      hue: (baseHue + Math.floor((Math.random() - 0.5) * 15) + 360) % 360
    };

    // Pack hunting role
    const roles = ['alpha', 'flanker_left', 'flanker_right', 'ambush_diver'];
    this.role = genome?.role || roles[this.id % roles.length];

    // Energetic State & Hunting Stats
    this.energy = 110;
    this.stamina = this.genome.staminaMax;
    this.alive = true;
    this.age = 0;
    this.kills = 0;
    this.timeSinceLastKill = 0;
    this.isPouncing = false;
    this.howlSignal = 0.0;
    this.strikeCooldown = 0.0;

    // Streamline ribbon trail history
    this.trail = [];
    this.maxTrailLen = 14;

    // Neural Network: 26 Inputs -> 20 -> 14 -> 5 Outputs
    // Outputs: [0] Pitch, [1] Yaw, [2] Roll, [3] Pounce/Sprint Boost, [4] Pack Howl Signal
    this.brain = brain || new NeuralNetwork([26, 20, 14, 5]);

    this.lastInputs = new Float32Array(26);
    this.lastOutputs = new Float32Array(5);
  }

  _updateOrientation() {
    const speed = Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z);
    if (speed > 1e-4) {
      this.forward.x = this.velocity.x / speed;
      this.forward.y = this.velocity.y / speed;
      this.forward.z = this.velocity.z / speed;
    } else {
      this.forward = { x: 0, y: 0, z: 1 };
    }

    const worldUp = { x: 0, y: 1, z: 0 };
    let rx = this.forward.y * worldUp.z - this.forward.z * worldUp.y;
    let ry = this.forward.z * worldUp.x - this.forward.x * worldUp.z;
    let rz = this.forward.x * worldUp.y - this.forward.y * worldUp.x;
    const rLen = Math.hypot(rx, ry, rz);
    if (rLen > 1e-4) {
      this.right.x = rx / rLen;
      this.right.y = ry / rLen;
      this.right.z = rz / rLen;
    } else {
      this.right = { x: 1, y: 0, z: 0 };
    }

    this.up.x = this.right.y * this.forward.z - this.right.z * this.forward.y;
    this.up.y = this.right.z * this.forward.x - this.right.x * this.forward.z;
    this.up.z = this.right.x * this.forward.y - this.right.y * this.forward.x;
  }

  toLocalFrame(dx, dy, dz) {
    return {
      x: dx * this.right.x + dy * this.right.y + dz * this.right.z,
      y: dx * this.up.x + dy * this.up.y + dz * this.up.z,
      z: dx * this.forward.x + dy * this.forward.y + dz * this.forward.z
    };
  }

  sense(boids, packmates, bounds) {
    const inp = this.lastInputs;
    inp.fill(0);

    const vRange = this.genome.visionRange;
    const invRange = 1.0 / vRange;

    // 1. SENSE TARGET PREY (Nearest and Most Vulnerable)
    let nearestBoid = null;
    let nearestDist = Infinity;
    let vulnerableBoid = null;
    let lowestStamina = Infinity;

    let flockX = 0, flockY = 0, flockZ = 0;
    let boidCount = 0;

    for (let i = 0; i < boids.length; i++) {
      const b = boids[i];
      if (!b.alive) continue;
      const dx = b.position.x - this.position.x;
      const dy = b.position.y - this.position.y;
      const dz = b.position.z - this.position.z;
      const d2 = dx * dx + dy * dy + dz * dz;

      // Camouflage factor reduces detection range
      const effectiveRange = vRange * (1.0 - (b.genome?.camouflage || 0) * 0.4);

      if (d2 < effectiveRange * effectiveRange) {
        const d = Math.sqrt(d2);
        flockX += b.position.x;
        flockY += b.position.y;
        flockZ += b.position.z;
        boidCount++;

        if (d < nearestDist) {
          nearestDist = d;
          nearestBoid = b;
        }

        if (b.stamina < lowestStamina) {
          lowestStamina = b.stamina;
          vulnerableBoid = b;
        }
      }
    }

    if (nearestBoid) {
      const dx = nearestBoid.position.x - this.position.x;
      const dy = nearestBoid.position.y - this.position.y;
      const dz = nearestBoid.position.z - this.position.z;
      const local = this.toLocalFrame(dx, dy, dz);
      inp[0] = Math.max(-1, Math.min(1, local.x * invRange));
      inp[1] = Math.max(-1, Math.min(1, local.y * invRange));
      inp[2] = Math.max(-1, Math.min(1, local.z * invRange));
      inp[3] = Math.max(0, Math.min(1, nearestDist * invRange));

      // Closing speed
      const rvx = nearestBoid.velocity.x - this.velocity.x;
      const rvy = nearestBoid.velocity.y - this.velocity.y;
      const rvz = nearestBoid.velocity.z - this.velocity.z;
      if (nearestDist > 1e-3) {
        const closingRate = -(rvx * dx + rvy * dy + rvz * dz) / nearestDist;
        inp[4] = Math.max(-1, Math.min(1, closingRate / 30.0));
      }
    }

    if (vulnerableBoid && vulnerableBoid !== nearestBoid) {
      const dx = vulnerableBoid.position.x - this.position.x;
      const dy = vulnerableBoid.position.y - this.position.y;
      const dz = vulnerableBoid.position.z - this.position.z;
      const local = this.toLocalFrame(dx, dy, dz);
      inp[5] = Math.max(-1, Math.min(1, local.x * invRange));
      inp[6] = Math.max(-1, Math.min(1, local.y * invRange));
      inp[7] = Math.max(-1, Math.min(1, local.z * invRange));
      inp[8] = vulnerableBoid.stamina / (vulnerableBoid.genome?.staminaMax || 100);
    }

    if (boidCount > 0) {
      flockX /= boidCount;
      flockY /= boidCount;
      flockZ /= boidCount;
      const fdx = flockX - this.position.x;
      const fdy = flockY - this.position.y;
      const fdz = flockZ - this.position.z;
      const fLocal = this.toLocalFrame(fdx, fdy, fdz);
      inp[9] = Math.max(-1, Math.min(1, fLocal.x * invRange));
      inp[10] = Math.max(-1, Math.min(1, fLocal.y * invRange));
      inp[11] = Math.max(-1, Math.min(1, fLocal.z * invRange));
      inp[12] = Math.min(1.0, boidCount / 15.0); // flock cluster size
    }

    // 2. SENSE PACKMATES (Cooperative hunting)
    let nearestPack = null;
    let packDist = Infinity;
    for (let i = 0; i < packmates.length; i++) {
      const p = packmates[i];
      if (p === this || !p.alive) continue;
      const dx = p.position.x - this.position.x;
      const dy = p.position.y - this.position.y;
      const dz = p.position.z - this.position.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < packDist) {
        packDist = d2;
        nearestPack = p;
      }
    }

    if (nearestPack) {
      const d = Math.sqrt(packDist);
      const dx = nearestPack.position.x - this.position.x;
      const dy = nearestPack.position.y - this.position.y;
      const dz = nearestPack.position.z - this.position.z;
      const local = this.toLocalFrame(dx, dy, dz);
      inp[13] = Math.max(-1, Math.min(1, local.x * invRange));
      inp[14] = Math.max(-1, Math.min(1, local.y * invRange));
      inp[15] = Math.max(-1, Math.min(1, local.z * invRange));
      inp[16] = Math.max(0, Math.min(1, d * invRange));

      // Packmate heading alignment (dot product)
      const mySpeed = Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z);
      const packSpeed = Math.hypot(nearestPack.velocity.x, nearestPack.velocity.y, nearestPack.velocity.z);
      if (mySpeed > 0.1 && packSpeed > 0.1) {
        inp[17] = (this.velocity.x * nearestPack.velocity.x +
                   this.velocity.y * nearestPack.velocity.y +
                   this.velocity.z * nearestPack.velocity.z) / (mySpeed * packSpeed);
      }

      // Packmate howl / strike signal
      inp[18] = nearestPack.howlSignal;
      inp[19] = nearestPack.isPouncing ? 1.0 : 0.0;
    }

    // 3. WHISKER & ENVIRONMENT
    const lookahead = 50.0;
    const fx = this.position.x + this.forward.x * lookahead;
    const fy = this.position.y + this.forward.y * lookahead;
    const fz = this.position.z + this.forward.z * lookahead;
    let whiskerDanger = 0;
    if (bounds) {
      if (fx < bounds.minX || fx > bounds.maxX) whiskerDanger += 0.5;
      if (fy < bounds.minY || fy > bounds.maxY) whiskerDanger += 0.5;
      if (fz < bounds.minZ || fz > bounds.maxZ) whiskerDanger += 0.5;
    }
    inp[21] = Math.min(1.0, whiskerDanger);

    // 4. INTERNAL PHYSIOLOGY
    inp[22] = Math.max(0, 1.0 - (this.energy / 120.0)); // Hunger urgency
    inp[23] = this.stamina / this.genome.staminaMax;
    const currentSpeed = Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z);
    inp[24] = Math.min(1.0, currentSpeed / 36.0);
    inp[25] = Math.min(1.0, this.timeSinceLastKill / 35.0); // Starvation pressure

    return inp;
  }

  update(dt, boids, packmates, bounds) {
    if (!this.alive) return;

    this.age += dt;
    this.timeSinceLastKill += dt;
    if (this.strikeCooldown > 0) this.strikeCooldown -= dt;

    // Forward pass
    const inputs = this.sense(boids, packmates, bounds);
    const outputs = this.brain.forward(inputs);
    this.lastOutputs.set(outputs);

    // Outputs: [0] Pitch, [1] Yaw, [2] Roll, [3] Pounce/Sprint, [4] Howl
    const pitchCmd = outputs[0];
    const yawCmd = outputs[1];
    const rollCmd = outputs[2];
    const sprintCmd = outputs[3];
    const howlCmd = outputs[4];

    this.howlSignal = howlCmd;

    // Pounce / sprint mechanics
    let thrust = 16.0;
    if (sprintCmd > 0.45 && this.stamina > 8.0) {
      this.isPouncing = true;
      const sprintFactor = (sprintCmd - 0.45) * 2.0;
      thrust += sprintFactor * this.genome.pouncePower;
      this.stamina -= sprintFactor * 28.0 * dt;
    } else {
      this.isPouncing = false;
      this.stamina = Math.min(this.genome.staminaMax, this.stamina + this.genome.staminaRecovery * dt);
    }

    // Role-based tactical assistance
    if (this.role === 'ambush_diver' && this.forward.y < -0.15) {
      // Gravity dive acceleration
      thrust += 16.0 * Math.abs(this.forward.y);
      this.acceleration.y -= 18.0;
    }

    // Steering torque
    const steerPower = 32.0;
    this.acceleration.x = (this.right.x * yawCmd + this.up.x * pitchCmd) * steerPower + this.forward.x * thrust;
    this.acceleration.y = (this.right.y * yawCmd + this.up.y * pitchCmd) * steerPower + this.forward.y * thrust;
    this.acceleration.z = (this.right.z * yawCmd + this.up.z * pitchCmd) * steerPower + this.forward.z * thrust;

    // Boundary repulsion
    if (bounds) {
      const margin = 40.0;
      const repForce = 50.0;
      if (this.position.x < bounds.minX + margin) this.acceleration.x += (bounds.minX + margin - this.position.x) * repForce / margin;
      if (this.position.x > bounds.maxX - margin) this.acceleration.x -= (this.position.x - (bounds.maxX - margin)) * repForce / margin;
      if (this.position.y < bounds.minY + margin) this.acceleration.y += (bounds.minY + margin - this.position.y) * repForce / margin;
      if (this.position.y > bounds.maxY - margin) this.acceleration.y -= (this.position.y - (bounds.maxY - margin)) * repForce / margin;
      if (this.position.z < bounds.minZ + margin) this.acceleration.z += (bounds.minZ + margin - this.position.z) * repForce / margin;
      if (this.position.z > bounds.maxZ - margin) this.acceleration.z -= (this.position.z - (bounds.maxZ - margin)) * repForce / margin;
    }

    // Integrate Kinematics
    this.velocity.x += this.acceleration.x * dt;
    this.velocity.y += this.acceleration.y * dt;
    this.velocity.z += this.acceleration.z * dt;

    const speed = Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z);
    const maxSpeed = (28.0 * this.genome.diveSpeed) + (this.isPouncing ? 14.0 : 0.0);
    const minSpeed = 10.0;

    if (speed > maxSpeed) {
      const scale = maxSpeed / speed;
      this.velocity.x *= scale;
      this.velocity.y *= scale;
      this.velocity.z *= scale;
    } else if (speed < minSpeed && speed > 1e-4) {
      const scale = minSpeed / speed;
      this.velocity.x *= scale;
      this.velocity.y *= scale;
      this.velocity.z *= scale;
    }

    this.position.x += this.velocity.x * dt;
    this.position.y += this.velocity.y * dt;
    this.position.z += this.velocity.z * dt;

    this._updateOrientation();

    // Streamline trail recording
    this.trail.push({ x: this.position.x, y: this.position.y, z: this.position.z });
    if (this.trail.length > this.maxTrailLen) {
      this.trail.shift();
    }

    // Starvation & Metabolism
    const baseMetabolism = 1.8;
    const speedCost = (speed * speed) * 0.0012;
    this.energy -= (baseMetabolism + speedCost) * dt;

    if (this.energy <= 0 || this.timeSinceLastKill > 48.0) {
      this.alive = false; // Starvation death
    }
  }

  /**
   * Attempts to strike and eat nearby prey.
   * Returns true if a catch occurred.
   */
  attemptCatch(boid) {
    if (!this.alive || !boid.alive) return false;
    const dx = boid.position.x - this.position.x;
    const dy = boid.position.y - this.position.y;
    const dz = boid.position.z - this.position.z;
    const d2 = dx * dx + dy * dy + dz * dz;

    const reach = this.genome.jawReach;
    if (d2 < reach * reach) {
      // Caught!
      boid.alive = false;
      this.kills++;
      this.timeSinceLastKill = 0;
      this.energy = Math.min(200, this.energy + 70);
      this.strikeCooldown = 0.8;
      return true;
    }
    return false;
  }

  canReproduce() {
    return this.alive && this.energy >= 155 && this.kills >= 2;
  }

  reproduce(partner = null) {
    this.energy -= 75;
    let childBrain;
    let childGenome;

    if (partner) {
      childBrain = this.brain.crossover(partner.brain);
      childGenome = {
        lineageId: Math.random() < 0.5 ? this.lineageId : partner.lineageId,
        generation: Math.max(this.generation, partner.generation) + 1,
        size: (this.genome.size + partner.genome.size) * 0.5,
        diveSpeed: (this.genome.diveSpeed + partner.genome.diveSpeed) * 0.5,
        jawReach: (this.genome.jawReach + partner.genome.jawReach) * 0.5,
        pouncePower: (this.genome.pouncePower + partner.genome.pouncePower) * 0.5,
        packCooperation: (this.genome.packCooperation + partner.genome.packCooperation) * 0.5,
        visionRange: (this.genome.visionRange + partner.genome.visionRange) * 0.5,
        staminaMax: (this.genome.staminaMax + partner.genome.staminaMax) * 0.5,
        staminaRecovery: (this.genome.staminaRecovery + partner.genome.staminaRecovery) * 0.5,
        hue: Math.random() < 0.5 ? this.genome.hue : partner.genome.hue
      };
    } else {
      childBrain = this.brain.clone();
      childGenome = { ...this.genome, generation: this.generation + 1 };
    }

    childBrain.mutate(0.09, 0.35);
    childGenome.diveSpeed = Math.max(0.7, Math.min(1.5, childGenome.diveSpeed + (Math.random() - 0.5) * 0.1));
    childGenome.jawReach = Math.max(4.0, Math.min(9.0, childGenome.jawReach + (Math.random() - 0.5) * 0.5));
    childGenome.pouncePower = Math.max(16, Math.min(40, childGenome.pouncePower + (Math.random() - 0.5) * 4));
    childGenome.hue = (childGenome.hue + Math.floor((Math.random() - 0.5) * 15) + 360) % 360;

    const child = new Predator(
      this.position.x + (Math.random() - 0.5) * 8,
      this.position.y + (Math.random() - 0.5) * 8,
      this.position.z + (Math.random() - 0.5) * 8,
      childBrain,
      childGenome
    );
    child.energy = 85;
    return child;
  }
}
