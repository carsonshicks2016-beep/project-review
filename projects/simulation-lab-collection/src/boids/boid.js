/**
 * ApexFlock - Prey (Boid) Agent
 * Deep Neural Network Brain + Aerodynamic 3D Flight Physics + Morphological Genetics
 */

import { NeuralNetwork } from './neuralNet.js';

let nextBoidId = 1;

export class Boid {
  constructor(x = 0, y = 0, z = 0, brain = null, genome = null) {
    this.id = nextBoidId++;
    this.type = 'boid';
    this.lineageId = genome?.lineageId || this.id;
    this.generation = genome?.generation || 1;

    // 3D Kinematics
    this.position = { x, y, z };
    const initialSpeed = 12 + Math.random() * 8;
    const theta = Math.random() * Math.PI * 2;
    const phi = (Math.random() - 0.5) * Math.PI * 0.5;
    this.velocity = {
      x: Math.cos(theta) * Math.cos(phi) * initialSpeed,
      y: Math.sin(phi) * initialSpeed,
      z: Math.sin(theta) * Math.cos(phi) * initialSpeed
    };
    this.acceleration = { x: 0, y: 0, z: 0 };
    
    // Orientation vectors (body frame)
    this.forward = { x: 0, y: 0, z: 1 };
    this.up = { x: 0, y: 1, z: 0 };
    this.right = { x: 1, y: 0, z: 0 };
    this._updateOrientation();

    // Morphological Genome (Physical traits subject to evolution)
    // Organic artistic palette: Pearl (190), Celadon (160), Champagne (42), Amethyst (280), Oyster (210)
    const organicHues = [42, 160, 190, 210, 280];
    const baseHue = organicHues[Math.floor(Math.random() * organicHues.length)];
    this.genome = genome || {
      lineageId: this.id,
      generation: 1,
      wingspan: 1.0 + (Math.random() - 0.5) * 0.3,          // 0.7 - 1.3: speed vs turn rate
      agility: 1.0 + (Math.random() - 0.5) * 0.3,           // Turn torque multiplier
      visionRange: 55 + (Math.random() - 0.5) * 20,         // Sensory sphere radius
      visionAngle: 4.2 + (Math.random() - 0.5) * 0.8,       // Radians (~240° field of view)
      staminaMax: 100 + (Math.random() - 0.5) * 30,
      staminaRecovery: 4.0 + (Math.random() - 0.5) * 1.5,
      camouflage: 0.2 + Math.random() * 0.4,                // Reduces predator detection distance
      hue: (baseHue + Math.floor((Math.random() - 0.5) * 20) + 360) % 360
    };

    // Metabolic & Energetic State
    this.energy = 85;
    this.stamina = this.genome.staminaMax;
    this.alive = true;
    this.age = 0;
    this.foodEaten = 0;
    this.evasions = 0;
    this.panicLevel = 0.0;
    this.wingFlapPhase = Math.random() * Math.PI * 2;
    this.bankingRoll = 0.0;
    this.isDrafting = false;
    this.draftingEfficiency = 0.0;

    // Position history for silky streamline ribbon trails
    this.trail = [];
    this.maxTrailLen = 10;

    // Neural Network Brain: 26 Sensory Inputs -> 20 -> 14 -> 5 Motor Outputs
    // Inputs:
    // [0..3]:   Nearest flockmate (local X, Y, Z, dist)
    // [4]:      Flockmate alignment match
    // [5..8]:   Flock center of mass (local X, Y, Z, density)
    // [9..13]:  Nearest predator (local X, Y, Z, invDist, closingSpeed)
    // [14..17]: 2nd nearest predator (local X, Y, Z, invDist)
    // [18..21]: Nearest food spore (local X, Y, Z, dist)
    // [22]:     Alarm pheromone intensity
    // [23]:     Forward whisker obstacle depth
    // [24]:     Stamina ratio (0..1)
    // [25]:     Speed ratio (0..1)
    // Outputs:
    // [0]: Pitch (-1..1), [1]: Yaw (-1..1), [2]: Roll (-1..1)
    // [3]: Boost / Sprint (0..1), [4]: Emit Alarm Pheromone (0..1)
    this.brain = brain || new NeuralNetwork([26, 20, 14, 5]);

    // Cached sensory buffer for real-time telemetry inspection
    this.lastInputs = new Float32Array(26);
    this.lastOutputs = new Float32Array(5);

    // Instinct vs Neural blend ratio (0 = pure neural, 1 = pure Reynolds instinct)
    this.instinctWeight = 0.25;
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

    // World Up approx
    const worldUp = { x: 0, y: 1, z: 0 };
    // Right = Forward x WorldUp
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

    // Up = Right x Forward
    this.up.x = this.right.y * this.forward.z - this.right.z * this.forward.y;
    this.up.y = this.right.z * this.forward.x - this.right.x * this.forward.z;
    this.up.z = this.right.x * this.forward.y - this.right.y * this.forward.x;
  }

  /**
   * Transforms a world relative displacement vector into this boid's local body frame.
   */
  toLocalFrame(dx, dy, dz) {
    return {
      x: dx * this.right.x + dy * this.right.y + dz * this.right.z,     // Lateral (Right/Left)
      y: dx * this.up.x + dy * this.up.y + dz * this.up.z,             // Vertical (Up/Down)
      z: dx * this.forward.x + dy * this.forward.y + dz * this.forward.z // Longitudinal (Forward/Back)
    };
  }

  /**
   * Senses environment and populates input vector for the neural network.
   */
  sense(flockmates, predators, foods, pheromoneField, bounds) {
    const inp = this.lastInputs;
    inp.fill(0);

    const vRange = this.genome.visionRange;
    const vRangeInv = 1.0 / vRange;

    // 1. SENSE FLOCKMATES
    let nearestFlockDist = Infinity;
    let nearestFlock = null;
    let flockCenterX = 0, flockCenterY = 0, flockCenterZ = 0;
    let flockCount = 0;

    for (let i = 0; i < flockmates.length; i++) {
      const other = flockmates[i];
      if (other === this || !other.alive) continue;
      const dx = other.position.x - this.position.x;
      const dy = other.position.y - this.position.y;
      const dz = other.position.z - this.position.z;
      const d2 = dx * dx + dy * dy + dz * dz;

      if (d2 < vRange * vRange) {
        const d = Math.sqrt(d2);
        flockCenterX += other.position.x;
        flockCenterY += other.position.y;
        flockCenterZ += other.position.z;
        flockCount++;

        if (d < nearestFlockDist) {
          nearestFlockDist = d;
          nearestFlock = other;
        }
      }
    }

    if (nearestFlock) {
      const dx = nearestFlock.position.x - this.position.x;
      const dy = nearestFlock.position.y - this.position.y;
      const dz = nearestFlock.position.z - this.position.z;
      const local = this.toLocalFrame(dx, dy, dz);
      inp[0] = Math.max(-1, Math.min(1, local.x * vRangeInv));
      inp[1] = Math.max(-1, Math.min(1, local.y * vRangeInv));
      inp[2] = Math.max(-1, Math.min(1, local.z * vRangeInv));
      inp[3] = Math.max(0, Math.min(1, nearestFlockDist * vRangeInv));

      // Alignment match
      const speed = Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z);
      const otherSpeed = Math.hypot(nearestFlock.velocity.x, nearestFlock.velocity.y, nearestFlock.velocity.z);
      if (speed > 0.1 && otherSpeed > 0.1) {
        inp[4] = (this.velocity.x * nearestFlock.velocity.x + 
                  this.velocity.y * nearestFlock.velocity.y + 
                  this.velocity.z * nearestFlock.velocity.z) / (speed * otherSpeed);
      }
    }

    if (flockCount > 0) {
      flockCenterX /= flockCount;
      flockCenterY /= flockCount;
      flockCenterZ /= flockCount;
      const fdx = flockCenterX - this.position.x;
      const fdy = flockCenterY - this.position.y;
      const fdz = flockCenterZ - this.position.z;
      const fLocal = this.toLocalFrame(fdx, fdy, fdz);
      inp[5] = Math.max(-1, Math.min(1, fLocal.x * vRangeInv));
      inp[6] = Math.max(-1, Math.min(1, fLocal.y * vRangeInv));
      inp[7] = Math.max(-1, Math.min(1, fLocal.z * vRangeInv));
      inp[8] = Math.min(1, flockCount / 12.0); // local density
    }

    // 2. SENSE PREDATORS (Nearest & 2nd Nearest)
    let p1 = null, p2 = null;
    let d1 = Infinity, d2 = Infinity;
    const predRange = vRange * 1.5;

    for (let i = 0; i < predators.length; i++) {
      const pred = predators[i];
      if (!pred.alive) continue;
      const dx = pred.position.x - this.position.x;
      const dy = pred.position.y - this.position.y;
      const dz = pred.position.z - this.position.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist < predRange) {
        if (dist < d1) {
          d2 = d1; p2 = p1;
          d1 = dist; p1 = pred;
        } else if (dist < d2) {
          d2 = dist; p2 = pred;
        }
      }
    }

    if (p1) {
      const dx = p1.position.x - this.position.x;
      const dy = p1.position.y - this.position.y;
      const dz = p1.position.z - this.position.z;
      const local = this.toLocalFrame(dx, dy, dz);
      inp[9] = Math.max(-1, Math.min(1, local.x / predRange));
      inp[10] = Math.max(-1, Math.min(1, local.y / predRange));
      inp[11] = Math.max(-1, Math.min(1, local.z / predRange));
      inp[12] = 1.0 - Math.min(1.0, d1 / predRange); // Closer = higher urgency

      // Closing speed
      const rvx = p1.velocity.x - this.velocity.x;
      const rvy = p1.velocity.y - this.velocity.y;
      const rvz = p1.velocity.z - this.velocity.z;
      if (d1 > 1e-3) {
        const closingSpeed = -(rvx * dx + rvy * dy + rvz * dz) / d1;
        inp[13] = Math.max(-1, Math.min(1, closingSpeed / 25.0));
      }
      this.panicLevel = Math.max(this.panicLevel, inp[12]);
    } else {
      this.panicLevel *= 0.95;
    }

    if (p2) {
      const dx = p2.position.x - this.position.x;
      const dy = p2.position.y - this.position.y;
      const dz = p2.position.z - this.position.z;
      const local = this.toLocalFrame(dx, dy, dz);
      inp[14] = Math.max(-1, Math.min(1, local.x / predRange));
      inp[15] = Math.max(-1, Math.min(1, local.y / predRange));
      inp[16] = Math.max(-1, Math.min(1, local.z / predRange));
      inp[17] = 1.0 - Math.min(1.0, d2 / predRange);
    }

    // 3. SENSE NEAREST FOOD SPORE
    let nearestFoodDist = Infinity;
    let nearestFood = null;
    for (let i = 0; i < foods.length; i++) {
      const f = foods[i];
      if (!f.active) continue;
      const dx = f.x - this.position.x;
      const dy = f.y - this.position.y;
      const dz = f.z - this.position.z;
      const d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < nearestFoodDist) {
        nearestFoodDist = d2;
        nearestFood = f;
      }
    }

    if (nearestFood) {
      const dist = Math.sqrt(nearestFoodDist);
      const dx = nearestFood.x - this.position.x;
      const dy = nearestFood.y - this.position.y;
      const dz = nearestFood.z - this.position.z;
      const local = this.toLocalFrame(dx, dy, dz);
      const foodRange = vRange * 2.0;
      inp[18] = Math.max(-1, Math.min(1, local.x / foodRange));
      inp[19] = Math.max(-1, Math.min(1, local.y / foodRange));
      inp[20] = Math.max(-1, Math.min(1, local.z / foodRange));
      inp[21] = Math.max(0, Math.min(1, dist / foodRange));
    }

    // 4. PHEROMONE & WHISKER DEPTH
    if (pheromoneField) {
      inp[22] = Math.min(1.0, pheromoneField.sampleAlarm(this.position.x, this.position.y, this.position.z));
    }

    // Forward boundary proximity whisker
    const lookahead = 40.0;
    const fx = this.position.x + this.forward.x * lookahead;
    const fy = this.position.y + this.forward.y * lookahead;
    const fz = this.position.z + this.forward.z * lookahead;
    let whiskerDanger = 0;
    if (bounds) {
      if (fx < bounds.minX || fx > bounds.maxX) whiskerDanger += 0.5;
      if (fy < bounds.minY || fy > bounds.maxY) whiskerDanger += 0.5;
      if (fz < bounds.minZ || fz > bounds.maxZ) whiskerDanger += 0.5;
    }
    inp[23] = Math.min(1.0, whiskerDanger);

    // Internal physiological stats
    inp[24] = this.stamina / this.genome.staminaMax;
    const currentSpeed = Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z);
    inp[25] = Math.min(1.0, currentSpeed / 30.0);

    return inp;
  }

  /**
   * Updates flight physics, metabolic energy, neural steering, and boundary avoidance.
   */
  update(dt, flockmates, predators, foods, pheromoneField, bounds, instinctBlend = 0.25) {
    if (!this.alive) return;

    this.age += dt;
    this.instinctWeight = instinctBlend;

    // 1. SENSE & FORWARD PASS
    const inputs = this.sense(flockmates, predators, foods, pheromoneField, bounds);
    const outputs = this.brain.forward(inputs);
    this.lastOutputs.set(outputs);

    // Motor interpretation:
    // [0]: Pitch (-1..1), [1]: Yaw (-1..1), [2]: Roll (-1..1)
    // [3]: Sprint Boost (0..1), [4]: Alarm Pheromone (0..1)
    const pitchCmd = outputs[0];
    const yawCmd = outputs[1];
    const rollCmd = outputs[2];
    const boostCmd = outputs[3];
    const alarmCmd = outputs[4];

    // Check emergency sprint
    let thrust = 14.0 * (1.0 / this.genome.wingspan);
    if (boostCmd > 0.5 && this.stamina > 5.0) {
      const boostAmount = (boostCmd - 0.5) * 2.0;
      thrust += boostAmount * 16.0;
      this.stamina -= boostAmount * 22.0 * dt;
    } else {
      this.stamina = Math.min(this.genome.staminaMax, this.stamina + this.genome.staminaRecovery * dt);
    }

    // Alarm pheromone emission
    if (alarmCmd > 0.65 && pheromoneField) {
      pheromoneField.emitAlarm(this.position.x, this.position.y, this.position.z, alarmCmd);
    }

    // Neural steering torque vector in world frame
    const agility = this.genome.agility * 28.0;
    const neuralSteerX = (this.right.x * yawCmd + this.up.x * pitchCmd) * agility;
    const neuralSteerY = (this.right.y * yawCmd + this.up.y * pitchCmd) * agility;
    const neuralSteerZ = (this.right.z * yawCmd + this.up.z * pitchCmd) * agility;

    // Optional instinctual Reynolds baseline (Separation, Cohesion, Alignment)
    let instinctX = 0, instinctY = 0, instinctZ = 0;
    if (this.instinctWeight > 0.01) {
      // Separation from nearby boids
      for (let i = 0; i < flockmates.length; i++) {
        const other = flockmates[i];
        if (other === this || !other.alive) continue;
        const dx = this.position.x - other.position.x;
        const dy = this.position.y - other.position.y;
        const dz = this.position.z - other.position.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 64.0 && d2 > 1e-4) { // < 8 units
          const invD = 1.0 / Math.sqrt(d2);
          instinctX += (dx * invD) * 18.0;
          instinctY += (dy * invD) * 18.0;
          instinctZ += (dz * invD) * 18.0;
        }
      }

      // Evade predators directly
      for (let i = 0; i < predators.length; i++) {
        const pred = predators[i];
        if (!pred.alive) continue;
        const dx = this.position.x - pred.position.x;
        const dy = this.position.y - pred.position.y;
        const dz = this.position.z - pred.position.z;
        const d2 = dx * dx + dy * dy + dz * dz;
        if (d2 < 3600.0 && d2 > 1e-4) { // < 60 units
          const invD = 1.0 / Math.sqrt(d2);
          const panicMult = 45.0 * (1.0 - Math.sqrt(d2) / 60.0);
          instinctX += (dx * invD) * panicMult;
          instinctY += (dy * invD) * panicMult;
          instinctZ += (dz * invD) * panicMult;
        }
      }
    }

    // 3. AERODYNAMIC DRAFTING & COLLECTIVE MANEUVERS
    this.isDrafting = false;
    this.draftingEfficiency = 0.0;
    const myWingspan = this.genome.wingspan * 3.0;

    for (let i = 0; i < flockmates.length; i++) {
      const leader = flockmates[i];
      if (leader === this || !leader.alive) continue;
      const dx = leader.position.x - this.position.x;
      const dy = leader.position.y - this.position.y;
      const dz = leader.position.z - this.position.z;
      const local = this.toLocalFrame(dx, dy, dz);

      // Leader must be ahead (local.z > 0)
      if (local.z > 5.0 && local.z < 22.0 && Math.abs(local.y) < 4.0) {
        const latDist = Math.abs(local.x);
        // Upwash sweet-spot outside wingtip: ~ 0.8 to 1.6 x wingspan
        if (latDist >= myWingspan * 0.7 && latDist <= myWingspan * 1.6) {
          this.isDrafting = true;
          this.draftingEfficiency = 0.32; // 32% energy savings!
          // Aerodynamic suction pulling boid smoothly into the V-formation slot
          this.acceleration.x += (leader.forward.x * 4.0 + this.right.x * (local.x > 0 ? 1 : -1) * 2.0);
          this.acceleration.y += leader.forward.y * 3.0;
          this.acceleration.z += leader.forward.z * 4.0;
          break;
        } else if (latDist < myWingspan * 0.5) {
          // Direct downwash turbulence: push outward away from vortex centerline
          const pushDir = local.x >= 0 ? 1 : -1;
          this.acceleration.x += this.right.x * pushDir * 6.0;
        }
      }
    }

    // Emergent Fountain / Bifurcation Evasion Maneuver
    for (let i = 0; i < predators.length; i++) {
      const pred = predators[i];
      if (!pred.alive) continue;
      const dx = pred.position.x - this.position.x;
      const dy = pred.position.y - this.position.y;
      const dz = pred.position.z - this.position.z;
      const dist = Math.hypot(dx, dy, dz);
      if (dist < 45.0) {
        const local = this.toLocalFrame(dx, dy, dz);
        // If predator is directly along flight axis (charging), peel off perpendicular
        const sideSign = (local.x >= 0) ? -1 : 1;
        const urgency = (1.0 - dist / 45.0);
        this.acceleration.x += (this.right.x * sideSign * 35.0 + this.up.x * 12.0) * urgency;
        this.acceleration.y += (this.right.y * sideSign * 35.0 + this.up.y * 12.0) * urgency;
        this.acceleration.z += (this.right.z * sideSign * 35.0 + this.up.z * 12.0) * urgency;
      }
    }

    // Blend neural and instinct
    const wN = 1.0 - this.instinctWeight;
    const wI = this.instinctWeight;
    this.acceleration.x += neuralSteerX * wN + instinctX * wI + this.forward.x * thrust;
    this.acceleration.y += neuralSteerY * wN + instinctY * wI + this.forward.y * thrust;
    this.acceleration.z += neuralSteerZ * wN + instinctZ * wI + this.forward.z * thrust;

    // Soft Boundary Repulsion
    if (bounds) {
      const margin = 35.0;
      const repForce = 40.0;
      if (this.position.x < bounds.minX + margin) this.acceleration.x += (bounds.minX + margin - this.position.x) * repForce / margin;
      if (this.position.x > bounds.maxX - margin) this.acceleration.x -= (this.position.x - (bounds.maxX - margin)) * repForce / margin;
      if (this.position.y < bounds.minY + margin) this.acceleration.y += (bounds.minY + margin - this.position.y) * repForce / margin;
      if (this.position.y > bounds.maxY - margin) this.acceleration.y -= (this.position.y - (bounds.maxY - margin)) * repForce / margin;
      if (this.position.z < bounds.minZ + margin) this.acceleration.z += (bounds.minZ + margin - this.position.z) * repForce / margin;
      if (this.position.z > bounds.maxZ - margin) this.acceleration.z -= (this.position.z - (bounds.maxZ - margin)) * repForce / margin;
    }

    // Integrate Kinematics (Euler)
    this.velocity.x += this.acceleration.x * dt;
    this.velocity.y += this.acceleration.y * dt;
    this.velocity.z += this.acceleration.z * dt;

    // Aerodynamic Drag & Speed Clamping
    const speed = Math.hypot(this.velocity.x, this.velocity.y, this.velocity.z);
    const maxSpeed = 26.0 * this.genome.wingspan + (boostCmd > 0.5 ? 12.0 : 0.0);
    const minSpeed = 8.0;

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

    // Banking roll effect
    this.bankingRoll += (rollCmd * 0.8 - this.bankingRoll) * 5.0 * dt;
    this._updateOrientation();

    // Wing flap animation
    const flapFreq = (speed / 10.0) * 14.0;
    this.wingFlapPhase += flapFreq * dt;

    // Streamline ribbon trail history
    this.trail.push({ x: this.position.x, y: this.position.y, z: this.position.z });
    if (this.trail.length > this.maxTrailLen) {
      this.trail.shift();
    }

    // Metabolism & Energy with Drafting Discount
    const baseMetabolism = 1.2;
    const speedCost = (speed * speed) * 0.001;
    const energyDiscount = this.isDrafting ? (1.0 - this.draftingEfficiency) : 1.0;
    this.energy -= (baseMetabolism + speedCost) * energyDiscount * dt;

    if (this.energy <= 0) {
      this.alive = false;
    }
  }

  /**
   * Feeds on an energy spore.
   */
  feed(energyAmount = 35) {
    this.energy = Math.min(180, this.energy + energyAmount);
    this.foodEaten++;
  }

  /**
   * Checks if ready to reproduce.
   */
  canReproduce() {
    return this.alive && this.energy >= 140 && this.age > 8.0;
  }

  /**
   * Reproduces a child boid via mutation and inherited genome.
   */
  reproduce(partner = null) {
    this.energy -= 65; // High reproductive energetic cost
    let childBrain;
    let childGenome;

    if (partner) {
      childBrain = this.brain.crossover(partner.brain);
      childGenome = {
        lineageId: Math.random() < 0.5 ? this.lineageId : partner.lineageId,
        generation: Math.max(this.generation, partner.generation) + 1,
        wingspan: (this.genome.wingspan + partner.genome.wingspan) * 0.5,
        agility: (this.genome.agility + partner.genome.agility) * 0.5,
        visionRange: (this.genome.visionRange + partner.genome.visionRange) * 0.5,
        visionAngle: (this.genome.visionAngle + partner.genome.visionAngle) * 0.5,
        staminaMax: (this.genome.staminaMax + partner.genome.staminaMax) * 0.5,
        staminaRecovery: (this.genome.staminaRecovery + partner.genome.staminaRecovery) * 0.5,
        camouflage: (this.genome.camouflage + partner.genome.camouflage) * 0.5,
        hue: Math.random() < 0.5 ? this.genome.hue : partner.genome.hue
      };
    } else {
      childBrain = this.brain.clone();
      childGenome = { ...this.genome, generation: this.generation + 1 };
    }

    // Apply genetic mutation
    childBrain.mutate(0.09, 0.35);
    childGenome.wingspan = Math.max(0.6, Math.min(1.5, childGenome.wingspan + (Math.random() - 0.5) * 0.1));
    childGenome.agility = Math.max(0.6, Math.min(1.6, childGenome.agility + (Math.random() - 0.5) * 0.1));
    childGenome.visionRange = Math.max(30, Math.min(90, childGenome.visionRange + (Math.random() - 0.5) * 6));
    childGenome.staminaMax = Math.max(70, Math.min(160, childGenome.staminaMax + (Math.random() - 0.5) * 10));
    childGenome.camouflage = Math.max(0.05, Math.min(0.75, childGenome.camouflage + (Math.random() - 0.5) * 0.05));
    childGenome.hue = (childGenome.hue + Math.floor((Math.random() - 0.5) * 12) + 360) % 360;

    const child = new Boid(
      this.position.x + (Math.random() - 0.5) * 6,
      this.position.y + (Math.random() - 0.5) * 6,
      this.position.z + (Math.random() - 0.5) * 6,
      childBrain,
      childGenome
    );
    child.energy = 60;
    return child;
  }
}
