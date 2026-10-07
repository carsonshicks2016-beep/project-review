/**
 * ApexFlock - Biosphere & Co-Evolutionary Ecosystem Manager
 * Handles multi-agent life cycles, 3D spatial partitioning,
 * chemical pheromone diffusion, food spore spawning, and ecological analytics.
 */

import { SpatialGrid } from './spatialGrid.js';
import { Boid } from './boid.js';
import { Predator } from './predator.js';

export class Ecosystem {
  constructor(options = {}) {
    this.bounds = {
      minX: -240, maxX: 240,
      minY: -100, maxY: 140,
      minZ: -240, maxZ: 240
    };

    this.spatialGrid = new SpatialGrid(this.bounds, 35);

    // Populations
    this.boids = [];
    this.predators = [];
    this.foods = [];
    this.maxFoods = options.maxFoods || 90;
    this.targetBoids = options.targetBoids || 120;
    this.targetPredators = options.targetPredators || 8;

    // Instinct vs Neural blend ratio (0 = pure neural, 1 = pure Reynolds instinct)
    this.instinctBlend = 0.25;

    // Environmental Pheromone Grid (low-res 3D field: 16 x 10 x 16)
    this._initPheromoneField();

    // Statistics & Telemetry
    this.time = 0;
    this.totalKills = 0;
    this.totalBirthsBoid = 0;
    this.totalBirthsPred = 0;
    this.history = [];
    this.historyTimer = 0;
    this.selectedAgent = null;

    // External turbulent wind vortices (God-Mode force fields)
    this.vortices = [];

    // Obstacles (crystalline pillars in biosphere)
    this.obstacles = [
      { x: 0, y: 0, z: 0, radius: 22, height: 180 },
      { x: -110, y: 0, z: -90, radius: 16, height: 160 },
      { x: 120, y: 0, z: 80, radius: 18, height: 170 },
      { x: -90, y: 0, z: 110, radius: 15, height: 150 },
      { x: 100, y: 0, z: -100, radius: 16, height: 160 }
    ];

    this.initPopulations();
  }

  _initPheromoneField() {
    this.pCols = 16;
    this.pRows = 10;
    this.pLayers = 16;
    this.pSize = this.pCols * this.pRows * this.pLayers;
    this.alarmPheromone = new Float32Array(this.pSize);
    this.pCellX = (this.bounds.maxX - this.bounds.minX) / this.pCols;
    this.pCellY = (this.bounds.maxY - this.bounds.minY) / this.pRows;
    this.pCellZ = (this.bounds.maxZ - this.bounds.minZ) / this.pLayers;
  }

  sampleAlarm(x, y, z) {
    const cx = Math.floor((x - this.bounds.minX) / this.pCellX);
    const cy = Math.floor((y - this.bounds.minY) / this.pCellY);
    const cz = Math.floor((z - this.bounds.minZ) / this.pCellZ);
    if (cx < 0 || cx >= this.pCols || cy < 0 || cy >= this.pRows || cz < 0 || cz >= this.pLayers) return 0;
    return this.alarmPheromone[cx + cy * this.pCols + cz * this.pCols * this.pRows];
  }

  emitAlarm(x, y, z, strength) {
    const cx = Math.floor((x - this.bounds.minX) / this.pCellX);
    const cy = Math.floor((y - this.bounds.minY) / this.pCellY);
    const cz = Math.floor((z - this.bounds.minZ) / this.pCellZ);
    if (cx < 0 || cx >= this.pCols || cy < 0 || cy >= this.pRows || cz < 0 || cz >= this.pLayers) return;
    const idx = cx + cy * this.pCols + cz * this.pCols * this.pRows;
    this.alarmPheromone[idx] = Math.min(2.0, this.alarmPheromone[idx] + strength * 0.4);
  }

  updatePheromones(dt) {
    const decay = Math.exp(-0.45 * dt); // Exponential dissipation
    for (let i = 0; i < this.pSize; i++) {
      this.alarmPheromone[i] *= decay;
    }
  }

  initPopulations() {
    this.boids = [];
    this.predators = [];
    this.foods = [];

    // Seed Prey
    for (let i = 0; i < this.targetBoids; i++) {
      const rx = (Math.random() - 0.5) * 260;
      const ry = (Math.random() - 0.5) * 120 + 20;
      const rz = (Math.random() - 0.5) * 260;
      this.boids.push(new Boid(rx, ry, rz));
    }

    // Seed Apex Predators
    for (let i = 0; i < this.targetPredators; i++) {
      const rx = (Math.random() - 0.5) * 320;
      const ry = (Math.random() - 0.5) * 140 + 20;
      const rz = (Math.random() - 0.5) * 320;
      this.predators.push(new Predator(rx, ry, rz));
    }

    // Seed Food Spores
    for (let i = 0; i < this.maxFoods; i++) {
      this.spawnFood();
    }
  }

  spawnFood(originX = null, originY = null, originZ = null) {
    let x, y, z;
    if (originX !== null) {
      // Clustered bloom
      x = originX + (Math.random() - 0.5) * 45;
      y = originY + (Math.random() - 0.5) * 30;
      z = originZ + (Math.random() - 0.5) * 45;
    } else {
      x = (Math.random() - 0.5) * (this.bounds.maxX - this.bounds.minX) * 0.85;
      y = (Math.random() - 0.5) * (this.bounds.maxY - this.bounds.minY) * 0.75 + 20;
      z = (Math.random() - 0.5) * (this.bounds.maxZ - this.bounds.minZ) * 0.85;
    }

    this.foods.push({
      id: Math.random(),
      x, y, z,
      active: true,
      energy: 40,
      radius: 2.2,
      pulsePhase: Math.random() * Math.PI * 2
    });
  }

  /**
   * Applies vortex wind physics to all entities.
   */
  applyVortices(dt) {
    for (let i = this.vortices.length - 1; i >= 0; i--) {
      const v = this.vortices[i];
      v.life -= dt;
      if (v.life <= 0) {
        this.vortices.splice(i, 1);
        continue;
      }

      const r2 = v.radius * v.radius;
      // Affect boids
      for (let j = 0; j < this.boids.length; j++) {
        const b = this.boids[j];
        const dx = b.position.x - v.x;
        const dy = b.position.y - v.y;
        const dz = b.position.z - v.z;
        const d2 = dx * dx + dz * dz;
        if (d2 < r2 && d2 > 1.0) {
          const d = Math.sqrt(d2);
          const force = v.strength * (1.0 - d / v.radius) * 45.0;
          // Swirling tangential force
          b.velocity.x += (-dz / d) * force * dt;
          b.velocity.z += (dx / d) * force * dt;
          b.velocity.y += (v.upDraft || 0) * dt;
        }
      }
    }
  }

  /**
   * Main simulation tick.
   * @param {number} dt - Timestep in seconds
   */
  update(dt) {
    this.time += dt;
    this.updatePheromones(dt);
    this.applyVortices(dt);

    // 1. REBUILD SPATIAL HASH GRID
    this.spatialGrid.clear();
    for (let i = 0; i < this.boids.length; i++) {
      const b = this.boids[i];
      if (b.alive) {
        this.spatialGrid.insert(b, b.position.x, b.position.y, b.position.z);
      }
    }
    for (let i = 0; i < this.predators.length; i++) {
      const p = this.predators[i];
      if (p.alive) {
        this.spatialGrid.insert(p, p.position.x, p.position.y, p.position.z);
      }
    }

    // 2. UPDATE BOIDS (Prey)
    const newBoids = [];
    const queryBuf = [];

    for (let i = 0; i < this.boids.length; i++) {
      const b = this.boids[i];
      if (!b.alive) continue;

      // Avoid crystalline obstacles
      for (let o = 0; o < this.obstacles.length; o++) {
        const obs = this.obstacles[o];
        const odx = b.position.x - obs.x;
        const odz = b.position.z - obs.z;
        const od2 = odx * odx + odz * odz;
        const safeR = obs.radius + 15;
        if (od2 < safeR * safeR && od2 > 0.1) {
          const od = Math.sqrt(od2);
          const push = (safeR - od) * 25.0;
          b.velocity.x += (odx / od) * push * dt;
          b.velocity.z += (odz / od) * push * dt;
        }
      }

      // Query neighbors within boid's vision range
      const nearby = this.spatialGrid.queryRadius(
        b.position.x, b.position.y, b.position.z,
        b.genome.visionRange * 1.5,
        queryBuf
      );

      const flockmates = [];
      const localPreds = [];
      for (let k = 0; k < nearby.length; k++) {
        const entity = nearby[k];
        if (entity.type === 'boid') flockmates.push(entity);
        else if (entity.type === 'predator') localPreds.push(entity);
      }

      // Check food consumption
      for (let f = 0; f < this.foods.length; f++) {
        const food = this.foods[f];
        if (!food.active) continue;
        const fdx = food.x - b.position.x;
        const fdy = food.y - b.position.y;
        const fdz = food.z - b.position.z;
        const fd2 = fdx * fdx + fdy * fdy + fdz * fdz;
        if (fd2 < 18.0) { // Caught food spore!
          food.active = false;
          b.feed(food.energy);
          break;
        }
      }

      b.update(dt, flockmates, localPreds, this.foods, this, this.bounds, this.instinctBlend);

      // Reproduction check
      if (b.canReproduce()) {
        const child = b.reproduce();
        newBoids.push(child);
        this.totalBirthsBoid++;
      }
    }

    // 3. UPDATE PREDATORS
    const newPredators = [];
    for (let i = 0; i < this.predators.length; i++) {
      const pred = this.predators[i];
      if (!pred.alive) continue;

      // Obstacle avoidance
      for (let o = 0; o < this.obstacles.length; o++) {
        const obs = this.obstacles[o];
        const odx = pred.position.x - obs.x;
        const odz = pred.position.z - obs.z;
        const od2 = odx * odx + odz * odz;
        const safeR = obs.radius + 18;
        if (od2 < safeR * safeR && od2 > 0.1) {
          const od = Math.sqrt(od2);
          const push = (safeR - od) * 35.0;
          pred.velocity.x += (odx / od) * push * dt;
          pred.velocity.z += (odz / od) * push * dt;
        }
      }

      const nearby = this.spatialGrid.queryRadius(
        pred.position.x, pred.position.y, pred.position.z,
        pred.genome.visionRange,
        queryBuf
      );

      const localBoids = [];
      const localPack = [];
      for (let k = 0; k < nearby.length; k++) {
        const entity = nearby[k];
        if (entity.type === 'boid') localBoids.push(entity);
        else if (entity.type === 'predator') localPack.push(entity);
      }

      // Hunt strike evaluation
      for (let b = 0; b < localBoids.length; b++) {
        const prey = localBoids[b];
        if (pred.attemptCatch(prey)) {
          this.totalKills++;
          // Cooperative hunting: distribute minor energy to adjacent packmates
          for (let p = 0; p < localPack.length; p++) {
            const mate = localPack[p];
            if (mate !== pred) {
              mate.energy = Math.min(180, mate.energy + 18);
            }
          }
          break;
        }
      }

      pred.update(dt, localBoids, localPack, this.bounds);

      // Predator reproduction
      if (pred.canReproduce()) {
        const child = pred.reproduce();
        newPredators.push(child);
        this.totalBirthsPred++;
      }
    }

    // 4. CLEANUP DEAD & INSERT OFFSPRING
    this.boids = this.boids.filter(b => b.alive);
    this.predators = this.predators.filter(p => p.alive);
    this.foods = this.foods.filter(f => f.active);

    for (let i = 0; i < newBoids.length; i++) {
      if (this.boids.length < 320) this.boids.push(newBoids[i]);
    }
    for (let i = 0; i < newPredators.length; i++) {
      if (this.predators.length < 35) this.predators.push(newPredators[i]);
    }

    // 5. AUTO IMMIGRATION / MINIMUM POPULATION REGULATION
    if (this.boids.length < 40) {
      // Spawn fresh boid immigrants to preserve evolutionary pool
      for (let i = 0; i < 15; i++) {
        this.boids.push(new Boid(
          (Math.random() - 0.5) * 200,
          (Math.random() - 0.5) * 80 + 20,
          (Math.random() - 0.5) * 200
        ));
      }
    }

    if (this.predators.length < 3) {
      // Spawn new apex challenger
      this.predators.push(new Predator(
        (Math.random() - 0.5) * 240,
        (Math.random() - 0.5) * 100 + 20,
        (Math.random() - 0.5) * 240
      ));
    }

    // Respawn food spores dynamically
    while (this.foods.length < this.maxFoods) {
      this.spawnFood();
    }

    // Animate food spore pulses
    for (let i = 0; i < this.foods.length; i++) {
      this.foods[i].pulsePhase += 3.0 * dt;
    }

    // 6. RECORD TELEMETRY FOR PHASE-SPACE PLOTS
    this.historyTimer += dt;
    if (this.historyTimer >= 0.5) {
      this.historyTimer = 0;
      this._recordSnapshot();
    }
  }

  _recordSnapshot() {
    let meanBoidSpeed = 0;
    let meanBoidStamina = 0;
    for (let i = 0; i < this.boids.length; i++) {
      const b = this.boids[i];
      meanBoidSpeed += Math.hypot(b.velocity.x, b.velocity.y, b.velocity.z);
      meanBoidStamina += b.stamina;
    }
    if (this.boids.length > 0) {
      meanBoidSpeed /= this.boids.length;
      meanBoidStamina /= this.boids.length;
    }

    let meanPredSpeed = 0;
    let meanPredEnergy = 0;
    for (let i = 0; i < this.predators.length; i++) {
      const p = this.predators[i];
      meanPredSpeed += Math.hypot(p.velocity.x, p.velocity.y, p.velocity.z);
      meanPredEnergy += p.energy;
    }
    if (this.predators.length > 0) {
      meanPredSpeed /= this.predators.length;
      meanPredEnergy /= this.predators.length;
    }

    // Calculate unique genetic lineages
    const lineages = new Set();
    for (let i = 0; i < this.boids.length; i++) lineages.add(this.boids[i].lineageId);

    const snapshot = {
      time: Math.round(this.time * 10) / 10,
      preyCount: this.boids.length,
      predCount: this.predators.length,
      foodCount: this.foods.length,
      meanBoidSpeed: Math.round(meanBoidSpeed * 10) / 10,
      meanPredSpeed: Math.round(meanPredSpeed * 10) / 10,
      meanBoidStamina: Math.round(meanBoidStamina),
      meanPredEnergy: Math.round(meanPredEnergy),
      totalKills: this.totalKills,
      lineages: lineages.size
    };

    this.history.push(snapshot);
    if (this.history.length > 120) {
      this.history.shift();
    }
  }

  /**
   * Interactive God-Mode: Trigger a mass mutation pulse.
   */
  triggerRadiationPulse(rate = 0.35, power = 0.8) {
    for (let i = 0; i < this.boids.length; i++) {
      this.boids[i].brain.mutate(rate, power);
      this.boids[i].genome.wingspan = Math.max(0.6, Math.min(1.5, this.boids[i].genome.wingspan + (Math.random() - 0.5) * 0.3));
      this.boids[i].genome.agility = Math.max(0.6, Math.min(1.6, this.boids[i].genome.agility + (Math.random() - 0.5) * 0.3));
    }
    for (let i = 0; i < this.predators.length; i++) {
      this.predators[i].brain.mutate(rate, power);
      this.predators[i].genome.diveSpeed = Math.max(0.7, Math.min(1.6, this.predators[i].genome.diveSpeed + (Math.random() - 0.5) * 0.3));
    }
  }

  /**
   * Interactive God-Mode: Creates a swirling wind vortex at (x, y, z).
   */
  stirWindVortex(x, y, z, strength = 1.8, radius = 90, duration = 6.0) {
    this.vortices.push({
      x, y, z,
      strength,
      radius,
      life: duration,
      upDraft: 15.0
    });
  }

  /**
   * Exports top champion brains as JSON.
   */
  exportChampions() {
    const bestBoid = [...this.boids].sort((a, b) => (b.age + b.evasions * 10) - (a.age + a.evasions * 10))[0];
    const bestPred = [...this.predators].sort((a, b) => (b.kills * 20 + b.age) - (a.kills * 20 + a.age))[0];
    return {
      bestBoid: bestBoid ? { brain: bestBoid.brain.toJSON(), genome: bestBoid.genome, stats: { age: bestBoid.age, evasions: bestBoid.evasions } } : null,
      bestPredator: bestPred ? { brain: bestPred.brain.toJSON(), genome: bestPred.genome, stats: { age: bestPred.age, kills: bestPred.kills } } : null
    };
  }
}
