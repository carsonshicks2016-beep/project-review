// Player Vehicle Physics & AI Traffic Simulation

import { ROAD_CONSTANTS } from './road.js';

export class PlayerCar {
  constructor() {
    this.reset();
  }

  reset() {
    this.x = 0;                     // -1 (left road edge) to 1 (right road edge)
    this.z = 0;                     // Position along circuit
    this.speed = 0;                 // Current velocity
    this.maxSpeed = 12500;          // Max forward speed (~250 km/h)
    this.accel = 3800;              // Acceleration rate
    this.braking = 8500;            // Footbrake deceleration
    this.handbraking = 12000;       // Handbrake deceleration
    this.decel = 1200;              // Natural rolling resistance
    this.offroadDecel = 6500;       // Grass friction drag
    this.offroadLimit = 3500;       // Max speed on grass
    
    // Steering & drift dynamics
    this.steer = 0;                 // Current steer input (-1 to 1)
    this.heading = 0;               // Visual tilt
    this.isDrifting = false;
    this.driftAngle = 0;
    this.driftScore = 0;
    this.driftMultiplier = 1;
    this.driftTimer = 0;
    this.smokeParticles = [];

    // Transmission & RPM
    this.gear = 1;
    this.rpm = 900;
    this.gears = [
      { minSpeed: 0, maxSpeed: 2800, ratio: 2.8 },    // 1st (~55 km/h)
      { minSpeed: 2200, maxSpeed: 5200, ratio: 1.9 }, // 2nd (~105 km/h)
      { minSpeed: 4500, maxSpeed: 7800, ratio: 1.4 }, // 3rd (~155 km/h)
      { minSpeed: 7000, maxSpeed: 10200, ratio: 1.1 },// 4th (~205 km/h)
      { minSpeed: 9500, maxSpeed: 13000, ratio: 0.85} // 5th (~260 km/h)
    ];

    // Camera view ('chase', 'cockpit', 'bumper')
    this.cameraMode = 'chase';
  }

  update(dt, input, currentSegment, totalTrackLength) {
    const isAccelerating = input.up;
    const isBraking = input.down;
    const isHandbraking = input.space;
    const isTurningLeft = input.left;
    const isTurningRight = input.right;

    const speedRatio = this.speed / this.maxSpeed;
    const isOnRoad = Math.abs(this.x) <= 1.0;

    // Steering input smoothing
    let targetSteer = 0;
    if (isTurningLeft) targetSteer -= 1;
    if (isTurningRight) targetSteer += 1;
    this.steer += (targetSteer - this.steer) * dt * 10;

    // Centrifugal curve force
    const curveEffect = currentSegment ? currentSegment.curve : 0;
    const centrifugalForce = curveEffect * (speedRatio * speedRatio) * 1.8;
    this.x -= centrifugalForce * dt;

    // Lateral steering displacement
    if (this.speed > 0) {
      const steerPower = (1.2 - speedRatio * 0.4); // More responsive at mid speeds
      this.x += this.steer * steerPower * dt * (this.speed / 2800);
    }

    // Drift Detection & Physics
    const turningHard = Math.abs(this.steer) > 0.45;
    const canInitiateDrift = this.speed > 3500 && turningHard && (isBraking || isHandbraking);
    
    if (canInitiateDrift || (this.isDrifting && this.speed > 2500 && turningHard)) {
      this.isDrifting = true;
      const driftDir = Math.sign(this.steer);
      this.driftAngle += (driftDir * 35 - this.driftAngle) * dt * 6;
      this.driftTimer += dt;
      this.driftMultiplier = Math.min(5.0, 1.0 + Math.floor(this.driftTimer * 1.5) * 0.5);
      this.driftScore += Math.floor(dt * 1500 * this.driftMultiplier);

      // Emit tire smoke
      if (Math.random() < 0.8) {
        this.smokeParticles.push({
          x: this.x + (Math.random() - 0.5) * 0.15,
          z: this.z - 50,
          y: (currentSegment ? currentSegment.p1.world.y : 0) + 15,
          size: 10 + Math.random() * 8,
          alpha: 0.85,
          life: 0.45
        });
      }
    } else {
      this.isDrifting = false;
      this.driftAngle *= 0.85;
      this.driftTimer = 0;
    }

    // Longitudinal Acceleration / Braking
    if (isAccelerating) {
      this.speed += this.accel * dt;
    } else if (isBraking) {
      this.speed -= this.braking * dt;
    } else if (isHandbraking) {
      this.speed -= this.handbraking * dt;
    } else {
      this.speed -= this.decel * dt;
    }

    // Offroad grass penalty
    if (!isOnRoad) {
      if (this.speed > this.offroadLimit) {
        this.speed -= this.offroadDecel * dt;
      }
      // Camera shake / bump
      this.heading += (Math.random() - 0.5) * 0.04;
    }

    // Boundary clamp
    this.speed = Math.max(0, Math.min(this.maxSpeed, this.speed));
    this.x = Math.max(-2.2, Math.min(2.2, this.x)); // Road verges barrier

    // Update distance along track
    this.z += this.speed * dt;
    if (this.z >= totalTrackLength) {
      this.z -= totalTrackLength;
    }

    // Transmission & Gear Calculation
    this.updateGears(dt, isAccelerating);

    // Update smoke particles
    for (let i = this.smokeParticles.length - 1; i >= 0; i--) {
      const p = this.smokeParticles[i];
      p.life -= dt;
      p.size += dt * 45;
      p.alpha = Math.max(0, p.life / 0.45);
      if (p.life <= 0) {
        this.smokeParticles.splice(i, 1);
      }
    }
  }

  updateGears(dt, isAccelerating) {
    // Determine gear from speed
    let targetGear = 1;
    for (let g = 0; g < this.gears.length; g++) {
      if (this.speed >= this.gears[g].minSpeed) {
        targetGear = g + 1;
      }
    }
    this.gear = targetGear;

    const currentGearInfo = this.gears[this.gear - 1];
    const gearSpeedRange = currentGearInfo.maxSpeed - currentGearInfo.minSpeed;
    const gearSpeedProgress = Math.max(0, (this.speed - currentGearInfo.minSpeed) / gearSpeedRange);

    // Compute RPM between 900 (idle) and 8200 (redline)
    if (this.speed < 100) {
      this.rpm = isAccelerating ? 2500 : 900 + Math.random() * 80;
    } else {
      let targetRpm = 1800 + gearSpeedProgress * 6200;
      if (targetRpm > 8000) {
        // Rev limiter stutter
        targetRpm = 8000 + (Math.random() > 0.5 ? -250 : 200);
      }
      this.rpm += (targetRpm - this.rpm) * dt * 15;
    }
  }

  getSpeedKmh() {
    // Scale speed unit to realistic ~0-260 km/h
    return Math.floor((this.speed / this.maxSpeed) * 260);
  }

  cycleCamera() {
    const modes = ['chase', 'cockpit', 'bumper'];
    const idx = modes.indexOf(this.cameraMode);
    this.cameraMode = modes[(idx + 1) % modes.length];
    return this.cameraMode;
  }
}

export class TrafficCar {
  constructor(z, lane, speed, spriteType, color) {
    this.z = z;
    this.x = lane; // -0.8 to 0.8
    this.targetX = lane;
    this.speed = speed;
    this.spriteType = spriteType; // 'car_sedan', 'car_sports', 'car_van'
    this.color = color;
    this.laneChangeTimer = 2 + Math.random() * 5;
  }

  update(dt, totalTrackLength) {
    this.z += this.speed * dt;
    if (this.z >= totalTrackLength) {
      this.z -= totalTrackLength;
    }

    // AI lane changing
    this.laneChangeTimer -= dt;
    if (this.laneChangeTimer <= 0) {
      this.laneChangeTimer = 4 + Math.random() * 6;
      const lanes = [-0.65, 0, 0.65];
      this.targetX = lanes[Math.floor(Math.random() * lanes.length)];
    }

    this.x += (this.targetX - this.x) * dt * 1.5;
  }
}

export class TrafficSystem {
  constructor(totalTrackLength, count = 24) {
    this.cars = [];
    this.totalTrackLength = totalTrackLength;
    this.initCars(count);
  }

  initCars(count) {
    this.cars = [];
    const types = ['car_sedan', 'car_sports', 'car_van'];
    const colors = ['#e63946', '#457b9d', '#2a9d8f', '#e9c46a', '#f4a261', '#e0e1dd'];
    const lanes = [-0.65, 0, 0.65];

    for (let i = 0; i < count; i++) {
      const z = 1000 + (i / count) * (this.totalTrackLength - 1200);
      const lane = lanes[i % lanes.length];
      const speed = 4000 + Math.random() * 5000; // 80 - 180 km/h
      const type = types[Math.floor(Math.random() * types.length)];
      const color = colors[Math.floor(Math.random() * colors.length)];
      this.cars.push(new TrafficCar(z, lane, speed, type, color));
    }
  }

  update(dt, player) {
    for (const car of this.cars) {
      car.update(dt, this.totalTrackLength);

      // Check collision with player
      const dz = Math.abs(car.z - player.z);
      const trackLoopDz = Math.min(dz, this.totalTrackLength - dz);
      if (trackLoopDz < 120) {
        const dx = Math.abs(car.x - player.x);
        if (dx < 0.45) {
          // Collision occurred!
          if (player.speed > car.speed) {
            player.speed = Math.max(1000, car.speed * 0.75);
            // Bump player sideways
            player.x += (player.x > car.x ? 0.35 : -0.35);
          }
        }
      }
    }
  }
}
