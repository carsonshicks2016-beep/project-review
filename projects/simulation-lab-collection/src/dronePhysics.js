export class DronePhysics {
  constructor(config = {}) {
    this.mass = config.mass !== undefined ? config.mass : 0.5;
    this.gravity = config.gravity !== undefined ? config.gravity : 9.81;
    this.maxThrust = config.maxThrust !== undefined ? config.maxThrust : 20.0;
    this.dragCoeff = config.dragCoeff !== undefined ? config.dragCoeff : 0.15;
    this.dragQuadratic = config.dragQuadratic !== undefined ? config.dragQuadratic : 0.01;
    this.maxRate = config.maxRate !== undefined ? config.maxRate : Math.PI * 3;
    this.maxYawRate = config.maxYawRate !== undefined ? config.maxYawRate : this.maxRate * 0.7;
    this.angularDamping = config.angularDamping !== undefined ? config.angularDamping : 12.0;
    this.getTerrainHeight = typeof config.getTerrainHeight === 'function' ? config.getTerrainHeight : null;

    this.pos = new Float32Array(3);
    this.vel = new Float32Array(3);
    this.quat = new Float32Array(4); // [w, x, y, z]
    this.angularVel = new Float32Array(3);
    this.alive = true;
    this.crashed = false;

    this.reset();
  }

  reset(startPos = [0, 1, 0], startYaw = 0) {
    this.pos[0] = startPos[0];
    this.pos[1] = startPos[1];
    this.pos[2] = startPos[2];

    this.vel.fill(0);
    this.angularVel.fill(0);

    this.quat = this.eulerToQuat(0, startYaw, 0);

    this.alive = true;
    this.crashed = false;
    this.hitWall = false;
  }

  step(controls, dt = 1/60) {
    if (!this.alive) return;

    // Clamp controls
    const thrustCmd = Math.max(0, Math.min(1, controls.thrust));
    const pitchCmd = Math.max(-1, Math.min(1, controls.pitch));
    const rollCmd = Math.max(-1, Math.min(1, controls.roll));
    const yawCmd = Math.max(-1, Math.min(1, controls.yaw));

    // Target angular rates
    const targetPitchRate = pitchCmd * this.maxRate;
    const targetRollRate = rollCmd * this.maxRate;
    const targetYawRate = yawCmd * this.maxYawRate;

    // Damped angular dynamics
    this.angularVel[0] += (targetPitchRate - this.angularVel[0]) * this.angularDamping * dt;
    this.angularVel[1] += (targetYawRate - this.angularVel[1]) * this.angularDamping * dt;
    this.angularVel[2] += (targetRollRate - this.angularVel[2]) * this.angularDamping * dt;

    // Integrate quaternion from angular velocity
    this.quat = this.integrateQuat(this.quat, this.angularVel, dt);

    // Compute thrust vector: rotate local [0, 1, 0] by quaternion
    const localUp = [0, 1, 0];
    const thrustDir = this.rotateVectorByQuat(localUp, this.quat);
    const thrustMag = thrustCmd * this.maxThrust;
    const thrustAcc = [
      (thrustDir[0] * thrustMag) / this.mass,
      (thrustDir[1] * thrustMag) / this.mass,
      (thrustDir[2] * thrustMag) / this.mass
    ];

    const gravAcc = [0, -this.gravity, 0];

    // Compute drag
    const speed = Math.sqrt(this.vel[0]**2 + this.vel[1]**2 + this.vel[2]**2);
    const dragAcc = [
      -(this.dragCoeff * this.vel[0] + this.dragQuadratic * speed * this.vel[0]) / this.mass,
      -(this.dragCoeff * this.vel[1] + this.dragQuadratic * speed * this.vel[1]) / this.mass,
      -(this.dragCoeff * this.vel[2] + this.dragQuadratic * speed * this.vel[2]) / this.mass
    ];

    // Compute acceleration
    const acc = [
      thrustAcc[0] + gravAcc[0] + dragAcc[0],
      thrustAcc[1] + gravAcc[1] + dragAcc[1],
      thrustAcc[2] + gravAcc[2] + dragAcc[2]
    ];

    // Semi-implicit Euler: update vel then pos
    this.vel[0] += acc[0] * dt;
    this.vel[1] += acc[1] * dt;
    this.vel[2] += acc[2] * dt;

    this.pos[0] += this.vel[0] * dt;
    this.pos[1] += this.vel[1] * dt;
    this.pos[2] += this.vel[2] * dt;

    // World collision against the scan. `worldGrid` is optional: leaving it unset keeps
    // the original behaviour where the point cloud is decorative and the drone flies
    // through it. Checked before the ground so a wall hit is attributed correctly.
    if (this.worldGrid && this.worldGrid.isSolid(this.pos[0], this.pos[1], this.pos[2])) {
      this.alive = false;
      this.crashed = true;
      this.hitWall = true;
      this.vel.fill(0);
      this.angularVel.fill(0);
      return;
    }

    // Ground / terrain collision
    const groundY = this.getTerrainHeight ? this.getTerrainHeight(this.pos[0], this.pos[2]) : 0.0;
    if (this.pos[1] <= groundY + 0.1) {
      this.pos[1] = groundY + 0.1;
      this.alive = false;
      this.crashed = true;
      this.vel.fill(0);
      this.angularVel.fill(0);
    }
  }

  worldToLocal(v) {
    const qInv = new Float32Array([this.quat[0], -this.quat[1], -this.quat[2], -this.quat[3]]);
    return this.rotateVectorByQuat(v, qInv);
  }

  getLocalUp() {
    return this.rotateVectorByQuat([0, 1, 0], this.quat);
  }

  // Helpers
  eulerToQuat(pitch, yaw, roll) {
    const cy = Math.cos(yaw * 0.5);
    const sy = Math.sin(yaw * 0.5);
    const cp = Math.cos(pitch * 0.5);
    const sp = Math.sin(pitch * 0.5);
    const cr = Math.cos(roll * 0.5);
    const sr = Math.sin(roll * 0.5);

    // This is the ZYX aerospace formula, which assumes Z-up: yaw lands in z, pitch in y,
    // roll in x. This simulator is Y-UP -- thrust is body [0,1,0], altitude is pos[1], and
    // angularVel maps [x,y,z] to [pitch,yaw,roll]. Under the original ordering a non-zero
    // start yaw rotated the drone about the world forward axis, so every episode began
    // rolled over (at the default start yaw of -2.54 rad, fully inverted). Permuted so
    // pitch->x, yaw->y, roll->z.
    return new Float32Array([
      cr * cp * cy + sr * sp * sy, // w
      cr * sp * cy + sr * cp * sy, // x = pitch
      cr * cp * sy - sr * sp * cy, // y = yaw
      sr * cp * cy - cr * sp * sy  // z = roll
    ]);
  }

  quatMultiply(a, b) {
    return new Float32Array([
      a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3], // w
      a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2], // x
      a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1], // y
      a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]  // z
    ]);
  }

  integrateQuat(q, omega, dt) {
    // Body-frame rates: q_dot = 0.5 * q (x) omega. The original used omega (x) q, which
    // integrates WORLD-frame rates -- so once the drone banked, a pitch command rotated it
    // about the world x axis instead of its own nose axis and control effectively inverted.
    const qOmega = new Float32Array([0, omega[0], omega[1], omega[2]]);
    const dq = this.quatMultiply(q, qOmega);

    let res = new Float32Array([
      q[0] + 0.5 * dt * dq[0],
      q[1] + 0.5 * dt * dq[1],
      q[2] + 0.5 * dt * dq[2],
      q[3] + 0.5 * dt * dq[3]
    ]);

    const mag = Math.sqrt(res[0]**2 + res[1]**2 + res[2]**2 + res[3]**2);
    if (mag > 0) {
      res[0] /= mag; res[1] /= mag; res[2] /= mag; res[3] /= mag;
    } else {
      res = new Float32Array([1, 0, 0, 0]);
    }
    return res;
  }

  rotateVectorByQuat(v, q) {
    const qInv = new Float32Array([q[0], -q[1], -q[2], -q[3]]);
    const qV = new Float32Array([0, v[0], v[1], v[2]]);
    const temp = this.quatMultiply(q, qV);
    const res = this.quatMultiply(temp, qInv);
    return [res[1], res[2], res[3]];
  }
}
