import { DronePhysics } from '../dronePhysics.js';
import { checkGatePassing } from '../gateCollision.js';
import { clipAction } from './policy.js';
import { OBS_DIM as SHARED_OBS_DIM, buildObservation } from '../observation.js';

// Dense per-step reward for the drone task.
//
// The GA optimised a single episodic scalar. PPO needs reward at every tick, which is
// also a chance to fix what that scalar got wrong: there is no payment for merely staying
// airborne (that bug taught the GA to crawl), and time costs something every tick, so
// speed is rewarded continuously rather than through one small end-of-episode bonus.
export const REWARD = {
  progress: 1.0,      // Per metre of closure on the current gate -- the workhorse term
  gatePass: 10.0,
  completion: 50.0,
  crash: -10.0,
  timeCost: -0.01,    // Per tick: ~-36 over a full 3600-tick episode vs +160 for 16 gates
  actionCost: -0.02,  // Discourages the buzzing the GA's jitter penalty targeted
  progressClamp: 2.0  // Guards against a teleport-sized delta on gate re-basing
};

export const OBS_DIM = SHARED_OBS_DIM;
export const ACT_DIM = 4;

export class DroneEnv {
  constructor(gates, { maxTicks = 3600, dt = 1 / 60, startGateIdx = 0, getTerrainHeight = null, grid = null } = {}) {
    this.gates = gates;
    this.maxTicks = maxTicks;
    this.dt = dt;
    this.startGateIdx = startGateIdx;
    this.getTerrainHeight = typeof getTerrainHeight === 'function' ? getTerrainHeight : null;
    this.grid = grid;
    this.drone = new DronePhysics({ getTerrainHeight: this.getTerrainHeight });
    this.drone.worldGrid = grid;
    this.obs = new Float32Array(OBS_DIM);
    this._prevAction = new Float32Array(ACT_DIM);
    this._prevPos = [0, 0, 0];
    this.reset();
  }

  static startPose(gates, gateIdx) {
    const g = gates[Math.max(0, Math.min(gateIdx, gates.length - 1))];
    const n = g.normal || [0, 0, 1];
    return {
      pos: [g.position[0] - n[0] * 15, g.position[1], g.position[2] - n[2] * 15],
      yaw: Math.atan2(-n[0], -n[2])
    };
  }

  reset(startGateIdx = this.startGateIdx) {
    this.startGateIdx = startGateIdx;
    const { pos, yaw } = DroneEnv.startPose(this.gates, startGateIdx);
    this.drone.reset(pos, yaw);
    this.gateIdx = startGateIdx;
    this.tick = 0;
    this.gatesCleared = 0;
    this.topSpeed = 0;
    this._prevAction.fill(0);
    this.prevDist = this._distToGate(this.gateIdx);
    return this._observe();
  }

  _distToGate(i) {
    const g = this.gates[Math.min(i, this.gates.length - 1)];
    const p = this.drone.pos;
    return Math.hypot(p[0] - g.position[0], p[1] - g.position[1], p[2] - g.position[2]);
  }

  _observe() {
    return buildObservation(this.drone, this.gates, this.gateIdx, this.grid, this.obs, this.getTerrainHeight);
  }

  step(rawAction) {
    const controls = clipAction(rawAction);
    const p = this.drone.pos;
    this._prevPos[0] = p[0]; this._prevPos[1] = p[1]; this._prevPos[2] = p[2];

    this.drone.step(controls, this.dt);
    this.tick++;

    const speed = Math.hypot(this.drone.vel[0], this.drone.vel[1], this.drone.vel[2]);
    if (speed > this.topSpeed) this.topSpeed = speed;

    let reward = REWARD.timeCost;

    // Action smoothness, on the raw (pre-clip) action so the gradient is well defined.
    let ac = 0;
    for (let i = 0; i < ACT_DIM; i++) {
      const d = rawAction[i] - this._prevAction[i];
      ac += d * d;
      this._prevAction[i] = rawAction[i];
    }
    reward += REWARD.actionCost * (ac / ACT_DIM);

    let done = false;
    const hit = checkGatePassing(this._prevPos, [p[0], p[1], p[2]], this.gates[this.gateIdx]);

    if (hit.status === 'PASSED') {
      reward += REWARD.gatePass;
      this.gateIdx++;
      this.gatesCleared++;
      if (this.gateIdx >= this.gates.length) {
        reward += REWARD.completion;
        done = true;
      } else {
        // Re-base distance to the new target, and skip progress this tick: the switch is
        // a discontinuity, not motion the policy earned.
        this.prevDist = this._distToGate(this.gateIdx);
      }
    } else {
      const dist = this._distToGate(this.gateIdx);
      const delta = this.prevDist - dist;
      reward += REWARD.progress * Math.max(-REWARD.progressClamp, Math.min(REWARD.progressClamp, delta));
      this.prevDist = dist;
    }

    if (!this.drone.alive || this.drone.crashed) {
      reward += REWARD.crash;
      done = true;
    }
    if (this.tick >= this.maxTicks) done = true;

    return { obs: this._observe(), reward, done, info: { gatesCleared: this.gatesCleared, topSpeed: this.topSpeed, tick: this.tick } };
  }
}
