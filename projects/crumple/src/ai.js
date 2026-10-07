// Pursuit AI. Steers toward the player, feels ahead with three whisker probes
// to slide around buildings, and reverses out when it wedges itself somewhere.

const PROBE_MARGIN = 2.6;

export class Chaser {
  constructor(vehicle) {
    this.v = vehicle;
    this.stuckTime = 0;
    this.reverseTime = 0;
    this.aggression = 0.85 + Math.random() * 0.3;
  }

  blocked(world, x, z) {
    for (const b of world.boxes) {
      if (x > b.min.x - PROBE_MARGIN && x < b.max.x + PROBE_MARGIN &&
          z > b.min.z - PROBE_MARGIN && z < b.max.z + PROBE_MARGIN) {
        return true;
      }
    }
    return false;
  }

  think(world, target, dt) {
    const v = this.v;
    const c = v.controls;

    if (v.disabled) {
      c.throttle = 0; c.steer = 0; c.brake = 1; c.handbrake = true;
      return;
    }

    const dx = target.pos.x - v.pos.x;
    const dz = target.pos.z - v.pos.z;
    const range = Math.hypot(dx, dz);

    // Aim slightly ahead of where the player is going, so it cuts corners.
    const lead = Math.min(range / 22, 1.4);
    const ax = dx + target.vel.x * lead;
    const az = dz + target.vel.z * lead;

    const fwdDot = ax * v.fwd.x + az * v.fwd.z;
    const rightDot = ax * v.right.x + az * v.right.z;

    let steer = Math.atan2(rightDot, Math.abs(fwdDot) + 3) * 1.5;
    let throttle = this.aggression;

    // Whiskers: centre, and two angled outward.
    const px = v.pos.x, pz = v.pos.z;
    const reach = 7 + v.speed * 0.7;
    const cFwd = this.blocked(world, px + v.fwd.x * reach, pz + v.fwd.z * reach);
    const cL = this.blocked(world,
      px + v.fwd.x * reach * 0.7 - v.right.x * 4.5,
      pz + v.fwd.z * reach * 0.7 - v.right.z * 4.5);
    const cR = this.blocked(world,
      px + v.fwd.x * reach * 0.7 + v.right.x * 4.5,
      pz + v.fwd.z * reach * 0.7 + v.right.z * 4.5);

    if (cFwd || cL || cR) {
      let avoid = 0;
      if (cL) avoid += 1;
      if (cR) avoid -= 1;
      if (cFwd && avoid === 0) avoid = steer >= 0 ? -1 : 1;
      steer = steer * 0.25 + avoid * 1.1;
      if (cFwd) throttle *= 0.55;
    }

    // Reverse out of a wedge.
    if (v.speed < 1.4 && this.reverseTime <= 0) {
      this.stuckTime += dt;
      if (this.stuckTime > 1.3) { this.reverseTime = 1.1; this.stuckTime = 0; }
    } else if (v.speed > 3) {
      this.stuckTime = 0;
    }

    if (this.reverseTime > 0) {
      this.reverseTime -= dt;
      c.throttle = -1;
      c.steer = -Math.sign(steer || 1) * 0.8;
      c.brake = 0;
      c.handbrake = false;
      return;
    }

    // Back off the gas for tight turns so it doesn't understeer into a wall.
    const turnPenalty = Math.min(Math.abs(steer), 1) * Math.min(v.speed / 26, 1);
    throttle *= 1 - turnPenalty * 0.55;

    // Close the last few metres hard — that's the ram.
    if (range < 16 && Math.abs(steer) < 0.5) throttle = 1;

    c.steer = Math.max(-1, Math.min(1, steer));
    c.throttle = Math.max(-1, Math.min(1, throttle));
    c.brake = 0;
    c.handbrake = false;
  }
}
