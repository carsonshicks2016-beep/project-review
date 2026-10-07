// Frame timeline: normalizes authoritative telemetry frames and plays them
// back on a smoothed sim-time playhead.  Pure math, no DOM, no three.js —
// unit tested under node.
//
// Truth rules honored here:
//  * Positions, attitude, and road height come only from the server frame.
//  * Interpolation may blend between two authoritative frames; it never
//    extrapolates geometry the simulator did not produce.
//  * Any playback discontinuity (reset / scrub / episode boundary) snaps the
//    playhead instead of animating an impossible glide.

const num = (value, fallback = 0) => (Number.isFinite(Number(value)) ? Number(value) : fallback);

export function normalizeFrame(raw) {
  if (!raw || typeof raw !== "object" || raw.schema !== "fable-observatory-frame-v1") {
    throw new Error("frame schema is not fable-observatory-frame-v1");
  }
  const vehicle = raw.vehicle ?? {};
  const pose = vehicle.pose ?? {};
  const dynamics = vehicle.dynamics ?? {};
  const powertrain = vehicle.powertrain ?? {};
  const track = raw.track ?? {};
  const controls = raw.controls ?? {};
  const playback = raw.playback ?? {};
  return {
    seq: num(raw.sequence),
    simTime: num(raw.sim_time),
    episode: num(raw.episode),
    episodeTime: num(raw.episode_time),
    pose: {
      x: num(pose.x), y: num(pose.y), z: num(pose.z),
      yaw: num(pose.yaw), pitch: num(pose.pitch), roll: num(pose.roll),
    },
    roadZ: num(track.road_z_m, num(pose.z)),
    progress: num(track.progress),
    arcM: num(track.arc_m),
    lateralM: num(track.lateral_m),
    offTrack: Boolean(track.off_track),
    grade: num(track.grade),
    speedMps: num(dynamics.speed_mps),
    speedKmh: num(dynamics.speed_kmh),
    worldVel: Array.isArray(dynamics.world_velocity_mps)
      ? dynamics.world_velocity_mps.map((v) => num(v)) : [0, 0, 0],
    lateralG: num((dynamics.acceleration_mps2 ?? [])[1]) / 9.81,
    airborne: Boolean(dynamics.airborne),
    surfaceGrip: num(dynamics.surface_grip, 1),
    rpm: num(powertrain.rpm),
    gear: Math.trunc(num(powertrain.gear)),
    boost: num(powertrain.boost),
    hybridSoc: num(powertrain.hybrid_soc),
    mguFraction: num(powertrain.mgu_power_fraction),
    aeroEngaged: Boolean(powertrain.active_aero_engaged),
    steer: num(controls.steer),
    throttle: num(controls.throttle),
    brake: num(controls.brake),
    wheels: Array.isArray(vehicle.wheels) ? vehicle.wheels.map((w) => ({
      id: String(w?.id ?? "?"),
      rotation: num(w?.rotation_rad),
      steer: num(w?.steer_rad),
      load: num(w?.load_n),
      slipRatio: num(w?.slip_ratio),
      slipAngle: num(w?.slip_angle_rad),
      grip: num(w?.grip, 1),
      contact: Boolean(w?.contact),
    })) : [],
    footprint: Array.isArray(vehicle.geometry?.footprint_world_xy)
      ? vehicle.geometry.footprint_world_xy : null,
    fable: raw.fable ?? {},
    brain: raw.brain ?? null,
    observations: raw.observations ?? null,
    sensors: raw.sensor_geometry ?? null,
    playback: {
      paused: Boolean(playback.paused),
      speed: num(playback.speed, 1),
      revision: num(playback.revision, 0),
      discontinuity: playback.discontinuity ?? null,
    },
  };
}

/** Shortest-arc angle blend so yaw never spins the long way around. */
export function mixAngle(a, b, t) {
  const delta = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + delta * t;
}

const mix = (a, b, t) => a + (b - a) * t;

/** Blend two normalized frames at t∈[0,1]; scalar channels lerp, poses too. */
export function blendFrames(a, b, t) {
  if (t <= 0) return a;
  if (t >= 1) return b;
  return {
    ...b,
    simTime: mix(a.simTime, b.simTime, t),
    pose: {
      x: mix(a.pose.x, b.pose.x, t),
      y: mix(a.pose.y, b.pose.y, t),
      z: mix(a.pose.z, b.pose.z, t),
      yaw: mixAngle(a.pose.yaw, b.pose.yaw, t),
      pitch: mix(a.pose.pitch, b.pose.pitch, t),
      roll: mix(a.pose.roll, b.pose.roll, t),
    },
    roadZ: mix(a.roadZ, b.roadZ, t),
    speedMps: mix(a.speedMps, b.speedMps, t),
    speedKmh: mix(a.speedKmh, b.speedKmh, t),
    rpm: mix(a.rpm, b.rpm, t),
    steer: mix(a.steer, b.steer, t),
    throttle: mix(a.throttle, b.throttle, t),
    brake: mix(a.brake, b.brake, t),
    lateralG: mix(a.lateralG, b.lateralG, t),
    progress: mix(a.progress, b.progress, t),
    wheels: b.wheels.map((wheel, i) => {
      const prev = a.wheels[i];
      if (!prev) return wheel;
      return {
        ...wheel,
        rotation: mixAngle(prev.rotation, wheel.rotation, t),
        steer: mix(prev.steer, wheel.steer, t),
        load: mix(prev.load, wheel.load, t),
      };
    }),
  };
}

/**
 * Carry control state from the newest frame onto the rendered one.
 *
 * The playhead deliberately trails the newest frame by `latency`, so the frame
 * we render is always ~140 ms old. Playback state is control-plane truth, not
 * something to play out on that delay: the pause flag only ever lands on the
 * newest frame, so reading it off the interpolated frame meant `paused` was
 * permanently false. That left the Pause button stuck on "Pause" and made
 * resume unreachable — the session could only be freed by reloading.
 */
function withLatestPlayback(frame, newest) {
  if (!frame || frame === newest) return frame;
  if (frame.playback === newest.playback) return frame;
  return { ...frame, playback: newest.playback };
}

export class Timeline {
  constructor({ latency = 0.14, capacity = 90 } = {}) {
    this.latency = latency;
    this.capacity = capacity;
    this.frames = [];
    this.playhead = null;
    this.revision = -1;
    this.episode = -1;
  }

  /** Accept an authoritative frame; returns a discontinuity kind or null. */
  push(frame) {
    let breakKind = null;
    if (frame.playback.discontinuity) breakKind = String(frame.playback.discontinuity.kind ?? "discontinuity");
    if (this.episode >= 0 && frame.episode !== this.episode) breakKind = breakKind ?? "episode_change";
    if (this.revision >= 0 && frame.playback.revision !== this.revision
        && this.frames.length && frame.simTime < this.frames[this.frames.length - 1].simTime - 1e-9) {
      breakKind = breakKind ?? "scrub";
    }
    this.revision = frame.playback.revision;
    this.episode = frame.episode;
    if (breakKind) {
      this.frames.length = 0;
      this.playhead = null;
    }
    if (this.frames.length && frame.simTime <= this.frames[this.frames.length - 1].simTime + 1e-9) {
      // A paused / state frame repeats sim-time; replace the tail so the
      // latest control state (pause flag, revision) is what renders.
      this.frames[this.frames.length - 1] = frame;
    } else {
      this.frames.push(frame);
      if (this.frames.length > this.capacity) this.frames.splice(0, this.frames.length - this.capacity);
    }
    return breakKind;
  }

  get latest() {
    return this.frames.length ? this.frames[this.frames.length - 1] : null;
  }

  /**
   * Advance the playhead by wall dt and return the frame to render.
   * The playhead chases (latest sim-time − latency) with gentle rate
   * correction so late/early websocket delivery never causes rubber-banding.
   */
  sample(wallDt) {
    if (!this.frames.length) return null;
    const newest = this.frames[this.frames.length - 1];
    const target = newest.simTime - this.latency;
    if (this.playhead === null || Math.abs(target - this.playhead) > 1.5) {
      this.playhead = target;
    } else {
      const speed = newest.playback.paused ? 0 : newest.playback.speed || 1;
      const drift = target - this.playhead;
      this.playhead += wallDt * speed + drift * Math.min(1, wallDt * 1.6);
    }
    this.playhead = Math.min(this.playhead, newest.simTime);
    if (this.frames.length === 1) return newest;
    const oldest = this.frames[0];
    this.playhead = Math.max(this.playhead, oldest.simTime);
    let hi = this.frames.length - 1;
    while (hi > 0 && this.frames[hi - 1].simTime > this.playhead) hi -= 1;
    if (hi === 0) return withLatestPlayback(oldest, newest);
    const a = this.frames[hi - 1];
    const b = this.frames[hi];
    const span = b.simTime - a.simTime;
    const t = span > 1e-9 ? (this.playhead - a.simTime) / span : 1;
    return withLatestPlayback(blendFrames(a, b, t), newest);
  }
}
