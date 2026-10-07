import * as THREE from "three";

const lerp = (a, b, t) => a + (b - a) * t;

function lerpAngle(a, b, t) {
  const delta = Math.atan2(Math.sin(b - a), Math.cos(b - a));
  return a + delta * t;
}

function interpolateArray(a, b, t) {
  return a.map((value, index) => lerp(value, b[index] ?? value, t));
}

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

/**
 * Monotone cubic interpolation for road height.
 *
 * Linear interpolation keeps position continuous, but changes vertical speed
 * abruptly at every 30 Hz telemetry boundary. Those tiny slope corners read
 * as a rhythmic body hop on steep grades. Hermite interpolation matches the
 * slope on both sides of each boundary; the monotonicity limiter prevents the
 * curve from lifting the car above a crest or dipping it below a trough.
 */
function interpolateRoadHeight(previous, a, b, next, t, valueForFrame) {
  const aValue = finiteNumber(valueForFrame(a));
  const bValue = finiteNumber(valueForFrame(b), aValue);
  const span = finiteNumber(b.simTime) - finiteNumber(a.simTime);
  if (!(span > 1e-6)) return lerp(aValue, bValue, t);

  const secant = (bValue - aValue) / span;
  if (Math.abs(secant) < 1e-9) return aValue;

  let slopeA = secant;
  if (previous && finiteNumber(a.simTime) - finiteNumber(previous.simTime) > 1e-6) {
    slopeA = (bValue - finiteNumber(valueForFrame(previous), aValue))
      / (finiteNumber(b.simTime) - finiteNumber(previous.simTime));
  }

  let slopeB = secant;
  if (next && finiteNumber(next.simTime) - finiteNumber(b.simTime) > 1e-6) {
    slopeB = (finiteNumber(valueForFrame(next), bValue) - aValue)
      / (finiteNumber(next.simTime) - finiteNumber(a.simTime));
  }

  // Fritsch-Carlson limiting: retain C1 slopes without overshooting the two
  // authoritative road-height samples that bound this render interval.
  let ratioA = slopeA / secant;
  let ratioB = slopeB / secant;
  if (ratioA < 0) ratioA = 0;
  if (ratioB < 0) ratioB = 0;
  const magnitude = Math.hypot(ratioA, ratioB);
  if (magnitude > 3) {
    const scale = 3 / magnitude;
    ratioA *= scale;
    ratioB *= scale;
  }
  slopeA = ratioA * secant;
  slopeB = ratioB * secant;

  const t2 = t * t;
  const t3 = t2 * t;
  return (2 * t3 - 3 * t2 + 1) * aValue
    + (t3 - 2 * t2 + t) * span * slopeA
    + (-2 * t3 + 3 * t2) * bValue
    + (t3 - t2) * span * slopeB;
}

export function interpolateFrames(a, b, alpha, neighbors = {}) {
  if (!a) return b;
  if (!b) return a;
  const t = THREE.MathUtils.clamp(alpha, 0, 1);
  const grounded = !a.vehicle.airborne && !b.vehicle.airborne;
  const position = interpolateArray(a.vehicle.position, b.vehicle.position, t);
  let roadZ = lerp(
    a.vehicle.roadZ ?? a.vehicle.position[2],
    b.vehicle.roadZ ?? b.vehicle.position[2],
    t,
  );
  if (grounded) {
    const roadValue = (frame) => frame.vehicle.roadZ ?? frame.vehicle.position[2];
    roadZ = interpolateRoadHeight(neighbors.previous, a, b, neighbors.next, t, roadValue);
    position[2] = roadZ;
  }
  return {
    ...b,
    simTime: lerp(a.simTime, b.simTime, t),
    vehicle: {
      ...b.vehicle,
      position,
      yaw: lerpAngle(a.vehicle.yaw, b.vehicle.yaw, t),
      pitch: lerpAngle(a.vehicle.pitch, b.vehicle.pitch, t),
      roll: lerpAngle(a.vehicle.roll, b.vehicle.roll, t),
      roadZ,
      steer: lerp(a.vehicle.steer, b.vehicle.steer, t),
      wheelRotation: interpolateArray(a.vehicle.wheelRotation, b.vehicle.wheelRotation, t),
      wheelTravel: interpolateArray(a.vehicle.wheelTravel, b.vehicle.wheelTravel, t),
    },
    telemetry: {
      ...b.telemetry,
      speedMps: lerp(a.telemetry.speedMps, b.telemetry.speedMps, t),
      rpm: lerp(a.telemetry.rpm, b.telemetry.rpm, t),
      progress: lerp(a.telemetry.progress, b.telemetry.progress, t),
    },
  };
}

/**
 * A jitter-absorbing playout buffer for authoritative telemetry.
 *
 * Frames stream from Python at a fixed policy cadence (control_hz) but arrive
 * over a lock-step WebSocket with real jitter (tick cost, scheduler grain, and
 * CPU contention with training). Sampling on wall-clock arrival time makes the
 * car freeze then leap whenever the buffer briefly underruns — the "stepping /
 * bumping forward" artifact.
 *
 * Instead we key everything on the frame's own authoritative ``simTime`` and
 * advance a local playhead at the real playback rate (wall dt x replay speed),
 * kept a fixed ``delaySeconds`` behind the freshest frame. A gentle correction
 * absorbs arrival jitter, and the playhead is clamped to the data we actually
 * hold, so motion is continuous and never extrapolates past a real sample.
 * Physics and pose remain server-authoritative; this is render-only smoothing.
 */
export class FrameBuffer {
  constructor({ delaySeconds = 0.14, capacity = 240, catchUpRate = 4, maxSnapGap = 0.75 } = {}) {
    // How far behind the live edge we render, in authoritative sim seconds.
    // ~4 frames at 30 Hz — deep enough to ride out delivery jitter, shallow
    // enough to stay responsive for a spectator view.
    this.delaySeconds = delaySeconds;
    this.capacity = capacity;
    // Rate (1/s) at which the playhead corrects residual drift toward target.
    this.catchUpRate = catchUpRate;
    // A sim-time jump larger than this (or any backward step) is a scrub,
    // reset, or episode change rather than continuous motion — resync instead
    // of interpolating across the discontinuity.
    this.maxSnapGap = maxSnapGap;
    this.frames = [];
    this.playhead = null;
    this.lastNow = 0;
  }

  push(frame) {
    if (!frame || !Number.isFinite(frame.simTime)) return;
    const previous = this.frames[this.frames.length - 1];
    if (previous) {
      const gap = frame.simTime - previous.simTime;
      const episodeChanged = Number.isFinite(frame.episode)
        && Number.isFinite(previous.episode)
        && frame.episode !== previous.episode;
      const checkpointChanged = frame.identity?.policyHash
        && previous.identity?.policyHash
        && frame.identity.policyHash !== previous.identity.policyHash;
      if (gap < 0 || gap > this.maxSnapGap || episodeChanged
          || checkpointChanged || frame.replay?.discontinuity) {
        this.frames.length = 0;
        this.playhead = null;
      }
    }
    this.frames.push(frame);
    if (this.frames.length > this.capacity) {
      this.frames.splice(0, this.frames.length - this.capacity);
    }
  }

  clear() {
    this.frames.length = 0;
    this.playhead = null;
    this.lastNow = 0;
  }

  sample(now = performance.now()) {
    const count = this.frames.length;
    if (count === 0) return null;
    const newest = this.frames[count - 1];
    if (count === 1) {
      this.playhead = newest.simTime;
      this.lastNow = now;
      return newest;
    }
    const oldest = this.frames[0];
    const dt = this.lastNow ? THREE.MathUtils.clamp((now - this.lastNow) / 1000, 0, 0.25) : 0;
    this.lastNow = now;

    const speed = Number.isFinite(newest.replay?.speed) ? Math.max(0, newest.replay.speed) : 1;
    const paused = Boolean(newest.replay?.paused);
    const target = newest.simTime - this.delaySeconds;

    if (this.playhead === null) {
      this.playhead = target;
    } else if (paused) {
      // Hold on the freshest frame; no forward motion while paused/scrubbing.
      this.playhead = newest.simTime;
    } else {
      // Feed-forward at the true playback rate carries smooth motion between
      // the discrete authoritative frames; the gentle correction keeps the
      // playhead locked ~delaySeconds behind the live edge despite jitter.
      this.playhead += dt * speed;
      this.playhead += (target - this.playhead) * (1 - Math.exp(-dt * this.catchUpRate));
    }

    // Never render outside the trajectory we actually hold.
    this.playhead = THREE.MathUtils.clamp(this.playhead, oldest.simTime, newest.simTime);

    let hi = 1;
    while (hi < count - 1 && this.frames[hi].simTime < this.playhead) hi += 1;
    const a = this.frames[hi - 1];
    const b = this.frames[hi];
    const span = b.simTime - a.simTime;
    const alpha = span > 1e-6 ? (this.playhead - a.simTime) / span : 1;
    return interpolateFrames(a, b, alpha, {
      previous: hi >= 2 ? this.frames[hi - 2] : null,
      next: hi + 1 < count ? this.frames[hi + 1] : null,
    });
  }
}
