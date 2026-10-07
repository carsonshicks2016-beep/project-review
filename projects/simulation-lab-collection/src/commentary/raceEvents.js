import { DronePhysics } from '../dronePhysics.js';
import { checkGatePassing } from '../gateCollision.js';
import { readSensors, SENSOR_COUNT, SENSOR_RANGE } from '../collision/sensors.js';

// Turns a recorded trajectory into a timestamped stream of things worth remarking on.
//
// This is where commentary quality actually comes from. A language model with no
// situational awareness produces generic filler; given "gate 7 cleared at 112 km/h, 0.4s
// up on your record, 1.2m off the left wall" it has something to say. Every field here is
// measured from the run, so no voice built on top can invent a fact.

export const KEYFRAME_HZ = 20; // Worker samples every 3rd tick at dt=1/60

const NEAR_MISS_M = 4.0;       // Wall clearance below this is worth mentioning
const STALL_SPEED = 4.0;       // m/s
const STALL_SECONDS = 1.5;
const BIG_BANK_DEG = 55;

const SENSOR_LABEL = ['ahead', 'front-left', 'front-right', 'left', 'right', 'above', 'below', 'behind'];

function speedAt(traj, f) {
  if (f <= 0) return 0;
  const a = (f - 1) * 7, b = f * 7;
  return Math.hypot(traj[b] - traj[a], traj[b + 1] - traj[a + 1], traj[b + 2] - traj[a + 2]) * KEYFRAME_HZ;
}

// Roll angle about the drone's own forward axis, in degrees.
function bankDeg(drone) {
  const up = drone.rotateVectorByQuat([0, 1, 0], drone.quat);
  return Math.acos(Math.max(-1, Math.min(1, up[1]))) * 180 / Math.PI;
}

export function extractEvents(run, { gates, grid = null, reference = null } = {}) {
  const traj = run.trajectory || run.data;
  if (!traj || traj.length < 14) return [];
  const numFrames = traj.length / 7;
  const startGate = run.startGateIdx || 0;

  const events = [];
  const push = (frame, type, data = {}) =>
    events.push({ frame, t: frame / KEYFRAME_HZ, type, ...data });

  // A throwaway body used purely to evaluate orientation-dependent quantities
  // (wall rays, bank angle) at each keyframe, so none of that maths is duplicated here.
  const probe = new DronePhysics();
  const sensors = new Float32Array(SENSOR_COUNT);

  const prev = [0, 0, 0], cur = [0, 0, 0];
  let gi = startGate;
  const gateFrames = [];
  let topSpeed = 0, topSpeedFrame = 0;
  let stallStart = -1;
  let lastNearMiss = -999;
  let reportedBank = false;

  push(0, 'launch', { gate: startGate, totalGates: gates.length });

  for (let f = 1; f < numFrames; f++) {
    const po = (f - 1) * 7, co = f * 7;
    prev[0] = traj[po]; prev[1] = traj[po + 1]; prev[2] = traj[po + 2];
    cur[0] = traj[co];  cur[1] = traj[co + 1];  cur[2] = traj[co + 2];

    const spd = speedAt(traj, f);

    probe.pos[0] = cur[0]; probe.pos[1] = cur[1]; probe.pos[2] = cur[2];
    probe.quat[0] = traj[co + 3]; probe.quat[1] = traj[co + 4];
    probe.quat[2] = traj[co + 5]; probe.quat[3] = traj[co + 6];

    // ── Gate cleared ──
    if (gi < gates.length && checkGatePassing(prev, cur, gates[gi]).status === 'PASSED') {
      const split = gateFrames.length ? (f - gateFrames[gateFrames.length - 1]) / KEYFRAME_HZ : f / KEYFRAME_HZ;
      let delta = null;
      if (reference && reference.gateFrames && reference.gateFrames[gateFrames.length] !== undefined) {
        delta = (f - reference.gateFrames[gateFrames.length]) / KEYFRAME_HZ;
      }
      gateFrames.push(f);
      push(f, 'gate', {
        gate: gi,
        ordinal: gateFrames.length,
        totalGates: gates.length,
        speed: spd,
        split,
        elapsed: f / KEYFRAME_HZ,
        delta   // negative = ahead of the reference run
      });
      gi++;
    }

    // ── Near miss ──
    // Only while actually moving, and rate-limited: a drone hovering by a wall would
    // otherwise generate a line every frame.
    if (grid && spd > 6 && f - lastNearMiss > KEYFRAME_HZ * 1.5) {
      readSensors(grid, probe, sensors);
      let worst = -1, worstVal = 0;
      for (let i = 0; i < SENSOR_COUNT; i++) {
        if (sensors[i] > worstVal) { worstVal = sensors[i]; worst = i; }
      }
      const clearance = (1 - worstVal) * SENSOR_RANGE;
      if (worst >= 0 && clearance < NEAR_MISS_M) {
        lastNearMiss = f;
        push(f, 'nearMiss', { clearance, side: SENSOR_LABEL[worst], speed: spd });
      }
    }

    // ── Top speed ──
    if (spd > topSpeed) { topSpeed = spd; topSpeedFrame = f; }

    // ── Stall ──
    if (spd < STALL_SPEED) {
      if (stallStart < 0) stallStart = f;
      else if ((f - stallStart) / KEYFRAME_HZ > STALL_SECONDS) {
        push(f, 'stall', { seconds: (f - stallStart) / KEYFRAME_HZ });
        stallStart = -1;
      }
    } else {
      stallStart = -1;
    }

    // ── Style: a big committed bank, mentioned once ──
    if (!reportedBank && spd > 15 && bankDeg(probe) > BIG_BANK_DEG) {
      reportedBank = true;
      push(f, 'bank', { degrees: bankDeg(probe), speed: spd });
    }
  }

  if (topSpeed > 0) push(topSpeedFrame, 'topSpeed', { speed: topSpeed });

  // ── How it ended ──
  const cleared = gateFrames.length;
  const lastFrame = numFrames - 1;
  if (startGate + cleared >= gates.length) {
    push(lastFrame, 'finish', { cleared, totalGates: gates.length, time: lastFrame / KEYFRAME_HZ, topSpeed });
  } else if (run.hitWall) {
    push(lastFrame, 'wallHit', { cleared, totalGates: gates.length, nextGate: gi });
  } else if (run.crashed) {
    push(lastFrame, 'groundHit', { cleared, totalGates: gates.length, nextGate: gi });
  } else {
    push(lastFrame, 'timeout', { cleared, totalGates: gates.length, nextGate: gi, time: lastFrame / KEYFRAME_HZ });
  }

  if (run.isRecord) push(lastFrame, 'record', { cleared, totalGates: gates.length });

  events.sort((a, b) => a.frame - b.frame);
  return events;
}

// Gate frames alone, for use as the reference in a later run's delta comparison.
export function gateFramesOf(run, gates) {
  return extractEvents(run, { gates }).filter(e => e.type === 'gate').map(e => e.frame);
}
