import test from "node:test";
import assert from "node:assert/strict";

import { FrameBuffer, interpolateFrames } from "../src/frame-buffer.js";

const SPEED_MPS = 40;
const HZ = 30;
const FRAME_DT = 1 / HZ;
const RENDER_DT_MS = 1000 / 60;

// A frame whose x-position is a linear function of authoritative sim time, so
// smooth / correct playback of x is equivalent to smooth / correct playback of
// the trajectory. Only the fields the buffer reads or interpolates are set.
function makeFrame(simTime, { speed = 1, paused = false } = {}) {
  return {
    simTime,
    seq: Math.round(simTime * HZ),
    vehicle: {
      position: [simTime * SPEED_MPS, 0, 0],
      yaw: 0, pitch: 0, roll: 0, steer: 0,
      wheelRotation: [0, 0, 0, 0],
      wheelTravel: [0, 0, 0, 0],
    },
    telemetry: { speedMps: SPEED_MPS, rpm: 6000, progress: simTime * 0.01 },
    replay: { speed, paused },
  };
}

function makeHeightFrame(simTime, height) {
  const frame = makeFrame(simTime);
  frame.vehicle.position[2] = height;
  frame.vehicle.roadZ = height;
  frame.vehicle.airborne = false;
  return frame;
}

test("grounded road height crosses telemetry boundaries with continuous slope", () => {
  const frames = [0, 1, 4, 9, 16].map((height, index) => makeHeightFrame(index, height));
  const epsilon = 1e-4;
  const left = interpolateFrames(frames[1], frames[2], 1 - epsilon, {
    previous: frames[0],
    next: frames[3],
  }).vehicle.position[2];
  const boundary = frames[2].vehicle.position[2];
  const right = interpolateFrames(frames[2], frames[3], epsilon, {
    previous: frames[1],
    next: frames[4],
  }).vehicle.position[2];
  const leftSlope = (boundary - left) / epsilon;
  const rightSlope = (right - boundary) / epsilon;

  assert.ok(Math.abs(leftSlope - rightSlope) < 0.01, "vertical speed must not jump at a telemetry boundary");
  assert.ok(left <= boundary && right >= boundary, "monotone road interpolation must not overshoot");
});

test("playback stays smooth and correct-speed under in-order arrival jitter", () => {
  // Fixed jitter pattern (ms) applied to each frame's ideal arrival, including
  // a 210 ms stall — the lock-step + CPU-contention delivery that made the old
  // wall-clock buffer freeze then leap. Delivery stays in sim-time order.
  const jitter = [0, 48, -22, 61, 12, -33, 210, 8, -25, 44, 6, -30];
  const basePipelineMs = 30;
  const total = 200;
  const arrivals = [];
  let previous = -1;
  for (let k = 0; k < total; k += 1) {
    const ideal = k * FRAME_DT * 1000 + basePipelineMs + jitter[k % jitter.length];
    const arriveMs = Math.max(previous + 1, ideal);
    arrivals.push(arriveMs);
    previous = arriveMs;
  }

  const buffer = new FrameBuffer();
  let nextK = 0;
  const samples = [];
  for (let ms = 0; ms <= 5000; ms += RENDER_DT_MS) {
    while (nextK < total && arrivals[nextK] <= ms) {
      buffer.push(makeFrame(nextK * FRAME_DT));
      nextK += 1;
    }
    const frame = buffer.sample(ms);
    if (frame) samples.push({ ms, x: frame.vehicle.position[0] });
  }

  // Judge the steady region, past the initial buffer fill / ramp-in.
  const steady = samples.filter((s) => s.ms >= 600 && s.ms <= 4600);
  assert.ok(steady.length > 200, "expected a populated steady region");

  const idealStep = SPEED_MPS * (RENDER_DT_MS / 1000);
  let maxStep = 0;
  let minStep = Infinity;
  for (let i = 1; i < steady.length; i += 1) {
    const step = steady[i].x - steady[i - 1].x;
    assert.ok(step >= -1e-6, `motion must never run backward (step ${step} at ${steady[i].ms}ms)`);
    maxStep = Math.max(maxStep, step);
    minStep = Math.min(minStep, step);
  }
  // No single render may lurch forward: the anti-"bump" guarantee. A stall is
  // absorbed as a brief hold + eased catch-up, never a teleport.
  assert.ok(maxStep <= idealStep * 3, `max render step ${maxStep.toFixed(3)}m exceeded 3x ideal ${idealStep.toFixed(3)}m`);

  // Average velocity tracks the authoritative rate (no systematic drift/lag).
  const elapsedS = (steady.at(-1).ms - steady[0].ms) / 1000;
  const travelled = steady.at(-1).x - steady[0].x;
  const meanSpeed = travelled / elapsedS;
  assert.ok(Math.abs(meanSpeed - SPEED_MPS) < SPEED_MPS * 0.1, `mean speed ${meanSpeed.toFixed(2)} strayed from ${SPEED_MPS}`);
});

test("pause holds the freshest pose instead of drifting forward", () => {
  const buffer = new FrameBuffer();
  buffer.push(makeFrame(0.0));
  buffer.push(makeFrame(FRAME_DT));
  buffer.sample(0);
  buffer.sample(20);
  // Server pauses: the newest frame is flagged paused and sim time stops.
  const held = 2 * FRAME_DT;
  buffer.push(makeFrame(held, { paused: true }));
  const first = buffer.sample(40).vehicle.position[0];
  const later = buffer.sample(400).vehicle.position[0];
  const muchLater = buffer.sample(4000).vehicle.position[0];
  assert.equal(first, held * SPEED_MPS, "pause snaps to the newest authoritative pose");
  assert.equal(later, first, "a paused viewer must not creep forward");
  assert.equal(muchLater, first, "a paused viewer must stay put indefinitely");
});

test("a backward sim-time jump resyncs instead of interpolating across a scrub", () => {
  const buffer = new FrameBuffer();
  for (let k = 0; k <= 30; k += 1) buffer.push(makeFrame(k * FRAME_DT));
  buffer.sample(0);
  buffer.sample(500);
  // Scrub back near the start: sim time leaps backward.
  buffer.push(makeFrame(0.1));
  const frame = buffer.sample(520);
  assert.equal(frame.vehicle.position[0], 0.1 * SPEED_MPS, "scrub target renders directly, no cross-fade from the old cursor");
});
