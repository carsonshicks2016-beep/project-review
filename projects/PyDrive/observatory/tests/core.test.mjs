import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import { CONDITION_IDS, ambienceWord, condition } from "../src/conditions.js";
import { tyreSlipStrength, tyreSprayStrength } from "../src/fx.js";
import { blendFrames, mixAngle, normalizeFrame, Timeline } from "../src/timeline.js";
import { FOA1_HEADER, parseFoa1 } from "../src/audio.js";
import { simToThree, vehicleOrientation } from "../src/coords.js";

function bareFrame(overrides = {}) {
  return {
    schema: "fable-observatory-frame-v1",
    sequence: 1,
    sim_time: 1.0,
    episode: 0,
    episode_time: 1.0,
    vehicle: {
      pose: { x: 10, y: 20, z: 5, yaw: 0.5, pitch: 0.01, roll: -0.02 },
      dynamics: {
        speed_mps: 40, speed_kmh: 144,
        world_velocity_mps: [40, 0, 0],
        acceleration_mps2: [1, 0],
      },
      powertrain: { rpm: 7000, gear: 3 },
      wheels: [
        { id: "FL", rotation_rad: 1, steer_rad: 0.1, load_n: 2000, slip_ratio: 0, grip: 1, contact: true },
        { id: "FR", rotation_rad: 1.1, steer_rad: 0.1, load_n: 2000, slip_ratio: 0, grip: 1, contact: true },
        { id: "RL", rotation_rad: 1.2, steer_rad: 0, load_n: 2200, slip_ratio: 0, grip: 1, contact: true },
        { id: "RR", rotation_rad: 1.3, steer_rad: 0, load_n: 2200, slip_ratio: 0, grip: 1, contact: true },
      ],
    },
    controls: { steer: 0.2, throttle: 0.8, brake: 0 },
    track: { progress: 0.1, road_z_m: 5, arc_m: 100, lateral_m: 0, off_track: false },
    fable: { valid: true, pace_ratio: 0.95 },
    playback: { paused: false, speed: 1, revision: 0 },
    ...overrides,
  };
}

test("condition table covers acceptance weather ids", () => {
  for (const id of ["clear-day", "wet-dusk", "rain-day", "rain-night"]) {
    assert.equal(condition(id).id, id);
  }
  assert.ok(CONDITION_IDS.length >= 9);
});

test("ambienceWord maps lighting to dry|dusk|night", () => {
  assert.equal(ambienceWord("clear-day"), "dry");
  assert.equal(ambienceWord("rain-day"), "dry");
  assert.equal(ambienceWord("wet-dusk"), "dusk");
  assert.equal(ambienceWord("rain-night"), "night");
});

test("wet-dusk is obviously wetter than clear-day for asphalt", () => {
  const dry = condition("clear-day");
  const wet = condition("wet-dusk");
  assert.equal(dry.wetness, 0);
  assert.ok(wet.wetness >= 0.6);
  assert.notEqual(dry.fogColor, wet.fogColor);
});

test("tyreSprayStrength needs wetness and speed", () => {
  assert.equal(tyreSprayStrength(0, 40), 0);
  assert.ok(tyreSprayStrength(0.9, 5) < 0.05);
  assert.ok(tyreSprayStrength(0.9, 40) > 0.7);
});

test("tyreSlipStrength needs both slip and speed", () => {
  const gripping = [{ slipRatio: 0.01, slipAngle: 0.0, contact: true }];
  const sliding = [{ slipRatio: 0.7, slipAngle: 0.2, contact: true }];
  assert.equal(tyreSlipStrength(gripping, 60), 0);
  // A spinning wheel at a standstill is not a slide worth drawing.
  assert.ok(tyreSlipStrength(sliding, 1) < 0.05);
  assert.ok(tyreSlipStrength(sliding, 60) > 0.9);
  // Airborne wheels report no contact and must not smoke.
  assert.equal(tyreSlipStrength([{ slipRatio: 0.9, contact: false }], 60), 0);
  assert.equal(tyreSlipStrength([], 60), 0);
});

test("every condition carries a stylized sky palette", () => {
  for (const id of CONDITION_IDS) {
    const c = condition(id);
    assert.ok(typeof c.cloudAmount === "number", `${id} cloudAmount`);
    assert.ok(c.cloudAmount >= 0 && c.cloudAmount <= 1, `${id} cloudAmount range`);
    assert.ok(typeof c.cloudTint === "number", `${id} cloudTint`);
    // Banded light only reads against a hard key/fill ratio.
    assert.ok(c.ambientIntensity < c.sunIntensity, `${id} key must beat fill`);
  }
});

test("normalizeFrame requires schema and remaps pose", () => {
  const frame = normalizeFrame(bareFrame());
  assert.equal(frame.pose.x, 10);
  assert.equal(frame.roadZ, 5);
  assert.equal(frame.speedKmh, 144);
  assert.equal(frame.wheels.length, 4);
  assert.throws(() => normalizeFrame({ schema: "nope" }));
});

test("mixAngle takes the short arc", () => {
  // 3 → -3 short path crosses ±π (~3.14), not zero.
  const mid = mixAngle(3.0, -3.0, 0.5);
  assert.ok(Math.abs(Math.abs(mid) - Math.PI) < 0.05);
  assert.ok(Math.abs(mixAngle(0.1, 0.5, 0.5) - 0.3) < 1e-12);
});

test("Timeline interpolates without inventing beyond newest", () => {
  const tl = new Timeline({ latency: 0.1, capacity: 30 });
  tl.push(normalizeFrame(bareFrame({ sequence: 1, sim_time: 1.0 })));
  tl.push(normalizeFrame(bareFrame({
    sequence: 2, sim_time: 1.2,
    vehicle: {
      ...bareFrame().vehicle,
      pose: { x: 20, y: 20, z: 5, yaw: 0.5, pitch: 0, roll: 0 },
    },
  })));
  const sample = tl.sample(0.05);
  assert.ok(sample.pose.x >= 10 && sample.pose.x <= 20);
  assert.ok(sample.simTime <= 1.2);
});

test("Timeline snaps on discontinuity", () => {
  const tl = new Timeline({ latency: 0.05 });
  tl.push(normalizeFrame(bareFrame({ sequence: 1, sim_time: 5 })));
  const kind = tl.push(normalizeFrame(bareFrame({
    sequence: 2, sim_time: 0.1, episode: 1,
    playback: { paused: false, speed: 1, revision: 0, discontinuity: { kind: "episode_reset" } },
  })));
  assert.equal(kind, "episode_reset");
  assert.equal(tl.frames.length, 1);
});

test("blendFrames lerps scalars and angles", () => {
  const a = normalizeFrame(bareFrame({ sim_time: 0 }));
  const b = normalizeFrame(bareFrame({
    sim_time: 1,
    vehicle: {
      ...bareFrame().vehicle,
      pose: { x: 30, y: 20, z: 5, yaw: 1.0, pitch: 0, roll: 0 },
      dynamics: { speed_mps: 50, speed_kmh: 180, world_velocity_mps: [50, 0, 0], acceleration_mps2: [0, 0] },
    },
  }));
  const mid = blendFrames(a, b, 0.5);
  assert.ok(Math.abs(mid.pose.x - 20) < 1e-6);
  assert.ok(Math.abs(mid.speedKmh - 162) < 1e-6);
});

test("Timeline surfaces the newest playback state, not the delayed one", () => {
  // Regression: the playhead trails `latency`, so a pause landing on the newest
  // frame was never observed by the renderer and resume became unreachable.
  const tl = new Timeline({ latency: 0.1, capacity: 30 });
  for (let i = 0; i < 6; i += 1) {
    tl.push(normalizeFrame(bareFrame({ sequence: i + 1, sim_time: 1 + i * 0.05 })));
  }
  tl.push(normalizeFrame(bareFrame({
    sequence: 7, sim_time: 1.35,
    playback: { paused: true, speed: 1, revision: 2 },
  })));
  const sample = tl.sample(0.016);
  // The rendered pose is still behind the newest frame...
  assert.ok(sample.simTime < 1.35);
  // ...but the control state is the live one.
  assert.equal(sample.playback.paused, true);
  assert.equal(sample.playback.revision, 2);
});

test("vehicleOrientation points the body along its direction of travel", () => {
  // Regression: yaw and roll were negated, which crabbed the car up to ~58°
  // away from its velocity. cameras.js derives forward as (cos y, 0, -sin y).
  for (const yaw of [0, 0.7, -2.6, 3.0]) {
    const q = vehicleOrientation(yaw, 0, 0);
    const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(q);
    const expected = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
    assert.ok(forward.dot(expected) > 0.9999, `yaw ${yaw} -> ${forward.toArray()}`);
  }
  // Sim velocity remapped through simToThree must agree with that forward.
  const yaw = 1.1;
  const vel = simToThree(Math.cos(yaw), Math.sin(yaw), 0).normalize();
  const forward = new THREE.Vector3(1, 0, 0).applyQuaternion(vehicleOrientation(yaw, 0, 0));
  assert.ok(forward.dot(vel) > 0.9999);
});

test("vehicleOrientation lifts the nose on climb and the left side on bank", () => {
  // supra/physics.py: +pitch = nose up (grade_body), +roll = left side up.
  const up = new THREE.Vector3(0, 1, 0);
  const nose = new THREE.Vector3(1, 0, 0).applyQuaternion(vehicleOrientation(0, 0.2, 0));
  assert.ok(nose.y > 0.15, `nose should rise, got ${nose.y}`);
  // Car's left in three space is -Z; rolling left-side-up gives it +Y.
  const left = new THREE.Vector3(0, 0, -1).applyQuaternion(vehicleOrientation(0, 0, 0.2));
  assert.ok(left.dot(up) > 0.15, `left should rise, got ${left.dot(up)}`);
});

test("parseFoa1 reads 32-byte header + PCM", () => {
  const frames = 4;
  const channels = 2;
  const header = new ArrayBuffer(FOA1_HEADER.BYTES);
  const view = new DataView(header);
  const magic = [70, 79, 65, 49]; // FOA1
  magic.forEach((b, i) => view.setUint8(i, b));
  view.setUint16(4, 1, true);
  view.setUint16(6, channels, true);
  view.setUint32(8, 48000, true);
  view.setUint32(12, frames, true);
  view.setBigUint64(16, 7n, true);
  view.setFloat64(24, 1.25, true);
  const pcm = new Int16Array(frames * channels);
  pcm[0] = 1000;
  const bytes = new Uint8Array(FOA1_HEADER.BYTES + pcm.byteLength);
  bytes.set(new Uint8Array(header), 0);
  bytes.set(new Uint8Array(pcm.buffer), FOA1_HEADER.BYTES);
  const parsed = parseFoa1(bytes.buffer);
  assert.equal(parsed.sampleRate, 48000);
  assert.equal(parsed.frames, 4);
  assert.equal(parsed.sequence, 7);
  assert.equal(parsed.simTime, 1.25);
  assert.equal(parsed.pcm.byteLength, 16);
});
