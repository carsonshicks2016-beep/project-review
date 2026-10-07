import test from "node:test";
import assert from "node:assert/strict";

import * as THREE from "three";

import { RenderAttitudeFilter, RenderHeightFilter, vehiclePoseQuaternion } from "../src/vehicle-pose.js";

const EPSILON = 1e-12;

function orientedAxes(pose) {
  const quaternion = vehiclePoseQuaternion(pose);
  return {
    quaternion,
    forward: new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion),
    left: new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion),
    up: new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion),
  };
}

test("vehicle pose follows uphill and downhill pitch with simulator sign", () => {
  for (const pitch of [Math.atan(0.15), Math.atan(-0.18)]) {
    const { forward } = orientedAxes({ yaw: 0.73, pitch, roll: 0 });
    const horizontal = Math.hypot(forward.x, forward.z);
    assert.ok(Math.abs(forward.y / horizontal - Math.tan(pitch)) < EPSILON);
    assert.equal(Math.sign(forward.y), Math.sign(pitch));
  }
});

test("positive simulator roll raises the model's left side", () => {
  const roll = Math.atan(0.1);
  const { left } = orientedAxes({ yaw: -1.2, pitch: 0, roll });
  const horizontal = Math.hypot(left.x, left.z);
  assert.ok(Math.abs(left.y / horizontal - Math.tan(roll)) < EPSILON);
  assert.ok(left.y > 0);
});

test("combined yaw, pitch, and roll retain an orthonormal vehicle basis", () => {
  const { quaternion, forward, left, up } = orientedAxes({
    yaw: 2.31,
    pitch: Math.atan(0.1495),
    roll: Math.atan(-0.08),
  });
  assert.ok(Math.abs(quaternion.length() - 1) < EPSILON);
  assert.ok(Math.abs(forward.length() - 1) < EPSILON);
  assert.ok(Math.abs(left.length() - 1) < EPSILON);
  assert.ok(Math.abs(up.length() - 1) < EPSILON);
  assert.ok(Math.abs(forward.dot(left)) < EPSILON);
  assert.ok(Math.abs(forward.dot(up)) < EPSILON);
  assert.ok(Math.abs(left.dot(up)) < EPSILON);
  assert.ok(forward.y > 0, "uphill pitch must still raise the nose after yaw and roll");
  assert.ok(left.y < 0, "negative roll must lower the left side after yaw and pitch");
});

test("render attitude filtering damps road-plane chatter without delaying yaw", () => {
  const filter = new RenderAttitudeFilter(11);
  const initial = filter.update({ yaw: 0.2, pitch: 0.08, roll: -0.04 }, 1 / 60);
  assert.equal(initial.pitch, 0.08);
  assert.equal(initial.roll, -0.04);

  const reversed = filter.update({ yaw: 0.7, pitch: -0.08, roll: 0.04 }, 1 / 60);
  assert.equal(reversed.yaw, 0.7, "render yaw must remain authoritative");
  assert.ok(reversed.pitch > -0.08 && reversed.pitch < 0.08);
  assert.ok(reversed.roll > -0.04 && reversed.roll < 0.04);

  const snapped = filter.update({ yaw: -1, pitch: -0.12, roll: 0.09 }, 1 / 60, true);
  assert.deepEqual(
    { yaw: snapped.yaw, pitch: snapped.pitch, roll: snapped.roll },
    { yaw: -1, pitch: -0.12, roll: 0.09 },
  );
});

test("render height filtering rounds uphill steps while staying road-tight", () => {
  const filter = new RenderHeightFilter();
  const raw = [];
  const smooth = [];
  for (let frame = 0; frame < 120; frame += 1) {
    // Deliberately quantized 30 Hz climb sampled by a 60 Hz renderer.
    const target = 500 + Math.floor(frame / 2) * 0.035;
    raw.push(target);
    smooth.push(filter.update(target, 1 / 60));
  }
  const rawSteps = raw.slice(1).map((value, index) => Math.abs(value - raw[index]));
  const smoothSteps = smooth.slice(1).map((value, index) => Math.abs(value - smooth[index]));
  assert.ok(Math.max(...smoothSteps) < Math.max(...rawSteps));
  assert.ok(smooth.every((value, index) => Math.abs(value - raw[index]) <= 0.09 + EPSILON));
  assert.ok(Math.abs(smooth.at(-1) - raw.at(-1)) < 0.04, "predictive lead must avoid hill-climb lag");

  assert.equal(filter.update(610, 1 / 60, true), 610, "resets and teleports must snap");
});
