import test from "node:test";
import assert from "node:assert/strict";

import { audioPerspectiveForShot, CAMERA_MODES } from "../src/cameras.js";
import { encodeControl } from "../src/protocol.js";

test("only the roof camera selects the close onboard acoustic perspective", () => {
  assert.equal(audioPerspectiveForShot("roof"), "onboard");
  for (const shot of [
    ...CAMERA_MODES.filter((mode) => mode !== "roof"),
    "trackside-long", "trackside-close", "chase-low", "apex", "helicopter",
  ]) assert.equal(audioPerspectiveForShot(shot), "external", shot);
});

test("listener controls preserve scene-acoustic metadata", () => {
  const listener = {
    x: 1, y: 2, z: 3, vx: 4, vy: 5, yaw: 0.6,
    camera: "trackside-close", perspective: "external", cut: 9, weather: "dusk",
  };
  assert.deepEqual(JSON.parse(encodeControl("listener", listener)), {
    type: "listener", listener,
  });
});
