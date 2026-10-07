import test from "node:test";
import assert from "node:assert/strict";

import {
  approvedShotForProgress,
  audioPerspectiveForShot,
  CAMERA_MODES,
  normalizeEditorialShot,
  normalizeEditorialShots,
} from "../src/cameras.js";
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

test("editorial shots are bounded, local presentation contracts", () => {
  const shot = normalizeEditorialShot({
    id: "crest <approved>", label: "Crest <North>", approved: true,
    anchorProgress: 1.14, coverage: 0.4, side: "right", offset: 100,
    height: 0, lookAhead: 90, fov: 8, foreground: "foliage",
  });
  assert.equal(shot.id, "crest-approved");
  assert.equal(shot.label, "Crest North");
  assert.ok(Math.abs(shot.anchorProgress - 0.14) < 1e-9);
  assert.equal(shot.coverage, 0.08);
  assert.equal(shot.side, "right");
  assert.equal(shot.offset, 32);
  assert.equal(shot.height, 1.35);
  assert.equal(shot.lookAhead, 24);
  assert.equal(shot.fov, 24);
  assert.equal(shot.foreground, "foliage");
});

test("approved editorial shots win only inside their wrapped track coverage", () => {
  const shots = normalizeEditorialShots([
    { id: "finish", label: "FINISH", approved: true, anchorProgress: 0.99, coverage: 0.02 },
    { id: "draft", label: "DRAFT", approved: false, anchorProgress: 0.5, coverage: 0.08 },
  ]);
  assert.equal(approvedShotForProgress(shots, 0.005)?.id, "finish");
  assert.equal(approvedShotForProgress(shots, 0.5), null);
  assert.equal(approvedShotForProgress(shots, 0.7), null);
});
