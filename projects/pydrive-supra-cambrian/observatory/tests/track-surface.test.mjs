import test from "node:test";
import assert from "node:assert/strict";

import { sampleTrackSurface } from "../src/track-surface.js";

test("visual road projection fills the height between authoritative samples", () => {
  const track = [
    [0, 0, 100],
    [3, 0, 101],
    [6, 0, 103],
    [9, 5, 106],
  ];
  const surface = sampleTrackSurface(track, [4.5, 0.8, 0], 0.5, 3);
  assert.ok(Math.abs(surface.height - 102) < 1e-12);
  assert.equal(surface.index, 1);
  assert.ok(Math.abs(surface.along - 0.5) < 1e-12);
  assert.ok(Math.abs(surface.pitch - Math.atan2(2, 3)) < 1e-12);
});

test("visual road projection wraps safely across the lap boundary", () => {
  const track = [
    [0, 0, 10],
    [3, 0, 11],
    [3, 3, 12],
    [0, 3, 13],
  ];
  const surface = sampleTrackSurface(track, [0, 1.5, 0], 0.99, 2);
  assert.equal(surface.index, 3);
  assert.ok(Math.abs(surface.height - 11.5) < 1e-12);
});
