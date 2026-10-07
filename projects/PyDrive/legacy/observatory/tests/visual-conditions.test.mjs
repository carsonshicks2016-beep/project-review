import test from "node:test";
import assert from "node:assert/strict";
import { VISUAL_CONDITIONS, resolveVisualCondition } from "../src/visual-conditions.js";
import { tyreSprayStrength } from "../src/precipitation.js";

test("visual conditions are presentation-only combinations with rain always wet", () => {
  assert.equal(Object.keys(VISUAL_CONDITIONS).length, 9);
  for (const condition of Object.values(VISUAL_CONDITIONS)) {
    assert.ok(["day", "dusk", "night"].includes(condition.lighting));
    assert.ok(["clear", "overcast", "rain"].includes(condition.precipitation));
    if (condition.precipitation === "rain") assert.equal(condition.surface, "wet");
  }
  assert.equal(resolveVisualCondition("dry").id, "clear-day");
  assert.equal(resolveVisualCondition("dusk").id, "clear-dusk");
  assert.equal(resolveVisualCondition("unknown").id, "clear-day");
});

test("tyre spray remains a wet visual response with a speed threshold", () => {
  assert.equal(tyreSprayStrength(resolveVisualCondition("clear-day"), 80), 0);
  assert.equal(tyreSprayStrength(resolveVisualCondition("rain-day"), 4), 0);
  assert.ok(tyreSprayStrength(resolveVisualCondition("wet-dusk"), 36) > 0.25);
  assert.ok(tyreSprayStrength(resolveVisualCondition("rain-night"), 52) > 0.8);
});
