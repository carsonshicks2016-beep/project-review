import test from "node:test";
import assert from "node:assert/strict";

import {
  PROTOCOL_VERSION,
  classifyMessage,
  encodeControl,
  normalizeFrame,
  normalizeIdentity,
  sessionIdFromLocation,
  socketUrl,
  vehicleFromIdentity,
} from "../src/protocol.js";
import { browserToken, createObservatorySession, ObservatoryClient, releaseObservatorySession } from "../src/observatory-client.js";
import { AdaptiveQuality, QUALITY_LEVELS } from "../src/quality.js";
import { audioSocketPath, EngineAudioStream } from "../src/audio-stream.js";

const checkpoint = {
  checkpoint_id: "best-lap",
  filename: "fable5_919_best.pt",
  edition: "919-evo",
  car: "porsche_919evo",
  stage: "frontier",
  drivetrain: "919evo-v2",
  compatibility_class: "fable-playback",
  observation_layout: "fable-v2",
  classification: "current",
  policy_sha256: "a".repeat(64),
  warnings: ["Legacy 919 authority warning"],
};

test("session paths and protocol use the production contract", () => {
  const location = { protocol: "https:", host: "pit.local", pathname: "/observatory/abc-123", search: "" };
  assert.equal(PROTOCOL_VERSION, "fable-observatory-v1");
  assert.equal(socketUrl("/api/observatory/ws/abc-123", location), "wss://pit.local/api/observatory/ws/abc-123");
  assert.equal(sessionIdFromLocation(location), "abc-123");
  assert.equal(vehicleFromIdentity({ checkpoint: { car: "mazda787b" } }), "mazda787b");
  assert.equal(vehicleFromIdentity({ checkpoint: { car: "porsche_919evo" } }), "porsche_919evo");
  assert.throws(() => vehicleFromIdentity({ car: "../query-invented-car" }), /server did not provide/i);
});

test("hello checkpoint identity never drops policy provenance", () => {
  assert.deepEqual(normalizeIdentity({ protocol: PROTOCOL_VERSION, checkpoint, track: { hash: "track-hash" } }), {
    car: "porsche_919evo",
    checkpoint: "fable5_919_best.pt",
    policyHash: "a".repeat(64),
    classification: "current",
    stage: "frontier",
    drivetrain: "919evo-v2",
    edition: "919-evo",
    compatibility: "fable-playback",
    observationLayout: "fable-v2",
    warnings: ["Legacy 919 authority warning"],
    trackHash: "track-hash",
    protocol: PROTOCOL_VERSION,
  });
});

test("authoritative nested telemetry maps into renderer state", () => {
  const frame = normalizeFrame({
    schema: "fable-observatory-frame-v1",
    sequence: 91,
    sim_time: 12.5,
    episode: 2,
    episode_time: 4.25,
    checkpoint,
    vehicle: {
      pose: { x: 10, y: -4, z: 488, yaw: 0.3, pitch: 0.02, roll: -0.01 },
      dynamics: { speed_mps: 71 },
      powertrain: { rpm: 8300, gear: 6, hybrid_soc: 0.64, mgu_power_w: 225000, boost: 0.7, active_aero_engaged: true, active_aero_low_drag: 0.8 },
      geometry: { collision_length_m: 4.65, collision_width_m: 1.9, footprint_world_xy: [[8, -5], [12, -5], [12, -3], [8, -3]] },
      wheels: ["FL", "FR", "RL", "RR"].map((id, index) => ({ id, rotation_rad: index + 0.5, steer_rad: index < 2 ? 0.08 : 0, load_n: 2000 + index, grip: 0.95, slip_ratio: 0.04, slip_angle_rad: 0.02, contact: true })),
    },
    controls: { steer: 0.2, longitudinal: 0.7, throttle: 0.7, brake: 0, gear_offset: 0.1 },
    track: { progress: 0.42, off_track: false },
    fable: { stage: "frontier", pace_ratio: 1.04, valid: true, footprint_valid: true },
    observations: { normalized_vector: [0.1, -0.3, 0.8], groups: { rays: { name: "Rays", values: [{ label: "ray_0", policy_value: 0.1 }] } } },
    brain: {
      output_means: [-0.12, 0.9, 0.02],
      hidden_layers: [[0.1, -0.2], [0.8, 0.4]],
      policy_std: [0.12, 0.23, 0.34],
      critic_value: 8.2,
      predicted_path: [{ x: 10, y: -4, z: 488 }, { x: 14, y: -2, z: 488.2 }],
      sensitivity: { label: "Local finite-difference sensitivity, not intent" },
    },
    sensor_geometry: { beam_origin: [10, -4, 488], beam_points: [[18, -8, 488], [21, -1, 488]] },
    playback: { paused: true, speed: 0.5, scrubbing: true },
  });
  assert.equal(frame.seq, 91);
  assert.deepEqual(frame.vehicle.position, [10, -4, 488]);
  assert.deepEqual(frame.vehicle.wheelRotation, [0.5, 1.5, 2.5, 3.5]);
  assert.equal(frame.vehicle.wheels[0].loadN, 2000);
  assert.equal(frame.vehicle.collisionLength, 4.65);
  assert.equal(frame.vehicle.footprint.length, 4);
  assert.equal(frame.telemetry.speedMps, 71);
  assert.equal(frame.telemetry.mguPower, 225000);
  assert.deepEqual(frame.brain.actions, [-0.12, 0.9, 0.02]);
  assert.deepEqual(frame.brain.hiddenLayers, [[0.1, -0.2], [0.8, 0.4]]);
  assert.deepEqual(frame.brain.policyStd, [0.12, 0.23, 0.34]);
  assert.equal(frame.brain.observationGroups[0].name, "Rays");
  assert.equal(frame.brain.rays.length, 2);
  assert.deepEqual(frame.brain.predictedPath[1], [14, -2, 488.2]);
  assert.equal(frame.replay.live, false);
  assert.equal(frame.identity.policyHash, "a".repeat(64));
  assert.equal(classifyMessage({ type: "frame", frame: { checkpoint, vehicle: { pose: { x: 1 } } } }).type, "frame");
});

test("grounded rendering follows current road height while airborne z stays ballistic", () => {
  const grounded = normalizeFrame({
    checkpoint,
    vehicle: {
      pose: { x: 1, y: 2, z: 499.72 },
      dynamics: { airborne: false },
    },
    track: { road_z_m: 500.0 },
  });
  assert.equal(grounded.vehicle.position[2], 500.0);
  assert.equal(grounded.vehicle.roadZ, 500.0);
  assert.equal(grounded.vehicle.airborne, false);

  const airborne = normalizeFrame({
    checkpoint,
    vehicle: {
      pose: { x: 1, y: 2, z: 503.5 },
      dynamics: { airborne: true },
    },
    track: { road_z_m: 500.0 },
  });
  assert.equal(airborne.vehicle.position[2], 503.5);
  assert.equal(airborne.vehicle.airborne, true);
});

test("controls match FablePlaybackSession.handle_control wire messages", () => {
  assert.deepEqual(JSON.parse(encodeControl("pause")), { type: "pause" });
  assert.deepEqual(JSON.parse(encodeControl("resume")), { type: "resume" });
  assert.deepEqual(JSON.parse(encodeControl("step")), { type: "step" });
  assert.deepEqual(JSON.parse(encodeControl("speed", 2)), { type: "speed", speed: 2 });
  assert.deepEqual(JSON.parse(encodeControl("scrub", 18.5)), { type: "scrub", seconds_ago: 18.5 });
});

test("user playback intent survives a brief telemetry reconnect", () => {
  const client = new ObservatoryClient("session-7", { WebSocketImpl: { OPEN: 1 } });
  client.send("pause");
  client.send("speed", 2);
  client.send("listener", { x: 1, y: 2, z: 3 });
  assert.deepEqual(client.pendingControls.map((payload) => JSON.parse(payload)), [
    { type: "pause" },
    { type: "speed", speed: 2 },
  ]);
});

test("session lease uses one stable browser token and server-returned sockets", async () => {
  const values = new Map();
  const storage = { getItem: (key) => values.get(key), setItem: (key, value) => values.set(key, value) };
  const originalStorage = globalThis.localStorage;
  globalThis.localStorage = storage;
  const first = browserToken(storage, { randomUUID: () => "browser-token-1" });
  assert.equal(browserToken(storage, { randomUUID: () => "different" }), first);
  const calls = [];
  const fakeFetch = async (url, options) => {
    calls.push({ url, options });
    return {
      ok: true,
      json: async () => ({ session_id: "session-7", track_hash: "track-7", telemetry_socket: "/api/observatory/ws/session-7", audio_socket: "/api/observatory/audio/session-7" }),
    };
  };
  const session = await createObservatorySession({ search: "?edition=919&checkpoint=best&mode=replay&car=query-invention" }, fakeFetch);
  assert.equal(session.sessionId, "session-7");
  assert.equal(session.wsUrl, "/api/observatory/ws/session-7");
  assert.equal(session.audioUrl, "/api/observatory/audio/session-7");
  assert.equal(session.trackHash, "track-7");
  assert.equal(calls[0].options.headers["X-Observatory-Browser"], first);
  assert.deepEqual(JSON.parse(calls[0].options.body), {
    edition: "919", checkpoint: "best", mode: "replay",
  });
  releaseObservatorySession(session, fakeFetch);
  assert.equal(calls[1].url, "/api/observatory/sessions/session-7");
  assert.equal(calls[1].options.method, "DELETE");
  globalThis.localStorage = originalStorage;
});

test("adaptive fallback changes rendering only through declared visual knobs", () => {
  const changes = [];
  const quality = new AdaptiveQuality((name, settings) => changes.push([name, settings]));
  for (let index = 0; index < 240; index += 1) quality.observe(45);
  assert.ok(changes.length >= 1);
  assert.equal(changes[0][0], "medium");
  assert.deepEqual(Object.keys(QUALITY_LEVELS.medium).sort(), ["forestDensity", "pixelRatioCap", "shadows"]);
});

test("audio uses the server route and surfaces sustained jitter underflow", () => {
  assert.equal(audioSocketPath("session 7"), "/api/observatory/audio/session%207");
  assert.equal(audioSocketPath("session-7", "/custom/audio"), "/custom/audio");
  const stream = new EngineAudioStream("session-7");
  const statuses = [];
  stream.addEventListener("status", (event) => statuses.push(event.detail));
  stream.handleWorkletStatus({ type: "underrun" });
  assert.equal(statuses.length, 0, "muted audio ignores stale worklet status");
  stream.enabled = true;
  stream.handleWorkletStatus({ type: "underrun" });
  stream.handleWorkletStatus({ type: "underrun" });
  stream.handleWorkletStatus({ type: "underrun" });
  assert.equal(statuses.at(-1).degraded, true);
  stream.handleWorkletStatus({ type: "recovered" });
  assert.equal(statuses.at(-1).degraded, false);
});
