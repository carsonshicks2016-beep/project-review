// 3D Viewer V2 — thin orchestrator: renderer, net, input, frame loop.
// All look/feel lives in v2/ modules (see VIEWER3D_V2_PLAN.md).
import * as THREE from "./vendor/three.module.min.js";
import { V2 } from "./v2/palette.js";
import { createMoodController } from "./v2/materials.js";
import { buildSky } from "./v2/sky.js";
import { buildWorld, disposeWorld } from "./v2/world.js";
import { sampleRoadAt, roadPitchAt } from "./v2/corridor.js";
import { createSupra } from "./v2/supra.js";
import { createCameraRig } from "./v2/camera.js";
import { createHud } from "./v2/hud.js";

const CAR_LIFT = 0.02; // road surface -> tyre contact fudge

const canvas = document.querySelector("#scene");
const hud = createHud();

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: true,
  powerPreference: "high-performance",
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.setSize(window.innerWidth, window.innerHeight, false);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(
  V2.fov,
  window.innerWidth / window.innerHeight,
  0.1,
  6000,
);
camera.position.set(-10, 4, 0);

const mood = createMoodController("night");
mood.bindNumber(renderer, "toneMappingExposure", "exposure");

// ── Lights (all mood-bound) ──
const hemi = new THREE.HemisphereLight(0xffffff, 0x000000, 1);
mood.bindColor(hemi, "color", "hemiSky");
mood.bindColor(hemi, "groundColor", "hemiGround");
mood.bindNumber(hemi, "intensity", "hemiIntensity");
scene.add(hemi);

const sun = new THREE.DirectionalLight(0xffffff, 1);
mood.bindColor(sun, "color", "sunColor");
mood.bindNumber(sun, "intensity", "sunIntensity");
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -90;
sun.shadow.camera.right = 90;
sun.shadow.camera.top = 90;
sun.shadow.camera.bottom = -90;
sun.shadow.camera.far = 700;
sun.shadow.bias = -0.0004;
scene.add(sun);
scene.add(sun.target);
// The sun/moon follows the car (shadow map stays focused); mood drives the
// offset direction only.
const sunRig = { offset: new THREE.Vector3(-120, 140, 60) };
mood.bindVec3(sunRig, "offset", "sunPos");

const ambient = new THREE.AmbientLight(0xffffff, 0.3);
mood.bindColor(ambient, "color", "ambient");
mood.bindNumber(ambient, "intensity", "ambientIntensity");
scene.add(ambient);

scene.fog = new THREE.FogExp2(0x1c2735, 0.0030);
mood.bindColor(scene.fog, "color", "fog");
mood.bindNumber(scene.fog, "density", "fogDensity");

const sky = buildSky(scene, mood);
const car = createSupra(mood);
scene.add(car.group);
// roll (X) innermost, then pitch (Z), then yaw (Y) — model-frame attitude
car.group.rotation.order = "YZX";
// the contact shadow lives in the WORLD, pinned to the road mesh — it stays
// behind on the asphalt when the car jumps (PHYSICS_3D_PLAN Stage 4)
car.group.remove(car.blob);
scene.add(car.blob);
const rig = createCameraRig(camera);

const state = {
  ws: null,
  connected: false,
  latest: null,
  world: null,
  keys: new Set(),
  auto: true,
  carTarget: new THREE.Vector3(0, 0.5, 0),
  carPos: new THREE.Vector3(0, 0.5, 0),
  yawTarget: 0,
  yaw: 0,
  pitchTarget: 0,
  pitch: 0,
  rollTarget: 0,
  roll: 0,
  roadY: 0, // road MESH height under the car (shadow anchor + air height)
  snapNext: true, // teleport car+camera on first state after load/reset
};

// URL params for screenshots/links: ?day=1 starts in day mode, ?track=akina
// picks the track.
const urlParams = new URLSearchParams(location.search);
if (urlParams.get("day") === "1") {
  mood.set("day", true);
}
const trackParam = urlParams.get("track");
if (trackParam) {
  // any named/special track works server-side; add missing ones to the picker
  if (![...hud.el.track.options].some((o) => o.value === trackParam)) {
    const opt = document.createElement("option");
    opt.value = trackParam;
    opt.textContent = trackParam;
    hud.el.track.appendChild(opt);
  }
  hud.el.track.value = trackParam;
}
// ?checkpoint=name.pt -> the AI drives (watch a champion, or point it at a
// TRAINING run's rolling checkpoint and watch the brain evolve live)
state.checkpoint = urlParams.get("checkpoint") || null;
hud.el.daynight.textContent = mood.current === "day" ? "☀ Day" : "☾ Night";
mood.refresh();

// ── Net ──
function connect() {
  if (state.ws) state.ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/3d/ws`);
  state.ws = ws;
  hud.setConn("connecting", "warn");

  ws.addEventListener("open", () => {
    state.connected = true;
    hud.setConn("connected", "good");
    startSession();
  });

  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "track") {
      onTrack(msg);
    } else if (msg.type === "state") {
      receiveState(msg);
    } else if (msg.type === "event" && msg.event === "reset") {
      state.snapNext = true;
    } else if (msg.type === "event" && msg.event === "notice") {
      hud.toast(msg.message || "");
    } else if (msg.type === "error") {
      hud.toast(msg.message || "3D viewer error");
    }
  });

  ws.addEventListener("close", () => {
    state.connected = false;
    hud.setConn("disconnected", "warn");
    setTimeout(() => {
      if (!state.connected && state.ws === ws) connect();
    }, 1200);
  });
}

function send(msg) {
  if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  state.ws.send(JSON.stringify(msg));
}

function startSession() {
  send({
    type: "start",
    car: hud.el.car.value,
    track: hud.el.track.value,
    seed: 7,
    ...(state.checkpoint ? { checkpoint: state.checkpoint } : {}),
  });
}

function onTrack(msg) {
  if (state.world) {
    scene.remove(state.world.group);
    disposeWorld(state.world, mood);
  }
  state.world = buildWorld(msg, mood);
  scene.add(state.world.group);
  mood.refresh();
  state.snapNext = true;
  hud.el.trackName.textContent = msg.meta?.name || "track";
  hud.toast(`loaded ${msg.meta?.name || "track"}`);
}

function receiveState(msg) {
  state.latest = msg;
  const p = msg.pose;
  // Sim: X fwd, Y left, Z up -> Three: (x, z, -y)
  state.carTarget.set(p.x, p.z + CAR_LIFT, -p.y);
  state.yawTarget = p.yaw;
  const airborne = !!(msg.vehicle && msg.vehicle.airborne);
  const corridor = state.world?.corridor;
  const sample = corridor
    ? sampleRoadAt(corridor, state.carTarget.x, state.carTarget.z)
    : null;
  state.roadY = sample ? sample.y : p.z;
  if (typeof p.pitch === "number") {
    // server-authoritative attitude (hills physics). Grounded cars drape on
    // the road MESH (smooth between samples); a flying car uses raw sim z.
    state.pitchTarget = p.pitch;
    state.rollTarget = p.roll || 0;
    if (!airborne && sample) state.carTarget.y = sample.y + CAR_LIFT;
  } else if (sample) {
    // legacy payloads: client-side road sampling
    state.carTarget.y = sample.y + CAR_LIFT;
    state.pitchTarget = roadPitchAt(
      corridor, state.carTarget.x, state.carTarget.z, state.yawTarget,
    );
  }
  hud.update(msg);
}

// ── Input ──
function inputPayload() {
  const k = state.keys;
  const left = k.has("ArrowLeft") || k.has("KeyA");
  const right = k.has("ArrowRight") || k.has("KeyD");
  return {
    type: "input",
    steer: (left ? 1 : 0) - (right ? 1 : 0),
    throttle: k.has("ArrowUp") || k.has("KeyW") ? 1 : 0,
    brake: k.has("ArrowDown") || k.has("KeyS") ? 1 : 0,
    handbrake: k.has("Space") ? 1 : 0,
    clutch: k.has("ShiftLeft") || k.has("ShiftRight") ? 0 : 1,
    shiftUp: k.has("KeyE"),
    shiftDown: k.has("KeyQ"),
    auto: state.auto,
  };
}

function keyAllowed(e) {
  const tag = ((e.target && e.target.tagName) || "").toLowerCase();
  return tag !== "select" && tag !== "input" && tag !== "button";
}

window.addEventListener("keydown", (e) => {
  if (!keyAllowed(e)) return;
  if (["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(e.code)) {
    e.preventDefault();
  }
  if (e.code === "KeyR") {
    send({ type: "reset" });
    hud.toast("reset");
  } else if (e.code === "KeyC") {
    hud.el.cameraMode.textContent = rig.cycle();
  } else if (e.code === "KeyT") {
    state.auto = !state.auto;
    hud.toast(state.auto ? "automatic" : "manual");
  } else if (e.code === "KeyL") {
    toggleMood();
  }
  state.keys.add(e.code);
});

window.addEventListener("keyup", (e) => {
  state.keys.delete(e.code);
});

setInterval(() => send(inputPayload()), 1000 / 30);

hud.el.restart.addEventListener("click", () => startSession());
hud.el.track.addEventListener("change", () => startSession());
hud.el.daynight.addEventListener("click", toggleMood);

function toggleMood() {
  const next = mood.toggle();
  hud.el.daynight.textContent = next === "day" ? "☀ Day" : "☾ Night";
  hud.toast(next === "day" ? "daytime" : "nighttime");
}

// ── Frame loop ──
function dampAngle(a, b, t) {
  let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
  if (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

let lastFrame = performance.now();
let fpsFrames = 0;
let fpsLast = performance.now();

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;

  mood.update(dt);

  if (state.latest) {
    const snap = state.snapNext;
    if (snap) {
      state.carPos.copy(state.carTarget);
      state.yaw = state.yawTarget;
      state.pitch = state.pitchTarget;
      state.roll = state.rollTarget;
      state.snapNext = false;
    } else {
      state.carPos.lerp(state.carTarget, 0.18);
      state.yaw = dampAngle(state.yaw, state.yawTarget, 0.18);
      state.pitch += (state.pitchTarget - state.pitch) * 0.16;
      state.roll += (state.rollTarget - state.roll) * 0.16;
    }
    car.group.position.copy(state.carPos);
    car.group.rotation.set(state.roll, state.yaw, state.pitch);

    // shadow stays on the asphalt; shrinks as the car gets air under it
    const lift = Math.max(0, state.carPos.y - CAR_LIFT - state.roadY);
    car.blob.position.set(state.carPos.x, state.roadY + 0.035, state.carPos.z);
    car.blob.rotation.y = state.yaw;
    const sh = 1 / (1 + 0.5 * lift);
    car.blob.scale.set(sh, 1, sh);

    const steer = state.latest.vehicle.steerAngle || 0;
    const spin = state.latest.vehicle.wheelSpin || 0;
    for (const w of car.wheels) {
      w.rotation.y = w.userData.steer ? steer : 0;
      if (w.userData.roll) w.userData.roll.rotation.z = -spin;
    }

    rig.update({
      carPos: state.carPos,
      yaw: state.yaw,
      speed: state.latest.vehicle.speed,
      time: state.latest.time,
      snap,
    });

    if (state.world && state.world.update) state.world.update(state.carPos);
  }

  // Sun/moon and sky follow the car/camera
  sun.position.copy(state.carPos).add(sunRig.offset);
  sun.target.position.copy(state.carPos);
  sky.position.copy(camera.position);

  fpsFrames++;
  if (now - fpsLast >= 1000) {
    hud.setFps(Math.round((fpsFrames * 1000) / (now - fpsLast)));
    fpsFrames = 0;
    fpsLast = now;
  }

  renderer.render(scene, camera);
}

function resize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
}
window.addEventListener("resize", resize);

connect();
animate();
