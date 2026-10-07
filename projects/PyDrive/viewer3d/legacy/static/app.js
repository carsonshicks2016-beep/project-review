import * as THREE from "./vendor/three.module.min.js";
import {
  MOUNTAIN_PASS,
  PALETTE,
  buildSkyDome,
  createMountainPassMaterials,
  createMountainPassPost,
  createMountainPassTextures,
  setTimeOfDay,
  setupMountainPassScene,
} from "./mountain-pass-style.js?v=stage1-corridor11";
import {
  buildMountainPassTrack,
  roadHeightAt,
  roadPitchAt,
  sampleRoadAt,
} from "./mountain-pass-world.js?v=stage1-corridor11";

const $ = (s) => document.querySelector(s);
const canvas = $("#scene");

const ui = {
  conn: $("#conn"),
  trackName: $("#track-name"),
  cameraMode: $("#camera-mode"),
  speed: $("#speed"),
  rpm: $("#rpm"),
  gear: $("#gear"),
  progress: $("#progress"),
  throttle: $("#throttle"),
  brake: $("#brake"),
  steer: $("#steer"),
  car: $("#car"),
  track: $("#track"),
  restart: $("#restart"),
  toast: $("#toast"),
  daynight: $("#daynight"),
};

const state = {
  ws: null,
  latest: null,
  connected: false,
  cameraMode: "chase",
  auto: true,
  keys: new Set(),
  carTarget: new THREE.Vector3(),
  carPos: new THREE.Vector3(),
  yawTarget: 0,
  yaw: 0,
  roadGroup: new THREE.Group(),
  roadsideGroup: new THREE.Group(),
  car: null,
  headlightPool: null,
  headlightWallWash: null,
  carContactShadow: null,
  tailRoadGlow: null,
  wheels: [],
  trackProfile: null,
  pitchTarget: 0,
  pitch: 0,
  isDay: false,
  fogPlanes: [],
  skyDome: null,
  lighting: null,
};

const textures = createMountainPassTextures();
const mats = createMountainPassMaterials(textures);

const scene = new THREE.Scene();
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, powerPreference: "high-performance" });
renderer.setPixelRatio(1);
renderer.setSize(window.innerWidth, window.innerHeight, false);
state.lighting = setupMountainPassScene(scene, renderer);
state.skyDome = buildSkyDome(scene);

const camera = new THREE.PerspectiveCamera(MOUNTAIN_PASS.cameraFov, window.innerWidth / window.innerHeight, 0.1, 5000);
camera.position.set(-8, 5, 8);

const post = createMountainPassPost();
resizeRenderer();

scene.add(state.roadGroup);
scene.add(state.roadsideGroup);
state.car = createCarModel("supra");
scene.add(state.car);
state.headlightPool = createHeadlightPool();
scene.add(state.headlightPool);
state.headlightWallWash = createWallWashPlane(mats.headlightWallWash);
scene.add(state.headlightWallWash);
state.carContactShadow = createGroundCircle(mats.carContactShadow, 24);
scene.add(state.carContactShadow);
state.tailRoadGlow = createGroundCircle(mats.tailRoadGlow, 18);
scene.add(state.tailRoadGlow);

function toast(msg) {
  ui.toast.textContent = msg;
  ui.toast.classList.add("show");
  clearTimeout(ui.toast._t);
  ui.toast._t = setTimeout(() => ui.toast.classList.remove("show"), 2200);
}

function setConn(label, cls = "") {
  ui.conn.textContent = label;
  ui.conn.className = `pill ${cls}`;
}

// Sim: X forward, Y left, yaw positive left. Three: X forward, Y up, Z mirrored.
function simPointToThree(p) {
  return new THREE.Vector3(p[0], p[2], -p[1]);
}

function simPoseToThree(p) {
  return new THREE.Vector3(
    p.x,
    p.z + MOUNTAIN_PASS.roadSurfaceLift + MOUNTAIN_PASS.carSurfaceOffset,
    -p.y
  );
}

function simYawToThreeYaw(yaw) {
  return yaw;
}

function connect() {
  if (state.ws) state.ws.close();
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/api/3d/ws`);
  state.ws = ws;
  setConn("connecting", "warn");

  ws.addEventListener("open", () => {
    state.connected = true;
    setConn("connected", "good");
    startSession();
  });

  ws.addEventListener("message", (event) => {
    const msg = JSON.parse(event.data);
    if (msg.type === "track") {
      buildTrack(msg);
      if (msg.meta && msg.meta.car) swapCarModel(msg.meta.car);
      ui.trackName.textContent = msg.meta.name || "track";
      toast(`loaded ${msg.meta.name || "track"}`);
    } else if (msg.type === "state") {
      receiveState(msg);
    } else if (msg.type === "event" && msg.event === "mute") {
      toast(msg.muted ? "audio muted" : "audio unmuted");
    } else if (msg.type === "error") {
      toast(msg.message || "3D viewer error");
    }
  });

  ws.addEventListener("close", () => {
    state.connected = false;
    setConn("disconnected", "warn");
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
  const urlParams = new URLSearchParams(window.location.search);
  const trackName = urlParams.get("track") || ui.track.value;
  const checkpoint = urlParams.get("checkpoint");

  if (ui.track && trackName && [...ui.track.options].some(o => o.value === trackName)) {
      ui.track.value = trackName;
  }

  send({
    type: "start",
    car: ui.car ? ui.car.value : "supra",
    track: trackName,
    seed: 7,
    checkpoint: checkpoint
  });
}

function buildTrack(data) {
  state.roadGroup.clear();
  state.roadsideGroup.clear();

  const left = data.left.map(simPointToThree);
  const right = data.right.map(simPointToThree);
  const center = data.center.map(simPointToThree);
  const result = buildMountainPassTrack({
    roadGroup: state.roadGroup,
    roadsideGroup: state.roadsideGroup,
    center,
    left,
    right,
    mats,
    surfaceLift: MOUNTAIN_PASS.roadSurfaceLift,
  });
  state.trackProfile = result.profile;
  state.fogPlanes = result.fogPlanes || [];
  state.pitchTarget = 0;
  state.pitch = 0;

  // Re-apply current day/night mode to the new fog planes
  setTimeOfDay(state.isDay, {
    skyDome: state.skyDome,
    lighting: state.lighting,
    scene,
    post,
    fogPlanes: state.fogPlanes,
    mats,
  });
}

// ── Car model dispatcher ────────────────────────────────────────────── //
function createCarModel(name) {
  if (name === "skyline") return createSkyline();
  return createSupra(); // default fallback for supra, rx7, etc.
}

function swapCarModel(name) {
  if (state.currentCarName === name) return; // already the right model
  state.currentCarName = name;
  const pos = state.car.position.clone();
  const rot = state.car.rotation.clone();
  scene.remove(state.car);
  state.car = createCarModel(name);
  state.car.position.copy(pos);
  state.car.rotation.copy(rot);
  scene.add(state.car);
}

// ── Supra A80 ───────────────────────────────────────────────────────── //
function createSupra() {
  const group = new THREE.Group();
  group.name = "procedural-supra";

  const red = mats.carBody || mats.body;
  const redDark = mats.carBodyDark || mats.bodyDark;
  const glass = mats.carGlass || mats.glass;
  const black = mats.carTrim || mats.black;
  const tire = mats.carTire || mats.tire;
  const rim = mats.carRim || mats.rim;
  const lamp = mats.lamp;
  const tail = mats.tail;
  const highlight = mats.carHighlight || red;

  addBox(group, [0.05, 0.56, 0], [4.55, 0.44, 1.78], red);
  addBox(group, [0.40, 0.80, 0.56], [2.30, 0.026, 0.16], highlight, false);
  addBox(group, [0.40, 0.80, -0.56], [2.30, 0.026, 0.16], highlight, false);
  addBox(group, [0.35, 0.85, 0], [2.65, 0.20, 1.64], redDark);
  addBox(group, [0.82, 0.78, 0], [1.8, 0.18, 1.62], red);
  addBox(group, [0.94, 0.91, 0.42], [0.95, 0.022, 0.12], highlight, false);
  addBox(group, [0.94, 0.91, -0.42], [0.95, 0.022, 0.12], highlight, false);
  addBox(group, [-0.48, 1.12, 0], [1.62, 0.70, 1.36], glass);
  addBox(group, [-1.45, 0.92, 0], [0.98, 0.18, 1.58], red);
  addBox(group, [2.36, 0.48, 0], [0.18, 0.28, 1.72], black);
  addBox(group, [2.50, 0.62, 0.54], [0.07, 0.18, 0.38], lamp, false);
  addBox(group, [2.50, 0.62, -0.54], [0.07, 0.18, 0.38], lamp, false);
  addBox(group, [2.53, 0.50, 0.0], [0.05, 0.10, 1.15], black, false);
  addBox(group, [-2.32, 0.58, 0.56], [0.08, 0.20, 0.42], tail, false);
  addBox(group, [-2.32, 0.58, -0.56], [0.08, 0.20, 0.42], tail, false);
  addWing(group, black);

  const headGlow = new THREE.PointLight(PALETTE.amber, 1.65, 28, 1.85);
  headGlow.position.set(2.35, 0.82, 0);
  group.add(headGlow);
  const tailGlow = new THREE.PointLight(PALETTE.tail, 0.55, 11, 2.1);
  tailGlow.position.set(-2.22, 0.72, 0);
  group.add(tailGlow);

  attachWheels(group, [
    [1.33, 0.39, 0.92, true],
    [1.33, 0.39, -0.92, true],
    [-1.35, 0.39, 0.92, false],
    [-1.35, 0.39, -0.92, false],
  ]);

  finalizeCar(group);
  return group;
}

// ── Nissan Skyline GT-R R34 ─────────────────────────────────────────── //
function createSkyline() {
  const group = new THREE.Group();
  group.name = "procedural-skyline";

  // R34 palette: Bayside Blue body, dark metallic accents
  const blueMat = new THREE.MeshLambertMaterial({ color: 0x1a4a8a, emissive: 0x071e3f, emissiveIntensity: 0.45 });
  const blueDark = new THREE.MeshLambertMaterial({ color: 0x0e2e5a, emissive: 0x050e22, emissiveIntensity: 0.40 });
  const glass = mats.carGlass || mats.glass;
  const black = mats.carTrim || mats.black;
  const tire = mats.carTire || mats.tire;
  const rim = mats.carRim || mats.rim;
  const lamp = mats.lamp;
  const tail = mats.tail;
  const silver = new THREE.MeshLambertMaterial({ color: 0xb8bcc0, emissive: 0x3a3e42, emissiveIntensity: 0.35 });

  // ── Main body: R34 is boxier and slightly shorter than the Supra ───
  // Lower body slab
  addBox(group, [0.0, 0.52, 0], [4.60, 0.42, 1.82], blueMat);
  // Fender flares — R34 GT-R has wider rear fenders
  addBox(group, [1.15, 0.52, 0.88], [1.05, 0.38, 0.14], blueMat);
  addBox(group, [1.15, 0.52, -0.88], [1.05, 0.38, 0.14], blueMat);
  addBox(group, [-1.20, 0.52, 0.92], [1.15, 0.40, 0.16], blueMat); // wider rear
  addBox(group, [-1.20, 0.52, -0.92], [1.15, 0.40, 0.16], blueMat);

  // Shoulder line — boxy R34 crease
  addBox(group, [0.30, 0.82, 0], [2.80, 0.22, 1.68], blueDark);
  addBox(group, [0.75, 0.76, 0], [1.90, 0.18, 1.72], blueMat);

  // Hood — flatter and boxier than Supra
  addBox(group, [1.55, 0.78, 0], [1.40, 0.10, 1.58], blueMat);
  // Hood vents (NACA ducts)
  addBox(group, [1.50, 0.84, 0.28], [0.42, 0.025, 0.14], black, false);
  addBox(group, [1.50, 0.84, -0.28], [0.42, 0.025, 0.14], black, false);

  // Cabin — R34's greenhouse is upright and boxy
  addBox(group, [-0.30, 1.08, 0], [1.75, 0.65, 1.42], glass);
  // C-pillar thick
  addBox(group, [-1.05, 1.00, 0.60], [0.32, 0.48, 0.12], blueMat);
  addBox(group, [-1.05, 1.00, -0.60], [0.32, 0.48, 0.12], blueMat);

  // Rear deck / trunk — flat and wide
  addBox(group, [-1.55, 0.88, 0], [1.10, 0.20, 1.66], blueMat);

  // ── Front fascia — multi-element headlights ───
  addBox(group, [2.34, 0.46, 0], [0.16, 0.26, 1.76], black);
  // Headlights — R34's projectors
  addBox(group, [2.42, 0.60, 0.58], [0.08, 0.16, 0.34], lamp, false);
  addBox(group, [2.42, 0.60, -0.58], [0.08, 0.16, 0.34], lamp, false);
  // Bumper indicator lamps
  addBox(group, [2.44, 0.44, 0.72], [0.05, 0.10, 0.14], lamp, false);
  addBox(group, [2.44, 0.44, -0.72], [0.05, 0.10, 0.14], lamp, false);
  // Front splitter / lip
  addBox(group, [2.42, 0.32, 0], [0.08, 0.04, 1.72], black, false);
  // Intercooler peeking through
  addBox(group, [2.38, 0.40, 0], [0.04, 0.12, 0.90], silver, false);

  // ── Rear — the iconic quad circular taillights ───
  // Tail panel
  addBox(group, [-2.32, 0.62, 0], [0.06, 0.24, 1.74], black);
  // Quad taillights (2 per side, circular look via small boxes)
  for (const side of [1, -1]) {
    addBox(group, [-2.36, 0.66, side * 0.42], [0.06, 0.14, 0.14], tail, false);
    addBox(group, [-2.36, 0.66, side * 0.60], [0.06, 0.14, 0.14], tail, false);
  }
  // Rear diffuser
  addBox(group, [-2.34, 0.38, 0], [0.08, 0.06, 1.20], black, false);
  // Exhaust tips — twin pipes
  addBox(group, [-2.40, 0.38, 0.30], [0.10, 0.08, 0.08], silver, false);
  addBox(group, [-2.40, 0.38, -0.30], [0.10, 0.08, 0.08], silver, false);

  // ── R34 GT wing — taller and chunkier than the Supra's ───
  const wingMat = black;
  const wingBlade = addBox(group, [-2.10, 1.12, 0], [0.16, 0.07, 1.44], wingMat);
  wingBlade.rotation.z = -0.05;
  // Wing endplates
  addBox(group, [-2.10, 1.08, 0.72], [0.22, 0.14, 0.03], wingMat);
  addBox(group, [-2.10, 1.08, -0.72], [0.22, 0.14, 0.03], wingMat);
  // Wing mounts (swan-neck style)
  addBox(group, [-1.95, 0.92, 0.48], [0.06, 0.28, 0.06], wingMat);
  addBox(group, [-1.95, 0.92, -0.48], [0.06, 0.28, 0.06], wingMat);

  // ── Side skirts ───
  addBox(group, [0.0, 0.30, 0.90], [3.60, 0.06, 0.06], black, false);
  addBox(group, [0.0, 0.30, -0.90], [3.60, 0.06, 0.06], black, false);

  // ── Lights ───
  const headGlow = new THREE.PointLight(0xffeedd, 1.55, 26, 1.85);
  headGlow.position.set(2.35, 0.80, 0);
  group.add(headGlow);
  const tailGlow = new THREE.PointLight(PALETTE.tail, 0.65, 12, 2.0);
  tailGlow.position.set(-2.28, 0.72, 0);
  group.add(tailGlow);

  // ── Wheels — wider track than Supra, bigger rubber ───
  attachWheels(group, [
    [1.33, 0.39, 0.95, true],
    [1.33, 0.39, -0.95, true],
    [-1.33, 0.39, 0.98, false],  // wider rear track
    [-1.33, 0.39, -0.98, false],
  ]);

  finalizeCar(group);
  return group;
}

// ── Shared wheel + finalize helpers ─────────────────────────────────── //
function attachWheels(group, wheelPositions) {
  const tire = mats.carTire || mats.tire;
  const rim = mats.carRim || mats.rim;
  state.wheels = wheelPositions.map(([x, y, z, steer]) => {
    const pivot = new THREE.Group();
    pivot.position.set(x, y, z);
    pivot.userData.steer = steer;
    const roll = createWheelAssembly(tire, rim);
    pivot.userData.roll = roll;
    pivot.add(roll);
    group.add(pivot);
    return pivot;
  });
}

function finalizeCar(group) {
  group.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = !obj.userData.noCastShadow;
      obj.receiveShadow = false;
    }
  });
}

function createWheelAssembly(tireMat, rimMat) {
  const roll = new THREE.Group();
  const treadMat = mats.carTread || mats.tread;

  const tireGeo = new THREE.CylinderGeometry(0.38, 0.38, 0.30, 32);
  tireGeo.rotateX(Math.PI / 2);
  const tire = new THREE.Mesh(tireGeo, tireMat);
  tire.castShadow = true;
  roll.add(tire);

  const sidewallGeo = new THREE.TorusGeometry(0.305, 0.018, 8, 32);
  for (const z of [-0.166, 0.166]) {
    const sidewall = new THREE.Mesh(sidewallGeo, rimMat);
    sidewall.position.z = z;
    roll.add(sidewall);
  }

  const rimGeo = new THREE.CylinderGeometry(0.22, 0.22, 0.322, 18);
  rimGeo.rotateX(Math.PI / 2);
  const rim = new THREE.Mesh(rimGeo, rimMat);
  roll.add(rim);

  for (let i = 0; i < 16; i++) {
    const angle = i * Math.PI * 2 / 16;
    const tread = new THREE.Mesh(new THREE.BoxGeometry(0.055, 0.15, 0.34), treadMat);
    tread.position.set(Math.cos(angle) * 0.36, Math.sin(angle) * 0.36, 0);
    tread.rotation.z = angle;
    roll.add(tread);
  }

  for (const z of [-0.184, 0.184]) {
    for (let i = 0; i < 5; i++) {
      const spoke = new THREE.Mesh(new THREE.BoxGeometry(0.31, 0.045, 0.035), rimMat);
      spoke.position.z = z;
      spoke.rotation.z = i * Math.PI * 2 / 5;
      roll.add(spoke);
    }
  }

  return roll;
}

function addBox(parent, pos, scale, mat, cast = true) {
  const mesh = new THREE.Mesh(new THREE.BoxGeometry(scale[0], scale[1], scale[2]), mat);
  mesh.position.set(pos[0], pos[1], pos[2]);
  mesh.castShadow = cast;
  mesh.userData.noCastShadow = !cast;
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function addWing(parent, mat) {
  const blade = addBox(parent, [-2.08, 1.04, 0], [0.13, 0.065, 1.34], mat);
  blade.rotation.z = -0.045;
  addBox(parent, [-1.94, 0.86, 0.51], [0.075, 0.31, 0.07], mat);
  addBox(parent, [-1.94, 0.86, -0.51], [0.075, 0.31, 0.07], mat);
}

function createHeadlightPool() {
  const stations = [3.0, 4.8, 7.2, 10.2, 13.2, 16.4];
  const positions = new Float32Array(stations.length * 2 * 3);
  const indices = [];
  for (let i = 0; i < stations.length - 1; i++) {
    const a = i * 2;
    const b = a + 1;
    const c = a + 2;
    const d = a + 3;
    indices.push(a, c, b, b, c, d);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geo.setIndex(indices);
  geo.userData.stations = stations;
  geo.getAttribute("position").setUsage(THREE.DynamicDrawUsage);
  const mesh = new THREE.Mesh(geo, mats.headlightPool);
  mesh.frustumCulled = false;
  mesh.renderOrder = 5;
  mesh.visible = false;
  return mesh;
}

function createGroundCircle(material, segments = 18) {
  const geo = new THREE.CircleGeometry(1, segments);
  geo.rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geo, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 7;
  mesh.visible = false;
  return mesh;
}

function createGroundPlane(material) {
  const geo = new THREE.PlaneGeometry(1, 1, 1, 1);
  geo.rotateX(-Math.PI / 2);
  const mesh = new THREE.Mesh(geo, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 7;
  mesh.visible = false;
  return mesh;
}

function createWallWashPlane(material) {
  const geo = new THREE.PlaneGeometry(1, 1, 1, 1);
  const mesh = new THREE.Mesh(geo, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = 8;
  mesh.visible = false;
  return mesh;
}

function profilePointAt(points, index, t) {
  const next = Math.min(points.length - 1, index + 1);
  return points[index].clone().lerp(points[next], t);
}

function updateCarGroundEffects() {
  if (!state.trackProfile) return;
  const forward = new THREE.Vector3(Math.cos(state.yaw), 0, -Math.sin(state.yaw));
  const roadY = roadHeightAt(state.trackProfile, state.carPos.x, state.carPos.z, state.carPos.y) + MOUNTAIN_PASS.roadSurfaceLift;

  if (state.carContactShadow) {
    state.carContactShadow.position.set(state.carPos.x, roadY + 0.078, state.carPos.z);
    state.carContactShadow.rotation.y = state.yaw;
    state.carContactShadow.scale.set(2.35, 1, 1.03);
    state.carContactShadow.visible = true;
  }

  if (state.tailRoadGlow) {
    const glowPos = state.carPos.clone().add(forward.clone().multiplyScalar(-2.15));
    const glowY = roadHeightAt(state.trackProfile, glowPos.x, glowPos.z, roadY) + MOUNTAIN_PASS.roadSurfaceLift;
    state.tailRoadGlow.position.set(glowPos.x, glowY + 0.084, glowPos.z);
    state.tailRoadGlow.rotation.y = state.yaw;
    state.tailRoadGlow.scale.set(1.28, 1, 1.86);
    state.tailRoadGlow.visible = true;
  }
}

function updateHeadlightWallWash() {
  if (!state.headlightWallWash || !state.trackProfile) return;
  const forward = new THREE.Vector3(Math.cos(state.yaw), 0, -Math.sin(state.yaw));
  const probe = state.carPos.clone().add(forward.multiplyScalar(8.6));
  const roadSample = sampleRoadAt(state.trackProfile, probe.x, probe.z);
  if (!roadSample) {
    state.headlightWallWash.visible = false;
    return;
  }

  const center = profilePointAt(state.trackProfile.center, roadSample.index, roadSample.t);
  const rightEdge = profilePointAt(state.trackProfile.right, roadSample.index, roadSample.t);
  const outward = rightEdge.clone().sub(center).setY(0).normalize();
  const pos = rightEdge.clone().add(outward.multiplyScalar(3.25));
  const roadY = roadHeightAt(state.trackProfile, pos.x, pos.z, rightEdge.y) + MOUNTAIN_PASS.roadSurfaceLift;
  pos.y = roadY + 1.72;
  state.headlightWallWash.position.copy(pos);
  state.headlightWallWash.rotation.set(0, state.yaw, 0);
  state.headlightWallWash.scale.set(4.9, 2.7, 1);
  state.headlightWallWash.visible = true;
}

function updateHeadlightPool() {
  if (!state.headlightPool || !state.trackProfile) return;
  const geo = state.headlightPool.geometry;
  const stations = geo.userData.stations;
  const pos = geo.getAttribute("position");
  const forward = new THREE.Vector3(Math.cos(state.yaw), 0, -Math.sin(state.yaw));
  const lateral = new THREE.Vector3(Math.sin(state.yaw), 0, Math.cos(state.yaw));
  for (let i = 0; i < stations.length; i++) {
    const d = stations[i];
    const width = 0.34 + d * 0.105;
    const center = state.carPos.clone().add(forward.clone().multiplyScalar(d));
    const left = center.clone().add(lateral.clone().multiplyScalar(width));
    const right = center.clone().add(lateral.clone().multiplyScalar(-width));
    for (const [j, p] of [[i * 2, left], [i * 2 + 1, right]]) {
      const sample = sampleRoadAt(state.trackProfile, p.x, p.z);
      p.y = (sample ? sample.y : center.y) + MOUNTAIN_PASS.roadSurfaceLift + 0.072;
      pos.setXYZ(j, p.x, p.y, p.z);
    }
  }
  pos.needsUpdate = true;
  geo.computeBoundingSphere();
  state.headlightPool.visible = true;
}

function receiveState(msg) {
  state.latest = msg;
  const p = msg.pose;
  const pose = simPoseToThree(p);
  state.yawTarget = simYawToThreeYaw(p.yaw);
  const roadSample = sampleRoadAt(state.trackProfile, pose.x, pose.z);
  if (roadSample) {
    pose.y = roadSample.y + MOUNTAIN_PASS.roadSurfaceLift + MOUNTAIN_PASS.carSurfaceOffset;
    state.pitchTarget = roadPitchAt(state.trackProfile, pose.x, pose.z, state.yawTarget);
  } else {
    state.pitchTarget = 0;
  }
  state.carTarget.copy(pose);
  updateHud(msg);
}

function updateHud(msg) {
  ui.speed.textContent = Math.round(msg.vehicle.speedKmh);
  ui.rpm.textContent = Math.round(msg.vehicle.rpm);
  ui.gear.textContent = msg.vehicle.gear;
  ui.progress.textContent = `${Math.round(msg.track.progress * 100)}%`;
  ui.throttle.style.width = `${Math.round(msg.controls.throttle * 100)}%`;
  ui.brake.style.width = `${Math.round(msg.controls.brake * 100)}%`;
  const steer = Math.max(-1, Math.min(1, msg.controls.steer));
  ui.steer.style.width = `${Math.abs(steer) * 50}%`;
  ui.steer.style.marginLeft = steer >= 0 ? "50%" : `${50 - Math.abs(steer) * 50}%`;
  ui.trackName.className = msg.track.offTrack ? "pill warn" : "pill";
}

function inputPayload() {
  const k = state.keys;
  const left = k.has("ArrowLeft") || k.has("KeyA");
  const right = k.has("ArrowRight") || k.has("KeyD");
  return {
    type: "input",
    steer: (left ? 1 : 0) - (right ? 1 : 0),
    throttle: (k.has("ArrowUp") || k.has("KeyW")) ? 1 : 0,
    brake: (k.has("ArrowDown") || k.has("KeyS")) ? 1 : 0,
    handbrake: k.has("Space") ? 1 : 0,
    clutch: k.has("ShiftLeft") || k.has("ShiftRight") ? 0 : 1,
    shiftUp: k.has("KeyE"),
    shiftDown: k.has("KeyQ"),
    auto: state.auto,
  };
}

function keyAllowed(e) {
  const tag = (e.target && e.target.tagName || "").toLowerCase();
  return tag !== "select" && tag !== "input" && tag !== "button";
}

window.addEventListener("keydown", (e) => {
  if (!keyAllowed(e)) return;
  if (["KeyW", "KeyA", "KeyS", "KeyD", "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Space"].includes(e.code)) {
    e.preventDefault();
  }
  if (e.code === "KeyR") {
    send({ type: "reset" });
    toast("reset");
  } else if (e.code === "KeyC") {
    cycleCamera();
  } else if (e.code === "KeyT") {
    state.auto = !state.auto;
    toast(state.auto ? "automatic" : "manual");
  } else if (e.code === "KeyL") {
    toggleDayNight();
  } else if (e.code === "KeyM") {
    send({ type: "mute" });
  }
  state.keys.add(e.code);
});

window.addEventListener("keyup", (e) => {
  state.keys.delete(e.code);
});

setInterval(() => send(inputPayload()), 1000 / 30);

ui.restart.addEventListener("click", () => {
  startSession();
});
ui.track.addEventListener("change", () => startSession());
if (ui.car) ui.car.addEventListener("change", () => startSession());

// Day/night toggle
if (ui.daynight) {
  ui.daynight.addEventListener("click", toggleDayNight);
}

function toggleDayNight() {
  state.isDay = !state.isDay;
  setTimeOfDay(state.isDay, {
    skyDome: state.skyDome,
    lighting: state.lighting,
    scene,
    post,
    fogPlanes: state.fogPlanes,
    mats,
  });
  if (ui.daynight) {
    ui.daynight.textContent = state.isDay ? "☀ Day" : "☾ Night";
  }
  toast(state.isDay ? "daytime" : "nighttime");
}

function cycleCamera() {
  const modes = ["chase", "cinematic", "hood"];
  const i = modes.indexOf(state.cameraMode);
  state.cameraMode = modes[(i + 1) % modes.length];
  ui.cameraMode.textContent = state.cameraMode;
  send({ type: "camera", mode: state.cameraMode });
}

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now() * 0.001;

  if (state.latest) {
    state.carPos.lerp(state.carTarget, 0.18);
    state.yaw = dampAngle(state.yaw, state.yawTarget, 0.18);
    state.pitch += (state.pitchTarget - state.pitch) * 0.16;
    state.car.position.copy(state.carPos);
    state.car.rotation.set(0, state.yaw, state.pitch);
    updateHeadlightPool();
    updateHeadlightWallWash();
    updateCarGroundEffects();
    const steer = state.latest.vehicle.steerAngle || 0;
    const spin = state.latest.vehicle.wheelSpin || 0;
    state.wheels.forEach((w) => {
      w.rotation.y = w.userData.steer ? steer : 0;
      if (w.userData.roll) w.userData.roll.rotation.z = -spin;
    });
    updateCamera();
  }

  // Animate fog planes — gentle bobbing
  for (const fp of state.fogPlanes) {
    if (fp.userData.baseY !== undefined) {
      fp.position.y = fp.userData.baseY + Math.sin(now * fp.userData.bobSpeed) * fp.userData.bobAmount;
    }
  }

  // Keep sky dome centered on camera
  if (state.skyDome) {
    state.skyDome.position.copy(camera.position);
  }

  renderMountainPassFrame();
}

function updateCamera() {
  const yaw = state.yaw;
  const forward = new THREE.Vector3(Math.cos(yaw), 0, -Math.sin(yaw));
  const right = new THREE.Vector3(Math.sin(yaw), 0, Math.cos(yaw));
  const base = state.carPos;
  const speed = state.latest ? state.latest.vehicle.speed : 0;
  const t = state.latest ? state.latest.time : 0;
  const bob = Math.sin(t * 8.0) * Math.min(speed * 0.010, 0.13);
  let desired;
  let look;
  if (state.cameraMode === "hood") {
    desired = base.clone().add(forward.clone().multiplyScalar(1.66)).add(new THREE.Vector3(0, 1.05 + bob, 0));
    look = base.clone().add(forward.clone().multiplyScalar(25)).add(new THREE.Vector3(0, 0.82, 0));
  } else if (state.cameraMode === "cinematic") {
    desired = base.clone().add(forward.clone().multiplyScalar(-7.2)).add(right.clone().multiplyScalar(4.2)).add(new THREE.Vector3(0, 2.35 + bob, 0));
    look = base.clone().add(forward.clone().multiplyScalar(7.2)).add(new THREE.Vector3(0, 0.82, 0));
  } else {
    desired = base.clone().add(forward.clone().multiplyScalar(-8.15 - Math.min(speed * 0.08, 3.6))).add(new THREE.Vector3(0, 2.58 + bob, 0));
    look = base.clone().add(forward.clone().multiplyScalar(13.8)).add(new THREE.Vector3(0, 0.88, 0));
  }
  camera.position.lerp(desired, 0.095);
  camera.lookAt(look);
}

function dampAngle(a, b, t) {
  let d = ((b - a + Math.PI) % (Math.PI * 2)) - Math.PI;
  if (d < -Math.PI) d += Math.PI * 2;
  return a + d * t;
}

function resizeRenderer() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  const w = Math.max(1, Math.floor(window.innerWidth * MOUNTAIN_PASS.renderScale));
  const h = Math.max(1, Math.floor(window.innerHeight * MOUNTAIN_PASS.renderScale));
  post.target.setSize(w, h);
  post.uniforms.resolution.value.set(w, h);
}

function renderMountainPassFrame() {
  post.uniforms.time.value = performance.now() * 0.001;
  renderer.setRenderTarget(post.target);
  renderer.clear();
  renderer.render(scene, camera);
  renderer.setRenderTarget(null);
  renderer.clear();
  renderer.render(post.scene, post.camera);
}

window.addEventListener("resize", resizeRenderer);

connect();
animate();
