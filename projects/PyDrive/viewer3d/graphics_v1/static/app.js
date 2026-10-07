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
} from "./mountain-pass-style.js?v=ps2-pass-reference6";
import {
  buildMountainPassTrack,
  roadPitchAt,
  sampleRoadAt,
} from "./mountain-pass-world.js?v=ps2-pass-reference6";

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
state.car = createSupra();
scene.add(state.car);
state.headlightPool = createHeadlightPool();
scene.add(state.headlightPool);

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
      ui.trackName.textContent = msg.meta.name || "track";
      toast(`loaded ${msg.meta.name || "track"}`);
    } else if (msg.type === "state") {
      receiveState(msg);
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
  send({
    type: "start",
    car: ui.car.value,
    track: ui.track.value,
    seed: 7,
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
  });
}

function createSupra() {
  const group = new THREE.Group();
  group.name = "procedural-supra";

  const red = mats.body;
  const redDark = mats.bodyDark;
  const glass = mats.glass;
  const black = mats.black;
  const tire = mats.tire;
  const rim = mats.rim;
  const lamp = mats.lamp;
  const tail = mats.tail;

  addBox(group, [0.05, 0.56, 0], [4.55, 0.44, 1.78], red);
  addBox(group, [0.35, 0.85, 0], [2.65, 0.20, 1.64], redDark);
  addBox(group, [0.82, 0.78, 0], [1.8, 0.18, 1.62], red);
  addBox(group, [-0.48, 1.12, 0], [1.62, 0.70, 1.36], glass);
  addBox(group, [-1.45, 0.92, 0], [0.98, 0.18, 1.58], red);
  addBox(group, [2.36, 0.48, 0], [0.18, 0.28, 1.72], black);
  addBox(group, [2.50, 0.62, 0.54], [0.07, 0.18, 0.38], lamp, false);
  addBox(group, [2.50, 0.62, -0.54], [0.07, 0.18, 0.38], lamp, false);
  addBox(group, [2.53, 0.50, 0.0], [0.05, 0.10, 1.15], black, false);
  addBox(group, [-2.32, 0.58, 0.56], [0.08, 0.20, 0.42], tail, false);
  addBox(group, [-2.32, 0.58, -0.56], [0.08, 0.20, 0.42], tail, false);
  addWing(group, redDark);

  const headGlow = new THREE.PointLight(PALETTE.amber, 1.65, 28, 1.85);
  headGlow.position.set(2.35, 0.82, 0);
  group.add(headGlow);
  const tailGlow = new THREE.PointLight(PALETTE.tail, 0.55, 11, 2.1);
  tailGlow.position.set(-2.22, 0.72, 0);
  group.add(tailGlow);

  const wheelPositions = [
    [1.33, 0.39, 0.92, true],
    [1.33, 0.39, -0.92, true],
    [-1.35, 0.39, 0.92, false],
    [-1.35, 0.39, -0.92, false],
  ];
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

  group.traverse((obj) => {
    if (obj.isMesh) {
      obj.castShadow = true;
      obj.receiveShadow = true;
    }
  });
  return group;
}

function createWheelAssembly(tireMat, rimMat) {
  const roll = new THREE.Group();
  const treadMat = mats.tread;

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
  mesh.receiveShadow = true;
  parent.add(mesh);
  return mesh;
}

function addWing(parent, mat) {
  const blade = addBox(parent, [-2.12, 1.18, 0], [0.18, 0.12, 1.72], mat);
  blade.rotation.z = -0.08;
  addBox(parent, [-1.92, 0.92, 0.62], [0.12, 0.48, 0.10], mat);
  addBox(parent, [-1.92, 0.92, -0.62], [0.12, 0.48, 0.10], mat);
}

function createHeadlightPool() {
  const stations = [1.8, 3.4, 5.6, 8.1, 11.0, 14.2];
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
  mesh.renderOrder = 8;
  mesh.visible = false;
  return mesh;
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
    const width = 0.58 + d * 0.22;
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
    desired = base.clone().add(forward.clone().multiplyScalar(1.58)).add(new THREE.Vector3(0, 1.16 + bob, 0));
    look = base.clone().add(forward.clone().multiplyScalar(21)).add(new THREE.Vector3(0, 0.92, 0));
  } else if (state.cameraMode === "cinematic") {
    desired = base.clone().add(forward.clone().multiplyScalar(-5.9)).add(right.clone().multiplyScalar(5.1)).add(new THREE.Vector3(0, 2.18 + bob, 0));
    look = base.clone().add(forward.clone().multiplyScalar(4.8)).add(new THREE.Vector3(0, 0.88, 0));
  } else {
    desired = base.clone().add(forward.clone().multiplyScalar(-5.6 - Math.min(speed * 0.15, 5.8))).add(new THREE.Vector3(0, 2.50 + bob, 0));
    look = base.clone().add(forward.clone().multiplyScalar(7.4)).add(new THREE.Vector3(0, 0.92, 0));
  }
  camera.position.lerp(desired, 0.105);
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
