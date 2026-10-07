import * as THREE from "three";
import "./styles.css";
import { AudioBridge } from "./audio.js";
import { WatchCamera } from "./cameras.js";
import { CrtOverlay } from "./crt-overlay.js";
import { simToThree, threeToSim, vehicleOrientation } from "./coords.js";
import { launchParams, openSession, releaseSession, TelemetryLink } from "./net.js";
import {
  syncPs1Fog,
  syncPs1Resolution,
  syncPs1Time,
  syncPs1Tint,
} from "./ps1-material.js";
import { CameraRain } from "./rain.js";
import { normalizeFrame, Timeline } from "./timeline.js";
import { loadTrackWorld } from "./track-world.js";
import { createVehicleProxy, resolveVehicleKind } from "./vehicle.js";
import { nextWeather, resolveWeather } from "./weather.js";

const canvas = document.getElementById("stage");
const hudConn = document.getElementById("hud-conn");
const hudSeq = document.getElementById("hud-seq");
const hudSpeed = document.getElementById("hud-speed");
const hudGear = document.getElementById("hud-gear");
const hudCam = document.getElementById("hud-cam");
const hudWx = document.getElementById("hud-wx");
const hudCrt = document.getElementById("hud-crt");
const hudProgress = document.getElementById("hud-progress");
const hudLaptime = document.getElementById("hud-laptime");
const hudNote = document.getElementById("hud-note");
const hudStatus = document.getElementById("hud-status");
const audioBtn = document.getElementById("toggle-audio");

const FOG_COLOR = 0x6e7c74;

const renderer = new THREE.WebGLRenderer({
  canvas,
  antialias: false,
  powerPreference: "high-performance",
  alpha: false,
});
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(FOG_COLOR, 1);

const scene = new THREE.Scene();
// Short Rally-Championship-ish draw distance — presentation only.
scene.fog = new THREE.FogExp2(FOG_COLOR, 0.011);
scene.background = new THREE.Color(FOG_COLOR);

const camera = new THREE.PerspectiveCamera(58, 1, 0.3, 380);
const watchCam = new WatchCamera(camera);

const hemi = new THREE.HemisphereLight(0xb8c4b8, 0x3a4038, 0.55);
scene.add(hemi);

const carRoot = new THREE.Group();
scene.add(carRoot);

const timeline = new Timeline();
const audio = new AudioBridge();
const crt = new CrtOverlay();
const rain = new CameraRain(scene);
const _p = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _tint = new THREE.Color(1, 1, 1);

let vehicle = null;
let vehicleKind = "919";
let link = null;
let session = null;
let currentFrame = null;
let trackWorld = null;
let lastTs = performance.now();
let lastListenerSent = 0;
let disposed = false;
let weather = resolveWeather("clear");
let audioArmed = false;
let audioAutoStarted = false;
let bootElapsed = 0;

function setConn(state) {
  hudConn.dataset.state = state;
  hudConn.textContent = String(state || "—").toUpperCase();
}

function setStatus(text) {
  hudStatus.textContent = text;
}

function setCamHud() {
  if (hudCam) hudCam.textContent = String(watchCam.mode).toUpperCase();
}

function setWxHud() {
  if (hudWx) hudWx.textContent = weather.label;
}

function setCrtHud() {
  if (hudCrt) hudCrt.textContent = crt.label;
}

function setAudioButton(status) {
  if (!audioBtn) return;
  const on = status !== "off" && status !== "unavailable";
  audioBtn.setAttribute("aria-pressed", on ? "true" : "false");
  audioBtn.dataset.state = status || "off";
  audioBtn.classList.toggle("degraded", status === "buffering"
    || status === "offline" || status === "muted" || status === "unavailable");
  const label = status === "live" ? "AUDIO ON"
    : status === "buffering" ? "AUDIO…"
    : status === "muted" ? "AUDIO MUTE"
    : status === "offline" || status === "unavailable" ? "AUDIO —"
    : "AUDIO";
  audioBtn.textContent = label;
}

function applyWeather(palette) {
  weather = palette;
  scene.fog.color.setHex(palette.fog);
  scene.fog.density = palette.density;
  scene.background.setHex(palette.fog);
  renderer.setClearColor(palette.fog, 1);
  hemi.color.setHex(palette.hemiSky);
  hemi.groundColor.setHex(palette.hemiGround);
  hemi.intensity = palette.hemiIntensity;
  _tint.setRGB(
    palette.asphaltTint[0],
    palette.asphaltTint[1],
    palette.asphaltTint[2],
  );
  if (trackWorld?.materials) {
    syncPs1Fog(trackWorld.materials, scene.fog);
    syncPs1Tint(trackWorld.materials, _tint);
  }
  rain.setEnabled(palette.rain);
  setWxHud();
}

function formatLapTime(seconds) {
  const s = Math.max(0, Number(seconds) || 0);
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  const whole = Math.floor(rem);
  const tenths = Math.floor((rem - whole) * 10);
  return `${m}:${String(whole).padStart(2, "0")}.${tenths}`;
}

function updateHud(frame) {
  if (!frame) return;
  hudSeq.textContent = String(frame.seq);
  const kmh = Math.round(Math.max(0, frame.speedKmh));
  hudSpeed.textContent = String(kmh).padStart(3, "0");
  const gear = frame.gear;
  hudGear.textContent = gear === 0 ? "N" : String(gear);
  const pct = Math.max(0, Math.min(1, frame.progress)) * 100;
  hudProgress.textContent = `${pct.toFixed(1)}%`;
  hudLaptime.textContent = formatLapTime(frame.episodeTime);
}

function mountVehicle(kind) {
  const next = createVehicleProxy(kind);
  if (vehicle?.group) carRoot.remove(vehicle.group);
  vehicle = next;
  vehicleKind = next.kind;
  carRoot.add(vehicle.group);
  if (hudNote) {
    hudNote.hidden = true;
    hudNote.textContent = "";
  }
  return vehicle;
}

function resize() {
  const w = window.innerWidth;
  const h = Math.max(1, window.innerHeight);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  // Cap DPR for a crunchier PS1 raster; still no AA.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.0));
  renderer.setSize(w, h, false);
  if (trackWorld?.materials) {
    syncPs1Resolution(trackWorld.materials, w, h);
  }
}
window.addEventListener("resize", resize);

function applyCar(frame) {
  if (!vehicle) return;
  // Road height is Python truth (road_z_m); do not invent elevation.
  simToThree(frame.pose.x, frame.pose.y, frame.roadZ, _p);
  vehicleOrientation(frame.pose.yaw, frame.pose.pitch, frame.pose.roll, _q);
  carRoot.position.copy(_p);
  carRoot.quaternion.copy(_q);
  vehicle.applyWheels(frame);
  watchCam.update(_p, frame.pose.yaw);
}

function pushListener(frame) {
  if (!link || !frame || !audio.enabled) return;
  const now = performance.now();
  if (now - lastListenerSent < 50) return;
  lastListenerSent = now;
  const pos = camera.position;
  const sim = threeToSim(pos.x, pos.y, pos.z);
  const audioState = watchCam.audioState();
  link.sendListener({
    x: sim.x,
    y: sim.y,
    z: sim.z,
    vx: frame.worldVel[0] || 0,
    vy: frame.worldVel[1] || 0,
    yaw: frame.pose.yaw,
    camera: audioState.camera,
    perspective: audioState.perspective,
    cut: audioState.cut,
  });
}

async function enableAudio() {
  try {
    await audio.enable(link?.audioUrl);
    setAudioButton(audio.status);
    if (audio.status === "unavailable") {
      setStatus("audio unavailable (no FOA1 socket)");
    }
  } catch {
    setAudioButton("unavailable");
    setStatus("audio unavailable");
  }
}

async function disableAudio() {
  await audio.disable();
  setAudioButton("off");
}

async function toggleAudio() {
  if (!session) return;
  if (audio.enabled) {
    await disableAudio();
    return;
  }
  // User gesture — safe to resume AudioContext.
  await enableAudio();
}

/** When ?audio=1, first page gesture auto-enables once (no second AUDIO click). */
function tryAutoEnableAudio() {
  if (!audioArmed || audioAutoStarted || !session || !link) return;
  if (audio.enabled) {
    audioAutoStarted = true;
    return;
  }
  audioAutoStarted = true;
  enableAudio().then(() => {
    if (audio.status === "live" || audio.status === "buffering") {
      setStatus(`${hudStatus.textContent.replace(/ · gesture to start sound| · tap AUDIO to start sound/g, "")} · audio live`);
    }
  });
}

window.addEventListener("keydown", (event) => {
  if (event.repeat) return;
  if (event.key === "a" || event.key === "A") {
    // Manual toggle owns this gesture — don't auto-enable then flip off.
    audioAutoStarted = true;
    toggleAudio();
    return;
  }
  tryAutoEnableAudio();
  if (event.key === "c" || event.key === "C") {
    watchCam.cycle();
    setCamHud();
  }
  if (event.key === "g" || event.key === "G") {
    applyWeather(nextWeather(weather.id));
  }
  if (event.key === "v" || event.key === "V") {
    crt.cycle();
    setCrtHud();
  }
});

window.addEventListener("pointerdown", (event) => {
  if (event.target?.closest?.("#toggle-audio")) return;
  tryAutoEnableAudio();
}, { passive: true });

audioBtn?.addEventListener("click", (event) => {
  event.preventDefault();
  audioAutoStarted = true; // manual toggle owns the gesture
  toggleAudio();
});

audio.onStatus = (status) => {
  setAudioButton(status);
};

async function boot() {
  setConn("connecting");
  setStatus("loading track ribbon…");
  setCamHud();
  setCrtHud();
  setAudioButton("off");

  const params = launchParams();
  audioArmed = Boolean(params.audio);
  applyWeather(resolveWeather(params.weather));
  mountVehicle(resolveVehicleKind(params.car, params.edition));

  try {
    trackWorld = await loadTrackWorld("./assets/track/nordschleife_ribbon.json");
    scene.add(trackWorld.group);
    applyWeather(weather);
    resize();
    setStatus(
      `ribbon ${trackWorld.meta.id} · ${trackWorld.meta.sampleCount} samples · opening session…`,
    );
  } catch (error) {
    setConn("error");
    setStatus(`track load failed: ${error.message || error}`);
    return;
  }

  try {
    session = await openSession(params);
  } catch (error) {
    setConn("error");
    setStatus(String(error.message || error));
    return;
  }

  const hello = session.hello || {};
  const car = hello.checkpoint?.car || params.car || params.edition;
  const kind = resolveVehicleKind(car, params.car, params.edition);
  if (kind !== vehicleKind) mountVehicle(kind);
  const ckpt = hello.checkpoint?.path || hello.checkpoint?.name || params.checkpoint;
  setStatus(`session ${session.session_id.slice(0, 8)}… · ${vehicle.meta.displayName} · ${ckpt}`);
  if (params.audio) {
    setStatus(`${hudStatus.textContent} · gesture to start sound`);
    setAudioButton("off");
  }

  link = new TelemetryLink(session);
  link.onState = (state) => setConn(state);
  link.onHello = (message) => {
    const c = message.checkpoint?.car || car;
    const nextKind = resolveVehicleKind(c, params.car, params.edition);
    if (nextKind !== vehicleKind) mountVehicle(nextKind);
    setStatus(`hello · ${vehicle.meta.displayName} · waiting for frames`);
    if (params.audio && !audio.enabled) {
      setStatus(`${hudStatus.textContent} · gesture to start sound`);
    }
  };
  link.onFrame = (raw) => {
    try {
      const frame = normalizeFrame(raw);
      timeline.push(frame);
    } catch (error) {
      console.warn(error);
    }
  };
  link.onWarning = (message) => {
    setStatus(message.message || message.code || "warning");
  };
  link.onError = (message) => {
    setConn("error");
    setStatus(message.message || message.code || "telemetry error");
  };
  link.connect();
}

function tick(now) {
  if (disposed) return;
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, Math.max(0.001, (now - lastTs) / 1000));
  lastTs = now;
  bootElapsed += dt;

  const frame = timeline.sample(dt);
  if (frame) {
    currentFrame = frame;
    applyCar(frame);
    updateHud(frame);
    pushListener(frame);
  }
  if (trackWorld?.materials) {
    syncPs1Time(trackWorld.materials, bootElapsed);
  }
  rain.update(camera.position, bootElapsed);
  renderer.render(scene, camera);
}
requestAnimationFrame(tick);

async function shutdown() {
  disposed = true;
  link?.disconnect();
  await audio.disable();
  await releaseSession(session?.session_id);
}

window.addEventListener("pagehide", (event) => {
  if (event.persisted) return;
  shutdown();
});
window.addEventListener("beforeunload", () => {
  shutdown();
});

boot().catch((error) => {
  setConn("error");
  setStatus(String(error.message || error));
});

// Debug probe for Phase acceptance.
window.__watch25d = {
  get frame() { return currentFrame; },
  get session() { return session; },
  get track() { return trackWorld; },
  get vehicle() { return vehicle; },
  get vehicleKind() { return vehicleKind; },
  get audio() { return audio; },
  get weather() { return weather; },
  get crt() { return crt; },
  applyWeather,
  timeline,
  camera,
  watchCam,
  renderer,
  resolveVehicleKind,
};
