import * as THREE from "three";
import "./styles.css";
import { AtmosphereRig } from "./atmosphere.js";
import { createObservatorySession, ObservatoryClient, releaseObservatorySession } from "./observatory-client.js";
import { EngineAudioStream } from "./audio-stream.js";
import { CameraDirector } from "./cameras.js";
import { FrameBuffer } from "./frame-buffer.js";
import { bindPanel, recordReplayEvent, setConnection, toast, updateFrame, updateIdentity } from "./overlays.js";
import { CinematicPipeline } from "./postfx.js";
import { normalizeFrame, normalizeIdentity, vehicleFromIdentity } from "./protocol.js";
import { AdaptiveQuality } from "./quality.js";
import { ObservatoryWorld } from "./world.js";

const app = document.getElementById("app");
const canvas = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance", stencil: false, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.96;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0xa6bbc2, 0.00145);
const camera = new THREE.PerspectiveCamera(48, 1, 0.5, 400000);
camera.position.set(8, 4, 10);
const atmosphere = new AtmosphereRig(scene);
const director = new CameraDirector(camera, canvas, (label) => {
  document.getElementById("camera-shot").textContent = label;
});
const world = new ObservatoryWorld(scene, renderer);
const pipeline = new CinematicPipeline(renderer, scene, camera);
const frames = new FrameBuffer();
const clock = new THREE.Clock();
let currentFrame = null;
let lastUiFrameKey = "";
let viewportWidth = 0;
let viewportHeight = 0;

const hemisphere = new THREE.HemisphereLight(0xcbe3ec, 0x243325, 1.65);
scene.add(hemisphere);
const sun = new THREE.DirectionalLight(0xfff2d1, 3.25);
sun.position.set(-240, 360, 180);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -70;
sun.shadow.camera.right = 70;
sun.shadow.camera.top = 70;
sun.shadow.camera.bottom = -70;
sun.shadow.camera.near = 1;
sun.shadow.camera.far = 900;
sun.shadow.bias = -0.00012;
sun.shadow.normalBias = 0.035;
scene.add(sun, sun.target);

let client = null;
let audio = null;
let sessionLease = null;
let released = false;
let assetTrackHash = "";
let trackMismatchReported = false;
let lastPolicyHash = "";
let lastIdentityRenderKey = "";
const seenWarnings = new Set();
let lastListenerUpdate = 0;
const LISTENER_UPDATE_SECONDS = 0.1;
let currentWeather = "dry";
const lastListenerPosition = new THREE.Vector3();
const cameraDirection = new THREE.Vector3();

const quality = new AdaptiveQuality((name, settings) => {
  renderer.setPixelRatio(Math.min(devicePixelRatio, settings.pixelRatioCap));
  renderer.shadowMap.enabled = settings.shadows;
  world.setForestDensity(settings.forestDensity);
  pipeline.setQuality(name);
  app.dataset.quality = name;
  viewportWidth = 0;
  toast(`Visual quality adjusted to ${name}. Physics and playback are unchanged.`);
});
app.dataset.quality = quality.level;
pipeline.setQuality(quality.level);

const engineerPanel = bindPanel("toggle-engineer", "engineer-overlay", false);
const brainPanel = bindPanel("toggle-xray", "brain-overlay", false);
const replayPanel = bindPanel("toggle-replay", "replay-overlay", false);
const controlsPanel = { open: false };

function renderControlsPanel() {
  const panel = document.getElementById("view-controls");
  const button = document.getElementById("toggle-controls");
  panel.dataset.open = String(controlsPanel.open);
  button.classList.toggle("active", controlsPanel.open);
  button.setAttribute("aria-pressed", String(controlsPanel.open));
}

function closeOtherWorkspaces(keep) {
  if (keep !== "engineer") engineerPanel.set(false);
  if (keep !== "brain") brainPanel.set(false);
  if (keep !== "replay") replayPanel.set(false);
  if (keep !== "view") controlsPanel.open = false;
  renderControlsPanel();
}

function syncWorkspace() {
  const workspace = brainPanel.open ? "brain"
    : engineerPanel.open ? "engineer"
      : replayPanel.open ? "replay"
        : controlsPanel.open ? "view"
          : "cinematic";
  app.dataset.workspace = workspace;
  document.getElementById("mode-cinematic").classList.toggle("active", workspace === "cinematic");
  document.getElementById("mode-cinematic").setAttribute("aria-pressed", String(workspace === "cinematic"));
  world.setEngineeringVisible(engineerPanel.open);
}

document.getElementById("mode-cinematic").addEventListener("click", () => {
  closeOtherWorkspaces("cinematic");
  syncWorkspace();
});
document.getElementById("toggle-engineer").addEventListener("click", () => {
  if (engineerPanel.open) closeOtherWorkspaces("engineer");
  syncWorkspace();
});
document.getElementById("toggle-xray").addEventListener("click", () => {
  if (brainPanel.open) closeOtherWorkspaces("brain");
  syncWorkspace();
});
document.getElementById("toggle-replay").addEventListener("click", () => {
  if (replayPanel.open) closeOtherWorkspaces("replay");
  syncWorkspace();
});
document.getElementById("toggle-controls").addEventListener("click", () => {
  controlsPanel.open = !controlsPanel.open;
  if (controlsPanel.open) closeOtherWorkspaces("view");
  renderControlsPanel();
  syncWorkspace();
});
document.getElementById("close-controls").addEventListener("click", () => {
  controlsPanel.open = false;
  renderControlsPanel();
  syncWorkspace();
});
document.querySelectorAll("[data-toggle-panel]").forEach((button) => button.addEventListener("click", () => queueMicrotask(syncWorkspace)));
renderControlsPanel();
syncWorkspace();

document.getElementById("warning-toggle").addEventListener("click", (event) => {
  const expanded = event.currentTarget.getAttribute("aria-expanded") === "true";
  event.currentTarget.setAttribute("aria-expanded", String(!expanded));
  document.getElementById("identity-warnings").dataset.hidden = String(expanded);
});
document.getElementById("camera-mode").addEventListener("change", (event) => director.setMode(event.target.value));
document.getElementById("weather-mode").addEventListener("change", (event) => setWeather(event.target.value));
document.getElementById("control-pause").addEventListener("click", () => client?.send(currentFrame?.replay.paused ? "resume" : "pause"));
document.getElementById("control-step").addEventListener("click", () => client?.send("step"));
document.getElementById("control-reset").addEventListener("click", () => client?.send("reset"));
document.getElementById("control-speed").addEventListener("change", (event) => client?.send("speed", Number(event.target.value)));
document.getElementById("control-scrub").addEventListener("change", (event) => client?.send("scrub", Number(event.target.value)));
document.getElementById("toggle-audio").addEventListener("click", async (event) => {
  const button = event.currentTarget;
  const enabled = audio ? await audio.toggle() : false;
  button.classList.toggle("active", enabled);
  button.setAttribute("aria-pressed", String(enabled));
  button.textContent = enabled ? "AUDIO LIVE" : "AUDIO OFF";
});

window.addEventListener("keydown", (event) => {
  if (event.metaKey || event.ctrlKey || event.altKey || /INPUT|SELECT|TEXTAREA/.test(event.target?.tagName || "")) return;
  const buttons = { "1": "mode-cinematic", "2": "toggle-engineer", "3": "toggle-xray", "4": "toggle-replay", "5": "toggle-controls" };
  if (buttons[event.key]) document.getElementById(buttons[event.key]).click();
});

function attachSession(session) {
  sessionLease = session;
  client = new ObservatoryClient(session);
  audio = new EngineAudioStream(session.sessionId, session.audioUrl);
  client.addEventListener("status", (event) => setConnection(event.detail));
  client.addEventListener("warning", (event) => {
    toast(event.detail.message);
    recordReplayEvent({ type: "warning", message: event.detail.message });
  });
  client.addEventListener("event", (event) => recordReplayEvent(event.detail));
  client.addEventListener("events", (event) => recordReplayEvent(event.detail));
  client.addEventListener("identity", (event) => acceptIdentity(event.detail));
  client.addEventListener("frame", (event) => {
    acceptIdentity(event.detail.identity);
    frames.push(event.detail);
  });
  audio.addEventListener("status", (event) => {
    const button = document.getElementById("toggle-audio");
    button.classList.toggle("active", event.detail.enabled && !event.detail.degraded);
    button.setAttribute("aria-pressed", String(event.detail.enabled));
    button.textContent = event.detail.degraded ? "AUDIO BUFFER" : event.detail.enabled ? "AUDIO LIVE" : "AUDIO OFF";
    if (!event.detail.enabled || event.detail.degraded) toast(event.detail.label);
  });
  if (session.hello) acceptIdentity(normalizeIdentity(session.hello));
  if (session.trackHash && assetTrackHash && session.trackHash !== assetTrackHash) acceptIdentity(normalizeIdentity({ track: { hash: session.trackHash } }));
  recordReplayEvent({ type: "session_created", checkpoint: session.hello?.checkpoint });
  client.connect();
}

function acceptIdentity(identity) {
  const renderKey = JSON.stringify([
    identity.car, identity.edition, identity.stage, identity.checkpoint,
    identity.policyHash, identity.classification, identity.drivetrain,
    identity.observationLayout, identity.compatibility, identity.warnings,
  ]);
  if (renderKey !== lastIdentityRenderKey) {
    updateIdentity(identity);
    lastIdentityRenderKey = renderKey;
  }
  try {
    world.ensureVehicle(vehicleFromIdentity(identity)).catch((error) => {
      setConnection({ state: "error", label: "VEHICLE MISMATCH" });
      toast(error.message);
    });
  } catch (error) {
    setConnection({ state: "error", label: "IDENTITY ERROR" });
    toast(error.message);
  }
  if (lastPolicyHash && identity.policyHash !== "HASH UNAVAILABLE" && identity.policyHash !== lastPolicyHash) {
    recordReplayEvent({ type: "checkpoint_swap", checkpoint: { filename: identity.checkpoint } });
  }
  if (identity.policyHash !== "HASH UNAVAILABLE") lastPolicyHash = identity.policyHash;
  identity.warnings.forEach((warning) => {
    if (seenWarnings.has(warning)) return;
    seenWarnings.add(warning);
    recordReplayEvent({ type: "warning", message: warning });
  });
  if (identity.trackHash && assetTrackHash && identity.trackHash !== assetTrackHash) {
    setConnection({ state: "error", label: "TRACK MISMATCH" });
    if (!trackMismatchReported) {
      trackMismatchReported = true;
      toast("Playback track hash does not match the displayed truth geometry. Playback remains authoritative; rebuild Observatory assets before trusting alignment.");
    }
  }
}

function setWeather(mode) {
  currentWeather = mode;
  app.dataset.weather = mode;
  const preset = atmosphere.setPreset(mode);
  scene.fog.color.setHex(preset.fog);
  scene.fog.density = preset.fogDensity;
  sun.color.setHex(preset.sunColor);
  sun.intensity = preset.sunIntensity;
  hemisphere.color.setHex(preset.hemiSky);
  hemisphere.groundColor.setHex(preset.hemiGround);
  hemisphere.intensity = preset.hemiIntensity;
  renderer.toneMappingExposure = preset.exposure;
  pipeline.setWeatherStrength(preset.bloom);
  world.setWeather(mode);
}

async function boot() {
  const progress = document.getElementById("loading-progress");
  const detail = document.getElementById("loading-detail");
  let pendingSession = null;
  try {
    const session = await createObservatorySession();
    pendingSession = session;
    const serverIdentity = session.hello ? normalizeIdentity(session.hello) : null;
    const initialVehicle = serverIdentity ? vehicleFromIdentity(serverIdentity) : null;
    const loaded = await world.load((ratio, label) => {
      progress.style.width = `${Math.round(ratio * 100)}%`;
      detail.textContent = label;
    }, initialVehicle);
    director.setTrack(loaded.data.track.centerline);
    assetTrackHash = loaded.manifest.track.hash;
    const initial = loaded.data.track.centerline[0];
    if (serverIdentity) {
      frames.push(normalizeFrame({
        checkpoint: session.hello.checkpoint,
        car: initialVehicle,
        vehicle: { position: initial, yaw: loaded.data.track.heading[0] },
      }));
    }
    attachSession(session);
    pendingSession = null;
    document.getElementById("loading-card").dataset.hidden = "true";
  } catch (error) {
    if (pendingSession) releaseObservatorySession(pendingSession);
    setConnection({ state: "error", label: "ASSET ERROR" });
    document.getElementById("loading-title").textContent = "Observatory assets unavailable";
    detail.textContent = `${error.message}. Run npm run assets from observatory/.`;
    console.error(error);
  }
}

function resize() {
  const width = Math.max(1, canvas.clientWidth);
  const height = Math.max(1, canvas.clientHeight);
  if (width === viewportWidth && height === viewportHeight) return;
  viewportWidth = width;
  viewportHeight = height;
  renderer.setSize(width, height, false);
  pipeline.setSize(width, height, renderer.getPixelRatio());
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  const delta = Math.min(clock.getDelta(), 0.1);
  const elapsed = clock.elapsedTime;
  resize();
  quality.observe(delta * 1000);
  const frame = frames.sample();
  if (frame) {
    currentFrame = frame;
    const frameKey = `${frame.identity.policyHash}:${frame.seq}:${frame.replay.paused}:${frame.replay.live}:${frame.replay.speed}`;
    const telemetryChanged = frameKey !== lastUiFrameKey;
    const state = world.applyFrame(frame, brainPanel.open, telemetryChanged, delta);
    director.update(state, elapsed, delta);
    world.updateTerrainLod(camera.position);
    if (telemetryChanged) {
      updateFrame(frame);
      lastUiFrameKey = frameKey;
    }
    sun.target.position.copy(state.position);
    sun.position.copy(state.position).addScaledVector(atmosphere.sunDirection, 520);
  }
  atmosphere.update(camera.position);
  if (client?.connected && elapsed - lastListenerUpdate >= LISTENER_UPDATE_SECONDS) {
    const interval = lastListenerUpdate ? elapsed - lastListenerUpdate : LISTENER_UPDATE_SECONDS;
    const audioCamera = director.audioState();
    camera.getWorldDirection(cameraDirection);
    client.send("listener", {
      x: camera.position.x,
      y: -camera.position.z,
      z: camera.position.y,
      vx: (camera.position.x - lastListenerPosition.x) / interval,
      vy: -(camera.position.z - lastListenerPosition.z) / interval,
      yaw: Math.atan2(-cameraDirection.z, cameraDirection.x),
      camera: audioCamera.camera,
      perspective: audioCamera.perspective,
      cut: audioCamera.cut,
      weather: currentWeather,
    });
    lastListenerPosition.copy(camera.position);
    lastListenerUpdate = elapsed;
  }
  pipeline.render(delta);
}

async function releaseSession() {
  if (released) return;
  released = true;
  await Promise.allSettled([client?.close(), audio?.shutdown()]);
  releaseObservatorySession(sessionLease);
}
window.addEventListener("pagehide", releaseSession);
window.addEventListener("beforeunload", releaseSession);

// Opt-in QA hook: `?debug` exposes render internals so smoothness and pose can
// be sampled from the console. It exposes nothing that changes playback.
if (new URLSearchParams(location.search).has("debug")) {
  window.__observatory = { world, frames, camera, director, quality, scene, getFrame: () => currentFrame };
}

setWeather("dry");
boot();
animate();
