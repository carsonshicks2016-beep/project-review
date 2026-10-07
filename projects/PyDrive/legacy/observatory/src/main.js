import * as THREE from "three";
import "./styles.css";
import { AtmosphereRig } from "./atmosphere.js";
import { createObservatorySession, ObservatoryClient, releaseObservatorySession } from "./observatory-client.js";
import { EngineAudioStream } from "./audio-stream.js";
import { CameraDirector, normalizeEditorialShot, normalizeEditorialShots } from "./cameras.js";
import { FrameBuffer } from "./frame-buffer.js";
import { bindPanel, recordReplayEvent, setConnection, toast, updateFrame, updateIdentity } from "./overlays.js";
import { CinematicPipeline } from "./postfx.js";
import { BroadcastPrecipitation, TyreSpray } from "./precipitation.js";
import { normalizeFrame, normalizeIdentity, vehicleFromIdentity } from "./protocol.js";
import { AdaptiveQuality } from "./quality.js";
import { ObservatoryWorld } from "./world.js";

const app = document.getElementById("app");
const canvas = document.getElementById("stage");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance", stencil: false, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.92;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x9aafb4, 0.00155);
const camera = new THREE.PerspectiveCamera(46, 1, 0.5, 400000);
camera.position.set(8, 4, 10);
const atmosphere = new AtmosphereRig(scene);
const precipitation = new BroadcastPrecipitation(scene);
const tyreSpray = new TyreSpray(scene);
const director = new CameraDirector(camera, canvas, (label) => {
  document.getElementById("camera-shot").textContent = label;
});
const EDITORIAL_SHOT_STORAGE_KEY = "fable-observatory-editorial-shots-v1";
const world = new ObservatoryWorld(scene, renderer);
director.setGroundHeightSampler((x, z, fallback) => world.visualGroundHeightAt(x, z, fallback));
const pipeline = new CinematicPipeline(renderer, scene, camera);
const frames = new FrameBuffer();
const clock = new THREE.Clock();
let currentFrame = null;
let playbackPaused = false;
let playbackRevision = -1;
let lastUiFrameKey = "";
let viewportWidth = 0;
let viewportHeight = 0;

function setPlaybackPaused(value, revision = undefined) {
  const parsedRevision = Number(revision);
  if (Number.isFinite(parsedRevision) && parsedRevision < playbackRevision) return false;
  if (Number.isFinite(parsedRevision)) playbackRevision = parsedRevision;
  playbackPaused = Boolean(value);
  // Control replies are authoritative frames, but a paused viewer may have no
  // later simulation frame to drive the render sampler. Update this one
  // control directly so Pause becomes Resume as soon as that reply arrives.
  const button = document.getElementById("control-pause");
  if (button) button.textContent = playbackPaused ? "Resume" : "Pause";
  return true;
}

function handleSessionEvent(payload) {
  if (payload?.playback) {
    setPlaybackPaused(payload.playback.paused, payload.playback.revision);
  }
  recordReplayEvent(payload);
}

const hemisphere = new THREE.HemisphereLight(0xd0e4ea, 0x2a3828, 1.48);
scene.add(hemisphere);
const sun = new THREE.DirectionalLight(0xfff0d4, 2.45);
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
let currentWeather = "clear-day";
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
let editorialShots = loadEditorialShots();
let selectedEditorialShotId = editorialShots[0]?.id || null;
director.setShotLibrary(editorialShots);

function loadEditorialShots() {
  try {
    const parsed = JSON.parse(localStorage.getItem(EDITORIAL_SHOT_STORAGE_KEY) || "[]");
    return normalizeEditorialShots(parsed);
  } catch {
    return [];
  }
}

function persistEditorialShots() {
  try {
    localStorage.setItem(EDITORIAL_SHOT_STORAGE_KEY, JSON.stringify(editorialShots));
  } catch {
    // Editorial framing is optional; private-mode storage must not affect playback.
  }
}

function selectedEditorialShot() {
  return editorialShots.find((shot) => shot.id === selectedEditorialShotId) || null;
}

function editorElement(id) {
  return document.getElementById(`shot-editor-${id}`);
}

function nextEditorialId() {
  return `editorial-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 6)}`;
}

function newEditorialShot() {
  const ordinal = editorialShots.length + 1;
  return normalizeEditorialShot({
    id: nextEditorialId(),
    label: `EDITORIAL ${ordinal}`,
    anchorProgress: currentFrame?.vehicle?.progress || 0,
    approved: false,
  }, ordinal - 1);
}

function renderShotEditor() {
  const selector = editorElement("select");
  const shot = selectedEditorialShot();
  selector.replaceChildren(...editorialShots.map((candidate) => {
    const option = document.createElement("option");
    option.value = candidate.id;
    option.textContent = `${candidate.approved ? "✓ " : ""}${candidate.label}`;
    return option;
  }));
  selector.value = shot?.id || "";
  const fields = {
    label: shot?.label || "",
    progress: shot ? (shot.anchorProgress * 100).toFixed(1) : "0",
    coverage: shot ? (shot.coverage * 100).toFixed(1) : "1.8",
    side: shot?.side || "left",
    foreground: shot?.foreground || "clear",
    offset: shot?.offset ?? 14,
    height: shot?.height ?? 2.4,
    "look-ahead": shot?.lookAhead ?? 4.5,
    fov: shot?.fov ?? 38,
  };
  for (const [field, value] of Object.entries(fields)) editorElement(field).value = value;
  const editable = Boolean(shot);
  document.querySelectorAll("#view-controls .shot-editor input, #view-controls .shot-editor select:not(#shot-editor-select)")
    .forEach((control) => { control.disabled = !editable; });
  const status = editorElement("status");
  status.textContent = `LOCAL LIBRARY · ${editorialShots.length}${shot?.approved ? " · APPROVED" : ""}`;
  const preview = editorElement("preview");
  const isPreviewing = director.mode === "editor" && director.previewShotId === shot?.id;
  preview.textContent = isPreviewing ? "Return" : "Preview";
  preview.dataset.preview = String(isPreviewing);
  preview.disabled = !editable;
  const approve = editorElement("approve");
  approve.textContent = shot?.approved ? "Approved" : "Approve";
  approve.dataset.approved = String(Boolean(shot?.approved));
  approve.disabled = !editable;
  editorElement("delete").disabled = !editable;
}

function commitEditorialShots() {
  editorialShots = normalizeEditorialShots(editorialShots);
  if (!selectedEditorialShot()) selectedEditorialShotId = editorialShots[0]?.id || null;
  director.setShotLibrary(editorialShots);
  persistEditorialShots();
  renderShotEditor();
}

function updateSelectedEditorialShot() {
  const current = selectedEditorialShot();
  if (!current) return;
  const replacement = normalizeEditorialShot({
    ...current,
    label: editorElement("label").value,
    anchorProgress: Number(editorElement("progress").value) / 100,
    coverage: Number(editorElement("coverage").value) / 100,
    side: editorElement("side").value,
    foreground: editorElement("foreground").value,
    offset: Number(editorElement("offset").value),
    height: Number(editorElement("height").value),
    lookAhead: Number(editorElement("look-ahead").value),
    fov: Number(editorElement("fov").value),
  }, editorialShots.indexOf(current));
  editorialShots = editorialShots.map((shot) => shot.id === current.id ? replacement : shot);
  commitEditorialShots();
}

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
if (!editorialShots.length) {
  const initialShot = newEditorialShot();
  editorialShots = [initialShot];
  selectedEditorialShotId = initialShot.id;
}
commitEditorialShots();

document.getElementById("warning-toggle").addEventListener("click", (event) => {
  const expanded = event.currentTarget.getAttribute("aria-expanded") === "true";
  event.currentTarget.setAttribute("aria-expanded", String(!expanded));
  document.getElementById("identity-warnings").dataset.hidden = String(expanded);
});
document.getElementById("identity-inspector-toggle").addEventListener("click", (event) => {
  const expanded = event.currentTarget.getAttribute("aria-expanded") === "true";
  event.currentTarget.setAttribute("aria-expanded", String(!expanded));
  document.getElementById("identity-inspector").dataset.hidden = String(expanded);
});
document.getElementById("camera-mode").addEventListener("change", (event) => director.setMode(event.target.value));
document.getElementById("weather-mode").addEventListener("change", (event) => setWeather(event.target.value));
editorElement("select").addEventListener("change", (event) => {
  selectedEditorialShotId = event.target.value || null;
  renderShotEditor();
});
[
  "label", "progress", "coverage", "side", "foreground", "offset", "height", "look-ahead", "fov",
].forEach((field) => editorElement(field).addEventListener("change", updateSelectedEditorialShot));
editorElement("new").addEventListener("click", () => {
  const shot = newEditorialShot();
  editorialShots.push(shot);
  selectedEditorialShotId = shot.id;
  commitEditorialShots();
  toast(`Created ${shot.label}. Preview it before approval.`);
});
editorElement("preview").addEventListener("click", () => {
  const shot = selectedEditorialShot();
  if (!shot) return;
  if (director.mode === "editor" && director.previewShotId === shot.id) {
    director.clearEditorialPreview();
    toast("Returned to broadcast director.");
  } else if (director.previewEditorialShot(shot.id)) {
    toast(`Previewing ${shot.label}.`);
  }
  renderShotEditor();
});
editorElement("approve").addEventListener("click", () => {
  const shot = selectedEditorialShot();
  if (!shot) return;
  editorialShots = editorialShots.map((candidate) => candidate.id === shot.id ? { ...candidate, approved: !candidate.approved } : candidate);
  commitEditorialShots();
  toast(shot.approved ? `${shot.label} removed from broadcast rotation.` : `${shot.label} approved for its track segment.`);
});
editorElement("delete").addEventListener("click", () => {
  const shot = selectedEditorialShot();
  if (!shot) return;
  if (director.previewShotId === shot.id) director.clearEditorialPreview();
  editorialShots = editorialShots.filter((candidate) => candidate.id !== shot.id);
  selectedEditorialShotId = editorialShots[0]?.id || null;
  commitEditorialShots();
  toast(`Deleted ${shot.label}.`);
});
const debugToggle = document.getElementById("toggle-visual-debug");
if (new URLSearchParams(location.search).has("debug") && debugToggle) {
  debugToggle.closest("label").hidden = false;
  debugToggle.addEventListener("change", (event) => world.setDebugVisuals(event.target.checked));
}
document.getElementById("control-pause").addEventListener("click", () => client?.send(playbackPaused ? "resume" : "pause"));
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
  client.addEventListener("event", (event) => handleSessionEvent(event.detail));
  client.addEventListener("events", (event) => handleSessionEvent(event.detail));
  client.addEventListener("identity", (event) => {
    acceptIdentity(event.detail);
    setPlaybackPaused(event.detail.playback?.paused, event.detail.playback?.revision);
  });
  client.addEventListener("frame", (event) => {
    acceptIdentity(event.detail.identity);
    if (!setPlaybackPaused(event.detail.replay?.paused, event.detail.replay?.revision)) return;
    frames.push(event.detail);
  });
  audio.addEventListener("status", (event) => {
    const button = document.getElementById("toggle-audio");
    button.classList.toggle("active", event.detail.enabled && !event.detail.degraded);
    button.setAttribute("aria-pressed", String(event.detail.enabled));
    button.textContent = event.detail.degraded ? "AUDIO BUFFER" : event.detail.enabled ? "AUDIO LIVE" : "AUDIO OFF";
    if (!event.detail.enabled || event.detail.degraded) toast(event.detail.label);
  });
  if (session.hello) {
    const identity = normalizeIdentity(session.hello);
    acceptIdentity(identity);
    setPlaybackPaused(identity.playback?.paused, identity.playback?.revision);
  }
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
  const preset = atmosphere.setVisualCondition(mode);
  currentWeather = preset.id;
  app.dataset.weather = preset.id;
  app.dataset.lighting = preset.lighting;
  app.dataset.precipitation = preset.precipitation;
  sun.color.setHex(preset.sunColor);
  sun.intensity = preset.sunIntensity;
  hemisphere.color.setHex(preset.hemiSky);
  hemisphere.groundColor.setHex(preset.hemiGround);
  hemisphere.intensity = preset.hemiIntensity;
  renderer.toneMappingExposure = preset.exposure;
  if (scene.fog) {
    scene.fog.color.setHex(preset.fog);
    scene.fog.density = preset.fogDensity;
  }
  pipeline.setVisualCondition(preset.id);
  world.setVisualCondition(preset.id);
  precipitation.setVisualCondition(preset);
  tyreSpray.setVisualCondition(preset);
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
    world.updateSurfaceDetails(elapsed);
    tyreSpray.update(state, elapsed);
    if (telemetryChanged) {
      updateFrame(frame);
      lastUiFrameKey = frameKey;
    }
    sun.target.position.copy(state.position);
    sun.position.copy(state.position).addScaledVector(atmosphere.sunDirection, 520);
  }
  atmosphere.update(camera.position, elapsed);
  precipitation.update(camera.position, elapsed);
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

setWeather("clear-day");
boot();
animate();
