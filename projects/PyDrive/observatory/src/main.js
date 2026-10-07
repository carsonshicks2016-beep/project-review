import * as THREE from "three";
import "./theme.css";
import { ambienceWord } from "./conditions.js";
import { threeToSim } from "./coords.js";
import { CameraDirector } from "./cameras.js";
import {
  bindChrome, recordReplayEvent, setAudioButton, setConnection, setLoading,
  toast, updateFrame, updateIdentity,
} from "./chrome.js";
import { AudioBridge } from "./audio.js";
import { GroundMist, RainField, TyreSpray, QualityGovernor } from "./fx.js";
import { launchParams, openSession, releaseSession, TelemetryLink } from "./net.js";
import { CinematicPipeline } from "./postfx.js";
import { SkyRig } from "./sky.js";
import { normalizeFrame, Timeline } from "./timeline.js";
import { WorldStage } from "./world.js";

const app = document.getElementById("app");
const canvas = document.getElementById("stage");

const renderer = new THREE.WebGLRenderer({
  canvas,
  // Canvas MSAA is dead weight now that everything composites through
  // EffectComposer; the multisampling that matters is on its render target.
  antialias: false,
  powerPreference: "high-performance",
  alpha: false,
});
renderer.outputColorSpace = THREE.SRGBColorSpace;
// Neutral, not ACES: ACES desaturates and rolls highlights off hard, which
// fights a committed banded palette. Neutral keeps colour separation so the
// stylized bands stay readable. OutputPass applies this — see postfx.js.
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 0.92;
renderer.shadowMap.enabled = true;
// PCFSoftShadowMap was removed in three 0.185 and silently downgrades with a
// console warning; softness now comes from DirectionalLight.shadow.radius.
renderer.shadowMap.type = THREE.PCFShadowMap;

const scene = new THREE.Scene();
// SkyRig owns background/fog; avoid a black clear that fights the dome.
scene.background = new THREE.Color(0xc5d8e0);
const camera = new THREE.PerspectiveCamera(48, 1, 0.2, 12000);
camera.position.set(0, 8, 16);

const sky = new SkyRig(scene);
const world = new WorldStage(scene);
const rain = new RainField(scene);
const mist = new GroundMist(scene);
const spray = new TyreSpray(scene);
const director = new CameraDirector(camera);
const post = new CinematicPipeline(renderer, scene, camera);
const quality = new QualityGovernor(renderer, app);
quality.onQuality = (level) => {
  sky.setShadowQuality(level);
  world.setForestQuality(level);
  post.setQuality(level);
};
quality.apply();
const timeline = new Timeline();
const audio = new AudioBridge();

let link = null;
let session = null;
let weather = "clear-day";
let currentFrame = null;
let pausedLocal = false;
let lastListenerSent = 0;
let lastTs = performance.now();
let disposed = false;

function resize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / Math.max(1, h);
  camera.updateProjectionMatrix();
  quality.apply();
  renderer.setSize(w, h, false);
  post.setSize(w, h, quality.pixelRatio);
}
window.addEventListener("resize", resize);
resize();

function applyWeather(id) {
  weather = id;
  app.dataset.weather = id;
  sky.apply(id);
  world.setCondition(id);
  rain.set(id);
  mist.set(id);
  spray.set(id);
  post.setCondition(id);
  const select = document.getElementById("weather-mode");
  if (select && select.value !== id) select.value = id;
}

function vehicleFolder(car) {
  const key = String(car || "").toLowerCase();
  if (key.includes("919") || key.includes("porsche")) return "porsche_919evo";
  return "mazda787b";
}

function pushListener(frame) {
  if (!link || !frame) return;
  const now = performance.now();
  if (now - lastListenerSent < 50) return;
  lastListenerSent = now;
  const pos = camera.position;
  const sim = threeToSim(pos.x, pos.y, pos.z);
  const audioState = director.audioState();
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
    weather: ambienceWord(weather),
  });
}

bindChrome({
  onCamera: (mode) => {
    director.setMode(mode);
    recordReplayEvent(`camera ${mode}`);
  },
  onWeather: (id) => {
    applyWeather(id);
    recordReplayEvent(`weather ${id}`);
  },
  onPause: () => {
    if (!link) return;
    if (currentFrame?.playback?.paused) link.control("resume");
    else link.control("pause");
  },
  onStep: () => link?.control("step"),
  onReset: () => link?.control("reset"),
  onSpeed: (speed) => link?.control("speed", { speed }),
  onScrub: (secondsAgo) => link?.control("scrub", { seconds_ago: secondsAgo }),
  onAudio: async () => {
    if (!session) return;
    const pressed = document.getElementById("toggle-audio")?.getAttribute("aria-pressed") === "true";
    if (pressed) {
      await audio.disable();
      setAudioButton("off");
      return;
    }
    try {
      await audio.enable(link?.audioUrl);
      setAudioButton(audio.status);
      if (audio.status === "unavailable" || audio.degraded) {
        toast(audio.status === "unavailable" ? "audio unavailable" : "audio stream offline");
      }
    } catch {
      setAudioButton("unavailable");
      toast("audio unavailable");
    }
  },
  onPanel: (name) => {
    world.setBrainVisible(name === "brain");
  },
});

audio.onStatus = (status, detail) => {
  setAudioButton(status);
  if (detail && (status === "unavailable" || status === "offline" || status === "muted")) {
    toast(detail);
  }
};

async function boot() {
  setLoading(false, "Bringing the Nordschleife online", "opening playback session", 0.08);
  const params = launchParams();
  try {
    session = await openSession(params);
  } catch (error) {
    setConnection("error");
    setLoading(false, "Session failed", String(error.message || error), 0);
    toast(String(error.message || error));
    return;
  }

  const hello = session.hello || {};
  updateIdentity(hello.checkpoint);
  setLoading(false, "Loading world assets", "truth road · visual scenery · vehicle", 0.25);

  const carId = vehicleFolder(hello.checkpoint?.car);
  try {
    await world.boot(carId);
  } catch (error) {
    setConnection("error");
    setLoading(false, "Asset load failed", String(error.message || error), 0);
    toast(String(error.message || error));
    return;
  }

  applyWeather("clear-day");
  setLoading(false, "Connecting telemetry", "waiting for first frame", 0.7);

  link = new TelemetryLink(session);
  link.onState = (state) => setConnection(state);
  link.onHello = (message) => {
    updateIdentity(message.checkpoint);
    recordReplayEvent("hello");
  };
  link.onFrame = (raw) => {
    try {
      const frame = normalizeFrame(raw);
      const breakKind = timeline.push(frame);
      if (breakKind) {
        world.snapFilters(frame);
        director.snap(frame);
        recordReplayEvent(breakKind);
      }
      if (frame.playback.paused !== pausedLocal) {
        pausedLocal = frame.playback.paused;
        recordReplayEvent(pausedLocal ? "paused" : "resumed");
      }
    } catch (error) {
      console.warn(error);
    }
  };
  link.onEvent = (message) => {
    const name = message.event || message.type || "event";
    recordReplayEvent(name);
    if (name === "speed_changed") recordReplayEvent(`speed_changed ${message.speed ?? ""}`.trim());
    if (name === "reset" || message.reason === "reset") recordReplayEvent("reset");
    if (message.checkpoint) updateIdentity(message.checkpoint);
  };
  link.onWarning = (message) => {
    recordReplayEvent(message.message || message.code || "warning", true);
    toast(message.message || message.code || "warning");
  };
  link.onError = (message) => {
    setConnection("error");
    toast(message.message || message.code || "telemetry error");
  };
  link.connect();

  setLoading(true, "Live", "ready", 1);
}

function tick(now) {
  if (disposed) return;
  requestAnimationFrame(tick);
  const dt = Math.min(0.05, Math.max(0.001, (now - lastTs) / 1000));
  lastTs = now;

  const frame = timeline.sample(dt);
  if (frame) {
    currentFrame = frame;
    world.applyFrame(frame, dt);
    director.update(frame, dt);
    // Shadow volume must track the car, not the camera — otherwise the
    // subject falls outside the tight ortho frustum at speed.
    sky.follow(world.carRoot.position);
    world.syncTrackLighting(sky);
    world.updateSceneryLod(frame, camera);
    rain.update(camera.position, dt);
    mist.update(world.carRoot.position, dt);
    spray.update({
      position: world.carRoot.position,
      yaw: frame.pose.yaw,
      speedMps: frame.speedMps,
      wheels: frame.wheels,
    }, dt);
    updateFrame(frame, { shotLabel: director.shotLabel() });
    pushListener(frame);
  }
  quality.sample();
  post.render(dt);
}
requestAnimationFrame(tick);

async function shutdown() {
  disposed = true;
  link?.disconnect();
  await audio.disable();
  await releaseSession(session?.session_id);
}

// `persisted` means the page went into the bfcache and can come back; tearing
// the session down there leaves a dead client on restore.
window.addEventListener("pagehide", (event) => {
  if (event.persisted) return;
  shutdown();
});
window.addEventListener("beforeunload", () => {
  shutdown();
});
window.addEventListener("pageshow", (event) => {
  if (!event.persisted) return;
  // Restored from bfcache: the rAF clock is stale, so reset it before the
  // next tick turns the gap into one enormous dt.
  lastTs = performance.now();
});

boot().catch((error) => {
  setConnection("error");
  toast(String(error.message || error));
});

// Debug probe for acceptance / live QA (read-only).
window.__observatory = {
  get frame() { return currentFrame; },
  get weather() { return weather; },
  director,
  world,
  timeline,
  camera,
  post,
  quality,
  sky,
  renderer,
};
