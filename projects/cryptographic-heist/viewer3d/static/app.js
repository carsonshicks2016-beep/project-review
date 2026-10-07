const canvas = document.getElementById("scene");
const ctx = canvas.getContext("2d");
const THREE_MODULE_URL = "https://cdn.jsdelivr.net/npm/three@0.165.0/build/three.module.js";

const els = {
  status: document.getElementById("status"),
  replaySelect: document.getElementById("replay-select"),
  fileReplay: document.getElementById("file-replay"),
  fileInput: document.getElementById("file-input"),
  playToggle: document.getElementById("play-toggle"),
  audioToggle: document.getElementById("audio-toggle"),
  scrub: document.getElementById("scrub"),
  simStats: document.getElementById("sim-stats"),
  radio: document.getElementById("radio"),
  radioCount: document.getElementById("radio-count"),
  predictions: document.getElementById("predictions"),
  confidenceFill: document.getElementById("confidence-fill"),
  confidenceLabel: document.getElementById("confidence-label"),
  soundtrackLabel: document.getElementById("soundtrack-label"),
  agentSelect: document.getElementById("agent-select"),
  rpmVal: document.getElementById("rpm-val"),
  rpmArc: document.getElementById("rpm-arc"),
  speedVal: document.getElementById("speed-val"),
  speedArc: document.getElementById("speed-arc"),
  gforceVal: document.getElementById("gforce-val"),
  timelineCanvas: document.getElementById("timeline-canvas"),
  tires: {
    fl: document.getElementById("tire-fl"),
    fr: document.getElementById("tire-fr"),
    rl: document.getElementById("tire-rl"),
    rr: document.getElementById("tire-rr"),
  }
};

const state = {
  meta: null,
  summary: {},
  frames: [],
  frameIndex: 0,
  frameAccumulator: 0,
  playing: true,
  speed: 1,
  replayName: "",
  replayPath: "",
  camera: { x: 0, y: 0, zoom: 2.4 },
  lastAnimationTime: 0,
  selectedAgent: null,
  timelineCtx: els.timelineCanvas.getContext("2d"),
  audio: {
    enabled: false,
    context: null,
    gain: null,
    oscillators: [],
  },
  three: {
    ready: false,
    failed: false,
    module: null,
    renderer: null,
    scene: null,
    camera: null,
    root: null,
    city: null,
    cars: new Map(),
    trails: new Map(),
    dynamic: null,
    waypoint: null,
    flare: null,
    smokeSystems: new Map(),
    materials: {},
  },
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function resize() {
  const stage = document.querySelector(".stage");
  const cssW = Math.max(1, Math.floor(stage?.clientWidth || canvas.clientWidth || window.innerWidth || 1));
  const cssH = Math.max(1, Math.floor(stage?.clientHeight || canvas.clientHeight || window.innerHeight || 1));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  canvas.width = Math.max(1, Math.floor(cssW * dpr));
  canvas.height = Math.max(1, Math.floor(cssH * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  if (state.three.ready && state.three.renderer && state.three.camera) {
    state.three.renderer.setPixelRatio(dpr);
    state.three.renderer.setSize(cssW, cssH, false);
    state.three.camera.aspect = cssW / cssH;
    state.three.camera.updateProjectionMatrix();
  }

  if (els.timelineCanvas) {
    const pRect = els.timelineCanvas.parentElement.getBoundingClientRect();
    els.timelineCanvas.width = Math.floor(Math.max(1, pRect.width) * dpr);
    els.timelineCanvas.height = Math.floor(Math.max(1, pRect.height) * dpr);
    state.timelineCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
}

async function api(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json();
}

async function loadReplayIndex() {
  try {
    const replays = (await api("/api/replays")).filter((item) => item.valid !== false);
    els.replaySelect.innerHTML = "";
    for (const replay of replays) {
      const option = document.createElement("option");
      option.value = replay.path || `replays/${replay.name}`;
      option.textContent = replay.name;
      els.replaySelect.appendChild(option);
    }
    const requested = new URLSearchParams(window.location.search).get("replay");
    if (requested) els.replaySelect.value = requested;
    if (!els.replaySelect.value && replays[0]) els.replaySelect.value = replays[0].path;
    if (els.replaySelect.value) await loadReplayFromApi(els.replaySelect.value);
  } catch (err) {
    els.status.textContent = `Replay index unavailable: ${err.message}`;
  }
}

async function loadReplayFromApi(path) {
  const payload = await api(`/api/replay?path=${encodeURIComponent(path)}&max_frames=2400`);
  installReplay(payload);
}

function installReplay(payload) {
  state.meta = payload.meta || null;
  state.summary = payload.summary || {};
  state.frames = payload.frames || [];
  state.frameIndex = 0;
  state.frameAccumulator = 0;
  state.replayName = payload.name || "local replay";
  state.replayPath = payload.path || "";
  state.playing = true;
  els.playToggle.textContent = "⏸ PAUSE";
  els.scrub.max = Math.max(0, state.frames.length - 1);
  els.scrub.value = "0";
  
  const first = state.frames[0];
  if (first?.agents?.length) {
    const evader = first.agents.find((agent) => agent.faction === "evader") || first.agents[0];
    state.camera.x = evader.x;
    state.camera.y = evader.y;
    state.selectedAgent = evader.name;
  }
  updateDashboard();
  renderTimeline();
}

function parseReplayJsonl(text, name = "local replay") {
  const lines = text
    .trim()
    .split(/\n+/)
    .map((line) => JSON.parse(line));
  const meta = lines.find((line) => line.type === "meta") || null;
  const frames = lines.filter((line) => line.type === "frame");
  return {
    name,
    path: name,
    meta,
    frames,
    summary: summarizeFrames(frames),
    frame_count: frames.length,
    returned_frames: frames.length,
    stride: 1,
  };
}

function summarizeFrames(frames) {
  const last = frames[frames.length - 1] || {};
  const metrics = last.metrics || {};
  return {
    frames: frames.length,
    duration: frames.length > 1 ? frames[frames.length - 1].time - frames[0].time : 0,
    captures: metrics.captures || 0,
    waypoints_hit: metrics.waypoints_hit || 0,
    deception_score: metrics.deception_score || 0,
    avg_confidence: frames.reduce((sum, frame) => sum + Number(frame.confidence || 0), 0) / Math.max(1, frames.length),
  };
}

function currentFrame() {
  return state.frames[Math.min(state.frameIndex, state.frames.length - 1)];
}

async function initThreeRenderer() {
  try {
    const THREE = await import(THREE_MODULE_URL);
    state.three.module = THREE;
    state.three.scene = new THREE.Scene();
    state.three.scene.background = new THREE.Color(0x030508);
    state.three.scene.fog = new THREE.FogExp2(0x030508, 0.002);
    
    state.three.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1800);
    state.three.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false, powerPreference: "high-performance" });
    state.three.renderer.domElement.className = "three-canvas";
    state.three.renderer.domElement.dataset.renderer = "threejs";
    state.three.renderer.shadowMap.enabled = true;
    state.three.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    
    const stage = document.querySelector(".stage");
    stage.innerHTML = "";
    stage.appendChild(state.three.renderer.domElement);

    state.three.materials = {
      evader: new THREE.MeshPhysicalMaterial({ color: 0xec4655, roughness: 0.1, metalness: 0.8, clearcoat: 1.0, clearcoatRoughness: 0.1 }),
      pursuer: new THREE.MeshPhysicalMaterial({ color: 0x111111, roughness: 0.2, metalness: 0.6, clearcoat: 0.8, clearcoatRoughness: 0.2 }),
      glass: new THREE.MeshPhysicalMaterial({ color: 0x000000, roughness: 0.1, metalness: 0.9, transparent: true, opacity: 0.8, clearcoat: 1.0 }),
      road: new THREE.MeshStandardMaterial({ color: 0x080a0c, roughness: 0.4, metalness: 0.2 }),
      building: new THREE.MeshStandardMaterial({ color: 0x0a0c10, roughness: 0.8, metalness: 0.1 }),
      waypoint: new THREE.MeshBasicMaterial({ color: 0x00f3ff, transparent: true, opacity: 0.6 }),
      flare: new THREE.MeshBasicMaterial({ color: 0xf06b60, transparent: true, opacity: 0.0 }),
      smoke: new THREE.MeshBasicMaterial({ color: 0x444444, transparent: true, opacity: 0.3, depthWrite: false }),
    };

    const ambient = new THREE.HemisphereLight(0x0a101f, 0x030508, 0.6);
    state.three.scene.add(ambient);
    const moonlight = new THREE.DirectionalLight(0x1a2e4c, 0.8);
    moonlight.position.set(-100, 200, -100);
    state.three.scene.add(moonlight);

    state.three.root = new THREE.Group();
    state.three.scene.add(state.three.root);
    state.three.city = buildThreeCity(THREE);
    state.three.root.add(state.three.city);
    state.three.dynamic = new THREE.Group();
    state.three.root.add(state.three.dynamic);

    const ring = new THREE.Mesh(new THREE.TorusGeometry(12, 0.4, 8, 48), state.three.materials.waypoint);
    ring.rotation.x = Math.PI / 2;
    ring.visible = false;
    state.three.waypoint = ring;
    state.three.root.add(ring);

    const flare = new THREE.Mesh(new THREE.SphereGeometry(10, 24, 12), state.three.materials.flare);
    flare.visible = false;
    state.three.flare = flare;
    state.three.root.add(flare);

    state.three.ready = true;
    resize();
  } catch (err) {
    state.three.failed = true;
    els.status.textContent = `Three.js unavailable, using 2D fallback: ${err.message}`;
  }
}

function buildThreeCity(THREE) {
  const group = new THREE.Group();
  const road = new THREE.Mesh(new THREE.PlaneGeometry(2200, 2200), state.three.materials.road);
  road.rotation.x = -Math.PI / 2;
  road.receiveShadow = true;
  group.add(road);

  const grid = new THREE.GridHelper(2200, 44, 0x11161d, 0x0a0d11);
  grid.position.y = 0.05;
  group.add(grid);

  const cell = 104;
  const roadWidth = 34;
  const buildingSize = cell - roadWidth;
  const geometry = new THREE.BoxGeometry(buildingSize, 1, buildingSize);
  
  for (let x = -936; x <= 936; x += cell) {
    for (let z = -936; z <= 936; z += cell) {
      if (Math.abs(x) < roadWidth || Math.abs(z) < roadWidth) continue;
      const hash = Math.abs(Math.sin(x * 12.9898 + z * 78.233));
      const height = 16 + Math.floor(hash * 60);
      const mesh = new THREE.Mesh(geometry, state.three.materials.building);
      mesh.position.set(x + roadWidth / 2 + buildingSize / 2, height / 2, z + roadWidth / 2 + buildingSize / 2);
      mesh.scale.y = height;
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      group.add(mesh);
      
      // Add street lights
      if (hash > 0.8) {
        const streetLight = new THREE.PointLight(0xffaa00, 1.5, 80);
        streetLight.position.set(x + roadWidth, 10, z + roadWidth);
        group.add(streetLight);
      }
    }
  }
  return group;
}

function renderThree(frame) {
  const THREE = state.three.module;
  const scene = state.three.scene;
  const camera = state.three.camera;
  const renderer = state.three.renderer;
  if (!THREE || !scene || !camera || !renderer) return;
  canvas.classList.add("fallback-hidden");
  updateThreeCamera(frame, THREE);
  updateThreeCars(frame, THREE);
  updateThreeTrails(frame, THREE);
  updateThreeVectors(frame, THREE);
  updateThreeWaypoint(frame);
  updateThreeFlares(frame);
  renderer.render(scene, camera);
}

function updateThreeCamera(frame, THREE) {
  const agents = frame.agents || [];
  const evader = agents.find((agent) => agent.faction === "evader") || agents[0];
  if (!evader) return;
  const speed = clamp(evader.speed || 0, 0, 42);
  const yaw = evader.yaw || 0;
  const chaseDistance = 70 + speed * 1.2;
  const height = 60 + speed * 0.5;
  const target = new THREE.Vector3(evader.x, 2, evader.y);
  
  // Look slightly ahead of the evader
  target.x += Math.cos(yaw) * speed * 0.5;
  target.z += Math.sin(yaw) * speed * 0.5;

  const cameraTarget = new THREE.Vector3(
    evader.x - Math.cos(yaw) * chaseDistance,
    height,
    evader.y - Math.sin(yaw) * chaseDistance,
  );
  const current = state.three.camera.position;
  current.lerp(cameraTarget, 0.05); // smooth drone follow
  state.three.camera.lookAt(target);
}

function updateThreeCars(frame, THREE) {
  const live = new Set();
  const time = frame.time || 0;

  for (const agent of frame.agents || []) {
    live.add(agent.name);
    let group = state.three.cars.get(agent.name);
    if (!group) {
      group = createThreeCar(THREE, agent);
      state.three.cars.set(agent.name, group);
      state.three.root.add(group);
      
      const smokeSystem = new THREE.Group();
      state.three.smokeSystems.set(agent.name, smokeSystem);
      state.three.root.add(smokeSystem);
    }
    group.visible = true;
    group.position.set(agent.x, 1.1, agent.y);
    group.rotation.y = -(agent.yaw || 0);

    // Sirens
    if (agent.faction === "pursuer" && group.userData.sirens) {
      const flash = Math.sin(time * 20);
      group.userData.sirens.red.intensity = flash > 0 ? 5 : 0;
      group.userData.sirens.blue.intensity = flash < 0 ? 5 : 0;
    }

    // Smoke
    const smokeSystem = state.three.smokeSystems.get(agent.name);
    if (smokeSystem) {
      // Age existing smoke
      for (let i = smokeSystem.children.length - 1; i >= 0; i--) {
        const p = smokeSystem.children[i];
        p.scale.addScalar(0.2);
        p.material.opacity -= 0.02;
        p.position.y += 0.2;
        if (p.material.opacity <= 0) {
          p.geometry.dispose();
          p.material.dispose();
          smokeSystem.remove(p);
        }
      }
      
      // Emit new smoke if drifting
      const grip = Math.max(...(agent.wheel_grip || [1]));
      const slip = Math.abs(agent.slip_angle || 0);
      if (slip > 0.1 || grip < 0.8) {
        if (Math.random() > 0.3) {
          const smoke = new THREE.Mesh(new THREE.BoxGeometry(2,2,2), state.three.materials.smoke.clone());
          smoke.position.set(
            agent.x - Math.cos(agent.yaw) * 5 + (Math.random()-0.5)*2,
            1.0,
            agent.y - Math.sin(agent.yaw) * 5 + (Math.random()-0.5)*2
          );
          smoke.rotation.y = Math.random() * Math.PI;
          smokeSystem.add(smoke);
        }
      }
    }
  }

  for (const [name, group] of state.three.cars.entries()) {
    if (!live.has(name)) {
      group.visible = false;
      const ss = state.three.smokeSystems.get(name);
      if (ss) ss.children.forEach(c => ss.remove(c));
    }
  }
}

function createThreeCar(THREE, agent) {
  const group = new THREE.Group();
  const length = agent.faction === "evader" ? 10.5 : 12.5;
  const width = agent.faction === "evader" ? 4.8 : 5.4;
  const height = agent.faction === "evader" ? 2.0 : 2.6;
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(length, height, width),
    agent.faction === "evader" ? state.three.materials.evader : state.three.materials.pursuer,
  );
  body.castShadow = true;
  body.receiveShadow = true;
  group.add(body);
  
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(length * 0.4, height * 0.7, width * 0.8), state.three.materials.glass);
  cabin.position.set(length * 0.1, height * 0.65, 0);
  group.add(cabin);

  if (agent.faction === "pursuer") {
    const lightBar = new THREE.Mesh(new THREE.BoxGeometry(1.6, 0.4, width * 0.8), new THREE.MeshBasicMaterial({ color: 0x222222 }));
    lightBar.position.set(0.2, height * 1.1, 0);
    group.add(lightBar);

    const redSiren = new THREE.PointLight(0xff0000, 0, 40);
    redSiren.position.set(0.2, height * 1.5, width * 0.3);
    const blueSiren = new THREE.PointLight(0x0000ff, 0, 40);
    blueSiren.position.set(0.2, height * 1.5, -width * 0.3);
    group.add(redSiren);
    group.add(blueSiren);
    group.userData.sirens = { red: redSiren, blue: blueSiren };
  } else {
    // Tail lights
    const tailLight = new THREE.Mesh(new THREE.BoxGeometry(0.2, 0.4, width*0.8), new THREE.MeshBasicMaterial({color: 0xff0000}));
    tailLight.position.set(-length/2, 0, 0);
    group.add(tailLight);
  }

  return group;
}

function updateThreeTrails(frame, THREE) {
  for (const agent of frame.agents || []) {
    const points = [];
    for (let i = Math.max(0, state.frameIndex - 90); i <= state.frameIndex; i += 4) {
      const past = state.frames[i];
      const pastAgent = past?.agents?.find((item) => item.name === agent.name);
      if (pastAgent) points.push(new THREE.Vector3(pastAgent.x, 0.3, pastAgent.y));
    }
    let line = state.three.trails.get(agent.name);
    if (!line) {
      const material = new THREE.LineBasicMaterial({
        color: agent.faction === "evader" ? 0xec4655 : 0x00f3ff,
        transparent: true,
        opacity: agent.faction === "evader" ? 0.8 : 0.4,
      });
      line = new THREE.Line(new THREE.BufferGeometry(), material);
      state.three.trails.set(agent.name, line);
      state.three.root.add(line);
    }
    line.geometry.dispose();
    line.geometry = new THREE.BufferGeometry().setFromPoints(points.length >= 2 ? points : []);
    line.visible = points.length >= 2;
  }
}

function updateThreeVectors(frame, THREE) {
  const dynamic = state.three.dynamic;
  while (dynamic.children.length) {
    const child = dynamic.children.pop();
    child.geometry?.dispose?.();
    child.material?.dispose?.();
    dynamic.remove(child);
  }
  const confidence = frame.confidence || 0;
  const color = confidence < 0.3 ? 0xf06b60 : confidence < 0.7 ? 0xe8c95c : 0x62d99a;
  for (const [name, pos] of Object.entries(frame.predictions || {})) {
    const agent = frame.agents?.find((item) => item.name === name);
    if (!agent || !Array.isArray(pos)) continue;
    const line = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(agent.x, 2.6, agent.y),
        new THREE.Vector3(pos[0], 2.6, pos[1]),
      ]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.72 }),
    );
    dynamic.add(line);
  }
  const evader = frame.agents?.find((agent) => agent.faction === "evader");
  if (!evader) return;
  for (const agent of frame.agents || []) {
    if (agent.faction !== "pursuer") continue;
    dynamic.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(agent.x, 1.8, agent.y),
        new THREE.Vector3(evader.x, 1.8, evader.y),
      ]),
      new THREE.LineBasicMaterial({ color: 0x4d96ff, transparent: true, opacity: 0.18 }),
    ));
  }
}

function updateThreeWaypoint(frame) {
  if (!state.three.waypoint) return;
  if (!frame.waypoint) {
    state.three.waypoint.visible = false;
    return;
  }
  const pulse = 1 + Math.sin((frame.time || 0) * 10) * 0.2;
  state.three.waypoint.visible = true;
  state.three.waypoint.position.set(frame.waypoint[0], 0.5, frame.waypoint[1]);
  state.three.waypoint.scale.setScalar(pulse);
}

function updateThreeFlares(frame) {
  if (!state.three.flare) return;
  const metrics = frame.metrics || {};
  const impact = Math.max(Number(metrics.impact || 0), Number(metrics.last_collision_impact || 0));
  const eventActive = impact > 0.01 || metrics.waypoint_event || metrics.capture_event;
  const evader = frame.agents?.find((agent) => agent.faction === "evader") || frame.agents?.[0];
  
  if (!eventActive || !evader) {
    state.three.flare.visible = false;
    return;
  }
  state.three.flare.visible = true;
  state.three.flare.position.set(evader.x, 2.8, evader.y);
  const scale = metrics.capture_event ? 4.0 : metrics.waypoint_event ? 3.0 : clamp(impact, 0.8, 5.0);
  state.three.flare.scale.setScalar(scale);
  state.three.flare.material.color.setHex(metrics.waypoint_event ? 0x00f3ff : 0xffaa00);
  state.three.flare.material.opacity = metrics.capture_event ? 0.6 : metrics.waypoint_event ? 0.4 : 0.3;
}

function animate(ts) {
  resize();
  const delta = state.lastAnimationTime ? ts - state.lastAnimationTime : 16;
  state.lastAnimationTime = ts;
  if (state.playing && state.frames.length > 1) {
    state.frameAccumulator += (delta / 1000) * 60 * state.speed;
    const advance = Math.floor(state.frameAccumulator);
    if (advance > 0) {
      state.frameAccumulator -= advance;
      state.frameIndex = (state.frameIndex + advance) % state.frames.length;
      els.scrub.value = String(state.frameIndex);
      updateDashboard();
    }
  }
  
  const frame = currentFrame();
  draw();
  renderTimeline();
  requestAnimationFrame(animate);
}

function draw() {
  const w = canvas.clientWidth;
  const h = canvas.clientHeight;
  ctx.clearRect(0, 0, w, h);
  const frame = currentFrame();
  if (!frame) {
    canvas.classList.remove("fallback-hidden");
    drawEmptyCity(w, h);
    return;
  }
  updateCamera(frame, w, h);
  if (state.three.ready) {
    renderThree(frame);
    return;
  }
  canvas.classList.remove("fallback-hidden");
  drawCity(w, h);
  drawWaypoint(frame);
  drawTrails(frame);
  drawPredictions(frame);
  drawContainment(frame);
  for (const agent of frame.agents || []) drawCar(agent);
  drawEventFlares(frame);
  drawVignette(w, h);
}

function updateCamera(frame, w, h) {
  const agents = frame.agents || [];
  const evader = agents.find((agent) => agent.faction === "evader") || agents[0];
  if (!evader) return;
  const avg = agents.reduce((acc, agent) => {
    acc.x += agent.x;
    acc.y += agent.y;
    return acc;
  }, { x: 0, y: 0 });
  avg.x /= Math.max(1, agents.length);
  avg.y /= Math.max(1, agents.length);
  const lead = clamp(evader.speed || 0, 0, 38) * 0.55;
  const targetX = evader.x * 0.78 + avg.x * 0.22 + Math.cos(evader.yaw || 0) * lead;
  const targetY = evader.y * 0.78 + avg.y * 0.22 + Math.sin(evader.yaw || 0) * lead;
  const targetZoom = clamp(Math.min(w, h) / (260 + clamp(evader.speed || 0, 0, 38) * 8), 1.15, 3.2);
  state.camera.x += (targetX - state.camera.x) * 0.08;
  state.camera.y += (targetY - state.camera.y) * 0.08;
  state.camera.zoom += (targetZoom - state.camera.zoom) * 0.05;
}

function worldToScreen(x, y) {
  return {
    x: canvas.clientWidth / 2 + (x - state.camera.x) * state.camera.zoom,
    y: canvas.clientHeight / 2 + (y - state.camera.y) * state.camera.zoom,
  };
}

function drawEmptyCity(w, h) {
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "#202a35";
  ctx.lineWidth = 1;
  for (let y = 40; y < h; y += 80) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  for (let x = 40; x < w; x += 100) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
  }
}

function drawCity(w, h) {
  ctx.fillStyle = "#0b0f14";
  ctx.fillRect(0, 0, w, h);
  const cell = 104;
  const road = 34;
  const left = state.camera.x - w / state.camera.zoom / 2 - cell;
  const right = state.camera.x + w / state.camera.zoom / 2 + cell;
  const top = state.camera.y - h / state.camera.zoom / 2 - cell;
  const bottom = state.camera.y + h / state.camera.zoom / 2 + cell;
  const startX = Math.floor(left / cell) * cell;
  const startY = Math.floor(top / cell) * cell;

  ctx.fillStyle = "#121821";
  for (let x = startX; x <= right; x += cell) {
    for (let y = startY; y <= bottom; y += cell) {
      const p = worldToScreen(x + road / 2, y + road / 2);
      const size = (cell - road) * state.camera.zoom;
      ctx.fillRect(p.x, p.y, size, size);
      ctx.strokeStyle = "rgba(255,255,255,0.025)";
      ctx.strokeRect(p.x, p.y, size, size);
    }
  }

  ctx.strokeStyle = "#273340";
  ctx.lineWidth = 1;
  for (let x = startX; x <= right; x += cell) {
    const p = worldToScreen(x, state.camera.y);
    ctx.beginPath();
    ctx.moveTo(p.x, 0);
    ctx.lineTo(p.x, h);
    ctx.stroke();
  }
  for (let y = startY; y <= bottom; y += cell) {
    const p = worldToScreen(state.camera.x, y);
    ctx.beginPath();
    ctx.moveTo(0, p.y);
    ctx.lineTo(w, p.y);
    ctx.stroke();
  }
}

function drawWaypoint(frame) {
  if (!frame.waypoint) return;
  const p = worldToScreen(frame.waypoint[0], frame.waypoint[1]);
  const pulse = 0.5 + 0.5 * Math.sin((frame.time || 0) * 7);
  ctx.save();
  ctx.strokeStyle = `rgba(232, 201, 92, ${0.45 + pulse * 0.35})`;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(p.x, p.y, 16 + pulse * 7, 0, Math.PI * 2);
  ctx.stroke();
  ctx.fillStyle = "#e8c95c";
  ctx.fillRect(p.x - 4, p.y - 4, 8, 8);
  ctx.restore();
}

function drawTrails(frame) {
  const agents = frame.agents || [];
  for (const agent of agents) {
    const color = agent.faction === "evader" ? "236,70,85" : "77,150,255";
    ctx.strokeStyle = `rgba(${color}, ${agent.faction === "evader" ? 0.38 : 0.20})`;
    ctx.lineWidth = agent.faction === "evader" ? 2 : 1.4;
    ctx.beginPath();
    let started = false;
    for (let i = Math.max(0, state.frameIndex - 160); i <= state.frameIndex; i += 5) {
      const past = state.frames[i];
      const pastAgent = past?.agents?.find((item) => item.name === agent.name);
      if (!pastAgent) continue;
      const p = worldToScreen(pastAgent.x, pastAgent.y);
      if (!started) {
        ctx.moveTo(p.x, p.y);
        started = true;
      } else {
        ctx.lineTo(p.x, p.y);
      }
    }
    ctx.stroke();
  }
}

function drawPredictions(frame) {
  const predictions = frame.predictions || {};
  ctx.save();
  ctx.setLineDash([6, 6]);
  for (const [name, pos] of Object.entries(predictions)) {
    const agent = frame.agents?.find((item) => item.name === name);
    if (!agent || !Array.isArray(pos)) continue;
    const from = worldToScreen(agent.x, agent.y);
    const to = worldToScreen(pos[0], pos[1]);
    ctx.strokeStyle = confidenceColor(frame.confidence || 0, 0.62);
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.fillStyle = confidenceColor(frame.confidence || 0, 0.90);
    ctx.fillRect(to.x - 3, to.y - 3, 6, 6);
  }
  ctx.restore();
}

function drawContainment(frame) {
  const evader = frame.agents?.find((agent) => agent.faction === "evader");
  if (!evader) return;
  const ev = worldToScreen(evader.x, evader.y);
  ctx.strokeStyle = "rgba(77, 150, 255, 0.18)";
  ctx.lineWidth = 1;
  for (const agent of frame.agents || []) {
    if (agent.faction !== "pursuer") continue;
    const p = worldToScreen(agent.x, agent.y);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
    ctx.lineTo(ev.x, ev.y);
    ctx.stroke();
  }
}

function drawCar(agent) {
  const p = worldToScreen(agent.x, agent.y);
  const length = agent.faction === "evader" ? 25 : 30;
  const width = agent.faction === "evader" ? 12 : 14;
  const color = agent.faction === "evader" ? "#ec4655" : "#4d96ff";
  const glow = Math.max(...(agent.wheel_grip || [0]));
  ctx.save();
  ctx.translate(p.x, p.y);
  ctx.rotate(agent.yaw || 0);
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.fillRect(-length / 2 + 2, -width / 2 + 3, length, width);
  ctx.fillStyle = color;
  ctx.fillRect(-length / 2, -width / 2, length, width);
  ctx.fillStyle = agent.faction === "evader" ? "#ffd6db" : "#d8ecff";
  ctx.fillRect(length * 0.12, -width * 0.28, length * 0.28, width * 0.56);
  ctx.fillStyle = glow > 0.85 ? "#f4a84e" : "#080a0d";
  ctx.fillRect(-length * 0.34, -width / 2 - 2, 6, 3);
  ctx.fillRect(-length * 0.34, width / 2 - 1, 6, 3);
  ctx.fillRect(length * 0.20, -width / 2 - 2, 6, 3);
  ctx.fillRect(length * 0.20, width / 2 - 1, 6, 3);
  if (agent.faction === "pursuer") {
    ctx.fillStyle = "#ff5b6a";
    ctx.fillRect(-2, -2, 4, 4);
  }
  ctx.restore();
  ctx.fillStyle = "#dce6f2";
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace";
  ctx.fillText(agent.role || agent.name, p.x + 12, p.y - 10);
}

function drawEventFlares(frame) {
  const metrics = frame.metrics || {};
  const impact = Math.max(Number(metrics.impact || 0), Number(metrics.last_collision_impact || 0));
  if (impact <= 0.01 && !metrics.waypoint_event && !metrics.capture_event) return;
  const evader = frame.agents?.find((agent) => agent.faction === "evader") || frame.agents?.[0];
  if (!evader) return;
  const p = worldToScreen(evader.x, evader.y);
  const radius = metrics.capture_event ? 70 : metrics.waypoint_event ? 48 : clamp(impact * 18, 22, 80);
  ctx.save();
  ctx.strokeStyle = metrics.capture_event ? "rgba(240,107,96,0.72)" : "rgba(232,201,92,0.68)";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
  ctx.stroke();
  ctx.restore();
}

function drawVignette(w, h) {
  const gradient = ctx.createRadialGradient(w / 2, h / 2, Math.min(w, h) * 0.2, w / 2, h / 2, Math.max(w, h) * 0.65);
  gradient.addColorStop(0, "rgba(0,0,0,0)");
  gradient.addColorStop(1, "rgba(0,0,0,0.48)");
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, w, h);
}

function confidenceColor(confidence, alpha) {
  if (confidence < 0.3) return `rgba(240, 107, 96, ${alpha})`;
  if (confidence < 0.7) return `rgba(232, 201, 92, ${alpha})`;
  return `rgba(98, 217, 154, ${alpha})`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function renderTimeline() {
  if (!els.timelineCanvas || !state.frames.length) return;
  const c = els.timelineCanvas;
  const ctx2 = state.timelineCtx;
  const w = c.clientWidth;
  const h = c.clientHeight;
  ctx2.clearRect(0, 0, w, h);

  ctx2.beginPath();
  ctx2.moveTo(0, h / 2);
  for (let i = 0; i < state.frames.length; i++) {
    const x = (i / state.frames.length) * w;
    const f = state.frames[i];
    const m = f.metrics || {};
    let val = 2;
    if (m.waypoint_event) val = 15;
    else if (m.capture_event) val = 25;
    else if (m.impact > 0) val = clamp(m.impact * 10, 5, 20);
    if (f.radio?.some((r) => r.spoofed)) val += 10;
    ctx2.lineTo(x, h / 2 - val);
  }
  ctx2.strokeStyle = "rgba(0, 243, 255, 0.4)";
  ctx2.lineWidth = 1;
  ctx2.stroke();

  const currX = (state.frameIndex / state.frames.length) * w;
  ctx2.beginPath();
  ctx2.moveTo(currX, 0);
  ctx2.lineTo(currX, h);
  ctx2.strokeStyle = "#ff00ff";
  ctx2.lineWidth = 2;
  ctx2.stroke();
}

function updateDashboard() {
  const frame = currentFrame();
  if (!frame) {
    els.status.textContent = "No replay loaded";
    return;
  }

  const metrics = frame.metrics || {};
  const confidence = clamp(Number(frame.confidence || 0), 0, 1);
  const radioEvents = uniqueRadio(frame.radio || []);

  els.status.textContent = `${state.replayName} | TIME: ${Number(frame.time || 0).toFixed(2)}s`;
  els.simStats.textContent = `${Math.floor((frame.time || 0) / 60).toString().padStart(2, "0")}:${(Math.floor(frame.time || 0) % 60).toString().padStart(2, "0")}`;

  const agents = frame.agents || [];
  if (els.agentSelect.options.length !== agents.length) {
    els.agentSelect.innerHTML = agents.map((a) => `<option value="${a.name}">${a.role || a.name}</option>`).join("");
    if (!state.selectedAgent) state.selectedAgent = agents[0]?.name;
    els.agentSelect.value = state.selectedAgent;
  }

  const selectedAgentData = agents.find((a) => a.name === state.selectedAgent) || agents[0];
  if (selectedAgentData) {
    const speed = selectedAgentData.speed || 0;
    const rpm = speed * 120 + 1000 + Math.random() * 200;
    els.speedVal.textContent = Math.floor(speed * 3.6);
    els.rpmVal.textContent = Math.floor(rpm / 100) * 100;
    const speedPct = clamp(speed / 45, 0, 1);
    els.speedArc.style.strokeDashoffset = 125.6 - speedPct * 125.6;
    const rpmPct = clamp(rpm / 8000, 0, 1);
    els.rpmArc.style.strokeDashoffset = 125.6 - rpmPct * 125.6;
    const gforce = selectedAgentData.gforce || speed * (selectedAgentData.slip_angle || 0) * 0.1 + 1.0;
    els.gforceVal.textContent = `${Math.abs(gforce).toFixed(1)}G`;
    const grips = selectedAgentData.wheel_grip || [1, 1, 1, 1];
    els.tires.fl.style.backgroundColor = grips[0] < 0.6 ? "#ffaa00" : "#333";
    els.tires.fr.style.backgroundColor = grips[1] < 0.6 ? "#ffaa00" : "#333";
    els.tires.rl.style.backgroundColor = grips[2] < 0.6 ? "#f06b60" : "#333";
    els.tires.rr.style.backgroundColor = grips[3] < 0.6 ? "#f06b60" : "#333";
  }

  els.confidenceLabel.textContent = `${Math.round(confidence * 100)}%`;
  els.confidenceFill.style.width = `${confidence * 100}%`;
  if (confidence < 0.3) els.confidenceFill.classList.add("pulsing-low");
  else els.confidenceFill.classList.remove("pulsing-low");
  els.soundtrackLabel.textContent = soundtrackMode(confidence);

  els.radioCount.textContent = `${radioEvents.length}`;
  const recentRadio = radioEvents.slice(-10);
  let html = "";
  for (const event of recentRadio) {
    html += `
      <div class="radio-line ${event.spoofed ? "spoof" : ""}">
        <span>${Number(event.time || 0).toFixed(2)}</span>
        <strong>${escapeHtml(event.speaker || "?")}</strong>
        <code>${event.spoofed ? "[SPOOF_INJECT]: " : ""}${escapeHtml(event.word || "")}</code>
      </div>
    `;
  }
  if (els.radio.innerHTML !== html) {
    els.radio.innerHTML = html;
    els.radio.scrollTop = els.radio.scrollHeight;
  }

  if (els.predictions) {
    els.predictions.innerHTML = Object.entries(frame.predictions || {}).slice(0, 5).map(([name, pos]) => `
      <div class="prediction">
        <span>${escapeHtml(name)}</span>
        <code>${Number(pos[0] || 0).toFixed(1)}, ${Number(pos[1] || 0).toFixed(1)}</code>
      </div>
    `).join("");
  }

  void metrics;
  updateAudio(confidence);
}

function uniqueRadio(events) {
  const seen = new Set();
  const out = [];
  for (const event of events) {
    const key = `${Number(event.time || 0).toFixed(4)}|${event.speaker}|${event.word}|${event.spoofed}`;
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(event);
  }
  return out;
}

function soundtrackMode(confidence) {
  if (!state.audio.enabled) {
    if (confidence < 0.3) return "low confidence: dissonant jazz map";
    if (confidence < 0.7) return "mid confidence: hybrid tension map";
    return "high confidence: structured classical map";
  }
  if (confidence < 0.3) return "dissonant jazz engine";
  if (confidence < 0.7) return "hybrid tension engine";
  return "structured classical engine";
}

async function toggleAudio() {
  if (!state.audio.context) {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    state.audio.context = new AudioContext();
    state.audio.gain = state.audio.context.createGain();
    state.audio.gain.gain.value = 0.045;
    state.audio.gain.connect(state.audio.context.destination);
    state.audio.oscillators = [0, 1, 2, 3].map(() => {
      const osc = state.audio.context.createOscillator();
      const gain = state.audio.context.createGain();
      gain.gain.value = 0.18;
      osc.type = "sine";
      osc.connect(gain);
      gain.connect(state.audio.gain);
      osc.start();
      return { osc, gain };
    });
  }
  if (state.audio.context.state === "suspended") await state.audio.context.resume();
  state.audio.enabled = !state.audio.enabled;
  state.audio.gain.gain.setTargetAtTime(state.audio.enabled ? 0.045 : 0.0001, state.audio.context.currentTime, 0.08);
  els.audioToggle.textContent = state.audio.enabled ? "Audio On" : "Audio Off";
  updateDashboard();
}

function updateAudio(confidence) {
  if (!state.audio.context || !state.audio.oscillators.length) return;
  const now = state.audio.context.currentTime;
  const low = [93, 139, 211, 307];
  const mid = [110, 165, 247, 330];
  const high = [196, 247, 294, 392];
  const bank = confidence < 0.3 ? low : confidence < 0.7 ? mid : high;
  state.audio.oscillators.forEach((node, index) => {
    const wobble = confidence < 0.3 ? Math.sin(now * (7 + index)) * 9 : Math.sin(now * 0.8) * 1.5;
    node.osc.type = confidence < 0.3 ? "sawtooth" : confidence < 0.7 ? "triangle" : "sine";
    node.osc.frequency.setTargetAtTime(bank[index] + wobble, now, 0.05);
    node.gain.gain.setTargetAtTime(state.audio.enabled ? 0.16 : 0.0001, now, 0.08);
  });
}

function bindControls() {
  els.replaySelect.addEventListener("change", () => {
    if (els.replaySelect.value) loadReplayFromApi(els.replaySelect.value).catch((err) => {
      els.status.textContent = `Load failed: ${err.message}`;
    });
  });
  els.fileReplay.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", async () => {
    const file = els.fileInput.files?.[0];
    if (!file) return;
    installReplay(parseReplayJsonl(await file.text(), file.name));
  });
  els.playToggle.addEventListener("click", () => {
    state.playing = !state.playing;
    els.playToggle.textContent = state.playing ? "⏸ PAUSE" : "▶ PLAY";
  });
  els.audioToggle.addEventListener("click", toggleAudio);
  els.scrub.addEventListener("input", () => {
    state.frameIndex = Number(els.scrub.value || 0);
    state.frameAccumulator = 0;
    updateDashboard();
    renderTimeline();
  });
  els.agentSelect.addEventListener("change", () => {
    state.selectedAgent = els.agentSelect.value;
    updateDashboard();
  });
  window.addEventListener("keydown", (event) => {
    if (event.code !== "Space") return;
    event.preventDefault();
    els.playToggle.click();
  });
}

bindControls();
initThreeRenderer();
loadReplayIndex();
requestAnimationFrame(animate);
