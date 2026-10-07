/**
 * Autonomous Satellite Rendezvous & Docking 3D Visualizer & Mission Control
 * Using Three.js, WebGL, and Clohessy-Wiltshire relative orbital mechanics.
 */

// Scene Globals
let scene, camera, renderer, controls;
let targetSatellite, chaserSatellite, approachCone, earthMesh;
let thrusterPlumes = {};
let trajectoryLine, trajectoryGeometry;

// Simulation / Playback State
let trajectoryData = null;
let currentEpisodeIndex = 0;
let currentStep = 0;
let isPlaying = true;
let playbackSpeed = 1.0;
let lastFrameTime = performance.now();
let stepAccumulator = 0;

// Camera Mode: "chase", "dock", "orbit"
let cameraMode = "chase";

// Manual Pilot Mode
let isManualMode = false;
let manualState = [0, 50, 0, 0, -0.2, 0]; // [x, y, z, vx, vy, vz]
let manualFuel = 50.0;
let manualDeltaV = 0.0;
let keysPressed = {};

// Constant orbital rate for 400km LEO (approx 0.00113 rad/s)
const OMEGA = 0.0011311;

// Init on DOM load
window.addEventListener("DOMContentLoaded", () => {
  initThree();
  initUI();
  loadTrajectories();
  animate();
});

function initThree() {
  const container = document.getElementById("canvas-container");
  const width = window.innerWidth;
  const height = window.innerHeight;

  // Scene
  scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x030712, 0.0015);

  // Camera
  camera = new THREE.PerspectiveCamera(55, width / height, 0.1, 5000);
  camera.position.set(15, 25, 45);

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;
  container.appendChild(renderer.domElement);

  // Orbit Controls
  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxDistance = 1200;
  controls.minDistance = 0.5;

  // Lights
  const sunLight = new THREE.DirectionalLight(0xffffff, 1.8);
  sunLight.position.set(200, 150, 300);
  scene.add(sunLight);

  const ambientLight = new THREE.AmbientLight(0x223355, 0.7);
  scene.add(ambientLight);

  const dockLight = new THREE.PointLight(0x38bdf8, 1.2, 30);
  dockLight.position.set(0, 2, 0);
  scene.add(dockLight);

  // Build Environment
  createStarfield();
  createEarth();
  createTargetSatellite();
  createApproachCorridor();
  createChaserSatellite();
  createTrajectoryRibbon();

  // Resize handler
  window.addEventListener("resize", () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
  });
}

function createStarfield() {
  const count = 2500;
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);

  for (let i = 0; i < count; i++) {
    const r = 2000 + Math.random() * 800;
    const theta = Math.random() * Math.PI * 2;
    const phi = Math.acos(2 * Math.random() - 1);

    positions[i * 3] = r * Math.sin(phi) * Math.cos(theta);
    positions[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
    positions[i * 3 + 2] = r * Math.cos(phi);

    // Subtle star colors (blue, yellow, white)
    const tint = Math.random();
    colors[i * 3] = tint > 0.8 ? 0.8 : 1.0;
    colors[i * 3 + 1] = tint > 0.8 ? 0.9 : 1.0;
    colors[i * 3 + 2] = tint > 0.5 ? 1.0 : 0.8;
  }

  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));

  const material = new THREE.PointsMaterial({
    size: 2.2,
    vertexColors: true,
    transparent: true,
    opacity: 0.85
  });

  const stars = new THREE.Points(geometry, material);
  scene.add(stars);
}

function createEarth() {
  // In LVLH frame: -X points towards Earth Center.
  // We place Earth below the orbit along -X (in Three.js coordinates: Y is up/radial X, so -Y in Three.js)
  const earthRadius = 450;
  const earthGeo = new THREE.SphereGeometry(earthRadius, 48, 48);

  // Procedural Earth shader / material
  const earthMat = new THREE.MeshStandardMaterial({
    color: 0x1d4ed8,
    roughness: 0.8,
    metalness: 0.1
  });

  earthMesh = new THREE.Mesh(earthGeo, earthMat);
  // Positioned along -X radial (mapping: X_lvlh -> X, Y_lvlh -> Z, Z_lvlh -> Y)
  earthMesh.position.set(-earthRadius - 80, -80, -200);
  scene.add(earthMesh);

  // Atmosphere glow ring
  const atmoGeo = new THREE.SphereGeometry(earthRadius + 6, 32, 32);
  const atmoMat = new THREE.MeshBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.12,
    side: THREE.BackSide
  });
  const atmoMesh = new THREE.Mesh(atmoGeo, atmoMat);
  earthMesh.add(atmoMesh);
}

function createTargetSatellite() {
  targetSatellite = new THREE.Group();

  // Main Bus (Titanium gold foil cuboid)
  const busGeo = new THREE.BoxGeometry(2.4, 2.4, 3.2);
  const busMat = new THREE.MeshStandardMaterial({
    color: 0xd97706,
    metalness: 0.85,
    roughness: 0.25
  });
  const bus = new THREE.Mesh(busGeo, busMat);
  targetSatellite.add(bus);

  // Solar Array Truss & Panels (+/- Z cross-track)
  const panelGeo = new THREE.BoxGeometry(0.1, 1.8, 6.5);
  const panelMat = new THREE.MeshStandardMaterial({
    color: 0x0284c7,
    roughness: 0.3,
    metalness: 0.6
  });

  const panelLeft = new THREE.Mesh(panelGeo, panelMat);
  panelLeft.position.set(0, 0, 5.0);
  targetSatellite.add(panelLeft);

  const panelRight = new THREE.Mesh(panelGeo, panelMat);
  panelRight.position.set(0, 0, -5.0);
  targetSatellite.add(panelRight);

  // High Gain Dish Antenna
  const dishGeo = new THREE.CylinderGeometry(0.9, 0.1, 0.4, 24);
  const dishMat = new THREE.MeshStandardMaterial({ color: 0xe2e8f0, roughness: 0.4 });
  const dish = new THREE.Mesh(dishGeo, dishMat);
  dish.position.set(1.4, 0, 0);
  dish.rotation.z = Math.PI / 2;
  targetSatellite.add(dish);

  // Docking Adapter Ring on +Y (Along-track V-bar face)
  // Mapping: LVLH Y is forward along-track. In Three.js, we align target along Y axis!
  const dockRingGeo = new THREE.TorusGeometry(0.65, 0.08, 16, 32);
  const dockRingMat = new THREE.MeshStandardMaterial({
    color: 0x22c55e,
    emissive: 0x14532d,
    roughness: 0.2,
    metalness: 0.8
  });
  const dockRing = new THREE.Mesh(dockRingGeo, dockRingMat);
  dockRing.position.set(0, 1.65, 0);
  dockRing.rotation.x = Math.PI / 2;
  targetSatellite.add(dockRing);

  // Docking Guide Petals (Crosshairs)
  const petalMat = new THREE.MeshBasicMaterial({ color: 0x4ade80 });
  for (let i = 0; i < 4; i++) {
    const angle = (i * Math.PI) / 2;
    const petalGeo = new THREE.BoxGeometry(0.08, 0.25, 0.04);
    const petal = new THREE.Mesh(petalGeo, petalMat);
    petal.position.set(Math.cos(angle) * 0.75, 1.65, Math.sin(angle) * 0.75);
    petal.rotation.y = -angle;
    targetSatellite.add(petal);
  }

  // Blinking Nav Beacon
  const beaconGeo = new THREE.SphereGeometry(0.12, 12, 12);
  const beaconMat = new THREE.MeshBasicMaterial({ color: 0x22c55e });
  const beacon = new THREE.Mesh(beaconGeo, beaconMat);
  beacon.position.set(0, 1.75, 0);
  targetSatellite.add(beacon);

  scene.add(targetSatellite);
}

function createApproachCorridor() {
  approachCone = new THREE.Group();

  // 25 degree cone opening along +Y
  const maxRange = 120;
  const halfAngle = 25 * (Math.PI / 180);
  const radius = maxRange * Math.tan(halfAngle);

  // 4 guideline rays
  const lineMat = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.35
  });

  for (let i = 0; i < 4; i++) {
    const angle = (i * Math.PI) / 2;
    const x = Math.cos(angle) * radius;
    const z = Math.sin(angle) * radius;
    const pts = [new THREE.Vector3(0, 0, 0), new THREE.Vector3(x, maxRange, z)];
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    approachCone.add(new THREE.Line(geo, lineMat));
  }

  // Distance range rings (10m, 25m, 50m, 100m)
  const ranges = [10, 25, 50, 100];
  ranges.forEach(r => {
    const ringRadius = r * Math.tan(halfAngle);
    const ringGeo = new THREE.BufferGeometry();
    const ringPts = [];
    const segments = 48;
    for (let j = 0; j <= segments; j++) {
      const theta = (j / segments) * Math.PI * 2;
      ringPts.push(new THREE.Vector3(Math.cos(theta) * ringRadius, r, Math.sin(theta) * ringRadius));
    }
    ringGeo.setFromPoints(ringPts);
    const ringMat = new THREE.LineBasicMaterial({
      color: 0x0ea5e9,
      transparent: true,
      opacity: 0.4
    });
    approachCone.add(new THREE.Line(ringGeo, ringMat));
  });

  scene.add(approachCone);
}

function createChaserSatellite() {
  chaserSatellite = new THREE.Group();

  // Chaser Main Body
  const bodyGeo = new THREE.CylinderGeometry(0.7, 0.85, 1.4, 8);
  const bodyMat = new THREE.MeshStandardMaterial({
    color: 0x94a3b8,
    metalness: 0.8,
    roughness: 0.3
  });
  const body = new THREE.Mesh(bodyGeo, bodyMat);
  chaserSatellite.add(body);

  // Docking Probe Cone on -Y face (facing target when moving forward)
  const probeGeo = new THREE.CylinderGeometry(0.15, 0.45, 0.5, 16);
  const probeMat = new THREE.MeshStandardMaterial({ color: 0x38bdf8, metalness: 0.9 });
  const probe = new THREE.Mesh(probeGeo, probeMat);
  probe.position.set(0, -0.85, 0);
  chaserSatellite.add(probe);

  // 6 RCS Thruster Blocks & Animated Plumes
  // We place nozzles on 6 sides: +/-X, +/-Y, +/-Z
  const directions = [
    { name: "x_pos", dir: [1, 0, 0], rot: [0, 0, -Math.PI / 2] },
    { name: "x_neg", dir: [-1, 0, 0], rot: [0, 0, Math.PI / 2] },
    { name: "y_pos", dir: [0, 1, 0], rot: [0, 0, 0] },
    { name: "y_neg", dir: [0, -1, 0], rot: [Math.PI, 0, 0] },
    { name: "z_pos", dir: [0, 0, 1], rot: [Math.PI / 2, 0, 0] },
    { name: "z_neg", dir: [0, 0, -1], rot: [-Math.PI / 2, 0, 0] }
  ];

  const plumeGeo = new THREE.ConeGeometry(0.18, 0.7, 12);
  const plumeMat = new THREE.MeshBasicMaterial({
    color: 0x67e8f9,
    transparent: true,
    opacity: 0.0
  });

  directions.forEach(d => {
    const nozzleGeo = new THREE.CylinderGeometry(0.06, 0.1, 0.2, 8);
    const nozzleMat = new THREE.MeshStandardMaterial({ color: 0x334155 });
    const nozzle = new THREE.Mesh(nozzleGeo, nozzleMat);
    nozzle.position.set(d.dir[0] * 0.8, d.dir[1] * 0.8, d.dir[2] * 0.8);
    nozzle.rotation.set(d.rot[0], d.rot[1], d.rot[2]);
    chaserSatellite.add(nozzle);

    const plume = new THREE.Mesh(plumeGeo, plumeMat.clone());
    plume.position.set(0, 0.45, 0);
    nozzle.add(plume);
    thrusterPlumes[d.name] = plume;
  });

  scene.add(chaserSatellite);
}

function createTrajectoryRibbon() {
  const maxPts = 500;
  trajectoryGeometry = new THREE.BufferGeometry();
  const positions = new Float32Array(maxPts * 3);
  trajectoryGeometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  trajectoryGeometry.setDrawRange(0, 0);

  const mat = new THREE.LineBasicMaterial({
    color: 0x38bdf8,
    transparent: true,
    opacity: 0.75,
    linewidth: 2
  });

  trajectoryLine = new THREE.Line(trajectoryGeometry, mat);
  scene.add(trajectoryLine);
}

// Coordinate mapping from LVLH to Three.js world coordinates:
// In LVLH:
//   x: Radial (away from Earth) -> Three.js X
//   y: Along-track / V-bar -> Three.js Y (so target docking port faces +Y)
//   z: Cross-track -> Three.js Z
function lvlhToThree(x, y, z) {
  return new THREE.Vector3(x, y, z);
}

function loadTrajectories() {
  fetch("trajectories.json")
    .then(res => res.json())
    .then(data => {
      trajectoryData = data;
      populateEpisodeDropdown();
      loadEpisode(0);
    })
    .catch(err => {
      console.warn("Could not load trajectories.json via HTTP fetch, generating orbital preview trajectory...", err);
      generateFallbackTrajectory();
    });
}

function generateFallbackTrajectory() {
  const steps = 180;
  const states = [];
  const actions = [];
  const distances = [];
  const speeds = [];

  let x = 1.2, y = 30.0, z = 0.8;
  let vx = -0.01, vy = -0.16, vz = -0.01;

  for (let i = 0; i < steps; i++) {
    const dist = Math.sqrt(x*x + y*y + z*z);
    const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);
    states.push({ x: +x.toFixed(3), y: +y.toFixed(3), z: +z.toFixed(3), vx: +vx.toFixed(4), vy: +vy.toFixed(4), vz: +vz.toFixed(4) });
    distances.push(+dist.toFixed(3));
    speeds.push(+speed.toFixed(4));

    // Simple braking & centering
    const ux = -x * 0.4 - vx * 1.2;
    const uy = (dist > 1.0) ? -0.15 : -vy * 1.5;
    const uz = -z * 0.4 - vz * 1.2;
    actions.push({ ux: +ux.toFixed(2), uy: +uy.toFixed(2), uz: +uz.toFixed(2) });

    vx += ux * 0.02;
    vy += uy * 0.02;
    vz += uz * 0.02;
    x += vx; y += vy; z += vz;
    if (dist < 0.25) break;
  }

  trajectoryData = {
    summary: { num_episodes: 1, success_rate: 100.0, mean_delta_v: 0.82 },
    episodes: [{
      episode: 1,
      success: true,
      collision: false,
      initial_distance: 30.0,
      final_distance: 0.22,
      final_speed: 0.04,
      delta_v: 0.82,
      propellant_remaining: 48.2,
      states, actions, distances, speeds
    }]
  };
  populateEpisodeDropdown();
  loadEpisode(0);
}

function populateEpisodeDropdown() {
  if (!trajectoryData || !trajectoryData.episodes) return;
  const select = document.getElementById("episode-select");
  select.innerHTML = "";

  trajectoryData.episodes.forEach((ep, idx) => {
    const opt = document.createElement("option");
    opt.value = idx;
    const statusStr = ep.success ? "✓ SOFT DOCK" : (ep.collision ? "✗ CRASH" : "⏱ TIMEOUT");
    opt.textContent = `Ep #${ep.episode} (${ep.initial_distance}m) - ${statusStr} [Δv ${ep.delta_v}m/s]`;
    select.appendChild(opt);
  });
}

function loadEpisode(index) {
  if (!trajectoryData || !trajectoryData.episodes[index]) return;
  currentEpisodeIndex = index;
  currentStep = 0;
  const ep = trajectoryData.episodes[index];

  // Update scrubber
  const scrubber = document.getElementById("time-scrubber");
  scrubber.max = ep.states.length - 1;
  scrubber.value = 0;

  // Build full trajectory ribbon for this episode
  const positions = trajectoryGeometry.attributes.position.array;
  for (let i = 0; i < ep.states.length; i++) {
    const s = ep.states[i];
    const pt = lvlhToThree(s.x, s.y, s.z);
    positions[i * 3] = pt.x;
    positions[i * 3 + 1] = pt.y;
    positions[i * 3 + 2] = pt.z;
  }
  trajectoryGeometry.attributes.position.needsUpdate = true;
  trajectoryGeometry.setDrawRange(0, ep.states.length);

  updateFrame(0);
}

function updateFrame(stepIdx) {
  if (!trajectoryData || isManualMode) return;
  const ep = trajectoryData.episodes[currentEpisodeIndex];
  if (!ep || !ep.states[stepIdx]) return;

  currentStep = stepIdx;
  const s = ep.states[stepIdx];
  const a = ep.actions[stepIdx] || { ux: 0, uy: 0, uz: 0 };
  const dist = ep.distances[stepIdx];
  const speed = ep.speeds[stepIdx];
  const fuel = ep.fuel_levels ? ep.fuel_levels[stepIdx] : 50.0;
  const inCorridor = ep.corridor_flags ? ep.corridor_flags[stepIdx] : true;

  // Position Chaser Spacecraft
  const pos = lvlhToThree(s.x, s.y, s.z);
  chaserSatellite.position.copy(pos);

  // Orient chaser towards target port
  chaserSatellite.lookAt(targetSatellite.position);
  chaserSatellite.rotateX(Math.PI / 2); // align forward docking probe

  // Update RCS Thruster Plumes based on action [ux, uy, uz]
  updateThrusterPlumes(a.ux, a.uy, a.uz);

  // Update Telemetry HUD
  updateHUD(dist, speed, s, fuel, inCorridor, a);

  // Update Scrubber
  document.getElementById("time-scrubber").value = stepIdx;
  document.getElementById("step-counter").textContent = `STEP: ${stepIdx}/${ep.states.length - 1}`;
}

function updateThrusterPlumes(ux, uy, uz) {
  // Reset all
  for (let k in thrusterPlumes) {
    thrusterPlumes[k].material.opacity = 0.0;
  }

  const threshold = 0.15;
  if (ux > threshold) thrusterPlumes["x_pos"].material.opacity = Math.min(1.0, ux);
  if (ux < -threshold) thrusterPlumes["x_neg"].material.opacity = Math.min(1.0, -ux);
  if (uy > threshold) thrusterPlumes["y_pos"].material.opacity = Math.min(1.0, uy);
  if (uy < -threshold) thrusterPlumes["y_neg"].material.opacity = Math.min(1.0, -uy);
  if (uz > threshold) thrusterPlumes["z_pos"].material.opacity = Math.min(1.0, uz);
  if (uz < -threshold) thrusterPlumes["z_neg"].material.opacity = Math.min(1.0, -uz);

  // Thruster HUD bars
  document.getElementById("thrust-x-bar").style.width = `${Math.abs(ux) * 100}%`;
  document.getElementById("thrust-y-bar").style.width = `${Math.abs(uy) * 100}%`;
  document.getElementById("thrust-z-bar").style.width = `${Math.abs(uz) * 100}%`;
}

function updateHUD(dist, speed, s, fuel, inCorridor, a) {
  const rangeEl = document.getElementById("val-range");
  const speedEl = document.getElementById("val-speed");
  const latSpeedEl = document.getElementById("val-lat-speed");
  const corridorEl = document.getElementById("val-corridor");
  const fuelEl = document.getElementById("val-fuel");
  const fuelBar = document.getElementById("fuel-bar");
  const phaseTitle = document.getElementById("phase-title");
  const phaseDesc = document.getElementById("phase-desc");
  const statusDot = document.getElementById("status-indicator");

  rangeEl.textContent = `${dist.toFixed(2)} m`;
  speedEl.textContent = `${speed.toFixed(3)} m/s`;

  const latSpeed = Math.sqrt(s.vx * s.vx + s.vz * s.vz);
  latSpeedEl.textContent = `${latSpeed.toFixed(3)} m/s`;

  if (inCorridor) {
    corridorEl.textContent = "INSIDE CONE";
    corridorEl.className = "metric-value good";
  } else {
    corridorEl.textContent = "CORRIDOR DRIFT";
    corridorEl.className = "metric-value warning";
  }

  fuelEl.textContent = `${fuel.toFixed(1)} kg`;
  fuelBar.style.width = `${(fuel / 50.0) * 100}%`;

  // Status & Mission Phase updates
  if (dist <= 0.20 && speed <= 0.06) {
    phaseTitle.textContent = "SOFT DOCK ACHIEVED";
    phaseDesc.textContent = "Docking latch engaged | Zero relative drift";
    statusDot.style.backgroundColor = "#22c55e";
    statusDot.style.boxShadow = "0 0 16px #22c55e";
  } else if (dist < 5.0) {
    phaseTitle.textContent = "FINAL CONTACT PHASE";
    phaseDesc.textContent = "Laser rangefinders aligned | Speed < 0.06 m/s";
    statusDot.style.backgroundColor = "#38bdf8";
  } else if (dist < 30.0) {
    phaseTitle.textContent = "V-BAR CORRIDOR APPROACH";
    phaseDesc.textContent = "Glideslope velocity throttling active";
    statusDot.style.backgroundColor = "#eab308";
  } else {
    phaseTitle.textContent = "ORBITAL PHASING & RENDEZVOUS";
    phaseDesc.textContent = "Countering Coriolis & tidal gravity drift";
    statusDot.style.backgroundColor = "#38bdf8";
  }
}

// Manual Flight Step (Euler / RK integration with CW equations)
function updateManualPhysics(dt) {
  if (!isManualMode) return;

  // Read inputs
  let ux = 0, uy = 0, uz = 0;
  if (keysPressed["KeyW"]) ux += 1.0;
  if (keysPressed["KeyS"]) ux -= 1.0;
  if (keysPressed["KeyR"]) uy += 1.0;
  if (keysPressed["KeyF"]) uy -= 1.0;
  if (keysPressed["KeyD"]) uz += 1.0;
  if (keysPressed["KeyA"]) uz -= 1.0;
  if (keysPressed["Space"]) {
    // Kill drift burn
    ux = -Math.sign(manualState[3]) * 1.0;
    uy = -Math.sign(manualState[4]) * 1.0;
    uz = -Math.sign(manualState[5]) * 1.0;
  }

  const maxThrust = 10.0; // N
  const mass = 500.0;     // kg
  const ax_thrust = (ux * maxThrust) / mass;
  const ay_thrust = (uy * maxThrust) / mass;
  const az_thrust = (uz * maxThrust) / mass;

  // CW equations
  let [x, y, z, vx, vy, vz] = manualState;
  const w = OMEGA;
  const w2 = w * w;

  const ax = 3.0 * w2 * x + 2.0 * w * vy + ax_thrust;
  const ay = -2.0 * w * vx + ay_thrust;
  const az = -w2 * z + az_thrust;

  vx += ax * dt;
  vy += ay * dt;
  vz += az * dt;
  x += vx * dt;
  y += vy * dt;
  z += vz * dt;

  manualState = [x, y, z, vx, vy, vz];

  // Fuel consumption
  const thrustMag = Math.sqrt(ux*ux + uy*uy + uz*uz) * maxThrust;
  const dm = (thrustMag / (220.0 * 9.80665)) * dt;
  manualFuel = Math.max(0, manualFuel - dm);
  manualDeltaV += (thrustMag / mass) * dt;

  // Position Chaser
  const pos = lvlhToThree(x, y, z);
  chaserSatellite.position.copy(pos);
  chaserSatellite.lookAt(targetSatellite.position);
  chaserSatellite.rotateX(Math.PI / 2);

  updateThrusterPlumes(ux, uy, uz);

  const dist = Math.sqrt(x*x + y*y + z*z);
  const speed = Math.sqrt(vx*vx + vy*vy + vz*vz);
  const inCorridor = (y > 0) && (Math.sqrt(x*x + z*z) <= y * Math.tan(25 * Math.PI / 180) + 0.2);

  updateHUD(dist, speed, { vx, vy, vz }, manualFuel, inCorridor, { ux, uy, uz });
  document.getElementById("val-deltav").textContent = `${manualDeltaV.toFixed(2)} m/s`;
}

// Camera Modes
function updateCamera() {
  if (cameraMode === "chase") {
    // Position camera behind and slightly above chaser
    const chaserPos = chaserSatellite.position;
    camera.position.set(chaserPos.x + 8, chaserPos.y + 12, chaserPos.z + 18);
    controls.target.copy(targetSatellite.position);
  } else if (cameraMode === "dock") {
    // First-person view from docking ring
    camera.position.set(0, 1.8, 0);
    camera.lookAt(chaserSatellite.position);
    controls.target.copy(chaserSatellite.position);
  }
}

// Animation Loop
function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const dt = Math.min((now - lastFrameTime) / 1000, 0.1);
  lastFrameTime = now;

  // Rotate Earth slowly
  if (earthMesh) {
    earthMesh.rotation.y += 0.0003;
  }

  // Playback or Manual update
  if (isManualMode) {
    updateManualPhysics(dt);
  } else if (isPlaying && trajectoryData) {
    const ep = trajectoryData.episodes[currentEpisodeIndex];
    if (ep) {
      stepAccumulator += dt * playbackSpeed;
      if (stepAccumulator >= 1.0) {
        stepAccumulator = 0;
        if (currentStep < ep.states.length - 1) {
          updateFrame(currentStep + 1);
        } else {
          // Loop episode after short pause
          setTimeout(() => { if (isPlaying) updateFrame(0); }, 1500);
        }
      }
    }
  }

  updateCamera();
  controls.update();
  renderer.render(scene, camera);
}

// UI Event Handlers
function initUI() {
  const btnPlay = document.getElementById("btn-play");
  const btnRestart = document.getElementById("btn-restart");
  const timeScrubber = document.getElementById("time-scrubber");
  const speedSelect = document.getElementById("speed-select");
  const epSelect = document.getElementById("episode-select");
  const btnMode = document.getElementById("btn-mode");
  const modeText = document.getElementById("mode-text");
  const manualHint = document.getElementById("manual-hint");

  btnPlay.addEventListener("click", () => {
    isPlaying = !isPlaying;
    document.getElementById("play-text").textContent = isPlaying ? "Pause" : "Play";
  });

  btnRestart.addEventListener("click", () => {
    if (isManualMode) {
      manualState = [0, 50, 0, 0, -0.2, 0];
      manualFuel = 50.0;
      manualDeltaV = 0.0;
    } else {
      updateFrame(0);
    }
  });

  timeScrubber.addEventListener("input", (e) => {
    if (!isManualMode) {
      isPlaying = false;
      document.getElementById("play-text").textContent = "Play";
      updateFrame(parseInt(e.target.value, 10));
    }
  });

  speedSelect.addEventListener("change", (e) => {
    playbackSpeed = parseFloat(e.target.value);
  });

  epSelect.addEventListener("change", (e) => {
    loadEpisode(parseInt(e.target.value, 10));
  });

  // Toggle Pilot Mode
  btnMode.addEventListener("click", () => {
    isManualMode = !isManualMode;
    if (isManualMode) {
      modeText.textContent = "Manual Pilot";
      manualHint.classList.add("show");
      manualState = [0, 45, 0, 0, -0.15, 0];
      manualFuel = 50.0;
      manualDeltaV = 0.0;
    } else {
      modeText.textContent = "AI Autopilot";
      manualHint.classList.remove("show");
      loadEpisode(currentEpisodeIndex);
    }
  });

  // Camera buttons
  const camChase = document.getElementById("cam-chase");
  const camDock = document.getElementById("cam-dock");
  const camOrbit = document.getElementById("cam-orbit");

  function setActiveCam(btn, mode) {
    [camChase, camDock, camOrbit].forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    cameraMode = mode;
  }

  camChase.addEventListener("click", () => setActiveCam(camChase, "chase"));
  camDock.addEventListener("click", () => setActiveCam(camDock, "dock"));
  camOrbit.addEventListener("click", () => setActiveCam(camOrbit, "orbit"));

  // Keyboard controls for manual pilot mode
  window.addEventListener("keydown", (e) => {
    keysPressed[e.code] = true;
  });
  window.addEventListener("keyup", (e) => {
    keysPressed[e.code] = false;
  });
}
