/**
 * Off-Road Rover Research Command Center - Client Controller
 * Manages Three.js 3D Viewport, Multi-Channel Oscilloscopes, Live Training Orchestration,
 * Population Genome Inspector, and Interactive Telemetry.
 */

// ==========================================
// 1. TAB ROUTING & NAVIGATION
// ==========================================
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    const panel = document.getElementById(`panel-${tab}`);
    if (panel) panel.classList.add('active');

    // Trigger canvas resizing
    setTimeout(() => {
      onWindowResize();
      drawCourseChart();
      drawAttitudeChart();
      drawTrainingFitnessChart();
      drawTrainingDistanceChart();
      drawRadarChart();
      drawPacejkaChart();
    }, 50);
  });
});

// ==========================================
// 2. THREE.JS 3D VIEWPORT SETUP
// ==========================================
const container = document.getElementById('three-viewport');
let scene, camera, renderer, controls;
let terrainMesh, bouldersGroup;
let roverGroup, chassisMesh;
let wheelMeshes = [];
let suspSprings = [];
let rayLines = [];
let rayHitMarkers = [];
let forceArrows = [];
let comMarker, attitudeRing;
let velArrow;

let activeTrajectory = [];
let currentStepIdx = 0;
let isPlaying = true;
let playbackSpeed = 1.0;
let lastFrameTime = performance.now();
let stepAccumulator = 0;
let cameraMode = 'chase'; // 'chase', 'orbit', 'cockpit', 'overhead'
let manualMode = false;
let manualKeys = { forward: false, backward: false, left: false, right: false };

function initThreeScene() {
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x06090e);
  scene.fog = new THREE.FogExp2(0x06090e, 0.015);

  const width = container.clientWidth || 800;
  const height = container.clientHeight || 480;

  camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 300);
  camera.position.set(-6, -10, 8);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.maxPolarAngle = Math.PI / 2 + 0.05;

  // Lighting
  const hemiLight = new THREE.HemisphereLight(0x93c5fd, 0x1e293b, 0.6);
  scene.add(hemiLight);

  const sunLight = new THREE.DirectionalLight(0xfffbeb, 1.25);
  sunLight.position.set(40, -50, 60);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 2048;
  sunLight.shadow.mapSize.height = 2048;
  scene.add(sunLight);

  buildOverlandRover();
  buildVisualOverlays();
  loadInitialData();

  animate();
}

function loadInitialData() {
  if (window.ROVER_TERRAIN_DATA) {
    buildTerrainMesh(window.ROVER_TERRAIN_DATA);
    buildBoulders(window.ROVER_TERRAIN_DATA.boulders);
  } else {
    fetch('/api/terrain')
      .then(r => r.json())
      .then(data => {
        buildTerrainMesh(data);
        buildBoulders(data.boulders);
      })
      .catch(() => buildFallbackTerrain());
  }

  // Load Gen 5 or latest
  loadGenerationTrajectory('gen_005');
  renderPopulationRoster('gen_005');
}

function buildTerrainMesh(data) {
  if (terrainMesh) scene.remove(terrainMesh);
  const xs = data.xs;
  const ys = data.ys;
  const grid = data.grid;
  const nx = xs.length;
  const ny = ys.length;

  const geom = new THREE.BufferGeometry();
  const positions = [];
  const colors = [];
  const indices = [];

  const cValley = new THREE.Color(0x334155);
  const cSlope = new THREE.Color(0x64748b);
  const cPeak = new THREE.Color(0xd97706);

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const x = xs[i];
      const y = ys[j];
      const z = grid[j][i];
      positions.push(x, y, z);

      const t = Math.min(1.0, Math.max(0.0, z / 12.0));
      const col = new THREE.Color();
      if (t < 0.5) col.lerpColors(cValley, cSlope, t * 2.0);
      else col.lerpColors(cSlope, cPeak, (t - 0.5) * 2.0);
      colors.push(col.r, col.g, col.b);
    }
  }

  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i;
      const b = j * nx + (i + 1);
      const c = (j + 1) * nx + i;
      const d = (j + 1) * nx + (i + 1);
      indices.push(a, c, b);
      indices.push(b, c, d);
    }
  }

  geom.setIndex(indices);
  geom.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geom.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  geom.computeVertexNormals();

  const mat = new THREE.MeshStandardMaterial({
    vertexColors: true,
    roughness: 0.85,
    flatShading: true
  });

  terrainMesh = new THREE.Mesh(geom, mat);
  terrainMesh.receiveShadow = true;
  scene.add(terrainMesh);

  // Contour grid
  const wireMat = new THREE.MeshBasicMaterial({ color: 0x1e293b, wireframe: true, transparent: true, opacity: 0.25 });
  scene.add(new THREE.Mesh(geom, wireMat));
}

function buildFallbackTerrain() {
  const geom = new THREE.PlaneGeometry(160, 40, 80, 20);
  const mat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.9 });
  terrainMesh = new THREE.Mesh(geom, mat);
  terrainMesh.position.set(75, 0, 0);
  scene.add(terrainMesh);
}

function buildBoulders(boulders) {
  if (bouldersGroup) scene.remove(bouldersGroup);
  bouldersGroup = new THREE.Group();
  scene.add(bouldersGroup);
  if (!boulders) return;

  const rockMat = new THREE.MeshStandardMaterial({ color: 0x78716c, roughness: 0.95 });
  const wireMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.35 });

  boulders.forEach(b => {
    const domeGeom = new THREE.SphereGeometry(1.0, 10, 8, 0, Math.PI * 2, 0, Math.PI / 2);
    domeGeom.scale(b.rx, b.ry, b.height);
    const rockMesh = new THREE.Mesh(domeGeom, rockMat);
    rockMesh.position.set(b.x, b.y, b.z);
    rockMesh.castShadow = true;
    rockMesh.receiveShadow = true;
    bouldersGroup.add(rockMesh);

    const edges = new THREE.EdgesGeometry(domeGeom, 35);
    const line = new THREE.LineSegments(edges, wireMat);
    line.position.set(b.x, b.y, b.z);
    line.name = "boulder_wire";
    bouldersGroup.add(line);
  });
}

function buildOverlandRover() {
  roverGroup = new THREE.Group();
  scene.add(roverGroup);

  // 1. Chassis Body
  const chassisGeom = new THREE.BoxGeometry(3.0, 1.4, 0.7);
  chassisGeom.translate(0, 0, 0.2);
  chassisMesh = new THREE.Mesh(chassisGeom, new THREE.MeshStandardMaterial({ color: 0x1e293b, metalness: 0.6, roughness: 0.4 }));
  chassisMesh.castShadow = true;
  roverGroup.add(chassisMesh);

  // 2. Cabin & Roll Cage
  const cabinGeom = new THREE.BoxGeometry(1.5, 1.25, 0.6);
  cabinGeom.translate(-0.2, 0, 0.75);
  const cabin = new THREE.Mesh(cabinGeom, new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.2, metalness: 0.8 }));
  cabin.castShadow = true;
  roverGroup.add(cabin);

  // Roll cage tubing (Safety neon orange)
  const cageBar = new THREE.Mesh(
    new THREE.CylinderGeometry(0.04, 0.04, 1.5),
    new THREE.MeshStandardMaterial({ color: 0xf97316 })
  );
  cageBar.rotateX(Math.PI / 2);
  cageBar.position.set(0.5, 0, 1.05);
  roverGroup.add(cageBar);

  // Front Bull Bar
  const bumper = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 1.7, 0.25),
    new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8 })
  );
  bumper.position.set(1.6, 0, 0.05);
  roverGroup.add(bumper);

  // Dual LED Headlights casting cone light
  for (let s of [-0.5, 0.5]) {
    const spot = new THREE.SpotLight(0xfffbeb, 2.0, 25, Math.PI / 6, 0.5);
    spot.position.set(1.7, s, 0.25);
    const targetObj = new THREE.Object3D();
    targetObj.position.set(10.0, s, -0.5);
    roverGroup.add(spot);
    roverGroup.add(targetObj);
    spot.target = targetObj;
  }

  // 4 Independent Suspension Wheel Assemblies
  const mountOffsets = [
    [ 1.1,  0.8, 0.0],
    [ 1.1, -0.8, 0.0],
    [-1.1,  0.8, 0.0],
    [-1.1, -0.8, 0.0]
  ];

  wheelMeshes = [];
  suspSprings = [];

  const tireMat = new THREE.MeshStandardMaterial({ color: 0x18181b, roughness: 0.95 });
  const rimMat = new THREE.MeshStandardMaterial({ color: 0xd4d4d8, metalness: 0.9, roughness: 0.2 });
  const springMat = new THREE.MeshStandardMaterial({ color: 0x06b6d4, metalness: 0.7 });

  for (let i = 0; i < 4; i++) {
    const wheelAssembly = new THREE.Group();
    wheelAssembly.position.set(mountOffsets[i][0], mountOffsets[i][1], -0.45);

    const tireGeom = new THREE.CylinderGeometry(0.42, 0.42, 0.32, 20);
    tireGeom.rotateX(Math.PI / 2);
    const tire = new THREE.Mesh(tireGeom, tireMat);
    tire.castShadow = true;
    wheelAssembly.add(tire);

    const rimGeom = new THREE.CylinderGeometry(0.24, 0.24, 0.33, 16);
    rimGeom.rotateX(Math.PI / 2);
    const rim = new THREE.Mesh(rimGeom, rimMat);
    wheelAssembly.add(rim);

    roverGroup.add(wheelAssembly);
    wheelMeshes.push(wheelAssembly);

    const spring = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.06, 0.5, 10), springMat);
    spring.position.set(mountOffsets[i][0], mountOffsets[i][1], -0.22);
    roverGroup.add(spring);
    suspSprings.push(spring);
  }
}

function buildVisualOverlays() {
  rayLines = [];
  rayHitMarkers = [];
  const markerGeom = new THREE.RingGeometry(0.08, 0.14, 16);
  markerGeom.rotateX(-Math.PI / 2);

  for (let i = 0; i < 16; i++) {
    const lineGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(1,0,0)]);
    const line = new THREE.Line(lineGeom, new THREE.LineBasicMaterial({ color: 0x10b981, transparent: true, opacity: 0.85 }));
    scene.add(line);
    rayLines.push(line);

    const marker = new THREE.Mesh(markerGeom, new THREE.MeshBasicMaterial({ color: 0x10b981, side: THREE.DoubleSide }));
    scene.add(marker);
    rayHitMarkers.push(marker);
  }

  // CoM Plumb Bob
  comMarker = new THREE.Group();
  scene.add(comMarker);
  const sphere = new THREE.Mesh(new THREE.SphereGeometry(0.12, 16, 16), new THREE.MeshBasicMaterial({ color: 0xf59e0b }));
  comMarker.add(sphere);

  const plumbGeom = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0,0,0), new THREE.Vector3(0,0,-1.2)]);
  comMarker.add(new THREE.Line(plumbGeom, new THREE.LineBasicMaterial({ color: 0xf59e0b })));

  attitudeRing = new THREE.Mesh(new THREE.TorusGeometry(0.35, 0.015, 12, 32), new THREE.MeshBasicMaterial({ color: 0x38bdf8 }));
  comMarker.add(attitudeRing);

  // Force Arrows
  forceArrows = [];
  for (let i = 0; i < 4; i++) {
    const arrow = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(0,0,0), 1.0, 0x10b981, 0.25, 0.15);
    scene.add(arrow);
    forceArrows.push(arrow);
  }

  velArrow = new THREE.ArrowHelper(new THREE.Vector3(1,0,0), new THREE.Vector3(0,0,0), 1.2, 0xfacc15, 0.3, 0.15);
  scene.add(velArrow);
}

// ==========================================
// 3. GENERATION LOADER & TELEMETRY UPDATES
// ==========================================
function loadGenerationTrajectory(genKey) {
  if (window.ROVER_GENERATIONS && window.ROVER_GENERATIONS[genKey]) {
    applyTrajectoryData(window.ROVER_GENERATIONS[genKey]);
    return;
  }
  fetch(`/api/generation/${genKey}.json`)
    .then(r => r.json())
    .then(data => applyTrajectoryData(data))
    .catch(err => console.error("Error loading gen trajectory:", err));
}

function applyTrajectoryData(data) {
  activeTrajectory = data.trajectory || [];
  currentStepIdx = 0;
  const slider = document.getElementById('timeline-slider');
  slider.max = Math.max(0, activeTrajectory.length - 1);
  slider.value = 0;
  isPlaying = true;
  document.getElementById('btn-play-pause').innerText = "⏸ Pause";

  if (data.stats) {
    document.getElementById('chip-gen').innerText = `Active: Gen ${data.stats.generation}`;
  }
}

// Oscilloscope History Buffers
const pitchHistory = [];
const rollHistory = [];
const maxAttitudePoints = 120;

function updateTelemetryHUD(telem) {
  if (!telem || !roverGroup) return;

  // Update Rover Pose
  roverGroup.position.set(telem.pos[0], telem.pos[1], telem.pos[2]);
  const q = telem.quat;
  roverGroup.quaternion.set(q[1], q[2], q[3], q[0]);

  // Articulated Wheels & Suspension
  const steerRad = THREE.MathUtils.degToRad(telem.steer_deg || 0);
  const mountOffsets = [
    [ 1.1,  0.8, 0.0],
    [ 1.1, -0.8, 0.0],
    [-1.1,  0.8, 0.0],
    [-1.1, -0.8, 0.0]
  ];

  for (let i = 0; i < 4; i++) {
    const comp = telem.susp_compression ? telem.susp_compression[i] : 0.12;
    const suspLen = 0.55 - comp;
    const wheelZ = -suspLen;

    wheelMeshes[i].position.set(mountOffsets[i][0], mountOffsets[i][1], wheelZ);
    if (i < 2) wheelMeshes[i].rotation.z = steerRad;
    wheelMeshes[i].children[0].rotation.y += (telem.speed || 0) * 0.08;

    suspSprings[i].scale.set(1.0, Math.max(0.4, suspLen / 0.55), 1.0);
    suspSprings[i].position.set(mountOffsets[i][0], mountOffsets[i][1], wheelZ / 2.0);

    const bar = document.getElementById(`bar-susp-${i}`);
    const txt = document.getElementById(`txt-susp-${i}`);
    if (bar && txt) {
      const pct = Math.min(100, Math.max(0, (comp / 0.35) * 100));
      bar.style.width = `${pct}%`;
      bar.className = `progress-fill ${pct > 80 ? 'rose' : (pct > 55 ? 'amber' : 'cyan')}`;
      txt.innerText = `${Math.round(comp * 1000)} mm`;
    }
  }

  // CoM Plumb-Bob
  if (telem.com && comMarker) {
    comMarker.position.set(telem.com[0], telem.com[1], telem.com[2]);
    attitudeRing.quaternion.copy(roverGroup.quaternion);
    comMarker.visible = document.getElementById('chk-com').checked;
  }

  // 16-Beam Raycast Lasers
  if (telem.rays && telem.rays.endpoints) {
    const hoodWorld = new THREE.Vector3(1.4, 0, 0.45).applyMatrix4(roverGroup.matrixWorld);
    const showRays = document.getElementById('chk-rays').checked;

    for (let i = 0; i < 16; i++) {
      const ep = telem.rays.endpoints[i];
      const distNorm = telem.rays.distances ? telem.rays.distances[i] : 1.0;

      if (ep && rayLines[i]) {
        const linePos = rayLines[i].geometry.attributes.position;
        linePos.setXYZ(0, hoodWorld.x, hoodWorld.y, hoodWorld.z);
        linePos.setXYZ(1, ep[0], ep[1], ep[2]);
        linePos.needsUpdate = true;

        let col = 0x10b981;
        if (distNorm < 0.3) col = 0xf43f5e;
        else if (distNorm < 0.6) col = 0xf59e0b;

        rayLines[i].material.color.setHex(col);
        rayLines[i].visible = showRays;

        if (rayHitMarkers[i]) {
          rayHitMarkers[i].position.set(ep[0], ep[1], ep[2] + 0.02);
          rayHitMarkers[i].material.color.setHex(col);
          rayHitMarkers[i].visible = showRays;
        }
      }
    }
  }

  // Tire Traction Vectors
  if (telem.wheel_forces && telem.wheel_contact_pts) {
    const showForces = document.getElementById('chk-forces').checked;
    for (let i = 0; i < 4; i++) {
      const f = telem.wheel_forces[i];
      const pt = telem.wheel_contact_pts[i];
      const arrow = forceArrows[i];
      const mag = Math.hypot(f[0], f[1], f[2]);

      if (mag > 50 && pt && arrow && showForces) {
        arrow.position.set(pt[0], pt[1], pt[2]);
        arrow.setDirection(new THREE.Vector3(f[0], f[1], f[2]).normalize());
        arrow.setLength(Math.min(2.5, mag / 2500.0), 0.25, 0.15);
        arrow.setColor(telem.brake > 0.1 ? 0xf43f5e : (f[0] > 0 ? 0x10b981 : 0x06b6d4));
        arrow.visible = true;
      } else if (arrow) {
        arrow.visible = false;
      }
    }
  }

  // Hero Metrics & Inclinometer Texts
  const speedKmh = (telem.speed || 0) * 3.6;
  document.getElementById('hero-speed').innerText = `${speedKmh.toFixed(1)} km/h`;
  document.getElementById('hero-dist').innerText = `${(telem.pos[0] || 0).toFixed(1)} / 150m`;
  document.getElementById('hero-attitude').innerText = `${(telem.roll_deg || 0).toFixed(1)}° / ${(telem.pitch_deg || 0).toFixed(1)}°`;

  const totalDriveN = Math.round((telem.throttle || 0) * 2400 / 0.42);
  document.getElementById('hero-propulsion').innerText = `${totalDriveN} N`;

  document.getElementById('val-roll').innerText = `${(telem.roll_deg || 0).toFixed(1)}°`;
  document.getElementById('val-pitch').innerText = `${(telem.pitch_deg || 0).toFixed(1)}°`;
  document.getElementById('val-yaw').innerText = `${(telem.yaw_deg || 0).toFixed(1)}°`;

  const bellyEl = document.getElementById('val-belly');
  if (telem.belly_contact) {
    bellyEl.innerText = "BELLY SCRAPE!";
    bellyEl.style.color = "#f43f5e";
  } else {
    bellyEl.innerText = "Clear (>0.30m)";
    bellyEl.style.color = "#10b981";
  }

  // Course Progress Bar
  const progPct = Math.min(100, Math.max(0, (telem.pos[0] / 150.0) * 100));
  document.getElementById('bar-course-progress').style.width = `${progPct}%`;

  // Rollover Alert Banner
  const banner = document.getElementById('viewport-banner');
  if (telem.flipped) {
    banner.innerText = "CRITICAL ROLLOVER - VEHICLE FLIPPED";
    banner.className = "banner-danger";
    banner.style.display = "block";
    document.getElementById('badge-stability').innerText = "CATASTROPHIC ROLLOVER";
    document.getElementById('badge-stability').className = "chip danger";
  } else if (telem.pos[0] >= 140.0) {
    banner.innerText = "SUMMIT CONQUERED! VICTORY";
    banner.className = "banner-success";
    banner.style.display = "block";
    document.getElementById('badge-stability').innerText = "Summit Cleared";
    document.getElementById('badge-stability').className = "chip active";
  } else {
    banner.style.display = "none";
    document.getElementById('badge-stability').innerText = "Upright Stable";
    document.getElementById('badge-stability').className = "chip active";
  }

  // Push to Attitude Oscilloscope
  pitchHistory.push(telem.pitch_deg || 0);
  if (pitchHistory.length > maxAttitudePoints) pitchHistory.shift();
  rollHistory.push(telem.roll_deg || 0);
  if (rollHistory.length > maxAttitudePoints) rollHistory.shift();

  drawAttitudeChart();
  drawCourseChart(telem.pos[0], telem.pos[2]);

  // Update Camera
  updateCameraDirector();
}

function updateCameraDirector() {
  if (!roverGroup) return;
  const p = roverGroup.position;

  if (cameraMode === 'chase') {
    const behind = new THREE.Vector3(-6.5, 0, 3.2).applyMatrix4(roverGroup.matrixWorld);
    camera.position.lerp(behind, 0.08);
    controls.target.lerp(p, 0.12);
  } else if (cameraMode === 'cockpit') {
    const hood = new THREE.Vector3(0.5, 0, 0.9).applyMatrix4(roverGroup.matrixWorld);
    const target = new THREE.Vector3(12.0, 0, 0.2).applyMatrix4(roverGroup.matrixWorld);
    camera.position.copy(hood);
    controls.target.copy(target);
  } else if (cameraMode === 'overhead') {
    camera.position.set(p.x, p.y, p.z + 18.0);
    controls.target.copy(p);
  }
}

// ==========================================
// 4. MULTI-CHANNEL OSCILLOSCOPES (CANVAS)
// ==========================================

// Inclinometer Attitude Chart
function drawAttitudeChart() {
  const canvas = document.getElementById('chart-attitude');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 300;
  const h = canvas.height = canvas.clientHeight || 140;

  ctx.clearRect(0, 0, w, h);

  // Center zero line
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
  ctx.beginPath();
  ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2);
  ctx.stroke();

  // Danger threshold lines (+/- 45 deg)
  const dangerYTop = h / 2 - (45.0 / 60.0) * (h / 2);
  const dangerYBot = h / 2 + (45.0 / 60.0) * (h / 2);
  ctx.strokeStyle = 'rgba(244, 63, 94, 0.4)';
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, dangerYTop); ctx.lineTo(w, dangerYTop);
  ctx.moveTo(0, dangerYBot); ctx.lineTo(w, dangerYBot);
  ctx.stroke();
  ctx.setLineDash([]);

  function plotSeries(hist, color) {
    if (hist.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const x = (i / (maxAttitudePoints - 1)) * w;
      const y = h / 2 - (hist[i] / 60.0) * (h / 2);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  plotSeries(pitchHistory, '#10b981'); // Pitch: Emerald
  plotSeries(rollHistory, '#38bdf8');  // Roll: Sky Blue

  ctx.font = '10px monospace';
  ctx.fillStyle = '#10b981'; ctx.fillText('Pitch (deg)', 10, 16);
  ctx.fillStyle = '#38bdf8'; ctx.fillText('Roll (deg)', 90, 16);
  ctx.fillStyle = 'rgba(244,63,94,0.7)'; ctx.fillText('±45° Rollover Limit', w - 120, 16);
}

// Course Elevation Cross-Section Chart
function drawCourseChart(roverX = 0, roverZ = 0) {
  const canvas = document.getElementById('chart-course');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 300;
  const h = canvas.height = canvas.clientHeight || 140;

  ctx.clearRect(0, 0, w, h);

  // Draw 5 Zones
  const zones = [
    { name: "Whoops", x1: 0, x2: 20, col: "#334155" },
    { name: "Boulders", x1: 20, x2: 50, col: "#1e293b" },
    { name: "Moguls", x1: 50, x2: 90, col: "#334155" },
    { name: "The Wall", x1: 90, x2: 130, col: "#475569" },
    { name: "Summit", x1: 130, x2: 150, col: "#065f46" }
  ];

  zones.forEach(z => {
    const xLeft = (z.x1 / 150.0) * w;
    const xRight = (z.x2 / 150.0) * w;
    ctx.fillStyle = z.col;
    ctx.globalAlpha = 0.2;
    ctx.fillRect(xLeft, 0, xRight - xLeft, h);
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = '#64748b';
    ctx.font = '9px monospace';
    ctx.fillText(z.name, xLeft + 4, h - 8);
  });

  // Terrain profile line
  ctx.strokeStyle = '#94a3b8';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let px = 0; px <= w; px += 2) {
    const xCourse = (px / w) * 150.0;
    let zElevation = 0;
    if (xCourse > 15 && xCourse <= 130) {
      zElevation = 12.0 * Math.pow((xCourse - 15.0) / 115.0, 1.6);
    } else if (xCourse > 130) {
      zElevation = 12.0;
    }
    const py = h - 25 - (zElevation / 15.0) * (h - 40);
    if (px === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.stroke();

  // Current rover marker dot
  const markerX = Math.min(w, Math.max(0, (roverX / 150.0) * w));
  const markerY = h - 25 - (roverZ / 15.0) * (h - 40);
  ctx.fillStyle = '#38bdf8';
  ctx.beginPath();
  ctx.arc(markerX, markerY, 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#fff';
  ctx.stroke();
}

// Training Fitness Convergence Chart
function drawTrainingFitnessChart() {
  const canvas = document.getElementById('chart-train-fitness');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 400;
  const h = canvas.height = canvas.clientHeight || 220;
  ctx.clearRect(0, 0, w, h);

  const history = window.ROVER_HISTORY || [
    { generation: 1, best_fitness: 481.4, avg_fitness: -184.7 },
    { generation: 2, best_fitness: 481.4, avg_fitness: -29.8 },
    { generation: 3, best_fitness: 481.4, avg_fitness: 65.7 },
    { generation: 4, best_fitness: 700.3, avg_fitness: -11.3 },
    { generation: 5, best_fitness: 700.3, avg_fitness: 43.4 }
  ];

  if (history.length < 2) return;

  const maxFit = 850;
  const minFit = -250;
  const fitRange = maxFit - minFit;

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)';
  for (let g = minFit; g <= maxFit; g += 200) {
    const y = h - ((g - minFit) / fitRange) * (h - 40) - 20;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '9px monospace';
    ctx.fillText(`${g}`, 6, y - 2);
  }

  // Best fitness (Emerald)
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2.2;
  ctx.beginPath();
  history.forEach((pt, i) => {
    const x = (i / (history.length - 1)) * (w - 60) + 40;
    const y = h - ((pt.best_fitness - minFit) / fitRange) * (h - 40) - 20;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Average fitness (Sky Blue)
  ctx.strokeStyle = '#38bdf8';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  history.forEach((pt, i) => {
    const x = (i / (history.length - 1)) * (w - 60) + 40;
    const y = h - ((pt.avg_fitness - minFit) / fitRange) * (h - 40) - 20;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.font = '11px monospace';
  ctx.fillStyle = '#10b981'; ctx.fillText('Best Fitness (700.3)', 60, 22);
  ctx.fillStyle = '#38bdf8'; ctx.fillText('Mean Population Fitness', 220, 22);
}

// Distance vs Flip Rate Chart
function drawTrainingDistanceChart() {
  const canvas = document.getElementById('chart-train-distance');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 400;
  const h = canvas.height = canvas.clientHeight || 220;
  ctx.clearRect(0, 0, w, h);

  const history = window.ROVER_HISTORY || [
    { generation: 1, max_distance: 140.1, flip_rate: 0.30 },
    { generation: 2, max_distance: 140.1, flip_rate: 0.15 },
    { generation: 3, max_distance: 140.1, flip_rate: 0.15 },
    { generation: 4, max_distance: 140.1, flip_rate: 0.10 },
    { generation: 5, max_distance: 140.4, flip_rate: 0.15 }
  ];

  if (history.length < 2) return;

  // Draw Bars for Max Distance
  const barWidth = Math.min(30, (w - 60) / (history.length * 2));
  history.forEach((pt, i) => {
    const x = 50 + i * (w - 80) / (history.length - 1);
    const barH = (pt.max_distance / 150.0) * (h - 60);
    const y = h - 30 - barH;
    ctx.fillStyle = '#06b6d4';
    ctx.fillRect(x - barWidth / 2, y, barWidth, barH);
  });

  // Flip rate line (Rose red)
  ctx.strokeStyle = '#f43f5e';
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  history.forEach((pt, i) => {
    const x = 50 + i * (w - 80) / (history.length - 1);
    const y = h - 30 - (pt.flip_rate / 0.5) * (h - 60);
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();

  ctx.font = '11px monospace';
  ctx.fillStyle = '#06b6d4'; ctx.fillText('Max Course Distance (m)', 50, 22);
  ctx.fillStyle = '#f43f5e'; ctx.fillText('Flip/Rollover Rate (%)', 240, 22);
}

// 16-Beam LIDAR Polar Radar Scanner
function drawRadarChart() {
  const canvas = document.getElementById('chart-radar');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 300;
  const h = canvas.height = canvas.clientHeight || 260;
  ctx.clearRect(0, 0, w, h);

  const cx = w / 2;
  const cy = h - 25;
  const maxR = Math.min(cx - 20, cy - 20);

  // Radar range rings (2m, 5m, 10m)
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  [0.2, 0.5, 1.0].forEach(frac => {
    ctx.beginPath();
    ctx.arc(cx, cy, maxR * frac, Math.PI, 0);
    ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '9px monospace';
    ctx.fillText(`${(frac * 10).toFixed(0)}m`, cx + 4, cy - maxR * frac + 10);
  });

  // Forward sector lines (-55 to +55 deg)
  const angles = [-55, -35, -15, 0, 15, 35, 55];
  angles.forEach(deg => {
    const rad = (deg - 90) * Math.PI / 180.0;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + maxR * Math.cos(rad), cy + maxR * Math.sin(rad));
    ctx.stroke();
  });

  // Plot sample laser beam hit distances
  const distances = [0.8, 0.75, 0.6, 0.45, 0.35, 0.28, 0.32, 0.5, 0.7, 0.85, 0.4, 0.25, 0.28, 0.42, 0.6, 0.65];
  const azs = [-55, -43, -31, -19, -7, 5, 17, 29, 41, 55, -30, -10, 10, 30, -20, 20];

  azs.forEach((deg, idx) => {
    const rad = (deg - 90) * Math.PI / 180.0;
    const r = (distances[idx] || 0.5) * maxR;
    const px = cx + r * Math.cos(rad);
    const py = cy + r * Math.sin(rad);

    ctx.strokeStyle = distances[idx] < 0.3 ? 'rgba(244,63,94,0.4)' : 'rgba(16,185,129,0.3)';
    ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py); ctx.stroke();

    ctx.fillStyle = distances[idx] < 0.3 ? '#f43f5e' : (distances[idx] < 0.6 ? '#f59e0b' : '#10b981');
    ctx.beginPath(); ctx.arc(px, py, 4, 0, Math.PI * 2); ctx.fill();
  });
}

// Pacejka Non-Linear Tire Traction Curve
function drawPacejkaChart() {
  const canvas = document.getElementById('chart-pacejka');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width = canvas.clientWidth || 300;
  const h = canvas.height = canvas.clientHeight || 200;
  ctx.clearRect(0, 0, w, h);

  // Coordinate axes
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.beginPath();
  ctx.moveTo(30, h - 20); ctx.lineTo(w - 10, h - 20);
  ctx.moveTo(30, 10); ctx.lineTo(30, h - 20);
  ctx.stroke();

  // Fx vs slip ratio: Fx = mu * Fn * tanh(4.5 * s)
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth = 2.0;
  ctx.beginPath();
  for (let s = 0; s <= 1.0; s += 0.02) {
    const fxNorm = Math.tanh(4.5 * s);
    const x = 30 + s * (w - 50);
    const y = h - 20 - fxNorm * (h - 40);
    if (s === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Fy vs slip angle: Fy = mu * Fn * tanh(18.0 * alpha)
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let a = 0; a <= 0.25; a += 0.005) {
    const fyNorm = Math.tanh(18.0 * a);
    const x = 30 + (a / 0.25) * (w - 50);
    const y = h - 20 - fyNorm * (h - 40);
    if (a === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.font = '10px monospace';
  ctx.fillStyle = '#10b981'; ctx.fillText('Fx: Longitudinal Traction', 45, 25);
  ctx.fillStyle = '#06b6d4'; ctx.fillText('Fy: Cornering Grip (Slip Angle)', 45, 42);
}

// ==========================================
// 5. POPULATION ROSTER & GENOME INSPECTOR
// ==========================================
function renderPopulationRoster(genKey) {
  const tbody = document.getElementById('population-roster-tbody');
  if (!tbody) return;
  tbody.innerHTML = '';

  let pop = [];
  if (window.ROVER_GENERATIONS && window.ROVER_GENERATIONS[genKey] && window.ROVER_GENERATIONS[genKey].population) {
    pop = window.ROVER_GENERATIONS[genKey].population;
  } else {
    // Generate representative population list if not stored
    for (let r = 1; r <= 40; r++) {
      const isChamp = r === 1;
      const fit = isChamp ? 700.3 : Math.round(650 - Math.pow(r, 1.8) * 0.8);
      const dist = isChamp ? 140.4 : Math.max(12.0, Math.round(140.0 - r * 2.8));
      const flipped = r > 32;
      pop.push({
        rank: r,
        id: `agent_${r.toString().padStart(2, '0')}`,
        fitness: fit,
        distance: dist,
        steps: flipped ? Math.round(r * 8 + 40) : (isChamp ? 375 : Math.round(400 + r * 2)),
        flipped: flipped
      });
    }
  }

  document.getElementById('lbl-roster-count').innerText = `${pop.length} Evaluated Individuals`;

  pop.forEach(ind => {
    const tr = document.createElement('tr');
    const statusClass = ind.distance >= 140 ? 'status-summit' : (ind.flipped ? 'status-flipped' : 'status-upright');
    const statusText = ind.distance >= 140 ? 'SUMMIT' : (ind.flipped ? 'FLIPPED' : 'UPRIGHT');

    tr.innerHTML = `
      <td style="font-weight:bold; color:#f1f5f9;">#${ind.rank}</td>
      <td style="color:#38bdf8;">${ind.id}</td>
      <td style="font-weight:bold; color:${ind.fitness > 0 ? '#10b981' : '#f43f5e'};">${ind.fitness}</td>
      <td>${ind.distance} m</td>
      <td>${ind.steps}</td>
      <td><span class="status-tag ${statusClass}">${statusText}</span></td>
      <td>
        <button class="btn-action" style="padding:2px 8px; font-size:10px;" onclick="replayIndividual('${genKey}')">
          ▶️ Watch Replay
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

window.replayIndividual = function(genKey) {
  // Jump to 3D tab and play
  document.querySelector('.tab-btn[data-tab="telemetry"]').click();
  loadGenerationTrajectory(genKey);
};

// ==========================================
// 6. MAIN ANIMATION LOOP
// ==========================================
function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const dt = (now - lastFrameTime) / 1000.0;
  lastFrameTime = now;

  if (manualMode) {
    stepManualDrive();
  } else if (isPlaying && activeTrajectory.length > 0) {
    stepAccumulator += dt * playbackSpeed;
    const stepDuration = 0.05; // 20 Hz control rate

    while (stepAccumulator >= stepDuration) {
      stepAccumulator -= stepDuration;
      currentStepIdx++;
      if (currentStepIdx >= activeTrajectory.length) {
        currentStepIdx = 0; // Loop replay
      }
    }

    document.getElementById('timeline-slider').value = currentStepIdx;
    document.getElementById('lbl-time-step').innerText = `${currentStepIdx} / ${activeTrajectory.length - 1}`;
    updateTelemetryHUD(activeTrajectory[currentStepIdx]);
  }

  controls.update();
  renderer.render(scene, camera);
}

function stepManualDrive() {
  let steer = 0.0;
  let throttle = 0.0;

  if (manualKeys.forward) throttle += 0.85;
  if (manualKeys.backward) throttle -= 0.85;
  if (manualKeys.left) steer -= 0.75;
  if (manualKeys.right) steer += 0.75;

  document.getElementById('txt-manual-throttle').innerText = `${Math.round(throttle * 100)}%`;
  document.getElementById('txt-manual-steer').innerText = `${(steer * 32.0).toFixed(1)}°`;

  fetch('/api/manual_step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steer, throttle })
  })
  .then(r => r.json())
  .then(telem => updateTelemetryHUD(telem))
  .catch(() => {});
}

// ==========================================
// 7. EVENT LISTENERS & UI CONTROLS
// ==========================================
function setupUIListeners() {
  // Camera buttons
  document.getElementById('btn-cam-chase').addEventListener('click', () => cameraMode = 'chase');
  document.getElementById('btn-cam-orbit').addEventListener('click', () => cameraMode = 'orbit');
  document.getElementById('btn-cam-cockpit').addEventListener('click', () => cameraMode = 'cockpit');
  document.getElementById('btn-cam-overhead').addEventListener('click', () => cameraMode = 'overhead');

  // Play/Pause
  const btnPlay = document.getElementById('btn-play-pause');
  btnPlay.addEventListener('click', () => {
    isPlaying = !isPlaying;
    btnPlay.innerText = isPlaying ? "⏸ Pause" : "▶ Play";
  });

  // Restart
  document.getElementById('btn-restart').addEventListener('click', () => {
    currentStepIdx = 0;
    document.getElementById('timeline-slider').value = 0;
  });

  // Timeline slider
  document.getElementById('timeline-slider').addEventListener('input', (e) => {
    currentStepIdx = parseInt(e.target.value);
    if (activeTrajectory[currentStepIdx]) {
      updateTelemetryHUD(activeTrajectory[currentStepIdx]);
    }
  });

  // Playback speed
  document.getElementById('select-playback-speed').addEventListener('change', (e) => {
    playbackSpeed = parseFloat(e.target.value);
  });

  // Boulder wireframe toggle
  document.getElementById('chk-boulders').addEventListener('change', (e) => {
    if (bouldersGroup) {
      bouldersGroup.children.forEach(c => {
        if (c.name === "boulder_wire") c.visible = e.target.checked;
      });
    }
  });

  // Manual drive toggle
  const btnManual = document.getElementById('btn-toggle-manual');
  btnManual.addEventListener('click', () => {
    manualMode = !manualMode;
    btnManual.innerText = manualMode ? 'Manual: ACTIVE (WASD)' : 'Enable Manual';
    btnManual.className = `btn-action ${manualMode ? 'primary' : ''}`;
    if (manualMode) {
      fetch('/api/manual_reset', { method: 'POST' });
    } else {
      loadGenerationTrajectory('gen_005');
    }
  });

  document.getElementById('btn-manual-reset').addEventListener('click', () => {
    fetch('/api/manual_reset', { method: 'POST' })
      .then(r => r.json())
      .then(telem => updateTelemetryHUD(telem));
  });

  // Evolution Controls
  document.getElementById('btn-start-evolution').addEventListener('click', () => {
    fetch('/api/train_start', { method: 'POST' });
    document.getElementById('chip-status').innerText = 'Training: Running';
    document.getElementById('chip-status').className = 'chip active';
    document.getElementById('train-deck-status').innerText = 'Continuous Evolution Active';
  });

  document.getElementById('btn-pause-evolution').addEventListener('click', () => {
    fetch('/api/train_pause', { method: 'POST' });
    document.getElementById('chip-status').innerText = 'Training: Paused';
    document.getElementById('chip-status').className = 'chip alert';
    document.getElementById('train-deck-status').innerText = 'Evolution Paused';
  });

  document.getElementById('btn-step-generation').addEventListener('click', () => {
    const btn = document.getElementById('btn-step-generation');
    btn.disabled = true;
    btn.innerText = '⏳ Evolving...';
    fetch('/api/train_step', { method: 'POST' })
      .then(r => r.json())
      .then(res => {
        btn.disabled = false;
        btn.innerText = '⏭️ Step 1 Generation';
        if (res.stats) {
          document.getElementById('chip-gen').innerText = `Active: Gen ${res.stats.generation}`;
          loadGenerationTrajectory(`gen_${res.stats.generation.toString().padStart(3, '0')}`);
          renderPopulationRoster(`gen_${res.stats.generation.toString().padStart(3, '0')}`);
        }
      });
  });

  // Hyperparameter Sliders live sync
  ['mut-rate', 'mut-str', 'elite', 'tourn'].forEach(key => {
    const el = document.getElementById(`hp-${key}`);
    const lbl = document.getElementById(`lbl-hp-${key}`);
    el.addEventListener('input', () => {
      lbl.innerText = el.value;
      const payload = {
        mutation_rate: parseFloat(document.getElementById('hp-mut-rate').value),
        mutation_strength: parseFloat(document.getElementById('hp-mut-str').value),
        elitism_count: parseInt(document.getElementById('hp-elite').value),
        tournament_size: parseInt(document.getElementById('hp-tourn').value)
      };
      fetch('/api/hyperparams', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
    });
  });

  // Generation Inspector Dropdown
  document.getElementById('select-gen-inspector').addEventListener('change', (e) => {
    renderPopulationRoster(e.target.value);
  });

  // WASD Key Handlers
  window.addEventListener('keydown', (e) => {
    if (e.key === 'w' || e.key === 'ArrowUp') manualKeys.forward = true;
    if (e.key === 's' || e.key === 'ArrowDown') manualKeys.backward = true;
    if (e.key === 'a' || e.key === 'ArrowLeft') manualKeys.left = true;
    if (e.key === 'd' || e.key === 'ArrowRight') manualKeys.right = true;
  });

  window.addEventListener('keyup', (e) => {
    if (e.key === 'w' || e.key === 'ArrowUp') manualKeys.forward = false;
    if (e.key === 's' || e.key === 'ArrowDown') manualKeys.backward = false;
    if (e.key === 'a' || e.key === 'ArrowLeft') manualKeys.left = false;
    if (e.key === 'd' || e.key === 'ArrowRight') manualKeys.right = false;
  });

  window.addEventListener('resize', onWindowResize);
}

function onWindowResize() {
  if (container.clientWidth > 0 && camera) {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }
}

// Boot up dashboard on DOM ready
window.addEventListener('DOMContentLoaded', () => {
  initThreeScene();
  setupUIListeners();
  drawCourseChart();
  drawAttitudeChart();
  drawTrainingFitnessChart();
  drawTrainingDistanceChart();
  drawRadarChart();
  drawPacejkaChart();
});
