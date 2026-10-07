/**
 * Off-Road Rover 3D WebGL Simulation & Telemetry Visualizer
 * Powered by Three.js (r128)
 * Renders procedural terrain, articulated overland rover with 4-wheel independent
 * suspension, 16 neon raycast lasers, Center-of-Mass plumb-bob, and 3D tire traction vectors.
 */

// Scene Globals
let scene, camera, renderer, controls;
let terrainMesh, bouldersGroup;
let roverGroup, chassisMesh;
let wheelMeshes = [];
let suspSprings = [];
let rayLines = [];
let rayHitMarkers = [];
let forceArrows = [];
let comMarker, comPlumbLine, attitudeRing;
let velArrow;

// Playback & Simulation State
let activeTrajectory = [];
let currentStepIdx = 0;
let isPlaying = true;
let playbackSpeed = 1.0;
let lastFrameTime = performance.now();
let stepAccumulator = 0;
let cameraMode = 'chase'; // 'chase', 'orbit', 'cockpit', 'overhead'
let manualMode = false;
let manualKeys = { forward: false, backward: false, left: false, right: false };

// UI Elements
const elSpeed = document.getElementById('val-speed');
const elDist = document.getElementById('val-dist');
const elThrottle = document.getElementById('val-throttle');
const elSteer = document.getElementById('val-steer');
const elRoll = document.getElementById('val-roll');
const elPitch = document.getElementById('val-pitch');
const elTimeline = document.getElementById('timeline-slider');
const elTimeStep = document.getElementById('lbl-time-step');
const elProgressFill = document.getElementById('course-progress-fill');
const elPlayPause = document.getElementById('btn-play-pause');
const elAlert = document.getElementById('alert-banner');
const elGenSelect = document.getElementById('select-generation');
const elBestFit = document.getElementById('gen-best-fit');
const elFlipRate = document.getElementById('gen-flip-rate');
const elModeTag = document.getElementById('sim-mode-tag');

// Overlay Toggles
const toggleRays = document.getElementById('toggle-rays');
const toggleCom = document.getElementById('toggle-com');
const toggleForces = document.getElementById('toggle-forces');
const toggleBoulders = document.getElementById('toggle-boulders');

function initScene() {
  const container = document.getElementById('canvas-container');
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0a0e17);
  scene.fog = new THREE.FogExp2(0x0a0e17, 0.015);

  camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 300);
  camera.position.set(-6, -10, 8);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
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

  const sunLight = new THREE.DirectionalLight(0xfffbeb, 1.2);
  sunLight.position.set(40, -50, 60);
  sunLight.castShadow = true;
  sunLight.shadow.mapSize.width = 2048;
  sunLight.shadow.mapSize.height = 2048;
  sunLight.shadow.camera.near = 10;
  sunLight.shadow.camera.far = 160;
  sunLight.shadow.camera.left = -60;
  sunLight.shadow.camera.right = 60;
  sunLight.shadow.camera.top = 60;
  sunLight.shadow.camera.bottom = -60;
  scene.add(sunLight);

  buildOverlandRover();
  buildVisualOverlays();
  loadTerrainData();
  setupEventListeners();

  animate();
}

/**
 * Procedural Terrain Loader
 */
function loadTerrainData() {
  if (window.ROVER_TERRAIN_DATA) {
    buildTerrainMesh(window.ROVER_TERRAIN_DATA);
    buildBoulders(window.ROVER_TERRAIN_DATA.boulders);
    loadGeneration(elGenSelect.value);
    return;
  }
  fetch('/api/terrain')
    .then(res => res.json())
    .catch(() => fetch('terrain.json').then(r => r.json()))
    .then(data => {
      buildTerrainMesh(data);
      buildBoulders(data.boulders);
      loadGeneration(elGenSelect.value);
    })
    .catch(err => {
      console.warn("Could not fetch remote terrain. Building fallback procedural mesh.", err);
      buildFallbackTerrain();
      loadGeneration(elGenSelect.value);
    });
}

function buildTerrainMesh(data) {
  if (terrainMesh) scene.remove(terrainMesh);

  const xs = data.xs;
  const ys = data.ys;
  const grid = data.grid; // [y_idx][x_idx]
  const nx = xs.length;
  const ny = ys.length;

  const geom = new THREE.BufferGeometry();
  const positions = [];
  const colors = [];
  const indices = [];

  // Elevation color palette
  const cValley = new THREE.Color(0x334155); // Slate rock
  const cSlope = new THREE.Color(0x64748b);  // Granite gray
  const cPeak = new THREE.Color(0xd97706);   // Sandstone orange crest

  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const x = xs[i];
      const y = ys[j];
      const z = grid[j][i];
      positions.push(x, y, z);

      // Height-based tinting
      const t = Math.min(1.0, Math.max(0.0, z / 12.0));
      const col = new THREE.Color();
      if (t < 0.5) {
        col.lerpColors(cValley, cSlope, t * 2.0);
      } else {
        col.lerpColors(cSlope, cPeak, (t - 0.5) * 2.0);
      }
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
    metalness: 0.1,
    flatShading: true
  });

  terrainMesh = new THREE.Mesh(geom, mat);
  terrainMesh.receiveShadow = true;
  scene.add(terrainMesh);

  // Add terrain contour grid lines
  const wireMat = new THREE.MeshBasicMaterial({ color: 0x1e293b, wireframe: true, transparent: true, opacity: 0.25 });
  const wireMesh = new THREE.Mesh(geom, wireMat);
  scene.add(wireMesh);
}

function buildFallbackTerrain() {
  const geom = new THREE.PlaneGeometry(160, 40, 100, 30);
  const mat = new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.9, wireframe: false });
  terrainMesh = new THREE.Mesh(geom, mat);
  terrainMesh.position.set(75, 0, 0);
  scene.add(terrainMesh);
}

function buildBoulders(boulders) {
  if (bouldersGroup) scene.remove(bouldersGroup);
  bouldersGroup = new THREE.Group();
  scene.add(bouldersGroup);

  if (!boulders) return;

  const rockMat = new THREE.MeshStandardMaterial({
    color: 0x78716c,
    roughness: 0.95,
    metalness: 0.05
  });

  const wireMat = new THREE.LineBasicMaterial({ color: 0x38bdf8, transparent: true, opacity: 0.4 });

  boulders.forEach(b => {
    const domeGeom = new THREE.SphereGeometry(1.0, 12, 10, 0, Math.PI * 2, 0, Math.PI / 2);
    domeGeom.scale(b.rx, b.ry, b.height);
    const rockMesh = new THREE.Mesh(domeGeom, rockMat);
    rockMesh.position.set(b.x, b.y, b.z);
    rockMesh.castShadow = true;
    rockMesh.receiveShadow = true;
    bouldersGroup.add(rockMesh);

    // Subtle blue outline for spectator visibility
    const edges = new THREE.EdgesGeometry(domeGeom, 35);
    const line = new THREE.LineSegments(edges, wireMat);
    line.position.set(b.x, b.y, b.z);
    line.name = "boulder_wire";
    bouldersGroup.add(line);
  });
}

/**
 * Overland 4x4 Rover 3D Model
 */
function buildOverlandRover() {
  roverGroup = new THREE.Group();
  scene.add(roverGroup);

  // 1. Main Chassis Box
  const chassisGeom = new THREE.BoxGeometry(3.0, 1.4, 0.7);
  chassisGeom.translate(0, 0, 0.2);
  const chassisMat = new THREE.MeshStandardMaterial({
    color: 0x1e293b,
    metalness: 0.6,
    roughness: 0.4
  });
  chassisMesh = new THREE.Mesh(chassisGeom, chassisMat);
  chassisMesh.castShadow = true;
  roverGroup.add(chassisMesh);

  // 2. Cabin / Greenhouse
  const cabinGeom = new THREE.BoxGeometry(1.5, 1.25, 0.6);
  cabinGeom.translate(-0.2, 0, 0.75);
  const cabinMat = new THREE.MeshStandardMaterial({
    color: 0x0f172a,
    roughness: 0.2,
    metalness: 0.8
  });
  const cabin = new THREE.Mesh(cabinGeom, cabinMat);
  cabin.castShadow = true;
  roverGroup.add(cabin);

  // 3. Roll Cage Bars (Safety Neon Orange)
  const cageMat = new THREE.MeshStandardMaterial({ color: 0xf97316, roughness: 0.5 });
  const cageBarGeom = new THREE.CylinderGeometry(0.04, 0.04, 1.5);
  cageBarGeom.rotateX(Math.PI / 2);
  cageBarGeom.translate(0.5, 0, 1.05);
  const cageBar = new THREE.Mesh(cageBarGeom, cageMat);
  roverGroup.add(cageBar);

  // 4. Heavy Duty Bull Bar Front Bumper
  const bumperGeom = new THREE.BoxGeometry(0.3, 1.7, 0.25);
  bumperGeom.translate(1.6, 0, 0.05);
  const bumperMat = new THREE.MeshStandardMaterial({ color: 0x475569, metalness: 0.8 });
  const bumper = new THREE.Mesh(bumperGeom, bumperMat);
  roverGroup.add(bumper);

  // 5. Dual LED Spotlights (Cast real cone light on rocks)
  for (let s of [-0.5, 0.5]) {
    const lightMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.08, 0.08, 0.1),
      new THREE.MeshBasicMaterial({ color: 0xffffff })
    );
    lightMesh.rotateZ(Math.PI / 2);
    lightMesh.position.set(1.65, s, 0.25);
    roverGroup.add(lightMesh);

    const spot = new THREE.SpotLight(0xfffbeb, 2.5, 25, Math.PI / 6, 0.5);
    spot.position.set(1.7, s, 0.25);
    const targetObj = new THREE.Object3D();
    targetObj.position.set(10.0, s, -0.5);
    roverGroup.add(spot);
    roverGroup.add(targetObj);
    spot.target = targetObj;
  }

  // 6. 4 Independent Articulated Wheels & Suspension Struts
  // Hardpoints: FL, FR, RL, RR
  const mountOffsets = [
    [ 1.1,  0.8, 0.0], // FL
    [ 1.1, -0.8, 0.0], // FR
    [-1.1,  0.8, 0.0], // RL
    [-1.1, -0.8, 0.0]  // RR
  ];

  wheelMeshes = [];
  suspSprings = [];

  const tireMat = new THREE.MeshStandardMaterial({
    color: 0x18181b,
    roughness: 0.95
  });
  const rimMat = new THREE.MeshStandardMaterial({
    color: 0xd4d4d8,
    metalness: 0.9,
    roughness: 0.2
  });
  const springMat = new THREE.MeshStandardMaterial({
    color: 0x06b6d4, // Cyan spring
    metalness: 0.7,
    roughness: 0.3
  });

  for (let i = 0; i < 4; i++) {
    // Wheel assembly group
    const wheelAssembly = new THREE.Group();
    wheelAssembly.position.set(mountOffsets[i][0], mountOffsets[i][1], -0.45);

    // Tire mesh (Cylinder oriented along Y axis)
    const tireGeom = new THREE.CylinderGeometry(0.42, 0.42, 0.32, 20);
    tireGeom.rotateX(Math.PI / 2);
    const tire = new THREE.Mesh(tireGeom, tireMat);
    tire.castShadow = true;
    wheelAssembly.add(tire);

    // Rim mesh
    const rimGeom = new THREE.CylinderGeometry(0.24, 0.24, 0.33, 16);
    rimGeom.rotateX(Math.PI / 2);
    const rim = new THREE.Mesh(rimGeom, rimMat);
    wheelAssembly.add(rim);

    roverGroup.add(wheelAssembly);
    wheelMeshes.push(wheelAssembly);

    // Suspension Spring Cylinder (Connecting chassis mount to wheel hub)
    const springGeom = new THREE.CylinderGeometry(0.06, 0.06, 0.5, 10);
    const spring = new THREE.Mesh(springGeom, springMat);
    spring.position.set(mountOffsets[i][0], mountOffsets[i][1], -0.22);
    roverGroup.add(spring);
    suspSprings.push(spring);
  }
}

/**
 * Visual Overlays: Raycast Lasers, CoM Plumb-Bob, Force Vectors
 */
function buildVisualOverlays() {
  // 1. 16 Raycast Laser Beams
  rayLines = [];
  rayHitMarkers = [];
  const markerGeom = new THREE.RingGeometry(0.08, 0.14, 16);
  markerGeom.rotateX(-Math.PI / 2);

  for (let i = 0; i < 16; i++) {
    const lineGeom = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(1, 0, 0)
    ]);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x10b981,
      linewidth: 2,
      transparent: true,
      opacity: 0.85
    });
    const line = new THREE.Line(lineGeom, lineMat);
    scene.add(line);
    rayLines.push(line);

    const hitMat = new THREE.MeshBasicMaterial({
      color: 0x10b981,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.9
    });
    const marker = new THREE.Mesh(markerGeom, hitMat);
    scene.add(marker);
    rayHitMarkers.push(marker);
  }

  // 2. Center of Mass & Plumb-Bob
  const comGroup = new THREE.Group();
  scene.add(comGroup);
  comMarker = comGroup;

  const sphereGeom = new THREE.SphereGeometry(0.12, 16, 16);
  const sphereMat = new THREE.MeshBasicMaterial({ color: 0xf59e0b });
  const sphere = new THREE.Mesh(sphereGeom, sphereMat);
  comGroup.add(sphere);

  // Plumb-bob line pointing down to gravity
  const plumbGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0, 0, -1.2)
  ]);
  const plumbMat = new THREE.LineDashedMaterial({
    color: 0xf59e0b,
    dashSize: 0.1,
    gapSize: 0.05
  });
  comPlumbLine = new THREE.Line(plumbGeom, plumbMat);
  comPlumbLine.computeLineDistances();
  comGroup.add(comPlumbLine);

  // Attitude gimbal ring
  const ringGeom = new THREE.TorusGeometry(0.35, 0.015, 12, 32);
  const ringMat = new THREE.MeshBasicMaterial({ color: 0x38bdf8 });
  attitudeRing = new THREE.Mesh(ringGeom, ringMat);
  comGroup.add(attitudeRing);

  // 3. 4-Corner Tire Force Arrows (Drive, Brake, Lateral)
  forceArrows = [];
  for (let i = 0; i < 4; i++) {
    const dir = new THREE.Vector3(1, 0, 0);
    const origin = new THREE.Vector3(0, 0, 0);
    const arrow = new THREE.ArrowHelper(dir, origin, 1.0, 0x10b981, 0.25, 0.15);
    scene.add(arrow);
    forceArrows.push(arrow);
  }

  // 4. Velocity Vector Arrow at Front Bumper
  velArrow = new THREE.ArrowHelper(
    new THREE.Vector3(1, 0, 0),
    new THREE.Vector3(0, 0, 0),
    1.2,
    0xfacc15,
    0.3,
    0.15
  );
  scene.add(velArrow);
}

/**
 * Generation Replay Loader
 */
function loadGeneration(filename) {
  const genKey = filename.replace('.json', '');
  if (window.ROVER_GENERATIONS && window.ROVER_GENERATIONS[genKey]) {
    applyGenerationData(window.ROVER_GENERATIONS[genKey], filename);
    return;
  }
  fetch(`/api/generation/${filename}`)
    .then(r => r.json())
    .catch(() => fetch(`checkpoints/${filename}`).then(r => r.json()))
    .then(data => applyGenerationData(data, filename))
    .catch(err => console.error("Error loading checkpoint:", err));
}

function applyGenerationData(data, filename) {
  activeTrajectory = data.trajectory || [];
  if (data.stats) {
    elBestFit.innerText = data.stats.best_fitness;
    elFlipRate.innerText = Math.round(data.stats.flip_rate * 100) + '%';
  }
  currentStepIdx = 0;
  elTimeline.max = Math.max(0, activeTrajectory.length - 1);
  elTimeline.value = 0;
  isPlaying = true;
  elPlayPause.innerText = "⏸ Pause";
  elModeTag.innerText = `REPLAYING ${filename.toUpperCase().replace('.JSON', '')}`;
  elModeTag.style.color = "#38bdf8";
  manualMode = false;
}

/**
 * State Update & Visualization Loop
 */
function updateSimulationState(telem) {
  if (!telem || !roverGroup) return;

  // 1. Update Rover Pose
  roverGroup.position.set(telem.pos[0], telem.pos[1], telem.pos[2]);
  // Quaternion: [w, x, y, z] in Python -> Three.js (x, y, z, w)
  const q = telem.quat;
  roverGroup.quaternion.set(q[1], q[2], q[3], q[0]);

  // 2. Update Articulated Wheels & Suspension
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
    // Front wheel steering
    if (i < 2) {
      wheelMeshes[i].rotation.z = steerRad;
    }
    // Dynamic wheel spin
    wheelMeshes[i].children[0].rotation.y += (telem.speed || 0) * 0.08;

    // Scale spring strut visually
    suspSprings[i].scale.set(1.0, Math.max(0.4, suspLen / 0.55), 1.0);
    suspSprings[i].position.set(mountOffsets[i][0], mountOffsets[i][1], wheelZ / 2.0);

    // Update Suspension UI bars
    const bar = document.getElementById(`bar-susp-${i}`);
    const txt = document.getElementById(`txt-susp-${i}`);
    if (bar && txt) {
      const pct = Math.min(100, Math.max(0, (comp / 0.35) * 100));
      bar.style.width = `${pct}%`;
      bar.style.backgroundColor = pct > 85 ? '#ef4444' : (pct > 60 ? '#f59e0b' : '#06b6d4');
      txt.innerText = `${Math.round(comp * 1000)} mm`;
    }
  }

  // 3. Update Center of Mass & Plumb Line
  if (telem.com && comMarker) {
    comMarker.position.set(telem.com[0], telem.com[1], telem.com[2]);
    // Attitude ring tilts with vehicle roll and pitch
    attitudeRing.quaternion.copy(roverGroup.quaternion);
    comMarker.visible = toggleCom.checked;
  }

  // 4. Update 16-Beam Raycast Lasers
  if (telem.rays && telem.rays.endpoints) {
    const hoodWorld = new THREE.Vector3(1.4, 0, 0.45).applyMatrix4(roverGroup.matrixWorld);

    for (let i = 0; i < 16; i++) {
      const ep = telem.rays.endpoints[i];
      const hit = telem.rays.hits ? telem.rays.hits[i] : true;
      const distNorm = telem.rays.distances ? telem.rays.distances[i] : 1.0;

      if (ep && rayLines[i]) {
        const linePos = rayLines[i].geometry.attributes.position;
        linePos.setXYZ(0, hoodWorld.x, hoodWorld.y, hoodWorld.z);
        linePos.setXYZ(1, ep[0], ep[1], ep[2]);
        linePos.needsUpdate = true;

        // Color dynamics: Green (>6m) -> Yellow (3-6m) -> Red (<3m)
        let colorHex = 0x10b981;
        if (distNorm < 0.3) {
          colorHex = 0xef4444; // Proximity hazard!
        } else if (distNorm < 0.6) {
          colorHex = 0xf59e0b; // Approaching boulder/incline
        }

        rayLines[i].material.color.setHex(colorHex);
        rayLines[i].visible = toggleRays.checked;

        if (rayHitMarkers[i]) {
          rayHitMarkers[i].position.set(ep[0], ep[1], ep[2] + 0.03);
          rayHitMarkers[i].material.color.setHex(colorHex);
          rayHitMarkers[i].visible = toggleRays.checked && hit;
        }
      }
    }
  }

  // 5. Update Tire Force Vectors
  if (telem.wheel_forces && telem.wheel_contact_pts) {
    for (let i = 0; i < 4; i++) {
      const f = telem.wheel_forces[i];
      const pt = telem.wheel_contact_pts[i];
      const arrow = forceArrows[i];
      const mag = Math.hypot(f[0], f[1], f[2]);

      if (mag > 50 && pt && arrow) {
        arrow.position.set(pt[0], pt[1], pt[2]);
        arrow.setDirection(new THREE.Vector3(f[0], f[1], f[2]).normalize());
        arrow.setLength(Math.min(2.5, mag / 2500.0), 0.25, 0.15);
        // Green for forward drive, red for braking, cyan for cornering
        const isBraking = telem.brake > 0.1;
        arrow.setColor(isBraking ? 0xef4444 : (f[0] > 0 ? 0x10b981 : 0x06b6d4));
        arrow.visible = toggleForces.checked;
      } else if (arrow) {
        arrow.visible = false;
      }
    }
  }

  // 6. Velocity Vector Arrow
  if (velArrow && telem.vel) {
    const vMag = Math.hypot(telem.vel[0], telem.vel[1], telem.vel[2]);
    if (vMag > 0.2) {
      const bumperPos = new THREE.Vector3(1.6, 0, 0.2).applyMatrix4(roverGroup.matrixWorld);
      velArrow.position.copy(bumperPos);
      velArrow.setDirection(new THREE.Vector3(telem.vel[0], telem.vel[1], telem.vel[2]).normalize());
      velArrow.setLength(Math.min(3.0, vMag * 0.4), 0.3, 0.15);
      velArrow.visible = true;
    } else {
      velArrow.visible = false;
    }
  }

  // 7. Update HUD Telemetry
  elSpeed.innerText = ((telem.speed || 0) * 3.6).toFixed(1);
  elDist.innerText = (telem.pos[0] || 0).toFixed(1);
  elThrottle.innerText = Math.round((telem.throttle || 0) * 100);
  elSteer.innerText = (telem.steer_deg || 0).toFixed(1);

  const rollDeg = telem.roll_deg || 0;
  const pitchDeg = telem.pitch_deg || 0;
  elRoll.innerText = `${rollDeg.toFixed(1)}°`;
  elPitch.innerText = `${pitchDeg.toFixed(1)}°`;

  // Attitude hazard warning colors
  const absRoll = Math.abs(rollDeg);
  const absPitch = Math.abs(pitchDeg);
  elRoll.className = `angle-num ${absRoll > 45 ? 'angle-danger' : (absRoll > 25 ? 'angle-warn' : 'angle-safe')}`;
  elPitch.className = `angle-num ${absPitch > 40 ? 'angle-danger' : (absPitch > 25 ? 'angle-warn' : 'angle-safe')}`;

  // Course progress bar (0 to 150m)
  const progressPct = Math.min(100, Math.max(0, (telem.pos[0] / 150.0) * 100));
  elProgressFill.style.width = `${progressPct}%`;

  // Status Alerts
  if (telem.flipped) {
    elAlert.innerText = "CRITICAL ROLLOVER - VEHICLE FLIPPED";
    elAlert.className = "alert-danger";
    elAlert.style.display = "block";
  } else if (telem.belly_contact) {
    elAlert.innerText = "BELLY CONTACT - HIGH-CENTERING RISK";
    elAlert.className = "alert-warning";
    elAlert.style.display = "block";
  } else if (telem.pos[0] >= 140.0) {
    elAlert.innerText = "VICTORY! SUMMIT CONQUERED";
    elAlert.className = "alert-success";
    elAlert.style.display = "block";
  } else {
    elAlert.style.display = "none";
  }

  // Camera Follow logic
  updateCamera(telem);
}

function updateCamera(telem) {
  if (!roverGroup) return;
  const roverPos = roverGroup.position;

  if (cameraMode === 'chase') {
    const behindOffset = new THREE.Vector3(-6.5, 0, 3.2).applyMatrix4(roverGroup.matrixWorld);
    camera.position.lerp(behindOffset, 0.08);
    controls.target.lerp(roverPos, 0.12);
  } else if (cameraMode === 'cockpit') {
    const hoodPos = new THREE.Vector3(0.5, 0, 0.9).applyMatrix4(roverGroup.matrixWorld);
    const lookAtTarget = new THREE.Vector3(12.0, 0, 0.2).applyMatrix4(roverGroup.matrixWorld);
    camera.position.copy(hoodPos);
    controls.target.copy(lookAtTarget);
  } else if (cameraMode === 'overhead') {
    camera.position.set(roverPos.x, roverPos.y, roverPos.z + 18.0);
    controls.target.copy(roverPos);
  }
}

/**
 * Main Animation Loop
 */
function animate() {
  requestAnimationFrame(animate);

  const now = performance.now();
  const dt = (now - lastFrameTime) / 1000.0;
  lastFrameTime = now;

  if (manualMode) {
    // Send manual step to backend
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

    elTimeline.value = currentStepIdx;
    elTimeStep.innerText = `${currentStepIdx} / ${activeTrajectory.length - 1}`;
    updateSimulationState(activeTrajectory[currentStepIdx]);
  }

  controls.update();
  renderer.render(scene, camera);
}

/**
 * Interactive Manual Drive (WASD)
 */
function stepManualDrive() {
  let steer = 0.0;
  let throttle = 0.0;

  if (manualKeys.forward) throttle += 0.85;
  if (manualKeys.backward) throttle -= 0.85;
  if (manualKeys.left) steer -= 0.75;
  if (manualKeys.right) steer += 0.75;

  fetch('/api/manual_step', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ steer, throttle })
  })
  .then(r => r.json())
  .then(telem => updateSimulationState(telem))
  .catch(() => {});
}

/**
 * UI Event Listeners
 */
function setupEventListeners() {
  // Play/Pause
  elPlayPause.addEventListener('click', () => {
    isPlaying = !isPlaying;
    elPlayPause.innerText = isPlaying ? "⏸ Pause" : "▶ Play";
  });

  // Restart
  document.getElementById('btn-restart').addEventListener('click', () => {
    currentStepIdx = 0;
    elTimeline.value = 0;
  });

  // Timeline scrubber
  elTimeline.addEventListener('input', (e) => {
    currentStepIdx = parseInt(e.target.value);
    if (activeTrajectory[currentStepIdx]) {
      updateSimulationState(activeTrajectory[currentStepIdx]);
    }
  });

  // Playback speed
  document.getElementById('select-speed').addEventListener('change', (e) => {
    playbackSpeed = parseFloat(e.target.value);
  });

  // Generation select dropdown
  elGenSelect.addEventListener('change', (e) => {
    loadGeneration(e.target.value);
  });

  // Toggle wireframe boulders
  toggleBoulders.addEventListener('change', (e) => {
    if (bouldersGroup) {
      bouldersGroup.children.forEach(c => {
        if (c.name === "boulder_wire") c.visible = e.target.checked;
      });
    }
  });

  // Camera Mode buttons
  document.querySelectorAll('.cam-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      cameraMode = btn.dataset.cam;
    });
  });

  // Manual drive toggle
  const btnManual = document.getElementById('btn-toggle-manual');
  btnManual.addEventListener('click', () => {
    manualMode = !manualMode;
    if (manualMode) {
      btnManual.classList.add('active');
      btnManual.innerText = "🛑 Stop Manual Drive";
      elModeTag.innerText = "MANUAL CONTROL ACTIVE (WASD)";
      elModeTag.style.color = "#f59e0b";
      fetch('/api/manual_reset', { method: 'POST' });
    } else {
      btnManual.classList.remove('active');
      btnManual.innerText = "🕹️ Manual Drive (WASD)";
      loadGeneration(elGenSelect.value);
    }
  });

  // Evolve Next Generation button
  const btnEvolve = document.getElementById('btn-evolve-next');
  btnEvolve.addEventListener('click', () => {
    btnEvolve.disabled = true;
    btnEvolve.innerText = "⏳ Evolving in Background...";
    fetch('/api/evolve_step', { method: 'POST' })
      .then(r => r.json())
      .then(() => {
        setTimeout(() => {
          btnEvolve.disabled = false;
          btnEvolve.innerText = "⚡ Evolve Next Generation";
          refreshGenerationsList();
        }, 5000);
      });
  });

  // Keyboard controls for Manual Mode
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

  // Window resize
  window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
  });
}

function refreshGenerationsList() {
  fetch('/api/generations')
    .then(r => r.json())
    .then(list => {
      elGenSelect.innerHTML = '';
      list.forEach((item, idx) => {
        const opt = document.createElement('option');
        opt.value = item.filename;
        opt.innerText = `Gen ${idx + 1} (Fit: ${item.stats.best_fitness || '--'})`;
        elGenSelect.appendChild(opt);
      });
      elGenSelect.selectedIndex = list.length - 1;
      loadGeneration(elGenSelect.value);
    });
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', initScene);
