// Flapping Flight RL - Interactive Command Center Client
// Supports both live API connection (port 8775) and standalone offline simulation

// --- TAB SWITCHING ---
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    const tab = btn.dataset.tab;
    const panel = document.getElementById(`panel-${tab}`);
    if (panel) panel.classList.add('active');
  });
});

// --- THREE.JS 3D VIEWPORT SETUP ---
const container = document.getElementById('three-viewport');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x06090e);
scene.fog = new THREE.FogExp2(0x06090e, 0.2);

const camera = new THREE.PerspectiveCamera(45, (container.clientWidth || 800) / (container.clientHeight || 480), 0.01, 20);
camera.position.set(0.18, -0.32, 0.62);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(container.clientWidth || 800, container.clientHeight || 480);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(0, 0, 0.5);

// Lighting
scene.add(new THREE.AmbientLight(0xffffff, 0.65));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
dirLight.position.set(2, -2, 4);
scene.add(dirLight);

// Ground & Target Hover Ring
const grid = new THREE.GridHelper(2, 20, 0x1e293b, 0x0f172a);
grid.rotation.x = Math.PI / 2;
scene.add(grid);

const targetRing = new THREE.Mesh(
  new THREE.RingGeometry(0.035, 0.040, 32),
  new THREE.MeshBasicMaterial({ color: 0x10b981, side: THREE.DoubleSide })
);
targetRing.position.set(0, 0, 0.5);
scene.add(targetRing);

// Insect Body Construction
const insectGroup = new THREE.Group();
scene.add(insectGroup);

// Thorax
const thoraxGeom = new THREE.SphereGeometry(0.011, 16, 16);
thoraxGeom.scale(1.2, 0.8, 0.8);
const thorax = new THREE.Mesh(thoraxGeom, new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.4 }));
insectGroup.add(thorax);

// Abdomen
const abdomenGeom = new THREE.ConeGeometry(0.008, 0.032, 16);
abdomenGeom.rotateX(Math.PI / 2);
abdomenGeom.translate(0, -0.024, -0.002);
const abdomen = new THREE.Mesh(abdomenGeom, new THREE.MeshStandardMaterial({ color: 0x475569, roughness: 0.5 }));
insectGroup.add(abdomen);

// Head
const headGeom = new THREE.SphereGeometry(0.006, 12, 12);
headGeom.translate(0, 0.015, 0.002);
const head = new THREE.Mesh(headGeom, new THREE.MeshStandardMaterial({ color: 0x1e293b }));
insectGroup.add(head);

// Wing Generator
function createWing(isLeft) {
  const group = new THREE.Group();
  const shape = new THREE.Shape();
  shape.moveTo(0, 0);
  shape.bezierCurveTo(0.01, 0.016, 0.035, 0.022, 0.05, 0.006);
  shape.bezierCurveTo(0.045, -0.008, 0.02, -0.014, 0, 0);

  const wingMesh = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshPhysicalMaterial({
      color: 0x38bdf8,
      transparent: true,
      opacity: 0.65,
      roughness: 0.1,
      transmission: 0.8,
      side: THREE.DoubleSide
    })
  );

  const levLine = new THREE.Line(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0.001),
      new THREE.Vector3(0.018, 0.018, 0.001),
      new THREE.Vector3(0.05, 0.006, 0.001)
    ]),
    new THREE.LineBasicMaterial({ color: 0x06b6d4, linewidth: 2 })
  );

  group.add(wingMesh);
  group.add(levLine);
  return group;
}

const hingeL = new THREE.Group();
hingeL.position.set(-0.008, 0.005, 0.004);
const wingMeshL = createWing(true);
hingeL.add(wingMeshL);
insectGroup.add(hingeL);

const hingeR = new THREE.Group();
hingeR.position.set(0.008, 0.005, 0.004);
const wingMeshR = createWing(false);
wingMeshR.scale.set(-1, 1, 1);
hingeR.add(wingMeshR);
insectGroup.add(hingeR);

// Responsive canvas resize
window.addEventListener('resize', () => {
  if (container.clientWidth > 0) {
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
  }
});

// --- TELEMETRY OSCILLOSCOPES (CANVAS CHARTS) ---
const canvasLift = document.getElementById('chart-lift');
const ctxLift = canvasLift.getContext('2d');
const liftHistory = [];
const maxHistory = 150;

function drawLiftChart() {
  const w = canvasLift.width = canvasLift.clientWidth;
  const h = canvasLift.height = canvasLift.clientHeight;
  ctxLift.clearRect(0, 0, w, h);

  // Background grid line
  ctxLift.strokeStyle = 'rgba(255,255,255,0.06)';
  ctxLift.lineWidth = 1;
  ctxLift.beginPath();
  ctxLift.moveTo(0, h * 0.5); ctxLift.lineTo(w, h * 0.5);
  ctxLift.stroke();

  // Weight Reference Line (15.69 mN)
  const weightY = h - (15.69 / 40.0) * h;
  ctxLift.strokeStyle = '#f59e0b';
  ctxLift.setLineDash([4, 4]);
  ctxLift.beginPath();
  ctxLift.moveTo(0, weightY); ctxLift.lineTo(w, weightY);
  ctxLift.stroke();
  ctxLift.setLineDash([]);
  ctxLift.fillStyle = '#f59e0b';
  ctxLift.font = '10px monospace';
  ctxLift.fillText('Weight Thresh: 15.69 mN', 10, weightY - 4);

  // Lift trace
  if (liftHistory.length > 1) {
    ctxLift.strokeStyle = '#06b6d4';
    ctxLift.lineWidth = 2;
    ctxLift.beginPath();
    for (let i = 0; i < liftHistory.length; i++) {
      const x = (i / (maxHistory - 1)) * w;
      const y = h - (Math.min(40.0, Math.max(0.0, liftHistory[i])) / 40.0) * h;
      if (i === 0) ctxLift.moveTo(x, y); else ctxLift.lineTo(x, y);
    }
    ctxLift.stroke();
  }
}

// 6-DOF Attitude Chart
const canvasAtt = document.getElementById('chart-attitude');
const ctxAtt = canvasAtt.getContext('2d');
const pitchHist = [];
const rollHist = [];

function drawAttitudeChart() {
  const w = canvasAtt.width = canvasAtt.clientWidth;
  const h = canvasAtt.height = canvasAtt.clientHeight;
  ctxAtt.clearRect(0, 0, w, h);

  ctxAtt.strokeStyle = 'rgba(255,255,255,0.08)';
  ctxAtt.beginPath();
  ctxAtt.moveTo(0, h / 2); ctxAtt.lineTo(w, h / 2);
  ctxAtt.stroke();

  function drawSeries(hist, color) {
    if (hist.length < 2) return;
    ctxAtt.strokeStyle = color;
    ctxAtt.lineWidth = 1.5;
    ctxAtt.beginPath();
    for (let i = 0; i < hist.length; i++) {
      const x = (i / (maxHistory - 1)) * w;
      const y = (h / 2) - (hist[i] / 45.0) * (h / 2);
      if (i === 0) ctxAtt.moveTo(x, y); else ctxAtt.lineTo(x, y);
    }
    ctxAtt.stroke();
  }

  drawSeries(pitchHist, '#10b981'); // Pitch
  drawSeries(rollHist, '#8b5cf6');  // Roll

  ctxAtt.font = '10px monospace';
  ctxAtt.fillStyle = '#10b981'; ctxAtt.fillText('Pitch (deg)', 10, 16);
  ctxAtt.fillStyle = '#8b5cf6'; ctxAtt.fillText('Roll (deg)', 90, 16);
}

// Polar Curves Static Canvas
const canvasPolars = document.getElementById('chart-polars');
if (canvasPolars) {
  const ctxP = canvasPolars.getContext('2d');
  const w = canvasPolars.width = canvasPolars.clientWidth || 300;
  const h = canvasPolars.height = canvasPolars.clientHeight || 200;

  ctxP.strokeStyle = '#06b6d4';
  ctxP.lineWidth = 2;
  ctxP.beginPath();
  for (let a = 0; a <= 90; a++) {
    const rad = (a * Math.PI) / 180.0;
    const cl = 1.95 * Math.sin(2 * rad);
    const x = (a / 90.0) * (w - 40) + 30;
    const y = h - (cl / 2.5) * (h - 30) - 20;
    if (a === 0) ctxP.moveTo(x, y); else ctxP.lineTo(x, y);
  }
  ctxP.stroke();

  ctxP.strokeStyle = '#f43f5e';
  ctxP.beginPath();
  for (let a = 0; a <= 90; a++) {
    const rad = (a * Math.PI) / 180.0;
    const cd = 0.15 + 3.0 * (1 - Math.cos(2 * rad));
    const x = (a / 90.0) * (w - 40) + 30;
    const y = h - (cd / 4.0) * (h - 30) - 20;
    if (a === 0) ctxP.moveTo(x, y); else ctxP.lineTo(x, y);
  }
  ctxP.stroke();

  ctxP.font = '11px monospace';
  ctxP.fillStyle = '#06b6d4'; ctxP.fillText('C_L (Lift: LEV Peak ~1.95)', 35, 25);
  ctxP.fillStyle = '#f43f5e'; ctxP.fillText('C_D (Drag)', 35, 42);
}

// --- TELEMETRY UPDATE LOGIC ---
function updateUI(d) {
  document.getElementById('hero-alt').innerText = d.altitude_m.toFixed(3) + ' m';
  document.getElementById('hero-lift').innerText = d.lift_ratio.toFixed(2) + 'x';
  document.getElementById('hero-freq').innerText = d.frequency_hz.toFixed(1) + ' Hz';
  document.getElementById('hero-net-lift').innerText = d.lift_mn.toFixed(1) + ' mN';

  document.getElementById('val-cur-lift').innerText = d.lift_mn.toFixed(2) + ' mN';
  document.getElementById('val-vz').innerText = d.vel_z_mps.toFixed(2) + ' m/s';
  
  const liftPct = Math.min(100, Math.max(0, (d.lift_mn / 25.0) * 100));
  document.getElementById('bar-lift').style.width = liftPct + '%';

  const clapChip = document.getElementById('chip-clap');
  clapChip.style.display = d.clap_and_fling ? 'inline-block' : 'none';

  const statTrans = document.getElementById('stat-trans-lift');
  if (statTrans) statTrans.innerText = (d.lift_mn * 0.85).toFixed(1) + ' mN';
  const statRot = document.getElementById('stat-rot-force');
  if (statRot) statRot.innerText = d.rotational_mn.toFixed(1) + ' mN';
  const statMass = document.getElementById('stat-added-mass');
  if (statMass) statMass.innerText = d.added_mass_mn.toFixed(1) + ' mN';

  liftHistory.push(d.lift_mn);
  if (liftHistory.length > maxHistory) liftHistory.shift();

  pitchHist.push(d.pitch_deg);
  if (pitchHist.length > maxHistory) pitchHist.shift();

  rollHist.push(d.roll_deg);
  if (rollHist.length > maxHistory) rollHist.shift();

  drawLiftChart();
  drawAttitudeChart();

  // Update 3D Insect Pose
  insectGroup.position.set(d.pos_x_m, d.pos_y_m, d.altitude_m);
  insectGroup.rotation.x = THREE.MathUtils.degToRad(d.pitch_deg + 20.0);
  insectGroup.rotation.y = THREE.MathUtils.degToRad(d.roll_deg);
  insectGroup.rotation.z = THREE.MathUtils.degToRad(d.yaw_deg);

  // Update Wings Articulation
  hingeL.rotation.y = THREE.MathUtils.degToRad(d.phi_L_deg);
  hingeL.rotation.x = THREE.MathUtils.degToRad(d.theta_L_deg);
  wingMeshL.rotation.z = THREE.MathUtils.degToRad(d.psi_L_deg);

  hingeR.rotation.y = -THREE.MathUtils.degToRad(d.phi_R_deg);
  hingeR.rotation.x = THREE.MathUtils.degToRad(d.theta_R_deg);
  wingMeshR.rotation.z = -THREE.MathUtils.degToRad(d.psi_R_deg);
}

// Client-side simulation fallback when server is offline
let clientSimTime = 0.0;
function runClientSim() {
  clientSimTime += 0.016;
  const f = 27.0;
  const omega = 2 * Math.PI * f;
  const phase = omega * clientSimTime;

  const pL = manualActive ? parseFloat(document.getElementById('rng-pitch-l').value) * 25.0 : 0.0;
  const pR = manualActive ? parseFloat(document.getElementById('rng-pitch-r').value) * 25.0 : 0.0;
  const sBias = manualActive ? parseFloat(document.getElementById('rng-stroke-bias').value) * 15.0 : 0.0;

  const phiL = 20.0 + 62.5 * Math.cos(phase) + sBias;
  const phiR = 20.0 + 62.5 * Math.cos(phase) + sBias;
  const psiL = 45.0 * Math.tanh(3.0 * Math.sin(phase)) + pL;
  const psiR = 45.0 * Math.tanh(3.0 * Math.sin(phase)) + pR;
  const thetaL = 8.0 * Math.sin(2 * phase);
  const thetaR = 8.0 * Math.sin(2 * phase);

  const atDorsal = Math.cos(phase) > 0.85;
  const liftRatio = atDorsal ? 1.48 : (1.16 + 0.35 * Math.sin(phase));
  const lift_mn = liftRatio * 15.69;

  const d = {
    altitude_m: 0.50 + 0.012 * Math.sin(clientSimTime * 4.0) + 0.003 * Math.sin(2 * phase),
    pos_x_m: 0.002 * Math.sin(clientSimTime * 2.0),
    pos_y_m: 0.001 * Math.cos(clientSimTime * 2.0),
    vel_z_mps: 0.048 * Math.cos(clientSimTime * 4.0),
    pitch_deg: 20.0 + 1.5 * Math.sin(phase),
    roll_deg: (pL - pR) * 0.4,
    yaw_deg: (pL - pR) * 0.2,
    phi_L_deg: phiL,
    phi_R_deg: phiR,
    psi_L_deg: psiL,
    psi_R_deg: psiR,
    theta_L_deg: thetaL,
    theta_R_deg: thetaR,
    lift_mn: lift_mn,
    weight_mn: 15.69,
    lift_ratio: liftRatio,
    rotational_mn: 2.8 * Math.abs(Math.cos(phase)),
    added_mass_mn: 1.2 * Math.abs(Math.sin(phase)),
    clap_and_fling: atDorsal,
    frequency_hz: 27.0
  };
  updateUI(d);
}

async function pollTelemetry() {
  try {
    const res = await fetch('/api/telemetry');
    if (!res.ok) throw new Error('API offline');
    const d = await res.json();
    updateUI(d);
  } catch (err) {
    // Run real-time client physics fallback
    runClientSim();
  }
}

setInterval(pollTelemetry, 30); // ~33 Hz

function render() {
  requestAnimationFrame(render);
  controls.update();
  renderer.render(scene, camera);
}
render();

// --- MANUAL CONTROLS ---
let manualActive = false;
const btnManual = document.getElementById('btn-toggle-manual');
btnManual.addEventListener('click', () => {
  manualActive = !manualActive;
  btnManual.innerText = manualActive ? 'Manual: ON' : 'Enable Manual';
  btnManual.style.background = manualActive ? '#0284c7' : '#1e293b';
});

function updateSliders() {
  const pL = parseFloat(document.getElementById('rng-pitch-l').value);
  const pR = parseFloat(document.getElementById('rng-pitch-r').value);
  const sb = parseFloat(document.getElementById('rng-stroke-bias').value);

  document.getElementById('lbl-pitch-l').innerText = (pL * 25).toFixed(1) + '°';
  document.getElementById('lbl-pitch-r').innerText = (pR * 25).toFixed(1) + '°';
  document.getElementById('lbl-stroke-bias').innerText = (sb * 15).toFixed(1) + '°';
}

document.getElementById('rng-pitch-l').addEventListener('input', updateSliders);
document.getElementById('rng-pitch-r').addEventListener('input', updateSliders);
document.getElementById('rng-stroke-bias').addEventListener('input', updateSliders);

document.getElementById('btn-reset-flight').addEventListener('click', () => {
  clientSimTime = 0.0;
  fetch('/api/reset').catch(() => {});
});
