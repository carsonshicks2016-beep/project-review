import * as THREE from 'three';
import { World, clamp } from './physics.js';
import { Vehicle, BODY_SKIN, GLASS_SKIN, WHEEL_RADIUS } from './vehicle.js';
import { buildCity, randomRoadPoint, WORLD_EXTENT, PITCH, BLOCK_HALF, GRID } from './city.js';
import { Chaser } from './ai.js';
import { Sound } from './audio.js';

const INVERT_STEERING = true;
const MAX_CHASERS = 7;
const START_CHASERS = 2;
const ESCALATE_EVERY = 24; // seconds between new pursuers

// ---------------------------------------------------------------- textures

function cityGroundTexture() {
  const S = 2048;
  const cv = document.createElement('canvas');
  cv.width = cv.height = S;
  const g = cv.getContext('2d');
  const E = WORLD_EXTENT;
  const toPx = (v) => ((v + E) / (2 * E)) * S;
  const scale = S / (2 * E);

  g.fillStyle = '#212429';
  g.fillRect(0, 0, S, S);

  // Blocks (sidewalk / lot surface)
  for (let i = -GRID; i <= GRID; i++) {
    for (let j = -GRID; j <= GRID; j++) {
      const h = (BLOCK_HALF + 1.4) * scale;
      g.fillStyle = '#31353c';
      g.fillRect(toPx(i * PITCH) - h, toPx(j * PITCH) - h, h * 2, h * 2);
      g.fillStyle = '#3b4048';
      g.fillRect(toPx(i * PITCH) - h + 2, toPx(j * PITCH) - h + 2, h * 2 - 4, h * 2 - 4);
    }
  }

  // Lane dashes down every road centreline.
  g.strokeStyle = 'rgba(226, 200, 92, 0.55)';
  g.lineWidth = Math.max(2, 0.35 * scale);
  g.setLineDash([6 * scale, 6 * scale]);
  for (let k = -GRID; k <= GRID; k++) {
    const c = toPx(PITCH / 2 + k * PITCH);
    g.beginPath(); g.moveTo(c, 0); g.lineTo(c, S); g.stroke();
    g.beginPath(); g.moveTo(0, c); g.lineTo(S, c); g.stroke();
  }

  // Crosswalk ladders at each intersection.
  g.setLineDash([]);
  g.fillStyle = 'rgba(235,238,244,0.34)';
  for (let a = -GRID; a <= GRID; a++) {
    for (let b = -GRID; b <= GRID; b++) {
      const cx = PITCH / 2 + a * PITCH, cz = PITCH / 2 + b * PITCH;
      for (let s = -1; s <= 1; s += 2) {
        for (let n = -3; n <= 3; n++) {
          g.fillRect(toPx(cx + n * 1.6) - 0.4 * scale, toPx(cz + s * 9) - 1.6 * scale,
            0.8 * scale, 3.2 * scale);
          g.fillRect(toPx(cx + s * 9) - 1.6 * scale, toPx(cz + n * 1.6) - 0.4 * scale,
            3.2 * scale, 0.8 * scale);
        }
      }
    }
  }

  const tex = new THREE.CanvasTexture(cv);
  tex.anisotropy = 8;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function windowTexture() {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 128;
  const g = cv.getContext('2d');
  // Kept light: this multiplies the building albedo, so a dark texture turns
  // every facade black under a daylight sky.
  g.fillStyle = '#8b929c';
  g.fillRect(0, 0, 128, 128);
  for (let y = 0; y < 4; y++) {
    for (let x = 0; x < 4; x++) {
      const lit = Math.random();
      if (lit > 0.78) g.fillStyle = '#ffe0ad';
      else if (lit > 0.45) g.fillStyle = '#69727e';
      else g.fillStyle = '#525b66';
      g.fillRect(x * 32 + 6, y * 32 + 8, 20, 17);
    }
  }
  const tex = new THREE.CanvasTexture(cv);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function rimTexture() {
  const cv = document.createElement('canvas');
  cv.width = cv.height = 128;
  const g = cv.getContext('2d');
  g.fillStyle = '#15171b';
  g.fillRect(0, 0, 128, 128);
  g.translate(64, 64);
  g.fillStyle = '#9aa3ad';
  g.beginPath(); g.arc(0, 0, 40, 0, Math.PI * 2); g.fill();
  g.fillStyle = '#15171b';
  for (let i = 0; i < 5; i++) {
    g.rotate((Math.PI * 2) / 5);
    g.beginPath();
    g.moveTo(6, 12); g.lineTo(-6, 12); g.lineTo(-13, 36); g.lineTo(13, 36);
    g.closePath(); g.fill();
  }
  g.fillStyle = '#c3ccd6';
  g.beginPath(); g.arc(0, 0, 11, 0, Math.PI * 2); g.fill();
  const tex = new THREE.CanvasTexture(cv);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// ---------------------------------------------------------------- scene

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.4;
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x9db4cc);
scene.fog = new THREE.Fog(0x9db4cc, 90, 460);

const camera = new THREE.PerspectiveCamera(68, innerWidth / innerHeight, 0.2, 900);

// Strong sky/bounce fill: street level is mostly shadowed by towers, and
// unlifted shadows read as night against a daylight sky.
scene.add(new THREE.HemisphereLight(0xc6dcf5, 0x6b7078, 2.1));
const sun = new THREE.DirectionalLight(0xffe6c4, 2.6);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.near = 10;
sun.shadow.camera.far = 260;
sun.shadow.camera.left = -80;
sun.shadow.camera.right = 80;
sun.shadow.camera.top = 80;
sun.shadow.camera.bottom = -80;
sun.shadow.bias = -0.0008;
scene.add(sun);
scene.add(sun.target);

const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(WORLD_EXTENT * 2, WORLD_EXTENT * 2),
  new THREE.MeshStandardMaterial({ map: cityGroundTexture(), roughness: 0.95, metalness: 0 })
);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// ---------------------------------------------------------------- world

const world = new World();
const city = buildCity(world);

const winTex = windowTexture();
const boxGeo = new THREE.BoxGeometry(1, 1, 1);
for (const b of city.buildings) {
  const w = b.max.x - b.min.x, h = b.max.y - b.min.y, d = b.max.z - b.min.z;
  const map = winTex.clone();
  map.needsUpdate = true;
  map.repeat.set(Math.max(1, Math.round(w / 4.5)), Math.max(1, Math.round(h / 4)));
  const mat = new THREE.MeshStandardMaterial({
    color: b.color, map, emissiveMap: map, emissive: 0xffcf90,
    emissiveIntensity: 0.12, roughness: 0.82, metalness: 0.05,
  });
  const m = new THREE.Mesh(boxGeo, mat);
  m.scale.set(w, h, d);
  m.position.set((b.min.x + b.max.x) / 2, h / 2, (b.min.z + b.max.z) / 2);
  m.castShadow = true;
  m.receiveShadow = true;
  scene.add(m);
}

const propMat = new THREE.MeshStandardMaterial({ color: 0xc4ad55, roughness: 0.8 });
const propMesh = new THREE.InstancedMesh(boxGeo, propMat, city.props.length);
propMesh.castShadow = true;
propMesh.receiveShadow = true;
{
  const m4 = new THREE.Matrix4();
  city.props.forEach((p, i) => {
    m4.makeScale(p.max.x - p.min.x, p.max.y - p.min.y, p.max.z - p.min.z);
    m4.setPosition((p.min.x + p.max.x) / 2, (p.min.y + p.max.y) / 2, (p.min.z + p.max.z) / 2);
    propMesh.setMatrixAt(i, m4);
  });
}
scene.add(propMesh);

// ---------------------------------------------------------------- car views

const rimTex = rimTexture();
const wheelGeo = new THREE.CylinderGeometry(WHEEL_RADIUS, WHEEL_RADIUS, 0.30, 20);
wheelGeo.rotateZ(Math.PI / 2); // axle along local X, then oriented by basis below

function skinGeometry(quads) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(quads.length * 6 * 3), 3));
  return g;
}

class CarView {
  constructor(vehicle) {
    this.v = vehicle;
    this.root = new THREE.Group();

    const paint = new THREE.MeshStandardMaterial({
      color: vehicle.color, roughness: 0.36, metalness: 0.5,
      side: THREE.DoubleSide, flatShading: true,
    });
    this.paint = paint;
    this.bodyGeo = skinGeometry(BODY_SKIN);
    this.body = new THREE.Mesh(this.bodyGeo, paint);
    this.body.castShadow = true;
    scene.add(this.body);

    const glassMat = new THREE.MeshStandardMaterial({
      color: 0x10161d, roughness: 0.08, metalness: 0.85,
      transparent: true, opacity: 0.82, side: THREE.DoubleSide, flatShading: true,
    });
    this.glassGeo = skinGeometry(GLASS_SKIN);
    this.glass = new THREE.Mesh(this.glassGeo, glassMat);
    scene.add(this.glass);

    const tire = new THREE.MeshStandardMaterial({ color: 0x14161a, roughness: 0.95 });
    const rim = new THREE.MeshStandardMaterial({ map: rimTex, roughness: 0.4, metalness: 0.7 });
    this.wheels = vehicle.wheelNodes.map(() => {
      const m = new THREE.Mesh(wheelGeo, [tire, rim, rim]);
      m.castShadow = true;
      scene.add(m);
      return m;
    });

    // Beam-view debug lattice.
    this.lineGeo = new THREE.BufferGeometry();
    this.lineGeo.setAttribute('position',
      new THREE.BufferAttribute(new Float32Array(vehicle.beams.length * 6), 3));
    this.lineGeo.setAttribute('color',
      new THREE.BufferAttribute(new Float32Array(vehicle.beams.length * 6), 3));
    this.lines = new THREE.LineSegments(this.lineGeo,
      new THREE.LineBasicMaterial({ vertexColors: true }));
    this.lines.visible = false;
    scene.add(this.lines);
  }

  setVisible(on) {
    this.body.visible = on;
    this.glass.visible = on;
    for (const w of this.wheels) w.visible = on;
    if (!on) this.lines.visible = false;
  }

  writeSkin(geo, quads) {
    const arr = geo.attributes.position.array;
    const N = this.v.chassisNodes;
    let p = 0;
    const put = (n) => { arr[p++] = n.x; arr[p++] = n.y; arr[p++] = n.z; };
    for (const [a, b, c, d] of quads) {
      put(N[a]); put(N[b]); put(N[c]);
      put(N[a]); put(N[c]); put(N[d]);
    }
    geo.attributes.position.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
  }

  update(beamView) {
    this.writeSkin(this.bodyGeo, BODY_SKIN);
    this.writeSkin(this.glassGeo, GLASS_SKIN);

    const v = this.v;
    for (let i = 0; i < 4; i++) {
      const n = v.wheelNodes[i];
      const mesh = this.wheels[i];
      const steer = i < 2 ? v.steerAngle : 0;
      const ca = Math.cos(steer), sa = Math.sin(steer);

      // Axle = body right, rotated by the steering angle about body up.
      const ax = v.right.x * ca - v.fwd.x * sa;
      const ay = v.right.y * ca - v.fwd.y * sa;
      const az = v.right.z * ca - v.fwd.z * sa;
      const fx = v.fwd.x * ca + v.right.x * sa;
      const fy = v.fwd.y * ca + v.right.y * sa;
      const fz = v.fwd.z * ca + v.right.z * sa;

      // Third axis completes the frame; spin rolls f/u about the axle.
      const ux = ay * fz - az * fy;
      const uy = az * fx - ax * fz;
      const uz = ax * fy - ay * fx;

      const s = -v.wheelSpin[i], cs = Math.cos(s), ss = Math.sin(s);
      mesh.matrixAutoUpdate = false;
      mesh.matrix.set(
        ax, fx * cs + ux * ss, -fx * ss + ux * cs, n.x,
        ay, fy * cs + uy * ss, -fy * ss + uy * cs, n.y,
        az, fz * cs + uz * ss, -fz * ss + uz * cs, n.z,
        0, 0, 0, 1
      );
      mesh.matrixWorld.copy(mesh.matrix);
      mesh.matrixWorldNeedsUpdate = false;
    }

    this.lines.visible = beamView && this.body.visible;
    if (this.lines.visible) {
      const pos = this.lineGeo.attributes.position.array;
      const col = this.lineGeo.attributes.color.array;
      let p = 0, c = 0;
      for (const bm of v.beams) {
        pos[p++] = bm.a.x; pos[p++] = bm.a.y; pos[p++] = bm.a.z;
        pos[p++] = bm.b.x; pos[p++] = bm.b.y; pos[p++] = bm.b.z;
        // green intact -> yellow yielding -> red broken
        const strain = bm.broken ? 1 : clamp(Math.abs(bm.rest - bm.restOrig) / (bm.restOrig * 0.25), 0, 1);
        const r = bm.broken ? 1 : strain, gch = bm.broken ? 0.05 : 1 - strain * 0.8;
        for (let k = 0; k < 2; k++) { col[c++] = r; col[c++] = gch; col[c++] = 0.15; }
      }
      this.lineGeo.attributes.position.needsUpdate = true;
      this.lineGeo.attributes.color.needsUpdate = true;
      this.lineGeo.computeBoundingSphere();
    }
  }
}

// ---------------------------------------------------------------- sparks

const SPARKS = 900;
const sparkPos = new Float32Array(SPARKS * 3);
const sparkVel = new Float32Array(SPARKS * 3);
const sparkCol = new Float32Array(SPARKS * 3);
const sparkLife = new Float32Array(SPARKS);
let sparkHead = 0;
sparkPos.fill(-9999);

const sparkGeo = new THREE.BufferGeometry();
sparkGeo.setAttribute('position', new THREE.BufferAttribute(sparkPos, 3));
sparkGeo.setAttribute('color', new THREE.BufferAttribute(sparkCol, 3));
const sparks = new THREE.Points(sparkGeo, new THREE.PointsMaterial({
  size: 0.22, vertexColors: true, transparent: true, opacity: 0.95,
  depthWrite: false, blending: THREE.AdditiveBlending,
}));
sparks.frustumCulled = false;
scene.add(sparks);

function emitSparks(x, y, z, speed) {
  const n = Math.min(34, 5 + Math.floor(speed * 1.6));
  for (let i = 0; i < n; i++) {
    const k = sparkHead;
    sparkHead = (sparkHead + 1) % SPARKS;
    sparkPos[k * 3] = x; sparkPos[k * 3 + 1] = y; sparkPos[k * 3 + 2] = z;
    const sp = 2 + Math.random() * speed * 0.55;
    const th = Math.random() * Math.PI * 2;
    const ph = Math.random() * Math.PI * 0.5;
    sparkVel[k * 3] = Math.cos(th) * Math.sin(ph) * sp;
    sparkVel[k * 3 + 1] = Math.abs(Math.cos(ph)) * sp * 0.9 + 1;
    sparkVel[k * 3 + 2] = Math.sin(th) * Math.sin(ph) * sp;
    const warm = Math.random();
    sparkCol[k * 3] = 1;
    sparkCol[k * 3 + 1] = 0.55 + warm * 0.4;
    sparkCol[k * 3 + 2] = 0.15 + warm * 0.25;
    sparkLife[k] = 0.45 + Math.random() * 0.55;
  }
}

function updateSparks(dt) {
  for (let i = 0; i < SPARKS; i++) {
    if (sparkLife[i] <= 0) continue;
    sparkLife[i] -= dt;
    if (sparkLife[i] <= 0) { sparkPos[i * 3 + 1] = -9999; continue; }
    sparkVel[i * 3 + 1] -= 22 * dt;
    sparkPos[i * 3] += sparkVel[i * 3] * dt;
    sparkPos[i * 3 + 1] += sparkVel[i * 3 + 1] * dt;
    sparkPos[i * 3 + 2] += sparkVel[i * 3 + 2] * dt;
    if (sparkPos[i * 3 + 1] < 0.05) {
      sparkPos[i * 3 + 1] = 0.05;
      sparkVel[i * 3 + 1] *= -0.32;
      sparkVel[i * 3] *= 0.6;
      sparkVel[i * 3 + 2] *= 0.6;
    }
    const f = Math.min(1, sparkLife[i] * 2);
    sparkCol[i * 3 + 1] *= 0.995;
    sparkCol[i * 3 + 2] *= 0.99;
  }
  sparkGeo.attributes.position.needsUpdate = true;
  sparkGeo.attributes.color.needsUpdate = true;
}

// ---------------------------------------------------------------- vehicles

const player = new Vehicle(world, { x: 0, z: 0, heading: 0, color: 0xdc3c2a, isPlayer: true, name: 'player' });
const playerView = new CarView(player);

const COP_COLORS = [0x1f2c44, 0x22303f, 0x1b2740, 0x2a3550, 0x1e2b3c, 0x243247, 0x1a2438];
const chasers = [];
for (let i = 0; i < MAX_CHASERS; i++) {
  const v = new Vehicle(world, { x: 9999, z: 9999 + i * 12, heading: 0, color: COP_COLORS[i], name: `cop${i}` });
  v.group.active = false;
  const view = new CarView(v);
  view.setVisible(false);
  chasers.push({ v, ai: new Chaser(v), view, deployed: false });
}

// Roof light bars, so pursuers read at a glance.
for (const c of chasers) {
  const bar = new THREE.Mesh(
    new THREE.BoxGeometry(1.0, 0.16, 0.3),
    new THREE.MeshStandardMaterial({ color: 0x111417, emissive: 0xff2222, emissiveIntensity: 2 })
  );
  c.lightBar = bar;
  scene.add(bar);
}

// ---------------------------------------------------------------- impacts

const impacts = [];
world.onImpact = (node, speed, kind) => {
  if (speed < 6 || impacts.length > 24) return;
  impacts.push({ x: node.x, y: node.y, z: node.z, speed, group: node.group });
};
world.onBeamBreak = (beam, g) => {
  if (impacts.length > 24) return;
  impacts.push({ x: beam.a.x, y: beam.a.y, z: beam.a.z, speed: 9, group: g, tear: true });
};

// ---------------------------------------------------------------- input

const keys = new Set();
addEventListener('keydown', (e) => {
  if ([' ', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.key)) e.preventDefault();
  keys.add(e.key.toLowerCase());
  if (e.key.toLowerCase() === 'c') camMode = (camMode + 1) % 3;
  if (e.key.toLowerCase() === 'b') beamView = !beamView;
  if (e.key.toLowerCase() === 'r') resetRun();
  if (e.key.toLowerCase() === 't') respawnPlayer();
});
addEventListener('keyup', (e) => keys.delete(e.key.toLowerCase()));
addEventListener('blur', () => keys.clear());
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

const sound = new Sound();
const boot = document.getElementById('boot');
boot.addEventListener('click', () => {
  sound.start();
  boot.style.display = 'none';
  running = true;
  startTime = performance.now() / 1000;
});

// ---------------------------------------------------------------- game state

let camMode = 0;
let beamView = false;
let running = false;
let gameOver = false;
let startTime = 0;
let elapsed = 0;
let nextSpawn = ESCALATE_EVERY;
let shake = 0;
let flashAmt = 0;
let bestTime = 0;

const camPos = new THREE.Vector3(0, 6, -12);
const camLook = new THREE.Vector3();

function deployChaser() {
  const idle = chasers.find((c) => !c.deployed);
  if (!idle) return;
  let p, tries = 0;
  do {
    p = randomRoadPoint();
    tries++;
  } while (Math.hypot(p.x - player.pos.x, p.z - player.pos.z) < 70 && tries < 40);
  idle.v.reset(p.x, p.z, p.heading);
  idle.v.group.active = true;
  idle.deployed = true;
  idle.view.setVisible(true);
  idle.lightBar.visible = true;
}

function respawnPlayer() {
  const p = randomRoadPoint();
  player.reset(p.x, p.z, p.heading);
  shake = 0;
}

function resetRun() {
  player.reset(0, 0, 0);
  for (const c of chasers) {
    c.deployed = false;
    c.v.group.active = false;
    c.v.reset(9999, 9999, 0);
    c.view.setVisible(false);
    c.lightBar.visible = false;
    c.ai.stuckTime = 0;
    c.ai.reverseTime = 0;
  }
  for (let i = 0; i < START_CHASERS; i++) deployChaser();
  elapsed = 0;
  startTime = performance.now() / 1000;
  nextSpawn = ESCALATE_EVERY;
  gameOver = false;
  document.getElementById('overlay').classList.remove('show');
  running = true;
}

for (let i = 0; i < START_CHASERS; i++) deployChaser();

// ---------------------------------------------------------------- HUD

const el = {
  mph: document.getElementById('mph'),
  gear: document.getElementById('gear'),
  time: document.getElementById('time'),
  cops: document.getElementById('cops'),
  near: document.getElementById('near'),
  dmg: document.getElementById('dmg'),
  dmgpct: document.getElementById('dmgpct'),
  flash: document.getElementById('flash'),
  overlay: document.getElementById('overlay'),
  ovScore: document.getElementById('ovScore'),
};

// ---------------------------------------------------------------- loop

let last = performance.now();

function frame(now) {
  requestAnimationFrame(frame);
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;

  if (running && !gameOver) elapsed = now / 1000 - startTime;

  // --- player input
  const c = player.controls;
  const fwdKey = keys.has('w') || keys.has('arrowup');
  const backKey = keys.has('s') || keys.has('arrowdown');
  c.throttle = (fwdKey ? 1 : 0) + (backKey ? -1 : 0);
  const left = keys.has('a') || keys.has('arrowleft');
  const right = keys.has('d') || keys.has('arrowright');
  // Inverted steering: A/left steers right, D/right steers left. Player input
  // only — the AI computes its steering geometrically and must stay unflipped.
  c.steer = INVERT_STEERING
    ? (left ? 1 : 0) + (right ? -1 : 0)
    : (right ? 1 : 0) + (left ? -1 : 0);
  c.handbrake = keys.has(' ');
  c.brake = 0;
  if (gameOver) { c.throttle = 0; c.steer = 0; c.brake = 1; }

  // --- AI
  for (const ch of chasers) {
    if (ch.deployed) ch.ai.think(world, player, dt);
  }

  // --- physics
  world.refreshBroadphase();
  world.step(dt);

  // --- impacts
  let playerHit = 0;
  for (const im of impacts) {
    emitSparks(im.x, im.y, im.z, im.speed);
    if (im.group === player.group) playerHit = Math.max(playerHit, im.speed);
    const d = Math.hypot(im.x - camera.position.x, im.z - camera.position.z);
    if (d < 90) sound.crash(im.speed * (im.tear ? 0.7 : 1) * clamp(1 - d / 90, 0.15, 1));
  }
  impacts.length = 0;
  if (playerHit > 0) {
    shake = Math.min(1.2, shake + playerHit * 0.035);
    flashAmt = Math.min(0.5, flashAmt + playerHit * 0.012);
  }
  shake *= Math.pow(0.02, dt);
  flashAmt *= Math.pow(0.01, dt);
  el.flash.style.opacity = flashAmt.toFixed(3);

  // --- views
  playerView.update(beamView);
  for (const ch of chasers) {
    if (!ch.deployed) continue;
    ch.view.update(beamView);
    const N = ch.v.chassisNodes;
    ch.lightBar.position.set(
      (N[16].x + N[17].x) / 2, (N[16].y + N[17].y) / 2 + 0.12, (N[16].z + N[17].z) / 2);
    ch.lightBar.material.emissive.setHex(Math.floor(now / 180) % 2 ? 0xff2222 : 0x2244ff);
  }
  updateSparks(dt);

  // --- camera
  const p = player.pos;
  const fh = new THREE.Vector3(player.fwd.x, 0, player.fwd.z);
  if (fh.lengthSq() < 1e-4) fh.set(0, 0, 1);
  fh.normalize();

  let want;
  if (camMode === 0) {
    const back = 8.5 + player.speed * 0.12;
    want = new THREE.Vector3(p.x - fh.x * back, p.y + 3.4, p.z - fh.z * back);
    camLook.set(p.x + fh.x * 7, p.y + 0.9, p.z + fh.z * 7);
  } else if (camMode === 1) {
    want = new THREE.Vector3(p.x + fh.x * 0.4, p.y + 1.25, p.z + fh.z * 0.4);
    camLook.set(p.x + fh.x * 20, p.y + 1.1, p.z + fh.z * 20);
  } else {
    want = new THREE.Vector3(p.x - fh.x * 20, p.y + 16, p.z - fh.z * 20);
    camLook.set(p.x, p.y, p.z);
  }
  const lerp = camMode === 1 ? 1 : 1 - Math.pow(0.0015, dt);
  camPos.lerp(want, lerp);
  camera.position.copy(camPos);
  if (shake > 0.001) {
    camera.position.x += (Math.random() - 0.5) * shake * 0.8;
    camera.position.y += (Math.random() - 0.5) * shake * 0.8;
    camera.position.z += (Math.random() - 0.5) * shake * 0.8;
  }
  camera.lookAt(camLook);

  sun.position.set(p.x + 55, 95, p.z + 35);
  sun.target.position.set(p.x, 0, p.z);
  sun.target.updateMatrixWorld();

  // --- audio
  let slip = 0;
  for (let i = 0; i < 4; i++) slip = Math.max(slip, player.wheelSlip[i]);
  sound.engine(player.rpm, player.controls.throttle, !player.disabled);
  sound.squeal(player.speed > 3 ? slip : 0);

  // --- escalation
  if (running && !gameOver && elapsed > nextSpawn) {
    deployChaser();
    nextSpawn += ESCALATE_EVERY;
  }

  // --- HUD
  const dmg = player.damage;
  el.mph.textContent = Math.round(player.speed * 2.2369);
  el.gear.textContent = player.disabled ? 'ENGINE DEAD'
    : player.controls.handbrake ? 'HANDBRAKE'
    : player.controls.throttle < 0 ? 'REVERSE / BRAKE' : 'ENGINE OK';
  el.time.textContent = elapsed.toFixed(1) + 's';
  const live = chasers.filter((x) => x.deployed && !x.v.disabled).length;
  el.cops.textContent = String(live);
  let nearest = Infinity;
  for (const ch of chasers) {
    if (!ch.deployed || ch.v.disabled) continue;
    nearest = Math.min(nearest, Math.hypot(ch.v.pos.x - p.x, ch.v.pos.z - p.z));
  }
  el.near.textContent = nearest === Infinity ? '—' : Math.round(nearest) + 'm';
  el.dmg.style.width = (dmg * 100).toFixed(0) + '%';
  el.dmg.style.background = dmg > 0.6 ? '#ef4444' : dmg > 0.3 ? '#facc15' : '#4ade80';
  el.dmgpct.textContent = Math.round(dmg * 100) + '%';

  if (!gameOver && player.disabled && running) {
    gameOver = true;
    bestTime = Math.max(bestTime, elapsed);
    el.ovScore.textContent = `survived ${elapsed.toFixed(1)}s  ·  best ${bestTime.toFixed(1)}s`;
    el.overlay.classList.add('show');
  }

  renderer.render(scene, camera);
}

requestAnimationFrame(frame);

// Debug handle: lets the sim be stepped and inspected without the render loop.
window.CRUMPLE = {
  world, player, chasers, scene, camera,
  simulate(seconds, dt = 1 / 60) {
    const steps = Math.round(seconds / dt);
    for (let i = 0; i < steps; i++) {
      for (const ch of chasers) if (ch.deployed) ch.ai.think(world, player, dt);
      world.refreshBroadphase();
      world.step(dt);
      impacts.length = 0;
    }
  },
  stats() {
    let nan = 0;
    for (const g of world.groups) {
      if (!g.active) continue;
      for (const n of g.nodes) if (!isFinite(n.x + n.y + n.z + n.vx + n.vy + n.vz)) nan++;
    }
    return {
      nan,
      speed: +player.speed.toFixed(2),
      mph: Math.round(player.speed * 2.2369),
      damage: +player.damage.toFixed(3),
      pos: [player.pos.x, player.pos.z].map((v) => +v.toFixed(1)),
      upY: +player.up.y.toFixed(2),
      disabled: player.disabled,
      beamsBroken: player.beams.filter((b) => b.broken).length,
      nodes: world.groups.reduce((a, g) => a + (g.active ? g.nodes.length : 0), 0),
      beams: world.groups.reduce((a, g) => a + (g.active ? g.beams.length : 0), 0),
    };
  },
};
