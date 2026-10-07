/**
 * app.js — Simulation Manager, Renderer & UI Controller
 * Siege Mech Evolution
 *
 * Orchestrates the full game loop: physics stepping, GA evolution,
 * 3-D wireframe rendering, HUD overlays, telemetry charts, and
 * all DOM/UI bindings.
 */

/* ================================================================== */
/*  Simulation State                                                  */
/* ================================================================== */

let camera = new Camera();
let terrain;

/** @type {{ body: MechBody, dna: MechDNA }[]} */
let mechs = [];

/** @type {Projectile[]} */
let projectiles = [];

/** @type {Target[]} */
let targets = [];

let generation = 0;
let genTimer = 0;
let genTimerMax = 15;       // seconds per generation
let isPlaying = true;
let cameraMode = 'free';    // 'free' | 'follow' | 'target'
let trackedMech = null;

/** @type {{ best: number, avg: number }[]} */
let fitnessHistory = [];

/** @type {{ x: number, z: number, hit: boolean }[]} */
let impactScatter = [];

let simTime = 0;

/** Wind vector (world-space) */
let wind = { x: 0, z: 0 };

/** Cached slider / UI values */
let simSpeed = 1;
let popSize = 12;
let mutationRate = 0.1;
let crossoverRate = 0.5;
let locoWeight = 1;
let targetWeight = 1;
let gravityVal = 0.25;
let frictionVal = 0.99;
let selectionMethod = 'tournament';

/* ================================================================== */
/*  DOM References (resolved once on load)                            */
/* ================================================================== */

let canvas, ctx;
let fitnessCanvas, fitnessCtx;
let scatterCanvas, scatterCtx;

function el(id) { return document.getElementById(id); }

/* ================================================================== */
/*  UI Binding                                                        */
/* ================================================================== */

function bindUI() {
  canvas = el('viewport-canvas');
  ctx = canvas.getContext('2d');
  fitnessCanvas = el('fitness-chart');
  fitnessCtx = fitnessCanvas ? fitnessCanvas.getContext('2d') : null;
  scatterCanvas = el('scatter-chart');
  scatterCtx = scatterCanvas ? scatterCanvas.getContext('2d') : null;

  // ---- Sliders ----
  const sliders = [
    'pop-slider', 'speed-slider', 'timer-slider', 'mutation-slider',
    'crossover-slider', 'loco-weight-slider', 'target-weight-slider',
    'gravity-slider', 'wind-x-slider', 'wind-z-slider', 'friction-slider'
  ];
  sliders.forEach(id => {
    const slider = el(id);
    const valSpan = el(id + '-val');
    if (slider) {
      const updateLabel = () => {
        if (!valSpan) return;
        if (id === 'speed-slider') {
          valSpan.textContent = slider.value + '×';
        } else if (id === 'timer-slider') {
          valSpan.textContent = slider.value + 's';
        } else if (
          id === 'mutation-slider' ||
          id === 'crossover-slider' ||
          id === 'loco-weight-slider' ||
          id === 'target-weight-slider' ||
          id === 'friction-slider'
        ) {
          valSpan.textContent = slider.value + '%';
        } else {
          valSpan.textContent = slider.value;
        }
      };
      slider.addEventListener('input', () => {
        updateLabel();
        _readSliders();
      });
      // Initialise display
      updateLabel();
    }
  });
  _readSliders();

  // ---- Buttons ----
  _btn('play-pause-btn', () => {
    isPlaying = !isPlaying;
    const b = el('play-pause-btn');
    if (b) b.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
  });
  _btn('reset-btn', () => {
    generation = 0;
    fitnessHistory = [];
    impactScatter = [];
    initGeneration(true);
  });
  _btn('next-gen-btn', () => { endGeneration(); });

  // Camera mode buttons
  _btn('cam-free-btn', () => { cameraMode = 'free'; _highlightCamBtn('cam-free-btn'); });
  _btn('cam-follow-btn', () => { cameraMode = 'follow'; _highlightCamBtn('cam-follow-btn'); });
  _btn('cam-target-btn', () => { cameraMode = 'target'; _highlightCamBtn('cam-target-btn'); });

  // ---- Tabs ----
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const tabId = btn.getAttribute('data-tab');
      ['tab-simulation', 'tab-ga', 'tab-env'].forEach(tid => {
        const panel = el(tid);
        if (panel) panel.style.display = tid === tabId ? '' : 'none';
      });
    });
  });

  // ---- Selection method ----
  const selEl = el('selection-select');
  if (selEl) {
    selEl.addEventListener('change', () => { selectionMethod = selEl.value; });
  }

  // ---- Canvas mouse interaction ----
  _bindCanvasMouse();
}

function _btn(id, fn) {
  const b = el(id);
  if (b) b.addEventListener('click', fn);
}

function _highlightCamBtn(activeId) {
  ['cam-free-btn', 'cam-follow-btn', 'cam-target-btn'].forEach(id => {
    const b = el(id);
    if (b) b.classList.toggle('active', id === activeId);
  });
}

function _readSliders() {
  popSize       = _sliderInt('pop-slider', 20);
  simSpeed      = _sliderInt('speed-slider', 1);
  genTimerMax   = _sliderFloat('timer-slider', 15);
  mutationRate  = _sliderFloat('mutation-slider', 12) / 100;
  crossoverRate = _sliderFloat('crossover-slider', 80) / 100;
  locoWeight    = _sliderFloat('loco-weight-slider', 50) / 100;
  targetWeight  = _sliderFloat('target-weight-slider', 50) / 100;
  gravityVal    = _sliderFloat('gravity-slider', 10) * 0.025;
  wind.x        = _sliderFloat('wind-x-slider', 0);
  wind.z        = _sliderFloat('wind-z-slider', 0);
  frictionVal   = 1 - (_sliderFloat('friction-slider', 70) / 100) * 0.1;
}

function _sliderFloat(id, fallback) {
  const s = el(id);
  return s ? parseFloat(s.value) : fallback;
}
function _sliderInt(id, fallback) {
  const s = el(id);
  return s ? parseInt(s.value, 10) : fallback;
}

/* ================================================================== */
/*  Canvas Mouse / Wheel                                              */
/* ================================================================== */

function _bindCanvasMouse() {
  let dragging = false;
  let lastMX = 0, lastMY = 0;

  canvas.addEventListener('mousedown', e => {
    dragging = true;
    lastMX = e.clientX;
    lastMY = e.clientY;
  });
  window.addEventListener('mouseup', () => { dragging = false; });
  window.addEventListener('mousemove', e => {
    if (!dragging) return;
    const dx = e.clientX - lastMX;
    const dy = e.clientY - lastMY;
    lastMX = e.clientX;
    lastMY = e.clientY;
    camera.yaw   += dx * 0.005;
    camera.pitch  += dy * 0.005;
    camera.pitch = Math.max(-1.2, Math.min(0.6, camera.pitch));
    camera.updatePosition();
  });
  canvas.addEventListener('wheel', e => {
    e.preventDefault();
    camera.distance += e.deltaY * 0.4;
    camera.distance = Math.max(80, Math.min(800, camera.distance));
    camera.updatePosition();
  }, { passive: false });

  // Double-click to track nearest mech
  canvas.addEventListener('dblclick', e => {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let bestDist = Infinity;
    let bestMech = null;
    for (const m of mechs) {
      const com = m.body.getCenterOfMass();
      const p = camera.project(com.x, com.y, com.z, canvas.width, canvas.height);
      if (!p.visible) continue;
      const d = Math.hypot(p.x - mx, p.y - my);
      if (d < bestDist) { bestDist = d; bestMech = m; }
    }
    trackedMech = bestMech;
    if (bestMech) { cameraMode = 'follow'; _highlightCamBtn('cam-follow-btn'); }
  });
}

/* ================================================================== */
/*  Generation Management                                             */
/* ================================================================== */

/**
 * Initialise a new generation of mechs, targets, terrain.
 * @param {boolean} [forceRandom=false]  If true, ignore existing DNA
 */
function initGeneration(forceRandom = false) {
  _readSliders();

  const tmpl = _getSelect('template-select', 'biped');
  const tgtType = _getSelect('target-select', 'static');
  const terrType = _getSelect('terrain-select', 'flat');

  // Terrain
  terrain = new Terrain(terrType, 800, 400);

  // Targets
  targets = _createTargets(tgtType, terrain);

  // Population
  if (generation === 0 || forceRandom) {
    mechs = [];
    for (let i = 0; i < popSize; i++) {
      const dna = MechGA.createRandom(tmpl);
      const sx = -150 + (i % 6) * 50;
      const sz = -40 + Math.floor(i / 6) * 60;
      const config = dna.decode();
      const terrainHeight = terrain ? terrain.getHeight(sx, sz) : 0;
      const body = new MechBody(tmpl, sx, sz, config, terrainHeight);
      mechs.push({ body, dna });
    }
  }
  // else: mechs already populated by evolveGeneration path

  projectiles = [];
  genTimer = 0;
  simTime = 0;
  trackedMech = mechs.length > 0 ? mechs[0] : null;

  _updateCounters();
}

function _getSelect(id, fallback) {
  const s = el(id);
  return s ? s.value : fallback;
}

/**
 * Create targets based on type selector.
 */
function _createTargets(type, terrain) {
  const arr = [];
  const positions = [
    { x: 200, z: 0 }, { x: 280, z: 80 }, { x: 320, z: -60 },
    { x: 150, z: 120 }, { x: 260, z: -120 }
  ];

  const isDrone = (type === 'drone');
  const isMixed = (type === 'mixed');

  for (let i = 0; i < positions.length; i++) {
    const p = positions[i];
    const ty = isMixed ? (i < 3 ? 'static' : 'drone') : (isDrone ? 'drone' : 'static');
    const gy = terrain.getHeight(p.x, p.z);
    const baseY = ty === 'drone' ? gy + 50 + Math.random() * 30 : gy + 20;
    arr.push(new Target(p.x, baseY, p.z, ty));
  }
  return arr;
}

/**
 * End the current generation: evaluate, record, evolve.
 */
function endGeneration() {
  const fitnesses = mechs.map(m => MechGA.evaluateFitness(m.body, locoWeight, targetWeight));

  // Record history
  const best = Math.max(...fitnesses);
  const avg = fitnesses.reduce((a, b) => a + b, 0) / fitnesses.length;
  fitnessHistory.push({ best, avg });

  // Evolve DNA
  const dnaPool = mechs.map(m => m.dna);
  const selEl = el('selection-select');
  const method = selEl ? selEl.value : 'tournament';
  const nextDNA = MechGA.evolveGeneration(dnaPool, fitnesses, {
    mutationRate,
    mutationStrength: 0.3,
    crossoverRate,
    selectionMethod: method
  });

  generation++;

  // Rebuild mechs from new DNA
  const tmpl = _getSelect('template-select', 'biped');
  const tgtType = _getSelect('target-select', 'static');
  const terrType = _getSelect('terrain-select', 'flat');
  terrain = new Terrain(terrType, 800, 400);
  targets = _createTargets(tgtType, terrain);
  projectiles = [];
  impactScatter = [];
  genTimer = 0;
  simTime = 0;

  mechs = [];
  for (let i = 0; i < nextDNA.length; i++) {
    const dna = nextDNA[i];
    const sx = -150 + (i % 6) * 50;
    const sz = -40 + Math.floor(i / 6) * 60;
    const config = dna.decode();
    const terrainHeight = terrain ? terrain.getHeight(sx, sz) : 0;
    const body = new MechBody(tmpl, sx, sz, config, terrainHeight);
    mechs.push({ body, dna });
  }
  trackedMech = mechs[0] || null;
  _updateCounters();
}

/* ================================================================== */
/*  Simulation Update                                                 */
/* ================================================================== */

function updateSimulation(dt) {
  simTime += dt;
  genTimer += dt;

  const windVec = { x: wind.x * 0.01, z: wind.z * 0.01 };

  // --- Mechs ---
  for (let i = 0; i < mechs.length; i++) {
    const m = mechs[i];
    if (m.body.isDead) continue;

    m.body.update(simTime, gravityVal, frictionVal, windVec, terrain);

    // Brain: pick nearest alive target
    const nearestTarget = _nearestTarget(m.body);
    if (nearestTarget) {
      _runBrain(m, nearestTarget);
    }
  }

  // --- Projectiles ---
  for (let i = projectiles.length - 1; i >= 0; i--) {
    const p = projectiles[i];
    const result = p.update(gravityVal, windVec, terrain, targets);

    if (!p.alive) {
      if (result.impactPos) {
        const hit = result.hit !== null;
        impactScatter.push({ x: result.impactPos.x, z: result.impactPos.z, hit });
        // Attribute to owning mech
        if (p._ownerIdx !== undefined && p._ownerIdx < mechs.length) {
          if (hit) {
            mechs[p._ownerIdx].body.hits++;
          } else {
            mechs[p._ownerIdx].body.misses++;
          }
        }
      }
      projectiles.splice(i, 1);
    }
  }

  // --- Targets ---
  for (let i = 0; i < targets.length; i++) {
    targets[i].update(simTime);
  }

  // --- Generation timer ---
  if (genTimer >= genTimerMax) {
    endGeneration();
  }

  _updateCounters();
}

/**
 * Find the nearest non-destroyed target to a mech.
 */
function _nearestTarget(body) {
  const com = body.getCenterOfMass();
  let best = null, bestD = Infinity;
  for (const t of targets) {
    if (t.isDestroyed) continue;
    const d = Math.hypot(t.x - com.x, t.y - com.y, t.z - com.z);
    if (d < bestD) { bestD = d; best = t; }
  }
  return best;
}

/**
 * Run the turret brain for one mech and optionally fire.
 */
function _runBrain(m, target) {
  const body = m.body;
  const dna = m.dna;
  const tn = body.turretNode;

  // Relative position
  const rx = target.x - tn.x;
  const ry = target.y - tn.y;
  const rz = target.z - tn.z;
  const dist = Math.sqrt(rx * rx + ry * ry + rz * rz) || 1;

  // Target velocity estimate (drone motion)
  const tvx = target.type === 'drone' ? -Math.sin(simTime * target.pathSpeed + target.pathPhase) * target.pathRadius * target.pathSpeed : 0;
  const tvy = 0;
  const tvz = target.type === 'drone' ? Math.cos(simTime * target.pathSpeed + target.pathPhase) * target.pathRadius * target.pathSpeed : 0;

  const inputs = [
    rx / 300,    // normalised relative position
    ry / 300,
    rz / 300,
    dist / 400,
    tvx / 50,
    tvy / 50,
    tvz / 50,
    wind.x / 10,
    wind.z / 10
  ];

  const out = dna.forward(inputs);
  // out: [pitchDelta, yawDelta, fireImpulse, fireTrigger]

  body.turretAnglePitch += out[0] * 0.05;
  body.turretAngleYaw   += out[1] * 0.05;

  // Clamp pitch
  body.turretAnglePitch = Math.max(-0.8, Math.min(0.8, body.turretAnglePitch));

  // Fire if trigger > 0.3
  if (out[3] > 0.3) {
    const proj = body.fire(targets, terrain, wind);
    if (proj) {
      // Scale velocity by fire impulse
      const scale = 0.6 + (out[2] + 1) * 0.5 * 1.4; // map [-1,1] → [0.6, 2.0]
      proj.vx *= scale;
      proj.vy *= scale;
      proj.vz *= scale;
      proj._ownerIdx = mechs.indexOf(m);
      projectiles.push(proj);
    }
  }
}

/* ================================================================== */
/*  UI Counter Updates                                                */
/* ================================================================== */

function _updateCounters() {
  _setText('gen-count', generation);

  const fits = mechs.map(m => MechGA.evaluateFitness(m.body, locoWeight, targetWeight));
  const bestFit = fits.length ? Math.max(...fits).toFixed(1) : '0';
  _setText('best-fitness', bestFit);

  const alive = mechs.filter(m => !m.body.isDead && !m.body.isFallen).length;
  _setText('pop-alive', alive + '/' + mechs.length);

  // Wind badge
  const wb = el('wind-badge');
  if (wb) {
    const mag = Math.sqrt(wind.x * wind.x + wind.z * wind.z);
    wb.textContent = mag < 0.1 ? 'Calm' : mag.toFixed(1);
  }

  // Best mech stats
  if (mechs.length) {
    const bestIdx = fits.indexOf(Math.max(...fits));
    const bm = mechs[bestIdx].body;
    _setText('stat-distance', bm.distanceMoved.toFixed(1));
    _setText('stat-hits', bm.hits);
    _setText('stat-misses', bm.misses);
    const total = bm.hits + bm.misses;
    const acc = total > 0 ? ((bm.hits / total) * 100).toFixed(0) + '%' : '—';
    _setText('stat-accuracy', acc);

    // Trait bars
    _setTraitBar('trait-limb', _avgPhysicalTrait(mechs[bestIdx].dna, 'limb'));
    _setTraitBar('trait-muscle', _avgPhysicalTrait(mechs[bestIdx].dna, 'muscle'));
    _setTraitBar('trait-turret', _avgBrainMagnitude(mechs[bestIdx].dna));
  }

  // Gen timer
  const remaining = Math.max(0, genTimerMax - genTimer);
  _setText('gen-timer-val', remaining.toFixed(1) + 's');
  const fill = el('gen-progress-fill');
  if (fill) fill.style.width = ((genTimer / genTimerMax) * 100) + '%';
}

function _setText(id, val) {
  const e = el(id);
  if (e) e.textContent = val;
}

function _setTraitBar(prefix, value01) {
  const fill = el(prefix + '-fill');
  const val  = el(prefix + '-val');
  if (fill) fill.style.width = (value01 * 100) + '%';
  if (val)  val.textContent = (value01 * 100).toFixed(0) + '%';
}

function _avgPhysicalTrait(dna, type) {
  const genes = dna.physicalGenes;
  if (!genes.length) return 0.5;
  const layout = _geneLayout(dna.template);
  if (type === 'limb') {
    let sum = 0;
    for (let i = 0; i < layout.limbCount && i < genes.length; i++) sum += genes[i];
    return sum / layout.limbCount;
  }
  // muscle
  let sum = 0;
  const start = layout.limbCount;
  const count = layout.muscleCount * 3;
  for (let i = start; i < start + count && i < genes.length; i++) sum += genes[i];
  return count > 0 ? sum / count : 0.5;
}

function _avgBrainMagnitude(dna) {
  if (!dna.brainGenes.length) return 0;
  let sum = 0;
  for (let i = 0; i < dna.brainGenes.length; i++) sum += Math.abs(dna.brainGenes[i]);
  return Math.min(1, sum / dna.brainGenes.length);
}

/* ================================================================== */
/*  Rendering                                                         */
/* ================================================================== */

function render() {
  const cw = canvas.width;
  const ch = canvas.height;

  // --- Camera follow ---
  if (cameraMode === 'follow' && trackedMech) {
    const com = trackedMech.body.getCenterOfMass();
    camera.updateFollow(com.x, com.y + 10, com.z);
  } else if (cameraMode === 'target' && targets.length) {
    const t = targets.find(t => !t.isDestroyed) || targets[0];
    camera.updateFollow(t.x, t.y, t.z);
  } else {
    camera.updatePosition();
  }

  // --- Clear ---
  ctx.fillStyle = '#0A0E17';
  ctx.fillRect(0, 0, cw, ch);

  // --- Collect draw calls for depth sorting ---
  const drawCalls = [];

  // Terrain
  _collectTerrainDrawCalls(drawCalls, cw, ch);

  // Targets
  _collectTargetDrawCalls(drawCalls, cw, ch);

  // Mechs
  const fits = mechs.map(m => MechGA.evaluateFitness(m.body, locoWeight, targetWeight));
  const bestIdx = fits.length ? fits.indexOf(Math.max(...fits)) : -1;
  for (let i = 0; i < mechs.length; i++) {
    _collectMechDrawCalls(drawCalls, mechs[i], cw, ch, i === bestIdx);
  }

  // Projectiles
  _collectProjectileDrawCalls(drawCalls, cw, ch);

  // Execute depth-sorted draws
  depthSort(drawCalls);

  // --- HUD overlay ---
  _drawHUD(cw, ch);

  // --- Tracking reticle ---
  if (trackedMech) {
    const com = trackedMech.body.getCenterOfMass();
    const p = camera.project(com.x, com.y + 10, com.z, cw, ch);
    if (p.visible) {
      ctx.strokeStyle = '#00FF88';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 22, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
      // Cross-hairs
      ctx.beginPath();
      ctx.moveTo(p.x - 28, p.y); ctx.lineTo(p.x - 16, p.y);
      ctx.moveTo(p.x + 16, p.y); ctx.lineTo(p.x + 28, p.y);
      ctx.moveTo(p.x, p.y - 28); ctx.lineTo(p.x, p.y - 16);
      ctx.moveTo(p.x, p.y + 16); ctx.lineTo(p.x, p.y + 28);
      ctx.stroke();
    }
  }

  // --- Charts ---
  if (fitnessCtx) _drawFitnessChart();
  if (scatterCtx) _drawScatterChart();
}

/* -------- Terrain rendering -------- */

function _collectTerrainDrawCalls(calls, cw, ch) {
  const lines = terrain.getGridLines();
  for (let i = 0; i < lines.length; i++) {
    const l = lines[i];
    const proj = camera.projectLine(l.x1, l.y1, l.z1, l.x2, l.y2, l.z2, cw, ch);
    if (!proj.a.visible && !proj.b.visible) continue;
    const depth = proj.depth;
    const a = proj.a, b = proj.b;
    calls.push({
      depth,
      drawFn: () => {
        const fade = Math.max(0.08, Math.min(0.5, 200 / depth));
        ctx.strokeStyle = terrain.type === 'craters'
          ? `rgba(100,200,180,${fade})`
          : `rgba(50,220,140,${fade})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();
      }
    });
  }
}

/* -------- Target rendering -------- */

function _collectTargetDrawCalls(calls, cw, ch) {
  for (const t of targets) {
    if (t.isDestroyed) continue;
    const p = camera.project(t.x, t.y, t.z, cw, ch);
    if (!p.visible) continue;
    const depth = p.z;
    const size = Math.max(4, (t.radius * 600) / depth);
    calls.push({
      depth,
      drawFn: () => {
        // Glow
        const grad = ctx.createRadialGradient(p.x, p.y, size * 0.2, p.x, p.y, size * 2);
        grad.addColorStop(0, t.color + 'CC');
        grad.addColorStop(1, t.color + '00');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(p.x, p.y, size * 2, 0, Math.PI * 2);
        ctx.fill();

        // Core shape
        ctx.fillStyle = t.color;
        if (t.type === 'drone') {
          // Diamond
          ctx.beginPath();
          ctx.moveTo(p.x, p.y - size);
          ctx.lineTo(p.x + size, p.y);
          ctx.lineTo(p.x, p.y + size);
          ctx.lineTo(p.x - size, p.y);
          ctx.closePath();
          ctx.fill();
        } else {
          // Square tower
          ctx.fillRect(p.x - size * 0.6, p.y - size, size * 1.2, size * 2);
        }

        // Label
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '9px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(t.type === 'drone' ? 'DRN' : 'TGT', p.x, p.y - size - 4);
      }
    });
  }
}

/* -------- Mech rendering -------- */

function _collectMechDrawCalls(calls, m, cw, ch, isBest) {
  const body = m.body;
  if (body.isDead) return;

  const com = body.getCenterOfMass();
  const comP = camera.project(com.x, com.y, com.z, cw, ch);
  const mechDepth = comP.z;

  // Constraints (bones + muscles)
  for (const c of body.constraints) {
    const proj = camera.projectLine(c.a.x, c.a.y, c.a.z, c.b.x, c.b.y, c.b.z, cw, ch);
    if (!proj.a.visible && !proj.b.visible) continue;

    const pa = proj.a, pb = proj.b, d = proj.depth;
    const isMuscle = c.type === 'muscle';
    const isMinBone = c.type === 'min-bone';

    calls.push({
      depth: d,
      drawFn: () => {
        if (body.isFallen) {
          ctx.strokeStyle = '#2d3748';
          ctx.lineWidth = isMuscle ? 1.5 : (isMinBone ? 1 : 2);
        } else if (isMuscle) {
          // Color by contraction state
          const curLen = Math.sqrt(
            (c.b.x - c.a.x) ** 2 + (c.b.y - c.a.y) ** 2 + (c.b.z - c.a.z) ** 2
          );
          const ratio = curLen / c.restLength;
          // < 1 = contracted (green), > 1 = extended (red)
          const t = Math.max(0, Math.min(1, (ratio - 0.7) / 0.6));
          ctx.strokeStyle = lerpColor('#33FF66', '#FF3333', t);
          ctx.lineWidth = 2;
        } else if (isMinBone) {
          ctx.strokeStyle = '#445566';
          ctx.lineWidth = 1;
          ctx.setLineDash([2, 4]);
        } else {
          ctx.strokeStyle = isBest ? '#FFD700' : '#88AACC';
          ctx.lineWidth = 3;
        }
        ctx.beginPath();
        ctx.moveTo(pa.x, pa.y);
        ctx.lineTo(pb.x, pb.y);
        ctx.stroke();
        if (isMinBone) {
          ctx.setLineDash([]);
        }
      }
    });
  }

  // Nodes
  for (const n of body.nodes) {
    const p = camera.project(n.x, n.y, n.z, cw, ch);
    if (!p.visible) continue;
    const d = p.z;
    calls.push({
      depth: d - 0.1, // slightly in front of constraints
      drawFn: () => {
        const r = Math.max(2, (n.radius * 600) / d);
        if (body.isFallen) {
          ctx.fillStyle = '#4a5568';
        } else {
          ctx.fillStyle = n === body.turretNode ? '#FF4488' : (isBest ? '#FFDD44' : '#AACCEE');
        }
        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }

  // Turret barrel
  if (body.turretNode) {
    const tn = body.turretNode;
    const barrelLen = 18;
    const cosP = Math.cos(body.turretAnglePitch);
    const sinP = Math.sin(body.turretAnglePitch);
    const cosY = Math.cos(body.turretAngleYaw);
    const sinY = Math.sin(body.turretAngleYaw);
    const bx = tn.x + cosP * sinY * barrelLen;
    const by = tn.y + sinP * barrelLen;
    const bz = tn.z + cosP * cosY * barrelLen;

    const proj = camera.projectLine(tn.x, tn.y, tn.z, bx, by, bz, cw, ch);
    if (proj.a.visible || proj.b.visible) {
      calls.push({
        depth: proj.depth - 0.2,
        drawFn: () => {
          ctx.strokeStyle = body.isFallen ? '#2d3748' : '#FF6688';
          ctx.lineWidth = 3;
          ctx.beginPath();
          ctx.moveTo(proj.a.x, proj.a.y);
          ctx.lineTo(proj.b.x, proj.b.y);
          ctx.stroke();
          // Muzzle dot
          ctx.fillStyle = body.isFallen ? '#4a5568' : '#FF88AA';
          ctx.beginPath();
          ctx.arc(proj.b.x, proj.b.y, 3, 0, Math.PI * 2);
          ctx.fill();
        }
      });
    }
  }

  // Best mech gold outline halo
  if (isBest && comP.visible) {
    calls.push({
      depth: mechDepth - 0.3,
      drawFn: () => {
        ctx.strokeStyle = 'rgba(255,215,0,0.35)';
        ctx.lineWidth = 2;
        ctx.setLineDash([6, 3]);
        ctx.beginPath();
        ctx.arc(comP.x, comP.y, 30, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#FFD700';
        ctx.font = 'bold 10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText('★ BEST', comP.x, comP.y - 35);
      }
    });
  }
}

/* -------- Projectile rendering -------- */

function _collectProjectileDrawCalls(calls, cw, ch) {
  for (const p of projectiles) {
    if (!p.alive) continue;
    if (p.trail.length < 2) continue;

    // Project trail head for depth
    const headP = camera.project(p.x, p.y, p.z, cw, ch);
    if (!headP.visible) continue;

    const depth = headP.z;
    // Capture trail projection
    const projTrail = p.trail.map(tp => camera.project(tp.x, tp.y, tp.z, cw, ch));

    calls.push({
      depth,
      drawFn: () => {
        // Trail gradient
        for (let i = 1; i < projTrail.length; i++) {
          const a = projTrail[i - 1];
          const b = projTrail[i];
          if (!a.visible || !b.visible) continue;
          const alpha = (i / projTrail.length) * 0.8;
          ctx.strokeStyle = `rgba(255,180,50,${alpha})`;
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.stroke();
        }

        // Head glow
        const grad = ctx.createRadialGradient(headP.x, headP.y, 1, headP.x, headP.y, 8);
        grad.addColorStop(0, 'rgba(255,255,200,0.9)');
        grad.addColorStop(1, 'rgba(255,140,20,0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(headP.x, headP.y, 8, 0, Math.PI * 2);
        ctx.fill();
      }
    });
  }
}

/* -------- HUD overlay -------- */

function _drawHUD(cw, ch) {
  ctx.fillStyle = 'rgba(255,255,255,0.7)';
  ctx.font = '12px monospace';
  ctx.textAlign = 'left';
  ctx.fillText(`Gen ${generation}`, 12, 20);

  const remaining = Math.max(0, genTimerMax - genTimer);
  ctx.fillText(`Time: ${remaining.toFixed(1)}s`, 12, 36);
  ctx.fillText(`Mechs: ${mechs.filter(m => !m.body.isDead).length}/${mechs.length}`, 12, 52);

  // Projectile count
  ctx.fillText(`Shells: ${projectiles.length}`, 12, 68);

  // Wind arrow (top-right)
  const arrowCx = cw - 50;
  const arrowCy = 40;
  const wMag = Math.sqrt(wind.x * wind.x + wind.z * wind.z);
  if (wMag > 0.05) {
    const angle = Math.atan2(wind.z, wind.x);
    const len = Math.min(20, wMag * 3);
    ctx.strokeStyle = '#66CCFF';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(arrowCx, arrowCy);
    ctx.lineTo(arrowCx + Math.cos(angle) * len, arrowCy + Math.sin(angle) * len);
    ctx.stroke();
    // Arrowhead
    const ax = arrowCx + Math.cos(angle) * len;
    const ay = arrowCy + Math.sin(angle) * len;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - Math.cos(angle - 0.4) * 6, ay - Math.sin(angle - 0.4) * 6);
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax - Math.cos(angle + 0.4) * 6, ay - Math.sin(angle + 0.4) * 6);
    ctx.stroke();
  }
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.textAlign = 'center';
  ctx.font = '9px monospace';
  ctx.fillText('WIND', arrowCx, arrowCy + 30);
}

/* ================================================================== */
/*  Telemetry Charts                                                  */
/* ================================================================== */

function _drawFitnessChart() {
  const c = fitnessCanvas;
  const cx = fitnessCtx;
  const w = c.width, h = c.height;
  cx.fillStyle = '#111520';
  cx.fillRect(0, 0, w, h);

  if (fitnessHistory.length < 2) {
    cx.fillStyle = '#556';
    cx.font = '11px monospace';
    cx.textAlign = 'center';
    cx.fillText('Waiting for data...', w / 2, h / 2);
    return;
  }

  const maxVal = Math.max(1, ...fitnessHistory.map(f => f.best));
  const minVal = Math.min(0, ...fitnessHistory.map(f => f.avg));
  const range = maxVal - minVal || 1;
  const pad = 30;
  const plotW = w - pad * 2;
  const plotH = h - pad * 2;

  // Axes
  cx.strokeStyle = '#334';
  cx.lineWidth = 1;
  cx.beginPath();
  cx.moveTo(pad, pad);
  cx.lineTo(pad, h - pad);
  cx.lineTo(w - pad, h - pad);
  cx.stroke();

  // Labels
  cx.fillStyle = '#778';
  cx.font = '9px monospace';
  cx.textAlign = 'right';
  cx.fillText(maxVal.toFixed(0), pad - 4, pad + 4);
  cx.fillText(minVal.toFixed(0), pad - 4, h - pad);
  cx.textAlign = 'center';
  cx.fillText('Generation', w / 2, h - 4);

  const n = fitnessHistory.length;
  const dx = plotW / Math.max(1, n - 1);

  // Best line
  cx.strokeStyle = '#00FF88';
  cx.lineWidth = 2;
  cx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad + i * dx;
    const y = h - pad - ((fitnessHistory[i].best - minVal) / range) * plotH;
    if (i === 0) cx.moveTo(x, y); else cx.lineTo(x, y);
  }
  cx.stroke();

  // Avg line
  cx.strokeStyle = '#4488CC';
  cx.lineWidth = 1.5;
  cx.beginPath();
  for (let i = 0; i < n; i++) {
    const x = pad + i * dx;
    const y = h - pad - ((fitnessHistory[i].avg - minVal) / range) * plotH;
    if (i === 0) cx.moveTo(x, y); else cx.lineTo(x, y);
  }
  cx.stroke();

  // Legend
  cx.fillStyle = '#00FF88';
  cx.font = '9px monospace';
  cx.textAlign = 'left';
  cx.fillText('● Best', pad + 4, pad - 4);
  cx.fillStyle = '#4488CC';
  cx.fillText('● Avg', pad + 60, pad - 4);
}

function _drawScatterChart() {
  const c = scatterCanvas;
  const cx = scatterCtx;
  const w = c.width, h = c.height;
  cx.fillStyle = '#111520';
  cx.fillRect(0, 0, w, h);

  if (!impactScatter.length) {
    cx.fillStyle = '#556';
    cx.font = '11px monospace';
    cx.textAlign = 'center';
    cx.fillText('No impacts yet', w / 2, h / 2);
    return;
  }

  // Center on first target
  const refTarget = targets[0] || { x: 200, z: 0 };
  const scale = 1.2;
  const cx0 = w / 2;
  const cy0 = h / 2;

  // Target cross-hair
  cx.strokeStyle = '#444';
  cx.lineWidth = 1;
  cx.beginPath();
  cx.moveTo(cx0 - 30, cy0); cx.lineTo(cx0 + 30, cy0);
  cx.moveTo(cx0, cy0 - 30); cx.lineTo(cx0, cy0 + 30);
  cx.stroke();
  cx.strokeStyle = '#555';
  cx.beginPath();
  cx.arc(cx0, cy0, 15 * scale, 0, Math.PI * 2);
  cx.stroke();

  // Impact dots
  for (const imp of impactScatter) {
    const dx = (imp.x - refTarget.x) * scale;
    const dz = (imp.z - refTarget.z) * scale;
    const sx = cx0 + dx;
    const sy = cy0 + dz;
    if (sx < 0 || sx > w || sy < 0 || sy > h) continue;

    cx.fillStyle = imp.hit ? '#00FF88' : '#FF4444';
    cx.globalAlpha = 0.7;
    cx.beginPath();
    cx.arc(sx, sy, 3, 0, Math.PI * 2);
    cx.fill();
  }
  cx.globalAlpha = 1;

  // Legend
  cx.font = '9px monospace';
  cx.textAlign = 'left';
  cx.fillStyle = '#00FF88';
  cx.fillText('● Hit', 6, 12);
  cx.fillStyle = '#FF4444';
  cx.fillText('● Miss', 46, 12);
}

/* ================================================================== */
/*  Canvas Resize & Main Loop                                         */
/* ================================================================== */

function resizeCanvases() {
  const list = [
    { c: canvas },
    { c: fitnessCanvas },
    { c: scatterCanvas }
  ];
  list.forEach(item => {
    if (!item.c) return;
    const rect = item.c.getBoundingClientRect();
    const w = Math.round(rect.width) || 300;
    const h = Math.round(rect.height) || 150;
    if (item.c.width !== w || item.c.height !== h) {
      item.c.width = w;
      item.c.height = h;
    }
  });
}

window.addEventListener('resize', resizeCanvases);

window.addEventListener('load', () => {
  bindUI();
  resizeCanvases();
  initGeneration(true);

  function loop() {
    if (isPlaying) {
      const speed = _sliderInt('speed-slider', 1);
      for (let i = 0; i < speed; i++) {
        updateSimulation(1 / 60);
      }
    }
    render();
    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
});
