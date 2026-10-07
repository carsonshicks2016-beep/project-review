// VECTOR DUEL — replay renderer for self-play PPO duels.
// Loads recorded matches (manifest.json + match_XX.json) and renders them with
// interpolation, tracer tails, bloom, particles and screen-shake.

const THEME = {
  bg0: '#05060e', bg1: '#0b1024',
  p: ['#37e8ff', '#ff3d7f'],
  rgb: [[55, 232, 255], [255, 61, 127]],
  grid: 'rgba(90,120,190,0.06)',
  edge: 'rgba(130,165,240,0.55)',
  text: '#e6ecff',
};
const NAMES = ['ARC', 'HEX'];
const rgba = (i, a) => `rgba(${THEME.rgb[i][0]},${THEME.rgb[i][1]},${THEME.rgb[i][2]},${a})`;
const lerp = (a, b, t) => a + (b - a) * t;
function alerp(a, b, t) {                 // angle lerp (shortest arc)
  let d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI;
  return a + d * t;
}

// ---------------------------------------------------------------- DOM
const cv = document.getElementById('scene');
const ctx = cv.getContext('2d');
const stage = document.getElementById('stage');
const el = (id) => document.getElementById(id);

// ---------------------------------------------------------------- state
let manifest = null, mi = 0;
let match = null, frames = [], meta = null, nframe = 0;
let playhead = 0, playing = true, speed = 1, auto = true, lastEv = -1;
let shake = 0, verdictShown = false;
let trails = [[], []], parts = [], rings = [], flashes = [];
let prevBullets = [];
const SPEEDS = [0.5, 1, 2];

// world transform
let scale = 1, ox = 0, oy = 0, dpr = 1;
function fit() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = stage.clientWidth, h = stage.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + 'px'; cv.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!meta) return;
  const pad = 46;
  scale = Math.min((w - pad * 2) / meta.arena_w, (h - pad * 2) / meta.arena_h);
  ox = (w - meta.arena_w * scale) / 2;
  oy = (h - meta.arena_h * scale) / 2;
}
const sx = (x) => ox + x * scale;
const sy = (y) => oy + y * scale;
window.addEventListener('resize', fit);

// ---------------------------------------------------------------- loading
async function boot() {
  const m = await fetch('replays/manifest.json?t=' + Date.now()).then(r => r.json());
  manifest = m;
  meta = m.meta;
  await loadMatch(0);
  requestAnimationFrame(loop);
}
async function loadMatch(i) {
  mi = (i + manifest.matches.length) % manifest.matches.length;
  const entry = manifest.matches[mi];
  match = await fetch('replays/' + entry.file + '?t=' + Date.now()).then(r => r.json());
  meta = match.meta; frames = match.frames; nframe = frames.length;
  playhead = 0; lastEv = -1; playing = true; verdictShown = false;
  trails = [[], []]; parts = []; rings = []; flashes = []; prevBullets = [];
  el('verdict').classList.remove('show');
  el('n0').textContent = NAMES[0]; el('n1').textContent = NAMES[1];
  el('matchlabel').textContent = `MATCH ${mi + 1} / ${manifest.matches.length}`;
  const w = match.winner;
  el('stat').innerHTML = `HITS <b>${entry.hits}</b> · <b>${w < 0 ? 'DRAW' : NAMES[w] + ' WON'}</b>`;
  el('play').textContent = '❚❚';
  fit();
}

// ---------------------------------------------------------------- events
function frameEvents(f) {
  if (f < 1 || f >= nframe) return;
  const cur = frames[f], prev = frames[f - 1];
  for (let p = 0; p < 2; p++) {
    const a = cur.a[p], pa = prev.a[p];
    const fl = a[4];
    if (fl & 4) {                                   // fired
      flashes.push({ x: a[0], y: a[1], ang: a[2], c: p, life: 0.12, max: 0.12 });
    }
    if (a[3] < pa[3]) {                             // took damage
      spawnHit(a[0], a[1], p);
      shake = Math.min(shake + 7, 16);
      rings.push({ x: a[0], y: a[1], r: 0.6, maxr: 3.4, life: 0.5, max: 0.5, c: p });
      if (a[3] <= 0 && pa[3] > 0) {                 // killed
        spawnDeath(a[0], a[1], p);
        shake = 26;
        showVerdict(1 - p);
      }
    }
  }
}
function showVerdict(winner) {
  if (verdictShown) return;
  verdictShown = true;
  const v = el('verdict');
  v.textContent = winner < 0 ? 'DRAW' : NAMES[winner] + ' WINS';
  v.style.color = winner < 0 ? THEME.text : THEME.p[winner];
  v.style.textShadow = winner < 0 ? '0 0 30px #889' : `0 0 40px ${THEME.p[winner]}`;
  v.classList.add('show');
}

// ---------------------------------------------------------------- particles
function spawnHit(x, y, p) {
  for (let i = 0; i < 16; i++) {
    const a = Math.random() * Math.PI * 2, s = 4 + Math.random() * 11;
    parts.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s,
      life: 0.4 + Math.random() * 0.3, max: 0.7, size: 1 + Math.random() * 2, c: p });
  }
}
function spawnDeath(x, y, p) {
  for (let i = 0; i < 60; i++) {
    const a = Math.random() * Math.PI * 2, s = 6 + Math.random() * 26;
    parts.push({ x, y, vx: Math.cos(a) * s, vy: Math.sin(a) * s,
      life: 0.7 + Math.random() * 0.8, max: 1.5, size: 1.5 + Math.random() * 3, c: p });
  }
  rings.push({ x, y, r: 0.5, maxr: 9, life: 0.9, max: 0.9, c: p });
}
function updateParts(dt) {
  for (const q of parts) { q.x += q.vx * dt; q.y += q.vy * dt; q.vx *= 0.92; q.vy *= 0.92; q.life -= dt; }
  parts = parts.filter(q => q.life > 0);
  for (const r of rings) { r.r += (r.maxr - r.r) * dt * 6; r.life -= dt; }
  rings = rings.filter(r => r.life > 0);
  for (const f of flashes) f.life -= dt;
  flashes = flashes.filter(f => f.life > 0);
}

// ---------------------------------------------------------------- bullets
function interpBullets(A, B, t) {
  const out = [], usedB = new Array(B.length).fill(false);
  const TH = (meta.max_speed ? 2.2 : 2.2);
  for (const ba of A) {
    let best = -1, bd = TH * TH;
    for (let j = 0; j < B.length; j++) {
      if (usedB[j] || B[j][2] !== ba[2]) continue;
      const dx = B[j][0] - ba[0], dy = B[j][1] - ba[1], d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = j; }
    }
    if (best >= 0) {
      const bb = B[best]; usedB[best] = true;
      out.push({ x: lerp(ba[0], bb[0], t), y: lerp(ba[1], bb[1], t),
        vx: bb[0] - ba[0], vy: bb[1] - ba[1], c: ba[2] });
    }
  }
  for (let j = 0; j < B.length; j++)
    if (!usedB[j]) out.push({ x: B[j][0], y: B[j][1], vx: 0, vy: 0, c: B[j][2] });
  return out;
}

// ---------------------------------------------------------------- render
function loop(ts) {
  const dt = Math.min((ts - (loop.t || ts)) / 1000, 0.05); loop.t = ts;

  if (playing && nframe > 1) {
    playhead += dt * 30 * speed;
    const tgt = Math.floor(playhead);
    while (lastEv < tgt) { lastEv++; frameEvents(lastEv); }
    if (playhead >= nframe - 1) {
      playhead = nframe - 1; playing = false;
      if (!verdictShown) showVerdict(match.winner);
      if (auto) setTimeout(() => { if (auto) loadMatch(mi + 1); }, 1700);
    }
  }
  updateParts(dt);
  shake *= Math.pow(0.0025, dt);

  const f0 = Math.min(Math.floor(playhead), nframe - 1);
  const f1 = Math.min(f0 + 1, nframe - 1);
  const t = playhead - f0;
  const A = frames[f0], B = frames[f1];

  draw(A, B, t, dt);
  syncHUD(A, B, t);
  el('scrub').value = Math.round((playhead / Math.max(nframe - 1, 1)) * 1000);
  requestAnimationFrame(loop);
}

function draw(A, B, t, dt) {
  const W = stage.clientWidth, H = stage.clientHeight;
  // background
  const g = ctx.createRadialGradient(W / 2, H * 0.46, 60, W / 2, H / 2, Math.max(W, H) * 0.75);
  g.addColorStop(0, THEME.bg1); g.addColorStop(1, THEME.bg0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

  ctx.save();
  if (shake > 0.2) ctx.translate((Math.random() - .5) * shake, (Math.random() - .5) * shake);

  drawArena();

  // interpolated agents
  const ag = [0, 1].map(p => ({
    x: lerp(A.a[p][0], B.a[p][0], t),
    y: lerp(A.a[p][1], B.a[p][1], t),
    ang: alerp(A.a[p][2], B.a[p][2], t),
    hp: A.a[p][3], fl: A.a[p][4],
  }));
  for (let p = 0; p < 2; p++) {
    trails[p].push([ag[p].x, ag[p].y]);
    if (trails[p].length > 16) trails[p].shift();
  }

  // bullets (under agents)
  const bul = interpBullets(A.b, B.b, t);
  ctx.globalCompositeOperation = 'lighter';
  for (const b of bul) drawBullet(b);
  ctx.globalCompositeOperation = 'source-over';

  for (let p = 0; p < 2; p++) drawTrail(p);
  for (let p = 0; p < 2; p++) drawAgent(ag[p], p);

  // effects
  ctx.globalCompositeOperation = 'lighter';
  for (const f of flashes) drawFlash(f);
  for (const r of rings) drawRing(r);
  for (const q of parts) drawPart(q);
  ctx.globalCompositeOperation = 'source-over';

  ctx.restore();
  prevBullets = bul;
}

function drawArena() {
  const x = sx(0), y = sy(0), w = meta.arena_w * scale, h = meta.arena_h * scale;
  // floor grid
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.strokeStyle = THEME.grid; ctx.lineWidth = 1;
  const step = 1.5 * scale;
  const drift = (performance.now() * 0.006) % step;
  ctx.beginPath();
  for (let gx = x + drift; gx < x + w; gx += step) { ctx.moveTo(gx, y); ctx.lineTo(gx, y + h); }
  for (let gy = y + drift; gy < y + h; gy += step) { ctx.moveTo(x, gy); ctx.lineTo(x + w, gy); }
  ctx.stroke();
  // center seam
  ctx.strokeStyle = 'rgba(120,150,230,0.10)';
  ctx.beginPath(); ctx.moveTo(x + w / 2, y); ctx.lineTo(x + w / 2, y + h); ctx.stroke();
  ctx.beginPath(); ctx.arc(x + w / 2, y + h / 2, 2.2 * scale, 0, 7); ctx.stroke();
  ctx.restore();
  // glowing border
  ctx.save();
  ctx.strokeStyle = THEME.edge; ctx.lineWidth = 2;
  ctx.shadowColor = 'rgba(120,170,255,0.8)'; ctx.shadowBlur = 18;
  roundRect(x, y, w, h, 10); ctx.stroke();
  ctx.restore();
}

function drawTrail(p) {
  const tr = trails[p]; if (tr.length < 2) return;
  ctx.globalCompositeOperation = 'lighter';
  for (let i = 1; i < tr.length; i++) {
    const a = (i / tr.length) * 0.5;
    ctx.strokeStyle = rgba(p, a * (tr.length === 16 ? 1 : 1));
    ctx.lineWidth = (i / tr.length) * 3;
    ctx.beginPath(); ctx.moveTo(sx(tr[i - 1][0]), sy(tr[i - 1][1]));
    ctx.lineTo(sx(tr[i][0]), sy(tr[i][1])); ctx.stroke();
  }
  ctx.globalCompositeOperation = 'source-over';
}

function drawAgent(a, p) {
  const cx = sx(a.x), cy = sy(a.y), r = meta.agent_r * scale;
  const dashing = a.fl & 2, inv = a.fl & 1;
  ctx.save();
  ctx.translate(cx, cy);

  // aim laser sight
  ctx.save(); ctx.rotate(a.ang);
  ctx.globalCompositeOperation = 'lighter';
  const lg = ctx.createLinearGradient(r, 0, r * 9, 0);
  lg.addColorStop(0, rgba(p, 0.5)); lg.addColorStop(1, rgba(p, 0));
  ctx.strokeStyle = lg; ctx.lineWidth = 1.4;
  ctx.beginPath(); ctx.moveTo(r * 1.3, 0); ctx.lineTo(r * 9, 0); ctx.stroke();
  ctx.restore();

  // health ring
  ctx.beginPath(); ctx.arc(0, 0, r * 1.9, 0, Math.PI * 2);
  ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 3; ctx.stroke();
  const frac = Math.max(a.hp, 0) / meta.health;
  ctx.beginPath();
  ctx.arc(0, 0, r * 1.9, -Math.PI / 2, -Math.PI / 2 + frac * Math.PI * 2);
  ctx.strokeStyle = THEME.p[p]; ctx.lineWidth = 3;
  ctx.shadowColor = THEME.p[p]; ctx.shadowBlur = 10; ctx.stroke();
  ctx.shadowBlur = 0;

  // body (arrowhead)
  ctx.rotate(a.ang);
  if (dashing) {                                  // dash streak
    ctx.globalCompositeOperation = 'lighter';
    ctx.fillStyle = rgba(p, 0.25);
    ctx.beginPath(); ctx.ellipse(-r * 2, 0, r * 2.6, r * 0.7, 0, 0, 7); ctx.fill();
    ctx.globalCompositeOperation = 'source-over';
  }
  const grd = ctx.createLinearGradient(-r, 0, r * 1.6, 0);
  grd.addColorStop(0, rgba(p, 0.25)); grd.addColorStop(1, THEME.p[p]);
  ctx.fillStyle = grd;
  ctx.strokeStyle = THEME.p[p]; ctx.lineWidth = 1.5;
  ctx.shadowColor = THEME.p[p]; ctx.shadowBlur = dashing ? 22 : 12;
  ctx.beginPath();
  ctx.moveTo(r * 1.6, 0);
  ctx.lineTo(-r * 0.95, r);
  ctx.lineTo(-r * 0.45, 0);
  ctx.lineTo(-r * 0.95, -r);
  ctx.closePath(); ctx.fill(); ctx.stroke();
  // bright core
  ctx.shadowBlur = 14;
  ctx.fillStyle = '#fff';
  ctx.beginPath(); ctx.arc(0, 0, r * 0.32, 0, 7); ctx.fill();
  ctx.restore();

  // invuln shield
  if (inv) {
    ctx.save(); ctx.translate(cx, cy);
    ctx.globalCompositeOperation = 'lighter';
    const a2 = 0.35 + 0.25 * Math.sin(performance.now() * 0.04);
    ctx.strokeStyle = rgba(p, a2); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(0, 0, r * 2.5, 0, 7); ctx.stroke();
    ctx.restore();
  }
}

function drawBullet(b) {
  const x = sx(b.x), y = sy(b.y);
  const tx = sx(b.x - b.vx), ty = sy(b.y - b.vy);
  const grd = ctx.createLinearGradient(tx, ty, x, y);
  grd.addColorStop(0, rgba(b.c, 0)); grd.addColorStop(1, rgba(b.c, 0.8));
  ctx.strokeStyle = grd; ctx.lineWidth = 2.5; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(x, y); ctx.stroke();
  ctx.fillStyle = '#fff';
  ctx.shadowColor = THEME.p[b.c]; ctx.shadowBlur = 12;
  ctx.beginPath(); ctx.arc(x, y, Math.max(meta.bullet_r * scale, 2.2), 0, 7); ctx.fill();
  ctx.shadowBlur = 0;
}

function drawFlash(f) {
  const k = f.life / f.max, x = sx(f.x), y = sy(f.y);
  ctx.save(); ctx.translate(x, y); ctx.rotate(f.ang);
  ctx.fillStyle = rgba(f.c, k * 0.9);
  const r = (meta.agent_r * scale) * (1.3 + (1 - k) * 1.5);
  ctx.beginPath(); ctx.moveTo(r * 1.4, 0);
  ctx.lineTo(0, r * 0.7); ctx.lineTo(0, -r * 0.7); ctx.closePath(); ctx.fill();
  ctx.restore();
}
function drawRing(r) {
  const k = r.life / r.max;
  ctx.strokeStyle = rgba(r.c, k * 0.7); ctx.lineWidth = 2 + k * 2;
  ctx.beginPath(); ctx.arc(sx(r.x), sy(r.y), r.r * scale, 0, 7); ctx.stroke();
}
function drawPart(q) {
  const k = q.life / q.max;
  ctx.fillStyle = rgba(q.c, k);
  ctx.beginPath(); ctx.arc(sx(q.x), sy(q.y), q.size * (0.5 + k), 0, 7); ctx.fill();
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function syncHUD(A, B, t) {
  el('h0').style.width = Math.max(A.a[0][3], 0) + '%';
  el('h1').style.width = Math.max(A.a[1][3], 0) + '%';
  el('timer').textContent = (playhead * meta.dt).toFixed(1) + 's';
}

// ---------------------------------------------------------------- controls
el('play').onclick = () => {
  if (playhead >= nframe - 1) { playhead = 0; lastEv = -1; verdictShown = false;
    el('verdict').classList.remove('show'); trails = [[], []]; }
  playing = !playing; el('play').textContent = playing ? '❚❚' : '▶';
};
el('next').onclick = () => loadMatch(mi + 1);
el('prev').onclick = () => loadMatch(mi - 1);
el('speed').onclick = () => {
  speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length];
  el('speed').textContent = speed.toFixed(1) + '×';
};
el('auto').onclick = () => { auto = !auto; el('auto').classList.toggle('active', auto); };
el('scrub').oninput = (e) => {
  playhead = (e.target.value / 1000) * (nframe - 1);
  lastEv = Math.floor(playhead); parts = []; rings = []; flashes = [];
  if (playhead < nframe - 1) { verdictShown = false; el('verdict').classList.remove('show'); }
};
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); el('play').onclick(); }
  if (e.code === 'ArrowRight') el('next').onclick();
  if (e.code === 'ArrowLeft') el('prev').onclick();
});

// debug/verification hook: DUEL.go(matchIndex, frac) jumps into a match
window.DUEL = {
  go: async (m, frac = 0.4, pause = false) => {
    await loadMatch(m);
    playhead = (nframe - 1) * frac;
    lastEv = Math.floor(playhead);
    // backfill trails so a paused still frame still shows motion streaks
    const f = Math.floor(playhead);
    trails = [[], []];
    for (let k = Math.max(0, f - 15); k <= f; k++)
      for (let p = 0; p < 2; p++) trails[p].push([frames[k].a[p][0], frames[k].a[p][1]]);
    playing = !pause; el('play').textContent = playing ? '❚❚' : '▶';
  },
  get info() { return { mi, nframe, playhead: +playhead.toFixed(1), playing }; },
};

boot();
