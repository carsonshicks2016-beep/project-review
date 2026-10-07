// FOLDSPACE — renders a strip folding itself into target silhouettes.
// Loads folds.json (per-shape clips of strip vertices) and animates each fold
// with the target outline ghosted behind it.

const THEME = {
  bg0: '#060912', bg1: '#0e1636',
  grid: 'rgba(90,125,210,0.09)',
  edge: 'rgba(120,150,235,0.30)',
  ghost: 'rgba(130,165,255,0.30)',
  ink: '#e6ecff',
  // strip gradient stops (head -> tail)
  stops: [[75, 227, 255], [154, 123, 255], [255, 93, 176]],
};
const lerp = (a, b, t) => a + (b - a) * t;
function mix(c0, c1, t) { return [Math.round(lerp(c0[0], c1[0], t)), Math.round(lerp(c0[1], c1[1], t)), Math.round(lerp(c0[2], c1[2], t))]; }
function stripColor(u) {  // u in [0,1] along the strip
  const s = THEME.stops;
  const k = u * (s.length - 1), i = Math.min(Math.floor(k), s.length - 2);
  return mix(s[i], s[i + 1], k - i);
}

const cv = document.getElementById('scene'), ctx = cv.getContext('2d');
const stage = document.getElementById('stage'), el = (id) => document.getElementById(id);

let data = null, clips = [], meta = null, ci = 0;
let clip = null, frames = [], target = [], bounds = null;
let head = 0, playing = true, speed = 1, auto = true, ghost = true;
let scale = 1, ox = 0, oy = 0, dpr = 1;
const SPEEDS = [0.5, 1, 2], HOLD = 34;   // frames held on the finished shape

function fit() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = stage.clientWidth, h = stage.clientHeight;
  cv.width = w * dpr; cv.height = h * dpr; cv.style.width = w + 'px'; cv.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  if (!bounds) return;
  const pad = 60;
  const bw = bounds.maxx - bounds.minx, bh = bounds.maxy - bounds.miny;
  scale = Math.min((w - pad * 2) / bw, (h - pad * 2) / bh);
  ox = (w - bw * scale) / 2 - bounds.minx * scale;
  oy = (h - bh * scale) / 2 - bounds.miny * scale;
}
const sx = (x) => ox + x * scale, sy = (y) => oy - y * scale;   // flip y (math up)
window.addEventListener('resize', fit);

async function boot() {
  data = await fetch('replays/folds.json?t=' + Date.now()).then(r => r.json());
  meta = data.meta; clips = data.clips;
  el('ctot').textContent = clips.length;
  loadClip(0);
  requestAnimationFrame(loop);
}
function loadClip(i) {
  ci = (i + clips.length) % clips.length;
  clip = clips[ci];
  frames = clip.frames;
  const si = meta.shapes.indexOf(clip.shape);
  target = meta.targets[si];
  // stable bounds = union of straight strip (frame 0), final shape, and target
  let minx = 1e9, miny = 1e9, maxx = -1e9, maxy = -1e9;
  const acc = (p) => { for (const [x, y] of p) { minx = Math.min(minx, x); miny = Math.min(miny, y); maxx = Math.max(maxx, x); maxy = Math.max(maxy, y); } };
  acc(frames[0]); acc(frames[frames.length - 1]); acc(target);
  bounds = { minx, miny, maxx, maxy };
  head = 0; playing = true;
  el('shape').textContent = clip.shape;
  el('cidx').textContent = ci + 1;
  el('scorebig').textContent = Math.round(clip.score * 100) + '%';
  el('play').textContent = '❚❚';
  fit();
}

function loop(ts) {
  const dt = Math.min((ts - (loop.t || ts)) / 1000, 0.05); loop.t = ts;
  const total = frames.length - 1;
  if (playing) {
    head += dt * 30 * speed;
    if (head >= total + HOLD) {
      if (auto) { loadClip(ci + 1); } else { head = total; playing = false; el('play').textContent = '▶'; }
    }
  }
  draw();
  const shown = Math.min(head, total);
  el('scrub').value = Math.round((shown / Math.max(total, 1)) * 1000);
  requestAnimationFrame(loop);
}

function draw() {
  const W = stage.clientWidth, H = stage.clientHeight;
  const g = ctx.createRadialGradient(W / 2, H * 0.46, 50, W / 2, H / 2, Math.max(W, H) * 0.7);
  g.addColorStop(0, THEME.bg1); g.addColorStop(1, THEME.bg0);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

  drawGrid(W, H);
  if (ghost) drawTarget();

  const total = frames.length - 1;
  const h = Math.min(head, total);
  const f0 = Math.floor(h), f1 = Math.min(f0 + 1, total), t = h - f0;
  const A = frames[f0], B = frames[f1];
  const pts = A.map((p, i) => [lerp(p[0], B[i][0], t), lerp(p[1], B[i][1], t)]);
  drawStrip(pts, h >= total);
}

function drawGrid(W, H) {
  ctx.strokeStyle = THEME.grid; ctx.lineWidth = 1;
  const step = 1.4 * scale;
  if (step < 6) return;
  const x0 = ox % step, y0 = oy % step;
  ctx.beginPath();
  for (let x = x0; x < W; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, H); }
  for (let y = y0; y < H; y += step) { ctx.moveTo(0, y); ctx.lineTo(W, y); }
  ctx.stroke();
}

function drawTarget() {
  ctx.save();
  ctx.setLineDash([5, 6]); ctx.lineWidth = 1.5; ctx.strokeStyle = THEME.ghost;
  ctx.shadowColor = 'rgba(120,160,255,0.5)'; ctx.shadowBlur = 8;
  ctx.beginPath();
  target.forEach(([x, y], i) => i ? ctx.lineTo(sx(x), sy(y)) : ctx.moveTo(sx(x), sy(y)));
  ctx.closePath(); ctx.stroke();
  ctx.restore();
}

function drawStrip(pts, done) {
  ctx.save();
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.globalCompositeOperation = 'lighter';
  const n = pts.length;
  // glowing gradient segments
  for (let i = 1; i < n; i++) {
    const c = stripColor((i - 0.5) / (n - 1));
    ctx.strokeStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.shadowColor = ctx.strokeStyle; ctx.shadowBlur = done ? 18 : 12;
    ctx.lineWidth = 3.2;
    ctx.beginPath(); ctx.moveTo(sx(pts[i - 1][0]), sy(pts[i - 1][1])); ctx.lineTo(sx(pts[i][0]), sy(pts[i][1])); ctx.stroke();
  }
  // crease nodes
  ctx.shadowBlur = 8;
  for (let i = 0; i < n; i++) {
    const c = stripColor(i / (n - 1));
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.beginPath(); ctx.arc(sx(pts[i][0]), sy(pts[i][1]), i === 0 || i === n - 1 ? 3.4 : 1.7, 0, 7); ctx.fill();
  }
  ctx.restore();
}

// controls
el('play').onclick = () => {
  const total = frames.length - 1;
  if (head >= total) { head = 0; }
  playing = !playing; el('play').textContent = playing ? '❚❚' : '▶';
};
el('next').onclick = () => loadClip(ci + 1);
el('prev').onclick = () => loadClip(ci - 1);
el('speed').onclick = () => { speed = SPEEDS[(SPEEDS.indexOf(speed) + 1) % SPEEDS.length]; el('speed').textContent = speed.toFixed(1) + '×'; };
el('auto').onclick = () => { auto = !auto; el('auto').classList.toggle('active', auto); };
el('ghost').onclick = () => { ghost = !ghost; el('ghost').textContent = 'GHOST ' + (ghost ? 'ON' : 'OFF'); el('ghost').style.color = ghost ? '' : '#444c70'; };
el('scrub').oninput = (e) => { head = (e.target.value / 1000) * (frames.length - 1); playing = false; el('play').textContent = '▶'; };
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space') { e.preventDefault(); el('play').onclick(); }
  if (e.code === 'ArrowRight') el('next').onclick();
  if (e.code === 'ArrowLeft') el('prev').onclick();
});

window.FOLD = { go: (i) => loadClip(i), get info() { return { ci, shape: clip && clip.shape, head: +head.toFixed(1), nf: frames.length }; } };
boot();
