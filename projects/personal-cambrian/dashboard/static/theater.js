import * as THREE from "https://esm.sh/three@0.160.0";
import { OrbitControls } from "https://esm.sh/three@0.160.0/examples/jsm/controls/OrbitControls.js";

const $ = (s) => document.querySelector(s);
const PAL = [0x22d3ee, 0x9b7bff, 0x1fd6a6, 0x3b9eff, 0xf4b740, 0x6be29a];
const Q = (q) => new THREE.Quaternion(q[1], q[2], q[3], q[0]);   // wxyz -> xyzw
const V = (p) => new THREE.Vector3(p[0], p[1], p[2]);

// ---------- a render stage (scene + camera + controls) ----------
function makeStage(canvas, { autorotate = false, ground = true } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true; renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.15;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  const scene = new THREE.Scene();
  if (ground) scene.fog = new THREE.Fog(0x080b11, 7, 18);
  const camera = new THREE.PerspectiveCamera(42, 1, 0.01, 100);
  camera.position.set(2.4, 1.7, 2.8);
  const controls = new OrbitControls(camera, canvas);
  controls.enableDamping = true; controls.autoRotate = autorotate; controls.autoRotateSpeed = 1.0;
  controls.minDistance = 0.4; controls.maxDistance = 24;
  scene.add(new THREE.HemisphereLight(0x9fc4ff, 0x141a22, 0.6));
  const key = new THREE.DirectionalLight(0xfff4e6, 2.3);
  key.position.set(4, 8, 3); key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024); key.shadow.camera.near = 1; key.shadow.camera.far = 26;
  Object.assign(key.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6 });
  key.shadow.bias = -0.0006; scene.add(key);
  const rim = new THREE.DirectionalLight(0x66b0ff, 0.6); rim.position.set(-5, 2.5, -3); scene.add(rim);
  const fill = new THREE.PointLight(0x9b7bff, 5, 18); fill.position.set(2, 1.2, 3); scene.add(fill);
  if (ground) {
    const floor = new THREE.Mesh(new THREE.PlaneGeometry(60, 60),
      new THREE.MeshStandardMaterial({ color: 0x0c1018, roughness: 0.95, metalness: 0.0 }));
    floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true; scene.add(floor);
    const grid = new THREE.GridHelper(60, 60, 0x1c2836, 0x121922);
    grid.position.y = 0.002; scene.add(grid);
  }
  const root = new THREE.Group(); root.rotation.x = -Math.PI / 2; scene.add(root);   // z-up -> y-up
  return { renderer, scene, camera, controls, root, frames: null, meshes: null, muscles: null, fi: 0 };
}

function clearGroup(g) { while (g.children.length) { const c = g.children.pop(); c.geometry?.dispose?.(); } }

function bodyMesh(g, transparent) {
  const d = g.dims; let geo, fix = null;
  if (g.shape === "capsule") { geo = new THREE.CapsuleGeometry(d.radius, d.length, 14, 28); fix = "cap"; }
  else if (g.shape === "box") geo = new THREE.BoxGeometry(d.x * 2, d.y * 2, d.z * 2);
  else if (g.shape === "sphere") geo = new THREE.SphereGeometry(d.radius, 30, 22);
  else geo = new THREE.SphereGeometry(1, 30, 22);
  const col = PAL[g.depth % PAL.length];
  const mat = new THREE.MeshStandardMaterial({
    color: col, roughness: 0.55, metalness: 0.12,
    transparent, opacity: transparent ? 0.2 : 1,
    emissive: col, emissiveIntensity: transparent ? 0.4 : 0.0,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.castShadow = !transparent; mesh.receiveShadow = !transparent;
  if (fix === "cap") mesh.rotation.x = Math.PI / 2;          // three capsule Y -> our local Z
  if (g.shape === "ellipsoid") mesh.scale.set(d.x, d.y, d.z);
  const node = new THREE.Group(); node.add(mesh);
  return node;
}

function buildBodies(root, plan, transparent) {
  clearGroup(root);
  const meshes = {}; let lo = new THREE.Vector3(1e9, 1e9, 1e9), hi = new THREE.Vector3(-1e9, -1e9, -1e9);
  for (const g of plan.geoms) {
    const node = bodyMesh(g, transparent);
    node.position.copy(V(g.pos)); node.quaternion.copy(Q(g.quat));
    root.add(node); meshes[g.name] = node;
    lo.min(node.position); hi.max(node.position);
  }
  return { meshes, center: lo.clone().add(hi).multiplyScalar(0.5), size: hi.distanceTo(lo) || 1 };
}

function frameCamera(st, center, size) {
  // center is in MuJoCo coords (inside rotated root) -> bring to world y-up
  const c = center.clone().applyEuler(st.root.rotation);
  st.controls.target.copy(c);
  const d = Math.max(size * 1.7, 1.2);
  st.camera.position.set(c.x + d, c.y + d * 0.7, c.z + d);
}

// muscle cylinder between two world points (in MuJoCo coords inside root)
function setSeg(mesh, a, b, radius) {
  const av = V(a), bv = V(b), dir = new THREE.Vector3().subVectors(bv, av), len = dir.length() || 1e-4;
  mesh.position.copy(av).add(bv).multiplyScalar(0.5);
  mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
  mesh.scale.set(radius, len, radius);
}

// ---------- viewport (locomotion) ----------
const STAGE = makeStage($("#view"));
const ANAT = makeStage($("#anat"), { autorotate: true, ground: false });

function setCreatureStatic(plan) {
  const b = buildBodies(STAGE.root, plan, false);
  STAGE.meshes = b.meshes; STAGE.frames = null;
  frameCamera(STAGE, b.center, b.size);
}
function animateLoco(traj) {
  const b = buildBodies(STAGE.root, traj.plan, false);
  STAGE.meshes = b.meshes; STAGE.frames = traj.frames; STAGE.fi = 0;
  frameCamera(STAGE, b.center, b.size);
}
function buildAnatomy(bio) {
  const b = buildBodies(ANAT.root, bio.plan, true);
  ANAT.meshes = b.meshes; ANAT.frames = bio.frames; ANAT.fi = 0; ANAT.fmax = bio.force_max || 1;
  // muscle segments
  ANAT.muscles = [];
  const nM = bio.frames[0] ? bio.frames[0].muscles.length : 0;
  for (let i = 0; i < nM; i++) {
    const geo = new THREE.CylinderGeometry(1, 1, 1, 8);
    const mat = new THREE.MeshBasicMaterial({ color: 0xff4444, transparent: true, opacity: 0.95 });
    const m = new THREE.Mesh(geo, mat); ANAT.root.add(m); ANAT.muscles.push(m);
  }
  frameCamera(ANAT, b.center, b.size); ANAT.controls.target.applyEuler(new THREE.Euler());
}

function applyFrame(st) {
  if (!st.frames || !st.frames.length) return;
  const fr = st.frames[Math.floor(st.fi) % st.frames.length];
  const bodies = fr.bodies || fr;
  for (const id in bodies) { const n = st.meshes[id]; if (n) { n.position.copy(V(bodies[id][0])); n.quaternion.copy(Q(bodies[id][1])); } }
  if (st.muscles && fr.muscles) {
    fr.muscles.forEach((mu, i) => {
      const seg = st.muscles[i]; if (!seg) return;
      const f = Math.min(mu.force / st.fmax, 1);
      setSeg(seg, mu.org, mu.ins, 0.006 + 0.03 * f);
      seg.material.color.setRGB(0.6 + 0.4 * f, 0.12 + 0.1 * f, 0.12);
      seg.material.opacity = 0.45 + 0.55 * f;
    });
  }
}

// ---------- gauges ----------
function gaugeSVG(g) {
  const r = 26, c = 2 * Math.PI * r, frac = Math.min(g.ratio, 1);
  const col = g.over ? "#ff5a52" : (g.ratio > 0.7 ? "#f4b740" : "#1fd6a6");
  return `<div class="gauge"><svg viewBox="0 0 64 64">
    <circle cx="32" cy="32" r="${r}" fill="none" stroke="rgba(120,160,210,.16)" stroke-width="6"/>
    <circle cx="32" cy="32" r="${r}" fill="none" stroke="${col}" stroke-width="6" stroke-linecap="round"
      stroke-dasharray="${(c * frac).toFixed(1)} ${c.toFixed(1)}" transform="rotate(-90 32 32)"/>
    <text x="32" y="36" text-anchor="middle" fill="#dfe9f4" font-size="13" font-weight="500">${Math.round(g.ratio * 100)}%</text>
  </svg><div class="gl">${g.label}</div><div class="gv" style="color:${col}">${g.value}${g.unit}</div></div>`;
}
function renderGauges(b) {
  $("#gauges").innerHTML = gaugeSVG(b.heat) + gaugeSVG(b.energy) + gaugeSVG(b.neural);
}
function renderStats(s, viable, ref) {
  const rows = [["bodies", s.bodies], ["muscles", s.muscles], ["mass", s.mass_kg + " kg"],
    ["height", s.height_m + " m"], ["aspect", s.aspect], ["symmetry", s.symmetry]];
  $("#stats").innerHTML = rows.map(([k, v]) => `<div class="row"><span>${k}</span><span>${v}</span></div>`).join("") +
    `<span class="taglet">${viable ? "viable" : "non-viable"}</span>`;
}

// ---------- creature loading ----------
let CURRENT = "seed:quadruped";
async function loadCreature(ref) {
  CURRENT = ref; $("#loading").style.display = "flex"; $("#loading").textContent = "developing creature…";
  const info = await fetch("/api/creature?ref=" + encodeURIComponent(ref)).then((r) => r.json());
  setCreatureStatic(info.plan);
  renderGauges(info.budgets); renderStats(info.stats, info.viable, ref);
  $("#cname").textContent = ref.startsWith("node:") ? "Lineage " + ref.slice(5, 9) : (ref.split(":")[1] || ref);
  $("#ov-score").innerHTML = "Mass: <b>" + info.stats.mass_kg + " kg</b>";
  $("#loading").style.display = "none";
  $("#ov-phys").innerHTML = "Phys: <b>animating…</b>";
  // animated locomotion + anatomy (parallel, may take a couple seconds each)
  fetch(`/api/creature/trajectory?ref=${encodeURIComponent(ref)}&niche=${SESS.niche || "locomotion"}&steps=90`)
    .then((r) => r.json()).then((t) => {
      if (CURRENT !== ref) return;
      animateLoco(t);
      $("#ov-phys").innerHTML = "Phys: <b>active</b>";
      $("#ov-score").innerHTML = "Travelled: <b>" + t.root_displacement_m + " m</b>";
      $("#ctrait").textContent = SESS.traitOf?.(ref) || "evolved";
    }).catch(() => { $("#ov-phys").innerHTML = "Phys: <b>idle</b>"; });
  fetch(`/api/creature/biomechanics?ref=${encodeURIComponent(ref)}&steps=70`)
    .then((r) => r.json()).then((b) => { if (CURRENT === ref) buildAnatomy(b); }).catch(() => {});
}

// ---------- phylogeny tree (SVG) ----------
const TRAIT_COLOR = { serpentine: "#9b7bff", quadruped: "#1fd6a6", compact: "#3b9eff", radial: "#f4b740" };
let SESS = { niche: "locomotion", nodes: [], traitMap: {} };
SESS.traitOf = (ref) => ref.startsWith("node:") ? SESS.traitMap[ref.slice(5)] : "human seed";

function renderTree(data) {
  SESS.nodes = data.nodes; SESS.traitMap = {};
  $("#sess").innerHTML = `<span class="dot" style="background:${data.playing ? "#1fd6a6" : "#5f7488"}"></span>gen ${data.generation} · ${data.n_nodes} creatures`;
  $("#ov-gen").innerHTML = "Gen: <b>" + data.generation + "</b>";
  SESS.bestNode = data.best_node;
  if (data.best_distance != null) $("#ov-best").innerHTML = "Best: <b>" + data.best_distance.toFixed(3) + " m</b>";
  drawSpark(data.history || []);
  if (SESS.followBest && data.best_node && data.best_node !== SESS.shownBest) {
    SESS.shownBest = data.best_node; loadCreature("node:" + data.best_node);
  }
  const dt = $("#deeptime"); dt.max = Math.max(data.max_gen, 1); dt.value = dt.max;
  const svg = $("#tree"); const nodes = data.nodes;
  if (!nodes.length) { svg.innerHTML = `<text x="12" y="24" class="tlabel">no creatures yet — press Step or Play</text>`; svg.setAttribute("height", 60); return; }
  const ys = nodes.map((n) => n.y), gens = nodes.map((n) => n.gen);
  const rowH = 34, colW = 30, padX = 24, padY = 22;
  const W = $("#tree-wrap").clientWidth - 6;
  const ymax = Math.max(...ys, 1), gmax = Math.max(...gens, 1);
  const X = (y) => padX + (ys.length > 1 ? (y / ymax) * (W - 2 * padX) : 0);
  const Y = (g) => padY + g * rowH;
  svg.setAttribute("height", padY * 2 + gmax * rowH);
  svg.setAttribute("width", W);
  const byId = {}; nodes.forEach((n) => (byId[n.id] = n, SESS.traitMap[n.id] = n.trait));
  let edges = "", marks = "";
  for (const n of nodes) {
    if (n.parent && byId[n.parent]) {
      const p = byId[n.parent];
      edges += `<path d="M ${X(p.y)} ${Y(p.gen)} L ${X(n.y)} ${Y(p.gen)} L ${X(n.y)} ${Y(n.gen)}" fill="none" stroke="rgba(120,170,230,.22)" stroke-width="1.4" data-edge="${n.id}"/>`;
    }
  }
  for (const n of nodes) {
    const isRoot = n.parent === null;
    const col = isRoot ? "#22d3ee" : (TRAIT_COLOR[n.trait] || "#8fa4ba");
    const rad = isRoot ? 8 : 5.5;
    const innov = n.innov.length ? `<circle cx="${X(n.y) + 7}" cy="${Y(n.gen) - 6}" r="2.5" fill="#f4b740"/>` : "";
    marks += `<g class="tnode" data-id="${n.id}" data-gen="${n.gen}">
      <circle cx="${X(n.y)}" cy="${Y(n.gen)}" r="${rad}" fill="${col}" fill-opacity="0.9" stroke="rgba(255,255,255,.25)" stroke-width="1"/>${innov}</g>`;
  }
  svg.innerHTML = edges + marks;
  svg.querySelectorAll(".tnode").forEach((el) => {
    const n = byId[el.dataset.id];
    el.onclick = () => loadCreature("node:" + n.id);
    el.onmousemove = (e) => showTip(e, n);
    el.onmouseleave = () => ($("#tip").style.display = "none");
  });
  applyDeepTime(+dt.value);
}
function showTip(e, n) {
  const t = $("#tip");
  t.innerHTML = `<b>${n.parent === null ? "Agent Zero" : n.trait}</b> · gen ${n.gen}<br>` +
    `limbs ${n.limbs} · muscles ${n.muscles} · ${n.mass}kg · fit ${n.fitness}` +
    (n.innov.length ? `<br><b>${n.innov.join(" · ")}</b>` : "");
  t.style.display = "block"; t.style.left = (e.clientX + 12) + "px"; t.style.top = (e.clientY + 10) + "px";
}
function drawSpark(hist) {
  const cv = $("#spark"); if (!cv) return;
  const ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = "#6b7d92"; ctx.font = "9px ui-monospace,monospace"; ctx.fillText("best distance ↑", 6, 12);
  if (hist.length < 2) return;
  const ys = hist.map((h) => h.best_dist);
  let lo = Math.min(...ys), hi = Math.max(...ys); if (hi - lo < 1e-4) { hi += 0.01; lo -= 0.01; }
  ctx.strokeStyle = "#9b7bff"; ctx.lineWidth = 1.7; ctx.beginPath();
  hist.forEach((h, i) => {
    const x = 5 + (i / (hist.length - 1)) * (W - 10);
    const y = H - 5 - ((h.best_dist - lo) / (hi - lo)) * (H - 16);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#9b7bff"; ctx.fillText(ys[ys.length - 1].toFixed(2) + "m", W - 38, 12);
}
function applyDeepTime(maxGen) {
  $("#genlabel").textContent = "generation ≤ " + maxGen;
  $("#tree").querySelectorAll(".tnode").forEach((el) => (el.style.opacity = (+el.dataset.gen <= maxGen) ? 1 : 0.12));
}

// ---------- session control ----------
async function refreshTree() { renderTree(await fetch("/api/session/tree").then((r) => r.json())); }
let playPoll = null;
$("#btn-play").onclick = async () => {
  const on = $("#btn-play").dataset.on === "1";
  await fetch(on ? "/api/session/pause" : "/api/session/play", { method: "POST" });
  $("#btn-play").dataset.on = on ? "0" : "1";
  $("#btn-play").textContent = on ? "▶ Play" : "⏸ Pause";
  SESS.followBest = !on;                         // while playing, the viewport tracks the frontier
  if (!on) { playPoll = setInterval(refreshTree, 2000); refreshTree(); } else { clearInterval(playPoll); }
};
$("#btn-best").onclick = () => { if (SESS.bestNode) { SESS.shownBest = SESS.bestNode; loadCreature("node:" + SESS.bestNode); } };
$("#btn-step").onclick = async () => {
  $("#ov-phys").innerHTML = "Phys: <b>evolving…</b>";
  renderTree(await fetch("/api/session/step?n=1", { method: "POST" }).then((r) => r.json()));
  $("#ov-phys").innerHTML = "Phys: <b>active</b>";
};
$("#btn-reset").onclick = async () => {
  const seed = $("#seed-pick").value;
  await fetch("/api/session/reset?seed_creature=" + seed, { method: "POST" });
  clearInterval(playPoll); SESS.followBest = false; SESS.shownBest = null;
  $("#btn-play").dataset.on = "0"; $("#btn-play").textContent = "▶ Play";
  $("#ov-best").innerHTML = "Best: <b>—</b>"; drawSpark([]);
  await refreshTree(); loadCreature("seed:" + seed);
};
$("#deeptime").oninput = (e) => applyDeepTime(+e.target.value);

// ---------- render loop + resize ----------
let last = performance.now();
function loop(now) {
  const dt = (now - last) / 1000; last = now;
  for (const st of [STAGE, ANAT]) {
    if (st.frames) { st.fi += dt * 26; applyFrame(st); }
    st.controls.update(); st.renderer.render(st.scene, st.camera);
  }
  requestAnimationFrame(loop);
}
function resize() {
  for (const [st, el] of [[STAGE, $("#view")], [ANAT, $("#anat")]]) {
    const w = el.clientWidth, h = el.clientHeight;
    st.renderer.setSize(w, h, false); st.camera.aspect = w / h; st.camera.updateProjectionMatrix();
  }
}
window.addEventListener("resize", resize);

// ---------- boot ----------
$("#ov-gen").innerHTML = "Gen: <b>0</b>";
resize(); requestAnimationFrame(loop);
refreshTree();
loadCreature("seed:quadruped");
