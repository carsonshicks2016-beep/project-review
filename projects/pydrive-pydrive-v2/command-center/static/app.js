/* Supra Command Center — frontend logic */
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
const api = (u, opt) => fetch(u, opt).then(r => r.json());
let CARS = ["supra", "rx7", "skyline", "lr4", "f150"];   // fallback; replaced by /api/cars on load
const CAR_LABELS = {
  supra: "Supra",
  rx7: "RX-7",
  skyline: "Skyline",
  lr4: "LR4",
  f150: "F-150 XLT",
  mazda787b: "Mazda 787B"
};
const carOptions = () => CARS.map(v => ({v, l: CAR_LABELS[v] || v}));
const defaultCar = () => CARS.includes("supra") ? "supra" : (CARS[0] || "supra");
const CAR_PREVIEW = {
  supra: {label: "Supra", type: "turbo coupe"},
  rx7: {label: "RX-7", type: "rotary coupe"},
  skyline: {label: "Skyline", type: "AWD grand tourer"},
  lr4: {label: "LR4", type: "boxy AWD SUV"},
  f150: {label: "F-150 XLT", type: "full-size pickup"},
  mazda787b: {label: "Mazda 787B", type: "Group C prototype"}
};
const HTML_ESC = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"};
const esc = v => String(v ?? "").replace(/[&<>"']/g, ch => HTML_ESC[ch]);
const REAL_ELEVATION_TRACKS = new Set(["nordschleife", "nurburgring", "nuerburgring"]);
const isRealElevationTrack = name => REAL_ELEVATION_TRACKS.has(String(name || "").toLowerCase());
const RING_TRACK = "nordschleife";
const RING_DEFAULT_OUT = "nordschleife_787b_ppo.pt";
const RING_RACE_BEST = "ring_787b_best.pt";
const RING_CAR = "mazda787b";
const FABLE_BEST = "fable5_ring_best.pt";
const FABLE_STAGES = ["foundation", "flow", "finish", "fast", "frontier"];
const SPRITE_FRAME_COUNT = 360;
const SPRITE_FRAME_MS = 1000 / 120;
const CAR_PREVIEW_IMAGES = new Set();
const PRELOADED_SPRITES = new Set();
let carPreviewLoopStarted = false;
let carPreviewPhase = 0;
let LAST_STATUS = {tasks: [], cpu: 0};
let DIAGNOSTICS = [];

function normalizeCarId(car) {
  const raw = String(car || "").toLowerCase().replace(/[^a-z0-9]/g, "");
  const aliases = {
    toyotasupra: "supra",
    supra: "supra",
    mazdarx7: "rx7",
    rx7: "rx7",
    nissanskyline: "skyline",
    skyline: "skyline",
    landroverlr4: "lr4",
    lr4: "lr4",
    fordf150: "f150",
    f150xlt: "f150",
    f150: "f150",
    mazda787b: "mazda787b",
    "787b": "mazda787b"
  };
  return aliases[raw] || (CAR_PREVIEW[raw] ? raw : defaultCar());
}

function carSpriteSrc(car, frame) {
  return `/api/car-sprite/${encodeURIComponent(car)}/${frame % SPRITE_FRAME_COUNT}.png?v=3`;
}

function preloadCarSprites(car) {
  if (PRELOADED_SPRITES.has(car)) return;
  PRELOADED_SPRITES.add(car);
  for (let i = 0; i < SPRITE_FRAME_COUNT; i += 1) {
    const img = new Image();
    img.src = carSpriteSrc(car, i);
  }
}

function startCarPreviewLoop() {
  if (carPreviewLoopStarted) return;
  carPreviewLoopStarted = true;
  const tick = ts => {
    const baseFrame = Math.floor(ts / SPRITE_FRAME_MS) % SPRITE_FRAME_COUNT;
    CAR_PREVIEW_IMAGES.forEach(img => {
      if (!img.isConnected) {
        CAR_PREVIEW_IMAGES.delete(img);
        return;
      }
      const phase = Number(img.dataset.phase || 0);
      const frame = (baseFrame + phase) % SPRITE_FRAME_COUNT;
      if (img.dataset.frame !== String(frame)) {
        img.dataset.frame = String(frame);
        img.src = carSpriteSrc(img.dataset.car, frame);
      }
    });
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function renderCarPreview(el, car, context, detail) {
  if (!el) return;
  const id = normalizeCarId(car);
  const meta = CAR_PREVIEW[id] || CAR_PREVIEW[defaultCar()] || CAR_PREVIEW.supra;
  preloadCarSprites(id);
  el.innerHTML = `
    <div class="preview-head">
      <span>${esc(context || "Selected car")}</span>
      <strong>${esc(meta.label)}</strong>
    </div>
    <div class="preview-stage">
      <img class="preview-car-sprite" data-car="${esc(id)}" data-frame="0"
        src="${carSpriteSrc(id, 0)}" alt="${esc(meta.label)} preview" decoding="async">
    </div>
    <div class="preview-caption">
      <span>${esc(detail || meta.type)}</span>
      <code>${esc(id)}</code>
    </div>`;
  const img = $(".preview-car-sprite", el);
  if (img) {
    img.dataset.phase = String(carPreviewPhase % SPRITE_FRAME_COUNT);
    carPreviewPhase += 7;
    CAR_PREVIEW_IMAGES.add(img);
    startCarPreviewLoop();
  }
}

function toast(msg) {
  let t = $(".toast") || Object.assign(document.body.appendChild(document.createElement("div")), {className: "toast"});
  t.textContent = msg; t.classList.add("show");
  clearTimeout(t._t); t._t = setTimeout(() => t.classList.remove("show"), 2200);
}

const fishEgg = $("#fish-egg");
if (fishEgg) {
  fishEgg.addEventListener("click", () => {
    fishEgg.classList.add("revealed");
    clearTimeout(fishEgg._hide);
    fishEgg._hide = setTimeout(() => fishEgg.classList.remove("revealed"), 2400);
  });
}

/* segmented control: seg(el, options, default, cb) -> get current via segVal(el) */
function segVal(elm) {
  return elm._val || elm.dataset.val;
}
function seg(elm, opts, def, cb) {
  elm.innerHTML = "";
  elm._val = def;
  elm.dataset.val = def;
  opts.forEach(o => {
    const v = typeof o === "string" ? o : o.v, lbl = typeof o === "string" ? o : o.l;
    const b = document.createElement("button");
    b.textContent = lbl; if (v === def) b.classList.add("on");
    b.onclick = () => { elm._val = v; elm.dataset.val = v; $$("button", elm).forEach(x => x.classList.remove("on")); b.classList.add("on"); cb && cb(v); };
    elm.appendChild(b);
  });
}

/* ---------- tabs ---------- */
function activateTab(name) {
  $$(".tab").forEach(x => x.classList.remove("active"));
  $$(".panel").forEach(x => x.classList.remove("active"));
  const tab = $(`.tab[data-tab="${name}"]`);
  const panel = $("#tab-" + name);
  if (tab) tab.classList.add("active");
  if (panel) panel.classList.add("active");
  if (name === "checkpoints") loadCheckpoints();
  if (name === "watch") loadWatchCkpts();
  if (name === "diagnostics") { loadDiagnostics(); loadWatchCkpts(); populateBrainCkpts(); loadBrainReports(); }
  if (name === "ringrace") { loadWatchCkpts(); loadRingRaceState(); }
  if (name === "fable") { loadWatchCkpts(); loadFableState(); }
  if (name === "fablega") { loadWatchCkpts(); loadFableGAState(); }
  if (name === "ops") renderOps();
}
$$(".tab").forEach(t => t.onclick = () => activateTab(t.dataset.tab));

/* ---------- track grids ---------- */
let TRACKS = [];
function trackCard(t, onSel) {
  const c = document.createElement("div");
  c.className = "tcard"; c.dataset.name = t.name;
  c.innerHTML = `<img loading="lazy" src="/api/thumb/${t.name}" alt="${t.name}">
    <div class="tname">${t.name}</div><div class="tdesc">${t.desc}</div>`;
  c.onclick = () => onSel(c, t.name);
  return c;
}
function fillGrid(elm, sel, def) {
  elm.innerHTML = "";
  TRACKS.forEach(t => {
    const c = trackCard(t, (card, name) => {
      $$(".tcard", elm).forEach(x => x.classList.remove("sel"));
      card.classList.add("sel"); elm._sel = name;
    });
    if (t.name === def) { c.classList.add("sel"); elm._sel = def; }
    elm.appendChild(c);
  });
}

/* ---------- DRIVE ---------- */
// #drive-car is built in init() after /api/cars resolves (so the list is authoritative)
seg($("#drive-audio"), [{v: "on", l: "On"}, {v: "off", l: "Muted"}], "on");
seg($("#gen-style"), ["gp", "technical", "speedway", "touge"], "gp");
$("#gen-diff").oninput = e => $("#gen-diff-v").textContent = (+e.target.value).toFixed(2);
$("#gen-len").oninput = e => $("#gen-len-v").textContent = e.target.value;
$("#drive-go").onclick = () => launch("drive", {
  car: segVal($("#drive-car")), track: $("#drive-tracks")._sel || "club",
  noaudio: segVal($("#drive-audio")) === "off"}, "drive " + ($("#drive-tracks")._sel || "club"));
$("#drive-gen-go").onclick = () => launch("drive_gen", {
  car: segVal($("#drive-car")), style: segVal($("#gen-style")),
  difficulty: $("#gen-diff").value, length: $("#gen-len").value}, "drive " + segVal($("#gen-style")));

/* ---------- WATCH ---------- */
let WATCH_CK = [];
const WATCH_ACTION = {ga: "ga_watch", race: "ppo_watch", drift: "drift_watch", hybrid: "hybrid_watch"};
const isBestFile = n => /_best\.(pt|npz)$/.test(n);
const bestNameOf = n => n.replace(/\.(pt|npz)$/, "_best.$1");
const ptName = n => String(n || RING_DEFAULT_OUT).endsWith(".pt") ? String(n || RING_DEFAULT_OUT) : `${n}.pt`;
// default to the ★best peak — the "latest" is usually the over-trained tail and
// will look far worse (or spin out) than the saved peak.
seg($("#watch-which"), [{v: "best", l: "Best ★"}, {v: "latest", l: "Latest"}], "best", updateWatchPreview);
function loadWatchCkpts() {
  api("/api/checkpoints").then(list => {
    WATCH_CK = list;        // full list (incl. _best) for resolution
    const byName = Object.fromEntries(list.map(c => [c.name, c]));
    const sel = $("#watch-ckpt"); const prev = sel.value;
    const perf = x => x.kind === "race" ? `lap ${x.laps}`
      : x.kind === "drift" ? `drift ${x.drift}·lap ${x.laps}`
      : x.kind === "hybrid" ? `lap ${x.laps}·style ${x.drift}`
      : `fit ${x.fitness}`;
    // dropdown shows the primary runs; the _best siblings are reached via the
    // Best★ toggle — so list BOTH the latest's and the ★best's numbers (the
    // latest can read lower: later snapshot AND a harder curriculum difficulty).
    sel.innerHTML = list.filter(c => !isBestFile(c.name)).map(c => {
      const head = c.kind === "ga" ? `GA · gen ${c.generation}`
        : `${c.kind} · ${c.updates}u${c.track ? " · 🎯" + c.track : ""}`;
      const best = byName[bestNameOf(c.name)];
      const bestStr = best ? `   ·   ★best ${perf(best)}` : "";
      const warn = c.layout === "pre-hills" ? "⚠pre-hills · " : "";
      return `<option value="${c.name}">${warn}${c.name}  —  ${head} · latest ${perf(c)}${bestStr}</option>`;
    }).join("");
    if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;

    // Populate watch opponent dropdowns with competitive race brains only
    const racePts = list.filter(c => c.kind === "race" && c.name.endsWith(".pt"));
    const opts = `<option value="none">none</option>` + racePts.map(c => `<option value="${c.name}">${c.name}</option>`).join("");
    ["watch-opp1", "watch-opp2", "watch-opp3"].forEach(id => {
       const s = $(`#${id}`);
       if (s) {
           const prevOpp = s.value;
           s.innerHTML = opts;
           if (prevOpp && [...s.options].some(o => o.value === prevOpp)) s.value = prevOpp;
       }
    });

    // Populate competitive race tab dropdowns
    ["race-opp1", "race-opp2", "race-opp3", "race-opp4", "race-opp5"].forEach(id => {
       const s = $(`#${id}`);
       if (s) {
           const prevOpp = s.value;
           s.innerHTML = opts;
           if (prevOpp && [...s.options].some(o => o.value === prevOpp)) s.value = prevOpp;
       }
    });

    watchSelectTrainedTrack(true);
    updateWatchPreview();
    populateRingSelect(list);
    populateRingRaceSelects(list);
    populateFableSelects(list);
    populateGaSelects(list);
    loadRingRaceState();
    renderOps();
  });
}
$("#watch-refresh").onclick = loadWatchCkpts;
function watchSelectTrainedTrack(silent=false) {        // specialist -> auto-pick its track
  const c = WATCH_CK.find(x => x.name === $("#watch-ckpt").value);
  if (!c || !c.track) return;
  const grid = $("#watch-tracks");
  const card = [...grid.children].find(el => el.dataset.name === c.track);
  if (card) {
    [...grid.children].forEach(x => x.classList.remove("sel"));
    card.classList.add("sel"); grid._sel = c.track;
    if (!silent) toast("🎯 specialist — watching on its trained track: " + c.track);
  }
}
function currentWatchRun() {
  const base = $("#watch-ckpt").value;
  if (!base) return null;
  let c = WATCH_CK.find(x => x.name === base);
  if (segVal($("#watch-which")) === "best") {
    const best = WATCH_CK.find(x => x.name === bestNameOf(base));
    if (best) c = best;
  }
  return c || null;
}
function updateWatchPreview() {
  const c = currentWatchRun();
  const detail = c ? `${c.name}${c.track ? " · " + c.track : ""}` : "pick a checkpoint";
  renderCarPreview($("#watch-car-preview"), c?.car || defaultCar(), "Checkpoint car", detail);
}
$("#watch-ckpt").addEventListener("change", () => {
  watchSelectTrainedTrack();
  updateWatchPreview();
});

function getWatchConfig() {
  const base = $("#watch-ckpt").value;
  const c = WATCH_CK.find(x => x.name === base);
  if (!c) {
    toast("pick a checkpoint to watch");
    return null;
  }
  let file = base;
  if (segVal($("#watch-which")) === "best") {
    const bn = bestNameOf(base);
    if (WATCH_CK.some(x => x.name === bn)) file = bn;
    else toast("no 'best' snapshot for this run — using latest");
  }
  return { c, file };
}

$("#watch-go").onclick = () => {
  const conf = getWatchConfig();
  if (!conf) return;

  const o1 = $("#watch-opp1").value, o2 = $("#watch-opp2").value, o3 = $("#watch-opp3").value;
  const opps = [o1, o2, o3].filter(x => x !== "none").join(",");

  let act = WATCH_ACTION[conf.c.kind] || "drift_watch";
  const params = {checkpoint: conf.file, track: $("#watch-tracks")._sel || "club"};

  if (opps) {
      act = "multiagent_watch";
      params.opponents = opps;
  }

  launch(act, params, "watch " + conf.file);
};
$("#watch-3d-go").onclick = () => {
  const conf = getWatchConfig();
  if (!conf) return;
  const o1 = $("#watch-opp1").value, o2 = $("#watch-opp2").value, o3 = $("#watch-opp3").value;
  const opps = [o1, o2, o3].filter(x => x !== "none").join(",");
  const trk = $("#watch-tracks")._sel || "club";
  let url = `/3d/?checkpoint=${encodeURIComponent(conf.file)}&track=${encodeURIComponent(trk)}`;
  if (opps) url += `&opponents=${encodeURIComponent(opps)}`;
  window.open(url, "_blank");
};

// Competitive Race Mode Handlers
function raceSelections() {
  const o1 = $("#race-opp1").value, o2 = $("#race-opp2").value, o3 = $("#race-opp3").value, o4 = $("#race-opp4").value, o5 = $("#race-opp5").value;
  return [o1, o2, o3, o4, o5].filter(x => x !== "none");
}
$("#race-refresh").onclick = loadWatchCkpts;
$("#race-go").onclick = () => {
  const opps = raceSelections();
  if (opps.length < 2) {
    toast("Please select at least 2 competitors to race.");
    return;
  }

  const params = {
    opponents: opps.join(","),
    track: $("#race-tracks")._sel || "national"
  };

  launch("watch_race", params, "symmetric race");
};
$("#race-diagnose").onclick = () => {
  const selected = raceSelections();
  if (!selected.length) {
    toast("Select at least one policy to diagnose.");
    return;
  }
  const checkpoint = selected[0];
  const opponents = selected.slice(1);
  const track = $("#race-tracks")._sel || "national";
  const meta = WATCH_CK.find(c => c.name === checkpoint) || {};
  const episodes = prompt("Diagnostic episodes:", "4") || "4";
  const maxSteps = prompt("Max control steps per episode (blank = full episode):", "1200");
  launch("diagnose_checkpoint", {
    checkpoint,
    track,
    scenario: opponents.length ? "frozen_opponents" : "auto",
    episodes,
    max_steps: maxSteps || "",
    opponents: opponents.join(","),
    ...(isRealElevationTrack(track) ? {} : {flat: true}),
  }, `diag-race:${checkpoint}`);
};

/* ---------- TRAIN ---------- */
const TRAIN = [
  {key: "ga", title: "🧬 Genetic Algorithm", sub: "Evolve NumPy brains (race)",
   extra: true, spec: true, ck: "ga_champion.npz", dflt: "ga_champion.npz", ext: ".npz",
   iters: {min: 20, max: 500, step: 10, def: 120, label: "Generations"},
   modes: [{l: "New run (charts)", a: "ga_train"}, {l: "Live swarm", a: "ga_live"}, {l: "Continue ↻", a: "ga_continue", resume: true}]},
  {key: "race", title: "🏁 PPO Race", sub: "Grip-racing policy", ck: "ppo_race.pt", spec: true, dflt: "ppo_race.pt", ext: ".pt",
   anneal: true, workers: true, reseed: true,
   iters: {min: 100, max: 20000, step: 100, def: 600, label: "Iterations"},
   modes: [{l: "New run (charts)", a: "ppo_train"}, {l: "Live (S/L)", a: "ppo_live"}, {l: "Continue ↻", a: "ppo_continue", resume: true}]},
  {key: "drift", title: "💨 PPO Drift", sub: "Drift policy", ck: "ppo_drift.pt", spec: true, dflt: "ppo_drift.pt", ext: ".pt",
   anneal: true, workers: true, target: true,
   iters: {min: 100, max: 20000, step: 100, def: 1000, label: "Iterations"},
   modes: [{l: "New run (charts)", a: "drift_train"}, {l: "Live (S/L)", a: "drift_live"}, {l: "Continue ↻", a: "drift_continue", resume: true}]},
  {key: "hybrid", title: "🏁💨 PPO Hybrid", sub: "Race speed + drift the corners", ck: "ppo_hybrid.pt", spec: true, dflt: "ppo_hybrid.pt", ext: ".pt",
   anneal: true, workers: true, target: true, style: true,
   iters: {min: 100, max: 20000, step: 100, def: 2000, label: "Iterations"},
   modes: [{l: "New run (charts)", a: "hybrid_train"}, {l: "Live (S/L)", a: "hybrid_live"}, {l: "Continue ↻", a: "hybrid_continue", resume: true}]},
  {key: "multiagent", title: "🏁 Multi-Agent Race", sub: "Race vs historical frozen policies", ck: "test_race.pt", spec: true, dflt: "test_race.pt", ext: ".pt",
   anneal: true, workers: true, opponents: true, reseed: true,
   iters: {min: 100, max: 20000, step: 100, def: 1000, label: "Iterations"},
   modes: [{l: "New run (charts)", a: "multiagent_train"}, {l: "Live (S/L)", a: "multiagent_live"}, {l: "Continue ↻", a: "multiagent_continue", resume: true}]},
  {key: "selfplay", title: "⚔️ True Multi-Agent", sub: "Race & learn simultaneously against clones of itself", ck: "ppo_selfplay.pt", spec: true, dflt: "ppo_selfplay.pt", ext: ".pt",
   anneal: true, workers: true, reseed: true,
   iters: {min: 100, max: 20000, step: 100, def: 1000, label: "Iterations"},
   modes: [{l: "New run (charts)", a: "selfplay_train"}, {l: "Live (S/L)", a: "selfplay_live"}, {l: "Continue ↻", a: "selfplay_continue", resume: true}]},
];
function ringOutName() {
  const raw = ($("#ring-out")?.value || RING_DEFAULT_OUT).trim();
  return ptName(raw || RING_DEFAULT_OUT);
}
function ringBestName() {
  return bestNameOf(ringOutName());
}
function ringCheckpoint() {
  const picked = $("#ring-ck")?.value;
  return picked || ringBestName();
}
function ringParams(extra={}) {
  return {
    track: RING_TRACK,
    car: CARS.includes(RING_CAR) ? RING_CAR : defaultCar(),
    out: ringOutName(),
    iters: $("#ring-iters")?.value || "8000",
    workers: $("#ring-workers")?.value || "8",
    pop: $("#ring-envs")?.value || "32",
    patience: $("#ring-patience")?.value || "850",
    max_restarts: $("#ring-maxrestarts")?.value || "6",
    anneal: true,
    ...extra,
  };
}
function populateRingSelect(list=WATCH_CK) {
  const sel = $("#ring-ck");
  if (!sel) return;
  const prev = sel.value;
  const pts = (list || []).filter(c => c.name && c.name.endsWith(".pt"));
  const ringish = pts.filter(c =>
    c.track === RING_TRACK ||
    String(c.track_profile || "").includes("nordschleife") ||
    /nord|ring|nurb|nuer/i.test(c.name));
  const shown = ringish.length ? ringish : pts.filter(c => c.kind === "race").slice(0, 30);
  const fallback = ringBestName();
  sel.innerHTML = `<option value="${esc(fallback)}">${esc(fallback)} — expected best</option>` +
    shown.map(c => `<option value="${esc(c.name)}">${isBestFile(c.name) ? "★ " : ""}${esc(c.name)}${c.track ? " — " + esc(c.track) : ""}</option>`).join("");
  if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
  else if ([...sel.options].some(o => o.value === fallback)) sel.value = fallback;
}
function initRingPipeline() {
  if (!$("#ring-lane")) return;
  const bindRange = (id, out, fmt=v=>v) => {
    const el = $(`#${id}`), label = $(`#${out}`);
    if (!el || !label) return;
    el.oninput = e => { label.textContent = fmt(e.target.value); };
  };
  bindRange("ring-iters", "ring-iv");
  bindRange("ring-workers", "ring-wv");
  bindRange("ring-envs", "ring-ev");
  bindRange("ring-patience", "ring-patv");
  bindRange("ring-maxrestarts", "ring-mrv");
  $("#ring-out").onchange = () => populateRingSelect();
  $("#ring-train").onclick = () => launch("ring_ppo_train", ringParams(), "ring ppo train");
  $("#ring-live").onclick = () => launch("ring_ppo_live", ringParams(), "ring ppo live");
  $("#ring-continue").onclick = () => launch("ring_ppo_continue",
    ringParams({checkpoint: ringCheckpoint()}), "ring ppo continue");
  $("#ring-watch").onclick = () => launch("ring_ppo_watch",
    ringParams({checkpoint: ringCheckpoint()}), "watch ring best");
  $("#ring-diagnose").onclick = () => {
    const episodes = prompt("Diagnostic episodes:", "1") || "1";
    const maxSteps = prompt("Max control steps per episode (blank = full episode):", "") || "";
    launch("ring_ppo_diagnose", ringParams({
      checkpoint: ringCheckpoint(),
      episodes,
      max_steps: maxSteps,
    }), "diag ring best");
  };
  populateRingSelect();
}

function ringRaceStage() {
  return segVal($("#rr-stage")) || "survive";
}
function ringRaceOutFor(stage=ringRaceStage()) {
  const raw = ($("#rr-out")?.value || "").trim();
  if (raw) return ptName(raw);
  return `ring_787b_${stage === "auto" ? "survive" : stage}.pt`;
}
function ringRaceCheckpoint() {
  return $("#rr-watch-ck")?.value || $("#rr-resume")?.value || RING_RACE_BEST;
}
function ringRaceParams(extra={}) {
  const stage = extra.stage || ringRaceStage();
  return {
    track: RING_TRACK,
    car: CARS.includes(RING_CAR) ? RING_CAR : defaultCar(),
    stage,
    out: ringRaceOutFor(stage),
    iters: $("#rr-iters")?.value || "8000",
    workers: $("#rr-workers")?.value || "8",
    pop: $("#rr-envs")?.value || "32",
    patience: $("#rr-patience")?.value || "850",
    max_restarts: $("#rr-maxrestarts")?.value || "6",
    anneal: ($("#rr-anneal") ? segVal($("#rr-anneal")) : "on") !== "off",
    ...extra,
  };
}
function ringRaceCkpts(list=WATCH_CK) {
  return (list || []).filter(c => c.name && c.name.endsWith(".pt") && (
    c.ring_pipeline || c.track === RING_TRACK ||
    String(c.track_profile || "").includes("nordschleife") ||
    /^ring_787b|ring_smoke/i.test(c.name)
  ));
}
function populateRingRaceSelects(list=WATCH_CK) {
  const pts = ringRaceCkpts(list);
  const opts = `<option value="${esc(RING_RACE_BEST)}">${esc(RING_RACE_BEST)} — pipeline best</option>` +
    pts.map(c => `<option value="${esc(c.name)}">${isBestFile(c.name) ? "★ " : ""}${esc(c.name)}${c.ring_stage ? " — " + esc(c.ring_stage) : ""}</option>`).join("");
  ["rr-resume", "rr-watch-ck"].forEach(id => {
    const sel = $(`#${id}`);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = opts;
    if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
  });
}
function setRingRaceOutForStage(stage) {
  const el = $("#rr-out");
  if (!el) return;
  const current = el.value || "";
  if (/^ring_787b_(survive|fast|attack)\.pt$/.test(current) || !current) {
    el.value = `ring_787b_${stage === "auto" ? "survive" : stage}.pt`;
  }
}
function fmtMaybeTime(v) {
  if (!v) return "--";
  const n = Number(v);
  if (!isFinite(n) || n <= 0) return "--";
  return fmtLap(n);
}
function loadRingRaceState() {
  api("/api/ring-race").then(s => {
    const m = s.manifest || {};
    const ev = m.latest_eval || {};
    $("#rr-state-stage").textContent = m.current_stage || "--";
    $("#rr-state-active").textContent = m.active_checkpoint || "--";
    $("#rr-state-best").textContent = m.best_checkpoint || (s.best_exists ? RING_RACE_BEST : "--");
    $("#rr-state-score").textContent = ev.metric == null ? "--" : Number(ev.metric).toFixed(3);
    $("#rr-state-gate").textContent = ev.clean_sectors != null
      ? `${ev.clean_sectors}/${ev.sector_count || 16} sectors`
      : ev.clean_progress_frac != null ? `${(ev.clean_progress_frac * 100).toFixed(1)}% clean` : "--";
    $("#rr-state-rec").textContent = m.recommended_next_action || ev.recommendation || "--";
    $("#rr-state-reseed").textContent = m.reseed_count == null ? "0" : String(m.reseed_count);
    const diag = s.latest_diagnostic;
    $("#rr-diag-path").value = diag?.path || "none yet";
    $("#rr-summary-path").value = diag?.path ? `${diag.path}/summary.md` : "none yet";
    $("#rr-readme-path").value = diag?.path ? `${diag.path}/README_FOR_AI.md` : "none yet";
    $("#rr-best-lap").textContent = fmtMaybeTime(ev.best_clean_lap);
    $("#rr-clean-progress").textContent = ev.max_clean_progress_m == null ? "--" : `${Math.round(ev.max_clean_progress_m)} m`;
    $("#rr-offtrack").textContent = ev.offtrack_seconds == null ? "--" : `${Number(ev.offtrack_seconds).toFixed(2)} s`;
    $("#rr-invalid").textContent = ev.invalid_laps == null ? "--" : String(ev.invalid_laps);
    $("#rr-dottinger").textContent = ev.max_dottinger_speed == null ? "--" : `${(Number(ev.max_dottinger_speed) * 3.6).toFixed(0)} km/h`;
    $("#rr-terms").textContent = ev.termination_counts ? Object.entries(ev.termination_counts).map(([k,v]) => `${k}:${v}`).join(" · ") : "--";
    const mini = $("#rr-mini-metrics");
    if (mini) {
      mini.innerHTML = [
        ["clean progress", ev.max_clean_progress_m == null ? "--" : `${Math.round(ev.max_clean_progress_m)} m`],
        ["terminal rate", ev.terminal_rate == null ? "--" : Number(ev.terminal_rate).toFixed(2)],
        ["clean chain", ev.clean_chain ?? "--"],
        ["mean speed", ev.mean_speed == null ? "--" : `${(Number(ev.mean_speed) * 3.6).toFixed(0)} km/h`],
      ].map(([a,b]) => `<div><span>${esc(a)}</span><b>${esc(b)}</b></div>`).join("");
    }
  }).catch(() => {});
}
function initRingRace() {
  if (!$("#tab-ringrace")) return;
  seg($("#rr-stage"), [{v:"auto", l:"Auto"}, {v:"survive", l:"Survive"}, {v:"fast", l:"Fast"}, {v:"attack", l:"Attack"}], "survive", setRingRaceOutForStage);
  seg($("#rr-anneal"), [{v:"on", l:"On"}, {v:"off", l:"Off"}], "on");
  const bindRange = (id, out) => {
    const el = $(`#${id}`), label = $(`#${out}`);
    if (!el || !label) return;
    el.oninput = e => { label.textContent = e.target.value; };
  };
  bindRange("rr-iters", "rr-iv");
  bindRange("rr-workers", "rr-wv");
  bindRange("rr-envs", "rr-ev");
  bindRange("rr-patience", "rr-patv");
  bindRange("rr-maxrestarts", "rr-mrv");
  $("#rr-refresh").onclick = () => { loadWatchCkpts(); loadRingRaceState(); };
  $("#rr-start-survive").onclick = () => launch("ring_race_survive", ringRaceParams({stage:"survive", out:"ring_787b_survive.pt"}), "ring survive");
  $("#rr-start-fast").onclick = () => launch("ring_race_fast", ringRaceParams({stage:"fast", out:"ring_787b_fast.pt"}), "ring fast");
  $("#rr-start-attack").onclick = () => launch("ring_race_attack", ringRaceParams({stage:"attack", out:"ring_787b_attack.pt"}), "ring attack");
  $("#rr-start-auto").onclick = () => launch("ring_race_auto", ringRaceParams({stage:"auto"}), "ring auto");
  $("#rr-continue").onclick = () => launch("ring_race_continue", ringRaceParams({checkpoint: $("#rr-resume").value || RING_RACE_BEST}), "ring continue");
  $("#rr-live").onclick = () => launch("ring_race_live", ringRaceParams({checkpoint: $("#rr-resume").value || ""}), "ring live");
  $("#rr-stop").onclick = () => { if (activePid && TASKS[activePid]?.live) api(`/api/stop/${activePid}`, {method:"POST"}).then(()=>toast("■ stopping Ring Race run")); };
  $("#rr-watch-best").onclick = () => launch("ring_race_watch", ringRaceParams({checkpoint: RING_RACE_BEST}), "watch ring race best");
  $("#rr-watch-selected").onclick = () => launch("ring_race_watch", ringRaceParams({checkpoint: ringRaceCheckpoint()}), "watch ring race");
  $("#rr-diagnose-best").onclick = () => launch("ring_race_diagnose", ringRaceParams({checkpoint: RING_RACE_BEST, episodes:"1", max_steps:""}), "diagnose ring race best");
  $("#rr-diagnose-selected").onclick = () => launch("ring_race_diagnose", ringRaceParams({checkpoint: ringRaceCheckpoint(), episodes:"1", max_steps:""}), "diagnose ring race");
  populateRingRaceSelects();
  loadRingRaceState();
}

/* ---------- FABLE FIVE (superhuman Nordschleife pipeline) ---------- */
function fableStage() {
  return segVal($("#fb-stage")) || "foundation";
}
function fableOutFor(stage=fableStage()) {
  const raw = ($("#fb-out")?.value || "").trim();
  if (raw) return ptName(raw);
  return `fable5_ring_${stage === "auto" ? "foundation" : stage}.pt`;
}
function fableCheckpoint() {
  return $("#fb-watch-ck")?.value || $("#fb-resume")?.value || FABLE_BEST;
}
function fableParams(extra={}) {
  const stage = extra.stage || fableStage();
  const p = {
    track: RING_TRACK,
    car: CARS.includes(RING_CAR) ? RING_CAR : defaultCar(),
    stage,
    out: fableOutFor(stage),
    iters: $("#fb-iters")?.value || "6000",
    workers: $("#fb-workers")?.value || "8",
    pop: $("#fb-envs")?.value || "32",
    ...extra,
  };
  const pat = $("#fb-patience")?.value;
  if (pat && +pat > 0) p.patience = pat;         // 0 = stage default
  const sc = $("#fb-scale")?.value;
  if (sc && +sc > 0) p.envelope_scale = sc;      // 0 = stage default
  if ($("#fb-open-floor")?.checked) p.open_best_floor = true;
  const rs = $("#fb-resume")?.value;
  if (rs) p.resume = rs;
  return p;
}
function fableCkpts(list=WATCH_CK) {
  return (list || []).filter(c => c.name && c.name.endsWith(".pt") && (
    c.fable_pipeline || /^fable5/i.test(c.name)
  ));
}
function populateFableSelects(list=WATCH_CK) {
  const pts = fableCkpts(list);
  const stageTag = c => c.fable_stage ? ` — ${esc(c.fable_stage)}` : "";
  const opts = `<option value="">(stage default resume)</option>` +
    `<option value="${esc(FABLE_BEST)}">${esc(FABLE_BEST)} — pipeline best</option>` +
    pts.map(c => `<option value="${esc(c.name)}">${isBestFile(c.name) ? "★ " : ""}${esc(c.name)}${stageTag(c)}</option>`).join("");
  ["fb-resume", "fb-watch-ck"].forEach(id => {
    const sel = $(`#${id}`);
    if (!sel) return;
    const prev = sel.value;
    sel.innerHTML = id === "fb-watch-ck"
      ? opts.replace('<option value="">(stage default resume)</option>', "")
      : opts;
    if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
  });
}
function setFableOutForStage(stage) {
  const el = $("#fb-out");
  if (!el) return;
  const current = el.value || "";
  // only overwrite defaults the picker itself wrote, never a custom run name
  if (/^fable5_ring(_(foundation|flow|finish|fast|frontier))?(\.pt)?$/.test(current) || !current) {
    // auto: the name is a RUN PREFIX (files become <prefix>_<stage>.pt)
    el.value = stage === "auto" ? "fable5_ring" : `fable5_ring_${stage}.pt`;
  }
}
function loadFableState() {
  api("/api/fable-five").then(s => {
    const m = s.manifest || {};
    const ev = m.latest_eval || {};
    $("#fb-state-stage").textContent = m.current_stage || "--";
    $("#fb-state-active").textContent = m.active_checkpoint || "--";
    $("#fb-state-best").textContent = m.best_checkpoint || (s.best_exists ? FABLE_BEST : "--");
    $("#fb-state-score").textContent = ev.metric == null ? "--" : Number(ev.metric).toFixed(3);
    $("#fb-state-clean").textContent = ev.clean_sectors != null
      ? `${ev.clean_sectors}/${ev.sector_count || 16} (chain ${ev.clean_chain ?? "?"})` : "--";
    $("#fb-state-pace").textContent = ev.pace_ratio == null ? "--"
      : `${(Number(ev.pace_ratio) * 100).toFixed(0)}% of envelope`;
    $("#fb-state-rec").textContent = m.recommended_next_action || ev.recommendation || "--";
    const theo = ev.theoretical_lap ?? m.theoretical_lap;
    $("#fb-best-lap").textContent = fmtMaybeTime(ev.lap_time);
    $("#fb-theo-lap").textContent = theo ? fmtLap(Number(theo)) : "--";
    const bellof = (ev.benchmarks || m.benchmarks || {}).bellof_956_1983_qualifying || 371.13;
    $("#fb-vs-human").textContent = ev.lap_time
      ? `${(Number(ev.lap_time) / bellof * 100).toFixed(1)}% (${Number(ev.lap_time) < bellof ? "-" : "+"}${Math.abs(Number(ev.lap_time) - bellof).toFixed(1)}s)`
      : "--";
    const shEl = $("#fb-superhuman");
    if (ev.superhuman) { shEl.textContent = "🏆 SUPERHUMAN"; shEl.style.color = "#ffd75e"; }
    else if (ev.lap_time) { shEl.textContent = "clean lap banked — compressing"; shEl.style.color = ""; }
    else { shEl.textContent = "no clean lap yet"; shEl.style.color = ""; }
    // superhuman progress bar: right end = the 919 Evo outright record (fastest),
    // markers at the envelope lap and the Bellof human bar. fill = evo/lap.
    const evo = (ev.benchmarks || m.benchmarks || {}).porsche_919_evo_2018_record || 319.55;
    const fill = $("#fb-lapbar-fill");
    if (fill) {
      const pct = ev.lap_time
        ? Math.max(0, Math.min(100, evo / Number(ev.lap_time) * 100)) : 0;
      fill.style.width = pct.toFixed(1) + "%";
      fill.classList.toggle("superhuman", !!ev.superhuman);
      const mt = $("#fb-mark-theo"), mb = $("#fb-mark-bellof");
      if (mt && theo) mt.style.left = Math.min(99, evo / Number(theo) * 100).toFixed(1) + "%";
      if (mb) mb.style.left = Math.min(99, evo / bellof * 100).toFixed(1) + "%";
    }
    const diag = s.latest_diagnostic;
    $("#fb-diag-path").value = diag?.path || "none yet";
    $("#fb-progress").textContent = ev.max_progress_m == null ? "--"
      : `${Math.round(ev.max_progress_m)} m (${((ev.progress_frac || 0) * 100).toFixed(1)}%)`;
    $("#fb-terminal").textContent = ev.terminal_rate == null ? "--" : Number(ev.terminal_rate).toFixed(2);
    $("#fb-offtrack").textContent = ev.offtrack_seconds == null ? "--" : `${Number(ev.offtrack_seconds).toFixed(2)} s`;
    $("#fb-terms").textContent = ev.termination_counts
      ? Object.entries(ev.termination_counts).map(([k,v]) => `${k}:${v}`).join(" · ") : "--";
    const mini = $("#fb-mini-metrics");
    if (mini) {
      const worst = (ev.worst_sectors || []).map(w =>
        `S${w.sector}${w.clean ? "" : "✗"} ${(Number(w.pace || 0) * 100).toFixed(0)}%`).join(" · ");
      const auto = m.auto || {};
      const hist = auto.history || [];
      const lastH = hist[hist.length - 1];
      let ladder = "--";
      if (lastH) {
        ladder = lastH.stage + (lastH.segment ? ` seg ${lastH.segment}` : "")
          + (lastH.attempts > 1 ? ` (retry ${lastH.attempts})` : "")
          + (lastH.soft_advanced ? " · soft-adv" : "")
          + (lastH.scale != null ? ` @ ${Number(lastH.scale).toFixed(2)}` : "");
      }
      const pit = m.pit || {};
      const pitStr = `${pit.reseeds ?? 0} reseeds`
        + (pit.consolidations ? ` · ${pit.consolidations} consol` : "");
      mini.innerHTML = [
        ["mean speed", ev.mean_speed == null ? "--" : `${(Number(ev.mean_speed) * 3.6).toFixed(0)} km/h`],
        ["invalid laps", ev.invalid_laps ?? "--"],
        ["envelope lap", theo ? fmtLap(Number(theo)) : "--"],
        ["reward ver", m.fable_reward_version || "--"],
        ["envelope scale", m.envelope_scale == null ? "--" : Number(m.envelope_scale).toFixed(2)],
        ["frontier scale", auto.frontier_scale == null ? "--" : Number(auto.frontier_scale).toFixed(2)],
        ["ladder", ladder],
        ["ladder budget", auto.total ? `${auto.spent ?? 0}/${auto.total} iters` : "--"],
        ["lap style", ev.lap_style || "--"],
        ["active best", m.active_best_checkpoint
          ? String(m.active_best_checkpoint).replace(/\.pt$/, "") : "--"],
        ["resumed from", m.resumed_from ? String(m.resumed_from).replace(/\.pt$/, "") : "scratch/unknown"],
        ["worst sectors", worst || "--"],
        ["pit calls", m.pit ? pitStr : "--"],
      ].map(([a,b]) => `<div><span>${esc(a)}</span><b>${esc(b)}</b></div>`).join("");
    }
    const pitEl = $("#fb-pitlog");
    if (pitEl) {
      const dec = (m.pit && m.pit.decisions) || [];
      pitEl.innerHTML = dec.length ? dec.slice().reverse().map(d =>
        `<div class="pitrow ${esc(d.decision)}"><span>${esc(d.t || "")}</span>` +
        `<b>${esc((d.decision || "").toUpperCase())}</b>` +
        `<i>${esc(d.reason || "")}</i></div>`).join("")
        : "<i>no decisions yet</i>";
    }
  }).catch(() => {});
}
function initFableFive() {
  if (!$("#tab-fable")) return;
  seg($("#fb-stage"), [{v:"auto", l:"Auto"},
    {v:"foundation", l:"1 Foundation"}, {v:"flow", l:"2 Flow"},
    {v:"finish", l:"3 Finish"}, {v:"fast", l:"4 Fast"},
    {v:"frontier", l:"5 Frontier"}], "foundation", setFableOutForStage);
  const bindRange = (id, out, dflt) => {
    const el = $(`#${id}`), label = $(`#${out}`);
    if (!el || !label) return;
    el.oninput = e => { label.textContent = +e.target.value > 0 ? e.target.value : (dflt || "stage default"); };
  };
  bindRange("fb-iters", "fb-iv");
  bindRange("fb-workers", "fb-wv");
  bindRange("fb-envs", "fb-ev");
  bindRange("fb-patience", "fb-patv", "stage default");
  bindRange("fb-scale", "fb-scalev", "stage default");
  $("#fb-refresh").onclick = () => { loadWatchCkpts(); loadFableState(); };
  $("#fb-start").onclick = () => launch("fable_train", fableParams(), `fable ${fableStage()}`);
  $("#fb-auto").onclick = () => launch("fable_auto", fableParams({stage:"auto"}), "fable auto ladder");
  $("#fb-continue").onclick = () => launch("fable_continue",
    fableParams({checkpoint: $("#fb-resume").value || FABLE_BEST}), "fable continue");
  $("#fb-live").onclick = () => launch("fable_live", fableParams(), "fable live");
  $("#fb-stop").onclick = () => { if (activePid && TASKS[activePid]?.live) api(`/api/stop/${activePid}`, {method:"POST"}).then(()=>toast("■ stopping Fable Five run")); };
  // no checkpoint -> the server resolves the TRUE current best
  // (active_best_checkpoint -> best_checkpoint -> FABLE_BEST), so "best" follows
  // the run in flight instead of a stale promoted champion.
  $("#fb-watch-best").onclick = () => launch("fable_watch", fableParams(), "watch fable best");
  $("#fb-watch-selected").onclick = () => launch("fable_watch", fableParams({checkpoint: fableCheckpoint()}), "watch fable");
  $("#fb-diagnose-best").onclick = () => launch("fable_diagnose", fableParams({episodes:"1", max_steps:""}), "diagnose fable best");
  $("#fb-diagnose-selected").onclick = () => launch("fable_diagnose", fableParams({checkpoint: fableCheckpoint(), episodes:"1", max_steps:""}), "diagnose fable");
  populateFableSelects();
  loadFableState();
}

// ---- Fable GA (evolution baseline vs PPO) ---------------------------------- #
function gaStage() { return segVal($("#fga-stage")) || "frontier"; }
function gaParams(extra={}) {
  const stage = extra.stage || gaStage();
  const p = {
    stage,
    car: CARS.includes(RING_CAR) ? RING_CAR : defaultCar(),
    out: ($("#fga-out")?.value || "ga_787b_ring.npz").trim() || "ga_787b_ring.npz",
    gens: $("#fga-gens")?.value || "5000",
    pop: $("#fga-pop")?.value || "96",
    workers: $("#fga-workers")?.value || "8",
    eval_every: $("#fga-eval")?.value || "5",
    hidden: ($("#fga-hidden")?.value || "128,128").trim(),
    ...extra,
  };
  const mins = $("#fga-minutes")?.value;
  if (mins && +mins > 0) p.minutes = mins;         // 0 = no wall-clock cap
  const rs = $("#fga-resume")?.value;
  if (rs) p.resume = rs;
  return p;
}
function gaCheckpoint() {
  return $("#fga-watch-ck")?.value || $("#fga-resume")?.value || "ga_787b_ring.npz";
}
function gaCkpts(list=WATCH_CK) {
  return (list || []).filter(c => c.name && c.name.endsWith(".npz") &&
    (c.fable_ga || c.layout === "fable-v1"));
}
function populateGaSelects(list=WATCH_CK) {
  const pts = gaCkpts(list);
  const tag = c => (c.fable_stage ? ` — ${esc(c.fable_stage)}` : "") +
    (c.lap_time ? ` · ${fmtLap(c.lap_time)}` : (c.metric != null ? ` · m${Number(c.metric).toFixed(1)}` : ""));
  const base = `<option value="ga_787b_ring.npz">ga_787b_ring.npz — default</option>`;
  const opts = pts.map(c => `<option value="${esc(c.name)}">${esc(c.name)}${tag(c)}</option>`).join("");
  ["fga-resume", "fga-watch-ck"].forEach(id => {
    const sel = $(`#${id}`); if (!sel) return;
    const prev = sel.value;
    const head = id === "fga-resume" ? `<option value="">(none — start fresh)</option>` : "";
    sel.innerHTML = head + (pts.length ? opts : (id === "fga-resume" ? "" : base));
    if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
  });
}
function loadFableGAState() {
  api("/api/fable-ga").then(s => {
    const snap = s.snapshot || {};
    const ev = snap.eval || {};
    const champ = s.champion || {};
    $("#fga-state-gen").textContent = snap.generation ?? champ.generation ?? "--";
    $("#fga-state-metric").textContent = snap.best_metric == null
      ? (champ.metric ?? "--") : Number(snap.best_metric).toFixed(3);
    $("#fga-state-fit").textContent = snap.best_fitness == null
      ? "--" : Number(snap.best_fitness).toFixed(1);
    $("#fga-state-clean").textContent = ev.clean_sectors != null
      ? `${ev.clean_sectors}/${ev.sector_count || 16} (chain ${ev.clean_chain ?? "?"})` : "--";
    $("#fga-state-pace").textContent = ev.pace_ratio == null ? "--"
      : `${(Number(ev.pace_ratio) * 100).toFixed(0)}% of envelope`;
    $("#fga-state-file").textContent = s.champion_exists ? s.checkpoint : "not saved yet";
    const theo = ev.theoretical_lap ?? champ.fable_theoretical_lap;
    const lap = ev.lap_time ?? (champ.lap_time || null);
    $("#fga-best-lap").textContent = fmtMaybeTime(lap);
    $("#fga-theo-lap").textContent = theo ? fmtLap(Number(theo)) : "--";
    const bellof = s.superhuman_lap || 371.13;
    $("#fga-vs-human").textContent = lap
      ? `${(Number(lap) / bellof * 100).toFixed(1)}% (${Number(lap) < bellof ? "-" : "+"}${Math.abs(Number(lap) - bellof).toFixed(1)}s)`
      : "--";
    const shEl = $("#fga-superhuman");
    if (ev.superhuman) { shEl.textContent = "🏆 SUPERHUMAN"; shEl.style.color = "#ffd75e"; }
    else if (lap) { shEl.textContent = "clean lap banked"; shEl.style.color = ""; }
    else { shEl.textContent = "no clean lap yet"; shEl.style.color = ""; }
    const evo = 319.55;
    const fill = $("#fga-lapbar-fill");
    if (fill) {
      const pct = lap ? Math.max(0, Math.min(100, evo / Number(lap) * 100)) : 0;
      fill.style.width = pct.toFixed(1) + "%";
      fill.classList.toggle("superhuman", !!ev.superhuman);
      const mt = $("#fga-mark-theo"), mb = $("#fga-mark-bellof");
      if (mt && theo) mt.style.left = Math.min(99, evo / Number(theo) * 100).toFixed(1) + "%";
      if (mb) mb.style.left = Math.min(99, evo / bellof * 100).toFixed(1) + "%";
    }
    $("#fga-progress").textContent = ev.max_progress_m == null ? "--"
      : `${Math.round(ev.max_progress_m)} m (${((ev.progress_frac || 0) * 100).toFixed(1)}%)`;
    $("#fga-terminal").textContent = ev.terminal_rate == null ? "--" : Number(ev.terminal_rate).toFixed(2);
    $("#fga-genome").textContent = snap.genome_size ? Number(snap.genome_size).toLocaleString() : "--";
    $("#fga-popsize").textContent = snap.pop_size ?? "--";
    const mini = $("#fga-mini-metrics");
    if (mini) {
      mini.innerHTML = [
        ["optimizer", "genetic algorithm"],
        ["hidden", (snap.hidden || []).join("×") || "--"],
        ["sigma", snap.sigma == null ? "--" : Number(snap.sigma).toFixed(3)],
        ["stage", snap.stage || ev.ga_stage || "--"],
        ["mean fitness", snap.mean_fitness == null ? "--" : Number(snap.mean_fitness).toFixed(1)],
        ["vs Bellof", ev.vs_bellof == null ? "--" : `${(Number(ev.vs_bellof) * 100).toFixed(1)}%`],
      ].map(([a, b]) => `<div><span>${esc(a)}</span><b>${esc(b)}</b></div>`).join("");
    }
  }).catch(() => {});
}
function initFableGA() {
  if (!$("#tab-fablega")) return;
  seg($("#fga-stage"), [
    {v:"foundation", l:"1 Foundation"}, {v:"flow", l:"2 Flow"},
    {v:"finish", l:"3 Finish"}, {v:"fast", l:"4 Fast"},
    {v:"frontier", l:"5 Frontier"}], "frontier");
  const bindRange = (id, out, dflt) => {
    const el = $(`#${id}`), label = $(`#${out}`);
    if (!el || !label) return;
    el.oninput = e => { label.textContent = +e.target.value > 0 ? e.target.value : (dflt || "off"); };
  };
  bindRange("fga-gens", "fga-gv");
  bindRange("fga-pop", "fga-pv");
  bindRange("fga-workers", "fga-wv");
  bindRange("fga-minutes", "fga-mv", "off");
  bindRange("fga-eval", "fga-ev");
  $("#fga-refresh").onclick = () => { loadWatchCkpts(); loadFableGAState(); };
  $("#fga-start").onclick = () => launch("fable_ga_train", gaParams(), `fable-ga ${gaStage()}`);
  $("#fga-continue").onclick = () => launch("fable_ga_continue",
    gaParams({checkpoint: $("#fga-resume").value || "ga_787b_ring.npz"}), "fable-ga continue");
  $("#fga-live").onclick = () => launch("fable_ga_live", gaParams(), "follow live GA training");
  $("#fga-stop").onclick = () => { if (activePid && TASKS[activePid]?.live) api(`/api/stop/${activePid}`, {method:"POST"}).then(() => toast("■ stopping Fable GA run")); };
  $("#fga-watch-best").onclick = () => launch("fable_ga_watch", gaParams(), "watch GA champion");
  $("#fga-watch-selected").onclick = () => launch("fable_ga_watch", gaParams({checkpoint: gaCheckpoint()}), "watch GA");
  populateGaSelects();
  loadFableGAState();
}

function fmtTime(ts) {
  if (!ts) return "unknown time";
  const d = new Date(ts * 1000);
  return d.toLocaleString([], {month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"});
}
function fmtLap(v) {
  if (v == null) return "none";
  const m = Math.floor(v / 60);
  const s = (v - m * 60).toFixed(2).padStart(5, "0");
  return `${m}:${s}`;
}
function checkpointSummary(c) {
  if (!c) return "No checkpoint selected";
  if (c.error) return c.error;
  if (c.kind === "ga") return `GA · gen ${c.generation} · fit ${c.fitness} · ${c.car}`;
  const arena = c.train_arena ? ` · ${c.train_arena}` : "";
  const track = c.track ? ` · ${c.track}` : " · generalist";
  const metric = c.kind === "drift" ? `drift ${c.drift} · lap ${c.laps}`
    : c.kind === "hybrid" ? `lap ${c.laps} · style ${c.drift}`
    : `lap ${c.laps}`;
  return `${c.kind} · ${c.updates || 0}u${track}${arena} · ${metric}`;
}
function recentCheckpoints(limit=4) {
  return (WATCH_CK || []).slice()
    .filter(c => c.name && c.name.endsWith(".pt"))
    .sort((a, b) => (b.mtime || 0) - (a.mtime || 0))
    .slice(0, limit);
}
function ringCandidate() {
  const pts = (WATCH_CK || []).filter(c => c.name && c.name.endsWith(".pt"));
  const ringish = pts.filter(c => c.track === RING_TRACK ||
    String(c.track_profile || "").includes("nordschleife") ||
    /nord|ring|nurb|nuer/i.test(c.name));
  return ringish.find(c => isBestFile(c.name)) || ringish[0] ||
    WATCH_CK.find(c => c.name === ringBestName()) || null;
}
function diagnosticVerdict(d) {
  const label = d.best_clean_lap_time ? "clean-lap"
    : d.verdict || (d.termination_count ? "failure" : "developing");
  return label;
}
function renderOps() {
  const runs = $("#ops-active-runs");
  if (runs) {
    const live = (LAST_STATUS.tasks || []).filter(t => t.running);
    runs.innerHTML = live.length ? live.slice(0, 5).map(t => `
      <div class="ops-run">
        <i class="ops-run-state"></i>
        <div><b>${esc(t.label || "run")}</b><span>${esc(t.cmd || "")}</span></div>
        <em class="ops-pill">${Math.floor((t.elapsed || 0) / 60)}m</em>
      </div>`).join("") : `<div class="ops-empty">No active runs. Launch from Train, Ring, Watch, or Race.</div>`;
  }

  const ck = $("#ops-checkpoints");
  if (ck) {
    const items = recentCheckpoints(5);
    ck.innerHTML = items.length ? items.map(c => `
      <div class="ops-mini">
        <span class="ops-pill">${esc(c.kind || "pt")}</span>
        <div><b>${esc(c.name)}${isBestFile(c.name) ? " ★" : ""}</b><span>${esc(checkpointSummary(c))}</span></div>
        <em class="ops-pill">${esc(c.car || "?")}</em>
      </div>`).join("") : `<div class="ops-empty">No checkpoints found yet.</div>`;
  }

  renderOpsDiagnostics();
}
function renderOpsDiagnostics() {
  const box = $("#ops-diagnostics");
  if (!box) return;
  const items = (DIAGNOSTICS || []).slice(0, 4);
  box.innerHTML = items.length ? items.map(d => {
    const verdict = diagnosticVerdict(d);
    const events = d.event_total ? `${d.event_total} events` : "no events";
    const term = d.run_count ? `${d.termination_count || 0}/${d.run_count} terminated` : "no runs";
    return `<div class="ops-mini">
      <span class="ops-pill ${esc(verdict)}">${esc(verdict)}</span>
      <div><b>${esc(d.checkpoint || d.name)}</b><span>${esc(d.track || "?")} · ${events} · ${term}</span></div>
      <em class="ops-pill">${esc(fmtTime(d.mtime))}</em>
    </div>`;
  }).join("") : `<div class="ops-empty">No diagnostic bundles exported yet.</div>`;
}
function renderDiagnostics() {
  const wrap = $("#diag-list");
  if (!wrap) return;
  if (!DIAGNOSTICS.length) {
    wrap.innerHTML = `<div class="ops-empty">No diagnostics found. Run Diagnose from a checkpoint or the Ring trainer.</div>`;
    return;
  }
  wrap.innerHTML = DIAGNOSTICS.map(d => {
    const verdict = diagnosticVerdict(d);
    const events = Object.entries(d.event_counts || {})
      .sort((a, b) => b[1] - a[1])
      .slice(0, 6)
      .map(([k, v]) => `<span>${esc(k)} ${esc(v)}</span>`)
      .join("") || "<span>no events</span>";
    return `<article class="diag-card">
      <div class="diag-card-head">
        <div>
          <h3>${esc(d.checkpoint || d.name)}</h3>
          <div class="diag-meta">${esc(d.name)}<br>${esc(d.track || "?")} · ${esc(d.scenario || "auto")} · ${esc(fmtTime(d.mtime))}</div>
        </div>
        <span class="ops-pill ${esc(verdict)}">${esc(verdict)}</span>
      </div>
      <div class="diag-stats">
        <div class="diag-stat"><span>Runs</span><b>${esc(d.run_count || 0)}</b></div>
        <div class="diag-stat"><span>Terminated</span><b>${esc(d.termination_count || 0)}</b></div>
        <div class="diag-stat"><span>Mean Speed</span><b>${esc((d.mean_speed * 3.6).toFixed(0))} km/h</b></div>
        <div class="diag-stat"><span>Clean Lap</span><b>${esc(fmtLap(d.best_clean_lap_time))}</b></div>
      </div>
      <div class="diag-events">${events}</div>
    </article>`;
  }).join("");
}
function loadDiagnostics() {
  api("/api/diagnostics").then(list => {
    DIAGNOSTICS = Array.isArray(list) ? list : [];
    renderDiagnostics();
    renderOpsDiagnostics();
  }).catch(() => {
    DIAGNOSTICS = [];
    renderDiagnostics();
    renderOpsDiagnostics();
  });
}

/* ---------- Fable Five Brain Lab ---------- */
let BRAIN_REPORTS = [];
let BRAIN_SELECTED = null;
let BRAIN_POLL = null;

function pct(v, d = 0) { return v == null ? "--" : `${(Number(v) * 100).toFixed(d)}%`; }

function populateBrainCkpts() {
  const sel = $("#brain-ck");
  if (!sel) return;
  const pts = fableCkpts(WATCH_CK);
  const prev = sel.value;
  const stageTag = c => c.fable_stage ? ` — ${esc(c.fable_stage)}` : "";
  sel.innerHTML = `<option value="${esc(FABLE_BEST)}">${esc(FABLE_BEST)} — pipeline best</option>` +
    pts.filter(c => c.name !== FABLE_BEST)
       .map(c => `<option value="${esc(c.name)}">${isBestFile(c.name) ? "★ " : ""}${esc(c.name)}${stageTag(c)}</option>`).join("");
  if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
}

function loadBrainReports(keepSelection = true) {
  return api("/api/fable-diag").then(list => {
    BRAIN_REPORTS = Array.isArray(list) ? list : [];
    renderBrainReports();
    if (!keepSelection || !BRAIN_SELECTED) {
      if (BRAIN_REPORTS.length && BRAIN_REPORTS[0].name) openBrainReport(BRAIN_REPORTS[0].name);
    }
  }).catch(() => { BRAIN_REPORTS = []; renderBrainReports(); });
}

function renderBrainReports() {
  const wrap = $("#brain-reports");
  if (!wrap) return;
  if (!BRAIN_REPORTS.length) {
    wrap.innerHTML = `<i class="brain-empty">No brain reports yet.</i>`;
    return;
  }
  wrap.innerHTML = BRAIN_REPORTS.map(d => {
    if (d.error) return `<div class="brain-repcard err">${esc(d.name)}<span>${esc(d.error)}</span></div>`;
    const fin = d.best_lap ? (d.best_lap < 371.13 ? "superhuman" : "finisher") : "nonfinisher";
    const head = d.best_lap ? fmtLap(d.best_lap) : `${pct(d.best_progress, 1)} lap`;
    const active = d.name === BRAIN_SELECTED ? " active" : "";
    return `<button class="brain-repcard${active}" data-rep="${esc(d.name)}">
      <div class="brain-repcard-top"><b>${esc(d.checkpoint || d.name)}</b>
        <span class="brain-pill ${fin}">${fin}</span></div>
      <div class="brain-repcard-meta">${esc(d.stage || "?")} · ${esc(head)} · ${esc(d.clean_sectors ?? "?")}/16 clean · ${esc(fmtTime(d.mtime))}</div>
    </button>`;
  }).join("");
  wrap.querySelectorAll("[data-rep]").forEach(b =>
    b.onclick = () => openBrainReport(b.dataset.rep));
}

function openBrainReport(name) {
  BRAIN_SELECTED = name;
  renderBrainReports();
  const box = $("#brain-report");
  if (box) box.innerHTML = `<i class="brain-empty">Loading ${esc(name)}…</i>`;
  api(`/api/fable-diag/${encodeURIComponent(name)}`).then(r => {
    if (r.error) { box.innerHTML = `<i class="brain-empty">${esc(r.error)}</i>`; return; }
    renderBrainReport(r);
  }).catch(e => { if (box) box.innerHTML = `<i class="brain-empty">${esc(String(e))}</i>`; });
}

function brainSpark(trace) {
  const s = trace && trace.s_m, v = trace && trace.v, vr = trace && trace.vref;
  if (!s || !v || !vr || s.length < 3) return "";
  const W = 640, H = 120, n = s.length;
  const maxS = s[n - 1] || 1, maxV = Math.max(...vr, ...v) || 1;
  const X = i => (s[i] / maxS) * W;
  const Y = val => H - (val / maxV) * (H - 6) - 3;
  const path = arr => arr.map((val, i) => `${i ? "L" : "M"}${X(i).toFixed(1)},${Y(val).toFixed(1)}`).join("");
  // off-track shading
  const off = trace.off || [];
  let bands = "", inOff = false, x0 = 0;
  for (let i = 0; i < n; i++) {
    if (off[i] && !inOff) { inOff = true; x0 = X(i); }
    else if (!off[i] && inOff) { inOff = false; bands += `<rect x="${x0.toFixed(1)}" y="0" width="${(X(i) - x0).toFixed(1)}" height="${H}" fill="rgba(255,90,90,.16)"/>`; }
  }
  return `<svg class="brain-spark" viewBox="0 0 ${W} ${H}" preserveAspectRatio="none">
    ${bands}
    <path d="${path(vr)}" fill="none" stroke="rgba(120,160,255,.55)" stroke-width="1.5"/>
    <path d="${path(v)}" fill="none" stroke="#ffd24a" stroke-width="1.8"/>
  </svg>`;
}

function brainBars(hist, opts = {}) {
  const entries = Object.entries(hist || {});
  if (!entries.length) return "<i>none</i>";
  const max = Math.max(...entries.map(e => e[1])) || 1;
  return `<div class="brain-bars">` + entries.map(([k, val]) =>
    `<div class="brain-bar"><span class="brain-bar-k">${esc(opts.klabel ? opts.klabel(k) : k)}</span>
      <span class="brain-bar-t"><i style="width:${(val / max * 100).toFixed(1)}%"></i></span>
      <span class="brain-bar-v">${pct(val, 0)}</span></div>`).join("") + `</div>`;
}

function renderBrainReport(r) {
  const box = $("#brain-report");
  if (!box) return;
  const cp = r.checkpoint || {}, beh = r.behavior || {}, env = r.envelope || {};
  const laps = r.line_laps || [], sectors = r.sectors || [], closure = r.closure || [];
  const deaths = r.deaths || [];
  const bestLap = laps.filter(l => l.lap_time).map(l => l.lap_time).sort((a, b) => a - b)[0];
  const fin = bestLap ? (bestLap < 371.13 ? "superhuman" : "finisher") : "nonfinisher";

  // death heat per sector
  const heat = {};
  deaths.forEach(d => { heat[d.sector] = (heat[d.sector] || 0) + 1; });
  const heatMax = Math.max(1, ...Object.values(heat));
  const deathMap = Array.from({length: 16}, (_, s) => {
    const c = heat[s] || 0;
    const a = c ? (0.2 + 0.8 * c / heatMax).toFixed(2) : 0;
    const nm = (sectors[s] && sectors[s].name) || "";
    return `<div class="brain-cell" title="S${s} ${esc(nm)} — ${c} death(s)"
      style="background:rgba(255,80,80,${a})">${s}${c ? `<b>${c}</b>` : ""}</div>`;
  }).join("");

  const findings = (r.findings || []).map(f => `<li>${esc(f)}</li>`).join("");
  const lapRows = laps.map(l => `<div class="brain-lap">
    <span class="brain-lap-n">${esc(l.name)}</span>
    <b>${l.clean_lap ? fmtLap(l.lap_time) : `${pct(l.progress_frac, 1)} then ${esc(l.termination_reason || "?")}`}</b>
    <span>pace ${pct(l.mean_pace_on_track, 0)}</span></div>`).join("");
  const closRows = closure.map(c => `<div class="brain-clos ${c.crossed_line && c.clean ? "ok" : "bad"}">
    <span>${Math.round(c.start_frac * 100)}%→line</span>
    <b>${c.crossed_line && c.clean ? "carried home" : (c.termination_reason || "incomplete")}</b></div>`).join("");

  const sectorRows = sectors.map(s => {
    const bad = !s.clean;
    return `<tr class="${bad ? "bad" : ""}">
      <td>${s.sector}</td><td class="brain-secname">${esc(s.name)}</td>
      <td>${s.clean ? "✓" : "✗ " + esc(s.reason || "")}</td>
      <td>${s.line_pace != null ? pct(s.line_pace, 0) : "--"}</td>
      <td>${s.pace != null ? pct(s.pace, 0) : "--"}</td>
      <td>${s.slip_p95 != null ? Number(s.slip_p95).toFixed(0) + "°" : "--"}</td></tr>`;
  }).join("");

  const deathRows = deaths.map(d => `<div class="brain-death">
    <span class="brain-dtag t-${esc(d.tag)}">${esc(d.tag)}</span>
    <b>S${d.sector} ${esc(d.sector_name)}</b>
    <span>${esc(d.reason)} · entry ${pct(d.entry_pace, 0)} · slip ${Number(d.max_slip_last3s).toFixed(0)}°</span></div>`).join("");

  box.innerHTML = `
    <div class="brain-rhead">
      <div><span class="brain-pill ${fin}">${fin}</span>
        <h3>${esc(cp.name || "brain")}</h3>
        <div class="brain-rmeta">stage ${esc(cp.stage || "?")} · scale ${esc(cp.envelope_scale ?? "?")} · ${esc(cp.updates ?? "?")} updates · act ${esc(cp.act_dim ?? "?")} · ${esc(r.wall_seconds ?? "?")}s battery</div></div>
      <div class="brain-verdict">${esc(r.verdict || "")}</div>
    </div>

    <div class="brain-sec"><h4>What it does</h4><ul class="brain-findings">${findings}</ul></div>

    <div class="brain-sec"><h4>Speed vs physics envelope <small>(gold = car, blue = envelope, red = off-track) — race-dropin line lap</small></h4>
      ${brainSpark(r.trace) || "<i>no trace</i>"}</div>

    <div class="brain-grid2">
      <div class="brain-sec"><h4>Flying line laps</h4>${lapRows || "<i>none</i>"}
        <h4 style="margin-top:12px">Lap-closure drills</h4><div class="brain-clos-row">${closRows || "<i>none</i>"}</div></div>
      <div class="brain-sec"><h4>Death map <small>(${deaths.length} terminations)</small></h4>
        <div class="brain-deathmap">${deathMap}</div>
        <div class="brain-deaths">${deathRows || "<i>zero terminations</i>"}</div></div>
    </div>

    <div class="brain-grid2">
      <div class="brain-sec"><h4>Behavior</h4>
        <div class="brain-stats">
          <div><span>On-track pace</span><b>${pct(beh.mean_pace_on_track, 1)}</b></div>
          <div><span>Over envelope</span><b>${pct(beh.time_over_envelope, 1)}</b></div>
          <div><span>Full throttle</span><b>${pct(beh.full_throttle_frac, 0)}</b></div>
          <div><span>Braking</span><b>${pct(beh.braking_frac, 0)}</b></div>
          <div><span>Coasting</span><b>${pct(beh.coasting_frac, 0)}</b></div>
          <div><span>Off-track time</span><b>${pct(beh.offtrack_time_frac, 2)}</b></div>
          <div><span>Steer reversals/km</span><b>${esc(beh.steer_reversals_per_km ?? "--")}</b></div>
          <div><span>Steer saturation</span><b>${pct(beh.steer_saturation_frac, 0)}</b></div>
          <div><span>Gear offset (mean)</span><b>${esc(beh.gear_offset_mean ?? "--")}</b></div>
          <div><span>Downshifts</span><b>${esc(beh.shifts_down ?? "--")}</b></div>
        </div>
        <h4 style="margin-top:12px">Pace distribution <small>(share of on-track time)</small></h4>
        ${brainBars(beh.pace_histogram)}
        <h4 style="margin-top:12px">Gear usage</h4>
        ${brainBars(beh.gear_histogram, {klabel: k => "G" + k})}
      </div>
      <div class="brain-sec"><h4>Sector battery</h4>
        <table class="brain-sectbl"><thead><tr><th>#</th><th>Sector</th><th>Result</th><th>line pace</th><th>start pace</th><th>slip</th></tr></thead>
        <tbody>${sectorRows}</tbody></table></div>
    </div>`;
}

function runBrainDiag() {
  const sel = $("#brain-ck");
  const cp = (sel && sel.value) || FABLE_BEST;
  const st = $("#brain-status");
  launch("fable_diagnose", {checkpoint: cp, car: RING_CAR}, `brain diag ${cp}`);
  if (st) if (st) st.textContent = `running battery on ${cp} — reports refresh automatically…`;
  // the battery takes ~2 min; poll the report list until a newer one lands
  if (BRAIN_POLL) clearInterval(BRAIN_POLL);
  const before = BRAIN_REPORTS.length ? BRAIN_REPORTS[0].name : null;
  let ticks = 0;
  BRAIN_POLL = setInterval(() => {
    ticks++;
    api("/api/fable-diag").then(list => {
      const top = Array.isArray(list) && list.length ? list[0].name : null;
      if (top && top !== before) {
        clearInterval(BRAIN_POLL); BRAIN_POLL = null;
        if (st) if (st) st.textContent = "";
        BRAIN_REPORTS = list; renderBrainReports(); openBrainReport(top);
      }
    }).catch(() => {});
    if (ticks > 120) { clearInterval(BRAIN_POLL); BRAIN_POLL = null; if (st) if (st) st.textContent = ""; }
  }, 4000);
}

function initBrainLab() {
  if (!$("#tab-diagnostics")) return;
  if ($("#brain-run")) $("#brain-run").onclick = runBrainDiag;
  if ($("#brain-refresh")) $("#brain-refresh").onclick = () => { loadWatchCkpts(); loadBrainReports(); };
  populateBrainCkpts();
  loadBrainReports(false);
}

function buildTrain() {
  const wrap = $("#train-cards"); wrap.innerHTML = "";
  TRAIN.forEach(t => {
    const c = document.createElement("div"); c.className = "tc";
    c.innerHTML = `<h2>${t.title}</h2><div class="sub">${t.sub}</div>
      <div class="ctl"><label>${t.iters.label}: <b id="${t.key}-iv">${t.iters.def}</b></label>
        <input type="range" id="${t.key}-iters" min="${t.iters.min}" max="${t.iters.max}" step="${t.iters.step}" value="${t.iters.def}" style="width:100%"></div>
      ${t.extra ? `<div class="ctl"><label>Population: <b id="${t.key}-pv">60</b></label><input type="range" id="${t.key}-pop" min="20" max="120" step="5" value="60" style="width:100%"></div>` : ""}
      <div class="ctl"><label>Chassis</label><div class="seg" id="${t.key}-car"></div></div>
      <div class="car-preview train-preview" id="${t.key}-car-preview" aria-live="polite"></div>
      <div class="ctl"><label>Terrain <span style="color:#7b8190;font-weight:400">(flat 2D physics by default)</span></label>
        <div class="seg" id="${t.key}-terrain"></div></div>
      <div class="ctl" id="${t.key}-hs-wrap"><label>Hill scale: <b id="${t.key}-hv">1.0</b> <span style="color:#7b8190;font-weight:400">(0.2 gentle → 1.5 exaggerated)</span></label>
        <input type="range" id="${t.key}-hillscale" min="0.2" max="1.5" step="0.1" value="1.0" style="width:100%"></div>
      ${t.spec ? `<div class="ctl"><label>Train on</label><select id="${t.key}-track">
          <option value="pool">🌐 Generalist — track pool (curriculum)</option>
          ${TRACKS.map(tr => `<option value="${tr.name}">🎯 Specialist — ${tr.name} (${tr.desc})</option>`).join("")}
        </select></div>
      <div class="ctl"><label>Run name (saved checkpoint)</label>
        <input id="${t.key}-out" placeholder="${t.dflt}" autocomplete="off"></div>` : ""}
      ${t.target ? `<div class="ctl"><label>Graduate to <span style="color:#7b8190;font-weight:400">generalist → fine-tune this track after the wide→tight curriculum</span></label>
        <select id="${t.key}-target"><option value="none">— none (stay generalist) —</option>
        ${TRACKS.filter(tr => !["oval","random","touge"].includes(tr.name)).map(tr => `<option value="${tr.name}">🎓 ${tr.name} (${tr.desc})</option>`).join("")}</select></div>` : ""}
      ${t.anneal ? `<div class="ctl"><label>Anneal LR + entropy <span style="color:#7b8190;font-weight:400">(sharpen late — best for long runs)</span></label><div class="seg" id="${t.key}-anneal"></div></div>` : ""}
      ${t.workers ? `<div class="ctl"><label>Parallel workers: <b id="${t.key}-wv">1</b> <span style="color:#7b8190;font-weight:400">(faster; n_envs auto-scales)</span></label>
        <input type="range" id="${t.key}-workers" min="1" max="8" step="1" value="1" style="width:100%"></div>` : ""}
      ${t.reseed ? `<div class="ctl"><label>Auto-reseed on plateau: <b id="${t.key}-mrv">3</b> tries <span style="color:#7b8190;font-weight:400">(no new ★best → reseed from best w/ fresh exploration; 0 = just stop)</span></label>
        <input type="range" id="${t.key}-maxrestarts" min="0" max="6" step="1" value="3" style="width:100%"></div>
      <div class="ctl"><label>Plateau patience: <b id="${t.key}-patv">500</b> iters <span style="color:#7b8190;font-weight:400">(applies once the curriculum is maxed / on a specialist track)</span></label>
        <input type="range" id="${t.key}-patience" min="100" max="1500" step="50" value="500" style="width:100%"></div>` : ""}
      ${t.style ? `<div class="ctl"><label>Drift style: <b id="${t.key}-sv">0.03</b> <span style="color:#7b8190;font-weight:400">(bigger = driftier corners, a bit slower)</span></label>
        <input type="range" id="${t.key}-style" min="0.005" max="0.12" step="0.005" value="0.03" style="width:100%"></div>` : ""}
      ${t.opponents ? `<div class="ctl"><label>Opponents <span style="color:#7b8190;font-weight:400">(historical checkpoints to race against)</span></label>
        <div style="display:flex;gap:4px">
          <select id="${t.key}-opp1" class="opp-select"><option value="none">none</option></select>
          <select id="${t.key}-opp2" class="opp-select"><option value="none">none</option></select>
          <select id="${t.key}-opp3" class="opp-select"><option value="none">none</option></select>
        </div></div>` : ""}
      <div class="ctl resume-ctl" id="${t.key}-resume-wrap" style="display:none"><label>Continue from <span style="color:#7b8190;font-weight:400">★ = saved peak — continue from this, not the latest (it can over-train)</span></label><select id="${t.key}-ck"></select></div>
      <div class="actions" id="${t.key}-acts"></div>`;
    wrap.appendChild(c);
    const trainDefault = defaultCar();
    seg($(`#${t.key}-car`), carOptions(), trainDefault, v => renderCarPreview($(`#${t.key}-car-preview`), v, "Training chassis"));
    renderCarPreview($(`#${t.key}-car-preview`), trainDefault, "Training chassis");
    $(`#${t.key}-iters`).oninput = e => $(`#${t.key}-iv`).textContent = e.target.value;
    seg($(`#${t.key}-terrain`), [{v: "flat", l: "▭ Flat"}, {v: "hills", l: "Hills"}], "hills",
      v => $(`#${t.key}-hs-wrap`).style.display = v === "flat" ? "none" : "block");
    $(`#${t.key}-hs-wrap`).style.display = "block";
    $(`#${t.key}-hillscale`).oninput = e => $(`#${t.key}-hv`).textContent = (+e.target.value).toFixed(1);
    if (t.extra) $(`#${t.key}-pop`).oninput = e => $(`#${t.key}-pv`).textContent = e.target.value;
    if (t.anneal) seg($(`#${t.key}-anneal`), [{v: "off", l: "Off"}, {v: "on", l: "On"}], "off");
    if (t.workers) $(`#${t.key}-workers`).oninput = e => $(`#${t.key}-wv`).textContent = e.target.value;
    if (t.reseed) {
      $(`#${t.key}-maxrestarts`).oninput = e => $(`#${t.key}-mrv`).textContent = e.target.value;
      $(`#${t.key}-patience`).oninput = e => $(`#${t.key}-patv`).textContent = e.target.value;
    }
    if (t.style) $(`#${t.key}-style`).oninput = e => $(`#${t.key}-sv`).textContent = (+e.target.value).toFixed(3);
    if (t.spec) {  // pick a specialist track -> suggest a matching run name
      const trkSel = $(`#${t.key}-track`), outIn = $(`#${t.key}-out`);
      trkSel.onchange = () => {
        const v = trkSel.value;
        if (v === "pool") { outIn.value = ""; outIn.placeholder = t.dflt; }
        else { outIn.value = `${v}_${t.key}${t.ext || ".pt"}`; }
      };
    }
    if (t.opponents) {
      api("/api/checkpoints").then(list => {
        const pts = list.filter(c => c.name.endsWith(".pt"));
        const opts = `<option value="none">none</option>` + pts.map(c => `<option value="${c.name}">${c.name}</option>`).join("");
        [1, 2, 3].forEach(i => {
           const s = $(`#${t.key}-opp${i}`);
           if(s) s.innerHTML = opts;
        });
      });
    }
    const acts = $(`#${t.key}-acts`);
    if (t.key !== "ga") {   // PPO policies drive the 3D viewer; GA brains are 2D
      const lv = document.createElement("button");
      lv.textContent = "🧠 Live 3D";
      lv.title = "Open the 3D viewer on this run's rolling checkpoint — the brain "
        + "hot-reloads every training iteration, so you watch it learn live";
      lv.onclick = () => {
        const out = (t.spec ? $(`#${t.key}-out`).value.trim() : "") || t.ck;
        const sel = t.spec ? $(`#${t.key}-track`).value : "pool";
        const trk = sel !== "pool" ? sel : "club";
        window.open(`/3d/?checkpoint=${encodeURIComponent(out)}&track=${encodeURIComponent(trk)}`, "_blank");
      };
      acts.appendChild(lv);
    }
    t.modes.forEach(m => {
      const b = document.createElement("button"); b.textContent = m.l;
      b.onclick = () => {
        const p = {iters: $(`#${t.key}-iters`).value, car: segVal($(`#${t.key}-car`))};
        if (t.extra) { p.pop = $(`#${t.key}-pop`).value; p.seed = 7; }
        if (t.spec) { p.track = $(`#${t.key}-track`).value; p.out = $(`#${t.key}-out`).value.trim(); }
        const realTrack = isRealElevationTrack(p.track);
        if (segVal($(`#${t.key}-terrain`)) === "flat" && !realTrack) p.flat = true;
        else { p.hills = true; p.hill_scale = $(`#${t.key}-hillscale`).value; }
        if (t.anneal && segVal($(`#${t.key}-anneal`)) === "on") p.anneal = true;
        if (t.workers) p.workers = $(`#${t.key}-workers`).value;
        if (t.reseed) { p.max_restarts = $(`#${t.key}-maxrestarts`).value; p.patience = $(`#${t.key}-patience`).value; }
        if (t.target) p.target = $(`#${t.key}-target`).value;
        if (t.style) p.style = $(`#${t.key}-style`).value;
        if (t.opponents) {
           const o1 = $(`#${t.key}-opp1`).value, o2 = $(`#${t.key}-opp2`).value, o3 = $(`#${t.key}-opp3`).value;
           const opps = [o1, o2, o3].filter(x => x !== "none").join(",");
           if (opps) p.opponents = opps;
        }
        if (m.resume) p.checkpoint = $(`#${t.key}-ck`).value || t.ck;
        const tag = (t.spec && p.track !== "pool") ? `${m.a}:${p.track}` : m.a;
        launch(m.a, p, tag);
      };
      acts.appendChild(b);
    });
    if (t.ck) {  // show resume picker
      $(`#${t.key}-resume-wrap`).style.display = "block";
      populateCkSelect(t.key, t.ck, t.ext || ".pt");
    }
  });
}
function ckPerf(c) {       // honest eval numbers for a checkpoint option
  if (c.kind === "ga") return `gen ${c.generation} · fit ${c.fitness}`;
  const m = c.kind === "drift" ? `drift ${c.drift} · lap ${c.laps}`
    : c.kind === "hybrid" ? `lap ${c.laps} · style ${c.drift}`
    : `lap ${c.laps}`;
  return `${c.updates || 0}u · ${m}${c.track ? " · 🎯" + c.track : ""}`;
}
function populateCkSelect(key, prefKind, ext) {
  api("/api/checkpoints").then(list => {
    const sel = $(`#${key}-ck`); if (!sel) return;
    const pts = list.filter(c => c.name.endsWith(ext || ".pt"));
    sel.innerHTML = pts.map(c =>
      `<option value="${c.name}">${c.layout === "pre-hills" ? "⚠pre-hills · " : ""}${isBestFile(c.name) ? "★ " : ""}${c.name} — ${ckPerf(c)}</option>`
    ).join("");
    // default to the ★best peak of the slot if it exists (continue from the PEAK,
    // not the latest — the latest can have over-trained); else the slot itself.
    const bestPref = bestNameOf(prefKind);
    if (pts.some(c => c.name === bestPref)) sel.value = bestPref;
    else if (pts.some(c => c.name === prefKind)) sel.value = prefKind;
  });
}

/* ---------- CHECKPOINTS ---------- */
function loadCheckpoints() {
  api("/api/checkpoints").then(list => {
    const wrap = $("#ck-list"); wrap.innerHTML = "";
    list.forEach(c => {
      const div = document.createElement("div");
      div.className = "ckrow" + (c.active ? " active" : "");
      const spec = c.track ? `🎯 ${c.track}` : "🌐 generalist";
      // surface the HONEST eval numbers (drift/lap for drift, lap for race)
      const perf = c.kind === "drift" ? ` · drift ${c.drift} · lap ${c.laps}`
                 : c.kind === "hybrid" ? ` · lap ${c.laps} · style ${c.drift}`
                 : c.kind === "race" ? ` · lap ${c.laps}` : "";
      const meta = c.kind === "ga"
        ? `gen ${c.generation} · best ${c.fitness} · ${c.car}`
        : `${c.kind} · ${spec} · ${c.updates}u · diff ${c.difficulty}${perf} · ${c.car}`;
      const preHills = c.layout === "pre-hills";
      const layoutBadge = c.layout
        ? (preHills
          ? `<span class="ck-badge prehills" title="incompatible — pre-hills obs ${c.obs}; the obs vector grew when hills/jumps landed. Retrain (see PHYSICS_3D_PLAN.md)">⚠ pre-hills</span>`
          : `<span class="ck-badge hills" title="hills-era checkpoint (obs ${c.obs})">hills-v1</span>`)
        : "";
      div.innerHTML = `<span class="ck-name">${c.name}${isBestFile(c.name) ? " ★" : ""}</span>
        <span class="ck-badge">${c.kind||"?"}</span>${layoutBadge}
        <span class="ck-meta">${c.error ? "⚠ "+c.error : meta} · ${c.size_kb}KB</span>
        <span class="ck-acts"></span>`;
      const acts = $(".ck-acts", div);
      const mk = (label, fn) => { const b = document.createElement("button"); b.className="mini"; b.textContent=label; b.onclick=fn; acts.appendChild(b); };
      mk("backup", () => ckOp("backup", c.name, prompt("Backup as:", c.name.replace(/\.(pt|npz)$/, "_copy.$1"))));
      mk("rename", () => ckOp("rename", c.name, prompt("Rename to:", c.name)));
      if (c.name.endsWith(".pt")) {
        const guard = fn => () => {
          if (preHills && !confirm(
            `${c.name} is a PRE-HILLS checkpoint (obs ${c.obs}) — the loaders ` +
            `will refuse it. It's a museum piece; retrain for hills.\n\nActivate anyway?`)) return;
          fn();
        };
        mk("→race", guard(() => ckOp("activate", c.name, "ppo_race.pt")));
        mk("→drift", guard(() => ckOp("activate", c.name, "ppo_drift.pt")));
      }
      mk("diagnose", () => diagnoseCk(c));
      if (!c.active) mk("delete", () => confirm(`Delete ${c.name}?`) && ckOp("delete", c.name));
      wrap.appendChild(div);
    });
  });
}
function diagnoseCk(c) {
  const track = prompt("Diagnostic track:", c.track || "club");
  if (!track) return;
  const scenario = prompt("Scenario: auto, solo, frozen_opponents, true_multi", "auto") || "auto";
  const episodes = prompt("Episodes:", "4") || "4";
  const maxSteps = prompt("Max control steps per episode (blank = full episode):", "1200");
  const opponents = prompt("Opponent checkpoints, comma-separated (optional):", "") || "";
  launch("diagnose_checkpoint", {
    checkpoint: c.name,
    track,
    scenario,
    episodes,
    max_steps: maxSteps || "",
    opponents,
    ...(isRealElevationTrack(track) ? {} : {flat: true}),
  }, `diag:${c.name}`);
}
function ckOp(op, name, dest) {
  if ((op === "backup" || op === "rename" || op === "activate") && !dest) return;
  api("/api/checkpoint", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({op, name, dest})}).then(r => {
    if (r.error) toast("✗ " + r.error); else { toast("✓ " + op); loadCheckpoints(); loadWatchCkpts(); populateRingSelect(); TRAIN.forEach(t => t.ck && populateCkSelect(t.key, t.ck, t.ext || ".pt")); }
  });
}

/* ---------- LAUNCH + DOCK ---------- */
const TASKS = {}; let activePid = null;
window.pwActiveTask = () => activePid != null ? TASKS[activePid] : null;   // Pit Wall (pitwall.js) reads the dock's live stream through this
let chartPerf = null; let chartHealth = null; let dockView = "log";
function launch(action, params, title) {
  params = {...(params || {})};
  if (!params.hills && !isRealElevationTrack(params.track)) params.flat = true;
  api("/api/launch", {method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify({action, params})}).then(r => {
    if (r.error) return toast("✗ " + r.error);
    toast((r.gui ? "🪟 window launching… " : "▶ ") + title);
    TASKS[r.pid] = {label: title, pid: r.pid, log: "", metrics: [], live: true};
    openStream(r.pid);
    $("#dock").classList.remove("hidden");
    if (window.PitWall) window.PitWall.setActive(true);
    selectTask(r.pid); renderDockTabs();
    // headless training -> jump straight to live charts (that's the payoff);
    // a GUI run (drive/live/watch) opens its own window, so show the log here.
    setDockView(r.gui ? "log" : "chart");
  });
}
function openStream(pid) {
  const T = TASKS[pid]; if (!T || T.es) return;
  const es = new EventSource(`/api/stream/${pid}`);
  T.es = es;
  es.onmessage = e => {
    const d = JSON.parse(e.data);
    if (!TASKS[pid]) return;
    if (d.snapshot) {                 // full resync (first frame / after reconnect)
      T.log = d.text || ""; T.metrics = (d.history || []).slice();
      if (d.running === false) { T.live = false; es.close(); T.es = null; renderDockTabs(); }
    } else {
      if (d.text) T.log += d.text;
      if (d.metric) {     // merge by iteration so the [eval] point rides on its iter
        const m = d.metric, arr = T.metrics, last = arr[arr.length - 1];
        // plateau auto-restart events: track on the task + flag it for the badge
        if (m.restart !== undefined) {
          T.restart = m.restart; T.maxRestarts = m.max_restarts;
          if (pid === activePid) toast(`↻ reseed ${m.restart}/${m.max_restarts} from best (plateau)`);
        }
        if (m.converged) {
          T.converged = true; T.convergedBest = m.converged_best;
          if (pid === activePid) toast(`✓ converged — no gain after ${T.maxRestarts || ""} reseeds`);
        }
        if (m.x !== undefined) { if (last && last.x === m.x) Object.assign(last, m); else arr.push(m); }
      }
      if (d.done) { T.live = false; es.close(); T.es = null; renderDockTabs(); }
    }
    if (pid === activePid) { renderConsole(); renderChart(); }
  };
  // a dropped connection (sleep / blip) must NOT kill the live view — reconnect
  // until the task is actually done. The server replays full history on connect,
  // so the display snaps back to the real current progress.
  es.onerror = () => {
    es.close(); T.es = null;
    if (T.live) setTimeout(() => openStream(pid), 2000);
  };
}
function reattach() {
  // on page load / refresh, re-bind the dock to any still-running tasks.
  api("/api/status").then(s => {
    (s.tasks || []).filter(t => t.running).forEach(t => {
      if (TASKS[t.pid]) return;
      TASKS[t.pid] = {label: t.label, pid: t.pid, log: "", metrics: [], live: true};
      openStream(t.pid);
    });
    const pids = Object.keys(TASKS);
    if (pids.length) {
      $("#dock").classList.remove("hidden");
      if (window.PitWall) window.PitWall.setActive(true);
      if (!activePid) selectTask(+pids[pids.length - 1]);
      renderDockTabs();
    }
  }).catch(() => {});
}
function renderDockTabs() {
  const wrap = $("#dock-tabs"); wrap.innerHTML = "";
  Object.values(TASKS).slice(-8).forEach(T => {
    const b = document.createElement("div");
    b.className = "dt" + (T.pid === activePid ? " on" : "") + (T.live ? " live" : "");
    b.textContent = T.label;
    b.onclick = () => { selectTask(T.pid); };
    wrap.appendChild(b);
  });
}
function selectTask(pid) { activePid = pid; renderDockTabs(); renderConsole(); renderChart(); }
const LOG_CLASSES = [
  [/\[pit\]/, "lg-pit"],
  [/\[eval/, "lg-eval"],
  [/\[best/, "lg-best"],
  [/\[gate\]|TRANSPLANTED|promoted|\[graduate\]/i, "lg-gate"],
  [/\[restart\]|\[converged\]|Traceback|Error|FAIL|not promoted/i, "lg-warn"],
];
function renderConsole() {
  const T = TASKS[activePid]; const c = $("#console");
  if (!T) { c.textContent = ""; return; }
  const lines = T.log.split("\n");
  // colorize the visible tail only (full log stays in memory)
  c.innerHTML = lines.slice(-400).map(l => {
    const cls = (LOG_CLASSES.find(([re]) => re.test(l)) || [0, ""])[1];
    return `<span class="lgl ${cls}">${esc(l)}</span>`;
  }).join("");
  c.scrollTop = c.scrollHeight;
  // "Pit Call" tile: surface the supervisor's latest decision + reason
  const pitEl = $("#td-pit");
  if (pitEl) {
    let pitLine = null;
    for (let i = lines.length - 1; i >= 0; i--) {
      if (lines[i].includes("[pit]")) { pitLine = lines[i]; break; }
    }
    if (pitLine) {
      const dec = (pitLine.match(/->\s*([A-Z]+)/) || [])[1] || "—";
      pitEl.textContent = dec;
      pitEl.style.color = dec === "RESEED" ? "#ffb84d" : "";
      const reason = (pitLine.match(/\(([^]*)\)\s*$/) || [])[1] || "pit wall";
      if ($("#td-pit-sub")) $("#td-pit-sub").textContent = reason.slice(0, 70);
    } else {
      pitEl.textContent = "—";
      if ($("#td-pit-sub")) $("#td-pit-sub").textContent = "no calls yet";
    }
  }
}
function renderChart() {
  const T = TASKS[activePid]; if (!T) return;
  const m = T.metrics;
  const statsPerf = $("#chart-stats-perf");
  const statsHealth = $("#chart-stats-health");
  if (!m.length) {
    if (statsPerf) statsPerf.innerHTML = "<div class='stat'>No metrics yet…</div>";
    if (chartPerf){chartPerf.destroy();chartPerf=null;}
    if (chartHealth){chartHealth.destroy();chartHealth=null;}
    return;
  }

  const keys = [...new Set(m.flatMap(o => Object.keys(o)))].filter(k => k !== "x");
  const colors = {
    eval_lap:"#3ddc84", eval_drift:"#39d0ff", eval_score:"#ffd24a",
    ring_metric:"#ff5a6f", fable_metric:"#ffb84d", pace_ratio:"#c78bff",
    clean_sectors:"#3ddc84", clean_progress_m:"#39d0ff",
    terminal_rate:"#ff8a4c", best_clean_lap:"#ffd24a", invalid_laps:"#ff5a6f",
    max_dottinger:"#a8d8ff", clean_chain:"#b7f26b", offtrack_seconds:"#ff8a4c",
    mean_speed:"#7cc7ff",
    return:"#5aa0e6", difficulty:"#c79a3a", laps:"#3f7a52",
    entropy:"#9a6fc0", value_loss:"#777", policy_loss:"#555"
  };
  const LABELS = {
    eval_lap:"eval lap ★", eval_drift:"eval drift ★", eval_score:"eval score ★",
    ring_metric:"ring score ★", fable_metric:"fable score ★", pace_ratio:"pace ratio",
    clean_sectors:"clean sectors", clean_progress_m:"clean progress m",
    terminal_rate:"terminal rate", best_clean_lap:"best clean lap", invalid_laps:"invalid laps",
    max_dottinger:"Dottinger m/s", clean_chain:"clean chain", offtrack_seconds:"off-track s",
    mean_speed:"mean speed",
    laps:"laps (rolling)", return:"return", difficulty:"difficulty",
    entropy:"entropy", value_loss:"value loss", policy_loss:"policy loss"
  };

  const perfKeys = ["ring_metric", "fable_metric", "clean_sectors", "clean_progress_m", "best_clean_lap", "mean_speed", "eval_lap", "eval_drift", "eval_score", "return", "laps"];
  const healthKeys = ["terminal_rate", "offtrack_seconds", "invalid_laps", "clean_chain", "pace_ratio", "max_dottinger", "entropy", "value_loss", "policy_loss", "difficulty"];

  const shownPerf = keys.filter(k => perfKeys.includes(k) && colors[k]).sort((a,b)=>(b.startsWith("eval_")-a.startsWith("eval_")));
  const shownHealth = keys.filter(k => healthKeys.includes(k) && colors[k]);

  const labels = m.map(o => o.x);
  const lastOf = k => { for (let i = m.length - 1; i >= 0; i--) if (m[i][k] != null) return m[i][k]; return null; };

  // Update big telemetry cards
  if ($("#td-laps")) {
    const evalV = lastOf("fable_metric") ?? lastOf("ring_metric")
      ?? lastOf("eval_lap") ?? lastOf("eval_drift") ?? lastOf("eval_score");
    $("#td-laps").textContent = evalV == null ? "--" : evalV.toFixed(2);
    if ($("#td-best")) {
      const series = m.map(o => o.fable_metric ?? o.ring_metric ?? o.eval_score
        ?? o.eval_lap).filter(v => v != null && isFinite(v));
      $("#td-best").textContent = series.length
        ? Math.max(...series).toFixed(2) : "--";
      const lapT = lastOf("eval_lap_time") ?? lastOf("best_clean_lap");
      $("#td-best-sub").textContent = lapT
        ? `best clean lap ${fmtLap(Number(lapT))}` : "banked peak";
    }

    const diffV = lastOf("difficulty");
    if ($("#td-diff")) $("#td-diff").textContent = diffV == null ? "--" : diffV.toFixed(2);

    // parse SPS from latest log line
    const lines = T.log.split("\n");
    let lastSps = "--";
    for(let i=lines.length-1; i>=0; i--) {
      if(lines[i].includes("sps")) {
        const match = lines[i].match(/(\d+)\s+sps/);
        if(match) { lastSps = match[1]; break; }
      }
    }
    if ($("#td-sps")) $("#td-sps").innerHTML = `${lastSps} <span style="font-size:0.5em;color:#7b8190">SPS</span>`;

    // plateau / reseed status (auto-restart-from-best pipeline)
    const rc = $("#td-reseed"), rsub = $("#td-reseed-sub");
    if (rc) {
      if (T.converged) {
        if (rc) rc.textContent = "converged ✓"; rc.style.color = "#3ddc84";
        if (rsub) if (rsub) rsub.textContent = T.convergedBest != null
          ? `peak ${(+T.convergedBest).toFixed(2)} · stopped` : "stopped";
      } else if (T.restart) {
        if (rc) rc.textContent = `${T.restart}/${T.maxRestarts}`; rc.style.color = "#ffd24a";
        if (rsub) if (rsub) rsub.textContent = "reseeds from best";
      } else {
        if (rc) rc.textContent = "—"; rc.style.color = "";
        if (rsub) if (rsub) rsub.textContent = "auto-restart from best";
      }
    }
    const st = $(".td-status-text");
    if (st) if (st) st.textContent = T.converged ? "CONVERGED" : (T.live ? "RUNNING" : "FINISHED");
  }

  const buildChart = (chartInstance, keysList, canvasId, statsEl) => {
    if (!keysList.length) return chartInstance;
    const datasets = keysList.map(k => {
      const isEval = k.startsWith("eval_");
      const label = LABELS[k] || k;
      let hidden = false;
      if (chartInstance) {
        const idx = chartInstance.data.datasets.findIndex(d => d.label === label);
        if (idx >= 0 && !chartInstance.isDatasetVisible(idx)) hidden = true;
      }
      return {label, data: m.map(o => o[k] ?? null), borderColor: colors[k],
        backgroundColor: "transparent", borderWidth: isEval ? 3 : 1.5,
        pointRadius: isEval ? 2.5 : 0, pointBackgroundColor: colors[k],
        tension: .25, spanGaps: true, hidden};
    });

    if (statsEl) {
      statsEl.innerHTML = keysList.map(k => {
        const v = lastOf(k);
        return `<div class="td-stat" style="color:${colors[k]}">${LABELS[k] || k}<b>${v == null ? "—" : (v.toFixed ? v.toFixed(2) : v)}</b></div>`;
      }).join("");
    }

    if (!chartInstance) {
      const mono = "ui-monospace, SFMono-Regular, Menlo, monospace";
      return new Chart($(canvasId), {type: "line", data: {labels, datasets},
        options: {animation: false, responsive: true, maintainAspectRatio: false,
          scales: {x: {ticks: {color: "#8a8578", maxTicksLimit: 8,
                               font: {family: mono, size: 10}},
                       grid: {color: "rgba(231,185,76,.07)"}},
            y: {ticks: {color: "#8a8578", font: {family: mono, size: 10}},
                grid: {color: "rgba(231,185,76,.07)"}}},
          plugins: {legend: {labels: {color: "#b9b29c", boxWidth: 10,
                                      font: {family: mono, size: 10}}}}}});
    } else {
      chartInstance.data.labels = labels;
      for (let i = 0; i < datasets.length; i++) {
        if (i < chartInstance.data.datasets.length) chartInstance.data.datasets[i].data = datasets[i].data;
        else chartInstance.data.datasets.push(datasets[i]);
      }
      if (chartInstance.data.datasets.length > datasets.length) chartInstance.data.datasets.splice(datasets.length);
      chartInstance.update("none");
      return chartInstance;
    }
  };

  chartPerf = buildChart(chartPerf, shownPerf, "#chart-canvas", statsPerf);
  chartHealth = buildChart(chartHealth, shownHealth, "#chart-canvas-health", statsHealth);
}
$("#dock-stop").onclick = () => { if (activePid && TASKS[activePid]?.live) api(`/api/stop/${activePid}`, {method:"POST"}).then(()=>toast("■ stopping (saving checkpoint)")); };
$("#dock-toggle").onclick = () => { 
  const d = $("#dock"); 
  const isHidden = d.classList.toggle("hidden"); 
  if (window.PitWall) window.PitWall.setActive(!isHidden);
};
const showTelemetry = () => {
  const d = $("#dock");
  d.classList.remove("hidden");
  if (window.PitWall) window.PitWall.setActive(true);
};
if ($("#ops-return-telemetry")) $("#ops-return-telemetry").onclick = showTelemetry;
if ($("#fb-return-telemetry")) $("#fb-return-telemetry").onclick = showTelemetry;
function setDockView(v) {
  // Obsolete: Full dashboard shows everything
}

/* ---------- status poll ---------- */
function pollStatus() {
  api("/api/status").then(s => {
    LAST_STATUS = s || {tasks: [], cpu: 0};
    const clock = $("#clock-chip");
    if (clock) clock.textContent = new Date().toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
    $("#cpu-chip").textContent = "CPU " + s.cpu + "%";
    const live = s.tasks.filter(t => t.running).length;
    $("#task-chip").textContent = live + " running";
    renderOps();
  }).catch(()=>{});
}
setInterval(pollStatus, 3000);

function bindOpsActions() {
  const byId = id => $(`#${id}`);
  if (byId("ops-refresh")) byId("ops-refresh").onclick = () => { pollStatus(); loadWatchCkpts(); loadDiagnostics(); };
  if (byId("ops-ring-live")) byId("ops-ring-live").onclick = () => launch("ring_race_live", ringRaceParams(), "ring race live");
  if (byId("ops-ring-diagnose")) byId("ops-ring-diagnose").onclick = () =>
    launch("ring_race_diagnose", ringRaceParams({checkpoint: RING_RACE_BEST, episodes: "1", max_steps: ""}), "diag ring race best");
  if (byId("ops-watch-best")) byId("ops-watch-best").onclick = () => {
    launch("ring_race_watch", ringRaceParams({checkpoint: RING_RACE_BEST}), "watch ring race best");
  };
  if (byId("ops-open-garage")) byId("ops-open-garage").onclick = () => activateTab("checkpoints");
  if (byId("ops-open-watch")) byId("ops-open-watch").onclick = () => activateTab("watch");
  if (byId("ops-open-diagnostics")) byId("ops-open-diagnostics").onclick = () => activateTab("diagnostics");
  if (byId("diag-refresh")) byId("diag-refresh").onclick = loadDiagnostics;
  if (byId("diag-ring-go")) byId("diag-ring-go").onclick = () =>
    launch("ring_ppo_diagnose", ringParams({checkpoint: ringCheckpoint(), episodes: "1", max_steps: ""}), "diag ring best");
  if (byId("diag-watch-go")) byId("diag-watch-go").onclick = () => activateTab("watch");
}

/* ---------- init ---------- */
function initAfterCars() {
  const driveDefault = defaultCar();
  seg($("#drive-car"), carOptions(), driveDefault, v => renderCarPreview($("#drive-car-preview"), v, "Drive chassis"));   // built from the authoritative car list
  renderCarPreview($("#drive-car-preview"), driveDefault, "Drive chassis");
  updateWatchPreview();
  api("/api/tracks").then(list => {
    TRACKS = list;
    fillGrid($("#drive-tracks"), null, "club");
    fillGrid($("#watch-tracks"), null, "club");
    fillGrid($("#race-tracks"), null, "national");
    const all = $("#all-tracks"); list.forEach(t => all.appendChild(trackCard(t, () => {})));
    initRingPipeline();
    initRingRace();
    initFableFive();
    initFableGA();
    initBrainLab();
    buildTrain();        // needs TRACKS for the specialist <select> AND CARS for chassis
  }).catch(() => { initRingPipeline(); initRingRace(); initFableFive(); initFableGA(); initBrainLab(); buildTrain(); });
}
// Cars come from the server (= real presets). Fall back to the hardcoded list if
// the endpoint is missing (older server), so the dashboard still works.
api("/api/cars").then(list => { if (Array.isArray(list) && list.length) CARS = list; })
  .catch(() => {})
  .finally(initAfterCars);
pollStatus();
loadWatchCkpts();    // populate the watch checkpoint picker
loadDiagnostics();   // populate the telemetry lab + ops summary
loadRingRaceState(); // populate Ring Race gate board
reattach();          // re-bind dock to running tasks after a refresh
bindOpsActions();
$("#ck-refresh").onclick = loadCheckpoints;

if ($("#td-start-viewer")) {
  $("#td-start-viewer").onclick = () => {
    $("#td-3d-iframe").src = "/3d/legacy/"; // Use legacy because it's stable and performs better for training overlay
    $("#td-viewer-overlay").classList.add("hidden");
  };
}

if ($("#quit-btn")) {
  $("#quit-btn").onclick = () => {
    if (confirm("Are you sure you want to quit the server?")) {
      fetch("/api/quit", { method: "POST" })
        .then(() => {
          document.body.innerHTML = "<h1 style='color:white;text-align:center;margin-top:20%'>Server has been shut down. You can close this tab.</h1>";
        })
        .catch(() => alert("Failed to quit server."));
    }
  };
}
