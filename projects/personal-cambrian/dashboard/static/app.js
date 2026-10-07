"use strict";
const $ = (s) => document.querySelector(s);
const api = (p, opt) => fetch(p, opt).then((r) => r.json());
const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? v : +v.toFixed(4)) : v);

const S = { actions: [], byCat: {}, cat: null, action: null, job: null, poll: null, since: 0, statusPoll: null };

// ---------- status ----------
async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const b = (label, on) => `<span class="b ${on ? "on" : "off"}">${label}</span>`;
    $("#status").innerHTML =
      `<span class="b">${s.cpus} CPU</span>` +
      b("torch", s.torch) + b("mujoco", s.mujoco) +
      `<span class="b ${s.mjx ? "on" : "off"}">MJX ${s.mjx ? "cpu" : "—"}</span>` +
      b("blender", s.blender) +
      `<span class="b jobs">${s.active_jobs} job${s.active_jobs === 1 ? "" : "s"}</span>`;
  } catch (e) {}
}

// ---------- load actions + rail ----------
async function init() {
  S.actions = await api("/api/actions");
  S.byCat = {};
  for (const a of S.actions) (S.byCat[a.category] = S.byCat[a.category] || []).push(a);
  const rail = $("#rail");
  rail.innerHTML = "";
  Object.keys(S.byCat).forEach((cat, i) => {
    const btn = document.createElement("button");
    btn.innerHTML = `<span class="dot"></span>${cat}`;
    btn.onclick = () => selectCat(cat, btn);
    rail.appendChild(btn);
    if (i === 0) selectCat(cat, btn);
  });
  refreshStatus();
  S.statusPoll = setInterval(refreshStatus, 5000);
  loadJobs();
  setInterval(() => { if ($('[data-tab="jobs"]').classList.contains("active")) loadJobs(); }, 4000);
  initTabs();
}

function selectCat(cat, btn) {
  S.cat = cat;
  document.querySelectorAll("#rail button").forEach((b) => b.classList.remove("active"));
  if (btn) btn.classList.add("active");
  $("#cat-title").textContent = cat;
  const list = $("#action-list");
  list.innerHTML = "";
  for (const a of S.byCat[cat]) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.innerHTML = a.label + (a.danger ? '<span class="danger-dot">●</span>' : "");
    chip.onclick = () => selectAction(a, chip);
    list.appendChild(chip);
  }
  selectAction(S.byCat[cat][0], list.firstChild);
}

// ---------- auto-generated form ----------
function selectAction(a, chip) {
  S.action = a;
  document.querySelectorAll("#action-list .chip").forEach((c) => c.classList.remove("active"));
  if (chip) chip.classList.add("active");
  $("#form-card").hidden = false;
  $("#action-label").textContent = a.label;
  $("#action-desc").textContent = a.description || "";
  const form = $("#form");
  form.innerHTML = "";
  for (const f of a.fields) form.appendChild(fieldEl(f));
  if (a.fields.length === 0) form.innerHTML = '<p class="muted" style="grid-column:1/-1;margin:0">No parameters.</p>';
}

function fieldEl(f) {
  const wrap = document.createElement("div");
  wrap.className = "field" + (f.type === "bool" ? " bool" : "");
  const id = "f_" + f.name;
  if (f.type === "bool") {
    wrap.innerHTML = `<label><input type="checkbox" id="${id}" ${f.default ? "checked" : ""}>${f.label}</label>`;
  } else if (f.type === "enum") {
    const opts = f.options.map((o) => `<option ${o === f.default ? "selected" : ""}>${o}</option>`).join("");
    wrap.innerHTML = `<label>${f.label}</label><select id="${id}">${opts}</select>`;
  } else if ((f.type === "int" || f.type === "float") && f.min != null && f.max != null) {
    const step = f.step != null ? f.step : (f.type === "int" ? 1 : 0.01);
    wrap.innerHTML = `<label>${f.label} <span class="rangeval" id="${id}_v">${f.default}</span></label>
      <div class="rangewrap"><input type="range" id="${id}" min="${f.min}" max="${f.max}" step="${step}" value="${f.default}"></div>`;
    setTimeout(() => { const r = $("#" + id); r.oninput = () => ($("#" + id + "_v").textContent = r.value); }, 0);
  } else {
    wrap.innerHTML = `<label>${f.label}</label><input id="${id}" type="${f.type === "str" ? "text" : "number"}" value="${f.default ?? ""}">`;
  }
  return wrap;
}

function collectParams() {
  const p = {};
  for (const f of S.action.fields) {
    const el = $("#f_" + f.name);
    if (!el) continue;
    p[f.name] = f.type === "bool" ? el.checked : el.value;
  }
  return p;
}

// ---------- run + poll ----------
async function runAction() {
  if (!S.action) return;
  if (S.action.danger && !confirm(`"${S.action.label}" can be heavy or use personal data. Run it?`)) return;
  if (S.poll) { clearInterval(S.poll); S.poll = null; }
  const { job_id } = await api(`/api/actions/${S.action.id}/run`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(collectParams()),
  });
  openJob(job_id, S.action);
  loadJobs();
}

function openJob(jobId, action) {
  S.job = jobId; S.since = 0; S.jobMetrics = []; S.jobAction = action;
  $("#job-card").hidden = false;
  $("#job-title").textContent = action ? action.label : jobId;
  $("#log").textContent = ""; $("#result").innerHTML = ""; $("#artifacts-inline").innerHTML = "";
  $("#chart").hidden = true;
  const ctx = $("#chart").getContext("2d"); ctx.clearRect(0, 0, $("#chart").width, $("#chart").height);
  if (S.poll) clearInterval(S.poll);
  pollJob();
  S.poll = setInterval(pollJob, 600);
  $("#job-card").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

async function pollJob() {
  if (!S.job) return;
  const d = await api(`/api/jobs/${S.job}/events?since=${S.since}`);
  S.since = d.next;
  const badge = $("#job-state");
  badge.className = "badge " + d.state; badge.textContent = d.state;
  $("#job-stage").textContent = d.stage || "";
  $("#cancel-btn").style.display = (d.state === "running" || d.state === "queued") ? "" : "none";
  const log = $("#log");
  for (const e of d.events) {
    if (e.kind === "metric") (S.jobMetrics = S.jobMetrics || []).push(e.data);
    if (e.kind === "log" || e.kind === "status") log.textContent += (e.msg || "") + "\n";
  }
  if ((S.jobMetrics || []).length) { $("#chart").hidden = false; drawChart(); }
  if (d.artifacts && d.artifacts.length) renderArtifacts($("#artifacts-inline"), d.artifacts);
  log.parentElement.scrollTop = 0; log.scrollTop = log.scrollHeight;
  if (["done", "error", "cancelled"].includes(d.state)) {
    clearInterval(S.poll); S.poll = null;
    renderResult(d.result, d.error);
    loadJobs(); refreshStatus(); loadGallery();
  }
}

async function cancelJob() { if (S.job) await api(`/api/jobs/${S.job}/cancel`, { method: "POST" }); }

// ---------- live chart ----------
function drawChart() {
  const cv = $("#chart"), ctx = cv.getContext("2d");
  const W = (cv.width = cv.clientWidth * devicePixelRatio), H = (cv.height = 170 * devicePixelRatio);
  ctx.scale(1, 1); ctx.clearRect(0, 0, W, H);
  const ms = S.jobMetrics || []; if (ms.length < 2) return;
  let keys = (S.jobAction && S.jobAction.streams) || [];
  keys = keys.filter((k) => ms.some((m) => typeof m[k] === "number"));
  if (!keys.length) keys = Object.keys(ms[ms.length - 1]).filter((k) => k !== "iter" && k !== "t" && typeof ms[ms.length - 1][k] === "number").slice(0, 3);
  const xs = ms.map((m, i) => (typeof m.iter === "number" ? m.iter : i));
  const xmin = Math.min(...xs), xmax = Math.max(...xs);
  const pad = 30 * devicePixelRatio, pr = 10 * devicePixelRatio;
  const colors = ["#3b9eff", "#1d9e75", "#efad3a", "#7f77dd"];
  ctx.font = `${11 * devicePixelRatio}px -apple-system,sans-serif`;
  keys.forEach((k, ki) => {
    const ys = ms.map((m) => m[k]).filter((v) => typeof v === "number");
    if (ys.length < 2) return;
    let ymin = Math.min(...ys), ymax = Math.max(...ys); if (ymin === ymax) { ymin -= 1; ymax += 1; }
    const X = (x) => pad + ((x - xmin) / (xmax - xmin || 1)) * (W - pad - pr);
    const Y = (y) => H - pad - ((y - ymin) / (ymax - ymin || 1)) * (H - pad - pr);
    ctx.strokeStyle = colors[ki % colors.length]; ctx.lineWidth = 1.6 * devicePixelRatio; ctx.beginPath();
    let started = false;
    ms.forEach((m, i) => { const v = m[k]; if (typeof v !== "number") return; const px = X(xs[i]), py = Y(v); started ? ctx.lineTo(px, py) : ctx.moveTo(px, py); started = true; });
    ctx.stroke();
    ctx.fillStyle = colors[ki % colors.length];
    ctx.fillText(`${k} ${fmt(ys[ys.length - 1])}`, pad + 4, (14 + ki * 14) * devicePixelRatio);
  });
  ctx.strokeStyle = "#28313d"; ctx.lineWidth = devicePixelRatio; ctx.beginPath();
  ctx.moveTo(pad, H - pad); ctx.lineTo(W - pr, H - pad); ctx.stroke();
  ctx.fillStyle = "#6b7d92"; ctx.fillText(`iter ${xmin}…${xmax}`, W - 90 * devicePixelRatio, H - 10 * devicePixelRatio);
}

// ---------- result rendering ----------
function renderResult(res, error) {
  const el = $("#result");
  if (error) { el.innerHTML = `<div class="tag bad">error</div><pre class="muted">${escapeHtml(error)}</pre>`; return; }
  if (!res) { el.innerHTML = ""; return; }
  // special: anchor metric table
  if (res.metrics && Array.isArray(res.metrics) && res.metrics[0] && res.metrics[0].status) {
    el.innerHTML = metricTable(res.metrics) + (res.note ? `<p class="muted" style="font-size:12px">${res.note}</p>` : "");
    return;
  }
  el.innerHTML = renderValue(res);
}

function renderValue(v) {
  if (v === null || v === undefined) return '<span class="muted">—</span>';
  if (Array.isArray(v)) {
    if (v.every((x) => typeof x !== "object")) return v.map((x) => `<span class="tag">${escapeHtml(String(fmt(x)))}</span>`).join("");
    return v.map((x) => `<div style="margin:4px 0">${renderValue(x)}</div>`).join("");
  }
  if (typeof v === "object") {
    let h = '<div class="kv">';
    for (const [k, val] of Object.entries(v)) {
      if (k === "artifacts") continue;
      let vh;
      if (typeof val === "boolean") vh = `<span class="tag ${val ? "ok" : "bad"}">${val}</span>`;
      else if (val && typeof val === "object") vh = renderValue(val);
      else vh = escapeHtml(String(fmt(val)));
      h += `<div class="k">${escapeHtml(k)}</div><div class="v">${vh}</div>`;
    }
    return h + "</div>";
  }
  return escapeHtml(String(fmt(v)));
}

function metricTable(rows) {
  let h = '<table class="tbl"><tr><th>metric</th><th>value</th><th>unit</th><th>status</th><th>n</th><th>source</th></tr>';
  for (const r of rows) {
    const ci = r.ci ? ` <span class="muted">[${r.ci[0]}–${r.ci[1]}]</span>` : "";
    h += `<tr><td>${r.name}</td><td>${fmt(r.value)}${ci}</td><td>${r.unit || ""}</td>
      <td class="st-${r.status}">${r.status}</td><td>${r.n}</td><td class="muted">${r.source}</td></tr>`;
  }
  return h + "</table>";
}

// ---------- artifacts ----------
function renderArtifacts(container, paths) {
  container.innerHTML = "";
  for (const p of paths) container.appendChild(thumb(p));
}
function thumb(path) {
  const ext = path.split(".").pop().toLowerCase();
  const url = "/api/file?path=" + encodeURIComponent(path);
  const d = document.createElement("div"); d.className = "thumb"; d.onclick = () => viewFile(path, ext);
  let inner;
  if (["png", "jpg", "jpeg", "gif"].includes(ext)) inner = `<img src="${url}" loading="lazy">`;
  else if (["mp4", "webm"].includes(ext)) inner = `<video src="${url}" muted></video>`;
  else inner = `<div class="ph">${ext}</div>`;
  d.innerHTML = inner + `<div class="cap">${path.split("/").pop()}</div>`;
  return d;
}
async function viewFile(path, ext) {
  const url = "/api/file?path=" + encodeURIComponent(path);
  let body;
  if (["png", "jpg", "jpeg", "gif"].includes(ext)) body = `<img src="${url}">`;
  else if (["mp4", "webm"].includes(ext)) body = `<video src="${url}" controls autoplay loop></video>`;
  else { const t = await fetch(url).then((r) => r.text()); body = `<pre>${escapeHtml(t.slice(0, 20000))}</pre>`; }
  $("#viewer-body").innerHTML = body; $("#viewer").hidden = false;
}

// ---------- jobs + gallery side panels ----------
async function loadJobs() {
  const jobs = await api("/api/jobs");
  $("#jobs").innerHTML = jobs.slice(0, 40).map((j) => {
    const ago = j.created ? timeAgo(j.created) : "";
    return `<div class="jrow" onclick="reopen('${j.id}')">
      <div class="jl"><span><span class="jdot ${j.state}"></span>${j.label}</span><span class="muted" style="font-size:11px">${j.state}</span></div>
      <div class="jt">${j.category} · ${ago}</div></div>`;
  }).join("") || '<p class="muted" style="font-size:12px">No jobs yet.</p>';
}
window.reopen = (id) => { const a = S.actions.find((x) => x.id === (S.jobActionFor || {})[id]); openJob(id, a || S.action); };

async function loadGallery() {
  const arts = await api("/api/artifacts");
  const g = $("#gallery"); g.innerHTML = "";
  arts.slice(0, 60).forEach((a) => g.appendChild(thumb(a.path)));
  if (!arts.length) g.innerHTML = '<p class="muted" style="font-size:12px">No artifacts yet.</p>';
}

function initTabs() {
  document.querySelectorAll(".side-tabs .tab").forEach((t) => {
    t.onclick = () => {
      document.querySelectorAll(".side-tabs .tab").forEach((x) => x.classList.remove("active"));
      t.classList.add("active");
      const which = t.dataset.tab;
      $("#jobs").hidden = which !== "jobs"; $("#gallery").hidden = which !== "gallery";
      if (which === "gallery") loadGallery();
    };
  });
}

// ---------- utils ----------
function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }
function timeAgo(ts) { const s = Math.max(0, (Date.now() / 1000 - ts)); if (s < 60) return `${Math.round(s)}s ago`; if (s < 3600) return `${Math.round(s / 60)}m ago`; return `${Math.round(s / 3600)}h ago`; }

$("#run-btn").onclick = runAction;
$("#cancel-btn").onclick = cancelJob;
$("#viewer-close").onclick = () => ($("#viewer").hidden = true);
$("#viewer").onclick = (e) => { if (e.target.id === "viewer") $("#viewer").hidden = true; };
init();
